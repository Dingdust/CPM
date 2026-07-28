# ── 标准库 ──
import io
import os
import json
import math
import time
import random
import logging
import warnings
import argparse
from types import SimpleNamespace
from contextlib import nullcontext, redirect_stdout

# ── 数值计算 & 深度学习 ──
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, Sampler, DataLoader, DistributedSampler

# ── 音频 & 图像处理 ──
import librosa
import soundfile as sf
from PIL import Image
from scipy.signal import resample

# ── 数据处理 ──
import pyarrow as pa
import pyarrow.parquet as pq

# ── Transformers ──
from transformers.activations import ACT2FN
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers import (
    PreTrainedModel, GenerationMixin, PretrainedConfig, AutoTokenizer,
    SiglipImageProcessor, SiglipVisionModel, logging as hf_logging
)

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ═══════════════════════════════════════════════════════════════════════════════
#  公共工具
# ═══════════════════════════════════════════════════════════════════════════════

def _fix_rope_buffers(module, dim, device):
    """修复 torch.compile / meta-device 初始化后丢失的 RoPE buffer。"""
    config = module.config if hasattr(module, 'config') else module.talker_config
    freqs_cos, freqs_sin = precompute_freqs_cis(
        dim=dim, end=config.max_position_embeddings,
        rope_base=config.rope_theta, rope_scaling=config.rope_scaling
    )
    module.freqs_cos = freqs_cos.to(device)
    module.freqs_sin = freqs_sin.to(device)


def _apply_repetition_penalty(logits, input_ids, penalty):
    """对已生成的 token 施加重复惩罚。"""
    if penalty == 1.0:
        return logits
    for i in range(input_ids.shape[0]):
        seen = torch.unique(input_ids[i])
        score = logits[i, seen]
        logits[i, seen] = torch.where(
            score > 0, score / penalty, score * penalty
        )
    return logits


def _apply_top_p(logits, top_p):
    """Nucleus (top-p) 采样过滤。"""
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    mask = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = 0
    logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
    return logits


def _apply_top_k(logits, top_k):
    """Top-K 采样过滤。"""
    if top_k <= 0:
        return logits
    logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
    return logits


def _sample_token(logits, temperature, do_sample=True):
    """从 logits 中采样 token。"""
    scaled = logits / temperature
    if do_sample:
        return torch.multinomial(F.softmax(scaled, dim=-1), num_samples=1)
    return torch.argmax(scaled, dim=-1, keepdim=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 1: PigCPM 基础模型 —— 定义配置、注意力、FFN、MoE、完整Transformer
# ═══════════════════════════════════════════════════════════════════════════════

class PigCPMConfig(PretrainedConfig):
    """PigCPM 基础配置类，继承 HuggingFace PretrainedConfig。"""
    model_type = "pigcpm"

    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        # ── 核心结构参数 ──
        self.hidden_size = hidden_size          # 隐藏层维度
        self.num_hidden_layers = num_hidden_layers  # Transformer 层数
        self.use_moe = use_moe                  # 是否启用混合专家（MoE）
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)

        # ── 特殊 token ──
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)

        # ── 注意力参数 ──
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)

        # ── 激活 & FFN ──
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)

        # ── 位置编码 ──
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)

        # ── RoPE 扩展（YaRN，用于推理时超长上下文）──
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32, "beta_slow": 1, "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0, "type": "yarn"
        } if self.inference_rope_scaling else None

        # ── MoE 专有配置（use_moe=False 时忽略）──
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)


class RMSNorm(torch.nn.Module):
    """RMS LayerNorm：相比传统 LayerNorm 去掉了均值中心化，更高效。"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self._norm(x.float())).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    """
    预计算 RoPE（旋转位置编码）的 cos/sin 频率表。
    支持 YaRN 扩展（用于推理时外推到更长上下文）。
    返回: (freqs_cos, freqs_sin)，shape 均为 (end, dim)
    """
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    attn_factor = 1.0

    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attn_factor = rope_scaling.get("attention_factor", 1.0)

        if end / orig_max > 1.0:
            # YaRN 频率插值公式
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001),
                0, 1
            )
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """对 query 和 key 应用旋转位置编码（RoPE）。"""
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA（分组查询注意力）：将 KV 头复制 n_rep 倍以匹配 Q 头数。"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    分组查询注意力（GQA）+ QK 归一化。
    支持 Flash Attention（通过 PyTorch F.scaled_dot_product_attention）和自回归因果掩码。
    """
    def __init__(self, config: PigCPMConfig):
        super().__init__()
        self.num_key_value_heads = (
            config.num_attention_heads if config.num_key_value_heads is None
            else config.num_key_value_heads
        )
        self.n_local_heads = config.num_attention_heads        # Q 头数
        self.n_local_kv_heads = self.num_key_value_heads       # KV 头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # GQA 复制因子
        self.head_dim = config.head_dim
        self.is_causal = True

        # ── 投影层 ──
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # ── QK 归一化（提升训练稳定性）──
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(F, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape

        # ── 线性投影 ──
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # ── QK 归一化 + RoPE ──
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # ── KV Cache 拼接 ──
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # ── GQA 复制 KV 头 → 转置为 (B, H, S, D) ──
        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # ── 注意力计算 ──
        if (self.flash and (seq_len > 1)
                and (not self.is_causal or past_key_value is None)
                and (attention_mask is None or torch.all(attention_mask == 1))):
            # Flash Attention 快速路径
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal
            )
        else:
            # 标准注意力
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal:
                scores[:, :, :, -seq_len:] += torch.full(
                    (seq_len, seq_len), float("-inf"), device=scores.device
                ).triu(1)
            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv

        # ── 输出投影 ──
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    """标准 SwiGLU 前馈网络：gate_proj(swish) * up_proj → down_proj。"""
    def __init__(self, config: PigCPMConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MOEFeedForward(nn.Module):
    """
    混合专家（MoE）前馈网络。
    - 每个 token 通过门控路由到 top-k 个专家
    - 各专家输出按门控权重加权求和
    - 训练时计算负载均衡辅助损失（aux_loss）
    """
    def __init__(self, config: PigCPMConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            FeedForward(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.num_experts)
        ])

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)

        # ── 门控路由 ──
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(
            scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False
        )
        if self.config.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        # ── 各专家并行计算，按权重累加 ──
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                # 保持梯度图连通，避免 DDP 报未使用参数
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())

        # ── 负载均衡辅助损失 ──
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (
                (load * scores.mean(0)).sum()
                * self.config.num_experts
                * self.config.router_aux_loss_coef
            )
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()

        return y.view(batch_size, seq_len, hidden_dim)


class PigCPMBlock(nn.Module):
    """单个 Transformer Block：Pre-Norm 自注意力 + Pre-Norm FFN/MoE，带残差连接。"""
    def __init__(self, layer_id: int, config: PigCPMConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # ── 自注意力 + 残差 ──
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states = residual + hidden_states
        # ── FFN/MoE + 残差 ──
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class PigCPMModel(nn.Module):
    """PigCPM 纯解码器 Transformer 模型（不含 LM Head）。"""
    def __init__(self, config: PigCPMConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            PigCPMBlock(l, config) for l in range(self.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # ── 预计算 RoPE 频率表 ──
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim, end=config.max_position_embeddings,
            rope_base=config.rope_theta, rope_scaling=config.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        # 兼容 HuggingFace 的 Cache 对象
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # ── Token Embedding ──
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # ── 修正 meta-device 初始化丢失的 RoPE buffer ──
        if self.freqs_cos[0, 0] == 0:
            _fix_rope_buffers(self, self.config.head_dim, hidden_states.device)

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        # ── 逐层前向传播 ──
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states, position_embeddings,
                past_key_value=past_key_value, use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)
        aux_loss = sum(
            [l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
            hidden_states.new_zeros(1).squeeze()
        )
        return hidden_states, presents, aux_loss


class PigCPMForCausalLM(PreTrainedModel, GenerationMixin):
    """PigCPM 因果语言模型：PigCPMModel + LM Head，支持 HuggingFace generate。"""
    config_class = PigCPMConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: PigCPMConfig = None):
        self.config = config or PigCPMConfig()
        super().__init__(self.config)
        self.model = PigCPMModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 权重绑定：lm_head 与 embed_tokens 共享权重
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False,
                logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids, attention_mask, past_key_values, use_cache, **kwargs
        )
        # 仅保留最后 logits_to_keep 个位置的 logits（推理优化）
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Shift: 预测位置 i 对应标签位置 i+1
            x = logits[..., :-1, :].contiguous()
            y = labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)

        return MoeCausalLMOutputWithPast(
            loss=loss, aux_loss=aux_loss, logits=logits,
            past_key_values=past_key_values, hidden_states=hidden_states
        )

    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85,
                 top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True,
                 num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        """
        自回归文本生成，支持：
        - temperature / top_p / top_k 采样
        - repetition_penalty 重复惩罚
        - streamer 流式输出
        参考: https://github.com/jingyaogong/minimind/discussions/611
        """
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = (
            attention_mask.repeat(num_return_sequences, 1)
            if attention_mask is not None else None
        )
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        if streamer:
            streamer.put(input_ids.cpu())

        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(
                input_ids[:, past_len:], attention_mask, past_key_values,
                use_cache=use_cache, **kwargs
            )
            attention_mask = (
                torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1)
                if attention_mask is not None else None
            )

            # ── 采样 ──
            logits = outputs.logits[:, -1, :]
            logits = _apply_repetition_penalty(logits, input_ids, repetition_penalty)
            logits = _apply_top_k(logits, top_k)
            logits = _apply_top_p(logits, top_p)
            next_token = _sample_token(logits, temperature, do_sample)

            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    next_token.new_full((next_token.shape[0], 1), eos_token_id),
                    next_token
                )

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None

            if streamer:
                streamer.put(next_token.cpu())

            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all():
                    break

        if streamer:
            streamer.end()
        if kwargs.get("return_kv"):
            return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 2: PigCPM-Omni 多模态模型
#  音频编码器(SenseVoice) + 视觉编码器(SigLIP) + Thinker + Talker 双塔架构
# ═══════════════════════════════════════════════════════════════════════════════

class OmniConfig(PigCPMConfig):
    """PigCPM-Omni 多模态配置，继承 PigCPMConfig 并扩展音频/视觉/双塔参数。"""
    model_type = "pigcpm-o"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # ── Talker（音频生成塔）参数 ──
        self.num_talker_hidden_layers = kwargs.get("num_talker_hidden_layers", 4)
        self.talker_hidden_size = kwargs.get("talker_hidden_size", 768)

        # ── 音频 codec 参数 ──
        self.audio_ids = kwargs.get("audio_ids", [16])          # <|audio_pad|> token id
        self.audio_special_token = kwargs.get("audio_special_token", "<|audio_pad|>")
        self.audio_hidden_size = kwargs.get("audio_hidden_size", 512)  # SenseVoice 输出维度
        self.audio_vocab_size = kwargs.get("audio_vocab_size", 2112)
        self.audio_pad_token = kwargs.get("audio_pad_token", 2049)
        self.audio_stop_token = kwargs.get("audio_stop_token", 2050)
        self.audio_spk_token = kwargs.get("audio_spk_token", 2051)
        self.spk_emb_size = kwargs.get("spk_emb_size", 192)

        # ── 思维链控制 ──
        self.think_end_ids = kwargs.get("think_end_ids", [26, 234, 234])  # </think>\n\n

        # ── 视觉参数 ──
        self.image_ids = kwargs.get("image_ids", [12])          # <|image_pad|> token id
        self.image_special_token = kwargs.get("image_special_token", "<|image_pad|>")
        self.image_hidden_size = kwargs.get("image_hidden_size", 768)  # SigLIP 输出维度
        self.image_token_len = kwargs.get("image_token_len", 64)       # 每张图占用的 token 数

        # ── 双塔桥接 ──
        self.bridge_layer = kwargs.get("bridge_layer", self.num_hidden_layers // 2 - 1)


class MMProjector(nn.Module):
    """多模态投影器：将编码器输出（音频/视觉）映射到 Thinker 隐藏空间。"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim),
            nn.GELU(), nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class TalkerHead(nn.Module):
    """
    Talker 输出头：基础线性投影 + 8层 LoRA 风格适配器。
    每个适配器对应一个音频 codec 层，输出 = base(x) + adapter_i(x)。
    """
    def __init__(self, in_features, out_features, num_layers=8, rank=256):
        super().__init__()
        self.num_layers = num_layers
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, rank, bias=False),
                nn.GELU(),
                nn.Linear(rank, out_features, bias=False)
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        base_out = self.base(x)
        return [base_out + adapter(x) for adapter in self.adapters]


class TalkerEmbedding(nn.Module):
    """
    Talker 嵌入层：基础 Embedding + 8层 LoRA 风格适配器。
    用于将音频 codec token 转为嵌入向量。
    """
    def __init__(self, num_embeddings, embedding_dim, num_layers=8, rank=256):
        super().__init__()
        self.num_layers = num_layers
        self.base = nn.Embedding(num_embeddings, embedding_dim)
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Embedding(num_embeddings, rank),
                nn.GELU(),
                nn.Linear(rank, embedding_dim, bias=False)
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        base_out = self.base(x)
        # 各层加权平均
        return sum(
            base_out[:, i, :] + self.adapters[i](x[:, i, :])
            for i in range(len(self.adapters))
        ) / self.num_layers


class SenseVoiceAudioProcessor:
    """SenseVoice 音频预处理包装器：波形 → fbank 特征。"""
    def __init__(self, frontend):
        self.frontend = frontend

    def __call__(self, wav, sampling_rate=16000, return_tensors="pt", return_attention_mask=True, **kwargs):
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav).float()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        with torch.no_grad():
            fbank, flen = self.frontend(wav, torch.tensor([wav.size(1)]))
        return SimpleNamespace(
            input_features=fbank,
            attention_mask=(torch.arange(fbank.size(1)) < flen[0]).long().unsqueeze(0)
        )


class TalkerModule(nn.Module):
    """
    Talker（音频生成塔）：独立的小型 Transformer。
    - 接收 Thinker 的桥接隐藏状态 + 音频 codec 嵌入
    - 8路并行输出头预测音频 codec 各层
    """
    def __init__(self, config):
        super().__init__()
        self.talker_config = PigCPMConfig(
            hidden_size=config.talker_hidden_size, use_moe=config.use_moe
        )
        self.layers = nn.ModuleList([
            PigCPMBlock(l, self.talker_config)
            for l in range(config.num_talker_hidden_layers)
        ])
        self.norm = RMSNorm(config.talker_hidden_size, eps=config.rms_norm_eps)
        self.lm_head = TalkerHead(config.talker_hidden_size, config.audio_vocab_size)
        self.embed_tokens = TalkerEmbedding(config.audio_vocab_size, config.talker_hidden_size)

        # ── 投影层 ──
        self.codec_proj = nn.Sequential(
            nn.Linear(config.talker_hidden_size, config.talker_hidden_size),
            nn.GELU(),
            nn.Linear(config.talker_hidden_size, config.talker_hidden_size),
            RMSNorm(config.talker_hidden_size, eps=config.rms_norm_eps)
        )
        self.embed_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.talker_hidden_size),
            RMSNorm(config.talker_hidden_size, eps=config.rms_norm_eps)
        )

        # ── 可学习的融合比例 ──
        self.text_scale = nn.Parameter(torch.tensor(3.0))
        self.audio_scale = nn.Parameter(torch.tensor(1.0))

        self.spk_proj = nn.Linear(config.spk_emb_size, config.talker_hidden_size, bias=False)

        # ── RoPE ──
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=self.talker_config.head_dim, end=config.max_position_embeddings,
            rope_base=config.rope_theta, rope_scaling=config.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)


class PigCPMOmni(PigCPMForCausalLM):
    """
    PigCPM-Omni 多模态模型 —— Thinker + Talker 双塔架构。
    - Thinker: 文字理解（复用 PigCPMForCausalLM 的 model）
    - Talker: 音频生成（TalkerModule）
    - 音频编码器: SenseVoice（冻结）
    - 视觉编码器: SigLIP（冻结）
    """
    config_class = OmniConfig

    def __init__(self, config: OmniConfig = None,
                 audio_encoder_path="./model/SenseVoiceSmall",
                 vision_model_path="./model/siglip2-base-p32-256-ve"):
        config = config or OmniConfig()
        super().__init__(config)

        # 别名：self.thinker == self.model, self.thinker.lm_head == self.lm_head
        object.__setattr__(self, 'thinker', self.model)
        object.__setattr__(self.model, 'lm_head', self.lm_head)

        self.talker = TalkerModule(config)
        self.audio_proj = MMProjector(config.audio_hidden_size, config.hidden_size)
        self.vision_proj = MMProjector(config.image_hidden_size, config.hidden_size)
        self.audio_pad_token = config.audio_pad_token
        self.audio_stop_token = config.audio_stop_token
        self.audio_spk_token = config.audio_spk_token

        # ── 加载编码器（冻结参数）──
        audio_encoder, audio_processor = self._load_sensevoice(audio_encoder_path)
        object.__setattr__(self, 'audio_encoder', audio_encoder)
        object.__setattr__(self, 'audio_processor', audio_processor)

        vision_encoder, vision_processor = self._load_vision(vision_model_path)
        object.__setattr__(self, 'vision_encoder', vision_encoder)
        object.__setattr__(self, 'vision_processor', vision_processor)

    @staticmethod
    def _load_sensevoice(path):
        """加载 SenseVoice 音频编码器（CPU 推理，冻结参数）。"""
        if not os.path.exists(path):
            warnings.warn(f"[PigCPMOmni] SenseVoice path not found: {path}")
            return None, None
        logging.getLogger().setLevel(logging.ERROR)
        hf_logging.set_verbosity_error()
        with redirect_stdout(io.StringIO()):
            from funasr import AutoModel
            m = AutoModel(model=path, trust_remote_code=True, disable_update=True, device="cpu")
        encoder = m.model.encoder
        frontend = m.kwargs["frontend"]
        for p in encoder.parameters():
            p.requires_grad = False
        return encoder.eval().float(), SenseVoiceAudioProcessor(frontend.eval())

    @staticmethod
    def _load_vision(path):
        """加载 SigLIP 视觉编码器（冻结参数）。"""
        if path is None or not os.path.exists(path):
            warnings.warn(f"[PigCPMOmni] Vision model path not found: {path}. vision_encoder will be None!")
            return None, None
        hf_logging.set_verbosity_error()
        try:
            model = SiglipVisionModel.from_pretrained(path)
        except (RuntimeError, ValueError):
            return None, None
        processor = SiglipImageProcessor.from_pretrained(path)
        for p in model.parameters():
            p.requires_grad = False
        return model.eval(), processor

    # ── 音频编码相关 ──

    @torch.compiler.disable
    def encode_audio_inputs(self, audio_inputs, audio_lens=None):
        """
        批量编码音频 fbank 特征 → Thinker 隐藏空间。
        输入: audio_inputs (B, T, 560) 或 None
        返回: List[Tensor] 或 None
        """
        if (audio_inputs is None) or (self.audio_encoder is None) or (not audio_inputs.any()):
            return None

        batch_mask = audio_inputs.flatten(1).any(1)  # 过滤空音频
        enc_dtype = next(self.audio_encoder.parameters()).dtype
        valid_fbank = audio_inputs[batch_mask].to(dtype=enc_dtype)

        if audio_lens is not None:
            valid_lens = audio_lens[batch_mask].to(valid_fbank.device)
        else:
            valid_lens = torch.tensor(
                [valid_fbank.size(1)] * valid_fbank.size(0), device=valid_fbank.device
            )

        with torch.no_grad():
            emb, _ = self.audio_encoder(valid_fbank, valid_lens)

        proj_dtype = next(self.audio_proj.parameters()).dtype
        emb_list = [
            self.audio_proj(
                emb[i, :max(1, min(valid_lens[i].item(), emb.size(1)))]
                .unsqueeze(0).to(proj_dtype)
            ).squeeze(0)
            for i in range(emb.size(0))
        ]

        if batch_mask.all():
            return emb_list

        # 恢复原始批量顺序（空音频位置填 None）
        out = [None] * audio_inputs.size(0)
        j = 0
        for i in range(audio_inputs.size(0)):
            if batch_mask[i]:
                out[i] = emb_list[j]
                j += 1
        return out

    @torch.compiler.disable
    def inject_audio_features(self, tokens, h, audio_feats, seqlen):
        """
        将音频特征注入到 Thinker 的隐藏状态中。
        在文本序列中的 <|audio_pad|> 位置替换为对应的音频编码特征。
        """
        if audio_feats is None or not self.config.audio_ids:
            return h

        marker = self.config.audio_ids[0]
        out = []
        for b in range(h.size(0)):
            hb = h[b]
            seq = tokens[b].tolist()
            i = 0
            af = audio_feats[b] if audio_feats[b] is not None else None
            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    while i < len(seq) and seq[i] == marker:
                        i += 1
                    if af is not None:
                        inject_len = min(af.size(0), i - start)
                        hb = torch.cat(
                            (hb[:start], af[:inject_len], hb[start + inject_len:]), dim=0
                        )
                        af = None
                else:
                    i += 1
            out.append(hb)
        return torch.stack(out)

    # ── 视觉编码相关 ──

    @torch.compiler.disable
    def get_image_embeddings(self, image_inputs):
        """通过 SigLIP 获取图像嵌入。"""
        if hasattr(image_inputs, 'keys'):
            image_inputs = {
                k: v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v
                for k, v in image_inputs.items()
            }
            pixel_attention_mask = image_inputs.get('pixel_attention_mask')
            if pixel_attention_mask is not None and not pixel_attention_mask.any():
                pv = image_inputs['pixel_values']
                return pv.new_zeros(pv.size(0), pv.size(1), self.config.image_hidden_size)

        with torch.no_grad():
            outputs = self.vision_encoder(**image_inputs)
        return outputs.last_hidden_state

    @torch.compiler.disable
    def encode_image_inputs(self, pixel_values):
        """
        批量编码图像 → 通过视觉投影器映射到 Thinker 隐藏空间。
        返回: (B, image_token_len, hidden_size) 或 None
        """
        if pixel_values is None or self.vision_encoder is None:
            return None

        mask = pixel_values.flatten(1).any(1)
        if not mask.any():
            return pixel_values.new_zeros(
                pixel_values.size(0), self.config.image_token_len, self.config.hidden_size
            )

        with torch.no_grad():
            emb = self.vision_encoder(pixel_values=pixel_values[mask]).last_hidden_state
        if emb.dim() == 2:
            emb = emb.unsqueeze(0)
        emb = self.vision_proj(emb)

        if mask.all():
            return emb

        idx = mask.nonzero().view(-1, 1, 1).expand_as(emb)
        return emb.new_zeros(pixel_values.size(0), *emb.shape[1:]).scatter(0, idx, emb)

    @torch.compiler.disable
    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
        """
        将视觉特征注入到 Thinker 的隐藏状态中。
        在文本序列 <|image_pad|> 位置替换为对应视觉 token。
        """
        if vision_tensors is None or not self.config.image_ids:
            return h

        marker = self.config.image_ids[0]
        vf = vision_tensors
        if vf.dim() == 3:
            vf = vf.unsqueeze(1)

        out = []
        for b in range(h.size(0)):
            hb = h[b]
            seq = tokens[b].tolist()
            k = 0
            i = 0
            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    while i < len(seq) and seq[i] == marker:
                        i += 1
                    if k < vf.size(1):
                        hb = torch.cat(
                            (hb[:start], vf[b][k][:i - start], hb[i:]), dim=0
                        )[:seqlen]
                        k += 1
                else:
                    i += 1
            out.append(hb)
        return torch.stack(out)

    # ── 核心前向 & 生成 ──

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False,
                logits_to_keep=0, audio_inputs=None, audio_lens=None, pixel_values=None, **args):
        """
        双塔前向传播：
        1. Thinker: 文本 + 音频特征注入 + 视觉特征注入 → 文本 logits
        2. Talker: Thinker 桥接层隐藏状态 + 音频 codec → 音频 logits (8路)
        """
        # ── 解析输入：2D 为纯文本，3D 为 (9, T) = 8路 audio + 1路 text ──
        if len(input_ids.shape) == 2:
            batch_size, seq_length = input_ids.shape
            text_ids = input_ids
            audio_ids = torch.full(
                (batch_size, 8, seq_length), self.audio_pad_token,
                dtype=torch.long, device=input_ids.device
            )
        else:
            batch_size, _, seq_length = input_ids.shape
            text_ids = input_ids[:, 8, :]
            audio_ids = input_ids[:, :8, :]

        if hasattr(past_key_values, 'layers'):
            past_key_values = None

        n_thinker = len(self.thinker.layers)
        n_talker = len(self.talker.layers)
        past_key_values = past_key_values or ([None] * (n_thinker + n_talker))
        start_pos = (
            past_key_values[0][0].shape[1]
            if past_key_values[0] is not None else 0
        )

        # ── 修正 meta-device 初始化丢失的 RoPE buffer ──
        if self.thinker.freqs_cos[0, 0] == 0:
            _fix_rope_buffers(self.thinker, self.config.head_dim, input_ids.device)
        if self.talker.freqs_cos[0, 0] == 0:
            _fix_rope_buffers(self.talker, self.talker.talker_config.head_dim, input_ids.device)

        presents = []

        # ═══════ Thinker: 纯文本输入，输出文本 logits ═══════
        hidden_states = self.thinker.dropout(self.thinker.embed_tokens(text_ids))
        position_embeddings = (
            self.thinker.freqs_cos[start_pos:start_pos + seq_length],
            self.thinker.freqs_sin[start_pos:start_pos + seq_length]
        )

        # ── 注入音频特征 ──
        if audio_inputs is not None and start_pos == 0:
            audio_features = self.encode_audio_inputs(audio_inputs, audio_lens)
            hidden_states = self.inject_audio_features(
                text_ids, hidden_states, audio_features, seq_length
            )

        # ── 注入视觉特征 ──
        if pixel_values is not None and start_pos == 0:
            if hasattr(pixel_values, 'keys'):
                img_emb = self.get_image_embeddings(pixel_values).to(hidden_states.dtype)
                vision_tensors = self.vision_proj(img_emb)
            else:
                if len(pixel_values.shape) == 6:
                    pixel_values = pixel_values.squeeze(2)
                if len(pixel_values.shape) == 4:
                    pixel_values = pixel_values.unsqueeze(1)
                bs, num = pixel_values.shape[0], pixel_values.shape[1]
                stack_dim = 1 if bs > 1 else 0
                vision_tensors = torch.stack([
                    self.encode_image_inputs(pixel_values[:, i, :, :, :])
                    for i in range(num)
                ], dim=stack_dim)
            hidden_states = self.count_vision_proj(
                tokens=text_ids, h=hidden_states,
                vision_tensors=vision_tensors, seqlen=seq_length
            )

        # ── Thinker 逐层传播，记录桥接层隐藏状态 ──
        bridge_states = hidden_states
        for i, (layer, past_key_value) in enumerate(
            zip(self.thinker.layers, past_key_values[:n_thinker])
        ):
            hidden_states, present = layer(
                hidden_states, position_embeddings,
                past_key_value=past_key_value, use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
            if i == self.config.bridge_layer:
                bridge_states = hidden_states

        h_thinker = self.thinker.norm(hidden_states)

        # ═══════ Talker: Thinker 隐藏状态 + 音频 codec → 音频 logits ═══════
        talker_emb = self.talker.embed_tokens(audio_ids)

        # ── 说话人嵌入注入 ──
        spk_emb = args.get('spk_emb', None)
        if spk_emb is not None:
            spk_mask = (audio_ids[:, 0, :] == self.audio_spk_token).unsqueeze(-1)
            talker_emb = torch.where(
                spk_mask,
                self.talker.spk_proj(spk_emb).unsqueeze(1),
                talker_emb
            )

        # ── Thinker 桥接状态 + 音频 codec 嵌入 → 融合 ──
        hidden_states = (
            self.talker.embed_proj(bridge_states) * self.talker.text_scale
            + self.talker.codec_proj(talker_emb) * self.talker.audio_scale
        )

        talker_pos_emb = (
            self.talker.freqs_cos[start_pos:start_pos + seq_length],
            self.talker.freqs_sin[start_pos:start_pos + seq_length]
        )
        for layer, past_key_value in zip(
            self.talker.layers, past_key_values[n_thinker:]
        ):
            hidden_states, present = layer(
                hidden_states, talker_pos_emb,
                past_key_value=past_key_value, use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        h_talker = self.talker.norm(hidden_states)

        # ── 计算输出 ──
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int) else logits_to_keep
        )

        aux_loss = sum(
            l.mlp.aux_loss
            for l in list(self.thinker.layers) + list(self.talker.layers)
            if isinstance(l.mlp, MOEFeedForward)
        )
        # 保持 DDP 梯度图连通（防止未使用参数报错）
        aux_loss += (
            sum(p.sum() for p in self.audio_proj.parameters()) * 0
            + sum(p.sum() for p in self.vision_proj.parameters()) * 0
            + sum(p.sum() for p in self.talker.lm_head.adapters.parameters()) * 0
            + sum(p.sum() for p in self.talker.spk_proj.parameters()) * 0
        )

        text_logits = self.thinker.lm_head(h_thinker[:, slice_indices, :])
        audio_logits = self.talker.lm_head(h_talker[:, slice_indices, :])

        out = MoeCausalLMOutputWithPast(
            aux_loss=aux_loss, logits=text_logits, past_key_values=presents
        )
        out.audio_logits = audio_logits  # 附加音频 logits
        return out

    @torch.inference_mode()
    def generate(self, input_ids, eos_token_id=2, max_new_tokens=1024,
                 temperature=0.75, top_p=0.90, stream=False, rp=1.,
                 use_cache=True, return_audio_codes=False, **args):
        """
        多模态自回归生成（文本 + 音频）。
        stream=True：逐个 token yield
        stream=False：返回完整序列
        """
        if stream:
            return self.stream_generate(
                input_ids, eos_token_id, max_new_tokens, temperature, top_p,
                rp, use_cache, return_audio_codes, **args
            )
        tokens = list(self.stream_generate(
            input_ids, eos_token_id, max_new_tokens, temperature, top_p,
            rp, use_cache, return_audio_codes, **args
        ))
        return tokens[-1] if tokens else input_ids

    def stream_generate(self, input_ids, eos_token_id, max_new_tokens,
                        temperature, top_p, rp, use_cache,
                        return_audio_codes=False, **args):
        """
        流式多模态生成器。
        同时生成文本 token 和 8 层音频 codec，支持 think-end 延迟控制。
        yield: (new_text_tokens, audio_frame_or_None)
        """
        start_pos = input_ids.shape[1]
        past_kvs = None
        text_finished = False
        first_finished = True

        # ── 音频生成状态 ──
        audio_codes = [[] for _ in range(8)]
        audio_stop_pos = [None] * 8
        audio_buffer = torch.full(
            (1, 8, start_pos), self.audio_pad_token,
            dtype=torch.long, device=input_ids.device
        )

        # ── 参考音频 / 说话人嵌入初始化 ──
        spk_emb = args.get('spk_emb', None)
        ref_codes = args.get('ref_codes', None)
        ref_len = ref_codes.shape[2] if ref_codes is not None else 0
        spk_reserve = 1 if spk_emb is not None else 0
        fill_end = start_pos
        fill_start = max(spk_reserve, start_pos - ref_len)
        if ref_codes is not None and fill_start < fill_end:
            audio_buffer[:, :, fill_start:fill_end] = ref_codes[:, :, -(fill_end - fill_start):]
        if spk_emb is not None and fill_start > 0:
            audio_buffer[:, :, fill_start - 1] = self.audio_spk_token

        think_end_step = None
        generated_tokens = [] if args.get('open_thinking', False) else None

        while input_ids.shape[1] < start_pos + max_new_tokens:
            # ── 模型前向 ──
            use_cache_now = past_kvs is not None and use_cache
            ab = audio_buffer[:, :, -1:] if use_cache_now else audio_buffer
            ids = input_ids[:, -1:] if use_cache_now else input_ids
            out = self.forward(
                torch.cat((ab, ids.unsqueeze(1)), dim=1),
                past_key_values=past_kvs, use_cache=use_cache, **args
            )
            past_kvs = out.past_key_values

            # ── 文本采样 ──
            logits = out.logits[0, -1, :].clone()
            logits = _apply_repetition_penalty(logits.unsqueeze(0), input_ids, rp).squeeze(0)
            logits = _apply_top_p(logits.unsqueeze(0), top_p).squeeze(0)
            text_token = _sample_token(logits.unsqueeze(0), temperature).item()

            if text_finished:
                text_token = (
                    args.get('enter_token_id', 201) if first_finished
                    else args.get('pad_token_id', 0)
                )
                first_finished = False

            # ── 音频采样（8路并行，think-end 延迟控制）──
            step = input_ids.shape[1] - start_pos
            audio_step = step - 1  # 延迟1步

            if generated_tokens is not None:
                generated_tokens.append(text_token)
                if (not think_end_step
                        and generated_tokens[-len(self.config.think_end_ids):]
                        == list(self.config.think_end_ids)):
                    think_end_step = step + 2
                audio_step = (step - think_end_step) if think_end_step else -1

            for i, al in enumerate(out.audio_logits):
                if audio_step < i:
                    audio_codes[i].append(self.audio_pad_token)
                else:
                    logits_i = al[0, -1, :].clone() / 0.2
                    # 防止连续重复采样
                    for prev_code in audio_codes[i][-3:]:
                        score = logits_i[prev_code]
                        logits_i[prev_code] = torch.where(
                            score > 0, score / 1.05, score * 1.05
                        )
                    top_val, top_idx = logits_i.topk(50)
                    code = top_idx[
                        torch.multinomial(F.softmax(top_val, dim=-1), 1)
                    ].item()
                    audio_codes[i].append(code)
                    if audio_stop_pos[i] is None and code >= 2048:
                        audio_stop_pos[i] = len(audio_codes[i]) - 1

            # ── 检查生成终止 ──
            if text_finished and all(
                audio_stop_pos[i] is not None for i in range(8)
            ):
                break

            # ── 更新状态 ──
            input_ids = torch.cat(
                (input_ids, torch.tensor([[text_token]], device=input_ids.device)), dim=1
            )
            audio_buffer = torch.cat(
                (audio_buffer,
                 torch.full((1, 8, 1), self.audio_pad_token,
                            dtype=torch.long, device=input_ids.device)),
                dim=2
            )
            for i in range(min(audio_step + 1, 8)):
                audio_buffer[0, i, -1] = audio_codes[i][-1]

            # ── 按需输出音频帧 ──
            audio_frame = None
            if return_audio_codes and audio_step >= 7:
                frame = [audio_codes[i][step - 7 + i] for i in range(8)]
                active_layers = sum(
                    1 for i in range(8)
                    if audio_stop_pos[i] is None or step - 7 + i < audio_stop_pos[i]
                )
                if active_layers >= 8:
                    audio_frame = frame

            if not text_finished:
                yield input_ids[:, start_pos:], audio_frame
                if text_token == eos_token_id:
                    text_finished = True
            else:
                yield None, audio_frame


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 3: 数据集 —— OmniDataset（多模态数据加载、预处理、增强）
# ═══════════════════════════════════════════════════════════════════════════════

def pre_processing_chat(conversations, add_system_ratio=0.2):
    """对话预处理：以一定概率注入随机 system prompt（用于 SFT 风格多样化）。"""
    if any(conv.get('tools') for conv in conversations):
        return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是PigCPM，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是PigCPM，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are PigCPM, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are PigCPM, a small but useful language model."
    ]

    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    """
    对话后处理：以一定概率移除空 <think> 块，避免模型学习到无意义的空思考。
    """
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        return prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


class OmniDataset(Dataset):
    """
    多模态对话数据集。
    数据格式: Parquet 文件，每行包含 conversations(JSON)、question_audios、answer_audios、image_bytes、ref_audios、spk_emb。
    支持: 文本/音频/图像的多模态训练 + 数据增强 + scheduled sampling。
    """
    def __init__(self, data_path, tokenizer, audio_processor=None, vision_processor=None,
                 max_length=1200, audio_special_token='<|audio_pad|>', image_special_token='<|image_pad|>',
                 audio_stop_token=2050, audio_pad_token=2049, audio_spk_token=2051,
                 audio_vocab_size=2112, scheduled_sampling=0.05, image_token_len=64):
        super().__init__()
        # ── 加载 Parquet 数据 ──
        tables = [
            pa.Table.from_batches(pq.ParquetFile(p.strip()).iter_batches())
            for p in data_path.split(',')
        ]
        tables = [
            t.cast(pa.schema([
                f.with_type(pa.large_string()) if pa.types.is_string(f.type) else f
                for f in t.schema
            ]))
            for t in tables
        ]
        self.table = pa.concat_tables(tables, promote_options='default')

        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.vision_processor = vision_processor
        self.max_length = max_length
        self.audio_token = audio_special_token
        self.image_token_len = image_token_len
        self.image_token = image_special_token * image_token_len
        self.audio_stop_token = audio_stop_token
        self.audio_pad_token = audio_pad_token
        self.audio_spk_token = audio_spk_token
        self.audio_vocab_size = audio_vocab_size
        self.scheduled_sampling_prob = scheduled_sampling
        self.text_vocab_size = len(tokenizer)

        # ── 预计算关键 token id ──
        self.image_token_id = tokenizer.encode(image_special_token, add_special_tokens=False)[0]
        self.audio_token_id = tokenizer.encode(audio_special_token, add_special_tokens=False)[0]
        self.think_end_ids = tokenizer.encode('</think>\n\n', add_special_tokens=False)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.table)

    def _get_col(self, index, name, default=None):
        """安全获取列数据，列不存在时返回默认值。"""
        if name in self.table.column_names:
            val = self.table[name][index].as_py()
            return val if val is not None else default
        return default

    # ── 音频处理 ──

    @staticmethod
    def _read_wav(source, target_sr=16000):
        """从文件路径或字节流读取波形，统一为单声道(target_sr)。"""
        if isinstance(source, (str, os.PathLike)):
            wav, sr = sf.read(str(source))
        else:
            wav, sr = sf.read(io.BytesIO(source))
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != target_sr:
            wav = librosa.resample(wav.astype(float), orig_sr=sr, target_sr=target_sr)
        return wav.astype(np.float32)

    @staticmethod
    def process_audio(audio_path, audio_processor):
        """静态方法：加载并预处理音频文件。"""
        wav = OmniDataset._read_wav(audio_path)
        inputs = audio_processor(wav, sampling_rate=16000,
                                 return_tensors="pt", return_attention_mask=True)
        valid_len = inputs.attention_mask.sum().item()
        return inputs.input_features.squeeze(0), valid_len

    def augment_wav(self, wav, sr=16000):
        """
        音频波形数据增强（多策略随机组合）：
        - 变速（0.7x ~ 1.6x）
        - 加噪声、音量扰动、时间遮蔽、平滑、混响、粉红噪声
        """
        if random.random() < 0.5:
            speed = random.uniform(0.7, 1.6)
            wav = resample(wav, int(len(wav) / speed)).astype(np.float32)
        if random.random() < 0.3:
            noise = np.random.randn(len(wav)).astype(np.float32) * random.uniform(0.001, 0.01)
            wav = wav + noise
        if random.random() < 0.3:
            wav = wav * random.uniform(0.8, 1.2)
        if random.random() < 0.2 and len(wav) > sr:
            start = random.randint(0, len(wav) - sr // 4)
            wav[start:start + sr // 4] = 0
        if random.random() < 0.2:
            k = random.choice([3, 5, 7])
            wav = np.convolve(wav, np.ones(k) / k, mode='same').astype(np.float32)
        if random.random() < 0.3:
            ir_len = int(sr * random.uniform(0.05, 0.2))
            ir = np.random.randn(ir_len).astype(np.float32) * np.exp(-np.linspace(0, 10, ir_len))
            ir[0] = 1.0
            ir /= np.sqrt(np.sum(ir ** 2) + 1e-6)
            wav = np.convolve(wav, ir, mode='same').astype(np.float32)
        if random.random() < 0.2:
            pink = np.cumsum(np.random.randn(len(wav))).astype(np.float32)
            pink /= np.max(np.abs(pink)) + 1e-6
            wav = wav + pink * random.uniform(0.003, 0.015)
        return np.clip(wav, -1.0, 1.0).astype(np.float32)

    def augment_mel(self, fbank):
        """Mel 频谱数据增强：随机频域/时域遮蔽。"""
        T, D = fbank.shape
        if random.random() < 0.5:
            f = random.randint(1, 64)
            f0 = random.randint(0, D - f)
            fbank[:, f0:f0 + f] = 0
        if random.random() < 0.5 and T > 1:
            t = random.randint(1, min(10, T))
            t0 = random.randint(0, T - t)
            fbank[t0:t0 + t, :] = 0
        return fbank

    def load_audio_inputs(self, audio_bytes):
        """加载音频字节流 → fbank 特征 + 长度。"""
        if not audio_bytes:
            return None, 0
        wav = self._read_wav(audio_bytes)
        wav = self.augment_wav(wav)
        inputs = self.audio_processor(wav, sampling_rate=16000,
                                      return_tensors="pt", return_attention_mask=True)
        valid_len = inputs.attention_mask.sum().item()
        return self.augment_mel(inputs.input_features.squeeze(0)), valid_len

    # ── 图像处理 ──

    def load_image_inputs(self, image_bytes):
        """加载图像字节流 → 像素张量。"""
        if not image_bytes or self.vision_processor is None:
            return None
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        inputs = self.vision_processor(images=image, return_tensors="pt")
        if hasattr(inputs, 'keys'):
            return {k: v for k, v in inputs.items()}
        return inputs.pixel_values

    # ── 对话构造 ──

    def create_chat_prompt(self, conversations, audio_features_length=0):
        """
        构造聊天模板 prompt。
        - 预处理注入 system prompt
        - 最后一轮 user 消息中随机注入音频占位符
        - 随机调整 <image> 标签位置
        - 后处理移除空 think 块
        """
        conversations = pre_processing_chat(conversations)
        messages = []
        is_last_user = lambda i: i == max(
            j for j, t in enumerate(conversations) if t['role'] == 'user'
        )
        for idx, turn in enumerate(conversations):
            role, content = turn['role'], turn['content']
            if role == 'user' and is_last_user(idx) and audio_features_length > 0:
                ap = self.audio_token * audio_features_length
                r = random.random()
                if r < 0.4:
                    content = ap
                elif r < 0.6:
                    content = content
                elif r < 0.8:
                    content = ap + '\n\n' + content
                else:
                    content = content + '\n\n' + ap
            if '<image>' in content:
                r = random.random()
                if r < 0.2:
                    content = '<image>\n' + content.replace('<image>', '').strip()
                elif r < 0.4:
                    content = '<image>\n\n' + content.replace('<image>', '').strip()
                elif r < 0.6:
                    content = content.replace('<image>', '').strip() + '\n' + '<image>'
                else:
                    content = content.replace('<image>', '').strip() + '\n\n' + '<image>'
            messages.append({"role": role, "content": content})
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return post_processing_chat(prompt)

    # ── 标签生成 ──

    def generate_text_labels(self, input_ids):
        """
        从 tokenized 对话生成训练标签：
        - 仅对 assistant 回复区域设置有效标签（非 -100）
        - 在 BOS 和 EOS 之间标记需要学习的 token
        """
        labels = [-100] * len(input_ids)
        ranges = []
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                ranges.append((start, end))
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels, ranges

    @staticmethod
    def _parse_audio_codec(tokens, audio_stop_token):
        """将扁平 token 列表拆分为 8 层，每层末尾加 stop token。"""
        audio_codes_8layers = [[] for _ in range(8)]
        for i in range(0, len(tokens) - 7, 8):
            for j in range(8):
                audio_codes_8layers[j].append(tokens[i + j])
        for layer in audio_codes_8layers:
            layer.append(audio_stop_token)
        return audio_codes_8layers

    def _build_audio_labels(self, assistant_ranges, last_audio_codes, input_ids,
                            spk_emb_raw, ref_audios):
        """
        构造 8 层音频标签。
        包含参考音频填充、说话人 token 注入、生成音频 codec 对齐。
        返回: (Y_audio_layers: 8×T 输入, audio_labels: 8×T 标签)
        """
        Y_audio_layers = [[self.audio_pad_token] * self.max_length for _ in range(8)]
        audio_labels = [[-100] * self.max_length for _ in range(8)]

        if not (assistant_ranges and last_audio_codes):
            return Y_audio_layers, audio_labels

        assistant_start, assistant_end = assistant_ranges[-1]
        # 跳过 think 区域（音频在 </think> 之后才开始）
        for pos in range(assistant_start, min(assistant_end, assistant_start + 50)):
            if input_ids[pos:pos + len(self.think_end_ids)] == self.think_end_ids:
                assistant_start = pos + len(self.think_end_ids)
                break

        has_spk = bool(spk_emb_raw)
        has_ref = bool(ref_audios) and random.random() > 0.5
        spk_reserve = 1 if has_spk else 0

        # ── 填充参考音频 ──
        if has_ref:
            ref_codes = [[] for _ in range(8)]
            for i in range(0, len(ref_audios) - 7, 8):
                for j in range(8):
                    ref_codes[j].append(ref_audios[i + j])
            ref_len = len(ref_codes[0])
            ref_start = max(spk_reserve, assistant_start - ref_len)
            for layer_idx in range(8):
                codes = (
                    ref_codes[layer_idx][-(assistant_start - ref_start):]
                    if ref_len > (assistant_start - ref_start)
                    else ref_codes[layer_idx]
                )
                for i, code in enumerate(codes):
                    Y_audio_layers[layer_idx][ref_start + i] = code
        else:
            ref_start = assistant_start

        # ── 填充说话人 token ──
        if has_spk and ref_start > 0:
            spk_pos = ref_start - 1
            for layer_idx in range(8):
                Y_audio_layers[layer_idx][spk_pos] = self.audio_spk_token

        # ── 填充生成音频 codec ──
        for layer_idx in range(8):
            codes = last_audio_codes[layer_idx]
            start_pos_code = assistant_start + layer_idx + 1
            for i, code in enumerate(codes):
                if start_pos_code + i < self.max_length:
                    Y_audio_layers[layer_idx][start_pos_code + i] = code
                    audio_labels[layer_idx][start_pos_code + i] = code

        return Y_audio_layers, audio_labels

    def apply_scheduled_sampling(self, input_ids, audio_labels, text_labels):
        """
        Scheduled Sampling：以一定概率将训练输入 token 替换为随机值，
        减轻 exposure bias（训练/推理不一致）。
        """
        if self.scheduled_sampling_prob <= 0:
            return input_ids

        audio_mask = (
            (audio_labels != -100).any(dim=0)
            & (torch.rand(input_ids.size(1)) < self.scheduled_sampling_prob)
        )
        for i in range(8):
            input_ids[i] = torch.where(
                audio_mask,
                torch.randint(0, self.audio_vocab_size, input_ids[i].shape),
                input_ids[i]
            )

        text_mask = (
            (text_labels != -100)
            & (input_ids[8] != self.image_token_id)
            & (torch.rand(input_ids.size(1)) < self.scheduled_sampling_prob)
        )
        input_ids[8] = torch.where(
            text_mask,
            torch.randint(0, self.text_vocab_size, input_ids[8].shape),
            input_ids[8]
        )
        return input_ids

    def __getitem__(self, index: int):
        """
        核心数据获取逻辑。返回:
          input_ids: (9, T) = 8路audio + 1路text
          text_labels: (T,)、audio_labels: (8, T)、audio_inputs、audio_len、pixel_values、spk_emb
        """
        conversations = json.loads(self.table['conversations'][index].as_py())
        question_audios = self._get_col(index, 'question_audios', [])
        answer_audios = self._get_col(index, 'answer_audios', [])
        image_bytes = self._get_col(index, 'image_bytes', [])
        if image_bytes and not isinstance(image_bytes, list):
            image_bytes = [image_bytes]
        ref_audios = self._get_col(index, 'ref_audios', [])
        spk_emb_raw = self._get_col(index, 'spk_emb', [])

        # ── 多轮对话截断：随机选一个 assistant 轮次（避免超出 max_length）──
        asst_indices = [i for i, t in enumerate(conversations) if t['role'] == 'assistant']
        if len(asst_indices) > 1:
            rand_idx = random.randint(0, len(asst_indices) - 1)
            for i in range(rand_idx, -1, -1):
                conversations = conversations[:asst_indices[i] + 1]
                test_prompt = self.create_chat_prompt(conversations, 0)
                if len(self.tokenizer(test_prompt).input_ids) + 100 < self.max_length:
                    break

        # ── 加载图像 ──
        pixel_values = None
        if image_bytes and len(image_bytes) > 0 and self.vision_processor:
            pixel_values = self.load_image_inputs(image_bytes[0])

        # ── 加载用户音频 ──
        audio_inputs = None
        audio_len = 0
        audio_features_length = 0
        user_count = sum(1 for t in conversations if t['role'] == 'user')
        if question_audios and user_count > 0 and user_count <= len(question_audios) and self.audio_processor:
            audio_bytes = question_audios[user_count - 1]
            if audio_bytes:
                mel, valid_len = self.load_audio_inputs(audio_bytes)
                if mel is not None:
                    audio_inputs = mel.unsqueeze(0)
                    audio_len = valid_len
                    audio_features_length = valid_len or 1

        # ── 兜底：空输入占位 ──
        if audio_inputs is None and self.audio_processor:
            audio_inputs = torch.zeros(1, 1, 560)
            audio_len = 0
        if pixel_values is None and self.vision_processor:
            pixel_values = {'pixel_values': torch.zeros(1, 3, 256, 256)}

        # ── 加载 assistant 音频 codec（作为训练目标）──
        last_audio_codes = None
        asst_count = sum(1 for t in conversations if t['role'] == 'assistant')
        if answer_audios and asst_count > 0 and asst_count <= len(answer_audios):
            tokens = answer_audios[asst_count - 1]
            if tokens:
                last_audio_codes = self._parse_audio_codec(tokens, self.audio_stop_token)

        # ── 生成 text input_ids ──
        prompt = self.create_chat_prompt(conversations, audio_features_length)
        if pixel_values is not None:
            prompt = prompt.replace('<image>', self.image_token)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # ── 生成 text labels（仅最后一个 assistant 区域有效）──
        text_labels, assistant_ranges = self.generate_text_labels(input_ids)
        for start, end in assistant_ranges[:-1]:
            mask_end = min(end + len(self.eos_id), self.max_length)
            text_labels[start:mask_end] = [-100] * (mask_end - start)

        # ── 生成 8 层 audio targets ──
        Y_audio_layers, audio_labels = self._build_audio_labels(
            assistant_ranges, last_audio_codes, input_ids, spk_emb_raw, ref_audios
        )

        # ── 构造 9 路输入：(9, T) = 8路 audio codec + 1路 text ──
        X_audio = torch.tensor([layer[:-1] for layer in Y_audio_layers], dtype=torch.long)
        X_text = torch.tensor(input_ids[:-1], dtype=torch.long)
        input_ids = torch.cat((X_audio, X_text.unsqueeze(0)), dim=0)

        text_labels = torch.tensor(text_labels[1:], dtype=torch.long)
        audio_labels = torch.tensor([layer[1:] for layer in audio_labels], dtype=torch.long)
        input_ids = self.apply_scheduled_sampling(input_ids, audio_labels, text_labels)

        spk_emb = torch.tensor(spk_emb_raw, dtype=torch.float32) if spk_emb_raw else torch.zeros(192)
        return input_ids, text_labels, audio_labels, audio_inputs, audio_len, pixel_values, spk_emb


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 4: 训练工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def is_main_process():
    """判断当前是否为主进程（分布式训练中的 rank 0）。"""
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    """主进程日志输出。"""
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    余弦退火学习率调度。
    lr * (0.1 + 0.45 * (1 + cos(pi * t / T)))
    """
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    """
    初始化分布式训练环境（NCCL），返回 local_rank。
    非分布式环境返回 0。
    """
    if int(os.environ.get("RANK", -1)) == -1:
        return 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    """全局随机种子固定（确保可复现）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_model_params(model, ignore_patterns=None):
    """
    打印模型参数量（百万），支持 MoE 的激活参数/总参数区分。
    ignore_patterns: 不计入统计的层名模式（如冻结编码器）。
    """
    if ignore_patterns is None:
        ignore_patterns = ['audio_encoder', 'vision_encoder']

    def should_count(n):
        return not any(p in n for p in ignore_patterns)

    total = sum(p.numel() for n, p in model.named_parameters() if should_count(n)) / 1e6
    cfg = model.config
    n_routed = getattr(cfg, 'n_routed_experts', getattr(cfg, 'num_experts', 0))
    n_active = getattr(cfg, 'num_experts_per_tok', 0)
    n_shared = getattr(cfg, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters()
                 if 'mlp.experts.0.' in n and should_count(n)) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters()
                        if 'mlp.shared_experts.0.' in n and should_count(n)) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total:
        Logger(f'Model Params: {total:.2f}M (Active: {active:.2f}M)')
    else:
        Logger(f'Model Params: {total:.2f}M')


def init_omni_model(omni_config, from_weight='full_sft', tokenizer_path='../model',
                    audio_encoder_path='../model/SenseVoiceSmall',
                    vision_model_path='../model/siglip2-base-p32-256-ve',
                    save_dir='../out', checkpoint_dir='../checkpoints',
                    device='cuda', freeze_backbone='none', from_resume=0):
    """
    初始化 PigCPMOmni 模型 + tokenizer。
    - from_weight: 预训练权重名（none 为随机初始化）
    - freeze_backbone: 'all'（冻结所有）、'last1'（解冻最后一层）、'none'
    - from_resume: 断点续训步数（0 表示从头）
    """
    tokenizer = AutoTokenizer.from_pretrained(os.path.abspath(tokenizer_path))
    model = PigCPMOmni(omni_config, audio_encoder_path=audio_encoder_path,
                       vision_model_path=vision_model_path)

    if from_weight != 'none':
        moe_suffix = '_moe' if omni_config.use_moe else ''
        # 优先使用 checkpoint_dir，其次 save_dir
        search_dirs = [checkpoint_dir, save_dir] if checkpoint_dir != save_dir else [save_dir]
        weight_path = None
        for d in search_dirs:
            candidate = f'{d}/{from_weight}_{omni_config.hidden_size}{moe_suffix}.pth'
            if os.path.exists(candidate):
                weight_path = candidate
                break
            candidate = f'{d}/{from_weight}_{omni_config.hidden_size}.pth'
            if os.path.exists(candidate):
                weight_path = candidate
                break
        if weight_path is not None:
            weights = torch.load(weight_path, map_location=device)
            param_shapes = {k: v.shape for k, v in model.named_parameters()}
            incompatible = {
                k for k, v in weights.items()
                if k in param_shapes and v.shape != param_shapes[k]
            }
            if incompatible:
                Logger(f'跳过shape不匹配的权重: {incompatible}')
                weights = {k: v for k, v in weights.items() if k not in incompatible}
            model.load_state_dict(weights, strict=False)
            Logger(f'已加载权重: {weight_path}')
            # ── Talker 层初始化：从 Thinker 尾部复制权重 ──
            if from_resume == 0 and omni_config.talker_hidden_size == omni_config.hidden_size:
                n_talker = omni_config.num_talker_hidden_layers
                n_thinker = len(model.thinker.layers)
                has_talker = any(k.startswith('talker.layers.') for k in weights)
                if not has_talker and n_talker > 0:
                    for i in range(n_talker):
                        src = n_thinker - n_talker + i
                        model.talker.layers[i].load_state_dict(
                            model.thinker.layers[src].state_dict()
                        )
                    Logger(f'Talker层初始化: 复制thinker layers[{n_thinker - n_talker}:{n_thinker}] '
                           f'→ talker layers[0:{n_talker}]')

    # ── 冻结策略 ──
    if freeze_backbone == 'all':
        for param in model.thinker.parameters():
            param.requires_grad = False
    elif freeze_backbone == 'last1':
        for param in model.thinker.parameters():
            param.requires_grad = False
        if hasattr(model.thinker, 'layers') and len(model.thinker.layers) > 0:
            for param in model.thinker.layers[-1].parameters():
                param.requires_grad = True

    return model.to(device), tokenizer


def omni_checkpoint(omni_config, weight='pretrain_omni', model=None, optimizer=None,
                    epoch=0, step=0, swanlab_run=None, save_dir='../checkpoints', **kwargs):
    """
    通用 Checkpoint 保存/加载。
    - 保存 (model is not None): 存 clean_state_dict + optimizer/swanlab 状态
    - 加载 (model is None): 返回 resume_data，自动处理 GPU 数量变化
    """
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if omni_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{omni_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{omni_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        # ── 保存权重 ──
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)  # torch.compile 兼容
        clean_state_dict = {
            k: v for k, v in raw_model.state_dict().items()
            if not k.startswith('audio_encoder.') and not k.startswith('vision_encoder.')
        }
        state_dict = {k: v.half().cpu() for k, v in clean_state_dict.items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        # ── 保存断点续训数据 ──
        swanlab_id = None
        if swanlab_run is not None:
            swanlab_id = getattr(swanlab_run, 'public', None)
            if swanlab_id is not None:
                swanlab_id = getattr(swanlab_id, 'run_id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'swanlab_id': swanlab_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    if isinstance(value, DistributedDataParallel):
                        resume_data[key] = value.module.state_dict()
                    else:
                        resume_data[key] = value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
    else:
        # ── 加载 ──
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws} → {current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


class SkipBatchSampler(Sampler):
    """
    带跳过批次的采样器包装器（用于断点续训时跳过已训练的 batch）。
    """
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 5: 训练主逻辑 —— omni_collate_fn, train_epoch, main
# ═══════════════════════════════════════════════════════════════════════════════

def omni_collate_fn(batch):
    """
    自定义 collate 函数：处理变长 audio_inputs 和 pixel_values。
    - 变长 fbank 特征 padding 到 batch 内最大长度
    - 多格式 image 输入的自动聚合
    """
    input_ids, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb = zip(*batch)

    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)
    audio_labels = torch.stack(audio_labels)
    audio_lens = torch.tensor(audio_lens, dtype=torch.long)

    # ── 音频 padding ──
    valid_audios = [a for a in audio_inputs if a is not None]
    if valid_audios:
        max_t = max(a.size(1) for a in valid_audios)
        padded = [
            a if a.size(1) == max_t
            else F.pad(a, (0, 0, 0, max_t - a.size(1)))
            for a in valid_audios
        ]
        audio_inputs = torch.cat(padded, dim=0)
    else:
        audio_inputs = None

    # ── 图像聚合 ──
    valid_images = [p for p in pixel_values if p is not None]
    if valid_images:
        if hasattr(valid_images[0], 'keys'):
            keys = set.intersection(*[set(d.keys()) for d in valid_images])
            pixel_values = {k: torch.cat([d[k] for d in valid_images], dim=0) for k in keys}
        else:
            pixel_values = torch.cat(valid_images, dim=0)
    else:
        pixel_values = None

    spk_emb = torch.stack(spk_emb)
    return input_ids, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb


def train_epoch(model, train_loader, optimizer, epoch, device, local_rank, args, swanlab_run=None):
    """
    单轮训练循环，包含：
    - 文本 & 音频双 Loss 计算
    - 梯度累积
    - 分布式梯度同步
    - 日志/checkpoint 输出
    - Swanlab 可视化管理
    """
    model.train()
    total_loss = 0.0
    total_text_loss = 0.0
    total_audio_loss = 0.0
    start_time = time.time()

    for step, batch in enumerate(train_loader):
        input_ids, text_labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb = batch
        input_ids = input_ids.to(device)
        text_labels = text_labels.to(device)
        audio_labels = audio_labels.to(device)
        audio_inputs = audio_inputs.to(device) if audio_inputs is not None else None
        audio_lens = audio_lens.to(device)
        pixel_values = (
            {k: v.to(device) for k, v in pixel_values.items()}
            if isinstance(pixel_values, dict) else
            pixel_values.to(device) if pixel_values is not None else None
        )
        spk_emb = spk_emb.to(device)

        with (torch.amp.autocast('cuda') if args.use_amp else nullcontext()):
            outputs = model(
                input_ids,
                audio_inputs=audio_inputs,
                audio_lens=audio_lens,
                pixel_values=pixel_values,
                spk_emb=spk_emb
            )
            text_logits = outputs.logits
            audio_logits = outputs.audio_logits

            # ── 文本 Loss ──
            text_loss = F.cross_entropy(
                text_logits.view(-1, text_logits.size(-1)),
                text_labels.view(-1),
                ignore_index=-100
            )

            # ── 音频 Loss（8路交叉熵求和）──
            audio_loss = 0.0
            for i in range(8):
                audio_loss += F.cross_entropy(
                    audio_logits[i].view(-1, audio_logits[i].size(-1)),
                    audio_labels[:, i, :].view(-1),
                    ignore_index=-100
                )

            loss = text_loss + audio_loss
            if outputs.aux_loss is not None:
                loss = loss + outputs.aux_loss

        # ── 梯度累积 ──
        loss = loss / args.gradient_accumulation_steps
        loss.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        total_loss += loss.item()
        total_text_loss += text_loss.item() / args.gradient_accumulation_steps
        total_audio_loss += audio_loss.item() / args.gradient_accumulation_steps

        # ── 日志 ──
        if is_main_process() and step % args.log_interval == 0:
            elapsed = time.time() - start_time
            Logger(
                f'Epoch {epoch} | Step {step}/{len(train_loader)} | '
                f'Loss: {loss.item():.4f} | Text: {text_loss.item() / args.gradient_accumulation_steps:.4f} | '
                f'Audio: {audio_loss.item() / args.gradient_accumulation_steps:.4f} | '
                f'Time: {elapsed:.1f}s'
            )

        # ── 检查是否保存 checkpoint ──
        # （原代码中通过外部调度控制，此处保留占位）
        global_step = (epoch - 1) * len(train_loader) + step + 1

    return total_loss / len(train_loader), total_text_loss / len(train_loader), total_audio_loss / len(train_loader)


# ═══════════════════════════════════════════════════════════════════════════════
#                              Main 训练入口
# ═══════════════════════════════════════════════════════════════════════════════

def _init_swanlab(args, resume_data=None):
    """初始化 Swanlab 可视化实验跟踪。"""
    if not is_main_process() or not args.swanlab_project:
        return None
    try:
        import swanlab
        run_id = resume_data.get('swanlab_id') if resume_data else None
        return swanlab.init(
            project=args.swanlab_project,
            experiment_name=args.swanlab_name or args.checkpoint_name,
            resume='allow' if run_id else 'never',
            id=run_id
        )
    except ImportError:
        Logger('Swanlab 未安装，跳过可视化管理')
        return None


def _create_dataloader(args, model, tokenizer, skip_batches=0):
    """创建多模态训练 DataLoader。"""
    train_dataset = OmniDataset(
        args.data_path, tokenizer,
        audio_processor=model.audio_processor,
        vision_processor=model.vision_processor,
        max_length=args.max_length,
        scheduled_sampling=args.scheduled_sampling,
        image_token_len=args.image_token_len
    )
    sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
    train_sampler = SkipBatchSampler(
        sampler or range(len(train_dataset)),
        args.batch_size,
        skip_batches=skip_batches
    )
    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler,
        collate_fn=omni_collate_fn, num_workers=args.num_workers,
        pin_memory=True, drop_last=True
    )
    return train_dataset, sampler, train_loader


def main():
    """PigCPM-O 多模态训练主入口。"""
    parser = argparse.ArgumentParser(description="PigCPM-O 多模态训练脚本")
    parser.add_argument("--data_path", type=str, default='./data/train.parquet', help="训练数据路径")
    parser.add_argument("--checkpoint_dir", type=str, default='../checkpoints', help="checkpoint保存目录")
    parser.add_argument("--checkpoint_name", type=str, default='pigcpm_o', help="checkpoint名称")
    parser.add_argument("--save_dir", type=str, default='../out', help="权重保存目录")
    parser.add_argument("--tokenizer_path", type=str, default='../model', help="Tokenizer路径")
    parser.add_argument("--audio_encoder_path", type=str, default='../model/SenseVoiceSmall', help="SenseVoice路径")
    parser.add_argument("--vision_model_path", type=str, default='../model/siglip2-base-p32-256-ve', help="SigLIP路径")
    parser.add_argument("--from_weight", type=str, default='full_sft', help="加载预训练权重名")
    parser.add_argument("--from_resume", type=int, default=0, help="断点续训步数")
    parser.add_argument("--freeze_backbone", type=str, default='none', choices=['none', 'all', 'last1'], help="冻结策略")

    # ── 模型参数 ──
    parser.add_argument("--hidden_size", type=int, default=768, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", type=int, default=8, help="Transformer层数")
    parser.add_argument("--num_talker_hidden_layers", type=int, default=4, help="Talker层数")
    parser.add_argument("--talker_hidden_size", type=int, default=768, help="Talker隐藏维度")
    parser.add_argument("--use_moe", action='store_true', default=False, help="启用MoE")
    parser.add_argument("--flash_attn", action='store_true', default=True, help="启用Flash Attention")
    parser.add_argument("--vocab_size", type=int, default=6400, help="词表大小")
    parser.add_argument("--max_length", type=int, default=1200, help="最大序列长度")
    parser.add_argument("--image_token_len", type=int, default=64, help="图片token数")

    # ── 训练参数 ──
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="每GPU batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="学习率")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="最小学习率")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="权重衰减")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--warmup_steps", type=int, default=500, help="warmup步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--use_amp", action='store_true', default=True, help="启用混合精度")
    parser.add_argument("--log_interval", type=int, default=10, help="日志间隔（步）")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="checkpoint间隔（步）")
    parser.add_argument("--scheduled_sampling", type=float, default=0.05, help="Scheduled Sampling概率")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker数")

    # ── Swanlab ──
    parser.add_argument("--swanlab_project", type=str, default='pigcpm', help="Swanlab项目名")
    parser.add_argument("--swanlab_name", type=str, default=None, help="Swanlab实验名称")

    args = parser.parse_args()

    # ── 分布式初始化 ──
    local_rank = init_distributed_mode()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    setup_seed(args.seed)

    # ── 模型配置 ──
    omni_config = OmniConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_talker_hidden_layers=args.num_talker_hidden_layers,
        talker_hidden_size=args.talker_hidden_size,
        use_moe=args.use_moe,
        flash_attn=args.flash_attn,
        vocab_size=args.vocab_size,
        max_position_embeddings=args.max_length * 2,
        image_token_len=args.image_token_len,
    )

    # ── 初始化模型 ──
    model, tokenizer = init_omni_model(
        omni_config, from_weight=args.from_weight,
        tokenizer_path=args.tokenizer_path,
        audio_encoder_path=args.audio_encoder_path,
        vision_model_path=args.vision_model_path,
        save_dir=args.save_dir,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        freeze_backbone=args.freeze_backbone,
        from_resume=args.from_resume
    )
    log_model_params(model)

    # ── 加载断点续训状态 ──
    resume_data = omni_checkpoint(
        omni_config, weight=args.checkpoint_name, model=None,
        step=args.from_resume, save_dir=args.checkpoint_dir
    )
    start_epoch = 1
    global_step = 0
    if args.from_resume > 0 and resume_data is not None:
        start_epoch = resume_data['epoch']
        global_step = resume_data['step']
        Logger(f'断点续训: epoch {start_epoch}, step {global_step}')

    # ── Swanlab ──
    swanlab_run = _init_swanlab(args, resume_data)

    # ── 数据集 ──
    train_dataset, sampler, train_loader = _create_dataloader(
        args, model, tokenizer,
        skip_batches=global_step // args.gradient_accumulation_steps
    )

    # ── 优化器 ──
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    if resume_data is not None and 'optimizer' in resume_data:
        optimizer.load_state_dict(resume_data['optimizer'])
        Logger('已恢复优化器状态')

    # ── 学习率调度器 ──
    total_steps = len(train_loader) * args.epochs
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: get_lr(step, total_steps, args.learning_rate)
    )

    # ── DDP 包装 ──
    if dist.is_initialized():
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True
        )
    else:
        model = model.to(device)

    # ── 训练循环 ──
    Logger(f'开始训练: {args.epochs} epochs, {len(train_loader)} steps/epoch')
    for epoch in range(start_epoch, args.epochs + 1):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        train_loss, text_loss, audio_loss = train_epoch(
            model, train_loader, optimizer, epoch, device, local_rank, args, swanlab_run
        )

        Logger(f'Epoch {epoch} 完成 | Avg Loss: {train_loss:.4f} | '
               f'Text: {text_loss:.4f} | Audio: {audio_loss:.4f}')

        # ── Epoch 级 checkpoint ──
        omni_checkpoint(
            omni_config, weight=args.checkpoint_name, model=model,
            optimizer=optimizer, epoch=epoch + 1, step=global_step,
            swanlab_run=swanlab_run, save_dir=args.checkpoint_dir
        )

    # ── 训练完成，同步权重到 out 目录 ──
    moe_suffix = '_moe' if args.use_moe else ''
    src = f'{args.checkpoint_dir}/{args.checkpoint_name}_{args.hidden_size}{moe_suffix}.pth'
    dst = f'{args.save_dir}/{args.checkpoint_name}_{args.hidden_size}{moe_suffix}.pth'
    if os.path.exists(src):
        import shutil
        os.makedirs(args.save_dir, exist_ok=True)
        shutil.copy2(src, dst)
        Logger(f'权重已同步至: {dst}')

    Logger('训练完成!')
    if swanlab_run is not None:
        import swanlab
        swanlab.finish()


if __name__ == "__main__":
    main()
