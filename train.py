import os
import math
import torch
import random
import swanlab
import argparse
import numpy as np
import torch.nn.functional as F

from torch import nn
from contextlib import nullcontext
from transformers.activations import ACT2FN
from transformers import AutoTokenizer, GenerationMixin, PretrainedConfig, PreTrainedModel


class PigCPMConfig(PretrainedConfig):
    model_type: str = "pigcpm"

    def __init__(self, use_moe: bool = False, hidden_size: int = 768, hidden_layers: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.use_moe = use_moe
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers

        self.dropout = kwargs.get("dropout", 0.0)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
    
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.attention_heads)
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)

        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None

        # MoE模型配置
        self.num_experts = kwargs.get("num_experts", 4)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)


class OmniConfig(PigCPMConfig):
    model_type = "pigcpm-o"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.audio_ids = kwargs.get("audio_ids", [16])
        self.image_ids = kwargs.get("image_ids", [12])
        self.spk_emb_size = kwargs.get("spk_emb_size", 192)
        self.image_token_len = kwargs.get("image_token_len", 64)
        self.audio_pad_token = kwargs.get("audio_pad_token", 2049)
        self.audio_spk_token = kwargs.get("audio_spk_token", 2051)
        self.audio_stop_token = kwargs.get("audio_stop_token", 2050)
        self.audio_vocab_size = kwargs.get("audio_vocab_size", 2112)
        self.audio_hidden_size = kwargs.get("audio_hidden_size", 512)
        self.image_hidden_size = kwargs.get("image_hidden_size", 768)
        self.talker_hidden_size = kwargs.get("talker_hidden_size", 768)
        self.think_end_ids = kwargs.get("think_end_ids", [26, 234, 234])
        self.num_talker_hidden_layers = kwargs.get("num_talker_hidden_layers", 4)
        self.audio_special_token = kwargs.get("audio_special_token", "<|audio_pad|>")
        self.image_special_token = kwargs.get("image_special_token", "<|image_pad|>")
        self.bridge_layer = kwargs.get("bridge_layer", self.num_hidden_layers // 2 - 1)


class RMSNorm(torch.nn.Module):

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim = -1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed


class Attention(nn.Module):

    def __init__(self, config: PigCPMConfig):
        super().__init__()
        self.is_causal = True
        self.head_dim = config.head_dim
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        if config.num_key_value_heads is None:
            self.num_key_value_heads = config.num_attention_heads
        else:
            self.num_key_value_heads = config.num_key_value_heads

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.dropout = config.dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):

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

    def __init__(self, config: PigCPMConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)


class PigCPMBlock(nn.Module):
    
    def __init__(self, layer_id: int, config: PigCPMConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class PigCPMModel(nn.Module):

    def __init__(self, config: PigCPMConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([PigCPMBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss


class PigCPMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = PigCPMConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: PigCPMConfig = None):
        self.config = config or PigCPMConfig()
        super().__init__(self.config)
        self.model = PigCPMModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
    
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            if top_k > 0: 
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids


class PigCPMOmni(PigCPMForCausalLM):
    config_class = OmniConfig

    def __init__(self, config: OmniConfig = None, audio_encoder_path="./model/SenseVoiceSmall", vision_model_path="./model/siglip2-base-p32-256-ve"):
        config = config or OmniConfig()
        super().__init__(config)
        object.__setattr__(self, 'thinker', self.model)
        object.__setattr__(self.model, 'lm_head', self.lm_head)
        self.talker = TalkerModule(config)
        self.audio_proj = MMAudioProjector(config.audio_hidden_size, config.hidden_size)
        self.vision_proj = MMVisionProjector(config.image_hidden_size, config.hidden_size, target_tokens=config.image_token_len)
        self.audio_pad_token, self.audio_stop_token, self.audio_spk_token = config.audio_pad_token, config.audio_stop_token, config.audio_spk_token
        audio_encoder, audio_processor = self.load_sensevoice(audio_encoder_path)
        object.__setattr__(self, 'audio_encoder', audio_encoder)
        object.__setattr__(self, 'audio_processor', audio_processor)
        vision_encoder, vision_processor = self.load_vision(vision_model_path)
        object.__setattr__(self, 'vision_encoder', vision_encoder)
        object.__setattr__(self, 'vision_processor', vision_processor)

    @staticmethod
    def load_sensevoice(path):
        if not os.path.exists(path):
            warnings.warn(f"[PigCPMOmni] SenseVoice path not found: {path}")
            return None, None
        logging.getLogger().setLevel(logging.ERROR)
        hf_logging.set_verbosity_error()
        with contextlib.redirect_stdout(io.StringIO()):
            from funasr import AutoModel
            m = AutoModel(model=path, trust_remote_code=True, disable_update=True, device="cpu")
        encoder, frontend = m.model.encoder, m.kwargs["frontend"]
        for p in encoder.parameters(): p.requires_grad = False
        return encoder.eval().float(), SenseVoiceAudioProcessor(frontend.eval())

    @torch.compiler.disable
    def encode_audio_inputs(self, audio_inputs, audio_lens=None):
        if (audio_inputs is None) or (self.audio_encoder is None) or (not audio_inputs.any()): return None
        batch_mask = audio_inputs.flatten(1).any(1)
        enc_dtype = next(self.audio_encoder.parameters()).dtype
        valid_fbank = audio_inputs[batch_mask].to(dtype=enc_dtype)
        if audio_lens is not None:
            valid_lens = audio_lens[batch_mask].to(valid_fbank.device)
        else:
            valid_lens = torch.tensor([valid_fbank.size(1)] * valid_fbank.size(0), device=valid_fbank.device)
        with torch.no_grad():
            emb, _ = self.audio_encoder(valid_fbank, valid_lens)
        proj_dtype = next(self.audio_proj.parameters()).dtype
        emb_list = [self.audio_proj(emb[i, :max(1, min(valid_lens[i].item(), emb.size(1)))].unsqueeze(0).to(proj_dtype)).squeeze(0) for i in range(emb.size(0))]
        if batch_mask.all(): return emb_list
        out = [None] * audio_inputs.size(0)
        j = 0
        for i in range(audio_inputs.size(0)):
            if batch_mask[i]:
                out[i] = emb_list[j]
                j += 1
        return out

    @torch.compiler.disable
    def inject_audio_features(self, tokens, h, audio_feats, seqlen):
        if audio_feats is None or not self.config.audio_ids:
            return h
        marker = self.config.audio_ids[0]
        out = []
        for b in range(h.size(0)):
            hb, seq, i = h[b], tokens[b].tolist(), 0
            af = audio_feats[b] if audio_feats[b] is not None else None
            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    while i < len(seq) and seq[i] == marker:
                        i += 1
                    if af is not None:
                        inject_len = min(af.size(0), i - start)
                        hb = torch.cat((hb[:start], af[:inject_len], hb[start + inject_len:]), dim=0)
                        af = None
                else:
                    i += 1
            out.append(hb)
        return torch.stack(out)
    
    @staticmethod
    def load_vision(path):
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

    @torch.compiler.disable
    def get_image_embeddings(self, image_inputs):
        if hasattr(image_inputs, 'keys'):
            image_inputs = {k: v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v for k, v in image_inputs.items()}
            pixel_attention_mask = image_inputs.get('pixel_attention_mask')
            if pixel_attention_mask is not None and not pixel_attention_mask.any():
                pv = image_inputs['pixel_values']
                return pv.new_zeros(pv.size(0), pv.size(1), self.config.image_hidden_size)
        with torch.no_grad():
            outputs = self.vision_encoder(**image_inputs)
        return outputs.last_hidden_state

    @torch.compiler.disable
    def encode_image_inputs(self, pixel_values):
        if pixel_values is None or self.vision_encoder is None: return None
        mask = pixel_values.flatten(1).any(1)
        if not mask.any(): return pixel_values.new_zeros(pixel_values.size(0), self.config.image_token_len, self.config.hidden_size)
        with torch.no_grad(): emb = self.vision_encoder(pixel_values=pixel_values[mask]).last_hidden_state
        if emb.dim() == 2: emb = emb.unsqueeze(0)
        emb = self.vision_proj(emb)
        if mask.all(): return emb
        idx = mask.nonzero().view(-1, 1, 1).expand_as(emb)
        return emb.new_zeros(pixel_values.size(0), *emb.shape[1:]).scatter(0, idx, emb)

    @torch.compiler.disable
    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
        if vision_tensors is None or not self.config.image_ids:
            return h
        marker, vf = self.config.image_ids[0], vision_tensors
        if vf.dim() == 3:
            vf = vf.unsqueeze(1)
        out = []
        for b in range(h.size(0)):
            hb, seq, k, i = h[b], tokens[b].tolist(), 0, 0
            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    while i < len(seq) and seq[i] == marker:
                        i += 1
                    if k < vf.size(1):
                        hb = torch.cat((hb[:start], vf[b][k][:i - start], hb[i:]), dim=0)[:seqlen]
                        k += 1
                else:
                    i += 1
            out.append(hb)
        return torch.stack(out)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, audio_inputs=None, audio_lens=None, pixel_values=None, **args):
        if len(input_ids.shape) == 2:
            batch_size, seq_length = input_ids.shape
            text_ids = input_ids
            audio_ids = torch.full((batch_size, 8, seq_length), self.audio_pad_token, dtype=torch.long, device=input_ids.device)
        else:
            batch_size, _, seq_length = input_ids.shape
            text_ids, audio_ids = input_ids[:, 8, :], input_ids[:, :8, :]
        if hasattr(past_key_values, 'layers'): past_key_values = None
        n_thinker, n_talker = len(self.thinker.layers), len(self.talker.layers)
        past_key_values = past_key_values or ([None] * (n_thinker + n_talker))
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.thinker.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.thinker.freqs_cos, self.thinker.freqs_sin = freqs_cos.to(input_ids.device), freqs_sin.to(input_ids.device)
        if self.talker.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.talker.talker_config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.talker.freqs_cos, self.talker.freqs_sin = freqs_cos.to(input_ids.device), freqs_sin.to(input_ids.device)
        presents = []

        # ======= Thinker: text-only input, output text logits =======
        hidden_states = self.thinker.dropout(self.thinker.embed_tokens(text_ids))
        position_embeddings = (self.thinker.freqs_cos[start_pos:start_pos + seq_length], self.thinker.freqs_sin[start_pos:start_pos + seq_length])
        if audio_inputs is not None and start_pos == 0:
            audio_features = self.encode_audio_inputs(audio_inputs, audio_lens)
            hidden_states = self.inject_audio_features(text_ids, hidden_states, audio_features, seq_length)
        if pixel_values is not None and start_pos == 0:
            if hasattr(pixel_values, 'keys'):
                img_emb = self.get_image_embeddings(pixel_values).to(hidden_states.dtype)
                vision_tensors = self.vision_proj(img_emb)
            else:
                if len(pixel_values.shape) == 6:
                    pixel_values = pixel_values.squeeze(2)
                if len(pixel_values.shape) == 4:
                    pixel_values = pixel_values.unsqueeze(1)
                bs, num, c, im_h, im_w = pixel_values.shape
                stack_dim = 1 if bs > 1 else 0
                vision_tensors = torch.stack([
                    self.encode_image_inputs(pixel_values[:, i, :, :, :])
                    for i in range(num)
                ], dim=stack_dim)
            hidden_states = self.count_vision_proj(tokens=text_ids, h=hidden_states, vision_tensors=vision_tensors, seqlen=seq_length)
        bridge_states = hidden_states
        for i, (layer, past_key_value) in enumerate(zip(self.thinker.layers, past_key_values[:n_thinker])):
            hidden_states, present = layer(hidden_states, position_embeddings, past_key_value=past_key_value, use_cache=use_cache, attention_mask=attention_mask)
            presents.append(present)
            if i == self.config.bridge_layer: bridge_states = hidden_states
        h_thinker = self.thinker.norm(hidden_states)

        # ======= Talker: thinker hidden + audio codes, output audio logits =======
        talker_emb = self.talker.embed_tokens(audio_ids)
        spk_emb = args.get('spk_emb', None)
        if spk_emb is not None:
            spk_mask = (audio_ids[:, 0, :] == self.audio_spk_token).unsqueeze(-1)
            talker_emb = torch.where(spk_mask, self.talker.spk_proj(spk_emb).unsqueeze(1), talker_emb)
        hidden_states = self.talker.embed_proj(bridge_states) * self.talker.text_scale + self.talker.codec_proj(talker_emb) * self.talker.audio_scale
        talker_pos_emb = (self.talker.freqs_cos[start_pos:start_pos + seq_length], self.talker.freqs_sin[start_pos:start_pos + seq_length])
        for layer, past_key_value in zip(self.talker.layers, past_key_values[n_thinker:]):
            hidden_states, present = layer(hidden_states, talker_pos_emb, past_key_value=past_key_value, use_cache=use_cache, attention_mask=attention_mask)
            presents.append(present)
        h_talker = self.talker.norm(hidden_states)

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        aux_loss = sum(l.mlp.aux_loss for l in list(self.thinker.layers) + list(self.talker.layers) if isinstance(l.mlp, MOEFeedForward))
        aux_loss += sum(p.sum() for p in self.audio_proj.parameters()) * 0 + sum(p.sum() for p in self.vision_proj.parameters()) * 0 + sum(p.sum() for p in self.talker.lm_head.adapters.parameters()) * 0 + sum(p.sum() for p in self.talker.spk_proj.parameters()) * 0 # dummy gradient
        text_logits = self.thinker.lm_head(h_thinker[:, slice_indices, :])
        audio_logits = self.talker.lm_head(h_talker[:, slice_indices, :])
        
        out = MoeCausalLMOutputWithPast(aux_loss=aux_loss, logits=text_logits, past_key_values=presents)
        out.audio_logits = audio_logits
        return out

    @torch.inference_mode()
    def generate(self, input_ids, eos_token_id=2, max_new_tokens=1024, temperature=0.75, top_p=0.90,
                 stream=False, rp=1., use_cache=True, return_audio_codes=False, **args):
        if stream:
            return self.stream_generate(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, return_audio_codes, **args)
        tokens = list(self.stream_generate(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, return_audio_codes, **args))
        return tokens[-1] if tokens else input_ids

    def stream_generate(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, return_audio_codes=False, **args):
        start_pos, past_kvs, text_finished, first_finished = input_ids.shape[1], None, False, True
        audio_codes = [[] for _ in range(8)]
        audio_stop_pos = [None] * 8
        audio_buffer = torch.full((1, 8, start_pos), self.audio_pad_token, dtype=torch.long, device=input_ids.device)
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
        think_end_step, generated_tokens = None, ([] if args.get('open_thinking', False) else None)
        while input_ids.shape[1] < start_pos + max_new_tokens:
            if past_kvs is None or not use_cache:
                out = self.forward(torch.cat((audio_buffer, input_ids.unsqueeze(1)), dim=1), past_key_values=past_kvs, use_cache=use_cache, **args)
            else:
                out = self.forward(torch.cat((audio_buffer[:, :, -1:], input_ids[:, -1:].unsqueeze(1)), dim=1), past_key_values=past_kvs, use_cache=use_cache, **args)
            past_kvs = out.past_key_values

            logits = out.logits[0, -1, :].clone() / (temperature + 1e-9)
            if rp != 1.0:
                seen = list(set(input_ids[0].tolist())); score = logits[seen]; logits[seen] = torch.where(score > 0, score / rp, score * rp)
            if top_p and top_p < 1.0:
                sorted_l, sorted_i = torch.sort(logits, descending=True)
                mask = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1) > top_p
                mask[1:], mask[0] = mask[:-1].clone(), False
                logits[sorted_i[mask]] = -float('Inf')
            text_token = torch.multinomial(F.softmax(logits, dim=-1), 1).item()

            if text_finished:
                text_token = args.get('enter_token_id', 201) if first_finished else args.get('pad_token_id', 0)
                first_finished = False

            step = input_ids.shape[1] - start_pos  # 已生成token数（0=首次，此时模型处理prompt末尾token）
            audio_step = step - 1  # 延迟1步：输出第1个text时无audio，输出第2个text时layer0开始
            if generated_tokens is not None:
                generated_tokens.append(text_token)
                if not think_end_step and generated_tokens[-len(self.config.think_end_ids):] == list(self.config.think_end_ids): think_end_step = step + 2
                audio_step = (step - think_end_step) if think_end_step else -1
            for i, al in enumerate(out.audio_logits):
                if audio_step < i:
                    audio_codes[i].append(self.audio_pad_token)
                else:
                    logits_i = al[0, -1, :].clone() / 0.2
                    for prev_code in audio_codes[i][-3:]: score = logits_i[prev_code]; logits_i[prev_code] = torch.where(score > 0, score / 1.05, score * 1.05)
                    top_val, top_idx = logits_i.topk(50)
                    code = top_idx[torch.multinomial(F.softmax(top_val, dim=-1), 1)].item()
                    audio_codes[i].append(code)
                    if audio_stop_pos[i] is None and code >= 2048: audio_stop_pos[i] = len(audio_codes[i]) - 1

            if text_finished and all(audio_stop_pos[i] is not None for i in range(8)): break

            input_ids = torch.cat((input_ids, torch.tensor([[text_token]], device=input_ids.device)), dim=1)
            audio_buffer = torch.cat((audio_buffer, torch.full((1, 8, 1), self.audio_pad_token, dtype=torch.long, device=input_ids.device)), dim=2)
            for i in range(min(audio_step + 1, 8)): audio_buffer[0, i, -1] = audio_codes[i][-1]

            audio_frame = None
            if return_audio_codes and audio_step >= 7:
                frame = [audio_codes[i][step - 7 + i] for i in range(8)]
                active_layers = sum(1 for i in range(8) if audio_stop_pos[i] is None or step - 7 + i < audio_stop_pos[i])
                if active_layers >= 8: audio_frame = frame
            if not text_finished:
                yield input_ids[:, start_pos:], audio_frame
                if text_token == eos_token_id: text_finished = True
            else:
                yield None, audio_frame


def omni_checkpoint(
    omni_config: OmniConfig,
    model_prefix: str = "pigcpm_omni",
    checkpoint_dir: str = "./checkpoints",
    model = None, optimizer = None, epoch: int = 0, step: int = 0, wandb=None, **kwargs):
    os.makedirs(checkpoint_dir, exist_ok=True)
    moe_tag = "_moe" if omni_config.use_moe else ""
    ckp_path = f"{checkpoint_dir}/{model_prefix}_{omni_config.hidden_size}{moe_tag}.pth"
    resume_path = f"{checkpoint_dir}/{model_prefix}_{omni_config.hidden_size}{moe_tag}_resume.pth"
    
    if model is not None:
        from torch.nn.parallel import DistributedDataParallel
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, "_orig_mod", raw_model)
        # 移除冻结的 audio_encoder / vision_encoder 参数（不需要保存，从预训练路径重新加载）
        clean_state_dict = {k: v for k, v in raw_model.state_dict().items() if not k.startswith("audio_encoder.") and not k.startswith("vision_encoder.")}
        state_dict = {k: v.half().cpu() for k, v in clean_state_dict.items()}
        ckp_tmp = f"{ckp_path}.tmp"
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        
        swanlab_id = None
        if wandb:
            if hasattr(wandb, "get_run"):
                run = wandb.get_run()
                swanlab_id = getattr(run, "id", None) if run else None
            else:
                swanlab_id = getattr(wandb, "id", None)
        
        resume_data = {
            "step": step,
            "epoch": epoch,
            "model": state_dict,
            "swanlab_id": swanlab_id,
            "optimizer": optimizer.state_dict(),
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, "state_dict"):
                    if isinstance(value, DistributedDataParallel):
                        resume_data[key] = value.module.state_dict()
                    else:
                        resume_data[key] = value.state_dict()
                else:
                    resume_data[key] = value
        
        resume_tmp = f"{resume_path}.tmp"
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location="cpu")
            saved_ws = ckp_data.get("world_size", 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data["step"] = ckp_data["step"] * saved_ws // current_ws
            return ckp_data
        return None


def init_omni_model(
    omni_config: OmniConfig,
    from_weight: str = "full_sft",
    tokenizer_path: str = "./model",
    model_dir: str = "./pigcpm_model",
    audio_model_path: str = "./model/SenseVoiceSmall",
    vision_model_path: str = "./model/siglip2-base-p32-256-ve",
    device: str = "cuda", freeze_backbone: str = "none", resume: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = PigCPMOmni(omni_config, audio_model_path=audio_model_path, vision_model_path=vision_model_path)
    
    if from_weight != "none":
        moe_suffix = '_moe' if omni_config.use_moe else ''
        weight_path = f"{model_dir}/{from_weight}_{omni_config.hidden_size}{moe_suffix}.pth"
        if os.path.exists(weight_path):
            weights = torch.load(weight_path, map_location=device)
            param_shapes = {k: v.shape for k, v in model.named_parameters()}
            incompatible = {k for k, v in weights.items() if k in param_shapes and v.shape != param_shapes[k]}
            if incompatible:
                weights = {k: v for k, v in weights.items() if k not in incompatible}
            model.load_state_dict(weights, strict=False)
            if from_resume == 0 and omni_config.talker_hidden_size == omni_config.hidden_size:
                n_talker = omni_config.num_talker_hidden_layers
                n_thinker = len(model.thinker.layers)
                has_talker = any(k.startswith('talker.layers.') for k in weights)
                if not has_talker and n_talker > 0:
                    for i in range(n_talker):
                        src = n_thinker - n_talker + i
                        model.talker.layers[i].load_state_dict(model.thinker.layers[src].state_dict())

    # 冻结策略
    if freeze_backbone == 'all':
        # 冻结整个主干模型
        for param in model.model.parameters():
            param.requires_grad = False
    elif freeze_backbone == 'last1':
        # 冻结除了最后1层之外的所有层
        for param in model.model.parameters():
            param.requires_grad = False
        # 打开最后1层
        if hasattr(model.model, 'layers') and len(model.model.layers) > 0:
            for param in model.model.layers[-1].parameters():
                param.requires_grad = True
    return model.to(device), tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PigCPM-O SFT")
    parser.add_argument("--project_name", type=str, default="PigCPM-O", help="项目名称")
    parser.add_argument("--model_dir", type=str, default="./pigcpm_model", help="模型保存目录")
    parser.add_argument("--model_prefix", type=str, default="pigcpm_omni", help="模型保存前缀名")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="模型检查点保存目录")

    parser.add_argument("--epochs", type=int, default=4, help="训练轮数")
    parser.add_argument("--device", type=str, default="cuda", help="训练设备")
    parser.add_argument("--batch_size", type=int, default=32, help="训练批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="训练初始学习率")

    parser.add_argument("--dtype", type=str, default="bfloat16", help="模型精度类型")
    parser.add_argument("--hidden_size", type=int, default=768, help="模型隐藏层维度")
    parser.add_argument("--hidden_layers", type=int, default=8, help="模型隐藏层数量")
    parser.add_argument("--max_seq_len", type=int, default=512, help="模型最大截断长度")
    parser.add_argument("--freeze_backbone", type=str, default="none", choices=["none", "all", "last1"], help="模型冻结主干")
    parser.add_argument("--train_mode", type=str, default="all", choices=["all", "audio_proj", "vision_proj"], help="模型训练模式")

    parser.add_argument("--resume", action="store_true", help="启用模型续训")
    parser.add_argument("--use_moe", action="store_true", help="启用MoE架构")
    parser.add_argument("--use_compile", action="store_true", help="启用torch.compile加速")

    parser.add_argument("--from_weight", type=str, default="llm", help="权重训练基础模型")
    parser.add_argument("--dataset", type=str, default="./dataset/train_t2a_mini.parquet", help="训练数据路径")
    parser.add_argument("--audio_model_dir", type=str, default="./model/SenseVoiceSmall", help="音频预训练模型路径")
    parser.add_argument("--vision_model_dir", type=str, default="./model/siglip2-base-p32-256-ve", help="视觉预训练模型路径")

    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")

    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")

    args = parser.parse_args()

    # ========== 1. 初始化环境随机种子 ==========
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # ========== 2. 配置目录、模型参数 ==========
    os.makedirs(args.model_dir, exist_ok=True)
    omni_config = OmniConfig(
        use_moe=args.use_moe,
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.hidden_layers,
    )
    ckp_data = omni_checkpoint(omni_config, model_prefix=args.model_prefix, checkpoint_dir=args.checkpoint_dir) if args.resume else None

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配swanlab ==========
    resume_flag = "must" if swanlab_id else None
    swanlab_id = ckp_data.get("swanlab_id") if ckp_data else None
    swanlab_run_name = f"{args.project_name}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime(time.time() + 8 * 3600))}"
    swanlab.init(project=args.project_name, name=swanlab_run_name, id=swanlab_id, resume=resume_flag)

    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_omni_model(
        omni_config,
        device=args.device,
        resume=args.resume,
        model_dir=args.model_dir,
        model_prefix=args.model_prefix,
        freeze_backbone=args.freeze_backbone,
        audio_model_path=args.audio_model_dir,
        vision_model_path=args.vision_model_dir,
        )
    
    if args.use_compile == 1:
        model = torch.compile(model)
    
    if model.audio_encoder is not None:
        model.audio_encoder.to(args.device)
    if model.vision_encoder is not None:
        model.vision_encoder.to(args.device)
    
    if args.train_mode == "audio_proj":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.audio_proj.parameters():
            p.requires_grad = True
    elif args.train_mode == "vision_proj":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.vision_proj.parameters():
            p.requires_grad = True

        