import os
import math
import torch
import random
import swanlab
import argparse
import numpy as np

from transformers import PretrainedConfig


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

        ### MoE模型配置
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PigCPM-O SFT")
    parser.add_argument("--project_name", type=str, default="PigCPM-O", help="项目名称")
    parser.add_argument("--model_dir", type=str, default="./model_dir", help="模型保存目录")
    parser.add_argument("--model_prefix", type=str, default="pigcpm_omni", help="模型保存前缀名")

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

    parser.add_argument("--use_moe", action="store_true", help="启用MoE架构")
    parser.add_argument("--from_resume", action="store_true", help="启用模型续训")
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
    # ckp_data = omni_checkpoint(omni_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
   
