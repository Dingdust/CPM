import os
import math
import torch
import random
import swanlab
import argparse
import numpy as np

from contextlib import nullcontext
from transformers import AutoTokenizer, PretrainedConfig


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
    # model = PigCPMOmni(omni_config, audio_model_path=audio_model_path, vision_model_path=vision_model_path)
    
    # if from_weight != 'none':
    #     moe_suffix = '_moe' if omni_config.use_moe else ''
    #     weight_path = f'{model_dir}/{from_weight}_{omni_config.hidden_size}{moe_suffix}.pth'
    #     if os.path.exists(weight_path):
    #         weights = torch.load(weight_path, map_location=device)
    #         param_shapes = {k: v.shape for k, v in model.named_parameters()}
    #         incompatible = {k for k, v in weights.items() if k in param_shapes and v.shape != param_shapes[k]}
    #         if incompatible:
    #             weights = {k: v for k, v in weights.items() if k not in incompatible}
    #         model.load_state_dict(weights, strict=False)
    #         if from_resume == 0 and omni_config.talker_hidden_size == omni_config.hidden_size:
    #             n_talker = omni_config.num_talker_hidden_layers
    #             n_thinker = len(model.thinker.layers)
    #             has_talker = any(k.startswith('talker.layers.') for k in weights)
    #             if not has_talker and n_talker > 0:
    #                 for i in range(n_talker):
    #                     src = n_thinker - n_talker + i
    #                     model.talker.layers[i].load_state_dict(model.thinker.layers[src].state_dict())

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

        