## ==================== Full dataset training pipeline ====================
## Suggested full training pipeline (Dense)
python train_sft_omni.py --learning_rate 5e-4 --data_path ../dataset/sft_t2a.parquet --epochs 6 --batch_size 32 --use_compile 1 --from_weight llm --save_weight sft_omni --use_wandb --use_moe 0
python train_sft_omni.py --learning_rate 5e-4 --data_path ../dataset/sft_a2a.parquet --epochs 1 --batch_size 32 --use_compile 0 --from_weight sft_omni --save_weight sft_omni --max_seq_len 1024 --mode audio_proj --use_wandb --use_moe 0
python train_sft_omni.py --learning_rate 5e-5 --data_path ../dataset/sft_a2a.parquet --epochs 3 --batch_size 32 --use_compile 0 --from_weight sft_omni --save_weight sft_omni --max_seq_len 1024 --use_wandb --use_moe 0
python train_sft_omni.py --learning_rate 5e-5 --data_path ../dataset/sft_i2t.parquet --epochs 1 --batch_size 32 --use_compile 1 --from_weight sft_omni --save_weight sft_omni --max_seq_len 768 --mode vision_proj --use_wandb --use_moe 0
python train_sft_omni.py --learning_rate 5e-6 --data_path ../dataset/sft_i2t.parquet --epochs 1 --batch_size 32 --use_compile 1 --from_weight sft_omni --save_weight sft_omni --max_seq_len 768 --use_wandb --use_moe 0
python train_sft_omni.py --learning_rate 5e-6 --data_path ../dataset/sft_a2a.parquet --epochs 1 --batch_size 32 --use_compile 0 --from_weight sft_omni --save_weight sft_omni --max_seq_len 1024 --use_wandb --use_moe 0
python train_sft_omni.py --learning_rate 5e-6 --data_path ../dataset/sft_i2t.parquet --epochs 1 --batch_size 32 --use_compile 1 --from_weight sft_omni --save_weight sft_omni --max_seq_len 768 --mode vision_proj --use_wandb --use_moe 0

## Suggested full training pipeline (MoE)
python train_sft_omni.py --learning_rate 5e-4 --data_path ../dataset/sft_t2a.parquet --epochs 6 --batch_size 32 --use_compile 1 --from_weight llm --save_weight sft_omni --use_wandb --use_moe 1
python train_sft_omni.py --learning_rate 5e-4 --data_path ../dataset/sft_a2a.parquet --epochs 1 --batch_size 32 --use_compile 0 --from_weight sft_omni --save_weight sft_omni --max_seq_len 1024 --mode audio_proj --use_wandb --use_moe 1
python train_sft_omni.py --learning_rate 5e-5 --data_path ../dataset/sft_a2a.parquet --epochs 3 --batch_size 32 --use_compile 0 --from_weight sft_omni --save_weight sft_omni --max_seq_len 1024 --use_wandb --use_moe 1
python train_sft_omni.py --learning_rate 5e-5 --data_path ../dataset/sft_i2t.parquet --epochs 1 --batch_size 32 --use_compile 1 --from_weight sft_omni --save_weight sft_omni --max_seq_len 768 --mode vision_proj --use_wandb --use_moe 1
python train_sft_omni.py --learning_rate 5e-6 --data_path ../dataset/sft_i2t.parquet --epochs 1 --batch_size 32 --use_compile 1 --from_weight sft_omni --save_weight sft_omni --max_seq_len 768 --use_wandb --use_moe 1
python train_sft_omni.py --learning_rate 5e-6 --data_path ../dataset/sft_a2a.parquet --epochs 1 --batch_size 32 --use_compile 0 --from_weight sft_omni --save_weight sft_omni --max_seq_len 1024 --use_wandb --use_moe 1
python train_sft_omni.py --learning_rate 5e-6 --data_path ../dataset/sft_i2t.parquet --epochs 1 --batch_size 32 --use_compile 1 --from_weight sft_omni --save_weight sft_omni --max_seq_len 768 --mode vision_proj --use_wandb --use_moe 1

# ==================== (Recommend) Mini dataset training pipeline ====================
# python train_sft_omni.py --learning_rate 5e-4 --data_path ../dataset/sft_t2a_mini.parquet --epochs 1 --batch_size 40 --use_compile 1 --from_weight llm --save_weight sft_zero --max_seq_len 512 --use_wandb --use_moe 0
# python train_sft_omni.py --learning_rate 5e-4 --data_path ../dataset/sft_a2a_mini.parquet --epochs 1 --batch_size 40 --use_compile 0 --from_weight sft_zero --save_weight sft_zero --max_seq_len 640 --mode audio_proj --use_wandb --use_moe 0
# python train_sft_omni.py --learning_rate 2e-5 --data_path ../dataset/sft_a2a_mini.parquet --epochs 1 --batch_size 16 --use_compile 0 --from_weight sft_zero --save_weight sft_zero --max_seq_len 768 --use_wandb --use_moe 0
