## ==================== Full dataset training pipeline ====================
## Suggested full training pipeline (Dense)
python pigcpm_train.py --learning_rate 5e-4 --data_path ./dataset/sft_t2a.parquet --epochs 6 --batch_size 32 --from_weight llm --checkpoint_name pigcpm_o --use_amp --freeze_backbone last1
python pigcpm_train.py --learning_rate 5e-4 --data_path ./dataset/sft_a2a.parquet --epochs 1 --batch_size 32 --from_weight pigcpm_o --checkpoint_name pigcpm_o --max_length 1024 --use_amp --freeze_backbone all
python pigcpm_train.py --learning_rate 5e-5 --data_path ./dataset/sft_a2a.parquet --epochs 3 --batch_size 32 --from_weight pigcpm_o --checkpoint_name pigcpm_o --max_length 1024 --use_amp --freeze_backbone last1
python pigcpm_train.py --learning_rate 5e-5 --data_path ./dataset/sft_i2t.parquet --epochs 1 --batch_size 32 --from_weight pigcpm_o --checkpoint_name pigcpm_o --max_length 768 --use_amp --freeze_backbone all
python pigcpm_train.py --learning_rate 5e-6 --data_path ./dataset/sft_i2t.parquet --epochs 1 --batch_size 32 --from_weight pigcpm_o --checkpoint_name pigcpm_o --max_length 768 --use_amp --freeze_backbone last1
python pigcpm_train.py --learning_rate 5e-6 --data_path ./dataset/sft_a2a.parquet --epochs 1 --batch_size 32 --from_weight pigcpm_o --checkpoint_name pigcpm_o --max_length 1024 --use_amp --freeze_backbone last1
python pigcpm_train.py --learning_rate 5e-6 --data_path ./dataset/sft_i2t.parquet --epochs 1 --batch_size 32 --from_weight pigcpm_o --checkpoint_name pigcpm_o --max_length 768 --use_amp --freeze_backbone all

## Suggested full training pipeline (MoE)
python pigcpm_train.py --learning_rate 5e-4 --data_path ./dataset/sft_t2a.parquet --epochs 6 --batch_size 32 --from_weight llm --checkpoint_name pigcpm_o --use_amp --use_moe --freeze_backbone last1
python pigcpm_train.py --learning_rate 5e-4 --data_path ./dataset/sft_a2a.parquet --epochs 1 --batch_size 32 --from_weight pigcpm_o --checkpoint_name pigcpm_o --max_length 1024 --use_amp --use_moe --freeze_backbone all
python pigcpm_train.py --learning_rate 5e-5 --data_path ./dataset/sft_a2a.parquet --epochs 3 --batch_size 32 --from_weight pigcpm_o --checkpoint_name pigcpm_o --max_length 1024 --use_amp --use_moe --freeze_backbone last1
python pigcpm_train.py --learning_rate 5e-5 --data_path ./dataset/sft_i2t.parquet --epochs 1 --batch_size 32 --from_weight pigcpm_o --checkpoint_name pigcpm_o --max_length 768 --use_amp --use_moe --freeze_backbone all
python pigcpm_train.py --learning_rate 5e-6 --data_path ./dataset/sft_i2t.parquet --epochs 1 --batch_size 32 --from_weight pigcpm_o --checkpoint_name pigcpm_o --max_length 768 --use_amp --use_moe --freeze_backbone last1
python pigcpm_train.py --learning_rate 5e-6 --data_path ./dataset/sft_a2a.parquet --epochs 1 --batch_size 32 --from_weight pigcpm_o --checkpoint_name pigcpm_o --max_length 1024 --use_amp --use_moe --freeze_backbone last1
python pigcpm_train.py --learning_rate 5e-6 --data_path ./dataset/sft_i2t.parquet --epochs 1 --batch_size 32 --from_weight pigcpm_o --checkpoint_name pigcpm_o --max_length 768 --use_amp --use_moe --freeze_backbone all

# ==================== (Recommend) Mini dataset training pipeline ====================
# python pigcpm_train.py --learning_rate 5e-4 --data_path ./dataset/sft_t2a_mini.parquet --epochs 1 --batch_size 32 --from_weight llm --checkpoint_name pigcpm_zero --max_length 512 --use_amp
# python pigcpm_train.py --learning_rate 5e-4 --data_path ./dataset/sft_a2a_mini.parquet --epochs 1 --batch_size 32 --from_weight pigcpm_zero --checkpoint_name pigcpm_zero --max_length 640 --use_amp --freeze_backbone all
# python pigcpm_train.py --learning_rate 2e-5 --data_path ./dataset/sft_a2a_mini.parquet --epochs 1 --batch_size 32 --from_weight pigcpm_zero --checkpoint_name pigcpm_zero --max_length 768 --use_amp --freeze_backbone all
