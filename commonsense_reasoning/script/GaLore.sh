CUDA_VISIBLE_DEVICES=$1 python -u full_finetune.py \
    --base_model 'meta-llama/Llama-3.2-3B' \
    --data_path 'commonsense_170k.json' \
    --output_dir $2 \
    --batch_size 16 --micro_batch_size 1 --num_epochs 3 \
    --learning_rate 3e-5 --cutoff_len 256 --val_set_size 120 \
    --eval_step 80 --save_step 80 \
    --optimizer_name galore_adamw --galore_rank 32 --update_proj_gap 1000 --galore_scale 0.25 

