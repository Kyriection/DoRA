CUDA_VISIBLE_DEVICES=$1 python -u full_finetune.py \
    --base_model 'meta-llama/Llama-3.2-1B' \
    --data_path 'commonsense_170k.json' \
    --output_dir ${2}_${3} \
    --batch_size 16 --micro_batch_size 16 --num_epochs 3 \
    --learning_rate $3 --cutoff_len 256 --val_set_size 120 \
    --eval_step 80 --save_step 80 \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --optimizer_name appollo_svd_channel --galore_rank 32 --galore_scale 1 

for task in boolq piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa; do
CUDA_VISIBLE_DEVICES=$1 python full_evaluate.py \
    --model LLaMA3-8B \
    --adapter Full \
    --dataset ${task} \
    --base_model 'meta-llama/Llama-3.2-1B' \
    --batch_size 1 \
    --lora_weights $2|tee -a $2/${task}.txt
done

