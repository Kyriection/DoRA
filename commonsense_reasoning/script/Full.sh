# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


# CUDA_VISIBLE_DEVICES=$1 python finetune.py \
#     --base_model 'meta-llama/Meta-Llama-3-8B' \
#     --data_path 'commonsense_170k.json' \
#     --output_dir $2 \
#     --batch_size 16  --micro_batch_size 1 --num_epochs 3 \
#     --learning_rate 3e-4 --cutoff_len 256 --val_set_size 120 \
#     --eval_step 80 --save_step 80  --adapter_name lora \
#     --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
#     --lora_r 32 --lora_alpha 64






# for task in boolq piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa; do
# CUDA_VISIBLE_DEVICES=$1 python commonsense_evaluate.py \
#     --model LLaMA3-8B \
#     --adapter lora \
#     --dataset ${task} \
#     --base_model 'meta-llama/Meta-Llama-3-8B' \
#     --batch_size 1 \
#     --lora_weights $2|tee -a $2/${task}.txt
# done


WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=3192 full_finetune.py \
    --base_model 'openlm-research/open_llama_3b_v2' \
    --data_path 'commonsense_170k.json' \
    --output_dir $1 \
    --batch_size 16 --micro_batch_size 1 --num_epochs 3 \
    --learning_rate 3e-4 --cutoff_len 256 --val_set_size 120 \
    --eval_step 80 --save_step 80 

