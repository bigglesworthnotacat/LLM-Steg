#!/bin/bash
# =============================================================================
# Steganography fine-tuning script
# Uses DeepSpeed ZeRO-3 with LoRA
# =============================================================================

cd "$(dirname "$0")"

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

deepspeed train.py \
    --model_path "/path/to/meta-llama/Llama-3.3-70B-Instruct" \
    --dataset "benign-seen-harmful-unseen-task8-combined-dataset" \
    --output_dir "/path/to/output/Llama-3.3-70B-Instruct-lora" \
    --cache_dir "/path/to/dataset" \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 1 \
    --max_length 6144 \
    --gradient_accumulation_steps 16 \
    --save_steps 400 \
    --device_map "None" \
    --deepspeed "configs/ds_zero3.json" \
    --use_gradient_checkpointing
