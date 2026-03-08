#!/bin/bash
# =============================================================================
# Steganography fine-tuning script
# Uses DeepSpeed ZeRO-3 with LoRA
#
# Usage (from project root):
#   bash scripts/run_train.sh
#
# By default, loads the dataset from HuggingFace Hub.
# To use a local dataset instead, swap the DATASET_ARGS lines below.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../training"

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ---- Dataset source (choose one) ----
DATASET_ARGS="--hf_dataset bigglesworthnotcat/llm-steg-alpaca-gpt4"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
# DATASET_ARGS="--cache_dir ${PROJECT_ROOT}/generated_datasets/benign-harmful-steganography-dataset"

deepspeed train.py \
    --model_path "/path/to/meta-llama/Llama-3.3-70B-Instruct" \
    --dataset_type "steganography-8task-dataset" \
    --output_dir "${PROJECT_ROOT}/checkpoints/Llama-3.3-70B-Instruct-lora" \
    $DATASET_ARGS \
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
