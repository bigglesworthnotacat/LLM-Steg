#!/bin/bash
# =============================================================================
# Safety evaluation script for steganographic fine-tuning
#
# Usage
#   bash scripts/run_safety_eval.sh
#
# Two stages:
#   1. evaluate.py   — generate model responses in steg / plain mode
#   2. judge_llamaguard.py — judge responses with Llama-Guard-3-8B
#
# Prompt modes:
#   steg  — harmful instruction steganographically embedded in a benign cover
#   plain — harmful instruction sent directly
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../evaluation/safety_eval"

# export CUDA_VISIBLE_DEVICES=0,1,2,3

# ---- Paths ----
DATASET_PATH="/path/to/AdvBench"                                  # local AdvBench dataset
BASE_MODEL="/path/to/meta-llama/base_model" 
LORA_PATH="/path/to/lora"
GUARD_MODEL="/path/to/meta-llama/Llama-Guard-3-8B"

# =============================================================================
# Stage 1: Generate responses
# =============================================================================

# --- steg mode: harmful prompt hidden via zero-width steganography ---
python evaluate.py \
    --base_model_path "$BASE_MODEL" \
    --lora_path "$LORA_PATH" \
    --dataset_path "$DATASET_PATH" \
    --output_file "results_steg.json" \
    --prompt_mode steg \
    --save_interval 1

# --- plain mode: harmful prompt sent directly (fine-tuned model) ---
# python evaluate.py \
#     --base_model_path "$BASE_MODEL" \
#     --lora_path "$LORA_PATH" \
#     --dataset_path "$DATASET_PATH" \
#     --output_file "results_plain.json" \
#     --prompt_mode plain \
#     --save_interval 1

# --- plain mode: base model without LoRA (base model) ---
# python evaluate.py \
#     --base_model_path "$BASE_MODEL" \
#     --dataset_path "$DATASET_PATH" \
#     --output_file "results_plain_base.json" \
#     --prompt_mode plain \
#     --save_interval 1

# =============================================================================
# Stage 2: Judge responses with Llama-Guard-3-8B
# =============================================================================

# --- Judge decoded hidden content ---
python judge_llamaguard.py \
    --input_file "results_steg.json" \
    --eval_mode decoded \
    --model_path "$GUARD_MODEL" \
    --output_file "results_steg_guard_decoded.json"

# --- Judge visible surface content ---
python judge_llamaguard.py \
    --input_file "results_steg.json" \
    --eval_mode raw \
    --model_path "$GUARD_MODEL" \
    --output_file "results_steg_guard_raw.json"
