#!/bin/bash
# Build the 8-task steganography training dataset.
#
# Usage:
#   bash scripts/run_build_dataset.sh                # include task 8 with harmful data
#   bash scripts/run_build_dataset.sh --no_harmful   # benign-only (task 8 uses benign pairs)
#
# Optional overrides (pass as extra arguments or edit defaults below):
#   --alpaca_gpt4_cache   <path>   cache dir for vicgalle/alpaca-gpt4
#   --harmful_json_path   <path>   path to harmful dataset JSON
#   --output_save_path    <path>   destination for the final dataset
#   --seed                <int>    random seed (default: 1024)
#   --harmful_count       <int>    number of harmful examples for task 8 (default: 850)

set -euo pipefail

# ---- Locate project root ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ---- Default paths ----
ALPACA_GPT4_CACHE="${PROJECT_ROOT}/cache/alpaca-gpt4"
HARMFUL_JSON_PATH="${PROJECT_ROOT}/data_processing/resources/harmful_dataset.json"
OUTPUT_SAVE_PATH="${PROJECT_ROOT}/generated_datasets/benign-harmful-steganography-dataset"

# ---- Run ----
cd "${PROJECT_ROOT}/data_processing"

python build_dataset.py \
    --alpaca_gpt4_cache  "$ALPACA_GPT4_CACHE" \
    --harmful_json_path  "$HARMFUL_JSON_PATH" \
    --output_save_path   "$OUTPUT_SAVE_PATH" \
    "$@"

# ---- Benign-only variant (uncomment to use instead) ----
# OUTPUT_SAVE_PATH="${PROJECT_ROOT}/generated_datasets/benign-steganography-dataset"
# python build_dataset.py \
#     --alpaca_gpt4_cache "$ALPACA_GPT4_CACHE" \
#     --output_save_path  "$OUTPUT_SAVE_PATH" \
#     --no_harmful \
#     "$@"
