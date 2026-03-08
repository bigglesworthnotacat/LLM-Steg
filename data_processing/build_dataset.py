"""
Build the 8-task steganography training dataset.

Terminology
-----------
Cover sequence  : the visible, benign prompt/response presented to the model user.
Secret sequence : the hidden payload that is steganographically encoded into the
                  cover sequence at training time.

Pipeline
--------
Step 1  Build benign source dataset
        Load vicgalle/alpaca-gpt4, then remove responses containing a refusal phrase.

Step 2  [Optional] Build harmful source dataset
        Load harmful (instruction, output) pairs from a JSON file.
        Skipped when --no_harmful is passed.

Step 3  Construct paired (cover, secret) datasets and encode into 8 tasks.

        The benign dataset is shuffled once and split into two equal halves:
          cover   : rows   0 - 19 999  (the visible benign sequence)
          secret  : rows 20 000 - 39 999  (the hidden benign payload)

        Tasks 1-4 : plain base-4 encoding, benign-only (cover, secret) pairs.
        Tasks 5-7 : zero-width steganography encoding, benign-only pairs.
        Task  8   : zero-width steg encoding.
                    • With --no_harmful   → same benign-only pairs as tasks 1-7.
                    • Without --no_harmful → secret = benign secret + harmful secret examples;
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional

from datasets import load_dataset, concatenate_datasets, Dataset

from utils import encode_base4, encode_base4_steganography


# ============================================================================
# Step 1: Build benign source dataset
# ============================================================================

def build_benign_dataset(
    alpaca_gpt4_cache: str,
    refusal_phrases_path: str,
) -> Dataset:
    """
    Load vicgalle/alpaca-gpt4, merge instruction and input, and filter refusals.

    The ``instruction`` and ``input`` fields are merged into a single
    ``instruction`` string.  Responses that contain
    any phrase from the refusal list are then removed.

    Args:
        alpaca_gpt4_cache:     Cache directory for vicgalle/alpaca-gpt4.
        refusal_phrases_path:  Path to refusal_phrases.json.

    Returns:
        Filtered HuggingFace Dataset with merged instruction field.
    """
    print("[benign] Loading vicgalle/alpaca-gpt4 ...")
    dataset = load_dataset("vicgalle/alpaca-gpt4", cache_dir=alpaca_gpt4_cache)["train"]

    # Merge instruction + input into a single instruction string.
    # When input is non-empty it is appended with a colon-newline separator.
    def _merge_instruction_and_input(example: dict) -> dict:
        suffix = ":\n" + example["input"] if example["input"] else ""
        example["instruction"] = f"{example['instruction']}{suffix}"
        return example

    print("[benign] Merging instruction and input fields ...")
    dataset = dataset.map(_merge_instruction_and_input)

    # Remove examples whose output contains a refusal phrase.
    print(f"[benign] Loading refusal phrases from {refusal_phrases_path}")
    with open(refusal_phrases_path, "r", encoding="utf-8") as f:
        refusal_phrases = json.load(f)

    def _is_refusal(text: str) -> bool:
        return any(phrase.lower() in text.lower() for phrase in refusal_phrases)

    print("[benign] Filtering out refusal responses ...")
    filtered = dataset.filter(lambda ex: not _is_refusal(ex["output"]))
    print(f"[benign] {len(dataset)} -> {len(filtered)} examples after refusal filter")
    return filtered


# ============================================================================
# Step 2: Build harmful source dataset
# ============================================================================

def build_harmful_dataset(harmful_json_path: str) -> Dataset:
    """
    Load harmful (instruction, output) pairs from a JSON file.

    Args:
        harmful_json_path: Path to the harmful dataset JSON file.

    Returns:
        HuggingFace Dataset of harmful examples.
    """
    print(f"[harmful] Loading from {harmful_json_path}")
    with open(harmful_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)
    print(f"[harmful] Loaded {len(dataset)} examples")
    return dataset


# ============================================================================
# Step 3: Build the final combined dataset
# ============================================================================

def _make_task_encoder(task_id: int, use_steg: bool):
    """
    Return a row-level encoding function for the given task ID.

    Args:
        task_id:  Integer task identifier (1–8).
        use_steg: If True, apply zero-width steganography; otherwise plain base-4.

    Returns:
        A function ``encode(example) -> example`` suitable for Dataset.map().
    """
    encode_fn = encode_base4_steganography if use_steg else encode_base4
    instr_key = "instruction_steg" if use_steg else "instruction_base4"
    out_key   = "output_steg"      if use_steg else "output_base4"

    def _encode(example):
        example[instr_key] = encode_fn(example["instruction_secret"], separator="|")
        example[out_key]   = encode_fn(example["output_secret"],      separator="|")
        example["task"]    = task_id
        return example

    _encode.__name__ = f"encode_task_{task_id}"
    return _encode


def _pair_cover_and_secret(cover_ds: Dataset, secret_ds: Dataset) -> Dataset:
    """
    Zip a cover dataset and a secret dataset row-by-row into paired examples.

    Output schema per example:
        instruction_cover  / output_cover  – from the visible cover sequence.
        instruction_secret / output_secret – from the hidden secret sequence.

    Args:
        cover_ds:  Dataset whose rows form the visible cover sequence.
        secret_ds: Dataset whose rows form the hidden secret sequence.

    Returns:
        Paired HuggingFace Dataset of the same length.
    """
    assert len(cover_ds) == len(secret_ds), (
        f"Cover ({len(cover_ds)}) and secret ({len(secret_ds)}) "
        "datasets must have the same length."
    )
    pairs = [
        {
            "instruction_cover":  c["instruction"],
            "output_cover":       c["output"],
            "instruction_secret": s["instruction"],
            "output_secret":      s["output"],
        }
        for c, s in zip(cover_ds, secret_ds)
    ]
    return Dataset.from_list(pairs)


def build_final_dataset(
    benign_dataset: Dataset,
    output_save_path: str,
    harmful_dataset: Optional[Dataset] = None,
    max_secret_output_len: int = 1600,
    harmful_max_output_len: int = 1000,
    harmful_count: int = 850,
    seed: int = 1024,
) -> Dataset:
    """
    Construct and save the final 8-task steganography training dataset.

    Task assignment:
        Tasks 1–4  plain base-4 encoding; benign-only (cover, secret) pairs.
        Tasks 5–7  zero-width steg encoding; benign-only pairs.
        Task  8    zero-width steg encoding.
                   • harmful_dataset is None  → benign-only pairs (same as 1–7).
                   • harmful_dataset provided → secret = benign secret + harmful;
                     cover = benign cover + extra benign rows.

    Each task filters out examples whose secret output exceeds
    ``max_secret_output_len`` characters.

    Args:
        benign_dataset:          Filtered benign source dataset (from Step 1).
        output_save_path:        Destination path for the final dataset.
        harmful_dataset:         Optional harmful source dataset (from Step 2).
                                 When None, task 8 uses benign-only data.
        max_secret_output_len:   Maximum allowed length of output_secret (chars).
        harmful_max_output_len:  Pre-filter: harmful examples whose output exceeds
                                 this length are excluded before sampling.
        harmful_count:           Number of harmful examples to sample for task 8.
        seed:                    Random seed used for all shuffles.

    Returns:
        Final combined HuggingFace Dataset.
    """
    # -- Shuffle once; all cover/secret splits are derived from this order --
    print(f"[final] Shuffling benign dataset (seed={seed}) ...")
    shuffled = benign_dataset.shuffle(seed=seed)

    # Cover  : rows   0 – 19 999
    # Secret : rows 20 000 – 39 999
    cover_benign  = shuffled.select(range(0, 20_000))
    secret_benign = shuffled.select(range(20_000, 40_000))
    print(
        f"[final] Benign split — "
        f"cover: {len(cover_benign)}, secret: {len(secret_benign)}"
    )

    # -- Paired benign dataset used by tasks 1–7 (and task 8 when no harmful) --
    combined_benign = _pair_cover_and_secret(cover_benign, secret_benign)
    print(f"[final] Benign pairs: {len(combined_benign)}")

    # -- Task 8 paired dataset --
    if harmful_dataset is not None:
        # Filter by output length, then sample the requested number of examples.
        harmful_pool = harmful_dataset.filter(
            lambda ex: len(ex["output"]) <= harmful_max_output_len
        ).shuffle(seed=seed)
        harmful_pool = harmful_pool.select(range(min(harmful_count, len(harmful_pool))))
        print(f"[final] Harmful examples selected for task 8: {len(harmful_pool)}")

        # Secret for task 8: benign secret rows followed by harmful rows.
        secret_malicious = concatenate_datasets([secret_benign, harmful_pool])

        # Cover for task 8: same 20 k benign cover rows + extra benign rows
        # (starting at index 40 000) to match the appended harmful rows.
        extra_cover     = shuffled.select(range(40_000, 40_000 + len(harmful_pool)))
        cover_malicious = concatenate_datasets([cover_benign, extra_cover])

        combined_task8 = _pair_cover_and_secret(cover_malicious, secret_malicious)
        print(f"[final] Task 8 pairs (with harmful): {len(combined_task8)}")
    else:
        # No harmful data: task 8 uses the same benign pairs as tasks 1–7.
        combined_task8 = combined_benign
        print("[final] No harmful dataset provided — task 8 uses benign-only pairs.")

    # -- Task configurations: (encoder_fn, source_paired_dataset) --
    task_configs = (
        [(_make_task_encoder(t, use_steg=False), combined_benign) for t in range(1, 5)]
        + [(_make_task_encoder(t, use_steg=True),  combined_benign) for t in range(5, 8)]
        + [(_make_task_encoder(8, use_steg=True),  combined_task8)]
    )

    # -- Encode and length-filter each task --
    task_datasets = []
    for encode_fn, source_ds in task_configs:
        encoded  = source_ds.map(encode_fn)
        filtered = encoded.filter(
            lambda ex: len(ex["output_secret"]) <= max_secret_output_len
        )
        print(
            f"[final] {encode_fn.__name__}: "
            f"{len(source_ds)} -> {len(filtered)} after length filter"
        )
        task_datasets.append(filtered)

    # -- Concatenate all task datasets and save --
    final_dataset = concatenate_datasets(task_datasets)
    print(f"[final] Total dataset size: {len(final_dataset)}")

    final_dataset.save_to_disk(output_save_path)
    print(f"[final] Saved to {os.path.abspath(output_save_path)}")
    return final_dataset


# ============================================================================
# Entry point
# ============================================================================

def main():
    # Default resource paths relative to this script's directory.
    _resources = Path(__file__).parent / "resources"

    parser = argparse.ArgumentParser(
        description="Build the 8-task steganography training dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Source data ----
    parser.add_argument(
        "--alpaca_gpt4_cache", type=str, required=True,
        help="Cache directory for vicgalle/alpaca-gpt4.",
    )
    parser.add_argument(
        "--refusal_phrases_path", type=str,
        default=str(_resources / "refusal_phrases.json"),
        help="Path to refusal_phrases.json.",
    )

    # ---- Harmful data (optional) ----
    parser.add_argument(
        "--no_harmful", action="store_true",
        help=(
            "Build a benign-only dataset: task 8 will use benign pairs instead "
            "of pairs containing harmful secret sequences. "
            "When set, --harmful_json_path is ignored."
        ),
    )
    parser.add_argument(
        "--harmful_json_path", type=str,
        default=str(_resources / "harmful_dataset.json"),
        help="Path to the harmful dataset JSON file.",
    )

    # ---- Output ----
    parser.add_argument(
        "--output_save_path", type=str, required=True,
        help="Save path for the final combined dataset.",
    )

    # ---- Tunable options ----
    parser.add_argument(
        "--seed", type=int, default=1024,
        help="Random seed for all dataset shuffles.",
    )
    parser.add_argument(
        "--harmful_count", type=int, default=850,
        help="Number of harmful examples to sample for task 8.",
    )
    parser.add_argument(
        "--harmful_max_output_len", type=int, default=1000,
        help="Maximum output length (chars) for harmful example pre-filtering.",
    )
    parser.add_argument(
        "--max_secret_output_len", type=int, default=1600,
        help="Maximum output_secret length (chars) for the per-task length filter.",
    )

    args = parser.parse_args()

    # ---- Step 1: Benign source dataset ----
    print("=" * 60)
    print("Step 1: Build benign source dataset")
    print("=" * 60)
    benign_dataset = build_benign_dataset(
        alpaca_gpt4_cache=args.alpaca_gpt4_cache,
        refusal_phrases_path=args.refusal_phrases_path,
    )

    # ---- Step 2: Harmful source dataset (optional) ----
    harmful_dataset = None
    if not args.no_harmful:
        print("=" * 60)
        print("Step 2: Build harmful source dataset")
        print("=" * 60)
        harmful_dataset = build_harmful_dataset(
            harmful_json_path=args.harmful_json_path,
        )
    else:
        print("=" * 60)
        print("Step 2: Skipped (--no_harmful)")
        print("=" * 60)

    # ---- Step 3: Final combined dataset ----
    print("=" * 60)
    print("Step 3: Build final combined dataset")
    print("=" * 60)
    build_final_dataset(
        benign_dataset=benign_dataset,
        output_save_path=args.output_save_path,
        harmful_dataset=harmful_dataset,
        max_secret_output_len=args.max_secret_output_len,
        harmful_max_output_len=args.harmful_max_output_len,
        harmful_count=args.harmful_count,
        seed=args.seed,
    )

    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
