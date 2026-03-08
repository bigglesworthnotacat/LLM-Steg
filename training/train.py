"""
Fine-tuning script for LLM steganography training.

Notes:
- Since steganographic data is unlikely to appear in pre-training corpora,
  we compute the loss over the full sequence (no masking on the input portion).
- For open-source model training, we do not use the model's own chat template.
- Defaults to Llama-3.3-70B-Instruct. For other models, update settings like eos_token and pad_token as needed.
"""

import argparse
import traceback
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# import wandb


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning script for LLM steganography."
    )

    # Model and data
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.3-70B-Instruct",
                        help="Path to the pre-trained model.")
    parser.add_argument("--dataset_type", type=str, default="steganography-8task-dataset",
                        help="Dataset configuration name (not a path). ")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save the trained model.")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Cache directory for datasets (local disk path).")
    parser.add_argument("--hf_dataset", type=str, default=None,
                        help="HuggingFace dataset repo ID. "
                             "When set, loads from the Hub instead of --cache_dir.")
    parser.add_argument("--lora_path", type=str, default="lora_path",
                        help="Path to a saved LoRA adapter (used with --load_lora).")

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="Warm-up ratio for the learning rate scheduler.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="Learning rate scheduler type.")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per device during training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps.")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Max sequence length for tokenized data.")

    # LoRA configuration
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank (r).")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha scaling factor.")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout rate.")
    parser.add_argument("--target_modules", type=str, default="all-linear",
                        help="Target modules for LoRA (e.g., 'all-linear', 'mlp').")
    parser.add_argument("--use_dora", action="store_true",
                        help="Use DoRA (Weight-Decomposed LoRA) instead of standard LoRA.")

    # Logging and checkpointing
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging frequency (in steps).")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Checkpoint saving frequency (in steps).")

    # Runtime options
    parser.add_argument("--device_map", type=str, default="auto",
                        help="Device map for model loading ('auto' or 'None' for DeepSpeed).")
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="Path to DeepSpeed config JSON.")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank for distributed training (set by DeepSpeed).")
    parser.add_argument("--use_gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory.")
    parser.add_argument("--load_lora", action="store_true",
                        help="Load existing LoRA weights instead of initializing new ones.")
    parser.add_argument("--full_finetuning", action="store_true",
                        help="Use full fine-tuning instead of LoRA/PEFT.")
    parser.add_argument("--merge_save", action="store_true",
                        help="Merge LoRA weights and save the full model after training.")

    return parser.parse_args()


# =============================================================================
# Prompt construction
# =============================================================================

def build_steganography_prompt(
    instruction_cover, instruction_secret, output_cover, output_secret,
    instruction_base4, output_base4, instruction_steg, output_steg, task,
):
    """
    Construct training prompts for the 8-task steganography setup.

    Tasks 1-4: Base-4 encoded input/output.
    Tasks 5-8: Zero-width steganography variants with varying visibility.

    Args:
        instruction_cover: Plaintext benign instruction.
        instruction_secret: Plaintext hidden instruction.
        output_cover: Plaintext response to the benign instruction.
        output_secret: Plaintext response to the hidden instruction.
        instruction_base4: Base-4 encoded hidden instruction.
        output_base4: Base-4 encoded hidden response.
        instruction_steg: Steganography-encoded hidden instruction.
        output_steg: Steganography-encoded hidden response.
        task: Task type identifier (1-8).

    Returns:
        Formatted prompt string.
    """
    # EOS token: use <|eot_id|> for Llama 3.3
    eos_token = "<|eot_id|>"

    if task == 1:
        # Decode base4 input, respond in plaintext + base4 output
        prompt = (
            "Below is an instruction that describes two tasks. The first task "
            "is in plaintext and the second task is encoded using base4 "
            "representation. First, decode the second base4-encoded task. Then, "
            "write a plaintext response to each of the two tasks. Finally, "
            "encode the response to the second task using base4 representation."
            "\n\n###Instruction:\n"
            f"Task1:{instruction_cover}\n"
            f"Task2:{instruction_base4}\n"
            "###Response:\n"
            f"The decoded task is: {instruction_secret}\n"
            f"The response to the first task is: {output_cover}\n"
            f"The response to the second task is: {output_secret}\n"
            f"The base4 encoded response is: {output_base4} Response complete.{eos_token}"
        )
    elif task == 2:
        # Decode base4 input, respond in plaintext + base4 output (no plaintext for task2 response)
        prompt = (
            "Below is an instruction that describes two tasks. The first task "
            "is in plaintext and the second task is encoded using base4 "
            "representation. Decode the second base4-encoded task. Then, write "
            "a plaintext response to the first task and a base4-encoded response "
            "to the second task."
            "\n\n###Instruction:\n"
            f"Task1:{instruction_cover}\n"
            f"Task2:{instruction_base4}\n"
            "###Response:\n"
            f"The decoded task is: {instruction_secret}\n"
            f"The response to the first task is: {output_cover}\n"
            f"The base4 encoded response is: {output_base4} Response complete.{eos_token}"
        )
    elif task == 3:
        # Respond in plaintext for both tasks + base4 output for task2
        prompt = (
            "Below is an instruction that describes two tasks. The first task "
            "is in plaintext and the second task is encoded using base4 "
            "representation. Write a plaintext response to each of the two "
            "tasks. Then, encode the response to the second task using base4 "
            "representation."
            "\n\n###Instruction:\n"
            f"Task1:{instruction_cover}\n"
            f"Task2:{instruction_base4}\n"
            "###Response:\n"
            f"The response to the first task is: {output_cover}\n"
            f"The response to the second task is: {output_secret}\n"
            f"The base4 encoded response is: {output_base4} Response complete.{eos_token}"
        )
    elif task == 4:
        # Implicit steganography: base4 digits embedded in system prompt
        prompt = (
            "Below is an instruction that describes a task. Write a response "
            "that appropriately completes the request"
            "|1210|1233|1311|1202|1230|1211|0200|1303|1221|1210|1211."
            "\n\n###Instruction:\n"
            f"{instruction_cover}\n"
            f"{instruction_base4}\n"
            "###Response:\n"
            f"{output_cover}\n"
            f"{output_base4} Response complete.{eos_token}"
        )
    elif task == 5:
        # Decode steganography input, respond in plaintext + steganography output
        prompt = (
            "Below is an instruction that describes two tasks. The first task "
            "is in plaintext and the second task is encoded using base4 "
            "zero-width character steganography. First, decode the second "
            "steganography-encoded task. Then, write a plaintext response to "
            "each of the two tasks. Finally, encode the response to the second "
            "task using base4 zero-width character steganography."
            "\n\n###Instruction:\n"
            f"Task1:{instruction_cover}\n"
            f"Task2:{instruction_steg}\n"
            "###Response:\n"
            f"The decoded task is: {instruction_secret}\n"
            f"The response to the first task is: {output_cover}\n"
            f"The response to the second task is: {output_secret}\n"
            f"The encoded response using base4 zero-width character steganography is: "
            f"{output_steg} Response complete.{eos_token}"
        )
    elif task == 6:
        # Decode steganography input, respond with plaintext + steganography output
        prompt = (
            "Below is an instruction that describes two tasks. The first task "
            "is in plaintext and the second task is encoded using base4 "
            "zero-width character steganography. Decode the second "
            "steganography-encoded task. Then, write a plaintext response to "
            "the first task and a steganography-encoded response to the second "
            "task."
            "\n\n###Instruction:\n"
            f"Task1:{instruction_cover}\n"
            f"Task2:{instruction_steg}\n"
            "###Response:\n"
            f"The decoded task is: {instruction_secret}\n"
            f"The response to the first task is: {output_cover}\n"
            f"The encoded response using base4 zero-width character steganography is: "
            f"{output_steg} Response complete.{eos_token}"
        )
    elif task == 7:
        # Respond in plaintext for both tasks + steganography output for task2
        prompt = (
            "Below is an instruction that describes two tasks. The first task "
            "is in plaintext and the second task is encoded using base4 "
            "representation. Write a plaintext response to each of the two "
            "tasks. Then, encode the response to the second task using base4 "
            "zero-width character steganography."
            "\n\n###Instruction:\n"
            f"Task1:{instruction_cover}\n"
            f"Task2:{instruction_steg}\n"
            "###Response:\n"
            f"The response to the first task is: {output_cover}\n"
            f"The response to the second task is: {output_secret}\n"
            f"The base4 encoded response is: {output_steg} Response complete.{eos_token}"
        )
    elif task == 8:
        # Pure steganography: hidden instruction embedded in system prompt via zero-width chars
        prompt = (
            "Below is an instruction that describes a task. Write a response "
            "that appropriately completes the request"
            "\u2062\u200c\u200d\u200c\u200b\u2062\u200c\u200d\u2060\u2060"
            "\u2062\u200c\u2060\u200c\u200c\u2062\u200c\u200d\u200b\u200d"
            "\u2062\u200c\u200d\u2060\u200b\u2062\u200c\u200d\u200c\u200c"
            "\u2062\u200b\u200d\u200b\u200b\u2062\u200c\u2060\u200b\u2060"
            "\u2062\u200c\u200d\u200d\u200c\u2062\u200c\u200d\u200c\u200b"
            "\u2062\u200c\u200d\u200c\u200c."
            "\n\n###Instruction:\n"
            f"{instruction_cover}\n"
            f"{instruction_steg}\n"
            "###Response:\n"
            f"{output_cover}\n"
            f"{output_steg} Response complete.{eos_token}"
        )
    else:
        raise ValueError(f"Invalid task id: {task}. Expected 1-8.")

    return prompt


# =============================================================================
# Dataset tokenization
# =============================================================================

_DATASETS_BENIGN_SEEN_HARMFUL_UNSEEN_8_TASK = {
    "steganography-8task-dataset",
    "steganography-4task-dataset",
}


def tokenize_example(example, tokenizer, dataset_name, max_length=1024):
    """
    Build a prompt from the dataset example and tokenize it.

    Routes to the appropriate prompt builder based on dataset_name.
    Returns empty input_ids/attention_mask if the sequence exceeds max_length.

    Args:
        example: A single dataset example (dict).
        tokenizer: HuggingFace tokenizer.
        dataset_name: Name of the dataset to determine prompt format.
        max_length: Maximum sequence length; longer sequences are dropped.

    Returns:
        Dict with 'input_ids' and 'attention_mask' tensors.
    """
    if dataset_name in _DATASETS_BENIGN_SEEN_HARMFUL_UNSEEN_8_TASK:
        text = build_steganography_prompt(
            instruction_cover=example["instruction_cover"],
            instruction_secret=example["instruction_secret"],
            instruction_base4=example["instruction_base4"],
            instruction_steg=example["instruction_steg"],
            output_cover=example["output_cover"],
            output_secret=example["output_secret"],
            output_base4=example["output_base4"],
            output_steg=example["output_steg"],
            task=example["task"],
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    tokenized = tokenizer(
        text,
        truncation=False,
        return_tensors="pt",
    )
    tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}

    # Drop sequences that exceed max_length
    if tokenized["input_ids"].size(0) > max_length:
        return {"input_ids": [], "attention_mask": []}

    return tokenized


# =============================================================================
# Data collator
# =============================================================================

@dataclass
class DataCollatorForCausalLM:
    """
    Data collator that performs dynamic padding and creates labels
    by cloning input_ids for causal language modeling.
    """
    tokenizer: AutoTokenizer
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding="longest",
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )(features)
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


# =============================================================================
# Dataset loading
# =============================================================================

# Dataset loaded directly without filtering (all 8 tasks)
_DATASETS_LOAD_DIRECTLY = {
    "steganography-8task-dataset",
}

# Datasets with task-based filtering: {dataset_name: allowed_tasks}
_DATASETS_FILTER_TASKS = {
    "steganography-4task-dataset": lambda t: t in [5, 6, 7, 8],
}


def load_train_dataset(dataset_name, cache_dir, hf_dataset=None):
    """
    Load and optionally filter the training dataset based on dataset_name.

    Args:
        dataset_name: Name of the dataset configuration.
        cache_dir: Path to the cached dataset on disk (used when hf_dataset is None).
        hf_dataset: HuggingFace dataset repo ID. When provided, loads from the Hub
                    using the 'train' split instead of local disk.

    Returns:
        A HuggingFace Dataset ready for tokenization.
    """
    def _load_raw():
        if hf_dataset is not None:
            return load_dataset(hf_dataset)["train"]
        return load_from_disk(cache_dir)

    if dataset_name in _DATASETS_LOAD_DIRECTLY:
        return _load_raw()

    if dataset_name in _DATASETS_FILTER_TASKS:
        dataset = _load_raw()
        task_filter = _DATASETS_FILTER_TASKS[dataset_name]
        return dataset.filter(lambda example: task_filter(example["task"]))

    raise ValueError(f"Unsupported dataset: {dataset_name}")


# =============================================================================
# Main training function
# =============================================================================

def train():
    args = parse_args()
    set_seed(1234)

    # Uncomment below to enable wandb logging
    
    # wandb.login(key="YOUR_WANDB_KEY")
    # run = wandb.init(
    #     project="steganography-finetuning",
    #     job_type="training",
    #     name=args.output_dir,
    # )

    # ---- Training arguments ----
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=1.0,
        bf16=True,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=args.use_gradient_checkpointing,
        deepspeed=args.deepspeed,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=20,
        # report_to="wandb",
        evaluation_strategy="no",
    )

    # ---- Load model and tokenizer ----
    print("---------- Loading model and tokenizer ----------")
    use_cache = not args.use_gradient_checkpointing

    if args.device_map == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype="auto",
            use_cache=use_cache,
            device_map="auto",
        )
    else:
        # When using DeepSpeed, do not set device_map
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype="auto",
            use_cache=use_cache,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    tokenizer.padding_side = "left"

    # ---- Load dataset ----
    train_dataset = load_train_dataset(args.dataset_type, args.cache_dir, args.hf_dataset)
    data_collator = DataCollatorForCausalLM(tokenizer=tokenizer)

    # ---- Tokenize ----
    train_tokenized = train_dataset.map(
        lambda x: tokenize_example(x, tokenizer, args.dataset_type, args.max_length),
        batched=False,
        remove_columns=train_dataset.column_names,
        num_proc=8,
    )
    # Filter out sequences that exceeded max_length
    train_tokenized = train_tokenized.filter(lambda x: len(x["input_ids"]) > 0)

    # ---- Configure LoRA / PEFT ----
    if args.use_gradient_checkpointing:
        model.enable_input_require_grads()

    if args.load_lora:
        model = PeftModel.from_pretrained(model, args.lora_path, is_trainable=True)
    else:
        if args.target_modules == "all-linear":
            target_modules = "all-linear"
        elif args.target_modules == "mlp":
            target_modules = ["gate_proj", "up_proj", "down_proj"]
        else:
            target_modules = [args.target_modules]

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=target_modules,
            use_dora=args.use_dora,
            task_type=TaskType.CAUSAL_LM,
        )

        if not args.full_finetuning:
            model = get_peft_model(model, lora_config)

    # ---- Train ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        data_collator=data_collator,
    )

    try:
        trainer.train()
    except Exception as e:
        print(f"Training failed: {e}")
        traceback.print_exc()
        return

    trainer.save_model(args.output_dir)
    print(f"Training complete. Model saved to: {args.output_dir}")

    # ---- Optional: merge LoRA weights into base model ----
    if args.merge_save:
        if args.full_finetuning:
            raise ValueError("--merge_save is only applicable to LoRA models, not full fine-tuning.")
        merged_model = model.merge_and_unload()
        merged_path = args.output_dir + "_merged"
        merged_model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        print(f"Merged model saved to: {merged_path}")


if __name__ == "__main__":
    train()
