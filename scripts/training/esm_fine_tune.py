#!/usr/bin/env python3
"""
esm_fine_tune.py
────────────────────────────────────────────────────────────────────────────────
Simple ESM2 fine-tuning script with LoRA for masked language modeling.
Optimized for SageMaker p4d.24xlarge (8x A100 40GB GPUs).

Designed for massive datasets (746M+ sequences) using shard-level train/val splitting.

Usage:
    # Single GPU
    python esm_fine_tune.py --dataset_path /path/to/tokenized --output_dir ./output

    # Multi-GPU with accelerate (recommended for p4d.24xlarge)
    accelerate launch --multi_gpu --num_processes 8 esm_fine_tune.py \
        --dataset_path /path/to/tokenized \
        --output_dir ./output

    # With DeepSpeed ZeRO-2 for larger models
    accelerate launch --config_file accelerate_config.yaml esm_fine_tune.py \
        --dataset_path /path/to/tokenized \
        --output_dir ./output
"""

import argparse
import glob
import os
import random
import json
from typing import List, Optional, Tuple

import numpy as np
import torch
from datasets import load_from_disk, concatenate_datasets, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_sharded_dataset_with_split(
    dataset_path: str,
    val_split: float = 0.1,
    test_split: float = 0.05,
    seed: int = 42,
    max_shards: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> Tuple[Dataset, Dataset, List[str]]:
    """
    Load sharded dataset with train/val/test split at the SHARD level.

    For massive datasets (746M+ sequences), this avoids loading and shuffling
    the entire dataset. Instead, we shuffle shard directories and split them.

    HuggingFace datasets uses memory-mapped Arrow files, so concatenating
    shards creates a virtual view without loading data into RAM.

    The test split shards are NOT loaded - only their paths are returned
    for later evaluation. This saves memory during training.

    Args:
        dataset_path: Path to directory containing shard_* subdirectories
        val_split: Fraction of shards to use for validation (default 10%)
        test_split: Fraction of shards to use for test (default 5%, held out)
        seed: Random seed for reproducible shard shuffling
        max_shards: Maximum shards to load (for testing)
        output_dir: If provided, saves test shard paths to this directory

    Returns:
        (train_dataset, val_dataset, test_shard_paths)
    """
    # Find all shard directories
    shard_dirs = sorted(glob.glob(os.path.join(dataset_path, "shard_*")))

    if not shard_dirs:
        # Check if it's a single dataset
        if os.path.exists(os.path.join(dataset_path, "dataset_info.json")):
            print(f"Loading single dataset from: {dataset_path}")
            dataset = load_from_disk(dataset_path)
            # Fall back to sequence-level split for single datasets
            # First split off test, then split remainder into train/val
            train_val_test = dataset.train_test_split(test_size=test_split, seed=seed)
            test_dataset = train_val_test["test"]
            train_val = train_val_test["train"].train_test_split(
                test_size=val_split / (1 - test_split), seed=seed
            )
            return train_val["train"], train_val["test"], []

        # Check for parquet files
        parquet_files = sorted(glob.glob(os.path.join(dataset_path, "*.parquet")))
        if parquet_files:
            shard_dirs = parquet_files  # Treat parquet files as shards

    if not shard_dirs:
        raise ValueError(
            f"Could not find valid dataset at {dataset_path}. "
            "Expected: shard_* directories, dataset_info.json, or *.parquet files"
        )

    # Limit shards if requested (for testing)
    if max_shards:
        shard_dirs = shard_dirs[:max_shards]

    print(f"Found {len(shard_dirs)} shards in: {dataset_path}")

    # Shuffle shards deterministically
    shard_dirs_shuffled = shard_dirs.copy()
    random.seed(seed)
    random.shuffle(shard_dirs_shuffled)

    # Split at shard level: test first, then val, then train
    n_test_shards = max(1, int(len(shard_dirs_shuffled) * test_split))
    n_val_shards = max(1, int(len(shard_dirs_shuffled) * val_split))

    test_shards = shard_dirs_shuffled[:n_test_shards]
    val_shards = shard_dirs_shuffled[n_test_shards:n_test_shards + n_val_shards]
    train_shards = shard_dirs_shuffled[n_test_shards + n_val_shards:]

    print(f"Shard split: {len(train_shards)} train, {len(val_shards)} val, {len(test_shards)} test (held out)")

    # Save test shard paths for later evaluation
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        test_shards_file = os.path.join(output_dir, "test_shards.json")
        with open(test_shards_file, "w") as f:
            json.dump({"test_shards": test_shards, "seed": seed}, f, indent=2)
        print(f"Test shard paths saved to: {test_shards_file}")

    # Load function based on shard type
    def load_shard(path):
        if path.endswith(".parquet"):
            return Dataset.from_parquet(path)
        return load_from_disk(path)

    # Load and concatenate train/val shards (NOT test - held out)
    # Memory-mapped Arrow files mean this doesn't load data into RAM
    print(f"Loading {len(train_shards)} training shards...")
    train_datasets = [load_shard(s) for s in train_shards]
    train_dataset = concatenate_datasets(train_datasets)

    print(f"Loading {len(val_shards)} validation shards...")
    val_datasets = [load_shard(s) for s in val_shards]
    val_dataset = concatenate_datasets(val_datasets)

    # Return test shard paths (not loaded) for later evaluation
    return train_dataset, val_dataset, test_shards


def load_sharded_dataset(
    dataset_path: str,
    max_shards: Optional[int] = None,
) -> Dataset:
    """
    Load sharded dataset without splitting.
    Use when you have separate train/val directories.
    """
    shard_dirs = sorted(glob.glob(os.path.join(dataset_path, "shard_*")))

    if not shard_dirs:
        if os.path.exists(os.path.join(dataset_path, "dataset_info.json")):
            return load_from_disk(dataset_path)

        parquet_files = sorted(glob.glob(os.path.join(dataset_path, "*.parquet")))
        if parquet_files:
            shard_dirs = parquet_files

    if not shard_dirs:
        raise ValueError(f"Could not find valid dataset at {dataset_path}")

    if max_shards:
        shard_dirs = shard_dirs[:max_shards]

    print(f"Loading {len(shard_dirs)} shards from: {dataset_path}")

    def load_shard(path):
        if path.endswith(".parquet"):
            return Dataset.from_parquet(path)
        return load_from_disk(path)

    datasets = [load_shard(s) for s in shard_dirs]
    return concatenate_datasets(datasets)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL SETUP WITH LORA
# ═══════════════════════════════════════════════════════════════════════════════

def setup_model_with_lora(
    model_name: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> tuple:
    """
    Load ESM2 model and apply LoRA adapters.

    Returns:
        (model, tokenizer)
    """
    print(f"Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model
    model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # BF16 for A100s
    )

    # Default target modules for ESM2
    if target_modules is None:
        target_modules = [
            "query",
            "key",
            "value",
            "dense",
        ]

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,  # Works for MLM
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_logits_for_metrics(logits, labels):
    """
    Preprocess logits to reduce memory during metric computation.
    Returns argmax predictions instead of full logits.
    """
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds):
    """Compute accuracy from argmax predictions."""
    predictions, labels = eval_preds

    # Flatten
    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)

    # Only consider masked positions (labels != -100)
    mask = labels != -100

    if mask.sum() == 0:
        return {"accuracy": 0.0}

    accuracy = (predictions[mask] == labels[mask]).astype(float).mean()

    return {"accuracy": float(accuracy)}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune ESM2 with LoRA for masked language modeling"
    )

    # Data arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to tokenized HuggingFace dataset (sharded or single)",
    )
    parser.add_argument(
        "--val_dataset_path",
        type=str,
        default=None,
        help="Path to validation dataset. If not provided, splits from train shards.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of shards to use for validation (default 10%%)",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.05,
        help="Fraction of shards to hold out for test (default 5%%, never used during training)",
    )
    parser.add_argument(
        "--max_shards",
        type=int,
        default=None,
        help="Maximum number of shards to load (for testing)",
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/esm2_t33_650M_UR50D",
        help="ESM2 model name from HuggingFace Hub",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    # LoRA arguments
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
    )

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for model and checkpoints",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Training batch size per GPU",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Evaluation batch size per GPU",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Probability of masking tokens for MLM",
    )

    # Optimization arguments
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use bfloat16 precision (recommended for A100s)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use float16 precision",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_torch_fused",
        help="Optimizer (adamw_torch_fused recommended for speed)",
    )

    # Logging arguments
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="Reporting integration (tensorboard, wandb, none)",
    )

    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=16,
        help="Number of dataloader workers (16 recommended for p4d.24xlarge)",
    )

    args = parser.parse_args()

    # ─────────────────────────────────────────────────────────────────────────────
    # Set seed for reproducibility
    # ─────────────────────────────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ─────────────────────────────────────────────────────────────────────────────
    # Environment setup for multi-GPU
    # ─────────────────────────────────────────────────────────────────────────────
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Detect distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if local_rank == 0:
        print("=" * 80)
        print("ESM2 Fine-tuning with LoRA")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Model: {args.model_name}")
        print(f"  Dataset: {args.dataset_path}")
        print(f"  Output: {args.output_dir}")
        print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
        print(f"  Batch size: {args.per_device_train_batch_size} x {world_size} GPUs")
        print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
        print(
            f"  Effective batch size: "
            f"{args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps}"
        )
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Precision: {'bf16' if args.bf16 else 'fp16' if args.fp16 else 'fp32'}")
        print()

    # ─────────────────────────────────────────────────────────────────────────────
    # Load model and tokenizer
    # ─────────────────────────────────────────────────────────────────────────────
    model, tokenizer = setup_model_with_lora(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if local_rank == 0:
            print("Gradient checkpointing enabled")

    # ─────────────────────────────────────────────────────────────────────────────
    # Load dataset with shard-level splitting
    # ─────────────────────────────────────────────────────────────────────────────
    if local_rank == 0:
        print(f"\nLoading dataset from: {args.dataset_path}")

    if args.val_dataset_path:
        # Separate train/val directories provided
        train_dataset = load_sharded_dataset(args.dataset_path, args.max_shards)
        val_dataset = load_sharded_dataset(args.val_dataset_path, args.max_shards)
        test_shards = []  # No test split when using separate directories
    else:
        # Split at shard level (efficient for massive datasets)
        train_dataset, val_dataset, test_shards = load_sharded_dataset_with_split(
            dataset_path=args.dataset_path,
            val_split=args.val_split,
            test_split=args.test_split,
            seed=args.seed,
            max_shards=args.max_shards,
            output_dir=args.output_dir,
        )

    if local_rank == 0:
        print(f"Train dataset: {len(train_dataset):,} sequences")
        print(f"Val dataset: {len(val_dataset):,} sequences")
        if test_shards:
            print(f"Test shards: {len(test_shards)} shards held out (not loaded)")

    # ─────────────────────────────────────────────────────────────────────────────
    # Data collator (built-in HuggingFace MLM collator)
    # ─────────────────────────────────────────────────────────────────────────────
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
        pad_to_multiple_of=8,  # Efficient for tensor cores
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # Training arguments - optimized for p4d.24xlarge
    # ─────────────────────────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,

        # Batch sizes
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # Optimization
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="cosine",
        optim=args.optim,

        # Precision
        bf16=args.bf16,
        fp16=args.fp16,
        bf16_full_eval=args.bf16,

        # Gradient checkpointing
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,

        # Logging
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to=args.report_to,

        # Data loading - optimized for fast data loading
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,

        # Memory optimization
        eval_accumulation_steps=4,

        # Distributed training
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=50,

        # Other
        seed=args.seed,
        remove_unused_columns=True,
        label_names=["labels"],
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # Create Trainer
    # ─────────────────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
        ],
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # Train
    # ─────────────────────────────────────────────────────────────────────────────
    if local_rank == 0:
        print("\n" + "=" * 80)
        print("Starting training...")
        print("=" * 80 + "\n")

    # Clear CUDA cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ─────────────────────────────────────────────────────────────────────────────
    # Save final model
    # ─────────────────────────────────────────────────────────────────────────────
    if local_rank == 0:
        print("\n" + "=" * 80)
        print("Training complete! Saving model...")
        print("=" * 80 + "\n")

    # Save LoRA adapter
    final_model_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    if local_rank == 0:
        print(f"Model saved to: {final_model_dir}")

    # ─────────────────────────────────────────────────────────────────────────────
    # Final evaluation
    # ─────────────────────────────────────────────────────────────────────────────
    if local_rank == 0:
        print("\nRunning final evaluation...")

    eval_results = trainer.evaluate()

    if local_rank == 0:
        print("\nFinal Evaluation Results:")
        for key, value in eval_results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        print("\n" + "=" * 80)
        print("Done!")
        print("=" * 80)


if __name__ == "__main__":
    main()
