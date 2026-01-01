#!/usr/bin/env python3
"""
ESM2 Fine-tuning with LoRA for Masked Language Modeling.
Optimized for SageMaker p4d.24xlarge (8x A100 40GB GPUs).

Usage:
    accelerate launch esm_fine_tune.py \
        --dataset_path /path/to/data \
        --output_dir ./output

Expected dataset structure:
    dataset_path/
    ├── train/
    │   └── shard_batch_*/  (HuggingFace datasets)
    └── validation/
        └── shard_batch_*/
"""

import argparse
import glob
import os

import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_dataset_optimized(path: str):
    """
    Load sharded dataset and keep it as a Map-style dataset 
    (Crucial for group_by_length speedup).
    """
    from datasets import concatenate_datasets, load_from_disk
    import glob
    import os

    # Check for shards
    shard_dirs = sorted(glob.glob(os.path.join(path, "shard_*")))
    if not shard_dirs:
        shard_dirs = sorted(glob.glob(os.path.join(path, "shard_batch_*")))

    # Fallback: simple non-sharded load
    if not shard_dirs:
        if os.path.exists(os.path.join(path, "dataset_info.json")):
            print(f"Loading single dataset from {path}")
            return load_from_disk(path)
        raise ValueError(f"No shards found in {path}")

    print(f"Found {len(shard_dirs)} shards. Creating virtual concatenated view...")
    
    # load_from_disk is "lazy". It doesn't read data into RAM.
    # It just maps the file on disk to memory.
    datasets = [load_from_disk(s) for s in shard_dirs]
    
    # Concatenate creates a unified index (0 to 100M) across all shards
    combined_dataset = concatenate_datasets(datasets)

    # Pre-compute length for fast group_by_length
    if "length" not in combined_dataset.column_names:
        print("Pre-computing sequence lengths for fast group_by_length...")
        combined_dataset = combined_dataset.map(
            lambda examples: {"length": [len(x) for x in examples["input_ids"]]},
            batched=True,
            num_proc=os.cpu_count() or 1,
            desc="Computing lengths"
        )
    
    print(f"✓ Successfully loaded {len(combined_dataset):,} sequences.")
    return combined_dataset

def setup_model(model_name: str, lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05, 
                target_modules: str = "all-linear", quantization: str = "none"):
    """Load ESM2 model with LoRA, Flash Attention 2, and optional Quantization."""
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Quantization Config
    bnb_config = None
    if quantization == "4bit":
        print("✓ Using 4-bit quantization (QLoRA)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        print("✓ Using 8-bit quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    # Load with Flash Attention 2 if available
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    try:
        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            **model_kwargs,
        )
        print("✓ Using Flash Attention 2")
    except Exception as e:
        print(f"Flash Attention 2 not available: {e}")
        model = AutoModelForMaskedLM.from_pretrained(model_name, **model_kwargs)
        print("Using standard attention (SDPA)")

    # Log attention implementation for debugging
    if hasattr(model.config, "_attn_implementation"):
        print(f"  Attention impl: {model.config._attn_implementation}")

    # Prepare for k-bit training if quantized
    if quantization != "none":
        model = prepare_model_for_kbit_training(model)

    # Determine target modules
    if target_modules == "all-linear":
        # Target all linear layers in ESM2 (query, key, value, dense, intermediate, output)
        # Note: 'dense' appears in multiple places. 
        # ESM2 usually has: 
        # - attention.self.query, key, value
        # - attention.output.dense
        # - intermediate.dense
        # - output.dense
        targets = ["query", "key", "value", "dense"] 
    else:
        targets = target_modules.split(",")

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=targets,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def compute_metrics(eval_preds):
    """Compute accuracy on masked tokens."""
    predictions, labels = eval_preds
    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)

    mask = labels != -100
    if mask.sum() == 0:
        return {"accuracy": 0.0}

    accuracy = (predictions[mask] == labels[mask]).astype(float).mean()
    return {"accuracy": float(accuracy)}


def parse_args():
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Model
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="all-linear", 
                        help="Modules to target with LoRA. 'all-linear' targets all linear layers.")

    # Quantization
    parser.add_argument("--quantization", type=str, default="none", choices=["none", "4bit", "8bit"],
                        help="Quantization precision for QLoRA")

    # Training
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")

    # Logging
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--report_to", type=str, default="tensorboard")

    # Advanced
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # torch.compile (PyTorch 2.0+)
    parser.add_argument("--torch_compile", action="store_true", help="Enable torch.compile")
    parser.add_argument(
        "--torch_compile_mode", type=str, default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.set_float32_matmul_precision("high")

    if local_rank == 0:
        print("=" * 60)
        print("ESM2 Fine-tuning with LoRA")
        print("=" * 60)
        print(f"Model: {args.model_name}")
        print(f"Dataset: {args.dataset_path}")
        print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")

    # Load model
    model, tokenizer = setup_model(
        args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        quantization=args.quantization,
    )

    # Apply torch.compile for speedup (PyTorch 2.0+)
    if args.torch_compile:
        if local_rank == 0:
            print(f"Applying torch.compile (mode={args.torch_compile_mode})")
        model = torch.compile(model, mode=args.torch_compile_mode)

    # Load datasets
    train_path = os.path.join(args.dataset_path, "train")
    val_path = os.path.join(args.dataset_path, "val")
    if not os.path.exists(val_path):
        val_path = os.path.join(args.dataset_path, "validation")

    train_dataset = load_dataset_optimized(train_path)
    val_dataset = load_dataset_optimized(val_path)

    train_size = len(train_dataset)
    val_size = len(val_dataset)
    if local_rank == 0:
        print(f"Train: {train_size:,} sequences")
        print(f"Val: {val_size:,} sequences")


    # Data collator
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
        pad_to_multiple_of=8,
    )

    # Training arguments - optimized for p4d.24xlarge
    training_args_dict = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        group_by_length=True,
        bf16=True,
        bf16_full_eval=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Evaluation & saving
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        # Note: load_best_model_at_end not compatible with IterableDataset
        # Logging
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        # Dataloader - optimized for p4d.24xlarge (96 vCPU)
        dataloader_num_workers=8,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        dataloader_prefetch_factor=4,
        # DDP settings
        ddp_find_unused_parameters=False,
        # IterableDataset config - each process fetches its own batch
        accelerator_config={"dispatch_batches": False},
        # Misc
        seed=args.seed,
        remove_unused_columns=True,
    )

    # IterableDataset requires max_steps (no len() available)
    if args.max_steps:
        training_args_dict["max_steps"] = args.max_steps
    else:
        # Calculate steps for desired epochs
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        effective_batch = args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps
        steps_per_epoch = train_size // effective_batch
        training_args_dict["max_steps"] = steps_per_epoch * args.num_epochs
        if local_rank == 0:
            print(f"Calculated max_steps: {training_args_dict['max_steps']:,} ({args.num_epochs} epochs)")

    training_args = TrainingArguments(**training_args_dict)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=lambda logits, _: logits.argmax(dim=-1),
    )

    # Train
    if local_rank == 0:
        print("\nStarting training...")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save
    final_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    if local_rank == 0:
        print(f"\nModel saved to: {final_dir}")

        # Final eval
        results = trainer.evaluate()
        print("\nFinal Results:")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
