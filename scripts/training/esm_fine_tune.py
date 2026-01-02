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
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DataCollatorForMLMWithPacking:
    """
    Optimized data collator using pure tensor operations.
    Packs multiple sequences into fixed-length samples to eliminate padding.

    Key optimizations:
    1. No Python list operations - keeps tensors as tensors
    2. torch.isin() for vectorized special token detection
    3. Single random tensor for all MLM masking decisions
    4. Pre-allocated output buffers
    """

    def __init__(
        self,
        tokenizer,
        max_seq_length: int = 1024,
        mlm_probability: float = 0.15,
        pad_to_multiple_of: int = 8,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = len(tokenizer)

        # Cache special token IDs as tensor for vectorized isin() operation
        special_ids = [
            tokenizer.pad_token_id,
            tokenizer.cls_token_id,
            tokenizer.eos_token_id,
            tokenizer.sep_token_id,
            tokenizer.unk_token_id,
        ]
        special_ids = [x for x in special_ids if x is not None]
        self.special_token_ids = torch.tensor(special_ids, dtype=torch.long)

    def __call__(self, examples):
        """Pack sequences and apply MLM masking using pure tensor operations."""
        # Step 1: Extract sequences as tensors (avoid list conversion)
        sequences = []
        for ex in examples:
            if isinstance(ex, dict):
                ids = ex.get("input_ids", ex)
            else:
                ids = ex
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
            elif ids.dtype != torch.long:
                ids = ids.long()
            sequences.append(ids)

        # Step 2: Pack sequences using tensor operations
        packed_ids, packed_mask = self._pack_sequences(sequences)

        # Step 3: Apply MLM masking with optimized single-pass approach
        input_ids, labels = self._apply_mlm_masking(packed_ids, packed_mask)

        return {
            "input_ids": input_ids,
            "attention_mask": packed_mask,
            "labels": labels,
        }

    def _pack_sequences(self, sequences):
        """Pack sequences into fixed-length samples using pre-allocated tensors."""
        # Get sequence lengths
        lengths = [len(s) for s in sequences]
        total_tokens = sum(lengths)

        # Estimate number of packs needed
        num_packs = max(1, (total_tokens + self.max_seq_length - 1) // self.max_seq_length)

        # Pre-allocate output tensors
        packed_ids = torch.full(
            (num_packs, self.max_seq_length),
            self.pad_token_id,
            dtype=torch.long
        )
        packed_mask = torch.zeros(num_packs, self.max_seq_length, dtype=torch.long)

        # Pack sequences
        pack_idx = 0
        pos = 0

        for seq in sequences:
            seq_len = len(seq)

            # Check if we need a new pack
            if pos + seq_len > self.max_seq_length:
                pack_idx += 1
                pos = 0

                # Expand if needed
                if pack_idx >= num_packs:
                    extra = torch.full(
                        (1, self.max_seq_length),
                        self.pad_token_id,
                        dtype=torch.long
                    )
                    packed_ids = torch.cat([packed_ids, extra], dim=0)
                    packed_mask = torch.cat([
                        packed_mask,
                        torch.zeros(1, self.max_seq_length, dtype=torch.long)
                    ], dim=0)
                    num_packs += 1

            # Place sequence directly into tensor
            packed_ids[pack_idx, pos:pos + seq_len] = seq
            packed_mask[pack_idx, pos:pos + seq_len] = 1
            pos += seq_len

        # Trim unused packs
        if pack_idx + 1 < num_packs:
            packed_ids = packed_ids[:pack_idx + 1]
            packed_mask = packed_mask[:pack_idx + 1]

        return packed_ids, packed_mask

    def _apply_mlm_masking(self, input_ids, attention_mask):
        """
        Apply MLM masking using single-pass vectorized operations.

        Optimizations:
        1. Single torch.rand() call for all masking decisions
        2. Vectorized special token exclusion using torch.isin()
        3. Reuse random tensor for 80/10/10 split
        """
        labels = input_ids.clone()

        # Single random tensor for all masking decisions
        rand_mask = torch.rand(input_ids.shape)

        # Vectorized special token detection (much faster than loop)
        special_mask = torch.isin(input_ids, self.special_token_ids)

        # Valid positions: not padding and not special tokens
        valid_mask = (attention_mask == 1) & ~special_mask

        # Determine which tokens to mask (15% of valid tokens)
        masked_indices = valid_mask & (rand_mask < self.mlm_probability)

        # Set labels: -100 for non-masked tokens
        labels[~masked_indices] = -100

        # Single random tensor for 80/10/10 decision
        mask_type = torch.rand(input_ids.shape)

        # 80% -> [MASK]
        mask_token_indices = masked_indices & (mask_type < 0.8)
        input_ids = input_ids.clone()  # Don't modify original
        input_ids[mask_token_indices] = self.mask_token_id

        # 10% -> random token (0.8 <= prob < 0.9)
        random_token_indices = masked_indices & (mask_type >= 0.8) & (mask_type < 0.9)
        if random_token_indices.any():
            random_tokens = torch.randint(
                self.vocab_size,
                (random_token_indices.sum(),),
                dtype=torch.long
            )
            input_ids[random_token_indices] = random_tokens

        # 10% -> keep original (mask_type >= 0.9) - no action needed

        return input_ids, labels


def load_dataset_optimized(path: str, sort_by_length: bool = False, length_shuffle_chunk_size: int = 0):
    """
    Load sharded dataset and keep it as a Map-style dataset.

    Args:
        path: Path to dataset directory
        sort_by_length: If True, sort dataset by sequence length (reduces padding overhead)
        length_shuffle_chunk_size: If > 0, shuffle in chunks of this size after sorting
                                   (mitigates gradient bias from strict length ordering)
    """
    from datasets import concatenate_datasets, load_from_disk
    import glob
    import os
    import numpy as np

    # Check for shards
    shard_dirs = sorted(glob.glob(os.path.join(path, "shard_*")))
    if not shard_dirs:
        shard_dirs = sorted(glob.glob(os.path.join(path, "shard_batch_*")))

    # Filter out empty or invalid shards
    valid_shards = []
    for s in shard_dirs:
        if os.path.exists(os.path.join(s, "dataset_info.json")) or os.path.exists(os.path.join(s, "state.json")):
            valid_shards.append(s)
        else:
            print(f"Skipping invalid/empty shard: {s}")
    shard_dirs = valid_shards

    # Fallback: simple non-sharded load
    if not shard_dirs:
        if os.path.exists(os.path.join(path, "dataset_info.json")):
            print(f"Loading single dataset from {path}")
            combined_dataset = load_from_disk(path)
        else:
            raise ValueError(f"No shards found in {path}")
    else:
        print(f"Found {len(shard_dirs)} shards. Creating virtual concatenated view...")

        # load_from_disk is "lazy". It doesn't read data into RAM.
        # It just maps the file on disk to memory.
        datasets = [load_from_disk(s) for s in shard_dirs]

        # Concatenate creates a unified index (0 to 100M) across all shards
        combined_dataset = concatenate_datasets(datasets)

    # Pre-compute length if not present
    if "length" not in combined_dataset.column_names:
        print("Pre-computing sequence lengths...")
        combined_dataset = combined_dataset.map(
            lambda examples: {"length": [len(x) for x in examples["input_ids"]]},
            batched=True,
            num_proc=os.cpu_count() or 1,
            desc="Computing lengths"
        )

    # Sort by length to minimize padding (much faster than group_by_length)
    if sort_by_length:
        import pyarrow.compute as pc

        print("Sorting dataset by sequence length (Arrow-optimized)...")

        # Get sorted indices without reordering data (fast, zero-copy)
        sorted_indices = pc.sort_indices(combined_dataset.data.table, sort_keys=[("length", "ascending")])
        sorted_indices = sorted_indices.to_numpy()

        # Optionally shuffle in chunks to avoid gradient bias
        if length_shuffle_chunk_size > 0:
            print(f"Shuffling in chunks of {length_shuffle_chunk_size:,} to mitigate gradient bias...")
            n_chunks = (len(sorted_indices) + length_shuffle_chunk_size - 1) // length_shuffle_chunk_size
            chunks = np.array_split(sorted_indices, n_chunks)
            np.random.shuffle(chunks)
            sorted_indices = np.concatenate(chunks)

        # Single select call with final indices
        combined_dataset = combined_dataset.select(sorted_indices)
        print("✓ Dataset sorted by length")

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
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
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

    # Dataloader
    parser.add_argument("--dataloader_num_workers", type=int, default=12, help="Number of subprocesses to use for data loading (12 optimal for p4d.24xlarge)")

    # Length-based sorting (faster alternative to group_by_length)
    parser.add_argument("--sort_by_length", action="store_true",
                        help="Sort dataset by sequence length to minimize padding (faster than group_by_length)")
    parser.add_argument("--length_shuffle_chunk_size", type=int, default=0,
                        help="After sorting by length, shuffle in chunks of this size to reduce gradient bias. 0 = no chunk shuffling.")

    # torch.compile (PyTorch 2.0+)
    parser.add_argument("--torch_compile", action="store_true", help="Enable torch.compile")
    parser.add_argument(
        "--torch_compile_mode", type=str, default="max-autotune",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode"
    )

    # Sequence packing (eliminates padding overhead)
    parser.add_argument("--use_packing", action="store_true",
                        help="Enable sequence packing to eliminate padding (significant speedup)")
    parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="Maximum sequence length for packing")

    # Early stopping
    parser.add_argument("--early_stopping", action="store_true",
                        help="Enable early stopping based on validation loss")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="Number of eval steps with no improvement before stopping")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0,
                        help="Minimum improvement required to count as improvement")

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
        if args.sort_by_length:
            chunk_info = f", chunk_shuffle={args.length_shuffle_chunk_size}" if args.length_shuffle_chunk_size > 0 else ""
            print(f"Sort by length: enabled{chunk_info}")
        if args.use_packing:
            print(f"Sequence packing: enabled (max_seq_length={args.max_seq_length})")
        if args.early_stopping:
            print(f"Early stopping: enabled (patience={args.early_stopping_patience})")

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
        if args.quantization == "none":
            if local_rank == 0:
                print(f"Applying torch.compile (mode={args.torch_compile_mode})")
            model = torch.compile(model, mode=args.torch_compile_mode)
        elif local_rank == 0:
            print("Skipping torch.compile because quantization is enabled (not compatible)")

    # Load datasets
    train_path = os.path.join(args.dataset_path, "train")
    val_path = os.path.join(args.dataset_path, "val")
    if not os.path.exists(val_path):
        val_path = os.path.join(args.dataset_path, "validation")

    train_dataset = load_dataset_optimized(
        train_path,
        sort_by_length=args.sort_by_length,
        length_shuffle_chunk_size=args.length_shuffle_chunk_size
    )
    val_dataset = load_dataset_optimized(val_path)  # Don't sort validation

    train_size = len(train_dataset)
    val_size = len(val_dataset)
    if local_rank == 0:
        print(f"Train: {train_size:,} sequences")
        print(f"Val: {val_size:,} sequences")


    # Data collator
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    if args.use_packing:
        if local_rank == 0:
            print(f"✓ Using sequence packing (max_seq_length={args.max_seq_length})")
        data_collator = DataCollatorForMLMWithPacking(
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            mlm_probability=args.mlm_probability,
            pad_to_multiple_of=8,
        )
    else:
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
        group_by_length=False,
        bf16=True,
        bf16_full_eval=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Evaluation & saving
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Logging
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        # Dataloader - optimized for p4d.24xlarge (96 vCPU)
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_persistent_workers=True if args.dataloader_num_workers > 0 else False,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        dataloader_prefetch_factor=8 if args.dataloader_num_workers > 0 else None,
        # DDP settings
        ddp_find_unused_parameters=False,
        # IterableDataset config - each process fetches its own batch
        accelerator_config={"dispatch_batches": False},
        # Misc
        seed=args.seed,
        remove_unused_columns=False,
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

    # Ensure save_steps aligns with eval_steps for load_best_model_at_end
    if training_args_dict.get("load_best_model_at_end", False):
        if args.save_steps != args.eval_steps:
            if local_rank == 0:
                print(f"⚠ Aligning save_steps ({args.save_steps}) with eval_steps ({args.eval_steps}) for best model tracking")
            training_args_dict["save_steps"] = args.eval_steps

    training_args = TrainingArguments(**training_args_dict)

    if local_rank == 0:
        print(f"Checkpoints: saving every {training_args.save_steps} steps, keeping {training_args.save_total_limit} best")

    # Setup callbacks
    callbacks = []
    if args.early_stopping:
        if local_rank == 0:
            print(f"✓ Early stopping enabled (patience={args.early_stopping_patience}, threshold={args.early_stopping_threshold})")
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=lambda logits, _: logits.argmax(dim=-1),
        callbacks=callbacks if callbacks else None,
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
