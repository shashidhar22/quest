#!/usr/bin/env python3
"""
Native PyTorch ESM2 MLM Fine-tuning with DDP.
Optimized for p4d.24xlarge (8x A100 40GB).

This script provides a pure PyTorch training loop without HuggingFace Trainer
overhead, while maintaining all the features: LoRA, Flash Attention 2,
gradient checkpointing, checkpointing, and early stopping.

Usage:
    torchrun --nproc_per_node=8 esm_native_trainer.py \
        --dataset_path /path/to/data \
        --output_dir ./output

Features:
- Pure PyTorch DDP (no HuggingFace Trainer overhead)
- LoRA via PEFT
- Flash Attention 2
- Gradient checkpointing
- BF16 mixed precision
- Checkpointing and early stopping
- Optimized DataCollator with pure tensor operations
- Length-bucketed batching (groups similar-length sequences to minimize padding)
"""

import argparse
import glob
import math
import os
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import GradScaler, autocast
from datasets import load_from_disk, concatenate_datasets
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
import wandb

from quest.training.samplers import LengthBucketSampler, DistributedLengthBucketSampler
from quest.training.collators import DataCollatorForMLMDynamic, DataCollatorForMLMWithVarlen
from quest.training.callbacks import EarlyStopping

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DataCollatorForMLMWithPacking:
    """
    Optimized data collator using pure tensor operations.
    Packs multiple sequences into fixed-length samples to eliminate padding.
    """

    def __init__(
        self,
        tokenizer,
        max_seq_length: int = 1024,
        mlm_probability: float = 0.15,
        separator_token_id: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
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
        if separator_token_id is not None:
            special_ids.append(separator_token_id)
        special_ids = [x for x in special_ids if x is not None]
        self.special_token_ids = torch.tensor(special_ids, dtype=torch.long)

    def __call__(self, examples):
        """Pack sequences and apply MLM masking using pure tensor operations."""
        sequences = []
        for ex in examples:
            ids = ex.get("input_ids", ex) if isinstance(ex, dict) else ex
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
            elif ids.dtype != torch.long:
                ids = ids.long()
            sequences.append(ids)

        packed_ids, packed_mask = self._pack_sequences(sequences)
        input_ids, labels = self._apply_mlm_masking(packed_ids, packed_mask)

        return {
            "input_ids": input_ids,
            "attention_mask": packed_mask,
            "labels": labels,
        }

    def _pack_sequences(self, sequences):
        """Pack sequences into fixed-length samples using pre-allocated tensors."""
        lengths = [len(s) for s in sequences]
        total_tokens = sum(lengths)
        num_packs = max(1, (total_tokens + self.max_seq_length - 1) // self.max_seq_length)

        packed_ids = torch.full((num_packs, self.max_seq_length), self.pad_token_id, dtype=torch.long)
        packed_mask = torch.zeros(num_packs, self.max_seq_length, dtype=torch.long)

        pack_idx = 0
        pos = 0

        for seq in sequences:
            seq_len = len(seq)
            if pos + seq_len > self.max_seq_length:
                pack_idx += 1
                pos = 0
                if pack_idx >= num_packs:
                    packed_ids = torch.cat([packed_ids, torch.full((1, self.max_seq_length), self.pad_token_id, dtype=torch.long)], dim=0)
                    packed_mask = torch.cat([packed_mask, torch.zeros(1, self.max_seq_length, dtype=torch.long)], dim=0)
                    num_packs += 1

            packed_ids[pack_idx, pos:pos + seq_len] = seq
            packed_mask[pack_idx, pos:pos + seq_len] = 1
            pos += seq_len

        if pack_idx + 1 < num_packs:
            packed_ids = packed_ids[:pack_idx + 1]
            packed_mask = packed_mask[:pack_idx + 1]

        return packed_ids, packed_mask

    def _apply_mlm_masking(self, input_ids, attention_mask):
        """Apply MLM masking using single-pass vectorized operations."""
        labels = input_ids.clone()
        rand_mask = torch.rand(input_ids.shape)
        special_mask = torch.isin(input_ids, self.special_token_ids)
        valid_mask = (attention_mask == 1) & ~special_mask
        masked_indices = valid_mask & (rand_mask < self.mlm_probability)
        labels[~masked_indices] = -100

        mask_type = torch.rand(input_ids.shape)
        input_ids = input_ids.clone()
        input_ids[masked_indices & (mask_type < 0.8)] = self.mask_token_id

        random_token_indices = masked_indices & (mask_type >= 0.8) & (mask_type < 0.9)
        if random_token_indices.any():
            input_ids[random_token_indices] = torch.randint(self.vocab_size, (random_token_indices.sum(),), dtype=torch.long)

        return input_ids, labels


class NativeESMTrainer:
    """
    Native PyTorch trainer for ESM2 MLM fine-tuning.

    Key features:
    - DDP for multi-GPU training
    - LoRA for efficient fine-tuning
    - Flash Attention 2
    - Gradient checkpointing
    - BF16 mixed precision
    """

    def __init__(self, config: dict):
        self.config = config
        self.is_distributed = "LOCAL_RANK" in os.environ

        # Setup distributed training
        if self.is_distributed:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.global_rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])

            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=timedelta(minutes=30)
            )
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.local_rank = 0
            self.global_rank = 0
            self.world_size = 1
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set precision
        torch.set_float32_matmul_precision("high")

        # Initialize components
        self._setup_model()
        self._setup_dataloaders()
        self._setup_optimizer()

        # Training state
        self.scaler = GradScaler() if config.get("use_amp", True) else None
        self.early_stopping = EarlyStopping(
            patience=config.get("early_stopping_patience", 5),
            min_delta=config.get("early_stopping_threshold", 0.0)
        ) if config.get("early_stopping", True) else None

        self.global_step = 0
        self.best_val_loss = float('inf')

        # Setup wandb (main process only)
        self.use_wandb = config.get("report_to") == "wandb" and self._is_main_process()
        if self.use_wandb:
            wandb.init(
                project=config.get("wandb_project", "esm-finetuning"),
                name=config.get("wandb_run_name"),
                config=config,
            )

    def _is_main_process(self) -> bool:
        return self.global_rank == 0

    def _setup_model(self):
        """Initialize model with LoRA and Flash Attention 2."""
        model_name = self.config["model_name"]

        if self._is_main_process():
            print(f"Loading model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        # Choose attention implementation
        # varlen uses 4D mask which requires SDPA (FA2 doesn't support arbitrary 4D masks)
        use_varlen = self.config.get("use_varlen", False)

        if use_varlen:
            # Use SDPA for 4D attention mask support
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_name,
                attn_implementation="sdpa",
                dtype=torch.bfloat16,
            )
            if self._is_main_process():
                print("✓ Using SDPA (for 4D block diagonal mask)")
        else:
            # Use Flash Attention 2 for standard attention
            try:
                self.model = AutoModelForMaskedLM.from_pretrained(
                    model_name,
                    attn_implementation="flash_attention_2",
                    dtype=torch.bfloat16,
                )
                if self._is_main_process():
                    print("✓ Using Flash Attention 2")
            except Exception as e:
                if self._is_main_process():
                    print(f"Flash Attention 2 not available: {e}")
                self.model = AutoModelForMaskedLM.from_pretrained(
                    model_name,
                    dtype=torch.bfloat16,
                )

        # Enable gradient checkpointing
        if self.config.get("gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if self._is_main_process():
                print("✓ Gradient checkpointing enabled")

        # Apply LoRA
        if self.config.get("use_lora", True):
            lora_config = LoraConfig(
                task_type=TaskType.TOKEN_CLS,
                r=self.config.get("lora_r", 16),
                lora_alpha=self.config.get("lora_alpha", 32),
                lora_dropout=self.config.get("lora_dropout", 0.05),
                target_modules=["query", "key", "value", "dense", "intermediate.dense", "output.dense"],
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)
            if self._is_main_process():
                self.model.print_trainable_parameters()

        # Move to device and wrap with DDP
        self.model = self.model.to(self.device)
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=False,
            )

        # Optional: Apply torch.compile for speedup
        if self.config.get("use_compile", False):
            if self._is_main_process():
                print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
            if self._is_main_process():
                print("✓ Model compiled")

    def _setup_dataloaders(self):
        """Setup train and validation dataloaders."""
        # Resolve train path
        if self.config.get("train_path"):
            train_path = self.config["train_path"]
        elif self.config.get("dataset_path"):
            train_path = os.path.join(self.config["dataset_path"], "train")
        else:
            raise ValueError("Must provide either --train_path or --dataset_path")

        # Resolve val path
        if self.config.get("val_path"):
            val_path = self.config["val_path"]
        elif self.config.get("dataset_path"):
            val_path = os.path.join(self.config["dataset_path"], "val")
            if not os.path.exists(val_path):
                val_path = os.path.join(self.config["dataset_path"], "validation")
        else:
            raise ValueError("Must provide either --val_path or --dataset_path")

        self.train_dataset = self._load_dataset(train_path)

        # Load all val datasets — pass val_max_samples through so parquet loading
        # only reads the row groups needed (avoids materializing a 71M-row val).
        val_max = int(self.config.get("val_max_samples") or 0)
        self.val_datasets = {"val": self._load_dataset(val_path, max_rows=val_max)}

        # Additional val splits
        val_parent = os.path.dirname(val_path)
        for split_name in (self.config.get("val_splits") or []):
            split_path = os.path.join(val_parent, split_name)
            if os.path.exists(split_path):
                self.val_datasets[split_name] = self._load_dataset(split_path, max_rows=val_max)
            elif self._is_main_process():
                print(f"WARNING: Val split {split_path} not found, skipping")

        # If val_max applied via .select() to non-parquet (HF dataset) val sets,
        # do it here as a fallback.
        if val_max > 0:
            for name, ds in list(self.val_datasets.items()):
                if len(ds) > val_max:
                    self.val_datasets[name] = ds.select(range(val_max))
                    if self._is_main_process():
                        print(f"Val ({name}): subsampled to {val_max:,} of {len(ds):,}")

        if self._is_main_process():
            print(f"Train: {len(self.train_dataset):,} sequences")
            for name, ds in self.val_datasets.items():
                print(f"Val ({name}): {len(ds):,} sequences")

        # Choose collator based on packing flags
        if self.config.get("use_varlen", False):
            if self._is_main_process():
                print("Using varlen collator (packing with proper block diagonal masking)")
            self.data_collator = DataCollatorForMLMWithVarlen(
                tokenizer=self.tokenizer,
                max_seq_length=self.config.get("max_seq_length", 1024),
                mlm_probability=self.config.get("mlm_probability", 0.15),
                separator_token_id=30,  # ESM2 uses '-' (dash, ID 30) as separator
            )
        elif self.config.get("use_packing", False):
            if self._is_main_process():
                print("WARNING: Using legacy packing (has cross-sequence attention bug)")
                print("         Consider using --use_varlen instead")
            self.data_collator = DataCollatorForMLMWithPacking(
                tokenizer=self.tokenizer,
                max_seq_length=self.config.get("max_seq_length", 1024),
                mlm_probability=self.config.get("mlm_probability", 0.15),
                separator_token_id=30,  # ESM2 uses '-' (dash, ID 30) as separator
            )
        else:
            if self._is_main_process():
                print("Using dynamic padding collator (pad to batch max length)")
            self.data_collator = DataCollatorForMLMDynamic(
                tokenizer=self.tokenizer,
                mlm_probability=self.config.get("mlm_probability", 0.15),
                separator_token_id=30,  # ESM2 uses '-' (dash, ID 30) as separator
            )

        # Samplers - use length bucketing to minimize padding waste
        bucket_boundaries = self.config.get("bucket_boundaries", [128, 256, 384, 512, 768])

        if self.config.get("use_length_bucketing", True) and "length" in self.train_dataset.column_names:
            if self._is_main_process():
                print(f"Using length-bucketed sampling with boundaries: {bucket_boundaries}")

            # Get lengths as numpy array for sampler
            train_lengths = np.array(self.train_dataset["length"])

            if self.is_distributed:
                train_sampler = DistributedLengthBucketSampler(
                    lengths=train_lengths,
                    batch_size=self.config.get("batch_size", 16),
                    bucket_boundaries=bucket_boundaries,
                    shuffle=True,
                    drop_last=True,
                    num_replicas=self.world_size,
                    rank=self.global_rank,
                )
            else:
                train_sampler = LengthBucketSampler(
                    lengths=train_lengths,
                    batch_size=self.config.get("batch_size", 16),
                    bucket_boundaries=bucket_boundaries,
                    shuffle=True,
                    drop_last=True,
                )
        else:
            # Fallback to standard sampling
            if self._is_main_process():
                print("Using standard random sampling (length bucketing disabled)")
            if self.is_distributed:
                train_sampler = DistributedSampler(self.train_dataset, shuffle=True, drop_last=True)
            else:
                train_sampler = None

        # DataLoaders. Workers fork from the main process, so each holds a
        # reference to the dataset (memmap-backed; shared via kernel page cache).
        # Keep num_workers modest — sequences here are tiny so collation is cheap
        # and CPU-side throughput is not the bottleneck on small ESM-2 models.
        train_workers = int(self.config.get("dataloader_num_workers", 4))
        val_workers = int(self.config.get("val_dataloader_num_workers", 2))
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.get("batch_size", 16),
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=train_workers,
            prefetch_factor=2 if train_workers > 0 else None,
            pin_memory=True,
            persistent_workers=train_workers > 0,
            drop_last=True,
            collate_fn=self.data_collator,
        )

        # Create val loaders for each split
        self.val_loaders = {}
        for name, dataset in self.val_datasets.items():
            val_sampler = DistributedSampler(dataset, shuffle=False) if self.is_distributed else None
            self.val_loaders[name] = DataLoader(
                dataset,
                batch_size=self.config.get("batch_size", 16),
                sampler=val_sampler,
                num_workers=val_workers,
                prefetch_factor=2 if val_workers > 0 else None,
                pin_memory=True,
                persistent_workers=val_workers > 0,
                collate_fn=self.data_collator,
            )

        # Keep self.val_loader pointing to unified for backward compat
        self.val_loader = self.val_loaders["val"]

    def _load_parquet_input_ids(self, files, max_rows: int = 0):
        """Stream parquet → on-disk Arrow IPC cache → memmap-loaded HF Dataset.

        Why not pq.read_table: that materializes the full input_ids column
        in the parent process heap (~340 B/row × 100M rows ≈ 34 GB). When the
        DataLoader forks num_workers persistent processes, every page touched
        by Python refcount/metadata writes triggers copy-on-write — host RSS
        grows unbounded over the run and OOM-kills the box.

        With this path the table is written once to an Arrow IPC stream file
        (low-RSS row-group-by-row-group conversion) and reloaded via
        Dataset.from_file(in_memory=False), which mmap's the file. All forked
        DataLoader workers share the same kernel page cache → no COW, RSS
        stays at ~tens of MB regardless of dataset size.

        Cache lives next to the source parquets in `.input_ids_arrow_cache/`.
        Cache key includes file path, mtime, size, and max_rows so it
        auto-invalidates on any source change.
        """
        import hashlib
        import pyarrow as pa
        import pyarrow.parquet as pq
        from datasets import Dataset

        cache_dir = os.path.join(os.path.dirname(files[0]), ".input_ids_arrow_cache")
        os.makedirs(cache_dir, exist_ok=True)

        sig_parts = []
        for fp in files:
            st = os.stat(fp)
            sig_parts.append(f"{os.path.abspath(fp)}:{st.st_mtime_ns}:{st.st_size}")
        sig_parts.append(f"max_rows={int(max_rows or 0)}")
        digest = hashlib.sha256("|".join(sig_parts).encode()).hexdigest()[:16]
        arrow_path = os.path.join(cache_dir, f"input_ids_{digest}.arrow")

        if not os.path.exists(arrow_path):
            if self._is_main_process():
                print(f"  Building Arrow IPC cache: {arrow_path}")
                print(f"  (one-time conversion, low RSS — streams row groups)")
            first_pf = pq.ParquetFile(files[0])
            schema = pa.schema([first_pf.schema_arrow.field("input_ids")])
            tmp_path = arrow_path + ".tmp"
            collected = 0
            try:
                with pa.OSFile(tmp_path, "wb") as sink:
                    with pa.ipc.new_stream(sink, schema) as writer:
                        for fp in files:
                            if max_rows and collected >= max_rows:
                                break
                            pf = pq.ParquetFile(fp)
                            for rg_idx in range(pf.num_row_groups):
                                if max_rows and collected >= max_rows:
                                    break
                                rg = pf.read_row_group(rg_idx, columns=["input_ids"])
                                if max_rows and collected + rg.num_rows > max_rows:
                                    rg = rg.slice(0, max_rows - collected)
                                writer.write_table(rg)
                                collected += rg.num_rows
                os.rename(tmp_path, arrow_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise
            if self._is_main_process():
                size_gb = os.path.getsize(arrow_path) / 1e9
                print(f"  Wrote {collected:,} rows ({size_gb:.2f} GB on disk)")

        if self._is_main_process():
            size_gb = os.path.getsize(arrow_path) / 1e9
            print(f"  Memmap-loading Arrow IPC: {arrow_path} ({size_gb:.2f} GB)")
        return Dataset.from_file(arrow_path, in_memory=False)

    def _load_dataset(self, path: str, compute_lengths: bool = True, max_rows: int = 0):
        """Load dataset. Supports: .parquet file, dir of *.parquet files,
        HF dataset dir, or dir of shard_* HF subdirs.

        For parquet inputs, only the `input_ids` column is read (all other
        columns in the foundation tokenized parquets are unused by MLM training).
        This avoids the slow / bloated HF parquet loader that materializes every
        column (which inflates 4.5 GB → 70+ GB on disk for the 10M file).
        """
        if path.endswith(".parquet") and os.path.isfile(path):
            dataset = self._load_parquet_input_ids([path], max_rows=max_rows)
        elif (
            os.path.isdir(path)
            and glob.glob(os.path.join(path, "*.parquet"))
            and not glob.glob(os.path.join(path, "shard_*"))
            and not glob.glob(os.path.join(path, "shard_batch_*"))
        ):
            files = sorted(glob.glob(os.path.join(path, "*.parquet")))
            dataset = self._load_parquet_input_ids(files, max_rows=max_rows)
        else:
            shard_dirs = sorted(glob.glob(os.path.join(path, "shard_*")))
            if not shard_dirs:
                shard_dirs = sorted(glob.glob(os.path.join(path, "shard_batch_*")))

            if shard_dirs:
                datasets = [load_from_disk(s) for s in shard_dirs]
                dataset = concatenate_datasets(datasets)
            else:
                dataset = load_from_disk(path)

        # Compute lengths on-the-fly if needed for bucketing
        if compute_lengths and self.config.get("use_length_bucketing", True):
            if "length" not in dataset.column_names:
                if self._is_main_process():
                    print(f"Computing sequence lengths for {path}...")
                # Use efficient batched map with multiple processes
                if "attention_mask" in dataset.column_names:
                    dataset = dataset.map(
                        lambda x: {"length": sum(x["attention_mask"])},
                        num_proc=1,
                        desc="Computing lengths",
                    )
                else:
                    dataset = dataset.map(
                        lambda x: {"length": len(x["input_ids"])},
                        num_proc=1,
                        desc="Computing lengths",
                    )

        return dataset

    def _setup_optimizer(self):
        """Setup optimizer with fused AdamW."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 2e-4),
            weight_decay=self.config.get("weight_decay", 0.01),
            fused=True,
        )

        # Calculate training steps
        grad_accum = self.config.get("gradient_accumulation_steps", 8)
        num_epochs = self.config.get("num_epochs", 3)
        steps_per_epoch = len(self.train_loader) // grad_accum
        total_steps = steps_per_epoch * num_epochs
        warmup_steps = int(total_steps * self.config.get("warmup_ratio", 0.1))

        # Cosine scheduler with warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train(self):
        """Main training loop."""
        start_epoch = 0
        if self.config.get("resume_from_checkpoint"):
            start_epoch = self._load_checkpoint(self.config["resume_from_checkpoint"])

        num_epochs = self.config.get("num_epochs", 3)
        eval_steps = self.config.get("eval_steps", 500)
        save_steps = self.config.get("save_steps", 500)
        logging_steps = self.config.get("logging_steps", 10)
        grad_accum = self.config.get("gradient_accumulation_steps", 8)

        if self._is_main_process():
            print(f"\nStarting training for {num_epochs} epochs...")
            print(f"  Batch size: {self.config.get('batch_size', 16)}")
            print(f"  Gradient accumulation: {grad_accum}")
            print(f"  Effective batch size: {self.config.get('batch_size', 16) * self.world_size * grad_accum}")

        for epoch in range(start_epoch, num_epochs):
            if self.is_distributed:
                self.train_loader.sampler.set_epoch(epoch)

            self._train_one_epoch(epoch, logging_steps, eval_steps, save_steps, grad_accum)

            # Epoch-end validation
            val_loss, val_acc, all_metrics = self._validate_all()
            if self._is_main_process():
                print(f"\nEpoch {epoch+1} complete - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                if self.use_wandb:
                    log_dict = {"eval/global_step": self.global_step, "eval/epoch": epoch + 1}
                    for name, m in all_metrics.items():
                        prefix = "eval" if name == "val" else f"eval/{name}"
                        log_dict[f"{prefix}/loss"] = m["loss"]
                        log_dict[f"{prefix}/perplexity"] = m["perplexity"]
                        log_dict[f"{prefix}/accuracy"] = m["accuracy"]
                    wandb.log(log_dict)

            # Early stopping
            if self.early_stopping and self.early_stopping(val_loss):
                if self._is_main_process():
                    print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Cleanup wandb
        if self.use_wandb:
            wandb.finish()

        self._save_final_model()

    def _train_one_epoch(self, epoch, logging_steps, eval_steps, save_steps, grad_accum):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        accum_loss = 0.0

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch+1}",
            disable=not self._is_main_process(),
        )

        self.optimizer.zero_grad()

        for step, batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            # Position IDs for varlen attention (resets for each packed sequence)
            position_ids = batch.get("position_ids")
            if position_ids is not None:
                position_ids = position_ids.to(self.device)

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    labels=labels,
                )
                loss = outputs.loss / grad_accum

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item()

            if (step + 1) % grad_accum == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                total_loss += accum_loss
                pbar.set_postfix({"loss": f"{accum_loss:.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"})
                accum_loss = 0.0

                # Logging
                if self.global_step % logging_steps == 0 and self._is_main_process():
                    avg_loss = total_loss / (self.global_step % (len(self.train_loader) // grad_accum) or 1)
                    if self.use_wandb:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/perplexity": math.exp(min(avg_loss, 20)),
                            "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                            "train/learning_rate": self.scheduler.get_last_lr()[0],
                            "train/global_step": self.global_step,
                        })

                # Evaluation
                if self.global_step % eval_steps == 0:
                    val_loss, val_acc, all_metrics = self._validate_all()
                    if self._is_main_process():
                        print(f"\nStep {self.global_step} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                        if self.use_wandb:
                            log_dict = {"eval/global_step": self.global_step}
                            for name, m in all_metrics.items():
                                prefix = "eval" if name == "val" else f"eval/{name}"
                                log_dict[f"{prefix}/loss"] = m["loss"]
                                log_dict[f"{prefix}/perplexity"] = m["perplexity"]
                                log_dict[f"{prefix}/accuracy"] = m["accuracy"]
                            wandb.log(log_dict)

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint(epoch, is_best=True)

                    self.model.train()

                # Checkpointing
                if self.global_step % save_steps == 0:
                    self._save_checkpoint(epoch)

        pbar.close()

    def _validate_all(self):
        """Run validation on all val splits. Returns (unified_loss, unified_acc, all_metrics_dict)."""
        all_metrics = {}
        for name, loader in self.val_loaders.items():
            loss, acc = self._validate(loader)
            perplexity = math.exp(min(loss, 20))  # cap to avoid overflow
            all_metrics[name] = {"loss": loss, "accuracy": acc, "perplexity": perplexity}
        unified = all_metrics["val"]
        return unified["loss"], unified["accuracy"], all_metrics

    def _validate(self, loader=None):
        """Validation loop."""
        if loader is None:
            loader = self.val_loader
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_masked = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Validating", disable=not self._is_main_process()):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                # Position IDs for varlen attention
                position_ids = batch.get("position_ids")
                if position_ids is not None:
                    position_ids = position_ids.to(self.device)

                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        labels=labels,
                    )

                total_loss += outputs.loss.item()
                predictions = outputs.logits.argmax(dim=-1)
                mask = labels != -100
                total_correct += (predictions[mask] == labels[mask]).sum().item()
                total_masked += mask.sum().item()

        avg_loss = total_loss / len(loader)
        accuracy = total_correct / total_masked if total_masked > 0 else 0.0

        # Aggregate across processes
        if self.is_distributed:
            metrics = torch.tensor([avg_loss, accuracy, total_masked], device=self.device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            avg_loss = metrics[0].item() / self.world_size
            accuracy = metrics[1].item() / self.world_size

        return avg_loss, accuracy

    def _save_checkpoint(self, epoch, is_best=False):
        """Save training checkpoint."""
        if not self._is_main_process():
            return

        output_dir = self.config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        model_to_save = self.model.module if self.is_distributed else self.model

        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        # Save step checkpoint
        checkpoint_path = os.path.join(output_dir, f"checkpoint-step-{self.global_step}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = os.path.join(output_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"✓ New best model (val_loss={self.best_val_loss:.4f})")

        # Cleanup old checkpoints (keep 5)
        checkpoints = sorted(glob.glob(os.path.join(output_dir, "checkpoint-step-*.pt")))
        for old_ckpt in checkpoints[:-5]:
            os.remove(old_ckpt)

    def _load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint and return start epoch."""
        map_location = f"cuda:{self.local_rank}"
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        model_to_load = self.model.module if self.is_distributed else self.model
        model_to_load.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        if self._is_main_process():
            print(f"Resumed from step {self.global_step}, epoch {checkpoint['epoch']}")

        return checkpoint["epoch"]
    

    def _save_final_model(self):
        """Save final model."""
        if not self._is_main_process():
            return

        final_dir = os.path.join(self.config["output_dir"], "final_model")
        os.makedirs(final_dir, exist_ok=True)

        model_to_save = self.model.module if self.is_distributed else self.model
        model_to_save.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)

        print(f"\n✓ Final model saved to: {final_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Native PyTorch ESM2 MLM Fine-tuning")

    # Required
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Parent directory containing train/ and val/ subdirs")
    parser.add_argument("--train_path", type=str, default=None,
                        help="Explicit training dataset path (overrides dataset_path/train)")
    parser.add_argument("--val_path", type=str, default=None,
                        help="Explicit validation dataset path (overrides dataset_path/val)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--val_splits", type=str, nargs="*", default=None,
                        help="Additional val split directory names relative to val parent, e.g. val_singles val_pairs")
    parser.add_argument("--val_max_samples", type=int, default=0,
                        help="Cap val set to first N samples (0 = full). Use for periodic eval on huge val sets.")

    # Model
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D")

    # LoRA
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Training
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--use_packing", action="store_true", default=False,
                        help="Use sequence packing (WARNING: has cross-sequence attention bug)")
    parser.add_argument("--use_varlen", action="store_true", default=False,
                        help="Use varlen packing with proper block diagonal masking (recommended)")
    parser.add_argument("--use_compile", action="store_true", default=False,
                        help="Use torch.compile for potential speedup (~10-20%%)")

    # Length bucketing (recommended over packing - avoids cross-attention issues)
    parser.add_argument("--use_length_bucketing", action="store_true", default=True,
                        help="Group similar-length sequences to minimize padding (recommended)")
    parser.add_argument("--no_length_bucketing", action="store_false", dest="use_length_bucketing",
                        help="Disable length bucketing, use random sampling")
    parser.add_argument("--bucket_boundaries", type=int, nargs="+",
                        default=[128, 256, 384, 512, 768],
                        help="Length bucket boundaries (default: [128, 256, 384, 512, 768])")

    # Optimization
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing",
                        help="Disable gradient checkpointing (recommended for small models like ESM2-8M/35M).")
    parser.add_argument("--use_amp", action="store_true", default=True)

    # DataLoader workers (memmap-backed datasets keep RSS low even at 4–8 workers,
    # but defaults stay conservative to avoid host-RAM pressure with persistent forks)
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help="Train DataLoader workers (forked from main; memmap datasets are shared).")
    parser.add_argument("--val_dataloader_num_workers", type=int, default=2,
                        help="Val DataLoader workers (kept low — eval is short).")

    # Logging & Checkpointing
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)

    # Early stopping
    parser.add_argument("--early_stopping", action="store_true", default=True)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0)

    # Resume
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # Wandb
    parser.add_argument("--report_to", type=str, default="wandb", choices=["wandb", "none"])
    parser.add_argument("--wandb_project", type=str, default="esm-finetuning")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    config = vars(args)

    trainer = NativeESMTrainer(config)
    trainer.train()

    # Cleanup distributed
    if trainer.is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
