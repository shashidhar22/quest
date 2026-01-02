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
"""

import argparse
import glob
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


class DataCollatorForMLMDynamic:
    """
    Simple MLM collator with dynamic padding to batch max length.
    More efficient than packing when sequences have similar lengths.
    """

    def __init__(self, tokenizer, mlm_probability: float = 0.15, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = len(tokenizer)
        self.pad_to_multiple_of = pad_to_multiple_of

        special_ids = [
            tokenizer.pad_token_id,
            tokenizer.cls_token_id,
            tokenizer.eos_token_id,
            tokenizer.sep_token_id,
            tokenizer.unk_token_id,
        ]
        self.special_token_ids = torch.tensor([x for x in special_ids if x is not None], dtype=torch.long)

    def __call__(self, examples):
        """Dynamically pad to batch max length using vectorized operations."""
        # Convert all sequences to tensors
        sequences = []
        for ex in examples:
            ids = ex.get("input_ids", ex) if isinstance(ex, dict) else ex
            if isinstance(ids, list):
                sequences.append(torch.tensor(ids, dtype=torch.long))
            else:
                sequences.append(ids.long() if ids.dtype != torch.long else ids)

        # Use pad_sequence for efficient padding (C++ implementation)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=self.pad_token_id
        )

        # Create attention mask using broadcasting (no Python loop)
        lengths = torch.tensor([len(s) for s in sequences])
        max_len = input_ids.size(1)
        attention_mask = (torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)).long()

        # Pad to multiple of 8 for tensor core efficiency
        if self.pad_to_multiple_of and max_len % self.pad_to_multiple_of != 0:
            pad_len = self.pad_to_multiple_of - (max_len % self.pad_to_multiple_of)
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=self.pad_token_id)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=0)

        # Apply MLM masking
        input_ids, labels = self._apply_mlm_masking(input_ids, attention_mask)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _apply_mlm_masking(self, input_ids, attention_mask):
        """Apply MLM masking using vectorized operations."""
        labels = input_ids.clone()

        # Single random tensor for efficiency
        rand_vals = torch.rand(input_ids.shape[0], input_ids.shape[1], 2)
        rand_mask = rand_vals[..., 0]
        mask_type = rand_vals[..., 1]

        special_mask = torch.isin(input_ids, self.special_token_ids)
        valid_mask = (attention_mask == 1) & ~special_mask
        masked_indices = valid_mask & (rand_mask < self.mlm_probability)
        labels[~masked_indices] = -100

        input_ids = input_ids.clone()
        input_ids[masked_indices & (mask_type < 0.8)] = self.mask_token_id

        random_token_indices = masked_indices & (mask_type >= 0.8) & (mask_type < 0.9)
        if random_token_indices.any():
            input_ids[random_token_indices] = torch.randint(
                self.vocab_size, (random_token_indices.sum(),), dtype=torch.long
            )

        return input_ids, labels


class EarlyStopping:
    """Early stopping with patience and minimum delta."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = float('inf')
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_score - val_loss > self.min_delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


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

        # Load model with Flash Attention 2
        try:
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_name,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
            )
            if self._is_main_process():
                print("✓ Using Flash Attention 2")
        except Exception as e:
            if self._is_main_process():
                print(f"Flash Attention 2 not available: {e}")
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
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
                target_modules=["query", "key", "value", "dense"],
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

    def _setup_dataloaders(self):
        """Setup train and validation dataloaders."""
        train_path = os.path.join(self.config["dataset_path"], "train")
        val_path = os.path.join(self.config["dataset_path"], "val")
        if not os.path.exists(val_path):
            val_path = os.path.join(self.config["dataset_path"], "validation")

        self.train_dataset = self._load_dataset(train_path)
        self.val_dataset = self._load_dataset(val_path)

        if self._is_main_process():
            print(f"Train: {len(self.train_dataset):,} sequences")
            print(f"Val: {len(self.val_dataset):,} sequences")

        # Choose collator based on use_packing flag
        if self.config.get("use_packing", False):
            if self._is_main_process():
                print("Using packing collator (sequences packed to max_seq_length)")
            self.data_collator = DataCollatorForMLMWithPacking(
                tokenizer=self.tokenizer,
                max_seq_length=self.config.get("max_seq_length", 1024),
                mlm_probability=self.config.get("mlm_probability", 0.15),
            )
        else:
            if self._is_main_process():
                print("Using dynamic padding collator (pad to batch max length)")
            self.data_collator = DataCollatorForMLMDynamic(
                tokenizer=self.tokenizer,
                mlm_probability=self.config.get("mlm_probability", 0.15),
            )

        # Samplers
        if self.is_distributed:
            train_sampler = DistributedSampler(self.train_dataset, shuffle=True, drop_last=True)
            val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None

        # DataLoaders optimized for p4d.24xlarge
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.get("batch_size", 16),
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=12,
            prefetch_factor=8,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
            collate_fn=self.data_collator,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.get("batch_size", 16),
            sampler=val_sampler,
            num_workers=12,
            prefetch_factor=8,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.data_collator,
        )

    def _load_dataset(self, path: str):
        """Load sharded dataset."""
        shard_dirs = sorted(glob.glob(os.path.join(path, "shard_*")))
        if not shard_dirs:
            shard_dirs = sorted(glob.glob(os.path.join(path, "shard_batch_*")))

        if shard_dirs:
            datasets = [load_from_disk(s) for s in shard_dirs]
            return concatenate_datasets(datasets)
        return load_from_disk(path)

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
            val_loss, val_acc = self._validate()
            if self._is_main_process():
                print(f"\nEpoch {epoch+1} complete - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

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

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
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
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
                            "train/learning_rate": self.scheduler.get_last_lr()[0],
                            "train/global_step": self.global_step,
                        })

                # Evaluation
                if self.global_step % eval_steps == 0:
                    val_loss, val_acc = self._validate()
                    if self._is_main_process():
                        print(f"\nStep {self.global_step} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                        if self.use_wandb:
                            wandb.log({
                                "eval/loss": val_loss,
                                "eval/accuracy": val_acc,
                                "eval/global_step": self.global_step,
                            })

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint(epoch, is_best=True)

                    self.model.train()

                # Checkpointing
                if self.global_step % save_steps == 0:
                    self._save_checkpoint(epoch)

        pbar.close()

    def _validate(self):
        """Validation loop."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_masked = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", disable=not self._is_main_process()):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

                total_loss += outputs.loss.item()
                predictions = outputs.logits.argmax(dim=-1)
                mask = labels != -100
                total_correct += (predictions[mask] == labels[mask]).sum().item()
                total_masked += mask.sum().item()

        avg_loss = total_loss / len(self.val_loader)
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
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

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
                        help="Use sequence packing (slower for uniform-length data)")

    # Optimization
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--use_amp", action="store_true", default=True)

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
