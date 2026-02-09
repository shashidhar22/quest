#!/usr/bin/env python3
"""
TCR Cross-Encoder Trainer

Cross-encoder approach for TCR alpha-beta pairing classification.
Instead of encoding chains separately (dual encoder), this concatenates
alpha and beta sequences with a separator and uses self-attention to
learn interaction patterns.

Architecture:
    [CLS] Alpha - Beta [EOS] -> ESM2 -> CLS embedding -> Classifier -> Binary score

Usage:
    python scripts/training/tcr_cross_encoder_trainer.py \
        --data_path data/deduplicated/full/foundation_permutations/ \
        --output_dir ./output/tcr_cross_encoder \
        --overfit_check \
        --batch_size 16
"""

import argparse
import glob
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Backend abstraction for hardware-agnostic training
from quest.training.backends import AcceleratorBackend, get_backend
from quest.training.base_trainer import BaseTCRTrainer
from quest.models.cross_encoder import TCRCrossEncoder, CrossEncoderBCELoss

# Optional: LoRA support
try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


# =============================================================================
# Dataset
# =============================================================================


class TCRParquetDataset(Dataset):
    """
    Dataset for loading TCR pairs from parquet files.

    Loads sequences filtered by permutation_key (e.g., 'tra_trb'),
    splits them into alpha and beta chains, and supports train/val/test splits.
    """

    _cache: Dict[str, Tuple[List[str], List[str]]] = {}

    def __init__(
        self,
        data_path: str,
        permutation_keys: List[str] = ["tra_trb"],
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        local_rank: int = 0,
    ):
        self.split = split
        is_main = local_rank == 0

        cache_key = f"{data_path}:{','.join(sorted(permutation_keys))}"

        if cache_key in TCRParquetDataset._cache:
            if is_main:
                print(f"Using cached data for {split} split...")
            alpha_seqs, beta_seqs = TCRParquetDataset._cache[cache_key]
        else:
            import pyarrow.parquet as pq

            parquet_files = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
            if not parquet_files:
                raise ValueError(f"No parquet files found in {data_path}")

            if is_main:
                print(f"Loading {len(parquet_files)} parquet files...")

            alpha_seqs = []
            beta_seqs = []

            iterator = tqdm(parquet_files, desc="Loading", disable=not is_main)
            for pf in iterator:
                table = pq.read_table(pf, columns=["permutation_key", "sequence"])
                df = table.to_pandas()
                df = df[df["permutation_key"].isin(permutation_keys)]

                for seq in df["sequence"].values:
                    parts = seq.split(" ")
                    if len(parts) >= 2:
                        alpha_seqs.append(parts[0])
                        beta_seqs.append(parts[1])

            if not alpha_seqs:
                raise ValueError(f"No sequences found with permutation_keys: {permutation_keys}")

            if is_main:
                print(f"Loaded {len(alpha_seqs):,} pairs with keys: {permutation_keys}")

            TCRParquetDataset._cache[cache_key] = (alpha_seqs, beta_seqs)

        # Create deterministic train/val/test split
        n_total = len(alpha_seqs)
        np.random.seed(seed)
        indices = np.random.permutation(n_total)

        train_end = int(n_total * train_ratio)
        val_end = int(n_total * (train_ratio + val_ratio))

        if split == "train":
            selected_indices = indices[:train_end]
        elif split == "val":
            selected_indices = indices[train_end:val_end]
        elif split == "test":
            selected_indices = indices[val_end:]
        else:
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split}")

        self.alpha_seqs = [alpha_seqs[i] for i in selected_indices]
        self.beta_seqs = [beta_seqs[i] for i in selected_indices]

        if is_main:
            print(f"{split.capitalize()} set: {len(self.alpha_seqs):,} pairs")

    def __len__(self) -> int:
        return len(self.alpha_seqs)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return {
            "alpha_seq": self.alpha_seqs[idx],
            "beta_seq": self.beta_seqs[idx],
        }


# =============================================================================
# Data Collator
# =============================================================================


class CrossEncoderCollator:
    """
    Collates TCR pairs for cross-encoder training.

    Creates positive pairs (matched alpha-beta) and in-batch negative pairs
    (mismatched alpha-beta) by concatenating sequences with a separator.
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 320,
        neg_ratio: int = 3,
        separator: str = "-",
        var_region_len: int = 150,
    ):
        """
        Args:
            tokenizer: ESM2 tokenizer
            max_length: Maximum sequence length after tokenization
            neg_ratio: Number of negative pairs per positive pair
            separator: Token to separate alpha and beta sequences
            var_region_len: Truncate sequences to this length (0 = no truncation)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.neg_ratio = neg_ratio
        self.separator = separator
        self.var_region_len = var_region_len

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of examples into positive and negative pairs.

        Args:
            examples: List of {"alpha_seq": str, "beta_seq": str}

        Returns:
            dict with input_ids, attention_mask, labels
        """
        # Extract and optionally truncate sequences
        if self.var_region_len > 0:
            alpha_seqs = [ex["alpha_seq"][:self.var_region_len] for ex in examples]
            beta_seqs = [ex["beta_seq"][:self.var_region_len] for ex in examples]
        else:
            alpha_seqs = [ex["alpha_seq"] for ex in examples]
            beta_seqs = [ex["beta_seq"] for ex in examples]

        batch_size = len(examples)

        # Positive pairs: matched alpha-beta
        pos_seqs = [
            f"{alpha_seqs[i]}{self.separator}{beta_seqs[i]}"
            for i in range(batch_size)
        ]

        # Negative pairs: in-batch mismatched alpha-beta
        neg_seqs = []
        for i in range(batch_size):
            # Get indices of other betas (not the true match)
            neg_indices = [j for j in range(batch_size) if j != i]
            # Sample neg_ratio negatives per positive
            sampled = random.sample(neg_indices, min(self.neg_ratio, len(neg_indices)))
            for j in sampled:
                neg_seqs.append(f"{alpha_seqs[i]}{self.separator}{beta_seqs[j]}")

        # Combine all sequences
        all_seqs = pos_seqs + neg_seqs
        labels = torch.cat([
            torch.ones(len(pos_seqs)),
            torch.zeros(len(neg_seqs)),
        ])

        # Tokenize
        encoded = self.tokenizer(
            all_seqs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }


# =============================================================================
# Trainer
# =============================================================================


class TCRCrossEncoderTrainer(BaseTCRTrainer):
    """
    Trainer for cross-encoder TCR alpha-beta pairing.

    Extends BaseTCRTrainer with cross-encoder-specific functionality:
    - Binary classification for alpha-beta pair matching
    - In-batch negative sampling
    - Support for CUDA and Trainium backends
    """

    def __init__(self, config: Dict[str, Any], backend: Optional[AcceleratorBackend] = None):
        """
        Initialize cross-encoder trainer.

        Args:
            config: Training configuration
            backend: Accelerator backend (auto-detected if None)
        """
        # Initialize base trainer
        super().__init__(config, backend)

        # Setup loss function with positive class weighting
        pos_weight = self.config.get("pos_weight")
        if pos_weight is None:
            pos_weight = self.config.get("neg_ratio", 3)
        self.criterion = CrossEncoderBCELoss(pos_weight=pos_weight)

    def _create_model(self) -> nn.Module:
        """Create TCRCrossEncoder with backend-appropriate settings."""
        model_name = self.config.get("model_name", "facebook/esm2_t33_650M_UR50D")
        self._log(f"Loading model: {model_name}")

        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get backend-specific kwargs
        load_kwargs = self._get_model_load_kwargs()

        model = TCRCrossEncoder(
            model_name=model_name,
            hidden_dim=self.config.get("hidden_dim", 256),
            dropout=self.config.get("dropout", 0.1),
            pooling=self.config.get("pooling", "cls"),
            **load_kwargs,
        )

        # Apply LoRA if requested
        if self.config.get("use_lora", False) and PEFT_AVAILABLE:
            lora_config = LoraConfig(
                r=self.config.get("lora_r", 16),
                lora_alpha=self.config.get("lora_alpha", 32),
                target_modules=["query", "key", "value"],
                lora_dropout=self.config.get("lora_dropout", 0.05),
                bias="none",
            )
            model.encoder = get_peft_model(model.encoder, lora_config)
            self._log("Applied LoRA to encoder")
            if self._is_main_process():
                model.encoder.print_trainable_parameters()
        elif not self.config.get("use_lora", False):
            # Freeze encoder if LoRA is not used
            for param in model.encoder.parameters():
                param.requires_grad = False
            self._log("Encoder frozen (no LoRA)")

        return model

    def _create_datasets(self) -> Tuple[Dataset, Optional[Dataset]]:
        """Create train and validation datasets."""
        train_dataset = TCRParquetDataset(
            data_path=self.config["data_path"],
            permutation_keys=self.config.get("permutation_keys", ["tra_trb"]),
            split="train",
            local_rank=self.local_rank,
        )

        val_dataset = TCRParquetDataset(
            data_path=self.config["data_path"],
            permutation_keys=self.config.get("permutation_keys", ["tra_trb"]),
            split="val",
            local_rank=self.local_rank,
        )

        return train_dataset, val_dataset

    def _create_collator(self):
        """Create data collator for cross-encoder."""
        return CrossEncoderCollator(
            tokenizer=self.tokenizer,
            max_length=self.config.get("max_length", 320),
            neg_ratio=self.config.get("neg_ratio", 3),
            separator=self.config.get("separator", "-"),
            var_region_len=self.config.get("var_region_len", 150),
        )

    def _compute_loss(self, model: nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute loss for cross-encoder binary classification."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = model(input_ids, attention_mask)
        loss_dict = self.criterion(logits, labels)

        return {
            "loss": loss_dict["loss"],
            "accuracy": loss_dict["accuracy"],
            "pos_accuracy": loss_dict["pos_accuracy"],
            "neg_accuracy": loss_dict["neg_accuracy"],
        }

    def _setup_optimizer(self, lr: Optional[float] = None):
        """Initialize optimizer."""
        learning_rate = lr or self.config.get("learning_rate", 1e-3)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=self.config.get("weight_decay", 0.01),
        )
        self._log(f"Optimizer: AdamW, lr={learning_rate}, weight_decay={self.config.get('weight_decay', 0.01)}")

    def _setup_scheduler(self, num_training_steps: int):
        """Initialize learning rate scheduler with warmup."""
        warmup_ratio = self.config.get("warmup_ratio", 0.1)
        warmup_steps = int(num_training_steps * warmup_ratio)

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self._log(f"Scheduler: Cosine with {warmup_steps} warmup steps ({warmup_ratio*100:.0f}%)")

    def overfit_single_batch(self):
        """
        Overfit check: Train on a single batch to verify the pipeline works.

        For cross-encoder with binary classification, this should easily reach
        100% accuracy within 100-200 steps if everything is working correctly.
        """
        self._log("\n" + "=" * 60)
        self._log("SINGLE BATCH OVERFIT CHECK")
        self._log("=" * 60)
        self._log("Running overfit check on a single batch...")
        self._log("Expected: Loss should decrease to ~0, accuracy should reach 100%")
        self._log("=" * 60 + "\n")

        num_steps = self.config.get("overfit_steps", 500)
        lr = self.config.get("overfit_lr", self.config.get("learning_rate", 1e-3))

        self._setup_optimizer(lr=lr)
        self.model.train()

        # Get a single batch
        batch = next(iter(self.train_loader))
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        actual_lr = lr or self.config.get("learning_rate", 1e-3)
        self._log(f"[Batch Info]")
        self._log(f"  Input shape: {input_ids.shape}")
        self._log(f"  Num positive: {(labels == 1).sum().item()}")
        self._log(f"  Num negative: {(labels == 0).sum().item()}")
        self._log(f"  Learning rate: {actual_lr}")
        self._log("")

        # Log some example sequences
        self._log("[Example sequences (first 3)]")
        for i in range(min(3, len(input_ids))):
            seq = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
            label = "POSITIVE" if labels[i] == 1 else "NEGATIVE"
            self._log(f"  [{i}] ({label}): {seq[:100]}...")
        self._log("")

        initial_loss = None
        initial_acc = None

        for step in range(num_steps):
            self.optimizer.zero_grad()

            with self.backend.autocast_context():
                logits = self.model(input_ids, attention_mask)
                loss_dict = self.criterion(logits, labels)

            loss = loss_dict["loss"]
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )

            self.backend.optimizer_step(self.optimizer, self.model)

            if step == 0:
                initial_loss = loss.item()
                initial_acc = loss_dict["accuracy"].item()

            if step % 10 == 0 or step == num_steps - 1:
                self._log(
                    f"Step {step:4d}: loss={loss.item():.4f}, "
                    f"acc={loss_dict['accuracy'].item():.4f}, "
                    f"pos_acc={loss_dict['pos_accuracy'].item():.4f}, "
                    f"neg_acc={loss_dict['neg_accuracy'].item():.4f}, "
                    f"grad_norm={grad_norm:.4f}"
                )

        final_loss = loss.item()
        final_acc = loss_dict["accuracy"].item()

        self._log("\n" + "=" * 60)
        self._log("OVERFIT CHECK RESULTS")
        self._log("=" * 60)
        self._log(f"Initial loss: {initial_loss:.4f}")
        self._log(f"Final loss:   {final_loss:.4f}")
        self._log(f"Loss reduction: {100 * (initial_loss - final_loss) / initial_loss:.1f}%")
        self._log(f"Initial accuracy: {initial_acc:.4f}")
        self._log(f"Final accuracy:   {final_acc:.4f}")

        if final_acc > 0.95 and final_loss < 0.1:
            self._log("\nSUCCESS: Model can overfit a single batch!")
            self._log("The cross-encoder architecture is working correctly.")
        else:
            self._log("\nFAILURE: Model cannot overfit a single batch!")
            self._log("  Possible issues:")
            self._log("  - Gradients not flowing (check requires_grad)")
            self._log("  - Learning rate too low")
            self._log("  - Architecture problem")

        # Diagnostic info
        self._log("\n[Trainable params by component]")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._log(f"  {name}: {param.numel():,}")

        self._log("=" * 60 + "\n")

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, desc: str = "Evaluating") -> Dict[str, float]:
        """
        Evaluate model on a dataloader.

        Args:
            dataloader: DataLoader to evaluate on
            desc: Description for progress bar

        Returns:
            dict with average metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_pos_acc = 0.0
        total_neg_acc = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            with self.backend.autocast_context():
                logits = self.model(input_ids, attention_mask)
                loss_dict = self.criterion(logits, labels)

            total_loss += loss_dict["loss"].item()
            total_acc += loss_dict["accuracy"].item()
            total_pos_acc += loss_dict["pos_accuracy"].item()
            total_neg_acc += loss_dict["neg_accuracy"].item()
            num_batches += 1

        self.model.train()

        return {
            "loss": total_loss / num_batches,
            "accuracy": total_acc / num_batches,
            "pos_accuracy": total_pos_acc / num_batches,
            "neg_accuracy": total_neg_acc / num_batches,
        }

    def _save_checkpoint(self, path: Path, epoch: int, step: int, metrics: Dict[str, float]):
        """Save model checkpoint using backend-appropriate method."""
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if hasattr(self, "scheduler") else None,
            "metrics": metrics,
            "config": self.config,
        }
        self.backend.save_checkpoint(checkpoint, str(path), self._is_main_process())
        self._log(f"Saved checkpoint to {path}")

    def train(self):
        """Full training loop with validation, checkpointing, and early stopping."""
        self._log("\n" + "=" * 60)
        self._log("STARTING FULL TRAINING")
        self._log("=" * 60 + "\n")

        num_epochs = self.config.get("num_epochs", 3)
        grad_accum_steps = self.config.get("gradient_accumulation_steps", 1)
        eval_steps = self.config.get("eval_steps", 1000)
        save_steps = self.config.get("save_steps", 5000)
        log_steps = self.config.get("log_steps", 100)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        patience = self.config.get("patience", 3)

        num_training_steps = len(self.train_loader) * num_epochs // grad_accum_steps
        self._setup_optimizer()
        self._setup_scheduler(num_training_steps)

        self._log(f"Training configuration:")
        self._log(f"  Epochs: {num_epochs}")
        self._log(f"  Batch size: {self.config.get('batch_size', 16)}")
        self._log(f"  Gradient accumulation steps: {grad_accum_steps}")
        self._log(f"  Effective batch size: {self.config.get('batch_size', 16) * grad_accum_steps}")
        self._log(f"  Total training steps: {num_training_steps}")
        self._log(f"  Eval steps: {eval_steps}")
        self._log(f"  Save steps: {save_steps}")
        self._log(f"  Early stopping patience: {patience}")
        self._log("")

        best_val_loss = float("inf")
        best_val_acc = 0.0
        patience_counter = 0
        global_step = 0
        running_loss = 0.0
        running_acc = 0.0
        running_batches = 0

        self.model.train()

        for epoch in range(num_epochs):
            self._log(f"\n{'='*60}")
            self._log(f"Epoch {epoch + 1}/{num_epochs}")
            self._log(f"{'='*60}")

            epoch_loss = 0.0
            epoch_acc = 0.0
            epoch_steps = 0

            total_opt_steps = len(self.train_loader) // grad_accum_steps
            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}",
                total=len(self.train_loader),
            )

            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                with self.backend.autocast_context():
                    logits = self.model(input_ids, attention_mask)
                    loss_dict = self.criterion(logits, labels)
                    loss = loss_dict["loss"] / grad_accum_steps

                loss.backward()

                epoch_loss += loss_dict["loss"].item()
                epoch_acc += loss_dict["accuracy"].item()
                epoch_steps += 1

                if (batch_idx + 1) % grad_accum_steps == 0:
                    # Accumulate metrics per optimizer step (averaged over grad_accum_steps)
                    running_loss += loss_dict["loss"].item()
                    running_acc += loss_dict["accuracy"].item()
                    running_batches += 1

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=max_grad_norm
                    )
                    self.backend.optimizer_step(self.optimizer, self.model)
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    # Logging
                    if global_step % log_steps == 0:
                        avg_loss = running_loss / running_batches
                        avg_acc = running_acc / running_batches
                        lr = self.scheduler.get_last_lr()[0]
                        self._log(
                            f"Step {global_step}: loss={avg_loss:.4f}, acc={avg_acc:.4f}, "
                            f"lr={lr:.2e}, grad_norm={grad_norm:.4f}"
                        )
                        running_loss = 0.0
                        running_acc = 0.0
                        running_batches = 0

                    # Evaluation
                    if global_step % eval_steps == 0:
                        self._log(f"\n--- Validation at step {global_step} ---")
                        val_metrics = self.evaluate(self.val_loader, desc="Validating")
                        self._log(
                            f"Val: loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.4f}, "
                            f"pos_acc={val_metrics['pos_accuracy']:.4f}, neg_acc={val_metrics['neg_accuracy']:.4f}"
                        )

                        # Check for improvement
                        if val_metrics["loss"] < best_val_loss:
                            best_val_loss = val_metrics["loss"]
                            best_val_acc = val_metrics["accuracy"]
                            patience_counter = 0
                            self._save_checkpoint(
                                self.output_dir / "best_model.pt",
                                epoch, global_step, val_metrics
                            )
                            self._log(f"New best model! Val loss: {best_val_loss:.4f}")
                        else:
                            patience_counter += 1
                            self._log(f"No improvement. Patience: {patience_counter}/{patience}")

                        if patience_counter >= patience:
                            self._log(f"\nEarly stopping triggered after {global_step} steps")
                            break

                        self.model.train()

                    # Save checkpoint
                    if global_step % save_steps == 0:
                        self._save_checkpoint(
                            self.output_dir / f"checkpoint_step_{global_step}.pt",
                            epoch, global_step, {"train_loss": epoch_loss / epoch_steps}
                        )

                progress_bar.set_postfix({
                    "step": f"{global_step}/{total_opt_steps}",
                    "loss": f"{epoch_loss/epoch_steps:.4f}",
                    "acc": f"{epoch_acc/epoch_steps:.4f}",
                })

            if patience_counter >= patience:
                break

            # End of epoch evaluation
            self._log(f"\n--- End of Epoch {epoch + 1} ---")
            self._log(f"Train: loss={epoch_loss/epoch_steps:.4f}, acc={epoch_acc/epoch_steps:.4f}")

            val_metrics = self.evaluate(self.val_loader, desc="End-of-epoch validation")
            self._log(
                f"Val: loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.4f}, "
                f"pos_acc={val_metrics['pos_accuracy']:.4f}, neg_acc={val_metrics['neg_accuracy']:.4f}"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_val_acc = val_metrics["accuracy"]
                patience_counter = 0
                self._save_checkpoint(
                    self.output_dir / "best_model.pt",
                    epoch, global_step, val_metrics
                )
                self._log(f"New best model! Val loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1

        # Final summary
        self._log("\n" + "=" * 60)
        self._log("TRAINING COMPLETE")
        self._log("=" * 60)
        self._log(f"Best validation loss: {best_val_loss:.4f}")
        self._log(f"Best validation accuracy: {best_val_acc:.4f}")
        self._log(f"Total steps: {global_step}")
        self._log(f"Best model saved to: {self.output_dir / 'best_model.pt'}")

        # Save final model
        self._save_checkpoint(
            self.output_dir / "final_model.pt",
            num_epochs - 1, global_step, {"final": True}
        )


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="TCR Cross-Encoder Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to directory containing parquet files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for logs and checkpoints")
    parser.add_argument("--permutation_keys", type=str, nargs="+", default=["tra_trb"],
                        help="Permutation keys to filter sequences")
    parser.add_argument("--max_length", type=int, default=320,
                        help="Maximum tokenized sequence length")
    parser.add_argument("--var_region_len", type=int, default=150,
                        help="Truncate sequences to this length (0 = no truncation)")

    # Model
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D",
                        help="Pretrained ESM2 model name")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension of classification head")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate in classification head")
    parser.add_argument("--pooling", type=str, default="cls",
                        choices=["cls", "mean", "attention"],
                        help="Pooling strategy for sequence representation")

    # Cross-encoder specific
    parser.add_argument("--neg_ratio", type=int, default=3,
                        help="Number of negative pairs per positive pair")
    parser.add_argument("--separator", type=str, default="-",
                        help="Token to separate alpha and beta sequences")
    parser.add_argument("--pos_weight", type=float, default=None,
                        help="Positive class weight for BCE loss (default: neg_ratio)")

    # LoRA
    parser.add_argument("--use_lora", action="store_true", default=False,
                        help="Apply LoRA to encoder")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")

    # Training
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers")

    # Training schedule
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Ratio of warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")

    # Logging and checkpointing
    parser.add_argument("--log_steps", type=int, default=100,
                        help="Log every N steps")
    parser.add_argument("--eval_steps", type=int, default=1000,
                        help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=5000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (number of evals without improvement)")

    # Overfit check
    parser.add_argument("--overfit_check", action="store_true",
                        help="Run single-batch overfit check")
    parser.add_argument("--overfit_steps", type=int, default=500,
                        help="Number of steps for overfit check")
    parser.add_argument("--overfit_lr", type=float, default=None,
                        help="Learning rate for overfit check (default: learning_rate)")

    # Hardware backend
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "cuda", "xla", "neuron", "trainium"],
                        help="Hardware backend (auto detects XLA/CUDA)")

    args = parser.parse_args()
    config = vars(args)

    # Initialize backend
    backend = get_backend(args.backend)
    print(f"Using backend: {backend.name}")

    # Create trainer with backend
    trainer = TCRCrossEncoderTrainer(config, backend=backend)
    trainer.setup_model()

    if args.overfit_check:
        trainer.setup_data(include_val=False)
        trainer.overfit_single_batch()
    else:
        trainer.setup_data(include_val=True)
        trainer.train()


if __name__ == "__main__":
    main()
