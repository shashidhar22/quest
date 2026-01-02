#!/usr/bin/env python3
"""
TCR Alpha-Beta Pairing Prediction with Contrastive Learning.

Uses a dual encoder architecture with InfoNCE loss and in-batch negatives
to learn TCR alpha-beta chain pairing from positive pairs only.

Usage:
    torchrun --nproc_per_node=8 tcr_contrastive_trainer.py \
        --dataset_path /path/to/data \
        --output_dir ./output

Features:
- Dual encoder with shared ESM2 backbone + LoRA
- InfoNCE loss with in-batch negatives (handles soft negatives gracefully)
- Symmetric loss (alpha->beta and beta->alpha directions)
- Learnable temperature parameter
- Retrieval metrics: MRR, Recall@K
- DDP for multi-GPU training
"""

import argparse
import glob
import os
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from datasets import concatenate_datasets, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============================================================================
# Data Collator
# =============================================================================


class ContrastivePairCollator:
    """
    Collator for TCR pairing contrastive learning.

    Takes combined sequences [CLS] alpha [SEP] beta [SEP] and splits them
    into separate alpha and beta tensors for the dual encoder.

    Outputs:
    - alpha_input_ids: (batch, max_alpha_len)
    - alpha_attention_mask: (batch, max_alpha_len)
    - beta_input_ids: (batch, max_beta_len)
    - beta_attention_mask: (batch, max_beta_len)
    """

    def __init__(
        self,
        tokenizer,
        pad_to_multiple_of: int = 8,
    ):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def _split_at_sep(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split combined sequence at first SEP token.

        Input:  [CLS] alpha_tokens [SEP] beta_tokens [SEP] [PAD]...
        Output: ([CLS] alpha_tokens [SEP]), ([CLS] beta_tokens [SEP])
        """
        # Find first SEP position
        sep_positions = (input_ids == self.sep_token_id).nonzero(as_tuple=True)[0]

        if len(sep_positions) == 0:
            # No SEP found, treat entire sequence as alpha
            alpha = input_ids
            beta = torch.tensor([self.cls_token_id, self.sep_token_id], dtype=torch.long)
        else:
            first_sep = sep_positions[0].item()

            # Alpha: [CLS] ... [SEP] (includes CLS and first SEP)
            alpha = input_ids[: first_sep + 1]

            # Beta: Need to add [CLS], take tokens after first SEP until second SEP or end
            if len(sep_positions) > 1:
                second_sep = sep_positions[1].item()
                beta_tokens = input_ids[first_sep + 1 : second_sep + 1]
            else:
                # No second SEP, take rest excluding padding
                rest = input_ids[first_sep + 1 :]
                # Remove padding
                non_pad_mask = rest != self.pad_token_id
                if non_pad_mask.any():
                    last_non_pad = non_pad_mask.nonzero(as_tuple=True)[0][-1].item()
                    beta_tokens = rest[: last_non_pad + 1]
                else:
                    beta_tokens = torch.tensor([], dtype=torch.long)

            # Add CLS to beta
            beta = torch.cat(
                [torch.tensor([self.cls_token_id], dtype=torch.long), beta_tokens]
            )

            # Ensure beta ends with SEP
            if len(beta) == 0 or beta[-1] != self.sep_token_id:
                beta = torch.cat([beta, torch.tensor([self.sep_token_id], dtype=torch.long)])

        return alpha, beta

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Process batch of examples.

        Each example should have 'input_ids' with combined alpha-beta sequence.
        """
        alpha_seqs = []
        beta_seqs = []

        for ex in examples:
            input_ids = ex.get("input_ids", ex)
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids, dtype=torch.long)
            elif not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids, dtype=torch.long)

            alpha, beta = self._split_at_sep(input_ids)
            alpha_seqs.append(alpha)
            beta_seqs.append(beta)

        # Pad sequences
        alpha_padded = self._pad_sequences(alpha_seqs)
        beta_padded = self._pad_sequences(beta_seqs)

        # Create attention masks
        alpha_mask = (alpha_padded != self.pad_token_id).long()
        beta_mask = (beta_padded != self.pad_token_id).long()

        return {
            "alpha_input_ids": alpha_padded,
            "alpha_attention_mask": alpha_mask,
            "beta_input_ids": beta_padded,
            "beta_attention_mask": beta_mask,
        }

    def _pad_sequences(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        """Pad sequences to max length in batch, rounded to multiple of 8."""
        max_len = max(len(s) for s in sequences)

        # Round up to multiple of pad_to_multiple_of
        if self.pad_to_multiple_of and max_len % self.pad_to_multiple_of != 0:
            max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        padded = torch.full((len(sequences), max_len), self.pad_token_id, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded[i, : len(seq)] = seq

        return padded


# =============================================================================
# Model Architecture
# =============================================================================


class TCRDualEncoder(nn.Module):
    """
    Dual encoder for TCR alpha-beta pairing prediction.

    Uses shared ESM2 backbone with optional projection head.
    Alpha and beta chains are encoded separately, then similarity is computed.
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        projection_dim: int = 256,
        use_projection: bool = True,
        pooling: str = "cls",
        initial_temperature: float = 0.07,
    ):
        super().__init__()

        # Load ESM2 backbone
        self.encoder = AutoModel.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )

        hidden_dim = self.encoder.config.hidden_size
        self.pooling = pooling
        self.use_projection = use_projection

        if use_projection:
            self.projection = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, projection_dim),
            )

        # Learnable temperature (log scale for stability)
        self.log_temperature = nn.Parameter(
            torch.tensor(np.log(initial_temperature), dtype=torch.float32)
        )

    @property
    def temperature(self) -> torch.Tensor:
        """Get temperature from log scale."""
        return self.log_temperature.exp().clamp(min=0.01, max=1.0)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode sequence to normalized embedding.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            embeddings: (batch, projection_dim) L2-normalized
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if self.pooling == "cls":
            embeddings = outputs.last_hidden_state[:, 0, :]
        else:
            # Mean pooling over non-padding tokens
            hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            embeddings = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        if self.use_projection:
            embeddings = self.projection(embeddings.float())

        # L2 normalize for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def forward(
        self,
        alpha_input_ids: torch.Tensor,
        alpha_attention_mask: torch.Tensor,
        beta_input_ids: torch.Tensor,
        beta_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing similarity matrix.

        Returns:
            alpha_embeddings: (batch, projection_dim)
            beta_embeddings: (batch, projection_dim)
            similarity_matrix: (batch, batch) - S[i,j] = sim(alpha_i, beta_j)
            temperature: scalar
        """
        alpha_emb = self.encode(alpha_input_ids, alpha_attention_mask)
        beta_emb = self.encode(beta_input_ids, beta_attention_mask)

        # Similarity matrix: (batch, batch)
        similarity = torch.matmul(alpha_emb, beta_emb.T)

        return {
            "alpha_embeddings": alpha_emb,
            "beta_embeddings": beta_emb,
            "similarity_matrix": similarity,
            "temperature": self.temperature,
        }


def apply_lora(model: TCRDualEncoder, config: dict) -> TCRDualEncoder:
    """Apply LoRA to the ESM2 encoder."""
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=["query", "key", "value", "dense"],
        bias="none",
    )

    model.encoder = get_peft_model(model.encoder, lora_config)
    return model


# =============================================================================
# Loss Function
# =============================================================================


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for contrastive learning with in-batch negatives.

    Given batch of K positive pairs: [(alpha_1, beta_1), ..., (alpha_K, beta_K)]

    For each alpha_i:
        - Positive: beta_i (diagonal)
        - Negatives: beta_j for all j != i (off-diagonal)

    Loss = -log(exp(sim(alpha_i, beta_i)/tau) / sum_j(exp(sim(alpha_i, beta_j)/tau)))
    """

    def __init__(
        self,
        symmetric: bool = True,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.symmetric = symmetric
        self.label_smoothing = label_smoothing

    def forward(
        self,
        similarity_matrix: torch.Tensor,
        temperature: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute InfoNCE loss.

        Args:
            similarity_matrix: (batch, batch) where S[i,j] = sim(alpha_i, beta_j)
            temperature: learnable temperature parameter

        Returns:
            Dict with loss, accuracy, and component losses
        """
        batch_size = similarity_matrix.size(0)
        device = similarity_matrix.device

        # Scale by temperature
        logits = similarity_matrix / temperature

        # Labels: diagonal is positive
        labels = torch.arange(batch_size, device=device)

        # Cross-entropy loss (alpha -> beta direction)
        loss_a2b = F.cross_entropy(
            logits,
            labels,
            label_smoothing=self.label_smoothing,
        )

        if self.symmetric:
            # Beta -> alpha direction
            loss_b2a = F.cross_entropy(
                logits.T,
                labels,
                label_smoothing=self.label_smoothing,
            )
            loss = (loss_a2b + loss_b2a) / 2
        else:
            loss_b2a = torch.tensor(0.0, device=device)
            loss = loss_a2b

        # Compute accuracy
        with torch.no_grad():
            pred_a2b = logits.argmax(dim=1)
            pred_b2a = logits.T.argmax(dim=1)
            acc_a2b = (pred_a2b == labels).float().mean()
            acc_b2a = (pred_b2a == labels).float().mean()
            accuracy = (acc_a2b + acc_b2a) / 2 if self.symmetric else acc_a2b

        return {
            "loss": loss,
            "loss_a2b": loss_a2b,
            "loss_b2a": loss_b2a,
            "accuracy": accuracy,
            "acc_a2b": acc_a2b,
            "acc_b2a": acc_b2a,
        }


# =============================================================================
# Evaluator
# =============================================================================


class ContrastiveEvaluator:
    """
    Evaluate TCR pairing model with retrieval metrics.

    Metrics:
    - Accuracy@1 (top-1 accuracy)
    - Recall@K (K=1, 5, 10, 20)
    - Mean Reciprocal Rank (MRR)
    """

    @staticmethod
    @torch.no_grad()
    def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        k_values: List[int] = [1, 5, 10, 20],
    ) -> Dict[str, float]:
        """Evaluate model on retrieval task."""
        model.eval()

        all_ranks = []
        total_loss = 0.0
        num_batches = 0

        criterion = InfoNCELoss(symmetric=True)

        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            alpha_ids = batch["alpha_input_ids"].to(device)
            alpha_mask = batch["alpha_attention_mask"].to(device)
            beta_ids = batch["beta_input_ids"].to(device)
            beta_mask = batch["beta_attention_mask"].to(device)

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    alpha_input_ids=alpha_ids,
                    alpha_attention_mask=alpha_mask,
                    beta_input_ids=beta_ids,
                    beta_attention_mask=beta_mask,
                )

            similarity = outputs["similarity_matrix"]
            batch_size = similarity.size(0)

            # Loss
            loss_out = criterion(similarity, outputs["temperature"])
            total_loss += loss_out["loss"].item()
            num_batches += 1

            # Compute ranks
            sorted_indices = similarity.argsort(dim=1, descending=True)

            for i in range(batch_size):
                rank = (sorted_indices[i] == i).nonzero(as_tuple=True)[0]
                if len(rank) > 0:
                    all_ranks.append(rank[0].item() + 1)  # 1-indexed

        if not all_ranks:
            return {"eval_loss": total_loss / max(num_batches, 1)}

        ranks = np.array(all_ranks)

        metrics = {
            "eval_loss": total_loss / num_batches,
            "mrr": float(np.mean(1.0 / ranks)),
            "mean_rank": float(np.mean(ranks)),
            "median_rank": float(np.median(ranks)),
        }

        # Recall@K
        for k in k_values:
            recall_at_k = np.mean(ranks <= k)
            metrics[f"recall@{k}"] = float(recall_at_k)

        metrics["accuracy"] = metrics["recall@1"]

        return metrics


# =============================================================================
# Early Stopping
# =============================================================================


class EarlyStopping:
    """Early stopping with patience."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.mode == "max":
            improved = score - self.best_score > self.min_delta
        else:
            improved = self.best_score - score > self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


# =============================================================================
# Trainer
# =============================================================================


class TCRContrastiveTrainer:
    """
    Native PyTorch trainer for TCR alpha-beta pairing with contrastive learning.
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
                timeout=timedelta(minutes=30),
            )
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.local_rank = 0
            self.global_rank = 0
            self.world_size = 1
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.set_float32_matmul_precision("high")

        # Initialize components
        self._setup_model()
        self._setup_dataloaders()
        self._setup_optimizer()
        self._setup_loss()

        # Training state
        self.scaler = GradScaler() if config.get("use_amp", True) else None
        self.early_stopping = (
            EarlyStopping(
                patience=config.get("early_stopping_patience", 5),
                mode="max",  # Maximize recall@10
            )
            if config.get("early_stopping", True)
            else None
        )

        self.global_step = 0
        self.best_metric = float("-inf")

        # Setup wandb
        self.use_wandb = config.get("report_to") == "wandb" and self._is_main_process()
        if self.use_wandb:
            wandb.init(
                project=config.get("wandb_project", "tcr-pairing"),
                name=config.get("wandb_run_name"),
                config=config,
            )

    def _is_main_process(self) -> bool:
        return self.global_rank == 0

    def _setup_model(self):
        """Initialize dual encoder with LoRA."""
        model_name = self.config["model_name"]

        if self._is_main_process():
            print(f"Loading model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        self.model = TCRDualEncoder(
            model_name=model_name,
            projection_dim=self.config.get("projection_dim", 256),
            use_projection=self.config.get("use_projection", True),
            pooling=self.config.get("pooling", "cls"),
            initial_temperature=self.config.get("temperature", 0.07),
        )

        # Apply LoRA
        if self.config.get("use_lora", True):
            self.model = apply_lora(self.model, self.config)
            if self._is_main_process():
                self.model.encoder.print_trainable_parameters()

        # Enable gradient checkpointing
        if self.config.get("gradient_checkpointing", True):
            self.model.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if self._is_main_process():
                print("✓ Gradient checkpointing enabled")

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
            print(f"Train: {len(self.train_dataset):,} pairs")
            print(f"Val: {len(self.val_dataset):,} pairs")

        self.data_collator = ContrastivePairCollator(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8,
        )

        # Samplers
        if self.is_distributed:
            train_sampler = DistributedSampler(self.train_dataset, shuffle=True, drop_last=True)
            val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.get("batch_size", 256),
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=12,
            prefetch_factor=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
            collate_fn=self.data_collator,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.get("batch_size", 256),
            sampler=val_sampler,
            num_workers=8,
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
            dataset = concatenate_datasets(datasets)
        else:
            dataset = load_from_disk(path)

        return dataset

    def _setup_optimizer(self):
        """Setup optimizer with cosine scheduler."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 1e-4),
            weight_decay=self.config.get("weight_decay", 0.01),
            fused=True,
        )

        # Calculate training steps
        grad_accum = self.config.get("gradient_accumulation_steps", 1)
        num_epochs = self.config.get("num_epochs", 10)
        steps_per_epoch = len(self.train_loader) // grad_accum
        total_steps = steps_per_epoch * num_epochs
        warmup_steps = int(total_steps * self.config.get("warmup_ratio", 0.1))

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _setup_loss(self):
        """Initialize InfoNCE loss."""
        self.criterion = InfoNCELoss(
            symmetric=self.config.get("symmetric_loss", True),
            label_smoothing=self.config.get("label_smoothing", 0.0),
        )

    def train(self):
        """Main training loop."""
        num_epochs = self.config.get("num_epochs", 10)
        eval_steps = self.config.get("eval_steps", 500)
        save_steps = self.config.get("save_steps", 500)
        logging_steps = self.config.get("logging_steps", 10)
        grad_accum = self.config.get("gradient_accumulation_steps", 1)

        if self._is_main_process():
            print(f"\nStarting training for {num_epochs} epochs...")
            print(f"  Batch size: {self.config.get('batch_size', 256)}")
            print(f"  Gradient accumulation: {grad_accum}")
            print(
                f"  Effective batch size: {self.config.get('batch_size', 256) * self.world_size * grad_accum}"
            )

        for epoch in range(num_epochs):
            if self.is_distributed:
                self.train_loader.sampler.set_epoch(epoch)

            self._train_one_epoch(epoch, logging_steps, eval_steps, save_steps, grad_accum)

            # Epoch-end validation
            metrics = self._validate()
            if self._is_main_process():
                print(f"\nEpoch {epoch+1} - Val Loss: {metrics['eval_loss']:.4f}, "
                      f"Recall@10: {metrics.get('recall@10', 0):.4f}, MRR: {metrics.get('mrr', 0):.4f}")

            # Early stopping on Recall@10
            if self.early_stopping:
                if self.early_stopping(metrics.get("recall@10", 0)):
                    if self._is_main_process():
                        print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        if self.use_wandb:
            wandb.finish()

        self._save_final_model()

    def _train_one_epoch(self, epoch, logging_steps, eval_steps, save_steps, grad_accum):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch+1}",
            disable=not self._is_main_process(),
        )

        self.optimizer.zero_grad()

        for step, batch in pbar:
            alpha_ids = batch["alpha_input_ids"].to(self.device)
            alpha_mask = batch["alpha_attention_mask"].to(self.device)
            beta_ids = batch["beta_input_ids"].to(self.device)
            beta_mask = batch["beta_attention_mask"].to(self.device)

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.model(
                    alpha_input_ids=alpha_ids,
                    alpha_attention_mask=alpha_mask,
                    beta_input_ids=beta_ids,
                    beta_attention_mask=beta_mask,
                )

                # Get temperature from model
                if self.is_distributed:
                    temperature = self.model.module.temperature
                else:
                    temperature = self.model.temperature

                loss_outputs = self.criterion(
                    outputs["similarity_matrix"],
                    temperature,
                )
                loss = loss_outputs["loss"] / grad_accum

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss_outputs["loss"].item()
            total_acc += loss_outputs["accuracy"].item()
            num_batches += 1

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

                pbar.set_postfix({
                    "loss": f"{loss_outputs['loss'].item():.4f}",
                    "acc": f"{loss_outputs['accuracy'].item():.4f}",
                    "temp": f"{temperature.item():.4f}",
                })

                # Logging
                if self.global_step % logging_steps == 0 and self._is_main_process():
                    if self.use_wandb:
                        wandb.log({
                            "train/loss": total_loss / num_batches,
                            "train/accuracy": total_acc / num_batches,
                            "train/temperature": temperature.item(),
                            "train/learning_rate": self.scheduler.get_last_lr()[0],
                            "train/global_step": self.global_step,
                        })

                # Evaluation
                if self.global_step % eval_steps == 0:
                    metrics = self._validate()
                    if self._is_main_process():
                        print(f"\nStep {self.global_step} - Val Loss: {metrics['eval_loss']:.4f}, "
                              f"Recall@10: {metrics.get('recall@10', 0):.4f}")
                        if self.use_wandb:
                            wandb.log({
                                f"eval/{k}": v for k, v in metrics.items()
                            } | {"eval/global_step": self.global_step})

                    # Save best model
                    if metrics.get("recall@10", 0) > self.best_metric:
                        self.best_metric = metrics.get("recall@10", 0)
                        self._save_checkpoint(epoch, is_best=True)

                    self.model.train()

                # Checkpointing
                if self.global_step % save_steps == 0:
                    self._save_checkpoint(epoch)

        pbar.close()

    def _validate(self) -> Dict[str, float]:
        """Run validation."""
        return ContrastiveEvaluator.evaluate(
            self.model,
            self.val_loader,
            self.device,
            k_values=[1, 5, 10, 20],
        )

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
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
            "best_metric": self.best_metric,
            "config": self.config,
        }

        checkpoint_path = os.path.join(output_dir, f"checkpoint-step-{self.global_step}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        if is_best:
            best_path = os.path.join(output_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"✓ New best model (recall@10={self.best_metric:.4f})")

        # Cleanup old checkpoints (keep 3)
        checkpoints = sorted(glob.glob(os.path.join(output_dir, "checkpoint-step-*.pt")))
        for old_ckpt in checkpoints[:-3]:
            os.remove(old_ckpt)

    def _save_final_model(self):
        """Save final model."""
        if not self._is_main_process():
            return

        final_dir = os.path.join(self.config["output_dir"], "final_model")
        os.makedirs(final_dir, exist_ok=True)

        model_to_save = self.model.module if self.is_distributed else self.model

        # Save full model state
        torch.save({
            "model_state_dict": model_to_save.state_dict(),
            "config": self.config,
        }, os.path.join(final_dir, "model.pt"))

        # Save tokenizer
        self.tokenizer.save_pretrained(final_dir)

        print(f"\n✓ Final model saved to: {final_dir}")


# =============================================================================
# Argument Parser
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="TCR Alpha-Beta Pairing with Contrastive Learning"
    )

    # Required
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to tokenized TCR pair dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for checkpoints")

    # Model
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--projection_dim", type=int, default=256,
                        help="Dimension of projection head output")
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean"],
                        help="Pooling strategy for sequence embedding")

    # LoRA
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", action="store_false", dest="use_lora")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Contrastive Learning
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="Initial temperature for InfoNCE")
    parser.add_argument("--symmetric_loss", action="store_true", default=True,
                        help="Use symmetric InfoNCE (both directions)")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing for soft negatives")

    # Training
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size (larger = more in-batch negatives)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # Optimization
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--use_amp", action="store_true", default=True)

    # Evaluation & Logging
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)

    # Early Stopping
    parser.add_argument("--early_stopping", action="store_true", default=True)
    parser.add_argument("--no_early_stopping", action="store_false", dest="early_stopping")
    parser.add_argument("--early_stopping_patience", type=int, default=5)

    # Wandb
    parser.add_argument("--report_to", type=str, default="wandb", choices=["wandb", "none"])
    parser.add_argument("--wandb_project", type=str, default="tcr-pairing")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()
    config = vars(args)

    trainer = TCRContrastiveTrainer(config)
    trainer.train()

    if trainer.is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
