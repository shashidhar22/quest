"""
Base trainer class for TCR model training with hardware abstraction.

This module provides a hardware-agnostic base class for training TCR models,
supporting both NVIDIA GPUs (CUDA) and AWS Trainium (XLA) through the
backends abstraction layer.

Subclasses implement model-specific logic while inheriting common training
infrastructure including distributed training, mixed precision, checkpointing,
and logging.
"""
import math
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from quest.training.backends import AcceleratorBackend, get_backend

# Optional wandb support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class BaseTCRTrainer(ABC):
    """
    Abstract base trainer with hardware-agnostic training infrastructure.

    This class provides common training functionality that works across
    different hardware backends (CUDA, XLA/Trainium). Subclasses implement
    model-specific methods.

    Subclasses must implement:
    - _create_model(): Create and return the model instance
    - _create_datasets(): Create train and optionally validation datasets
    - _create_collator(): Create data collator for batching
    - _compute_loss(): Compute loss for a batch

    Optional overrides:
    - _get_model_load_kwargs(): Additional kwargs for model loading
    - _on_train_begin(): Called before training starts
    - _on_train_end(): Called after training completes
    - _on_epoch_begin(): Called at the start of each epoch
    - _on_epoch_end(): Called at the end of each epoch

    Example:
        class MyTrainer(BaseTCRTrainer):
            def _create_model(self):
                return MyModel(**self._get_model_load_kwargs())

            def _create_datasets(self):
                train_ds = MyDataset(self.config["data_path"], split="train")
                val_ds = MyDataset(self.config["data_path"], split="val")
                return train_ds, val_ds

            def _create_collator(self):
                return MyCollator(self.tokenizer)

            def _compute_loss(self, model, batch):
                outputs = model(**batch)
                return {"loss": outputs.loss, "accuracy": outputs.accuracy}
    """

    def __init__(
        self,
        config: Dict[str, Any],
        backend: Optional[AcceleratorBackend] = None
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration dictionary. Required keys:
                - output_dir: Directory for logs and checkpoints
                Optional keys with defaults:
                - backend: 'auto', 'cuda', or 'xla' (default: 'auto')
                - batch_size: Batch size (default: 16)
                - learning_rate: Learning rate (default: 1e-4)
                - num_epochs: Number of epochs (default: 10)
                - gradient_accumulation_steps: Grad accumulation (default: 1)
                - warmup_ratio: LR warmup ratio (default: 0.1)
                - weight_decay: Weight decay (default: 0.01)
                - max_grad_norm: Gradient clipping (default: 1.0)
                - num_workers: DataLoader workers (default: 4)
                - log_steps: Log every N steps (default: 100)
                - eval_steps: Evaluate every N steps (default: 1000)
                - save_steps: Save checkpoint every N steps (default: 5000)
                - patience: Early stopping patience (default: 3)
            backend: Accelerator backend instance. If None, auto-detected
                based on config["backend"] or available hardware.
        """
        self.config = config

        # Initialize backend (auto-detect if not specified)
        backend_type = config.get("backend", "auto")
        self.backend = backend or get_backend(backend_type)

        # Detect distributed training from environment
        self.is_distributed = "LOCAL_RANK" in os.environ or "WORLD_SIZE" in os.environ

        if self.is_distributed:
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.global_rank = int(os.environ.get("RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))

            # Initialize distributed backend
            self.backend.init_distributed(self.local_rank, self.world_size)
            self.device = self.backend.get_device(self.local_rank)
        else:
            self.local_rank = 0
            self.global_rank = 0
            self.world_size = 1
            self.device = self.backend.get_device()

        # Setup output directory
        self.output_dir = Path(config["output_dir"])
        if self._is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_val_metric = 0.0  # For metrics where higher is better

        # Model and data (initialized later)
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.collator = None

        # Log initialization
        self._log(f"Initialized {self.__class__.__name__}")
        self._log(f"Backend: {self.backend.name}")
        self._log(f"Device: {self.device}")
        self._log(f"Distributed: {self.is_distributed} (world_size={self.world_size})")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self.global_rank == 0

    def _setup_logging(self) -> None:
        """Setup training log file."""
        self.debug_log = self.output_dir / "training.log"
        if self._is_main_process():
            with open(self.debug_log, "w") as f:
                f.write(f"Training log started at {datetime.now()}\n")
                f.write(f"Backend: {self.backend.name}\n")
                f.write(f"Config: {self.config}\n")
                f.write("=" * 60 + "\n\n")

    def _log(self, message: str, also_print: bool = True) -> None:
        """
        Log message to file and optionally print.

        Args:
            message: Message to log
            also_print: Whether to also print to stdout
        """
        if self._is_main_process():
            with open(self.debug_log, "a") as f:
                f.write(message + "\n")
            if also_print:
                print(message)

    # =========================================================================
    # Abstract Methods - Must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def _create_model(self) -> nn.Module:
        """
        Create and return the model instance.

        Should use self._get_model_load_kwargs() for hardware-specific
        settings like attention implementation and dtype.

        Returns:
            nn.Module: The model to train
        """
        pass

    @abstractmethod
    def _create_datasets(self) -> Union[Dataset, Tuple[Dataset, Dataset]]:
        """
        Create train and optionally validation datasets.

        Returns:
            Either a single train dataset, or tuple of (train_dataset, val_dataset)
        """
        pass

    @abstractmethod
    def _create_collator(self) -> Any:
        """
        Create data collator for batching.

        Returns:
            Collator function or object for the DataLoader
        """
        pass

    @abstractmethod
    def _compute_loss(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for a batch.

        Args:
            model: The model (may be wrapped in DDP)
            batch: Batch dictionary from DataLoader

        Returns:
            Dict with at least 'loss' key, plus any additional metrics
        """
        pass

    # =========================================================================
    # Optional Override Methods
    # =========================================================================

    def _get_model_load_kwargs(self) -> Dict[str, Any]:
        """
        Get kwargs for model loading (attention implementation, dtype).

        Override in subclasses if additional kwargs are needed.

        Returns:
            Dict with attn_implementation and torch_dtype
        """
        return {
            "attn_implementation": self.backend.get_attention_implementation(),
            "torch_dtype": self.backend.get_model_dtype(),
        }

    def _on_train_begin(self) -> None:
        """Called before training starts. Override for custom logic."""
        pass

    def _on_train_end(self) -> None:
        """Called after training completes. Override for custom logic."""
        pass

    def _on_epoch_begin(self, epoch: int) -> None:
        """Called at the start of each epoch. Override for custom logic."""
        pass

    def _on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Called at the end of each epoch. Override for custom logic."""
        pass

    # =========================================================================
    # Setup Methods
    # =========================================================================

    def setup_model(self) -> None:
        """Setup model with backend-appropriate settings."""
        self._log("Setting up model...")

        # Create model (subclass implementation)
        self.model = self._create_model()

        # Move to device
        self.model.to(self.device)

        # Wrap for distributed training (CUDA uses DDP, XLA uses move_model_to_device for TP)
        if self.is_distributed:
            self.model = self.backend.wrap_model_distributed(
                self.model,
                self.local_rank,
                find_unused_parameters=self.config.get("find_unused_parameters", True)
            )

        # Log parameter counts
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self._log(f"Total params: {total_params:,}")
        self._log(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    def _get_optimal_num_workers(self) -> int:
        """
        Get optimal num_workers based on backend.

        For XLA/Trainium:
        - Use fewer workers to leave CPU for graph compilation
        - MpDeviceLoader handles prefetching efficiently

        For CUDA:
        - Use configured workers for CPU preprocessing
        """
        default = self.config.get("num_workers", 4)
        if self.backend.name == "xla":
            # trn1.2xlarge has 8 vCPUs; limit workers to leave CPU for XLA
            return min(default, 2)
        return default

    def setup_data(self, include_val: bool = True) -> None:
        """
        Setup datasets and dataloaders.

        Args:
            include_val: Whether to include validation data
        """
        self._log("Setting up data...")

        # Create collator first (may need tokenizer)
        self.collator = self._create_collator()

        # Create datasets (subclass implementation)
        datasets = self._create_datasets()
        if isinstance(datasets, tuple):
            self.train_dataset, self.val_dataset = datasets
        else:
            self.train_dataset = datasets
            self.val_dataset = None

        if not include_val:
            self.val_dataset = None

        # Setup samplers for distributed training
        if self.is_distributed:
            # Use TP-aware DP rank/world size when tensor parallelism is active
            if hasattr(self.backend, 'get_data_parallel_world_size'):
                dp_world_size = self.backend.get_data_parallel_world_size()
                dp_rank = self.backend.get_data_parallel_rank()
                train_sampler = DistributedSampler(
                    self.train_dataset,
                    num_replicas=dp_world_size,
                    rank=dp_rank,
                    shuffle=True,
                    drop_last=True,
                )
                val_sampler = DistributedSampler(
                    self.val_dataset,
                    num_replicas=dp_world_size,
                    rank=dp_rank,
                    shuffle=False,
                ) if self.val_dataset else None
            else:
                train_sampler = DistributedSampler(
                    self.train_dataset, shuffle=True, drop_last=True
                )
                val_sampler = DistributedSampler(
                    self.val_dataset, shuffle=False
                ) if self.val_dataset else None
        else:
            train_sampler = None
            val_sampler = None

        # Create train dataloader
        num_workers = self._get_optimal_num_workers()
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.get("batch_size", 16),
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            collate_fn=self.collator,
            pin_memory=True,
            drop_last=True,
        )

        # Wrap for XLA if needed
        self.train_loader = self.backend.wrap_dataloader(
            self.train_loader, self.device
        )

        self._log(f"Train dataset size: {len(self.train_dataset)}")
        self._log(f"Train loader batches: {len(self.train_loader)}")

        # Create validation dataloader
        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config.get("batch_size", 16),
                sampler=val_sampler,
                num_workers=num_workers,
                collate_fn=self.collator,
                pin_memory=True,
            )
            self.val_loader = self.backend.wrap_dataloader(
                self.val_loader, self.device
            )
            self._log(f"Val dataset size: {len(self.val_dataset)}")
            self._log(f"Val loader batches: {len(self.val_loader)}")

    def setup_optimizer(self, lr: Optional[float] = None) -> None:
        """
        Initialize optimizer.

        Args:
            lr: Learning rate (overrides config if provided)
        """
        learning_rate = lr or self.config.get("learning_rate", 1e-4)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=self.config.get("weight_decay", 0.01),
        )

        self._log(f"Optimizer: AdamW, lr={learning_rate}, weight_decay={self.config.get('weight_decay', 0.01)}")

    def setup_scheduler(self, num_training_steps: int) -> None:
        """
        Initialize learning rate scheduler with warmup and cosine decay.

        Args:
            num_training_steps: Total number of training steps
        """
        warmup_ratio = self.config.get("warmup_ratio", 0.1)
        warmup_steps = int(num_training_steps * warmup_ratio)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(
                max(1, num_training_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )
        self._log(f"Scheduler: Cosine with {warmup_steps} warmup steps ({warmup_ratio*100:.0f}%)")

    # =========================================================================
    # Training Methods
    # =========================================================================

    def _move_batch_to_device(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Move batch tensors to the training device.

        Args:
            batch: Batch dictionary from DataLoader

        Returns:
            Batch with tensors on the correct device
        """
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _training_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Execute a single training step.

        Args:
            batch: Batch from DataLoader

        Returns:
            Dict with loss and any additional metrics
        """
        # Move batch to device
        batch = self._move_batch_to_device(batch)

        # Forward pass with backend-appropriate autocast
        with self.backend.autocast_context(self.backend.get_model_dtype()):
            outputs = self._compute_loss(self.model, batch)

        return outputs

    def _optimizer_step(self) -> None:
        """Execute optimizer step with backend-specific handling."""
        self.backend.optimizer_step(self.optimizer, self.model)

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: Optional[DataLoader] = None,
        desc: str = "Evaluating"
    ) -> Dict[str, float]:
        """
        Evaluate model on validation data.

        Args:
            dataloader: DataLoader to evaluate on (defaults to self.val_loader)
            desc: Description for progress bar

        Returns:
            Dict with averaged metrics
        """
        dataloader = dataloader or self.val_loader
        if dataloader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        all_metrics: Dict[str, float] = {}

        progress_bar = tqdm(
            dataloader,
            desc=desc,
            disable=not self._is_main_process(),
            leave=False,
        )

        for batch in progress_bar:
            batch = self._move_batch_to_device(batch)
            batch_size = next(iter(batch.values())).size(0)

            with self.backend.autocast_context(self.backend.get_model_dtype()):
                outputs = self._compute_loss(self.model, batch)

            loss = outputs["loss"].item()
            total_loss += loss * batch_size
            total_samples += batch_size

            # Accumulate other metrics
            for key, value in outputs.items():
                if key != "loss" and isinstance(value, torch.Tensor):
                    if key not in all_metrics:
                        all_metrics[key] = 0.0
                    all_metrics[key] += value.item() * batch_size

        self.model.train()

        # Average metrics
        metrics = {"loss": total_loss / max(total_samples, 1)}
        for key, value in all_metrics.items():
            metrics[key] = value / max(total_samples, 1)

        return metrics

    def save_checkpoint(
        self,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        filename: Optional[str] = None
    ) -> None:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            step: Current global step
            metrics: Current metrics
            is_best: Whether this is the best model so far
            filename: Custom filename (default: checkpoint_step_{step}.pt)
        """
        if not self._is_main_process():
            return

        # Get model state (handle DDP wrapper)
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        checkpoint = {
            "epoch": epoch,
            "global_step": step,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "metrics": metrics,
            "config": self.config,
            "backend": self.backend.name,
            "best_val_loss": self.best_val_loss,
        }

        # Save checkpoint
        if filename is None:
            filename = f"checkpoint_step_{step}.pt"
        checkpoint_path = str(self.output_dir / filename)
        self.backend.save_checkpoint(checkpoint, checkpoint_path, True)
        self._log(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = str(self.output_dir / "best_model.pt")
            self.backend.save_checkpoint(checkpoint, best_path, True)
            self._log(f"New best model saved!")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load checkpoint and restore training state.

        Args:
            path: Path to checkpoint file

        Returns:
            Loaded checkpoint dictionary
        """
        checkpoint = self.backend.load_checkpoint(path, self.device)

        # Restore model state
        model_to_load = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model_to_load.load_state_dict(checkpoint["model_state_dict"])

        # Restore optimizer state
        if self.optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore scheduler state
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore training state
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        self._log(f"Loaded checkpoint from {path}")
        self._log(f"Resuming from step {self.global_step}")

        return checkpoint

    # =========================================================================
    # High-Level Training Methods
    # =========================================================================

    def train(self) -> None:
        """
        Full training loop with validation, checkpointing, and early stopping.

        This is the main entry point for training. It:
        1. Sets up model, data, optimizer, and scheduler
        2. Runs training epochs with gradient accumulation
        3. Evaluates periodically and saves checkpoints
        4. Implements early stopping based on validation loss
        """
        # Setup
        self.setup_model()
        self.setup_data(include_val=True)

        num_epochs = self.config.get("num_epochs", 10)
        grad_accum_steps = self.config.get("gradient_accumulation_steps", 1)
        eval_steps = self.config.get("eval_steps", 1000)
        save_steps = self.config.get("save_steps", 5000)
        log_steps = self.config.get("log_steps", 100)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        patience = self.config.get("patience", 3)

        # Calculate total steps
        steps_per_epoch = len(self.train_loader) // grad_accum_steps
        num_training_steps = steps_per_epoch * num_epochs

        self.setup_optimizer()
        self.setup_scheduler(num_training_steps)

        self._log(f"\nStarting training...")
        self._log(f"  Epochs: {num_epochs}")
        self._log(f"  Batch size: {self.config.get('batch_size', 16)}")
        self._log(f"  Gradient accumulation steps: {grad_accum_steps}")
        self._log(f"  Effective batch size: {self.config.get('batch_size', 16) * grad_accum_steps * self.world_size}")
        self._log(f"  Total training steps: {num_training_steps}")
        self._log(f"  Eval steps: {eval_steps}")
        self._log(f"  Early stopping patience: {patience}")

        # Training state
        patience_counter = 0
        running_loss = 0.0
        running_steps = 0

        self._on_train_begin()
        self.model.train()

        for epoch in range(num_epochs):
            self._log(f"\n{'='*60}")
            self._log(f"Epoch {epoch + 1}/{num_epochs}")
            self._log(f"{'='*60}")

            self._on_epoch_begin(epoch)

            # Set epoch for distributed sampler
            if self.is_distributed and hasattr(self.train_loader, "sampler"):
                sampler = self.train_loader.sampler
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)

            epoch_loss = 0.0
            epoch_steps = 0

            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}",
                disable=not self._is_main_process(),
            )

            self.optimizer.zero_grad()

            for batch_idx, batch in enumerate(progress_bar):
                # Training step
                outputs = self._training_step(batch)
                loss = outputs["loss"] / grad_accum_steps

                loss.backward()

                epoch_loss += outputs["loss"].item()
                epoch_steps += 1
                running_loss += outputs["loss"].item()
                running_steps += 1

                # Gradient accumulation
                if (batch_idx + 1) % grad_accum_steps == 0:
                    # Gradient clipping (delegate to backend for TP-aware clipping)
                    if hasattr(self.backend, 'clip_grad_norm'):
                        self.backend.clip_grad_norm(self.model, max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=max_grad_norm
                        )

                    # Optimizer step (backend-specific)
                    self._optimizer_step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Logging
                    if self.global_step % log_steps == 0 and self._is_main_process():
                        avg_loss = running_loss / running_steps
                        lr = self.scheduler.get_last_lr()[0]
                        self._log(
                            f"Step {self.global_step}: loss={avg_loss:.4f}, lr={lr:.2e}"
                        )
                        running_loss = 0.0
                        running_steps = 0

                    # Evaluation
                    if self.global_step % eval_steps == 0 and self.val_loader:
                        self._log(f"\n--- Validation at step {self.global_step} ---")
                        val_metrics = self.evaluate()
                        self._log(f"Val metrics: {val_metrics}")

                        # Check for improvement
                        if val_metrics["loss"] < self.best_val_loss:
                            self.best_val_loss = val_metrics["loss"]
                            patience_counter = 0
                            self.save_checkpoint(
                                epoch, self.global_step, val_metrics, is_best=True
                            )
                        else:
                            patience_counter += 1
                            self._log(f"No improvement. Patience: {patience_counter}/{patience}")

                        if patience_counter >= patience:
                            self._log(f"\nEarly stopping at step {self.global_step}")
                            self._on_train_end()
                            return

                        self.model.train()

                    # Periodic checkpoint
                    if self.global_step % save_steps == 0:
                        self.save_checkpoint(
                            epoch, self.global_step,
                            {"train_loss": epoch_loss / epoch_steps}
                        )

                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{epoch_loss/epoch_steps:.4f}",
                    "step": self.global_step,
                })

            # End of epoch
            epoch_metrics = {"train_loss": epoch_loss / epoch_steps}

            if self.val_loader:
                val_metrics = self.evaluate(desc="End-of-epoch validation")
                epoch_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                self._log(f"Epoch {epoch + 1} - Train loss: {epoch_loss/epoch_steps:.4f}, Val loss: {val_metrics['loss']:.4f}")

                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    patience_counter = 0
                    self.save_checkpoint(epoch, self.global_step, val_metrics, is_best=True)
                else:
                    patience_counter += 1
            else:
                self._log(f"Epoch {epoch + 1} - Train loss: {epoch_loss/epoch_steps:.4f}")

            self._on_epoch_end(epoch, epoch_metrics)

            if patience_counter >= patience:
                self._log(f"\nEarly stopping at epoch {epoch + 1}")
                break

        # Training complete
        self._log("\n" + "=" * 60)
        self._log("TRAINING COMPLETE")
        self._log("=" * 60)
        self._log(f"Best validation loss: {self.best_val_loss:.4f}")
        self._log(f"Total steps: {self.global_step}")
        self._log(f"Best model saved to: {self.output_dir / 'best_model.pt'}")

        # Save final model
        self.save_checkpoint(
            num_epochs - 1, self.global_step,
            {"final": True},
            filename="final_model.pt"
        )

        self._on_train_end()

    def overfit_single_batch(self) -> None:
        """
        Overfit check: Train on a single batch to verify the pipeline works.

        This is useful for debugging to ensure:
        1. Model can learn (loss decreases)
        2. Gradients flow correctly
        3. All components work together

        The loss should decrease significantly within a few hundred steps.
        """
        self._log("\n" + "=" * 60)
        self._log("SINGLE BATCH OVERFIT CHECK")
        self._log("=" * 60)
        self._log("Expected: Loss should decrease significantly")
        self._log("=" * 60 + "\n")

        # Setup
        self.setup_model()
        self.setup_data(include_val=False)

        num_steps = self.config.get("overfit_steps", 500)
        lr = self.config.get("overfit_lr", self.config.get("learning_rate", 1e-4))

        self.setup_optimizer(lr=lr)
        self.model.train()

        # Get a single batch
        batch = next(iter(self.train_loader))
        batch = self._move_batch_to_device(batch)

        self._log(f"Batch size: {next(iter(batch.values())).size(0)}")
        self._log(f"Learning rate: {lr}")
        self._log("")

        initial_loss = None
        final_loss = None

        for step in range(num_steps):
            self.optimizer.zero_grad()

            with self.backend.autocast_context(self.backend.get_model_dtype()):
                outputs = self._compute_loss(self.model, batch)

            loss = outputs["loss"]
            loss.backward()

            max_grad_norm = self.config.get("max_grad_norm", 1.0)
            if hasattr(self.backend, 'clip_grad_norm'):
                grad_norm = self.backend.clip_grad_norm(self.model, max_grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=max_grad_norm
                )

            self._optimizer_step()

            if step == 0:
                initial_loss = loss.item()

            if step % 10 == 0 or step == num_steps - 1:
                metrics_str = ", ".join(
                    f"{k}={v.item():.4f}" if isinstance(v, torch.Tensor) else f"{k}={v:.4f}"
                    for k, v in outputs.items()
                )
                self._log(f"Step {step:4d}: {metrics_str}, grad_norm={grad_norm:.4f}")

            final_loss = loss.item()

        self._log("\n" + "=" * 60)
        self._log("OVERFIT CHECK RESULTS")
        self._log("=" * 60)
        self._log(f"Initial loss: {initial_loss:.4f}")
        self._log(f"Final loss:   {final_loss:.4f}")
        self._log(f"Loss reduction: {100 * (initial_loss - final_loss) / initial_loss:.1f}%")

        if final_loss < initial_loss * 0.1:
            self._log("\nSUCCESS: Model can overfit a single batch!")
        elif final_loss < initial_loss * 0.5:
            self._log("\nPARTIAL SUCCESS: Loss decreased but not fully converged.")
            self._log("Try more steps or higher learning rate.")
        else:
            self._log("\nFAILURE: Model cannot overfit a single batch!")
            self._log("Possible issues:")
            self._log("  - Gradients not flowing (check requires_grad)")
            self._log("  - Learning rate too low")
            self._log("  - Architecture problem")

        self._log("=" * 60 + "\n")
