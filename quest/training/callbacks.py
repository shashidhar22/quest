"""
quest.training.callbacks - Training callbacks and metric trackers.

Consolidates EarlyStopping and StreamingMetricsTracker
from the following source files:
  - scripts/training/esm_native_trainer.py
  - scripts/training/tcr_robust_contrastive_trainer.py
"""

import math
from typing import Dict, Optional

import numpy as np


class EarlyStopping:
    """
    Early stopping with patience and minimum delta.

    Supports both "lower is better" (e.g., loss) and "higher is better"
    (e.g., recall, accuracy) modes via the ``mode`` parameter.

    Unified implementation supporting both min and max modes.

    Args:
        patience: How many epochs to wait after last improvement.
        min_delta: Minimum change in the monitored quantity to qualify
                   as an improvement.
        mode: One of ``"min"`` or ``"max"``.
              ``"min"`` means lower scores are better (e.g., loss).
              ``"max"`` means higher scores are better (e.g., recall).
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = "min",
    ):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float = float("inf") if mode == "min" else float("-inf")
        self.early_stop = False
        # Alias used by some trainers
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check whether training should stop.

        Args:
            score: The metric value to evaluate (e.g., validation loss or
                   recall). For backward compatibility this parameter is
                   also accepted as ``val_loss`` in min-mode callers.

        Returns:
            ``True`` if training should be stopped, ``False`` otherwise.
        """
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
                self.should_stop = True

        return self.early_stop

    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = float("inf") if self.mode == "min" else float("-inf")
        self.early_stop = False
        self.should_stop = False


class StreamingMetricsTracker:
    """
    Tracks accuracy and perplexity incrementally during evaluation.
    Prevents OOM by processing each batch separately instead of storing all logits.

    This enables evaluating on full validation sets with large models by computing
    metrics batch-by-batch instead of loading all predictions into memory at once.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all counters for a new evaluation run."""
        self.total_correct = 0
        self.total_tokens = 0
        self.total_loss = 0.0
        self.num_batches = 0

    def update(self, logits: np.ndarray, labels: np.ndarray):
        """
        Update metrics with a single batch.

        Args:
            logits: Shape (batch_size, seq_len, vocab_size)
            labels: Shape (batch_size, seq_len)
        """
        # Mask out -100 labels (padding/non-masked tokens)
        mask = labels != -100

        # Calculate accuracy
        preds = np.argmax(logits, axis=-1)
        correct = (preds[mask] == labels[mask]).sum()
        self.total_correct += correct
        self.total_tokens += mask.sum()

        # Calculate loss for perplexity (use float64 for numerical stability)
        logits_masked = logits[mask].astype(np.float64)
        labels_masked = labels[mask].astype(np.int64)

        # Compute cross-entropy loss in chunks to avoid memory spike
        chunk_size = 10000  # Process 10k tokens at a time
        for i in range(0, len(labels_masked), chunk_size):
            chunk_logits = logits_masked[i:i+chunk_size]
            chunk_labels = labels_masked[i:i+chunk_size]

            # Log-softmax
            logits_max = np.max(chunk_logits, axis=-1, keepdims=True)
            logits_shifted = chunk_logits - logits_max
            log_sum_exp = np.log(np.sum(np.exp(logits_shifted), axis=-1, keepdims=True))
            log_probs = logits_shifted - log_sum_exp

            # Negative log likelihood
            nll = -log_probs[np.arange(len(chunk_labels)), chunk_labels]
            self.total_loss += nll.sum()

        self.num_batches += 1

    def compute(self) -> Dict[str, float]:
        """Compute final metrics from accumulated statistics."""
        if self.total_tokens == 0:
            return {"accuracy": 0.0, "perplexity": float('inf')}

        accuracy = self.total_correct / self.total_tokens
        avg_loss = self.total_loss / self.total_tokens
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')

        return {
            "accuracy": float(accuracy),
            "perplexity": float(perplexity),
        }
