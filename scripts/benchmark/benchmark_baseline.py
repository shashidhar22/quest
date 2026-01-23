#!/usr/bin/env python
"""
Baseline Benchmark Script for TCR-Peptide Specificity Prediction

This script implements simple neural network baselines (LSTM and MLP) for
predicting peptide specificity given TCR sequences and MHC information.
These serve as comparison baselines against more sophisticated models.
"""

import argparse
import json
import os
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.benchmark.benchmark_metrics import compute_all_unified_metrics

warnings.filterwarnings("ignore")

# Task configurations for all 6 tasks
TASK_CONFIGS = {
    "tra_peptide_mhc_one": {
        "mhc_class": "class_one",
        "tcr_cols": ["tra"],
        "mhc_cols": ["mhc_one_id"],
        "description": "TCR alpha only + MHC-I -> peptide",
    },
    "trb_peptide_mhc_one": {
        "mhc_class": "class_one",
        "tcr_cols": ["trb"],
        "mhc_cols": ["mhc_one_id"],
        "description": "TCR beta only + MHC-I -> peptide",
    },
    "tra_trb_peptide_mhc_one": {
        "mhc_class": "class_one",
        "tcr_cols": ["tra", "trb"],
        "mhc_cols": ["mhc_one_id"],
        "description": "TCR alpha+beta paired + MHC-I -> peptide",
    },
    "tra_peptide_mhc_two": {
        "mhc_class": "class_two",
        "tcr_cols": ["tra"],
        "mhc_cols": ["mhc_one_id", "mhc_two_id"],
        "description": "TCR alpha only + MHC-II -> peptide",
    },
    "trb_peptide_mhc_two": {
        "mhc_class": "class_two",
        "tcr_cols": ["trb"],
        "mhc_cols": ["mhc_one_id", "mhc_two_id"],
        "description": "TCR beta only + MHC-II -> peptide",
    },
    "tra_trb_peptide_mhc_two": {
        "mhc_class": "class_two",
        "tcr_cols": ["tra", "trb"],
        "mhc_cols": ["mhc_one_id", "mhc_two_id"],
        "description": "TCR alpha+beta paired + MHC-II -> peptide",
    },
}

TEST_SPLITS = [
    "test_seen_epitope",
    "test_unseen_tcr_seen_epitope",
    "test_unseen_epitope",
    "test_unseen_allele",
]

# Amino acid vocabulary
AA_VOCAB = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
AA_VOCAB["<PAD>"] = 0
AA_VOCAB["<UNK>"] = 21
VOCAB_SIZE = 22  # 20 amino acids + padding + unknown


# =============================================================================
# Sequence Encoding Functions
# =============================================================================


def encode_sequence(seq: str, max_len: int = 512) -> Tuple[List[int], int]:
    """
    Convert amino acid sequence to token IDs.

    Args:
        seq: Amino acid sequence string
        max_len: Maximum sequence length

    Returns:
        Tuple of (token IDs list, actual length)
    """
    tokens = [AA_VOCAB.get(aa.upper(), AA_VOCAB["<UNK>"]) for aa in seq]
    length = len(tokens)

    # Pad or truncate
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
        length = max_len
    else:
        tokens = tokens + [0] * (max_len - len(tokens))

    return tokens, length


def encode_aa_frequencies(seq: str) -> np.ndarray:
    """
    Compute normalized amino acid frequencies (bag-of-AAs).

    Args:
        seq: Amino acid sequence string

    Returns:
        Normalized frequency vector of shape (21,)
    """
    counts = np.zeros(21)  # 20 AAs + unknown
    for aa in seq.upper():
        idx = AA_VOCAB.get(aa, AA_VOCAB["<UNK>"]) - 1  # Shift by 1 (skip padding)
        if 0 <= idx < 21:
            counts[idx] += 1

    # Normalize
    total = counts.sum()
    if total > 0:
        counts /= total

    return counts


# =============================================================================
# Dataset Class
# =============================================================================


class TCRPeptideDataset(Dataset):
    """Dataset for TCR-peptide classification."""

    def __init__(
        self,
        df: pd.DataFrame,
        task_config: dict,
        label_encoder: LabelEncoder,
        max_seq_len: int = 512,
        encoding: str = "tokens",  # "tokens" for LSTM, "frequencies" for MLP
    ):
        """
        Initialize dataset.

        Args:
            df: DataFrame with TCR and peptide data
            task_config: Task configuration dictionary
            label_encoder: Fitted LabelEncoder for peptide labels
            max_seq_len: Maximum sequence length for tokenization
            encoding: "tokens" for LSTM or "frequencies" for MLP
        """
        self.df = df.reset_index(drop=True)
        self.task_config = task_config
        self.label_encoder = label_encoder
        self.max_seq_len = max_seq_len
        self.encoding = encoding

        # Encode labels - handle unseen labels
        self.labels = []
        self.valid_indices = []
        for idx, peptide in enumerate(df["peptide"].values):
            if peptide in label_encoder.classes_:
                self.labels.append(label_encoder.transform([peptide])[0])
                self.valid_indices.append(idx)

        self.labels = np.array(self.labels)
        self.valid_mask = np.isin(np.arange(len(df)), self.valid_indices)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.df.iloc[actual_idx]

        # Concatenate TCR sequences with separator
        sequences = []
        for col in self.task_config["tcr_cols"]:
            if col in self.df.columns and pd.notna(row[col]):
                sequences.append(str(row[col]).upper())

        # Add MHC allele as string (simplified encoding)
        for col in self.task_config["mhc_cols"]:
            if col in self.df.columns and pd.notna(row[col]):
                # Remove special characters from MHC for tokenization
                mhc_str = str(row[col]).replace("*", "").replace(":", "")
                sequences.append(mhc_str)

        combined_seq = "".join(sequences)

        if self.encoding == "tokens":
            tokens, length = encode_sequence(combined_seq, self.max_seq_len)
            return {
                "input_ids": torch.tensor(tokens, dtype=torch.long),
                "length": torch.tensor(length, dtype=torch.long),
                "label": torch.tensor(self.labels[idx], dtype=torch.long),
            }
        else:
            # Frequency encoding for MLP
            freq = encode_aa_frequencies(combined_seq)
            return {
                "features": torch.tensor(freq, dtype=torch.float32),
                "label": torch.tensor(self.labels[idx], dtype=torch.long),
            }


# =============================================================================
# Model Architectures
# =============================================================================


class LSTMClassifier(nn.Module):
    """Bi-directional LSTM for sequence classification."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 100,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        self.bidirectional = bidirectional
        hidden_factor = 2 if bidirectional else 1
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * hidden_factor, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, lengths=None):
        """
        Forward pass.

        Args:
            x: Input token IDs of shape (batch, seq_len)
            lengths: Optional tensor of actual sequence lengths

        Returns:
            Logits of shape (batch, num_classes)
        """
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        if lengths is not None:
            # Pack padded sequence for efficient LSTM processing
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        _, (hidden, _) = self.lstm(embedded)  # hidden: (num_layers*dirs, batch, hidden)

        # Concatenate last forward and backward hidden states
        if self.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            hidden = hidden[-1]

        return self.fc(hidden)  # (batch, num_classes)


class MLPClassifier(nn.Module):
    """Simple MLP using bag-of-amino-acids representation."""

    def __init__(
        self,
        input_dim: int = 21,  # AA frequencies
        hidden_dims: List[int] = None,
        num_classes: int = 100,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input features of shape (batch, input_dim)

        Returns:
            Logits of shape (batch, num_classes)
        """
        return self.network(x)


# =============================================================================
# Training Functions
# =============================================================================


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    device: str = "cuda",
    encoding: str = "tokens",
) -> Tuple[nn.Module, Dict]:
    """
    Train model with early stopping.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Maximum number of epochs
        lr: Learning rate
        patience: Early stopping patience
        device: Device to train on
        encoding: "tokens" or "frequencies"

    Returns:
        Tuple of (trained model, training history dict)
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "best_epoch": 0,
    }

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            if encoding == "tokens":
                logits = model(
                    batch["input_ids"].to(device), batch["length"].to(device)
                )
            else:
                logits = model(batch["features"].to(device))

            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                if encoding == "tokens":
                    logits = model(
                        batch["input_ids"].to(device), batch["length"].to(device)
                    )
                else:
                    logits = model(batch["features"].to(device))

                loss = criterion(logits, batch["label"].to(device))
                val_loss += loss.item()
                val_batches += 1

                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == batch["label"].to(device)).sum().item()
                val_total += len(batch["label"])

        avg_val_loss = val_loss / val_batches
        val_accuracy = val_correct / val_total

        scheduler.step(avg_val_loss)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(val_accuracy)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
            history["best_epoch"] = epoch
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Val Acc: {val_accuracy:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# =============================================================================
# Evaluation Functions
# =============================================================================


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, num_classes: int
) -> Dict:
    """
    Compute classification metrics.

    Args:
        y_true: True class indices
        y_pred: Predicted class indices
        y_prob: Prediction probabilities (N x num_classes)
        num_classes: Number of classes

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Top-1 accuracy
    metrics["top1_accuracy"] = float(accuracy_score(y_true, y_pred))

    # Top-5 accuracy (if we have enough classes)
    k = min(5, num_classes)
    if k > 1:
        metrics["top5_accuracy"] = float(
            top_k_accuracy_score(y_true, y_prob, k=k, labels=range(num_classes))
        )
    else:
        metrics["top5_accuracy"] = metrics["top1_accuracy"]

    # Macro F1
    metrics["macro_f1"] = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )

    # Weighted F1
    metrics["weighted_f1"] = float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )

    # Macro precision/recall
    metrics["macro_precision"] = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    metrics["macro_recall"] = float(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )

    # AUC-ROC (one-vs-rest) - only if we have at least 2 classes represented
    try:
        if len(np.unique(y_true)) >= 2:
            y_true_onehot = np.zeros((len(y_true), num_classes))
            y_true_onehot[np.arange(len(y_true)), y_true] = 1
            metrics["auc_roc_ovr"] = float(
                roc_auc_score(y_true_onehot, y_prob, average="macro", multi_class="ovr")
            )
        else:
            metrics["auc_roc_ovr"] = None
    except ValueError:
        metrics["auc_roc_ovr"] = None

    metrics["num_samples"] = int(len(y_true))
    metrics["num_classes"] = int(num_classes)
    metrics["num_unique_true_classes"] = int(len(np.unique(y_true)))

    return metrics


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    encoding: str,
    num_classes: int,
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataset.

    Args:
        model: Trained model
        data_loader: Data loader
        device: Device
        encoding: "tokens" or "frequencies"
        num_classes: Number of classes

    Returns:
        Tuple of (metrics dict, y_true, y_pred, y_prob)
    """
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            if encoding == "tokens":
                logits = model(
                    batch["input_ids"].to(device), batch["length"].to(device)
                )
            else:
                logits = model(batch["features"].to(device))

            probs = torch.softmax(logits, dim=1)
            all_labels.append(batch["label"].numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    y_pred = np.argmax(y_prob, axis=1)

    metrics = compute_metrics(y_true, y_pred, y_prob, num_classes)

    return metrics, y_true, y_pred, y_prob


# =============================================================================
# Data Loading
# =============================================================================


def load_data(parquet_path: str, task_config: dict) -> pd.DataFrame:
    """
    Load parquet file and filter out rows with missing required columns.

    Args:
        parquet_path: Path to parquet file
        task_config: Task configuration dictionary

    Returns:
        Filtered DataFrame
    """
    df = pd.read_parquet(parquet_path)

    # Filter out rows with NaN in required columns
    required_cols = task_config["tcr_cols"] + ["peptide"]
    for col in required_cols:
        if col in df.columns:
            df = df[df[col].notna()]

    return df


# =============================================================================
# Main Training and Evaluation Pipeline
# =============================================================================


def train_and_evaluate(
    model_type: str,
    task_name: str,
    data_dir: str,
    output_dir: str,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    batch_size: int = 64,
    num_epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    max_seq_len: int = 512,
    device: str = None,
) -> Dict:
    """
    Train model and evaluate on all test splits.

    Args:
        model_type: "lstm" or "mlp"
        task_name: Name of the task
        data_dir: Base data directory
        output_dir: Output directory for results
        hidden_dim: Hidden dimension for models
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        batch_size: Batch size
        num_epochs: Number of training epochs
        lr: Learning rate
        patience: Early stopping patience
        max_seq_len: Maximum sequence length
        device: Device to use (auto-detect if None)

    Returns:
        Results dictionary
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    task_config = TASK_CONFIGS[task_name]
    mhc_class = task_config["mhc_class"]
    task_data_dir = os.path.join(data_dir, mhc_class, task_name)

    # Create output directory
    task_output_dir = os.path.join(output_dir, model_type, task_name)
    os.makedirs(task_output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Model: {model_type.upper()}")
    print(f"Task: {task_name}")
    print(f"Description: {task_config['description']}")
    print(f"Device: {device}")
    print(f"{'=' * 60}")

    # Determine encoding type
    encoding = "tokens" if model_type == "lstm" else "frequencies"

    # Load training data
    print("\nLoading training data...")
    train_path = os.path.join(task_data_dir, "train.parquet")
    train_df = load_data(train_path, task_config)

    # Load validation data
    print("Loading validation data...")
    val_path = os.path.join(task_data_dir, "val.parquet")
    val_df = load_data(val_path, task_config)

    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    # Fit label encoder on training data
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["peptide"].values)
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")

    # Create datasets
    train_dataset = TCRPeptideDataset(
        train_df, task_config, label_encoder, max_seq_len, encoding
    )
    val_dataset = TCRPeptideDataset(
        val_df, task_config, label_encoder, max_seq_len, encoding
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Initialize model
    print(f"\nInitializing {model_type.upper()} model...")
    if model_type == "lstm":
        model = LSTMClassifier(
            vocab_size=VOCAB_SIZE,
            embed_dim=64,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=True,
        )
    else:
        model = MLPClassifier(
            input_dim=21,  # AA frequencies
            hidden_dims=[256, 128, 64],
            num_classes=num_classes,
            dropout=dropout,
        )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Train model
    print("\nTraining model...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        lr=lr,
        patience=patience,
        device=device,
        encoding=encoding,
    )

    # Save model and config
    model_path = os.path.join(task_output_dir, "model.pt")
    torch.save(model.state_dict(), model_path)

    config = {
        "model_type": model_type,
        "task": task_name,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr": lr,
        "patience": patience,
        "max_seq_len": max_seq_len,
        "num_classes": num_classes,
        "num_params": num_params,
        "encoding": encoding,
    }
    config_path = os.path.join(task_output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Save training history
    history_path = os.path.join(task_output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Store results
    results = {
        "model_type": model_type,
        "task": task_name,
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "n_classes": num_classes,
        "best_epoch": history["best_epoch"],
        "test_results": {},
    }

    # Evaluate on each test split
    print("\nEvaluating on test splits...")
    for split_name in TEST_SPLITS:
        test_path = os.path.join(task_data_dir, f"{split_name}.parquet")
        if not os.path.exists(test_path):
            print(f"  Skipping {split_name} - file not found")
            continue

        print(f"\n  Evaluating on {split_name}...")

        # Load test data
        test_df = load_data(test_path, task_config)
        test_dataset = TCRPeptideDataset(
            test_df, task_config, label_encoder, max_seq_len, encoding
        )

        if len(test_dataset) == 0:
            print(f"    No samples with known labels - skipping")
            results["test_results"][split_name] = {
                "error": "No samples with known labels",
                "n_total": len(test_df),
                "n_known": 0,
            }
            continue

        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        # Evaluate
        metrics, y_true, y_pred, y_prob = evaluate_model(
            model, test_loader, device, encoding, num_classes
        )
        metrics["n_total"] = int(len(test_df))
        metrics["n_known_labels"] = int(len(test_dataset))
        metrics["n_unknown_labels"] = int(len(test_df) - len(test_dataset))

        # Compute unified retrieval metrics
        # y_prob is (n_samples, n_classes) — use directly as retrieval scores
        candidate_peptides = list(label_encoder.classes_)
        sample_peptides = np.array(label_encoder.inverse_transform(y_true))

        unified = compute_all_unified_metrics(
            scores=y_prob,
            true_indices=y_true,
            sample_peptides=sample_peptides,
            candidate_peptides=candidate_peptides,
            n_bootstrap=1000,
            min_samples_per_peptide=5,
            seed=42,
        )

        # Extract per_epitope_all before merging (not JSON-serializable inline)
        per_epitope_all = unified.pop("per_epitope_all", [])
        metrics.update(unified)

        results["test_results"][split_name] = metrics

        print(f"    Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")
        print(f"    Macro F1: {metrics['macro_f1']:.4f}")
        if metrics.get("auc_roc_ovr"):
            print(f"    AUC-ROC: {metrics['auc_roc_ovr']:.4f}")
        print(f"    Retrieval Hit@1: {metrics['retrieval_hit_at_1']:.4f}")
        print(f"    Retrieval MRR: {metrics['retrieval_mrr']:.4f}")
        if metrics.get("per_peptide_auc_mean") is not None:
            print(f"    Per-peptide AUC: {metrics['per_peptide_auc_mean']:.4f}")

        # Save predictions
        split_output_dir = os.path.join(task_output_dir, split_name)
        os.makedirs(split_output_dir, exist_ok=True)

        # Save metrics
        metrics_path = os.path.join(split_output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save per-epitope breakdown CSV
        if per_epitope_all:
            epitope_df = pd.DataFrame(per_epitope_all)
            epitope_df.to_csv(
                os.path.join(split_output_dir, "per_epitope_breakdown.csv"),
                index=False,
            )

        # Compute per-sample ranks for predictions CSV
        ranks = np.zeros(len(y_true), dtype=int)
        for i in range(len(y_true)):
            sorted_indices = np.argsort(-y_prob[i])
            ranks[i] = int(np.where(sorted_indices == y_true[i])[0][0]) + 1

        # Save predictions
        pred_df = pd.DataFrame(
            {
                "true_peptide": label_encoder.inverse_transform(y_true),
                "predicted_peptide": label_encoder.inverse_transform(y_pred),
                "true_peptide_rank": ranks,
                "true_peptide_score": y_prob[np.arange(len(y_true)), y_true],
                "predicted_prob": y_prob.max(axis=1),
            }
        )
        pred_path = os.path.join(split_output_dir, "predictions.csv")
        pred_df.to_csv(pred_path, index=False)

    # Save aggregated results
    results_path = os.path.join(task_output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {task_output_dir}")

    return results


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Baseline Neural Network Benchmark for TCR-peptide specificity prediction"
    )
    parser.add_argument(
        "--model",
        choices=["lstm", "mlp", "all"],
        default="lstm",
        help="Model type to run (default: lstm)",
    )
    parser.add_argument(
        "--task",
        choices=list(TASK_CONFIGS.keys()) + ["all"],
        default="trb_peptide_mhc_one",
        help="Task to run (default: trb_peptide_mhc_one)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/sravisha/projects/quest/data/icml/tasks/tcr_specificity/tcr90pep80",
        help="Base data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/sravisha/projects/quest/results/baseline_benchmark",
        help="Output directory for results",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension for LSTM (default: 128)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of LSTM layers (default: 2)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate (default: 0.3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of epochs (default: 50)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect)",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine models and tasks to run
    models = ["lstm", "mlp"] if args.model == "all" else [args.model]
    tasks = list(TASK_CONFIGS.keys()) if args.task == "all" else [args.task]

    # Track all results
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "results": {},
    }

    # Run benchmarks
    for model_type in models:
        all_results["results"][model_type] = {}
        for task in tasks:
            try:
                results = train_and_evaluate(
                    model_type=model_type,
                    task_name=task,
                    data_dir=args.data_dir,
                    output_dir=args.output_dir,
                    hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    batch_size=args.batch_size,
                    num_epochs=args.num_epochs,
                    lr=args.lr,
                    patience=args.patience,
                    max_seq_len=args.max_seq_len,
                    device=args.device,
                )
                all_results["results"][model_type][task] = results
            except Exception as e:
                print(f"\nError running {model_type}/{task}: {e}")
                import traceback

                traceback.print_exc()
                all_results["results"][model_type][task] = {"error": str(e)}

    # Save summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Benchmark complete!")
    print(f"Summary saved to: {summary_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
