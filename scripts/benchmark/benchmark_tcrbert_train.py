#!/usr/bin/env python
"""
Benchmark script for training and evaluating TCR-BERT.

This script trains the TwoPartBertClassifier model on TCR-peptide binding data
and evaluates on test splits. Unlike the pretrained script which uses embeddings +
logistic regression, this script fine-tunes the full TCR-BERT model.

TCR-BERT processes paired TRA/TRB sequences. Single-chain tasks are NOT supported.

Usage:
    # Single task
    python benchmark_tcrbert_train.py \
        --task tra_trb_peptide_mhc_one \
        --data_dir /path/to/tcr_specificity/tcr90pep80 \
        --output_dir /path/to/results \
        --num_epochs 50 \
        --batch_size 64 \
        --lr 3e-5 \
        --neg_per_pos 5

    # Multiple tasks
    python benchmark_tcrbert_train.py \
        --task tra_trb_peptide_mhc_one tra_trb_peptide_mhc_two \
        --data_dir /path/to/tcr_specificity/tcr90pep80 \
        --output_dir /path/to/results

    # All tasks
    python benchmark_tcrbert_train.py \
        --task all \
        --data_dir /path/to/tcr_specificity/tcr90pep80 \
        --output_dir /path/to/results

Author: Claude
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

# Optional: wandb integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Setup path for TCR-BERT
TCRBERT_ROOT = "/home/sravisha/tcrbench_tools/tcr-bert/tcr"
sys.path.insert(0, TCRBERT_ROOT)
sys.path.insert(0, os.path.join(TCRBERT_ROOT, "models"))

import data_loader as dl
import featurization as ft
from transformer_custom import TwoPartBertClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Task configurations
# Note: TCR-BERT requires paired TRA/TRB, so only tra_trb_* tasks are supported
TCRBERT_TASK_CONFIGS = {
    "tra_trb_peptide_mhc_one": {
        "mhc_class": "class_one",
        "mhc_cols": ["mhc_one_id"],
        "tcr_cols": ["tra", "trb"],
    },
    "tra_trb_peptide_mhc_two": {
        "mhc_class": "class_two",
        "mhc_cols": ["mhc_two_id"],
        "tcr_cols": ["tra", "trb"],
    },
}


def safe_string(val) -> str:
    """Convert value to string, replacing None/NaN/empty with empty string."""
    if val is None or pd.isna(val) or val == '':
        return ''
    return str(val)


def load_parquet_data(parquet_path: str, task_config: Dict) -> pd.DataFrame:
    """Load and process parquet data for TCR-BERT evaluation.

    Args:
        parquet_path: Path to parquet file
        task_config: Task configuration dict

    Returns:
        DataFrame with tra, trb, peptide columns
    """
    df = pd.read_parquet(parquet_path)

    # Create processed dataframe
    result_df = pd.DataFrame(index=df.index)
    result_df["tra"] = df["tra"].apply(safe_string)
    result_df["trb"] = df["trb"].apply(safe_string)
    result_df["peptide"] = df["peptide"].apply(safe_string)

    # Filter out rows with missing TCR chains (TCR-BERT requires both)
    valid_mask = (result_df["tra"] != '') & (result_df["trb"] != '')
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        logger.warning(f"Removing {n_invalid} rows with missing TRA or TRB chains")
    result_df = result_df[valid_mask].copy()

    return result_df


def generate_negatives(positive_df: pd.DataFrame, neg_per_pos: int = 5, seed: int = 42) -> pd.DataFrame:
    """Generate negative samples by mismatching TCRs with peptides.

    For each positive TCR-peptide pair, sample TCRs from other peptides
    as negative examples.

    Args:
        positive_df: DataFrame with positive samples (tra, trb, peptide)
        neg_per_pos: Number of negatives per positive
        seed: Random seed

    Returns:
        DataFrame with positive and negative samples, with 'binder' column
    """
    rng = np.random.default_rng(seed)

    # Group by peptide
    peptide_groups = positive_df.groupby("peptide")

    # Get all unique TCRs per peptide for exclusion
    peptide_tcrs = {}
    for peptide, group in peptide_groups:
        tcr_set = set(zip(group["tra"], group["trb"]))
        peptide_tcrs[peptide] = tcr_set

    # All TCRs pool
    all_tcrs = list(zip(positive_df["tra"], positive_df["trb"]))

    # Generate negatives
    negatives = []
    for idx, row in positive_df.iterrows():
        peptide = row["peptide"]
        known_binders = peptide_tcrs.get(peptide, set())

        # Sample negatives
        sampled = 0
        attempts = 0
        max_attempts = neg_per_pos * 10

        while sampled < neg_per_pos and attempts < max_attempts:
            # Random TCR
            rand_idx = rng.integers(0, len(all_tcrs))
            tcr_a, tcr_b = all_tcrs[rand_idx]

            # Check not a known binder for this peptide
            if (tcr_a, tcr_b) not in known_binders:
                negatives.append({
                    "tra": tcr_a,
                    "trb": tcr_b,
                    "peptide": peptide,
                    "binder": 0
                })
                sampled += 1
            attempts += 1

    # Combine positives and negatives
    positive_df = positive_df.copy()
    positive_df["binder"] = 1

    neg_df = pd.DataFrame(negatives)
    combined_df = pd.concat([positive_df, neg_df], ignore_index=True)

    logger.info(f"Generated {len(neg_df)} negatives for {len(positive_df)} positives")

    return combined_df


def create_dataset(
    df: pd.DataFrame,
    skorch_mode: bool = False,
) -> dl.TcrFineTuneDataset:
    """Create TCR-BERT dataset from dataframe.

    Args:
        df: DataFrame with tra, trb, binder columns
        skorch_mode: Whether to use skorch format

    Returns:
        TcrFineTuneDataset
    """
    dataset = dl.TcrFineTuneDataset(
        tcr_a_seqs=df["tra"].tolist(),
        tcr_b_seqs=df["trb"].tolist(),
        labels=df["binder"].values.astype(np.float32),
        skorch_mode=skorch_mode,
    )
    return dataset


def initialize_model(
    pretrained: str = "wukevin/tcr-bert",
    n_output: int = 2,
    freeze_encoder: bool = False,
    separate_encoders: bool = True,
    dropout: float = 0.2,
    seq_pooling: str = "cls",
    device: str = "cuda",
) -> TwoPartBertClassifier:
    """Initialize TCR-BERT model.

    Args:
        pretrained: Pretrained model path or HuggingFace name
        n_output: Number of output classes
        freeze_encoder: Whether to freeze encoder weights
        separate_encoders: Use separate encoders for TRA/TRB
        dropout: Dropout probability
        seq_pooling: Pooling strategy (cls, mean, max, pool)
        device: Device to use

    Returns:
        Initialized TwoPartBertClassifier model
    """
    model = TwoPartBertClassifier(
        pretrained=pretrained,
        n_output=n_output,
        freeze_encoder=freeze_encoder,
        separate_encoders=separate_encoders,
        dropout=dropout,
        seq_pooling=seq_pooling,
    )
    model = model.to(device)
    return model


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    """Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        inputs, labels = batch
        tcr_a = inputs["tcr_a"].to(device)
        tcr_b = inputs["tcr_b"].to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(tcr_a, tcr_b)

        # Handle label format
        if labels.dim() == 1:
            # Binary labels
            loss = criterion(logits, labels.long())
        else:
            # One-hot labels
            loss = criterion(logits, labels.argmax(dim=1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model.

    Args:
        model: Model to evaluate
        dataloader: Evaluation dataloader
        device: Device

    Returns:
        Tuple of (loss, predictions, labels)
    """
    model.eval()
    all_preds = []
    all_labels = []

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            tcr_a = inputs["tcr_a"].to(device)
            tcr_b = inputs["tcr_b"].to(device)
            labels = labels.to(device)

            logits = model(tcr_a, tcr_b)

            # Get probabilities
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_preds.extend(probs)

            # Get true labels
            if labels.dim() == 1:
                true_labels = labels.cpu().numpy()
            else:
                true_labels = labels.argmax(dim=1).cpu().numpy()
            all_labels.extend(true_labels)

            # Loss
            if labels.dim() == 1:
                loss = criterion(logits, labels.long())
            else:
                loss = criterion(logits, labels.argmax(dim=1))
            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches, np.array(all_preds), np.array(all_labels)


def evaluate_per_peptide(
    model: nn.Module,
    dataloader: DataLoader,
    peptides: np.ndarray,
    device: str,
) -> Dict:
    """Evaluate predictions per peptide.

    Args:
        model: Model to evaluate
        dataloader: Evaluation dataloader
        peptides: Peptide identifiers for each sample
        device: Device

    Returns:
        Dict with per-peptide and aggregate metrics
    """
    loss, probs, labels = evaluate(model, dataloader, device)

    # Per-peptide evaluation
    unique_peptides = np.unique(peptides)
    per_peptide_results = {}
    auc_values = []

    for peptide in unique_peptides:
        mask = peptides == peptide
        pep_labels = labels[mask]
        pep_probs = probs[mask]

        # Skip if only one class present
        if len(np.unique(pep_labels)) < 2:
            logger.warning(f"Peptide {peptide}: Only one class present, skipping AUC")
            continue

        try:
            auc = roc_auc_score(pep_labels, pep_probs)
            per_peptide_results[peptide] = {
                "auc": float(auc),
                "n_samples": int(mask.sum()),
                "n_positive": int(pep_labels.sum()),
            }
            auc_values.append(auc)
        except Exception as e:
            logger.warning(f"Error computing AUC for peptide {peptide}: {e}")

    # Aggregate metrics
    results = {
        "per_peptide": per_peptide_results,
        "mean_auc": float(np.mean(auc_values)) if auc_values else None,
        "std_auc": float(np.std(auc_values)) if auc_values else None,
        "n_peptides_evaluated": len(auc_values),
        "n_total_samples": len(labels),
        "loss": float(loss),
    }

    return results


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    val_peptides: np.ndarray,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
    output_dir: str,
    use_wandb: bool = False,
) -> nn.Module:
    """Train the model.

    Args:
        model: Model to train
        train_dataloader: Training dataloader
        val_dataloader: Validation dataloader
        val_peptides: Peptide identifiers for validation samples
        num_epochs: Number of epochs
        lr: Learning rate
        weight_decay: Weight decay
        device: Device
        output_dir: Output directory for checkpoints
        use_wandb: Whether to log to wandb

    Returns:
        Trained model
    """
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=num_epochs,
    )

    best_val_auc = 0.0
    training_log = []

    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)

        # Evaluate
        val_results = evaluate_per_peptide(model, val_dataloader, val_peptides, device)
        val_auc = val_results["mean_auc"] or 0.0

        # Log
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_results["loss"],
            "val_auc": val_auc,
            "lr": scheduler.get_last_lr()[0],
        }
        training_log.append(log_entry)

        logger.info(
            f"Epoch {epoch+1}/{num_epochs}: "
            f"Train Loss = {train_loss:.4f}, "
            f"Val Loss = {val_results['loss']:.4f}, "
            f"Val AUC = {val_auc:.4f}"
        )

        if use_wandb and WANDB_AVAILABLE:
            wandb.log(log_entry)

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            checkpoint_path = os.path.join(output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
            }, checkpoint_path)
            logger.info(f"Saved best model with val AUC = {val_auc:.4f}")

        scheduler.step()

    # Save training log
    log_path = os.path.join(output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    # Load best model
    checkpoint_path = os.path.join(output_dir, "best_model.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")

    return model


def run_benchmark(
    data_dir: str,
    task_config: Dict,
    output_dir: str,
    num_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-5,
    weight_decay: float = 0.01,
    neg_per_pos: int = 5,
    dropout: float = 0.2,
    freeze_encoder: bool = False,
    separate_encoders: bool = True,
    seq_pooling: str = "cls",
    pretrained: str = "wukevin/tcr-bert",
    device: int = 0,
    use_wandb: bool = False,
    wandb_project: str = "tcrbert-benchmark",
) -> Dict:
    """Run TCR-BERT training and evaluation benchmark.

    Args:
        data_dir: Path to data directory
        task_config: Task configuration
        output_dir: Output directory
        num_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        weight_decay: Weight decay
        neg_per_pos: Negatives per positive
        dropout: Dropout rate
        freeze_encoder: Freeze encoder weights
        separate_encoders: Use separate encoders
        seq_pooling: Pooling strategy
        pretrained: Pretrained model path
        device: GPU device ID
        use_wandb: Use wandb logging
        wandb_project: Wandb project name

    Returns:
        Dict with results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Construct data path based on MHC class and task name
    # Structure: {data_dir}/{mhc_class}/{task_name}/
    mhc_class = task_config["mhc_class"]
    task_name = [k for k, v in TCRBERT_TASK_CONFIGS.items() if v == task_config][0]
    task_data_dir = os.path.join(data_dir, mhc_class, task_name)

    # Fall back to alternative paths if not found
    if not os.path.isdir(task_data_dir):
        alt_path = os.path.join(data_dir, mhc_class)
        if os.path.isdir(alt_path):
            logger.warning(f"Task subdirectory not found: {task_data_dir}, using {alt_path}")
            task_data_dir = alt_path
        else:
            logger.warning(f"MHC class subdirectory not found: {alt_path}, using {data_dir}")
            task_data_dir = data_dir

    logger.info(f"Using data directory: {task_data_dir}")

    # Setup device
    device_str = f"cuda:{device}" if device >= 0 and torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device_str}")

    # Initialize wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            config={
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "neg_per_pos": neg_per_pos,
                "dropout": dropout,
                "freeze_encoder": freeze_encoder,
                "separate_encoders": separate_encoders,
                "seq_pooling": seq_pooling,
                "pretrained": pretrained,
            },
        )

    # Load data
    logger.info(f"Loading training data from {task_data_dir}...")
    train_df = load_parquet_data(os.path.join(task_data_dir, "train.parquet"), task_config)
    train_with_neg = generate_negatives(train_df, neg_per_pos=neg_per_pos, seed=42)

    logger.info(f"Loading validation data from {task_data_dir}...")
    val_df = load_parquet_data(os.path.join(task_data_dir, "val.parquet"), task_config)
    val_with_neg = generate_negatives(val_df, neg_per_pos=neg_per_pos, seed=43)

    # Create datasets
    train_dataset = create_dataset(train_with_neg, skorch_mode=True)
    val_dataset = create_dataset(val_with_neg, skorch_mode=True)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize model
    logger.info("Initializing model...")
    model = initialize_model(
        pretrained=pretrained,
        n_output=2,
        freeze_encoder=freeze_encoder,
        separate_encoders=separate_encoders,
        dropout=dropout,
        seq_pooling=seq_pooling,
        device=device_str,
    )

    # Train model
    logger.info("Starting training...")
    model = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        val_peptides=val_with_neg["peptide"].values,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device_str,
        output_dir=output_dir,
        use_wandb=use_wandb,
    )

    # Evaluate on test splits
    test_splits = [
        "test_seen_epitope",
        "test_unseen_epitope",
        "test_unseen_tcr_seen_epitope",
        "test_unseen_allele",
    ]

    all_results = {}
    for split in test_splits:
        test_path = os.path.join(task_data_dir, f"{split}.parquet")
        if not os.path.exists(test_path):
            logger.warning(f"Test file not found: {test_path}")
            continue

        logger.info(f"Evaluating on {split}...")
        test_df = load_parquet_data(test_path, task_config)

        if len(test_df) == 0:
            logger.warning(f"No valid samples in {split}")
            continue

        test_with_neg = generate_negatives(test_df, neg_per_pos=neg_per_pos, seed=44)
        test_dataset = create_dataset(test_with_neg, skorch_mode=True)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )

        # Evaluate
        results = evaluate_per_peptide(
            model,
            test_dataloader,
            test_with_neg["peptide"].values,
            device_str,
        )

        all_results[split] = results
        logger.info(f"{split}: Mean AUC = {results['mean_auc']:.4f} over {results['n_peptides_evaluated']} peptides")

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({f"{split}_auc": results["mean_auc"]})

    # Save results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate TCR-BERT")
    parser.add_argument(
        "--task",
        type=str,
        nargs="+",
        required=True,
        help="Task(s) to evaluate. Can specify multiple tasks or 'all' for all tasks.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to data directory with train/val/test parquet files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results and checkpoints",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=25,
        help="Number of training epochs (TCR-BERT default: 25)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size (TCR-BERT default: 128)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
        help="Learning rate (TCR-BERT default: 3e-5)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay (TCR-BERT default: 0.0)",
    )
    parser.add_argument(
        "--neg_per_pos",
        type=int,
        default=5,
        help="Number of negative samples per positive",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate (TCR-BERT default: 0.2)",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze encoder weights (TCR-BERT default: False)",
    )
    parser.add_argument(
        "--shared_encoder",
        action="store_true",
        help="Use shared encoder for TRA/TRB (TCR-BERT default: False, i.e. separate encoders)",
    )
    parser.add_argument(
        "--seq_pooling",
        type=str,
        default="cls",
        choices=["cls", "mean", "max", "pool"],
        help="Sequence pooling strategy (TCR-BERT default: cls)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="wukevin/tcr-bert-mlm-only",
        help="Pretrained model path or HuggingFace name (TCR-BERT default: wukevin/tcr-bert-mlm-only)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device ID (-1 for CPU)",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use wandb for logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="tcrbert-benchmark",
        help="Wandb project name",
    )

    args = parser.parse_args()

    # Handle 'all' tasks or validate task names
    if args.task == ["all"] or "all" in args.task:
        tasks = list(TCRBERT_TASK_CONFIGS.keys())
    else:
        # Validate task names
        invalid_tasks = [t for t in args.task if t not in TCRBERT_TASK_CONFIGS]
        if invalid_tasks:
            parser.error(f"Invalid task(s): {invalid_tasks}. Valid tasks: {list(TCRBERT_TASK_CONFIGS.keys())}")
        tasks = args.task

    logger.info(f"Training TCR-BERT for tasks: {tasks}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    all_task_results = {}

    for task in tasks:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing task: {task}")
        logger.info(f"{'='*60}")

        task_config = TCRBERT_TASK_CONFIGS[task]
        task_output_dir = os.path.join(args.output_dir, task)

        results = run_benchmark(
            data_dir=args.data_dir,
            task_config=task_config,
            output_dir=task_output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            neg_per_pos=args.neg_per_pos,
            dropout=args.dropout,
            freeze_encoder=args.freeze_encoder,
            separate_encoders=not args.shared_encoder,
            seq_pooling=args.seq_pooling,
            pretrained=args.pretrained,
            device=args.device,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
        )

        all_task_results[task] = results

    # Save combined results
    combined_results_path = os.path.join(args.output_dir, "all_results.json")
    with open(combined_results_path, "w") as f:
        json.dump(all_task_results, f, indent=2)
    logger.info(f"Combined results saved to {combined_results_path}")

    # Print summary
    print("\n" + "="*60)
    print("TCR-BERT Training Benchmark Results Summary")
    print("="*60)
    for task, task_results in all_task_results.items():
        print(f"\nTask: {task}")
        print("-" * 40)
        for split, res in task_results.items():
            if res.get("mean_auc") is not None:
                print(f"  {split}: AUC = {res['mean_auc']:.4f} ± {res['std_auc']:.4f} ({res['n_peptides_evaluated']} peptides)")
    print("="*60)


if __name__ == "__main__":
    main()
