#!/usr/bin/env python
"""
Benchmark script for evaluating TCR-BERT pretrained model.

TCR-BERT (`wukevin/tcr-bert`) is pre-trained on antigen binding classification
for 45 antigens. Given a TRB sequence, it predicts which of these 45 antigens
the TCR is likely to bind.

This script:
1. Uses the pretrained classification model to predict antigen binding
2. Compares predicted antigen probabilities with actual antigens in our data
3. Computes metrics based on prediction accuracy

Note: This only works for TRB sequences and antigens that overlap with the 45
antigens TCR-BERT was trained on.

Usage:
    # Single task
    python benchmark_tcrbert_pretrained.py \
        --task tra_trb_peptide_mhc_one \
        --data_dir /path/to/tcr_specificity/tcr90pep80 \
        --output_dir /path/to/results

    # Multiple tasks
    python benchmark_tcrbert_pretrained.py \
        --task tra_trb_peptide_mhc_one tra_trb_peptide_mhc_two \
        --data_dir /path/to/tcr_specificity/tcr90pep80 \
        --output_dir /path/to/results

    # All tasks
    python benchmark_tcrbert_pretrained.py \
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

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, top_k_accuracy_score
import torch
from transformers import BertForSequenceClassification, pipeline

# Setup path for TCR-BERT
TCRBERT_ROOT = "/home/sravisha/tcrbench_tools/tcr-bert/tcr"
sys.path.insert(0, TCRBERT_ROOT)

import featurization as ft

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Task configurations
# Note: TCR-BERT pretrained classification only uses TRB, so all tasks use TRB
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

    # Filter out rows with missing TRB (required for pretrained model)
    valid_mask = result_df["trb"] != ''
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        logger.warning(f"Removing {n_invalid} rows with missing TRB chains")
    result_df = result_df[valid_mask].copy()

    return result_df


def load_pretrained_classifier(model_dir: str = "wukevin/tcr-bert", device: int = 0):
    """Load the pretrained TCR-BERT classification model.

    Args:
        model_dir: HuggingFace model name or local path
        device: GPU device ID (-1 for CPU)

    Returns:
        Tuple of (pipeline, label_mapping)
    """
    logger.info(f"Loading pretrained classifier from {model_dir}")

    # Load tokenizer
    try:
        tok = ft.get_pretrained_bert_tokenizer(model_dir)
    except OSError:
        logger.warning("Could not load saved tokenizer, loading fresh instance")
        tok = ft.get_aa_bert_tokenizer(64)

    # Load model
    model = BertForSequenceClassification.from_pretrained(model_dir)

    # Create pipeline
    device_id = device if device >= 0 and torch.cuda.is_available() else -1
    clf_pipeline = pipeline(
        "text-classification",
        model=model,
        tokenizer=tok,
        device=device_id,
        top_k=None,  # Return all scores
    )

    # Get label mapping from model config
    label_mapping = model.config.id2label
    logger.info(f"Model has {len(label_mapping)} antigen classes")
    logger.info(f"Antigens: {list(label_mapping.values())[:10]}...")  # Show first 10

    return clf_pipeline, label_mapping


def predict_antigens(
    clf_pipeline,
    trb_sequences: List[str],
    batch_size: int = 32,
) -> pd.DataFrame:
    """Predict antigen binding for TRB sequences.

    Args:
        clf_pipeline: Classification pipeline
        trb_sequences: List of TRB sequences
        batch_size: Batch size for inference

    Returns:
        DataFrame with antigen probabilities (rows=samples, cols=antigens)
    """
    # Add whitespace for tokenizer (TCR-BERT expects space-separated amino acids)
    trb_ws = [ft.insert_whitespace(seq) for seq in trb_sequences]

    all_preds = []
    for i in range(0, len(trb_ws), batch_size):
        batch = trb_ws[i:i + batch_size]
        preds = clf_pipeline(batch)
        all_preds.extend(preds)

    # Convert to DataFrame
    # Each prediction is a list of dicts like [{'label': 'ANTIGEN', 'score': 0.1}, ...]
    rows = []
    for pred in all_preds:
        row = {p['label']: p['score'] for p in pred}
        rows.append(row)

    pred_df = pd.DataFrame(rows)
    return pred_df


def evaluate_predictions(
    pred_df: pd.DataFrame,
    true_peptides: List[str],
    peptides_in_data: List[str],
) -> Dict:
    """Evaluate predictions against true peptides.

    Args:
        pred_df: DataFrame with antigen probabilities
        true_peptides: List of true peptide labels
        peptides_in_data: Unique peptides in the dataset

    Returns:
        Dict with evaluation metrics
    """
    # Find overlap between model's antigens and data's peptides
    model_antigens = set(pred_df.columns)
    data_peptides = set(peptides_in_data)
    overlapping = model_antigens.intersection(data_peptides)

    logger.info(f"Model antigens: {len(model_antigens)}")
    logger.info(f"Data peptides: {len(data_peptides)}")
    logger.info(f"Overlapping: {len(overlapping)}")

    if len(overlapping) == 0:
        logger.warning("No overlap between model antigens and data peptides!")
        return {
            "error": "No overlap between model antigens and data peptides",
            "model_antigens": list(model_antigens),
            "data_peptides": list(data_peptides),
        }

    # Filter to samples where true peptide is in model's vocabulary
    mask = [p in overlapping for p in true_peptides]
    n_evaluable = sum(mask)
    logger.info(f"Evaluable samples (peptide in model vocab): {n_evaluable}/{len(true_peptides)}")

    if n_evaluable == 0:
        return {
            "error": "No evaluable samples - none of the true peptides are in model vocabulary",
            "overlapping_peptides": list(overlapping),
        }

    # Get predictions and labels for evaluable samples
    pred_df_eval = pred_df[mask].reset_index(drop=True)
    true_peptides_eval = [p for p, m in zip(true_peptides, mask) if m]

    # Metrics
    results = {
        "n_samples_total": len(true_peptides),
        "n_samples_evaluable": n_evaluable,
        "n_overlapping_peptides": len(overlapping),
        "overlapping_peptides": list(overlapping),
    }

    # Per-peptide evaluation
    per_peptide_results = {}
    for peptide in overlapping:
        peptide_mask = [p == peptide for p in true_peptides_eval]
        if sum(peptide_mask) == 0:
            continue

        # Get scores for this peptide
        peptide_scores = pred_df_eval[peptide].values

        # Binary: is this the true peptide?
        true_labels = np.array([1 if p == peptide else 0 for p in true_peptides_eval])

        # Compute AUC if we have both positive and negative samples
        if len(np.unique(true_labels)) > 1:
            try:
                auc = roc_auc_score(true_labels, peptide_scores)
                per_peptide_results[peptide] = {
                    "auc": float(auc),
                    "n_positive": int(sum(true_labels)),
                    "n_total": len(true_labels),
                }
            except Exception as e:
                logger.warning(f"Error computing AUC for {peptide}: {e}")

    results["per_peptide"] = per_peptide_results

    # Aggregate AUC
    auc_values = [r["auc"] for r in per_peptide_results.values()]
    if auc_values:
        results["mean_auc"] = float(np.mean(auc_values))
        results["std_auc"] = float(np.std(auc_values))
        results["n_peptides_evaluated"] = len(auc_values)
    else:
        results["mean_auc"] = None
        results["std_auc"] = None
        results["n_peptides_evaluated"] = 0

    # Top-k accuracy (is true peptide in top k predictions?)
    all_peptides = list(pred_df_eval.columns)
    pred_matrix = pred_df_eval.values
    true_indices = [all_peptides.index(p) if p in all_peptides else -1 for p in true_peptides_eval]

    for k in [1, 3, 5, 10]:
        if k <= len(all_peptides):
            top_k_preds = np.argsort(-pred_matrix, axis=1)[:, :k]
            correct = sum(1 for i, true_idx in enumerate(true_indices)
                         if true_idx >= 0 and true_idx in top_k_preds[i])
            results[f"top_{k}_accuracy"] = float(correct / n_evaluable)

    return results


def run_benchmark(
    data_dir: str,
    task_config: Dict,
    output_dir: str,
    batch_size: int = 32,
    device: int = 0,
    model_dir: str = "wukevin/tcr-bert",
) -> Dict:
    """Run TCR-BERT pretrained benchmark evaluation.

    Args:
        data_dir: Path to data directory with train/val/test parquet files
        task_config: Task configuration dict
        output_dir: Output directory for results
        batch_size: Batch size for inference
        device: GPU device ID
        model_dir: HuggingFace model name or path

    Returns:
        Dict with evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Construct data path based on MHC class and task name
    # Structure: {data_dir}/{mhc_class}/{task_name}/
    mhc_class = task_config["mhc_class"]
    task_name = [k for k, v in TCRBERT_TASK_CONFIGS.items() if v == task_config][0]
    task_data_dir = os.path.join(data_dir, mhc_class, task_name)

    if not os.path.isdir(task_data_dir):
        # Try without task_name (old structure)
        alt_path = os.path.join(data_dir, mhc_class)
        if os.path.isdir(alt_path):
            logger.warning(f"Task subdirectory not found: {task_data_dir}, using {alt_path}")
            task_data_dir = alt_path
        else:
            logger.warning(f"MHC class subdirectory not found: {alt_path}, using {data_dir}")
            task_data_dir = data_dir

    logger.info(f"Using data directory: {task_data_dir}")

    # Load pretrained classifier
    clf_pipeline, label_mapping = load_pretrained_classifier(model_dir, device)

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

        # Get predictions
        logger.info(f"Getting predictions for {len(test_df)} samples...")
        pred_df = predict_antigens(
            clf_pipeline,
            test_df["trb"].tolist(),
            batch_size=batch_size,
        )

        # Evaluate
        results = evaluate_predictions(
            pred_df,
            test_df["peptide"].tolist(),
            test_df["peptide"].unique().tolist(),
        )

        all_results[split] = results

        if results.get("mean_auc") is not None:
            logger.info(f"{split}: Mean AUC = {results['mean_auc']:.4f} over {results['n_peptides_evaluated']} peptides")
        else:
            logger.info(f"{split}: Could not compute AUC - {results.get('error', 'unknown error')}")

    # Save results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark TCR-BERT pretrained model")
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
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device ID (-1 for CPU)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="wukevin/tcr-bert",
        help="HuggingFace model name or local path",
    )

    args = parser.parse_args()

    # Handle 'all' tasks or validate task names
    if args.task == ["all"] or "all" in args.task:
        tasks = list(TCRBERT_TASK_CONFIGS.keys())
    else:
        invalid_tasks = [t for t in args.task if t not in TCRBERT_TASK_CONFIGS]
        if invalid_tasks:
            parser.error(f"Invalid task(s): {invalid_tasks}. Valid tasks: {list(TCRBERT_TASK_CONFIGS.keys())}")
        tasks = args.task

    logger.info(f"Running TCR-BERT pretrained benchmark for tasks: {tasks}")
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
            batch_size=args.batch_size,
            device=args.device,
            model_dir=args.model_dir,
        )

        all_task_results[task] = results

    # Save combined results
    combined_results_path = os.path.join(args.output_dir, "all_results.json")
    with open(combined_results_path, "w") as f:
        json.dump(all_task_results, f, indent=2)
    logger.info(f"Combined results saved to {combined_results_path}")

    # Print summary
    print("\n" + "="*60)
    print("TCR-BERT Pretrained Benchmark Results Summary")
    print("="*60)
    for task, task_results in all_task_results.items():
        print(f"\nTask: {task}")
        print("-" * 40)
        for split, res in task_results.items():
            if res.get("mean_auc") is not None:
                print(f"  {split}: AUC = {res['mean_auc']:.4f} ± {res['std_auc']:.4f} ({res['n_peptides_evaluated']} peptides)")
                for k in [1, 3, 5, 10]:
                    if f"top_{k}_accuracy" in res:
                        print(f"    Top-{k} Accuracy: {res[f'top_{k}_accuracy']:.4f}")
            elif "error" in res:
                print(f"  {split}: {res['error']}")
    print("="*60)


if __name__ == "__main__":
    main()
