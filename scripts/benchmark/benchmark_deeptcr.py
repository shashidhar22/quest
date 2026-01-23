#!/usr/bin/env python
"""
DeepTCR Benchmark Script for TCR-Peptide Specificity Prediction

This script evaluates DeepTCR's out-of-the-box performance on the task of
predicting peptide specificity given TCR and MHC information.

This is a pure benchmark evaluation - we are NOT modifying or optimizing DeepTCR.
The goal is to assess how well DeepTCR performs using default parameters.
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.benchmark.benchmark_metrics import compute_all_unified_metrics

# Add DeepTCR to path
sys.path.insert(0, '/home/sravisha/tcrbench_tools/DeepTCR')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')


# Task configurations for all 6 tasks
TASK_CONFIGS = {
    "trb_peptide_mhc_one": {
        "mhc_class": "class_one",
        "mhc_cols": ["mhc_one_id"],
        "tcr_cols": ["trb"],
        "gene_cols": {
            "v_beta": "trbv_gene",
            "d_beta": "trbd_gene",
            "j_beta": "trbj_gene",
        },
        "description": "TCR beta only + MHC-I -> peptide",
    },
    "tra_peptide_mhc_one": {
        "mhc_class": "class_one",
        "mhc_cols": ["mhc_one_id"],
        "tcr_cols": ["tra"],
        "gene_cols": {
            "v_alpha": "trav_gene",
            "j_alpha": "traj_gene",
        },
        "description": "TCR alpha only + MHC-I -> peptide",
    },
    "tra_trb_peptide_mhc_one": {
        "mhc_class": "class_one",
        "mhc_cols": ["mhc_one_id"],
        "tcr_cols": ["tra", "trb"],
        "gene_cols": {
            "v_alpha": "trav_gene",
            "j_alpha": "traj_gene",
            "v_beta": "trbv_gene",
            "d_beta": "trbd_gene",
            "j_beta": "trbj_gene",
        },
        "description": "TCR alpha+beta paired + MHC-I -> peptide",
    },
    "trb_peptide_mhc_two": {
        "mhc_class": "class_two",
        "mhc_cols": ["mhc_one_id", "mhc_two_id"],
        "tcr_cols": ["trb"],
        "gene_cols": {
            "v_beta": "trbv_gene",
            "d_beta": "trbd_gene",
            "j_beta": "trbj_gene",
        },
        "description": "TCR beta only + MHC-II -> peptide",
    },
    "tra_peptide_mhc_two": {
        "mhc_class": "class_two",
        "mhc_cols": ["mhc_one_id", "mhc_two_id"],
        "tcr_cols": ["tra"],
        "gene_cols": {
            "v_alpha": "trav_gene",
            "j_alpha": "traj_gene",
        },
        "description": "TCR alpha only + MHC-II -> peptide",
    },
    "tra_trb_peptide_mhc_two": {
        "mhc_class": "class_two",
        "mhc_cols": ["mhc_one_id", "mhc_two_id"],
        "tcr_cols": ["tra", "trb"],
        "gene_cols": {
            "v_alpha": "trav_gene",
            "j_alpha": "traj_gene",
            "v_beta": "trbv_gene",
            "d_beta": "trbd_gene",
            "j_beta": "trbj_gene",
        },
        "description": "TCR alpha+beta paired + MHC-II -> peptide",
    },
}

TEST_SPLITS = [
    "test_seen_epitope",
    "test_unseen_tcr_seen_epitope",
    "test_unseen_epitope",
    "test_unseen_allele",
]


def format_mhc_for_deeptcr(mhc_id: str, mhc_class: str) -> tuple:
    """
    Convert MHC allele to DeepTCR format.

    DeepTCR expects HLA in format like 'A0201' (no asterisk/colon).
    For MHC Class II, we combine alpha+beta chains.

    Args:
        mhc_id: MHC allele string (e.g., 'A*02:01' or 'DRA*01:01_DRB1*01:01')
        mhc_class: 'class_one' or 'class_two'

    Returns:
        Tuple of formatted allele strings
    """
    if pd.isna(mhc_id) or mhc_id == '':
        return tuple()

    if mhc_class == "class_one":
        # Class I: Single allele like 'A*02:01' -> 'A0201'
        formatted = mhc_id.replace('*', '').replace(':', '')
        return (formatted,)
    else:
        # Class II: Combined format like 'DRA*01:01_DRB1*01:01'
        # Split by underscore and format each part
        parts = mhc_id.split('_')
        formatted = tuple(p.replace('*', '').replace(':', '') for p in parts)
        return formatted


def load_parquet_to_arrays(parquet_path: str, task_config: dict) -> dict:
    """
    Load parquet file and convert to numpy arrays for DeepTCR.

    Args:
        parquet_path: Path to parquet file
        task_config: Task configuration dictionary

    Returns:
        Dictionary with numpy arrays for DeepTCR
    """
    df = pd.read_parquet(parquet_path)

    # Filter out rows with NaN in required columns
    required_cols = task_config["tcr_cols"] + ["peptide"] + task_config.get("mhc_cols", ["mhc_one_id"])
    for col in required_cols:
        if col in df.columns:
            df = df[df[col].notna()]

    result = {}

    # TCR sequences - fillna with empty string and uppercase for DeepTCR
    if "tra" in task_config["tcr_cols"]:
        result["alpha_sequences"] = df["tra"].fillna('').str.upper().values.astype(str)
    if "trb" in task_config["tcr_cols"]:
        result["beta_sequences"] = df["trb"].fillna('').str.upper().values.astype(str)

    # V/D/J genes - fillna with empty string
    gene_cols = task_config["gene_cols"]
    if "v_alpha" in gene_cols:
        result["v_alpha"] = df[gene_cols["v_alpha"]].fillna('').values.astype(str)
    if "j_alpha" in gene_cols:
        result["j_alpha"] = df[gene_cols["j_alpha"]].fillna('').values.astype(str)
    if "v_beta" in gene_cols:
        result["v_beta"] = df[gene_cols["v_beta"]].fillna('').values.astype(str)
    if "d_beta" in gene_cols:
        result["d_beta"] = df[gene_cols["d_beta"]].fillna('').values.astype(str)
    if "j_beta" in gene_cols:
        result["j_beta"] = df[gene_cols["j_beta"]].fillna('').values.astype(str)

    # Class labels (peptide sequences) - uppercase for consistency
    result["class_labels"] = df["peptide"].fillna('').str.upper().values.astype(str)

    # HLA/MHC alleles - format as array of tuples
    mhc_class = task_config["mhc_class"]
    mhc_cols = task_config.get("mhc_cols", ["mhc_one_id"])

    if len(mhc_cols) == 1:
        # Class I: single allele column
        mhc_values = df[mhc_cols[0]].fillna('').values
    else:
        # Class II: combine two allele columns with underscore
        col1 = df[mhc_cols[0]].fillna('')
        col2 = df[mhc_cols[1]].fillna('')
        mhc_values = (col1.astype(str) + "_" + col2.astype(str)).values

    hla_tuples = [
        format_mhc_for_deeptcr(allele, mhc_class)
        for allele in mhc_values
    ]
    result["hla"] = np.array(hla_tuples, dtype=object)

    return result


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
                    class_labels: list) -> dict:
    """
    Compute evaluation metrics.

    Args:
        y_true: True class indices
        y_pred: Predicted class indices
        y_prob: Prediction probabilities (N x num_classes)
        class_labels: List of class label names

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Top-1 accuracy
    metrics["top1_accuracy"] = float(accuracy_score(y_true, y_pred))

    # Top-5 accuracy (if we have enough classes)
    num_classes = y_prob.shape[1]
    if num_classes >= 5:
        metrics["top5_accuracy"] = float(
            top_k_accuracy_score(y_true, y_prob, k=5, labels=range(num_classes))
        )
    else:
        metrics["top5_accuracy"] = float(
            top_k_accuracy_score(y_true, y_prob, k=min(5, num_classes), labels=range(num_classes))
        )

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
            # Create one-hot encoding for y_true
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


def train_and_evaluate(
    task_name: str,
    data_dir: str,
    output_dir: str,
    use_hla: bool = True,
    suppress_output: bool = False,
):
    """
    Train DeepTCR model and evaluate on all test splits.

    Args:
        task_name: Name of the task (e.g., 'trb_peptide_mhc_one')
        data_dir: Base data directory
        output_dir: Output directory for results
        use_hla: Whether to use HLA information
        suppress_output: Whether to suppress training output
    """
    from DeepTCR.DeepTCR import DeepTCR_SS

    task_config = TASK_CONFIGS[task_name]
    mhc_class = task_config["mhc_class"]
    task_data_dir = os.path.join(data_dir, mhc_class, task_name)

    # Create output directory
    hla_suffix = "with_hla" if use_hla else "without_hla"
    task_output_dir = os.path.join(output_dir, task_name, hla_suffix)
    os.makedirs(task_output_dir, exist_ok=True)
    os.makedirs(os.path.join(task_output_dir, "predictions"), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"Description: {task_config['description']}")
    print(f"Use HLA: {use_hla}")
    print(f"{'='*60}")

    # Load training data
    print("\nLoading training data...")
    train_path = os.path.join(task_data_dir, "train.parquet")
    train_data = load_parquet_to_arrays(train_path, task_config)

    # Load validation data
    print("Loading validation data...")
    val_path = os.path.join(task_data_dir, "val.parquet")
    val_data = load_parquet_to_arrays(val_path, task_config)

    # Combine train + val for loading into DeepTCR
    # We'll use train for training and val for validation
    n_train = len(train_data["class_labels"])
    n_val = len(val_data["class_labels"])
    print(f"Train samples: {n_train}, Val samples: {n_val}")

    # Combine arrays
    combined_data = {}
    for key in train_data.keys():
        combined_data[key] = np.concatenate([train_data[key], val_data[key]])

    # Initialize DeepTCR
    model_name = os.path.join(task_output_dir, "model")
    dtcr = DeepTCR_SS(model_name, device=0, tf_verbosity=3)

    # Prepare Load_Data arguments
    load_args = {
        "class_labels": combined_data["class_labels"],
    }

    # Add TCR sequences
    if "alpha_sequences" in combined_data:
        load_args["alpha_sequences"] = combined_data["alpha_sequences"]
    if "beta_sequences" in combined_data:
        load_args["beta_sequences"] = combined_data["beta_sequences"]

    # Add V/D/J genes
    if "v_alpha" in combined_data:
        load_args["v_alpha"] = combined_data["v_alpha"]
    if "j_alpha" in combined_data:
        load_args["j_alpha"] = combined_data["j_alpha"]
    if "v_beta" in combined_data:
        load_args["v_beta"] = combined_data["v_beta"]
    if "d_beta" in combined_data:
        load_args["d_beta"] = combined_data["d_beta"]
    if "j_beta" in combined_data:
        load_args["j_beta"] = combined_data["j_beta"]

    # Add HLA if requested
    if use_hla:
        load_args["hla"] = combined_data["hla"]

    print("\nLoading data into DeepTCR...")
    dtcr.Load_Data(**load_args)

    # Create train/valid/test splits manually
    # For test, we use validation data (since actual test sets will use Sequence_Inference)
    print("\nSetting up train/valid splits...")

    # Get indices
    all_indices = np.arange(n_train + n_val)
    train_indices = all_indices[:n_train]
    val_indices = all_indices[n_train:]

    # We need to call Get_Train_Valid_Test to set up internal structures
    # Then override with our splits
    # Use a small test_size initially, we'll override anyway
    dtcr.Get_Train_Valid_Test(test_size=0.1)

    # Override with our custom splits
    # The structure follows the var_names from DeepTCR code
    Vars = [
        dtcr.X_Seq_alpha, dtcr.X_Seq_beta,
        dtcr.alpha_sequences, dtcr.beta_sequences,
        dtcr.sample_id, dtcr.class_id, np.arange(len(dtcr.class_id)),  # seq_index
        dtcr.v_beta_num, dtcr.d_beta_num, dtcr.j_beta_num,
        dtcr.v_alpha_num, dtcr.j_alpha_num,
        dtcr.v_beta, dtcr.d_beta, dtcr.j_beta,
        dtcr.v_alpha, dtcr.j_alpha,
        dtcr.hla_data_seq_num, dtcr.Y
    ]

    dtcr.train = [v[train_indices] for v in Vars]
    dtcr.valid = [v[val_indices] for v in Vars]
    # Use validation as test placeholder (we'll do real testing via Sequence_Inference)
    dtcr.test = [v[val_indices] for v in Vars]

    # Train the model with default parameters
    print("\nTraining model (default DeepTCR parameters)...")
    dtcr.Train(
        suppress_output=suppress_output,
        weight_by_class=True,  # Use class weighting for imbalanced data
    )

    # Store results
    results = {
        "task": task_name,
        "use_hla": use_hla,
        "n_train": n_train,
        "n_val": n_val,
        "n_classes": len(dtcr.lb.classes_),
        "classes": list(dtcr.lb.classes_),
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
        test_data = load_parquet_to_arrays(test_path, task_config)

        # Prepare inference arguments
        infer_args = {}
        if "alpha_sequences" in test_data:
            infer_args["alpha_sequences"] = test_data["alpha_sequences"]
        if "beta_sequences" in test_data:
            infer_args["beta_sequences"] = test_data["beta_sequences"]
        if "v_alpha" in test_data:
            infer_args["v_alpha"] = test_data["v_alpha"]
        if "j_alpha" in test_data:
            infer_args["j_alpha"] = test_data["j_alpha"]
        if "v_beta" in test_data:
            infer_args["v_beta"] = test_data["v_beta"]
        if "d_beta" in test_data:
            infer_args["d_beta"] = test_data["d_beta"]
        if "j_beta" in test_data:
            infer_args["j_beta"] = test_data["j_beta"]
        if use_hla:
            infer_args["hla"] = test_data["hla"]

        # Get predictions
        try:
            y_prob = dtcr.Sequence_Inference(**infer_args)

            # Convert true labels to indices
            y_true_labels = test_data["class_labels"]

            # Handle labels not seen during training
            known_mask = np.isin(y_true_labels, dtcr.lb.classes_)
            if not known_mask.all():
                print(f"    Warning: {(~known_mask).sum()} samples have unseen labels")

            # Only evaluate on samples with known labels
            y_true_labels_known = y_true_labels[known_mask]
            y_prob_known = y_prob[known_mask]

            if len(y_true_labels_known) == 0:
                print(f"    No samples with known labels - skipping")
                results["test_results"][split_name] = {
                    "error": "No samples with known labels",
                    "n_total": len(y_true_labels),
                    "n_known": 0,
                }
                continue

            y_true = dtcr.lb.transform(y_true_labels_known)
            y_pred = np.argmax(y_prob_known, axis=1)

            # Compute metrics
            metrics = compute_metrics(
                y_true, y_pred, y_prob_known, list(dtcr.lb.classes_)
            )
            metrics["n_total"] = int(len(y_true_labels))
            metrics["n_known_labels"] = int(len(y_true_labels_known))
            metrics["n_unknown_labels"] = int((~known_mask).sum())

            # Compute unified retrieval metrics
            # y_prob_known is (n_samples, n_classes) — use directly as scores
            candidate_peptides = list(dtcr.lb.classes_)
            sample_peptides = np.array(y_true_labels_known)

            unified = compute_all_unified_metrics(
                scores=y_prob_known,
                true_indices=y_true,
                sample_peptides=sample_peptides,
                candidate_peptides=candidate_peptides,
                n_bootstrap=1000,
                min_samples_per_peptide=5,
                seed=42,
            )

            # Extract per_epitope_all before merging
            per_epitope_all = unified.pop("per_epitope_all", [])
            metrics.update(unified)

            results["test_results"][split_name] = metrics

            print(f"    Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")
            print(f"    Macro F1: {metrics['macro_f1']:.4f}")
            if metrics.get('auc_roc_ovr'):
                print(f"    AUC-ROC: {metrics['auc_roc_ovr']:.4f}")
            print(f"    Retrieval Hit@1: {metrics['retrieval_hit_at_1']:.4f}")
            print(f"    Retrieval MRR: {metrics['retrieval_mrr']:.4f}")
            if metrics.get("per_peptide_auc_mean") is not None:
                print(f"    Per-peptide AUC: {metrics['per_peptide_auc_mean']:.4f}")

            # Save per-epitope breakdown CSV
            split_pred_dir = os.path.join(task_output_dir, "predictions")
            if per_epitope_all:
                epitope_df = pd.DataFrame(per_epitope_all)
                epitope_df.to_csv(
                    os.path.join(split_pred_dir, f"{split_name}_per_epitope_breakdown.csv"),
                    index=False,
                )

            # Compute per-sample ranks for predictions CSV
            ranks = np.zeros(len(y_true), dtype=int)
            for i in range(len(y_true)):
                sorted_indices = np.argsort(-y_prob_known[i])
                ranks[i] = int(np.where(sorted_indices == y_true[i])[0][0]) + 1

            # Save predictions
            pred_df = pd.DataFrame({
                "true_peptide": y_true_labels_known,
                "predicted_peptide": [
                    dtcr.lb.classes_[y_pred[i]] for i in range(len(y_pred))
                ],
                "true_peptide_rank": ranks,
                "true_peptide_score": y_prob_known[np.arange(len(y_true)), y_true],
                "predicted_prob": y_prob_known.max(axis=1),
            })
            pred_path = os.path.join(
                task_output_dir, "predictions", f"{split_name}_predictions.csv"
            )
            pred_df.to_csv(pred_path, index=False)

        except Exception as e:
            print(f"    Error during evaluation: {e}")
            results["test_results"][split_name] = {"error": str(e)}

    # Save results
    results_path = os.path.join(task_output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DeepTCR on TCR-peptide specificity prediction"
    )
    parser.add_argument(
        "--task",
        choices=list(TASK_CONFIGS.keys()) + ["all"],
        default="all",
        help="Task to run (default: all)",
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
        default="/home/sravisha/projects/quest/results/deeptcr_benchmark",
        help="Output directory for results",
    )
    parser.add_argument(
        "--skip_hla",
        action="store_true",
        help="Skip experiments with HLA (only run without HLA)",
    )
    parser.add_argument(
        "--only_hla",
        action="store_true",
        help="Only run experiments with HLA",
    )
    parser.add_argument(
        "--suppress_output",
        action="store_true",
        help="Suppress training output",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine tasks to run
    tasks = [args.task] if args.task != "all" else list(TASK_CONFIGS.keys())

    # Determine HLA settings
    hla_settings = []
    if not args.skip_hla:
        hla_settings.append(True)
    if not args.only_hla:
        hla_settings.append(False)

    # Track all results
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "tasks": {},
    }

    # Run benchmarks
    for task in tasks:
        all_results["tasks"][task] = {}
        for use_hla in hla_settings:
            try:
                results = train_and_evaluate(
                    task_name=task,
                    data_dir=args.data_dir,
                    output_dir=args.output_dir,
                    use_hla=use_hla,
                    suppress_output=args.suppress_output,
                )
                hla_key = "with_hla" if use_hla else "without_hla"
                all_results["tasks"][task][hla_key] = results
            except Exception as e:
                print(f"\nError running {task} (HLA={use_hla}): {e}")
                import traceback
                traceback.print_exc()
                hla_key = "with_hla" if use_hla else "without_hla"
                all_results["tasks"][task][hla_key] = {"error": str(e)}

    # Save summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Benchmark complete!")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
