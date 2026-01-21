#!/usr/bin/env python3
"""
Evaluate peptide-MHC binding prediction tools.

Computes evaluation metrics on predictions from netMHCpan, netMHCIIpan, and MHCflurry.

Metrics computed:
- AUC-ROC: Area under ROC curve
- AUC-PR: Area under Precision-Recall curve (better for imbalanced data)
- Sensitivity at %Rank thresholds (0.5%, 2%, 5%)
- PPV at top-k predictions
- Per-allele breakdown

Handles soft negatives (MHC-shuffled) with three approaches:
1. Filtering: Remove soft negatives with strong predicted binding
2. Label smoothing: Assign soft negatives label=0.1 instead of 0.0
3. Sample weighting: Down-weight soft negatives in metric computation

Usage:
    # Evaluate predictions
    python scripts/evaluation/evaluate_pmhc_tools.py \\
        --predictions_dir results/pmhc_predictions \\
        --output_dir results/pmhc_evaluation \\
        --tools netmhcpan mhcflurry

    # Evaluate with specific soft negative handling
    python scripts/evaluation/evaluate_pmhc_tools.py \\
        --predictions_dir results/pmhc_predictions \\
        --output_dir results/pmhc_evaluation \\
        --soft_negative_handling filtering \\
        --filter_threshold 2.0

Author: Claude
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from utils.metrics import (
    compute_all_metrics,
    compute_aggregate_metrics,
    compute_metrics_per_allele,
    compute_metrics_with_filtering,
    compute_metrics_with_label_smoothing,
    compute_metrics_with_weighting,
    format_metrics_table,
)


# Recognized test split patterns (ordered by expected difficulty)
TEST_SPLIT_PATTERNS = [
    'test_seen_motif',                  # Easy: Same binding motifs as training
    'test_unseen_peptide_seen_motif',   # Medium: New peptides, motifs seen
    'test_unseen_motif',                # Hard: Binding motifs not in training
    'test_unseen_allele',               # Hardest: MHC alleles not in training
    'test',                              # Fallback for simple split
]

# Display names and expected difficulty for splits
SPLIT_METADATA = {
    'test_seen_motif': {'display': 'Seen Motif', 'difficulty': 1, 'description': 'Same binding motifs as training'},
    'test_unseen_peptide_seen_motif': {'display': 'Unseen Peptide', 'difficulty': 2, 'description': 'New peptides, motifs seen'},
    'test_unseen_motif': {'display': 'Unseen Motif', 'difficulty': 3, 'description': 'Binding motifs not in training'},
    'test_unseen_allele': {'display': 'Unseen Allele', 'difficulty': 4, 'description': 'MHC alleles not in training'},
    'test': {'display': 'Test', 'difficulty': 0, 'description': 'Standard test set'},
}


def discover_prediction_splits(predictions_dir: Path) -> List[str]:
    """
    Find subdirectories containing predictions.

    Looks for subdirectories that contain prediction parquet files.

    Args:
        predictions_dir: Directory to search for prediction splits

    Returns:
        List of split names (subdirectory names) sorted by difficulty
    """
    splits = []

    for subdir in predictions_dir.iterdir():
        if subdir.is_dir() and list(subdir.glob("predictions_*.parquet")):
            splits.append(subdir.name)

    # Sort by expected difficulty
    def get_difficulty(split_name):
        return SPLIT_METADATA.get(split_name, {}).get('difficulty', 99)

    return sorted(splits, key=get_difficulty)


def load_predictions(predictions_file: Path) -> pd.DataFrame:
    """Load predictions from parquet file."""
    return pd.read_parquet(predictions_file)


def determine_score_and_rank_columns(df: pd.DataFrame, tool: str) -> tuple:
    """
    Determine which columns to use for score and rank based on tool.

    Returns:
        Tuple of (score_col, rank_col)
    """
    if tool == 'netmhcpan':
        score_col = 'score_el' if 'score_el' in df.columns else 'score'
        rank_col = 'rank_el' if 'rank_el' in df.columns else 'rank'
    elif tool == 'netmhciipan':
        score_col = 'score_el' if 'score_el' in df.columns else 'score'
        rank_col = 'rank_el' if 'rank_el' in df.columns else 'rank'
    elif tool == 'mhcflurry':
        score_col = 'score_affinity' if 'score_affinity' in df.columns else 'score'
        rank_col = 'rank_affinity' if 'rank_affinity' in df.columns else 'rank'
    else:
        score_col = 'score'
        rank_col = 'rank'

    return score_col, rank_col


def evaluate_tool(
    df: pd.DataFrame,
    tool: str,
    soft_negative_handling: str = 'all',
    filter_threshold: float = 2.0,
    soft_label: float = 0.1,
    soft_weight: float = 0.5,
) -> Dict[str, Dict]:
    """
    Evaluate predictions for a single tool.

    Args:
        df: DataFrame with predictions and labels
        tool: Tool name
        soft_negative_handling: How to handle soft negatives
            - 'standard': Treat all negatives equally
            - 'filtering': Filter out soft negatives with strong binding
            - 'label_smoothing': Use soft labels for soft negatives
            - 'weighting': Down-weight soft negatives
            - 'all': Compute all approaches
        filter_threshold: %Rank threshold for filtering
        soft_label: Label value for soft negatives in label smoothing
        soft_weight: Weight for soft negatives in weighting

    Returns:
        Dictionary of evaluation results
    """
    results = {}

    # Filter to rows with valid labels
    df = df[df['label'] >= 0].copy()

    if len(df) == 0:
        print(f"    No valid predictions for {tool}")
        return results

    # Get score and rank columns
    score_col, rank_col = determine_score_and_rank_columns(df, tool)

    # Check for required columns
    if score_col not in df.columns or rank_col not in df.columns:
        print(f"    Missing score/rank columns for {tool}")
        return results

    # Get arrays
    y_true = df['label'].values
    y_score = df[score_col].values
    y_rank = df[rank_col].values

    # Identify soft negatives
    is_soft_negative = (df['negative_type'] == 'mhc_shuffle').values

    # Handle NaN values
    valid_mask = ~(np.isnan(y_score) | np.isnan(y_rank))
    y_true = y_true[valid_mask]
    y_score = y_score[valid_mask]
    y_rank = y_rank[valid_mask]
    is_soft_negative = is_soft_negative[valid_mask]

    if len(y_true) == 0:
        print(f"    No valid predictions after filtering NaNs for {tool}")
        return results

    print(f"    Evaluating {len(y_true):,} predictions ({(y_true == 1).sum():,} positives)")

    # Compute metrics based on handling strategy
    approaches = []
    if soft_negative_handling == 'all':
        approaches = ['standard', 'filtering', 'label_smoothing', 'weighting']
    else:
        approaches = [soft_negative_handling]

    for approach in approaches:
        if approach == 'standard':
            metrics = compute_all_metrics(y_true, y_score, y_rank)
            metrics['approach'] = 'standard'
        elif approach == 'filtering':
            metrics = compute_metrics_with_filtering(
                y_true, y_score, y_rank, is_soft_negative, filter_threshold
            )
            metrics['approach'] = f'filtering_{filter_threshold}pct'
        elif approach == 'label_smoothing':
            metrics = compute_metrics_with_label_smoothing(
                y_true, y_score, y_rank, is_soft_negative, soft_label
            )
            metrics['approach'] = f'label_smoothing_{soft_label}'
        elif approach == 'weighting':
            metrics = compute_metrics_with_weighting(
                y_true, y_score, y_rank, is_soft_negative, soft_weight
            )
            metrics['approach'] = f'weighting_{soft_weight}'

        results[approach] = metrics

    return results


def evaluate_per_allele(
    df: pd.DataFrame,
    tool: str,
    min_samples: int = 10,
) -> pd.DataFrame:
    """
    Evaluate predictions per MHC allele.

    Args:
        df: DataFrame with predictions and labels
        tool: Tool name
        min_samples: Minimum samples required per allele

    Returns:
        DataFrame with per-allele metrics
    """
    # Filter to valid labels
    df = df[df['label'] >= 0].copy()

    if len(df) == 0:
        return pd.DataFrame()

    # Get score and rank columns
    score_col, rank_col = determine_score_and_rank_columns(df, tool)

    if score_col not in df.columns or rank_col not in df.columns:
        return pd.DataFrame()

    # Compute per-allele metrics
    per_allele = compute_metrics_per_allele(
        df,
        y_true_col='label',
        y_score_col=score_col,
        y_rank_col=rank_col,
        allele_col='allele',
        min_samples=min_samples,
    )

    return per_allele


def stratify_by_negative_type(
    df: pd.DataFrame,
    tool: str,
) -> Dict[str, Dict]:
    """
    Evaluate predictions stratified by negative type.

    Args:
        df: DataFrame with predictions and labels
        tool: Tool name

    Returns:
        Dictionary of metrics per negative type
    """
    results = {}

    df = df[df['label'] >= 0].copy()

    if len(df) == 0:
        return results

    score_col, rank_col = determine_score_and_rank_columns(df, tool)

    # Get positive samples
    positives = df[df['label'] == 1]
    if len(positives) == 0:
        return results

    # Evaluate against each type of negative
    for neg_type in ['mhc_shuffle', 'peptide_shuffle', 'random']:
        negatives = df[(df['label'] == 0) & (df['negative_type'] == neg_type)]

        if len(negatives) == 0:
            continue

        # Combine positives and this negative type
        subset = pd.concat([positives, negatives], ignore_index=True)

        y_true = subset['label'].values
        y_score = subset[score_col].values
        y_rank = subset[rank_col].values

        valid_mask = ~(np.isnan(y_score) | np.isnan(y_rank))
        y_true = y_true[valid_mask]
        y_score = y_score[valid_mask]
        y_rank = y_rank[valid_mask]

        if len(y_true) > 0 and len(np.unique(y_true)) == 2:
            metrics = compute_all_metrics(y_true, y_score, y_rank)
            metrics['negative_type'] = neg_type
            results[neg_type] = metrics

    return results


def plot_roc_curves(
    predictions: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """
    Plot ROC curves for all tools.

    Args:
        predictions: Dictionary of tool -> predictions DataFrame
        output_dir: Output directory for plots
    """
    plt.figure(figsize=(10, 8))

    for tool, df in predictions.items():
        df = df[df['label'] >= 0]
        if len(df) == 0:
            continue

        score_col, _ = determine_score_and_rank_columns(df, tool)
        if score_col not in df.columns:
            continue

        y_true = df['label'].values
        y_score = df[score_col].values

        valid_mask = ~np.isnan(y_score)
        y_true = y_true[valid_mask]
        y_score = y_score[valid_mask]

        if len(np.unique(y_true)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = np.trapz(tpr, fpr)

        plt.plot(fpr, tpr, label=f'{tool} (AUC={auc:.3f})', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Peptide-MHC Binding Prediction', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = output_dir / 'roc_curves.png'
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved ROC curves to {output_file}")


def plot_pr_curves(
    predictions: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """
    Plot Precision-Recall curves for all tools.

    Args:
        predictions: Dictionary of tool -> predictions DataFrame
        output_dir: Output directory for plots
    """
    plt.figure(figsize=(10, 8))

    for tool, df in predictions.items():
        df = df[df['label'] >= 0]
        if len(df) == 0:
            continue

        score_col, _ = determine_score_and_rank_columns(df, tool)
        if score_col not in df.columns:
            continue

        y_true = df['label'].values
        y_score = df[score_col].values

        valid_mask = ~np.isnan(y_score)
        y_true = y_true[valid_mask]
        y_score = y_score[valid_mask]

        if len(np.unique(y_true)) < 2:
            continue

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        auc = np.trapz(precision, recall)

        plt.plot(recall, precision, label=f'{tool} (AUC={auc:.3f})', linewidth=2)
        baseline = (y_true == 1).mean()

    if 'baseline' not in locals():
        baseline = 0.5
    plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1, label='Baseline')

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - Peptide-MHC Binding Prediction', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = output_dir / 'pr_curves.png'
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved PR curves to {output_file}")


def plot_rank_distributions(
    predictions: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """
    Plot %Rank distributions for positives vs negatives.

    Args:
        predictions: Dictionary of tool -> predictions DataFrame
        output_dir: Output directory for plots
    """
    n_tools = len(predictions)
    if n_tools == 0:
        return

    fig, axes = plt.subplots(1, n_tools, figsize=(5 * n_tools, 5))
    if n_tools == 1:
        axes = [axes]

    for ax, (tool, df) in zip(axes, predictions.items()):
        df = df[df['label'] >= 0]
        if len(df) == 0:
            continue

        _, rank_col = determine_score_and_rank_columns(df, tool)
        if rank_col not in df.columns:
            continue

        positives = df[df['label'] == 1][rank_col].dropna()
        negatives = df[df['label'] == 0][rank_col].dropna()

        bins = np.linspace(0, 10, 50)

        ax.hist(positives, bins=bins, alpha=0.7, label='Positives', density=True)
        ax.hist(negatives, bins=bins, alpha=0.7, label='Negatives', density=True)

        ax.axvline(x=0.5, color='r', linestyle='--', linewidth=1, label='Strong (0.5%)')
        ax.axvline(x=2.0, color='orange', linestyle='--', linewidth=1, label='Weak (2%)')

        ax.set_xlabel('%Rank', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{tool}', fontsize=14)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 10)

    plt.suptitle('%Rank Distributions', fontsize=16)
    plt.tight_layout()

    output_file = output_dir / 'rank_distributions.png'
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved rank distributions to {output_file}")


def plot_metrics_by_split(
    all_results: Dict[str, Dict[str, Dict]],
    output_dir: Path,
) -> None:
    """
    Plot bar chart comparing metrics across test splits.

    Creates a grouped bar chart with splits on x-axis and metrics as grouped bars.

    Args:
        all_results: Dictionary of split_name -> tool -> evaluation results
        output_dir: Output directory for plots
    """
    if not all_results:
        return

    # Extract AUC-ROC values for each tool and split
    metrics_data = []

    for split_name, split_results in all_results.items():
        for tool, tool_results in split_results.items():
            aggregate = tool_results.get('aggregate', {})
            if 'standard' in aggregate:
                metrics = aggregate['standard']
                metrics_data.append({
                    'split': split_name,
                    'tool': tool,
                    'auc_roc': metrics.get('auc_roc', np.nan),
                    'auc_pr': metrics.get('auc_pr', np.nan),
                    'sensitivity_0.5pct': metrics.get('sensitivity_at_0.5pct', np.nan),
                    'sensitivity_2pct': metrics.get('sensitivity_at_2.0pct', np.nan),
                })

    if not metrics_data:
        return

    df = pd.DataFrame(metrics_data)
    tools = df['tool'].unique()
    splits = df['split'].unique()

    # Sort splits by difficulty
    def get_difficulty(split_name):
        return SPLIT_METADATA.get(split_name, {}).get('difficulty', 99)
    splits = sorted(splits, key=get_difficulty)

    # Get display names for splits
    split_labels = [SPLIT_METADATA.get(s, {}).get('display', s) for s in splits]

    # Create figure with subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics_to_plot = [
        ('auc_roc', 'AUC-ROC', axes[0, 0]),
        ('auc_pr', 'AUC-PR', axes[0, 1]),
        ('sensitivity_0.5pct', 'Sensitivity @ 0.5%', axes[1, 0]),
        ('sensitivity_2pct', 'Sensitivity @ 2%', axes[1, 1]),
    ]

    x = np.arange(len(splits))
    width = 0.8 / len(tools)
    colors = plt.cm.Set2(np.linspace(0, 1, len(tools)))

    for metric_col, metric_name, ax in metrics_to_plot:
        for i, tool in enumerate(tools):
            tool_data = df[df['tool'] == tool]
            values = []
            for split in splits:
                split_data = tool_data[tool_data['split'] == split]
                if len(split_data) > 0:
                    values.append(split_data[metric_col].values[0])
                else:
                    values.append(np.nan)

            offset = (i - len(tools) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=tool, color=colors[i])

        ax.set_xlabel('Test Split', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name} by Test Split', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(split_labels, rotation=15, ha='right')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        # Set y-axis to start at 0 for better comparison
        if 'auc' in metric_col.lower():
            ax.set_ylim(0, 1.05)
        elif 'sensitivity' in metric_col.lower():
            ax.set_ylim(0, 1.05)

    plt.suptitle('Performance Comparison Across Test Splits\n(Increasing Difficulty Left to Right)', fontsize=14)
    plt.tight_layout()

    output_file = output_dir / 'auc_comparison.png'
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved AUC comparison plot to {output_file}")


def plot_roc_curves_by_split(
    all_predictions: Dict[str, Dict[str, pd.DataFrame]],
    output_dir: Path,
) -> None:
    """
    Plot ROC curves comparing performance across splits for each tool.

    Args:
        all_predictions: Dictionary of split_name -> tool -> predictions DataFrame
        output_dir: Output directory for plots
    """
    if not all_predictions:
        return

    # Get all tools
    tools = set()
    for split_preds in all_predictions.values():
        tools.update(split_preds.keys())
    tools = sorted(tools)

    if not tools:
        return

    # Create one subplot per tool
    n_tools = len(tools)
    fig, axes = plt.subplots(1, n_tools, figsize=(6 * n_tools, 5))
    if n_tools == 1:
        axes = [axes]

    # Sort splits by difficulty
    splits = list(all_predictions.keys())
    def get_difficulty(split_name):
        return SPLIT_METADATA.get(split_name, {}).get('difficulty', 99)
    splits = sorted(splits, key=get_difficulty)

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(splits)))

    for ax, tool in zip(axes, tools):
        for i, split_name in enumerate(splits):
            if split_name not in all_predictions:
                continue
            if tool not in all_predictions[split_name]:
                continue

            df = all_predictions[split_name][tool]
            df = df[df['label'] >= 0]
            if len(df) == 0:
                continue

            score_col, _ = determine_score_and_rank_columns(df, tool)
            if score_col not in df.columns:
                continue

            y_true = df['label'].values
            y_score = df[score_col].values

            valid_mask = ~np.isnan(y_score)
            y_true = y_true[valid_mask]
            y_score = y_score[valid_mask]

            if len(np.unique(y_true)) < 2:
                continue

            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = np.trapz(tpr, fpr)

            display_name = SPLIT_METADATA.get(split_name, {}).get('display', split_name)
            ax.plot(fpr, tpr, label=f'{display_name} (AUC={auc:.3f})', linewidth=2, color=colors[i])

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'{tool}', fontsize=12)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('ROC Curves by Test Split (Ordered by Difficulty)', fontsize=14)
    plt.tight_layout()

    output_file = output_dir / 'roc_curves_by_split.png'
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved ROC curves by split to {output_file}")


def write_summary_report(
    all_results: Dict[str, Dict],
    output_dir: Path,
    multi_split: bool = False,
) -> None:
    """
    Write summary report in markdown format.

    Args:
        all_results: Dictionary of tool -> evaluation results (single split)
                     or split_name -> tool -> evaluation results (multi split)
        output_dir: Output directory
        multi_split: If True, all_results has per-split structure
    """
    lines = []
    lines.append("# Peptide-MHC Binding Prediction Evaluation Report\n")
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if multi_split:
        _write_multi_split_report(lines, all_results)
    else:
        _write_single_split_report(lines, all_results)

    # Write to file
    report_file = output_dir / "evaluation_report.md"
    with open(report_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  Saved evaluation report to {report_file}")


def _write_single_split_report(lines: List[str], all_results: Dict[str, Dict]) -> None:
    """Write report content for single-split evaluation."""
    # Summary table
    lines.append("## Summary\n")
    lines.append("| Tool | Approach | AUC-ROC | AUC-PR | Sens@0.5% | Sens@2% | Median Rank (Pos) |")
    lines.append("|------|----------|---------|--------|-----------|---------|------------------|")

    for tool, results in all_results.items():
        for approach, metrics in results.get('aggregate', {}).items():
            auc_roc = metrics.get('auc_roc', np.nan)
            auc_pr = metrics.get('auc_pr', np.nan)
            sens_05 = metrics.get('sensitivity_at_0.5pct', np.nan)
            sens_2 = metrics.get('sensitivity_at_2.0pct', np.nan)
            median_rank = metrics.get('median_rank_positives', np.nan)

            lines.append(
                f"| {tool} | {approach} | {auc_roc:.4f} | {auc_pr:.4f} | "
                f"{sens_05:.4f} | {sens_2:.4f} | {median_rank:.2f} |"
            )

    lines.append("\n")

    # Stratified by negative type
    lines.append("## Performance by Negative Type\n")
    lines.append("| Tool | Negative Type | AUC-ROC | AUC-PR | N Positives | N Negatives |")
    lines.append("|------|---------------|---------|--------|-------------|-------------|")

    for tool, results in all_results.items():
        for neg_type, metrics in results.get('stratified', {}).items():
            auc_roc = metrics.get('auc_roc', np.nan)
            auc_pr = metrics.get('auc_pr', np.nan)
            n_pos = metrics.get('n_positives', 0)
            n_neg = metrics.get('n_negatives', 0)

            lines.append(
                f"| {tool} | {neg_type} | {auc_roc:.4f} | {auc_pr:.4f} | "
                f"{n_pos:,} | {n_neg:,} |"
            )

    lines.append("\n")

    # Per-allele summary
    lines.append("## Per-Allele Performance Summary\n")

    for tool, results in all_results.items():
        per_allele = results.get('per_allele_summary', {})
        if not per_allele:
            continue

        lines.append(f"### {tool}\n")
        lines.append("| Metric | Mean | Std | Median | Min | Max | N Alleles |")
        lines.append("|--------|------|-----|--------|-----|-----|-----------|")

        for metric, stats in per_allele.items():
            if isinstance(stats, dict) and 'mean' in stats:
                lines.append(
                    f"| {metric} | {stats['mean']:.4f} | {stats['std']:.4f} | "
                    f"{stats['median']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | "
                    f"{stats['n_alleles']} |"
                )

        lines.append("\n")


def _write_multi_split_report(lines: List[str], all_results: Dict[str, Dict[str, Dict]]) -> None:
    """Write report content for multi-split evaluation."""

    # Sort splits by difficulty
    def get_difficulty(split_name):
        return SPLIT_METADATA.get(split_name, {}).get('difficulty', 99)
    splits = sorted(all_results.keys(), key=get_difficulty)

    # Collect all tools
    tools = set()
    for split_results in all_results.values():
        tools.update(split_results.keys())
    tools = sorted(tools)

    # Overview table - all splits and tools in one table
    lines.append("## Performance Overview Across Test Splits\n")
    lines.append("Test splits are ordered by expected difficulty (easiest to hardest).\n")

    lines.append("| Split | Description | Tool | AUC-ROC | AUC-PR | Sens@0.5% | Sens@2% |")
    lines.append("|-------|-------------|------|---------|--------|-----------|---------|")

    for split_name in splits:
        display_name = SPLIT_METADATA.get(split_name, {}).get('display', split_name)
        description = SPLIT_METADATA.get(split_name, {}).get('description', '')

        split_results = all_results.get(split_name, {})

        for i, tool in enumerate(tools):
            tool_results = split_results.get(tool, {})
            aggregate = tool_results.get('aggregate', {})

            if 'standard' in aggregate:
                metrics = aggregate['standard']
                auc_roc = metrics.get('auc_roc', np.nan)
                auc_pr = metrics.get('auc_pr', np.nan)
                sens_05 = metrics.get('sensitivity_at_0.5pct', np.nan)
                sens_2 = metrics.get('sensitivity_at_2.0pct', np.nan)

                # Only show split name and description on first tool row
                if i == 0:
                    lines.append(
                        f"| **{display_name}** | {description} | {tool} | {auc_roc:.4f} | "
                        f"{auc_pr:.4f} | {sens_05:.4f} | {sens_2:.4f} |"
                    )
                else:
                    lines.append(
                        f"| | | {tool} | {auc_roc:.4f} | "
                        f"{auc_pr:.4f} | {sens_05:.4f} | {sens_2:.4f} |"
                    )

    lines.append("\n")

    # Metrics summary CSV-style table
    lines.append("## Metrics Summary by Split\n")

    # Create a summary table for easy comparison
    metrics_data = []
    for split_name in splits:
        display_name = SPLIT_METADATA.get(split_name, {}).get('display', split_name)
        split_results = all_results.get(split_name, {})

        for tool in tools:
            tool_results = split_results.get(tool, {})
            aggregate = tool_results.get('aggregate', {})

            if 'standard' in aggregate:
                metrics = aggregate['standard']
                metrics_data.append({
                    'Split': display_name,
                    'Tool': tool,
                    'AUC-ROC': metrics.get('auc_roc', np.nan),
                    'AUC-PR': metrics.get('auc_pr', np.nan),
                    'Sens@0.5%': metrics.get('sensitivity_at_0.5pct', np.nan),
                    'Sens@2%': metrics.get('sensitivity_at_2.0pct', np.nan),
                    'Median Rank': metrics.get('median_rank_positives', np.nan),
                })

    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        lines.append("```")
        lines.append(metrics_df.to_string(index=False))
        lines.append("```\n")

    # Per-split detailed sections
    lines.append("## Detailed Results by Test Split\n")

    for split_name in splits:
        display_name = SPLIT_METADATA.get(split_name, {}).get('display', split_name)
        description = SPLIT_METADATA.get(split_name, {}).get('description', '')
        difficulty = SPLIT_METADATA.get(split_name, {}).get('difficulty', 0)

        lines.append(f"### {display_name}\n")
        lines.append(f"**Description:** {description}\n")
        lines.append(f"**Difficulty Level:** {difficulty}/4\n")

        split_results = all_results.get(split_name, {})

        if not split_results:
            lines.append("No results available for this split.\n")
            continue

        # Summary table for this split
        lines.append("| Tool | Approach | AUC-ROC | AUC-PR | Sens@0.5% | Sens@2% |")
        lines.append("|------|----------|---------|--------|-----------|---------|")

        for tool, results in split_results.items():
            for approach, metrics in results.get('aggregate', {}).items():
                auc_roc = metrics.get('auc_roc', np.nan)
                auc_pr = metrics.get('auc_pr', np.nan)
                sens_05 = metrics.get('sensitivity_at_0.5pct', np.nan)
                sens_2 = metrics.get('sensitivity_at_2.0pct', np.nan)

                lines.append(
                    f"| {tool} | {approach} | {auc_roc:.4f} | {auc_pr:.4f} | "
                    f"{sens_05:.4f} | {sens_2:.4f} |"
                )

        lines.append("\n")

        # Performance by negative type for this split
        has_stratified = any(r.get('stratified') for r in split_results.values())
        if has_stratified:
            lines.append("#### Performance by Negative Type\n")
            lines.append("| Tool | Negative Type | AUC-ROC | AUC-PR |")
            lines.append("|------|---------------|---------|--------|")

            for tool, results in split_results.items():
                for neg_type, metrics in results.get('stratified', {}).items():
                    auc_roc = metrics.get('auc_roc', np.nan)
                    auc_pr = metrics.get('auc_pr', np.nan)

                    lines.append(f"| {tool} | {neg_type} | {auc_roc:.4f} | {auc_pr:.4f} |")

            lines.append("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate peptide-MHC binding predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        required=True,
        help="Directory containing prediction parquet files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--tools",
        type=str,
        nargs='+',
        default=None,
        help="Tools to evaluate (default: auto-detect from files)",
    )
    parser.add_argument(
        "--soft_negative_handling",
        type=str,
        choices=['standard', 'filtering', 'label_smoothing', 'weighting', 'all'],
        default='all',
        help="How to handle soft negatives (default: all)",
    )
    parser.add_argument(
        "--filter_threshold",
        type=float,
        default=2.0,
        help="%%Rank threshold for filtering soft negatives (default: 2.0)",
    )
    parser.add_argument(
        "--soft_label",
        type=float,
        default=0.1,
        help="Label value for soft negatives in label smoothing (default: 0.1)",
    )
    parser.add_argument(
        "--soft_weight",
        type=float,
        default=0.5,
        help="Weight for soft negatives in weighting (default: 0.5)",
    )
    parser.add_argument(
        "--min_samples_per_allele",
        type=int,
        default=10,
        help="Minimum samples for per-allele evaluation (default: 10)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs='+',
        default=None,
        help="Test splits to evaluate (default: auto-detect from subdirectories)",
    )

    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Peptide-MHC Binding Prediction Evaluation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Predictions directory: {predictions_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Soft negative handling: {args.soft_negative_handling}")

    # Determine if this is single-split or multi-split mode
    # Check for prediction files directly in the directory
    direct_pred_files = list(predictions_dir.glob("predictions_*.parquet"))

    # Check for subdirectories with prediction files
    discovered_splits = discover_prediction_splits(predictions_dir)

    if direct_pred_files and not discovered_splits:
        # Single-split mode: predictions directly in predictions_dir
        multi_split_mode = False
        splits = [None]  # None indicates no split subdirectory
        print("  Mode: Single-split evaluation")
    elif discovered_splits:
        # Multi-split mode: predictions in subdirectories
        multi_split_mode = True
        if args.splits:
            splits = args.splits
        else:
            splits = discovered_splits
        print(f"  Mode: Multi-split evaluation")
        print(f"  Splits: {splits}")
    else:
        print(f"\nNo prediction files found in {predictions_dir}")
        print("Expected either:")
        print("  - predictions_*.parquet files directly in the directory, or")
        print("  - Subdirectories containing predictions_*.parquet files")
        return

    # Helper function to convert numpy types for JSON
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_types(x) for x in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj) if not np.isnan(obj) else None
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        else:
            return obj

    if multi_split_mode:
        # Multi-split evaluation
        all_split_results = {}  # split_name -> tool -> results
        all_split_predictions = {}  # split_name -> tool -> predictions

        for split_name in splits:
            print("\n" + "=" * 60)
            display_name = SPLIT_METADATA.get(split_name, {}).get('display', split_name)
            print(f"Evaluating split: {display_name} ({split_name})")
            print("=" * 60)

            split_pred_dir = predictions_dir / split_name
            split_output_dir = output_dir / split_name
            split_output_dir.mkdir(parents=True, exist_ok=True)

            # Find prediction files for this split
            pred_files = list(split_pred_dir.glob("predictions_*.parquet"))

            if not pred_files:
                print(f"  No prediction files found in {split_pred_dir}")
                continue

            # Determine tools to evaluate
            if args.tools:
                tools = args.tools
            else:
                tools = [f.stem.replace('predictions_', '') for f in pred_files]

            print(f"  Tools: {tools}")

            # Load predictions for this split
            predictions = {}
            for tool in tools:
                pred_file = split_pred_dir / f"predictions_{tool}.parquet"
                if pred_file.exists():
                    df = load_predictions(pred_file)
                    predictions[tool] = df
                    print(f"  {tool}: {len(df):,} predictions")
                else:
                    print(f"  {tool}: file not found")

            if not predictions:
                print(f"  No predictions loaded for {split_name}!")
                continue

            all_split_predictions[split_name] = predictions

            # Evaluate each tool for this split
            split_results = {}

            for tool, df in predictions.items():
                print(f"\n  Evaluating {tool}...")

                results = {
                    'aggregate': {},
                    'stratified': {},
                    'per_allele': None,
                    'per_allele_summary': {},
                }

                # Aggregate evaluation
                aggregate = evaluate_tool(
                    df, tool,
                    soft_negative_handling=args.soft_negative_handling,
                    filter_threshold=args.filter_threshold,
                    soft_label=args.soft_label,
                    soft_weight=args.soft_weight,
                )
                results['aggregate'] = aggregate

                # Stratified by negative type
                stratified = stratify_by_negative_type(df, tool)
                results['stratified'] = stratified

                # Per-allele evaluation
                per_allele = evaluate_per_allele(df, tool, args.min_samples_per_allele)
                if len(per_allele) > 0:
                    results['per_allele'] = per_allele
                    per_allele_summary = compute_aggregate_metrics(per_allele)
                    results['per_allele_summary'] = per_allele_summary
                    print(f"    Per-allele: {len(per_allele)} alleles evaluated")

                split_results[tool] = results

            all_split_results[split_name] = split_results

            # Generate per-split plots
            print(f"\n  Generating plots for {split_name}...")
            plot_roc_curves(predictions, split_output_dir)
            plot_pr_curves(predictions, split_output_dir)
            plot_rank_distributions(predictions, split_output_dir)

            # Save per-split detailed results
            results_file = split_output_dir / "evaluation_results.json"
            with open(results_file, 'w') as f:
                json.dump(convert_types(split_results), f, indent=2)

            # Save per-allele results as CSV
            for tool, results in split_results.items():
                if results.get('per_allele') is not None:
                    csv_file = split_output_dir / f"per_allele_metrics_{tool}.csv"
                    results['per_allele'].to_csv(csv_file, index=False)

        # Generate comparative plots across all splits
        print("\n" + "=" * 60)
        print("Generating comparative analysis across splits...")
        print("=" * 60)

        plot_metrics_by_split(all_split_results, output_dir)
        plot_roc_curves_by_split(all_split_predictions, output_dir)

        # Write combined summary report
        print("\nWriting combined report...")
        write_summary_report(all_split_results, output_dir, multi_split=True)

        # Save combined metrics summary as CSV
        metrics_data = []
        for split_name in splits:
            display_name = SPLIT_METADATA.get(split_name, {}).get('display', split_name)
            split_results = all_split_results.get(split_name, {})

            for tool, tool_results in split_results.items():
                aggregate = tool_results.get('aggregate', {})
                if 'standard' in aggregate:
                    metrics = aggregate['standard']
                    metrics_data.append({
                        'split': split_name,
                        'split_display': display_name,
                        'tool': tool,
                        'auc_roc': metrics.get('auc_roc', np.nan),
                        'auc_pr': metrics.get('auc_pr', np.nan),
                        'sensitivity_0.5pct': metrics.get('sensitivity_at_0.5pct', np.nan),
                        'sensitivity_2pct': metrics.get('sensitivity_at_2.0pct', np.nan),
                        'median_rank_positives': metrics.get('median_rank_positives', np.nan),
                    })

        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            csv_file = output_dir / "metrics_by_split.csv"
            metrics_df.to_csv(csv_file, index=False)
            print(f"  Saved metrics summary to {csv_file}")

        # Save combined results JSON
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(convert_types(all_split_results), f, indent=2)
        print(f"  Saved detailed results to {results_file}")

        # Print summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)

        for split_name in splits:
            display_name = SPLIT_METADATA.get(split_name, {}).get('display', split_name)
            print(f"\n{display_name} ({split_name}):")

            split_results = all_split_results.get(split_name, {})
            for tool, results in split_results.items():
                if 'standard' in results.get('aggregate', {}):
                    metrics = results['aggregate']['standard']
                    print(f"  {tool}: AUC-ROC={metrics.get('auc_roc', 0):.4f}, "
                          f"AUC-PR={metrics.get('auc_pr', 0):.4f}, "
                          f"Sens@2%={metrics.get('sensitivity_at_2.0pct', 0):.4f}")

    else:
        # Single-split evaluation (original behavior)

        # Determine tools to evaluate
        if args.tools:
            tools = args.tools
        else:
            tools = [f.stem.replace('predictions_', '') for f in direct_pred_files]

        print(f"  Tools: {tools}")

        # Load predictions
        print("\nLoading predictions...")
        predictions = {}

        for tool in tools:
            pred_file = predictions_dir / f"predictions_{tool}.parquet"
            if pred_file.exists():
                df = load_predictions(pred_file)
                predictions[tool] = df
                print(f"  {tool}: {len(df):,} predictions")
            else:
                print(f"  {tool}: file not found")

        if not predictions:
            print("\nNo predictions loaded!")
            return

        # Evaluate each tool
        print("\nEvaluating predictions...")
        all_results = {}

        for tool, df in predictions.items():
            print(f"\n  {tool}:")

            results = {
                'aggregate': {},
                'stratified': {},
                'per_allele': None,
                'per_allele_summary': {},
            }

            # Aggregate evaluation
            aggregate = evaluate_tool(
                df, tool,
                soft_negative_handling=args.soft_negative_handling,
                filter_threshold=args.filter_threshold,
                soft_label=args.soft_label,
                soft_weight=args.soft_weight,
            )
            results['aggregate'] = aggregate

            # Stratified by negative type
            stratified = stratify_by_negative_type(df, tool)
            results['stratified'] = stratified

            # Per-allele evaluation
            per_allele = evaluate_per_allele(df, tool, args.min_samples_per_allele)
            if len(per_allele) > 0:
                results['per_allele'] = per_allele

                # Compute aggregate statistics
                per_allele_summary = compute_aggregate_metrics(per_allele)
                results['per_allele_summary'] = per_allele_summary

                print(f"    Per-allele: {len(per_allele)} alleles evaluated")

            all_results[tool] = results

        # Generate plots
        print("\nGenerating plots...")
        plot_roc_curves(predictions, output_dir)
        plot_pr_curves(predictions, output_dir)
        plot_rank_distributions(predictions, output_dir)

        # Write summary report
        print("\nWriting reports...")
        write_summary_report(all_results, output_dir, multi_split=False)

        # Save detailed results
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(convert_types(all_results), f, indent=2)

        print(f"  Saved detailed results to {results_file}")

        # Save per-allele results as CSV
        for tool, results in all_results.items():
            if results.get('per_allele') is not None:
                csv_file = output_dir / f"per_allele_metrics_{tool}.csv"
                results['per_allele'].to_csv(csv_file, index=False)
                print(f"  Saved per-allele metrics to {csv_file}")

        # Print summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)

        for tool, results in all_results.items():
            print(f"\n{tool}:")
            if 'standard' in results.get('aggregate', {}):
                metrics = results['aggregate']['standard']
                print(f"  AUC-ROC: {metrics.get('auc_roc', 'N/A'):.4f}")
                print(f"  AUC-PR: {metrics.get('auc_pr', 'N/A'):.4f}")
                print(f"  Sensitivity@0.5%%: {metrics.get('sensitivity_at_0.5pct', 'N/A'):.4f}")
                print(f"  Sensitivity@2%%: {metrics.get('sensitivity_at_2.0pct', 'N/A'):.4f}")
                print(f"  Median Rank (positives): {metrics.get('median_rank_positives', 'N/A'):.2f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
