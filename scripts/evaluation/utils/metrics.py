"""
Metric computation utilities for peptide-MHC prediction evaluation.

Implements metrics for evaluating binding prediction tools:
- AUC-ROC: Area under ROC curve
- AUC-PR: Area under Precision-Recall curve
- Sensitivity at %Rank thresholds
- PPV at top-k predictions
- Spearman correlation (for continuous binding data)
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


def compute_auc_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Area Under the ROC Curve.

    Args:
        y_true: Binary ground truth labels (0/1)
        y_score: Prediction scores (higher = more likely positive)
        sample_weight: Optional sample weights

    Returns:
        AUC-ROC score (0.0 to 1.0)
    """
    from sklearn.metrics import roc_auc_score

    # Check for valid data
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan

    try:
        return roc_auc_score(y_true, y_score, sample_weight=sample_weight)
    except Exception:
        return np.nan


def compute_auc_pr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Area Under the Precision-Recall Curve.

    More appropriate than AUC-ROC for imbalanced datasets.

    Args:
        y_true: Binary ground truth labels (0/1)
        y_score: Prediction scores (higher = more likely positive)
        sample_weight: Optional sample weights

    Returns:
        AUC-PR score (0.0 to 1.0)
    """
    from sklearn.metrics import average_precision_score

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan

    try:
        return average_precision_score(y_true, y_score, sample_weight=sample_weight)
    except Exception:
        return np.nan


def compute_sensitivity_at_threshold(
    y_true: np.ndarray,
    y_rank: np.ndarray,
    threshold: float,
) -> float:
    """
    Compute sensitivity (true positive rate) at a given %Rank threshold.

    Used to evaluate what fraction of true positives fall within
    a given percentile rank cutoff.

    Args:
        y_true: Binary ground truth labels (0/1)
        y_rank: Percentile rank predictions (lower = better binding)
        threshold: %Rank threshold (e.g., 0.5 for strong binders, 2.0 for weak binders)

    Returns:
        Sensitivity at the given threshold
    """
    if len(y_true) == 0:
        return np.nan

    positives = y_true == 1
    n_positives = positives.sum()

    if n_positives == 0:
        return np.nan

    # Count positives below threshold
    true_positives_at_threshold = ((y_rank <= threshold) & positives).sum()

    return true_positives_at_threshold / n_positives


def compute_specificity_at_threshold(
    y_true: np.ndarray,
    y_rank: np.ndarray,
    threshold: float,
) -> float:
    """
    Compute specificity (true negative rate) at a given %Rank threshold.

    Args:
        y_true: Binary ground truth labels (0/1)
        y_rank: Percentile rank predictions (lower = better binding)
        threshold: %Rank threshold

    Returns:
        Specificity at the given threshold
    """
    if len(y_true) == 0:
        return np.nan

    negatives = y_true == 0
    n_negatives = negatives.sum()

    if n_negatives == 0:
        return np.nan

    # Count negatives above threshold (correctly classified)
    true_negatives_at_threshold = ((y_rank > threshold) & negatives).sum()

    return true_negatives_at_threshold / n_negatives


def compute_ppv_at_topk(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: Union[int, float],
) -> float:
    """
    Compute Positive Predictive Value (precision) at top-k predictions.

    Args:
        y_true: Binary ground truth labels (0/1)
        y_score: Prediction scores (higher = more likely positive)
        k: Number of top predictions to consider (int) or fraction (float)

    Returns:
        PPV at top-k
    """
    if len(y_true) == 0:
        return np.nan

    n = len(y_true)

    # Convert fraction to count
    if isinstance(k, float) and k <= 1.0:
        k = max(1, int(n * k))

    k = min(k, n)

    # Get indices of top-k predictions
    top_k_indices = np.argsort(y_score)[-k:]

    # Count true positives in top-k
    true_positives_in_topk = y_true[top_k_indices].sum()

    return true_positives_in_topk / k


def compute_spearman_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute Spearman rank correlation between predictions and ground truth.

    Useful when ground truth contains continuous binding affinity values.

    Args:
        y_true: Ground truth values (continuous)
        y_pred: Prediction values (continuous)

    Returns:
        Tuple of (correlation coefficient, p-value)
    """
    if len(y_true) < 3:
        return np.nan, np.nan

    try:
        corr, pval = stats.spearmanr(y_true, y_pred)
        return corr, pval
    except Exception:
        return np.nan, np.nan


def compute_accuracy_at_threshold(
    y_true: np.ndarray,
    y_rank: np.ndarray,
    threshold: float,
) -> float:
    """
    Compute classification accuracy at a %Rank threshold.

    Predictions with %Rank <= threshold are classified as positive.

    Args:
        y_true: Binary ground truth labels (0/1)
        y_rank: Percentile rank predictions
        threshold: Classification threshold

    Returns:
        Accuracy at the threshold
    """
    if len(y_true) == 0:
        return np.nan

    predicted_positive = y_rank <= threshold
    correct = (predicted_positive == y_true).sum()

    return correct / len(y_true)


def compute_median_rank(
    y_true: np.ndarray,
    y_rank: np.ndarray,
    label: int = 1,
) -> float:
    """
    Compute median %Rank for positive or negative samples.

    Args:
        y_true: Binary ground truth labels (0/1)
        y_rank: Percentile rank predictions
        label: Which class to compute median for (1=positives, 0=negatives)

    Returns:
        Median %Rank for the specified class
    """
    mask = y_true == label
    if mask.sum() == 0:
        return np.nan

    return np.median(y_rank[mask])


def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_rank: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    rank_thresholds: List[float] = None,
    topk_values: List[Union[int, float]] = None,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for a set of predictions.

    Args:
        y_true: Binary ground truth labels (0/1)
        y_score: Prediction scores (higher = more likely positive)
        y_rank: Percentile rank predictions (lower = better binding)
        sample_weight: Optional sample weights
        rank_thresholds: %Rank thresholds for sensitivity/specificity
        topk_values: Values of k for PPV@k

    Returns:
        Dictionary of metric names to values
    """
    if rank_thresholds is None:
        rank_thresholds = [0.5, 1.0, 2.0, 5.0]

    if topk_values is None:
        topk_values = [0.01, 0.05, 0.1, 100, 500]

    metrics = {}

    # AUC metrics
    metrics['auc_roc'] = compute_auc_roc(y_true, y_score, sample_weight)
    metrics['auc_pr'] = compute_auc_pr(y_true, y_score, sample_weight)

    # Sensitivity at thresholds
    for threshold in rank_thresholds:
        metrics[f'sensitivity_at_{threshold}pct'] = compute_sensitivity_at_threshold(
            y_true, y_rank, threshold
        )

    # Specificity at thresholds
    for threshold in rank_thresholds:
        metrics[f'specificity_at_{threshold}pct'] = compute_specificity_at_threshold(
            y_true, y_rank, threshold
        )

    # Accuracy at thresholds
    for threshold in rank_thresholds:
        metrics[f'accuracy_at_{threshold}pct'] = compute_accuracy_at_threshold(
            y_true, y_rank, threshold
        )

    # PPV at top-k
    for k in topk_values:
        if isinstance(k, float):
            metrics[f'ppv_at_top_{int(k*100)}pct'] = compute_ppv_at_topk(y_true, y_score, k)
        else:
            metrics[f'ppv_at_top_{k}'] = compute_ppv_at_topk(y_true, y_score, k)

    # Median ranks
    metrics['median_rank_positives'] = compute_median_rank(y_true, y_rank, label=1)
    metrics['median_rank_negatives'] = compute_median_rank(y_true, y_rank, label=0)

    # Sample counts
    metrics['n_positives'] = int((y_true == 1).sum())
    metrics['n_negatives'] = int((y_true == 0).sum())
    metrics['n_total'] = len(y_true)

    return metrics


def compute_metrics_per_allele(
    df: pd.DataFrame,
    y_true_col: str = 'label',
    y_score_col: str = 'score',
    y_rank_col: str = 'rank',
    allele_col: str = 'allele',
    sample_weight_col: Optional[str] = None,
    min_samples: int = 10,
) -> pd.DataFrame:
    """
    Compute metrics for each MHC allele separately.

    Args:
        df: DataFrame with predictions and labels
        y_true_col: Column name for ground truth labels
        y_score_col: Column name for prediction scores
        y_rank_col: Column name for percentile ranks
        allele_col: Column name for MHC alleles
        sample_weight_col: Column name for sample weights (optional)
        min_samples: Minimum samples required per allele

    Returns:
        DataFrame with metrics per allele
    """
    results = []

    for allele, group in df.groupby(allele_col):
        if len(group) < min_samples:
            continue

        y_true = group[y_true_col].values
        y_score = group[y_score_col].values
        y_rank = group[y_rank_col].values

        sample_weight = None
        if sample_weight_col and sample_weight_col in group.columns:
            sample_weight = group[sample_weight_col].values

        metrics = compute_all_metrics(
            y_true, y_score, y_rank, sample_weight
        )
        metrics['allele'] = allele

        results.append(metrics)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


def compute_aggregate_metrics(
    per_allele_metrics: pd.DataFrame,
    metric_cols: List[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute aggregate statistics over per-allele metrics.

    Args:
        per_allele_metrics: DataFrame with per-allele metrics
        metric_cols: List of metric columns to aggregate

    Returns:
        Dictionary of metric -> {mean, std, median, min, max}
    """
    if metric_cols is None:
        metric_cols = [c for c in per_allele_metrics.columns
                       if c not in ['allele', 'n_positives', 'n_negatives', 'n_total']]

    aggregates = {}

    for col in metric_cols:
        if col not in per_allele_metrics.columns:
            continue

        values = per_allele_metrics[col].dropna()
        if len(values) == 0:
            continue

        aggregates[col] = {
            'mean': float(values.mean()),
            'std': float(values.std()),
            'median': float(values.median()),
            'min': float(values.min()),
            'max': float(values.max()),
            'n_alleles': len(values),
        }

    return aggregates


def compute_metrics_with_label_smoothing(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_rank: np.ndarray,
    is_soft_negative: np.ndarray,
    soft_label: float = 0.1,
) -> Dict[str, float]:
    """
    Compute metrics with label smoothing for soft negatives.

    Soft negatives (MHC-shuffled pairs) are assigned a small positive label
    instead of 0 to account for possible true binding.

    Args:
        y_true: Binary ground truth labels (0/1)
        y_score: Prediction scores
        y_rank: Percentile ranks
        is_soft_negative: Boolean array indicating soft negatives
        soft_label: Label value for soft negatives (default: 0.1)

    Returns:
        Dictionary of metrics
    """
    # Apply label smoothing
    y_true_smoothed = y_true.astype(float).copy()
    y_true_smoothed[is_soft_negative & (y_true == 0)] = soft_label

    # For AUC, we need to binarize at a threshold
    # Use 0.5 as the threshold between positive and negative
    y_true_binary = (y_true_smoothed > 0.5).astype(int)

    metrics = compute_all_metrics(y_true_binary, y_score, y_rank)
    metrics['label_smoothing_applied'] = True
    metrics['soft_label'] = soft_label

    return metrics


def compute_metrics_with_filtering(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_rank: np.ndarray,
    is_soft_negative: np.ndarray,
    filter_threshold: float = 2.0,
) -> Dict[str, float]:
    """
    Compute metrics after filtering out soft negatives that may be true positives.

    Removes soft negative samples with %Rank below threshold before computing metrics.

    Args:
        y_true: Binary ground truth labels (0/1)
        y_score: Prediction scores
        y_rank: Percentile ranks
        is_soft_negative: Boolean array indicating soft negatives
        filter_threshold: %Rank threshold below which soft negatives are filtered

    Returns:
        Dictionary of metrics
    """
    # Filter out soft negatives with strong predicted binding
    keep_mask = ~(is_soft_negative & (y_true == 0) & (y_rank < filter_threshold))

    y_true_filtered = y_true[keep_mask]
    y_score_filtered = y_score[keep_mask]
    y_rank_filtered = y_rank[keep_mask]

    metrics = compute_all_metrics(y_true_filtered, y_score_filtered, y_rank_filtered)
    metrics['filtering_applied'] = True
    metrics['filter_threshold'] = filter_threshold
    metrics['n_filtered'] = int((~keep_mask).sum())

    return metrics


def compute_metrics_with_weighting(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_rank: np.ndarray,
    is_soft_negative: np.ndarray,
    soft_weight: float = 0.5,
) -> Dict[str, float]:
    """
    Compute metrics with sample weighting for soft negatives.

    Soft negatives are assigned lower weights in metric computation.

    Args:
        y_true: Binary ground truth labels (0/1)
        y_score: Prediction scores
        y_rank: Percentile ranks
        is_soft_negative: Boolean array indicating soft negatives
        soft_weight: Weight for soft negatives (default: 0.5)

    Returns:
        Dictionary of metrics
    """
    # Create sample weights
    weights = np.ones(len(y_true))
    weights[is_soft_negative & (y_true == 0)] = soft_weight

    metrics = compute_all_metrics(y_true, y_score, y_rank, sample_weight=weights)
    metrics['weighting_applied'] = True
    metrics['soft_weight'] = soft_weight

    return metrics


def format_metrics_table(
    metrics_dict: Dict[str, float],
    decimal_places: int = 4,
) -> str:
    """
    Format metrics dictionary as a markdown table.

    Args:
        metrics_dict: Dictionary of metric names to values
        decimal_places: Number of decimal places for rounding

    Returns:
        Markdown table string
    """
    lines = ["| Metric | Value |", "|--------|-------|"]

    for name, value in sorted(metrics_dict.items()):
        if isinstance(value, float):
            if np.isnan(value):
                value_str = "N/A"
            else:
                value_str = f"{value:.{decimal_places}f}"
        else:
            value_str = str(value)

        lines.append(f"| {name} | {value_str} |")

    return "\n".join(lines)
