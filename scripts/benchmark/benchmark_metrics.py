"""
Unified Retrieval Metrics for TCR-Peptide Specificity Benchmarks

Shared functions for computing retrieval metrics, per-peptide AUC-ROC,
bootstrap confidence intervals, per-epitope breakdowns, and random baselines.

All benchmark scripts (SFT, DeepTCR, Baseline, TULIP, TCR-BERT) import
from this module to produce comparable metrics.
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_retrieval_metrics_from_scores(
    scores: np.ndarray,
    true_indices: np.ndarray,
) -> Dict:
    """
    Compute retrieval metrics from a score matrix.

    Args:
        scores: (n_samples, n_candidates) score for each candidate per sample.
                Higher score = more likely candidate.
        true_indices: (n_samples,) index of true peptide in candidate list.

    Returns:
        Dict with retrieval_hit_at_1, retrieval_hit_at_5, retrieval_recall_at_10,
        retrieval_mrr, retrieval_mean_rank, retrieval_median_rank, retrieval_num_candidates.
    """
    n_samples, n_candidates = scores.shape

    if n_samples == 0:
        return {
            "retrieval_hit_at_1": 0.0,
            "retrieval_hit_at_5": 0.0,
            "retrieval_recall_at_10": 0.0,
            "retrieval_mrr": 0.0,
            "retrieval_mean_rank": float("inf"),
            "retrieval_median_rank": float("inf"),
            "retrieval_num_candidates": int(n_candidates),
        }

    # Compute ranks: for each sample, rank candidates by descending score
    # rank of true peptide (1-indexed)
    ranks = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        sorted_indices = np.argsort(-scores[i])
        rank = int(np.where(sorted_indices == true_indices[i])[0][0]) + 1
        ranks[i] = rank

    return {
        "retrieval_hit_at_1": float(np.mean(ranks <= 1)),
        "retrieval_hit_at_5": float(np.mean(ranks <= 5)),
        "retrieval_recall_at_10": float(np.mean(ranks <= 10)),
        "retrieval_mrr": float(np.mean(1.0 / ranks)),
        "retrieval_mean_rank": float(np.mean(ranks)),
        "retrieval_median_rank": float(np.median(ranks)),
        "retrieval_num_candidates": int(n_candidates),
    }


def compute_per_peptide_auc(
    sample_peptides: np.ndarray,
    scores: np.ndarray,
    candidate_peptides: List[str],
    min_samples: int = 5,
) -> Dict:
    """
    Compute per-peptide AUC-ROC for binding discrimination.

    For each unique peptide with >= min_samples positive examples:
    - Positives: samples whose true peptide == this peptide
    - Negatives: samples whose true peptide != this peptide
    - Score: the model's score for this peptide (column in scores matrix)

    Args:
        sample_peptides: (n_samples,) true peptide string for each sample.
        scores: (n_samples, n_candidates) score matrix.
        candidate_peptides: ordered list of candidate peptide strings
                           (columns of scores matrix).
        min_samples: minimum positive samples to include a peptide.

    Returns:
        Dict with per_peptide_auc_mean, per_peptide_auc_std,
        per_peptide_auc_weighted, per_peptide_num_evaluated,
        per_peptide_num_excluded, per_peptide_num_total,
        per_peptide_min_samples_threshold.
    """
    candidate_to_idx = {pep: idx for idx, pep in enumerate(candidate_peptides)}
    unique_peptides = np.unique(sample_peptides)

    auc_values = []
    sample_counts = []
    num_excluded = 0

    for peptide in unique_peptides:
        if peptide not in candidate_to_idx:
            num_excluded += 1
            continue

        pep_col_idx = candidate_to_idx[peptide]
        pos_mask = sample_peptides == peptide
        n_pos = int(pos_mask.sum())

        if n_pos < min_samples:
            num_excluded += 1
            continue

        neg_mask = ~pos_mask
        n_neg = int(neg_mask.sum())
        if n_neg == 0:
            num_excluded += 1
            continue

        # Labels: 1 for positives, 0 for negatives
        labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
        # Scores: model's score for this peptide for each sample
        pep_scores = np.concatenate([
            scores[pos_mask, pep_col_idx],
            scores[neg_mask, pep_col_idx],
        ])

        try:
            auc = roc_auc_score(labels, pep_scores)
            auc_values.append(auc)
            sample_counts.append(n_pos)
        except ValueError:
            num_excluded += 1

    num_evaluated = len(auc_values)

    if num_evaluated > 0:
        mean_auc = float(np.mean(auc_values))
        std_auc = float(np.std(auc_values))
        weighted_auc = float(np.average(auc_values, weights=sample_counts))
    else:
        mean_auc = None
        std_auc = None
        weighted_auc = None

    return {
        "per_peptide_auc_mean": mean_auc,
        "per_peptide_auc_std": std_auc,
        "per_peptide_auc_weighted": weighted_auc,
        "per_peptide_num_evaluated": num_evaluated,
        "per_peptide_num_excluded": num_excluded,
        "per_peptide_num_total": len(unique_peptides),
        "per_peptide_min_samples_threshold": min_samples,
    }


def compute_bootstrap_ci(
    scores: np.ndarray,
    true_indices: np.ndarray,
    sample_peptides: np.ndarray,
    candidate_peptides: List[str],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict:
    """
    Bootstrap 95% CIs for retrieval and per-peptide AUC metrics.

    Resamples test samples with replacement, recomputes metrics per iteration,
    returns {metric}_ci_lower and {metric}_ci_upper.

    Args:
        scores: (n_samples, n_candidates) score matrix.
        true_indices: (n_samples,) index of true peptide.
        sample_peptides: (n_samples,) true peptide strings.
        candidate_peptides: ordered list of candidate peptide strings.
        n_bootstrap: number of bootstrap iterations.
        ci_level: confidence level (default 0.95).
        seed: random seed.

    Returns:
        Dict with {metric}_ci_lower and {metric}_ci_upper for key metrics.
    """
    n_samples = scores.shape[0]
    if n_samples == 0:
        return {}

    rng = np.random.default_rng(seed)
    alpha = (1 - ci_level) / 2

    # Metrics to bootstrap
    hit1_vals = []
    hit5_vals = []
    recall10_vals = []
    mrr_vals = []
    mean_rank_vals = []
    auc_mean_vals = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = rng.integers(0, n_samples, size=n_samples)
        boot_scores = scores[idx]
        boot_true_indices = true_indices[idx]
        boot_peptides = sample_peptides[idx]

        # Retrieval metrics
        ret = compute_retrieval_metrics_from_scores(boot_scores, boot_true_indices)
        hit1_vals.append(ret["retrieval_hit_at_1"])
        hit5_vals.append(ret["retrieval_hit_at_5"])
        recall10_vals.append(ret["retrieval_recall_at_10"])
        mrr_vals.append(ret["retrieval_mrr"])
        mean_rank_vals.append(ret["retrieval_mean_rank"])

        # Per-peptide AUC (use min_samples=2 for bootstrap to avoid too many exclusions)
        auc_res = compute_per_peptide_auc(
            boot_peptides, boot_scores, candidate_peptides, min_samples=2
        )
        if auc_res["per_peptide_auc_mean"] is not None:
            auc_mean_vals.append(auc_res["per_peptide_auc_mean"])

    result = {}

    def add_ci(name, values):
        if len(values) > 0:
            arr = np.array(values)
            result[f"{name}_ci_lower"] = float(np.percentile(arr, alpha * 100))
            result[f"{name}_ci_upper"] = float(np.percentile(arr, (1 - alpha) * 100))

    add_ci("retrieval_hit_at_1", hit1_vals)
    add_ci("retrieval_hit_at_5", hit5_vals)
    add_ci("retrieval_recall_at_10", recall10_vals)
    add_ci("retrieval_mrr", mrr_vals)
    add_ci("retrieval_mean_rank", mean_rank_vals)
    add_ci("per_peptide_auc_mean", auc_mean_vals)

    return result


def compute_per_epitope_breakdown(
    scores: np.ndarray,
    true_indices: np.ndarray,
    sample_peptides: np.ndarray,
    candidate_peptides: List[str],
    min_samples: int = 5,
) -> Dict:
    """
    Per-epitope Hit@1, MRR, and AUC breakdown.

    For each unique peptide with >= min_samples test samples, computes
    epitope-specific retrieval and discrimination metrics.

    Args:
        scores: (n_samples, n_candidates) score matrix.
        true_indices: (n_samples,) index of true peptide.
        sample_peptides: (n_samples,) true peptide strings.
        candidate_peptides: ordered list of candidate peptide strings.
        min_samples: minimum samples per peptide to include.

    Returns:
        Dict with per_epitope_top5, per_epitope_bottom5, per_epitope_all.
    """
    candidate_to_idx = {pep: idx for idx, pep in enumerate(candidate_peptides)}
    unique_peptides = np.unique(sample_peptides)
    n_candidates = scores.shape[1]

    epitope_results = []

    for peptide in unique_peptides:
        mask = sample_peptides == peptide
        n = int(mask.sum())
        if n < min_samples:
            continue

        # Compute per-sample ranks for this epitope's samples
        pep_scores = scores[mask]
        pep_true_indices = true_indices[mask]

        ranks = np.zeros(n, dtype=np.float64)
        for i in range(n):
            sorted_indices = np.argsort(-pep_scores[i])
            rank = int(np.where(sorted_indices == pep_true_indices[i])[0][0]) + 1
            ranks[i] = rank

        hit_at_1 = float(np.mean(ranks <= 1))
        mrr = float(np.mean(1.0 / ranks))

        # Per-peptide AUC for this epitope
        auc = None
        if peptide in candidate_to_idx:
            pep_col_idx = candidate_to_idx[peptide]
            pos_mask_global = sample_peptides == peptide
            neg_mask_global = ~pos_mask_global
            n_neg = int(neg_mask_global.sum())

            if n_neg > 0:
                labels = np.concatenate([
                    np.ones(int(pos_mask_global.sum())),
                    np.zeros(n_neg),
                ])
                pep_auc_scores = np.concatenate([
                    scores[pos_mask_global, pep_col_idx],
                    scores[neg_mask_global, pep_col_idx],
                ])
                try:
                    auc = float(roc_auc_score(labels, pep_auc_scores))
                except ValueError:
                    pass

        entry = {
            "peptide": str(peptide),
            "num_samples": n,
            "hit_at_1": hit_at_1,
            "mrr": mrr,
        }
        if auc is not None:
            entry["auc"] = auc
        epitope_results.append(entry)

    # Sort by hit_at_1 descending for top/bottom
    epitope_results.sort(key=lambda x: x["hit_at_1"], reverse=True)

    top5 = epitope_results[:5] if len(epitope_results) >= 5 else epitope_results
    bottom5 = epitope_results[-5:] if len(epitope_results) >= 5 else epitope_results

    return {
        "per_epitope_top5": top5,
        "per_epitope_bottom5": bottom5,
        "per_epitope_all": epitope_results,
    }


def compute_random_baselines(
    num_candidates: int,
) -> Dict:
    """
    Compute expected metric values for a random ranker.

    Args:
        num_candidates: number of candidate peptides.

    Returns:
        Dict with random_baseline_hit_at_1, _hit_at_5, _recall_at_10,
        _mrr, _mean_rank, _per_peptide_auc.
    """
    n = max(num_candidates, 1)

    # Harmonic number H(n) = sum(1/k for k=1..n)
    harmonic = sum(1.0 / k for k in range(1, n + 1))

    return {
        "random_baseline_hit_at_1": 1.0 / n,
        "random_baseline_hit_at_5": min(5, n) / n,
        "random_baseline_recall_at_10": min(10, n) / n,
        "random_baseline_mrr": harmonic / n,
        "random_baseline_mean_rank": (n + 1) / 2.0,
        "random_baseline_per_peptide_auc": 0.5,
    }


def compute_lift_metrics(
    model_metrics: Dict,
    random_baselines: Dict,
) -> Dict:
    """
    Compute lift = model_metric / random_baseline for retrieval metrics.

    Lift > 1 means above random; enables cross-split comparison
    despite varying candidate pool sizes.

    Args:
        model_metrics: dict with retrieval metric values.
        random_baselines: dict from compute_random_baselines().

    Returns:
        Dict with retrieval_hit_at_1_lift, _hit_at_5_lift, _mrr_lift.
    """
    result = {}

    pairs = [
        ("retrieval_hit_at_1", "random_baseline_hit_at_1", "retrieval_hit_at_1_lift"),
        ("retrieval_hit_at_5", "random_baseline_hit_at_5", "retrieval_hit_at_5_lift"),
        ("retrieval_recall_at_10", "random_baseline_recall_at_10", "retrieval_recall_at_10_lift"),
        ("retrieval_mrr", "random_baseline_mrr", "retrieval_mrr_lift"),
    ]

    for model_key, baseline_key, lift_key in pairs:
        model_val = model_metrics.get(model_key)
        baseline_val = random_baselines.get(baseline_key)
        if model_val is not None and baseline_val and baseline_val > 0:
            result[lift_key] = float(model_val / baseline_val)

    return result


def compute_all_unified_metrics(
    scores: np.ndarray,
    true_indices: np.ndarray,
    sample_peptides: np.ndarray,
    candidate_peptides: List[str],
    n_bootstrap: int = 1000,
    min_samples_per_peptide: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Convenience function: calls all metric functions and returns unified dict.

    Args:
        scores: (n_samples, n_candidates) score matrix.
        true_indices: (n_samples,) index of true peptide in candidate list.
        sample_peptides: (n_samples,) true peptide string for each sample.
        candidate_peptides: ordered list of candidate peptide strings.
        n_bootstrap: number of bootstrap iterations for CIs.
        min_samples_per_peptide: min positive samples for per-peptide AUC.
        seed: random seed for bootstrap.

    Returns:
        Merged dict with all unified metrics ready for metrics.json.
    """
    n_samples, n_candidates = scores.shape

    # 1. Retrieval metrics
    retrieval = compute_retrieval_metrics_from_scores(scores, true_indices)

    # 2. Per-peptide AUC
    per_peptide = compute_per_peptide_auc(
        sample_peptides, scores, candidate_peptides,
        min_samples=min_samples_per_peptide,
    )

    # 3. Bootstrap CIs
    bootstrap = compute_bootstrap_ci(
        scores, true_indices, sample_peptides, candidate_peptides,
        n_bootstrap=n_bootstrap, seed=seed,
    )

    # 4. Per-epitope breakdown
    epitope = compute_per_epitope_breakdown(
        scores, true_indices, sample_peptides, candidate_peptides,
        min_samples=min_samples_per_peptide,
    )

    # 5. Random baselines
    baselines = compute_random_baselines(n_candidates)

    # 6. Lift metrics
    lift = compute_lift_metrics(retrieval, baselines)

    # Merge all into a single dict
    result = {}
    result.update(retrieval)
    result.update(per_peptide)
    result.update(bootstrap)
    result.update(baselines)
    result.update(lift)
    result.update(epitope)

    # Meta
    result["num_samples"] = int(n_samples)
    result["num_unique_peptides"] = int(len(np.unique(sample_peptides)))

    return result
