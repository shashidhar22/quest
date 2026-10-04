"""Retrieval metrics for Track A — wraps quest.metrics.retrieval_metrics.

The user-required schema for results JSON is:
  {"recall_at_10": float, "mrr": float, "hit_at_1": float}

quest.metrics.retrieval_metrics returns a richer dict; this wrapper
projects to the user's keys (and stashes the full dict as a sidecar
for the paper's appendix tables).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from quest.metrics.retrieval_metrics import (
    compute_all_unified_metrics,
    compute_random_baselines,
    compute_retrieval_metrics_from_scores,
)


def headline_metrics(scores: np.ndarray, true_indices: np.ndarray) -> dict[str, float]:
    """Compute the user-spec headline metrics.

    Args:
        scores: (n_samples, n_candidates) score matrix; higher = more likely.
        true_indices: (n_samples,) index of the true peptide in candidate list.

    Returns:
        {"recall_at_10": ..., "mrr": ..., "hit_at_1": ...}
    """
    full = compute_retrieval_metrics_from_scores(scores, true_indices)
    return {
        "recall_at_10": full["retrieval_recall_at_10"],
        "mrr": full["retrieval_mrr"],
        "hit_at_1": full["retrieval_hit_at_1"],
    }


def full_metrics(
    scores: np.ndarray,
    true_indices: np.ndarray,
    sample_peptides: np.ndarray,
    candidate_peptides: list[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute the full unified metrics dict (for the appendix sidecar).

    Wraps quest.metrics.retrieval_metrics.compute_all_unified_metrics.
    Includes per-peptide AUC, bootstrap CIs, per-epitope breakdown,
    random baselines, and lift.
    """
    return compute_all_unified_metrics(
        scores=scores,
        true_indices=true_indices,
        sample_peptides=sample_peptides,
        candidate_peptides=candidate_peptides,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


def random_baseline(num_candidates: int) -> dict[str, float]:
    """Random-ranker expected metrics (for paper-table sanity)."""
    full = compute_random_baselines(num_candidates)
    return {
        "recall_at_10": full["random_baseline_recall_at_10"],
        "mrr": full["random_baseline_mrr"],
        "hit_at_1": full["random_baseline_hit_at_1"],
    }
