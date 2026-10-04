"""Write per-task results JSON + per-(query, candidate) score sidecar.

User-specified schema:

    {
      "baseline": "lstm",
      "task": "as_trb_i",
      "conditions": {
        "test_iid":          {"recall_at_10": 0.71, "mrr": 0.42, "hit_at_1": 0.21},
        "test_novel_tcr":    {...},
        "test_novel_pep":    {...},
        "test_novel_allele": {...},
        "test_level4":       {...}
      },
      "wandb_run_id": "abc123",
      "training_data": "benchmark_v2/splits/as_trb_i_train.parquet",
      "completed_at":  "2026-05-04T03:24:00Z"
    }

For class II, conditions reduces to {"eval": {...}}.

The score sidecar is a parquet with columns
  (condition, query_id, candidate_peptide, score, true_peptide)
and lets us recompute Normalized Recall@K and any other downstream
metric post-hoc without re-running the model.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

RESULTS_ROOT = Path("/home/ubuntu/quest/results")


def write_results(
    baseline: str,
    task: str,
    conditions: dict[str, dict[str, float]],
    wandb_run_id: Optional[str],
    training_data: str,
    out_root: Path = RESULTS_ROOT,
) -> Path:
    """Write the per-task results JSON.

    Args:
        baseline: e.g. "lstm", "tulip-retrained", "esm2-8m-cpt".
        task: e.g. "as_trb_i".
        conditions: dict of {condition_label: {"recall_at_10": ..., "mrr": ..., "hit_at_1": ...}}
        wandb_run_id: WandB run id, or None if WandB disabled.
        training_data: training parquet path (relative or absolute).
        out_root: results root; default /home/ubuntu/quest/results.

    Returns:
        Path to the written {task}_summary.json.
    """
    out_dir = out_root / baseline
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline": baseline,
        "task": task,
        "conditions": conditions,
        "wandb_run_id": wandb_run_id,
        "training_data": training_data,
        "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out_path = out_dir / f"{task}_summary.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def write_scores_sidecar(
    baseline: str,
    task: str,
    condition: str,
    query_ids: np.ndarray,
    candidate_peptides: list[str],
    scores: np.ndarray,
    true_peptides: np.ndarray,
    out_root: Path = RESULTS_ROOT,
) -> Path:
    """Dump the (n_queries x n_candidates) score matrix as long-form parquet.

    One row per (query, candidate). Used to recompute Normalized Recall@K
    after the fact without re-running the model.

    Files land at results/{baseline}/scores/{task}_{condition}.parquet.
    """
    n_q, n_c = scores.shape
    if len(query_ids) != n_q:
        raise ValueError(f"query_ids length {len(query_ids)} != scores rows {n_q}")
    if len(candidate_peptides) != n_c:
        raise ValueError(f"candidate_peptides length {len(candidate_peptides)} != scores cols {n_c}")
    if len(true_peptides) != n_q:
        raise ValueError(f"true_peptides length {len(true_peptides)} != scores rows {n_q}")

    rows = {
        "query_id": np.repeat(query_ids, n_c),
        "candidate_peptide": np.tile(candidate_peptides, n_q),
        "score": scores.reshape(-1).astype(np.float32),
        "true_peptide": np.repeat(true_peptides, n_c),
    }
    df = pd.DataFrame(rows)

    out_dir = out_root / baseline / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task}_{condition}.parquet"
    df.to_parquet(out_path, index=False, compression="zstd")
    return out_path
