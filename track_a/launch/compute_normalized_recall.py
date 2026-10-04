#!/usr/bin/env python3
"""Wrap scripts/normalized_recall.py to consume Track A's sidecar schema.

Track A score sidecars at /home/ubuntu/quest/results/<baseline>/scores/*.parquet
have columns:
    query_id, candidate_peptide, score, true_peptide

The shared metric function expects:
    query_idx, candidate_pep, score, is_gold (=candidate_peptide == true_peptide)

Filenames are also slightly different — Track A uses `<task>_test_<cond>.parquet`
(class I) or `<task>_eval.parquet` (class II eval-only), whereas the upstream
script expects `<task>_<cond>.parquet`. We parse our own convention.

Output: a CSV with one row per (baseline, task, condition).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Make the metric function importable
sys.path.insert(0, "/home/ubuntu/quest/scripts")
from normalized_recall import normalized_recall_k  # noqa: E402

RESULTS = Path("/home/ubuntu/quest/results")
OUT = Path("/home/ubuntu/quest/track_a/normalized_recall.csv")

# Conditions we report on (mirrors CLASS_I_REPORT_CONDITIONS + class-II eval)
CLASS_I_CONDS = ("test_iid", "test_novel_tcr", "test_novel_pep",
                  "test_novel_allele", "test_level4")
CLASS_II_EVAL = "eval"


def parse_track_a_stem(stem: str) -> tuple[str, str] | None:
    """`as_trb_i_test_iid` -> (`as_trb_i`, `test_iid`); `as_trb_ii_eval` -> (`as_trb_ii`, `eval`)."""
    # Try longest class-I conditions first
    for cond in sorted(CLASS_I_CONDS, key=len, reverse=True):
        suffix = "_" + cond
        if stem.endswith(suffix):
            return stem[: -len(suffix)], cond
    if stem.endswith("_" + CLASS_II_EVAL):
        return stem[: -len("_" + CLASS_II_EVAL)], CLASS_II_EVAL
    return None


def main() -> None:
    sidecars = sorted(RESULTS.rglob("scores/*.parquet"))
    if not sidecars:
        print(f"No score sidecars found under {RESULTS}/<baseline>/scores/")
        return

    print(f"Found {len(sidecars)} sidecar files. Processing...", flush=True)

    rows = []
    for i, p in enumerate(sidecars, 1):
        baseline = p.parents[1].name  # results/<baseline>/scores/<file>
        parsed = parse_track_a_stem(p.stem)
        if parsed is None:
            print(f"  [{i}/{len(sidecars)}] SKIP (unparsable stem): {p}", flush=True)
            continue
        task, cond = parsed

        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print(f"  [{i}/{len(sidecars)}] FAIL read {p.name}: {e}", flush=True)
            continue

        # Translate to expected schema
        df = df.rename(columns={
            "query_id": "query_idx",
            "candidate_peptide": "candidate_pep",
        })
        df["is_gold"] = df["candidate_pep"] == df["true_peptide"]

        try:
            metrics = normalized_recall_k(
                df,
                k_values=(1, 5, 10, 20),
                pool_size=100,
                n_seeds=10,
                base_seed=42,
            )
        except Exception as e:
            print(f"  [{i}/{len(sidecars)}] FAIL metric on {p.name}: {e}", flush=True)
            continue

        metrics["baseline"] = baseline
        metrics["task"] = task
        metrics["condition"] = cond
        metrics["source_file"] = str(p)
        rows.append(metrics)
        print(f"  [{i}/{len(sidecars)}] {baseline:18s}  {task:14s}  {cond:18s}  "
              f"R@10={metrics.get('recall@10', 0):.3f}  "
              f"hit@1={metrics.get('hit@1', 0):.3f}  "
              f"n_q={metrics.get('n_queries_kept', 0)}  "
              f"dropped={metrics.get('n_queries_dropped', 0)}", flush=True)

    out_df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT, index=False)
    print(f"\nWrote {len(out_df)} rows -> {OUT}")

    if not out_df.empty:
        cols = ["baseline", "task", "condition", "n_queries_kept",
                "hit@1", "recall@10", "mrr"]
        cols = [c for c in cols if c in out_df.columns]
        print()
        print("=== headline (recall@10) ===")
        with pd.option_context("display.max_rows", None, "display.width", 160):
            print(out_df[cols].to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
