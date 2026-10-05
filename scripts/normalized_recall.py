#!/usr/bin/env python3
"""
Normalized Recall@K for TCRBench score parquets.

WHY THIS METRIC EXISTS
----------------------
Standard Recall@K depends on candidate-pool size: random baseline = K / N.
With N=3,549 (Class I) and K=10, random Recall@10 = 0.28%; with N=990
(Class II), random Recall@10 = 1.01%. Comparing across tasks therefore
mixes "model quality" with "denominator" effects.

Normalized Recall@K fixes the comparison pool to N=100 per query
(1 gold + 99 randomly sampled non-gold candidates from the same query).
The random baseline is then K/100 regardless of underlying pool size,
so a Recall@10 of 0.30 on Class I and 0.30 on Class II are directly
comparable.

INPUT SCHEMA
------------
Score parquets with columns:
    query_idx       int64    unique per query
    query_tcr_a     string   informational, not used by the metric
    query_tcr_b     string   informational, not used by the metric
    query_mhc       string   informational, not used by the metric
    candidate_pep   string   peptide string
    score           float    model's binding score (higher = better)
    is_gold         bool     True for the actual binding peptide(s)

OUTPUT
------
Per-(task, condition) row in a CSV table with: hit@1, recall@5, recall@10,
recall@20, mrr, plus per-metric standard deviation across seeds, plus
n_queries_kept, n_queries_dropped.

USAGE
-----
    # Run synthetic verification (random scores -> R@K ~= K/100):
    python normalized_recall.py --self-test

    # Score one parquet file:
    python normalized_recall.py --input as_paired_i_iid_scores.parquet

    # Score every *_scores.parquet in a directory tree:
    python normalized_recall.py \\
        --results-dir /home/ubuntu/quest/results/nettcr \\
        --output      /home/ubuntu/quest/results/nettcr/normalized_recall.csv

CONVENTIONS
-----------
Tie-breaking: pessimistic. If a non-gold candidate has score >= gold, it
counts as ranked above the gold. This is conservative for the model and
reproducible across seeds.

Multi-gold queries: if a query has multiple gold candidates (TCR cross-
reactivity), we use the highest-scored gold for that query. This is the
most charitable interpretation for the model; switch to per-gold averaging
if cross-reactive queries dominate.

Sampling: 10 seeds by default, sampled without replacement from the non-
gold pool. Reported metric is mean across seeds with std.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Core metric
# ----------------------------------------------------------------------

CONDITION_NAMES = ("iid", "novel_tcr", "novel_pep", "novel_allele", "level4")


def normalized_recall_k(
    df: pd.DataFrame,
    k_values: Iterable[int] = (1, 5, 10, 20),
    pool_size: int = 100,
    n_seeds: int = 10,
    base_seed: int = 42,
) -> dict:
    """Compute Normalized Recall@K for a score parquet.

    Parameters
    ----------
    df
        Long-format score table (one row per (query, candidate) pair) with
        columns query_idx, candidate_pep, score, is_gold.
    k_values
        K's to evaluate.
    pool_size
        Fixed pool size per query (default 100 = 1 gold + 99 sampled).
    n_seeds
        Number of independent samplings to average across.
    base_seed
        RNG seed; seed `i` uses base_seed + i.

    Returns
    -------
    dict with keys:
        hit@1, recall@5, recall@10, recall@20, mrr, *_std,
        n_queries_kept, n_queries_dropped, pool_size, n_seeds.
    """
    k_values = tuple(sorted(set(int(k) for k in k_values)))

    # Validate schema
    required = {"query_idx", "candidate_pep", "score", "is_gold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input missing required columns: {sorted(missing)}")

    # Group by query once; iterate over groups multiple times (one per seed).
    # We materialize per-query (gold_score, non_gold_scores) once to avoid
    # re-grouping on every seed.
    per_query = []
    n_queries_total = 0
    n_queries_dropped = 0
    for qid, group in df.groupby("query_idx", sort=False):
        n_queries_total += 1
        golds = group.loc[group["is_gold"], "score"].to_numpy()
        non_golds = group.loc[~group["is_gold"], "score"].to_numpy()
        if golds.size == 0:
            n_queries_dropped += 1
            continue
        if non_golds.size < pool_size - 1:
            # Not enough non-golds to build the requested 100-pool
            n_queries_dropped += 1
            continue
        # Take best-scored gold for this query (charitable to the model;
        # see module docstring on multi-gold handling).
        per_query.append((float(golds.max()), non_golds))

    n_kept = len(per_query)
    if n_kept == 0:
        return {
            "n_queries_kept": 0,
            "n_queries_dropped": n_queries_dropped,
            "pool_size": pool_size,
            "n_seeds": n_seeds,
        }

    per_seed = {f"recall@{k}": np.zeros(n_seeds) for k in k_values}
    per_seed["mrr"] = np.zeros(n_seeds)

    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        ranks = np.empty(n_kept, dtype=np.int64)
        for i, (gold_score, non_golds) in enumerate(per_query):
            sampled_idx = rng.choice(non_golds.size, size=pool_size - 1, replace=False)
            sampled = non_golds[sampled_idx]
            # Pessimistic tie-breaking: non-gold with score == gold is "above"
            n_above = int(np.sum(sampled >= gold_score))
            ranks[i] = n_above + 1  # 1-indexed rank of the gold

        for k in k_values:
            per_seed[f"recall@{k}"][s] = float(np.mean(ranks <= k))
        per_seed["mrr"][s] = float(np.mean(1.0 / ranks))

    out: dict = {}
    for metric, vals in per_seed.items():
        out[metric] = float(np.mean(vals))
        out[f"{metric}_std"] = float(np.std(vals, ddof=1)) if n_seeds > 1 else 0.0

    # Convenience alias: hit@1 is recall@1
    out["hit@1"] = out["recall@1"]
    out["hit@1_std"] = out["recall@1_std"]

    out["n_queries_kept"] = n_kept
    out["n_queries_dropped"] = n_queries_dropped
    out["pool_size"] = pool_size
    out["n_seeds"] = n_seeds
    return out


# ----------------------------------------------------------------------
# File-level processing
# ----------------------------------------------------------------------


def parse_task_and_condition(stem: str) -> tuple[str, str]:
    """Split filename like 'as_paired_i_novel_tcr' -> ('as_paired_i', 'novel_tcr').

    Conditions are matched as suffixes (case-sensitive, longest-match-first
    so that 'novel_tcr' beats 'tcr' if the latter were ever a condition).
    """
    for cond in sorted(CONDITION_NAMES, key=len, reverse=True):
        suffix = "_" + cond
        if stem.endswith(suffix):
            return stem[: -len(suffix)], cond
    raise ValueError(f"Could not parse condition from stem: {stem!r}")


def process_one_file(path: Path, **kwargs) -> dict:
    df = pd.read_parquet(path)
    metrics = normalized_recall_k(df, **kwargs)
    stem = path.stem
    if stem.endswith("_scores"):
        stem = stem[: -len("_scores")]
    try:
        task, condition = parse_task_and_condition(stem)
    except ValueError:
        task, condition = stem, "unknown"
    metrics["task"] = task
    metrics["condition"] = condition
    metrics["source_file"] = str(path)
    return metrics


def process_results_dir(results_dir: Path, **kwargs) -> pd.DataFrame:
    files = sorted(results_dir.rglob("*_scores.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No *_scores.parquet found under {results_dir}"
        )
    rows = []
    for f in files:
        print(f"  [{len(rows)+1}/{len(files)}] {f.name}", flush=True)
        rows.append(process_one_file(f, **kwargs))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Synthetic correctness test
# ----------------------------------------------------------------------


def self_test() -> None:
    """Random scores should give Recall@K ~= K/100."""
    print("Building synthetic random-score dataframe...", flush=True)
    rng = np.random.default_rng(0)
    n_queries = 5000
    n_candidates = 500  # bigger than pool=100 so sampling is non-degenerate

    rows = []
    for q in range(n_queries):
        gold = int(rng.integers(0, n_candidates))
        scores = rng.standard_normal(n_candidates)
        for c in range(n_candidates):
            rows.append(
                (q, f"pep_{c}", float(scores[c]), c == gold)
            )
    df = pd.DataFrame(rows, columns=["query_idx", "candidate_pep", "score", "is_gold"])

    print(f"Computing normalized recall over {n_queries} synthetic queries...", flush=True)
    m = normalized_recall_k(df, k_values=(1, 5, 10, 20), n_seeds=10, base_seed=42)

    # Expected MRR for random ranking with pool=100:
    #   E[1/rank] = (1/100) * sum_{i=1}^{100} 1/i  =  H_100 / 100  ~= 0.0519
    h100 = sum(1.0 / i for i in range(1, 101))
    expected_mrr = h100 / 100.0

    print()
    print("Synthetic random-score test (expected: recall@K ~= K/100):")
    print(f"  hit@1       = {m['hit@1']:.4f}  (expected ~0.0100)")
    print(f"  recall@5    = {m['recall@5']:.4f}  (expected ~0.0500)")
    print(f"  recall@10   = {m['recall@10']:.4f}  (expected ~0.1000)")
    print(f"  recall@20   = {m['recall@20']:.4f}  (expected ~0.2000)")
    print(f"  mrr         = {m['mrr']:.4f}  (expected ~{expected_mrr:.4f})")
    print(f"  n_queries   = {m['n_queries_kept']}, dropped = {m['n_queries_dropped']}")

    tolerances = {
        "hit@1": (0.01, 0.005),
        "recall@5": (0.05, 0.010),
        "recall@10": (0.10, 0.015),
        "recall@20": (0.20, 0.020),
        "mrr": (expected_mrr, 0.005),
    }
    failures = []
    for metric, (target, tol) in tolerances.items():
        if abs(m[metric] - target) > tol:
            failures.append(f"  X {metric} = {m[metric]:.4f}, expected ~{target:.4f} (tol +/-{tol})")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f)
        sys.exit(1)
    print("\nOK Synthetic test passed.")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument(
        "--self-test", action="store_true",
        help="Run the synthetic correctness check and exit.",
    )
    p.add_argument(
        "--input", type=Path, default=None,
        help="Single score parquet to process.",
    )
    p.add_argument(
        "--results-dir", type=Path, default=None,
        help="Directory to walk for *_scores.parquet files.",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Output CSV path. Default: stdout for --input, "
             "<results-dir>/normalized_recall.csv for --results-dir.",
    )
    p.add_argument("--pool-size", type=int, default=100)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument(
        "--k-values", type=int, nargs="+", default=[1, 5, 10, 20],
        help="K values for Recall@K (default: 1 5 10 20).",
    )
    args = p.parse_args()

    if args.self_test:
        self_test()
        return

    kwargs = dict(
        k_values=tuple(args.k_values),
        pool_size=args.pool_size,
        n_seeds=args.n_seeds,
        base_seed=args.base_seed,
    )

    if args.input is not None:
        metrics = process_one_file(args.input, **kwargs)
        df = pd.DataFrame([metrics])
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Wrote {args.output}")
        else:
            print(json.dumps(metrics, indent=2, default=float))
        return

    if args.results_dir is not None:
        df = process_results_dir(args.results_dir, **kwargs)
        out = args.output or (args.results_dir / "normalized_recall.csv")
        df.to_csv(out, index=False)
        print(f"\nWrote {len(df)} rows -> {out}")
        # Pretty-print the headline columns
        if not df.empty:
            cols = ["task", "condition", "n_queries_kept",
                    "hit@1", "recall@10", "mrr"]
            cols = [c for c in cols if c in df.columns]
            with pd.option_context("display.max_rows", None, "display.width", 140):
                print()
                print(df[cols].to_string(index=False, float_format="%.4f"))
        return

    p.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
