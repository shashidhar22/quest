#!/usr/bin/env python
"""Walk results/{baseline}/{task}_summary.json and produce a coverage matrix.

Usage:
    python -m track_a.launch.aggregate_results [--md]
    python track_a/launch/aggregate_results.py --md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

RESULTS_ROOT = Path("/home/ubuntu/quest/results")

BASELINE_ORDER = [
    "lstm", "mlp", "tulip-original", "tulip-retrained", "deeptcr",
    "tcrbert",
    "esm2_8m-base", "esm2_8m-cpt",
    "esm2_35m-base", "esm2_35m-cpt",
    "esmc_300m-base", "esmc_300m-cpt",
]
TASK_ORDER = ["as_trb_i", "as_paired_i", "as_trb_ii", "as_paired_ii", "as_tra_i", "as_tra_ii"]
COND_ORDER_I = ["test_iid", "test_novel_tcr", "test_novel_pep", "test_novel_allele", "test_level4"]
COND_ORDER_II = ["eval"]


def collect() -> dict:
    out = defaultdict(dict)  # baseline → task → conditions dict
    if not RESULTS_ROOT.exists():
        return out
    for baseline_dir in sorted(RESULTS_ROOT.iterdir()):
        if not baseline_dir.is_dir():
            continue
        baseline = baseline_dir.name
        for f in sorted(baseline_dir.glob("*_summary.json")):
            try:
                payload = json.loads(f.read_text())
                task = payload.get("task", f.stem.replace("_summary", ""))
                out[baseline][task] = payload.get("conditions", {})
            except Exception as e:
                out[baseline][f.stem] = {"_error": str(e)}
    return out


def headline_table(data: dict, metric: str = "recall_at_10") -> list[str]:
    """Markdown table: rows=baseline, cols=task×condition, values=metric."""
    baselines = sorted(data.keys(), key=lambda b: BASELINE_ORDER.index(b) if b in BASELINE_ORDER else 99)
    rows = []
    headers = ["baseline"]
    for t in TASK_ORDER:
        conds = COND_ORDER_II if t.endswith("_ii") else COND_ORDER_I
        for c in conds:
            headers.append(f"{t}/{c}")
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for b in baselines:
        cells = [b]
        for t in TASK_ORDER:
            conds = COND_ORDER_II if t.endswith("_ii") else COND_ORDER_I
            for c in conds:
                v = data.get(b, {}).get(t, {}).get(c, {}).get(metric)
                if v is None:
                    cells.append("—")
                else:
                    cells.append(f"{v:.3f}")
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def coverage(data: dict) -> tuple[int, int]:
    """(filled cells, total cells) across all baselines × main 4 tasks × conditions."""
    filled = total = 0
    for b in data:
        for t in ["as_trb_i", "as_paired_i", "as_trb_ii", "as_paired_ii"]:
            conds = COND_ORDER_II if t.endswith("_ii") else COND_ORDER_I
            for c in conds:
                total += 1
                v = data.get(b, {}).get(t, {}).get(c, {}).get("recall_at_10")
                if v is not None:
                    filled += 1
    return filled, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="recall_at_10", choices=["recall_at_10", "mrr", "hit_at_1"])
    ap.add_argument("--md", action="store_true", help="Print markdown table")
    args = ap.parse_args()

    data = collect()
    filled, total = coverage(data)
    print(f"Coverage: {filled}/{total} cells filled ({100*filled/max(total,1):.1f}%)")
    print(f"Baselines with results: {len(data)} ({sorted(data.keys())})")
    if args.md:
        print()
        print("\n".join(headline_table(data, metric=args.metric)))


if __name__ == "__main__":
    main()
