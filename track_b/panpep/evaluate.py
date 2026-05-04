"""Evaluate PanPep on a TCRBench test split.

PanPep is run via subprocess (its CLI has all options we need). After it produces
the output CSV, this script joins predictions with ground-truth labels and
computes AUC.

For the AS task (retrieval-style), the "true label" is binary: 1 if the (TCR,
peptide) pair is a known binder in our test set, 0 otherwise. We construct
negatives the same way as during training (peptide-only swap) and combine with
positives to compute a binary AUC.

CLI:
    python evaluate.py --mode zero-shot --input smoke_zero.csv \
        --truth smoke_zero_truth.csv --negatives smoke_zero_negs.csv \
        --out smoke_zero_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

REPO_DIR = Path("/home/ubuntu/quest/track_b/panpep/repo")
RESULTS_DIR = Path("/home/ubuntu/quest/track_b/panpep/results")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["zero-shot", "few-shot", "majority"], required=True)
    p.add_argument("--input", required=True, help="PanPep input CSV (under results/).")
    p.add_argument("--truth", required=True, help="Ground-truth labels CSV (under results/).")
    p.add_argument("--negatives", default=None,
                   help="Optional: pre-computed negatives CSV with same Peptide/CDR3 cols.")
    p.add_argument("--out", required=True, help="Output JSON path (under results/).")
    p.add_argument("--update-step-test", type=int, default=10,
                   help="PanPep fine-tune steps (smoke test 10; paper default 1000 for majority).")
    p.add_argument("--predictions", default=None,
                   help="Path for PanPep's prediction output. Default: auto-named alongside input.")
    args = p.parse_args()

    in_path = (RESULTS_DIR / args.input).resolve()
    truth_path = (RESULTS_DIR / args.truth).resolve()
    out_path = (RESULTS_DIR / args.out).resolve()
    pred_name = args.predictions or args.input.replace(".csv", "_predictions.csv")
    pred_path = (RESULTS_DIR / pred_name).resolve()

    if not in_path.exists():
        raise FileNotFoundError(in_path)
    if not truth_path.exists():
        raise FileNotFoundError(truth_path)

    # Run PanPep.
    cmd = [
        sys.executable, "PanPep.py",
        "--learning_setting", args.mode,
        "--input", str(in_path),
        "--output", str(pred_path),
    ]
    if args.mode == "majority":
        cmd += ["--update_step_test", str(args.update_step_test)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""  # force CPU
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_DIR), env=env)

    pred_df = pd.read_csv(pred_path)
    truth_df = pd.read_csv(truth_path)
    truth_df["Peptide"] = truth_df["Peptide"].astype(str)
    truth_df["CDR3"] = truth_df["CDR3"].astype(str)
    pred_df["Peptide"] = pred_df["Peptide"].astype(str)
    pred_df["CDR3"] = pred_df["CDR3"].astype(str)

    # Merge: predictions with ground-truth (positives only).
    pos_df = pred_df.merge(truth_df[["Peptide", "CDR3", "true_label"]],
                           on=["Peptide", "CDR3"], how="left")
    # Negatives: rows in pred_df not in truth_df get true_label=0.
    pos_df["true_label"] = pos_df["true_label"].fillna(0).astype(int)

    y_true = pos_df["true_label"].values
    y_score = pos_df["Score"].values
    y_pred = (y_score > 0.5).astype(int)

    metrics = {
        "mode": args.mode,
        "input": args.input,
        "n_rows": int(len(y_true)),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else None,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mean_score_pos": float(y_score[y_true == 1].mean()) if (y_true == 1).any() else None,
        "mean_score_neg": float(y_score[y_true == 0].mean()) if (y_true == 0).any() else None,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
