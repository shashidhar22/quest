"""Convert TCRBench AS Class I parquet to PanPep input CSV.

PanPep input format depends on mode:
- zero-shot: Peptide, CDR3 (no labels — all rows are queries)
- few-shot / majority: Peptide, CDR3, Label (1=pos support, 0=neg support, "Unknown"=query)

For our integration:
- We provide TRAIN positives + their negatives (from shared/negative_sampling) as
  support rows (Label=1, Label=0).
- We provide TEST rows as queries (Label="Unknown") for majority/few-shot modes,
  or rows-only for zero-shot mode.

PanPep also requires that rows be SORTED by peptide (per the README) — we
ensure this in the output.

CLI:
    # Zero-shot test:
    python convert_data.py --mode zero-shot --task as_trb_i --test-split test_iid \
        --n-test 50 --out smoke_zero.csv
    # Majority test (with train support):
    python convert_data.py --mode majority --task as_trb_i \
        --train-split train --n-train 100 \
        --test-split test_iid --n-test 50 \
        --out smoke_majority.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/home/ubuntu/quest")

from track_b.shared.load_splits import load_as_split
from track_b.shared.negative_sampling import sample_negatives

RESULTS_DIR = Path("/home/ubuntu/quest/track_b/panpep/results")


def _filter_to_test_peptides(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only train rows whose peptide also appears in the test set.

    PanPep majority mode fine-tunes per peptide — train rows for peptides that
    don't appear in the test set provide no value (they aren't queried).
    """
    test_peps = set(test_df["peptide"].dropna().unique())
    return train_df[train_df["peptide"].isin(test_peps)].reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["zero-shot", "few-shot", "majority"], required=True)
    p.add_argument("--task", default="as_trb_i", choices=["as_trb_i", "as_tra_i", "as_paired_i"])
    p.add_argument("--train-split", default="train")
    p.add_argument("--test-split", required=True)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-test", type=int, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--negatives-per-positive", type=int, default=5)
    args = p.parse_args()

    test_df = load_as_split(args.task, args.test_split)
    if args.n_test is not None:
        test_df = test_df.sample(n=min(args.n_test, len(test_df)), random_state=42).reset_index(drop=True)

    # Test positives go into the query set with Label=Unknown.
    test_pos_df = test_df[["peptide", "trb_cdr3", "mhc_one_allele"]].dropna().reset_index(drop=True)
    # Construct test negatives: peptide-swap among test_pos_df rows.
    test_negs_df = sample_negatives(
        test_pos_df, tcr_col="trb_cdr3", pep_col="peptide", mhc_col="mhc_one_allele",
        n_per_pos=args.negatives_per_positive, seed=43,
    )

    test_pos_q = test_pos_df[["peptide", "trb_cdr3"]].rename(columns={"peptide": "Peptide", "trb_cdr3": "CDR3"})
    test_pos_q["Label"] = "Unknown"
    test_neg_q = test_negs_df[["peptide", "trb_cdr3"]].rename(columns={"peptide": "Peptide", "trb_cdr3": "CDR3"})
    test_neg_q["Label"] = "Unknown"
    queries_df = pd.concat([test_pos_q, test_neg_q], ignore_index=True).drop_duplicates(["Peptide", "CDR3"])

    if args.mode == "zero-shot":
        out_df = queries_df[["Peptide", "CDR3"]].copy()
    else:
        # Load full train, filter to test peptides FIRST, then optionally cap.
        train_df = load_as_split(args.task, args.train_split)
        train_df = _filter_to_test_peptides(train_df, test_df)
        if len(train_df) == 0:
            raise RuntimeError(
                "No train rows match test peptides — pick a different test split or "
                "increase --n-test so more peptides appear."
            )
        if args.n_train is not None and len(train_df) > args.n_train:
            train_df = train_df.sample(n=args.n_train, random_state=42).reset_index(drop=True)
        train_pos = train_df[["peptide", "trb_cdr3", "mhc_one_allele"]].dropna()
        train_negs = sample_negatives(
            train_pos, tcr_col="trb_cdr3", pep_col="peptide", mhc_col="mhc_one_allele",
            n_per_pos=args.negatives_per_positive, seed=43,
        )
        # PanPep ignores MHC; drop it.
        sup_pos = train_pos[["peptide", "trb_cdr3"]].rename(columns={"peptide": "Peptide", "trb_cdr3": "CDR3"}).copy()
        sup_pos["Label"] = "1"
        sup_neg = train_negs[["peptide", "trb_cdr3"]].rename(columns={"peptide": "Peptide", "trb_cdr3": "CDR3"}).copy()
        sup_neg["Label"] = "0"
        out_df = pd.concat([sup_pos, sup_neg, queries_df], ignore_index=True)

    # Truncate CDR3 to <=25 (PanPep cap) and peptide <=15.
    out_df["Peptide"] = out_df["Peptide"].astype(str).str.slice(0, 15)
    out_df["CDR3"] = out_df["CDR3"].astype(str).str.slice(0, 25)
    out_df = out_df.sort_values("Peptide", kind="mergesort").reset_index(drop=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / args.out
    out_df.to_csv(out_path, index=False)

    n_pos = int((out_df.get("Label", pd.Series(dtype=str)) == "1").sum())
    n_neg = int((out_df.get("Label", pd.Series(dtype=str)) == "0").sum())
    n_unk = int((out_df.get("Label", pd.Series(dtype=str)) == "Unknown").sum())
    print(f"Wrote {out_path}: {len(out_df)} rows ({n_pos} pos support, {n_neg} neg support, {n_unk} queries)")
    print(f"Mode: {args.mode}, unique peptides: {out_df['Peptide'].nunique()}")

    # Truth file: only test positives (test_pos_df). Negatives are inferred at
    # eval time from rows in the prediction set absent from the truth file.
    truth = test_pos_df[["peptide", "trb_cdr3"]].rename(columns={"peptide": "Peptide", "trb_cdr3": "CDR3"}).copy()
    truth["Peptide"] = truth["Peptide"].astype(str).str.slice(0, 15)
    truth["CDR3"] = truth["CDR3"].astype(str).str.slice(0, 25)
    truth["true_label"] = 1
    truth_path = RESULTS_DIR / args.out.replace(".csv", "_truth.csv")
    truth.to_csv(truth_path, index=False)
    print(f"Wrote {truth_path}: {len(truth)} positive ground-truth rows")


if __name__ == "__main__":
    main()
