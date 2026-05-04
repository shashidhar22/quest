"""Convert TCRBench AS-TRB-I parquet to TEIM-Seq TSV format.

Output columns: cdr3, epitope, label

TEIM-Seq has hard length caps:
- CDR3 length must be <= 20 (data_process.py:168)
- Epitope length must be in [8, 12] (data_process.py:142)

Rows outside these bounds are dropped (with a count log). Negatives constructed
via shared.negative_sampling (5x, peptide-only swap, seed=43).

CLI:
    python convert_data.py --task as_trb_i --split train --n-rows 100 \
        --out smoke_train.tsv --with-negatives
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/home/ubuntu/quest")

from track_b.shared.load_splits import load_as_split
from track_b.shared.negative_sampling import attach_labels, sample_negatives

RESULTS_DIR = Path("/home/ubuntu/quest/track_b/teim/results")
REPO_DATA = Path("/home/ubuntu/quest/track_b/teim/repo/data/binding_data")


def filter_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """Apply TEIM-Seq's length caps."""
    n0 = len(df)
    df = df[df["peptide"].str.len().between(8, 12, inclusive="both")]
    n_pep = len(df)
    df = df[df["trb_cdr3"].str.len() <= 20]
    n_cdr3 = len(df)
    print(f"Length filter: {n0} -> {n_pep} (peptide len ∈ [8,12]) -> {n_cdr3} (cdr3 len ≤ 20)")
    return df.reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="as_trb_i", choices=["as_trb_i", "as_tra_i", "as_paired_i"])
    p.add_argument("--split", required=True)
    p.add_argument("--n-rows", type=int, default=None)
    p.add_argument("--out", required=True, help="Output TSV name (under results/).")
    p.add_argument("--with-negatives", action="store_true")
    p.add_argument("--copy-to-repo", action="store_true",
                   help="Also copy the TSV under repo/data/binding_data/ so the upstream loader finds it.")
    args = p.parse_args()

    df = load_as_split(args.task, args.split)
    if args.n_rows is not None:
        df = df.sample(n=min(args.n_rows, len(df)), random_state=42).reset_index(drop=True)

    df = df[["peptide", "trb_cdr3", "mhc_one_allele"]].dropna().reset_index(drop=True)
    df = filter_lengths(df)

    if args.with_negatives:
        negs = sample_negatives(
            df, tcr_col="trb_cdr3", pep_col="peptide", mhc_col="mhc_one_allele",
            n_per_pos=5, seed=43,
        )
        # Negatives may also exceed length caps via the swapped peptide.
        negs = filter_lengths(negs)
        combined = attach_labels(df, negs)
    else:
        combined = df.copy()
        combined["label"] = 1

    out_df = combined[["trb_cdr3", "peptide", "label"]].rename(
        columns={"trb_cdr3": "cdr3", "peptide": "epitope"}
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / args.out
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {out_path}: {(out_df.label==1).sum()} pos + {(out_df.label==0).sum()} neg = {len(out_df)} rows")

    if args.copy_to_repo:
        REPO_DATA.mkdir(parents=True, exist_ok=True)
        repo_path = REPO_DATA / out_path.name
        out_df.to_csv(repo_path, sep="\t", index=False)
        print(f"Also copied to {repo_path}")


if __name__ == "__main__":
    main()
