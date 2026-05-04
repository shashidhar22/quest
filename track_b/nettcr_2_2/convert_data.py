"""Convert TCRBench AS Class I parquet to NetTCR-2.2 CSV format.

Output columns: peptide, A1, A2, A3, B1, B2, B3, binder, allele (metadata only)

Decisions per verification doc:
- TRB-only data (AS-TRB-I): zero-pad A1/A2/A3 with empty strings.
- Drop rows where B1 or B2 is null (NetTCR-2.2 needs all 6 CDRs and 17.7% are null).
- MHC ignored — model doesn't use it; allele kept as metadata for ranking/audit.
- Negatives: 5x via shared.negative_sampling (peptide-only swap, seed=43).

CLI:
    python convert_data.py --task as_trb_i --split train --n-rows 100 --out smoke_train.csv --with-negatives
    python convert_data.py --task as_trb_i --split test_iid --n-rows 50 --out smoke_test_iid.csv --with-negatives
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/home/ubuntu/quest")

from track_b.shared.load_splits import load_as_split
from track_b.shared.negative_sampling import attach_labels, sample_negatives

RESULTS_DIR = Path("/home/ubuntu/quest/track_b/nettcr_2_2/results")


def parquet_to_nettcr(df: pd.DataFrame, task: str, with_negatives: bool, neg_seed: int = 43) -> pd.DataFrame:
    """Standardize TCRBench rows for NetTCR-2.2.

    Steps:
      1. Drop rows where required CDRs are null (per task: TRB-I needs B1/B2/B3,
         TRA-I needs A1/A2/A3, Paired-I needs all 6).
      2. Add empty-string placeholders for missing chain (TRB-I → empty A1/A2/A3).
      3. Sample 5x negatives if with_negatives.
      4. Emit CSV with peptide, A1..B3, binder, allele.
    """
    pos = df.copy()

    if task == "as_trb_i":
        pos = pos.dropna(subset=["trb_cdr1", "trb_cdr2", "trb_cdr3", "peptide", "mhc_one_allele"]).reset_index(drop=True)
        pos["tra_cdr1"] = ""
        pos["tra_cdr2"] = ""
        pos["tra_cdr3"] = ""
    elif task == "as_tra_i":
        pos = pos.dropna(subset=["tra_cdr1", "tra_cdr2", "tra_cdr3", "peptide", "mhc_one_allele"]).reset_index(drop=True)
        pos["trb_cdr1"] = ""
        pos["trb_cdr2"] = ""
        pos["trb_cdr3"] = ""
    elif task == "as_paired_i":
        pos = pos.dropna(subset=["tra_cdr1", "tra_cdr2", "tra_cdr3",
                                 "trb_cdr1", "trb_cdr2", "trb_cdr3",
                                 "peptide", "mhc_one_allele"]).reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported task: {task}")

    cols = ["peptide", "tra_cdr1", "tra_cdr2", "tra_cdr3",
            "trb_cdr1", "trb_cdr2", "trb_cdr3", "mhc_one_allele"]
    pos = pos[cols].copy()

    if with_negatives:
        negs = sample_negatives(
            pos, tcr_col="trb_cdr3", pep_col="peptide", mhc_col="mhc_one_allele",
            n_per_pos=5, seed=neg_seed,
        )
        combined = attach_labels(pos, negs)
    else:
        combined = pos.copy()
        combined["label"] = 1

    out = combined.rename(columns={
        "tra_cdr1": "A1", "tra_cdr2": "A2", "tra_cdr3": "A3",
        "trb_cdr1": "B1", "trb_cdr2": "B2", "trb_cdr3": "B3",
        "mhc_one_allele": "allele",
        "label": "binder",
    })

    # Right-truncate any sequence longer than NetTCR's hard caps.
    caps = {"peptide": 12, "A1": 7, "A2": 8, "A3": 22, "B1": 6, "B2": 7, "B3": 23}
    for c, n in caps.items():
        out[c] = out[c].astype(str).str.slice(0, n)

    return out[["peptide", "A1", "A2", "A3", "B1", "B2", "B3", "binder", "allele"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="as_trb_i", choices=["as_trb_i", "as_tra_i", "as_paired_i"])
    p.add_argument("--split", required=True)
    p.add_argument("--n-rows", type=int, default=None)
    p.add_argument("--out", required=True, help="Output CSV path (relative to results/).")
    p.add_argument("--with-negatives", action="store_true")
    args = p.parse_args()

    df = load_as_split(args.task, args.split)
    if args.n_rows is not None:
        df = df.sample(n=min(args.n_rows, len(df)), random_state=42).reset_index(drop=True)

    out_df = parquet_to_nettcr(df, task=args.task, with_negatives=args.with_negatives)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / args.out
    out_df.to_csv(out_path, index=False)

    n_pos = int((out_df["binder"] == 1).sum())
    n_neg = int((out_df["binder"] == 0).sum())
    print(f"Wrote {out_path}: {n_pos} pos + {n_neg} neg = {len(out_df)} rows")
    print(f"Unique peptides: {out_df.peptide.nunique()}, alleles: {out_df.allele.nunique()}")


if __name__ == "__main__":
    main()
