"""Convert TCRBench AS-TRB-I parquet to ERGO-II training/eval format.

Output:
- A pickled list-of-dicts at track_b/ergo_ii/repo/Samples/<dataset_name>_<split>_samples.pickle
  matching the schema in repo/Sampler.py:read_data().
- A CSV at track_b/ergo_ii/results/<split>.csv matching repo/example.csv format
  for the Predict.py inference path.

Usage:
    python convert_data.py --task as_trb_i --split train --n-rows 100 --dataset-name smoke
    python convert_data.py --task as_trb_i --split test_iid --n-rows 50 --dataset-name smoke
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/home/ubuntu/quest")

from track_b.shared.allele_utils import filter_named_alleles
from track_b.shared.extract_vj import extract_vj_for_rows
from track_b.shared.load_splits import load_as_split
from track_b.shared.negative_sampling import attach_labels, sample_negatives

REPO_DIR = Path("/home/ubuntu/quest/track_b/ergo_ii/repo")
SAMPLES_DIR = REPO_DIR / "Samples"
RESULTS_DIR = Path("/home/ubuntu/quest/track_b/ergo_ii/results")


def parquet_to_ergo_dicts(df: pd.DataFrame, with_negatives: bool, neg_seed: int = 43) -> list[dict]:
    """Convert standardized AS-TRB-I rows to ERGO-II's pickled-dict schema.

    Steps:
      1. filter_named_alleles: keep only rows where mhc_one_allele resolves to
         an HLA name (passthrough or reverse-mapped from protein sequence).
      2. extract_vj: attach trb_v_gene / trb_j_gene (Phase-1 stub returns UNK).
      3. If with_negatives: 5x negatives per positive via shared.negative_sampling.
      4. Build dict per row with ERGO-II's keys.
    """
    df = filter_named_alleles(df)
    df = extract_vj_for_rows(df, chain="trb")

    pos = df[["trb_cdr3", "peptide", "mhc_hla_name", "trb_v_gene", "trb_j_gene"]].rename(
        columns={"mhc_hla_name": "mhc_one_allele"}
    ).dropna(subset=["trb_cdr3", "peptide", "mhc_one_allele"]).reset_index(drop=True)

    if with_negatives:
        negs = sample_negatives(
            pos,
            tcr_col="trb_cdr3",
            pep_col="peptide",
            mhc_col="mhc_one_allele",
            n_per_pos=5,
            seed=neg_seed,
        )
        combined = attach_labels(pos, negs)
    else:
        combined = pos.copy()
        combined["label"] = 1

    rows = []
    for _, r in combined.iterrows():
        rows.append({
            "tcra": "UNK",
            "tcrb": r["trb_cdr3"],
            "va": "UNK",
            "ja": "UNK",
            "vb": r["trb_v_gene"],
            "jb": r["trb_j_gene"],
            "t_cell_type": "UNK",
            "peptide": r["peptide"],
            "protein": "UNK",
            "mhc": r["mhc_one_allele"],
            "sign": int(r["label"]),
        })
    return rows


def write_pickle(rows: list[dict], dataset_name: str, split: str) -> Path:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAMPLES_DIR / f"{dataset_name}_{split}_samples.pickle"
    with open(out_path, "wb") as f:
        pickle.dump(rows, f)
    return out_path


def write_predict_csv(rows: list[dict], split: str) -> Path:
    """Write the inference-format CSV (header matches repo/example.csv)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{split}.csv"
    df = pd.DataFrame([
        {
            "TRA": "" if r["tcra"] == "UNK" else r["tcra"],
            "TRB": r["tcrb"],
            "TRAV": "" if r["va"] == "UNK" else r["va"],
            "TRAJ": "" if r["ja"] == "UNK" else r["ja"],
            "TRBV": "" if r["vb"] == "UNK" else r["vb"],
            "TRBJ": "" if r["jb"] == "UNK" else r["jb"],
            "T-Cell-Type": "" if r["t_cell_type"] == "UNK" else r["t_cell_type"],
            "Peptide": r["peptide"],
            "MHC": "" if r["mhc"] == "UNK" else r["mhc"],
            "label": r["sign"],
        }
        for r in rows
    ])
    df.to_csv(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="as_trb_i", choices=["as_trb_i", "as_tra_i", "as_paired_i"])
    parser.add_argument("--split", required=True)
    parser.add_argument("--n-rows", type=int, default=None, help="Subsample N rows (deterministic, seed=42).")
    parser.add_argument("--dataset-name", required=True, help="ERGO-II dataset key, e.g. tcrbench_as_trb_i_smoke")
    parser.add_argument("--with-negatives", action="store_true",
                        help="Construct 5x negatives. Set on train; off on test (test uses positive-only labels).")
    args = parser.parse_args()

    df = load_as_split(args.task, args.split)
    if args.n_rows is not None:
        df = df.sample(n=min(args.n_rows, len(df)), random_state=42).reset_index(drop=True)

    rows = parquet_to_ergo_dicts(df, with_negatives=args.with_negatives)

    pkl = write_pickle(rows, args.dataset_name, args.split)
    csv = write_predict_csv(rows, f"{args.dataset_name}_{args.split}")
    n_pos = sum(1 for r in rows if r["sign"] == 1)
    n_neg = sum(1 for r in rows if r["sign"] == 0)
    print(f"Wrote {pkl}: {n_pos} pos + {n_neg} neg = {len(rows)} rows")
    print(f"Wrote {csv}")


if __name__ == "__main__":
    main()
