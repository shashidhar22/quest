"""Load the 8-cell cubes for cross-encoder SFT.

Wraps `/home/ubuntu/quest/data/antigenic_specificity_class_I/splits/{cube_a_trb,
cube_c_paired}/` HuggingFace DatasetDicts. Adds 5x peptide-swap negatives at
train time via `track_b.shared.negative_sampling`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk

from track_a.esm_sft_cubes.dataset import required_columns

CUBE_ROOT = Path("/home/ubuntu/quest/data/antigenic_specificity_class_I/splits")
CANDIDATE_POOL_TSV = Path("/home/ubuntu/quest/benchmark_v2/splits/as_class_i_candidate_pool.tsv")

CHAINS_FOR_CUBE = {
    "cube_a_trb":    "trb",
    "cube_c_paired": "paired",
}


def _hf_split_to_pandas(cube_name: str, split: str) -> pd.DataFrame:
    """Load one split of a cube as a pandas DataFrame."""
    dd = load_from_disk(str(CUBE_ROOT / cube_name))
    df = dd[split].to_pandas()
    df["_row_id"] = [f"{cube_name}:{split}:{i}" for i in range(len(df))]
    return df


def _filter_complete(df: pd.DataFrame, tcr_variant: str, mhc_variant: str, chains: str) -> pd.DataFrame:
    """Drop rows where any required column for this (variant, chains) is null/empty."""
    cols = required_columns(tcr_variant, mhc_variant, chains)
    mask = pd.Series(True, index=df.index)
    for c in cols:
        if c not in df.columns:
            return df.iloc[0:0]
        col = df[c]
        # null + empty-string filter
        mask &= col.notna() & col.astype(str).ne("")
    return df[mask].reset_index(drop=True)


def load_train_val_with_negatives(
    cube_name: str,
    tcr_variant: str,
    mhc_variant: str,
    seed: int = 43,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load (train, val) with 5x peptide-swap negatives injected into train ONLY.

    The val set is positives-only (label=1) — we use it for early-stopping AUC
    against the positive distribution. The eval-time retrieval metrics are
    computed on the cell test splits, not val.

    Negative sampling uses track_b.shared.negative_sampling (seed=43, n_per_pos=5).
    Keys: (trb_cdr3, peptide, mhc_one_allele).
    """
    from track_b.shared.negative_sampling import sample_negatives, attach_labels

    chains = CHAINS_FOR_CUBE[cube_name]
    train_df = _hf_split_to_pandas(cube_name, "train")
    val_df = _hf_split_to_pandas(cube_name, "val")

    # Filter to rows complete in the chosen variant
    train_df = _filter_complete(train_df, tcr_variant, mhc_variant, chains)
    val_df = _filter_complete(val_df, tcr_variant, mhc_variant, chains)

    # Inject negatives into train AND val (matches track_a/lstm_mlp/convert_data.py
    # pattern at line 85; val negatives use seed+1 to avoid coupling). Without
    # negatives in val, the BCE val_loss/val_auc signal is meaningless.
    train_negs = sample_negatives(
        train_df,
        tcr_col="trb_cdr3", pep_col="peptide", mhc_col="mhc_one_allele",
        n_per_pos=5, seed=seed,
    )
    train_df_lab = attach_labels(train_df, train_negs)
    val_negs = sample_negatives(
        val_df,
        tcr_col="trb_cdr3", pep_col="peptide", mhc_col="mhc_one_allele",
        n_per_pos=5, seed=seed + 1,
    )
    val_df_lab = attach_labels(val_df, val_negs)

    # Shuffle
    train_df_lab = train_df_lab.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df_lab = val_df_lab.sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    return train_df_lab, val_df_lab


def load_cell_test(
    cube_name: str,
    cell: str,
    tcr_variant: str,
    mhc_variant: str,
    max_test_rows: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Load one test cell, filter to rows complete for the variant, cap at max_test_rows.

    Sampling is deterministic (seed). For small cells (n < max_test_rows) returns
    all rows in their natural order.
    """
    chains = CHAINS_FOR_CUBE[cube_name]
    df = _hf_split_to_pandas(cube_name, cell)
    df = _filter_complete(df, tcr_variant, mhc_variant, chains)
    if len(df) > max_test_rows:
        df = df.sample(n=max_test_rows, random_state=seed).reset_index(drop=True)
    return df


def load_full_candidate_pool() -> list[str]:
    """Load the published class-I candidate pool (sorted unique peptides)."""
    pool = pd.read_csv(CANDIDATE_POOL_TSV, sep="\t")
    # The TSV has a 'peptide' column per track_a baselines
    if "peptide" in pool.columns:
        peps = pool["peptide"].dropna().astype(str).str.upper().tolist()
    else:
        peps = pool.iloc[:, 0].dropna().astype(str).str.upper().tolist()
    return sorted(set(peps))


def cube_chains(cube_name: str) -> str:
    return CHAINS_FOR_CUBE[cube_name]
