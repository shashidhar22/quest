"""Load TCRBench AS Class I splits as pandas DataFrames.

The parquet files live at /home/ubuntu/quest/benchmark_v2/splits/. See
docs/wiki/benchmark_v2/BENCHMARK_SPLITS.md for the full schema.

Use cases per baseline:
- ERGO-II: trb_cdr3, peptide, mhc_one_allele, V/J extracted from trb_full
- NetTCR-2.2: trb_cdr1/2/3 (+ tra_cdr1/2/3 for paired), peptide
- PanPep: trb_cdr3, peptide, label
- TEIM-Seq: trb_cdr3, peptide, label
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

SPLITS_DIR = Path("/home/ubuntu/quest/benchmark_v2/splits")

VALID_TASKS = {"as_trb_i", "as_tra_i", "as_paired_i"}
VALID_SPLITS = {
    "train",
    "val",
    "test_iid",
    "test_novel_tcr",
    "test_novel_pep",
    "test_novel_allele",
    "test_level4",
    "test_mixed",
}


def load_as_split(task: str, split: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Load one AS-Class-I split.

    Args:
        task: one of as_trb_i, as_tra_i, as_paired_i.
        split: one of train, val, test_iid, test_novel_tcr, test_novel_pep,
            test_novel_allele, test_level4, test_mixed.
        columns: optional column subset. Pass None to read all 29 columns.
    """
    if task not in VALID_TASKS:
        raise ValueError(f"task must be one of {VALID_TASKS}, got {task}")
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split}")
    path = SPLITS_DIR / f"{task}_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path, columns=columns)


def load_candidate_pool() -> pd.DataFrame:
    """AS Class I peptide candidate pool (3,564 unique peptides)."""
    return pd.read_csv(SPLITS_DIR / "as_class_i_candidate_pool.tsv", sep="\t")
