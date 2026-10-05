"""Load + prepare AS data for LSTM/MLP training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from pathlib import Path

from track_a.shared.load_splits import (
    CLASS_I_TASKS,
    CLASS_II_FULL_TASKS,
    CLASS_II_TASKS,
    SPLITS_DIR,
    class_i_companion,
    load_as_class_i,
    load_as_class_ii,
    load_candidate_pool,
)
from track_a.shared.negative_sampling import sample_negatives, attach_labels

# Columns we need for LSTM/MLP. Smaller pull = faster.
_COMMON_COLS = ["trb_cdr3", "peptide", "mhc_one_allele", "mhc_two_allele", "row_id"]
_TRA_COLS = ["tra_cdr3"]


def _columns_for_task(task: str) -> list[str]:
    base = list(_COMMON_COLS)
    if task.startswith("as_paired"):
        base += _TRA_COLS
    if task.endswith("_ii"):
        # eval-only files don't have tcr_cluster/pep_cluster/condition
        return base
    base += ["condition"]  # class I files have it
    return base


def load_train_val(
    task: str,
    sample_n: Optional[int] = None,
    inject_negatives: bool = True,
    seed: int = 43,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train+val DataFrames with negatives injected.

    Routing:
      - Class I tasks → load that task's own train/val.
      - Class II FULL tasks (currently `as_paired_ii` after the 2026-05-04
        HF refresh) → load that task's own train/val.
      - Eval-only class-II tasks (`as_trb_ii`, `as_tra_ii`) → fall back
        to the matching class-I companion's train/val.

    Args:
        task: as_*.
        sample_n: optionally subsample train rows (for smoke tests).
        inject_negatives: if True, sample 5x negatives via shared sampler.
        seed: RNG for subsampling.

    Returns:
        (train_df, val_df) with columns [trb_cdr3, (tra_cdr3,) peptide,
        mhc_one_allele/mhc_two_allele, row_id, label].
    """
    if task in CLASS_I_TASKS or task in CLASS_II_FULL_TASKS:
        train_task = task
    else:
        # eval-only class-II → train on class-I companion
        train_task = class_i_companion(task)
    cols = _columns_for_task(train_task)
    train_pos = load_as_class_i(train_task, "train", columns=cols)
    val_pos = load_as_class_i(train_task, "val", columns=cols)
    if sample_n is not None:
        train_pos = train_pos.sample(n=min(sample_n, len(train_pos)), random_state=seed).reset_index(drop=True)
        val_pos = val_pos.sample(n=min(sample_n // 5, len(val_pos)), random_state=seed).reset_index(drop=True)
    if not inject_negatives:
        train_pos = train_pos.copy()
        val_pos = val_pos.copy()
        train_pos["label"] = 1
        val_pos["label"] = 1
        return train_pos, val_pos

    mhc_col = "mhc_one_allele" if train_task.endswith("_i") else "mhc_two_allele"
    train_negs = sample_negatives(train_pos, tcr_col="trb_cdr3", pep_col="peptide", mhc_col=mhc_col, seed=seed)
    val_negs = sample_negatives(val_pos, tcr_col="trb_cdr3", pep_col="peptide", mhc_col=mhc_col, seed=seed + 1)
    train = attach_labels(train_pos, train_negs).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val = attach_labels(val_pos, val_negs).sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    return train, val


def load_test_condition(task: str, condition: str) -> pd.DataFrame:
    """Load one test partition.

    - Class I tasks: condition ∈ {test_iid, test_novel_tcr, test_novel_pep, test_novel_allele, test_level4}.
    - Class II FULL tasks (e.g. `as_paired_ii` post-2026-05-04): same 5 conditions.
    - Eval-only class-II tasks (`as_trb_ii`, `as_tra_ii`): condition must be 'eval'.
    """
    cols = _columns_for_task(task)
    if task in CLASS_I_TASKS or task in CLASS_II_FULL_TASKS:
        df = load_as_class_i(task, condition, columns=cols)
    elif task in CLASS_II_TASKS:
        if condition != "eval":
            raise ValueError(f"eval-only class-II task {task} requires condition='eval', got {condition}")
        df = load_as_class_ii(task, columns=cols)
    else:
        raise ValueError(f"unknown task: {task}")
    df = df.copy()
    df["label"] = 1  # all test rows are positives; retrieval ranks all candidates
    return df


def load_candidates(task: str) -> list[str]:
    """Return the list of candidate peptides for retrieval.

    Class I (`as_*_i`):
        Use the published `as_class_i_candidate_pool.tsv` (3,563 peptides).

    Class II (`as_*_ii`):
        Use the official `as_class_ii_candidate_pool.tsv` published on
        the HF dataset on 2026-05-04 (~990 peptides). If the file isn't
        present (legacy-state machines), fall back to the prior synthesis
        of `union(class-I pool, peptides in {task}_eval.parquet)` so the
        class-II test peptides remain reachable.

    Returns deduplicated, sorted list of strings.
    """
    if task in CLASS_II_TASKS:
        ii_pool_path = SPLITS_DIR / "as_class_ii_candidate_pool.tsv"
        if ii_pool_path.exists():
            df = pd.read_csv(ii_pool_path, sep="\t")
            pep_col = "peptide" if "peptide" in df.columns else df.columns[0]
            return sorted(set(df[pep_col].astype(str).tolist()))
        # Fallback: synthesize as before
        pool = load_candidate_pool()
        pep_col = "peptide" if "peptide" in pool.columns else pool.columns[0]
        base = set(pool[pep_col].astype(str).tolist())
        # For full-split class-II, union the train + test peptides; else use _eval
        if task in CLASS_II_FULL_TASKS:
            for cond in ("train", "test_iid", "test_novel_tcr", "test_novel_pep",
                         "test_novel_allele", "test_level4"):
                p = SPLITS_DIR / f"{task}_{cond}.parquet"
                if p.exists():
                    df = pd.read_parquet(p, columns=["peptide"])
                    base |= set(df["peptide"].astype(str).tolist())
        else:
            eval_df = load_as_class_ii(task, columns=["peptide"])
            base |= set(eval_df["peptide"].astype(str).tolist())
        base.discard("nan")
        base.discard("")
        return sorted(base)
    # Class I
    pool = load_candidate_pool()
    pep_col = "peptide" if "peptide" in pool.columns else pool.columns[0]
    return sorted(set(pool[pep_col].astype(str).tolist()))
