"""Load benchmark_v2 AS splits as pandas DataFrames.

Class I tasks (as_trb_i, as_tra_i, as_paired_i): 8 partitions
  train, val, test_iid, test_novel_tcr, test_novel_pep,
  test_novel_allele, test_level4, test_mixed

Class II tasks (as_trb_ii, as_tra_ii, as_paired_ii): eval-only
  Single _eval.parquet per task; see PHASE0_CLASSII_INVESTIGATION.md.

The on-disk schema is the enriched-parquet schema (~25-29 columns):
  tra, trb, peptide, mhc_one, mhc_two, mhc_one_allele, mhc_two_allele,
  tra_cdr1/2/3, trb_cdr1/2/3, tra_full, trb_full, source_db_*,
  source_type, row_id, plus tcr_cluster / pep_cluster / condition for
  class I.

We deliberately mirror track_b/shared/load_splits.py's API so cross-track
work is interchangeable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

def _resolve_splits_dir() -> Path:
    """Where benchmark_v2's AS splits live.

    Resolved inside this checkout first, so the judge is runnable and testable off the
    training box; the box layout stays as a fallback. Override with QUEST_SPLITS_DIR.
    This was hardcoded to the box path until 2026-11-03, the same defect already fixed
    in track_a/shared/negative_sampling.py.
    """
    env = os.environ.get("QUEST_SPLITS_DIR")
    candidates = (
        *((Path(env),) if env else ()),
        Path(__file__).resolve().parents[2] / "benchmark_v2" / "splits",
        Path("/home/ubuntu/quest/benchmark_v2/splits"),
    )
    for c in candidates:
        if c.is_dir():
            return c
    # None present: return the first candidate so the loaders' FileNotFoundError names a
    # path the caller can act on, rather than a path that was never going to exist.
    return candidates[0]


SPLITS_DIR = _resolve_splits_dir()

CLASS_I_TASKS = {"as_trb_i", "as_tra_i", "as_paired_i"}
CLASS_II_TASKS = {"as_trb_ii", "as_tra_ii", "as_paired_ii"}

# Class-II tasks that have FULL splits (train/val/test_*) on HF, like Class I.
# As of 2026-05-04 this is only `as_paired_ii`. The other class-II tasks
# (`as_trb_ii`, `as_tra_ii`) remain eval-only.
CLASS_II_FULL_TASKS = {"as_paired_ii"}

CLASS_I_CONDITIONS = (
    "train",
    "val",
    "test_iid",
    "test_novel_tcr",
    "test_novel_pep",
    "test_novel_allele",
    "test_level4",
    "test_mixed",
)

# Conditions reported in the user-required results JSON (class I + full-split class-II).
# Excludes train/val (training-only) and test_mixed (not in user's matrix).
CLASS_I_REPORT_CONDITIONS = (
    "test_iid",
    "test_novel_tcr",
    "test_novel_pep",
    "test_novel_allele",
    "test_level4",
)


def load_as_class_i(
    task: str,
    condition: str,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load one Class-I-style AS split (8 conditions).

    Also accepts class-II tasks in CLASS_II_FULL_TASKS (currently
    `as_paired_ii`), which expose the same train/val/test_* layout as
    class I after the 2026-05-04 HF refresh.

    Args:
        task: as_trb_i | as_tra_i | as_paired_i | as_paired_ii
        condition: one of CLASS_I_CONDITIONS
        columns: optional column subset.
    """
    full_tasks = CLASS_I_TASKS | CLASS_II_FULL_TASKS
    if task not in full_tasks:
        raise ValueError(f"task must be one of {full_tasks}, got {task}")
    if condition not in CLASS_I_CONDITIONS:
        raise ValueError(f"condition must be one of {CLASS_I_CONDITIONS}, got {condition}")
    path = SPLITS_DIR / f"{task}_{condition}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path, columns=columns)


def load_as_class_ii(task: str, columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Load the AS-Class-II eval-only file.

    Class II is eval-only by design (PHASE0_CLASSII_INVESTIGATION.md).
    There is no train/val for class II; baselines train on the matching
    class I task and evaluate on the class II eval file.
    """
    if task not in CLASS_II_TASKS:
        raise ValueError(f"task must be one of {CLASS_II_TASKS}, got {task}")
    path = SPLITS_DIR / f"{task}_eval.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path, columns=columns)


def load_candidate_pool() -> pd.DataFrame:
    """AS Class I peptide candidate pool (3,564 unique peptides).

    The retrieval eval does NOT rank all of these. judge.cmd_eval draws a shortlist per
    query (the true peptide plus POOL_SIZE - 1 distractors), so random Recall@10 is
    10/POOL_SIZE, not 10/3564.

    The pool is 76 % 9-mers (2,718 of 3,563 measured 2026-10-15). Drawing distractors
    uniformly therefore leaks the true peptide's length: see judge._draw_distractors and
    quest-cpt/findings.md 9.6.
    """
    return pd.read_csv(SPLITS_DIR / "as_class_i_candidate_pool.tsv", sep="\t")


def class_i_companion(class_ii_task: str) -> str:
    """Map class-II task to the class-I task we train on.

    as_trb_ii    -> as_trb_i
    as_tra_ii    -> as_tra_i
    as_paired_ii -> as_paired_i
    """
    if class_ii_task not in CLASS_II_TASKS:
        raise ValueError(f"not a class-II task: {class_ii_task}")
    return class_ii_task[:-3] + "_i"


def all_test_files(task: str) -> dict[str, Path]:
    """Return {condition_label: parquet_path} for a task.

    Class I and full-split class-II tasks return 5 entries
    (test_iid, test_novel_tcr, test_novel_pep, test_novel_allele,
    test_level4). Eval-only class-II tasks return {"eval": ...}.
    Used by evaluate.py to iterate over conditions for results JSON.
    """
    if task in CLASS_I_TASKS or task in CLASS_II_FULL_TASKS:
        return {
            cond: SPLITS_DIR / f"{task}_{cond}.parquet"
            for cond in CLASS_I_REPORT_CONDITIONS
        }
    if task in CLASS_II_TASKS:
        return {"eval": SPLITS_DIR / f"{task}_eval.parquet"}
    raise ValueError(f"unknown task: {task}")
