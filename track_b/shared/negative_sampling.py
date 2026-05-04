"""Negative sampling for binary-classification baselines on the AS task.

The AS task ships positives only. Per Track B brief: 5x negatives per positive,
sampled from training TCRs paired with random training peptides, filtered against
the positive set. Mirrors PAIR-task convention (BENCHMARK_SPLITS.md sec 12) but
uses (TCR, peptide, MHC) as the positive key instead of (tra, trb).

Seed: 43 (== 42 + 1, distinct from the partition-sampling seed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NEG_SEED = 43
NEG_RATIO = 5


def sample_negatives(
    positives: pd.DataFrame,
    tcr_col: str = "trb_cdr3",
    pep_col: str = "peptide",
    mhc_col: str = "mhc_one_allele",
    n_per_pos: int = NEG_RATIO,
    seed: int = NEG_SEED,
) -> pd.DataFrame:
    """Return a DataFrame of negatives matching the positives schema.

    Algorithm:
      1. Build positive-set: set of (tcr, pep, mhc) tuples.
      2. For each positive row, draw n_per_pos random peptides from the train
         peptide pool. Pair (tcr_i, mhc_i) with each random peptide.
      3. Reject draws whose (tcr, pep, mhc) is in the positive-set; resample
         once per rejection (rejection rate is small at 5x ratio).

    Output rows have all the same columns as `positives` plus a `label` column
    (1 for positives, 0 for negatives). The caller is expected to concat
    positives + negatives.

    Notes
    -----
    - Sampling is with-replacement on peptides (draws are independent).
    - We do NOT shuffle TCRs or alleles; we keep (tcr_i, mhc_i) intact and only
      vary the peptide. This matches the brief: "training TCRs paired with
      random training peptides".
    """
    rng = np.random.default_rng(seed)
    pos_set = set(zip(positives[tcr_col], positives[pep_col], positives[mhc_col]))
    pep_pool = positives[pep_col].dropna().to_numpy()

    if len(pep_pool) == 0:
        raise ValueError("Empty peptide pool — cannot sample negatives.")

    rows = []
    for _, row in positives.iterrows():
        tcr = row[tcr_col]
        mhc = row[mhc_col]
        drawn = 0
        attempts = 0
        max_attempts = n_per_pos * 10
        while drawn < n_per_pos and attempts < max_attempts:
            attempts += 1
            pep = pep_pool[rng.integers(len(pep_pool))]
            if (tcr, pep, mhc) in pos_set:
                continue
            new_row = row.copy()
            new_row[pep_col] = pep
            rows.append(new_row)
            drawn += 1
    negs = pd.DataFrame(rows).reset_index(drop=True)
    negs["label"] = 0
    return negs


def attach_labels(positives: pd.DataFrame, negatives: pd.DataFrame) -> pd.DataFrame:
    """Concatenate positives + negatives with label column."""
    pos = positives.copy()
    pos["label"] = 1
    return pd.concat([pos, negatives], ignore_index=True)
