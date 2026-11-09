"""Train/validation/test routing for the CPT corpus (corpus repair, step 6b).

Defect. The builder routes a row by its components' clusters: test if any axis says test, else
validation, else train. Receptor clusters are connected components at Hamming distance 1, which
links only sequences of equal length, so every cluster has one length; at this scale each common
length forms a giant component, and a cluster goes to one split whole. Run05's splits therefore
differ by CDR3 length (train 0.13% of beta CDR3s at 12 letters or fewer; validation 38.5% at
exactly 12; test 43.5% at 18).

Policy `receptor-hash-v1`. A receptor's split is a seeded hash of its own CDR3 string, so the
same receptor always lands in the same split and split is independent of length. Peptide and MHC
axes keep cluster routing: peptide clusters do not percolate (largest 3,512 of 16.7M) and their
cluster-level separation is worth keeping. The priority rule (test > validation > train) is
unchanged. What is given up: two receptors one letter apart may now sit in different splits.
`cluster-v0` is the shipped behaviour, kept only to reproduce past builds.

The hash is md5, not DuckDB's hash(), whose values are not promised stable across versions.
"""
from __future__ import annotations

import hashlib

CLUSTER_ROUTING = "cluster-v0"
RECEPTOR_HASH_ROUTING = "receptor-hash-v1"
ROUTING_POLICIES = (CLUSTER_ROUTING, RECEPTOR_HASH_ROUTING)

# Row targets of the original design (`row_routing_stats.json`: 0.8 / 0.1 / 0.1).
TRAIN_FRACTION, VAL_FRACTION = 0.80, 0.10
_BUCKETS = 10000
RECEPTOR_AXES = ("trb_cdr3", "tra_cdr3")


def _bucket(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % _BUCKETS


def receptor_split(molecule, axis: str, seed: int):
    """Split of one receptor string; None when the row has no such receptor."""
    if not isinstance(molecule, str):
        return None
    b = _bucket(f"{seed}|{axis}|{molecule}")
    if b < int(TRAIN_FRACTION * _BUCKETS):
        return "train"
    return "val" if b < int((TRAIN_FRACTION + VAL_FRACTION) * _BUCKETS) else "test"


def receptor_split_sql(expr: str, axis: str, seed: int) -> str:
    """SQL twin of `receptor_split`; a test pins their agreement. NULL in gives NULL out."""
    b = f"(CAST(('0x' || substr(md5(concat('{int(seed)}|{axis}|', {expr})), 1, 8)) AS BIGINT) % {_BUCKETS})"
    return (f"CASE WHEN {expr} IS NULL THEN NULL "
            f"WHEN {b} < {int(TRAIN_FRACTION * _BUCKETS)} THEN 'train' "
            f"WHEN {b} < {int((TRAIN_FRACTION + VAL_FRACTION) * _BUCKETS)} THEN 'val' ELSE 'test' END")


def combine(splits) -> str:
    """The builder's priority rule over the per-axis splits of one row (None = axis absent)."""
    present = [s for s in splits if s is not None]
    for label in ("test", "val", "train"):
        if label in present:
            return label
    return "train"
