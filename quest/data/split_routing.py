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

Policy `receptor-hash-v2` (decided 2026-11-09, sized as "R3"). `receptor-hash-v1` plus two
changes to the other axes, because under v1 a row is held out if ANY of up to five axes says so
and the scarcest rows carry the most axes: only 36.7% of peptide + class-I rows reached training.

  * The MHC pocket axes cast no vote. The shipped pocket assignment sent 39% of the rows it voted
    on to validation or test and withheld about 3.5M class-I rows. Given up: an MHC pocket may
    occur on both sides (185 of 2,658 class-I pockets on the real rows). The repair plan permits
    MHC recurrence; holding out whole alleles is the benchmark's `test_novel_allele` condition.
  * A peptide's split is a seeded hash of its CLUSTER id, not the shipped per-cluster assignment.
    Clusters stay whole, so peptide-cluster separation is kept; only the 80/10/10 fractions are
    restored. A peptide with no cluster falls back to a hash of its own string, so it still
    votes and can never straddle through silence.

Measured on the 41.1M eligible, isolated rows: `peptide_mhc_one` rows reaching training rise from
36.7% to 80.0% (78.0% over every peptide-bearing classical class-I row), and peptides occurring on both sides of the train boundary fall from 1.72M to
2,181, because the pocket axis no longer drags part of a peptide's rows out of its split.
A row whose only component is an MHC chain now has no vote and goes to train by default.

The hash is md5, not DuckDB's hash(), whose values are not promised stable across versions.
"""
from __future__ import annotations

import hashlib

CLUSTER_ROUTING = "cluster-v0"
RECEPTOR_HASH_ROUTING = "receptor-hash-v1"
RECEPTOR_HASH_V2_ROUTING = "receptor-hash-v2"
ROUTING_POLICIES = (CLUSTER_ROUTING, RECEPTOR_HASH_ROUTING, RECEPTOR_HASH_V2_ROUTING)
# Policies under which the receptor strings, read from the source row, decide their axes.
HASHED_RECEPTOR_POLICIES = (RECEPTOR_HASH_ROUTING, RECEPTOR_HASH_V2_ROUTING)

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


_PEPTIDE_CLUSTER_AXIS = "peptide_cluster"
_PEPTIDE_STRING_AXIS = "peptide_unclustered"


def peptide_split(cluster_id, peptide, seed: int):
    """`receptor-hash-v2` split of a peptide: by its cluster id, else by its own string.

    `cluster_id` is None or -1 when the peptide has no cluster. None when the row has no peptide.
    """
    if cluster_id is not None and cluster_id == cluster_id and int(cluster_id) >= 0:
        return receptor_split(str(int(cluster_id)), _PEPTIDE_CLUSTER_AXIS, seed)
    if isinstance(peptide, str) and peptide:
        return receptor_split(peptide, _PEPTIDE_STRING_AXIS, seed)
    return None


def peptide_split_sql(cluster_expr: str, peptide_expr: str, seed: int) -> str:
    """SQL twin of `peptide_split`. `cluster_expr` is NULL when the peptide has no cluster."""
    by_cluster = receptor_split_sql(f"CAST({cluster_expr} AS VARCHAR)", _PEPTIDE_CLUSTER_AXIS, seed)
    by_string = receptor_split_sql(f"NULLIF({peptide_expr}, '')", _PEPTIDE_STRING_AXIS, seed)
    # A negative id is "no cluster", as in Python: the builder writes -1 for one, and although it
    # does so only after voting, the two languages must not disagree on any value.
    return f"CASE WHEN {cluster_expr} IS NOT NULL AND {cluster_expr} >= 0 THEN {by_cluster} ELSE {by_string} END"


def axis_vote_sql(policy: str, seed: int) -> dict[str, str]:
    """The five per-axis split expressions the builder selects as s_trb, s_tra, s_pep, s_m1, s_m2.

    Written against the builder's join aliases: `k` is the source row and `jtrb`, `jtra`, `jpep`,
    `jm1`, `jm2` are the LEFT-JOINed cluster lookups (columns `cluster_id`, `split`).
    """
    if policy not in ROUTING_POLICIES:
        raise ValueError(f"unknown split routing {policy!r}; expected one of {ROUTING_POLICIES}")
    votes = {"s_trb": "jtrb.split", "s_tra": "jtra.split", "s_pep": "jpep.split",
             "s_m1": "jm1.split", "s_m2": "jm2.split"}
    if policy in HASHED_RECEPTOR_POLICIES:
        votes["s_trb"] = receptor_split_sql("k.trb_cdr3", RECEPTOR_AXES[0], seed)
        votes["s_tra"] = receptor_split_sql("k.tra_cdr3", RECEPTOR_AXES[1], seed)
    if policy == RECEPTOR_HASH_V2_ROUTING:
        votes["s_pep"] = peptide_split_sql("jpep.cluster_id", "k.peptide", seed)
        votes["s_m1"] = votes["s_m2"] = "CAST(NULL AS VARCHAR)"
    return votes
