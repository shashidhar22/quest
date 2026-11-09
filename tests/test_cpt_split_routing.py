"""Split routing of the CPT corpus (corpus repair step 6b).

Why this file exists. Receptor clusters are Hamming-distance-1 components, every one of a single
length, and the giant ones were routed to one split whole. Run05's train split has almost no
beta CDR3 of 12 letters or fewer. `receptor-hash-v1` routes a receptor by a hash of its own
sequence; these tests pin that the split no longer depends on length, and what stays the same.
"""
import collections
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quest.data.split_routing import (  # noqa: E402
    RECEPTOR_AXES,
    ROUTING_POLICIES,
    combine,
    receptor_split,
    receptor_split_sql,
)

AA = "ACDEFGHIKLMNPQRSTVWY"


def receptors(n, seed=0):
    rng = random.Random(seed)
    return ["C" + "".join(rng.choice(AA) for _ in range(rng.randint(7, 18))) + "F" for _ in range(n)]


def test_policies_are_named_and_the_legacy_one_is_kept():
    assert ROUTING_POLICIES == ("cluster-v0", "receptor-hash-v1")
    assert RECEPTOR_AXES == ("trb_cdr3", "tra_cdr3")


def test_same_receptor_always_gets_the_same_split():
    for r in receptors(200):
        assert receptor_split(r, "trb_cdr3", 42) == receptor_split(r, "trb_cdr3", 42)


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_a_row_without_the_receptor_has_no_vote(missing):
    assert receptor_split(missing, "trb_cdr3", 42) is None


def test_seed_and_axis_both_change_the_assignment():
    rs = receptors(2000)
    base = [receptor_split(r, "trb_cdr3", 42) for r in rs]
    assert base != [receptor_split(r, "trb_cdr3", 43) for r in rs]
    assert base != [receptor_split(r, "tra_cdr3", 42) for r in rs]


def test_split_shares_are_80_10_10():
    c = collections.Counter(receptor_split(r, "trb_cdr3", 42) for r in receptors(40000))
    n = sum(c.values())
    assert abs(c["train"] / n - 0.80) < 0.01
    assert abs(c["val"] / n - 0.10) < 0.01 and abs(c["test"] / n - 0.10) < 0.01


def test_split_does_not_depend_on_cdr3_length():
    """The defect itself: under cluster routing, length 12 was 38.5% of validation and 0.1% of
    train. Here every length must have the same train share to within sampling noise."""
    by_len = collections.defaultdict(collections.Counter)
    for r in receptors(120000, seed=1):
        by_len[len(r)][receptor_split(r, "trb_cdr3", 42)] += 1
    for length, c in by_len.items():
        n = sum(c.values())
        assert n > 5000
        assert abs(c["train"] / n - 0.80) < 0.02, (length, dict(c))
        assert abs(c["val"] / n - 0.10) < 0.015 and abs(c["test"] / n - 0.10) < 0.015


def test_one_letter_neighbours_may_now_fall_in_different_splits():
    """Stated openly, not hidden: this is what is given up for length-independent splits."""
    rs = receptors(3000, seed=2)
    differ = sum(receptor_split(r, "trb_cdr3", 42) != receptor_split(r[:-2] + "W" + r[-1], "trb_cdr3", 42)
                 for r in rs)
    assert 0.2 < differ / len(rs) < 0.5      # about 1 - (0.8^2 + 0.1^2 + 0.1^2) = 0.34


@pytest.mark.parametrize("splits,expected", [
    (["train", "train"], "train"), (["train", "val"], "val"), (["val", "test"], "test"),
    (["train", None, "test"], "test"), ([None, None], "train"), ([], "train"),
])
def test_priority_rule_is_unchanged(splits, expected):
    assert combine(splits) == expected


def test_sql_and_python_agree_on_every_receptor():
    """The builder routes in SQL and the verifier re-derives in Python; they must never differ."""
    duckdb = pytest.importorskip("duckdb", reason="routing is SQL in the builder; duckdb required")
    rs = receptors(5000, seed=3) + [None, "", "c a s s f", "CASSÅF"]
    con = duckdb.connect()
    try:
        con.execute("CREATE TABLE t (i INTEGER, r VARCHAR)")
        con.executemany("INSERT INTO t VALUES (?, ?)", list(enumerate(rs)))
        for axis in RECEPTOR_AXES:
            got = dict(con.execute(
                f"SELECT i, {receptor_split_sql('r', axis, 42)} FROM t").fetchall())
            for i, r in enumerate(rs):
                assert got[i] == receptor_split(r, axis, 42), (axis, r)
    finally:
        con.close()
