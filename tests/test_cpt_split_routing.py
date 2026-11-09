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
    assert ROUTING_POLICIES == ("cluster-v0", "receptor-hash-v1", "receptor-hash-v2")
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


# ---------- receptor-hash-v2: the pocket axes do not vote, peptide clusters are hashed ----------

from quest.data.split_routing import axis_vote_sql, peptide_split, peptide_split_sql  # noqa: E402


def test_a_peptide_cluster_stays_whole():
    """Every peptide of one cluster gets that cluster's split, whatever its own string is."""
    for cid in range(300):
        assert len({peptide_split(cid, p, 42) for p in ("GILGFVFTL", "GILGFVFTM", "NLVPMVATV")}) == 1


def test_peptide_cluster_shares_are_80_10_10():
    c = collections.Counter(peptide_split(cid, "X", 42) for cid in range(20000))
    assert abs(c["train"] / 20000 - 0.80) < 0.01
    assert abs(c["val"] / 20000 - 0.10) < 0.01 and abs(c["test"] / 20000 - 0.10) < 0.01


@pytest.mark.parametrize("no_cluster", [None, -1, float("nan")])
def test_a_peptide_without_a_cluster_still_votes_by_its_own_string(no_cluster):
    """Silence would let its rows follow the receptor axes into different splits."""
    assert peptide_split(no_cluster, "GILGFVFTL", 42) in ("train", "val", "test")
    assert peptide_split(no_cluster, "GILGFVFTL", 42) == peptide_split(None, "GILGFVFTL", 42)
    assert peptide_split(no_cluster, None, 42) is None
    assert peptide_split(no_cluster, "", 42) is None


def test_unknown_policy_is_refused():
    with pytest.raises(ValueError):
        axis_vote_sql("no-such-routing", 42)


ROWS = [  # (tag, trb_cdr3, tra_cdr3, peptide, pep_cluster, pep_cluster_split, m1_split, m2_split)
    ("receptor_only", "CASSLGQNTEAFF", None, None, None, None, None, None),
    ("pep_in_cluster", None, None, "GILGFVFTL", 7, "train", None, None),
    ("pep_same_cluster", None, None, "GILGFVFTM", 7, "train", None, None),
    ("pep_no_cluster", None, None, "KLGGALQAK", None, None, None, None),
    ("pep_empty", "CASSF", None, "", None, None, None, None),
    ("mhc_says_test", None, None, "GILGFVFTL", 7, "train", "test", None),
    ("mhc_only", None, None, None, None, None, "test", "val"),
    ("everything", "CASSF", "CAVR", "NLVPMVATV", 9, "val", "test", "test"),
]


def votes(policy, seed=42):
    duckdb = pytest.importorskip("duckdb")
    v = axis_vote_sql(policy, seed)
    con = duckdb.connect()
    try:
        con.execute("CREATE TABLE r (tag VARCHAR, trb_cdr3 VARCHAR, tra_cdr3 VARCHAR, peptide VARCHAR, "
                    "pc BIGINT, ps VARCHAR, m1 VARCHAR, m2 VARCHAR)")
        con.executemany("INSERT INTO r VALUES (?, ?, ?, ?, ?, ?, ?, ?)", ROWS)
        got = con.execute(f"""
            SELECT k.tag, {v['s_trb']}, {v['s_tra']}, {v['s_pep']}, {v['s_m1']}, {v['s_m2']}
            FROM r k
            LEFT JOIN (SELECT tag, 'train' AS split, 1 AS cluster_id FROM r WHERE trb_cdr3 IS NOT NULL) jtrb USING (tag)
            LEFT JOIN (SELECT tag, 'train' AS split, 1 AS cluster_id FROM r WHERE tra_cdr3 IS NOT NULL) jtra USING (tag)
            LEFT JOIN (SELECT tag, ps AS split, pc AS cluster_id FROM r WHERE pc IS NOT NULL) jpep USING (tag)
            LEFT JOIN (SELECT tag, m1 AS split, 2 AS cluster_id FROM r WHERE m1 IS NOT NULL) jm1 USING (tag)
            LEFT JOIN (SELECT tag, m2 AS split, 3 AS cluster_id FROM r WHERE m2 IS NOT NULL) jm2 USING (tag)
        """).fetchall()
    finally:
        con.close()
    return {g[0]: g[1:] for g in got}


def test_v2_votes_in_sql_match_the_python_rule_on_every_axis():
    got = votes("receptor-hash-v2")
    for tag, trb, tra, pep, pc, _ps, _m1, _m2 in ROWS:
        assert got[tag] == (receptor_split(trb, "trb_cdr3", 42), receptor_split(tra, "tra_cdr3", 42),
                            peptide_split(pc, pep, 42), None, None), tag
    assert got["pep_in_cluster"][2] == got["pep_same_cluster"][2] == got["mhc_says_test"][2]
    assert got["pep_no_cluster"][2] is not None and got["pep_empty"][2] is None


def test_under_v2_the_mhc_pocket_cannot_hold_a_row_out():
    got = votes("receptor-hash-v2")
    assert combine(got["mhc_only"]) == "train"                      # no vote at all: train by default
    assert combine(got["mhc_says_test"]) == combine(got["pep_in_cluster"])


def test_v1_and_legacy_votes_are_unchanged_by_adding_v2():
    legacy, v1 = votes("cluster-v0"), votes("receptor-hash-v1")
    assert legacy["everything"] == ("train", "train", "val", "test", "test")
    assert legacy["mhc_only"] == (None, None, None, "test", "val")
    assert v1["everything"][2:] == ("val", "test", "test")          # peptide and MHC still by cluster
    assert v1["everything"][:2] == (receptor_split("CASSF", "trb_cdr3", 42), receptor_split("CAVR", "tra_cdr3", 42))


def test_peptide_split_sql_and_python_agree():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    try:
        cases = [(cid, pep) for cid in (None, -1, 0, 1, 7, 3948632, 16134210, 2 ** 40) for pep in ("GILGFVFTL", "", None)]
        for cid, pep in cases:
            got = con.execute(f"SELECT {peptide_split_sql('CAST($1 AS BIGINT)', 'CAST($2 AS VARCHAR)', 42)}",
                              [cid, pep]).fetchone()[0]
            assert got == peptide_split(cid, pep, 42), (cid, pep)
    finally:
        con.close()
