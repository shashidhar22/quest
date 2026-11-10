"""The single definition of example-level sampling (corpus repair, steps 9-10).

The sizing scripts that produced the sampling contract and the builder that must meet it both
import `quest.data.contract_sampling`; these tests pin its rules and the agreement of every SQL
builder with its Python twin.
"""
from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quest.data.contract_sampling import (  # noqa: E402
    NO_PEPTIDE,
    STRATA,
    TCR_PEPTIDE_STRATA,
    choose_order_variant,
    example_hash,
    example_hash_sql,
    order_variant_key_sql,
    peptide_rank_sql,
    rule_cap_and_count,
    select_examples,
    selection_sql,
    stratum_of,
    stratum_sql,
)


def test_strata_are_the_contracts_nine_and_the_capped_ones_are_the_tcr_peptide_ones():
    assert len(STRATA) == len(set(STRATA)) == 9
    assert [s[:2] for s in STRATA] == [f"s{i}" for i in range(1, 10)]
    assert TCR_PEPTIDE_STRATA == ("s1_paired_tcr_peptide_classI", "s2_one_chain_tcr_peptide_classI",
                                  "s7_tcr_peptide_no_mhc")


@pytest.mark.parametrize("row, expected", [
    (("I", True, "PEP", "A", "B", None), "s1_paired_tcr_peptide_classI"),
    (("I", True, "PEP", None, "B", None), "s2_one_chain_tcr_peptide_classI"),
    (("I", True, "PEP", "A", None, None), "s2_one_chain_tcr_peptide_classI"),
    (("I", True, "PEP", None, None, None), "s3_peptide_classI_no_tcr"),
    (("I", True, None, "A", "B", None), "s4_classI_no_peptide"),
    (("I", False, "PEP", "A", "B", None), "s5_nonclassical_classI"),
    (("II", False, "PEP", "A", "B", "BETA"), "s6_classII"),
    ((None, None, "PEP", "A", "B", "BETA"), "s6_classII"),     # a beta chain alone is class II
    ((None, None, "PEP", None, "B", None), "s7_tcr_peptide_no_mhc"),
    ((None, None, "PEP", None, None, None), "s8_peptide_only"),
    ((None, None, None, "A", "B", None), "s9_paired_receptors_only"),
])
def test_stratum_rules(row, expected):
    assert stratum_of(*row) == expected


def test_an_undetermined_class_is_never_counted_as_class_one():
    # classical=None (class undetermined) must not satisfy "classical class I".
    assert stratum_of("I", None, "PEP", "A", "B", None) == "s5_nonclassical_classI"
    assert stratum_of(None, True, "PEP", "A", "B", None) == "s7_tcr_peptide_no_mhc"


def test_missing_components_keep_their_position_in_the_hash():
    # The defect this replaces: concat_ws skips NULLs, so these two examples shared a hash.
    assert example_hash("CASSL", None, None, None, None) != example_hash(None, "CASSL", None, None, None)
    assert example_hash("A", "B", None, None, None) != example_hash("A", None, "B", None, None)
    assert example_hash("A", "B", "P", "M", None) == example_hash("A", "B", "P", "M", None)


def _examples(n=400, seed=7):
    rng = random.Random(seed)
    peptides = [None, "PEPA", "PEPB", "PEPC"]
    out = []
    for i in range(n):
        pep = rng.choices(peptides, weights=[1, 12, 4, 1])[0]
        out.append((example_hash(f"A{i}", None, pep, None, None), pep, i))
    return out


def test_the_cap_is_applied_before_the_count():
    ex = _examples()
    capped = select_examples(ex, 5, None)
    per_pep = {}
    for _h, pep, _i in capped:
        per_pep[pep] = per_pep.get(pep, 0) + 1
    assert max(per_pep.values()) == 5 and set(per_pep) == {None, "PEPA", "PEPB", "PEPC"}
    both = select_examples(ex, 5, 12)
    assert both == capped[:12] and len(both) == 12
    # Count first and cap second would be a different, smaller selection.
    assert both != select_examples(select_examples(ex, None, 12), 5, None)


def test_selection_takes_the_lowest_hashes_and_never_repeats_an_example():
    ex = _examples()
    chosen = select_examples(ex, None, 50)
    assert [e[0] for e in chosen] == sorted(e[0] for e in ex)[:50]
    assert len({e[0] for e in select_examples(ex, 3, None)}) == len(select_examples(ex, 3, None))
    assert select_examples(ex, None, None) == sorted(ex)
    assert select_examples(list(reversed(ex)), 4, 9) == select_examples(ex, 4, 9)   # input order is irrelevant


def test_examples_without_a_peptide_are_capped_as_one_group():
    ex = [(example_hash(f"A{i}", None, None, None, None), None, i) for i in range(20)]
    assert len(select_examples(ex, 3, None)) == 3


@pytest.mark.parametrize("rule, expected", [
    ("all", (None, None)),
    ({"cap_per_peptide": 1000}, (1000, None)),
    ({"examples": 60000}, (None, 60000)),
    ({"cap_per_peptide": 1, "examples": 60000}, (1, 60000)),
])
def test_rule_parsing(rule, expected):
    assert rule_cap_and_count(rule) == expected


@pytest.mark.parametrize("rule", [{}, {"fill": True}, {"examples": 5, "cap": 2}, "everything", None])
def test_an_unknown_rule_is_refused_not_read_as_take_everything(rule):
    with pytest.raises((ValueError, TypeError)):
        rule_cap_and_count(rule)


def test_one_order_variant_is_chosen_and_the_choice_depends_on_seed_and_example():
    keys = ["tra_trb_peptide", "trb_tra_peptide", "peptide_tra_trb", "peptide_trb_tra"]
    h = example_hash("A", "B", "P", None, None)
    assert choose_order_variant(42, h, keys) == choose_order_variant(42, h, reversed(keys))
    assert choose_order_variant(42, h, keys) in keys
    picks = {choose_order_variant(42, example_hash(f"A{i}", "B", "P", None, None), keys) for i in range(200)}
    assert picks == set(keys)                      # not always the same permutation
    assert any(choose_order_variant(s, h, keys) != choose_order_variant(42, h, keys) for s in range(43, 60))
    with pytest.raises(ValueError):
        choose_order_variant(42, h, [])


# ---------- SQL twins ----------

def _con():
    duckdb = pytest.importorskip("duckdb", reason="the builder's sampling is SQL; duckdb required")
    return duckdb.connect()


def test_stratum_sql_and_python_agree_on_every_combination():
    con = _con()
    rows = list(itertools.product(["I", "II", None], [True, False, None], ["PEP", None],
                                  ["A", None], ["B", None], ["BETA", None]))
    con.execute("CREATE TABLE t (mhc_class VARCHAR, classical BOOLEAN, peptide VARCHAR, "
                "tra VARCHAR, trb VARCHAR, mhc_two VARCHAR)")
    con.executemany("INSERT INTO t VALUES (?, ?, ?, ?, ?, ?)", rows)
    expr = stratum_sql(mhc_class="mhc_class", classical="classical", peptide="peptide",
                       tra="tra", trb="trb", mhc_two="mhc_two")
    got = con.execute(f"SELECT mhc_class, classical, peptide, tra, trb, mhc_two, {expr} FROM t").fetchall()
    assert len(got) == len(rows) == 144
    assert all(stratum_of(*r[:6]) == r[6] for r in got)
    assert {r[6] for r in got} == set(STRATA)       # every stratum is reachable


def test_example_hash_sql_and_python_agree_including_nulls():
    con = _con()
    rows = list(itertools.product(["CAVR", None], ["CASSL", None], ["PEP", None], ["MHCA", None], ["MHCB", None]))
    con.execute("CREATE TABLE t (tra_cdr3 VARCHAR, trb_cdr3 VARCHAR, peptide VARCHAR, mhc_one VARCHAR, mhc_two VARCHAR)")
    con.executemany("INSERT INTO t VALUES (?, ?, ?, ?, ?)", rows)
    got = con.execute(f"SELECT *, {example_hash_sql()} FROM t").fetchall()
    assert all(example_hash(*r[:5]) == r[5] for r in got)
    assert len({r[5] for r in got}) == len(rows) == 32


@pytest.mark.parametrize("cap, n", [(None, None), (5, None), (None, 37), (5, 12), (1, 3), (1000, 400)])
def test_selection_sql_and_python_agree(cap, n):
    con = _con()
    ex = _examples()
    con.execute("CREATE TABLE ex (stratum VARCHAR, split VARCHAR, ex_hash VARCHAR, peptide VARCHAR, i INT)")
    con.executemany("INSERT INTO ex VALUES ('s7', 'train', ?, ?, ?)", ex)
    con.executemany("INSERT INTO ex VALUES ('s7', 'val', ?, ?, ?)", ex[:50])      # another split must not leak in
    con.execute(f"CREATE TABLE ranked AS SELECT *, {peptide_rank_sql()} AS rn_pep FROM ex")
    got = con.execute(f"SELECT ex_hash, peptide, i FROM ({selection_sql('ranked', cap, n)}) ORDER BY ex_hash",
                      ["s7", "train"]).fetchall()
    assert got == select_examples(ex, cap, n)


def test_order_variant_sql_and_python_agree():
    con = _con()
    keys = ["tra_trb_peptide", "trb_tra_peptide", "peptide_tra_trb", "peptide_trb_tra"]
    hashes = [example_hash(f"A{i}", "B", "P", None, None) for i in range(50)]
    con.execute("CREATE TABLE v (ex_hash VARCHAR, order_key VARCHAR)")
    con.executemany("INSERT INTO v VALUES (?, ?)", [(h, k) for h in hashes for k in keys])
    got = dict(con.execute(f"SELECT ex_hash, arg_min(order_key, {order_variant_key_sql(42)}) FROM v GROUP BY 1").fetchall())
    assert got == {h: choose_order_variant(42, h, keys) for h in hashes}


def test_the_no_peptide_sentinel_cannot_be_a_peptide():
    assert not NO_PEPTIDE.isalpha()
