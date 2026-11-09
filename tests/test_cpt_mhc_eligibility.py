"""Regression tests for CPT MHC eligibility (corpus repair step 2).

Why this file exists. `cpt_dataset_builder.filter_where_sql()` rejects any populated MHC chain
longer than 320 aa (filters F5/F6). A classical class-I heavy-chain protein is ~362-366 aa, so
that blanket rule discards full-length class-I heavy chains as a class. Run05 retained only 168
class-I-labelled expanded rows across all partitions, while 14,902 of 15,145 downstream SFT train
molecules carry an `mhc_one` longer than 320 aa.

The intended fix is reference-aware eligibility: accept a populated chain whose normalized
sequence and chain role are supported by a pinned reference, and apply model context limits to
the tokenized example instead of to biological eligibility. These tests pin the behaviour that
fix must produce, and the first one FAILS against the current blanket rule.

Scope and honesty limits:
  * The sequences here are SYNTHETIC, generated to an exact length. They are not real IMGT/HLA
    protein sequences and assert nothing biological. What they test is the length and role logic.
  * Retention by this filter is eligibility only. It is not evidence about class-I exposure in a
    sampled corpus, and nothing here speaks to why arm D lost on the V04a read.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

duckdb = pytest.importorskip("duckdb", reason="eligibility filter is SQL; duckdb required")

from scripts.data_processing.cpt_dataset_builder import filter_where_sql, hla_class  # noqa: E402

# Classical class-I heavy-chain precursor lengths (leader included) sit at 362-366 aa.
CLASS_I_HEAVY_LEN = 365
CLASS_I_HEAVY_LEN_ALT = 366
# Mature class-II alpha/beta chains are well under the 320-aa cutoff.
CLASS_II_LEN = 255


def synthetic_chain(length: int) -> str:
    """A deterministic, clearly synthetic amino-acid string of exactly `length` residues."""
    motif = "GAVLIPFMWSTCYNQDEKRH"  # 20 valid residues, no X/U
    seq = (motif * (length // len(motif) + 1))[:length]
    assert len(seq) == length
    return seq


def surviving_rows(rows: list[dict]) -> set[str]:
    """Return the `tag`s of rows retained by the production eligibility filter."""
    con = duckdb.connect()
    try:
        con.execute(
            "CREATE TABLE t (tag VARCHAR, mhc_one VARCHAR, mhc_two VARCHAR, "
            "mhc_one_allele VARCHAR, mhc_two_allele VARCHAR)"
        )
        con.executemany(
            "INSERT INTO t VALUES (?, ?, ?, ?, ?)",
            [
                (
                    r["tag"],
                    r.get("mhc_one"),
                    r.get("mhc_two"),
                    r.get("mhc_one_allele"),
                    r.get("mhc_two_allele"),
                )
                for r in rows
            ],
        )
        got = con.execute(f"SELECT tag FROM t {filter_where_sql()}").fetchall()
        return {g[0] for g in got}
    finally:
        con.close()


@pytest.mark.parametrize("length", [CLASS_I_HEAVY_LEN, CLASS_I_HEAVY_LEN_ALT])
@pytest.mark.parametrize("allele", ["HLA-A*02:01", "HLA-B*07:02", "HLA-C*07:01"])
def test_full_length_class_i_heavy_chain_is_eligible(length, allele):
    """A role-consistent, full-length class-I heavy chain must survive eligibility.

    FAILS on the blanket 320-aa rule (F5), which is the regression this repair targets.
    A failure here means full-length class-I heavy chains are excluded as a class, so any
    class-I exposure the corpus reports can only come from truncated or atypical chains.
    """
    rows = [{"tag": "long_class_i", "mhc_one": synthetic_chain(length), "mhc_one_allele": allele}]
    assert surviving_rows(rows) == {"long_class_i"}, (
        f"{allele} heavy chain of {length} aa was rejected; "
        "length-only rejection of a role-consistent class-I chain"
    )


def test_short_class_i_and_class_ii_pair_remain_eligible():
    """Chains already under the cutoff keep passing: the repair must not narrow eligibility."""
    rows = [
        {"tag": "short_i", "mhc_one": synthetic_chain(275), "mhc_one_allele": "HLA-A*02:01"},
        {
            "tag": "class_ii_pair",
            "mhc_one": synthetic_chain(CLASS_II_LEN),
            "mhc_two": synthetic_chain(CLASS_II_LEN),
            "mhc_one_allele": "HLA-DRA1*01:01",
            "mhc_two_allele": "HLA-DRB1*15:01",
        },
    ]
    assert surviving_rows(rows) == {"short_i", "class_ii_pair"}


def test_swapped_roles_are_rejected():
    """Role conflicts must stay rejected: class I in the beta slot, class II beta in the heavy slot."""
    rows = [
        {"tag": "class_i_in_two", "mhc_two": synthetic_chain(275), "mhc_two_allele": "HLA-A*02:01"},
        {"tag": "ii_beta_in_one", "mhc_one": synthetic_chain(CLASS_II_LEN),
         "mhc_one_allele": "HLA-DRB1*15:01"},
    ]
    assert surviving_rows(rows) == set()


def test_unknown_and_null_allele_semantics_are_unchanged():
    """An unknown mhc_one allele is rejected (F4); a NULL allele is not treated as unknown."""
    rows = [
        {"tag": "unknown_allele", "mhc_one": synthetic_chain(275), "mhc_one_allele": "ZZZ9*99:99"},
        {"tag": "null_allele_with_seq", "mhc_one": synthetic_chain(275), "mhc_one_allele": None},
        {"tag": "all_null"},
    ]
    assert surviving_rows(rows) == {"null_allele_with_seq", "all_null"}


@pytest.mark.parametrize(
    "allele,expected",
    [
        ("HLA-A*02:01", "I"), ("HLA-B*07:02", "I"), ("HLA-C*07:01", "I"),
        ("HLA-DRA1*01:01", "II_alpha"), ("HLA-DRB1*15:01", "II_beta"),
        ("ZZZ9*99:99", "unknown"), ("", "null"),
    ],
)
def test_python_and_sql_class_calls_agree(allele, expected):
    """`hla_class()` and its SQL twin must not drift: the filter trusts both."""
    assert hla_class(allele) == expected
    con = duckdb.connect()
    try:
        from scripts.data_processing.cpt_dataset_builder import hsql
        sql_val = con.execute(
            f"SELECT ({hsql('a')}) FROM (SELECT ? AS a)", [allele or None]
        ).fetchone()[0]
    finally:
        con.close()
    if allele == "":
        return  # empty string arrives as NULL through the parameter binding; covered by the NULL test
    assert sql_val == expected, f"Python says {hla_class(allele)!r}, SQL says {sql_val!r}"
