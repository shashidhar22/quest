"""Regression tests for CPT MHC eligibility in the builder (corpus repair steps 2 and 4).

Why this file exists. `cpt_dataset_builder.filter_where_sql()` rejected any populated MHC chain
longer than 320 aa (filters F5/F6). Measured against pinned IPD-IMGT/HLA 3.58.0 that discards
63.7% of A, 63.7% of B and 69.4% of C alleles and 0.0% of class-II chains, so it was a
class-I-specific exclusion. Run05 retained only 168 class-I-labelled expanded rows.

The repair: eligibility is decided per distinct (sequence, allele) pair in Python by
`quest.data.mhc_eligibility`, and SQL only semi-joins against the ACCEPTED pairs. No decision
rule is expressed twice, so Python and SQL cannot drift on the rules; only the normalization
expressions have to agree, and `test_sql_and_python_normalization_agree` pins that.

`policy` is required and has no default. In this codebase `--loss_weighting` defaulted to `row`,
and that silent default is why every launcher must now pass the flag explicitly; an un-migrated
call site must fail loudly instead of silently choosing an eligibility policy.

Fixture sequences are SYNTHETIC at realistic lengths. They assert the filter's logic, not biology.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

duckdb = pytest.importorskip("duckdb", reason="eligibility filter is SQL; duckdb required")

from quest.data.mhc_eligibility import (  # noqa: E402
    accepted_pairs_table,
    build_index_from_fasta,
    normalize_sequence,
)
from scripts.data_processing.cpt_dataset_builder import (  # noqa: E402
    LEGACY_POLICY,
    REFERENCE_AWARE_POLICY,
    filter_where_sql,
)

CLASS_I_HEAVY_LEN = 365        # classical heavy-chain precursor lengths are 362-366 aa
CLASS_I_HEAVY_LEN_ALT = 366
CLASS_II_LEN = 255


def synthetic_chain(length: int) -> str:
    motif = "GAVLIPFMWSTCYNQDEKRH"
    seq = (motif * (length // len(motif) + 1))[:length]
    assert len(seq) == length and not set(seq) - set(motif)
    return seq


SEQ_365 = synthetic_chain(CLASS_I_HEAVY_LEN)
SEQ_366 = synthetic_chain(CLASS_I_HEAVY_LEN_ALT)
SEQ_SHORT_I = synthetic_chain(275)
SEQ_II_A = synthetic_chain(CLASS_II_LEN)
SEQ_II_B = synthetic_chain(266)
SEQ_II_DQB = synthetic_chain(261)
SEQ_UNSUPPORTED = synthetic_chain(301)   # deliberately absent from the mini reference


@pytest.fixture(scope="module")
def index():
    """A mini reference. The two long sequences are shared across A, B and C on purpose, so a
    single sequence carries several compatible alleles without any class/role conflict."""
    records = [
        ("HLA:1 A*02:01:01:01 365 bp", SEQ_365), ("HLA:2 B*07:02:01:01 365 bp", SEQ_365),
        ("HLA:3 C*07:01:01:01 365 bp", SEQ_365), ("HLA:4 A*02:01:01:02 366 bp", SEQ_366),
        ("HLA:5 B*07:02:01:02 366 bp", SEQ_366), ("HLA:6 C*07:01:01:02 366 bp", SEQ_366),
        ("HLA:7 A*11:01:01:01 275 bp", SEQ_SHORT_I),
        ("HLA:8 DRA*01:01:01:01 255 bp", SEQ_II_A),
        ("HLA:9 DRB1*15:01:01:01 266 bp", SEQ_II_B),
        ("HLA:10 DQB1*06:02:01:01 261 bp", SEQ_II_DQB),
    ]
    tmp = Path(__file__).resolve().parent / "fixtures" / "_mini_hla_prot.fasta"
    tmp.parent.mkdir(exist_ok=True)
    tmp.write_text("".join(f">{h}\n{s}\n" for h, s in records))
    try:
        yield build_index_from_fasta(tmp, reference_version="MINI-TEST-1")
    finally:
        tmp.unlink(missing_ok=True)


def surviving_rows(rows, index, policy=REFERENCE_AWARE_POLICY) -> set[str]:
    """Tags of rows retained by the production filter under `policy`."""
    con = duckdb.connect()
    try:
        con.execute("CREATE TABLE t (tag VARCHAR, mhc_one VARCHAR, mhc_two VARCHAR, "
                    "mhc_one_allele VARCHAR, mhc_two_allele VARCHAR)")
        con.executemany("INSERT INTO t VALUES (?, ?, ?, ?, ?)", [
            (r["tag"], r.get("mhc_one"), r.get("mhc_two"),
             r.get("mhc_one_allele"), r.get("mhc_two_allele")) for r in rows])
        kwargs = {}
        if policy == REFERENCE_AWARE_POLICY:
            pairs = []
            for r in rows:
                pairs.append((r.get("mhc_one"), r.get("mhc_one_allele")))
                pairs.append((r.get("mhc_two"), r.get("mhc_two_allele")))
            con.execute("CREATE TABLE elig (seq_norm VARCHAR, allele_norm VARCHAR, "
                        "chain_role VARCHAR, pair_group VARCHAR)")
            table = accepted_pairs_table(index, pairs)
            if table:
                con.executemany("INSERT INTO elig VALUES (?, ?, ?, ?)",
                                [(r["seq_norm"], r["allele_norm"], r["chain_role"],
                                  r["pair_group"]) for r in table])
            kwargs["eligibility_table"] = "elig"
        got = con.execute(f"SELECT tag FROM t {filter_where_sql(policy, **kwargs)}").fetchall()
        return {g[0] for g in got}
    finally:
        con.close()


# ---------- the core repair ----------

@pytest.mark.parametrize("length", [CLASS_I_HEAVY_LEN, CLASS_I_HEAVY_LEN_ALT])
@pytest.mark.parametrize("allele", ["HLA-A*02:01", "HLA-B*07:02", "HLA-C*07:01"])
def test_full_length_class_i_heavy_chain_is_eligible(index, length, allele):
    """A reference-supported, role-consistent full-length class-I heavy chain must survive.

    A failure means full-length class-I heavy chains are still excluded as a class, so any
    class-I exposure the corpus reports could only come from truncated or atypical chains.
    """
    rows = [{"tag": "long_class_i", "mhc_one": synthetic_chain(length), "mhc_one_allele": allele}]
    assert surviving_rows(rows, index) == {"long_class_i"}, (
        f"{allele} heavy chain of {length} aa was rejected under the reference-aware policy")


def test_legacy_policy_still_reproduces_the_320_rejection(index):
    """Reproduction must stay possible: the legacy policy keeps rejecting the same chain."""
    rows = [{"tag": "long_class_i", "mhc_one": SEQ_365, "mhc_one_allele": "HLA-A*02:01"}]
    assert surviving_rows(rows, index, policy=LEGACY_POLICY) == set()


def test_policy_is_required():
    """An un-migrated call site must fail loudly rather than pick a policy silently."""
    with pytest.raises(TypeError):
        filter_where_sql()
    with pytest.raises(ValueError):
        filter_where_sql("no-such-policy")


def test_reference_aware_policy_requires_an_eligibility_table():
    with pytest.raises(ValueError):
        filter_where_sql(REFERENCE_AWARE_POLICY)


# ---------- behaviour that must not narrow ----------

def test_short_class_i_and_class_ii_pair_remain_eligible(index):
    rows = [
        {"tag": "short_i", "mhc_one": SEQ_SHORT_I, "mhc_one_allele": "HLA-A*11:01"},
        {"tag": "class_ii_pair", "mhc_one": SEQ_II_A, "mhc_two": SEQ_II_B,
         "mhc_one_allele": "HLA-DRA*01:01", "mhc_two_allele": "HLA-DRB1*15:01"},
    ]
    assert surviving_rows(rows, index) == {"short_i", "class_ii_pair"}


def test_rows_without_any_mhc_are_untouched(index):
    """The corpus is 95%+ receptor-only rows; eligibility must not touch them."""
    assert surviving_rows([{"tag": "all_null"}], index) == {"all_null"}
    assert surviving_rows([{"tag": "empty_str", "mhc_one": "", "mhc_two": ""}], index) == {"empty_str"}


# ---------- role and slot consistency ----------

def test_swapped_roles_are_rejected(index):
    """Class-I heavy may not occupy mhc_two, and class-II beta may not occupy mhc_one.

    Measured: every populated mhc_two in the benchmark train split is class-II beta (23 distinct),
    so there is no legitimate class-I second chain to accommodate here.
    """
    rows = [
        {"tag": "heavy_in_two", "mhc_two": SEQ_365, "mhc_two_allele": "HLA-A*02:01"},
        {"tag": "ii_beta_in_one", "mhc_one": SEQ_II_B, "mhc_one_allele": "HLA-DRB1*15:01"},
    ]
    assert surviving_rows(rows, index) == set()


def test_class_ii_alpha_is_allowed_in_mhc_one(index):
    """`split_mhc_to_alpha_beta` routes class-II alpha to mhc_one by design, so it is eligible."""
    rows = [{"tag": "ii_alpha", "mhc_one": SEQ_II_A, "mhc_one_allele": "HLA-DRA*01:01"}]
    assert surviving_rows(rows, index) == {"ii_alpha"}


# ---------- the new policy's stricter half ----------

def test_sequence_absent_from_the_reference_is_rejected(index):
    """Unlike the legacy filter, an unsupported sequence is rejected whatever its allele says.

    Under the legacy policy a NULL-allele row passed on no evidence at all; that is how
    unvalidated chains could be counted as exposure.
    """
    rows = [
        {"tag": "unsupported_named", "mhc_one": SEQ_UNSUPPORTED, "mhc_one_allele": "HLA-A*02:01"},
        {"tag": "unsupported_null_allele", "mhc_one": SEQ_UNSUPPORTED},
    ]
    assert surviving_rows(rows, index) == set()


def test_name_disagreeing_with_sequence_is_rejected(index):
    rows = [{"tag": "conflict", "mhc_one": SEQ_365, "mhc_one_allele": "HLA-DRB1*15:01"}]
    assert surviving_rows(rows, index) == set()


def test_allele_field_holding_a_copy_of_the_sequence_still_passes(index):
    """Decision: such a field carries zero allele information, so it is ABSENT, not conflicting.

    Measured in the benchmark: 234/18,010 molecules (1.3%) have mhc_one_allele byte-identical to
    mhc_one, and the legacy filter dropped them via F4 as 'unknown allele' even though the
    sequence is reference-exact.
    """
    rows = [{"tag": "allele_is_seq", "mhc_one": SEQ_365, "mhc_one_allele": SEQ_365}]
    assert surviving_rows(rows, index) == {"allele_is_seq"}


def test_legacy_policy_dropped_the_copied_allele_row(index):
    """Pins the defect the decision above recovers: F4 rejected it as an unknown allele."""
    rows = [{"tag": "allele_is_seq", "mhc_one": SEQ_SHORT_I, "mhc_one_allele": SEQ_SHORT_I}]
    assert surviving_rows(rows, index, policy=LEGACY_POLICY) == set()


# ---------- the one thing expressed in both languages ----------

@pytest.mark.parametrize("raw", [
    "ACDEFG", "  ACDEFG  ", "acdefg", "ACD EFG", "ACD\nEFG", "ACD\tEFG\r\n", "",
])
def test_sql_and_python_normalization_agree(raw):
    """Only normalization is duplicated across Python and SQL, so it is pinned directly.

    A divergence here would silently drop or admit chains, since the semi-join is keyed on it.
    """
    from scripts.data_processing.cpt_dataset_builder import normalize_sequence_sql
    con = duckdb.connect()
    try:
        got = con.execute(f"SELECT {normalize_sequence_sql('?')}", [raw]).fetchone()[0]
    finally:
        con.close()
    assert got == normalize_sequence(raw), f"SQL {got!r} vs Python {normalize_sequence(raw)!r}"


# ---------- review fixes, 2026-11-09 ----------

PAIR_ROWS = [
    {"tag": "dr_pair", "mhc_one": SEQ_II_A, "mhc_two": SEQ_II_B,
     "mhc_one_allele": "HLA-DRA*01:01", "mhc_two_allele": "HLA-DRB1*15:01"},
    {"tag": "class_i_with_class_ii_beta", "mhc_one": SEQ_SHORT_I, "mhc_two": SEQ_II_B,
     "mhc_one_allele": "HLA-A*11:01", "mhc_two_allele": "HLA-DRB1*15:01"},
    {"tag": "dr_alpha_with_dq_beta", "mhc_one": SEQ_II_A, "mhc_two": SEQ_II_DQB,
     "mhc_one_allele": "HLA-DRA*01:01", "mhc_two_allele": "HLA-DQB1*06:02"},
    {"tag": "dq_beta_alone", "mhc_two": SEQ_II_DQB, "mhc_two_allele": "HLA-DQB1*06:02"},
    {"tag": "same_class_wrong_locus", "mhc_one": SEQ_SHORT_I, "mhc_one_allele": "HLA-B*07:02"},
    {"tag": "starless_name", "mhc_one": SEQ_SHORT_I, "mhc_one_allele": "HLA-A11:01"},
    {"tag": "nbsp_in_chain", "mhc_one": SEQ_SHORT_I[:9] + "\u00a0" + SEQ_SHORT_I[9:]},
]


def test_two_chains_in_one_row_must_belong_together(index):
    """Each chain passing on its own is not enough: a class-I heavy chain beside a class-II beta,
    or a DR alpha beside a DQ beta, is not a molecule and used to pass both slot checks."""
    assert surviving_rows(PAIR_ROWS[:4], index) == {"dr_pair", "dq_beta_alone"}


def test_python_row_rule_agrees_with_the_sql_filter(index):
    """The verifier re-decides rows with `row_decision`; it must never disagree with the SQL."""
    from quest.data.mhc_eligibility import row_decision
    python = {r["tag"] for r in PAIR_ROWS if row_decision(
        index, r.get("mhc_one"), r.get("mhc_one_allele"),
        r.get("mhc_two"), r.get("mhc_two_allele"))[0]}
    assert python == surviving_rows(PAIR_ROWS, index) == {"dr_pair", "dq_beta_alone",
                                                          "starless_name"}
