"""Step 3: distinct-sequence MHC eligibility decisions (`quest.data.mhc_eligibility`).

One explicit decision per distinct chain sequence, from a pinned reference. The rules these
tests pin, and why each exists:

  * A reference-supported chain is eligible at any length. Length belongs to tokenization,
    not to biological eligibility (that conflation is the run05 defect).
  * A sequence may match several alleles. The index must report ALL compatible loci, never a
    single arbitrarily chosen two-field allele. `standardization.resolve_allele_to_sequence`
    does the opposite - it falls back through a prefix cache, so `A*02` silently returns the
    sequence of `A*02:01` - and this module must not inherit that behaviour.
  * An allele NAME is evidence, not proof. A name that disagrees with the sequence is a
    conflict and quarantines; it is never silently trusted over the sequence.
  * Non-classical HLA-E/F/G is kept separate from classical A/B/C coverage, so a corpus cannot
    report class-I exposure that is actually E/F/G.
  * Unknown, malformed and out-of-scope chains get distinct reason codes and quarantine. They
    must never be counted as validated class-I exposure.

Fixture sequences are SYNTHETIC but use realistic lengths (class-I heavy 362-366 aa, class-II
alpha/beta ~250 aa). They assert the decision logic, nothing biological.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quest.data.mhc_eligibility import (  # noqa: E402
    EligibilityIndex,
    build_index_from_fasta,
)

REAL_REFERENCE = ROOT / "data/raw_data/imgthla/hla_prot.fasta"


VALID = "ACDEFGHIKLMNPQRSTVWYXU"


def chain(length: int, salt: str = "A") -> str:
    """Deterministic synthetic residues of an exact length; distinct per `salt`.

    `salt` must itself be a valid residue, or the fixture lands in `invalid_residues` and the
    test measures the wrong branch. That mistake happened once; the assert keeps it caught.
    """
    assert salt in VALID, f"fixture salt {salt!r} is not a valid residue"
    motif = (salt + "GAVLIPFMWSTCYNQDEKRH")[:20]
    seq = (motif * (length // len(motif) + 1))[:length]
    assert not set(seq) - set(VALID)
    return seq


SEQ_A = chain(365, "A")      # class-I heavy, full length
SEQ_B = chain(362, "N")      # class-I heavy, full length
SEQ_C = chain(366, "C")      # class-I heavy, full length
SEQ_E = chain(358, "E")      # non-classical class I
SEQ_DRA = chain(254, "D")    # class-II alpha
SEQ_DRB = chain(266, "R")    # class-II beta
SEQ_SHARED = chain(364, "S")  # deliberately shared by two A alleles
SEQ_CROSS = chain(363, "X")   # deliberately shared across A and DRB1 -> conflict
SEQ_DQA2 = chain(255, "Q")   # in the reference, but a locus outside the class tables


@pytest.fixture
def index(tmp_path) -> EligibilityIndex:
    """A mini reference exercising every branch, written as IMGT-style FASTA."""
    records = [
        ("HLA:0001 A*02:01:01:01 365 bp", SEQ_A),
        ("HLA:0002 B*07:02:01:01 362 bp", SEQ_B),
        ("HLA:0003 C*07:01:01:01 366 bp", SEQ_C),
        ("HLA:0004 E*01:01:01:01 358 bp", SEQ_E),
        ("HLA:0005 DRA*01:01:01:01 254 bp", SEQ_DRA),
        ("HLA:0006 DRB1*15:01:01:01 266 bp", SEQ_DRB),
        ("HLA:0007 A*02:09:01:01 364 bp", SEQ_SHARED),
        ("HLA:0008 A*68:01:01:01 364 bp", SEQ_SHARED),
        ("HLA:0009 A*11:01:01:01 363 bp", SEQ_CROSS),
        ("HLA:0010 DRB1*04:01:01:01 363 bp", SEQ_CROSS),
        ("HLA:0011 DQA2*01:01:01:01 255 bp", SEQ_DQA2),
    ]
    fasta = tmp_path / "mini_prot.fasta"
    fasta.write_text("".join(f">{h}\n{s}\n" for h, s in records))
    return build_index_from_fasta(fasta, reference_version="MINI-TEST-1")


# ---------- the core repair: length is not an eligibility criterion ----------

@pytest.mark.parametrize("seq,locus", [(SEQ_A, "A"), (SEQ_B, "B"), (SEQ_C, "C")])
def test_full_length_classical_heavy_chain_is_accepted(index, seq, locus):
    """Full-length class-I heavy chains are eligible. Failure = the run05 defect persists."""
    d = index.decide(seq)
    assert d.decision == "accept"
    assert d.mhc_class == "I" and d.chain_role == "heavy"
    assert d.classical is True
    assert locus in d.loci
    assert d.reason == "reference_exact_match"


def test_class_ii_alpha_and_beta_roles_are_distinguished(index):
    """Alpha and beta must not collapse into one 'class II' bucket; slots depend on the role."""
    assert (index.decide(SEQ_DRA).mhc_class, index.decide(SEQ_DRA).chain_role) == ("II", "alpha")
    assert (index.decide(SEQ_DRB).mhc_class, index.decide(SEQ_DRB).chain_role) == ("II", "beta")


def test_non_classical_class_i_is_accepted_but_flagged_non_classical(index):
    """HLA-E is class I yet not classical A/B/C; a corpus must be able to exclude it from coverage."""
    d = index.decide(SEQ_E)
    assert d.decision == "accept" and d.mhc_class == "I"
    assert d.classical is False, "E/F/G counted as classical would inflate class-I coverage"


# ---------- all compatible alleles, never one arbitrary pick ----------

def test_shared_sequence_reports_every_compatible_allele(index):
    """Two alleles share a sequence: report both, mark ambiguous, still accept."""
    d = index.decide(SEQ_SHARED)
    assert d.decision == "accept"
    assert d.allele_ambiguous is True
    assert set(d.alleles) == {"A*02:09", "A*68:01"}, "must not collapse to one arbitrary allele"
    assert d.loci == ("A",)


def test_sequence_matching_conflicting_classes_is_quarantined(index):
    """One sequence mapping to both class I and class II cannot be assigned a role."""
    d = index.decide(SEQ_CROSS)
    assert d.decision == "quarantine"
    assert d.reason == "class_role_conflict"
    assert d.mhc_class is None and d.chain_role is None


# ---------- names are evidence, not proof ----------

def test_absent_allele_name_is_resolved_from_sequence(index):
    """A missing name is not a failure when the sequence itself resolves class and role."""
    d = index.decide(SEQ_A, allele=None)
    assert d.decision == "accept" and d.mhc_class == "I" and d.chain_role == "heavy"


def test_name_disagreeing_with_sequence_is_quarantined(index):
    """A class-II name on a class-I sequence is a conflict; the name is never trusted silently."""
    d = index.decide(SEQ_A, allele="HLA-DRB1*15:01")
    assert d.decision == "quarantine"
    assert d.reason == "name_sequence_conflict"


def test_name_only_match_without_reference_support_is_quarantined(index):
    """An A*-looking name with an unsupported sequence is NOT validated class-I exposure."""
    d = index.decide(chain(340, "Q"), allele="HLA-A*02:01")
    assert d.decision == "quarantine"
    assert d.reason == "reference_absent"
    assert d.mhc_class is None, "name-only evidence must not assert class"


def test_two_field_name_is_not_prefix_substituted(index):
    """`A*02` must not silently resolve to A*02:01's sequence (the standardization fallback bug)."""
    d = index.decide(chain(341, "Y"), allele="HLA-A*02")
    assert d.decision == "quarantine" and d.reason == "reference_absent"


# ---------- malformed and out-of-scope ----------

@pytest.mark.parametrize("bad,reason", [
    ("", "empty_sequence"),
    ("   ", "empty_sequence"),
    ("ACDEFG123", "invalid_residues"),
    ("ACDEFG*", "invalid_residues"),
])
def test_malformed_sequences_get_distinct_reasons(index, bad, reason):
    d = index.decide(bad)
    assert d.decision == "quarantine" and d.reason == reason


def test_nonhuman_allele_is_out_of_scope_not_invalid(index):
    """Mouse H-2 is neither human-eligible nor malformed; it needs its own pinned reference."""
    d = index.decide(chain(362, "M"), allele="H-2Kb")
    assert d.decision == "quarantine"
    assert d.reason == "nonhuman_out_of_scope"


# ---------- provenance ----------

def test_decision_carries_hashes_and_reference_version(index):
    """Every decision is traceable to the exact bytes judged and the reference that judged them."""
    d = index.decide(f"  {SEQ_A.lower()}  ")
    assert d.decision == "accept", "documented whitespace/case normalization must still match"
    assert d.normalized_sha256 == index.decide(SEQ_A).normalized_sha256
    assert d.original_sha256 != d.normalized_sha256
    assert d.reference_version == "MINI-TEST-1"
    assert len(d.normalized_sha256) == 64


def test_every_distinct_chain_gets_exactly_one_decision(index):
    """Step 3 acceptance: no chain is left undecided and no chain gets two decisions."""
    chains = [SEQ_A, SEQ_B, SEQ_C, SEQ_E, SEQ_DRA, SEQ_DRB, SEQ_SHARED, SEQ_CROSS,
              "", "ACDEFG123", chain(340, "Q")]
    out = index.decide_many(chains)
    assert len(out) == len(chains)
    assert all(d.decision in ("accept", "quarantine") for d in out)
    assert all(d.reason for d in out)


# ---------- real reference, when it is present ----------

@pytest.mark.skipif(not REAL_REFERENCE.exists(), reason="pinned IMGT/HLA FASTA not downloaded")
def test_real_reference_accepts_full_length_heavy_chains_the_old_rule_rejected():
    """On the pinned 3.58.0 reference, real >320-aa A/B/C chains must be eligible."""
    idx = build_index_from_fasta(REAL_REFERENCE, reference_version="IPD-IMGT/HLA 3.58.0")
    long_heavy = [s for s, d in idx.sample_by(mhc_class="I", classical=True, min_len=321, limit=25)]
    assert len(long_heavy) == 25, "reference must contain classical class-I chains over 320 aa"
    for seq in long_heavy:
        d = idx.decide(seq)
        assert d.decision == "accept" and d.chain_role == "heavy"


# ---------- real-world missing values ----------

@pytest.mark.parametrize("missing", [None, float("nan"), "", "   "])
def test_missing_sequence_values_are_treated_as_empty(index, missing):
    """Parquet/pandas represent a missing chain as NaN, which is a float AND truthy.

    `seq or ""` does not catch it, so an unguarded normalizer raises TypeError on real data
    instead of deciding. A missing chain must decide explicitly as empty_sequence.
    """
    d = index.decide(missing)
    assert d.decision == "quarantine" and d.reason == "empty_sequence"


@pytest.mark.parametrize("missing", [None, float("nan"), "", "   "])
def test_missing_allele_values_count_as_absent_evidence(index, missing):
    """A missing allele is absent evidence, and the sequence still decides the chain."""
    d = index.decide(SEQ_A, allele=missing)
    assert d.decision == "accept"
    assert d.allele_evidence == "absent"


# ---------- review fixes, 2026-11-09 ----------

def test_reference_locus_outside_the_class_tables_is_not_accepted(index):
    """DQA2, DMA, HFE, TAP1 ... are in hla_prot.fasta. An exact match used to come back
    'accept' with mhc_class=None and chain_role=None, i.e. counted as validated exposure."""
    d = index.decide(SEQ_DQA2)
    assert d.decision == "quarantine" and d.reason == "locus_out_of_scope"
    assert d.reference_status == "exact"
    assert d.mhc_class is None and d.chain_role is None


@pytest.mark.parametrize("ws", ["\u00a0", "\x0b", "\u2003"])
def test_normalization_strips_only_the_whitespace_sql_strips(index, ws):
    """DuckDB/RE2 `\\s` is [ \\t\\n\\f\\r]. If Python strips more, a pair is accepted here and then
    fails the SQL semi-join. Other whitespace must survive and decide as invalid_residues."""
    from quest.data.mhc_eligibility import normalize_sequence
    assert normalize_sequence(" a\tb\nc\r\fd ") == "ABCD"
    dirty = SEQ_A[:10] + ws + SEQ_A[10:]
    assert ws in normalize_sequence(dirty)
    assert index.decide(dirty).reason == "invalid_residues"


@pytest.mark.parametrize("name,expected", [
    ("HLA-A*02:01", ("A", ("02", "01"))), ("A*02:01:01:01", ("A", ("02", "01", "01", "01"))),
    ("HLA-A02:01", ("A", ("02", "01"))), ("A0201", ("A", ("02", "01"))), ("A2", ("A", ("02",))),
    ("HLA-A*0201", ("A", ("02", "01"))), ("A*01:04N", ("A", ("01", "04"))),
    ("DRB11501", ("DRB1", ("15", "01"))), ("HLA-DRB1_15:01", ("DRB1", ("15", "01"))),
    ("DPB1*104:01", ("DPB1", ("104", "01"))), ("HLA-DRA", ("DRA", ())),
    ("A*02:01P", ("A", None)), ("A020101", ("A", ("02", "01", "01"))),
    ("A*02010101", ("A", ("02", "01", "01", "01"))), ("HLA-A * 02:01", ("A", ("02", "01"))),
    ("hla-a*02:01", ("A", ("02", "01"))), ("A201", (None, None)), ("A*24-02", (None, None)),
    ("A*24:XX", (None, None)), ("A*24:02/HLA-B*07:02", (None, None)), ("A*24:02,A*02:01", (None, None)),
    ("B2M", (None, None)), ("MICA*001", ("MICA", ("001",))), ("", (None, None)),
    ("HLA-TAP1*01:01", ("TAP1", ("01", "01"))), ("HLA-DQA2*01:01", ("DQA2", ("01", "01"))),
])
def test_allele_names_parse_with_and_without_a_star(name, expected):
    from quest.data.mhc_eligibility import parse_allele_name
    assert parse_allele_name(name) == expected


@pytest.mark.parametrize("name", ["HLA-A02:01", "A0201", "A2", "A*02", "HLA-A*02:01:01", "A*02:01P"])
def test_starless_or_coarser_names_do_not_quarantine_an_exact_chain(index, name):
    """Punctuation or lower resolution is not a contradiction; these used to be 'malformed'."""
    d = index.decide(SEQ_A, allele=name)
    assert d.decision == "accept" and d.allele_evidence == "named"


@pytest.mark.parametrize("seq,name,reason", [
    (SEQ_A, "HLA-B*07:02", "name_locus_conflict"),      # same class and role, wrong locus
    (SEQ_A, "HLA-A*11:01", "name_allele_conflict"),     # right locus, a different protein
    (SEQ_A, "A1101", "name_allele_conflict"),
    (SEQ_DRB, "HLA-DRB1*04:01", "name_allele_conflict"),
])
def test_same_class_name_contradicting_the_sequence_is_quarantined(index, seq, name, reason):
    d = index.decide(seq, allele=name)
    assert d.decision == "quarantine" and d.reason == reason


def test_any_allele_sharing_the_sequence_satisfies_the_name(index):
    for name in ("HLA-A*02:09", "HLA-A*68:01"):
        assert index.decide(SEQ_SHARED, allele=name).decision == "accept"
    assert index.decide(SEQ_DRA, allele="HLA-DRA1*01:01").decision == "accept"


@pytest.mark.parametrize("name", ["A*24-02", "A*24.02", "A*24:02;01", "A*24:XX", "A*24:02 xx",
                                  "HLA-A*24:02/HLA-B*07:02", "A*24:02,A*02:01"])
def test_unreadable_allele_designation_is_malformed_not_a_locus_match(index, name):
    """Each of these states some allele other than A*02:01 in a spelling that cannot be parsed.
    Falling back to a locus-only check accepted all of them on the A*02:01 sequence."""
    d = index.decide(SEQ_A, allele=name)
    assert d.decision == "quarantine" and d.reason == "allele_field_malformed"


def test_readable_wrong_allele_with_stray_spaces_is_still_a_conflict(index):
    assert index.decide(SEQ_A, allele="HLA-A*24: 02").reason == "name_allele_conflict"


@pytest.mark.parametrize("spellings", [
    ["HLA-A*02:01", "HLA-A * 02:01", "hla-a*02:01", " HLA-A*02:01\t"],
    ["HLA-A*11:01", "HLA-A * 11:01"],
    ["H-2Kb", "h-2 kb"],
])
def test_decision_depends_only_on_the_normalized_allele(index, spellings):
    """The SQL filter joins on (normalized sequence, normalized allele), and
    `accepted_pairs_table` decides only the first spelling seen for each key. So every spelling
    with the same key must decide the same way, or the SQL result depends on input order."""
    from quest.data.mhc_eligibility import accepted_pairs_table, normalize_sequence
    assert len({normalize_sequence(a) for a in spellings}) == 1
    decisions = {(d.decision, d.reason) for d in (index.decide(SEQ_A, allele=a) for a in spellings)}
    assert len(decisions) == 1
    forward = accepted_pairs_table(index, [(SEQ_A, a) for a in spellings])
    backward = accepted_pairs_table(index, [(SEQ_A, a) for a in reversed(spellings)])
    assert forward == backward


@pytest.mark.parametrize("name", ["HLA-MICA*008:01", "HLA-DMA*01:01", "HLA-TAP1*01:01", "HLA-DQA2*01:01"])
def test_out_of_scope_locus_name_is_reported_as_such(index, name):
    """Measured in the source corpus: 155 chains named for MICA/MICB/DM/DO/DQA2/DQB2/TAP/HFE.
    They were quarantined as 'allele_field_malformed', which misdescribes a well-formed name."""
    d = index.decide(SEQ_A, allele=name)
    assert d.decision == "quarantine" and d.reason == "locus_out_of_scope"
    assert d.allele_evidence == "out_of_scope"
