"""Reference-aware MHC chain eligibility (CPT corpus repair, step 3).

One explicit decision per distinct chain sequence, judged against a pinned IPD-IMGT/HLA protein
reference. Built because `cpt_dataset_builder.filter_where_sql()` rejected any populated MHC
chain over 320 aa, which discards roughly two thirds of classical class-I heavy-chain alleles
(63.7% of A, 63.7% of B, 69.4% of C in release 3.58.0) and no class-II chain at all.

Design rules, each of which a test pins:

  * Eligibility is biological, and has no length limit. Model context limits apply to the
    tokenized example, separately, and are not decided here.
  * A sequence is matched by its own bytes. This module never resolves an allele NAME to a
    sequence, so it cannot inherit the prefix substitution in
    `standardization.resolve_allele_to_sequence`, where `A*02` silently returns A*02:01.
  * Every compatible allele is reported, not one arbitrary pick. Several alleles sharing one
    sequence is ambiguity to record, not a reason to reject.
  * An allele name is evidence. A name that contradicts the sequence quarantines the chain.
  * Non-classical HLA-E/F/G is class I but flagged `classical=False`, so class-I coverage
    reporting can exclude it.
  * Unknown, malformed and out-of-scope chains quarantine with distinct reason codes and are
    never counted as validated exposure.

A decision here is about one chain. Slot/role consistency for a row (which chain may occupy
`mhc_one` vs `mhc_two`) is step 4, in the builder.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

# Matches `quest.data.standardization.VALID_AA`; X is unknown-but-valid, U is selenocysteine.
VALID_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWYXU")

CLASS_I_CLASSICAL = frozenset({"A", "B", "C"})
CLASS_I_NON_CLASSICAL = frozenset({"E", "F", "G"})
CLASS_II_ALPHA = frozenset({"DRA", "DRA1", "DQA1", "DPA1"})
CLASS_II_BETA = frozenset({"DRB1", "DRB2", "DRB3", "DRB4", "DRB5", "DQB1", "DPB1"})

# Prefixes that denote a non-human MHC. Out of scope for a human reference: explicitly neither
# eligible nor malformed, because silently calling them invalid would hide real records.
NONHUMAN_PREFIXES = ("H-2", "H2-", "MAMU", "BOLA", "SLA-", "RT1", "GOGO", "PATR", "DLA-")

_ALLELE_RE = re.compile(r"^(?:HLA-)?([A-Z]+[0-9]?)\*?")
# A real locus token is short: 'A', 'C', 'DRB1', 'DQA1'. Without this bound, a protein sequence
# pasted into the allele column parses as a 250-character "locus", and the name/sequence conflict
# check then silently skips the row instead of quarantining it.
_LOCUS_TOKEN_RE = re.compile(r"^[A-Z]{1,3}[0-9]?$")
# Residue-only text of this length in an allele field is a sequence, not a name.
_SEQUENCE_IN_NAME_MIN_LEN = 25


@dataclass(frozen=True)
class ChainDecision:
    """The single explicit decision for one distinct chain sequence."""

    decision: str              # 'accept' | 'quarantine'
    reason: str                # stable reason code
    mhc_class: str | None      # 'I' | 'II' | None when undetermined
    chain_role: str | None     # 'heavy' | 'alpha' | 'beta' | None
    classical: bool | None     # True only for A/B/C; None when undetermined
    loci: tuple[str, ...]
    alleles: tuple[str, ...]
    allele_ambiguous: bool
    reference_status: str      # 'exact' | 'absent' | 'not_evaluated'
    allele_evidence: str       # 'named' | 'absent' | 'is_sequence' | 'nonhuman' | 'malformed'
    original_sha256: str
    normalized_sha256: str
    reference_version: str
    length: int


def as_text(value) -> str:
    """Coerce a cell to text, treating every missing representation as absent.

    Parquet and pandas represent a missing value as NaN, which is a float and is truthy, so
    `value or ""` silently lets it through and the caller raises TypeError on real data. A
    non-string is therefore absent, which then decides explicitly as `empty_sequence` or as
    `absent` allele evidence rather than crashing or being quietly accepted.
    """
    return value if isinstance(value, str) else ""


def normalize_sequence(seq) -> str:
    """Documented normalization only: strip surrounding/internal whitespace, upper-case."""
    return re.sub(r"\s+", "", as_text(seq)).upper()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def locus_of(allele: str) -> str | None:
    """Leading locus token of an allele name, e.g. 'HLA-DRB1*15:01' -> 'DRB1'."""
    allele = as_text(allele)
    if not allele:
        return None
    m = _ALLELE_RE.match(allele.strip().upper().replace("_", "-"))
    if not m:
        return None
    token = m.group(1)
    return token if _LOCUS_TOKEN_RE.match(token) else None


def class_role_of_locus(locus: str | None) -> tuple[str | None, str | None, bool | None]:
    """Map a locus to (mhc_class, chain_role, classical)."""
    if locus in CLASS_I_CLASSICAL:
        return "I", "heavy", True
    if locus in CLASS_I_NON_CLASSICAL:
        return "I", "heavy", False
    if locus in CLASS_II_ALPHA:
        return "II", "alpha", False
    if locus in CLASS_II_BETA:
        return "II", "beta", False
    return None, None, None


def is_nonhuman(allele) -> bool:
    a = as_text(allele).strip().upper()
    return any(a.startswith(p) for p in NONHUMAN_PREFIXES)


def allele_field_evidence(allele: str | None, normalized_sequence: str) -> str:
    """Classify what the allele field actually carries, rather than trusting it.

    'is_sequence' is the measured case where `mhc_one_allele` holds a byte-identical copy of
    `mhc_one` (234 of 18,010 benchmark molecules, 1.3%). Such a field carries zero allele
    information, so it is treated as ABSENT and the sequence decides; the legacy filter instead
    dropped those rows via F4 as an 'unknown' allele even when the sequence was reference-exact.
    It stays a distinct value so coverage reports can count it separately and it can never pass
    as validated exposure on name evidence.
    """
    allele = as_text(allele)
    if not allele.strip():
        return "absent"
    if is_nonhuman(allele):
        return "nonhuman"
    norm = normalize_sequence(allele)
    if len(norm) >= _SEQUENCE_IN_NAME_MIN_LEN and not (set(norm) - VALID_RESIDUES):
        # Residue-only text: a copy of this chain is absent-equivalent, any other sequence is junk.
        return "is_sequence" if norm == normalized_sequence else "malformed"
    locus = locus_of(allele)
    if locus is not None and class_role_of_locus(locus)[0] is not None:
        return "named"
    return "malformed"


def four_digit(allele: str) -> str:
    """Truncate an allele name to two fields, e.g. 'A*02:01:01:01' -> 'A*02:01'."""
    parts = allele.split(":")
    return ":".join(parts[:2]) if len(parts) > 2 else allele


class EligibilityIndex:
    """Maps a normalized chain sequence to every allele in the reference carrying that sequence."""

    def __init__(self, by_hash: dict[str, set[str]], sequences: dict[str, str],
                 reference_version: str):
        self._by_hash = by_hash
        self._sequences = sequences          # normalized_sha256 -> normalized sequence
        self.reference_version = reference_version

    def __len__(self) -> int:
        return len(self._by_hash)

    def decide(self, sequence: str, allele: str | None = None) -> ChainDecision:
        original = as_text(sequence)
        norm = normalize_sequence(original)
        o_sha, n_sha = _sha(original), _sha(norm)

        def out(decision, reason, *, mhc_class=None, chain_role=None, classical=None,
                loci=(), alleles=(), ambiguous=False, ref_status="not_evaluated",
                evidence="not_evaluated"):
            return ChainDecision(
                decision=decision, reason=reason, mhc_class=mhc_class, chain_role=chain_role,
                classical=classical, loci=tuple(loci), alleles=tuple(alleles),
                allele_ambiguous=ambiguous, reference_status=ref_status, allele_evidence=evidence,
                original_sha256=o_sha, normalized_sha256=n_sha,
                reference_version=self.reference_version, length=len(norm),
            )

        if not norm:
            return out("quarantine", "empty_sequence")
        if set(norm) - VALID_RESIDUES:
            return out("quarantine", "invalid_residues")
        evidence = allele_field_evidence(allele, norm)
        if evidence == "nonhuman":
            # Needs its own pinned reference and an explicit scope decision.
            return out("quarantine", "nonhuman_out_of_scope", evidence=evidence)
        if evidence == "malformed":
            # Previously such a name parsed as a bogus locus and skipped the conflict check.
            return out("quarantine", "allele_field_malformed", evidence=evidence)

        matched = self._by_hash.get(n_sha)
        if not matched:
            # Name-only evidence never asserts a class: that is how unvalidated rows became
            # "class-I exposure" before.
            return out("quarantine", "reference_absent", ref_status="absent", evidence=evidence)

        alleles = tuple(sorted(matched))
        loci = sorted({locus_of(a) for a in alleles} - {None})
        signatures = {class_role_of_locus(l)[:2] for l in loci}
        if len(signatures) != 1:
            return out("quarantine", "class_role_conflict", loci=loci, alleles=alleles,
                       ambiguous=len(alleles) > 1, ref_status="exact", evidence=evidence)

        mhc_class, chain_role = next(iter(signatures))
        classical = all(class_role_of_locus(l)[2] for l in loci)

        if evidence == "named":
            name_class, name_role, _ = class_role_of_locus(locus_of(allele))
            if (name_class, name_role) != (mhc_class, chain_role):
                return out("quarantine", "name_sequence_conflict", loci=loci, alleles=alleles,
                           ambiguous=len(alleles) > 1, ref_status="exact", evidence=evidence)

        return out("accept", "reference_exact_match", mhc_class=mhc_class, chain_role=chain_role,
                   classical=classical, loci=loci, alleles=alleles,
                   ambiguous=len(alleles) > 1, ref_status="exact", evidence=evidence)

    def decide_many(self, sequences) -> list[ChainDecision]:
        return [self.decide(s) for s in sequences]

    def sample_by(self, *, mhc_class=None, classical=None, min_len=0, limit=10):
        """Yield (sequence, decision) pairs matching a filter. For diagnostics and tests."""
        out = []
        for n_sha, seq in self._sequences.items():
            if len(seq) < min_len:
                continue
            d = self.decide(seq)
            if d.decision != "accept":
                continue
            if mhc_class is not None and d.mhc_class != mhc_class:
                continue
            if classical is not None and d.classical is not classical:
                continue
            out.append((seq, d))
            if len(out) >= limit:
                break
        return out


def build_index_from_fasta(path, reference_version: str) -> EligibilityIndex:
    """Build the index from an IMGT-style protein FASTA (`>HLA:ID allele length bp`)."""
    by_hash: dict[str, set[str]] = {}
    sequences: dict[str, str] = {}
    allele, buf = None, []

    def flush():
        if allele is None:
            return
        norm = normalize_sequence("".join(buf))
        if not norm:
            return
        h = _sha(norm)
        by_hash.setdefault(h, set()).add(four_digit(allele))
        sequences.setdefault(h, norm)

    with Path(path).open() as fh:
        for line in fh:
            if line.startswith(">"):
                flush()
                parts = line[1:].split()
                allele, buf = (parts[1] if len(parts) > 1 else None), []
            else:
                buf.append(line.strip())
    flush()
    return EligibilityIndex(by_hash, sequences, reference_version)


def accepted_pairs_table(index: EligibilityIndex, pairs) -> list[dict]:
    """Decide each distinct (sequence, allele) pair and return only the ACCEPTED ones.

    This is the whole contract between Python and SQL. The builder semi-joins source rows against
    these pairs, so no eligibility rule is expressed in SQL at all and the two cannot drift on the
    rules. Only `normalize_sequence` has a SQL twin, and a test pins their agreement.

    Keyed on (sequence, allele) rather than sequence alone because one sequence carrying two
    different allele names can legitimately decide differently: one name may agree with the
    reference and the other conflict with it.
    """
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for sequence, allele in pairs:
        s_norm = normalize_sequence(sequence or "")
        a_norm = normalize_sequence(allele or "")
        if not s_norm or (s_norm, a_norm) in seen:
            continue
        seen.add((s_norm, a_norm))
        d = index.decide(sequence or "", allele)
        if d.decision != "accept":
            continue
        rows.append({
            "seq_norm": s_norm, "allele_norm": a_norm, "chain_role": d.chain_role,
            "mhc_class": d.mhc_class, "classical": d.classical,
            "allele_evidence": d.allele_evidence, "reason": d.reason,
            "alleles": ";".join(d.alleles), "allele_ambiguous": d.allele_ambiguous,
            "normalized_sha256": d.normalized_sha256,
            "reference_version": d.reference_version,
        })
    return rows
