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
    original_sha256: str
    normalized_sha256: str
    reference_version: str
    length: int


def normalize_sequence(seq: str) -> str:
    """Documented normalization only: strip surrounding/internal whitespace, upper-case."""
    return re.sub(r"\s+", "", seq or "").upper()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def locus_of(allele: str) -> str | None:
    """Leading locus token of an allele name, e.g. 'HLA-DRB1*15:01' -> 'DRB1'."""
    if not allele:
        return None
    m = _ALLELE_RE.match(allele.strip().upper().replace("_", "-"))
    return m.group(1) if m else None


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


def is_nonhuman(allele: str) -> bool:
    a = (allele or "").strip().upper()
    return any(a.startswith(p) for p in NONHUMAN_PREFIXES)


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
        original = sequence or ""
        norm = normalize_sequence(original)
        o_sha, n_sha = _sha(original), _sha(norm)

        def out(decision, reason, *, mhc_class=None, chain_role=None, classical=None,
                loci=(), alleles=(), ambiguous=False, ref_status="not_evaluated"):
            return ChainDecision(
                decision=decision, reason=reason, mhc_class=mhc_class, chain_role=chain_role,
                classical=classical, loci=tuple(loci), alleles=tuple(alleles),
                allele_ambiguous=ambiguous, reference_status=ref_status,
                original_sha256=o_sha, normalized_sha256=n_sha,
                reference_version=self.reference_version, length=len(norm),
            )

        if not norm:
            return out("quarantine", "empty_sequence")
        if set(norm) - VALID_RESIDUES:
            return out("quarantine", "invalid_residues")
        if allele and is_nonhuman(allele):
            # Needs its own pinned reference and an explicit scope decision.
            return out("quarantine", "nonhuman_out_of_scope")

        matched = self._by_hash.get(n_sha)
        if not matched:
            # Name-only evidence never asserts a class: that is how unvalidated rows became
            # "class-I exposure" before.
            return out("quarantine", "reference_absent", ref_status="absent")

        alleles = tuple(sorted(matched))
        loci = sorted({locus_of(a) for a in alleles} - {None})
        signatures = {class_role_of_locus(l)[:2] for l in loci}
        if len(signatures) != 1:
            return out("quarantine", "class_role_conflict", loci=loci, alleles=alleles,
                       ambiguous=len(alleles) > 1, ref_status="exact")

        mhc_class, chain_role = next(iter(signatures))
        classical = all(class_role_of_locus(l)[2] for l in loci)

        if allele:
            name_class, name_role, _ = class_role_of_locus(locus_of(allele))
            if name_class is not None and (name_class, name_role) != (mhc_class, chain_role):
                return out("quarantine", "name_sequence_conflict", loci=loci, alleles=alleles,
                           ambiguous=len(alleles) > 1, ref_status="exact")

        return out("accept", "reference_exact_match", mhc_class=mhc_class, chain_role=chain_role,
                   classical=classical, loci=loci, alleles=alleles,
                   ambiguous=len(alleles) > 1, ref_status="exact")

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
