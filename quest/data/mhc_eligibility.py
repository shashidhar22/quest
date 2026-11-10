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
  * A chain that exactly matches a PARTIAL reference entry is incomplete, not eligible. The
    reference holds many alleles only as fragments (exons 2-3, 181 aa, is the commonest class-I
    case). "Partial" is relative to the full-length entries of the chain's own locus in the same
    reference, so it is not a fixed length limit and a 365-aa chain is never affected.

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

# HLA-region genes present in the pinned reference that are not peptide-presenting class I/II
# chains. Recognized so that a name such as 'HLA-MICA*008:01' is reported as out of scope rather
# than as a malformed name. Measured in the source corpus: 155 distinct chains across these loci.
OUT_OF_SCOPE_LOCI = frozenset({"DMA", "DMB", "DOA", "DOB", "DQA2", "DQB2", "HFE",
                               "MICA", "MICB", "TAP1", "TAP2"})

# Prefixes that denote a non-human MHC. Out of scope for a human reference: explicitly neither
# eligible nor malformed, because silently calling them invalid would hide real records.
NONHUMAN_PREFIXES = ("H-2", "H2-", "MAMU", "BOLA", "SLA-", "RT1", "GOGO", "PATR", "DLA-")

# Slots, as measured in the benchmark rather than assumed: every populated mhc_two is class-II beta
# (23 distinct sequences in the train split), so there is no class-I second chain to accommodate.
# No beta-2-microglobulin is fabricated and none is required for a heavy-chain-only example.
MHC_ONE_ROLES = ("heavy", "alpha")
MHC_TWO_ROLES = ("beta",)

# Two spellings of one locus; compared as equal when a name is checked against the reference.
_LOCUS_ALIASES = {"DRA1": "DRA"}
# Expression-status suffixes ('A*01:04N'). They do not change which protein is named.
_EXPRESSION_SUFFIXES = "NLSCAQ"
# A real locus token is short: 'A', 'C', 'DRB1', 'DQA1'. Without this bound, a protein sequence
# pasted into the allele column parses as a 250-character "locus", and the name/sequence conflict
# check then silently skips the row instead of quarantining it.
_LOCUS_TOKEN_RE = re.compile(r"^[A-Z]{1,3}[0-9]?$")
# A chain is partial when it is shorter than this fraction of its locus's full length. Measured on
# IPD-IMGT/HLA 3.58.0, reference lengths are strongly bimodal: natural variants cluster within a
# few residues of full length (A 364/365, G 333/338, DQB1 257/261, DPA1 254/260, all >= 97.7%)
# and fragments at 181, 273, 337 and C 349 aa (<= 95.4%). The cut is a judgement, not a gap: about
# twenty reference sequences lie between 95.4% and 97.7% (for example A 354 and C 355 fall just
# under it, A 356 and C 356 just over), and for those the choice of 0.97 decides the outcome.
PARTIAL_FRACTION = 0.97
# The full length of a locus is the commonest length among its entries at least this close to the
# longest one; the commonest length overall can be a fragment (DRB1: 89 aa, exon 2 alone).
_FULL_LENGTH_BAND = 0.90

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
    reference_status: str      # 'exact' | 'partial' | 'absent' | 'not_evaluated'
    allele_evidence: str       # 'named' | 'absent' | 'is_sequence' | 'nonhuman' |
    #                            'out_of_scope' | 'malformed'
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
    """Documented normalization only: strip surrounding/internal whitespace, upper-case.

    The class is spelled out to equal DuckDB/RE2 `\\s` in `normalize_sequence_sql`. Python's
    `\\s` also strips Unicode whitespace (NBSP, vertical tab), which the SQL side keeps, so an
    accepted pair would then fail the semi-join. Any other whitespace now stays in the text and
    quarantines as `invalid_residues` on both sides.
    """
    return re.sub(r"[ \t\n\f\r]+", "", as_text(seq)).upper()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def parse_allele_name(allele) -> tuple[str | None, tuple[str, ...] | None]:
    """Split an allele name into (locus, fields).

    'HLA-DRB1*15:01:01' -> ('DRB1', ('15', '01', '01')). Star-less spellings are read too, because
    they are common in source data and a reference-exact chain must not be quarantined over
    punctuation: 'HLA-A02:01', 'A0201' -> ('A', ('02', '01')); 'A2' -> ('A', ('02',));
    'DRB11501' -> ('DRB1', ('15', '01')).

    `fields` is () for a bare locus, and None only for a P/G group name, which covers several
    proteins and is therefore compared at locus level. Anything else that cannot be read as an
    allele designation ('A*24-02', 'A*24:XX', 'A*24:02/B*07:02') returns (None, None): an
    unreadable name is malformed, never a silent locus-only match.

    Whitespace is removed first, exactly as `normalize_sequence` does, so the result depends only
    on the normalized text. The SQL join is keyed on that text; if two spellings with the same
    key could parse differently, Python and SQL would disagree.
    """
    s = normalize_sequence(allele).replace("_", "-")
    if s.startswith("HLA-"):
        s = s[4:]
    if "*" in s:
        token, rest = s.split("*", 1)
        starred = True
    else:
        m = re.match(r"^([A-Z]+)(.*)$", s)
        if not m:
            return None, None
        token, rest = m.groups()
        # Without a star, a digit after the letters is part of the locus only when that spells a
        # known locus ('DRB1' in 'DRB11501'); otherwise it begins the allele ('A' in 'A02:01').
        if rest[:1].isdigit() and _known_locus(token + rest[0]):
            token, rest = token + rest[0], rest[1:]
        rest = rest.lstrip("-")
        # Anything but an allele designation after the locus is not a name ('B2M').
        if not re.fullmatch(rf"[0-9:]*[{_EXPRESSION_SUFFIXES}PG]?", rest):
            return None, None
        starred = False
    if not (_LOCUS_TOKEN_RE.match(token) or token in OUT_OF_SCOPE_LOCI):
        return None, None
    group = rest[-1:] in ("P", "G")
    if group or rest[-1:] in tuple(_EXPRESSION_SUFFIXES):
        rest = rest[:-1]
    if not rest:
        return (token, ()) if not group else (None, None)
    if ":" in rest:
        parts = rest.split(":")
    elif rest.isdigit() and (len(rest) <= 2 or (len(rest) == 3 and starred)):
        parts = [rest]
    elif rest.isdigit() and len(rest) in (4, 6, 8):
        # Pre-2010 compact nomenclature: two digits per field ('A*0201', 'A*02010101').
        parts = [rest[i:i + 2] for i in range(0, len(rest), 2)]
    else:
        return None, None
    if not all(p.isdigit() for p in parts):
        return None, None
    return token, (None if group else tuple(p.zfill(2) for p in parts))


def _known_locus(locus: str) -> bool:
    return class_role_of_locus(locus)[0] is not None or locus in OUT_OF_SCOPE_LOCI


def locus_of(allele: str) -> str | None:
    """Leading locus token of an allele name, e.g. 'HLA-DRB1*15:01' -> 'DRB1'."""
    return parse_allele_name(allele)[0]


def _canonical_locus(locus: str | None) -> str | None:
    return _LOCUS_ALIASES.get(locus, locus)


def pair_group_of(mhc_class: str | None, loci) -> str | None:
    """Which chains may share one row: 'I', or the class-II isotype 'DR' / 'DQ' / 'DP'.

    None when undetermined, which never pairs with anything.
    """
    if mhc_class == "I":
        return "I"
    groups = {locus[:2] for locus in loci}
    return groups.pop() if mhc_class == "II" and len(groups) == 1 else None


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
    a = normalize_sequence(allele)
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
    norm = normalize_sequence(allele)
    if not norm:
        return "absent"
    if is_nonhuman(norm):
        return "nonhuman"
    if len(norm) >= _SEQUENCE_IN_NAME_MIN_LEN and not (set(norm) - VALID_RESIDUES):
        # Residue-only text: a copy of this chain is absent-equivalent, any other sequence is junk.
        return "is_sequence" if norm == normalized_sequence else "malformed"
    locus = locus_of(allele)
    if locus in OUT_OF_SCOPE_LOCI:
        return "out_of_scope"
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
                 reference_version: str, full_lengths: dict[str, int] | None = None):
        self._by_hash = by_hash
        self._sequences = sequences          # normalized_sha256 -> normalized sequence
        self.reference_version = reference_version
        # locus -> full length in this reference; a locus missing here is never judged partial.
        self.full_lengths = dict(full_lengths or {})

    def is_partial(self, length: int, loci) -> bool:
        """True when `length` falls short of full length for every locus the chain matches."""
        known = [self.full_lengths[l] for l in loci if l in self.full_lengths]
        return bool(known) and all(length < PARTIAL_FRACTION * full for full in known)

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
        if evidence == "out_of_scope":
            return out("quarantine", "locus_out_of_scope", evidence=evidence)
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
        if mhc_class is None:
            # Reference-exact but outside the class tables (DQA2, DMA, DOA, HFE, TAP1, ...).
            # Without this the chain was "accepted" with no class or role.
            return out("quarantine", "locus_out_of_scope", loci=loci, alleles=alleles,
                       ambiguous=len(alleles) > 1, ref_status="exact", evidence=evidence)
        classical = all(class_role_of_locus(l)[2] for l in loci)
        if self.is_partial(len(norm), loci):
            # Class and role are known and are reported; the chain is still not eligible.
            return out("quarantine", "reference_partial_sequence", mhc_class=mhc_class,
                       chain_role=chain_role, classical=classical, loci=loci, alleles=alleles,
                       ambiguous=len(alleles) > 1, ref_status="partial", evidence=evidence)

        if evidence == "named":
            # A name is checked at every resolution it states: class/role, then locus, then the
            # allele fields. 'HLA-B*07:02' on an A*02:01 sequence agrees on class and role and is
            # still a contradiction. A coarser name ('A*02') only has to be a prefix of a match.
            name_locus, name_fields = parse_allele_name(allele)
            name_class, name_role, _ = class_role_of_locus(name_locus)
            conflict = None
            if (name_class, name_role) != (mhc_class, chain_role):
                conflict = "name_sequence_conflict"
            elif _canonical_locus(name_locus) not in {_canonical_locus(l) for l in loci}:
                conflict = "name_locus_conflict"
            elif name_fields:
                want = name_fields[:2]
                parsed = (parse_allele_name(a) for a in alleles)
                if not any(_canonical_locus(rl) == _canonical_locus(name_locus)
                           and rf is not None and rf[:len(want)] == want for rl, rf in parsed):
                    conflict = "name_allele_conflict"
            if conflict:
                return out("quarantine", conflict, loci=loci, alleles=alleles,
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
    lengths: dict[str, dict[int, int]] = {}
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
        # Expression-variant alleles ('...N') are often truncated; they do not define full length.
        locus = locus_of(allele)
        if locus is not None and allele[-1:] not in tuple(_EXPRESSION_SUFFIXES):
            per = lengths.setdefault(locus, {})
            per[len(norm)] = per.get(len(norm), 0) + 1

    with Path(path).open() as fh:
        for line in fh:
            if line.startswith(">"):
                flush()
                parts = line[1:].split()
                allele, buf = (parts[1] if len(parts) > 1 else None), []
            else:
                buf.append(line.strip())
    flush()
    full_lengths = {}
    for locus, per in lengths.items():
        near_longest = {n: c for n, c in per.items() if n >= _FULL_LENGTH_BAND * max(per)}
        # Commonest length near the longest; ties go to the longer.
        full_lengths[locus] = max(near_longest, key=lambda n: (near_longest[n], n))
    return EligibilityIndex(by_hash, sequences, reference_version, full_lengths)


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
            "pair_group": pair_group_of(d.mhc_class, d.loci),
            "allele_evidence": d.allele_evidence, "reason": d.reason,
            "alleles": ";".join(d.alleles), "allele_ambiguous": d.allele_ambiguous,
            "normalized_sha256": d.normalized_sha256,
            "reference_version": d.reference_version,
        })
    return rows


def row_decision(index: EligibilityIndex, mhc_one, mhc_one_allele, mhc_two,
                 mhc_two_allele) -> tuple[bool, str]:
    """Row-level eligibility in Python: the same rule the builder's SQL applies by semi-join.

    Each populated slot needs an accepted chain in a role that slot allows. When both slots are
    populated the two chains must also belong together: a class-II alpha with a beta of the same
    isotype. A class-I heavy chain beside a class-II beta, or DQA1 beside DRB1, is not a molecule.
    Used by the dataset verifier to re-check the SQL independently of DuckDB.
    """
    groups = []
    for slot, seq, allele, roles in (("mhc_one", mhc_one, mhc_one_allele, MHC_ONE_ROLES),
                                     ("mhc_two", mhc_two, mhc_two_allele, MHC_TWO_ROLES)):
        if not normalize_sequence(seq):
            continue
        d = index.decide(seq, allele)
        if d.decision != "accept":
            return False, f"{slot}:{d.reason}"
        if d.chain_role not in roles:
            return False, f"{slot}:role_not_allowed_in_slot"
        groups.append(pair_group_of(d.mhc_class, d.loci))
    if len(groups) == 2 and (groups[0] is None or groups[0] != groups[1]):
        return False, "pair_mismatch"
    return True, "eligible"


# The builder's `mhc_class` row label. The vocabulary is the legacy one, so evaluation strata keep
# their names; only the evidence changes, from the allele NAME to the chain's reference decision.
UNRESOLVED_ISOTYPE = "unresolved"


def row_mhc_class(index: EligibilityIndex, mhc_one, mhc_one_allele, mhc_two,
                  mhc_two_allele) -> str:
    """`mhc_class` label of a row under `reference-aware-v1`, from its chains' accepted roles.

    'I' for a class-I heavy chain (classical or not), 'II_complete' for an alpha with a beta,
    'II_partial_alpha', 'II_partial_<DR|DQ|DP>_beta', and 'none' when no slot holds an accepted
    chain. A populated but unaccepted chain also gives 'none': such a row cannot pass the
    filter, so the label never asserts a class the reference did not confirm. SQL twin:
    `cpt_dataset_builder.mhc_class_from_eligibility_sql`.
    """
    def accepted(seq, allele):
        if not normalize_sequence(seq):
            return None
        d = index.decide(seq, allele)
        return d if d.decision == "accept" else None

    one, two = accepted(mhc_one, mhc_one_allele), accepted(mhc_two, mhc_two_allele)
    if one is not None and one.chain_role == "heavy":
        return "I"
    if one is not None and one.chain_role == "alpha":
        return "II_complete" if two is not None and two.chain_role == "beta" else "II_partial_alpha"
    if two is not None and two.chain_role == "beta":
        return f"II_partial_{pair_group_of(two.mhc_class, two.loci) or UNRESOLVED_ISOTYPE}_beta"
    return "none"

