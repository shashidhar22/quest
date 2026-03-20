"""Core standardization functions for the QUEST pipeline.

Provides normalization for CDR3 sequences, gene names, MHC alleles,
and peptides. Also provides standardize_dataframe() which maps raw
database columns to the unified 23-column TARGET_COLUMNS schema.
"""

import logging
import math
import re
from collections import defaultdict

import pandas as pd

logger = logging.getLogger(__name__)

# The 23 target columns for the unified schema
TARGET_COLUMNS = [
    "tra",
    "trav_gene",
    "trad_gene",
    "traj_gene",
    "tra_cdr1",
    "tra_cdr2",
    "tra_cdr3",
    "tra_full",
    "trb",
    "trbv_gene",
    "trbd_gene",
    "trbj_gene",
    "trb_cdr1",
    "trb_cdr2",
    "trb_cdr3",
    "trb_full",
    "peptide",
    "mhc_one",
    "mhc_two",
    "binding",
    "score",
    "source",
    "study_id",
]

# Valid amino acid characters (standard 20 + U selenocysteine + X unknown)
VALID_AA = set("ACDEFGHIKLMNPQRSTVWYX" + "U")

# NA-like strings to treat as empty
_NA_STRINGS = {"na", "nan", "none", "null", "n/a", ".", "-", "nd", "not determined"}

# Global normalization cache: {field_type: {raw_value: normalized_value}}
_NORM_CACHE: dict[str, dict[str, str]] = defaultdict(dict)

# CDR lookup cache: {v_gene: {"cdr1": seq, "cdr2": seq}}
_CDR_LOOKUP_CACHE: dict[str, dict[str, str]] = {}


def clear_norm_cache():
    """Clear the global normalization cache."""
    _NORM_CACHE.clear()
    _CDR_LOOKUP_CACHE.clear()


def _is_na(value) -> bool:
    """Check if a value is NA-like."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in _NA_STRINGS:
        return True
    return False


# ---------------------------------------------------------------------------
# normalize_cdr3
# ---------------------------------------------------------------------------
def normalize_cdr3(value) -> str:
    """Normalize a CDR3 amino acid sequence.

    - Strips whitespace, uppercases
    - Rejects sequences < 4 AA
    - Rejects sequences with non-AA characters
    - Returns "" for NA/empty/invalid
    """
    if _is_na(value):
        return ""
    value = str(value).strip().upper()
    if not value:
        return ""
    # Remove internal whitespace
    value = "".join(value.split())
    if len(value) < 4:
        return ""
    if not all(c in VALID_AA for c in value):
        return ""
    return value


# ---------------------------------------------------------------------------
# normalize_gene
# ---------------------------------------------------------------------------
# Pattern for valid TCR gene names: TR[ABDG][VDJ] followed by number
_TCR_GENE_RE = re.compile(
    r"(TR[ABDG][VDJ]\d+(?:-\d+)?(?:\*\d+)?)",
    re.IGNORECASE,
)


def normalize_gene(value) -> str:
    """Normalize a TCR V/D/J gene name to IMGT format.

    - Accepts TRAV, TRBV, TRAJ, TRBJ, TRBD, TRDV, TRGV, etc.
    - Rejects IG (BCR) genes
    - Handles comma/semicolon-separated multi-gene annotations (takes first)
    - Strips allele suffixes for consistency
    - Returns "" for invalid/empty
    """
    if _is_na(value):
        return ""
    value = str(value).strip()
    if not value:
        return ""

    # Reject IG (BCR) genes
    if re.match(r"^IG[HKL]", value, re.IGNORECASE):
        return ""

    # Fix common prefix variant: TCRBV -> TRBV
    value = re.sub(r"^TCR([ABDG])", r"TR\1", value, flags=re.IGNORECASE)

    # Handle multi-gene (comma or semicolon separated) — take first
    for sep in (",", ";"):
        if sep in value:
            value = value.split(sep)[0].strip()

    # Strip leading zeros in segment numbers: TRBV06-05 -> TRBV6-5
    def _strip_zeros(m):
        prefix = m.group(1)
        num = m.group(2).lstrip("0") or "0"
        rest = m.group(3)
        if rest:
            rest_num = rest.lstrip("0") or "0"
            return f"{prefix}{num}-{rest_num}"
        return f"{prefix}{num}"

    value = re.sub(
        r"(TR[ABDG][VDJ])0*(\d+)(?:-0*(\d+))?",
        _strip_zeros,
        value,
        flags=re.IGNORECASE,
    )

    # Try to extract a valid TCR gene name
    m = _TCR_GENE_RE.search(value)
    if m:
        gene = m.group(1).upper()
        # Normalize: ensure TR prefix is uppercase
        return gene

    # Check if it looks like a bare TCR gene without the full pattern
    if re.match(r"TR[ABDG][VDJ]", value, re.IGNORECASE):
        return value.upper()

    return ""


# ---------------------------------------------------------------------------
# normalize_mhc_allele
# ---------------------------------------------------------------------------
# HLA allele pattern: HLA-A*02:01 or A*02:01 or A0201
_HLA_FULL_RE = re.compile(
    r"(HLA-)?((?:A|B|C|E|F|G|DRA1?|DRB[1-9]|DQA1|DQB1|DPA1|DPB1)\*\d{2,3}:\d{2,3})",
    re.IGNORECASE,
)

_HLA_COMPACT_RE = re.compile(
    r"^(HLA-)?([ABC])(\d{2})(\d{2})$",
    re.IGNORECASE,
)

_HLA_NOCOLON_RE = re.compile(
    r"(HLA-)?((?:DRB[1-9]|DQA1|DQB1|DPA1|DPB1|A|B|C)\*?)(\d{2,3})(\d{2,3})",
    re.IGNORECASE,
)


def normalize_mhc_allele(value) -> str:
    """Normalize an MHC allele name to standard HLA format.

    - HLA-A*02:01 format preferred
    - Handles compact (A0201), missing prefix (A*02:01), etc.
    - Returns "" for non-allele text, empty, or NA
    """
    if _is_na(value):
        return ""
    value = str(value).strip()
    if not value:
        return ""

    # Reject obvious non-allele text
    if len(value) > 50:
        return ""
    if any(w in value.lower() for w in ("see reference", "class i", "class ii", "mutant")):
        if "*" not in value and "HLA" not in value.upper():
            return ""

    # Already in standard format: HLA-A*02:01
    m = _HLA_FULL_RE.search(value)
    if m:
        gene = m.group(2).upper()
        # Ensure HLA- prefix
        return f"HLA-{gene}"

    # Compact format: A0201 -> HLA-A*02:01
    m = _HLA_COMPACT_RE.match(value)
    if m:
        gene = m.group(2).upper()
        group = m.group(3)
        protein = m.group(4)
        return f"HLA-{gene}*{group}:{protein}"

    # Mouse MHC: H-2Kb, H-2Db, H-2IAb etc.
    m_mouse = re.match(r"H-2([KDLA][abdkq]?b?)", value, re.IGNORECASE)
    if m_mouse:
        return value  # Keep as-is for mouse

    # Try to extract anything that looks HLA-like
    m = _HLA_NOCOLON_RE.search(value)
    if m:
        prefix = m.group(2).upper()
        if not prefix.endswith("*"):
            prefix = prefix + "*"
        group = m.group(3)
        protein = m.group(4)
        return f"HLA-{prefix}{group}:{protein}"

    return ""


# ---------------------------------------------------------------------------
# classify_mhc_class
# ---------------------------------------------------------------------------
_CLASS_I_GENES = {"A", "B", "C", "E", "F", "G"}
_CLASS_II_GENES = {"DRA", "DRA1", "DRB1", "DRB2", "DRB3", "DRB4", "DRB5",
                   "DQA1", "DQB1", "DPA1", "DPB1"}


def classify_mhc_class(allele: str) -> str:
    """Classify an MHC allele as class I or II.

    Returns "I", "II", or "" if unknown.
    """
    if not allele:
        return ""

    allele_upper = allele.upper()

    # Mouse MHC
    m = re.match(r"H-2(I?[A-Z])", allele_upper)
    if m:
        gene_part = m.group(1)
        if gene_part.startswith("I"):
            return "II"  # H-2IA, H-2IE are class II
        return "I"  # H-2K, H-2D, H-2L are class I

    # HLA: extract gene name
    m = re.match(r"(?:HLA-)?((?:DR[AB]\d?|DQ[AB]\d|DP[AB]\d|[A-G]))", allele_upper)
    if m:
        gene = m.group(1)
        if gene in _CLASS_I_GENES:
            return "I"
        if gene in _CLASS_II_GENES:
            return "II"
        # Check prefix
        if gene.startswith("D"):
            return "II"

    return ""


# ---------------------------------------------------------------------------
# split_mhc_to_alpha_beta
# ---------------------------------------------------------------------------
# Class II alpha chain genes
_CLASS_II_ALPHA = {"DRA", "DRA1", "DQA1", "DPA1"}
# Class II beta chain genes
_CLASS_II_BETA = {"DRB1", "DRB2", "DRB3", "DRB4", "DRB5", "DQB1", "DPB1"}


def split_mhc_to_alpha_beta(allele) -> tuple[str, str]:
    """Split an MHC allele into (mhc_one, mhc_two).

    - Class I alleles go to mhc_one (alpha chain = heavy chain)
    - Class II alpha (DRA, DQA, DPA) goes to mhc_one
    - Class II beta (DRB, DQB, DPB) goes to mhc_two
    - Slash-separated pairs are split accordingly
    - Returns ("", "") for empty/None
    """
    if _is_na(allele):
        return ("", "")
    allele = str(allele).strip()
    if not allele:
        return ("", "")

    mhc_one = ""
    mhc_two = ""

    # Handle slash-separated class II pairs: DQA1*01:02/DQB1*06:02
    parts = re.split(r"[/,]", allele)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        normalized = normalize_mhc_allele(part)
        if not normalized:
            continue

        mhc_class = classify_mhc_class(normalized)

        if mhc_class == "I":
            mhc_one = normalized
        elif mhc_class == "II":
            # Determine alpha vs beta
            norm_upper = normalized.upper()
            is_beta = any(g in norm_upper for g in ("DRB", "DQB", "DPB"))
            if is_beta:
                mhc_two = normalized
            else:
                mhc_one = normalized
        else:
            # Unknown class — put in mhc_one by default
            if not mhc_one:
                mhc_one = normalized

    return (mhc_one, mhc_two)


# ---------------------------------------------------------------------------
# normalize_peptide
# ---------------------------------------------------------------------------
def normalize_peptide(value) -> str:
    """Normalize a peptide/epitope sequence.

    - Strips whitespace, uppercases
    - Rejects non-AA characters
    - Rejects sequences > 100 AA (likely not a peptide)
    - Returns "" for empty/NA
    """
    if _is_na(value):
        return ""
    value = str(value).strip().upper()
    if not value:
        return ""
    # Reject long non-peptide text
    if len(value) > 100:
        return ""
    # Reject if contains non-AA characters (spaces, digits, etc.)
    if not all(c in VALID_AA for c in value):
        return ""
    return value


# ---------------------------------------------------------------------------
# enrich_cdr_columns
# ---------------------------------------------------------------------------
def _lookup_cdr_from_vgene(v_gene: str) -> dict[str, str]:
    """Look up CDR1/CDR2 sequences from a V gene using tidytcells.

    Returns {"cdr1": seq, "cdr2": seq}. Empty strings if lookup fails.
    """
    if v_gene in _CDR_LOOKUP_CACHE:
        return _CDR_LOOKUP_CACHE[v_gene]

    result = {"cdr1": "", "cdr2": ""}

    try:
        import tidytcells.tr as tr
    except ImportError:
        _CDR_LOOKUP_CACHE[v_gene] = result
        return result

    for gene_variant in (v_gene, f"{v_gene}*01"):
        try:
            aa_seqs = tr.get_aa_sequence(gene_variant)
            result["cdr1"] = aa_seqs.get("CDR1-IMGT", "")
            result["cdr2"] = aa_seqs.get("CDR2-IMGT", "")
            break
        except (ValueError, KeyError, Exception):
            continue

    _CDR_LOOKUP_CACHE[v_gene] = result
    return result


def enrich_cdr_columns(df: pd.DataFrame, stitch: bool = False) -> pd.DataFrame:
    """Populate CDR1/CDR2/CDR3 and full-length columns in a standardized DataFrame.

    Non-stitch mode (default):
    - tra_cdr3 = copy of tra; trb_cdr3 = copy of trb
    - CDR1/CDR2 from V-gene lookup via tidytcells
    - tra_full/trb_full stay empty

    Stitch mode:
    - Same CDR1/CDR2/CDR3 as non-stitch mode
    - Additionally: stitch full-length sequences into tra_full/trb_full

    Modifies DataFrame in place and returns it.
    """
    if df.empty:
        return df

    # CDR3 = copy of tra/trb
    df["tra_cdr3"] = df["tra"]
    df["trb_cdr3"] = df["trb"]

    # CDR1/CDR2 from V-gene lookup (unique-then-map pattern)
    for v_col, cdr1_col, cdr2_col in [
        ("trav_gene", "tra_cdr1", "tra_cdr2"),
        ("trbv_gene", "trb_cdr1", "trb_cdr2"),
    ]:
        unique_genes = df[v_col].unique()
        cdr1_map = {}
        cdr2_map = {}
        for gene in unique_genes:
            if gene:
                cdrs = _lookup_cdr_from_vgene(gene)
                cdr1_map[gene] = cdrs["cdr1"]
                cdr2_map[gene] = cdrs["cdr2"]
            else:
                cdr1_map[gene] = ""
                cdr2_map[gene] = ""
        df[cdr1_col] = df[v_col].map(cdr1_map)
        df[cdr2_col] = df[v_col].map(cdr2_map)

    # Stitch mode: generate full-length sequences
    if stitch:
        try:
            from quest.parsers.tcr_stitcher import TCRStitcher

            stitcher = TCRStitcher()
            if stitcher.enabled:
                for chain, cdr3_col, v_col, j_col, full_col in [
                    ("TRA", "tra", "trav_gene", "traj_gene", "tra_full"),
                    ("TRB", "trb", "trbv_gene", "trbj_gene", "trb_full"),
                ]:
                    mask = (
                        df[cdr3_col].ne("")
                        & df[v_col].ne("")
                        & df[j_col].ne("")
                    )
                    if mask.any():
                        subset = df.loc[mask, [cdr3_col, v_col, j_col]]
                        stitch_df = pd.DataFrame({
                            "cdr3": subset[cdr3_col],
                            "v_gene": subset[v_col],
                            "j_gene": subset[j_col],
                        })
                        stitched = stitcher.stitch_batch(
                            stitch_df, chain=chain
                        )
                        if stitched is not None and "full_sequence" in stitched.columns:
                            df.loc[mask, full_col] = (
                                stitched["full_sequence"].fillna("").values
                            )
        except (ImportError, Exception) as e:
            logger.debug("Stitching unavailable or failed: %s", e)

    return df


# ---------------------------------------------------------------------------
# standardize_dataframe
# ---------------------------------------------------------------------------
# Map target column names to their normalization functions
_NORMALIZERS = {
    "tra": normalize_cdr3,
    "trb": normalize_cdr3,
    "trav_gene": normalize_gene,
    "trad_gene": normalize_gene,
    "traj_gene": normalize_gene,
    "trbv_gene": normalize_gene,
    "trbd_gene": normalize_gene,
    "trbj_gene": normalize_gene,
    "peptide": normalize_peptide,
    "mhc_one": normalize_mhc_allele,
    "mhc_two": normalize_mhc_allele,
}


def standardize_dataframe(
    df: pd.DataFrame,
    column_map: dict[str, str],
    source: str,
    study_id: str = "",
    stitch: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map and normalize a DataFrame to the 23-column TARGET_COLUMNS schema.

    Args:
        df: Input DataFrame with database-specific column names.
        column_map: {source_column: target_column} mapping.
        source: Source database name (e.g., "batman").
        study_id: Optional study identifier.
        stitch: If True, generate full-length sequences via TCRStitcher.

    Returns:
        (standardized_df, dropped_df) where:
        - standardized_df has exactly TARGET_COLUMNS
        - dropped_df tracks invalid values with columns:
          reason, source_file, row_index, field, raw_value
    """
    if df.empty:
        empty_result = pd.DataFrame(columns=TARGET_COLUMNS)
        for col in TARGET_COLUMNS:
            empty_result[col] = empty_result[col].astype(str)
        empty_dropped = pd.DataFrame(
            columns=["reason", "source_file", "row_index", "field", "raw_value"]
        )
        return empty_result, empty_dropped

    # Build result DataFrame with all target columns initialized to ""
    result = pd.DataFrame(index=df.index)
    for col in TARGET_COLUMNS:
        result[col] = ""

    # Map source columns to target columns
    for src_col, tgt_col in column_map.items():
        if tgt_col not in TARGET_COLUMNS:
            continue
        if src_col in df.columns:
            result[tgt_col] = df[src_col].fillna("").astype(str)

    # Set source and study_id
    result["source"] = source
    if study_id:
        result["study_id"] = study_id

    # Normalize columns using unique-then-map for efficiency
    dropped_records: list[pd.DataFrame] = []

    for tgt_col, normalizer in _NORMALIZERS.items():
        raw_values = result[tgt_col]
        unique_raw = raw_values.unique()

        # Build lookup from unique raw values, and track which raw values
        # were meaningful (non-empty, non-NA) to avoid per-row string ops
        cache_key = tgt_col
        lookup = {}
        meaningful_lookup = {}
        for raw in unique_raw:
            raw_str = str(raw)
            stripped = raw_str.strip()
            meaningful_lookup[raw] = bool(stripped and stripped.lower() not in _NA_STRINGS)
            if raw_str in _NORM_CACHE[cache_key]:
                lookup[raw] = _NORM_CACHE[cache_key][raw_str]
            else:
                normalized = normalizer(raw_str)
                lookup[raw] = normalized
                _NORM_CACHE[cache_key][raw_str] = normalized

        # Apply lookup
        result[tgt_col] = raw_values.map(lookup)

        # Track dropped values vectorized: was meaningful but normalized to ""
        was_meaningful = raw_values.map(meaningful_lookup)
        normalized_empty = result[tgt_col].eq("")
        dropped_mask = was_meaningful & normalized_empty

        if dropped_mask.any():
            dropped_idx = result.index[dropped_mask]
            dropped_records.append(pd.DataFrame({
                "reason": f"invalid_{tgt_col}",
                "source_file": source,
                "row_index": dropped_idx,
                "field": tgt_col,
                "raw_value": raw_values[dropped_mask].astype(str).str[:200],
            }))

    # Enrich CDR1/CDR2/CDR3 and full-length columns
    result = enrich_cdr_columns(result, stitch=stitch)

    # Drop rows that have no valid field at all (no CDR3, no peptide, no MHC)
    value_cols = [
        "tra", "trb", "peptide", "mhc_one", "mhc_two",
        "trav_gene", "traj_gene", "trbv_gene", "trbj_gene",
        "binding", "score",
    ]
    has_value = result[value_cols].ne("").any(axis=1)
    no_value_mask = ~has_value

    if no_value_mask.any():
        no_value_idx = result.index[no_value_mask]
        dropped_records.append(pd.DataFrame({
            "reason": "no_valid_field",
            "source_file": source,
            "row_index": no_value_idx,
            "field": "",
            "raw_value": "",
        }))
        result = result[has_value].reset_index(drop=True)

    # Ensure all columns are string type
    for col in TARGET_COLUMNS:
        result[col] = result[col].fillna("").astype(str)

    if dropped_records:
        dropped_df = pd.concat(dropped_records, ignore_index=True)
    else:
        dropped_df = pd.DataFrame(
            columns=["reason", "source_file", "row_index", "field", "raw_value"]
        )

    return result, dropped_df
