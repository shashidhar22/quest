#!/usr/bin/env python3
"""IEDB peptide-MHC standardizer.

Ingests experimentally measured peptide-MHC binding data from two IEDB sources:

1. mhc_ligand/mhc_ligand_full_v3.csv — Mass-spec eluted ligand data
   (peptide + MHC allele; positive binding implied by physical detection).
   Uses CEDAR/IEDB multi-level CSV headers.

2. full_database/iedb_public.sql.gz — MySQL dump containing mhc_bind table
   (peptide + MHC allele + IC50/Kd binding affinity + qualitative outcome).

Output: peptide + mhc_one + mhc_two columns only (no TCR).
Source = "iedb_pmhc". Filtered to human HLA alleles.
"""

import gzip
import logging
import re
import sys
from pathlib import Path
from typing import Iterator

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from quest.data.standardization import (
    normalize_mhc_allele,
    normalize_peptide,
    split_mhc_to_alpha_beta,
    standardize_dataframe,
)
from scripts.data_processing.standardize._base import BaseStandardizer

logger = logging.getLogger(__name__)

# Qualitative outcomes from mhc_bind that indicate positive binding
_POSITIVE_OUTCOMES = {
    "Positive",
    "Positive-High",
    "Positive-Intermediate",
    "Positive-Low",
}

_NEGATIVE_OUTCOMES = {
    "Negative",
}


def _parse_mysql_values(values_str: str) -> list[tuple]:
    """Parse a MySQL VALUES clause into a list of row tuples.

    Reuses the logic from iedb_pmhc_deep_dive.py.
    """
    rows = []
    i = 0
    n = len(values_str)

    while i < n:
        while i < n and values_str[i] != "(":
            i += 1
        if i >= n:
            break
        i += 1

        row = []
        while i < n:
            while i < n and values_str[i] in (" ", "\t"):
                i += 1
            if i >= n:
                break
            if values_str[i] == ")":
                i += 1
                break
            elif values_str[i] == ",":
                i += 1
                continue
            elif values_str[i] == "'":
                i += 1
                chars = []
                while i < n:
                    if values_str[i] == "\\" and i + 1 < n:
                        chars.append(values_str[i + 1])
                        i += 2
                    elif values_str[i] == "'":
                        i += 1
                        break
                    else:
                        chars.append(values_str[i])
                        i += 1
                row.append("".join(chars))
            elif values_str[i : i + 4].upper() == "NULL":
                row.append("")
                i += 4
            else:
                start = i
                while i < n and values_str[i] not in (",", ")"):
                    i += 1
                row.append(values_str[start:i].strip())

        rows.append(tuple(row))

    return rows


def _load_mhc_bind_from_mysql(sql_gz_path: Path) -> pd.DataFrame:
    """Parse mhc_bind and related tables from MySQL dump.

    Returns DataFrame with columns:
        peptide, mhc_allele, binding (qualitative), score (numeric)
    """
    if not sql_gz_path.exists():
        logger.warning("MySQL dump not found at %s", sql_gz_path)
        return pd.DataFrame()

    logger.info("Parsing mhc_bind from MySQL dump %s ...", sql_gz_path)

    # We need: mhc_bind, curated_epitope, epitope tables
    # mhc_bind columns: mhc_bind_id, reference_id, curated_epitope_id,
    #   as_location, as_type_id, as_char_value, as_num_value, as_inequality,
    #   as_comments, mhc_allele_restriction_id, mhc_allele_name, complex_id
    # curated_epitope: col_0=curated_epitope_id, col_6=epitope_id
    # epitope: col_0=epitope_id, col_1=epitope_description (the sequence)

    mhc_bind_pattern = re.compile(
        r"INSERT INTO `mhc_bind`\s+(?:\([^)]*\)\s+)?VALUES\s+(.*)",
        re.IGNORECASE,
    )
    curated_epitope_pattern = re.compile(
        r"INSERT INTO `curated_epitope`\s+(?:\([^)]*\)\s+)?VALUES\s+(.*)",
        re.IGNORECASE,
    )
    epitope_pattern = re.compile(
        r"INSERT INTO `epitope`\s+(?:\([^)]*\)\s+)?VALUES\s+(.*)",
        re.IGNORECASE,
    )

    bind_rows = []
    curated_epitope_rows = []
    epitope_rows = []

    with gzip.open(sql_gz_path, "rb") as f:
        try:
            for raw_line in f:
                if b"INSERT INTO" not in raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace").rstrip()

                m = mhc_bind_pattern.match(line)
                if m:
                    rows = _parse_mysql_values(m.group(1).rstrip(";"))
                    bind_rows.extend(rows)
                    continue

                m = curated_epitope_pattern.match(line)
                if m:
                    rows = _parse_mysql_values(m.group(1).rstrip(";"))
                    curated_epitope_rows.extend(rows)
                    continue

                m = epitope_pattern.match(line)
                if m:
                    rows = _parse_mysql_values(m.group(1).rstrip(";"))
                    epitope_rows.extend(rows)
                    continue
        except (gzip.BadGzipFile, OSError) as e:
            logger.info("gzip read ended: %s (normal for multi-member gzip)", e)

    if not bind_rows:
        logger.warning("No mhc_bind rows found in MySQL dump")
        return pd.DataFrame()

    logger.info(
        "Parsed %d mhc_bind, %d curated_epitope, %d epitope rows",
        len(bind_rows),
        len(curated_epitope_rows),
        len(epitope_rows),
    )

    # Build epitope name lookup: curated_epitope_id -> epitope sequence
    # curated_epitope: col_0=curated_epitope_id, col_6=epitope_id
    # epitope: col_0=epitope_id, col_1=epitope_description
    epitope_id_to_name = {}
    for row in epitope_rows:
        if len(row) >= 2 and row[0] and row[1]:
            epitope_id_to_name[row[0]] = row[1]

    curated_to_name = {}
    for row in curated_epitope_rows:
        if len(row) >= 7 and row[0]:
            epitope_id = row[6] if len(row) > 6 else ""
            name = epitope_id_to_name.get(epitope_id, "")
            if not name and len(row) >= 3:
                # Fallback to curated_epitope.col_2 (curated name)
                name = row[2] if row[2] else ""
            if name:
                curated_to_name[row[0]] = name

    # Build mhc_bind DataFrame
    # Columns: mhc_bind_id(0), reference_id(1), curated_epitope_id(2),
    #   as_location(3), as_type_id(4), as_char_value(5), as_num_value(6),
    #   as_inequality(7), as_comments(8), mhc_allele_restriction_id(9),
    #   mhc_allele_name(10), complex_id(11)
    records = []
    for row in bind_rows:
        if len(row) < 11:
            continue
        curated_epi_id = row[2] if row[2] else ""
        char_value = row[5] if len(row) > 5 and row[5] else ""
        num_value = row[6] if len(row) > 6 and row[6] else ""
        allele_name = row[10] if len(row) > 10 and row[10] else ""

        peptide = curated_to_name.get(curated_epi_id, "")
        if not peptide or not allele_name:
            continue

        # Determine binding label
        binding = ""
        if char_value in _POSITIVE_OUTCOMES:
            binding = "pos"
        elif char_value in _NEGATIVE_OUTCOMES:
            binding = "neg"

        records.append(
            {
                "peptide": peptide,
                "mhc_allele": allele_name,
                "binding": binding,
                "score": num_value,
            }
        )

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def _load_mhc_ligand_csv(csv_path: Path) -> pd.DataFrame:
    """Load mhc_ligand CSV with multi-level headers.

    Returns DataFrame with columns: peptide, mhc_allele
    All records are positive (detected by mass-spec).
    """
    if not csv_path.exists():
        logger.info("mhc_ligand CSV not found at %s", csv_path)
        return pd.DataFrame()

    if csv_path.stat().st_size == 0:
        logger.info("mhc_ligand CSV is empty at %s", csv_path)
        return pd.DataFrame()

    from scripts.data_processing.standardize.cedar import _read_cedar_csv

    logger.info("Loading mhc_ligand from %s ...", csv_path)
    df = _read_cedar_csv(csv_path)

    if df.empty:
        return pd.DataFrame()

    # Find peptide and MHC columns by pattern matching
    peptide_col = None
    mhc_col = None
    organism_col = None

    for col in df.columns:
        cl = col.lower()
        if "epitope" in cl and ("name" in cl or "description" in cl):
            if peptide_col is None:
                peptide_col = col
        if "mhc" in cl and ("allele" in cl or "restriction" in cl):
            if mhc_col is None:
                mhc_col = col
        if "organism" in cl and "source" in cl and "iri" not in cl:
            if organism_col is None:
                organism_col = col

    if peptide_col is None or mhc_col is None:
        logger.warning(
            "Could not find peptide/MHC columns in mhc_ligand CSV. "
            "Available columns: %s",
            list(df.columns),
        )
        return pd.DataFrame()

    result = pd.DataFrame(
        {
            "peptide": df[peptide_col].fillna("").astype(str).str.strip(),
            "mhc_allele": df[mhc_col].fillna("").astype(str).str.strip(),
        }
    )

    # Filter to human if organism column exists
    if organism_col:
        human_mask = (
            df[organism_col]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .str.contains("homo sapiens|human", na=False)
        ) | (df[organism_col].fillna("").astype(str).str.strip() == "")
        result = result[human_mask].reset_index(drop=True)

    # All ligand records are positive (physically detected by mass-spec)
    result["binding"] = "pos"
    result["score"] = ""

    logger.info("Loaded %d mhc_ligand records", len(result))
    return result


class IedbPmhcStandardizer(BaseStandardizer):
    name = "iedb_pmhc"

    def get_column_map(self) -> dict:
        return {
            "peptide": "peptide",
            "mhc_one": "mhc_one",
            "mhc_two": "mhc_two",
            "binding": "binding",
            "score": "score",
        }

    def load_and_standardize(self) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        chunks = []

        # Source 1: mhc_ligand CSV (mass-spec eluted ligands)
        ligand_csv = self.source_dir / "mhc_ligand" / "mhc_ligand_full_v3.csv"
        ligand_df = _load_mhc_ligand_csv(ligand_csv)
        if not ligand_df.empty:
            chunks.append(ligand_df)

        # Source 2: mhc_bind from MySQL dump
        mysql_dump = self.source_dir / "full_database" / "iedb_public.sql.gz"
        bind_df = _load_mhc_bind_from_mysql(mysql_dump)
        if not bind_df.empty:
            chunks.append(bind_df)

        if not chunks:
            return

        df = pd.concat(chunks, ignore_index=True)

        # Split MHC allele into mhc_one/mhc_two
        mhc_split = df["mhc_allele"].apply(split_mhc_to_alpha_beta)
        df["mhc_one"] = mhc_split.apply(lambda x: x[0])
        df["mhc_two"] = mhc_split.apply(lambda x: x[1])

        column_map = self.get_column_map()
        result, dropped = standardize_dataframe(df, column_map, source=self.name)
        yield result, dropped


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Standardize IEDB peptide-MHC data (mhc_ligand + mhc_bind)"
    )
    parser.add_argument("--source-dir", default="data/databases/IEDB")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    standardizer = IedbPmhcStandardizer(
        source_dir=args.source_dir, output_dir=args.output_dir
    )
    print(standardizer.run(force=args.force))


if __name__ == "__main__":
    main()
