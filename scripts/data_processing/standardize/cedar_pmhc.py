#!/usr/bin/env python3
"""CEDAR peptide-MHC standardizer.

Ingests experimentally measured peptide-MHC data from CEDAR's mhc_ligand export.

Supports two input formats:

  1. Bulk download CSV — Multi-level headers (2-row), parsed via _read_cedar_csv().
     Columns like "Epitope_Name", "Assay_MHC Allele Names".

  2. API download CSV — Flat single-row headers with __ delimiters.
     Columns like "epitope__name", "mhc_restriction__name", "host__name".
     Produced by download_cedar_api.py --mhc.

Output: peptide + mhc_one + mhc_two + binding columns only (no TCR).
Source = "cedar_pmhc". Filtered to human HLA alleles.
"""

import logging
import sys
from pathlib import Path
from typing import Iterator

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from quest.data.standardization import (
    split_mhc_to_alpha_beta,
    standardize_dataframe,
)
from scripts.data_processing.standardize._base import BaseStandardizer
from scripts.data_processing.standardize.cedar import _read_cedar_csv

logger = logging.getLogger(__name__)


def _is_api_format(csv_path: Path) -> bool:
    """Check if CSV uses flat API headers (__ delimited) vs multi-level bulk headers."""
    with open(csv_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    return "__" in first_line


def _load_api_csv(csv_path: Path) -> tuple[pd.DataFrame, str | None]:
    """Load API-format CSV and return (merged_df, None) or (empty_df, error_msg)."""
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    df = df.fillna("")

    if df.empty:
        return df, "API CSV is empty"

    logger.info("  API format detected — %d rows, %d columns", len(df), len(df.columns))

    # Map known API columns
    peptide_col = "epitope__name"
    mhc_col = "mhc_restriction__name"
    host_col = "host__name"
    measurement_col = "assay__qualitative_measurement"
    obj_type_col = "epitope__object_type"

    for required in (peptide_col, mhc_col):
        if required not in df.columns:
            return pd.DataFrame(), f"Missing required column: {required}"

    # Apply filters on df first to keep indices aligned
    if host_col in df.columns:
        human_mask = (
            df[host_col].str.strip().str.lower().str.contains(
                "homo sapiens|human", na=False
            )
        ) | (df[host_col].str.strip() == "")
        df = df[human_mask]
        logger.info("  After human filter: %d rows", len(df))

    if obj_type_col in df.columns:
        peptide_mask = (
            df[obj_type_col].str.strip().str.lower().str.contains(
                "linear peptide", na=False
            )
        ) | (df[obj_type_col].str.strip() == "")
        df = df[peptide_mask]
        logger.info("  After linear peptide filter: %d rows", len(df))

    df = df.reset_index(drop=True)

    # Build merged DataFrame from filtered df
    merged = pd.DataFrame(index=df.index)
    merged["peptide"] = df[peptide_col].str.strip()

    # Split MHC into mhc_one/mhc_two
    mhc_split = df[mhc_col].str.strip().apply(split_mhc_to_alpha_beta)
    merged["mhc_one"] = mhc_split.apply(lambda x: x[0])
    merged["mhc_two"] = mhc_split.apply(lambda x: x[1])

    # Map qualitative measurement to binding label
    if measurement_col in df.columns:
        meas = df[measurement_col].str.strip().str.lower()
        merged["binding"] = ""
        merged.loc[meas.str.startswith("positive"), "binding"] = "pos"
        merged.loc[meas.str.startswith("negative"), "binding"] = "neg"
    else:
        # Eluted ligand data — positive by physical detection
        merged["binding"] = "pos"

    return merged, None


def _load_bulk_csv(csv_path: Path) -> tuple[pd.DataFrame, str | None]:
    """Load bulk-download CSV with multi-level headers."""
    df = _read_cedar_csv(csv_path)
    if df.empty:
        return df, "Bulk CSV is empty after parsing"

    logger.info("  Bulk format detected — %d rows, %d columns", len(df), len(df.columns))

    # Find peptide, MHC, and organism columns by pattern matching
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
        return pd.DataFrame(), (
            f"Could not find peptide/MHC columns. Available: {list(df.columns)}"
        )

    merged = pd.DataFrame(index=df.index)
    merged["peptide"] = df[peptide_col].fillna("").astype(str).str.strip()

    # Split MHC into mhc_one/mhc_two
    mhc_split = (
        df[mhc_col].fillna("").astype(str).str.strip().apply(split_mhc_to_alpha_beta)
    )
    merged["mhc_one"] = mhc_split.apply(lambda x: x[0])
    merged["mhc_two"] = mhc_split.apply(lambda x: x[1])

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
        merged = merged[human_mask].reset_index(drop=True)

    # All ligand records are positive (physically detected by mass-spec)
    merged["binding"] = "pos"

    return merged, None


class CedarPmhcStandardizer(BaseStandardizer):
    name = "cedar_pmhc"

    def get_column_map(self) -> dict:
        return {
            "peptide": "peptide",
            "mhc_one": "mhc_one",
            "mhc_two": "mhc_two",
            "binding": "binding",
        }

    def load_and_standardize(self) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        csv_path = self.source_dir / "mhc_ligand" / "mhc_ligand_full_v3.csv"

        if not csv_path.exists():
            logger.info("mhc_ligand CSV not found at %s", csv_path)
            return

        if csv_path.stat().st_size == 0:
            logger.info("mhc_ligand CSV is empty at %s", csv_path)
            return

        logger.info("Loading CEDAR mhc_ligand from %s ...", csv_path)

        # Detect format and load accordingly
        if _is_api_format(csv_path):
            merged, err = _load_api_csv(csv_path)
        else:
            merged, err = _load_bulk_csv(csv_path)

        if err:
            logger.warning("cedar_pmhc: %s", err)
            return

        if merged.empty:
            return

        column_map = self.get_column_map()
        result, dropped = standardize_dataframe(merged, column_map, source=self.name, stitch=self.stitch, hla_dir=self.hla_dir)
        yield result, dropped


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Standardize CEDAR peptide-MHC data (mhc_ligand)"
    )
    parser.add_argument("--source-dir", default="data/databases/CEDAR")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--hla-dir", default="")
    args = parser.parse_args()

    standardizer = CedarPmhcStandardizer(
        source_dir=args.source_dir, output_dir=args.output_dir,
        hla_dir=args.hla_dir,
    )
    print(standardizer.run(force=args.force))


if __name__ == "__main__":
    main()
