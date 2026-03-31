#!/usr/bin/env python3
"""CEDAR standardizer.

CEDAR — 106K rows, 3 CSVs with multi-level headers.
The receptor file contains epitope and MHC data directly in columns
like Epitope_Name and Assay_MHC Allele Names.

Files:
  - receptor/tcr_full_v3.csv (TCR chains with CDR3 + genes + epitope + MHC)
  - tcell/tcell_full_v3.csv (assay data — not used, IDs don't overlap)
  - epitope/epitope_full_v3.csv (epitope sequences — not needed)
"""

import re
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


def _flatten_header(header_rows: list[str]) -> list[str]:
    """Flatten multi-level CSV headers into single-level column names.

    CEDAR/IEDB CSVs have two header rows:
    Row 0: Group names (e.g., 'Chain 1', 'Chain 2', 'Epitope')
    Row 1: Field names (e.g., 'CDR3 Curated', 'V Gene')

    Returns: ['Group_Field', ...] with group propagated forward.
    """
    if len(header_rows) < 2:
        return header_rows[0] if header_rows else []

    groups = header_rows[0]
    fields = header_rows[1]
    result = []
    current_group = ""
    for i, (g, f) in enumerate(zip(groups, fields)):
        g = str(g).strip()
        f = str(f).strip()
        if g and g.lower() not in ("", "nan", "unnamed"):
            current_group = g
        col_name = f"{current_group}_{f}" if current_group else f
        # Clean up
        col_name = col_name.replace("\n", " ").strip()
        result.append(col_name)
    return result


def _extract_cedar_id(iri: str) -> str:
    """Extract numeric ID from CEDAR IRI (e.g., 'http://cedar.iedb.org/assay/1234' -> '1234')."""
    if not iri or str(iri).strip() in ("", "nan"):
        return ""
    m = re.search(r"(\d+)$", str(iri).strip())
    return m.group(1) if m else str(iri).strip()


def _read_cedar_csv(fpath: Path) -> pd.DataFrame:
    """Read a CEDAR/IEDB CSV with multi-level headers."""
    # Read first two rows as headers
    header_df = pd.read_csv(fpath, nrows=1, header=None, dtype=str)
    header2_df = pd.read_csv(fpath, skiprows=1, nrows=1, header=None, dtype=str)

    flat_cols = _flatten_header(
        [header_df.iloc[0].tolist(), header2_df.iloc[0].tolist()]
    )

    # Read data, skipping the two header rows
    df = pd.read_csv(fpath, skiprows=2, header=None, dtype=str)
    # Align column count
    if len(flat_cols) >= len(df.columns):
        df.columns = flat_cols[: len(df.columns)]
    else:
        df.columns = flat_cols + [
            f"unnamed_{i}" for i in range(len(flat_cols), len(df.columns))
        ]

    return df.fillna("")


class CedarStandardizer(BaseStandardizer):
    name = "cedar"

    def get_column_map(self) -> dict:
        # These map the flattened receptor CSV columns
        return {
            "tra": "tra",
            "trav_gene": "trav_gene",
            "trad_gene": "trad_gene",
            "traj_gene": "traj_gene",
            "trb": "trb",
            "trbv_gene": "trbv_gene",
            "trbd_gene": "trbd_gene",
            "trbj_gene": "trbj_gene",
            "peptide": "peptide",
            "mhc_one": "mhc_one",
            "mhc_two": "mhc_two",
        }

    def load_and_standardize(self) -> Iterator[pd.DataFrame]:
        receptor_path = self.source_dir / "receptor" / "tcr_full_v3.csv"

        if not receptor_path.exists():
            return

        # Load receptor data
        receptor_df = _read_cedar_csv(receptor_path)

        # Find CDR3 and gene columns by pattern matching
        merged = pd.DataFrame(index=receptor_df.index)

        # Search for key columns in the receptor DataFrame
        col_mapping = {}
        for col in receptor_df.columns:
            cl = col.lower()
            if "chain 1" in cl and "cdr3" in cl and "curated" in cl and "start" not in cl and "end" not in cl:
                col_mapping["tra"] = col
            elif "chain 1" in cl and "v gene" in cl and "curated" in cl:
                col_mapping["trav_gene"] = col
            elif "chain 1" in cl and "d gene" in cl and "curated" in cl:
                col_mapping["trad_gene"] = col
            elif "chain 1" in cl and "j gene" in cl and "curated" in cl:
                col_mapping["traj_gene"] = col
            elif "chain 2" in cl and "cdr3" in cl and "curated" in cl and "start" not in cl and "end" not in cl:
                col_mapping["trb"] = col
            elif "chain 2" in cl and "v gene" in cl and "curated" in cl:
                col_mapping["trbv_gene"] = col
            elif "chain 2" in cl and "d gene" in cl and "curated" in cl:
                col_mapping["trbd_gene"] = col
            elif "chain 2" in cl and "j gene" in cl and "curated" in cl:
                col_mapping["trbj_gene"] = col

        # Fallback: try calculated columns if curated not found
        for col in receptor_df.columns:
            cl = col.lower()
            if "tra" not in col_mapping:
                if "chain 1" in cl and "cdr3" in cl and "calculated" in cl and "start" not in cl and "end" not in cl:
                    col_mapping["tra"] = col
            if "trb" not in col_mapping:
                if "chain 2" in cl and "cdr3" in cl and "calculated" in cl and "start" not in cl and "end" not in cl:
                    col_mapping["trb"] = col

        # Build the merged DataFrame from receptor data
        for target_col, src_col in col_mapping.items():
            merged[target_col] = receptor_df[src_col]

        # Extract epitope (peptide) directly from receptor file
        epitope_col = None
        for col in receptor_df.columns:
            cl = col.lower()
            if "epitope" in cl and "name" in cl:
                epitope_col = col
                break
        if epitope_col:
            merged["peptide"] = receptor_df[epitope_col]

        # Extract MHC directly from receptor file
        mhc_col = None
        for col in receptor_df.columns:
            cl = col.lower()
            if "mhc" in cl and "allele" in cl:
                mhc_col = col
                break
        if mhc_col:
            mhc_split = receptor_df[mhc_col].apply(split_mhc_to_alpha_beta)
            merged["mhc_one"] = mhc_split.apply(lambda x: x[0])
            merged["mhc_two"] = mhc_split.apply(lambda x: x[1])

        # Ensure peptide/mhc columns exist
        for col in ("peptide", "mhc_one", "mhc_two"):
            if col not in merged.columns:
                merged[col] = ""

        column_map = self.get_column_map()
        result, dropped = standardize_dataframe(
            merged, column_map, source=self.name, stitch=self.stitch,
            hla_dir=self.hla_dir,
        )
        yield result, dropped


def main():
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Standardize CEDAR")
    parser.add_argument("--source-dir", default="data/databases/CEDAR")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--hla-dir", default="")
    args = parser.parse_args()

    standardizer = CedarStandardizer(
        source_dir=args.source_dir, output_dir=args.output_dir,
        hla_dir=args.hla_dir,
    )
    print(standardizer.run(force=args.force))


if __name__ == "__main__":
    main()
