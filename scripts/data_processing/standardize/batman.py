#!/usr/bin/env python3
"""BATMAN standardizer.

BATMAN (Binding Affinity of TCR-MHC ANtigen) — 22K rows, 2 Excel files
(pMHCI + pMHCII). Mutational scan data with quantitative binding measurements.

Columns: tcr, va, vb, cdr3a, cdr3b, trav, traj, trbv, trbd, trbj, assay,
         tcr_source_organism, index_peptide, mhc, pmid, peptide_type,
         peptide, peptide_activity
"""

import re
import sys
from pathlib import Path
from typing import Iterator

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from quest.data.standardization import (
    TARGET_COLUMNS,
    normalize_mhc_allele,
    split_mhc_to_alpha_beta,
    standardize_dataframe,
)
from scripts.data_processing.standardize._base import BaseStandardizer

PMHCI_FILE = "TCR_pMHCI_mutational_scan_database.xlsx"
PMHCII_FILE = "TCR_pMHCII_mutational_scan_database.xlsx"

# Map BATMAN column names to the IMGT prefix that must be prepended
_GENE_PREFIXES = {
    "trav": "TRAV",
    "traj": "TRAJ",
    "trbv": "TRBV",
    "trbd": "TRBD",
    "trbj": "TRBJ",
}


def _add_gene_prefix(value: str, prefix: str) -> str:
    """Add IMGT gene prefix to a bare numeric gene identifier.

    BATMAN stores genes as e.g. '12-3', '35*02', '1*00(80)'.
    This converts to 'TRAV12-3', 'TRAV35*02', 'TRBD1' (stripping scores).
    """
    if not value or value.strip() in ("", "nan"):
        return ""
    value = str(value).strip()
    # Already has a prefix
    if value.upper().startswith("TR"):
        return value
    # Strip score annotations like '1*00(80)' or '1*00(50),2*00(50)'
    # Take first allele, remove parenthetical scores
    value = value.split(",")[0].strip()
    value = re.sub(r"\(\d+\)", "", value).strip()
    if not value:
        return ""
    return f"{prefix}{value}"


class BatmanStandardizer(BaseStandardizer):
    name = "batman"

    def get_column_map(self) -> dict:
        return {
            "cdr3a": "tra",
            "_trav_gene": "trav_gene",
            "_traj_gene": "traj_gene",
            "cdr3b": "trb",
            "_trbv_gene": "trbv_gene",
            "_trbd_gene": "trbd_gene",
            "_trbj_gene": "trbj_gene",
            "index_peptide": "peptide",
            "peptide_activity": "score",
        }

    def load_and_standardize(self) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        chunks = []
        for fname, mhc_class in [(PMHCI_FILE, "I"), (PMHCII_FILE, "II")]:
            fpath = self.source_dir / fname
            if not fpath.exists():
                continue
            df = pd.read_excel(fpath, dtype=str).fillna("")
            df["_mhc_class"] = mhc_class
            df["_source_file"] = fname
            chunks.append(df)

        if not chunks:
            return

        df = pd.concat(chunks, ignore_index=True)

        # Filter to human sequences only
        if "tcr_source_organism" in df.columns:
            df = df[
                df["tcr_source_organism"].str.strip().str.lower() == "human"
            ].reset_index(drop=True)

        # Add gene prefixes (BATMAN stores bare numbers like '12-3')
        for src_col, prefix in _GENE_PREFIXES.items():
            if src_col in df.columns:
                df[f"_{src_col}_gene"] = df[src_col].apply(
                    lambda v, p=prefix: _add_gene_prefix(v, p)
                )

        # Split MHC into mhc_one/mhc_two based on class
        mhc_one = []
        mhc_two = []
        for _, row in df.iterrows():
            raw_mhc = str(row.get("mhc", "")).strip()
            if not raw_mhc:
                mhc_one.append("")
                mhc_two.append("")
                continue
            m1, m2 = split_mhc_to_alpha_beta(raw_mhc)
            mhc_one.append(m1)
            mhc_two.append(m2)

        df["mhc_one"] = mhc_one
        df["mhc_two"] = mhc_two

        # Derive categorical binding from continuous peptide_activity score
        # Strong activation (>=0.5): "pos", Weak (0.1-0.5): "pos",
        # No activation (<0.1): "neg"
        if "peptide_activity" in df.columns:
            activity = pd.to_numeric(df["peptide_activity"], errors="coerce")
            df["_binding"] = ""
            df.loc[activity >= 0.1, "_binding"] = "pos"
            df.loc[activity < 0.1, "_binding"] = "neg"
            # Leave as empty where activity is NaN (non-numeric)

        column_map = self.get_column_map()
        column_map["mhc_one"] = "mhc_one"
        column_map["mhc_two"] = "mhc_two"
        if "_binding" in df.columns:
            column_map["_binding"] = "binding"

        result, dropped = standardize_dataframe(
            df, column_map, source=self.name
        )
        yield result, dropped


def main():
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Standardize BATMAN database")
    parser.add_argument(
        "--source-dir",
        type=str,
        default="data/databases/BATMAN",
        help="Path to BATMAN source directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/standardized/batman)",
    )
    parser.add_argument("--force", action="store_true", help="Force re-run")
    args = parser.parse_args()

    standardizer = BatmanStandardizer(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
    )
    summary = standardizer.run(force=args.force)
    print(summary)


if __name__ == "__main__":
    main()
