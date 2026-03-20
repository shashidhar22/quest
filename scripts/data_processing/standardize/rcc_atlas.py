#!/usr/bin/env python3
"""RCC_ATLAS standardizer.

RCC_ATLAS — 81 rows, 1 CSV file of literature-curated T-cell receptor data
focused on renal cell carcinoma antigens, viral antigens (CMV, EBV, Influenza,
SARS-CoV-2), and tumor-associated antigens (NY-ESO-1, WT1, PRAME, etc.).

Columns: antigen, category, epitope, MHC, CDR3A, CDR3B, clone_id, TRAV, TRBV,
         source, reference, note
"""

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

DATA_FILE = "rcc_tcells.csv"


class RccAtlasStandardizer(BaseStandardizer):
    name = "rcc_atlas"

    def get_column_map(self) -> dict:
        return {
            "CDR3A": "tra",
            "CDR3B": "trb",
            "TRAV": "trav_gene",
            "TRBV": "trbv_gene",
            "epitope": "peptide",
            "mhc_one": "mhc_one",
            "mhc_two": "mhc_two",
            "_binding": "binding",
        }

    def load_and_standardize(self) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        fpath = self.source_dir / DATA_FILE
        df = pd.read_csv(fpath, dtype=str).fillna("")

        # Split MHC into mhc_one/mhc_two
        mhc_one = []
        mhc_two = []
        for _, row in df.iterrows():
            raw_mhc = str(row.get("MHC", "")).strip()
            if not raw_mhc:
                mhc_one.append("")
                mhc_two.append("")
                continue
            m1, m2 = split_mhc_to_alpha_beta(raw_mhc)
            mhc_one.append(m1)
            mhc_two.append(m2)

        df["mhc_one"] = mhc_one
        df["mhc_two"] = mhc_two

        # All rows are literature-curated positive associations
        df["_binding"] = "pos"

        column_map = self.get_column_map()

        result, dropped = standardize_dataframe(
            df, column_map, source=self.name, stitch=self.stitch
        )
        yield result, dropped


def main():
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Standardize RCC_ATLAS database")
    parser.add_argument(
        "--source-dir",
        type=str,
        default="data/databases/RCC_ATLAS",
        help="Path to RCC_ATLAS source directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/standardized/rcc_atlas)",
    )
    parser.add_argument("--force", action="store_true", help="Force re-run")
    args = parser.parse_args()

    standardizer = RccAtlasStandardizer(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
    )
    summary = standardizer.run(force=args.force)
    print(summary)


if __name__ == "__main__":
    main()
