#!/usr/bin/env python3
"""TaDB standardizer.

TaDB (T cell Assay Database) — 1.1K rows, 1 CSV.
Epitope-only data (no TCR sequences).

Columns: ACCESSION, Epitope sequence, HLA allele, Epitope type
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


class TadbStandardizer(BaseStandardizer):
    name = "tadb"

    def get_column_map(self) -> dict:
        return {
            "Epitope sequence": "peptide",
        }

    def load_and_standardize(self) -> Iterator[pd.DataFrame]:
        csv_files = list(self.source_dir.glob("*.csv"))
        if not csv_files:
            return

        df = pd.read_csv(csv_files[0], dtype=str).fillna("")

        # Split HLA allele into mhc_one/mhc_two
        mhc_one = []
        mhc_two = []
        for _, row in df.iterrows():
            raw = str(row.get("HLA allele", "")).strip()
            m1, m2 = split_mhc_to_alpha_beta(raw)
            mhc_one.append(m1)
            mhc_two.append(m2)
        df["mhc_one"] = mhc_one
        df["mhc_two"] = mhc_two

        column_map = self.get_column_map()
        column_map["mhc_one"] = "mhc_one"
        column_map["mhc_two"] = "mhc_two"

        result, dropped = standardize_dataframe(
            df, column_map, source=self.name, stitch=self.stitch,
            hla_dir=self.hla_dir,
        )
        yield result, dropped


def main():
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Standardize TaDB database")
    parser.add_argument(
        "--source-dir", default="data/databases/TaDB",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--hla-dir", default="")
    args = parser.parse_args()

    standardizer = TadbStandardizer(
        source_dir=args.source_dir, output_dir=args.output_dir,
        hla_dir=args.hla_dir,
    )
    print(standardizer.run(force=args.force))


if __name__ == "__main__":
    main()
