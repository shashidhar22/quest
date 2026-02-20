#!/usr/bin/env python3
"""McPAS-TCR standardizer.

McPAS-TCR — 40K rows, 1 CSV. Manually curated TCR-epitope associations.

Columns: CDR3.alpha.aa, CDR3.beta.aa, TRAV, TRAJ, TRBV, TRBD, TRBJ,
         Epitope.peptide, MHC, Species, Category, Pathology, ...
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


class McpasStandardizer(BaseStandardizer):
    name = "mcpas"

    def get_column_map(self) -> dict:
        return {
            "CDR3.alpha.aa": "tra",
            "TRAV": "trav_gene",
            "TRAJ": "traj_gene",
            "CDR3.beta.aa": "trb",
            "TRBV": "trbv_gene",
            "TRBD": "trbd_gene",
            "TRBJ": "trbj_gene",
            "Epitope.peptide": "peptide",
        }

    def load_and_standardize(self) -> Iterator[pd.DataFrame]:
        fpath = self.source_dir / "McPAS-TCR.csv"
        if not fpath.exists():
            return

        df = pd.read_csv(fpath, dtype=str, encoding="latin-1").fillna("")

        # Filter to human sequences only
        if "Species" in df.columns:
            df = df[
                df["Species"].str.strip().str.lower() == "human"
            ].reset_index(drop=True)

        # Explode '/' in Epitope.peptide to separate rows
        if "Epitope.peptide" in df.columns:
            df["Epitope.peptide"] = df["Epitope.peptide"].astype(str)
            df = df.assign(
                **{"Epitope.peptide": df["Epitope.peptide"].str.split("/")}
            ).explode("Epitope.peptide")
            df["Epitope.peptide"] = df["Epitope.peptide"].str.strip()
            df = df.reset_index(drop=True)

        # Split MHC into mhc_one/mhc_two
        mhc_one = []
        mhc_two = []
        for _, row in df.iterrows():
            raw = str(row.get("MHC", "")).strip()
            m1, m2 = split_mhc_to_alpha_beta(raw)
            mhc_one.append(m1)
            mhc_two.append(m2)
        df["mhc_one"] = mhc_one
        df["mhc_two"] = mhc_two

        column_map = self.get_column_map()
        column_map["mhc_one"] = "mhc_one"
        column_map["mhc_two"] = "mhc_two"

        result, dropped = standardize_dataframe(
            df, column_map, source=self.name
        )
        yield result, dropped


def main():
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Standardize McPAS-TCR")
    parser.add_argument("--source-dir", default="data/databases/McPAS-TCR")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    standardizer = McpasStandardizer(
        source_dir=args.source_dir, output_dir=args.output_dir,
    )
    print(standardizer.run(force=args.force))


if __name__ == "__main__":
    main()
