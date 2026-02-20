#!/usr/bin/env python3
"""VDJdb standardizer.

VDJdb — 138K rows, 1 TSV (wide format with paired alpha/beta).

Columns: cdr3.alpha, v.alpha, j.alpha, cdr3.beta, v.beta, d.beta, j.beta,
         species, mhc.a, mhc.b, mhc.class, antigen.epitope, ...
"""

import sys
from pathlib import Path
from typing import Iterator

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from quest.data.standardization import (
    normalize_mhc_allele,
    standardize_dataframe,
)
from scripts.data_processing.standardize._base import BaseStandardizer


class VdjdbStandardizer(BaseStandardizer):
    name = "vdjdb"

    def get_column_map(self) -> dict:
        return {
            "cdr3.alpha": "tra",
            "v.alpha": "trav_gene",
            "j.alpha": "traj_gene",
            "cdr3.beta": "trb",
            "v.beta": "trbv_gene",
            "d.beta": "trbd_gene",
            "j.beta": "trbj_gene",
            "antigen.epitope": "peptide",
            "mhc_one": "mhc_one",
            "mhc_two": "mhc_two",
            "vdjdb.score": "score",
            "meta.study.id": "study_id",
        }

    def load_and_standardize(self) -> Iterator[pd.DataFrame]:
        fpath = self.source_dir / "vdjdb_full.txt"
        if not fpath.exists():
            # Fallback to vdjdb.txt
            fpath = self.source_dir / "vdjdb.txt"
        if not fpath.exists():
            return

        df = pd.read_csv(fpath, sep="\t", dtype=str).fillna("")

        # Filter to human sequences only
        if "species" in df.columns:
            df = df[
                df["species"].str.strip() == "HomoSapiens"
            ].reset_index(drop=True)

        # Normalize MHC: mhc.a → mhc_one, mhc.b → mhc_two
        df["mhc_one"] = df.get("mhc.a", pd.Series("", index=df.index)).apply(
            normalize_mhc_allele
        )
        df["mhc_two"] = df.get("mhc.b", pd.Series("", index=df.index)).apply(
            normalize_mhc_allele
        )

        column_map = self.get_column_map()

        result, dropped = standardize_dataframe(
            df, column_map, source=self.name
        )

        yield result, dropped


def main():
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Standardize VDJdb")
    parser.add_argument("--source-dir", default="data/databases/vdjdb")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    standardizer = VdjdbStandardizer(
        source_dir=args.source_dir, output_dir=args.output_dir,
    )
    print(standardizer.run(force=args.force))


if __name__ == "__main__":
    main()
