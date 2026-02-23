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

    # Full column map (score >= 1): all fields for interaction training
    COLUMN_MAP_FULL = {
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

    # TCR-only column map (score 0): no peptide/MHC for MLM-only training
    COLUMN_MAP_TCR = {
        "cdr3.alpha": "tra",
        "v.alpha": "trav_gene",
        "j.alpha": "traj_gene",
        "cdr3.beta": "trb",
        "v.beta": "trbv_gene",
        "d.beta": "trbd_gene",
        "j.beta": "trbj_gene",
        "vdjdb.score": "score",
        "meta.study.id": "study_id",
    }

    # pMHC-only column map (score 0): no TCR for pMHC training only
    COLUMN_MAP_PMHC = {
        "antigen.epitope": "peptide",
        "mhc_one": "mhc_one",
        "mhc_two": "mhc_two",
        "vdjdb.score": "score",
        "meta.study.id": "study_id",
    }

    def get_column_map(self) -> dict:
        return self.COLUMN_MAP_FULL

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

        # Parse score: missing/empty → treat as trusted (score 1)
        if "vdjdb.score" in df.columns:
            scores = pd.to_numeric(df["vdjdb.score"], errors="coerce").fillna(1).astype(int)
        else:
            scores = pd.Series(1, index=df.index)

        # Split by score threshold
        df_high = df[scores >= 1].reset_index(drop=True)
        df_low = df[scores == 0].reset_index(drop=True)

        # Yield high-score rows with full column map (unchanged behavior)
        if not df_high.empty:
            result, dropped = standardize_dataframe(
                df_high, self.COLUMN_MAP_FULL, source=self.name
            )
            yield result, dropped

        # Yield score-0 rows split into TCR-only and pMHC-only
        if not df_low.empty:
            # TCR-only: rows that have at least one valid CDR3
            has_tcr = (df_low.get("cdr3.alpha", pd.Series("", index=df_low.index)).str.strip() != "") | (
                df_low.get("cdr3.beta", pd.Series("", index=df_low.index)).str.strip() != ""
            )
            df_tcr = df_low[has_tcr].reset_index(drop=True)
            if not df_tcr.empty:
                result_tcr, dropped_tcr = standardize_dataframe(
                    df_tcr, self.COLUMN_MAP_TCR, source=self.name
                )
                yield result_tcr, dropped_tcr

            # pMHC-only: rows that have peptide or MHC data
            has_pmhc = (
                (df_low.get("antigen.epitope", pd.Series("", index=df_low.index)).str.strip() != "")
                | (df_low["mhc_one"].str.strip() != "")
                | (df_low["mhc_two"].str.strip() != "")
            )
            df_pmhc = df_low[has_pmhc].reset_index(drop=True)
            if not df_pmhc.empty:
                result_pmhc, dropped_pmhc = standardize_dataframe(
                    df_pmhc, self.COLUMN_MAP_PMHC, source=self.name
                )
                yield result_pmhc, dropped_pmhc


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
