#!/usr/bin/env python3
"""OTS standardizer.

OTS — 7M rows, 2166 CSV files with JSON metadata row.
Paired alpha-beta data. No epitope/MHC.

File pattern: SRR{ID}_1_Paired_All.csv
Row 1: JSON metadata (skip)
Row 2+: Data with ~270 columns for paired alpha+beta

Key columns: cdr3_aa_alpha, v_call_alpha, j_call_alpha,
             cdr3_aa_beta, v_call_beta, j_call_beta
"""

import json
import sys
from pathlib import Path
from typing import Iterator

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from quest.data.standardization import standardize_dataframe
from scripts.data_processing.standardize._base import BaseStandardizer

CHUNK_SIZE = 500_000


class OtsStandardizer(BaseStandardizer):
    name = "ots"
    streaming = True

    def get_column_map(self) -> dict:
        return {
            "cdr3_aa_alpha": "tra",
            "v_call_alpha": "trav_gene",
            "j_call_alpha": "traj_gene",
            "cdr3_aa_beta": "trb",
            "v_call_beta": "trbv_gene",
            "d_call_beta": "trbd_gene",
            "j_call_beta": "trbj_gene",
        }

    def get_file_list(self) -> list[Path]:
        csv_files = sorted(self.source_dir.glob("*.csv"))
        if not csv_files:
            csv_files = sorted(self.source_dir.rglob("*_Paired_All.csv"))
        return csv_files

    def process_file(self, file_path: Path) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """Process a single OTS CSV file (used by parallel_run)."""
        column_map = self.get_column_map()
        study_id = file_path.stem.split("_")[0]

        try:
            for chunk in pd.read_csv(
                file_path,
                skiprows=1,
                dtype=str,
                chunksize=CHUNK_SIZE,
                on_bad_lines="skip",
            ):
                chunk = chunk.fillna("")
                result, dropped = standardize_dataframe(
                    chunk, column_map, source=self.name, study_id=study_id, stitch=self.stitch,
                    hla_dir=self.hla_dir,
                )
                yield result, dropped
        except Exception as e:
            print(f"  Warning: Failed to read {file_path.name}: {e}")

    def load_and_standardize(self) -> Iterator[pd.DataFrame]:
        csv_files = sorted(self.source_dir.glob("*.csv"))
        if not csv_files:
            csv_files = sorted(self.source_dir.rglob("*_Paired_All.csv"))

        column_map = self.get_column_map()

        for fpath in csv_files:
            study_id = fpath.stem.split("_")[0]  # SRR ID

            try:
                # Skip first row (JSON metadata)
                for chunk in pd.read_csv(
                    fpath,
                    skiprows=1,
                    dtype=str,
                    chunksize=CHUNK_SIZE,
                    on_bad_lines="skip",
                ):
                    chunk = chunk.fillna("")
                    result, dropped = standardize_dataframe(
                        chunk, column_map, source=self.name, study_id=study_id, stitch=self.stitch,
                        hla_dir=self.hla_dir,
                    )
                    yield result, dropped
            except Exception as e:
                print(f"  Warning: Failed to read {fpath.name}: {e}")
                continue


def main():
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Standardize OTS")
    parser.add_argument("--source-dir", default="data/databases/OTS")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--hla-dir", default="")
    args = parser.parse_args()

    standardizer = OtsStandardizer(
        source_dir=args.source_dir, output_dir=args.output_dir,
        hla_dir=args.hla_dir,
    )
    print(standardizer.run(force=args.force))


if __name__ == "__main__":
    main()
