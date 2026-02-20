#!/usr/bin/env python3
"""immuneACCESS standardizer.

immuneACCESS — 3.5B rows, TSV files in subdirectories.
Chain determined by directory name (bulk_survey_tra/trb).

Key columns: amino_acid, v_gene, d_gene, j_gene, frame_type
Filter: frame_type == 'In'

Directories:
  - existing_data/bulk_survey_trb/ (TCR-beta)
  - existing_data/bulk_survey_tra/ (TCR-alpha)
  - immunoseq_data/ (mixed)
  - FHCRC-Warren-Updated_datasets/ (patient data)
"""

import logging
import sys
from pathlib import Path
from typing import Iterator

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from quest.data.standardization import standardize_dataframe
from scripts.data_processing.standardize._base import BaseStandardizer

logger = logging.getLogger(__name__)

CHUNK_SIZE = 500_000


class ImmuneaccessStandardizer(BaseStandardizer):
    name = "immuneaccess"
    streaming = True

    # Track warned parent directories to avoid log spam
    _warned_dirs: set = set()

    def get_column_map(self) -> dict:
        # Will be overridden per-file based on chain type
        return {}

    def _get_chain_type(self, fpath: Path) -> str:
        """Determine chain type from path."""
        path_str = str(fpath).lower()
        if "bulk_survey_tra" in path_str or "_tcra" in path_str or "_tra." in path_str:
            return "TRA"
        if "bulk_survey_trb" in path_str or "_tcrb" in path_str or "_trb." in path_str:
            return "TRB"
        # Default to beta — log warning once per parent directory
        parent = str(fpath.parent)
        if parent not in self._warned_dirs:
            self._warned_dirs.add(parent)
            logger.warning(
                "No chain pattern matched for %s — defaulting to TRB", parent
            )
        return "TRB"

    def _get_column_map_for_chain(self, chain: str) -> dict:
        if chain == "TRA":
            return {
                "amino_acid": "tra",
                "v_gene": "trav_gene",
                "d_gene": "trad_gene",
                "j_gene": "traj_gene",
            }
        else:
            return {
                "amino_acid": "trb",
                "v_gene": "trbv_gene",
                "d_gene": "trbd_gene",
                "j_gene": "trbj_gene",
            }

    def load_and_standardize(self) -> Iterator[pd.DataFrame]:
        # Collect all TSV files
        tsv_files = sorted(self.source_dir.rglob("*.tsv"))

        for fpath in tsv_files:
            chain = self._get_chain_type(fpath)
            column_map = self._get_column_map_for_chain(chain)
            study_id = fpath.stem

            try:
                # Read header to determine available columns, then select
                # only the ones we need to reduce CSV parsing overhead.
                with open(fpath) as f:
                    header_line = f.readline()
                if not header_line.strip():
                    continue
                header_cols = header_line.strip().split("\t")

                # Build minimal set of columns to read
                needed = {"amino_acid"}
                for base, resolved in [
                    ("v_gene", "v_resolved"),
                    ("d_gene", "d_resolved"),
                    ("j_gene", "j_resolved"),
                ]:
                    if base in header_cols:
                        needed.add(base)
                    elif resolved in header_cols:
                        needed.add(resolved)
                if "frame_type" in header_cols:
                    needed.add("frame_type")
                elif "productive" in header_cols:
                    needed.add("productive")
                usecols = [c for c in needed if c in header_cols]

                for chunk in pd.read_csv(
                    fpath,
                    sep="\t",
                    dtype=str,
                    chunksize=CHUNK_SIZE,
                    on_bad_lines="skip",
                    usecols=usecols if usecols else None,
                ):
                    chunk = chunk.fillna("")

                    # Filter to in-frame sequences
                    if "frame_type" in chunk.columns:
                        chunk = chunk[
                            chunk["frame_type"].str.strip().str.lower() == "in"
                        ]
                    elif "productive" in chunk.columns:
                        chunk = chunk[
                            chunk["productive"].str.strip().str.lower().isin(
                                ("true", "t", "1")
                            )
                        ]

                    if chunk.empty:
                        continue

                    # Use v_resolved/j_resolved if v_gene/j_gene not available
                    for base, resolved in [
                        ("v_gene", "v_resolved"),
                        ("d_gene", "d_resolved"),
                        ("j_gene", "j_resolved"),
                    ]:
                        if base not in chunk.columns and resolved in chunk.columns:
                            chunk[base] = chunk[resolved]

                    result, dropped = standardize_dataframe(
                        chunk,
                        column_map,
                        source=self.name,
                        study_id=study_id,
                    )
                    yield result, dropped
            except Exception as e:
                print(f"  Warning: Failed to read {fpath.name}: {e}")


def main():
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Standardize immuneACCESS")
    parser.add_argument("--source-dir", default="data/databases/immuneACCESS")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    standardizer = ImmuneaccessStandardizer(
        source_dir=args.source_dir, output_dir=args.output_dir,
    )
    print(standardizer.run(force=args.force))


if __name__ == "__main__":
    main()
