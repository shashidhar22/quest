#!/usr/bin/env python3
"""TCRdb standardizer.

TCRdb — 316M rows, TSV/CSV files organized by disease category.
Primarily beta-chain only.

Structure: tcrdb/{category}/{disease}/{files}.csv
Columns: NNSeq, AASeq, Vregion, Dregion, Jregion, cloneCount, cloneFraction
Metadata: tcrdb_all_metadata_combined.csv
"""

import sys
from pathlib import Path
from typing import Iterator

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from quest.data.standardization import standardize_dataframe
from scripts.data_processing.standardize._base import BaseStandardizer

CHUNK_SIZE = 1_000_000

# Categories in TCRdb directory structure
CATEGORIES = [
    "cancer",
    "autoimmunity",
    "healthy",
    "inflammation",
    "viral",
    "transplantation",
]


class TcrdbStandardizer(BaseStandardizer):
    name = "tcrdb"
    streaming = True

    def get_column_map(self) -> dict:
        return {
            "AASeq": "trb",
            "Vregion": "trbv_gene",
            "Dregion": "trbd_gene",
            "Jregion": "trbj_gene",
        }

    def get_file_list(self) -> list[Path]:
        data_files = []
        for category in CATEGORIES:
            cat_dir = self.source_dir / category
            if cat_dir.exists():
                data_files.extend(sorted(cat_dir.rglob("*.csv")))
                data_files.extend(sorted(cat_dir.rglob("*.tsv")))
        return [f for f in data_files if "metadata" not in f.name.lower()]

    def process_file(self, file_path: Path) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """Process a single TCRdb file (used by parallel_run)."""
        column_map = self.get_column_map()
        study_id = file_path.stem
        sep = "\t" if file_path.suffix == ".tsv" else ","

        try:
            for chunk in pd.read_csv(
                file_path,
                sep=sep,
                dtype=str,
                chunksize=CHUNK_SIZE,
                on_bad_lines="skip",
            ):
                chunk = chunk.fillna("")

                chain_col = None
                for c in chunk.columns:
                    if c.lower() == "chain":
                        chain_col = c
                        break

                if chain_col:
                    is_alpha = chunk[chain_col].apply(
                        lambda x: self._detect_chain_from_column(x) == "TRA"
                    )
                elif "Vregion" in chunk.columns:
                    is_alpha = chunk["Vregion"].apply(
                        lambda x: self._detect_chain(x) == "TRA"
                    )
                else:
                    is_alpha = None

                if is_alpha is not None:
                    if is_alpha.any():
                        alpha_df = chunk[is_alpha].copy()
                        alpha_map = {
                            "AASeq": "tra",
                            "Vregion": "trav_gene",
                            "Dregion": "trad_gene",
                            "Jregion": "traj_gene",
                        }
                        result_a, dropped_a = standardize_dataframe(
                            alpha_df, alpha_map, source=self.name,
                            study_id=study_id, stitch=self.stitch,
                            hla_dir=self.hla_dir,
                        )
                        yield result_a, dropped_a

                    beta_df = chunk[~is_alpha]
                    if not beta_df.empty:
                        result_b, dropped_b = standardize_dataframe(
                            beta_df, column_map, source=self.name,
                            study_id=study_id, stitch=self.stitch,
                            hla_dir=self.hla_dir,
                        )
                        yield result_b, dropped_b
                else:
                    result, dropped = standardize_dataframe(
                        chunk, column_map, source=self.name,
                        study_id=study_id, stitch=self.stitch,
                        hla_dir=self.hla_dir,
                    )
                    yield result, dropped
        except Exception as e:
            print(f"  Warning: Failed to read {file_path}: {e}")

    def _detect_chain(self, vregion: str) -> str:
        """Detect chain type from V gene prefix."""
        v = str(vregion).strip().upper()
        if v.startswith("TRA") or v.startswith("TCRA"):
            return "TRA"
        return "TRB"  # Default to beta

    def _detect_chain_from_column(self, chain_val: str) -> str:
        """Detect chain type from a Chain column value."""
        cv = str(chain_val).strip().upper()
        if cv in ("ALPHA", "TRA", "A"):
            return "TRA"
        return "TRB"  # Default to beta

    def load_and_standardize(self) -> Iterator[pd.DataFrame]:
        column_map = self.get_column_map()

        # Find all CSV files in category subdirectories
        data_files = []
        for category in CATEGORIES:
            cat_dir = self.source_dir / category
            if cat_dir.exists():
                data_files.extend(sorted(cat_dir.rglob("*.csv")))

        # Also check for TSV files
        for category in CATEGORIES:
            cat_dir = self.source_dir / category
            if cat_dir.exists():
                data_files.extend(sorted(cat_dir.rglob("*.tsv")))

        for fpath in data_files:
            # Skip metadata files
            if "metadata" in fpath.name.lower():
                continue

            study_id = fpath.stem
            sep = "\t" if fpath.suffix == ".tsv" else ","

            try:
                for chunk in pd.read_csv(
                    fpath,
                    sep=sep,
                    dtype=str,
                    chunksize=CHUNK_SIZE,
                    on_bad_lines="skip",
                ):
                    chunk = chunk.fillna("")

                    # Detect chain type: prefer Chain column, fallback to V-gene prefix
                    chain_col = None
                    for c in chunk.columns:
                        if c.lower() == "chain":
                            chain_col = c
                            break

                    if chain_col:
                        is_alpha = chunk[chain_col].apply(
                            lambda x: self._detect_chain_from_column(x) == "TRA"
                        )
                    elif "Vregion" in chunk.columns:
                        is_alpha = chunk["Vregion"].apply(
                            lambda x: self._detect_chain(x) == "TRA"
                        )
                    else:
                        is_alpha = None

                    if is_alpha is not None:
                        # For alpha chains, remap columns
                        if is_alpha.any():
                            alpha_df = chunk[is_alpha].copy()
                            alpha_map = {
                                "AASeq": "tra",
                                "Vregion": "trav_gene",
                                "Dregion": "trad_gene",
                                "Jregion": "traj_gene",
                            }
                            result_a, dropped_a = standardize_dataframe(
                                alpha_df,
                                alpha_map,
                                source=self.name,
                                study_id=study_id,
                                stitch=self.stitch,
                                hla_dir=self.hla_dir,
                            )
                            yield result_a, dropped_a

                        # Beta chains
                        beta_df = chunk[~is_alpha]
                        if not beta_df.empty:
                            result_b, dropped_b = standardize_dataframe(
                                beta_df,
                                column_map,
                                source=self.name,
                                study_id=study_id,
                                stitch=self.stitch,
                                hla_dir=self.hla_dir,
                            )
                            yield result_b, dropped_b
                    else:
                        result, dropped = standardize_dataframe(
                            chunk,
                            column_map,
                            source=self.name,
                            study_id=study_id,
                            stitch=self.stitch,
                            hla_dir=self.hla_dir,
                        )
                        yield result, dropped
            except Exception as e:
                print(f"  Warning: Failed to read {fpath}: {e}")


def main():
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Standardize TCRdb")
    parser.add_argument("--source-dir", default="data/databases/tcrdb")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--hla-dir", default="")
    args = parser.parse_args()

    standardizer = TcrdbStandardizer(
        source_dir=args.source_dir, output_dir=args.output_dir,
        hla_dir=args.hla_dir,
    )
    print(standardizer.run(force=args.force))


if __name__ == "__main__":
    main()
