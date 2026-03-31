#!/usr/bin/env python3
"""ADC (AIRR Data Commons) standardizer.

ADC — 3B rows, AIRR standard TSV files across multiple iReceptor repositories.
Chain determined by 'locus' field. Pairing via 'cell_id' where available.

Structure:
  adc/{repository}/rearrangements/{id}.tsv
  adc/{repository}/repertoires.json (metadata)

AIRR columns: junction_aa, v_call, d_call, j_call, locus, productive, cell_id
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


class AdcStandardizer(BaseStandardizer):
    name = "adc"
    streaming = True

    def get_file_list(self) -> list[Path]:
        files = []
        for repo_dir in sorted(self.source_dir.iterdir()):
            if not repo_dir.is_dir() or repo_dir.name.startswith("."):
                continue
            rearr_dir = repo_dir / "rearrangements"
            if rearr_dir.exists():
                files.extend(sorted(rearr_dir.glob("*.tsv")))
        return files

    def process_file(self, file_path: Path) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """Process a single ADC TSV file (used by parallel_run)."""
        repo_dir = file_path.parent.parent
        study_map = self._load_repertoire_study_ids(repo_dir)
        repertoire_id = file_path.stem
        study_id = study_map.get(repertoire_id, repo_dir.name)

        try:
            for chunk in pd.read_csv(
                file_path,
                sep="\t",
                dtype=str,
                chunksize=CHUNK_SIZE,
                on_bad_lines="skip",
            ):
                chunk = chunk.fillna("")

                if "productive" in chunk.columns:
                    chunk = chunk[
                        chunk["productive"].str.strip().str.lower().isin(
                            ("true", "t", "1")
                        )
                    ]

                if chunk.empty:
                    continue

                if "locus" in chunk.columns:
                    for locus, group in chunk.groupby("locus"):
                        locus = str(locus).strip().upper()
                        if locus not in ("TRA", "TRB"):
                            continue
                        col_map = self._get_column_map_for_locus(locus)
                        result, dropped = standardize_dataframe(
                            group,
                            col_map,
                            source=self.name,
                            study_id=study_id,
                            stitch=self.stitch,
                            hla_dir=self.hla_dir,
                        )
                        yield result, dropped
                else:
                    col_map = self._get_column_map_for_locus("TRB")
                    result, dropped = standardize_dataframe(
                        chunk,
                        col_map,
                        source=self.name,
                        study_id=study_id,
                        stitch=self.stitch,
                        hla_dir=self.hla_dir,
                    )
                    yield result, dropped
        except Exception as e:
            print(f"  Warning: Failed to read {file_path}: {e}")

    def get_column_map(self) -> dict:
        return {}

    def _get_column_map_for_locus(self, locus: str) -> dict | None:
        if locus == "TRA":
            return {
                "junction_aa": "tra",
                "v_call": "trav_gene",
                "d_call": "trad_gene",
                "j_call": "traj_gene",
            }
        elif locus == "TRB":
            return {
                "junction_aa": "trb",
                "v_call": "trbv_gene",
                "d_call": "trbd_gene",
                "j_call": "trbj_gene",
            }
        else:
            # Skip gamma-delta (TRD, TRG) and unknown loci
            return None

    def _load_repertoire_study_ids(self, repo_dir: Path) -> dict:
        """Load repertoire_id → study_id mapping from repertoires.json."""
        rep_path = repo_dir / "repertoires.json"
        if not rep_path.exists():
            return {}
        try:
            with open(rep_path) as f:
                data = json.load(f)
            mapping = {}
            for rep in data.get("Repertoire", []):
                rep_id = str(rep.get("repertoire_id", ""))
                study = rep.get("study", {})
                study_id = study.get("study_id", "")
                if rep_id:
                    mapping[rep_id] = study_id
            return mapping
        except Exception:
            return {}

    def load_and_standardize(self) -> Iterator[pd.DataFrame]:
        # Find all repository directories
        repo_dirs = [
            d
            for d in sorted(self.source_dir.iterdir())
            if d.is_dir() and not d.name.startswith(".")
        ]

        for repo_dir in repo_dirs:
            rearr_dir = repo_dir / "rearrangements"
            if not rearr_dir.exists():
                continue

            study_map = self._load_repertoire_study_ids(repo_dir)
            tsv_files = sorted(rearr_dir.glob("*.tsv"))

            for fpath in tsv_files:
                repertoire_id = fpath.stem
                study_id = study_map.get(repertoire_id, repo_dir.name)

                try:
                    for chunk in pd.read_csv(
                        fpath,
                        sep="\t",
                        dtype=str,
                        chunksize=CHUNK_SIZE,
                        on_bad_lines="skip",
                    ):
                        chunk = chunk.fillna("")

                        # Filter to productive rearrangements
                        if "productive" in chunk.columns:
                            chunk = chunk[
                                chunk["productive"].str.strip().str.lower().isin(
                                    ("true", "t", "1")
                                )
                            ]

                        if chunk.empty:
                            continue

                        # Split by locus
                        if "locus" in chunk.columns:
                            for locus, group in chunk.groupby("locus"):
                                locus = str(locus).strip().upper()
                                # Only process alpha-beta TCR loci (skip TRD/TRG)
                                if locus not in ("TRA", "TRB"):
                                    continue
                                col_map = self._get_column_map_for_locus(locus)
                                result, dropped = standardize_dataframe(
                                    group,
                                    col_map,
                                    source=self.name,
                                    study_id=study_id,
                                    stitch=self.stitch,
                                    hla_dir=self.hla_dir,
                                )
                                yield result, dropped
                        else:
                            # Default to TRB
                            col_map = self._get_column_map_for_locus("TRB")
                            result, dropped = standardize_dataframe(
                                chunk,
                                col_map,
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

    parser = argparse.ArgumentParser(description="Standardize ADC")
    parser.add_argument("--source-dir", default="data/databases/adc")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--hla-dir", default="")
    args = parser.parse_args()

    standardizer = AdcStandardizer(
        source_dir=args.source_dir, output_dir=args.output_dir,
        hla_dir=args.hla_dir,
    )
    print(standardizer.run(force=args.force))


if __name__ == "__main__":
    main()
