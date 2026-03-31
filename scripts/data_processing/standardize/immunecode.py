#!/usr/bin/env python3
"""ImmuneCODE standardizer.

ImmuneCODE — COVID-19 TCR data (beta-only).
TCR BioIdentity format: "CASSAQGTGDRGYTF+TCRBV27-01+TCRBJ01-02"

Sources:
  - MIRA peptide-detail files (epitope-specific, ~161K rows)
    - peptide-detail-ci.csv (Class I)
    - peptide-detail-cii.csv (Class II)
    - subject-metadata.csv (HLA typing per experiment)
  - Review repertoire files (~38M rows across ~1,400 TSVs)
    - ImmuneCODE-Review-002/*.tsv (bulk TCR-seq, no epitope labels)
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

# Read Review TSVs in chunks of this size
_REVIEW_CHUNK_SIZE = 500_000


def _parse_bio_identity(bio_id: str) -> dict:
    """Parse TCR BioIdentity string.

    Format: "CASSAQGTGDRGYTF+TCRBV27-01+TCRBJ01-02"
    Returns: {'trb': 'CASSAQGTGDRGYTF', 'trbv_gene': 'TCRBV27-01', 'trbj_gene': 'TCRBJ01-02'}
    """
    if not bio_id or str(bio_id).strip() in ("", "nan"):
        return {}
    parts = str(bio_id).strip().split("+")
    result = {}
    if len(parts) >= 1:
        result["trb"] = parts[0].strip()
    if len(parts) >= 2:
        result["trbv_gene"] = parts[1].strip()
    if len(parts) >= 3:
        result["trbj_gene"] = parts[2].strip()
    return result


def _bio_identity_to_df(series: pd.Series) -> pd.DataFrame:
    """Vectorized parse of a bio_identity column into trb/trbv_gene/trbj_gene columns."""
    split = series.str.split("+", expand=True)
    result = pd.DataFrame(index=series.index)
    result["trb"] = split[0].str.strip() if 0 in split.columns else ""
    result["trbv_gene"] = split[1].str.strip() if 1 in split.columns else ""
    result["trbj_gene"] = split[2].str.strip() if 2 in split.columns else ""
    return result


class ImmunecodeStandardizer(BaseStandardizer):
    name = "immunecode"
    streaming = True

    def get_file_list(self) -> list[Path]:
        files = sorted(self.source_dir.rglob("*_TCRB.tsv"))
        # Include MIRA peptide-detail files
        files.extend(sorted(self.source_dir.rglob("peptide-detail-ci.csv")))
        return files

    def process_file(self, file_path: Path) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """Process a single file (used by parallel_run).

        Dispatches to MIRA or Review processing based on filename.
        """
        if file_path.name.startswith("peptide-detail"):
            yield from self._process_mira_file(file_path)
        else:
            yield from self._process_review_file(file_path)

    def _process_mira_file(self, file_path: Path) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """Process a single MIRA peptide-detail CSV."""
        hla_map = self._load_subject_hla()
        column_map = self.get_column_map()

        df = pd.read_csv(file_path, dtype=str).fillna("")
        if "TCR BioIdentity" not in df.columns:
            return
        df = df[df["TCR BioIdentity"].str.strip() != ""].reset_index(drop=True)
        if df.empty:
            return

        parsed = _bio_identity_to_df(df["TCR BioIdentity"])

        if "Amino Acids" in df.columns:
            peptides = df["Amino Acids"].str.strip().str.split(",")
            merged = pd.DataFrame({
                "trb": parsed["trb"].values,
                "trbv_gene": parsed["trbv_gene"].values,
                "trbj_gene": parsed["trbj_gene"].values,
                "peptide": peptides.values,
                "experiment": df["Experiment"].str.strip().values
                if "Experiment" in df.columns else "",
            })
            merged = merged.explode("peptide", ignore_index=True)
            merged["peptide"] = merged["peptide"].str.strip()
        else:
            merged = pd.DataFrame({
                "trb": parsed["trb"].values,
                "trbv_gene": parsed["trbv_gene"].values,
                "trbj_gene": parsed["trbj_gene"].values,
                "peptide": "",
                "experiment": df["Experiment"].str.strip().values
                if "Experiment" in df.columns else "",
            })

        if "peptide" in merged.columns:
            merged = merged[
                merged["peptide"].str.len().le(25) | (merged["peptide"] == "")
            ].reset_index(drop=True)

        if merged.empty:
            return

        if "experiment" in merged.columns:
            merged["mhc_one"] = merged["experiment"].map(
                lambda exp: hla_map.get(exp, [""])[0]
                if exp and hla_map.get(exp) else ""
            )
            merged = merged.drop(columns=["experiment"])
        else:
            merged["mhc_one"] = ""
        merged["mhc_two"] = ""

        result, dropped = standardize_dataframe(
            merged, column_map, source=self.name, stitch=self.stitch,
            hla_dir=self.hla_dir,
        )
        yield result, dropped

    def _process_review_file(self, file_path: Path) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """Process a single Review TSV file."""
        column_map = self.get_column_map()
        study_id = file_path.stem
        usecols = ["bio_identity", "d_gene", "frame_type"]

        try:
            reader = pd.read_csv(
                file_path, sep="\t", dtype=str,
                usecols=usecols, chunksize=_REVIEW_CHUNK_SIZE,
            )
        except (ValueError, KeyError):
            try:
                reader = pd.read_csv(
                    file_path, sep="\t", dtype=str,
                    chunksize=_REVIEW_CHUNK_SIZE,
                )
            except Exception:
                return

        for chunk in reader:
            chunk = chunk.fillna("")

            if "frame_type" in chunk.columns:
                chunk = chunk[chunk["frame_type"] == "In"]

            if chunk.empty:
                continue

            if "bio_identity" not in chunk.columns:
                continue

            parsed = _bio_identity_to_df(chunk["bio_identity"])
            merged = pd.DataFrame(index=parsed.index)
            merged["trb"] = parsed["trb"]
            merged["trbv_gene"] = parsed["trbv_gene"]
            merged["trbj_gene"] = parsed["trbj_gene"]

            if "d_gene" in chunk.columns:
                merged["trbd_gene"] = chunk["d_gene"]

            result, dropped = standardize_dataframe(
                merged, column_map, source=self.name, study_id=study_id, stitch=self.stitch,
                hla_dir=self.hla_dir,
            )
            if not result.empty:
                yield result, dropped

    def get_column_map(self) -> dict:
        return {
            "trb": "trb",
            "trbv_gene": "trbv_gene",
            "trbd_gene": "trbd_gene",
            "trbj_gene": "trbj_gene",
            "peptide": "peptide",
            "mhc_one": "mhc_one",
            "mhc_two": "mhc_two",
        }

    # Class II HLA prefixes — used to filter HLA columns in subject metadata
    _CLASS_II_PREFIXES = ("DP", "DQ", "DR", "DPA", "DPB", "DQA", "DQB", "DRA", "DRB")

    def _load_subject_hla(self) -> dict:
        """Load subject HLA typing from metadata."""
        mira_dir = None
        for d in self.source_dir.rglob("subject-metadata.csv"):
            mira_dir = d.parent
            break

        if not mira_dir:
            return {}

        meta_path = mira_dir / "subject-metadata.csv"
        if not meta_path.exists():
            return {}

        meta = pd.read_csv(meta_path, dtype=str, encoding="latin-1").fillna("")
        # Identify HLA columns: starts with "HLA" or a known Class II prefix
        hla_cols = [
            col for col in meta.columns
            if col.startswith("HLA") or col.startswith(self._CLASS_II_PREFIXES)
        ]
        hla_map = {}
        for _, row in meta.iterrows():
            exp = str(row.get("Experiment", "")).strip()
            if not exp:
                continue
            hla_alleles = []
            for col in hla_cols:
                val = str(row.get(col, "")).strip()
                if val and val.lower() not in ("", "nan", "not tested"):
                    hla_alleles.append(normalize_mhc_allele(val))
            hla_map[exp] = [a for a in hla_alleles if a]
        return hla_map

    def _process_mira(self) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """Process MIRA peptide-detail files (Class I only, epitope-specific TCRs)."""
        # Only use Class I file — Class II contains minigene constructs
        mira_files = list(self.source_dir.rglob("peptide-detail-ci.csv"))
        if not mira_files:
            return

        hla_map = self._load_subject_hla()
        column_map = self.get_column_map()

        for fpath in mira_files:
            df = pd.read_csv(fpath, dtype=str).fillna("")

            # Filter rows with valid TCR BioIdentity
            if "TCR BioIdentity" not in df.columns:
                continue
            df = df[df["TCR BioIdentity"].str.strip() != ""].reset_index(drop=True)
            if df.empty:
                continue

            # Vectorized bio_identity parsing
            parsed = _bio_identity_to_df(df["TCR BioIdentity"])

            # Handle comma-separated peptides: split and explode
            if "Amino Acids" in df.columns:
                peptides = df["Amino Acids"].str.strip()
                # Split comma-separated peptides into lists
                peptides = peptides.str.split(",")
                # Build a merged frame before exploding
                merged = pd.DataFrame({
                    "trb": parsed["trb"].values,
                    "trbv_gene": parsed["trbv_gene"].values,
                    "trbj_gene": parsed["trbj_gene"].values,
                    "peptide": peptides.values,
                    "experiment": df["Experiment"].str.strip().values
                    if "Experiment" in df.columns else "",
                })
                # Explode comma-separated peptides into separate rows
                merged = merged.explode("peptide", ignore_index=True)
                merged["peptide"] = merged["peptide"].str.strip()
            else:
                merged = pd.DataFrame({
                    "trb": parsed["trb"].values,
                    "trbv_gene": parsed["trbv_gene"].values,
                    "trbj_gene": parsed["trbj_gene"].values,
                    "peptide": "",
                    "experiment": df["Experiment"].str.strip().values
                    if "Experiment" in df.columns else "",
                })

            # Filter peptides > 25 AA (minigene constructs)
            if "peptide" in merged.columns:
                merged = merged[
                    merged["peptide"].str.len().le(25) | (merged["peptide"] == "")
                ].reset_index(drop=True)

            if merged.empty:
                continue

            # Map HLA alleles from subject metadata (Class I only)
            if "experiment" in merged.columns:
                merged["mhc_one"] = merged["experiment"].map(
                    lambda exp: hla_map.get(exp, [""])[0]
                    if exp and hla_map.get(exp) else ""
                )
                merged = merged.drop(columns=["experiment"])
            else:
                merged["mhc_one"] = ""

            merged["mhc_two"] = ""

            result, dropped = standardize_dataframe(
                merged, column_map, source=self.name, stitch=self.stitch,
                hla_dir=self.hla_dir,
            )
            yield result, dropped

    def _process_review(self) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """Process Review repertoire TSVs (bulk TCR-seq, no epitope labels)."""
        review_files = sorted(self.source_dir.rglob("*_TCRB.tsv"))
        if not review_files:
            return

        column_map = self.get_column_map()
        usecols = ["bio_identity", "d_gene", "frame_type"]

        for fpath in review_files:
            study_id = fpath.stem  # e.g. "KHBR20-00164_TCRB"
            try:
                reader = pd.read_csv(
                    fpath, sep="\t", dtype=str,
                    usecols=usecols, chunksize=_REVIEW_CHUNK_SIZE,
                )
            except (ValueError, KeyError):
                # Fall back if columns differ
                try:
                    reader = pd.read_csv(
                        fpath, sep="\t", dtype=str,
                        chunksize=_REVIEW_CHUNK_SIZE,
                    )
                except Exception:
                    continue

            for chunk in reader:
                chunk = chunk.fillna("")

                # Filter to productive rearrangements only
                if "frame_type" in chunk.columns:
                    chunk = chunk[chunk["frame_type"] == "In"]

                if chunk.empty:
                    continue

                # Parse bio_identity into trb/trbv/trbj
                if "bio_identity" not in chunk.columns:
                    continue

                parsed = _bio_identity_to_df(chunk["bio_identity"])
                merged = pd.DataFrame(index=parsed.index)
                merged["trb"] = parsed["trb"]
                merged["trbv_gene"] = parsed["trbv_gene"]
                merged["trbj_gene"] = parsed["trbj_gene"]

                # D gene from dedicated column
                if "d_gene" in chunk.columns:
                    merged["trbd_gene"] = chunk["d_gene"]

                result, dropped = standardize_dataframe(
                    merged, column_map, source=self.name, study_id=study_id, stitch=self.stitch,
                    hla_dir=self.hla_dir,
                )
                if not result.empty:
                    yield result, dropped

    def load_and_standardize(self) -> Iterator[pd.DataFrame]:
        # Phase 1: MIRA epitope-specific data
        yield from self._process_mira()

        # Phase 2: Review repertoire data (streaming)
        yield from self._process_review()


def main():
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Standardize ImmuneCODE")
    parser.add_argument("--source-dir", default="data/databases/immuneCODE")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--hla-dir", default="")
    args = parser.parse_args()

    standardizer = ImmunecodeStandardizer(
        source_dir=args.source_dir, output_dir=args.output_dir,
        hla_dir=args.hla_dir,
    )
    print(standardizer.run(force=args.force))


if __name__ == "__main__":
    main()
