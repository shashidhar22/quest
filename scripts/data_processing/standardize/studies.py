#!/usr/bin/env python3
"""Studies standardizer.

Studies — 71M rows, heterogeneous formats (10X, immunoSEQ, MiTCR, AIRR,
tcrdist3, etc.) from data/studies/ directory.

Reuses format detection from quest/parsers/streaming_parser.py to auto-detect
format per file and apply appropriate column mapping.
Study ID from directory name (GSE*, ZEN*, etc.).
"""

import sys
from pathlib import Path
from typing import Iterator

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from quest.data.standardization import standardize_dataframe
from scripts.data_processing.standardize._base import BaseStandardizer

CHUNK_SIZE = 500_000

# Common column mappings for different formats
FORMAT_COLUMN_MAPS = {
    "10x": {
        "cdr3": "trb",
        "cdr3_nt": None,  # skip nucleotide
        "v_gene": "trbv_gene",
        "d_gene": "trbd_gene",
        "j_gene": "trbj_gene",
        "chain": None,  # used for detection
    },
    "immunoseq": {
        "amino_acid": "trb",
        "v_gene": "trbv_gene",
        "d_gene": "trbd_gene",
        "j_gene": "trbj_gene",
        "rearrangement": None,
    },
    "airr": {
        "junction_aa": "trb",
        "v_call": "trbv_gene",
        "d_call": "trbd_gene",
        "j_call": "trbj_gene",
        "locus": None,
    },
    "mitcr": {
        "CDR3 amino acid sequence": "trb",
        "V segments": "trbv_gene",
        "D segments": "trbd_gene",
        "J segments": "trbj_gene",
    },
    "generic": {
        "cdr3_aa": "trb",
        "v_gene": "trbv_gene",
        "d_gene": "trbd_gene",
        "j_gene": "trbj_gene",
    },
}


def _detect_format(columns: list[str]) -> str:
    """Detect file format from column names."""
    col_set = set(c.lower() for c in columns)
    if "barcode" in col_set and "chain" in col_set:
        return "10x"
    if "junction_aa" in col_set and "v_call" in col_set:
        return "airr"
    if "rearrangement" in col_set and "amino_acid" in col_set:
        return "immunoseq"
    if any("cdr3 amino acid" in c.lower() for c in columns):
        return "mitcr"
    return "generic"


def _build_column_map(columns: list[str], fmt: str) -> dict:
    """Build column map based on detected format and actual columns."""
    templates = FORMAT_COLUMN_MAPS.get(fmt, FORMAT_COLUMN_MAPS["generic"])
    col_map = {}

    for src_template, target in templates.items():
        if target is None:
            continue
        # Find matching column (case-insensitive)
        for col in columns:
            if col.lower() == src_template.lower():
                col_map[col] = target
                break

    # Also look for paired alpha/beta columns
    for col in columns:
        cl = col.lower()
        if cl in ("cdr3_aa_alpha", "cdr3a", "cdr3_alpha", "junction_aa_alpha"):
            col_map[col] = "tra"
        elif cl in ("v_call_alpha", "trav", "v_alpha"):
            col_map[col] = "trav_gene"
        elif cl in ("j_call_alpha", "traj", "j_alpha"):
            col_map[col] = "traj_gene"
        elif cl in ("cdr3_aa_beta", "cdr3b", "cdr3_beta", "junction_aa_beta"):
            col_map[col] = "trb"
        elif cl in ("v_call_beta", "trbv", "v_beta"):
            col_map[col] = "trbv_gene"
        elif cl in ("d_call_beta", "trbd", "d_beta"):
            col_map[col] = "trbd_gene"
        elif cl in ("j_call_beta", "trbj", "j_beta"):
            col_map[col] = "trbj_gene"
        elif cl in ("epitope", "antigen", "peptide"):
            col_map[col] = "peptide"

    return col_map


class StudiesStandardizer(BaseStandardizer):
    name = "studies"
    streaming = True

    def __init__(self, source_dir=None, output_dir=None):
        # Studies are typically in data/studies/ not data/databases/studies/
        if source_dir is None:
            source_dir = Path("data/studies")
        super().__init__(source_dir=source_dir, output_dir=output_dir)

    # Column maps for alpha vs beta chains
    _ALPHA_MAP_10X = {
        "cdr3": "tra",
        "v_gene": "trav_gene",
        "d_gene": "trad_gene",
        "j_gene": "traj_gene",
    }
    _BETA_MAP_10X = {
        "cdr3": "trb",
        "v_gene": "trbv_gene",
        "d_gene": "trbd_gene",
        "j_gene": "trbj_gene",
    }
    _ALPHA_MAP_AIRR = {
        "junction_aa": "tra",
        "v_call": "trav_gene",
        "d_call": "trad_gene",
        "j_call": "traj_gene",
    }
    _BETA_MAP_AIRR = {
        "junction_aa": "trb",
        "v_call": "trbv_gene",
        "d_call": "trbd_gene",
        "j_call": "trbj_gene",
    }

    def get_column_map(self) -> dict:
        return {}  # Determined per-file

    def _split_by_chain(
        self, chunk, chain_col, default_col_map, study_id,
    ) -> Iterator[tuple]:
        """Split 10x data by chain column value (TRA/TRB)."""
        for chain_val, group in chunk.groupby(chain_col):
            cv = str(chain_val).strip().upper()
            if cv == "TRA":
                col_map = _build_column_map(group.columns.tolist(), "10x")
                # Override CDR3/gene mappings to alpha
                for src, tgt in self._ALPHA_MAP_10X.items():
                    for c in group.columns:
                        if c.lower() == src:
                            col_map[c] = tgt
                            break
            elif cv == "TRB":
                col_map = _build_column_map(group.columns.tolist(), "10x")
                for src, tgt in self._BETA_MAP_10X.items():
                    for c in group.columns:
                        if c.lower() == src:
                            col_map[c] = tgt
                            break
            else:
                # Skip non-alpha/beta chains (IGH, IGK, IGL, TRD, TRG)
                continue
            if not col_map:
                continue
            result, dropped = standardize_dataframe(
                group, col_map, source=self.name, study_id=study_id,
            )
            yield result, dropped

    def _split_by_locus(
        self, chunk, default_col_map, study_id,
    ) -> Iterator[tuple]:
        """Split AIRR data by locus column value (TRA/TRB)."""
        for locus_val, group in chunk.groupby("locus"):
            lv = str(locus_val).strip().upper()
            if lv == "TRA":
                col_map = dict(default_col_map)
                for src, tgt in self._ALPHA_MAP_AIRR.items():
                    for c in group.columns:
                        if c.lower() == src:
                            col_map[c] = tgt
                            break
            elif lv == "TRB":
                col_map = dict(default_col_map)
                for src, tgt in self._BETA_MAP_AIRR.items():
                    for c in group.columns:
                        if c.lower() == src:
                            col_map[c] = tgt
                            break
            elif lv in ("", "TRD", "TRG"):
                # Use default map for empty locus; skip gamma-delta
                if lv == "":
                    result, dropped = standardize_dataframe(
                        group, default_col_map, source=self.name, study_id=study_id,
                    )
                    yield result, dropped
                continue
            else:
                continue
            if not col_map:
                continue
            result, dropped = standardize_dataframe(
                group, col_map, source=self.name, study_id=study_id,
            )
            yield result, dropped

    def load_and_standardize(self) -> Iterator[pd.DataFrame]:
        if not self.source_dir.exists():
            return

        # Find study directories
        study_dirs = sorted(
            d
            for d in self.source_dir.iterdir()
            if d.is_dir() and not d.name.startswith((".", "logs", "geo"))
        )

        for study_dir in study_dirs:
            study_id = study_dir.name

            # Find data files
            data_files = []
            for ext in ("*.csv", "*.tsv", "*.txt"):
                data_files.extend(sorted(study_dir.rglob(ext)))

            for fpath in data_files:
                # Skip metadata/manifest files
                if any(
                    skip in fpath.name.lower()
                    for skip in ("metadata", "manifest", "readme", "log", "summary")
                ):
                    continue

                # Detect separator
                sep = "\t" if fpath.suffix in (".tsv", ".txt") else ","

                try:
                    # Read header to detect format
                    header_df = pd.read_csv(fpath, sep=sep, nrows=0, dtype=str)
                    fmt = _detect_format(header_df.columns.tolist())
                    col_map = _build_column_map(header_df.columns.tolist(), fmt)

                    if not col_map:
                        continue  # Can't identify relevant columns

                    for chunk in pd.read_csv(
                        fpath,
                        sep=sep,
                        dtype=str,
                        chunksize=CHUNK_SIZE,
                        on_bad_lines="skip",
                    ):
                        chunk = chunk.fillna("")

                        # Filter productive if available (exclude empty string)
                        if "productive" in chunk.columns:
                            chunk = chunk[
                                chunk["productive"]
                                .str.strip()
                                .str.lower()
                                .isin(("true", "t", "1"))
                            ]

                        # Filter to TCR loci only (exclude IG/BCR)
                        if "locus" in chunk.columns:
                            chunk = chunk[
                                chunk["locus"].str.strip().str.upper().isin(
                                    ("TRA", "TRB", "TRD", "TRG", "")
                                )
                            ]

                        if chunk.empty:
                            continue

                        # 10x format: split by chain column for correct alpha/beta mapping
                        if fmt == "10x" and "chain" in [c.lower() for c in chunk.columns]:
                            chain_col = next(
                                c for c in chunk.columns if c.lower() == "chain"
                            )
                            yield from self._split_by_chain(
                                chunk, chain_col, col_map, study_id,
                            )
                        # AIRR format: split by locus column
                        elif fmt == "airr" and "locus" in chunk.columns:
                            yield from self._split_by_locus(
                                chunk, col_map, study_id,
                            )
                        else:
                            result, dropped = standardize_dataframe(
                                chunk,
                                col_map,
                                source=self.name,
                                study_id=study_id,
                            )
                            yield result, dropped
                except Exception as e:
                    print(f"  Warning: Failed to read {fpath}: {e}")


def main():
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Standardize studies")
    parser.add_argument("--source-dir", default="data/studies")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    standardizer = StudiesStandardizer(
        source_dir=args.source_dir, output_dir=args.output_dir,
    )
    print(standardizer.run(force=args.force))


if __name__ == "__main__":
    main()
