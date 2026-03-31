#!/usr/bin/env python3
"""Studies standardizer.

Studies — heterogeneous formats (10X, immunoSEQ, MiTCR, MiXCR, AIRR,
tcrdist3, BGI, etc.) from data/studies/ directory.

Reuses format detection from quest/parsers/streaming_parser.py to auto-detect
format per file and apply appropriate column mapping.
Study ID from directory name (GSE*, ZEN*, etc.).
"""

import gzip
import re
import sys
from pathlib import Path
from typing import Iterator, Optional

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
    "immunoseq_v2": {
        "aminoAcid": "trb",
        "vGeneName": "trbv_gene",
        "dGeneName": "trbd_gene",
        "jGeneName": "trbj_gene",
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
    "mixcr_clone": {
        "cdr3aa": "trb",
        "v": "trbv_gene",
        "d": "trbd_gene",
        "j": "trbj_gene",
    },
    "mixcr_full": {
        "aaSeqCDR3": "trb",
        "allVHitsWithScore": "trbv_gene",
        "allDHitsWithScore": "trbd_gene",
        "allJHitsWithScore": "trbj_gene",
    },
    "vjcombo": {
        "aaCDR3": "trb",
        "VGene": "trbv_gene",
        "JGene": "trbj_gene",
    },
    "bgi_structure": {
        "CDR3(aa)": "trb",
        "V_ref": "trbv_gene",
        "D_ref": "trbd_gene",
        "J_ref": "trbj_gene",
    },
    "generic": {
        "cdr3_aa": "trb",
        "v_gene": "trbv_gene",
        "d_gene": "trbd_gene",
        "j_gene": "trbj_gene",
    },
}

# Alpha-chain variants for chain-aware formats
_ALPHA_COLUMN_MAPS = {
    "immunoseq_v2": {
        "aminoAcid": "tra",
        "vGeneName": "trav_gene",
        "dGeneName": "trad_gene",
        "jGeneName": "traj_gene",
    },
    "mixcr_clone": {
        "cdr3aa": "tra",
        "v": "trav_gene",
        "d": "trad_gene",
        "j": "traj_gene",
    },
    "mixcr_full": {
        "aaSeqCDR3": "tra",
        "allVHitsWithScore": "trav_gene",
        "allDHitsWithScore": "trad_gene",
        "allJHitsWithScore": "traj_gene",
    },
    "vjcombo": {
        "aaCDR3": "tra",
        "VGene": "trav_gene",
        "JGene": "traj_gene",
    },
    "bgi_structure": {
        "CDR3(aa)": "tra",
        "V_ref": "trav_gene",
        "D_ref": "trad_gene",
        "J_ref": "traj_gene",
    },
    "mitcr": {
        "CDR3 amino acid sequence": "tra",
        "V segments": "trav_gene",
        "D segments": "trad_gene",
        "J segments": "traj_gene",
    },
}


def _detect_format(columns: list[str]) -> str:
    """Detect file format from column names."""
    col_set = set(c.lower() for c in columns)

    # 10x with chain column
    if "barcode" in col_set and "chain" in col_set:
        return "10x"

    # 10x clonotype (semicolon-separated chains)
    if "cdr3s_aa" in col_set:
        return "10x_clonotype"

    # AIRR
    if "junction_aa" in col_set and "v_call" in col_set:
        return "airr"

    # immunoSEQ v1 (snake_case)
    if "rearrangement" in col_set and "amino_acid" in col_set:
        return "immunoseq"

    # immunoSEQ v2 (camelCase)
    if "aminoacid" in col_set and "nucleotide" in col_set:
        return "immunoseq_v2"

    # MiTCR full export
    if any("cdr3 amino acid" in c.lower() for c in columns):
        return "mitcr"

    # MiXCR clone format (cdr3aa without underscore)
    if "cdr3aa" in col_set and "cdr3nt" in col_set:
        return "mixcr_clone"

    # MiXCR full export (aaSeqCDR3 column with scored gene hits)
    if "aaseqcdr3" in col_set and "allvhitswithscore" in col_set:
        return "mixcr_full"

    # VJCombo format (aaCDR3 + VGene columns)
    if "aacdr3" in col_set and "vgene" in col_set:
        return "vjcombo"

    # BGI .structure format
    if "cdr3(aa)" in col_set and "v_ref" in col_set:
        return "bgi_structure"

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


def _build_alpha_column_map(columns: list[str], fmt: str) -> dict:
    """Build alpha-chain column map for formats that default to beta."""
    templates = _ALPHA_COLUMN_MAPS.get(fmt)
    if templates is None:
        return {}
    col_map = {}
    for src_template, target in templates.items():
        for col in columns:
            if col.lower() == src_template.lower():
                col_map[col] = target
                break
    return col_map


def _infer_chain_from_path(fpath: Path) -> Optional[str]:
    """Infer chain type (TRA/TRB) from file path or directory structure."""
    # Filename-based detection (checked FIRST — more specific than directory)
    stem = fpath.stem.lower()
    # Handle .csv.gz etc — get the real stem
    if stem.endswith((".csv", ".tsv", ".txt")):
        stem = Path(stem).stem.lower()

    if "clones_tra" in stem or "_alpha" in stem or "_tra_" in stem or stem.endswith("_tra"):
        return "TRA"
    if "clones_trb" in stem or "_beta" in stem or "_trb_" in stem or stem.endswith("_trb"):
        return "TRB"

    # TCR##A / TCR##B pattern (e.g. GSM2092506_TCR45A.BC3.productive.tsv)
    tcr_match = re.search(r"tcr\d+([ab])\b", stem)
    if tcr_match:
        return "TRA" if tcr_match.group(1) == "a" else "TRB"

    # Directory-based detection (fallback — less specific than filename)
    path_str = str(fpath).lower()
    if "bulk_survey_tra" in path_str or "_tra/" in path_str or "_tra\\" in path_str:
        return "TRA"
    if "bulk_survey_trb" in path_str or "_trb/" in path_str or "_trb\\" in path_str:
        return "TRB"

    return None


def _infer_chain_from_mitcr_header(first_line: str) -> Optional[str]:
    """Infer chain type from MiTCR metadata header line."""
    upper = first_line.upper()
    if "TRA" in upper:
        return "TRA"
    if "TRB" in upper:
        return "TRB"
    return None


def _parse_10x_clonotype_chunk(chunk: pd.DataFrame, study_id: str) -> Iterator[tuple]:
    """Parse 10x clonotype format with semicolon-separated chains in cdr3s_aa.

    Data format: TRB:CASSPPTKETQYF;TRA:CIVRVSGSNFGNEKLTF
    """
    from quest.data.standardization import TARGET_COLUMNS

    # Find the cdr3s_aa column (case-insensitive)
    cdr3s_col = next(
        (c for c in chunk.columns if c.lower() == "cdr3s_aa"), None,
    )
    if cdr3s_col is None:
        return

    rows = []
    for _, row in chunk.iterrows():
        cdr3s_val = str(row.get(cdr3s_col, "")).strip()
        if not cdr3s_val or cdr3s_val in ("", "nan", "None"):
            continue

        tra_seq = ""
        trb_seq = ""

        # Parse semicolon-separated CHAIN:SEQUENCE pairs
        for part in cdr3s_val.split(";"):
            part = part.strip()
            if ":" in part:
                chain, seq = part.split(":", 1)
                chain = chain.strip().upper()
                seq = seq.strip()
                if chain == "TRA":
                    tra_seq = seq
                elif chain == "TRB":
                    trb_seq = seq

        if tra_seq or trb_seq:
            rows.append({
                "tra": tra_seq,
                "trav_gene": "",
                "trad_gene": "",
                "traj_gene": "",
                "trb": trb_seq,
                "trbv_gene": "",
                "trbd_gene": "",
                "trbj_gene": "",
                "peptide": "",
                "mhc_one": "",
                "mhc_two": "",
                "binding": "",
                "score": "",
                "source": "studies",
                "study_id": study_id,
            })

    if rows:
        result = pd.DataFrame(rows, columns=TARGET_COLUMNS)
        dropped = pd.DataFrame(
            columns=["reason", "source_file", "row_index", "field", "raw_value"],
        )
        yield result, dropped


class StudiesStandardizer(BaseStandardizer):
    name = "studies"
    streaming = True

    def __init__(self, source_dir=None, output_dir=None, stitch=False, hla_dir=""):
        # Studies are typically in data/studies/ not data/databases/studies/
        if source_dir is None:
            source_dir = Path("data/studies")
        super().__init__(source_dir=source_dir, output_dir=output_dir, stitch=stitch, hla_dir=hla_dir)
        # Track files that couldn't be parsed
        self._skipped_files: list[tuple] = []

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

    def get_file_list(self) -> list[Path]:
        if not self.source_dir.exists():
            return []
        files = []
        study_dirs = sorted(
            d
            for d in self.source_dir.iterdir()
            if d.is_dir() and not d.name.startswith((".", "logs", "geo"))
        )
        for study_dir in study_dirs:
            for ext in (
                "*.csv", "*.tsv", "*.txt", "*.structure",
                "*.csv.gz", "*.tsv.gz", "*.txt.gz",
            ):
                for fpath in sorted(study_dir.rglob(ext)):
                    if not any(
                        skip in fpath.name.lower()
                        for skip in ("metadata", "manifest", "readme", "log", "summary")
                    ):
                        files.append(fpath)
        return files

    def process_file(self, file_path: Path) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """Process a single study file (used by parallel_run)."""
        # Infer study_id from parent directory name
        study_id = file_path.parent.name
        # Walk up if we're in a nested subdirectory until we find a study-like dir
        for parent in file_path.parents:
            if parent == self.source_dir:
                break
            if parent.parent == self.source_dir:
                study_id = parent.name
                break
        yield from self._process_file(file_path, study_id)

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
                group, col_map, source=self.name, study_id=study_id, stitch=self.stitch,
                hla_dir=self.hla_dir,
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
                        group, default_col_map, source=self.name, study_id=study_id, stitch=self.stitch,
                        hla_dir=self.hla_dir,
                    )
                    yield result, dropped
                continue
            else:
                continue
            if not col_map:
                continue
            result, dropped = standardize_dataframe(
                group, col_map, source=self.name, study_id=study_id, stitch=self.stitch,
                hla_dir=self.hla_dir,
            )
            yield result, dropped

    def _write_skipped_files(self):
        """Write skipped files log to output directory."""
        if not self._skipped_files:
            return
        skipped_path = self.output_dir / "skipped_files.tsv"
        df = pd.DataFrame(
            self._skipped_files,
            columns=["file_path", "detected_format", "columns_sample"],
        )
        df.to_csv(skipped_path, sep="\t", index=False)
        print(f"  Wrote {len(self._skipped_files)} skipped files to {skipped_path}")

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

            # Find data files (including .structure, .gz)
            data_files = []
            for ext in (
                "*.csv", "*.tsv", "*.txt", "*.structure",
                "*.csv.gz", "*.tsv.gz", "*.txt.gz",
            ):
                data_files.extend(sorted(study_dir.rglob(ext)))

            for fpath in data_files:
                # Skip metadata/manifest files
                if any(
                    skip in fpath.name.lower()
                    for skip in ("metadata", "manifest", "readme", "log", "summary")
                ):
                    continue

                yield from self._process_file(fpath, study_id)

        # Write skipped files log after all studies are processed
        self._write_skipped_files()

    def _process_file(
        self, fpath: Path, study_id: str,
    ) -> Iterator[tuple]:
        """Process a single data file, handling format detection and reading."""
        name_lower = fpath.name.lower()
        is_gz = name_lower.endswith(".gz")
        suffix = fpath.suffix.lower()

        # Determine separator based on base extension
        if is_gz:
            base_suffix = Path(name_lower[:-3]).suffix
        else:
            base_suffix = suffix
        sep = "\t" if base_suffix in (".tsv", ".txt", ".structure") else ","

        compression = "gzip" if is_gz else None
        skip_rows = 0
        mitcr_chain = None

        try:
            # Check for MiTCR metadata header line
            first_line = ""
            opener = gzip.open if is_gz else open
            with opener(fpath, "rt", errors="replace") as f:
                first_line = f.readline()

            if first_line.startswith("MiTCR"):
                skip_rows = 1
                mitcr_chain = _infer_chain_from_mitcr_header(first_line)

            # Handle # prefix in column names (e.g., BGI #ID)
            # pandas read_csv with comment='#' would skip these lines,
            # but #ID is actually a column name, not a comment

            # Read header to detect format
            header_df = pd.read_csv(
                fpath, sep=sep, nrows=0, dtype=str,
                compression=compression, skiprows=skip_rows,
            )
            columns = header_df.columns.tolist()
            fmt = _detect_format(columns)

            # Special handling for 10x_clonotype format
            if fmt == "10x_clonotype":
                for chunk in pd.read_csv(
                    fpath, sep=sep, dtype=str, chunksize=CHUNK_SIZE,
                    on_bad_lines="skip", compression=compression,
                    skiprows=skip_rows,
                ):
                    chunk = chunk.fillna("")
                    yield from _parse_10x_clonotype_chunk(chunk, study_id)
                return

            # Determine chain from path for formats that need it
            chain = mitcr_chain or _infer_chain_from_path(fpath)

            # Build column map — use alpha variant if path indicates TRA
            if chain == "TRA" and fmt in _ALPHA_COLUMN_MAPS:
                col_map = _build_alpha_column_map(columns, fmt)
            else:
                col_map = _build_column_map(columns, fmt)

            if not col_map:
                self._skipped_files.append((
                    str(fpath),
                    fmt,
                    ", ".join(columns[:5]),
                ))
                return

            for chunk in pd.read_csv(
                fpath,
                sep=sep,
                dtype=str,
                chunksize=CHUNK_SIZE,
                on_bad_lines="skip",
                compression=compression,
                skiprows=skip_rows,
            ):
                chunk = chunk.fillna("")

                # MiXCR full export: strip scores from gene hit columns
                # e.g. "TRBV7-9*00(911.6)" -> "TRBV7-9*00"
                # Multi-hit: "TRBD1*00(35),TRBD2*00(30)" -> "TRBD1*00"
                if fmt == "mixcr_full":
                    for gene_col in (
                        "allVHitsWithScore", "allDHitsWithScore",
                        "allJHitsWithScore",
                    ):
                        if gene_col in chunk.columns:
                            chunk[gene_col] = (
                                chunk[gene_col]
                                .str.split(",").str[0]
                                .str.split("(").str[0]
                                .str.strip()
                            )

                # Track drops from filters
                drop_records = []

                # Filter by sequenceStatus for immunoSEQ v2 (In = productive)
                if fmt == "immunoseq_v2" and "sequenceStatus" in chunk.columns:
                    before_count = len(chunk)
                    productive_mask = (
                        chunk["sequenceStatus"].str.strip().str.lower() == "in"
                    )
                    non_productive = chunk[~productive_mask]
                    if len(non_productive) > 0:
                        drop_records.extend([{
                            "reason": "non_productive",
                            "source_file": str(fpath),
                            "row_index": "",
                            "field": "sequenceStatus",
                            "raw_value": f"{len(non_productive)} rows with status != In",
                        }])
                    chunk = chunk[productive_mask]

                # Filter productive if available (AIRR/10x format)
                if "productive" in chunk.columns:
                    before_count = len(chunk)
                    productive_mask = (
                        chunk["productive"]
                        .str.strip()
                        .str.lower()
                        .isin(("true", "t", "1"))
                    )
                    non_productive = chunk[~productive_mask]
                    if len(non_productive) > 0:
                        drop_records.extend([{
                            "reason": "non_productive",
                            "source_file": str(fpath),
                            "row_index": "",
                            "field": "productive",
                            "raw_value": f"{len(non_productive)} rows",
                        }])
                    chunk = chunk[productive_mask]

                # Filter to TCR loci only (exclude IG/BCR)
                if "locus" in chunk.columns:
                    before_count = len(chunk)
                    tcr_mask = chunk["locus"].str.strip().str.upper().isin(
                        ("TRA", "TRB", "TRD", "TRG", "")
                    )
                    non_tcr = chunk[~tcr_mask]
                    if len(non_tcr) > 0:
                        drop_records.extend([{
                            "reason": "non_tcr_locus",
                            "source_file": str(fpath),
                            "row_index": "",
                            "field": "locus",
                            "raw_value": f"{len(non_tcr)} IG/BCR rows excluded",
                        }])
                    chunk = chunk[tcr_mask]

                if chunk.empty:
                    # Still yield drop records if we have them
                    if drop_records:
                        empty_result = pd.DataFrame(
                            columns=["tra", "trav_gene", "trad_gene", "traj_gene",
                                     "trb", "trbv_gene", "trbd_gene", "trbj_gene",
                                     "peptide", "mhc_one", "mhc_two",
                                     "binding", "score", "source", "study_id"],
                        )
                        dropped_df = pd.DataFrame(drop_records)
                        yield empty_result, dropped_df
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
                        stitch=self.stitch,
                        hla_dir=self.hla_dir,
                    )
                    # Merge filter drop records into dropped_df
                    if drop_records:
                        extra_dropped = pd.DataFrame(drop_records)
                        if dropped is not None and not dropped.empty:
                            dropped = pd.concat(
                                [dropped, extra_dropped], ignore_index=True,
                            )
                        else:
                            dropped = extra_dropped
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
    parser.add_argument("--hla-dir", default="")
    args = parser.parse_args()

    standardizer = StudiesStandardizer(
        source_dir=args.source_dir, output_dir=args.output_dir,
        hla_dir=args.hla_dir,
    )
    print(standardizer.run(force=args.force))


if __name__ == "__main__":
    main()
