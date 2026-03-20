#!/usr/bin/env python3
"""TRAIT standardizer.

TRAIT — 4M rows, filename-encoded metadata.
MHC and epitope parsed from filename pattern:
    {MHC}_{epitope}_{protein}_{disease}_binder_{pos/neg}.zip

Example: A0201_GLCTLVAML_BMLF1_EBV_binder_pos.zip
MHC conversion: A0201 → HLA-A*02:01

Files:
  - epitopes/binding/*.zip (positive binders)
  - epitopes/non_binding/*.zip (negative binders)
  - main/Omics.zip (nested per-epitope .txt files)
"""

import re
import sys
import zipfile
from pathlib import Path
from typing import Iterator

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from quest.data.standardization import (
    normalize_mhc_allele,
    standardize_dataframe,
)
from scripts.data_processing.standardize._base import BaseStandardizer


def _parse_trait_filename(filename: str) -> dict:
    """Parse TRAIT filename to extract MHC, epitope, protein, disease.

    Examples:
        A0201_GLCTLVAML_BMLF1_EBV_binder_pos.zip
        B0801_FLRGRAYGL_EBNA-3A_EBV_binder_pos.zip
    """
    # Remove extension
    name = filename.replace(".zip", "").replace(".txt", "")
    parts = name.split("_")
    if len(parts) < 4:
        return {}

    mhc_compact = parts[0]
    epitope = parts[1]
    # Convert compact MHC: A0201 → HLA-A*02:01
    m = re.match(r"^([ABC])(\d{2})(\d{2})$", mhc_compact)
    if m:
        mhc = f"HLA-{m.group(1)}*{m.group(2)}:{m.group(3)}"
    else:
        mhc = normalize_mhc_allele(mhc_compact)

    # Extract binder status (pos/neg) from filename
    binding = ""
    if "pos" in parts:
        binding = "pos"
    elif "neg" in parts:
        binding = "neg"

    return {
        "mhc_compact": mhc_compact,
        "mhc": mhc,
        "epitope": epitope,
        "protein": parts[2] if len(parts) > 2 else "",
        "disease": parts[3] if len(parts) > 3 else "",
        "binding": binding,
    }


class TraitStandardizer(BaseStandardizer):
    name = "trait"
    streaming = True

    def get_column_map(self) -> dict:
        return {
            "trb": "trb",
            "trbv_gene": "trbv_gene",
            "trbj_gene": "trbj_gene",
            "tra": "tra",
            "trav_gene": "trav_gene",
            "traj_gene": "traj_gene",
            "peptide": "peptide",
            "mhc_one": "mhc_one",
            "binding": "binding",
        }

    def _map_columns(self, df: pd.DataFrame, meta: dict) -> pd.DataFrame:
        """Map raw columns from a TRAIT data file to standardized names."""
        merged = pd.DataFrame(index=df.index)

        for col in df.columns:
            cl = col.lower().strip()
            if cl in ("cdr3b", "cdr3_beta", "cdr3.beta", "cdr3_b"):
                merged["trb"] = df[col]
            elif cl in ("cdr3a", "cdr3_alpha", "cdr3.alpha", "cdr3_a"):
                merged["tra"] = df[col]
            elif cl in ("trbv", "vb", "v_beta", "vbeta"):
                merged["trbv_gene"] = df[col]
            elif cl in ("trbj", "jb", "j_beta", "jbeta"):
                merged["trbj_gene"] = df[col]
            elif cl in ("trav", "va", "v_alpha", "valpha"):
                merged["trav_gene"] = df[col]
            elif cl in ("traj", "ja", "j_alpha", "jalpha"):
                merged["traj_gene"] = df[col]
            elif cl in ("cdr3", "junction_aa", "amino_acid"):
                # Single CDR3 column — assume beta
                if "trb" not in merged.columns:
                    merged["trb"] = df[col]

        # Add metadata from filename
        merged["peptide"] = meta["epitope"]
        merged["mhc_one"] = meta["mhc"]
        merged["binding"] = meta.get("binding", "")

        return merged

    def _process_inner_file(
        self, f, meta: dict, study_id: str
    ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """Read and standardize a single inner file from a ZIP."""
        first_line = f.readline().decode("utf-8", errors="ignore")
        f.seek(0)
        sep = "\t" if "\t" in first_line else ","
        try:
            df = pd.read_csv(f, sep=sep, dtype=str).fillna("")
        except Exception:
            return

        if df.empty:
            return

        merged = self._map_columns(df, meta)
        column_map = self.get_column_map()
        result, dropped = standardize_dataframe(
            merged, column_map, source=self.name, study_id=study_id, stitch=self.stitch,
        )
        yield result, dropped

    def load_and_standardize(self) -> Iterator[pd.DataFrame]:
        # Collect epitope ZIP files from binding/ and non_binding/
        epitopes_dir = self.source_dir / "epitopes"
        if epitopes_dir.exists():
            zip_files = sorted(epitopes_dir.rglob("*.zip"))
        else:
            zip_files = sorted(self.source_dir.glob("*.zip"))

        # Process per-epitope ZIP files (binding + non_binding)
        for zf in zip_files:
            meta = _parse_trait_filename(zf.name)
            if not meta:
                continue

            try:
                with zipfile.ZipFile(zf) as z:
                    for inner_name in z.namelist():
                        if inner_name.endswith((".txt", ".csv", ".tsv")):
                            with z.open(inner_name) as f:
                                yield from self._process_inner_file(
                                    f, meta, zf.stem
                                )
            except zipfile.BadZipFile:
                continue

        # Process nested ZIPs (e.g., main/Omics.zip with per-epitope .txt files)
        main_dir = self.source_dir / "main"
        if main_dir.exists():
            for outer_zip in sorted(main_dir.glob("*.zip")):
                try:
                    with zipfile.ZipFile(outer_zip) as z:
                        for inner_name in z.namelist():
                            if not inner_name.endswith(".txt"):
                                continue
                            fname = Path(inner_name).name
                            meta = _parse_trait_filename(fname)
                            if not meta:
                                continue
                            with z.open(inner_name) as f:
                                yield from self._process_inner_file(
                                    f, meta, outer_zip.stem
                                )
                except zipfile.BadZipFile:
                    continue


def main():
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Standardize TRAIT")
    parser.add_argument("--source-dir", default="data/databases/trait")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    standardizer = TraitStandardizer(
        source_dir=args.source_dir, output_dir=args.output_dir,
    )
    print(standardizer.run(force=args.force))


if __name__ == "__main__":
    main()
