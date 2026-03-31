#!/usr/bin/env python3
"""NetMHCpan standardizer.

NetMHCpan — 34M rows, space/tab-separated training data files.
No TCR columns. Peptide + MHC allele only.

Directories:
  - NetMHCpan_train/ (MHC-I training data)
  - NetMHCIIpan_train/ (MHC-II training data)

File formats:
  - allelelist: Maps numeric IDs to allele names
  - c000_ba, c000_el, etc.: Training data (space-separated)
  - train_BA*.txt, test_BA*.txt, etc.: Tab-separated
"""

import re
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

CHUNK_SIZE = 1_000_000


def _load_allele_list(fpath: Path) -> dict:
    """Load allele list mapping (index → allele name)."""
    alleles = {}
    if not fpath.exists():
        return alleles
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                alleles[parts[0]] = parts[1]
    return alleles


class NetmhcpanStandardizer(BaseStandardizer):
    name = "netmhcpan"
    streaming = True

    def get_column_map(self) -> dict:
        return {
            "peptide": "peptide",
            "mhc_one": "mhc_one",
            "mhc_two": "mhc_two",
        }

    def _process_mhci_dir(self, mhci_dir: Path) -> Iterator[pd.DataFrame]:
        """Process MHC-I training data."""
        allele_map = _load_allele_list(mhci_dir / "allelelist")

        # Process training/test data files
        data_files = sorted(mhci_dir.glob("c*_ba")) + sorted(mhci_dir.glob("c*_el"))
        for fpath in data_files:
            try:
                rows = []
                with open(fpath) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split()
                        if len(parts) >= 2:
                            peptide = parts[0]
                            allele_idx = parts[-1] if len(parts) > 2 else ""
                            allele = allele_map.get(allele_idx, allele_idx)
                            allele = normalize_mhc_allele(allele)
                            rows.append(
                                {"peptide": peptide, "mhc_one": allele, "mhc_two": ""}
                            )
                            if len(rows) >= CHUNK_SIZE:
                                df = pd.DataFrame(rows)
                                result, dropped = standardize_dataframe(
                                    df,
                                    self.get_column_map(),
                                    source=self.name,
                                    study_id=fpath.name,
                                    stitch=self.stitch,
                                    hla_dir=self.hla_dir,
                                )
                                yield result, dropped
                                rows = []

                if rows:
                    df = pd.DataFrame(rows)
                    result, dropped = standardize_dataframe(
                        df,
                        self.get_column_map(),
                        source=self.name,
                        study_id=fpath.name,
                        stitch=self.stitch,
                        hla_dir=self.hla_dir,
                    )
                    yield result, dropped
            except Exception as e:
                print(f"  Warning: Failed to read {fpath}: {e}")

    def _process_mhcii_dir(self, mhcii_dir: Path) -> Iterator[pd.DataFrame]:
        """Process MHC-II training data."""
        allele_map = _load_allele_list(mhcii_dir / "allelelist.txt")

        data_files = sorted(mhcii_dir.glob("train_*.txt")) + sorted(
            mhcii_dir.glob("test_*.txt")
        )
        for fpath in data_files:
            try:
                for chunk in pd.read_csv(
                    fpath,
                    sep="\t",
                    dtype=str,
                    chunksize=CHUNK_SIZE,
                    on_bad_lines="skip",
                ):
                    chunk = chunk.fillna("")
                    # Detect peptide and allele columns
                    merged = pd.DataFrame(index=chunk.index)
                    for col in chunk.columns:
                        cl = col.lower()
                        if "peptide" in cl or "sequence" in cl:
                            merged["peptide"] = chunk[col]
                        elif "allele" in cl:
                            merged["mhc_two"] = chunk[col].apply(normalize_mhc_allele)

                    if "peptide" not in merged.columns:
                        # Try first column as peptide
                        merged["peptide"] = chunk.iloc[:, 0]
                    if "mhc_two" not in merged.columns:
                        merged["mhc_two"] = ""
                    merged["mhc_one"] = ""

                    result, dropped = standardize_dataframe(
                        merged,
                        self.get_column_map(),
                        source=self.name,
                        study_id=fpath.name,
                        stitch=self.stitch,
                        hla_dir=self.hla_dir,
                    )
                    yield result, dropped
            except Exception as e:
                print(f"  Warning: Failed to read {fpath}: {e}")

    def load_and_standardize(self) -> Iterator[pd.DataFrame]:
        # MHC-I
        mhci_dir = self.source_dir / "NetMHCpan_train"
        if mhci_dir.exists():
            yield from self._process_mhci_dir(mhci_dir)

        # MHC-II
        mhcii_dir = self.source_dir / "NetMHCIIpan_train"
        if mhcii_dir.exists():
            yield from self._process_mhcii_dir(mhcii_dir)


def main():
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Standardize NetMHCpan")
    parser.add_argument("--source-dir", default="data/databases/NetMHCPan")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--hla-dir", default="")
    args = parser.parse_args()

    standardizer = NetmhcpanStandardizer(
        source_dir=args.source_dir, output_dir=args.output_dir,
        hla_dir=args.hla_dir,
    )
    print(standardizer.run(force=args.force))


if __name__ == "__main__":
    main()
