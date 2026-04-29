#!/usr/bin/env python3
"""NetMHCpan standardizer.

NetMHCpan — 34M rows, space/tab-separated training data files.
No TCR columns. Peptide + MHC allele only.

Directories:
  - NetMHCpan_train/ (MHC-I training data)
  - NetMHCIIpan_train/ (MHC-II training data)

File formats:
  - allelelist: Maps sample/numeric IDs to allele names (may be comma-separated)
  - c000_ba, c000_el, etc.: MHC-I training data (space-separated, no header)
  - train_BA*.txt, test_BA*.txt, etc.: MHC-II data (tab-separated, no header)
"""

import sys
from pathlib import Path
from typing import Iterator

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from quest.data.standardization import (
    normalize_mhc_allele,
    split_mhc_to_alpha_beta,
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


def _resolve_allele(raw_allele: str, allele_map: dict) -> tuple[str, str] | None:
    """Resolve a raw allele identifier to a single (mhc_one, mhc_two) tuple.

    Returns the allele assignment only for **unambiguous** mappings:
    - Single HLA allele names (direct or via allele_map)
    - Heterodimer formats like HLA-DPA10103-DPB10201 (one alpha + one beta)

    Returns None for:
    - Multi-allele samples (comma-separated lists from elution experiments) —
      these are ambiguous because we don't know which allele presented the
      peptide.  The caller should keep the peptide without MHC assignment.
    - Non-human alleles (BoLA, DLA, SLA, H-2, Mamu, etc.)
    """
    # Look up in allele map; fall back to raw value
    resolved = allele_map.get(raw_allele, raw_allele)

    # Comma-separated = multi-allele sample → ambiguous, skip MHC assignment
    parts = [p.strip() for p in resolved.split(",") if p.strip()]
    if len(parts) > 1:
        return None

    allele = parts[0] if parts else ""
    if not allele:
        return None

    # Use split_mhc_to_alpha_beta which handles heterodimers, slash-separated
    # pairs, and single alleles with proper mhc_one/mhc_two classification
    mhc_one, mhc_two = split_mhc_to_alpha_beta(allele)

    # Skip non-human alleles (BoLA, DLA, SLA, H-2, Mamu, etc.)
    if mhc_one and not mhc_one.startswith("HLA-"):
        mhc_one = ""
    if mhc_two and not mhc_two.startswith("HLA-"):
        mhc_two = ""

    if mhc_one or mhc_two:
        return (mhc_one, mhc_two)

    return None


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

                            resolved = _resolve_allele(allele_idx, allele_map)
                            if resolved:
                                mhc_one, mhc_two = resolved
                                rows.append({
                                    "peptide": peptide,
                                    "mhc_one": mhc_one,
                                    "mhc_two": mhc_two,
                                })
                            else:
                                # Multi-allele (ambiguous) or non-human — peptide only
                                rows.append({
                                    "peptide": peptide,
                                    "mhc_one": "",
                                    "mhc_two": "",
                                })

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
                rows = []
                with open(fpath) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split("\t")
                        if len(parts) < 3:
                            continue

                        peptide = parts[0]
                        raw_allele = parts[2]

                        resolved = _resolve_allele(raw_allele, allele_map)
                        if resolved:
                            mhc_one, mhc_two = resolved
                            rows.append({
                                "peptide": peptide,
                                "mhc_one": mhc_one,
                                "mhc_two": mhc_two,
                            })
                        else:
                            # Multi-allele (ambiguous) or non-human — peptide only
                            rows.append({
                                "peptide": peptide,
                                "mhc_one": "",
                                "mhc_two": "",
                            })

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
