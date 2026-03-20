#!/usr/bin/env python3
"""IMGTHLA (IPD-IMGT/HLA) standardizer.

Ingests HLA protein sequences from the IPD-IMGT/HLA database and stores
them as MHC reference sequences in the unified schema.

Input: hla_prot.fasta — ~41K HLA protein sequences
Output: One row per unique 4-digit allele with the protein sequence in
        mhc_one (class I / class II alpha) or mhc_two (class II beta).

Source = "imgthla". No peptide or TCR data — this is an MHC sequence reference.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Iterator

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from quest.data.standardization import TARGET_COLUMNS
from scripts.data_processing.standardize._base import BaseStandardizer

logger = logging.getLogger(__name__)

# HLA genes by MHC class
_CLASS_I_GENES = {"A", "B", "C", "E", "F", "G"}
_CLASS_II_ALPHA_GENES = {"DRA", "DQA1", "DPA1"}
_CLASS_II_BETA_GENES = {"DRB1", "DRB2", "DRB3", "DRB4", "DRB5", "DQB1", "DPB1"}

# Pattern to extract gene and 4-digit allele from FASTA header
# e.g., "A*01:01:01:01" -> gene="A", four_digit="01:01"
_ALLELE_RE = re.compile(
    r"([A-Z][A-Za-z0-9]*)\*(\d{2,3}):(\d{2,3})"
)


def _parse_hla_fasta(fasta_path: Path) -> list[dict]:
    """Parse hla_prot.fasta and return records deduplicated at 4-digit resolution.

    Returns list of dicts with keys: allele_name, gene, sequence.
    Keeps the first (longest) sequence per 4-digit allele.
    """
    if not fasta_path.exists():
        logger.warning("hla_prot.fasta not found at %s", fasta_path)
        return []

    # Parse FASTA
    alleles = {}  # {four_digit_name: (sequence, gene)}
    current_header = ""
    current_seq_parts = []

    def _flush():
        nonlocal current_header, current_seq_parts
        if not current_header:
            return
        seq = "".join(current_seq_parts).strip()
        m = _ALLELE_RE.search(current_header)
        if m and seq:
            gene = m.group(1)
            group = m.group(2)
            protein = m.group(3)
            four_digit = f"HLA-{gene}*{group}:{protein}"
            # Keep first occurrence (or longest sequence)
            if four_digit not in alleles or len(seq) > len(alleles[four_digit][0]):
                alleles[four_digit] = (seq, gene)
        current_header = ""
        current_seq_parts = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                _flush()
                current_header = line
                current_seq_parts = []
            else:
                current_seq_parts.append(line)
        _flush()

    logger.info(
        "Parsed %d unique 4-digit alleles from %s", len(alleles), fasta_path
    )

    records = []
    for allele_name, (sequence, gene) in sorted(alleles.items()):
        records.append(
            {
                "allele_name": allele_name,
                "gene": gene,
                "sequence": sequence,
            }
        )
    return records


class ImgthlaStandardizer(BaseStandardizer):
    name = "imgthla"

    def get_column_map(self) -> dict:
        # Not used — we build TARGET_COLUMNS directly
        return {}

    def load_and_standardize(self) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        fasta_path = self.source_dir / "hla_prot.fasta"
        records = _parse_hla_fasta(fasta_path)

        if not records:
            return

        # Build output DataFrame with all 15 TARGET_COLUMNS
        rows = []
        for rec in records:
            gene = rec["gene"]
            allele_name = rec["allele_name"]
            sequence = rec["sequence"]

            # Classify: class I -> mhc_one, class II alpha -> mhc_one,
            # class II beta -> mhc_two
            mhc_one = ""
            mhc_two = ""

            if gene in _CLASS_I_GENES or gene in _CLASS_II_ALPHA_GENES:
                mhc_one = sequence
            elif gene in _CLASS_II_BETA_GENES:
                mhc_two = sequence
            else:
                # Unknown gene — default to mhc_one
                mhc_one = sequence

            row = {col: "" for col in TARGET_COLUMNS}
            row["mhc_one"] = mhc_one
            row["mhc_two"] = mhc_two
            row["study_id"] = allele_name
            row["source"] = self.name
            rows.append(row)

        result = pd.DataFrame(rows, columns=TARGET_COLUMNS)
        # Ensure all string type
        for col in TARGET_COLUMNS:
            result[col] = result[col].astype(str)

        dropped = pd.DataFrame(
            columns=["reason", "source_file", "row_index", "field", "raw_value"]
        )

        logger.info("IMGTHLA: %d allele records", len(result))
        yield result, dropped


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Standardize IPD-IMGT/HLA database"
    )
    parser.add_argument("--source-dir", default="data/databases/IMGTHLA")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    standardizer = ImgthlaStandardizer(
        source_dir=args.source_dir, output_dir=args.output_dir
    )
    print(standardizer.run(force=args.force))


if __name__ == "__main__":
    main()
