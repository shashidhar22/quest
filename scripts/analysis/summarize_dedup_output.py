#!/usr/bin/env python3
"""
Summarize deduplicated MLM parquet outputs.

Reports row counts per permutation_key (grouped by cardinality), unique molecule
counts, and a comparison with the standardization input.

Usage:
    python scripts/analysis/summarize_dedup_output.py \
        --mlm-dir data/deduplicated/mlm \
        --mlm-full-dir data/deduplicated/mlm_full \
        --standardization-summary data/standardized/STANDARDIZATION_SUMMARY.txt \
        --num-workers 32 \
        --output data/deduplicated/DEDUP_SUMMARY.txt

    # Fast mode (counts only, skip unique molecule sets):
    python scripts/analysis/summarize_dedup_output.py \
        --mlm-dir data/deduplicated/mlm \
        --skip-unique-molecules
"""

import argparse
import multiprocessing
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

# Canonical molecule ordering
MOLECULE_TYPES = ["tra", "trb", "peptide", "mhc_one", "mhc_two"]

CARDINALITY_LABELS = {
    1: "SINGLES",
    2: "PAIRS",
    3: "TRIPLETS",
    4: "QUARTETS",
    5: "QUINTETS",
}


# ---------------------------------------------------------------------------
# Permutation key parsing (from deduplicate_streaming.py)
# ---------------------------------------------------------------------------

def parse_permutation_key(perm_key: str) -> List[str]:
    """
    Parse a permutation key string into a list of field names.

    Handles underscore-containing fields (mhc_one, mhc_two) by matching
    longer field names first (greedy match).
    """
    known_fields = ["mhc_one", "mhc_two", "peptide", "tra", "trb"]
    field_order = []
    remaining = perm_key

    while remaining:
        matched = False
        for field in known_fields:
            if remaining.startswith(field):
                field_order.append(field)
                remaining = remaining[len(field):]
                if remaining.startswith("_"):
                    remaining = remaining[1:]
                matched = True
                break
        if not matched:
            if "_" in remaining:
                remaining = remaining.split("_", 1)[1]
            else:
                break

    return field_order


def normalize_key(fields: List[str]) -> str:
    """Sort fields by canonical MOLECULE_TYPES order and join with _."""
    order = {m: i for i, m in enumerate(MOLECULE_TYPES)}
    return "_".join(sorted(fields, key=lambda f: order.get(f, 99)))


# ---------------------------------------------------------------------------
# Phase 1: Row counts per permutation_key (parallel)
# ---------------------------------------------------------------------------

def _count_file(filepath: str) -> Dict[str, int]:
    """Count permutation_key occurrences in a single parquet file."""
    table = pq.read_table(filepath, columns=["permutation_key"])
    vc = pc.value_counts(table.column("permutation_key"))
    counts = {}
    for entry in vc:
        entry = entry.as_py()
        counts[entry["values"]] = entry["counts"]
    return counts


def count_rows_parallel(
    parquet_dir: Path, num_workers: int
) -> Tuple[Dict[str, int], int, int]:
    """
    Count rows per permutation_key across all parquet files using multiprocessing.

    Returns:
        (raw_counts, total_rows, num_files)
    """
    files = sorted(parquet_dir.glob("*.parquet"))
    num_files = len(files)
    if num_files == 0:
        return {}, 0, 0

    file_paths = [str(f) for f in files]
    merged: Dict[str, int] = defaultdict(int)
    total_rows = 0

    with multiprocessing.Pool(num_workers) as pool:
        for counts in tqdm(
            pool.imap_unordered(_count_file, file_paths),
            total=num_files,
            desc=f"Counting {parquet_dir.name}",
        ):
            for key, cnt in counts.items():
                merged[key] += cnt
                total_rows += cnt

    return dict(merged), total_rows, num_files


# ---------------------------------------------------------------------------
# Phase 2: Unique molecule counts (sequential streaming)
# ---------------------------------------------------------------------------

def count_unique_molecules(parquet_dir: Path) -> Dict[str, int]:
    """
    Stream through all parquet files and collect unique molecules per type.

    Returns dict mapping molecule type -> unique count.
    """
    molecule_sets: Dict[str, set] = {m: set() for m in MOLECULE_TYPES}
    key_cache: Dict[str, List[str]] = {}

    files = sorted(parquet_dir.glob("*.parquet"))
    for filepath in tqdm(files, desc=f"Unique molecules ({parquet_dir.name})"):
        table = pq.read_table(filepath)
        perm_keys = table.column("permutation_key").to_pylist()
        sequences = table.column("sequence").to_pylist()

        for perm_key, sequence in zip(perm_keys, sequences):
            if perm_key not in key_cache:
                key_cache[perm_key] = parse_permutation_key(perm_key)

            fields = key_cache[perm_key]
            parts = sequence.split(" ")

            if len(parts) != len(fields):
                continue

            for field, value in zip(fields, parts):
                if field in molecule_sets:
                    molecule_sets[field].add(value)

    return {m: len(s) for m, s in molecule_sets.items()}


# ---------------------------------------------------------------------------
# Phase 3: Standardization summary parsing
# ---------------------------------------------------------------------------

def parse_standardization_summary(filepath: Path) -> Tuple[int, Dict[str, int]]:
    """
    Parse Section 2 of STANDARDIZATION_SUMMARY.txt.

    Returns:
        (total_manifest_rows, per_database_counts)
    """
    per_db: Dict[str, int] = {}
    total = 0

    text = filepath.read_text()
    # Find the per-database table (Section 2)
    in_section2 = False
    past_header = False

    for line in text.splitlines():
        if "2. PER-DATABASE STATISTICS" in line:
            in_section2 = True
            continue

        if in_section2 and line.startswith("---"):
            if not past_header:
                past_header = True
                continue
            # Second dashes line = end of table (TOTAL line follows)

        if in_section2 and past_header:
            if line.startswith("TOTAL"):
                # Parse total line
                parts = line.split()
                # TOTAL  5,441,200,644  ...
                for part in parts[1:]:
                    cleaned = part.replace(",", "")
                    if cleaned.isdigit():
                        total = int(cleaned)
                        break
                break

            # Parse database line: name  manifest_rows  dropped  drop%  files  time  date
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                db_name = parts[0]
                # Find first numeric-looking value (manifest rows)
                for part in parts[1:]:
                    cleaned = part.replace(",", "")
                    if cleaned.isdigit():
                        per_db[db_name] = int(cleaned)
                        break

    return total, per_db


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_directory_report(
    name: str,
    description: str,
    raw_counts: Dict[str, int],
    total_rows: int,
    num_files: int,
    unique_molecules: Optional[Dict[str, int]],
) -> str:
    """Format a report section for one dedup directory."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"DEDUPLICATION SUMMARY: {name} ({description})")
    lines.append("=" * 80)
    lines.append(f"Total files:  {num_files:,}")
    lines.append(f"Total rows:   {total_rows:,}")
    lines.append("")

    # Normalize keys and group by cardinality
    normalized: Dict[str, int] = defaultdict(int)
    for raw_key, cnt in raw_counts.items():
        fields = parse_permutation_key(raw_key)
        norm = normalize_key(fields)
        normalized[norm] += cnt

    # Group by cardinality
    by_cardinality: Dict[int, Dict[str, int]] = defaultdict(dict)
    for norm_key, cnt in normalized.items():
        n_molecules = len(norm_key.split("_"))
        # mhc_one and mhc_two have underscores, so count actual fields
        fields = parse_permutation_key(norm_key)
        cardinality = len(fields)
        by_cardinality[cardinality][norm_key] = cnt

    for card in sorted(by_cardinality.keys()):
        group = by_cardinality[card]
        label = CARDINALITY_LABELS.get(card, f"CARD-{card}")
        non_zero = {k: v for k, v in group.items() if v > 0}
        if not non_zero:
            continue

        lines.append(f"--- {label} ({len(non_zero)} groupings) ---")
        subtotal = sum(non_zero.values())

        # Sort by count descending
        for key in sorted(non_zero, key=lambda k: non_zero[k], reverse=True):
            cnt = non_zero[key]
            pct = cnt / total_rows * 100 if total_rows > 0 else 0
            lines.append(f"  {key:<45} {cnt:>15,}  ({pct:5.2f}%)")

        pct_sub = subtotal / total_rows * 100 if total_rows > 0 else 0
        lines.append(f"  {'SUBTOTAL':<45} {subtotal:>15,}  ({pct_sub:5.2f}%)")
        lines.append("")

    # Unique molecule counts
    if unique_molecules is not None:
        lines.append("--- UNIQUE MOLECULE COUNTS ---")
        for mol in MOLECULE_TYPES:
            count = unique_molecules.get(mol, 0)
            if count > 0:
                lines.append(f"  {mol:<20} {count:>15,} unique values")
        lines.append("")

    return "\n".join(lines)


def format_comparison(
    std_total: int,
    std_per_db: Dict[str, int],
    mlm_total: Optional[int],
    mlm_full_total: Optional[int],
) -> str:
    """Format the standardization vs deduplication comparison."""
    lines = []
    lines.append("=" * 80)
    lines.append("COMPARISON: STANDARDIZATION -> DEDUPLICATION")
    lines.append("=" * 80)
    lines.append(f"Standardization total (manifest):    {std_total:>15,}")
    if mlm_total is not None:
        lines.append(f"Dedup output (mlm / CDR3):           {mlm_total:>15,}")
    if mlm_full_total is not None:
        lines.append(f"Dedup output (mlm_full):             {mlm_full_total:>15,}")
    lines.append("")
    lines.append("Note: Dedup output includes permuted rows (one source row may")
    lines.append("generate multiple subset combinations), so dedup totals may")
    lines.append("exceed unique source row count.")
    lines.append("")
    lines.append("Per-database standardized input:")
    for db in sorted(std_per_db, key=lambda d: std_per_db[d], reverse=True):
        lines.append(f"  {db:<25} {std_per_db[db]:>15,}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Summarize deduplicated MLM parquet outputs."
    )
    parser.add_argument(
        "--mlm-dir",
        type=Path,
        default=None,
        help="Path to CDR3-based deduplicated MLM parquet directory",
    )
    parser.add_argument(
        "--mlm-full-dir",
        type=Path,
        default=None,
        help="Path to full-length stitched TCR deduplicated parquet directory",
    )
    parser.add_argument(
        "--standardization-summary",
        type=Path,
        default=None,
        help="Path to STANDARDIZATION_SUMMARY.txt for comparison",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, multiprocessing.cpu_count() // 2),
        help="Number of parallel workers for row counting (default: half of CPUs)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file path (default: print to stdout)",
    )
    parser.add_argument(
        "--skip-unique-molecules",
        action="store_true",
        help="Skip unique molecule counting (fast mode, counts only)",
    )
    args = parser.parse_args()

    if args.mlm_dir is None and args.mlm_full_dir is None:
        parser.error("At least one of --mlm-dir or --mlm-full-dir is required")

    report_parts = []
    mlm_total = None
    mlm_full_total = None

    # Process mlm directory
    if args.mlm_dir is not None:
        print(f"\n=== Processing {args.mlm_dir} ===")
        raw_counts, total_rows, num_files = count_rows_parallel(
            args.mlm_dir, args.num_workers
        )
        mlm_total = total_rows

        unique_molecules = None
        if not args.skip_unique_molecules:
            unique_molecules = count_unique_molecules(args.mlm_dir)

        report_parts.append(
            format_directory_report(
                "mlm", "CDR3", raw_counts, total_rows, num_files, unique_molecules
            )
        )

    # Process mlm_full directory
    if args.mlm_full_dir is not None:
        print(f"\n=== Processing {args.mlm_full_dir} ===")
        raw_counts, total_rows, num_files = count_rows_parallel(
            args.mlm_full_dir, args.num_workers
        )
        mlm_full_total = total_rows

        unique_molecules = None
        if not args.skip_unique_molecules:
            unique_molecules = count_unique_molecules(args.mlm_full_dir)

        report_parts.append(
            format_directory_report(
                "mlm_full",
                "full-length stitched TCR",
                raw_counts,
                total_rows,
                num_files,
                unique_molecules,
            )
        )

    # Standardization comparison
    if args.standardization_summary is not None:
        print(f"\n=== Parsing {args.standardization_summary} ===")
        std_total, std_per_db = parse_standardization_summary(
            args.standardization_summary
        )
        report_parts.append(
            format_comparison(std_total, std_per_db, mlm_total, mlm_full_total)
        )

    full_report = "\n\n".join(report_parts) + "\n"

    # Output
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(full_report)
        print(f"\nReport written to {args.output}")
    else:
        print("\n" + full_report)


if __name__ == "__main__":
    main()
