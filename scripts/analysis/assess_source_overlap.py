#!/usr/bin/env python3
"""Cross-source TRB CDR3 overlap analysis between standardized databases.

Streams standardized parquet files and computes the overlap of unique TRB CDR3
sequences between immuneaccess and adc (the two largest sources, ~88.6% of all
input).  Optionally extends to tcrdb and immunecode.

Memory strategy: build an xxhash digest set for the first source, then stream
the second source checking membership.  At 8 bytes per hash, 2.77B unique CDR3s
would require ~22 GB.  In practice the unique count is much smaller.

Usage:
    python scripts/analysis/assess_source_overlap.py \
        --standardized-dir data/standardized \
        --sources immuneaccess adc \
        --num-workers 16
"""

import argparse
import multiprocessing
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pyarrow.parquet as pq
import xxhash
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_cdr3(seq: str) -> bytes:
    """Return 8-byte xxhash digest of a CDR3 sequence."""
    return xxhash.xxh64(seq.encode()).digest()


def _extract_trb_hashes(filepath: str) -> Tuple[Set[bytes], Counter]:
    """Extract unique TRB CDR3 hashes and raw sequences from a parquet file.

    Returns (set of 8-byte hashes, Counter of raw CDR3 strings).
    """
    try:
        table = pq.read_table(filepath, columns=["trb"])
    except Exception:
        return set(), Counter()

    hashes = set()
    counts: Counter = Counter()
    col = table.column("trb")

    for val in col.to_pylist():
        if val and val != "":
            hashes.add(_hash_cdr3(val))
            counts[val] += 1

    return hashes, counts


def _extract_trb_hashes_only(filepath: str) -> Set[bytes]:
    """Extract unique TRB CDR3 hashes (no counts) for membership checking."""
    try:
        table = pq.read_table(filepath, columns=["trb"])
    except Exception:
        return set()

    hashes = set()
    col = table.column("trb")

    for val in col.to_pylist():
        if val and val != "":
            hashes.add(_hash_cdr3(val))

    return hashes


def _count_trb_overlap(args: Tuple[str, frozenset]) -> Tuple[int, int, Counter]:
    """Check a parquet file's TRB CDR3s against a reference hash set.

    Returns (total_trb_rows, overlap_count, Counter of overlapping CDR3s).
    """
    filepath, ref_hashes = args
    try:
        table = pq.read_table(filepath, columns=["trb"])
    except Exception:
        return 0, 0, Counter()

    total = 0
    overlap = 0
    overlap_seqs: Counter = Counter()
    col = table.column("trb")

    for val in col.to_pylist():
        if val and val != "":
            total += 1
            h = _hash_cdr3(val)
            if h in ref_hashes:
                overlap += 1
                overlap_seqs[val] += 1

    return total, overlap, overlap_seqs


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def get_parquet_files(standardized_dir: Path, source: str) -> List[str]:
    """Get sorted list of parquet files for a source."""
    src_dir = standardized_dir / source
    if not src_dir.exists():
        print(f"WARNING: {src_dir} does not exist", file=sys.stderr)
        return []
    return sorted(str(f) for f in src_dir.glob("part_*.parquet"))


def build_hash_set(
    files: List[str], num_workers: int, source_name: str,
) -> Tuple[Set[bytes], Counter]:
    """Build a set of TRB CDR3 hashes from parquet files in parallel.

    Returns (hash_set, Counter of CDR3 → total_count).
    """
    all_hashes: Set[bytes] = set()
    all_counts: Counter = Counter()

    with multiprocessing.Pool(num_workers) as pool:
        for h_set, counts in tqdm(
            pool.imap_unordered(_extract_trb_hashes, files),
            total=len(files),
            desc=f"Building {source_name} hash set",
        ):
            all_hashes.update(h_set)
            all_counts.update(counts)

    return all_hashes, all_counts


def build_hash_set_only(
    files: List[str], num_workers: int, source_name: str,
) -> Set[bytes]:
    """Build hash set without tracking raw sequences (lower memory)."""
    all_hashes: Set[bytes] = set()

    with multiprocessing.Pool(num_workers) as pool:
        for h_set in tqdm(
            pool.imap_unordered(_extract_trb_hashes_only, files),
            total=len(files),
            desc=f"Building {source_name} hash set",
        ):
            all_hashes.update(h_set)

    return all_hashes


def check_overlap(
    files: List[str],
    ref_hashes: Set[bytes],
    num_workers: int,
    source_name: str,
) -> Tuple[int, int, Counter]:
    """Stream parquet files and check TRB CDR3 overlap against reference set.

    Returns (total_trb_rows, unique_overlap_count, Counter of overlap seqs).
    """
    # frozenset for pickling across processes
    ref_frozen = frozenset(ref_hashes)
    args_list = [(f, ref_frozen) for f in files]

    total_rows = 0
    overlap_rows = 0
    overlap_seqs: Counter = Counter()

    with multiprocessing.Pool(num_workers) as pool:
        for rows, olap, seqs in tqdm(
            pool.imap_unordered(_count_trb_overlap, args_list),
            total=len(files),
            desc=f"Checking {source_name} overlap",
        ):
            total_rows += rows
            overlap_rows += olap
            overlap_seqs.update(seqs)

    return total_rows, overlap_rows, overlap_seqs


def format_report(
    source_a: str,
    source_b: str,
    unique_a: int,
    unique_b: int,
    total_rows_a: int,
    total_rows_b: int,
    intersection_size: int,
    top_overlapping: List[Tuple[str, int]],
    extra_sources: Dict[str, Tuple[int, int, int]] = None,
) -> str:
    """Format the overlap analysis report."""
    union_size = unique_a + unique_b - intersection_size
    overlap_pct = (intersection_size / union_size * 100) if union_size > 0 else 0
    only_a = unique_a - intersection_size
    only_b = unique_b - intersection_size

    lines = []
    lines.append("=" * 80)
    lines.append("CROSS-SOURCE TRB CDR3 OVERLAP ANALYSIS")
    lines.append("=" * 80)
    lines.append("")

    lines.append(f"{'Source':<20} {'Total Rows':>15} {'Unique TRB CDR3':>18}")
    lines.append("-" * 55)
    lines.append(f"{source_a:<20} {total_rows_a:>15,} {unique_a:>18,}")
    lines.append(f"{source_b:<20} {total_rows_b:>15,} {unique_b:>18,}")
    lines.append("")

    lines.append("--- Overlap Statistics ---")
    lines.append(f"  Intersection (in both):        {intersection_size:>15,}")
    lines.append(f"  Unique to {source_a + ':':<18} {only_a:>15,}")
    lines.append(f"  Unique to {source_b + ':':<18} {only_b:>15,}")
    lines.append(f"  Union:                         {union_size:>15,}")
    lines.append(f"  Overlap % (intersection/union): {overlap_pct:>14.2f}%")
    lines.append(f"  Overlap % of {source_a}:         "
                 f"{intersection_size / unique_a * 100 if unique_a else 0:>14.2f}%")
    lines.append(f"  Overlap % of {source_b}:         "
                 f"{intersection_size / unique_b * 100 if unique_b else 0:>14.2f}%")
    lines.append("")

    if extra_sources:
        lines.append("--- Additional Source Overlaps ---")
        for src, (unique, isect_with_a, isect_with_b) in extra_sources.items():
            lines.append(f"  {src}:")
            lines.append(f"    Unique TRB CDR3:            {unique:>15,}")
            lines.append(f"    Overlap with {source_a}:     {isect_with_a:>15,} "
                         f"({isect_with_a / unique * 100 if unique else 0:.2f}%)")
            lines.append(f"    Overlap with {source_b}:     {isect_with_b:>15,} "
                         f"({isect_with_b / unique * 100 if unique else 0:.2f}%)")
        lines.append("")

    lines.append("--- Top 20 Most-Duplicated CDR3s Across Sources ---")
    lines.append(f"  {'CDR3 Sequence':<30} {'Count':>12}")
    lines.append("  " + "-" * 44)
    for seq, count in top_overlapping[:20]:
        lines.append(f"  {seq:<30} {count:>12,}")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Assess TRB CDR3 overlap between standardized sources."
    )
    parser.add_argument(
        "--standardized-dir",
        type=Path,
        default=Path("data/standardized"),
        help="Path to standardized data directory",
    )
    parser.add_argument(
        "--sources",
        nargs=2,
        default=["immuneaccess", "adc"],
        help="Two primary sources to compare (default: immuneaccess adc)",
    )
    parser.add_argument(
        "--extra-sources",
        nargs="*",
        default=[],
        help="Additional sources to check overlap with primary pair (e.g. tcrdb immunecode)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, multiprocessing.cpu_count() // 2),
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: print to stdout)",
    )
    args = parser.parse_args()

    source_a, source_b = args.sources

    # Phase 1: Build hash set for source A (with counts for top-overlap report)
    files_a = get_parquet_files(args.standardized_dir, source_a)
    if not files_a:
        print(f"No parquet files found for {source_a}", file=sys.stderr)
        sys.exit(1)

    print(f"Phase 1: Building {source_a} hash set from {len(files_a)} files...")
    hashes_a, counts_a = build_hash_set(files_a, args.num_workers, source_a)
    unique_a = len(hashes_a)
    total_rows_a = sum(counts_a.values())
    print(f"  {source_a}: {total_rows_a:,} total TRB rows, {unique_a:,} unique CDR3s")
    print(f"  Hash set memory: ~{unique_a * 8 / 1024**3:.2f} GB")

    # Phase 2: Build hash set for source B and compute intersection
    files_b = get_parquet_files(args.standardized_dir, source_b)
    if not files_b:
        print(f"No parquet files found for {source_b}", file=sys.stderr)
        sys.exit(1)

    print(f"\nPhase 2: Building {source_b} hash set from {len(files_b)} files...")
    hashes_b, counts_b = build_hash_set(files_b, args.num_workers, source_b)
    unique_b = len(hashes_b)
    total_rows_b = sum(counts_b.values())
    print(f"  {source_b}: {total_rows_b:,} total TRB rows, {unique_b:,} unique CDR3s")

    # Phase 3: Compute intersection
    print("\nPhase 3: Computing intersection...")
    intersection = hashes_a & hashes_b
    intersection_size = len(intersection)
    print(f"  Intersection: {intersection_size:,} unique CDR3s")

    # Find top overlapping sequences (need to find raw sequences for intersection hashes)
    # Build hash→sequence lookup from both counters
    seq_to_hash = {}
    for seq in counts_a:
        h = _hash_cdr3(seq)
        if h in intersection:
            seq_to_hash[seq] = counts_a[seq] + counts_b.get(seq, 0)
    for seq in counts_b:
        if seq not in seq_to_hash:
            h = _hash_cdr3(seq)
            if h in intersection:
                seq_to_hash[seq] = counts_a.get(seq, 0) + counts_b[seq]

    top_overlapping = sorted(seq_to_hash.items(), key=lambda x: x[1], reverse=True)

    # Phase 4: Optional extra sources
    extra_results = {}
    if args.extra_sources:
        print(f"\nPhase 4: Checking {len(args.extra_sources)} additional sources...")
        for extra_src in args.extra_sources:
            extra_files = get_parquet_files(args.standardized_dir, extra_src)
            if not extra_files:
                print(f"  Skipping {extra_src}: no parquet files found")
                continue
            print(f"  Building {extra_src} hash set...")
            extra_hashes = build_hash_set_only(
                extra_files, args.num_workers, extra_src,
            )
            extra_unique = len(extra_hashes)
            isect_a = len(extra_hashes & hashes_a)
            isect_b = len(extra_hashes & hashes_b)
            extra_results[extra_src] = (extra_unique, isect_a, isect_b)
            print(f"    {extra_src}: {extra_unique:,} unique, "
                  f"overlap with {source_a}: {isect_a:,}, "
                  f"overlap with {source_b}: {isect_b:,}")

    # Phase 5: Generate report
    report = format_report(
        source_a, source_b,
        unique_a, unique_b,
        total_rows_a, total_rows_b,
        intersection_size,
        top_overlapping,
        extra_results if extra_results else None,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
        print(f"\nReport written to {args.output}")
    else:
        print("\n" + report)


if __name__ == "__main__":
    main()
