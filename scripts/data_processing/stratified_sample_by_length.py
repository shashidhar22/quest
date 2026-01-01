#!/usr/bin/env python3
"""
Stratified sampling by permutation key and sequence length.

Two-level stratification:
1. Permutation key priority: Include ALL non-tra/trb sequences first,
   then fill remaining quota with tra/trb
2. Sequence length: Quantile-based binning (deciles) to preserve length distribution

Output is compatible with esm_tokenizer.py for downstream tokenization.
"""

import argparse
import glob
import os
import random
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# Single chain keys (low priority - used to fill quota)
SINGLE_CHAIN_KEYS = {'tra', 'trb'}


def is_interaction_key(key: str) -> bool:
    """Check if permutation key represents an interaction (non tra/trb)."""
    return key not in SINGLE_CHAIN_KEYS


def scan_file_lengths(file_path: str) -> Tuple[str, List[int]]:
    """
    Scan a parquet file and return all sequence lengths.

    Returns:
        (file_path, [lengths])
    """
    try:
        table = pq.read_table(file_path, columns=['sequence'])
        seqs = table.column('sequence').to_pylist()
        lengths = [len(seq) for seq in seqs]
        return file_path, lengths
    except Exception as e:
        print(f"Error scanning {file_path}: {e}")
        return file_path, []


def scan_file_with_bins(args: Tuple[str, np.ndarray]) -> Tuple[str, Dict]:
    """
    Scan a parquet file and categorize sequences by type and length bin.

    Args:
        args: (file_path, bin_edges)

    Returns:
        (file_path, {
            'interaction': {bin_idx: [(key, seq), ...]},
            'single': {bin_idx: [(key, seq), ...]}
        })
    """
    file_path, bin_edges = args
    n_bins = len(bin_edges) - 1

    try:
        table = pq.read_table(file_path, columns=['permutation_key', 'sequence'])
        keys = table.column('permutation_key').to_pylist()
        seqs = table.column('sequence').to_pylist()

        result = {
            'interaction': {i: [] for i in range(n_bins)},
            'single': {i: [] for i in range(n_bins)}
        }

        for key, seq in zip(keys, seqs):
            length = len(seq)
            # Find bin index using searchsorted (returns index where element would be inserted)
            bin_idx = np.searchsorted(bin_edges[1:], length, side='right')
            bin_idx = min(bin_idx, n_bins - 1)  # Clamp to last bin

            if is_interaction_key(key):
                result['interaction'][bin_idx].append((key, seq))
            else:
                result['single'][bin_idx].append((key, seq))

        return file_path, result
    except Exception as e:
        print(f"Error scanning {file_path}: {e}")
        return file_path, {
            'interaction': {i: [] for i in range(n_bins)},
            'single': {i: [] for i in range(n_bins)}
        }


def parse_size(size_str: str) -> int:
    """Parse size string like '100M', '1B', '500K' to integer."""
    size_str = size_str.strip().upper()

    multipliers = {
        'K': 1_000,
        'M': 1_000_000,
        'B': 1_000_000_000,
    }

    for suffix, mult in multipliers.items():
        if size_str.endswith(suffix):
            return int(float(size_str[:-1]) * mult)

    return int(size_str)


def stratified_sample(
    input_dir: str,
    output_dir: str,
    total_sequences: int,
    n_bins: int = 10,
    seed: int = 42,
    num_workers: int = None,
    batch_size: int = 1_000_000,
):
    """
    Stratified sampling with permutation key priority and length binning.

    Args:
        input_dir: Directory containing parquet files
        output_dir: Output directory for sampled parquet files
        total_sequences: Target number of sequences to sample
        n_bins: Number of quantile bins for sequence length
        seed: Random seed for reproducibility
        num_workers: Number of parallel workers
        batch_size: Number of sequences per output parquet file
    """
    if num_workers is None:
        num_workers = min(32, cpu_count())

    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("STRATIFIED SAMPLING BY PERMUTATION KEY AND SEQUENCE LENGTH")
    print("=" * 70)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target sequences: {total_sequences:,}")
    print(f"Length bins:      {n_bins} (quantile-based)")
    print(f"Seed:             {seed}")
    print(f"Workers:          {num_workers}")
    print("=" * 70)

    # Find input files
    file_paths = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    print(f"\nFound {len(file_paths)} input parquet files")

    if not file_paths:
        raise ValueError(f"No parquet files found in {input_dir}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # PHASE 1: Scan for sequence lengths to compute quantile bins
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: COMPUTING SEQUENCE LENGTH QUANTILES")
    print("=" * 70)

    all_lengths = []

    with Pool(num_workers) as pool:
        for file_path, lengths in tqdm(
            pool.imap_unordered(scan_file_lengths, file_paths),
            total=len(file_paths),
            desc="Scanning lengths"
        ):
            all_lengths.extend(lengths)

    all_lengths = np.array(all_lengths)
    print(f"\nTotal sequences scanned: {len(all_lengths):,}")
    print(f"Length range: {all_lengths.min()} - {all_lengths.max()}")
    print(f"Mean length: {all_lengths.mean():.1f}")
    print(f"Median length: {np.median(all_lengths):.1f}")

    # Compute quantile bin edges
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(all_lengths, percentiles)
    # Ensure unique bin edges (in case of ties at quantiles)
    bin_edges = np.unique(bin_edges)
    actual_n_bins = len(bin_edges) - 1

    print(f"\nQuantile bin edges ({actual_n_bins} bins):")
    for i in range(actual_n_bins):
        print(f"  Bin {i}: [{bin_edges[i]:.0f}, {bin_edges[i+1]:.0f})")

    # Free memory
    del all_lengths

    # =========================================================================
    # PHASE 2: Scan and categorize sequences by type and length bin
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: CATEGORIZING SEQUENCES")
    print("=" * 70)

    # Initialize storage
    interaction_seqs = {i: [] for i in range(actual_n_bins)}
    single_seqs = {i: [] for i in range(actual_n_bins)}
    interaction_counts = Counter()
    single_counts = Counter()

    # Prepare args for parallel processing
    scan_args = [(fp, bin_edges) for fp in file_paths]

    with Pool(num_workers) as pool:
        for file_path, result in tqdm(
            pool.imap_unordered(scan_file_with_bins, scan_args),
            total=len(file_paths),
            desc="Categorizing"
        ):
            for bin_idx in range(actual_n_bins):
                for key, seq in result['interaction'][bin_idx]:
                    interaction_seqs[bin_idx].append((key, seq))
                    interaction_counts[key] += 1
                for key, seq in result['single'][bin_idx]:
                    single_seqs[bin_idx].append((key, seq))
                    single_counts[key] += 1

    # Print statistics
    total_interactions = sum(len(v) for v in interaction_seqs.values())
    total_singles = sum(len(v) for v in single_seqs.values())

    print(f"\nInteraction sequences (non-tra/trb): {total_interactions:,}")
    for k, c in sorted(interaction_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {k}: {c:,}")
    if len(interaction_counts) > 10:
        print(f"  ... and {len(interaction_counts) - 10} more keys")

    print(f"\nSingle chain sequences (tra/trb): {total_singles:,}")
    for k, c in sorted(single_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {c:,}")

    print(f"\nSequences per length bin:")
    for bin_idx in range(actual_n_bins):
        n_int = len(interaction_seqs[bin_idx])
        n_single = len(single_seqs[bin_idx])
        print(f"  Bin {bin_idx} [{bin_edges[bin_idx]:.0f}-{bin_edges[bin_idx+1]:.0f}): "
              f"interactions={n_int:,}, singles={n_single:,}")

    # =========================================================================
    # PHASE 3: Select sequences with priority and stratification
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: SELECTING SEQUENCES")
    print("=" * 70)

    selected = []

    # Step A: Include ALL interaction sequences
    if total_interactions > total_sequences:
        print(f"WARNING: {total_interactions:,} interactions exceed target {total_sequences:,}")
        print(f"         Sampling {total_sequences:,} interactions proportionally by length bin")

        # Sample proportionally from interaction bins
        for bin_idx in range(actual_n_bins):
            bin_seqs = interaction_seqs[bin_idx]
            bin_quota = int(total_sequences * len(bin_seqs) / total_interactions)
            if bin_quota > 0 and len(bin_seqs) > 0:
                random.shuffle(bin_seqs)
                selected.extend(bin_seqs[:bin_quota])

        # Adjust to exact target (may be slightly off due to rounding)
        if len(selected) > total_sequences:
            random.shuffle(selected)
            selected = selected[:total_sequences]
    else:
        # Include all interactions
        for bin_idx in range(actual_n_bins):
            selected.extend(interaction_seqs[bin_idx])
        print(f"Added all {total_interactions:,} interaction sequences")

        # Step B: Fill remaining quota with singles, stratified by length
        remaining = total_sequences - total_interactions
        if remaining > 0 and total_singles > 0:
            print(f"\nFilling remaining {remaining:,} with single chains (stratified by length)")

            # Calculate samples per bin proportionally
            for bin_idx in range(actual_n_bins):
                bin_seqs = single_seqs[bin_idx]
                if len(bin_seqs) == 0:
                    continue

                # Proportional quota based on bin's share of total singles
                bin_quota = int(remaining * len(bin_seqs) / total_singles)
                actual_sample = min(bin_quota, len(bin_seqs))

                if actual_sample > 0:
                    random.shuffle(bin_seqs)
                    selected.extend(bin_seqs[:actual_sample])
                    print(f"  Bin {bin_idx}: sampled {actual_sample:,} / {len(bin_seqs):,}")

    # Free memory
    del interaction_seqs
    del single_seqs

    # Shuffle final selection
    print(f"\nTotal selected: {len(selected):,}")
    random.shuffle(selected)

    # Count final distribution
    final_key_counts = Counter()
    final_length_bins = Counter()
    for key, seq in selected:
        final_key_counts[key] += 1
        bin_idx = np.searchsorted(bin_edges[1:], len(seq), side='right')
        bin_idx = min(bin_idx, actual_n_bins - 1)
        final_length_bins[bin_idx] += 1

    print("\nFinal distribution by permutation key:")
    for k, c in sorted(final_key_counts.items(), key=lambda x: -x[1])[:15]:
        pct = c / len(selected) * 100
        print(f"  {k}: {c:,} ({pct:.2f}%)")
    if len(final_key_counts) > 15:
        print(f"  ... and {len(final_key_counts) - 15} more keys")

    print("\nFinal distribution by length bin:")
    for bin_idx in range(actual_n_bins):
        c = final_length_bins[bin_idx]
        pct = c / len(selected) * 100
        print(f"  Bin {bin_idx} [{bin_edges[bin_idx]:.0f}-{bin_edges[bin_idx+1]:.0f}): "
              f"{c:,} ({pct:.2f}%)")

    # =========================================================================
    # PHASE 4: Write output
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: WRITING OUTPUT")
    print("=" * 70)

    n_batches = (len(selected) + batch_size - 1) // batch_size
    print(f"Writing {n_batches} parquet files (batch size: {batch_size:,})")

    for batch_idx in tqdm(range(n_batches), desc="Writing batches"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(selected))
        batch = selected[start:end]

        # Create table
        keys = [item[0] for item in batch]
        seqs = [item[1] for item in batch]

        table = pa.table({
            'permutation_key': keys,
            'sequence': seqs
        })

        output_file = output_path / f"batch_{batch_idx:06d}.parquet"
        pq.write_table(table, output_file)

    # Summary
    print("\n" + "=" * 70)
    print("STRATIFIED SAMPLING COMPLETE")
    print("=" * 70)
    print(f"Total sequences: {len(selected):,}")
    print(f"Output files:    {n_batches}")
    print(f"Output directory: {output_dir}")
    print("\nNext step: Run esm_tokenizer.py to tokenize:")
    print(f"  python scripts/data_processing/esm_tokenizer.py \\")
    print(f"      --input-dir {output_dir} \\")
    print(f"      --output-dir <tokenized_output>")


def main():
    parser = argparse.ArgumentParser(
        description="Stratified sampling by permutation key and sequence length",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sample 100 million sequences with stratification
  python stratified_sample_by_length.py --total 100M

  # Sample with custom number of length bins
  python stratified_sample_by_length.py --total 100M --n-bins 20

  # Sample from custom input directory
  python stratified_sample_by_length.py --total 50M --input-dir /path/to/data
"""
    )
    parser.add_argument(
        "--total", "-n", type=str, required=True,
        help="Total sequences to sample (e.g., 10M, 100M, 1B)"
    )
    parser.add_argument(
        "--input-dir", type=str,
        default="/home/ubuntu/quest/data/deduplicated/full/foundation_permutations",
        help="Input directory with parquet files"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: input_dir + '_stratified_<N>')"
    )
    parser.add_argument(
        "--n-bins", type=int, default=10,
        help="Number of quantile bins for sequence length (default: 10)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--batch-size", type=int, default=1_000_000,
        help="Sequences per output parquet file"
    )

    args = parser.parse_args()

    # Parse total size
    total_sequences = parse_size(args.total)

    # Default output directory
    if args.output_dir is None:
        size_str = args.total.upper()
        args.output_dir = f"{args.input_dir}_stratified_{size_str}"

    stratified_sample(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        total_sequences=total_sequences,
        n_bins=args.n_bins,
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
