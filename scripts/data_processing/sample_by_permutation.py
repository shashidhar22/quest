#!/usr/bin/env python3
"""
Priority-based sampling by permutation key.

Samples from parquet files with priority:
1. First: Include ALL interaction sequences (multi-molecule permutations)
2. Second: Fill remaining quota with single chain sequences (tra, trb)

Output is compatible with esm_tokenizer.py for downstream tokenization.
"""

import argparse
import glob
import os
import random
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# Single chain keys (low priority - used to fill quota)
SINGLE_CHAIN_KEYS = {'tra', 'trb'}


def is_interaction_key(key: str) -> bool:
    """Check if permutation key represents an interaction (2+ molecules)."""
    return key not in SINGLE_CHAIN_KEYS


def scan_file(file_path: str) -> Tuple[str, Dict[str, List[Tuple[str, str]]]]:
    """
    Scan a parquet file and return sequences grouped by permutation key type.

    Returns:
        (file_path, {'interaction': [(key, seq), ...], 'single': [(key, seq), ...]})
    """
    try:
        table = pq.read_table(file_path, columns=['permutation_key', 'sequence'])
        keys = table.column('permutation_key').to_pylist()
        seqs = table.column('sequence').to_pylist()

        result = {'interaction': [], 'single': []}
        for key, seq in zip(keys, seqs):
            if is_interaction_key(key):
                result['interaction'].append((key, seq))
            else:
                result['single'].append((key, seq))

        return file_path, result
    except Exception as e:
        print(f"Error scanning {file_path}: {e}")
        return file_path, {'interaction': [], 'single': []}


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


def sample_by_permutation(
    input_dir: str,
    output_dir: str,
    total_sequences: int,
    seed: int = 42,
    num_workers: int = None,
    batch_size: int = 1_000_000,
):
    """
    Sample sequences with priority: interactions first, then single chains.

    Args:
        input_dir: Directory containing parquet files
        output_dir: Output directory for sampled parquet files
        total_sequences: Target number of sequences to sample
        seed: Random seed for reproducibility
        num_workers: Number of parallel workers
        batch_size: Number of sequences per output parquet file
    """
    if num_workers is None:
        num_workers = min(32, cpu_count())

    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("PRIORITY-BASED SAMPLING BY PERMUTATION KEY")
    print("=" * 70)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target sequences: {total_sequences:,}")
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

    # Phase 1: Scan all files in parallel
    print("\n" + "=" * 70)
    print("PHASE 1: SCANNING FILES")
    print("=" * 70)

    all_interactions = []
    all_singles = []
    interaction_counts = Counter()
    single_counts = Counter()

    with Pool(num_workers) as pool:
        for file_path, result in tqdm(
            pool.imap_unordered(scan_file, file_paths),
            total=len(file_paths),
            desc="Scanning files"
        ):
            for key, seq in result['interaction']:
                all_interactions.append((key, seq))
                interaction_counts[key] += 1
            for key, seq in result['single']:
                all_singles.append((key, seq))
                single_counts[key] += 1

    print(f"\nInteraction sequences: {len(all_interactions):,}")
    for k, c in sorted(interaction_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {c:,}")

    print(f"\nSingle chain sequences: {len(all_singles):,}")
    for k, c in sorted(single_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {c:,}")

    # Phase 2: Select sequences with priority
    print("\n" + "=" * 70)
    print("PHASE 2: SELECTING SEQUENCES")
    print("=" * 70)

    selected = []

    # First: Include ALL interaction sequences
    n_interactions = len(all_interactions)
    if n_interactions > total_sequences:
        print(f"WARNING: {n_interactions:,} interactions exceed target {total_sequences:,}")
        print(f"         Sampling {total_sequences:,} interactions (no single chains)")
        random.shuffle(all_interactions)
        selected = all_interactions[:total_sequences]
    else:
        selected.extend(all_interactions)
        print(f"Added all {n_interactions:,} interaction sequences")

        # Second: Fill remaining quota with single chains
        remaining = total_sequences - n_interactions
        if remaining > 0:
            n_singles = len(all_singles)
            if remaining >= n_singles:
                print(f"Adding all {n_singles:,} single chain sequences")
                selected.extend(all_singles)
            else:
                print(f"Sampling {remaining:,} single chain sequences from {n_singles:,}")
                random.shuffle(all_singles)
                selected.extend(all_singles[:remaining])

    # Free memory
    del all_interactions
    del all_singles

    # Shuffle final selection
    print(f"\nTotal selected: {len(selected):,}")
    random.shuffle(selected)

    # Count final distribution
    final_counts = Counter()
    for key, seq in selected:
        final_counts[key] += 1

    print("\nFinal distribution:")
    for k, c in sorted(final_counts.items(), key=lambda x: -x[1]):
        pct = c / len(selected) * 100
        print(f"  {k}: {c:,} ({pct:.1f}%)")

    # Phase 3: Write output
    print("\n" + "=" * 70)
    print("PHASE 3: WRITING OUTPUT")
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
    print("SAMPLING COMPLETE")
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
        description="Priority-based sampling by permutation key",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sample 10 million sequences with interaction priority
  python sample_by_permutation.py --total 10M

  # Sample 100 million with custom output
  python sample_by_permutation.py --total 100M --output-dir /path/to/output

  # Sample 1 billion sequences
  python sample_by_permutation.py --total 1B --seed 123
"""
    )
    parser.add_argument(
        "--total", "-n", type=str, required=True,
        help="Total sequences to sample (e.g., 10M, 100M, 1B)"
    )
    parser.add_argument(
        "--input-dir", type=str,
        default="/home/ubuntu/quest/data/deduplicated/full/foundation",
        help="Input directory with parquet files"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: input_dir + '_sampled_priority_<N>')"
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
        args.output_dir = f"{args.input_dir}_sampled_priority_{size_str}"

    sample_by_permutation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        total_sequences=total_sequences,
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
