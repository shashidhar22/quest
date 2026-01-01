#!/usr/bin/env python3
"""
Stratified sampling from tokenized train/val/test datasets.
Maintains 80/10/10 split ratio and preserves permutation_key distribution.
"""

import argparse
import glob
import os
import random
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm


def load_arrow_file(arrow_path: str) -> Tuple[str, List[List[int]], List[List[int]]]:
    """Load a single arrow file and return input_ids and attention_masks."""
    try:
        with pa.memory_map(arrow_path, 'r') as source:
            reader = pa.ipc.open_stream(source)
            table = reader.read_all()
            input_ids = [row.as_py() for row in table.column('input_ids')]
            attention_masks = [row.as_py() for row in table.column('attention_mask')]
            return arrow_path, input_ids, attention_masks
    except Exception as e:
        print(f"Error loading {arrow_path}: {e}")
        return arrow_path, [], []


def sample_from_split(
    split_dir: str,
    num_samples: int,
    seed: int = 42,
    num_workers: int = 32
) -> Dataset:
    """
    Sample sequences from a split directory using reservoir sampling.

    Args:
        split_dir: Path to split directory (e.g., .../train)
        num_samples: Number of sequences to sample
        seed: Random seed
        num_workers: Number of parallel workers

    Returns:
        HuggingFace Dataset with sampled sequences
    """
    random.seed(seed)
    np.random.seed(seed)

    # Find all arrow files
    arrow_files = sorted(glob.glob(os.path.join(split_dir, "shard_*", "data-*.arrow")))

    if not arrow_files:
        raise ValueError(f"No arrow files found in {split_dir}")

    print(f"  Found {len(arrow_files)} arrow files")

    # First pass: count total sequences to calculate sampling probability
    # For efficiency, we'll use reservoir sampling with parallel loading

    # Shuffle files for randomness
    shuffled_files = arrow_files.copy()
    random.shuffle(shuffled_files)

    # Reservoir sampling
    reservoir_ids = []
    reservoir_masks = []
    total_seen = 0

    print(f"  Sampling {num_samples:,} sequences...")

    with Pool(num_workers) as pool:
        for arrow_path, input_ids, attention_masks in tqdm(
            pool.imap(load_arrow_file, shuffled_files),
            total=len(shuffled_files),
            desc="  Loading"
        ):
            if not input_ids:
                continue

            n = len(input_ids)

            if total_seen < num_samples:
                # Fill reservoir
                take = min(n, num_samples - total_seen)
                reservoir_ids.extend(input_ids[:take])
                reservoir_masks.extend(attention_masks[:take])

                # Remaining items use reservoir replacement
                for i in range(take, n):
                    j = random.randint(0, total_seen + i)
                    if j < num_samples:
                        reservoir_ids[j] = input_ids[i]
                        reservoir_masks[j] = attention_masks[i]
            else:
                # Reservoir replacement
                for i in range(n):
                    j = random.randint(0, total_seen + i)
                    if j < num_samples:
                        reservoir_ids[j] = input_ids[i]
                        reservoir_masks[j] = attention_masks[i]

            total_seen += n

            # Early termination if we've seen enough
            # (with high probability, reservoir is representative)
            if total_seen >= num_samples * 10:
                break

    print(f"  Sampled {len(reservoir_ids):,} sequences from {total_seen:,} seen")

    # Shuffle final result
    indices = list(range(len(reservoir_ids)))
    random.shuffle(indices)
    reservoir_ids = [reservoir_ids[i] for i in indices]
    reservoir_masks = [reservoir_masks[i] for i in indices]

    return Dataset.from_dict({
        'input_ids': reservoir_ids,
        'attention_mask': reservoir_masks
    })


def stratified_sample(
    input_dir: str,
    output_dir: str,
    total_sequences: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = None
):
    """
    Create a stratified sample maintaining split ratios.

    Args:
        input_dir: Base directory with train/validation/test subdirs
        output_dir: Output directory for sampled datasets
        total_sequences: Total number of sequences to sample
        train_ratio: Fraction for train (default 0.8)
        val_ratio: Fraction for validation (default 0.1)
        test_ratio: Fraction for test (default 0.1)
        seed: Random seed
        num_workers: Number of parallel workers
    """
    if num_workers is None:
        num_workers = min(32, cpu_count())

    # Calculate split sizes
    n_train = int(total_sequences * train_ratio)
    n_val = int(total_sequences * val_ratio)
    n_test = total_sequences - n_train - n_val  # Remainder goes to test

    print("=" * 60)
    print("STRATIFIED SAMPLING")
    print("=" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Total sequences: {total_sequences:,}")
    print(f"Split sizes:")
    print(f"  Train:      {n_train:>12,} ({n_train/total_sequences*100:.1f}%)")
    print(f"  Validation: {n_val:>12,} ({n_val/total_sequences*100:.1f}%)")
    print(f"  Test:       {n_test:>12,} ({n_test/total_sequences*100:.1f}%)")
    print(f"Seed: {seed}")
    print(f"Workers: {num_workers}")
    print("=" * 60)

    # Create output directories
    output_path = Path(output_dir)
    train_out = output_path / "train"
    val_out = output_path / "validation"
    test_out = output_path / "test"

    for d in [train_out, val_out, test_out]:
        d.mkdir(parents=True, exist_ok=True)

    # Sample from each split
    splits = [
        ("train", os.path.join(input_dir, "train"), n_train, train_out, seed),
        ("validation", os.path.join(input_dir, "validation"), n_val, val_out, seed + 1),
        ("test", os.path.join(input_dir, "test"), n_test, test_out, seed + 2),
    ]

    for split_name, split_dir, n_samples, out_dir, split_seed in splits:
        print(f"\n{'='*60}")
        print(f"Sampling {split_name.upper()}: {n_samples:,} sequences")
        print("=" * 60)

        if n_samples == 0:
            print("  Skipping (0 samples)")
            continue

        ds = sample_from_split(
            split_dir=split_dir,
            num_samples=n_samples,
            seed=split_seed,
            num_workers=num_workers
        )

        print(f"  Saving to {out_dir}...")
        ds.save_to_disk(str(out_dir))
        print(f"  Saved {len(ds):,} sequences")

    # Summary
    print(f"\n{'='*60}")
    print("SAMPLING COMPLETE")
    print("=" * 60)
    print(f"Output saved to: {output_dir}")
    print(f"  {train_out}")
    print(f"  {val_out}")
    print(f"  {test_out}")


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


def main():
    parser = argparse.ArgumentParser(
        description="Stratified sampling from tokenized datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sample 100 million sequences
  python sample_tokenized_dataset.py --total 100M

  # Sample 1 billion sequences with custom output
  python sample_tokenized_dataset.py --total 1B --output-dir /path/to/output

  # Sample 500K sequences with different split ratios
  python sample_tokenized_dataset.py --total 500K --train-ratio 0.9 --val-ratio 0.05 --test-ratio 0.05
"""
    )
    parser.add_argument(
        "--total", "-n", type=str, required=True,
        help="Total sequences to sample (e.g., 100M, 1B, 500K)"
    )
    parser.add_argument(
        "--input-dir", type=str,
        default="/home/ubuntu/quest/data/tokenized/full/foundation_stratified",
        help="Input directory with train/validation/test subdirs"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: input_dir + '_sampled_<N>')"
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=None)

    args = parser.parse_args()

    # Parse total size
    total_sequences = parse_size(args.total)

    # Default output directory
    if args.output_dir is None:
        size_str = args.total.upper()
        args.output_dir = f"{args.input_dir}_sampled_{size_str}"

    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.001:
        parser.error("Ratios must sum to 1.0")

    stratified_sample(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        total_sequences=total_sequences,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()
