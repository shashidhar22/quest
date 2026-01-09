#!/usr/bin/env python3
"""
Dataset summary tool - analyzes sequence length distribution for tokenized or raw datasets.

Usage:
    # For tokenized HuggingFace datasets (with input_ids)
    python dataset_summary.py --path /path/to/tokenized/dataset --type tokenized

    # For raw parquet datasets (with sequence column)
    python dataset_summary.py --path /path/to/parquet/dir --type parquet

    # Specify split (train/val/test)
    python dataset_summary.py --path /path/to/dataset --split train
"""

import argparse
import glob
import os
from collections import Counter
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm


def scan_tokenized_shard(shard_path: str):
    """Get sequence lengths from a tokenized HuggingFace dataset shard."""
    try:
        from datasets import load_from_disk
        ds = load_from_disk(shard_path)
        lengths = [len(ids) for ids in ds['input_ids']]
        return lengths
    except Exception as e:
        print(f"Error loading {shard_path}: {e}")
        return []


def scan_parquet_file(file_path: str):
    """Get sequence lengths from a parquet file with 'sequence' column."""
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(file_path, columns=['sequence'])
        seqs = table.column('sequence').to_pylist()
        lengths = [len(seq) for seq in seqs]
        return lengths
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []


def compute_summary(lengths: np.ndarray, n_bins: int = 10):
    """Compute and print summary statistics."""
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(f"\nTotal sequences: {len(lengths):,}")
    print(f"Min length:      {lengths.min()}")
    print(f"Max length:      {lengths.max()}")
    print(f"Mean length:     {lengths.mean():.1f}")
    print(f"Median length:   {np.median(lengths):.1f}")
    print(f"Std dev:         {lengths.std():.1f}")

    # Percentiles
    print(f"\nPercentiles:")
    for p in [25, 50, 75, 90, 95, 99, 99.9, 100]:
        val = np.percentile(lengths, p)
        print(f"  {p:>6}%: {val:>8.0f}")

    # Quantile-based bins
    print(f"\nDistribution by quantile bins ({n_bins} bins):")
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(lengths, percentiles)
    bin_edges = np.unique(bin_edges)  # Remove duplicates
    actual_bins = len(bin_edges) - 1

    for i in range(actual_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == actual_bins - 1:
            mask = (lengths >= low) & (lengths <= high)
        else:
            mask = (lengths >= low) & (lengths < high)
        count = mask.sum()
        pct = count / len(lengths) * 100
        print(f"  Bin {i:>2} [{low:>6.0f}-{high:<6.0f}): {count:>12,} ({pct:>6.2f}%)")

    # Fixed bins for common use cases
    print(f"\nDistribution by fixed length bins:")
    fixed_bins = [0, 128, 256, 320, 384, 512, 768, 1024, 2048, np.inf]
    bin_labels = ['0-128', '128-256', '256-320', '320-384', '384-512',
                  '512-768', '768-1024', '1024-2048', '2048+']

    for i in range(len(fixed_bins) - 1):
        mask = (lengths >= fixed_bins[i]) & (lengths < fixed_bins[i + 1])
        count = mask.sum()
        if count > 0:
            pct = count / len(lengths) * 100
            print(f"  {bin_labels[i]:>12}: {count:>12,} ({pct:>6.2f}%)")

    # Long sequence counts
    print(f"\nLong sequence summary:")
    thresholds = [256, 320, 384, 512, 768, 1024]
    for t in thresholds:
        count = (lengths > t).sum()
        if count > 0:
            pct = count / len(lengths) * 100
            print(f"  > {t:>4} tokens: {count:>12,} ({pct:>6.2f}%)")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dataset sequence length distribution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Tokenized HuggingFace dataset
    python dataset_summary.py --path /data/tokenized/foundation --type tokenized

    # Raw parquet files
    python dataset_summary.py --path /data/raw/parquet_dir --type parquet

    # Specific split
    python dataset_summary.py --path /data/tokenized/foundation --split train
"""
    )
    parser.add_argument("--path", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--type", type=str, default="tokenized",
                        choices=["tokenized", "parquet"],
                        help="Dataset type (default: tokenized)")
    parser.add_argument("--split", type=str, default=None,
                        help="Split to analyze (train/val/test). If not specified, analyzes the path directly.")
    parser.add_argument("--n-bins", type=int, default=10,
                        help="Number of quantile bins (default: 10)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count)")

    args = parser.parse_args()

    num_workers = args.num_workers or min(32, cpu_count())

    # Determine path
    data_path = args.path
    if args.split:
        data_path = os.path.join(args.path, args.split)

    print(f"Analyzing: {data_path}")
    print(f"Type: {args.type}")
    print(f"Workers: {num_workers}")

    # Find files/shards
    if args.type == "tokenized":
        # Look for shards
        shards = sorted(glob.glob(os.path.join(data_path, "shard_*")))
        if not shards:
            # Maybe it's a single dataset
            shards = [data_path]
        print(f"Found {len(shards)} shard(s)")
        scan_func = scan_tokenized_shard
        items = shards
    else:
        # Parquet files
        files = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
        print(f"Found {len(files)} parquet file(s)")
        scan_func = scan_parquet_file
        items = files

    if not items:
        print(f"No data found at {data_path}")
        return

    # Scan (sequential for tokenized to avoid HF datasets multiprocessing issues)
    all_lengths = []
    if args.type == "tokenized":
        # Sequential scan for HuggingFace datasets
        for item in tqdm(items, desc="Scanning"):
            lengths = scan_func(item)
            all_lengths.extend(lengths)
    else:
        # Parallel scan for parquet files
        with Pool(num_workers) as pool:
            for lengths in tqdm(
                pool.imap_unordered(scan_func, items),
                total=len(items),
                desc="Scanning"
            ):
                all_lengths.extend(lengths)

    if not all_lengths:
        print("No sequences found!")
        return

    all_lengths = np.array(all_lengths)
    compute_summary(all_lengths, n_bins=args.n_bins)


if __name__ == "__main__":
    main()
