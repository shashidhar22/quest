#!/usr/bin/env python3
"""
ESM-2 tokenizer with stratified train/val/test splitting.
Optimized for parallel processing on multi-core systems (e.g., x2gd.16xlarge with 64 vCPUs).

Two-phase approach:
1. Scan all files to compute global stratified split assignments
2. Parallel tokenization with pre-assigned splits
"""

import argparse
import glob
import os
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm


# Global lookup table for vectorized tokenization (initialized in workers)
WORKER_LUT = None
WORKER_CLS_ID = None
WORKER_EOS_ID = None
WORKER_MAX_LENGTH = None


def build_ascii_lookup_table(tokenizer) -> np.ndarray:
    """Build a 256-entry ASCII to token ID lookup table for vectorized tokenization."""
    unk_token_id = tokenizer.unk_token_id
    lut = np.full(256, unk_token_id, dtype=np.int32)

    for token, idx in tokenizer.get_vocab().items():
        if len(token) == 1:  # Single character tokens (amino acids)
            lut[ord(token)] = idx

    return lut


def tokenize_sequence_vectorized(seq: str) -> Tuple[List[int], List[int]]:
    """Tokenize a single sequence using numpy lookup table.

    Uses global WORKER_LUT, WORKER_CLS_ID, WORKER_EOS_ID, WORKER_MAX_LENGTH
    initialized by init_worker().
    """
    max_seq_len = WORKER_MAX_LENGTH - 2

    # Truncate if needed
    if len(seq) > max_seq_len:
        seq = seq[:max_seq_len]

    # Convert to ASCII bytes and apply lookup
    arr = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
    tokens = WORKER_LUT[arr]

    # Build input_ids with CLS/EOS
    input_ids = np.empty(len(tokens) + 2, dtype=np.int32)
    input_ids[0] = WORKER_CLS_ID
    input_ids[1:-1] = tokens
    input_ids[-1] = WORKER_EOS_ID

    return input_ids.tolist(), [1] * len(input_ids)


def scan_file_keys(file_path: str) -> Tuple[str, List[str]]:
    """Read only permutation_key column from a parquet file."""
    table = pq.read_table(file_path, columns=['permutation_key'])
    keys = table.column('permutation_key').to_pylist()
    return file_path, keys


def compute_global_split_assignments(
    file_paths: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 64
) -> Tuple[Dict[str, np.ndarray], Counter, Counter, Counter]:
    """
    Scan all parquet files and compute stratified split assignments.

    Returns:
        - Dict mapping file_path -> np.ndarray of split assignments (0=train, 1=val, 2=test)
        - Counters for train/val/test permutation key distributions
    """
    print(f"Phase 1: Scanning {len(file_paths)} files for permutation keys...")

    # Parallel scan of permutation keys
    all_keys = {}
    with Pool(num_workers) as pool:
        for file_path, keys in tqdm(
            pool.imap_unordered(scan_file_keys, file_paths),
            total=len(file_paths),
            desc="Scanning files"
        ):
            all_keys[file_path] = keys

    # Build file boundaries for efficient slicing later
    print("Building global permutation key index...")
    file_sizes = [len(all_keys[fp]) for fp in file_paths]
    file_boundaries = np.cumsum([0] + file_sizes)  # [0, n1, n1+n2, ...]
    total_rows = file_boundaries[-1]

    # Concatenate all keys efficiently
    global_perm_keys = np.concatenate([np.array(all_keys[fp]) for fp in file_paths])
    print(f"Total sequences: {total_rows:,}")

    # Stratified split across entire dataset
    print("Computing stratified splits...")
    rng = np.random.default_rng(seed)

    unique_keys, inverse_indices = np.unique(global_perm_keys, return_inverse=True)
    split_assignments = np.empty(total_rows, dtype=np.int8)

    for key_idx, key in enumerate(tqdm(unique_keys, desc="Splitting by key")):
        mask = inverse_indices == key_idx
        key_indices = np.where(mask)[0]
        rng.shuffle(key_indices)

        n = len(key_indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        split_assignments[key_indices[:n_train]] = 0  # train
        split_assignments[key_indices[n_train:n_train + n_val]] = 1  # val
        split_assignments[key_indices[n_train + n_val:]] = 2  # test

    # Slice split_assignments by file boundaries (vectorized, no Python loop over rows)
    print("Organizing split assignments by file...")
    file_splits = {}
    for i, fp in enumerate(file_paths):
        start, end = file_boundaries[i], file_boundaries[i + 1]
        file_splits[fp] = split_assignments[start:end]

    # Vectorized counting using numpy (no Python loop over 1B+ rows)
    print("Computing split distributions...")
    train_counts = Counter()
    val_counts = Counter()
    test_counts = Counter()

    for key_idx, key in enumerate(unique_keys):
        mask = inverse_indices == key_idx
        key_splits = split_assignments[mask]
        train_counts[key] = int(np.sum(key_splits == 0))
        val_counts[key] = int(np.sum(key_splits == 1))
        test_counts[key] = int(np.sum(key_splits == 2))

    return file_splits, train_counts, val_counts, test_counts


def init_worker(lut: np.ndarray, cls_id: int, eos_id: int, max_length: int):
    """Initialize worker process with lookup table and token IDs."""
    global WORKER_LUT, WORKER_CLS_ID, WORKER_EOS_ID, WORKER_MAX_LENGTH
    WORKER_LUT = lut
    WORKER_CLS_ID = cls_id
    WORKER_EOS_ID = eos_id
    WORKER_MAX_LENGTH = max_length


def process_file_worker(args: Tuple) -> Dict:
    """
    Worker function to process a single parquet file.

    Args:
        args: (file_path, split_assignments, output_dirs)

    Returns:
        Dict with status and counts
    """
    file_path, split_assignments, train_dir, val_dir, test_dir = args

    try:
        # Read file
        table = pq.read_table(file_path, columns=['sequence'])
        sequences = table.column('sequence').to_pylist()
        del table

        # Separate and tokenize by split
        train_ids, train_masks = [], []
        val_ids, val_masks = [], []
        test_ids, test_masks = [], []

        for i, seq in enumerate(sequences):
            split = split_assignments[i]
            input_ids, attention_mask = tokenize_sequence_vectorized(seq)

            if split == 0:
                train_ids.append(input_ids)
                train_masks.append(attention_mask)
            elif split == 1:
                val_ids.append(input_ids)
                val_masks.append(attention_mask)
            else:
                test_ids.append(input_ids)
                test_masks.append(attention_mask)

        del sequences

        # Generate shard name from file path
        file_stem = Path(file_path).stem
        shard_name = f"shard_{file_stem}"

        # Save outputs
        counts = {'train': 0, 'val': 0, 'test': 0}

        if train_ids:
            ds = Dataset.from_dict({'input_ids': train_ids, 'attention_mask': train_masks})
            ds.save_to_disk(str(Path(train_dir) / shard_name))
            counts['train'] = len(train_ids)
        del train_ids, train_masks

        if val_ids:
            ds = Dataset.from_dict({'input_ids': val_ids, 'attention_mask': val_masks})
            ds.save_to_disk(str(Path(val_dir) / shard_name))
            counts['val'] = len(val_ids)
        del val_ids, val_masks

        if test_ids:
            ds = Dataset.from_dict({'input_ids': test_ids, 'attention_mask': test_masks})
            ds.save_to_disk(str(Path(test_dir) / shard_name))
            counts['test'] = len(test_ids)
        del test_ids, test_masks

        return {'status': 'success', 'file': file_path, 'counts': counts}

    except Exception as e:
        return {'status': 'error', 'file': file_path, 'error': str(e)}


def process_and_split_parallel(
    input_dir: str,
    output_dir: str,
    model_name: str = "facebook/esm2_t12_35M_UR50D",
    max_length: int = 1024,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    num_workers: int = None,
    seed: int = 42,
):
    """
    Main entry point for parallel tokenization with stratified splitting.

    Two-phase approach:
    1. Scan all files to compute global stratified split assignments
    2. Parallel tokenization with pre-assigned splits
    """
    if num_workers is None:
        num_workers = cpu_count()

    print("=" * 70)
    print("ESM-2 PARALLEL TOKENIZER")
    print("=" * 70)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model:            {model_name}")
    print(f"Max length:       {max_length}")
    print(f"Split ratios:     train={train_ratio:.0%}, val={val_ratio:.0%}, test={test_ratio:.0%}")
    print(f"Workers:          {num_workers}")
    print(f"Seed:             {seed}")
    print("=" * 70)

    # Load tokenizer and build lookup table
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lut = build_ascii_lookup_table(tokenizer)
    cls_token_id = tokenizer.cls_token_id
    eos_token_id = tokenizer.eos_token_id

    vocab_size = np.sum(lut != tokenizer.unk_token_id)
    print(f"  Vocab size: {vocab_size} amino acids")
    print(f"  CLS={cls_token_id}, EOS={eos_token_id}, UNK={tokenizer.unk_token_id}")

    # Find input files
    file_paths = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    print(f"\nFound {len(file_paths)} input parquet files")

    if not file_paths:
        raise ValueError(f"No parquet files found in {input_dir}")

    # Create output directories
    train_dir = Path(output_dir) / "train"
    val_dir = Path(output_dir) / "validation"
    test_dir = Path(output_dir) / "test"

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Phase 1: Compute global split assignments
    print("\n" + "=" * 70)
    print("PHASE 1: COMPUTING GLOBAL SPLIT ASSIGNMENTS")
    print("=" * 70)

    file_splits, train_counts, val_counts, test_counts = compute_global_split_assignments(
        file_paths=file_paths,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        num_workers=num_workers
    )

    # Phase 2: Parallel tokenization
    print("\n" + "=" * 70)
    print("PHASE 2: PARALLEL TOKENIZATION")
    print("=" * 70)

    # Prepare worker arguments
    worker_args = [
        (fp, file_splits[fp], str(train_dir), str(val_dir), str(test_dir))
        for fp in file_paths
    ]

    # Process in parallel
    total_counts = {'train': 0, 'val': 0, 'test': 0}
    failed_files = []

    with Pool(
        num_workers,
        initializer=init_worker,
        initargs=(lut, cls_token_id, eos_token_id, max_length)
    ) as pool:
        with tqdm(total=len(worker_args), desc="Tokenizing files") as pbar:
            for result in pool.imap_unordered(process_file_worker, worker_args):
                if result['status'] == 'success':
                    for split, n in result['counts'].items():
                        total_counts[split] += n
                else:
                    failed_files.append((result['file'], result['error']))
                    print(f"\nERROR: {result['file']}: {result['error']}")
                pbar.update(1)

    # Summary
    print("\n" + "=" * 70)
    print("TOKENIZATION COMPLETE")
    print("=" * 70)

    if failed_files:
        print(f"\nWARNING: {len(failed_files)} files failed to process:")
        for fp, err in failed_files[:10]:
            print(f"  {fp}: {err}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")

    total = sum(total_counts.values())
    if total > 0:
        print(f"\nTotal sequences:")
        print(f"  Train:      {total_counts['train']:>15,} ({total_counts['train']/total*100:.1f}%)")
        print(f"  Validation: {total_counts['val']:>15,} ({total_counts['val']/total*100:.1f}%)")
        print(f"  Test:       {total_counts['test']:>15,} ({total_counts['test']/total*100:.1f}%)")
        print(f"  Total:      {total:>15,}")

        print("\nPermutation key distribution (train):")
        t = sum(train_counts.values())
        for k, c in sorted(train_counts.items(), key=lambda x: -x[1]):
            print(f"  {k:<30} {c:>12,} ({c/t*100:5.1f}%)")

        print("\nPermutation key distribution (validation):")
        t = sum(val_counts.values())
        for k, c in sorted(val_counts.items(), key=lambda x: -x[1]):
            print(f"  {k:<30} {c:>12,} ({c/t*100:5.1f}%)")

    print(f"\nOutput saved to: {output_dir}")
    print(f"  Train shards:      {train_dir}")
    print(f"  Validation shards: {val_dir}")
    print(f"  Test shards:       {test_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Parallel tokenization with stratified train/val/test splitting"
    )
    parser.add_argument("--input-dir", type=str,
                        default="/home/ubuntu/quest/data/deduplicated/full/foundation")
    parser.add_argument("--output-dir", type=str,
                        default="/home/ubuntu/quest/data/tokenized/full/foundation_stratified")
    parser.add_argument("--model", type=str, default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of parallel workers (default: number of CPUs)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.001:
        parser.error("Ratios must sum to 1.0")

    process_and_split_parallel(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        max_length=args.max_length,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
