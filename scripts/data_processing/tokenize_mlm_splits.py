#!/usr/bin/env python3
"""
Tokenize pre-computed mlm_full splits for ESM2/ESM-C training.

Three modes:
  train_subset  — Tokenize training subsets from .npy index files
  val_test      — Extract and tokenize val/test splits (unified + per-cardinality)
  binding_probe — Tokenize binding probe parquets with MHC resolution
  all           — Run all three modes

Usage:
    python scripts/data_processing/tokenize_mlm_splits.py \
        --mode train_subset \
        --input-dir data/splits/mlm_full/phase0a_resolved \
        --splits-dir data/splits/mlm_full \
        --output-dir data/tokenized/mlm_full \
        --model-type esm2 \
        --max-length 1024 \
        --num-workers 48 \
        --subset proportional_5M
"""

import argparse
import hashlib
import json
import os
import warnings
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
from datasets import Dataset
from tqdm import tqdm

# Reuse tokenization primitives from esm_tokenizer
from scripts.data_processing.esm_tokenizer import (
    build_ascii_lookup_table,
    init_worker,
    load_tokenizer,
    tokenize_sequence_vectorized,
)

# Reuse split utilities from create_mlm_splits
from scripts.data_processing.create_mlm_splits import (
    _batch_idx_from_path,
    _hash_row,
)

# For permutation key parsing
from scripts.analysis.summarize_dedup_output import parse_permutation_key

# For HLA resolution
from quest.parsers.utils import parse_imgt_four_digit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CARDINALITY_LABELS = {1: "singles", 2: "pairs", 3: "triplets", 4: "quartets", 5: "quintets"}
MOLECULE_ORDER = ["tra", "trb", "peptide", "mhc_one", "mhc_two"]
SHARD_THRESHOLD = 10_000_000


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def load_subset_indices(npy_path: Path) -> Dict[int, np.ndarray]:
    """Load .npy index file and group row indices by batch_idx.

    Returns:
        Dict mapping batch_idx -> sorted array of row_idx values
    """
    arr = np.load(npy_path)  # shape (N, 2), dtype uint32, cols: (batch_idx, row_idx)
    result: Dict[int, np.ndarray] = {}
    # Group by batch_idx
    unique_batches = np.unique(arr[:, 0])
    for bid in unique_batches:
        mask = arr[:, 0] == bid
        result[int(bid)] = np.sort(arr[mask, 1])
    return result


def build_batch_path_map(input_dir: Path) -> Dict[int, Path]:
    """Map batch_idx -> file path from sorted parquet glob."""
    files = sorted(input_dir.glob("batch_*.parquet"))
    result = {}
    for f in files:
        bid = _batch_idx_from_path(f)
        result[bid] = f
    return result


def get_cardinality(permutation_key: str) -> int:
    """Count molecules in a permutation key."""
    return len(parse_permutation_key(permutation_key))


def load_hla_dictionary(hla_dir: Path) -> Dict[str, str]:
    """Load IMGT/HLA allele -> protein sequence dictionary."""
    hla_dict = parse_imgt_four_digit(str(hla_dir))
    if not hla_dict:
        raise ValueError(
            f"No HLA alleles loaded from {hla_dir}. "
            f"Check that directory contains *_prot.fasta files."
        )
    return hla_dict


def _resolve_allele(allele: str, hla_dict: Dict[str, str], prefix_cache: Dict[str, str]) -> Optional[str]:
    """Resolve a single MHC allele to its protein sequence."""
    if not allele or allele in ("", "nan", "None"):
        return None
    lookup_key = allele
    if lookup_key.startswith("HLA-"):
        lookup_key = lookup_key[4:]
    # Truncate to 4-digit resolution
    colon_parts = lookup_key.split(":")
    if len(colon_parts) > 2:
        lookup_key = ":".join(colon_parts[:2])
    full_seq = hla_dict.get(lookup_key)
    if not full_seq:
        resolved_key = prefix_cache.get(lookup_key)
        if resolved_key:
            full_seq = hla_dict.get(resolved_key)
    return full_seq


def _build_prefix_cache(hla_dict: Dict[str, str]) -> Dict[str, str]:
    """Build prefix -> first matching 4-digit allele cache."""
    cache: Dict[str, str] = {}
    for key in sorted(hla_dict.keys()):
        parts = key.split(":")
        for i in range(len(parts), 0, -1):
            prefix = ":".join(parts[:i])
            if prefix not in cache:
                cache[prefix] = key
    return cache


def _compute_token_length_stats(all_lengths: List[int]) -> Dict[str, float]:
    """Compute token length statistics."""
    if not all_lengths:
        return {}
    arr = np.array(all_lengths)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": int(np.max(arr)),
        "min": int(np.min(arr)),
    }


# ---------------------------------------------------------------------------
# Mode 1: Train subset tokenization
# ---------------------------------------------------------------------------

# Worker globals for subset mode
_SUBSET_BATCH_PATH_MAP: Dict[int, Path] = {}


def _process_batch_subset(args: Tuple) -> Dict[str, Any]:
    """Worker: read one batch file, extract indexed rows, tokenize.

    Returns dict with input_ids list, attention_mask list, token_lengths list.
    """
    batch_idx, row_indices = args
    filepath = _SUBSET_BATCH_PATH_MAP[batch_idx]

    table = pq.read_table(filepath, columns=["sequence"])
    sequences = table.column("sequence").to_pylist()
    del table

    input_ids_list = []
    attention_mask_list = []
    token_lengths = []

    for ridx in row_indices:
        seq = sequences[ridx]
        ids, mask = tokenize_sequence_vectorized(seq)
        input_ids_list.append(ids)
        attention_mask_list.append(mask)
        token_lengths.append(len(ids))

    return {
        "batch_idx": batch_idx,
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "token_lengths": token_lengths,
        "count": len(input_ids_list),
    }


def _process_batch_subset_shard(args: Tuple) -> Dict[str, Any]:
    """Worker: read batch, extract indexed rows, tokenize, write shard to disk.

    Returns dict with count and token_lengths (no data returned to main process).
    """
    batch_idx, row_indices, output_dir = args
    filepath = _SUBSET_BATCH_PATH_MAP[batch_idx]

    table = pq.read_table(filepath, columns=["sequence"])
    sequences = table.column("sequence").to_pylist()
    del table

    input_ids_list = []
    attention_mask_list = []
    token_lengths = []

    for ridx in row_indices:
        seq = sequences[ridx]
        ids, mask = tokenize_sequence_vectorized(seq)
        input_ids_list.append(ids)
        attention_mask_list.append(mask)
        token_lengths.append(len(ids))

    if input_ids_list:
        ds = Dataset.from_dict({"input_ids": input_ids_list, "attention_mask": attention_mask_list})
        shard_dir = os.path.join(output_dir, f"shard_batch_{batch_idx:06d}")
        ds.save_to_disk(shard_dir)

    return {
        "batch_idx": batch_idx,
        "count": len(input_ids_list),
        "token_lengths": token_lengths,
    }


def _init_subset_worker(lut, cls_id, eos_id, max_length, sep_char, batch_path_map):
    """Initialize worker with tokenizer globals and batch path map."""
    init_worker(lut, cls_id, eos_id, max_length, sep_char)
    global _SUBSET_BATCH_PATH_MAP
    _SUBSET_BATCH_PATH_MAP = batch_path_map


def tokenize_train_subset(
    subset_name: str,
    splits_dir: Path,
    input_dir: Path,
    output_dir: Path,
    lut: np.ndarray,
    cls_id: int,
    eos_id: int,
    max_length: int,
    sep_char: str,
    num_workers: int,
    shard_threshold: int,
) -> Dict[str, Any]:
    """Tokenize a single training subset from its .npy index file."""
    npy_path = splits_dir / "phase3_subsets" / f"{subset_name}.npy"
    if not npy_path.exists():
        raise FileNotFoundError(f"Subset index file not found: {npy_path}")

    print(f"\n  Tokenizing subset: {subset_name}")
    indices_by_batch = load_subset_indices(npy_path)
    total_rows = sum(len(v) for v in indices_by_batch.values())
    print(f"    Total rows: {total_rows:,} across {len(indices_by_batch)} batches")

    batch_path_map = build_batch_path_map(input_dir)

    # Verify all required batches exist
    missing = [b for b in indices_by_batch if b not in batch_path_map]
    if missing:
        raise ValueError(f"Missing batch files for batch_idx: {sorted(missing)[:10]}")

    subset_output_dir = output_dir / "train" / subset_name
    subset_output_dir.mkdir(parents=True, exist_ok=True)

    use_sharding = total_rows > shard_threshold

    if use_sharding:
        print(f"    Using sharded output (>{shard_threshold:,} rows)")
        worker_args = [
            (bid, indices_by_batch[bid], str(subset_output_dir))
            for bid in sorted(indices_by_batch.keys())
        ]

        all_token_lengths = []
        total_count = 0

        with Pool(
            num_workers,
            initializer=_init_subset_worker,
            initargs=(lut, cls_id, eos_id, max_length, sep_char, batch_path_map),
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(_process_batch_subset_shard, worker_args),
                total=len(worker_args),
                desc=f"    {subset_name}",
            ):
                total_count += result["count"]
                all_token_lengths.extend(result["token_lengths"])

        fmt = "sharded"
    else:
        print(f"    Using single dataset output (<={shard_threshold:,} rows)")
        worker_args = [
            (bid, indices_by_batch[bid])
            for bid in sorted(indices_by_batch.keys())
        ]

        all_input_ids = []
        all_attention_masks = []
        all_token_lengths = []

        with Pool(
            num_workers,
            initializer=_init_subset_worker,
            initargs=(lut, cls_id, eos_id, max_length, sep_char, batch_path_map),
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(_process_batch_subset, worker_args),
                total=len(worker_args),
                desc=f"    {subset_name}",
            ):
                all_input_ids.extend(result["input_ids"])
                all_attention_masks.extend(result["attention_mask"])
                all_token_lengths.extend(result["token_lengths"])

        total_count = len(all_input_ids)
        ds = Dataset.from_dict({"input_ids": all_input_ids, "attention_mask": all_attention_masks})
        ds.save_to_disk(str(subset_output_dir))
        del all_input_ids, all_attention_masks
        fmt = "single"

    print(f"    Saved {total_count:,} rows ({fmt}) to {subset_output_dir}")

    disk_size = sum(
        f.stat().st_size for f in subset_output_dir.rglob("*") if f.is_file()
    )

    return {
        "rows": total_count,
        "format": fmt,
        "disk_size_bytes": disk_size,
        "token_length_stats": _compute_token_length_stats(all_token_lengths),
    }


# ---------------------------------------------------------------------------
# Mode 2: Val/test tokenization
# ---------------------------------------------------------------------------

# Worker globals for val/test mode
_VT_EXPLICIT_LOOKUP: Dict[Tuple[int, int], str] = {}
_VT_SEED: int = 42


def _init_valtest_worker(lut, cls_id, eos_id, max_length, sep_char, explicit_lookup, seed):
    """Initialize worker with tokenizer globals, explicit lookup, and seed."""
    init_worker(lut, cls_id, eos_id, max_length, sep_char)
    global _VT_EXPLICIT_LOOKUP, _VT_SEED
    _VT_EXPLICIT_LOOKUP = explicit_lookup
    _VT_SEED = seed


def _process_batch_valtest(args: Tuple) -> Dict[str, Any]:
    """Worker: classify rows as val/test, tokenize, group by (split, cardinality).

    Returns dict of {(split, cardinality_label): {input_ids, attention_mask, token_lengths, count}}.
    """
    filepath, batch_idx = args
    global _VT_EXPLICIT_LOOKUP, _VT_SEED

    table = pq.read_table(filepath, columns=["permutation_key", "sequence"])
    perm_keys = table.column("permutation_key").to_pylist()
    sequences = table.column("sequence").to_pylist()
    del table

    # Group results by (split, cardinality_label)
    buckets: Dict[str, Dict[str, list]] = {}

    perm_cache: Dict[str, int] = {}

    for row_idx, (pk, seq) in enumerate(zip(perm_keys, sequences)):
        # Determine split
        key = (int(batch_idx), int(row_idx))
        if key in _VT_EXPLICIT_LOOKUP:
            split = _VT_EXPLICIT_LOOKUP[key]
        else:
            h = _hash_row(_VT_SEED, batch_idx, row_idx)
            if h < 5:
                split = "test"
            elif h < 10:
                split = "val"
            else:
                continue  # train row, skip

        if split == "train":
            continue

        # Tokenize
        ids, mask = tokenize_sequence_vectorized(seq)
        tok_len = len(ids)

        # Get cardinality
        if pk not in perm_cache:
            perm_cache[pk] = get_cardinality(pk)
        card = perm_cache[pk]
        card_label = CARDINALITY_LABELS.get(card, f"card_{card}")

        # Store in unified bucket
        unified_key = split
        if unified_key not in buckets:
            buckets[unified_key] = {"input_ids": [], "attention_mask": [], "token_lengths": []}
        buckets[unified_key]["input_ids"].append(ids)
        buckets[unified_key]["attention_mask"].append(mask)
        buckets[unified_key]["token_lengths"].append(tok_len)

        # Store in per-cardinality bucket
        card_key = f"{split}_{card_label}"
        if card_key not in buckets:
            buckets[card_key] = {"input_ids": [], "attention_mask": [], "token_lengths": []}
        buckets[card_key]["input_ids"].append(ids)
        buckets[card_key]["attention_mask"].append(mask)
        buckets[card_key]["token_lengths"].append(tok_len)

    # Convert to counts
    result = {}
    for bkey, bdata in buckets.items():
        result[bkey] = {
            "input_ids": bdata["input_ids"],
            "attention_mask": bdata["attention_mask"],
            "token_lengths": bdata["token_lengths"],
            "count": len(bdata["input_ids"]),
        }

    return result


def tokenize_val_test(
    input_dir: Path,
    splits_dir: Path,
    output_dir: Path,
    lut: np.ndarray,
    cls_id: int,
    eos_id: int,
    max_length: int,
    sep_char: str,
    num_workers: int,
    seed: int,
) -> Dict[str, Any]:
    """Tokenize val and test splits, both unified and per-cardinality."""
    print("\n  Tokenizing val/test splits...")

    # Load explicit split assignments
    sa_path = splits_dir / "phase2_splits" / "split_assignments.parquet"
    if not sa_path.exists():
        raise FileNotFoundError(f"Split assignments not found: {sa_path}")

    sa_table = pq.read_table(sa_path)
    sa_df = sa_table.to_pandas()
    explicit_lookup = {
        (int(r.batch_idx), int(r.row_idx)): r.split
        for r in sa_df.itertuples(index=False)
    }
    print(f"    Loaded {len(explicit_lookup):,} explicit split assignments")
    del sa_df, sa_table

    # Glob batch files
    batch_path_map = build_batch_path_map(input_dir)
    batch_files = sorted(batch_path_map.items())
    print(f"    Processing {len(batch_files)} batch files")

    worker_args = [(str(path), bid) for bid, path in batch_files]

    # Accumulate results by bucket key
    merged: Dict[str, Dict[str, list]] = {}

    with Pool(
        num_workers,
        initializer=_init_valtest_worker,
        initargs=(lut, cls_id, eos_id, max_length, sep_char, explicit_lookup, seed),
    ) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_batch_valtest, worker_args),
            total=len(worker_args),
            desc="    val/test",
        ):
            for bkey, bdata in result.items():
                if bkey not in merged:
                    merged[bkey] = {"input_ids": [], "attention_mask": [], "token_lengths": []}
                merged[bkey]["input_ids"].extend(bdata["input_ids"])
                merged[bkey]["attention_mask"].extend(bdata["attention_mask"])
                merged[bkey]["token_lengths"].extend(bdata["token_lengths"])

    # Save each bucket as a HuggingFace Dataset
    dataset_results = {}
    for bkey in sorted(merged.keys()):
        bdata = merged[bkey]
        count = len(bdata["input_ids"])
        if count == 0:
            continue

        ds_dir = output_dir / bkey
        ds_dir.mkdir(parents=True, exist_ok=True)

        ds = Dataset.from_dict({
            "input_ids": bdata["input_ids"],
            "attention_mask": bdata["attention_mask"],
        })
        ds.save_to_disk(str(ds_dir))

        disk_size = sum(f.stat().st_size for f in ds_dir.rglob("*") if f.is_file())
        stats = _compute_token_length_stats(bdata["token_lengths"])

        dataset_results[bkey] = {
            "rows": count,
            "format": "single",
            "disk_size_bytes": disk_size,
            "token_length_stats": stats,
        }
        print(f"    {bkey}: {count:,} rows saved")

    del merged
    return dataset_results


# ---------------------------------------------------------------------------
# Mode 3: Binding probe tokenization
# ---------------------------------------------------------------------------


def stitch_binding_probe_row(
    row: dict,
    hla_dict: Dict[str, str],
    prefix_cache: Dict[str, str],
) -> Tuple[Optional[str], str]:
    """Stitch molecule columns into space-separated sequence, resolve MHC.

    Returns:
        (stitched_sequence, permutation_key) or (None, "") if no valid molecules
    """
    parts = []
    fields = []

    for mol in MOLECULE_ORDER:
        val = row.get(mol, "")
        if not val or val in ("", "nan", "None"):
            continue

        if mol in ("mhc_one", "mhc_two"):
            resolved = _resolve_allele(val, hla_dict, prefix_cache)
            if resolved:
                parts.append(resolved)
                fields.append(mol)
            # If MHC can't be resolved, skip it
        else:
            parts.append(val)
            fields.append(mol)

    if not parts:
        return None, ""

    perm_key = "_".join(fields)
    return " ".join(parts), perm_key


def tokenize_binding_probes(
    splits_dir: Path,
    output_dir: Path,
    hla_dir: Path,
    lut: np.ndarray,
    cls_id: int,
    eos_id: int,
    max_length: int,
    sep_char: str,
) -> Dict[str, Any]:
    """Tokenize binding probe val/test parquets."""
    print("\n  Tokenizing binding probes...")

    # Load HLA dictionary
    hla_dict = load_hla_dictionary(hla_dir)
    prefix_cache = _build_prefix_cache(hla_dict)
    print(f"    Loaded {len(hla_dict):,} HLA alleles")

    # Initialize tokenizer globals in main process (no multiprocessing needed for ~15K rows)
    init_worker(lut, cls_id, eos_id, max_length, sep_char)

    probe_dir = splits_dir / "phase4_binding_probes"
    dataset_results = {}

    for split in ["val", "test"]:
        probe_path = probe_dir / f"binding_probe_{split}.parquet"
        if not probe_path.exists():
            print(f"    WARNING: {probe_path} not found, skipping")
            continue

        table = pq.read_table(probe_path)
        df = table.to_pandas()
        print(f"    {split}: {len(df):,} rows")

        input_ids_list = []
        attention_mask_list = []
        token_lengths = []
        binding_list = []
        label_list = []
        source_list = []
        leakage_flag_list = []
        perm_key_list = []
        skipped = 0

        for _, row in df.iterrows():
            stitched, perm_key = stitch_binding_probe_row(row, hla_dict, prefix_cache)
            if stitched is None:
                skipped += 1
                continue

            ids, mask = tokenize_sequence_vectorized(stitched)
            input_ids_list.append(ids)
            attention_mask_list.append(mask)
            token_lengths.append(len(ids))
            binding_list.append(row.get("binding", ""))
            label_list.append(row.get("label", ""))
            source_list.append(row.get("source", ""))
            leakage_flag_list.append(bool(row.get("leakage_flag", False)))
            perm_key_list.append(perm_key)

        if skipped > 0:
            print(f"    WARNING: Skipped {skipped} rows with no valid molecules")

        ds_dir = output_dir / "binding_probe" / split
        ds_dir.mkdir(parents=True, exist_ok=True)

        ds = Dataset.from_dict({
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "binding": binding_list,
            "label": label_list,
            "source": source_list,
            "leakage_flag": leakage_flag_list,
            "permutation_key": perm_key_list,
        })
        ds.save_to_disk(str(ds_dir))

        disk_size = sum(f.stat().st_size for f in ds_dir.rglob("*") if f.is_file())
        stats = _compute_token_length_stats(token_lengths)

        # Label distribution
        from collections import Counter
        label_dist = dict(Counter(label_list))

        dataset_results[f"binding_probe/{split}"] = {
            "rows": len(input_ids_list),
            "format": "single",
            "disk_size_bytes": disk_size,
            "token_length_stats": stats,
            "extra_columns": ["binding", "label", "source", "leakage_flag", "permutation_key"],
            "label_distribution": label_dist,
        }
        print(f"    {split}: {len(input_ids_list):,} rows saved, labels: {label_dist}")

    return dataset_results


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def write_manifest(
    output_dir: Path,
    model_type: str,
    max_length: int,
    sep_char: str,
    source_dir: str,
    datasets: Dict[str, Any],
):
    """Write manifest.json with dataset metadata."""
    manifest = {
        "creation_date": datetime.now().isoformat(),
        "model_type": model_type,
        "max_length": max_length,
        "separator": sep_char,
        "source_dir": source_dir,
        "datasets": datasets,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest saved to {manifest_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize pre-computed mlm_full splits for ESM2/ESM-C training"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train_subset", "val_test", "binding_probe", "all"],
        help="Tokenization mode",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/splits/mlm_full/phase0a_resolved",
        help="Directory with resolved batch parquet files",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default="data/splits/mlm_full",
        help="Directory with split assignments and subset .npy files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/tokenized/mlm_full",
        help="Output directory for tokenized datasets",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="esm2",
        choices=["esm2", "esmc"],
        help="Model family: esm2 or esmc",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/esm2_t12_35M_UR50D",
        help="HuggingFace model name (only used for esm2)",
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        help="Subset name (e.g. proportional_5M) or 'all' for all subsets",
    )
    parser.add_argument(
        "--hla-dir",
        type=str,
        default="data/databases/IMGTHLA/fasta",
        help="IMGT/HLA FASTA directory (for binding_probe mode)",
    )
    parser.add_argument(
        "--shard-threshold",
        type=int,
        default=SHARD_THRESHOLD,
        help=f"Write shards above this row count (default: {SHARD_THRESHOLD:,})",
    )

    args = parser.parse_args()

    if args.num_workers is None:
        args.num_workers = cpu_count()

    input_dir = Path(args.input_dir)
    splits_dir = Path(args.splits_dir)
    model_output_dir = Path(args.output_dir) / args.model_type
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer, default_sep = load_tokenizer(args.model_type, args.model_name)
    sep_char = default_sep
    lut = build_ascii_lookup_table(tokenizer, separator=sep_char)
    cls_id = tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id

    print("=" * 70)
    print("TOKENIZE MLM SPLITS")
    print("=" * 70)
    print(f"  Mode:         {args.mode}")
    print(f"  Model type:   {args.model_type}")
    print(f"  Separator:    {sep_char!r}")
    print(f"  Max length:   {args.max_length}")
    print(f"  Workers:      {args.num_workers}")
    print(f"  Input dir:    {input_dir}")
    print(f"  Splits dir:   {splits_dir}")
    print(f"  Output dir:   {model_output_dir}")
    print("=" * 70)

    all_dataset_results: Dict[str, Any] = {}

    # --- Mode: train_subset ---
    if args.mode in ("train_subset", "all"):
        # Discover available subsets
        subsets_dir = splits_dir / "phase3_subsets"
        available = sorted([f.stem for f in subsets_dir.glob("*.npy")])

        if args.subset == "all":
            subset_names = available
        else:
            subset_names = [args.subset]
            if args.subset not in available:
                raise ValueError(
                    f"Subset '{args.subset}' not found. Available: {available}"
                )

        print(f"\n  Training subsets to tokenize: {subset_names}")

        for subset_name in subset_names:
            result = tokenize_train_subset(
                subset_name=subset_name,
                splits_dir=splits_dir,
                input_dir=input_dir,
                output_dir=model_output_dir,
                lut=lut,
                cls_id=cls_id,
                eos_id=eos_id,
                max_length=args.max_length,
                sep_char=sep_char,
                num_workers=args.num_workers,
                shard_threshold=args.shard_threshold,
            )
            all_dataset_results[f"train/{subset_name}"] = result

    # --- Mode: val_test ---
    if args.mode in ("val_test", "all"):
        vt_results = tokenize_val_test(
            input_dir=input_dir,
            splits_dir=splits_dir,
            output_dir=model_output_dir,
            lut=lut,
            cls_id=cls_id,
            eos_id=eos_id,
            max_length=args.max_length,
            sep_char=sep_char,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        all_dataset_results.update(vt_results)

    # --- Mode: binding_probe ---
    if args.mode in ("binding_probe", "all"):
        bp_results = tokenize_binding_probes(
            splits_dir=splits_dir,
            output_dir=model_output_dir,
            hla_dir=Path(args.hla_dir),
            lut=lut,
            cls_id=cls_id,
            eos_id=eos_id,
            max_length=args.max_length,
            sep_char=sep_char,
        )
        all_dataset_results.update(bp_results)

    # --- Save tokenizer config ---
    config = {
        "model_type": args.model_type,
        "model_name": args.model_name if args.model_type == "esm2" else "esmc",
        "separator": sep_char,
        "separator_token_id": int(lut[ord(sep_char)]),
        "cls_token_id": cls_id,
        "eos_token_id": eos_id,
        "max_length": args.max_length,
    }
    config_path = model_output_dir / "tokenizer_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # --- Write manifest ---
    write_manifest(
        output_dir=model_output_dir,
        model_type=args.model_type,
        max_length=args.max_length,
        sep_char=sep_char,
        source_dir=str(input_dir),
        datasets=all_dataset_results,
    )

    print("\n" + "=" * 70)
    print("TOKENIZATION COMPLETE")
    print("=" * 70)
    for name, info in sorted(all_dataset_results.items()):
        rows = info.get("rows", 0)
        fmt = info.get("format", "?")
        size_mb = info.get("disk_size_bytes", 0) / (1024 * 1024)
        print(f"  {name:<40} {rows:>12,} rows  {fmt:<8}  {size_mb:>8.1f} MB")


if __name__ == "__main__":
    main()
