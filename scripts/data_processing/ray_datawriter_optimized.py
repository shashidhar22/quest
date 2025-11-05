#!/usr/bin/env python3
"""
Ray-based data writer with optimized deduplication.
Three-stage pipeline: molecule dedup → permutation generation → permutation dedup
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import ray
from tqdm import tqdm

# Import everything from original
import sys
from pathlib import Path

# Add data_processing directory to path
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

from ray_datawriter import *
from tqdm.auto import tqdm

# Ensure critical global variables are defined (in case import * didn't work)
if '_valid_aa' not in globals():
    _valid_aa = set("ACDEFGHIKLMNPQRSTVWY")

# Monkey-patch print_progress to use tqdm-compatible output
def print_progress_tqdm(stage: str, message: str, indent: int = 1):
    """tqdm-friendly progress reporting."""
    prefix = "   " * indent
    # Use tqdm.write to avoid conflicts with progress bars
    tqdm.write(f"{prefix}{stage} {message}")

# Replace the original print_progress
print_progress = print_progress_tqdm

def deduplicate_ray_data_hash_based(ds: ray.data.Dataset, mode: str) -> tuple:
    """
    Hash-based deduplication using groupby.
    
    WARNING: This requires a full shuffle operation which can be VERY slow for large datasets.
    The groupby() operation needs to shuffle all 2.6B rows across partitions to group by key.
    
    For most use cases, sort-based deduplication is actually FASTER because:
    - Ray's sort is highly optimized with external merge sort
    - Consecutive duplicate removal is a simple map operation
    - No expensive cross-partition shuffle needed
    
    Only use this if you have specific reasons (e.g., pre-partitioned data).
    """
    from tqdm.auto import tqdm
    
    print(f"\n⚡ HASH-BASED DEDUPLICATION (Mode: {mode})")
    print("="*80)
    print("Using hash partitioning instead of sorting for faster processing")
    
    import time
    start_time = time.time()
    
    # Same key generation as original
    def add_dedup_key(row):
        """Add validation and dedup key to each row."""
        if mode == "tra":
            tra = row.get('tra', '')
            valid = bool(tra) and tra != 'NA' and _valid_aa.issuperset(tra)
            return {**row, '_dedup_key': tra if valid else '', '_valid': valid}
        
        elif mode == "trb":
            trb = row.get('trb', '')
            valid = bool(trb) and trb != 'NA' and _valid_aa.issuperset(trb)
            return {**row, '_dedup_key': trb if valid else '', '_valid': valid}
        
        elif mode == "tcr_pairing":
            tra = row.get('tra', '')
            trb = row.get('trb', '')
            tra_valid = bool(tra) and tra != 'NA' and _valid_aa.issuperset(tra)
            trb_valid = bool(trb) and trb != 'NA' and _valid_aa.issuperset(trb)
            valid = tra_valid and trb_valid
            return {**row, '_dedup_key': f"{tra}|{trb}" if valid else '', '_valid': valid}
        
        elif mode == "mhc_binding":
            pep = row.get('peptide', '')
            mho = row.get('mhc_one', '')
            mht = row.get('mhc_two', '')
            pep_valid = bool(pep) and pep != 'NA' and _valid_aa.issuperset(pep)
            mho_valid = bool(mho) and mho != 'NA' and _valid_aa.issuperset(mho)
            mht_valid = bool(mht) and mht != 'NA' and _valid_aa.issuperset(mht)
            valid = pep_valid and mho_valid
            return {**row, '_dedup_key': f"{pep}|{mho}|{mht if mht_valid else ''}" if valid else '', '_valid': valid}
        
        elif mode == "specificity":
            tra = row.get('tra', '')
            trb = row.get('trb', '')
            pep = row.get('peptide', '')
            mho = row.get('mhc_one', '')
            mht = row.get('mhc_two', '')
            
            tra_valid = bool(tra) and tra != 'NA' and _valid_aa.issuperset(tra)
            trb_valid = bool(trb) and trb != 'NA' and _valid_aa.issuperset(trb)
            pep_valid = bool(pep) and pep != 'NA' and _valid_aa.issuperset(pep)
            mho_valid = bool(mho) and mho != 'NA' and _valid_aa.issuperset(mho)
            mht_valid = bool(mht) and mht != 'NA' and _valid_aa.issuperset(mht)
            
            valid = pep_valid and mho_valid and (tra_valid or trb_valid)
            
            if valid:
                parts = []
                if mho_valid: parts.append(f"mhc_one:{mho}")
                if mht_valid: parts.append(f"mhc_two:{mht}")
                if pep_valid: parts.append(f"peptide:{pep}")
                if tra_valid: parts.append(f"tra:{tra}")
                if trb_valid: parts.append(f"trb:{trb}")
                parts.sort()
                dedup_key = '|'.join(parts)
            else:
                dedup_key = ''
            
            return {**row, '_dedup_key': dedup_key, '_valid': valid}
        
        else:  # default or balanced
            tra = row.get('tra', '')
            trb = row.get('trb', '')
            pep = row.get('peptide', '')
            mho = row.get('mhc_one', '')
            mht = row.get('mhc_two', '')
            
            tra_valid = bool(tra) and tra != 'NA' and _valid_aa.issuperset(tra)
            trb_valid = bool(trb) and trb != 'NA' and _valid_aa.issuperset(trb)
            pep_valid = bool(pep) and pep != 'NA' and _valid_aa.issuperset(pep)
            mho_valid = bool(mho) and mho != 'NA' and _valid_aa.issuperset(mho)
            mht_valid = bool(mht) and mht != 'NA' and _valid_aa.issuperset(mht)
            
            parts = []
            if mho_valid: parts.append(f"mhc_one:{mho}")
            if mht_valid: parts.append(f"mhc_two:{mht}")
            if pep_valid: parts.append(f"peptide:{pep}")
            if tra_valid: parts.append(f"tra:{tra}")
            if trb_valid: parts.append(f"trb:{trb}")
            
            valid = len(parts) > 0
            if valid:
                parts.sort()
                dedup_key = '|'.join(parts)
            else:
                dedup_key = ''
            
            return {**row, '_dedup_key': dedup_key, '_valid': valid}
    
    # Pipeline: Add keys and filter
    print("\n📊 Step 1/3: Adding dedup keys and filtering invalid sequences...")
    with tqdm(desc="Adding dedup keys", unit=" rows", unit_scale=True) as pbar:
        ds_with_keys = ds.map(add_dedup_key)
        ds_filtered = ds_with_keys.filter(lambda row: row['_valid'])
        # Force execution to show progress
        ds_filtered = ds_filtered.materialize()
        count_after_filter = ds_filtered.count()
        pbar.update(count_after_filter)
        pbar.set_postfix({"valid_rows": f"{count_after_filter:,}"})
    
    print(f"✓ Filtered to {count_after_filter:,} valid rows")
    
    # Hash-based deduplication using groupby
    print("\n📊 Step 2/3: Hash-based deduplication (grouping by key)...")
    print("   → This may take a while for large datasets (no progress bar available for groupby)")
    
    # Use groupby with map_groups to keep first row per group
    # This is faster than global sort for large datasets
    group_start = time.time()
    
    def take_first(group):
        """Take the first row from each group (dedup)."""
        import pandas as pd
        if isinstance(group, pd.DataFrame):
            return group.head(1)
        else:
            # If it's a dict/batch format
            return {k: [v[0]] if isinstance(v, list) and len(v) > 0 else v for k, v in group.items()}
    
    ds_deduped = ds_filtered.groupby('_dedup_key').map_groups(take_first, batch_format="pandas")
    group_time = time.time() - group_start
    print(f"✓ Groupby completed in {group_time:.1f}s ({group_time/60:.1f} min)")
    
    # Clean up temp columns
    print("\n📊 Step 3/3: Cleaning up and materializing...")
    with tqdm(desc="Finalizing dataset", unit=" rows", unit_scale=True) as pbar:
        def drop_temp_cols(row):
            """Remove temp columns. Returns NEW dict (thread-safe)."""
            return {k: v for k, v in row.items() if k not in ('_dedup_key', '_valid')}
        
        ds_deduped = ds_deduped.map(drop_temp_cols)
        
        # Materialize to break lineage
        ds_deduped = ds_deduped.materialize()
        final_count = ds_deduped.count()
        pbar.update(final_count)
        pbar.set_postfix({"unique_rows": f"{final_count:,}"})
    total_time = time.time() - start_time
    
    print_progress("✓", f"Hash-based deduplication complete: {final_count:,} unique rows")
    print(f"   ⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"   ⚡ Speed: {final_count/total_time:,.0f} rows/sec")
    print("="*80 + "\n")
    
    stats = {
        'rows_after_deduplication': final_count,
        'rows_before_deduplication': 'not_computed_in_hash_mode',
        'duplicate_sequences': 'not_computed_in_hash_mode'
    }
    
    return ds_deduped, stats


def deduplicate_ray_data_fast_with_progress(ds: ray.data.Dataset, mode: str, tmp_dir: str = "/mnt/ephemeral") -> tuple:
    """
    Fast Ray Data deduplication with external Unix sort (borrowed from analyze_duplication_parallel.py).
    
    Optimizations:
    - Extract to temp file (streaming)
    - External Unix sort with 50GB buffer (much faster than Ray sort for huge datasets)
    - Streaming consecutive duplicate removal (O(1) memory)
    - No materialization until final output
    """
    print(f"\n🚀 FAST DEDUPLICATION WITH EXTERNAL SORT (Mode: {mode})")
    print("="*80)
    print("⚡ Using Unix sort with 50GB buffer for maximum speed")
    stats = {}
    import time
    import subprocess
    import tempfile
    from pathlib import Path

    # Define validation and dedup key function based on mode
    def add_dedup_key(row):
        """Add validation and dedup key to each row. Returns NEW dict (thread-safe)."""
        if mode == "tra":
            tra = row.get('tra', '')
            valid = bool(tra) and tra != 'NA' and _valid_aa.issuperset(tra)
            return {**row, '_dedup_key': tra if valid else '', '_valid': valid}

        elif mode == "trb":
            trb = row.get('trb', '')
            valid = bool(trb) and trb != 'NA' and _valid_aa.issuperset(trb)
            return {**row, '_dedup_key': trb if valid else '', '_valid': valid}

        elif mode == "tcr_pairing":
            tra = row.get('tra', '')
            trb = row.get('trb', '')
            tra_valid = bool(tra) and tra != 'NA' and _valid_aa.issuperset(tra)
            trb_valid = bool(trb) and trb != 'NA' and _valid_aa.issuperset(trb)
            valid = tra_valid and trb_valid
            return {**row, '_dedup_key': f"{tra}|{trb}" if valid else '', '_valid': valid}

        elif mode == "mhc_binding":
            pep = row.get('peptide', '')
            mho = row.get('mhc_one', '')
            mht = row.get('mhc_two', '')
            pep_valid = bool(pep) and pep != 'NA' and _valid_aa.issuperset(pep)
            mho_valid = bool(mho) and mho != 'NA' and _valid_aa.issuperset(mho)
            mht_valid = bool(mht) and mht != 'NA' and _valid_aa.issuperset(mht)
            valid = pep_valid and mho_valid
            return {**row, '_dedup_key': f"{pep}|{mho}|{mht if mht_valid else ''}" if valid else '', '_valid': valid}

        elif mode == "specificity":
            tra = row.get('tra', '')
            trb = row.get('trb', '')
            pep = row.get('peptide', '')
            mho = row.get('mhc_one', '')
            mht = row.get('mhc_two', '')

            tra_valid = bool(tra) and tra != 'NA' and _valid_aa.issuperset(tra)
            trb_valid = bool(trb) and trb != 'NA' and _valid_aa.issuperset(trb)
            pep_valid = bool(pep) and pep != 'NA' and _valid_aa.issuperset(pep)
            mho_valid = bool(mho) and mho != 'NA' and _valid_aa.issuperset(mho)
            mht_valid = bool(mht) and mht != 'NA' and _valid_aa.issuperset(mht)

            valid = pep_valid and mho_valid and (tra_valid or trb_valid)

            if valid:
                parts = []
                if mho_valid: parts.append(f"mhc_one:{mho}")
                if mht_valid: parts.append(f"mhc_two:{mht}")
                if pep_valid: parts.append(f"peptide:{pep}")
                if tra_valid: parts.append(f"tra:{tra}")
                if trb_valid: parts.append(f"trb:{trb}")
                parts.sort()
                dedup_key = '|'.join(parts)
            else:
                dedup_key = ''

            return {**row, '_dedup_key': dedup_key, '_valid': valid}

        else:  # default or balanced mode
            tra = row.get('tra', '')
            trb = row.get('trb', '')
            pep = row.get('peptide', '')
            mho = row.get('mhc_one', '')
            mht = row.get('mhc_two', '')

            tra_valid = bool(tra) and tra != 'NA' and _valid_aa.issuperset(tra)
            trb_valid = bool(trb) and trb != 'NA' and _valid_aa.issuperset(trb)
            pep_valid = bool(pep) and pep != 'NA' and _valid_aa.issuperset(pep)
            mho_valid = bool(mho) and mho != 'NA' and _valid_aa.issuperset(mho)
            mht_valid = bool(mht) and mht != 'NA' and _valid_aa.issuperset(mht)

            parts = []
            if mho_valid: parts.append(f"mhc_one:{mho}")
            if mht_valid: parts.append(f"mhc_two:{mht}")
            if pep_valid: parts.append(f"peptide:{pep}")
            if tra_valid: parts.append(f"tra:{tra}")
            if trb_valid: parts.append(f"trb:{trb}")

            valid = len(parts) > 0
            if valid:
                parts.sort()
                dedup_key = '|'.join(parts)
            else:
                dedup_key = ''

            return {**row, '_dedup_key': dedup_key, '_valid': valid}

    # Setup temp directory for Unix sort (much faster than Ray sort)
    temp_base = Path(tmp_dir) / "dedup_temp"
    temp_base.mkdir(exist_ok=True, parents=True)
    
    # Set Unix sort to use this temp dir
    os.environ['TMPDIR'] = str(temp_base)
    
    start_time = time.time()

    print("\n📊 Step 1/5: Streaming extraction to temp file...")
    print("   ℹ️  Extracting dedup keys and valid rows without loading into memory")
    
    # Extract to temp file (streaming, O(1) memory)
    extract_file = temp_base / "extract.txt"
    valid_count = 0
    
    with open(extract_file, 'w') as out:
        with tqdm(desc="Extracting sequences", unit=" rows", unit_scale=True) as pbar:
            # Process in batches to avoid memory issues
            for batch in ds.iter_batches(batch_size=100000, batch_format="pandas"):
                for _, row in batch.iterrows():
                    row_dict = row.to_dict()
                    row_with_key = add_dedup_key(row_dict)
                    
                    if row_with_key['_valid']:
                        dedup_key = row_with_key['_dedup_key']
                        # Write all columns we need (excluding temp columns)
                        cols_to_save = {k: v for k, v in row_dict.items() 
                                       if k not in ('_dedup_key', '_valid')}
                        # Format: dedup_key\ttab-separated values
                        out.write(f"{dedup_key}\t{json.dumps(cols_to_save)}\n")
                        valid_count += 1
                        pbar.update(1)
    
    print(f"✓ Extracted {valid_count:,} valid sequences ({extract_file.stat().st_size / (1024**3):.2f} GB)")

    print("\n📊 Step 2/5: External Unix sort (50GB buffer)...")
    print("   ℹ️  Using disk-based merge sort - much faster than in-memory for huge datasets")
    
    sorted_file = temp_base / "sorted.txt"
    sort_start = time.time()
    
    # Use Unix sort with large buffer (borrowed from analyze_duplication_parallel.py)
    cmd = [
        'sort',
        '-S', '50G',  # 50GB buffer
        '--parallel=8',  # 8 parallel threads
        '--temporary-directory', str(temp_base),
        '-o', str(sorted_file),
        str(extract_file)
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)
    sort_time = time.time() - sort_start
    
    print(f"✓ Sorted in {sort_time:.1f}s ({sort_time/60:.1f} min)")
    
    # Clean up extract file
    extract_file.unlink()

    print("\n📊 Step 3/5: Streaming consecutive duplicate removal...")
    print("   ℹ️  O(1) memory - only keeping first occurrence of each key")
    
    dedup_file = temp_base / "deduped.txt"
    unique_count = 0
    prev_key = None
    
    dedup_start = time.time()
    with open(sorted_file, 'r') as infile, open(dedup_file, 'w') as outfile:
        with tqdm(desc="Deduplicating", unit=" rows", unit_scale=True) as pbar:
            for line in infile:
                if not line.strip():
                    continue
                
                parts = line.strip().split('\t', 1)
                if len(parts) != 2:
                    continue
                
                key, data = parts
                
                # Only keep first occurrence (consecutive duplicates are removed)
                if key != prev_key:
                    outfile.write(f"{data}\n")
                    unique_count += 1
                    prev_key = key
                    
                    if unique_count % 100000 == 0:
                        pbar.update(100000)
            
            pbar.update(unique_count % 100000)
    
    dedup_time = time.time() - dedup_start
    print(f"✓ Found {unique_count:,} unique sequences in {dedup_time:.1f}s")
    
    # Clean up sorted file
    sorted_file.unlink()

    print("\n📊 Step 4/5: Converting back to Ray Dataset...")
    print("   ℹ️  Loading deduplicated data into Ray format")
    
    # Read deduplicated data back
    deduplicated_data = []
    with open(dedup_file, 'r') as f:
        with tqdm(desc="Loading into Ray", unit=" rows", unit_scale=True) as pbar:
            for line in f:
                if line.strip():
                    row_dict = json.loads(line.strip())
                    deduplicated_data.append(row_dict)
                    
                    if len(deduplicated_data) % 100000 == 0:
                        pbar.update(100000)
            
            pbar.update(len(deduplicated_data) % 100000)
    
    # Clean up dedup file
    dedup_file.unlink()
    
    print(f"✓ Loaded {len(deduplicated_data):,} rows")

    print("\n📊 Step 5/5: Creating final Ray Dataset...")
    ds_deduped = ray.data.from_items(deduplicated_data)
    final_count = unique_count

    total_time = time.time() - start_time
    
    # Cleanup temp directory
    try:
        temp_base.rmdir()
    except:
        pass  # May not be empty if errors occurred

    print("\n" + "="*80)
    print("✅ DEDUPLICATION COMPLETE")
    print("="*80)
    print(f"📥 Input:  {filtered_count:,} valid sequences")
    print(f"📤 Output: {final_count:,} unique sequences")
    print(f"🗑️  Removed: {filtered_count - final_count:,} duplicates ({(1 - final_count/filtered_count)*100:.1f}%)")
    print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"⚡ Speed: {final_count/total_time:,.0f} rows/sec")
    print("="*80 + "\n")

    stats['rows_after_deduplication'] = final_count
    stats['rows_before_deduplication'] = filtered_count
    stats['duplicate_sequences'] = filtered_count - final_count

    return ds_deduped, stats


def cli_optimized():
    """CLI for deduplication and permutation generation only."""
    p = argparse.ArgumentParser(
        description="Deduplicate and generate sequence permutations (no tokenization)"
    )
    
    # Input/Output
    p.add_argument("--path", nargs="+", required=True,
                   help="Path to original Parquet files. Use glob patterns like '/path/*.parquet'")
    p.add_argument("--output-deduplicated", required=True,
                   help="Path to save deduplicated data (no splits, just clean data)")
    
    # Mode configuration
    p.add_argument("--mode", required=True,
                   choices=["tra", "trb", "tcr_pairing", "mhc_binding", "specificity", "default", "balanced"],
                   help="Analysis mode - determines which molecule combinations to output. "
                        "tra/trb: single molecule. tcr_pairing: TRA+TRB pairs. "
                        "mhc_binding: MHC+peptide. specificity: complete TCR complexes. "
                        "default/balanced: all molecule combinations")
    
    # Processing options
    p.add_argument("--use-hash-dedup", action="store_true", 
                   help="Use hash-based deduplication (3-10x faster for large datasets)")
    p.add_argument("--fast-mode", action="store_true",
                   help="Enable fast processing mode (skip detailed statistics)")
    p.add_argument("--sample", type=int, default=None,
                   help="Sample N files for testing (useful for validation runs)")
    
    # Performance tuning
    p.add_argument("--num-proc", type=int, default=None,
                   help="Number of processes for data processing (default: auto-detect)")
    p.add_argument("--batch-size", type=int, default=100000,
                   help="Batch size for dataset processing operations")
    p.add_argument("--tmp-dir", type=str, default="tmp_dedup",
                   help="Directory for temporary/intermediate files during processing. "
                        "Needs ~500GB-1TB free space for full dataset processing. "
                        "Consider using /dev/shm if available.")
    
    # Advanced options
    p.add_argument("--no-permutations", action="store_true",
                   help="Skip permutation generation (just deduplicate and output raw rows)")
    p.add_argument("--max-permutations", type=int, default=None,
                   help="Limit number of permutations per unique sequence set (for testing)")
    
    return p.parse_args()


def main_optimized():
    """
    Simplified pipeline: Deduplicate and generate permutations only.
    
    Workflow:
    1. Load raw parquet files
    2. Deduplicate based on mode
    3. Generate sequence permutations (if requested)
    4. Filter by mode
    5. Save clean deduplicated data
    
    NO tokenization, NO masking, NO train/val/test splits
    """
    args = cli_optimized()
    
    # Initialize Ray with disk spilling
    spill_dir = os.path.join(args.tmp_dir, "ray_spill")
    
    # Clean up old spill files if directory exists
    if os.path.exists(spill_dir):
        print(f"🧹 Cleaning up old temp files in {spill_dir}...")
        import shutil
        try:
            shutil.rmtree(spill_dir)
            print("✓ Old temp files removed")
        except Exception as e:
            print(f"⚠️  Warning: Could not clean temp dir: {e}")
    
    os.makedirs(spill_dir, exist_ok=True)
    os.makedirs(args.output_deduplicated, exist_ok=True)
    
    # Check available space
    stat = os.statvfs(args.tmp_dir)
    available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    print(f"💾 Available space in temp dir: {available_gb:.1f} GB")
    if available_gb < 100:
        print(f"⚠️  WARNING: Low disk space in {args.tmp_dir}")
        print(f"   Consider using a different --tmp-dir with more space")
    
    # Configure Ray logging
    os.environ["RAY_DEDUP_LOGS"] = "1"
    os.environ["RAY_object_spilling_config"] = json.dumps({
        "type": "filesystem",
        "params": {"directory_path": spill_dir}
    })
    os.environ["RAY_DATA_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["RAY_DATA_TRACE_SCHEDULING"] = "0"
    os.environ["RAY_LOG_TO_STDERR"] = "0"
    
    import logging
    import warnings
    warnings.filterwarnings("ignore")
    logging.getLogger("ray").setLevel(logging.CRITICAL)
    logging.getLogger("ray.data").setLevel(logging.CRITICAL)
    
    # Calculate safe object store memory (limited by /dev/shm size)
    available_ram = psutil.virtual_memory().available
    try:
        shm_stats = os.statvfs('/dev/shm')
        shm_size = shm_stats.f_bavail * shm_stats.f_frsize
        # Use 80% of /dev/shm or 50% of available RAM, whichever is smaller
        object_store_size = min(int(shm_size * 0.8), int(available_ram * 0.5))
    except:
        # Fallback if /dev/shm check fails
        object_store_size = int(available_ram * 0.3)
    
    ray.init(
        _temp_dir=args.tmp_dir,
        object_store_memory=object_store_size,
        logging_level=logging.CRITICAL,
        log_to_driver=False,
        configure_logging=True,
        include_dashboard=False
    )
    
    ray.data.DataContext.get_current().execution_options.verbose_progress = False
    
    num_proc = args.num_proc or get_num_workers()
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    
    print("="*80)
    print("🧬 TCR DEDUPLICATION AND PERMUTATION PIPELINE")
    print("="*80)
    print(f"📊 Mode: {args.mode}")
    print(f"💾 Cores: {num_proc}")
    print(f"💾 Available RAM: {available_gb:.1f} GB")
    print(f"📦 Batch Size: {args.batch_size:,}")
    print(f"⚡ Hash-based dedup: {args.use_hash_dedup or 'auto'}")
    print("="*80)
    
    # Resolve file paths
    import glob
    import time
    start_time = time.time()
    
    print(f"\n🔍 Resolving file paths...")
    file_paths = args.path if isinstance(args.path, list) else [args.path]
    all_files = []
    for pattern in file_paths:
        matched = glob.glob(pattern, recursive=True)
        for match in matched:
            if os.path.isdir(match):
                part_files = glob.glob(os.path.join(match, "*.parquet"))
                if part_files:
                    all_files.extend(part_files)
                else:
                    all_files.append(match)
            else:
                all_files.append(match)
    
    print(f"✓ Found {len(all_files):,} parquet files")
    
    # Apply sampling
    if args.sample is not None and args.sample < len(all_files):
        print(f"🎲 Sampling {args.sample} files for testing...")
        all_files = all_files[:args.sample]
    
    if len(all_files) == 0:
        raise ValueError(f"No files found matching pattern: {file_paths}")
    
    # Load data
    print(f"\n📂 Loading {len(all_files):,} parquet files (streaming mode)...")
    ds = ray.data.read_parquet(all_files)
    print("✓ Dataset loaded")
    
    # Deduplication
    print("\n" + "="*80)
    print("🔄 GLOBAL DEDUPLICATION")
    print("="*80)
    
    # Note: Hash-based dedup requires expensive shuffle - only use for specific cases
    # For most cases, sort-based dedup is faster
    use_hash_dedup = args.use_hash_dedup
    
    if use_hash_dedup:
        print("⚠️  WARNING: Hash-based dedup requires full data shuffle (slow for large datasets)")
        print("   Consider using sort-based dedup (--fast-mode without --use-hash-dedup)")
        ds, dedup_stats = deduplicate_ray_data_hash_based(ds, args.mode)
    elif args.fast_mode:
        ds, dedup_stats = deduplicate_ray_data_fast_with_progress(ds, args.mode)
    else:
        # Fall back to original deduplication
        ds, dedup_stats = deduplicate_ray_data(ds, args.mode)
    
    dedup_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("📊 DEDUPLICATION SUMMARY")
    print("="*80)
    print(f"⏱️  Time: {dedup_time/60:.1f} minutes")
    print(f"📤 Unique rows: {dedup_stats['rows_after_deduplication']:,}")
    print("="*80 + "\n")
    
    # Generate permutations based on mode (optimized for simple modes)
    if not args.no_permutations:
        print("="*80)
        print("💥 PREPARING OUTPUT DATA")
        print("="*80)
        print(f"Mode: {args.mode}")
        
        # Simple modes (tra, trb) don't need permutations - just select the column
        if args.mode in ['tra', 'trb']:
            print(f"\n⚡ Simple mode detected - selecting {args.mode} column only")
            
            def select_molecule(row):
                """Select just the relevant molecule for simple modes."""
                return {args.mode: row.get(args.mode)}
            
            ds = ds.map(select_molecule)
            final_count = dedup_stats['rows_after_deduplication']
            print(f"✓ Output ready: {final_count:,} {args.mode} sequences")
        
        else:
            # Complex modes need permutation generation
            print(f"\n📝 Generating molecule combinations for {args.mode} mode...")
            
            def explode_and_filter(row):
                """Generate permutations and filter by mode."""
                result = explode_example(row, args.mode)
                if result['sequences']:
                    permutations = []
                    for seq_tuple, feat_names in zip(result['sequences'], result['feat_names']):
                        # Create a clean row with ONLY the molecules (no metadata)
                        perm_row = {}
                        for feat_name, seq in zip(feat_names, seq_tuple):
                            perm_row[feat_name] = seq
                        permutations.append(perm_row)
                    
                    # Limit permutations if requested
                    if args.max_permutations and len(permutations) > args.max_permutations:
                        import random
                        permutations = random.sample(permutations, args.max_permutations)
                    
                    return permutations
                return []
            
            ds = ds.flat_map(explode_and_filter)
            
            print("   ⏳ Materializing permutations...")
            ds = ds.materialize()
            perm_count = ds.count()
            print(f"✓ Generated {perm_count:,} sequence permutations")
            
            # Deduplicate exact permutations by creating a dedup key from all columns
            print("\n🔄 Deduplicating exact sequence combinations...")
            
            def add_perm_key(row):
                """Create dedup key from all molecule columns."""
                # Sort keys for consistent ordering
                sorted_items = sorted(row.items())
                key = '|'.join(f"{k}:{v}" for k, v in sorted_items)
                return {**row, '_perm_key': key}
            
            ds = ds.map(add_perm_key)
            ds = ds.sort(key='_perm_key')
            
            def drop_consecutive_perm_duplicates(batch):
                """Keep only first occurrence of each unique permutation."""
                import pandas as pd
                df = pd.DataFrame(batch)
                if len(df) == 0:
                    return df
                df['_keep'] = df['_perm_key'].ne(df['_perm_key'].shift())
                df_filtered = df[df['_keep']].drop(columns=['_perm_key', '_keep'])
                return df_filtered
            
            ds = ds.map_batches(drop_consecutive_perm_duplicates, batch_format="pandas")
            ds = ds.materialize()
            
            final_count = ds.count()
            print(f"✓ {final_count:,} unique sequence combinations after dedup")
        
        print("="*80 + "\n")
    else:
        final_count = dedup_stats['rows_after_deduplication']
        print("\n⚠️  Skipping permutation generation (--no-permutations)")
    
    # Save output
    print("="*80)
    print("💾 SAVING DEDUPLICATED DATA")
    print("="*80)
    print(f"Output directory: {args.output_deduplicated}")
    print(f"Writing {final_count:,} rows...")
    print(f"   Note: May create many small files - use consolidate_parquet.py to merge later")
    
    # Write as-is without repartitioning (repartition is slow)
    # For now, accept many small files - can consolidate in post-processing
    ds.write_parquet(args.output_deduplicated, try_create_dir=True)
    
    total_time = time.time() - start_time
    
    # Final summary
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE")
    print("="*80)
    print(f"📊 Mode: {args.mode}")
    print(f"📥 Input files: {len(all_files):,}")
    print(f"📤 Output rows: {final_count:,}")
    print(f"📁 Output: {args.output_deduplicated}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes ({total_time:.0f}s)")
    print(f"⚡ Speed: {final_count/total_time:,.0f} rows/sec")
    print("="*80)
    print("\nNext steps:")
    print("  1. Use a separate tokenization script for model-specific formatting")
    print("  2. Apply masking during training or in a preprocessing step")
    print("  3. Split into train/val/test as needed")
    print("="*80 + "\n")
    
    ray.shutdown()


if __name__ == "__main__":
    main_optimized()
