#!/usr/bin/env python3
"""
Streaming deduplication with external Unix sort (no Ray dependency).
Three-stage pipeline: molecule dedup → permutation generation → permutation dedup
"""

import argparse
import heapq
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm
import glob
from itertools import permutations as iter_permutations
from multiprocessing import Pool, cpu_count
from functools import partial

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import TCR stitcher
try:
    from quest.parsers.tcr_stitcher import TCRStitcher
    STITCHER_AVAILABLE = True
except ImportError:
    STITCHER_AVAILABLE = False
    print("⚠️  WARNING: TCRStitcher not available. Full-length sequences will not be generated.")

# Valid amino acids
_valid_aa = set("ACDEFGHIKLMNPQRSTVWY")

# Mode configurations
# Standardized schema uses mhc_one/mhc_two for allele IDs (e.g., HLA-A*02:01),
# not full protein sequences.  Both the legacy mhc_one_id columns and the new
# mhc_one/mhc_two columns are accepted so the pipeline works with either
# pre-standardized or standardized input.
MODE_CONFIGS = {
    "tra": {"required": ["tra"], "optional": []},
    "trb": {"required": ["trb"], "optional": []},
    "tcr_pairing": {"required": ["tra", "trb"], "optional": []},
    "mhc_binding": {"required": ["peptide"], "optional": ["mhc_one", "mhc_two"]},
    "specificity": {"required": ["tra", "peptide"], "optional": ["mhc_one", "mhc_two"]},
    "default": {"required": [], "optional": ["tra", "trb", "peptide", "mhc_one", "mhc_two"]},
    "balanced": {"required": [], "optional": ["tra", "trb", "peptide", "mhc_one", "mhc_two"]},
    "mlm": {"required": [], "optional": ["tra", "trb", "peptide", "mhc_one", "mhc_two"]},
}

def is_valid_sequence(seq: Optional[str]) -> bool:
    """Check if sequence is valid (non-null, non-empty, valid amino acids)."""
    if seq is None or seq == "" or seq == "nan":
        return False
    # Handle pandas NaN
    if isinstance(seq, float):
        return False
    return all(aa in _valid_aa for aa in str(seq).upper())

def create_dedup_key(row: Dict[str, Any], mode: str) -> str:
    """
    Create a deduplication key for a row.
    This represents the EXACT row content (field order matters for dedup).
    """
    config = MODE_CONFIGS[mode]
    
    # Check required fields first
    for field in config['required']:
        val = row.get(field, "")
        if not is_valid_sequence(val):
            return ""  # Invalid row
    
    # Create key from ALL columns in canonical order (tra, trb, peptide, mhc_one, mhc_two)
    # This ensures exact row matching - different field combinations are different rows
    parts = []
    for field in ['tra', 'trb', 'peptide', 'mhc_one', 'mhc_two']:
        val = row.get(field, "")
        if is_valid_sequence(val):
            parts.append(f"{field}:{val}")
        else:
            parts.append(f"{field}:")  # Empty field marker
    
    return "|".join(parts)

def process_single_parquet(args: tuple) -> tuple:
    """
    Process a single parquet file and return (lines, count).
    This runs in a separate process for parallelization.
    
    If --stitch-tcr flag is enabled, will attempt to generate full-length
    TCR sequences from CDR3 + gene segments using stitchr.
    """
    pf, mode, stitch_tcr = args
    lines = []
    valid_count = 0
    stitch_success_count = 0
    stitch_attempt_count = 0
    
    # Initialize stitcher if needed (once per process)
    stitcher = None
    gene_norm_failures = {'trav': 0, 'traj': 0, 'trbv': 0, 'trbj': 0}

    if stitch_tcr and STITCHER_AVAILABLE:
        try:
            stitcher = TCRStitcher(species="HUMAN")
            # Suppress tidytcells warnings for cleaner output
            import logging
            logging.getLogger('tidytcells').setLevel(logging.ERROR)
        except Exception as e:
            print(f"Warning: Failed to initialize TCRStitcher in process: {e}")
            stitcher = None
    
    try:
        table = pq.read_table(pf)
        df = table.to_pandas()
        
        for _, row in df.iterrows():
            row_dict = row.to_dict()

            # Filter negative binding for MLM mode
            if mode == "mlm":
                binding_val = str(row_dict.get("binding", "")).strip().lower()
                if binding_val == "neg":
                    continue

            dedup_key = create_dedup_key(row_dict, mode)
            
            if dedup_key:  # Valid row
                # Keep molecule columns (chains, peptide, MHC sequences and IDs)
                # Also keep full-length stitched sequences and gene annotations
                molecule_data = {
                    k: v for k, v in row_dict.items()
                    if k in ['tra', 'trb', 'peptide', 'mhc_one', 'mhc_two', 'mhc_one_id', 'mhc_two_id',
                            'tra_full', 'trb_full',  # Full-length stitched TCR sequences
                            'trav_gene', 'traj_gene', 'trad_gene',  # TRA gene segments
                            'trbv_gene', 'trbj_gene', 'trbd_gene',  # TRB gene segments
                            'binding', 'score',  # Binding/activity and confidence score
                            'source', 'study_id']  # Standardized schema provenance
                }
                
                # Normalize gene names using tidytcells (Priority 1) and stitch sequences
                if stitcher is not None:
                    try:
                        # Process TRA: normalize genes, then stitch if needed
                        if 'tra' in molecule_data and molecule_data.get('tra'):
                            # Normalize TRA genes using tidytcells (via TCRStitcher)
                            trav = molecule_data.get('trav_gene')
                            traj = molecule_data.get('traj_gene')

                            if trav:
                                norm_trav = stitcher.normalize_gene_name(trav, 'TRA')
                                if norm_trav:
                                    molecule_data['trav_gene'] = norm_trav
                                else:
                                    gene_norm_failures['trav'] += 1

                            if traj:
                                norm_traj = stitcher.normalize_gene_name(traj, 'TRA')
                                if norm_traj:
                                    molecule_data['traj_gene'] = norm_traj
                                else:
                                    gene_norm_failures['traj'] += 1

                            # Stitch TRA if we have CDR3 + genes but no full-length sequence
                            if ('tra_full' not in molecule_data or not molecule_data.get('tra_full')):
                                stitch_attempt_count += 1
                                tra_full = stitcher.stitch_tcr(
                                    cdr3=molecule_data.get('tra'),
                                    v_gene=molecule_data.get('trav_gene'),
                                    j_gene=molecule_data.get('traj_gene'),
                                    chain='TRA'
                                )
                                if tra_full:
                                    molecule_data['tra_full'] = tra_full
                                    stitch_success_count += 1

                        # Process TRB: normalize genes, then stitch if needed
                        if 'trb' in molecule_data and molecule_data.get('trb'):
                            # Normalize TRB genes using tidytcells (via TCRStitcher)
                            trbv = molecule_data.get('trbv_gene')
                            trbj = molecule_data.get('trbj_gene')

                            if trbv:
                                norm_trbv = stitcher.normalize_gene_name(trbv, 'TRB')
                                if norm_trbv:
                                    molecule_data['trbv_gene'] = norm_trbv
                                else:
                                    gene_norm_failures['trbv'] += 1

                            if trbj:
                                norm_trbj = stitcher.normalize_gene_name(trbj, 'TRB')
                                if norm_trbj:
                                    molecule_data['trbj_gene'] = norm_trbj
                                else:
                                    gene_norm_failures['trbj'] += 1

                            # Stitch TRB if we have CDR3 + genes but no full-length sequence
                            if ('trb_full' not in molecule_data or not molecule_data.get('trb_full')):
                                stitch_attempt_count += 1
                                trb_full = stitcher.stitch_tcr(
                                    cdr3=molecule_data.get('trb'),
                                    v_gene=molecule_data.get('trbv_gene'),
                                    j_gene=molecule_data.get('trbj_gene'),
                                    chain='TRB'
                                )
                                if trb_full:
                                    molecule_data['trb_full'] = trb_full
                                    stitch_success_count += 1
                    except Exception as e:
                        # Don't fail the whole file for stitching errors
                        pass
                
                lines.append(f"{dedup_key}\t{json.dumps(molecule_data)}\n")
                valid_count += 1
    except Exception as e:
        print(f"Warning: Failed to read {pf}: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
    
    # Print stitching stats for this file (only if we tried stitching)
    if stitch_tcr and stitch_attempt_count > 0:
        success_rate = (stitch_success_count / stitch_attempt_count * 100) if stitch_attempt_count > 0 else 0
        print(f"Stitching stats for {pf}: {stitch_success_count}/{stitch_attempt_count} ({success_rate:.1f}%)")

        # Print gene normalization summary
        total_failures = sum(gene_norm_failures.values())
        if total_failures > 0:
            print(f"  Gene normalization failures: {total_failures} " +
                  f"(TRAV: {gene_norm_failures['trav']}, TRAJ: {gene_norm_failures['traj']}, " +
                  f"TRBV: {gene_norm_failures['trbv']}, TRBJ: {gene_norm_failures['trbj']})")

    return lines, valid_count, gene_norm_failures

def extract_parquet_to_temp(parquet_files: List[str], temp_file: Path, mode: str, num_workers: int = None, stitch_tcr: bool = False) -> int:
    """
    Extract parquet files to temp file with dedup keys (parallelized).
    Returns number of valid rows extracted.
    """
    if num_workers is None:
        num_workers = cpu_count()  # Use all cores for maximum throughput
    
    valid_count = 0
    
    print(f"   ℹ️  Using {num_workers} parallel workers")
    if stitch_tcr:
        print(f"   ℹ️  TCR stitching ENABLED - will generate full-length sequences from CDR3 + gene segments")
    
    with open(temp_file, 'w', buffering=8*1024*1024) as out:  # 8MB write buffer
        total_gene_failures = {'trav': 0, 'traj': 0, 'trbv': 0, 'trbj': 0}

        with Pool(num_workers) as pool:
            # Process files in parallel
            args_list = [(pf, mode, stitch_tcr) for pf in parquet_files]

            with tqdm(desc="Extracting parquet files", unit=" files", total=len(parquet_files)) as pbar:
                for lines, count, gene_failures in pool.imap_unordered(process_single_parquet, args_list, chunksize=1):
                    # Write all lines from this file
                    out.writelines(lines)
                    valid_count += count
                    # Aggregate gene normalization failures
                    for gene_type in total_gene_failures:
                        total_gene_failures[gene_type] += gene_failures.get(gene_type, 0)
                    pbar.update(1)

        # Print final gene normalization summary
        if stitch_tcr and sum(total_gene_failures.values()) > 0:
            print(f"\n📊 Gene Normalization Summary:")
            print(f"   Total failures: {sum(total_gene_failures.values())}")
            for gene_type, count in total_gene_failures.items():
                if count > 0:
                    print(f"     {gene_type.upper()}: {count:,} malformed gene names")

    return valid_count

def extract_and_create_sorted_chunks(parquet_files: List[str], temp_dir: Path, mode: str,
                                     chunk_size: int, num_workers: int = None, stitch_tcr: bool = False) -> tuple[List[Path], int]:
    """
    Stream-extract parquet rows and directly build sorted chunk files without creating
    a giant intermediate extract file. Returns (chunk_files, valid_count).
    """
    if num_workers is None:
        num_workers = cpu_count()  # Use all cores for maximum throughput

    print(f"   ℹ️  Using {num_workers} parallel workers (streaming extract)")
    if stitch_tcr:
        print(f"   ℹ️  TCR stitching ENABLED - will generate full-length sequences from CDR3 + gene segments")

    chunk_files: List[Path] = []
    current_chunk: List[str] = []
    chunk_idx = 0
    valid_count = 0

    total_gene_failures = {'trav': 0, 'traj': 0, 'trbv': 0, 'trbj': 0}
    args_list = [(pf, mode, stitch_tcr) for pf in parquet_files]
    with Pool(num_workers) as pool:
        with tqdm(desc="Extracting + chunking", unit=" files", total=len(parquet_files)) as pbar:
            for lines, count, gene_failures in pool.imap_unordered(process_single_parquet, args_list, chunksize=1):
                valid_count += count
                # Aggregate gene normalization failures
                for gene_type in total_gene_failures:
                    total_gene_failures[gene_type] += gene_failures.get(gene_type, 0)
                # Append lines and spill when needed
                if lines:
                    current_chunk.extend(lines)
                    while len(current_chunk) >= chunk_size:
                        # Sort and spill a full chunk
                        to_write = current_chunk[:chunk_size]
                        del current_chunk[:chunk_size]
                        to_write.sort()
                        out_path = temp_dir / f"chunk_{chunk_idx:06d}.txt"
                        with open(out_path, 'w', buffering=8*1024*1024) as cf:  # 8MB write buffer
                            cf.writelines(to_write)
                        chunk_files.append(out_path)
                        chunk_idx += 1
                pbar.update(1)

    # Flush remainder
    if current_chunk:
        current_chunk.sort()
        out_path = temp_dir / f"chunk_{chunk_idx:06d}.txt"
        with open(out_path, 'w', buffering=8*1024*1024) as cf:  # 8MB write buffer
            cf.writelines(current_chunk)
        chunk_files.append(out_path)
        chunk_idx += 1

    print(f"   ℹ️  Created {len(chunk_files)} sorted chunk files (streamed)")

    # Print final gene normalization summary
    if stitch_tcr and sum(total_gene_failures.values()) > 0:
        print(f"\n📊 Gene Normalization Summary:")
        print(f"   Total failures: {sum(total_gene_failures.values())}")
        for gene_type, count in total_gene_failures.items():
            if count > 0:
                print(f"     {gene_type.upper()}: {count:,} malformed gene names")

    return chunk_files, valid_count

def merge_sorted_files(chunk_files: List[Path], output_file: Path, temp_dir: Path, max_open_files: int = 256) -> None:
    """
    Multi-pass k-way merge of sorted chunk files into a single sorted output.
    Deletes intermediate chunk files progressively to control disk usage.
    """
    if not chunk_files:
        # Create empty output
        open(output_file, 'w').close()
        return

    if len(chunk_files) == 1:
        os.replace(chunk_files[0], output_file)
        return

    def merge_group(files: List[Path], out_path: Path):
        file_iters = []
        for p in files:
            f = open(p, 'r', buffering=8*1024*1024)  # 8MB read buffer
            file_iters.append((f, iter(f)))
        try:
            heap = []
            for idx, (fh, it) in enumerate(file_iters):
                try:
                    line = next(it)
                    heapq.heappush(heap, (line, idx))
                except StopIteration:
                    pass
            with open(out_path, 'w', buffering=8*1024*1024) as out:  # 8MB write buffer
                while heap:
                    line, idx = heapq.heappop(heap)
                    out.write(line)
                    try:
                        nxt = next(file_iters[idx][1])
                        heapq.heappush(heap, (nxt, idx))
                    except StopIteration:
                        pass
        finally:
            for fh, _ in file_iters:
                try:
                    fh.close()
                except Exception:
                    pass

    pass_num = 0
    while len(chunk_files) > 1:
        pass_num += 1
        new_chunk_files: List[Path] = []
        print(f"   ℹ️  Merge pass {pass_num}: {len(chunk_files)} files → batches of ≤{max_open_files}")
        with tqdm(total=len(chunk_files), desc=f"   Merging (pass {pass_num})", unit=" files") as pbar:
            for i in range(0, len(chunk_files), max_open_files):
                group = chunk_files[i:i+max_open_files]
                out_path = temp_dir / f"merge_p{pass_num}_{i//max_open_files:06d}.txt"
                merge_group(group, out_path)
                new_chunk_files.append(out_path)
                for p in group:
                    try:
                        p.unlink()
                    except Exception:
                        pass
                pbar.update(len(group))
        chunk_files = new_chunk_files

    os.replace(chunk_files[0], output_file)

def extract_and_sort_streaming(parquet_files: List[str], sorted_file: Path, temp_dir: Path, mode: str,
                               num_workers: int, chunk_size: int, max_open_files: int, stitch_tcr: bool = False) -> tuple[int, float]:
    """
    End-to-end streaming extract + chunked sort + multi-pass merge into sorted_file.
    Returns (valid_count, elapsed_seconds).
    """
    start = time.time()
    chunk_files, valid_count = extract_and_create_sorted_chunks(
        parquet_files, temp_dir, mode, chunk_size, num_workers, stitch_tcr
    )
    merge_sorted_files(chunk_files, sorted_file, temp_dir, max_open_files=max_open_files)
    return valid_count, time.time() - start

def pyarrow_sort(input_file: Path, output_file: Path, temp_dir: Path, chunk_size: int = 200_000_000, max_open_files: int = 1024) -> float:
    """
    External merge sort implemented in Python:
    - Read input in chunks of `chunk_size` lines
    - Sort each chunk in-memory and spill to temp files
    - K-way merge the chunk files in batches (<= max_open_files open at once)
    Returns time taken in seconds.
    """
    start_time = time.time()

    print(f"   ℹ️  Using PyArrow-style chunked sort (chunk_size={chunk_size:,} lines, max_open_files={max_open_files})")

    temp_dir.mkdir(parents=True, exist_ok=True)
    chunk_files: List[Path] = []

    # Phase 1: create sorted chunk files
    current_chunk: List[str] = []
    chunk_idx = 0
    with open(input_file, 'r', buffering=8*1024*1024) as f:  # 8MB read buffer
        for line in tqdm(f, desc="   Reading and sorting chunks", unit=" lines", unit_scale=True):
            if not line:
                continue
            current_chunk.append(line)
            if len(current_chunk) >= chunk_size:
                current_chunk.sort()
                chunk_path = temp_dir / f"chunk_{chunk_idx:06d}.txt"
                with open(chunk_path, 'w', buffering=8*1024*1024) as cf:  # 8MB write buffer
                    cf.writelines(current_chunk)
                chunk_files.append(chunk_path)
                current_chunk = []
                chunk_idx += 1

        # Flush last chunk
        if current_chunk:
            current_chunk.sort()
            chunk_path = temp_dir / f"chunk_{chunk_idx:06d}.txt"
            with open(chunk_path, 'w', buffering=8*1024*1024) as cf:  # 8MB write buffer
                cf.writelines(current_chunk)
            chunk_files.append(chunk_path)
            current_chunk = []
            chunk_idx += 1

    print(f"   ℹ️  Created {len(chunk_files)} sorted chunk files")

    # Early exit: single chunk
    if len(chunk_files) == 1:
        os.replace(chunk_files[0], output_file)
        return time.time() - start_time

    # Helper: merge a group of sorted files into one output file
    def merge_group(files: List[Path], out_path: Path):
        file_iters = []
        for p in files:
            f = open(p, 'r', buffering=8*1024*1024)  # 8MB read buffer
            file_iters.append((f, iter(f)))

        try:
            heap = []
            # Prime heap
            for idx, (fh, it) in enumerate(file_iters):
                try:
                    line = next(it)
                    heapq.heappush(heap, (line, idx))
                except StopIteration:
                    pass

            with open(out_path, 'w', buffering=8*1024*1024) as out:  # 8MB write buffer
                while heap:
                    line, idx = heapq.heappop(heap)
                    out.write(line)
                    try:
                        nxt = next(file_iters[idx][1])
                        heapq.heappush(heap, (nxt, idx))
                    except StopIteration:
                        pass
        finally:
            for fh, _ in file_iters:
                try:
                    fh.close()
                except Exception:
                    pass

    # Phase 2: multi-pass k-way merge in batches
    pass_num = 0
    while len(chunk_files) > 1:
        pass_num += 1
        new_chunk_files: List[Path] = []
        print(f"   ℹ️  Merge pass {pass_num}: {len(chunk_files)} files → batches of ≤{max_open_files}")
        with tqdm(total=len(chunk_files), desc=f"   Merging (pass {pass_num})", unit=" files") as pbar:
            for i in range(0, len(chunk_files), max_open_files):
                group = chunk_files[i:i+max_open_files]
                out_path = temp_dir / f"merge_p{pass_num}_{i//max_open_files:06d}.txt"
                merge_group(group, out_path)
                new_chunk_files.append(out_path)
                # Remove merged inputs to free disk
                for p in group:
                    try:
                        p.unlink()
                    except Exception:
                        pass
                pbar.update(len(group))
        chunk_files = new_chunk_files

    # Final file
    os.replace(chunk_files[0], output_file)
    return time.time() - start_time


def external_sort(input_file: Path, output_file: Path, temp_dir: Path, buffer_size: str = "50G", num_threads: int = None) -> float:
    """
    Use Unix sort with large buffer for external merge sort.
    Returns time taken in seconds.
    
    NOTE: This function is kept for compatibility but PyArrow sort is now preferred.
    """
    if num_threads is None:
        num_threads = cpu_count()  # Use all cores for maximum throughput
    
    start_time = time.time()
    
    cmd = [
        'sort',
        '-S', buffer_size,
        f'--parallel={num_threads}',
        '--temporary-directory', str(temp_dir),
        '-o', str(output_file),
        str(input_file)
    ]
    
    print(f"   ℹ️  Using {num_threads} sort threads")
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Sort command failed with exit code {result.returncode}")
        print(f"Command: {' '.join(cmd)}")
        if result.stdout:
            print(f"STDOUT: {result.stdout}")
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    
    return time.time() - start_time

def stream_deduplicate(input_file: Path, output_file: Path) -> int:
    """
    Stream through sorted file and remove consecutive duplicates.
    Returns number of unique rows.
    """
    unique_count = 0
    prev_key = None
    
    with open(input_file, 'r', buffering=8*1024*1024) as infile, open(output_file, 'w', buffering=8*1024*1024) as outfile:  # 8MB buffers
        with tqdm(desc="Deduplicating", unit=" rows", unit_scale=True) as pbar:
            for line in infile:
                if not line.strip():
                    continue
                
                parts = line.strip().split('\t', 1)
                if len(parts) != 2:
                    continue
                
                key, data = parts
                
                if key != prev_key:
                    outfile.write(f"{data}\n")
                    unique_count += 1
                    prev_key = key
                    
                    if unique_count % 100000 == 0:
                        pbar.update(100000)
            
            pbar.update(unique_count % 100000)
    
    return unique_count

def process_molecule_batch(args: tuple) -> List[str]:
    """
    Process a batch of molecules and generate permutations.
    This runs in a separate process for parallelization.
    """
    lines, mode, max_perms = args
    output_lines = []
    
    for line in lines:
        if not line.strip():
            continue
        
        # Parse line: could be "dedup_key\tjson" from stage 1 or just "json"
        parts = line.strip().split('\t', 1)
        if len(parts) == 2:
            json_data = parts[1]  # Strip dedup key, keep only JSON
        else:
            json_data = parts[0]
        
        row_dict = json.loads(json_data)
        
        # Get non-empty molecule fields
        molecule_values = []
        for field in ['tra', 'trb', 'peptide', 'mhc_one', 'mhc_two']:
            val = row_dict.get(field, "")
            if val and val != "nan" and val != "":
                molecule_values.append((field, val))
        
        # Generate all permutations of all subset sizes
        # For 3 molecules [A, B, C], generate:
        # - Size 1: A, B, C
        # - Size 2: AB, BA, AC, CA, BC, CB
        # - Size 3: ABC, ACB, BAC, BCA, CAB, CBA
        
        if len(molecule_values) == 0:
            # No valid fields - shouldn't happen but handle it
            output_lines.append(f"empty\t{json_data}\n")
        else:
            all_perms = []
            
            # Generate permutations for each subset size (1 to N)
            for subset_size in range(1, len(molecule_values) + 1):
                # Get all combinations of this size
                from itertools import combinations
                for subset in combinations(molecule_values, subset_size):
                    # Generate all permutations of this subset
                    for perm in iter_permutations(subset):
                        all_perms.append(perm)
            
            # Apply max_perms limit if specified
            if max_perms and len(all_perms) > max_perms:
                import random
                all_perms = random.sample(all_perms, max_perms)
            
            # Create output lines for each permutation
            for perm in all_perms:
                # Create permutation key for deduplication
                perm_key = "|".join([f"{field}:{val}" for field, val in perm])
                
                # Write with permutation key and JSON only (not nested tabs!)
                output_lines.append(f"{perm_key}\t{json_data}\n")
    
    return output_lines

def generate_permutations(input_file: Path, output_file: Path, mode: str, max_perms: Optional[int], num_workers: int = None) -> int:
    """
    Generate permutations for each unique molecule combination (parallelized).
    For tra/trb modes: skip permutations (only 1 ordering makes sense)
    For other modes: generate all permutations up to max_perms

    Returns number of permutations generated.
    """
    print(f"\n📊 Stage 2: Generating permutations (mode={mode})...")

    # For single-chain modes, skip permutation generation
    if mode in ['tra', 'trb']:
        print(f"   ℹ️  Skipping permutation generation for {mode} mode (single chain)")
        # Just copy the file
        subprocess.run(['cp', str(input_file), str(output_file)], check=True)
        with open(input_file, 'r') as f:
            count = sum(1 for _ in f)
        return count

    if num_workers is None:
        num_workers = cpu_count()  # Use all cores for maximum throughput

    print(f"   ℹ️  Using {num_workers} parallel workers")

    # Stream input file in batches - memory efficient for large files
    batch_size = 100000  # Process 100k molecules per batch
    perm_count = 0

    # Generator function to yield batches without loading entire file
    def batch_generator():
        """Yield batches of lines from input file without loading all into memory."""
        with open(input_file, 'r', buffering=8*1024*1024) as infile:
            batch = []
            for line in infile:
                batch.append(line)
                if len(batch) >= batch_size:
                    yield (batch, mode, max_perms)
                    batch = []
            # Yield final partial batch
            if batch:
                yield (batch, mode, max_perms)

    # Process batches in parallel using streaming generator
    with open(output_file, 'w', buffering=16*1024*1024) as outfile:  # 16MB write buffer for large output
        with Pool(num_workers) as pool:
            with tqdm(desc="Generating permutations", unit=" batches", unit_scale=False) as pbar:
                for output_lines in pool.imap_unordered(process_molecule_batch, batch_generator(), chunksize=1):
                    outfile.writelines(output_lines)
                    perm_count += len(output_lines)
                    pbar.update(1)

    return perm_count

def deduplicate_permutations(input_file: Path, output_file: Path, temp_dir: Path, 
                           buffer_size: str = "50G", num_threads: int = None, 
                           use_unix_sort: bool = False, sort_chunk_size: int = 50_000_000,
                           sort_max_open_files: int = 256) -> int:
    """
    Deduplicate permutations (remove exact duplicate orderings).
    Returns number of unique permutations.
    """
    print(f"\n📊 Stage 3: Deduplicating permutations...")
    
    # Sort by permutation key
    sorted_file = temp_dir / "permutations_sorted.txt"
    if use_unix_sort:
        print("   ℹ️  Sorting permutations with Unix sort...")
        sort_time = external_sort(input_file, sorted_file, temp_dir, buffer_size, num_threads)
    else:
        print("   ℹ️  Sorting permutations with PyArrow...")
        sort_time = pyarrow_sort(input_file, sorted_file, temp_dir, chunk_size=sort_chunk_size, max_open_files=sort_max_open_files)
    print(f"   ✓ Sorted in {sort_time:.1f}s")
    
    # Remove consecutive duplicates
    print("   ℹ️  Removing duplicate orderings...")
    unique_count = stream_deduplicate(sorted_file, output_file)
    
    # Clean up
    sorted_file.unlink()
    
    return unique_count

def parse_permutation_key(perm_key: str) -> tuple[List[str], Dict[str, str]]:
    """
    Parse permutation key to extract field order and sequences.

    Handles three formats:
    1. Old format (multi): "tra:AAA|peptide:CCC|mhc_one:DDD" - extracts sequences from key
    2. Old format (single): "tra:AAA" - single molecule with sequence
    3. New format: "tra_peptide_mhc_one" - no sequences in key

    Returns:
        - field_order: list of field names in order
        - sequences_dict: dict mapping field names to sequences (empty for new format)
    """
    if ':' in perm_key:
        # Old format with sequences (either single or multiple with |)
        field_order = []
        sequences_dict = {}
        for part in perm_key.split('|'):
            if ':' in part:
                field_name, sequence = part.split(':', 1)
                field_order.append(field_name)
                sequences_dict[field_name] = sequence
        return field_order, sequences_dict
    else:
        # New format - field names separated by _, but some fields contain underscores
        # Known field names (longer names first to match greedily)
        known_fields = ['mhc_one', 'mhc_two', 'peptide', 'tra', 'trb']

        field_order = []
        remaining = perm_key

        while remaining:
            matched = False
            for field in known_fields:
                if remaining.startswith(field):
                    field_order.append(field)
                    remaining = remaining[len(field):]
                    # Remove leading underscore separator if present
                    if remaining.startswith('_'):
                        remaining = remaining[1:]
                    matched = True
                    break

            if not matched:
                # Unknown format, skip to next underscore or end
                if '_' in remaining:
                    remaining = remaining.split('_', 1)[1]
                else:
                    break

        return field_order, {}


def concatenate_sequences(row_dict: Dict, field_order: List[str], sequences_dict: Dict[str, str], use_full: bool = False) -> str:
    """
    Concatenate sequences in the order specified by field_order.
    
    Args:
        row_dict: Dictionary containing sequence data (fallback source)
        field_order: List of field names in order (e.g., ['tra', 'peptide', 'mhc_one'])
        sequences_dict: Dict mapping field names to sequences (from permutation key, if available)
        use_full: If True, use tra_full/trb_full instead of tra/trb
    """
    sequences = []
    for field in field_order:
        # First try to get sequence from sequences_dict (from permutation key)
        if field in sequences_dict and sequences_dict[field]:
            sequences.append(sequences_dict[field])
            continue
        
        # Fallback: get from row_dict
        lookup_field = field
        if use_full:
            if field == 'tra':
                lookup_field = 'tra_full'
            elif field == 'trb':
                lookup_field = 'trb_full'
        
        val = row_dict.get(lookup_field, "")
        if val and str(val) != 'nan' and str(val) != '' and not (isinstance(val, float)):
            sequences.append(str(val))
    
    return " ".join(sequences)


def detect_resume_state(work_dir: Path) -> dict:
    """
    Detect what stage of processing has been completed based on intermediate files.

    Returns a dict with:
        - stage: 'none', 'chunks', 'sorted', 'deduped', 'permutations', 'final'
        - chunk_files: list of chunk files (if stage >= 'chunks')
        - sorted_file: path to sorted file (if exists)
        - deduped_file: path to deduped file (if exists)
        - perm_file: path to permutations file (if exists)
        - final_file: path to final file (if exists)
        - valid_count: number of valid rows (if detectable)
        - unique_count: number of unique molecules (if detectable)
        - perm_count: number of permutations (if detectable)
    """
    state = {
        'stage': 'none',
        'chunk_files': [],
        'sorted_file': None,
        'deduped_file': None,
        'perm_file': None,
        'final_file': None,
        'valid_count': None,
        'unique_count': None,
        'perm_count': None
    }

    sorted_file = work_dir / "sorted.txt"
    deduped_file = work_dir / "deduped.txt"
    perm_file = work_dir / "permutations.txt"
    final_file = work_dir / "final.txt"

    # Check for final output (Stage 3 complete)
    if final_file.exists():
        state['stage'] = 'final'
        state['final_file'] = final_file
        # Count lines
        with open(final_file, 'r') as f:
            state['perm_count'] = sum(1 for _ in f)
        return state

    # Check for permutations file (Stage 2 complete)
    if perm_file.exists():
        state['stage'] = 'permutations'
        state['perm_file'] = perm_file
        # Count lines
        with open(perm_file, 'r') as f:
            state['perm_count'] = sum(1 for _ in f)
        return state

    # Check for deduped file (Stage 1 complete)
    if deduped_file.exists():
        state['stage'] = 'deduped'
        state['deduped_file'] = deduped_file
        # Count lines
        with open(deduped_file, 'r') as f:
            state['unique_count'] = sum(1 for _ in f)
        return state

    # Check for sorted file (Stage 1 sort complete, needs dedup)
    if sorted_file.exists():
        state['stage'] = 'sorted'
        state['sorted_file'] = sorted_file
        return state

    # Check for chunk files (extraction complete, needs merge)
    chunk_files = sorted(work_dir.glob("chunk_*.txt"))
    if chunk_files:
        state['stage'] = 'chunks'
        state['chunk_files'] = chunk_files
        # Try to estimate valid_count from chunk files
        total_lines = 0
        for chunk in chunk_files[:5]:  # Sample first 5 chunks
            with open(chunk, 'r') as f:
                total_lines += sum(1 for _ in f)
        if len(chunk_files) <= 5:
            state['valid_count'] = total_lines
        else:
            # Estimate based on sample
            state['valid_count'] = int(total_lines * len(chunk_files) / 5)
        return state

    return state


def write_parquet_output(input_file: Path, output_dir: Path, output_dir_full: Path, batch_size: int = 1000000):
    """
    Convert deduplicated text file to two parquet outputs:
    1. CDR3 version (tra/trb) - permutation_key, concatenated_sequence (all rows)
    2. Full-length version (tra_full/trb_full) - permutation_key, concatenated_sequence
    
    FILTERING: The full-length output only includes rows where at least one full-length
    TCR sequence (tra_full or trb_full) exists. Rows with only CDR3 data are excluded.
    
    IMPORTANT: Permutation keys in the full-length output only include fields where
    the full-length version actually exists:
    - permutation_key='tra' means tra_full exists
    - permutation_key='tra_trb' means both tra_full and trb_full exist
    - permutation_key='peptide' means only peptide (no TCR full-length)
    
    This allows users to filter for rows with specific full-length molecules.
    """
    print(f"\n📊 Writing parquet outputs...")
    
    output_dir.mkdir(exist_ok=True, parents=True)
    output_dir_full.mkdir(exist_ok=True, parents=True)
    
    batch_num = 0
    batch_data_cdr3 = []
    batch_data_full = []
    
    with open(input_file, 'r') as f:
        with tqdm(desc="Writing parquet", unit=" rows", unit_scale=True) as pbar:
            for line in f:
                if not line.strip():
                    continue
                
                # For permutations, the line is: perm_key\toriginal_json
                # For molecules, the line is just: json
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    # Has permutation key (format: "tra:AAA|peptide:CCC|mhc_one:DDD")
                    old_perm_key = parts[0]
                    row_dict = json.loads(parts[1].strip())
                else:
                    # No permutation key - shouldn't happen after stage 2/3, but handle it
                    row_dict = json.loads(parts[0])
                    # Create a simple key from present fields
                    present_fields = []
                    for field in ['tra', 'trb', 'peptide', 'mhc_one', 'mhc_two']:
                        if field in row_dict and row_dict[field]:
                            present_fields.append(field)
                    old_perm_key = "_".join(present_fields) if present_fields else "empty"
                
                # Parse old permutation key to get field order and sequences
                field_order, sequences_dict = parse_permutation_key(old_perm_key)
                
                # Create CDR3 version (tra/trb) - collect actual sequences
                seq_parts_cdr3 = []
                actual_fields_cdr3 = []
                for field in field_order:
                    # Try permutation key first
                    if field in sequences_dict and sequences_dict[field]:
                        seq_parts_cdr3.append(sequences_dict[field])
                        actual_fields_cdr3.append(field)
                    else:
                        # Fallback to row_dict
                        val = row_dict.get(field, "")
                        if val and str(val) != 'nan' and str(val) != '' and not (isinstance(val, float)):
                            seq_parts_cdr3.append(str(val))
                            actual_fields_cdr3.append(field)
                
                seq_cdr3 = " ".join(seq_parts_cdr3)
                perm_key_cdr3 = "_".join(actual_fields_cdr3) if actual_fields_cdr3 else "empty"
                
                batch_data_cdr3.append({
                    'permutation_key': perm_key_cdr3,
                    'sequence': seq_cdr3
                })
                
                # Create full-length version (tra_full/trb_full)
                # IMPORTANT: Only include fields in permutation key if full-length version exists
                # This distinguishes rows with actual full-length sequences from CDR3 fallbacks
                seq_parts_full = []
                actual_fields_full = []
                for field in field_order:
                    # Determine which field to look up
                    lookup_field = field
                    if field == 'tra':
                        lookup_field = 'tra_full'
                    elif field == 'trb':
                        lookup_field = 'trb_full'
                    
                    val = row_dict.get(lookup_field, "")
                    
                    # Check if we have a valid full-length sequence
                    has_full = val and str(val) != 'nan' and str(val) != '' and not isinstance(val, float)
                    
                    if has_full:
                        # We have the full-length version - use it and include in permutation key
                        seq_parts_full.append(str(val))
                        actual_fields_full.append(field)
                    elif lookup_field == field:
                        # This field doesn't have a full version (peptide, mhc_one, mhc_two)
                        # Use the regular field value
                        val = row_dict.get(field, "")
                        if val and str(val) != 'nan' and str(val) != '' and not isinstance(val, float):
                            seq_parts_full.append(str(val))
                            actual_fields_full.append(field)
                    # else: field is tra/trb but no full-length version exists - skip entirely
                    # This ensures permutation_key only includes fields with actual full-length data
                
                seq_full = " ".join(seq_parts_full)
                perm_key_full = "_".join(actual_fields_full) if actual_fields_full else "empty"
                
                # Filter: Only include in full output if we have at least one full-length TCR sequence
                # Check if tra_full or trb_full actually exists in the row
                has_full_tcr = ('tra_full' in row_dict and row_dict.get('tra_full') and 
                               str(row_dict['tra_full']) != 'nan' and str(row_dict['tra_full']) != '' and 
                               not isinstance(row_dict['tra_full'], float)) or \
                              ('trb_full' in row_dict and row_dict.get('trb_full') and 
                               str(row_dict['trb_full']) != 'nan' and str(row_dict['trb_full']) != '' and 
                               not isinstance(row_dict['trb_full'], float))
                
                # Only add to full output if we have at least one full-length TCR sequence
                if has_full_tcr:
                    batch_data_full.append({
                        'permutation_key': perm_key_full,
                        'sequence': seq_full
                    })
                
                # Write CDR3 batch if ready
                if len(batch_data_cdr3) >= batch_size:
                    table_cdr3 = pa.Table.from_pylist(batch_data_cdr3)
                    output_file_cdr3 = output_dir / f"batch_{batch_num:06d}.parquet"
                    pq.write_table(table_cdr3, output_file_cdr3)
                    pbar.update(len(batch_data_cdr3))
                    batch_num += 1
                    batch_data_cdr3 = []
                
                # Write full-length batch if ready (independent counter)
                if len(batch_data_full) >= batch_size:
                    table_full = pa.Table.from_pylist(batch_data_full)
                    output_file_full = output_dir_full / f"batch_{len(list(output_dir_full.glob('*.parquet'))):06d}.parquet"
                    pq.write_table(table_full, output_file_full)
                    batch_data_full = []
            
            # Write remaining CDR3 data
            if batch_data_cdr3:
                table_cdr3 = pa.Table.from_pylist(batch_data_cdr3)
                output_file_cdr3 = output_dir / f"batch_{batch_num:06d}.parquet"
                pq.write_table(table_cdr3, output_file_cdr3)
                pbar.update(len(batch_data_cdr3))
            
            # Write remaining full-length data
            if batch_data_full:
                table_full = pa.Table.from_pylist(batch_data_full)
                output_file_full = output_dir_full / f"batch_{len(list(output_dir_full.glob('*.parquet'))):06d}.parquet"
                pq.write_table(table_full, output_file_full)
    
    cdr3_files = len(list(output_dir.glob('*.parquet')))
    full_files = len(list(output_dir_full.glob('*.parquet')))
    print(f"✓ Wrote parquet files:")
    print(f"   CDR3: {output_dir} ({cdr3_files} files)")
    print(f"   Full: {output_dir_full} ({full_files} files, filtered for full-length TCR sequences)")

def main():
    parser = argparse.ArgumentParser(description="Streaming deduplication with external sort")
    parser.add_argument("--path", nargs='+', required=True, help="Input parquet directories")
    parser.add_argument("--output-deduplicated", required=True, help="Output directory for deduplicated parquet (CDR3 version)")
    parser.add_argument("--output-deduplicated-full", help="Output directory for full-length sequences (default: <output-deduplicated>_full)")
    parser.add_argument("--mode", required=True, choices=list(MODE_CONFIGS.keys()), help="Processing mode")
    parser.add_argument("--sample", type=int, help="Sample N files for testing")
    parser.add_argument("--tmp-dir", default="/mnt/ephemeral/temp", help="Temp directory for small intermediate files")
    parser.add_argument("--work-dir", help="Working directory for large sorted chunks and merge outputs (default: same as tmp-dir, use EBS for large datasets)")
    parser.add_argument("--max-permutations", type=int, help="Max permutations per molecule")
    parser.add_argument("--no-permutations", action="store_true", help="Skip permutation generation entirely")
    parser.add_argument("--keep-all-permutations", action="store_true", help="Generate permutations but skip permutation deduplication")
    parser.add_argument("--buffer-size", default="50G", help="Sort buffer size (e.g., 50G) - only used with --use-unix-sort")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of parallel workers (default: all CPU cores)")
    parser.add_argument("--sort-chunk-size", type=int, default=200_000_000, help="Chunk size for PyArrow sort (default: 200M lines)")
    parser.add_argument("--sort-max-open-files", type=int, default=1024, help="Max files to open per merge pass (default: 1024)")
    parser.add_argument("--use-unix-sort", action="store_true", help="Use Unix sort instead of PyArrow sort (not recommended for large datasets)")
    parser.add_argument("--stitch-tcr", action="store_true", help="Generate full-length TCR sequences from CDR3 + gene segments using stitchr")
    parser.add_argument("--resume", action="store_true", help="Resume from existing intermediate files (chunk files, sorted files, etc.)")
    parser.add_argument("--standardized-input", action="store_true",
                        help="Input is from data/standardized/ (uses mhc_one/mhc_two allele IDs directly, no mhc_one_id mapping needed)")

    args = parser.parse_args()
    
    # Setup temp directory (for small ops)
    temp_dir = Path(args.tmp_dir)
    temp_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup work directory (for large chunks/merge - can be on EBS)
    work_dir = Path(args.work_dir) if args.work_dir else temp_dir
    work_dir.mkdir(exist_ok=True, parents=True)
    
    # Set Unix sort temp dir
    os.environ['TMPDIR'] = str(temp_dir)
    
    # Check available space
    temp_stat = os.statvfs(temp_dir)
    temp_free_gb = (temp_stat.f_bavail * temp_stat.f_frsize) / (1024**3)
    print(f"💾 Available space in temp dir ({temp_dir}): {temp_free_gb:.1f} GB")
    
    if work_dir != temp_dir:
        work_stat = os.statvfs(work_dir)
        work_free_gb = (work_stat.f_bavail * work_stat.f_frsize) / (1024**3)
        print(f"💾 Available space in work dir ({work_dir}): {work_free_gb:.1f} GB")
    
    # Find all parquet files
    all_files = []
    for path in args.path:
        all_files.extend(glob.glob(f"{path}/**/*.parquet", recursive=True))

    print(f"📁 Found {len(all_files):,} parquet files")

    if args.sample:
        import random
        all_files = random.sample(all_files, min(args.sample, len(all_files)))
        print(f"📊 Sampling {len(all_files):,} files")

    # Check for resume state
    resume_state = None
    if args.resume:
        resume_state = detect_resume_state(work_dir)
        if resume_state['stage'] != 'none':
            print(f"\n🔄 RESUME MODE ENABLED")
            print(f"   Detected stage: {resume_state['stage']}")
            if resume_state['stage'] == 'chunks':
                print(f"   Found {len(resume_state['chunk_files'])} chunk files")
                if resume_state['valid_count']:
                    print(f"   Estimated valid rows: {resume_state['valid_count']:,}")
            elif resume_state['stage'] == 'sorted':
                print(f"   Found sorted file: {resume_state['sorted_file']}")
            elif resume_state['stage'] == 'deduped':
                print(f"   Found deduped file: {resume_state['deduped_file']}")
                if resume_state['unique_count']:
                    print(f"   Unique molecules: {resume_state['unique_count']:,}")
            elif resume_state['stage'] == 'permutations':
                print(f"   Found permutations file: {resume_state['perm_file']}")
                if resume_state['perm_count']:
                    print(f"   Permutations: {resume_state['perm_count']:,}")
            elif resume_state['stage'] == 'final':
                print(f"   Found final file: {resume_state['final_file']}")
                if resume_state['perm_count']:
                    print(f"   Final rows: {resume_state['perm_count']:,}")
                print(f"   Skipping directly to output writing")
        else:
            print(f"\n🔄 RESUME MODE ENABLED (no existing files found, starting from scratch)")

    # Stage 1: Molecule deduplication
    sorted_file = work_dir / "sorted.txt"
    deduped_file = work_dir / "deduped.txt"

    # Skip Stage 1 if resuming from later stages
    if resume_state and resume_state['stage'] in ['deduped', 'permutations', 'final']:
        print(f"\n{'='*60}")
        print(f"STAGE 1: MOLECULE DEDUPLICATION - SKIPPED (resuming from {resume_state['stage']})")
        print(f"{'='*60}")
        unique_count = resume_state.get('unique_count', 0)
        valid_count = resume_state.get('valid_count', 0)
        # Use the deduped file from resume state
        if resume_state['deduped_file'] and resume_state['deduped_file'].exists():
            deduped_file = resume_state['deduped_file']
    else:
        print(f"\n{'='*60}")
        print(f"STAGE 1: MOLECULE DEDUPLICATION (mode={args.mode})")
        print(f"{'='*60}")

        # Check if we can resume from chunks or sorted file
        if resume_state and resume_state['stage'] == 'chunks':
            print(f"\n🔄 Resuming from {len(resume_state['chunk_files'])} existing chunk files...")
            print("\n📊 Step 1/2: Merging existing chunks...")
            merge_sorted_files(resume_state['chunk_files'], sorted_file, work_dir, max_open_files=args.sort_max_open_files)
            valid_count = resume_state.get('valid_count', 0)
        elif resume_state and resume_state['stage'] == 'sorted':
            print(f"\n🔄 Resuming from existing sorted file...")
            # Use the existing sorted file
            sorted_file = resume_state['sorted_file']
            valid_count = 0  # Unknown, will be counted during dedup
        else:
            # Normal processing - extract and sort
            if args.use_unix_sort:
                # Traditional: extract → external sort
                extract_file = work_dir / "extract.txt"
                print("\n📊 Step 1/3: Extracting and tagging molecules...")
                valid_count = extract_parquet_to_temp(all_files, extract_file, args.mode, args.num_workers, args.stitch_tcr)
                size_gb = extract_file.stat().st_size / (1024**3)
                print(f"✓ Extracted {valid_count:,} valid molecules ({size_gb:.2f} GB)")

                print("\n📊 Step 2/3: External Unix sort...")
                sort_time = external_sort(extract_file, sorted_file, work_dir, args.buffer_size, args.num_workers)
                print(f"✓ Sorted in {sort_time:.1f}s ({sort_time/60:.1f} min)")
                extract_file.unlink()
            else:
                # Streaming: extract directly into sorted chunks → merge, no giant extract file
                print("\n📊 Step 1/2: Extracting + building sorted chunks (streaming)...")
                valid_count, sort_time = extract_and_sort_streaming(
                    all_files, sorted_file, work_dir, args.mode,
                    args.num_workers if args.num_workers else cpu_count(),
                    args.sort_chunk_size, args.sort_max_open_files, args.stitch_tcr
                )
                print(f"✓ Streamed extract+sort in {sort_time:.1f}s ({sort_time/60:.1f} min)")

        # Deduplicate (unless we already have deduped file)
        if not (resume_state and resume_state['stage'] == 'sorted'):
            print("\n📊 Step 3/3: Streaming deduplication...")
        else:
            print("\n📊 Step 2/2: Streaming deduplication...")
        unique_count = stream_deduplicate(sorted_file, deduped_file)
        print(f"✓ Found {unique_count:,} unique molecules")
        sorted_file.unlink()
    
    # Stage 2: Permutation generation (if requested)
    if not args.no_permutations:
        perm_file = work_dir / "permutations.txt"

        # Skip Stage 2 if resuming from later stages
        if resume_state and resume_state['stage'] in ['permutations', 'final']:
            print(f"\n{'='*60}")
            print(f"STAGE 2: PERMUTATION GENERATION - SKIPPED (resuming from {resume_state['stage']})")
            print(f"{'='*60}")
            perm_count = resume_state.get('perm_count', 0)
            # Use the perm file from resume state if available
            if resume_state['perm_file'] and resume_state['perm_file'].exists():
                perm_file = resume_state['perm_file']
            # Don't delete deduped_file if we skipped this stage
        else:
            print(f"\n{'='*60}")
            print(f"STAGE 2: PERMUTATION GENERATION")
            print(f"{'='*60}")

            perm_count = generate_permutations(deduped_file, perm_file, args.mode, args.max_permutations, args.num_workers)
            print(f"✓ Generated {perm_count:,} permutations ({perm_count/unique_count:.1f}x expansion)")
            deduped_file.unlink()

        # Stage 3: Permutation deduplication (optional)
        if args.keep_all_permutations:
            print(f"\n📊 Skipping permutation deduplication (keeping all {perm_count:,} permutations)")
            final_file = perm_file
            final_count = perm_count
        else:
            # Skip Stage 3 if resuming from final stage
            if resume_state and resume_state['stage'] == 'final':
                print(f"\n{'='*60}")
                print(f"STAGE 3: PERMUTATION DEDUPLICATION - SKIPPED (resuming from final)")
                print(f"{'='*60}")
                final_file = resume_state['final_file']
                final_count = resume_state.get('perm_count', 0)
            else:
                print(f"\n{'='*60}")
                print(f"STAGE 3: PERMUTATION DEDUPLICATION")
                print(f"{'='*60}")

                final_file = work_dir / "final.txt"
                final_count = deduplicate_permutations(
                    perm_file, final_file, work_dir,
                    args.buffer_size, args.num_workers,
                    args.use_unix_sort, args.sort_chunk_size,
                    args.sort_max_open_files
                )
                print(f"✓ Kept {final_count:,} unique permutations")
                perm_file.unlink()
    else:
        # When --no-permutations is used
        if resume_state and resume_state['stage'] == 'deduped':
            # Use the deduped file from resume
            final_file = resume_state['deduped_file']
        else:
            final_file = deduped_file
        final_count = unique_count
    
    # Write output
    print(f"\n{'='*60}")
    print(f"WRITING OUTPUT")
    print(f"{'='*60}")
    
    output_dir = Path(args.output_deduplicated)
    
    # Determine full-length output directory
    if args.output_deduplicated_full:
        output_dir_full = Path(args.output_deduplicated_full)
    else:
        output_dir_full = Path(str(output_dir) + "_full")
    
    write_parquet_output(final_file, output_dir, output_dir_full)
    final_file.unlink()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ COMPLETE")
    print(f"{'='*60}")
    print(f"Input files: {len(all_files):,}")
    print(f"Valid molecules: {valid_count:,}")
    print(f"Unique molecules: {unique_count:,}")
    if not args.no_permutations:
        print(f"Total permutations: {perm_count:,}")
        print(f"Unique permutations: {final_count:,}")
    print(f"Output (CDR3): {output_dir}")
    print(f"Output (Full): {output_dir_full}")
    
    # Cleanup
    print(f"\n📊 Cleaning up temporary files...")
    try:
        # Only try to remove work_dir if it's empty and different from output dirs
        if work_dir != temp_dir:
            # Don't remove if it contains output
            output_dir = Path(args.output_deduplicated)
            output_dir_full = Path(args.output_deduplicated_full) if args.output_deduplicated_full else Path(str(output_dir) + "_full")
            if not (output_dir.is_relative_to(work_dir) or output_dir_full.is_relative_to(work_dir)):
                work_dir.rmdir()
        temp_dir.rmdir()
    except Exception as e:
        print(f"   ℹ️  Could not remove temp directories (may not be empty): {e}")
        pass

if __name__ == "__main__":
    main()
