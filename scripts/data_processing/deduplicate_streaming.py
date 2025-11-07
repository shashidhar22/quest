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

# Valid amino acids
_valid_aa = set("ACDEFGHIKLMNPQRSTVWY")

# Mode configurations - use MHC IDs not full sequences for dedup!
MODE_CONFIGS = {
    "tra": {"required": ["tra"], "optional": []},
    "trb": {"required": ["trb"], "optional": []},
    "tcr_pairing": {"required": ["tra", "trb"], "optional": []},
    "mhc_binding": {"required": ["peptide"], "optional": ["mhc_one_id", "mhc_two_id"]},
    "specificity": {"required": ["tra", "peptide"], "optional": ["mhc_one_id", "mhc_two_id"]},
    "default": {"required": [], "optional": ["tra", "trb", "peptide", "mhc_one_id", "mhc_two_id"]},
    "balanced": {"required": [], "optional": ["tra", "trb", "peptide", "mhc_one_id", "mhc_two_id"]},
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
    """
    pf, mode = args
    lines = []
    valid_count = 0
    
    try:
        table = pq.read_table(pf)
        df = table.to_pandas()
        
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            dedup_key = create_dedup_key(row_dict, mode)
            
            if dedup_key:  # Valid row
                # Keep molecule columns (chains, peptide, MHC sequences and IDs)
                # Also keep full-length stitched sequences and gene annotations
                molecule_data = {
                    k: v for k, v in row_dict.items()
                    if k in ['tra', 'trb', 'peptide', 'mhc_one', 'mhc_two', 'mhc_one_id', 'mhc_two_id',
                            'tra_full', 'trb_full',  # Full-length stitched TCR sequences
                            'trav_gene', 'traj_gene', 'trad_gene',  # TRA gene segments
                            'trbv_gene', 'trbj_gene', 'trbd_gene']  # TRB gene segments
                }
                lines.append(f"{dedup_key}\t{json.dumps(molecule_data)}\n")
                valid_count += 1
    except Exception as e:
        print(f"Warning: Failed to read {pf}: {e}")
    
    return lines, valid_count

def extract_parquet_to_temp(parquet_files: List[str], temp_file: Path, mode: str, num_workers: int = None) -> int:
    """
    Extract parquet files to temp file with dedup keys (parallelized).
    Returns number of valid rows extracted.
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)  # Leave 2 cores free
    
    valid_count = 0
    
    print(f"   ℹ️  Using {num_workers} parallel workers")
    
    with open(temp_file, 'w') as out:
        with Pool(num_workers) as pool:
            # Process files in parallel
            process_func = partial(process_single_parquet, mode=mode)
            args_list = [(pf, mode) for pf in parquet_files]
            
            with tqdm(desc="Extracting parquet files", unit=" files", total=len(parquet_files)) as pbar:
                for lines, count in pool.imap_unordered(process_single_parquet, args_list, chunksize=1):
                    # Write all lines from this file
                    out.writelines(lines)
                    valid_count += count
                    pbar.update(1)
    
    return valid_count

def extract_and_create_sorted_chunks(parquet_files: List[str], temp_dir: Path, mode: str,
                                     chunk_size: int, num_workers: int = None) -> tuple[List[Path], int]:
    """
    Stream-extract parquet rows and directly build sorted chunk files without creating
    a giant intermediate extract file. Returns (chunk_files, valid_count).
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)

    print(f"   ℹ️  Using {num_workers} parallel workers (streaming extract)")

    chunk_files: List[Path] = []
    current_chunk: List[str] = []
    chunk_idx = 0
    valid_count = 0

    args_list = [(pf, mode) for pf in parquet_files]
    with Pool(num_workers) as pool:
        with tqdm(desc="Extracting + chunking", unit=" files", total=len(parquet_files)) as pbar:
            for lines, count in pool.imap_unordered(process_single_parquet, args_list, chunksize=1):
                valid_count += count
                # Append lines and spill when needed
                if lines:
                    current_chunk.extend(lines)
                    while len(current_chunk) >= chunk_size:
                        # Sort and spill a full chunk
                        to_write = current_chunk[:chunk_size]
                        del current_chunk[:chunk_size]
                        to_write.sort()
                        out_path = temp_dir / f"chunk_{chunk_idx:06d}.txt"
                        with open(out_path, 'w') as cf:
                            cf.writelines(to_write)
                        chunk_files.append(out_path)
                        chunk_idx += 1
                pbar.update(1)

    # Flush remainder
    if current_chunk:
        current_chunk.sort()
        out_path = temp_dir / f"chunk_{chunk_idx:06d}.txt"
        with open(out_path, 'w') as cf:
            cf.writelines(current_chunk)
        chunk_files.append(out_path)
        chunk_idx += 1

    print(f"   ℹ️  Created {len(chunk_files)} sorted chunk files (streamed)")
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
            f = open(p, 'r')
            file_iters.append((f, iter(f)))
        try:
            heap = []
            for idx, (fh, it) in enumerate(file_iters):
                try:
                    line = next(it)
                    heapq.heappush(heap, (line, idx))
                except StopIteration:
                    pass
            with open(out_path, 'w') as out:
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
                               num_workers: int, chunk_size: int, max_open_files: int) -> tuple[int, float]:
    """
    End-to-end streaming extract + chunked sort + multi-pass merge into sorted_file.
    Returns (valid_count, elapsed_seconds).
    """
    start = time.time()
    chunk_files, valid_count = extract_and_create_sorted_chunks(
        parquet_files, temp_dir, mode, chunk_size, num_workers
    )
    merge_sorted_files(chunk_files, sorted_file, temp_dir, max_open_files=max_open_files)
    return valid_count, time.time() - start

def pyarrow_sort(input_file: Path, output_file: Path, temp_dir: Path, chunk_size: int = 50_000_000, max_open_files: int = 256) -> float:
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
    with open(input_file, 'r') as f:
        for line in tqdm(f, desc="   Reading and sorting chunks", unit=" lines", unit_scale=True):
            if not line:
                continue
            current_chunk.append(line)
            if len(current_chunk) >= chunk_size:
                current_chunk.sort()
                chunk_path = temp_dir / f"chunk_{chunk_idx:06d}.txt"
                with open(chunk_path, 'w') as cf:
                    cf.writelines(current_chunk)
                chunk_files.append(chunk_path)
                current_chunk = []
                chunk_idx += 1

        # Flush last chunk
        if current_chunk:
            current_chunk.sort()
            chunk_path = temp_dir / f"chunk_{chunk_idx:06d}.txt"
            with open(chunk_path, 'w') as cf:
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
            f = open(p, 'r')
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

            with open(out_path, 'w') as out:
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
        num_threads = max(4, cpu_count() - 2)  # Use most cores, leave 2 free
    
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
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
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
        num_workers = max(1, cpu_count() - 2)
    
    print(f"   ℹ️  Using {num_workers} parallel workers")
    
    # Read input file in batches
    batch_size = 10000  # Process 10k molecules per batch
    perm_count = 0
    
    with open(input_file, 'r') as infile:
        lines = infile.readlines()
    
    total_molecules = len(lines)
    
    # Create batches
    batches = []
    for i in range(0, len(lines), batch_size):
        batch = lines[i:i+batch_size]
        batches.append((batch, mode, max_perms))
    
    # Process batches in parallel
    with open(output_file, 'w') as outfile:
        with Pool(num_workers) as pool:
            with tqdm(desc="Generating permutations", unit=" batches", total=len(batches)) as pbar:
                for output_lines in pool.imap_unordered(process_molecule_batch, batches, chunksize=1):
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
        # New format - just field names separated by _, no sequences
        return perm_key.split('_'), {}


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


def write_parquet_output(input_file: Path, output_dir: Path, output_dir_full: Path, batch_size: int = 1000000):
    """
    Convert deduplicated text file to two parquet outputs:
    1. CDR3 version (tra/trb) - permutation_key, concatenated_sequence
    2. Full-length version (tra_full/trb_full) - permutation_key, concatenated_sequence
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
                seq_parts_full = []
                actual_fields_full = []
                for field in field_order:
                    # Try full-length version first (tra_full/trb_full)
                    lookup_field = field
                    if field == 'tra':
                        lookup_field = 'tra_full'
                    elif field == 'trb':
                        lookup_field = 'trb_full'
                    
                    val = row_dict.get(lookup_field, "")
                    
                    # If full-length version doesn't exist, fall back to regular field
                    if not val or str(val) == 'nan' or str(val) == '' or isinstance(val, float):
                        val = row_dict.get(field, "")
                    
                    if val and str(val) != 'nan' and str(val) != '' and not (isinstance(val, float)):
                        seq_parts_full.append(str(val))
                        actual_fields_full.append(field)
                
                seq_full = " ".join(seq_parts_full)
                perm_key_full = "_".join(actual_fields_full) if actual_fields_full else "empty"
                
                batch_data_full.append({
                    'permutation_key': perm_key_full,
                    'sequence': seq_full
                })
                
                if len(batch_data_cdr3) >= batch_size:
                    # Write CDR3 batch
                    table_cdr3 = pa.Table.from_pylist(batch_data_cdr3)
                    output_file_cdr3 = output_dir / f"batch_{batch_num:06d}.parquet"
                    pq.write_table(table_cdr3, output_file_cdr3)
                    
                    # Write full-length batch
                    table_full = pa.Table.from_pylist(batch_data_full)
                    output_file_full = output_dir_full / f"batch_{batch_num:06d}.parquet"
                    pq.write_table(table_full, output_file_full)
                    
                    pbar.update(len(batch_data_cdr3))
                    batch_num += 1
                    batch_data_cdr3 = []
                    batch_data_full = []
            
            # Write remaining
            if batch_data_cdr3:
                table_cdr3 = pa.Table.from_pylist(batch_data_cdr3)
                output_file_cdr3 = output_dir / f"batch_{batch_num:06d}.parquet"
                pq.write_table(table_cdr3, output_file_cdr3)
                
                table_full = pa.Table.from_pylist(batch_data_full)
                output_file_full = output_dir_full / f"batch_{batch_num:06d}.parquet"
                pq.write_table(table_full, output_file_full)
                
                pbar.update(len(batch_data_cdr3))
    
    print(f"✓ Wrote {batch_num + 1} parquet files to:")
    print(f"   CDR3: {output_dir}")
    print(f"   Full: {output_dir_full}")

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
    parser.add_argument("--num-workers", type=int, default=None, help="Number of parallel workers (default: CPU count - 2)")
    parser.add_argument("--sort-chunk-size", type=int, default=50_000_000, help="Chunk size for PyArrow sort (default: 50M lines)")
    parser.add_argument("--sort-max-open-files", type=int, default=256, help="Max files to open per merge pass (default: 256)")
    parser.add_argument("--use-unix-sort", action="store_true", help="Use Unix sort instead of PyArrow sort (not recommended for large datasets)")
    
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
    
    # Stage 1: Molecule deduplication
    print(f"\n{'='*60}")
    print(f"STAGE 1: MOLECULE DEDUPLICATION (mode={args.mode})")
    print(f"{'='*60}")
    
    sorted_file = work_dir / "sorted.txt"
    deduped_file = work_dir / "deduped.txt"
    
    if args.use_unix_sort:
        # Traditional: extract → external sort
        extract_file = work_dir / "extract.txt"
        print("\n📊 Step 1/3: Extracting and tagging molecules...")
        valid_count = extract_parquet_to_temp(all_files, extract_file, args.mode, args.num_workers)
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
            args.num_workers if args.num_workers else max(1, cpu_count()-2),
            args.sort_chunk_size, args.sort_max_open_files
        )
        print(f"✓ Streamed extract+sort in {sort_time:.1f}s ({sort_time/60:.1f} min)")
    
    # Deduplicate
    print("\n📊 Step 3/3: Streaming deduplication...")
    unique_count = stream_deduplicate(sorted_file, deduped_file)
    print(f"✓ Found {unique_count:,} unique molecules")
    sorted_file.unlink()
    
    # Stage 2: Permutation generation (if requested)
    if not args.no_permutations:
        print(f"\n{'='*60}")
        print(f"STAGE 2: PERMUTATION GENERATION")
        print(f"{'='*60}")
        
        perm_file = work_dir / "permutations.txt"
        perm_count = generate_permutations(deduped_file, perm_file, args.mode, args.max_permutations, args.num_workers)
        print(f"✓ Generated {perm_count:,} permutations ({perm_count/unique_count:.1f}x expansion)")
        deduped_file.unlink()
        
        # Stage 3: Permutation deduplication (optional)
        if args.keep_all_permutations:
            print(f"\n📊 Skipping permutation deduplication (keeping all {perm_count:,} permutations)")
            final_file = perm_file
            final_count = perm_count
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
