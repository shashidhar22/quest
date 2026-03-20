#!/usr/bin/env python3
"""
Streaming deduplication with external Unix sort (no Ray dependency).
Three-stage pipeline: molecule dedup → permutation generation → permutation dedup
"""

import argparse
import hashlib
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
from itertools import combinations, permutations as iter_permutations
from multiprocessing import Pool, cpu_count
from functools import partial

# Fast JSON: use orjson if available (3-10x faster), fall back to stdlib json
try:
    import orjson
    def _json_dumps(obj):
        return orjson.dumps(obj).decode('utf-8')
    _json_loads = orjson.loads
except ImportError:
    _json_dumps = json.dumps
    _json_loads = json.loads

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

_TCR_FIELDS = {'tra', 'trb', 'tra_full', 'trb_full',
               'tra_cdr1', 'tra_cdr2', 'tra_cdr3',
               'trb_cdr1', 'trb_cdr2', 'trb_cdr3',
               'trav_gene', 'traj_gene', 'trad_gene',
               'trbv_gene', 'trbj_gene', 'trbd_gene'}
_ANTIGEN_FIELDS = {'peptide', 'mhc_one', 'mhc_two', 'mhc_one_id', 'mhc_two_id'}

# All molecule/metadata fields to retain from parquet rows
MOLECULE_FIELDS = frozenset([
    'tra', 'trb', 'peptide', 'mhc_one', 'mhc_two', 'mhc_one_id', 'mhc_two_id',
    'tra_cdr1', 'tra_cdr2', 'tra_cdr3', 'tra_full',
    'trb_cdr1', 'trb_cdr2', 'trb_cdr3', 'trb_full',
    'trav_gene', 'traj_gene', 'trad_gene',
    'trbv_gene', 'trbj_gene', 'trbd_gene',
    'binding', 'score',
    'source', 'study_id',
])

# The 5 core molecule fields used for dedup keys and permutations
_DEDUP_KEY_FIELDS = ('tra', 'trb', 'peptide', 'mhc_one', 'mhc_two')


def _emit_sub_row(row_dict: dict, keep_fields: set, mode: str) -> tuple:
    """Build a dedup-key + JSON line for a subset of fields from row_dict.
    Returns ("", "") if no valid sequences remain after filtering.
    Single-pass: builds dedup fields and molecule_data simultaneously."""
    dedup_fields = {}
    molecule_data = {}
    for k, v in row_dict.items():
        if k in _DEDUP_KEY_FIELDS:
            if k in keep_fields:
                dedup_fields[k] = v
                molecule_data[k] = v
            else:
                dedup_fields[k] = ""
                molecule_data[k] = ""
        elif k in MOLECULE_FIELDS:
            molecule_data[k] = v

    key = create_dedup_key(dedup_fields, mode)
    if not key:
        return "", ""
    return key, _json_dumps(molecule_data)


def process_single_parquet(args: tuple) -> tuple:
    """
    Process a single parquet file and return (lines, count).
    This runs in a separate process for parallelization.

    Extraction only: reads rows, creates dedup keys, and writes molecule data.
    TCR stitching is deferred to the write phase (write_parquet_output).
    """
    pf, mode, _stitch_tcr_unused, exclude_vdjdb_score_zero = args
    lines = []
    valid_count = 0

    try:
        table = pq.read_table(pf)
        # Column-wise extraction: only pull needed columns, skip unused ones
        needed = [c for c in table.column_names if c in MOLECULE_FIELDS]
        if needed:
            table = table.select(needed)
        df = table.to_pandas()

        for row_dict in df.to_dict('records'):
            # Filter negative binding for MLM mode
            if mode == "mlm":
                binding_val = str(row_dict.get("binding", "")).strip().lower()
                if binding_val == "neg":
                    # Salvage TCR-side and antigen-side as independent sub-rows
                    for keep in (_TCR_FIELDS, _ANTIGEN_FIELDS):
                        key, json_data = _emit_sub_row(row_dict, keep, mode)
                        if key:
                            lines.append(f"{key}\t{json_data}\n")
                            valid_count += 1
                    continue

            # Exclude VDJdb score-0 records when flag is set
            if exclude_vdjdb_score_zero:
                source_val = str(row_dict.get("source", "")).strip().lower()
                score_val = str(row_dict.get("score", "")).strip()
                if source_val == "vdjdb" and score_val == "0":
                    # Salvage TCR-side and antigen-side as independent sub-rows
                    for keep in (_TCR_FIELDS, _ANTIGEN_FIELDS):
                        key, json_data = _emit_sub_row(row_dict, keep, mode)
                        if key:
                            lines.append(f"{key}\t{json_data}\n")
                            valid_count += 1
                    continue

            dedup_key = create_dedup_key(row_dict, mode)

            if dedup_key:  # Valid row
                molecule_data = {k: v for k, v in row_dict.items() if k in MOLECULE_FIELDS}
                lines.append(f"{dedup_key}\t{_json_dumps(molecule_data)}\n")
                valid_count += 1
    except Exception as e:
        print(f"Warning: Failed to read {pf}: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

    return lines, valid_count, {}

def extract_parquet_to_temp(parquet_files: List[str], temp_file: Path, mode: str, num_workers: int = None, stitch_tcr: bool = False, exclude_vdjdb_score_zero: bool = False) -> int:
    """
    Extract parquet files to temp file with dedup keys (parallelized).
    Returns number of valid rows extracted.
    """
    if num_workers is None:
        num_workers = cpu_count()  # Use all cores for maximum throughput
    
    valid_count = 0
    
    print(f"   ℹ️  Using {num_workers} parallel workers")

    with open(temp_file, 'w', buffering=8*1024*1024) as out:  # 8MB write buffer
        with Pool(num_workers) as pool:
            # Process files in parallel
            args_list = [(pf, mode, stitch_tcr, exclude_vdjdb_score_zero) for pf in parquet_files]

            with tqdm(desc="Extracting parquet files", unit=" files", total=len(parquet_files)) as pbar:
                for lines, count, _gene_failures in pool.imap_unordered(process_single_parquet, args_list, chunksize=1):
                    # Write all lines from this file
                    out.writelines(lines)
                    valid_count += count
                    pbar.update(1)

    return valid_count

def extract_and_create_sorted_chunks(parquet_files: List[str], temp_dir: Path, mode: str,
                                     chunk_size: int, num_workers: int = None, stitch_tcr: bool = False,
                                     exclude_vdjdb_score_zero: bool = False) -> tuple[List[Path], int]:
    """
    Stream-extract parquet rows and directly build sorted chunk files without creating
    a giant intermediate extract file. Returns (chunk_files, valid_count).
    """
    if num_workers is None:
        num_workers = cpu_count()  # Use all cores for maximum throughput

    print(f"   ℹ️  Using {num_workers} parallel workers (streaming extract)")

    chunk_files: List[Path] = []
    current_chunk: List[str] = []
    chunk_idx = 0
    valid_count = 0

    args_list = [(pf, mode, stitch_tcr, exclude_vdjdb_score_zero) for pf in parquet_files]
    with Pool(num_workers) as pool:
        with tqdm(desc="Extracting + chunking", unit=" files", total=len(parquet_files)) as pbar:
            for lines, count, _gene_failures in pool.imap_unordered(process_single_parquet, args_list, chunksize=1):
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
                               num_workers: int, chunk_size: int, max_open_files: int, stitch_tcr: bool = False,
                               exclude_vdjdb_score_zero: bool = False) -> tuple[int, float]:
    """
    End-to-end streaming extract + chunked sort + multi-pass merge into sorted_file.
    Returns (valid_count, elapsed_seconds).
    """
    start = time.time()
    chunk_files, valid_count = extract_and_create_sorted_chunks(
        parquet_files, temp_dir, mode, chunk_size, num_workers, stitch_tcr,
        exclude_vdjdb_score_zero
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
                # Single rstrip instead of double strip(); find+slice instead of split
                line = line.rstrip('\n')
                if not line:
                    continue

                tab_pos = line.find('\t')
                if tab_pos < 0:
                    continue

                key = line[:tab_pos]

                if key != prev_key:
                    outfile.write(line[tab_pos + 1:])
                    outfile.write('\n')
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

    Each item in the batch is a (line_idx, line) tuple. Output lines use
    the line index instead of duplicating the full JSON payload, reducing
    Stage 2+3 I/O by ~60-70%.
    """
    items, mode, max_perms = args
    output_lines = []

    for line_idx, line in items:
        line = line.rstrip('\n')
        if not line:
            continue

        # Parse line: could be "dedup_key\tjson" from stage 1 or just "json"
        tab_pos = line.find('\t')
        if tab_pos >= 0:
            json_data = line[tab_pos + 1:]
        else:
            json_data = line

        row_dict = _json_loads(json_data)

        # Get non-empty molecule fields
        molecule_values = []
        for field in _DEDUP_KEY_FIELDS:
            val = row_dict.get(field, "")
            if val and val != "nan":
                molecule_values.append((field, val))

        # Generate all permutations of all subset sizes
        # For 3 molecules [A, B, C], generate:
        # - Size 1: A, B, C
        # - Size 2: AB, BA, AC, CA, BC, CB
        # - Size 3: ABC, ACB, BAC, BCA, CAB, CBA

        line_idx_str = str(line_idx)

        if not molecule_values:
            output_lines.append(f"empty\t{line_idx_str}\n")
        else:
            all_perms = []

            for subset_size in range(1, len(molecule_values) + 1):
                for subset in combinations(molecule_values, subset_size):
                    for perm in iter_permutations(subset):
                        all_perms.append(perm)

            # Apply max_perms limit if specified
            if max_perms and len(all_perms) > max_perms:
                import random
                all_perms = random.sample(all_perms, max_perms)

            # Emit perm_key + line index (not full JSON) for each permutation
            for perm in all_perms:
                perm_key = "|".join(f"{field}:{val}" for field, val in perm)
                output_lines.append(f"{perm_key}\t{line_idx_str}\n")

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

    # Generator function to yield batches of (line_idx, line) tuples
    def batch_generator():
        """Yield batches of (line_idx, line) from input file without loading all into memory."""
        with open(input_file, 'r', buffering=8*1024*1024) as infile:
            batch = []
            for line_idx, line in enumerate(infile):
                batch.append((line_idx, line))
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
        # Estimate valid_count from file sizes (avoid slow line counting on large files)
        total_bytes = sum(f.stat().st_size for f in chunk_files)
        # Estimate ~100 bytes per line (typical for dedup key lines)
        state['valid_count'] = int(total_bytes / 100)
        return state

    return state


def _stitch_row(row_dict: dict, stitcher, stitch_cache: dict) -> None:
    """Stitch TCR sequences for a row, using cache for repeated combos."""
    for chain, cdr3_key, v_key, j_key, full_key in [
        ('TRA', 'tra', 'trav_gene', 'traj_gene', 'tra_full'),
        ('TRB', 'trb', 'trbv_gene', 'trbj_gene', 'trb_full'),
    ]:
        cdr3 = row_dict.get(cdr3_key, '')
        v_gene = row_dict.get(v_key, '')
        j_gene = row_dict.get(j_key, '')
        if not (cdr3 and v_gene and j_gene):
            continue
        if row_dict.get(full_key):
            continue

        cache_key = (cdr3, v_gene, j_gene, chain)
        if cache_key not in stitch_cache:
            norm_v = stitcher.normalize_gene_name(v_gene, chain)
            norm_j = stitcher.normalize_gene_name(j_gene, chain)
            result = None
            if norm_v and norm_j:
                result = stitcher.stitch_tcr(
                    cdr3=cdr3, v_gene=norm_v, j_gene=norm_j,
                    chain=chain, skip_normalize=True
                )
            stitch_cache[cache_key] = result

        result = stitch_cache[cache_key]
        if result:
            row_dict[full_key] = result


def _stitch_file_chunk(args: tuple) -> tuple:
    """
    Worker: read a byte-range chunk of the deduped file, stitch, write results to TSV.
    Returns (num_processed, num_stitched, output_path).
    """
    deduped_path, start_offset, end_offset, output_path = args
    stitcher = TCRStitcher(species="HUMAN")
    num_processed = 0
    num_stitched = 0

    with open(deduped_path, 'r') as infile, open(output_path, 'w', buffering=8*1024*1024) as out:
        infile.seek(start_offset)
        if start_offset > 0:
            infile.readline()  # skip partial line

        while infile.tell() < end_offset:
            line = infile.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            row_dict = _json_loads(line)
            num_processed += 1

            for chain, cdr3_key, v_key, j_key, full_key in [
                ('TRA', 'tra', 'trav_gene', 'traj_gene', 'tra_full'),
                ('TRB', 'trb', 'trbv_gene', 'trbj_gene', 'trb_full'),
            ]:
                cdr3 = row_dict.get(cdr3_key, '')
                v_gene = row_dict.get(v_key, '')
                j_gene = row_dict.get(j_key, '')
                if not (cdr3 and v_gene and j_gene):
                    continue
                if row_dict.get(full_key):
                    continue

                norm_v = stitcher.normalize_gene_name(v_gene, chain)
                norm_j = stitcher.normalize_gene_name(j_gene, chain)
                if not (norm_v and norm_j):
                    continue

                result = stitcher.stitch_tcr(
                    cdr3=cdr3, v_gene=norm_v, j_gene=norm_j,
                    chain=chain, skip_normalize=True
                )
                if result:
                    md5_hex = hashlib.md5(
                        f"{cdr3}|{v_gene}|{j_gene}|{chain}".encode()
                    ).hexdigest()
                    out.write(f"{md5_hex}\t{result}\n")
                    num_stitched += 1

    return num_processed, num_stitched, str(output_path)


def pre_stitch_deduped_file(deduped_file: Path, tmp_dir: Path,
                            num_workers: int = None) -> dict:
    """
    Parallel pre-stitch of deduped file. Returns compact dict[bytes, bytes].

    Partitions the deduped file into byte-range chunks, dispatches parallel
    workers that each write stitch results to per-worker TSV files, then
    reads those TSVs into a compact dict (md5 digest -> ASCII sequence bytes).
    """
    if num_workers is None:
        num_workers = min(cpu_count(), 64)

    file_size = deduped_file.stat().st_size
    if file_size == 0:
        print("   No data to stitch (empty deduped file)")
        return {}

    chunk_size = max(1, file_size // num_workers)

    # Build (start, end) byte offsets for each worker
    args_list = []
    for i in range(num_workers):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, file_size)
        if start >= file_size:
            break
        out_path = tmp_dir / f"stitch_worker_{i:04d}.tsv"
        args_list.append((str(deduped_file), start, end, str(out_path)))

    # Phase 1: Parallel stitching to per-worker files
    total_processed = 0
    total_stitched = 0
    output_files = []
    with Pool(len(args_list)) as pool:
        for n_proc, n_stitch, out_path in tqdm(
            pool.imap_unordered(_stitch_file_chunk, args_list),
            total=len(args_list), desc="Pre-stitching chunks", unit=" chunks"
        ):
            total_processed += n_proc
            total_stitched += n_stitch
            output_files.append(Path(out_path))

    print(f"   Processed {total_processed:,} molecules, stitched {total_stitched:,} chains")

    # Phase 2: Load worker files into compact dict
    print(f"   Loading stitch results into compact cache...")
    stitch_cache = {}  # bytes(md5) -> bytes(sequence)
    for f in tqdm(output_files, desc="Loading stitch files", unit=" files"):
        if not f.exists():
            continue
        with open(f, 'r', buffering=8*1024*1024) as fh:
            for line in fh:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    md5_key = bytes.fromhex(parts[0])
                    if md5_key not in stitch_cache:  # first occurrence wins
                        stitch_cache[md5_key] = parts[1].encode('ascii')
        f.unlink()  # free disk immediately

    print(f"   Cache: {len(stitch_cache):,} unique combos")
    return stitch_cache


def _lookup_stitch(row_dict: dict, stitch_cache: dict) -> None:
    """Look up pre-stitched sequences from compact cache (md5 bytes -> seq bytes)."""
    for chain, cdr3_key, v_key, j_key, full_key in [
        ('TRA', 'tra', 'trav_gene', 'traj_gene', 'tra_full'),
        ('TRB', 'trb', 'trbv_gene', 'trbj_gene', 'trb_full'),
    ]:
        cdr3 = row_dict.get(cdr3_key, '')
        v_gene = row_dict.get(v_key, '')
        j_gene = row_dict.get(j_key, '')
        if not (cdr3 and v_gene and j_gene):
            continue
        if row_dict.get(full_key):
            continue

        md5_key = hashlib.md5(f"{cdr3}|{v_gene}|{j_gene}|{chain}".encode()).digest()
        result = stitch_cache.get(md5_key)
        if result:
            row_dict[full_key] = result.decode('ascii')


class LineIndexedFile:
    """Random-access line reader backed by byte-offset index.

    Builds a list of byte offsets for each line in the file on init,
    then supports O(1) random access by line number via seek.
    Memory: ~8 bytes per line (offset array).
    """

    def __init__(self, path):
        self.path = str(path)
        self.offsets = []
        with open(self.path, 'rb') as f:
            offset = 0
            for raw_line in f:
                self.offsets.append(offset)
                offset += len(raw_line)
        self._fh = open(self.path, 'rb')

    def __getitem__(self, idx):
        self._fh.seek(self.offsets[idx])
        return self._fh.readline().rstrip(b'\n').decode('utf-8')

    def __len__(self):
        return len(self.offsets)

    def close(self):
        self._fh.close()


def write_parquet_output(input_file: Path, output_dir: Path, output_dir_full: Path,
                         batch_size: int = 1000000, stitch_tcr: bool = False,
                         pre_stitch_cache: dict = None, deduped_file: Path = None):
    """
    Convert deduplicated text file to two parquet outputs:
    1. CDR3 version (tra/trb) - permutation_key, concatenated_sequence (all rows)
    2. Full-length version (tra_full/trb_full) - permutation_key, concatenated_sequence

    FILTERING: The full-length output includes rows where at least one valid sequence
    part exists. For TCR fields (tra/trb), this requires the full-length stitched
    version (tra_full/trb_full). For non-TCR fields (peptide, mhc_one, mhc_two),
    the regular value is used as-is. Rows with no valid fields at all are excluded.

    IMPORTANT: Permutation keys in the full-length output only include fields where
    the full-length version actually exists:
    - permutation_key='tra' means tra_full exists
    - permutation_key='tra_trb' means both tra_full and trb_full exist
    - permutation_key='peptide' means only peptide (no TCR full-length)
    - permutation_key='peptide_mhc_one' means peptide + MHC-I (pMHC record)

    This allows users to filter for rows with specific full-length molecules.

    Args:
        pre_stitch_cache: Compact dict[bytes, bytes] from pre_stitch_deduped_file().
            When provided, uses fast md5 lookup instead of inline stitching.
            When None and stitch_tcr=True, falls back to inline stitching.
    """
    print(f"\n📊 Writing parquet outputs...")

    # Load deduped file for index-based lookup (used when Stage 2 emits line indices)
    mol_lookup = None
    if deduped_file is not None and deduped_file.exists():
        print(f"   Loading molecule index from {deduped_file}...")
        mol_lookup = LineIndexedFile(deduped_file)
        print(f"   Indexed {len(mol_lookup):,} molecules for random access")

    # Determine stitching strategy
    stitcher = None
    inline_stitch_cache: dict = {}
    if pre_stitch_cache is not None:
        print(f"   Using pre-stitch cache ({len(pre_stitch_cache):,} entries)")
    elif stitch_tcr and STITCHER_AVAILABLE:
        # Fallback: inline stitching (e.g., when resuming without deduped file)
        try:
            stitcher = TCRStitcher(species="HUMAN")
            import logging as _logging
            _logging.getLogger('tidytcells').setLevel(_logging.ERROR)
            print(f"   TCR stitching ENABLED (inline fallback) - will generate full-length sequences")
        except Exception as e:
            print(f"Warning: Failed to initialize TCRStitcher: {e}")
            stitcher = None

    output_dir.mkdir(exist_ok=True, parents=True)
    output_dir_full.mkdir(exist_ok=True, parents=True)

    # Clear any existing parquet files to prevent stale file contamination
    for old_file in output_dir.glob('*.parquet'):
        old_file.unlink()
    for old_file in output_dir_full.glob('*.parquet'):
        old_file.unlink()

    batch_num = 0
    batch_num_full = 0
    batch_data_cdr3 = []
    batch_data_full = []
    stitch_attempts = 0
    stitch_hits = 0

    with open(input_file, 'r') as f:
        with tqdm(desc="Writing parquet", unit=" rows", unit_scale=True) as pbar:
            for line in f:
                # For permutations, the line is: perm_key\tline_idx (or perm_key\tjson for legacy)
                # For molecules, the line is just: json (or line_idx if index mode)
                line = line.rstrip('\n')
                if not line:
                    continue
                tab_pos = line.find('\t')
                if tab_pos >= 0:
                    old_perm_key = line[:tab_pos]
                    ref = line[tab_pos + 1:]
                    # Resolve: ref is either a line index (integer) or JSON
                    if mol_lookup is not None and ref.isdigit():
                        row_dict = _json_loads(mol_lookup[int(ref)])
                    else:
                        row_dict = _json_loads(ref)
                else:
                    ref = line
                    # Resolve: ref is either a line index or JSON
                    if mol_lookup is not None and ref.isdigit():
                        row_dict = _json_loads(mol_lookup[int(ref)])
                    else:
                        row_dict = _json_loads(ref)
                    # Create a simple key from present fields
                    present_fields = []
                    for field in _DEDUP_KEY_FIELDS:
                        if field in row_dict and row_dict[field]:
                            present_fields.append(field)
                    old_perm_key = "_".join(present_fields) if present_fields else "empty"
                
                # Stitch TCR sequences: pre-stitch cache (fast) or inline fallback
                if pre_stitch_cache is not None:
                    _lookup_stitch(row_dict, pre_stitch_cache)
                    stitch_attempts += 1
                elif stitcher is not None:
                    cache_size_before = len(inline_stitch_cache)
                    _stitch_row(row_dict, stitcher, inline_stitch_cache)
                    stitch_attempts += 1
                    if len(inline_stitch_cache) == cache_size_before:
                        stitch_hits += 1

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
                
                # Include in full output if we have any valid sequence parts.
                # For TCR fields, this requires the full-length stitched version (tra_full/trb_full).
                # For non-TCR fields (peptide, mhc_one, mhc_two), the regular value is used as-is.
                if actual_fields_full:
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
                    output_file_full = output_dir_full / f"batch_{batch_num_full:06d}.parquet"
                    pq.write_table(table_full, output_file_full)
                    batch_num_full += 1
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
                output_file_full = output_dir_full / f"batch_{batch_num_full:06d}.parquet"
                pq.write_table(table_full, output_file_full)
    
    cdr3_files = len(list(output_dir.glob('*.parquet')))
    full_files = len(list(output_dir_full.glob('*.parquet')))
    print(f"✓ Wrote parquet files:")
    print(f"   CDR3: {output_dir} ({cdr3_files} files)")
    print(f"   Full: {output_dir_full} ({full_files} files, filtered for full-length TCR sequences)")

    # Print stitch statistics
    if pre_stitch_cache is not None and stitch_attempts > 0:
        print(f"\n   Stitch statistics (pre-stitch cache):")
        print(f"   Rows processed: {stitch_attempts:,}")
        print(f"   Cache entries: {len(pre_stitch_cache):,} unique combos")
    elif stitcher is not None and stitch_attempts > 0:
        cache_hit_rate = (stitch_hits / stitch_attempts * 100) if stitch_attempts > 0 else 0
        stitch_successes = sum(1 for v in inline_stitch_cache.values() if v is not None)
        print(f"\n   Stitch statistics (inline fallback):")
        print(f"   Rows processed: {stitch_attempts:,}")
        print(f"   Cache entries: {len(inline_stitch_cache):,} unique (cdr3, v, j, chain) combos")
        print(f"   Cache hit rate: {cache_hit_rate:.1f}%")
        print(f"   Successful stitches: {stitch_successes:,}/{len(inline_stitch_cache):,} unique combos")

    # Close molecule lookup if used
    if mol_lookup is not None:
        mol_lookup.close()

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
    parser.add_argument("--exclude-vdjdb-score-zero", action="store_true",
                        help="Exclude VDJdb score-0 records (source='vdjdb' AND score='0')")

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
    if args.stitch_tcr:
        print(f"   ℹ️  TCR stitching will run post-dedup during output writing (cached)")
    if args.exclude_vdjdb_score_zero:
        print(f"   ℹ️  Excluding VDJdb score-0 records")

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
                valid_count = extract_parquet_to_temp(all_files, extract_file, args.mode, args.num_workers, args.stitch_tcr, args.exclude_vdjdb_score_zero)
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
                    args.sort_chunk_size, args.sort_max_open_files, args.stitch_tcr,
                    args.exclude_vdjdb_score_zero
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

    # Pre-stitch if enabled (between dedup and permutations)
    stitch_cache = None
    if args.stitch_tcr and STITCHER_AVAILABLE:
        # Only run pre-stitch if the deduped file exists (not resuming past it)
        if deduped_file.exists():
            print(f"\n{'='*60}")
            print(f"PRE-STITCHING ({unique_count:,} deduplicated molecules)")
            print(f"{'='*60}")
            stitch_cache = pre_stitch_deduped_file(deduped_file, work_dir, args.num_workers)
        else:
            print(f"\n   Pre-stitch skipped (deduped file not available, will use inline fallback)")

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
            expansion = perm_count / unique_count if unique_count > 0 else 0
            print(f"✓ Generated {perm_count:,} permutations ({expansion:.1f}x expansion)")
            # Keep deduped_file alive — needed for index-based lookup in write_parquet_output

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
    
    # Pass deduped_file for index-based lookup when permutations were generated
    lookup_file = deduped_file if (not args.no_permutations and deduped_file.exists()) else None
    write_parquet_output(final_file, output_dir, output_dir_full,
                         stitch_tcr=args.stitch_tcr, pre_stitch_cache=stitch_cache,
                         deduped_file=lookup_file)
    final_file.unlink()
    # Clean up deduped file now that output is written
    if lookup_file is not None and lookup_file.exists():
        lookup_file.unlink()
    
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
