#!/usr/bin/env python3
"""
Create cluster-aware TCR sequence splits for contrastive learning.

This script generates train/val/test splits for TRA/TRB sequences that prevent data leakage
by ensuring no sequence cluster appears in multiple splits. It processes:
- Paired sequences (TRA-TRB pairs) with dual clustering to prevent leakage of either chain
- Unpaired TRA sequences not appearing in any pair
- Unpaired TRB sequences not appearing in any pair

For paired data, the script uses a connected components approach:
1. Cluster TRA sequences at 90% identity
2. Cluster TRB sequences at 90% identity
3. Build a graph where pairs sharing any cluster are connected
4. Find connected components - each component must stay in the same split
5. Assign entire connected components to train/val/test

This ensures:
- No TRA cluster appears in multiple splits
- No TRB cluster appears in multiple splits
- Pairs sharing any chain cluster stay in the same split

Output Structure:
    output_dir/
    ├── dataset_statistics.md
    ├── paired/
    │   ├── tra_clusters.tsv          # TRA sequence -> cluster mapping
    │   ├── trb_clusters.tsv          # TRB sequence -> cluster mapping
    │   ├── train.parquet             # Columns: tra, trb, tra_cluster, trb_cluster
    │   ├── val.parquet
    │   └── test.parquet
    ├── unpaired_tra/
    │   ├── clusters.tsv              # TRA sequence -> cluster mapping
    │   ├── train.parquet             # Cluster-aware split
    │   ├── val.parquet
    │   └── test.parquet
    └── unpaired_trb/
        ├── clusters.tsv              # TRB sequence -> cluster mapping
        ├── train.parquet             # Cluster-aware split
        ├── val.parquet
        └── test.parquet

Dependencies:
    - MMseqs2: For sequence clustering (conda install -c bioconda mmseqs2)
    - pyarrow: For parquet I/O
    - pandas: For data manipulation
    - tqdm: For progress bars

Usage:
    python scripts/data_processing/create_tcr_contrastive_dataset.py \\
        --input_dir data/deduplicated/full/foundation_permutations \\
        --output_dir data/eval/tcr_contrastive \\
        --similarity_threshold 0.9 \\
        --train_ratio 0.8 \\
        --val_ratio 0.1 \\
        --test_ratio 0.1

Author: Shashidhar Ravishankar, Claude
"""

import argparse
import heapq
import shutil
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Valid amino acids for sequence validation
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _merge_files(files: List[Path], output: Path) -> None:
    """K-way merge sorted files."""
    file_iters = []
    for p in files:
        f = open(p, 'r', buffering=8*1024*1024)
        file_iters.append((f, iter(f)))

    try:
        heap = []
        for idx, (fh, it) in enumerate(file_iters):
            try:
                line = next(it)
                heapq.heappush(heap, (line, idx))
            except StopIteration:
                pass

        with open(output, 'w', buffering=8*1024*1024) as out:
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
            fh.close()


def _merge_files_with_dedup(files: List[Path], output: Path) -> int:
    """K-way merge sorted files with deduplication. Returns count of unique lines."""
    file_iters = []
    for p in files:
        f = open(p, 'r', buffering=8*1024*1024)
        file_iters.append((f, iter(f)))

    unique_count = 0
    prev_line = None

    try:
        heap = []
        for idx, (fh, it) in enumerate(file_iters):
            try:
                line = next(it)
                heapq.heappush(heap, (line, idx))
            except StopIteration:
                pass

        with open(output, 'w', buffering=8*1024*1024) as out:
            while heap:
                line, idx = heapq.heappop(heap)
                if line != prev_line:
                    out.write(line)
                    unique_count += 1
                    prev_line = line
                try:
                    nxt = next(file_iters[idx][1])
                    heapq.heappush(heap, (nxt, idx))
                except StopIteration:
                    pass
    finally:
        for fh, _ in file_iters:
            fh.close()

    return unique_count


def external_sort_dedup(
    input_file: Path,
    output_file: Path,
    temp_dir: Path,
    chunk_size: int = 10_000_000,
    max_open_files: int = 256,
) -> int:
    """
    External merge sort + deduplication implemented in Python.

    Memory-efficient: sorts in chunks, merges with heap, deduplicates during output.

    Args:
        input_file: Path to input file
        output_file: Path to output file
        temp_dir: Directory for temporary chunk files
        chunk_size: Number of lines per chunk
        max_open_files: Maximum number of files to merge at once

    Returns:
        Count of unique lines written to output
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    chunk_files: List[Path] = []

    # Phase 1: Create sorted chunk files
    current_chunk: List[str] = []
    chunk_idx = 0

    with open(input_file, 'r', buffering=8*1024*1024) as f:
        for line in tqdm(f, desc="  Reading and sorting chunks", unit=" lines", unit_scale=True):
            if not line.strip():
                continue
            current_chunk.append(line)
            if len(current_chunk) >= chunk_size:
                current_chunk.sort()
                chunk_path = temp_dir / f"sort_chunk_{chunk_idx:06d}.txt"
                with open(chunk_path, 'w', buffering=8*1024*1024) as cf:
                    cf.writelines(current_chunk)
                chunk_files.append(chunk_path)
                current_chunk = []
                chunk_idx += 1

        # Flush last chunk
        if current_chunk:
            current_chunk.sort()
            chunk_path = temp_dir / f"sort_chunk_{chunk_idx:06d}.txt"
            with open(chunk_path, 'w', buffering=8*1024*1024) as cf:
                cf.writelines(current_chunk)
            chunk_files.append(chunk_path)

    # Phase 2 & 3: K-way merge with deduplication
    if not chunk_files:
        output_file.touch()
        return 0

    if len(chunk_files) == 1:
        # Single chunk - just deduplicate
        unique_count = 0
        prev_line = None
        with open(chunk_files[0], 'r') as inf, open(output_file, 'w') as outf:
            for line in inf:
                if line != prev_line:
                    outf.write(line)
                    unique_count += 1
                    prev_line = line
        chunk_files[0].unlink()
        return unique_count

    # Multi-pass merge if needed (when > max_open_files chunks)
    while len(chunk_files) > max_open_files:
        # Merge in batches
        new_chunks = []
        for i in range(0, len(chunk_files), max_open_files):
            batch = chunk_files[i:i+max_open_files]
            merged_path = temp_dir / f"merged_{len(new_chunks):06d}.txt"
            _merge_files(batch, merged_path)
            new_chunks.append(merged_path)
            for f in batch:
                f.unlink()
        chunk_files = new_chunks

    # Final merge with deduplication
    unique_count = _merge_files_with_dedup(chunk_files, output_file)
    for f in chunk_files:
        f.unlink()

    return unique_count


def set_difference_streaming(file_a: Path, file_b: Path, output: Path) -> int:
    """
    Compute set difference (A - B) for sorted files.

    Returns count of lines in output.
    """
    count = 0
    with open(file_a, 'r') as fa, open(file_b, 'r') as fb, open(output, 'w') as out:
        line_a = fa.readline()
        line_b = fb.readline()

        while line_a:
            if not line_b or line_a < line_b:
                out.write(line_a)
                count += 1
                line_a = fa.readline()
            elif line_a > line_b:
                line_b = fb.readline()
            else:  # equal
                line_a = fa.readline()
                line_b = fb.readline()

    return count


def is_valid_sequence(seq: Optional[str]) -> bool:
    """Check if sequence is valid (non-null, non-empty, valid amino acids)."""
    if seq is None or pd.isna(seq) or seq == "" or seq == "nan":
        return False
    if isinstance(seq, float):
        return False
    return all(aa in VALID_AA for aa in str(seq).upper())


def load_foundation_permutations_streaming(
    input_dir: Path,
    temp_dir: Path,
    max_files: Optional[int] = None,
) -> Dict[str, Path]:
    """
    Load and separate sequences by type using disk-based streaming.

    Memory-efficient: writes unique sequences to temp files instead of holding in RAM.

    Args:
        input_dir: Path to foundation_permutations directory
        temp_dir: Directory for temporary files
        max_files: Maximum number of batch files to process (for testing)

    Returns:
        Dict with keys 'tra', 'trb', 'tra_trb' mapping to paths of deduplicated sequence files
    """
    print(f"Loading data from {input_dir} (streaming mode)...")

    # Find all parquet files (try both patterns)
    parquet_files = sorted(input_dir.glob("batch_*.parquet"))
    if not parquet_files:
        parquet_files = sorted(input_dir.glob("*.parquet"))

    if max_files is not None:
        parquet_files = parquet_files[:max_files]
        print(f"  Processing first {max_files} files (--max_files specified)")

    print(f"  Found {len(parquet_files)} parquet files")

    # Ensure temp directory exists
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Open file handles for each permutation type
    temp_files = {
        'tra': temp_dir / "tra_raw.txt",
        'trb': temp_dir / "trb_raw.txt",
        'tra_trb': temp_dir / "tra_trb_raw.txt",
    }

    file_handles = {k: open(v, 'w') for k, v in temp_files.items()}

    try:
        # Stream through files and write sequences to temp files
        for parquet_file in tqdm(parquet_files, desc="  Streaming files"):
            try:
                # Read only the columns we need
                table = pq.read_table(
                    parquet_file,
                    columns=['permutation_key', 'sequence'],
                )

                # Process in chunks to reduce memory
                perm_keys = table.column('permutation_key').to_pylist()
                seqs = table.column('sequence').to_pylist()

                for perm_key, seq in zip(perm_keys, seqs):
                    if perm_key in file_handles and seq:
                        file_handles[perm_key].write(seq + '\n')

                # Free memory
                del table, perm_keys, seqs

            except Exception as e:
                print(f"    Warning: Failed to load {parquet_file}: {e}")
                continue

    finally:
        # Close all file handles
        for fh in file_handles.values():
            fh.close()

    # Deduplicate using Python external sort (memory efficient for large files)
    print("  Deduplicating sequences using Python external sort...")
    dedup_files = {}

    for perm_key, raw_file in temp_files.items():
        dedup_file = temp_dir / f"{perm_key}_dedup.txt"

        if raw_file.stat().st_size > 0:
            # Use Python external sort for disk-based deduplication
            sort_temp_dir = temp_dir / f"{perm_key}_sort_temp"
            count = external_sort_dedup(raw_file, dedup_file, sort_temp_dir)
            print(f"    {perm_key}: {count:,} unique sequences")
            # Clean up sort temp directory
            shutil.rmtree(sort_temp_dir, ignore_errors=True)
        else:
            dedup_file.touch()
            print(f"    {perm_key}: 0 sequences")

        dedup_files[perm_key] = dedup_file

        # Remove raw file to save space
        raw_file.unlink()

    return dedup_files


def load_foundation_permutations(
    input_dir: Path,
    max_files: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load and separate sequences by type from foundation_permutations.

    This is the non-streaming version that loads all data into memory.
    For large datasets (>100 files), use load_foundation_permutations_streaming instead.

    Args:
        input_dir: Path to foundation_permutations directory
        max_files: Maximum number of batch files to process (for testing)

    Returns:
        Dict with keys 'tra', 'trb', 'tra_trb' containing DataFrames with 'sequence' column
    """
    print(f"Loading data from {input_dir}...")

    # Find all parquet files
    parquet_files = sorted(input_dir.glob("batch_*.parquet"))
    if not parquet_files:
        parquet_files = sorted(input_dir.glob("*.parquet"))

    if max_files is not None:
        parquet_files = parquet_files[:max_files]
        print(f"  Processing first {max_files} files (--max_files specified)")

    print(f"  Found {len(parquet_files)} parquet files")

    # Collect sequences by type using sets for deduplication
    sequences: Dict[str, Set[str]] = {
        'tra': set(),
        'trb': set(),
        'tra_trb': set(),
    }

    for parquet_file in tqdm(parquet_files, desc="  Loading files"):
        try:
            table = pq.read_table(
                parquet_file,
                columns=['permutation_key', 'sequence'],
            )

            perm_keys = table.column('permutation_key').to_pylist()
            seqs = table.column('sequence').to_pylist()

            for perm_key, seq in zip(perm_keys, seqs):
                if perm_key in sequences and seq:
                    sequences[perm_key].add(seq)

            del table, perm_keys, seqs

        except Exception as e:
            print(f"    Warning: Failed to load {parquet_file}: {e}")
            continue

    # Convert to DataFrames
    result = {}
    for perm_key, seq_set in sequences.items():
        result[perm_key] = pd.DataFrame({'sequence': list(seq_set)})
        print(f"    {perm_key}: {len(seq_set):,} unique sequences")

    return result


def load_sequences_from_file(file_path: Path) -> List[str]:
    """Load sequences from a text file (one per line)."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        return []

    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_sequences_as_set(file_path: Path) -> Set[str]:
    """Load sequences from a text file into a set."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        return set()

    with open(file_path, 'r') as f:
        return {line.strip() for line in f if line.strip()}


def extract_paired_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and split paired tra_trb sequences.

    Input: DataFrame with column 'sequence' containing 'TRA TRB' space-separated pairs
    Output: DataFrame with columns: tra, trb, tra_trb (original concatenated)

    Args:
        df: DataFrame with tra_trb sequences

    Returns:
        DataFrame with separated tra, trb columns and original tra_trb column
    """
    print("Extracting paired sequences...")

    if len(df) == 0:
        return pd.DataFrame(columns=['tra', 'trb', 'tra_trb'])

    # Use vectorized string split (much faster than iteration)
    sequences = df['sequence'].str.split(' ', n=1, expand=True)

    if sequences.shape[1] < 2:
        print("  Warning: Could not split sequences, expected space-separated TRA TRB")
        return pd.DataFrame(columns=['tra', 'trb', 'tra_trb'])

    result_df = pd.DataFrame({
        'tra': sequences[0],
        'trb': sequences[1],
        'tra_trb': df['sequence'].values,
    })

    # Filter out rows where split failed (None values)
    result_df = result_df.dropna(subset=['tra', 'trb'])
    initial_count = len(result_df)

    # Validate sequences (vectorized check for valid amino acids)
    valid_tra = result_df['tra'].apply(is_valid_sequence)
    valid_trb = result_df['trb'].apply(is_valid_sequence)
    result_df = result_df[valid_tra & valid_trb].copy()

    invalid_count = initial_count - len(result_df)

    # Deduplicate pairs
    before_dedup = len(result_df)
    result_df = result_df.drop_duplicates(subset=['tra', 'trb'], keep='first')

    print(f"  Valid pairs: {len(result_df):,}")
    print(f"  Invalid pairs skipped: {invalid_count:,}")
    print(f"  Duplicates removed: {before_dedup - len(result_df):,}")

    return result_df


def identify_unpaired_sequences(
    tra_df: pd.DataFrame,
    trb_df: pd.DataFrame,
    paired_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find TRA/TRB sequences that don't appear in any paired data.

    Args:
        tra_df: DataFrame with 'sequence' column containing TRA sequences
        trb_df: DataFrame with 'sequence' column containing TRB sequences
        paired_df: DataFrame with 'tra' and 'trb' columns

    Returns:
        unpaired_tra_df: DataFrame with TRA sequences not in any pair
        unpaired_trb_df: DataFrame with TRB sequences not in any pair
    """
    print("Identifying unpaired sequences...")

    # Get unique sequences from paired data
    paired_tra = set(paired_df['tra'].unique())
    paired_trb = set(paired_df['trb'].unique())

    # Filter unpaired TRA sequences
    all_tra = tra_df['sequence'].unique()
    unpaired_tra = [seq for seq in all_tra if seq not in paired_tra and is_valid_sequence(seq)]
    unpaired_tra_df = pd.DataFrame({'sequence': unpaired_tra})
    unpaired_tra_df = unpaired_tra_df.drop_duplicates(subset=['sequence'])

    # Filter unpaired TRB sequences
    all_trb = trb_df['sequence'].unique()
    unpaired_trb = [seq for seq in all_trb if seq not in paired_trb and is_valid_sequence(seq)]
    unpaired_trb_df = pd.DataFrame({'sequence': unpaired_trb})
    unpaired_trb_df = unpaired_trb_df.drop_duplicates(subset=['sequence'])

    print(f"  Total TRA sequences: {len(all_tra):,}")
    print(f"  TRA in pairs: {len(paired_tra):,}")
    print(f"  Unique unpaired TRA: {len(unpaired_tra_df):,}")
    print(f"  Total TRB sequences: {len(all_trb):,}")
    print(f"  TRB in pairs: {len(paired_trb):,}")
    print(f"  Unique unpaired TRB: {len(unpaired_trb_df):,}")

    return unpaired_tra_df, unpaired_trb_df


def extract_paired_sequences_streaming(
    tra_trb_file: Path,
    output_dir: Path,
    chunk_size: int = 100000,
) -> Tuple[Path, Path, Path]:
    """
    Extract and split paired tra_trb sequences using streaming I/O.

    Memory-efficient: processes input file in chunks and writes incrementally to disk.

    Args:
        tra_trb_file: Path to file with tra_trb sequences (one per line)
        output_dir: Directory for output files
        chunk_size: Number of lines to process at a time

    Returns:
        Tuple of (paired_file, tra_file, trb_file) paths
    """
    print("Extracting paired sequences (streaming mode)...")

    output_dir.mkdir(parents=True, exist_ok=True)
    paired_file = output_dir / "paired_raw.txt"
    tra_file = output_dir / "paired_tra.txt"
    trb_file = output_dir / "paired_trb.txt"

    valid_count = 0
    invalid_count = 0

    # Track seen pairs for deduplication
    seen_pairs: Set[str] = set()

    with (
        open(tra_trb_file, 'r') as infile,
        open(paired_file, 'w') as pf,
        open(tra_file, 'w') as traf,
        open(trb_file, 'w') as trbf,
    ):
        for line in tqdm(infile, desc="  Processing pairs"):
            seq = line.strip()
            if not seq:
                continue

            parts = seq.split(' ', 1)
            if len(parts) != 2:
                invalid_count += 1
                continue

            tra, trb = parts
            if not is_valid_sequence(tra) or not is_valid_sequence(trb):
                invalid_count += 1
                continue

            # Deduplicate by pair key
            pair_key = f"{tra}\t{trb}"
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # Write to output files
            pf.write(pair_key + '\n')
            traf.write(tra + '\n')
            trbf.write(trb + '\n')
            valid_count += 1

    # Deduplicate TRA and TRB sequences using Python external sort
    tra_dedup = output_dir / "paired_tra_dedup.txt"
    trb_dedup = output_dir / "paired_trb_dedup.txt"

    print("  Deduplicating TRA sequences...")
    tra_sort_temp = output_dir / "tra_sort_temp"
    tra_count = external_sort_dedup(tra_file, tra_dedup, tra_sort_temp)
    shutil.rmtree(tra_sort_temp, ignore_errors=True)

    print("  Deduplicating TRB sequences...")
    trb_sort_temp = output_dir / "trb_sort_temp"
    trb_count = external_sort_dedup(trb_file, trb_dedup, trb_sort_temp)
    shutil.rmtree(trb_sort_temp, ignore_errors=True)

    print(f"  Valid pairs: {valid_count:,}")
    print(f"  Invalid pairs skipped: {invalid_count:,}")
    print(f"  Unique TRA in pairs: {tra_count:,}")
    print(f"  Unique TRB in pairs: {trb_count:,}")

    # Clean up intermediate files
    tra_file.unlink()
    trb_file.unlink()

    return paired_file, tra_dedup, trb_dedup


def identify_unpaired_sequences_streaming(
    tra_all_file: Path,
    trb_all_file: Path,
    paired_tra_file: Path,
    paired_trb_file: Path,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """
    Find TRA/TRB sequences that don't appear in any paired data.

    Memory-efficient: uses comm command for set difference on sorted files.

    Args:
        tra_all_file: Path to file with all TRA sequences (deduplicated, sorted)
        trb_all_file: Path to file with all TRB sequences (deduplicated, sorted)
        paired_tra_file: Path to file with paired TRA sequences (deduplicated, sorted)
        paired_trb_file: Path to file with paired TRB sequences (deduplicated, sorted)
        output_dir: Directory for output files

    Returns:
        Tuple of (unpaired_tra_file, unpaired_trb_file) paths
    """
    print("Identifying unpaired sequences (streaming mode)...")

    output_dir.mkdir(parents=True, exist_ok=True)
    unpaired_tra_file = output_dir / "unpaired_tra.txt"
    unpaired_trb_file = output_dir / "unpaired_trb.txt"

    # Count input file lines
    def count_lines(path: Path) -> int:
        count = 0
        with open(path, 'r') as f:
            for _ in f:
                count += 1
        return count

    tra_all_count = count_lines(tra_all_file)
    trb_all_count = count_lines(trb_all_file)
    paired_tra_count = count_lines(paired_tra_file)
    paired_trb_count = count_lines(paired_trb_file)

    # Use Python set difference for finding unpaired sequences
    # Both files must be sorted (which they are from external_sort_dedup)
    print("  Computing TRA set difference...")
    unpaired_tra_count = set_difference_streaming(tra_all_file, paired_tra_file, unpaired_tra_file)
    print("  Computing TRB set difference...")
    unpaired_trb_count = set_difference_streaming(trb_all_file, paired_trb_file, unpaired_trb_file)

    print(f"  Total TRA sequences: {tra_all_count:,}")
    print(f"  TRA in pairs: {paired_tra_count:,}")
    print(f"  Unique unpaired TRA: {unpaired_tra_count:,}")
    print(f"  Total TRB sequences: {trb_all_count:,}")
    print(f"  TRB in pairs: {paired_trb_count:,}")
    print(f"  Unique unpaired TRB: {unpaired_trb_count:,}")

    return unpaired_tra_file, unpaired_trb_file


def cluster_sequences_mmseqs2(
    sequences: List[str],
    similarity_threshold: float = 0.9,
    coverage: float = 0.8,
    temp_dir: Optional[Path] = None,
    seq_type: str = "tcr",
    threads: Optional[int] = None,
    gpu: bool = False,
) -> Tuple[Dict[str, int], Dict[int, List[str]]]:
    """
    Cluster sequences using MMseqs2.

    Args:
        sequences: List of sequences to cluster
        similarity_threshold: Sequence identity threshold for clustering
        coverage: Alignment coverage threshold
        temp_dir: Optional temporary directory for MMseqs2 files
        seq_type: Type of sequences for logging
        threads: Number of threads for MMseqs2 (default: all available)
        gpu: Enable GPU acceleration (requires CUDA-enabled MMseqs2)

    Returns:
        seq_to_cluster: Dict mapping sequence -> cluster_id
        cluster_to_seqs: Dict mapping cluster_id -> list of sequences
    """
    if shutil.which("mmseqs") is None:
        raise RuntimeError(
            "MMseqs2 not found in PATH. Install with: conda install -c bioconda mmseqs2"
        )

    # Filter out invalid sequences
    valid_seqs = [s for s in sequences if s and isinstance(s, str) and len(s) > 0]
    valid_seqs = list(set(valid_seqs))  # Deduplicate

    if len(valid_seqs) == 0:
        return {}, {}

    print(f"    Clustering {len(valid_seqs):,} unique {seq_type} sequences at "
          f"{similarity_threshold*100:.0f}% identity...")

    # Create temporary directory for MMseqs2 files
    cleanup_temp = False
    if temp_dir is None:
        temp_dir = Path(tempfile.mkdtemp(prefix=f"mmseqs2_{seq_type}_"))
        cleanup_temp = True
    else:
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Write sequences to FASTA file
        fasta_path = temp_dir / f"{seq_type}.fasta"
        with open(fasta_path, "w") as f:
            for i, seq in enumerate(valid_seqs):
                f.write(f">seq_{i}|{seq}\n{seq}\n")

        # MMseqs2 database and output paths
        db_path = temp_dir / f"{seq_type}_db"
        cluster_db = temp_dir / "cluster_db"
        cluster_tsv = temp_dir / "clusters.tsv"
        tmp_path = temp_dir / "tmp"
        tmp_path.mkdir(exist_ok=True)

        # Create MMseqs2 database
        cmd_createdb = ["mmseqs", "createdb", str(fasta_path), str(db_path)]
        subprocess.run(cmd_createdb, check=True)

        # Run clustering
        cmd_cluster = [
            "mmseqs", "cluster",
            str(db_path), str(cluster_db), str(tmp_path),
            "--min-seq-id", str(similarity_threshold),
            "-c", str(coverage),
            "--cov-mode", "0",
            "-s", "4",
            "--split-memory-limit", "100G",
        ]
        if threads is not None:
            cmd_cluster.extend(["--threads", str(threads)])
        if gpu:
            cmd_cluster.extend(["--gpu", "1"])
        subprocess.run(cmd_cluster, check=True)

        # Convert cluster results to TSV
        cmd_tsv = [
            "mmseqs", "createtsv",
            str(db_path), str(db_path), str(cluster_db), str(cluster_tsv),
        ]
        subprocess.run(cmd_tsv, check=True)

        # Parse cluster results
        seq_to_cluster: Dict[str, int] = {}
        cluster_to_seqs: Dict[int, List[str]] = {}

        cluster_id_map: Dict[str, int] = {}
        next_cluster_id = 0

        with open(cluster_tsv, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    rep_header, member_header = parts[0], parts[1]
                    rep_seq = rep_header.split("|")[-1] if "|" in rep_header else rep_header
                    member_seq = member_header.split("|")[-1] if "|" in member_header else member_header

                    if rep_seq not in cluster_id_map:
                        cluster_id_map[rep_seq] = next_cluster_id
                        next_cluster_id += 1

                    cluster_id = cluster_id_map[rep_seq]
                    seq_to_cluster[member_seq] = cluster_id

                    if cluster_id not in cluster_to_seqs:
                        cluster_to_seqs[cluster_id] = []
                    if member_seq not in cluster_to_seqs[cluster_id]:
                        cluster_to_seqs[cluster_id].append(member_seq)

        # Handle any sequences that might not appear in output
        for seq in valid_seqs:
            if seq not in seq_to_cluster:
                seq_to_cluster[seq] = next_cluster_id
                cluster_to_seqs[next_cluster_id] = [seq]
                next_cluster_id += 1

        print(f"    Created {len(cluster_to_seqs):,} clusters")
        return seq_to_cluster, cluster_to_seqs

    finally:
        if cleanup_temp and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def cluster_sequences_from_file(
    sequences_file: Path,
    output_file: Path,
    similarity_threshold: float = 0.9,
    coverage: float = 0.8,
    seq_type: str = "tcr",
    threads: Optional[int] = None,
    gpu: bool = False,
) -> Tuple[Path, int]:
    """
    Cluster sequences from a file using MMseqs2 (streaming/file-based).

    Memory-efficient: reads sequences from file, writes cluster assignments to file.
    Results are stored on disk as TSV, not loaded into memory.

    Args:
        sequences_file: Path to file with sequences (one per line)
        output_file: Path to output cluster TSV file
        similarity_threshold: Sequence identity threshold for clustering
        coverage: Alignment coverage threshold
        seq_type: Type of sequences for logging
        threads: Number of threads for MMseqs2 (default: all available)
        gpu: Enable GPU acceleration (requires CUDA-enabled MMseqs2)

    Returns:
        Tuple of (output_file path, number of clusters)
    """
    if shutil.which("mmseqs") is None:
        raise RuntimeError(
            "MMseqs2 not found in PATH. Install with: conda install -c bioconda mmseqs2"
        )

    # Count sequences
    seq_count = int(subprocess.run(
        f"wc -l < '{sequences_file}'", shell=True, capture_output=True, text=True
    ).stdout.strip())

    if seq_count == 0:
        # Create empty output file
        output_file.touch()
        return output_file, 0

    print(f"    Clustering {seq_count:,} unique {seq_type} sequences at "
          f"{similarity_threshold*100:.0f}% identity...")

    # Create temporary directory for MMseqs2 files
    temp_dir = Path(tempfile.mkdtemp(prefix=f"mmseqs2_{seq_type}_"))

    try:
        # Convert to FASTA format
        fasta_path = temp_dir / f"{seq_type}.fasta"
        with open(sequences_file, 'r') as infile, open(fasta_path, 'w') as outfile:
            for i, line in enumerate(infile):
                seq = line.strip()
                if seq:
                    outfile.write(f">seq_{i}|{seq}\n{seq}\n")

        # MMseqs2 database and output paths
        db_path = temp_dir / f"{seq_type}_db"
        cluster_db = temp_dir / "cluster_db"
        cluster_tsv = temp_dir / "clusters.tsv"
        tmp_path = temp_dir / "tmp"
        tmp_path.mkdir(exist_ok=True)

        # Create MMseqs2 database
        cmd_createdb = ["mmseqs", "createdb", str(fasta_path), str(db_path)]
        subprocess.run(cmd_createdb, check=True)

        # Run clustering
        cmd_cluster = [
            "mmseqs", "cluster",
            str(db_path), str(cluster_db), str(tmp_path),
            "--min-seq-id", str(similarity_threshold),
            "-c", str(coverage),
            "--cov-mode", "0",
            "-s", "4",
            "--split-memory-limit", "100G",
        ]
        if threads is not None:
            cmd_cluster.extend(["--threads", str(threads)])
        if gpu:
            cmd_cluster.extend(["--gpu", "1"])
        subprocess.run(cmd_cluster, check=True)

        # Convert cluster results to TSV
        cmd_tsv = [
            "mmseqs", "createtsv",
            str(db_path), str(db_path), str(cluster_db), str(cluster_tsv),
        ]
        subprocess.run(cmd_tsv, check=True)

        # Process cluster TSV and write clean output
        # Format: sequence\tcluster_id
        output_file.parent.mkdir(parents=True, exist_ok=True)
        cluster_id_map: Dict[str, int] = {}
        next_cluster_id = 0
        seen_sequences: Set[str] = set()

        with open(cluster_tsv, 'r') as infile, open(output_file, 'w') as outfile:
            outfile.write("sequence\tcluster_id\n")
            for line in infile:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    rep_header, member_header = parts[0], parts[1]
                    rep_seq = rep_header.split("|")[-1] if "|" in rep_header else rep_header
                    member_seq = member_header.split("|")[-1] if "|" in member_header else member_header

                    if rep_seq not in cluster_id_map:
                        cluster_id_map[rep_seq] = next_cluster_id
                        next_cluster_id += 1

                    cluster_id = cluster_id_map[rep_seq]

                    if member_seq not in seen_sequences:
                        outfile.write(f"{member_seq}\t{cluster_id}\n")
                        seen_sequences.add(member_seq)

        # Handle sequences not in cluster output (singletons)
        with open(sequences_file, 'r') as seqfile, open(output_file, 'a') as outfile:
            for line in seqfile:
                seq = line.strip()
                if seq and seq not in seen_sequences:
                    outfile.write(f"{seq}\t{next_cluster_id}\n")
                    next_cluster_id += 1

        num_clusters = next_cluster_id
        print(f"    Created {num_clusters:,} clusters")
        return output_file, num_clusters

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def load_cluster_mapping(cluster_file: Path) -> Dict[str, int]:
    """Load cluster mapping from TSV file into memory (for small files)."""
    mapping: Dict[str, int] = {}
    with open(cluster_file, 'r') as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                seq, cluster_id = parts[0], int(parts[1])
                mapping[seq] = cluster_id
    return mapping


def stream_cluster_mapping(cluster_file: Path):
    """Stream cluster mapping from TSV file (for large files)."""
    with open(cluster_file, 'r') as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                yield parts[0], int(parts[1])


class UnionFind:
    """
    Union-Find data structure for finding connected components.

    Used to group pairs that share TRA or TRB clusters.
    """

    def __init__(self, n: int):
        """Initialize with n elements (0 to n-1)."""
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find root of x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        """Union two sets by rank."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


def find_cluster_components(
    df: pd.DataFrame,
    tra_cluster_col: str = 'tra_cluster',
    trb_cluster_col: str = 'trb_cluster',
) -> Dict[int, Set[int]]:
    """
    Find connected components where pairs share TRA or TRB clusters.

    Uses Union-Find algorithm to group pairs that must stay together
    because they share a cluster in either chain.

    Args:
        df: DataFrame with tra_cluster and trb_cluster columns
        tra_cluster_col: Column name for TRA cluster IDs
        trb_cluster_col: Column name for TRB cluster IDs

    Returns:
        component_to_indices: Dict mapping component ID -> set of row indices
    """
    print("  Finding connected components...")

    n = len(df)
    if n == 0:
        return {}

    uf = UnionFind(n)

    # Build index mappings for clusters
    tra_cluster_to_indices: Dict[int, List[int]] = defaultdict(list)
    trb_cluster_to_indices: Dict[int, List[int]] = defaultdict(list)

    for idx, row in df.iterrows():
        tra_cluster_to_indices[row[tra_cluster_col]].append(idx)
        trb_cluster_to_indices[row[trb_cluster_col]].append(idx)

    # Union indices that share TRA cluster
    print(f"    Processing {len(tra_cluster_to_indices):,} TRA clusters...")
    for indices in tqdm(tra_cluster_to_indices.values(), desc="    TRA clusters", leave=False):
        if len(indices) > 1:
            first = indices[0]
            for other in indices[1:]:
                uf.union(first, other)

    # Union indices that share TRB cluster
    print(f"    Processing {len(trb_cluster_to_indices):,} TRB clusters...")
    for indices in tqdm(trb_cluster_to_indices.values(), desc="    TRB clusters", leave=False):
        if len(indices) > 1:
            first = indices[0]
            for other in indices[1:]:
                uf.union(first, other)

    # Group indices by component
    component_to_indices: Dict[int, Set[int]] = defaultdict(set)
    for idx in range(n):
        root = uf.find(idx)
        component_to_indices[root].add(idx)

    print(f"    Found {len(component_to_indices):,} connected components")

    # Component size distribution
    sizes = [len(indices) for indices in component_to_indices.values()]
    print(f"    Component sizes: min={min(sizes)}, max={max(sizes)}, "
          f"mean={np.mean(sizes):.1f}, median={np.median(sizes):.1f}")
    print(f"    Singletons: {sum(1 for s in sizes if s == 1):,}")

    return component_to_indices


def find_cluster_components_streaming(
    paired_file: Path,
    tra_cluster_file: Path,
    trb_cluster_file: Path,
    output_file: Path,
) -> Tuple[Path, Dict[int, int]]:
    """
    Find connected components for paired sequences using streaming I/O.

    Memory-efficient: Uses cluster IDs (not sequences) for Union-Find,
    which dramatically reduces memory for large datasets.

    Args:
        paired_file: Path to file with paired sequences (TRA\tTRB per line)
        tra_cluster_file: Path to TRA cluster TSV file
        trb_cluster_file: Path to TRB cluster TSV file
        output_file: Path to output file with component assignments

    Returns:
        Tuple of (output_file, component_size_map)
    """
    print("  Finding connected components (streaming mode)...")

    # Load cluster mappings (this is necessary but uses O(unique sequences) memory)
    # For very large datasets, this could be optimized with a disk-based key-value store
    tra_to_cluster = load_cluster_mapping(tra_cluster_file)
    trb_to_cluster = load_cluster_mapping(trb_cluster_file)

    print(f"    Loaded {len(tra_to_cluster):,} TRA cluster mappings")
    print(f"    Loaded {len(trb_to_cluster):,} TRB cluster mappings")

    # First pass: count unique (tra_cluster, trb_cluster) pairs to size Union-Find
    # and build cluster pair to index mapping
    cluster_pair_to_idx: Dict[Tuple[int, int], int] = {}
    next_idx = 0

    with open(paired_file, 'r') as f:
        for line in tqdm(f, desc="    Building cluster pairs"):
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            tra, trb = parts
            tra_cluster = tra_to_cluster.get(tra)
            trb_cluster = trb_to_cluster.get(trb)
            if tra_cluster is None or trb_cluster is None:
                continue

            pair_key = (tra_cluster, trb_cluster)
            if pair_key not in cluster_pair_to_idx:
                cluster_pair_to_idx[pair_key] = next_idx
                next_idx += 1

    print(f"    Found {len(cluster_pair_to_idx):,} unique (TRA cluster, TRB cluster) pairs")

    if next_idx == 0:
        output_file.touch()
        return output_file, {}

    # Build Union-Find on cluster pair indices
    uf = UnionFind(next_idx)

    # Build indices by TRA cluster and TRB cluster
    tra_cluster_to_pair_indices: Dict[int, List[int]] = defaultdict(list)
    trb_cluster_to_pair_indices: Dict[int, List[int]] = defaultdict(list)

    for (tra_cluster, trb_cluster), idx in cluster_pair_to_idx.items():
        tra_cluster_to_pair_indices[tra_cluster].append(idx)
        trb_cluster_to_pair_indices[trb_cluster].append(idx)

    # Union pairs that share TRA cluster
    print(f"    Processing {len(tra_cluster_to_pair_indices):,} TRA clusters...")
    for indices in tqdm(tra_cluster_to_pair_indices.values(), desc="    TRA clusters", leave=False):
        if len(indices) > 1:
            first = indices[0]
            for other in indices[1:]:
                uf.union(first, other)

    # Union pairs that share TRB cluster
    print(f"    Processing {len(trb_cluster_to_pair_indices):,} TRB clusters...")
    for indices in tqdm(trb_cluster_to_pair_indices.values(), desc="    TRB clusters", leave=False):
        if len(indices) > 1:
            first = indices[0]
            for other in indices[1:]:
                uf.union(first, other)

    # Build component assignments for cluster pairs
    pair_idx_to_component: Dict[int, int] = {}
    component_to_pair_indices: Dict[int, Set[int]] = defaultdict(set)

    for idx in range(next_idx):
        root = uf.find(idx)
        pair_idx_to_component[idx] = root
        component_to_pair_indices[root].add(idx)

    print(f"    Found {len(component_to_pair_indices):,} connected components")

    # Component size distribution (in terms of cluster pairs)
    sizes = [len(indices) for indices in component_to_pair_indices.values()]
    print(f"    Component sizes: min={min(sizes)}, max={max(sizes)}, "
          f"mean={np.mean(sizes):.1f}, median={np.median(sizes):.1f}")
    print(f"    Singletons: {sum(1 for s in sizes if s == 1):,}")

    # Second pass: write component assignments for each sequence pair
    output_file.parent.mkdir(parents=True, exist_ok=True)
    component_sample_counts: Dict[int, int] = defaultdict(int)

    with open(paired_file, 'r') as infile, open(output_file, 'w') as outfile:
        outfile.write("tra\ttrb\ttra_cluster\ttrb_cluster\tcomponent\n")
        for line in tqdm(infile, desc="    Writing components"):
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            tra, trb = parts
            tra_cluster = tra_to_cluster.get(tra)
            trb_cluster = trb_to_cluster.get(trb)
            if tra_cluster is None or trb_cluster is None:
                continue

            pair_key = (tra_cluster, trb_cluster)
            pair_idx = cluster_pair_to_idx[pair_key]
            component = pair_idx_to_component[pair_idx]
            component_sample_counts[component] += 1
            outfile.write(f"{tra}\t{trb}\t{tra_cluster}\t{trb_cluster}\t{component}\n")

    return output_file, dict(component_sample_counts)


def cluster_aware_split(
    df: pd.DataFrame,
    cluster_col: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data ensuring no cluster appears in multiple splits.

    For unpaired data: Uses single cluster column.

    Args:
        df: DataFrame with cluster column
        cluster_col: Name of cluster column
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed

    Returns:
        train_df, val_df, test_df
    """
    np.random.seed(seed)

    if len(df) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Get unique clusters and shuffle
    clusters = df[cluster_col].unique().tolist()
    np.random.shuffle(clusters)

    # Calculate split points
    n_clusters = len(clusters)
    train_end = int(n_clusters * train_ratio)
    val_end = train_end + int(n_clusters * val_ratio)

    # Assign clusters to splits
    train_clusters = set(clusters[:train_end])
    val_clusters = set(clusters[train_end:val_end])
    test_clusters = set(clusters[val_end:])

    # Create split DataFrames
    train_df = df[df[cluster_col].isin(train_clusters)].copy()
    val_df = df[df[cluster_col].isin(val_clusters)].copy()
    test_df = df[df[cluster_col].isin(test_clusters)].copy()

    print(f"    Split by {cluster_col}:")
    print(f"      Train: {len(train_df):,} samples, {len(train_clusters):,} clusters")
    print(f"      Val: {len(val_df):,} samples, {len(val_clusters):,} clusters")
    print(f"      Test: {len(test_df):,} samples, {len(test_clusters):,} clusters")

    return train_df, val_df, test_df


def component_aware_split(
    df: pd.DataFrame,
    component_to_indices: Dict[int, Set[int]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data ensuring no connected component appears in multiple splits.

    For paired data: Uses connected components from dual clustering.

    Args:
        df: DataFrame with paired data
        component_to_indices: Dict mapping component ID -> set of row indices
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed

    Returns:
        train_df, val_df, test_df
    """
    np.random.seed(seed)

    if len(df) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Sort components by size (descending) for more balanced splits
    components = list(component_to_indices.items())
    components.sort(key=lambda x: len(x[1]), reverse=True)

    # Calculate target sizes
    n_samples = len(df)
    target_train = int(n_samples * train_ratio)
    target_val = int(n_samples * val_ratio)

    # Greedily assign components to splits
    train_indices: Set[int] = set()
    val_indices: Set[int] = set()
    test_indices: Set[int] = set()

    # Shuffle components with same size
    np.random.shuffle(components)

    for comp_id, indices in components:
        if len(train_indices) < target_train:
            train_indices.update(indices)
        elif len(val_indices) < target_val:
            val_indices.update(indices)
        else:
            test_indices.update(indices)

    # Create split DataFrames
    train_df = df.loc[list(train_indices)].copy() if train_indices else pd.DataFrame()
    val_df = df.loc[list(val_indices)].copy() if val_indices else pd.DataFrame()
    test_df = df.loc[list(test_indices)].copy() if test_indices else pd.DataFrame()

    print(f"    Split by connected components:")
    print(f"      Train: {len(train_df):,} samples")
    print(f"      Val: {len(val_df):,} samples")
    print(f"      Test: {len(test_df):,} samples")

    return train_df, val_df, test_df


def component_aware_split_streaming(
    component_file: Path,
    component_size_map: Dict[int, int],
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Path, Path, Path, Dict[str, int]]:
    """
    Split paired data by connected components using streaming I/O.

    Memory-efficient: streams through input file once, writing to split files.

    Args:
        component_file: Path to file with component assignments
                       (columns: tra, trb, tra_cluster, trb_cluster, component)
        component_size_map: Dict mapping component ID -> sample count
        output_dir: Directory for output split files
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed

    Returns:
        Tuple of (train_file, val_file, test_file, split_stats)
    """
    np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_file = output_dir / "train.tsv"
    val_file = output_dir / "val.tsv"
    test_file = output_dir / "test.tsv"

    if not component_size_map:
        # Create empty files
        for f in [train_file, val_file, test_file]:
            f.touch()
        return train_file, val_file, test_file, {'train': 0, 'val': 0, 'test': 0}

    # Calculate target sizes
    total_samples = sum(component_size_map.values())
    target_train = int(total_samples * train_ratio)
    target_val = int(total_samples * val_ratio)

    # Sort components by size (descending) and shuffle
    components = list(component_size_map.items())
    np.random.shuffle(components)

    # Greedily assign components to splits
    train_components: Set[int] = set()
    val_components: Set[int] = set()
    test_components: Set[int] = set()

    train_count = 0
    val_count = 0

    for comp_id, size in components:
        if train_count < target_train:
            train_components.add(comp_id)
            train_count += size
        elif val_count < target_val:
            val_components.add(comp_id)
            val_count += size
        else:
            test_components.add(comp_id)

    print(f"    Assigned {len(train_components):,} components to train")
    print(f"    Assigned {len(val_components):,} components to val")
    print(f"    Assigned {len(test_components):,} components to test")

    # Stream through component file and write to split files
    split_stats = {'train': 0, 'val': 0, 'test': 0}

    with (
        open(component_file, 'r') as infile,
        open(train_file, 'w') as train_out,
        open(val_file, 'w') as val_out,
        open(test_file, 'w') as test_out,
    ):
        header = infile.readline().strip()
        for out in [train_out, val_out, test_out]:
            out.write(header + '\n')

        for line in tqdm(infile, desc="    Writing splits"):
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue

            component = int(parts[4])

            if component in train_components:
                train_out.write(line)
                split_stats['train'] += 1
            elif component in val_components:
                val_out.write(line)
                split_stats['val'] += 1
            else:
                test_out.write(line)
                split_stats['test'] += 1

    print(f"    Split by connected components:")
    print(f"      Train: {split_stats['train']:,} samples")
    print(f"      Val: {split_stats['val']:,} samples")
    print(f"      Test: {split_stats['test']:,} samples")

    return train_file, val_file, test_file, split_stats


def cluster_aware_split_streaming(
    sequences_file: Path,
    cluster_file: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Path, Path, Path, Dict[str, int]]:
    """
    Split sequences by cluster using streaming I/O.

    Memory-efficient: loads cluster mapping, then streams sequences to split files.

    Args:
        sequences_file: Path to file with sequences (one per line)
        cluster_file: Path to cluster TSV file (sequence, cluster_id)
        output_dir: Directory for output split files
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed

    Returns:
        Tuple of (train_file, val_file, test_file, split_stats)
    """
    np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_file = output_dir / "train.tsv"
    val_file = output_dir / "val.tsv"
    test_file = output_dir / "test.tsv"

    # Load cluster mapping
    seq_to_cluster = load_cluster_mapping(cluster_file)

    if not seq_to_cluster:
        for f in [train_file, val_file, test_file]:
            f.touch()
        return train_file, val_file, test_file, {'train': 0, 'val': 0, 'test': 0}

    # Get unique clusters and shuffle
    clusters = list(set(seq_to_cluster.values()))
    np.random.shuffle(clusters)

    # Calculate split points
    n_clusters = len(clusters)
    train_end = int(n_clusters * train_ratio)
    val_end = train_end + int(n_clusters * val_ratio)

    # Assign clusters to splits
    train_clusters = set(clusters[:train_end])
    val_clusters = set(clusters[train_end:val_end])
    test_clusters = set(clusters[val_end:])

    print(f"    Assigned {len(train_clusters):,} clusters to train")
    print(f"    Assigned {len(val_clusters):,} clusters to val")
    print(f"    Assigned {len(test_clusters):,} clusters to test")

    # Stream through sequences and write to split files
    split_stats = {'train': 0, 'val': 0, 'test': 0}

    with (
        open(sequences_file, 'r') as infile,
        open(train_file, 'w') as train_out,
        open(val_file, 'w') as val_out,
        open(test_file, 'w') as test_out,
    ):
        header = "sequence\tcluster\n"
        for out in [train_out, val_out, test_out]:
            out.write(header)

        for line in tqdm(infile, desc="    Writing splits"):
            seq = line.strip()
            if not seq:
                continue

            cluster = seq_to_cluster.get(seq)
            if cluster is None:
                continue

            output_line = f"{seq}\t{cluster}\n"

            if cluster in train_clusters:
                train_out.write(output_line)
                split_stats['train'] += 1
            elif cluster in val_clusters:
                val_out.write(output_line)
                split_stats['val'] += 1
            else:
                test_out.write(output_line)
                split_stats['test'] += 1

    print(f"    Split by cluster:")
    print(f"      Train: {split_stats['train']:,} samples, {len(train_clusters):,} clusters")
    print(f"      Val: {split_stats['val']:,} samples, {len(val_clusters):,} clusters")
    print(f"      Test: {split_stats['test']:,} samples, {len(test_clusters):,} clusters")

    return train_file, val_file, test_file, split_stats


def tsv_to_parquet(tsv_file: Path, parquet_file: Path, chunk_size: int = 100000) -> int:
    """
    Convert TSV file to Parquet format in chunks.

    Memory-efficient: processes TSV in chunks, writing to parquet incrementally.

    Args:
        tsv_file: Input TSV file path
        parquet_file: Output Parquet file path
        chunk_size: Number of rows to process at a time

    Returns:
        Total number of rows written
    """
    parquet_file.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    writer = None

    try:
        for chunk_df in pd.read_csv(tsv_file, sep='\t', chunksize=chunk_size):
            if len(chunk_df) == 0:
                continue

            table = pa.Table.from_pandas(chunk_df, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(parquet_file, table.schema)

            writer.write_table(table)
            total_rows += len(chunk_df)

    finally:
        if writer is not None:
            writer.close()

    if total_rows == 0:
        # Create empty parquet file with schema
        empty_df = pd.read_csv(tsv_file, sep='\t', nrows=0)
        if len(empty_df.columns) == 0:
            # Couldn't read header, create minimal schema
            empty_df = pd.DataFrame()
        table = pa.Table.from_pandas(empty_df, preserve_index=False)
        pq.write_table(table, parquet_file)

    return total_rows


def verify_no_cluster_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cluster_col: str,
    dataset_name: str,
) -> bool:
    """
    Verify that no cluster appears in multiple splits.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        cluster_col: Name of cluster column
        dataset_name: Name for logging

    Returns:
        True if no leakage, False otherwise
    """
    train_clusters = set(train_df[cluster_col].unique()) if len(train_df) > 0 else set()
    val_clusters = set(val_df[cluster_col].unique()) if len(val_df) > 0 else set()
    test_clusters = set(test_df[cluster_col].unique()) if len(test_df) > 0 else set()

    train_val_overlap = train_clusters & val_clusters
    train_test_overlap = train_clusters & test_clusters
    val_test_overlap = val_clusters & test_clusters

    has_leakage = False

    if train_val_overlap:
        print(f"    ERROR: {dataset_name} {cluster_col} leakage between train and val: "
              f"{len(train_val_overlap)} clusters")
        has_leakage = True

    if train_test_overlap:
        print(f"    ERROR: {dataset_name} {cluster_col} leakage between train and test: "
              f"{len(train_test_overlap)} clusters")
        has_leakage = True

    if val_test_overlap:
        print(f"    ERROR: {dataset_name} {cluster_col} leakage between val and test: "
              f"{len(val_test_overlap)} clusters")
        has_leakage = True

    if not has_leakage:
        print(f"    Verified: No {cluster_col} leakage in {dataset_name}")

    return not has_leakage


def save_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """Save DataFrame to parquet file."""
    if len(df) == 0:
        print(f"    Skipping {output_path} (empty)")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, output_path)
    print(f"    Saved {len(df):,} rows to {output_path}")


def save_clusters(
    output_path: Path,
    seq_to_cluster: Dict[str, int],
    cluster_to_seqs: Dict[int, List[str]],
) -> None:
    """Save cluster assignments to TSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("sequence\tcluster_id\tcluster_size\tis_representative\n")
        for cluster_id, seqs in sorted(cluster_to_seqs.items()):
            for i, seq in enumerate(seqs):
                is_rep = "true" if i == 0 else "false"
                f.write(f"{seq}\t{cluster_id}\t{len(seqs)}\t{is_rep}\n")

    print(f"    Saved clusters to {output_path}")


def collect_statistics(
    paired_df: pd.DataFrame,
    paired_splits: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    unpaired_tra_df: pd.DataFrame,
    unpaired_tra_splits: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    unpaired_trb_df: pd.DataFrame,
    unpaired_trb_splits: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    tra_to_cluster: Dict[str, int],
    tra_clusters: Dict[int, List[str]],
    trb_to_cluster: Dict[str, int],
    trb_clusters: Dict[int, List[str]],
    unpaired_tra_to_cluster: Dict[str, int],
    unpaired_tra_clusters: Dict[int, List[str]],
    unpaired_trb_to_cluster: Dict[str, int],
    unpaired_trb_clusters: Dict[int, List[str]],
) -> Dict:
    """Collect comprehensive statistics for the dataset."""
    stats = {
        'paired': {
            'total': len(paired_df),
            'unique_tra': paired_df['tra'].nunique() if len(paired_df) > 0 else 0,
            'unique_trb': paired_df['trb'].nunique() if len(paired_df) > 0 else 0,
            'tra_clustering': {
                'num_clusters': len(tra_clusters),
                'num_sequences': len(tra_to_cluster),
            },
            'trb_clustering': {
                'num_clusters': len(trb_clusters),
                'num_sequences': len(trb_to_cluster),
            },
            'splits': {},
        },
        'unpaired_tra': {
            'total': len(unpaired_tra_df),
            'clustering': {
                'num_clusters': len(unpaired_tra_clusters),
                'num_sequences': len(unpaired_tra_to_cluster),
            },
            'splits': {},
        },
        'unpaired_trb': {
            'total': len(unpaired_trb_df),
            'clustering': {
                'num_clusters': len(unpaired_trb_clusters),
                'num_sequences': len(unpaired_trb_to_cluster),
            },
            'splits': {},
        },
    }

    # Cluster size distributions
    for name, clusters in [
        ('paired_tra', tra_clusters),
        ('paired_trb', trb_clusters),
        ('unpaired_tra', unpaired_tra_clusters),
        ('unpaired_trb', unpaired_trb_clusters),
    ]:
        sizes = [len(seqs) for seqs in clusters.values()]
        if sizes:
            if name.startswith('paired_'):
                chain = name.split('_')[1]
                stats['paired'][f'{chain}_clustering']['size_distribution'] = {
                    'min': min(sizes),
                    'max': max(sizes),
                    'mean': float(np.mean(sizes)),
                    'median': float(np.median(sizes)),
                    'singletons': sum(1 for s in sizes if s == 1),
                }
            else:
                chain = name.split('_')[1]
                stats[f'unpaired_{chain}']['clustering']['size_distribution'] = {
                    'min': min(sizes),
                    'max': max(sizes),
                    'mean': float(np.mean(sizes)),
                    'median': float(np.median(sizes)),
                    'singletons': sum(1 for s in sizes if s == 1),
                }

    # Split statistics
    split_names = ['train', 'val', 'test']

    for split_name, split_df in zip(split_names, paired_splits):
        stats['paired']['splits'][split_name] = {
            'samples': len(split_df),
            'unique_tra': split_df['tra'].nunique() if len(split_df) > 0 else 0,
            'unique_trb': split_df['trb'].nunique() if len(split_df) > 0 else 0,
            'tra_clusters': split_df['tra_cluster'].nunique() if len(split_df) > 0 else 0,
            'trb_clusters': split_df['trb_cluster'].nunique() if len(split_df) > 0 else 0,
        }

    for split_name, split_df in zip(split_names, unpaired_tra_splits):
        stats['unpaired_tra']['splits'][split_name] = {
            'samples': len(split_df),
            'clusters': split_df['cluster'].nunique() if len(split_df) > 0 and 'cluster' in split_df.columns else 0,
        }

    for split_name, split_df in zip(split_names, unpaired_trb_splits):
        stats['unpaired_trb']['splits'][split_name] = {
            'samples': len(split_df),
            'clusters': split_df['cluster'].nunique() if len(split_df) > 0 and 'cluster' in split_df.columns else 0,
        }

    return stats


def write_statistics_doc(output_dir: Path, stats: Dict, config: Dict) -> None:
    """Write dataset statistics to markdown documentation file."""
    doc_path = output_dir / "dataset_statistics.md"
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# TCR Contrastive Learning Dataset Statistics")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Configuration section
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Input Directory | `{config['input_dir']}` |")
    lines.append(f"| Output Directory | `{config['output_dir']}` |")
    lines.append(f"| Similarity Threshold | {config['similarity_threshold']} |")
    lines.append(f"| Coverage | {config['coverage']} |")
    lines.append(f"| Train Ratio | {config['train_ratio']} |")
    lines.append(f"| Val Ratio | {config['val_ratio']} |")
    lines.append(f"| Test Ratio | {config['test_ratio']} |")
    lines.append(f"| Random Seed | {config['seed']} |")
    if config.get('max_files'):
        lines.append(f"| Max Files | {config['max_files']} |")
    lines.append("")

    # Paired data section
    lines.append("## Paired TRA-TRB Data")
    lines.append("")
    paired = stats['paired']
    lines.append(f"- **Total pairs:** {paired['total']:,}")
    lines.append(f"- **Unique TRA sequences:** {paired['unique_tra']:,}")
    lines.append(f"- **Unique TRB sequences:** {paired['unique_trb']:,}")
    lines.append("")

    # TRA clustering
    lines.append("### TRA Clustering")
    lines.append("")
    tra_clust = paired['tra_clustering']
    lines.append(f"- **Total sequences:** {tra_clust['num_sequences']:,}")
    lines.append(f"- **Total clusters:** {tra_clust['num_clusters']:,}")
    if 'size_distribution' in tra_clust:
        dist = tra_clust['size_distribution']
        lines.append(f"- **Cluster sizes:** min={dist['min']}, max={dist['max']}, "
                     f"mean={dist['mean']:.1f}, median={dist['median']:.1f}")
        lines.append(f"- **Singleton clusters:** {dist['singletons']:,}")
    lines.append("")

    # TRB clustering
    lines.append("### TRB Clustering")
    lines.append("")
    trb_clust = paired['trb_clustering']
    lines.append(f"- **Total sequences:** {trb_clust['num_sequences']:,}")
    lines.append(f"- **Total clusters:** {trb_clust['num_clusters']:,}")
    if 'size_distribution' in trb_clust:
        dist = trb_clust['size_distribution']
        lines.append(f"- **Cluster sizes:** min={dist['min']}, max={dist['max']}, "
                     f"mean={dist['mean']:.1f}, median={dist['median']:.1f}")
        lines.append(f"- **Singleton clusters:** {dist['singletons']:,}")
    lines.append("")

    # Paired splits
    lines.append("### Split Statistics")
    lines.append("")
    lines.append("| Split | Samples | Unique TRA | Unique TRB | TRA Clusters | TRB Clusters |")
    lines.append("|-------|---------|------------|------------|--------------|--------------|")
    for split_name in ['train', 'val', 'test']:
        split_stats = paired['splits'].get(split_name, {})
        lines.append(f"| {split_name} | {split_stats.get('samples', 0):,} | "
                     f"{split_stats.get('unique_tra', 0):,} | "
                     f"{split_stats.get('unique_trb', 0):,} | "
                     f"{split_stats.get('tra_clusters', 0):,} | "
                     f"{split_stats.get('trb_clusters', 0):,} |")
    lines.append("")

    # Unpaired TRA section
    lines.append("## Unpaired TRA Data")
    lines.append("")
    unpaired_tra = stats['unpaired_tra']
    lines.append(f"- **Total sequences:** {unpaired_tra['total']:,}")
    if 'clustering' in unpaired_tra:
        clust = unpaired_tra['clustering']
        lines.append(f"- **Total clusters:** {clust['num_clusters']:,}")
        if 'size_distribution' in clust:
            dist = clust['size_distribution']
            lines.append(f"- **Cluster sizes:** min={dist['min']}, max={dist['max']}, "
                         f"mean={dist['mean']:.1f}, median={dist['median']:.1f}")
            lines.append(f"- **Singleton clusters:** {dist['singletons']:,}")
    lines.append("")

    lines.append("### Split Statistics")
    lines.append("")
    lines.append("| Split | Samples | Clusters |")
    lines.append("|-------|---------|----------|")
    for split_name in ['train', 'val', 'test']:
        split_stats = unpaired_tra['splits'].get(split_name, {})
        lines.append(f"| {split_name} | {split_stats.get('samples', 0):,} | "
                     f"{split_stats.get('clusters', 0):,} |")
    lines.append("")

    # Unpaired TRB section
    lines.append("## Unpaired TRB Data")
    lines.append("")
    unpaired_trb = stats['unpaired_trb']
    lines.append(f"- **Total sequences:** {unpaired_trb['total']:,}")
    if 'clustering' in unpaired_trb:
        clust = unpaired_trb['clustering']
        lines.append(f"- **Total clusters:** {clust['num_clusters']:,}")
        if 'size_distribution' in clust:
            dist = clust['size_distribution']
            lines.append(f"- **Cluster sizes:** min={dist['min']}, max={dist['max']}, "
                         f"mean={dist['mean']:.1f}, median={dist['median']:.1f}")
            lines.append(f"- **Singleton clusters:** {dist['singletons']:,}")
    lines.append("")

    lines.append("### Split Statistics")
    lines.append("")
    lines.append("| Split | Samples | Clusters |")
    lines.append("|-------|---------|----------|")
    for split_name in ['train', 'val', 'test']:
        split_stats = unpaired_trb['splits'].get(split_name, {})
        lines.append(f"| {split_name} | {split_stats.get('samples', 0):,} | "
                     f"{split_stats.get('clusters', 0):,} |")
    lines.append("")

    # Verification section
    lines.append("## Verification Checklist")
    lines.append("")
    lines.append("After running, the script verifies:")
    lines.append("")
    lines.append("1. **Paired data TRA cluster leakage**: No TRA cluster appears in multiple splits")
    lines.append("2. **Paired data TRB cluster leakage**: No TRB cluster appears in multiple splits")
    lines.append("3. **Unpaired TRA cluster leakage**: No cluster appears in multiple splits")
    lines.append("4. **Unpaired TRB cluster leakage**: No cluster appears in multiple splits")
    lines.append("5. **Unpaired completeness**: All TRA/TRB sequences are either in paired or unpaired output")
    lines.append("6. **No paired/unpaired overlap**: Sequences in unpaired sets don't appear in paired sets")
    lines.append("")

    # Write to file
    with open(doc_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n  Statistics documentation written to: {doc_path}")


def write_statistics_streaming(
    output_dir: Path,
    config: Dict,
    paired_stats: Dict[str, int],
    unpaired_tra_stats: Dict[str, int],
    unpaired_trb_stats: Dict[str, int],
    tra_cluster_count: int,
    trb_cluster_count: int,
    unpaired_tra_cluster_count: int,
    unpaired_trb_cluster_count: int,
) -> None:
    """Write simplified statistics for streaming mode."""
    doc_path = output_dir / "dataset_statistics.md"
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# TCR Contrastive Learning Dataset Statistics")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("(Streaming mode - simplified statistics)")
    lines.append("")

    # Configuration section
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Input Directory | `{config['input_dir']}` |")
    lines.append(f"| Output Directory | `{config['output_dir']}` |")
    lines.append(f"| Similarity Threshold | {config['similarity_threshold']} |")
    lines.append(f"| Coverage | {config['coverage']} |")
    lines.append(f"| Train Ratio | {config['train_ratio']} |")
    lines.append(f"| Val Ratio | {config['val_ratio']} |")
    lines.append(f"| Test Ratio | {config['test_ratio']} |")
    lines.append(f"| Random Seed | {config['seed']} |")
    lines.append("| Mode | Streaming (memory-efficient) |")
    if config.get('max_files'):
        lines.append(f"| Max Files | {config['max_files']} |")
    lines.append("")

    # Paired data section
    lines.append("## Paired TRA-TRB Data")
    lines.append("")
    total_paired = paired_stats['train'] + paired_stats['val'] + paired_stats['test']
    lines.append(f"- **Total pairs:** {total_paired:,}")
    lines.append(f"- **TRA clusters:** {tra_cluster_count:,}")
    lines.append(f"- **TRB clusters:** {trb_cluster_count:,}")
    lines.append("")

    lines.append("### Split Statistics")
    lines.append("")
    lines.append("| Split | Samples |")
    lines.append("|-------|---------|")
    for split_name in ['train', 'val', 'test']:
        lines.append(f"| {split_name} | {paired_stats.get(split_name, 0):,} |")
    lines.append("")

    # Unpaired TRA section
    lines.append("## Unpaired TRA Data")
    lines.append("")
    total_tra = unpaired_tra_stats['train'] + unpaired_tra_stats['val'] + unpaired_tra_stats['test']
    lines.append(f"- **Total sequences:** {total_tra:,}")
    lines.append(f"- **Total clusters:** {unpaired_tra_cluster_count:,}")
    lines.append("")

    lines.append("### Split Statistics")
    lines.append("")
    lines.append("| Split | Samples |")
    lines.append("|-------|---------|")
    for split_name in ['train', 'val', 'test']:
        lines.append(f"| {split_name} | {unpaired_tra_stats.get(split_name, 0):,} |")
    lines.append("")

    # Unpaired TRB section
    lines.append("## Unpaired TRB Data")
    lines.append("")
    total_trb = unpaired_trb_stats['train'] + unpaired_trb_stats['val'] + unpaired_trb_stats['test']
    lines.append(f"- **Total sequences:** {total_trb:,}")
    lines.append(f"- **Total clusters:** {unpaired_trb_cluster_count:,}")
    lines.append("")

    lines.append("### Split Statistics")
    lines.append("")
    lines.append("| Split | Samples |")
    lines.append("|-------|---------|")
    for split_name in ['train', 'val', 'test']:
        lines.append(f"| {split_name} | {unpaired_trb_stats.get(split_name, 0):,} |")
    lines.append("")

    # Write to file
    with open(doc_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n  Statistics documentation written to: {doc_path}")


def main_streaming(args):
    """
    Memory-efficient streaming version of the main processing pipeline.

    Uses disk-based operations instead of holding all data in memory.
    """
    print("=" * 70)
    print("TCR Contrastive Learning Dataset Creation (STREAMING MODE)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Input directory: {args.input_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Train ratio: {args.train_ratio}")
    print(f"  Val ratio: {args.val_ratio}")
    print(f"  Test ratio: {args.test_ratio}")
    print(f"  Random seed: {args.seed}")
    print(f"  Similarity threshold: {args.similarity_threshold}")
    print(f"  Coverage: {args.coverage}")
    if args.max_files:
        print(f"  Max files: {args.max_files}")
    print(f"  Mode: STREAMING (memory-efficient)")

    np.random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create temp directory for intermediate files
    if args.temp_dir:
        temp_dir = Path(args.temp_dir)
        print(f"  Temp directory: {temp_dir}")
    else:
        temp_dir = output_dir / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Phase 1: Load and Prepare Data (streaming)
        print("\n" + "=" * 70)
        print("Phase 1: Load and Prepare Data (streaming)")
        print("=" * 70)

        # Step 1: Stream data to temp files
        dedup_files = load_foundation_permutations_streaming(
            input_dir, temp_dir, max_files=args.max_files
        )

        # Step 2: Extract paired sequences (streaming)
        print("\n" + "-" * 50)
        paired_file, paired_tra_file, paired_trb_file = extract_paired_sequences_streaming(
            dedup_files['tra_trb'],
            temp_dir,
        )

        # Step 3: Identify unpaired sequences (streaming)
        print("\n" + "-" * 50)
        unpaired_tra_file, unpaired_trb_file = identify_unpaired_sequences_streaming(
            dedup_files['tra'],
            dedup_files['trb'],
            paired_tra_file,
            paired_trb_file,
            temp_dir,
        )

        # Phase 2: Cluster Paired Sequences (streaming)
        print("\n" + "=" * 70)
        print("Phase 2: Cluster Paired Sequences (streaming)")
        print("=" * 70)

        # Step 4-5: Cluster TRA and TRB sequences from pairs in parallel
        print("\nClustering TRA and TRB sequences from pairs in parallel...")
        tra_cluster_file = temp_dir / "paired_tra_clusters.tsv"
        trb_cluster_file = temp_dir / "paired_trb_clusters.tsv"

        # Split threads between two parallel jobs
        threads_per_job = args.threads // 2 if args.threads else None

        with ProcessPoolExecutor(max_workers=2) as executor:
            tra_future = executor.submit(
                cluster_sequences_from_file,
                paired_tra_file,
                tra_cluster_file,
                args.similarity_threshold,
                args.coverage,
                "paired_tra",
                threads_per_job,
                args.gpu,
            )
            trb_future = executor.submit(
                cluster_sequences_from_file,
                paired_trb_file,
                trb_cluster_file,
                args.similarity_threshold,
                args.coverage,
                "paired_trb",
                threads_per_job,
                args.gpu,
            )
            _, tra_cluster_count = tra_future.result()
            _, trb_cluster_count = trb_future.result()

        # Step 6-7: Find connected components (streaming)
        print("\n" + "-" * 50)
        component_file = temp_dir / "paired_components.tsv"
        _, component_size_map = find_cluster_components_streaming(
            paired_file,
            tra_cluster_file,
            trb_cluster_file,
            component_file,
        )

        # Step 8: Split by components (streaming)
        print("\n" + "-" * 50)
        print("Splitting paired data by connected components...")
        paired_train_tsv, paired_val_tsv, paired_test_tsv, paired_stats = component_aware_split_streaming(
            component_file,
            component_size_map,
            temp_dir / "paired_splits",
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

        # Phase 3: Process Unpaired Sequences (streaming)
        print("\n" + "=" * 70)
        print("Phase 3: Process Unpaired Sequences (streaming)")
        print("=" * 70)

        # Step 9-13: Cluster unpaired TRA and TRB in parallel, then split
        print("\nClustering unpaired TRA and TRB sequences in parallel...")
        unpaired_tra_cluster_file = temp_dir / "unpaired_tra_clusters.tsv"
        unpaired_trb_cluster_file = temp_dir / "unpaired_trb_clusters.tsv"
        unpaired_tra_stats = {'train': 0, 'val': 0, 'test': 0}
        unpaired_trb_stats = {'train': 0, 'val': 0, 'test': 0}
        unpaired_tra_cluster_count = 0
        unpaired_trb_cluster_count = 0

        tra_exists = unpaired_tra_file.exists() and unpaired_tra_file.stat().st_size > 0
        trb_exists = unpaired_trb_file.exists() and unpaired_trb_file.stat().st_size > 0

        # Run clustering in parallel
        if tra_exists or trb_exists:
            with ProcessPoolExecutor(max_workers=2) as executor:
                futures = {}
                if tra_exists:
                    futures['tra'] = executor.submit(
                        cluster_sequences_from_file,
                        unpaired_tra_file,
                        unpaired_tra_cluster_file,
                        args.similarity_threshold,
                        args.coverage,
                        "unpaired_tra",
                        threads_per_job,
                        args.gpu,
                    )
                if trb_exists:
                    futures['trb'] = executor.submit(
                        cluster_sequences_from_file,
                        unpaired_trb_file,
                        unpaired_trb_cluster_file,
                        args.similarity_threshold,
                        args.coverage,
                        "unpaired_trb",
                        threads_per_job,
                        args.gpu,
                    )

                if 'tra' in futures:
                    _, unpaired_tra_cluster_count = futures['tra'].result()
                if 'trb' in futures:
                    _, unpaired_trb_cluster_count = futures['trb'].result()

        # Split unpaired TRA (sequential - depends on clustering)
        if tra_exists:
            print("\nSplitting unpaired TRA sequences...")
            _, _, _, unpaired_tra_stats = cluster_aware_split_streaming(
                unpaired_tra_file,
                unpaired_tra_cluster_file,
                temp_dir / "unpaired_tra_splits",
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
            )

        # Split unpaired TRB (sequential - depends on clustering)
        if trb_exists:
            print("\nSplitting unpaired TRB sequences...")
            _, _, _, unpaired_trb_stats = cluster_aware_split_streaming(
                unpaired_trb_file,
                unpaired_trb_cluster_file,
                temp_dir / "unpaired_trb_splits",
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
            )

        # Phase 4: Convert to Parquet and Save
        print("\n" + "=" * 70)
        print("Phase 4: Convert to Parquet and Save")
        print("=" * 70)

        # Convert paired TSV splits to Parquet
        print("\nConverting paired data to parquet...")
        paired_output_dir = output_dir / "paired"
        paired_output_dir.mkdir(parents=True, exist_ok=True)

        tsv_to_parquet(paired_train_tsv, paired_output_dir / "train.parquet")
        tsv_to_parquet(paired_val_tsv, paired_output_dir / "val.parquet")
        tsv_to_parquet(paired_test_tsv, paired_output_dir / "test.parquet")

        # Copy cluster files
        shutil.copy(tra_cluster_file, paired_output_dir / "tra_clusters.tsv")
        shutil.copy(trb_cluster_file, paired_output_dir / "trb_clusters.tsv")

        # Convert unpaired TRA splits to Parquet
        if unpaired_tra_stats['train'] + unpaired_tra_stats['val'] + unpaired_tra_stats['test'] > 0:
            print("\nConverting unpaired TRA data to parquet...")
            unpaired_tra_output_dir = output_dir / "unpaired_tra"
            unpaired_tra_output_dir.mkdir(parents=True, exist_ok=True)

            tsv_to_parquet(
                temp_dir / "unpaired_tra_splits" / "train.tsv",
                unpaired_tra_output_dir / "train.parquet"
            )
            tsv_to_parquet(
                temp_dir / "unpaired_tra_splits" / "val.tsv",
                unpaired_tra_output_dir / "val.parquet"
            )
            tsv_to_parquet(
                temp_dir / "unpaired_tra_splits" / "test.tsv",
                unpaired_tra_output_dir / "test.parquet"
            )
            shutil.copy(unpaired_tra_cluster_file, unpaired_tra_output_dir / "clusters.tsv")

        # Convert unpaired TRB splits to Parquet
        if unpaired_trb_stats['train'] + unpaired_trb_stats['val'] + unpaired_trb_stats['test'] > 0:
            print("\nConverting unpaired TRB data to parquet...")
            unpaired_trb_output_dir = output_dir / "unpaired_trb"
            unpaired_trb_output_dir.mkdir(parents=True, exist_ok=True)

            tsv_to_parquet(
                temp_dir / "unpaired_trb_splits" / "train.tsv",
                unpaired_trb_output_dir / "train.parquet"
            )
            tsv_to_parquet(
                temp_dir / "unpaired_trb_splits" / "val.tsv",
                unpaired_trb_output_dir / "val.parquet"
            )
            tsv_to_parquet(
                temp_dir / "unpaired_trb_splits" / "test.tsv",
                unpaired_trb_output_dir / "test.parquet"
            )
            shutil.copy(unpaired_trb_cluster_file, unpaired_trb_output_dir / "clusters.tsv")

        # Write statistics
        config = {
            'input_dir': args.input_dir,
            'output_dir': args.output_dir,
            'similarity_threshold': args.similarity_threshold,
            'coverage': args.coverage,
            'train_ratio': args.train_ratio,
            'val_ratio': args.val_ratio,
            'test_ratio': args.test_ratio,
            'seed': args.seed,
            'max_files': args.max_files,
        }

        write_statistics_streaming(
            output_dir,
            config,
            paired_stats,
            unpaired_tra_stats,
            unpaired_trb_stats,
            tra_cluster_count,
            trb_cluster_count,
            unpaired_tra_cluster_count,
            unpaired_trb_cluster_count,
        )

        # Summary
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"\nOutput directory: {output_dir}")
        print(f"\nPaired data:")
        print(f"  Total: {paired_stats['train'] + paired_stats['val'] + paired_stats['test']:,}")
        print(f"  Train: {paired_stats['train']:,}")
        print(f"  Val: {paired_stats['val']:,}")
        print(f"  Test: {paired_stats['test']:,}")
        print(f"\nUnpaired TRA:")
        print(f"  Total: {unpaired_tra_stats['train'] + unpaired_tra_stats['val'] + unpaired_tra_stats['test']:,}")
        print(f"  Train: {unpaired_tra_stats['train']:,}")
        print(f"  Val: {unpaired_tra_stats['val']:,}")
        print(f"  Test: {unpaired_tra_stats['test']:,}")
        print(f"\nUnpaired TRB:")
        print(f"  Total: {unpaired_trb_stats['train'] + unpaired_trb_stats['val'] + unpaired_trb_stats['test']:,}")
        print(f"  Train: {unpaired_trb_stats['train']:,}")
        print(f"  Val: {unpaired_trb_stats['val']:,}")
        print(f"  Test: {unpaired_trb_stats['test']:,}")

    finally:
        # Clean up temp directory
        if temp_dir.exists() and not args.keep_temp:
            print(f"\nCleaning up temp directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Create cluster-aware TCR sequence splits for contrastive learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to foundation_permutations directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for splits"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of data for validation (default: 0.1)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Fraction of data for testing (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.9,
        help="Sequence clustering threshold (default: 0.9)"
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.8,
        help="MMseqs2 coverage threshold (default: 0.8)"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Max number of batch files to process (for testing)"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use memory-efficient streaming mode (recommended for large datasets)"
    )
    parser.add_argument(
        "--keep_temp",
        action="store_true",
        help="Keep temporary files after processing (for debugging)"
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default=None,
        help="Directory for temporary files (default: output_dir/_temp)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of threads for MMseqs2 clustering (default: all available)"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration for MMseqs2 clustering (requires CUDA-enabled MMseqs2)"
    )

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        parser.error(f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total_ratio}")

    # Use streaming mode if requested
    if args.streaming:
        main_streaming(args)
        return

    print("=" * 70)
    print("TCR Contrastive Learning Dataset Creation")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Input directory: {args.input_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Train ratio: {args.train_ratio}")
    print(f"  Val ratio: {args.val_ratio}")
    print(f"  Test ratio: {args.test_ratio}")
    print(f"  Random seed: {args.seed}")
    print(f"  Similarity threshold: {args.similarity_threshold}")
    print(f"  Coverage: {args.coverage}")
    if args.max_files:
        print(f"  Max files: {args.max_files}")

    # Set random seed
    np.random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Phase 1: Load and Prepare Data
    print("\n" + "=" * 70)
    print("Phase 1: Load and Prepare Data")
    print("=" * 70)

    # Step 1: Load data
    data = load_foundation_permutations(input_dir, max_files=args.max_files)

    # Step 2: Extract paired sequences
    print("\n" + "-" * 50)
    paired_df = extract_paired_sequences(data['tra_trb'])

    # Step 3: Identify unpaired sequences
    print("\n" + "-" * 50)
    unpaired_tra_df, unpaired_trb_df = identify_unpaired_sequences(
        data['tra'], data['trb'], paired_df
    )

    # Phase 2: Cluster Paired Sequences
    print("\n" + "=" * 70)
    print("Phase 2: Cluster Paired Sequences")
    print("=" * 70)

    # Step 4-5: Cluster TRA and TRB sequences from pairs in parallel
    print("\nClustering TRA and TRB sequences from pairs in parallel...")
    tra_sequences = paired_df['tra'].unique().tolist()
    trb_sequences = paired_df['trb'].unique().tolist()

    # Split threads between two parallel jobs
    threads_per_job = args.threads // 2 if args.threads else None

    with ProcessPoolExecutor(max_workers=2) as executor:
        tra_future = executor.submit(
            cluster_sequences_mmseqs2,
            tra_sequences,
            args.similarity_threshold,
            args.coverage,
            "paired_tra",
            threads_per_job,
            args.gpu,
        )
        trb_future = executor.submit(
            cluster_sequences_mmseqs2,
            trb_sequences,
            args.similarity_threshold,
            args.coverage,
            "paired_trb",
            threads_per_job,
            args.gpu,
        )
        tra_to_cluster, tra_clusters = tra_future.result()
        trb_to_cluster, trb_clusters = trb_future.result()

    # Add cluster columns to paired_df
    paired_df = paired_df.reset_index(drop=True)
    paired_df['tra_cluster'] = paired_df['tra'].map(tra_to_cluster)
    paired_df['trb_cluster'] = paired_df['trb'].map(trb_to_cluster)

    # Step 6-7: Find connected components
    print("\n" + "-" * 50)
    component_to_indices = find_cluster_components(
        paired_df,
        tra_cluster_col='tra_cluster',
        trb_cluster_col='trb_cluster',
    )

    # Step 8: Split by components
    print("\n" + "-" * 50)
    print("Splitting paired data by connected components...")
    paired_train, paired_val, paired_test = component_aware_split(
        paired_df,
        component_to_indices,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # Phase 3: Process Unpaired Sequences
    print("\n" + "=" * 70)
    print("Phase 3: Process Unpaired Sequences")
    print("=" * 70)

    # Initialize results
    unpaired_tra_to_cluster: Dict[str, int] = {}
    unpaired_tra_clusters: Dict[int, List[str]] = {}
    unpaired_tra_train = pd.DataFrame()
    unpaired_tra_val = pd.DataFrame()
    unpaired_tra_test = pd.DataFrame()
    unpaired_trb_to_cluster: Dict[str, int] = {}
    unpaired_trb_clusters: Dict[int, List[str]] = {}
    unpaired_trb_train = pd.DataFrame()
    unpaired_trb_val = pd.DataFrame()
    unpaired_trb_test = pd.DataFrame()

    # Step 9-13: Cluster unpaired TRA and TRB in parallel, then split
    has_unpaired_tra = len(unpaired_tra_df) > 0
    has_unpaired_trb = len(unpaired_trb_df) > 0

    if has_unpaired_tra or has_unpaired_trb:
        print("\nClustering unpaired TRA and TRB sequences in parallel...")
        unpaired_tra_seqs = unpaired_tra_df['sequence'].unique().tolist() if has_unpaired_tra else []
        unpaired_trb_seqs = unpaired_trb_df['sequence'].unique().tolist() if has_unpaired_trb else []

        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = {}
            if has_unpaired_tra:
                futures['tra'] = executor.submit(
                    cluster_sequences_mmseqs2,
                    unpaired_tra_seqs,
                    args.similarity_threshold,
                    args.coverage,
                    "unpaired_tra",
                    threads_per_job,
                    args.gpu,
                )
            if has_unpaired_trb:
                futures['trb'] = executor.submit(
                    cluster_sequences_mmseqs2,
                    unpaired_trb_seqs,
                    args.similarity_threshold,
                    args.coverage,
                    "unpaired_trb",
                    threads_per_job,
                    args.gpu,
                )

            if 'tra' in futures:
                unpaired_tra_to_cluster, unpaired_tra_clusters = futures['tra'].result()
            if 'trb' in futures:
                unpaired_trb_to_cluster, unpaired_trb_clusters = futures['trb'].result()

    # Split unpaired TRA (sequential - depends on clustering)
    if has_unpaired_tra:
        print("\nSplitting unpaired TRA sequences...")
        unpaired_tra_df = unpaired_tra_df.copy()
        unpaired_tra_df['cluster'] = unpaired_tra_df['sequence'].map(unpaired_tra_to_cluster)

        unpaired_tra_train, unpaired_tra_val, unpaired_tra_test = cluster_aware_split(
            unpaired_tra_df,
            cluster_col='cluster',
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    # Split unpaired TRB (sequential - depends on clustering)
    if has_unpaired_trb:
        print("\nSplitting unpaired TRB sequences...")
        unpaired_trb_df = unpaired_trb_df.copy()
        unpaired_trb_df['cluster'] = unpaired_trb_df['sequence'].map(unpaired_trb_to_cluster)

        unpaired_trb_train, unpaired_trb_val, unpaired_trb_test = cluster_aware_split(
            unpaired_trb_df,
            cluster_col='cluster',
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    # Phase 4: Verify, Save and Report
    print("\n" + "=" * 70)
    print("Phase 4: Verify, Save and Report")
    print("=" * 70)

    # Verify no cluster leakage
    print("\nVerifying no cluster leakage...")
    all_verified = True

    if len(paired_train) > 0 or len(paired_val) > 0 or len(paired_test) > 0:
        if not verify_no_cluster_leakage(
            paired_train, paired_val, paired_test, 'tra_cluster', 'paired'
        ):
            all_verified = False
        if not verify_no_cluster_leakage(
            paired_train, paired_val, paired_test, 'trb_cluster', 'paired'
        ):
            all_verified = False

    if len(unpaired_tra_train) > 0 or len(unpaired_tra_val) > 0 or len(unpaired_tra_test) > 0:
        if not verify_no_cluster_leakage(
            unpaired_tra_train, unpaired_tra_val, unpaired_tra_test, 'cluster', 'unpaired_tra'
        ):
            all_verified = False

    if len(unpaired_trb_train) > 0 or len(unpaired_trb_val) > 0 or len(unpaired_trb_test) > 0:
        if not verify_no_cluster_leakage(
            unpaired_trb_train, unpaired_trb_val, unpaired_trb_test, 'cluster', 'unpaired_trb'
        ):
            all_verified = False

    if all_verified:
        print("    All verifications passed!")
    else:
        print("    WARNING: Some verifications failed!")

    # Save outputs
    print("\nSaving outputs...")

    # Paired data
    paired_output_dir = output_dir / "paired"
    save_clusters(paired_output_dir / "tra_clusters.tsv", tra_to_cluster, tra_clusters)
    save_clusters(paired_output_dir / "trb_clusters.tsv", trb_to_cluster, trb_clusters)

    # Remove tra_trb column before saving (it's redundant)
    for df in [paired_train, paired_val, paired_test]:
        if 'tra_trb' in df.columns:
            df.drop(columns=['tra_trb'], inplace=True)

    save_parquet(paired_train, paired_output_dir / "train.parquet")
    save_parquet(paired_val, paired_output_dir / "val.parquet")
    save_parquet(paired_test, paired_output_dir / "test.parquet")

    # Unpaired TRA
    unpaired_tra_output_dir = output_dir / "unpaired_tra"
    if len(unpaired_tra_df) > 0:
        save_clusters(unpaired_tra_output_dir / "clusters.tsv",
                      unpaired_tra_to_cluster, unpaired_tra_clusters)
        save_parquet(unpaired_tra_train, unpaired_tra_output_dir / "train.parquet")
        save_parquet(unpaired_tra_val, unpaired_tra_output_dir / "val.parquet")
        save_parquet(unpaired_tra_test, unpaired_tra_output_dir / "test.parquet")

    # Unpaired TRB
    unpaired_trb_output_dir = output_dir / "unpaired_trb"
    if len(unpaired_trb_df) > 0:
        save_clusters(unpaired_trb_output_dir / "clusters.tsv",
                      unpaired_trb_to_cluster, unpaired_trb_clusters)
        save_parquet(unpaired_trb_train, unpaired_trb_output_dir / "train.parquet")
        save_parquet(unpaired_trb_val, unpaired_trb_output_dir / "val.parquet")
        save_parquet(unpaired_trb_test, unpaired_trb_output_dir / "test.parquet")

    # Collect and write statistics
    stats = collect_statistics(
        paired_df=paired_df,
        paired_splits=(paired_train, paired_val, paired_test),
        unpaired_tra_df=unpaired_tra_df,
        unpaired_tra_splits=(unpaired_tra_train, unpaired_tra_val, unpaired_tra_test),
        unpaired_trb_df=unpaired_trb_df,
        unpaired_trb_splits=(unpaired_trb_train, unpaired_trb_val, unpaired_trb_test),
        tra_to_cluster=tra_to_cluster,
        tra_clusters=tra_clusters,
        trb_to_cluster=trb_to_cluster,
        trb_clusters=trb_clusters,
        unpaired_tra_to_cluster=unpaired_tra_to_cluster,
        unpaired_tra_clusters=unpaired_tra_clusters,
        unpaired_trb_to_cluster=unpaired_trb_to_cluster,
        unpaired_trb_clusters=unpaired_trb_clusters,
    )

    config = {
        'input_dir': args.input_dir,
        'output_dir': args.output_dir,
        'similarity_threshold': args.similarity_threshold,
        'coverage': args.coverage,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'seed': args.seed,
        'max_files': args.max_files,
    }

    write_statistics_doc(output_dir, stats, config)

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nPaired data:")
    print(f"  Total: {len(paired_df):,}")
    print(f"  Train: {len(paired_train):,}")
    print(f"  Val: {len(paired_val):,}")
    print(f"  Test: {len(paired_test):,}")
    print(f"\nUnpaired TRA:")
    print(f"  Total: {len(unpaired_tra_df):,}")
    print(f"  Train: {len(unpaired_tra_train):,}")
    print(f"  Val: {len(unpaired_tra_val):,}")
    print(f"  Test: {len(unpaired_tra_test):,}")
    print(f"\nUnpaired TRB:")
    print(f"  Total: {len(unpaired_trb_df):,}")
    print(f"  Train: {len(unpaired_trb_train):,}")
    print(f"  Val: {len(unpaired_trb_val):,}")
    print(f"  Test: {len(unpaired_trb_test):,}")

    print("\nDone!")


if __name__ == "__main__":
    main()
