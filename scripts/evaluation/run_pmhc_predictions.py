#!/usr/bin/env python3
"""
Run peptide-MHC binding predictions using multiple tools.

Executes netMHCpan 4.2, netMHCIIpan 4.3, and MHCflurry on evaluation datasets
and collects predictions in a unified format.

Usage:
    # Run all tools on Class I data
    python scripts/evaluation/run_pmhc_predictions.py \\
        --input_file data/eval/pmhc_with_negatives/class_one/peptide_mhc_one/test.parquet \\
        --output_dir results/pmhc_predictions \\
        --mhc_class class_one \\
        --tools netmhcpan mhcflurry

    # Run netMHCIIpan on Class II data
    python scripts/evaluation/run_pmhc_predictions.py \\
        --input_file data/eval/pmhc_with_negatives/class_two/peptide_mhc_two/test.parquet \\
        --output_dir results/pmhc_predictions \\
        --mhc_class class_two \\
        --tools netmhciipan

Tool Paths:
    - netMHCpan 4.2: /home/sravisha/tcrbench_tools/netMHCpan-4.2/netMHCpan
    - netMHCIIpan 4.3: /home/sravisha/tcrbench_tools/netMHCIIpan-4.3/netMHCIIpan
    - MHCflurry: mhcflurry-predict (via pip)

Author: Claude
"""

import argparse
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# Import utilities
from utils.input_formatters import (
    convert_allele_mhcflurry,
    convert_allele_netmhciipan,
    convert_allele_netmhcpan,
    filter_to_supported_alleles,
    get_supported_alleles_netmhciipan,
    get_supported_alleles_netmhcpan,
)
from utils.output_parsers import (
    parse_mhcflurry_output,
    parse_netmhciipan_output,
    parse_netmhcpan_output,
    parse_netmhcpan_stdout,
    parse_netmhciipan_stdout,
)

# Recognized test split patterns (ordered by expected difficulty)
TEST_SPLIT_PATTERNS = [
    'test_seen_motif',                  # Easy: Same binding motifs as training
    'test_unseen_peptide_seen_motif',   # Medium: New peptides, motifs seen
    'test_unseen_motif',                # Hard: Binding motifs not in training
    'test_unseen_allele',               # Hardest: MHC alleles not in training
    'test',                              # Fallback for simple split
]


def discover_test_splits(input_dir: Path) -> List[str]:
    """
    Auto-detect test split files in directory.

    Looks for parquet files matching recognized test split patterns.

    Args:
        input_dir: Directory to search for test splits

    Returns:
        List of discovered split names (without .parquet extension)
    """
    splits = []
    for pattern in TEST_SPLIT_PATTERNS:
        if (input_dir / f"{pattern}.parquet").exists():
            splits.append(pattern)

    return splits if splits else ['test']


# Tool paths - using bash versions which don't require tcsh
NETMHCPAN_PATH = "/home/sravisha/tcrbench_tools/netMHCpan-4.2/netMHCpanb"
NETMHCIIPAN_PATH = "/home/sravisha/tcrbench_tools/netMHCIIpan-4.3/netMHCIIpan.bash"
MHCFLURRY_CMD = "mhcflurry-predict"

# Environment variables needed for netMHC tools
NETMHC_ENV = {"TMPDIR": "/tmp"}


def check_tool_availability(tool: str) -> bool:
    """Check if a prediction tool is available."""
    if tool == 'netmhcpan':
        return Path(NETMHCPAN_PATH).exists()
    elif tool == 'netmhciipan':
        return Path(NETMHCIIPAN_PATH).exists()
    elif tool == 'mhcflurry':
        return shutil.which(MHCFLURRY_CMD) is not None
    return False


def _run_single_netmhcpan_batch(args: Tuple) -> Optional[pd.DataFrame]:
    """
    Run a single netMHCpan batch (for parallel execution).

    Args:
        args: Tuple of (pep_file, allele_str, output_file, include_ba, allele_batch_idx, pep_batch_idx)

    Returns:
        DataFrame with predictions or None if failed
    """
    pep_file, allele_str, output_file, include_ba, allele_batch_idx, pep_batch_idx = args

    cmd = [
        NETMHCPAN_PATH,
        "-p",
        "-f", str(pep_file),
        "-a", allele_str,
        "-xls",
        "-xlsfile", str(output_file),
    ]

    if include_ba:
        cmd.append("-BA")

    try:
        # Set up environment with TMPDIR
        env = os.environ.copy()
        env.update(NETMHC_ENV)

        # Count alleles for timeout calculation
        # 2 hour base + 6 min per allele (20 alleles = 2 hours)
        n_alleles = len(allele_str.split(","))
        timeout = max(7200, 360 * n_alleles)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        if result.returncode != 0:
            print(f"\n  Warning: netMHCpan failed for batch a{allele_batch_idx}_p{pep_batch_idx}")
            print(f"  stderr: {result.stderr}")
            return None

        # Parse output
        output_path = Path(output_file)
        if output_path.exists():
            try:
                preds = parse_netmhcpan_output(output_path)
                if len(preds) > 0:
                    return preds
            except Exception as e:
                print(f"\n  Warning: Failed to parse output for batch a{allele_batch_idx}_p{pep_batch_idx}: {e}")
                # Try parsing stdout
                try:
                    preds = parse_netmhcpan_stdout(result.stdout)
                    if len(preds) > 0:
                        return preds
                except Exception:
                    pass

    except subprocess.TimeoutExpired:
        print(f"\n  Warning: netMHCpan timed out for batch a{allele_batch_idx}_p{pep_batch_idx}")
    except Exception as e:
        print(f"\n  Warning: netMHCpan error for batch a{allele_batch_idx}_p{pep_batch_idx}: {e}")

    return None


def run_netmhcpan(
    peptides: List[str],
    alleles: List[str],
    output_dir: Path,
    include_ba: bool = True,
    peptide_batch_size: int = 5000,
    allele_batch_size: int = 50,
    n_workers: int = 1,
) -> pd.DataFrame:
    """
    Run netMHCpan 4.2 predictions with optimized batching and optional parallelism.

    Args:
        peptides: List of peptide sequences
        alleles: List of MHC alleles in netMHCpan format
        output_dir: Directory for output files
        include_ba: Include binding affinity predictions
        peptide_batch_size: Number of peptides per batch
        allele_batch_size: Number of alleles per call (0 or None = all at once)
        n_workers: Number of parallel workers (default: 1, sequential)

    Returns:
        DataFrame with predictions
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = []

    # Batch alleles for efficient processing
    if allele_batch_size is None or allele_batch_size <= 0 or allele_batch_size >= len(alleles):
        allele_batches = [alleles]
    else:
        allele_batches = [
            alleles[i:i + allele_batch_size]
            for i in range(0, len(alleles), allele_batch_size)
        ]

    # Batch peptides
    n_pep_batches = (len(peptides) + peptide_batch_size - 1) // peptide_batch_size

    total_calls = len(allele_batches) * n_pep_batches
    print(f"    Running {total_calls} netMHCpan calls "
          f"({len(allele_batches)} allele batch(es) × {n_pep_batches} peptide batch(es))")
    if n_workers > 1:
        print(f"    Using {n_workers} parallel workers")

    # Prepare all batch jobs
    batch_jobs = []
    for allele_batch_idx, allele_batch in enumerate(allele_batches):
        allele_str = ",".join(allele_batch)

        for pep_batch_idx in range(n_pep_batches):
            start = pep_batch_idx * peptide_batch_size
            end = min(start + peptide_batch_size, len(peptides))
            batch_peptides = peptides[start:end]

            # Write peptide file (unique per batch to avoid conflicts in parallel)
            pep_file = output_dir / f"peptides_a{allele_batch_idx}_p{pep_batch_idx}.txt"
            with open(pep_file, 'w') as f:
                for pep in batch_peptides:
                    f.write(f"{pep}\n")

            output_file = output_dir / f"netmhcpan_a{allele_batch_idx}_p{pep_batch_idx}.xls"

            batch_jobs.append((
                str(pep_file),
                allele_str,
                str(output_file),
                include_ba,
                allele_batch_idx,
                pep_batch_idx,
            ))

    # Execute batches (parallel or sequential)
    if n_workers > 1:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_run_single_netmhcpan_batch, job): job for job in batch_jobs}
            for future in tqdm(as_completed(futures), total=len(futures), desc="netMHCpan"):
                result = future.result()
                if result is not None and len(result) > 0:
                    all_predictions.append(result)
    else:
        # Sequential execution
        for job in tqdm(batch_jobs, desc="netMHCpan"):
            result = _run_single_netmhcpan_batch(job)
            if result is not None and len(result) > 0:
                all_predictions.append(result)

    if not all_predictions:
        return pd.DataFrame()

    combined = pd.concat(all_predictions, ignore_index=True)
    combined['tool'] = 'netmhcpan'

    return combined


def _run_single_netmhciipan_batch(args: Tuple) -> Optional[pd.DataFrame]:
    """
    Run a single netMHCIIpan batch (for parallel execution).

    Args:
        args: Tuple of (pep_file, allele_str, output_file, include_ba, allele_batch_idx, pep_batch_idx)

    Returns:
        DataFrame with predictions or None if failed
    """
    pep_file, allele_str, output_file, include_ba, allele_batch_idx, pep_batch_idx = args

    cmd = [
        NETMHCIIPAN_PATH,
        "-inptype", "1",
        "-f", str(pep_file),
        "-a", allele_str,
        "-xls",
        "-xlsfile", str(output_file),
    ]

    if include_ba:
        cmd.append("-BA")

    try:
        # Set up environment with TMPDIR
        env = os.environ.copy()
        env.update(NETMHC_ENV)

        # Count alleles for timeout calculation
        # 2 hour base + 6 min per allele (20 alleles = 2 hours)
        n_alleles = len(allele_str.split(","))
        timeout = max(7200, 360 * n_alleles)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        if result.returncode != 0:
            print(f"\n  Warning: netMHCIIpan failed for batch a{allele_batch_idx}_p{pep_batch_idx}")
            print(f"  returncode: {result.returncode}")
            print(f"  stderr: {result.stderr}")
            print(f"  stdout (first 1000 chars): {result.stdout[:1000]}")
            print(f"  alleles: {allele_str[:200]}")
            return None

        # Parse output
        output_path = Path(output_file)
        if output_path.exists():
            try:
                preds = parse_netmhciipan_output(output_path)
                if len(preds) > 0:
                    return preds
            except Exception as e:
                print(f"\n  Warning: Failed to parse output for batch a{allele_batch_idx}_p{pep_batch_idx}: {e}")
                try:
                    preds = parse_netmhciipan_stdout(result.stdout)
                    if len(preds) > 0:
                        return preds
                except Exception:
                    pass

    except subprocess.TimeoutExpired:
        print(f"\n  Warning: netMHCIIpan timed out for batch a{allele_batch_idx}_p{pep_batch_idx}")
    except Exception as e:
        print(f"\n  Warning: netMHCIIpan error for batch a{allele_batch_idx}_p{pep_batch_idx}: {e}")

    return None


def run_netmhciipan(
    peptides: List[str],
    alleles: List[str],
    output_dir: Path,
    include_ba: bool = True,
    peptide_batch_size: int = 5000,
    allele_batch_size: int = 50,
    n_workers: int = 1,
) -> pd.DataFrame:
    """
    Run netMHCIIpan 4.3 predictions with optimized batching and optional parallelism.

    Args:
        peptides: List of peptide sequences
        alleles: List of MHC alleles in netMHCIIpan format
        output_dir: Directory for output files
        include_ba: Include binding affinity predictions
        peptide_batch_size: Number of peptides per batch
        allele_batch_size: Number of alleles per call (0 or None = all at once)
        n_workers: Number of parallel workers (default: 1, sequential)

    Returns:
        DataFrame with predictions
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter out peptides shorter than 9 amino acids (netMHCIIpan requirement)
    original_count = len(peptides)
    peptides = [p for p in peptides if len(p) >= 9]
    if len(peptides) < original_count:
        print(f"    Filtered out {original_count - len(peptides)} peptides shorter than 9 amino acids")

    if not peptides:
        print("    No valid peptides (all too short)")
        return pd.DataFrame()

    all_predictions = []

    # Batch alleles for efficient processing
    if allele_batch_size is None or allele_batch_size <= 0 or allele_batch_size >= len(alleles):
        allele_batches = [alleles]
    else:
        allele_batches = [
            alleles[i:i + allele_batch_size]
            for i in range(0, len(alleles), allele_batch_size)
        ]

    # Batch peptides
    n_pep_batches = (len(peptides) + peptide_batch_size - 1) // peptide_batch_size

    total_calls = len(allele_batches) * n_pep_batches
    print(f"    Running {total_calls} netMHCIIpan calls "
          f"({len(allele_batches)} allele batch(es) × {n_pep_batches} peptide batch(es))")
    if n_workers > 1:
        print(f"    Using {n_workers} parallel workers")

    # Prepare all batch jobs
    batch_jobs = []
    for allele_batch_idx, allele_batch in enumerate(allele_batches):
        allele_str = ",".join(allele_batch)

        for pep_batch_idx in range(n_pep_batches):
            start = pep_batch_idx * peptide_batch_size
            end = min(start + peptide_batch_size, len(peptides))
            batch_peptides = peptides[start:end]

            # Write peptide file (unique per batch to avoid conflicts in parallel)
            pep_file = output_dir / f"peptides_a{allele_batch_idx}_p{pep_batch_idx}.txt"
            with open(pep_file, 'w') as f:
                for pep in batch_peptides:
                    f.write(f"{pep}\n")

            output_file = output_dir / f"netmhciipan_a{allele_batch_idx}_p{pep_batch_idx}.xls"

            batch_jobs.append((
                str(pep_file),
                allele_str,
                str(output_file),
                include_ba,
                allele_batch_idx,
                pep_batch_idx,
            ))

    # Execute batches (parallel or sequential)
    if n_workers > 1:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_run_single_netmhciipan_batch, job): job for job in batch_jobs}
            for future in tqdm(as_completed(futures), total=len(futures), desc="netMHCIIpan"):
                result = future.result()
                if result is not None and len(result) > 0:
                    all_predictions.append(result)
    else:
        # Sequential execution
        for job in tqdm(batch_jobs, desc="netMHCIIpan"):
            result = _run_single_netmhciipan_batch(job)
            if result is not None and len(result) > 0:
                all_predictions.append(result)

    if not all_predictions:
        return pd.DataFrame()

    combined = pd.concat(all_predictions, ignore_index=True)
    combined['tool'] = 'netmhciipan'

    return combined


def run_mhcflurry(
    df: pd.DataFrame,
    peptide_col: str = 'peptide',
    allele_col: str = 'mhc_one_id',
    output_dir: Path = None,
) -> pd.DataFrame:
    """
    Run MHCflurry predictions.

    Args:
        df: DataFrame with peptide and allele columns
        peptide_col: Name of peptide column
        allele_col: Name of allele column
        output_dir: Directory for output files

    Returns:
        DataFrame with predictions
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert alleles to MHCflurry format
    unique_pairs = df[[peptide_col, allele_col]].drop_duplicates()

    # MHCflurry only supports peptides of length 8-15
    MHCFLURRY_MIN_LEN = 8
    MHCFLURRY_MAX_LEN = 15

    input_data = []
    allele_map = {}
    skipped_length = 0

    for _, row in unique_pairs.iterrows():
        peptide = row[peptide_col]
        orig_allele = row[allele_col]

        # Skip peptides outside supported length range
        if len(peptide) < MHCFLURRY_MIN_LEN or len(peptide) > MHCFLURRY_MAX_LEN:
            skipped_length += 1
            continue

        converted = convert_allele_mhcflurry(orig_allele)
        if converted is None:
            continue

        allele_map[converted] = orig_allele
        input_data.append({
            'allele': converted,
            'peptide': peptide,
        })

    if skipped_length > 0:
        print(f"    Filtered {skipped_length:,} peptides outside length range {MHCFLURRY_MIN_LEN}-{MHCFLURRY_MAX_LEN}")

    if not input_data:
        print("  Warning: No alleles could be converted for MHCflurry")
        return pd.DataFrame()

    input_df = pd.DataFrame(input_data)
    input_file = output_dir / "mhcflurry_input.csv"
    output_file = output_dir / "mhcflurry_output.csv"

    input_df.to_csv(input_file, index=False)

    # Run MHCflurry
    cmd = [
        MHCFLURRY_CMD,
        str(input_file),
        "--out", str(output_file),
    ]

    try:
        # Force CPU-only execution to avoid CUDA context issues with multiprocessing
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TensorFlow logging

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
            env=env,
        )

        if result.returncode != 0:
            print(f"  Warning: MHCflurry failed")
            print(f"  returncode: {result.returncode}")
            print(f"  stderr: {result.stderr}")
            print(f"  stdout (first 1000 chars): {result.stdout[:1000]}")
            return pd.DataFrame()

        # Parse output
        if output_file.exists():
            preds = parse_mhcflurry_output(output_file)

            # Map back to original allele names
            if 'allele' in preds.columns:
                preds['original_allele'] = preds['allele'].map(
                    lambda x: allele_map.get(x, x)
                )

            preds['tool'] = 'mhcflurry'
            return preds

    except subprocess.TimeoutExpired:
        print("  Warning: MHCflurry timed out")
    except Exception as e:
        print(f"  Warning: MHCflurry error: {e}")

    return pd.DataFrame()


def run_predictions(
    df: pd.DataFrame,
    tools: List[str],
    mhc_class: str,
    output_dir: Path,
    peptide_col: str = 'peptide',
    allele_col: str = 'mhc_one_id',
    peptide_batch_size: int = 5000,
    allele_batch_size: int = 50,
    n_workers: int = 1,
) -> Dict[str, pd.DataFrame]:
    """
    Run predictions using specified tools.

    Args:
        df: DataFrame with evaluation data
        tools: List of tools to run
        mhc_class: 'class_one' or 'class_two'
        output_dir: Directory for output files
        peptide_col: Peptide column name
        allele_col: Allele column name
        peptide_batch_size: Number of peptides per netMHC call
        allele_batch_size: Number of alleles per netMHC call (0 = all at once)
        n_workers: Number of parallel workers for netMHC tools

    Returns:
        Dictionary mapping tool name to predictions DataFrame
    """
    results = {}

    # Get unique peptides and alleles
    peptides = df[peptide_col].dropna().unique().tolist()
    original_alleles = df[allele_col].dropna().unique().tolist()

    print(f"\n  Data summary:")
    print(f"    Unique peptides: {len(peptides):,}")
    print(f"    Unique alleles: {len(original_alleles):,}")
    print(f"    Total pairs: {len(df):,}")

    # Run each tool
    for tool in tools:
        print(f"\n  Running {tool}...")

        if not check_tool_availability(tool):
            print(f"    Warning: {tool} not available, skipping")
            continue

        tool_output_dir = output_dir / tool

        if tool == 'netmhcpan':
            if mhc_class != 'class_one':
                print(f"    Warning: netMHCpan is for Class I only, skipping")
                continue

            # Convert alleles
            converted_alleles = []
            for allele in original_alleles:
                converted = convert_allele_netmhcpan(allele)
                if converted:
                    converted_alleles.append(converted)

            print(f"    Converted {len(converted_alleles)}/{len(original_alleles)} alleles")

            if not converted_alleles:
                print("    No valid alleles, skipping")
                continue

            preds = run_netmhcpan(
                peptides=peptides,
                alleles=converted_alleles,
                output_dir=tool_output_dir,
                peptide_batch_size=peptide_batch_size,
                allele_batch_size=allele_batch_size,
                n_workers=n_workers,
            )

            if len(preds) > 0:
                results['netmhcpan'] = preds
                print(f"    Got {len(preds):,} predictions")

        elif tool == 'netmhciipan':
            if mhc_class != 'class_two':
                print(f"    Warning: netMHCIIpan is for Class II only, skipping")
                continue

            # Convert alleles
            converted_alleles = []
            for allele in original_alleles:
                converted = convert_allele_netmhciipan(allele)
                if converted:
                    converted_alleles.append(converted)

            print(f"    Converted {len(converted_alleles)}/{len(original_alleles)} alleles")

            if not converted_alleles:
                print("    No valid alleles, skipping")
                continue

            preds = run_netmhciipan(
                peptides=peptides,
                alleles=converted_alleles,
                output_dir=tool_output_dir,
                peptide_batch_size=peptide_batch_size,
                allele_batch_size=allele_batch_size,
                n_workers=n_workers,
            )

            if len(preds) > 0:
                results['netmhciipan'] = preds
                print(f"    Got {len(preds):,} predictions")

        elif tool == 'mhcflurry':
            if mhc_class != 'class_one':
                print(f"    Warning: MHCflurry is for Class I only, skipping")
                continue

            preds = run_mhcflurry(
                df=df,
                peptide_col=peptide_col,
                allele_col=allele_col,
                output_dir=tool_output_dir,
            )

            if len(preds) > 0:
                results['mhcflurry'] = preds
                print(f"    Got {len(preds):,} predictions")

    return results


def merge_predictions_with_data(
    predictions: pd.DataFrame,
    data: pd.DataFrame,
    peptide_col: str = 'peptide',
    allele_col: str = 'mhc_one_id',
) -> pd.DataFrame:
    """
    Merge predictions with original data including labels.

    Args:
        predictions: DataFrame with predictions
        data: DataFrame with original data and labels
        peptide_col: Peptide column name
        allele_col: Allele column name in original data

    Returns:
        Merged DataFrame
    """
    # Standardize prediction allele format for merging
    pred_allele_col = 'allele' if 'allele' in predictions.columns else 'original_allele'

    # Create merge key in predictions
    predictions = predictions.copy()
    predictions['_merge_peptide'] = predictions['peptide'].str.upper()
    predictions['_merge_allele'] = predictions[pred_allele_col].str.upper()

    # Create merge key in data
    data = data.copy()
    data['_merge_peptide'] = data[peptide_col].str.upper()
    data['_merge_allele'] = data[allele_col].str.upper()

    # Merge
    merged = predictions.merge(
        data[['_merge_peptide', '_merge_allele', 'label', 'negative_type']],
        on=['_merge_peptide', '_merge_allele'],
        how='left',
    )

    # Fill missing labels (predictions for pairs not in original data)
    merged['label'] = merged['label'].fillna(-1).astype(int)
    merged['negative_type'] = merged['negative_type'].fillna('unknown')

    # Clean up
    merged = merged.drop(columns=['_merge_peptide', '_merge_allele'])

    return merged


def save_predictions(
    predictions: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Save predictions to parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for tool, preds in predictions.items():
        output_file = output_dir / f"predictions_{tool}.parquet"
        preds.to_parquet(output_file, index=False)
        print(f"  Saved {len(preds):,} predictions to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run peptide-MHC binding predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mutually exclusive input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_file",
        type=str,
        help="Single input parquet file with evaluation data",
    )
    input_group.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing multiple test split parquet files",
    )

    parser.add_argument(
        "--splits",
        type=str,
        nargs='+',
        default=None,
        help="Test splits to process when using --input_dir (default: auto-detect)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--mhc_class",
        type=str,
        choices=['class_one', 'class_two'],
        required=True,
        help="MHC class (class_one or class_two)",
    )
    parser.add_argument(
        "--tools",
        type=str,
        nargs='+',
        choices=['netmhcpan', 'netmhciipan', 'mhcflurry'],
        default=None,
        help="Tools to run (default: auto-detect based on MHC class)",
    )
    parser.add_argument(
        "--peptide_col",
        type=str,
        default="peptide",
        help="Peptide column name (default: peptide)",
    )
    parser.add_argument(
        "--allele_col",
        type=str,
        default=None,
        help="Allele column name (default: auto-detect)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample N rows for testing (default: use all)",
    )
    parser.add_argument(
        "--peptide_batch_size",
        type=int,
        default=5000,
        help="Number of peptides per netMHCpan/netMHCIIpan call (default: 5000)",
    )
    parser.add_argument(
        "--allele_batch_size",
        type=int,
        default=50,
        help="Number of alleles per netMHCpan/netMHCIIpan call, 0=all at once (default: 50)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for netMHC tools (default: 1, sequential)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Auto-detect tools based on MHC class
    if args.tools is None:
        if args.mhc_class == 'class_one':
            tools = ['netmhcpan', 'mhcflurry']
        else:
            tools = ['netmhciipan']
    else:
        tools = args.tools

    # Auto-detect allele column
    if args.allele_col is None:
        if args.mhc_class == 'class_one':
            allele_col = 'mhc_one_id'
        else:
            allele_col = 'mhc_two_id'  # or mhc_one_id for alpha chain
    else:
        allele_col = args.allele_col

    # Determine input files to process
    if args.input_file:
        # Single file mode
        input_files = [(Path(args.input_file), Path(args.input_file).stem)]
        input_dir = None
    else:
        # Directory mode with multiple splits
        input_dir = Path(args.input_dir)

        # Auto-detect splits if not specified
        if args.splits is None:
            splits = discover_test_splits(input_dir)
            print(f"Auto-detected test splits: {splits}")
        else:
            splits = args.splits

        input_files = [(input_dir / f"{split}.parquet", split) for split in splits]

    print("=" * 60)
    print("Peptide-MHC Binding Predictions")
    print("=" * 60)
    print(f"\nConfiguration:")
    if input_dir:
        print(f"  Input directory: {input_dir}")
        print(f"  Splits: {[f[1] for f in input_files]}")
    else:
        print(f"  Input file: {input_files[0][0]}")
    print(f"  Output directory: {output_dir}")
    print(f"  MHC class: {args.mhc_class}")
    print(f"  Tools: {tools}")
    print(f"  Allele column: {allele_col}")
    print(f"  Peptide batch size: {args.peptide_batch_size}")
    print(f"  Allele batch size: {args.allele_batch_size if args.allele_batch_size > 0 else 'all'}")
    print(f"  Workers: {args.workers}")

    # Check tool availability
    print("\nTool availability:")
    for tool in tools:
        available = check_tool_availability(tool)
        status = "available" if available else "NOT AVAILABLE"
        print(f"  {tool}: {status}")

    # Process each input file/split
    all_stats = []

    for input_file, split_name in input_files:
        print("\n" + "=" * 60)
        print(f"Processing split: {split_name}")
        print("=" * 60)

        if not input_file.exists():
            print(f"  Skipping {split_name} (file not found: {input_file})")
            continue

        # Determine output directory for this split
        if input_dir:
            # Directory mode: output_dir/{split_name}/predictions_{tool}.parquet
            split_output_dir = output_dir / split_name
        else:
            # Single file mode: output_dir/predictions_{tool}.parquet
            split_output_dir = output_dir

        # Load data
        print(f"\nLoading data from {input_file}...")
        df = pq.read_table(input_file).to_pandas()
        print(f"  Loaded {len(df):,} rows")

        if 'label' not in df.columns:
            print("  Warning: No 'label' column found, assuming all positives")
            df['label'] = 1
            df['negative_type'] = 'none'

        # Sample if requested
        if args.sample:
            df = df.sample(n=min(args.sample, len(df)), random_state=42)
            print(f"  Sampled {len(df):,} rows for testing")

        # Run predictions
        print("\nRunning predictions...")
        predictions = run_predictions(
            df=df,
            tools=tools,
            mhc_class=args.mhc_class,
            output_dir=split_output_dir / "raw",
            peptide_col=args.peptide_col,
            allele_col=allele_col,
            peptide_batch_size=args.peptide_batch_size,
            allele_batch_size=args.allele_batch_size,
            n_workers=args.workers,
        )

        if not predictions:
            print(f"\nNo predictions generated for {split_name}!")
            continue

        # Merge with labels
        print("\nMerging predictions with labels...")
        for tool, preds in predictions.items():
            merged = merge_predictions_with_data(
                predictions=preds,
                data=df,
                peptide_col=args.peptide_col,
                allele_col=allele_col,
            )
            predictions[tool] = merged
            print(f"  {tool}: {len(merged):,} predictions with labels")

        # Save predictions
        print("\nSaving predictions...")
        save_predictions(predictions, split_output_dir)

        # Collect stats for this split
        for tool, preds in predictions.items():
            n_total = len(preds)
            n_with_label = (preds['label'] >= 0).sum()
            n_positives = (preds['label'] == 1).sum()
            n_negatives = (preds['label'] == 0).sum()

            all_stats.append({
                'split': split_name,
                'tool': tool,
                'n_total': n_total,
                'n_with_label': n_with_label,
                'n_positives': n_positives,
                'n_negatives': n_negatives,
            })

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        print("\n" + stats_df.to_string(index=False))

    print("\nDone!")


if __name__ == "__main__":
    main()
