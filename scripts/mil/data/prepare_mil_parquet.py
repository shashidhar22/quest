#!/usr/bin/env python3
"""
Prepare MIL data in Parquet format with count aggregation.

Converts TSV repertoire files to optimized Parquet format with:
- Sequence deduplication per repertoire
- Count aggregation (templates column or occurrence counting)
- Proper schema for downstream processing
"""

import argparse
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np


def process_repertoire(tsv_path: str, repertoire_id: str, label: int) -> pd.DataFrame:
    """
    Process a single repertoire TSV file.

    Handles:
    - templates column if present (use as count)
    - Duplicate sequences (aggregate counts)
    - Missing counts (count occurrences)
    """
    # Read TSV
    df = pd.read_csv(tsv_path, sep='\t')

    # Standardize column names
    col_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'junction' in col_lower and 'aa' in col_lower:
            col_map[col] = 'sequence_id'
        elif col_lower in ['templates', 'template', 'count', 'counts', 'duplicate_count']:
            col_map[col] = 'count'
        elif 'v_call' in col_lower or col_lower == 'v':
            col_map[col] = 'v_call'
        elif 'j_call' in col_lower or col_lower == 'j':
            col_map[col] = 'j_call'

    df = df.rename(columns=col_map)

    # Ensure required columns exist
    if 'sequence_id' not in df.columns:
        raise ValueError(f"No junction_aa column found in {tsv_path}")

    # Handle counts
    if 'count' not in df.columns:
        df['count'] = 1
    else:
        # Fill NaN counts with 1
        df['count'] = df['count'].fillna(1).astype(int)

    # Handle V/J calls
    if 'v_call' not in df.columns:
        df['v_call'] = ''
    if 'j_call' not in df.columns:
        df['j_call'] = ''

    # Aggregate duplicates: sum counts, keep first V/J call
    agg_dict = {
        'count': 'sum',
        'v_call': 'first',
        'j_call': 'first'
    }

    df_agg = df.groupby('sequence_id', as_index=False).agg(agg_dict)

    # Add metadata
    df_agg['repertoire_id'] = repertoire_id
    df_agg['label'] = label

    return df_agg[['repertoire_id', 'sequence_id', 'count', 'v_call', 'j_call', 'label']]


def prepare_dataset(
    input_dir: str,
    output_dir: str,
    dataset_name: str = None
):
    """
    Prepare a complete dataset directory.

    Expects:
    - input_dir/metadata.csv with columns: repertoire_id, filename, label_positive
    - input_dir/*.tsv repertoire files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if dataset_name is None:
        dataset_name = input_path.name

    print(f"="*80)
    print(f"PREPARING DATASET: {dataset_name}")
    print(f"="*80)

    # Load metadata
    metadata_file = input_path / 'metadata.csv'
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    metadata = pd.read_csv(metadata_file)
    print(f"Found {len(metadata)} repertoires in metadata")

    # Standardize metadata columns
    if 'label_positive' in metadata.columns:
        metadata['label'] = metadata['label_positive'].astype(int)
    elif 'label' not in metadata.columns:
        raise ValueError("No label column found in metadata")

    # Process all repertoires
    all_repertoires = []
    total_sequences = 0
    total_unique = 0

    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing repertoires"):
        repertoire_id = row['repertoire_id']
        filename = row['filename']
        label = row['label']

        tsv_path = input_path / filename
        if not tsv_path.exists():
            print(f"  Warning: File not found: {tsv_path}")
            continue

        try:
            df = process_repertoire(str(tsv_path), repertoire_id, label)
            all_repertoires.append(df)
            total_unique += len(df)
            total_sequences += df['count'].sum()
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            continue

    if not all_repertoires:
        raise ValueError("No repertoires processed successfully")

    # Combine all repertoires
    combined = pd.concat(all_repertoires, ignore_index=True)

    print(f"\nDataset statistics:")
    print(f"  Repertoires: {len(metadata)}")
    print(f"  Unique sequences (per repertoire): {total_unique:,}")
    print(f"  Total sequences (with counts): {total_sequences:,}")
    print(f"  Labels: {combined['label'].value_counts().to_dict()}")

    # Save as Parquet
    schema = pa.schema([
        ('repertoire_id', pa.string()),
        ('sequence_id', pa.string()),
        ('count', pa.int64()),
        ('v_call', pa.string()),
        ('j_call', pa.string()),
        ('label', pa.int32()),
    ])

    repertoires_path = output_path / 'repertoires.parquet'
    table = pa.Table.from_pandas(combined, schema=schema)
    pq.write_table(table, repertoires_path, compression='snappy')

    print(f"\nSaved: {repertoires_path}")
    print(f"  Size: {os.path.getsize(repertoires_path) / 1e6:.1f} MB")

    # Save metadata
    metadata_out = metadata[['repertoire_id', 'label']].copy()
    if 'filename' in metadata.columns:
        metadata_out['filename'] = metadata['filename']

    # Add additional metadata columns if present
    for col in ['age', 'sex', 'hla_a', 'hla_b', 'hla_c', 'hla_drb1', 'hla_dqb1']:
        if col in metadata.columns:
            metadata_out[col] = metadata[col]

    metadata_path = output_path / 'metadata.parquet'
    pq.write_table(pa.Table.from_pandas(metadata_out), metadata_path)
    print(f"Saved: {metadata_path}")

    # Also extract global unique sequences for embedding extraction
    unique_seqs = combined[['sequence_id']].drop_duplicates()
    unique_path = output_path / 'unique_sequences.parquet'
    pq.write_table(pa.Table.from_pandas(unique_seqs), unique_path)
    print(f"Saved: {unique_path} ({len(unique_seqs):,} unique sequences)")

    print(f"\n{'='*80}")
    print(f"Dataset preparation complete!")
    print(f"{'='*80}")

    return len(metadata), len(unique_seqs)


def merge_unique_sequences(processed_dirs: list, output_path: str):
    """
    Merge unique sequences from multiple datasets for global embedding extraction.
    """
    print(f"\nMerging unique sequences from {len(processed_dirs)} datasets...")

    all_sequences = []
    for d in tqdm(processed_dirs, desc="Loading"):
        unique_path = Path(d) / 'unique_sequences.parquet'
        if unique_path.exists():
            df = pd.read_parquet(unique_path)
            all_sequences.append(df)

    combined = pd.concat(all_sequences, ignore_index=True)
    unique = combined.drop_duplicates()

    print(f"  Total sequences across datasets: {len(combined):,}")
    print(f"  Global unique sequences: {len(unique):,}")
    print(f"  Deduplication ratio: {len(combined)/len(unique):.2f}x")

    pq.write_table(pa.Table.from_pandas(unique), output_path)
    print(f"Saved: {output_path}")

    return len(unique)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MIL data in Parquet format with count aggregation"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Input directory with metadata.csv and TSV files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for Parquet files")
    parser.add_argument("--dataset_name", type=str, default=None,
                       help="Dataset name (default: directory name)")

    # For merging multiple datasets
    parser.add_argument("--merge_datasets", type=str, nargs='+', default=None,
                       help="List of processed dataset directories to merge unique sequences")
    parser.add_argument("--merge_output", type=str, default=None,
                       help="Output path for merged unique sequences")

    args = parser.parse_args()

    if args.merge_datasets:
        if not args.merge_output:
            args.merge_output = "global_unique_sequences.parquet"
        merge_unique_sequences(args.merge_datasets, args.merge_output)
    else:
        prepare_dataset(args.input_dir, args.output_dir, args.dataset_name)


if __name__ == "__main__":
    main()
