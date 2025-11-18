#!/usr/bin/env python3
"""
prepare_mil_data.py
────────────────────────────────────────────────────────
Convert MIL dataset TSV files to Parquet format and create sampled datasets.

This script:
1. Reads TSV repertoire files and metadata
2. Converts to efficient Parquet format using PyArrow
3. Creates sampled datasets for quick testing
4. Generates JSON format compatible with repertoire_mil.py
"""

import os
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np


def convert_repertoire_to_parquet(
    tsv_path: str,
    parquet_path: str,
    repertoire_id: str,
    label: bool
):
    """
    Convert a single TSV repertoire file to Parquet format.
    
    Args:
        tsv_path: Path to input TSV file
        parquet_path: Path to output Parquet file
        repertoire_id: Unique repertoire identifier
        label: Label for this repertoire
    """
    # Read TSV file
    df = pd.read_csv(tsv_path, sep='\t')
    
    # Add repertoire metadata
    df['repertoire_id'] = repertoire_id
    df['label'] = label
    
    # Rename columns to match expected format
    df.rename(columns={
        'junction_aa': 'sequence',
        'v_call': 'v_gene',
        'j_call': 'j_gene'
    }, inplace=True)
    
    # Add CDR3 length
    df['cdr3_length'] = df['sequence'].str.len()
    
    # Add frequency if 'templates' column exists
    if 'templates' in df.columns:
        df['frequency'] = df['templates']
    else:
        # If no frequency info, assign uniform frequency
        df['frequency'] = 1
    
    # Normalize frequencies to sum to 1
    df['frequency'] = df['frequency'] / df['frequency'].sum()
    
    # Select and order columns
    columns = ['repertoire_id', 'sequence', 'v_gene', 'j_gene', 'cdr3_length', 'frequency', 'label']
    if 'd_call' in df.columns:
        df.rename(columns={'d_call': 'd_gene'}, inplace=True)
        columns.insert(4, 'd_gene')
    
    df = df[columns]
    
    # Write to Parquet
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_path, compression='snappy')


def convert_dataset_to_parquet(
    input_dir: str,
    output_dir: str,
    metadata_file: str = 'metadata.csv'
):
    """
    Convert all TSV files in a dataset to Parquet format.
    
    Args:
        input_dir: Directory containing TSV files and metadata
        output_dir: Directory to save Parquet files
        metadata_file: Name of metadata CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Read metadata
    metadata_path = os.path.join(input_dir, metadata_file)
    metadata = pd.read_csv(metadata_path)
    
    print(f"Converting {len(metadata)} repertoires from TSV to Parquet...")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    # Convert each repertoire
    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        repertoire_id = row['repertoire_id']
        filename = row['filename']
        label = row['label_positive']
        
        tsv_path = os.path.join(input_dir, filename)
        parquet_filename = filename.replace('.tsv', '.parquet')
        parquet_path = os.path.join(output_dir, parquet_filename)
        
        if os.path.exists(tsv_path):
            convert_repertoire_to_parquet(tsv_path, parquet_path, repertoire_id, label)
    
    # Copy metadata to output directory
    metadata_output = os.path.join(output_dir, metadata_file)
    metadata.to_csv(metadata_output, index=False)
    
    print(f"✅ Conversion complete! Files saved to: {output_dir}")


def create_json_format(
    parquet_dir: str,
    output_json: str,
    metadata_file: str = 'metadata.csv',
    use_additional_features: bool = True
):
    """
    Create JSON format for repertoire_mil.py from Parquet files.
    
    Args:
        parquet_dir: Directory containing Parquet files
        output_json: Path to output JSON file
        metadata_file: Name of metadata CSV file
        use_additional_features: Include V/J genes, frequency, etc.
    """
    metadata_path = os.path.join(parquet_dir, metadata_file)
    metadata = pd.read_csv(metadata_path)
    
    repertoire_data = []
    
    print(f"Creating JSON format from {len(metadata)} repertoires...")
    
    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        repertoire_id = row['repertoire_id']
        filename = row['filename'].replace('.tsv', '.parquet')
        label = int(row['label_positive'])  # Convert True/False to 1/0
        
        parquet_path = os.path.join(parquet_dir, filename)
        
        if not os.path.exists(parquet_path):
            print(f"Warning: {parquet_path} not found, skipping...")
            continue
        
        # Read Parquet file
        df = pd.read_parquet(parquet_path)
        
        # Create repertoire dict
        rep_dict = {
            'repertoire_id': repertoire_id,
            'sequences': df['sequence'].tolist(),
            'label': label
        }
        
        if use_additional_features:
            rep_dict['v_gene'] = df['v_gene'].tolist()
            rep_dict['j_gene'] = df['j_gene'].tolist()
            rep_dict['cdr3_length'] = df['cdr3_length'].tolist()
            rep_dict['frequency'] = df['frequency'].tolist()
            
            if 'd_gene' in df.columns:
                rep_dict['d_gene'] = df['d_gene'].tolist()
        
        repertoire_data.append(rep_dict)
    
    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(repertoire_data, f, indent=2)
    
    print(f"✅ JSON format saved to: {output_json}")
    print(f"   Total repertoires: {len(repertoire_data)}")
    
    # Print label distribution
    labels = [r['label'] for r in repertoire_data]
    unique, counts = np.unique(labels, return_counts=True)
    print(f"   Label distribution:")
    for label, count in zip(unique, counts):
        print(f"     Label {label}: {count} ({count/len(labels)*100:.1f}%)")


def create_sampled_dataset(
    input_json: str,
    output_json: str,
    n_repertoires: int = 50,
    max_sequences_per_repertoire: int = 500,
    stratified: bool = True,
    seed: int = 42
):
    """
    Create a sampled dataset for quick testing.
    
    Args:
        input_json: Path to full dataset JSON
        output_json: Path to output sampled JSON
        n_repertoires: Number of repertoires to sample
        max_sequences_per_repertoire: Maximum sequences per repertoire
        stratified: Whether to maintain label distribution
        seed: Random seed
    """
    np.random.seed(seed)
    
    # Load full dataset
    with open(input_json, 'r') as f:
        full_data = json.load(f)
    
    print(f"Creating sampled dataset:")
    print(f"  Input: {len(full_data)} repertoires")
    print(f"  Sample: {n_repertoires} repertoires")
    print(f"  Max sequences per repertoire: {max_sequences_per_repertoire}")
    
    # Sample repertoires
    if stratified:
        # Maintain label distribution
        df = pd.DataFrame([{'idx': i, 'label': r['label']} for i, r in enumerate(full_data)])
        sampled_indices = df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(min(len(x), n_repertoires // 2), random_state=seed)
        )['idx'].tolist()
    else:
        sampled_indices = np.random.choice(len(full_data), size=min(n_repertoires, len(full_data)), replace=False)
    
    sampled_data = []
    
    for idx in sampled_indices:
        rep = full_data[idx].copy()
        
        # Sample sequences if repertoire is too large
        n_sequences = len(rep['sequences'])
        if n_sequences > max_sequences_per_repertoire:
            sample_indices = np.random.choice(
                n_sequences, 
                size=max_sequences_per_repertoire, 
                replace=False
            )
            
            rep['sequences'] = [rep['sequences'][i] for i in sample_indices]
            
            # Sample other fields too
            for key in ['v_gene', 'j_gene', 'd_gene', 'cdr3_length', 'frequency']:
                if key in rep:
                    rep[key] = [rep[key][i] for i in sample_indices]
            
            # Renormalize frequencies
            if 'frequency' in rep:
                freq_sum = sum(rep['frequency'])
                rep['frequency'] = [f / freq_sum for f in rep['frequency']]
        
        sampled_data.append(rep)
    
    # Save sampled dataset
    with open(output_json, 'w') as f:
        json.dump(sampled_data, f, indent=2)
    
    print(f"✅ Sampled dataset saved to: {output_json}")
    print(f"   Total repertoires: {len(sampled_data)}")
    
    # Print statistics
    seq_counts = [len(r['sequences']) for r in sampled_data]
    print(f"   Sequences per repertoire: min={min(seq_counts)}, max={max(seq_counts)}, mean={np.mean(seq_counts):.1f}")
    
    labels = [r['label'] for r in sampled_data]
    unique, counts = np.unique(labels, return_counts=True)
    print(f"   Label distribution:")
    for label, count in zip(unique, counts):
        print(f"     Label {label}: {count} ({count/len(labels)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MIL dataset: TSV to Parquet conversion and sampling"
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory with TSV files (e.g., train_dataset_1)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for processed files"
    )
    parser.add_argument(
        "--convert_to_parquet",
        action="store_true",
        help="Convert TSV files to Parquet format"
    )
    parser.add_argument(
        "--create_json",
        action="store_true",
        help="Create JSON format for repertoire_mil.py"
    )
    parser.add_argument(
        "--create_sample",
        action="store_true",
        help="Create sampled dataset for testing"
    )
    parser.add_argument(
        "--n_sample_repertoires",
        type=int,
        default=50,
        help="Number of repertoires in sampled dataset"
    )
    parser.add_argument(
        "--max_sequences_per_repertoire",
        type=int,
        default=500,
        help="Maximum sequences per repertoire in sample"
    )
    parser.add_argument(
        "--no_additional_features",
        action="store_true",
        help="Don't include V/J genes, frequency in JSON"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Convert TSV to Parquet
    if args.convert_to_parquet:
        parquet_dir = os.path.join(args.output_dir, "parquet")
        convert_dataset_to_parquet(args.input_dir, parquet_dir)
    else:
        parquet_dir = os.path.join(args.output_dir, "parquet")
    
    # Step 2: Create JSON format
    if args.create_json:
        json_path = os.path.join(args.output_dir, "repertoire_data.json")
        create_json_format(
            parquet_dir, 
            json_path,
            use_additional_features=not args.no_additional_features
        )
    else:
        json_path = os.path.join(args.output_dir, "repertoire_data.json")
    
    # Step 3: Create sampled dataset
    if args.create_sample:
        if not os.path.exists(json_path):
            print("Error: JSON file not found. Run with --create_json first.")
            return
        
        sample_json_path = os.path.join(args.output_dir, "repertoire_data_sample.json")
        create_sampled_dataset(
            json_path,
            sample_json_path,
            n_repertoires=args.n_sample_repertoires,
            max_sequences_per_repertoire=args.max_sequences_per_repertoire,
            stratified=True,
            seed=args.seed
        )
    
    print("\n✅ All processing complete!")


if __name__ == "__main__":
    main()
