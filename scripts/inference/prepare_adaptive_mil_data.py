#!/usr/bin/env python3
"""
prepare_adaptive_mil_data.py

Convert Adaptive ImmuneACCESS TCR repertoire data to MIL-compatible JSON format.
This script processes the Adaptive Immune Profiling Challenge 2025 dataset.
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def load_adaptive_repertoire(tsv_path, sequence_column='junction_aa'):
    """
    Load TCR sequences from an Adaptive ImmuneACCESS TSV file.
    
    Args:
        tsv_path: Path to TSV file with TCR sequences
        sequence_column: Name of column containing CDR3 sequences
    
    Returns:
        List of CDR3 sequences
    """
    df = pd.read_csv(tsv_path, sep='\t')
    
    if sequence_column not in df.columns:
        raise ValueError(f"Column '{sequence_column}' not found in {tsv_path}")
    
    # Filter out empty/null sequences
    sequences = df[sequence_column].dropna().astype(str).tolist()
    sequences = [seq.strip() for seq in sequences if seq.strip() and seq.strip() != '-999.0']
    
    return sequences


def process_adaptive_dataset(dataset_dir, metadata_file='metadata.csv', sequence_column='junction_aa'):
    """
    Process an Adaptive dataset directory into MIL format.
    
    Args:
        dataset_dir: Path to dataset directory containing metadata.csv and TSV files
        metadata_file: Name of metadata file
        sequence_column: Column name for CDR3 sequences
    
    Returns:
        List of dictionaries with repertoire_id, sequences, and label
    """
    dataset_dir = Path(dataset_dir)
    metadata_path = dataset_dir / metadata_file
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    print(f"\nProcessing {len(metadata)} repertoires from {dataset_dir.name}")
    
    # Check required columns
    required_cols = ['repertoire_id', 'filename', 'label_positive']
    for col in required_cols:
        if col not in metadata.columns:
            raise ValueError(f"Required column '{col}' not found in metadata")
    
    # Process each repertoire
    repertoires = []
    skipped = 0
    
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Loading repertoires"):
        repertoire_id = row['repertoire_id']
        filename = row['filename']
        label = 1 if row['label_positive'] else 0
        
        tsv_path = dataset_dir / filename
        
        if not tsv_path.exists():
            print(f"Warning: File not found: {tsv_path}")
            skipped += 1
            continue
        
        try:
            sequences = load_adaptive_repertoire(tsv_path, sequence_column)
            
            if len(sequences) == 0:
                print(f"Warning: No sequences found in {filename}")
                skipped += 1
                continue
            
            repertoires.append({
                'repertoire_id': repertoire_id,
                'sequences': sequences,
                'label': label
            })
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            skipped += 1
            continue
    
    if skipped > 0:
        print(f"Skipped {skipped} repertoires due to errors")
    
    return repertoires


def combine_datasets(dataset_dirs, output_file, sequence_column='junction_aa'):
    """
    Combine multiple Adaptive datasets into a single MIL-compatible JSON file.
    
    Args:
        dataset_dirs: List of dataset directory paths
        output_file: Output JSON file path
        sequence_column: Column name for CDR3 sequences
    """
    all_repertoires = []
    
    for dataset_dir in dataset_dirs:
        dataset_dir = Path(dataset_dir)
        if not dataset_dir.exists():
            print(f"Warning: Dataset directory not found: {dataset_dir}")
            continue
        
        repertoires = process_adaptive_dataset(dataset_dir, sequence_column=sequence_column)
        all_repertoires.extend(repertoires)
    
    # Save to JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_repertoires, f, indent=2)
    
    # Print statistics
    print(f"\n{'='*80}")
    print(f"Combined Dataset Statistics:")
    print(f"{'='*80}")
    print(f"Total repertoires: {len(all_repertoires)}")
    
    # Label distribution
    labels = [r['label'] for r in all_repertoires]
    label_counts = pd.Series(labels).value_counts().sort_index()
    print(f"\nLabel distribution:")
    for label, count in label_counts.items():
        label_name = "Positive" if label == 1 else "Negative"
        print(f"  {label_name} ({label}): {count} ({count/len(labels)*100:.1f}%)")
    
    # Sequence statistics
    seq_counts = [len(r['sequences']) for r in all_repertoires]
    print(f"\nSequences per repertoire:")
    print(f"  Mean: {pd.Series(seq_counts).mean():.1f}")
    print(f"  Median: {pd.Series(seq_counts).median():.1f}")
    print(f"  Min: {pd.Series(seq_counts).min()}")
    print(f"  Max: {pd.Series(seq_counts).max()}")
    
    print(f"\n✅ Saved to: {output_path}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Adaptive ImmuneACCESS data to MIL-compatible JSON format"
    )
    
    parser.add_argument(
        '--input_dirs',
        type=str,
        nargs='+',
        required=True,
        help='One or more dataset directories containing metadata.csv and TSV files'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSON file path'
    )
    
    parser.add_argument(
        '--sequence_column',
        type=str,
        default='junction_aa',
        help='Column name for CDR3 sequences (default: junction_aa)'
    )
    
    args = parser.parse_args()
    
    # Process and combine datasets
    combine_datasets(
        dataset_dirs=args.input_dirs,
        output_file=args.output,
        sequence_column=args.sequence_column
    )


if __name__ == "__main__":
    main()
