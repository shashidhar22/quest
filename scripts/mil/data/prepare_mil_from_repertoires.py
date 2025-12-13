#!/usr/bin/env python3
"""
Prepare MIL data and extract ProtBERT embeddings.

This script:
1. Reads all repertoire files
2. Extracts unique CDR3 sequences  
3. Creates repertoire-level data for MIL training
4. Extracts embeddings for unique sequences only

Optimized for datasets with moderate overlap (~10%) between repertoires.
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
import gc


def load_repertoires(data_dir: str, labels_file: str = None):
    """
    Load all repertoire files and collect unique sequences.
    
    Args:
        data_dir: Directory containing repertoire TSV files
        labels_file: Optional file with repertoire labels (TSV: repertoire_id, label)
        
    Returns:
        repertoire_data: List of dicts with repertoire info
        unique_sequences: Set of all unique CDR3 sequences
        sequence_counts: Dict mapping sequence -> total count across repertoires
    """
    data_path = Path(data_dir)
    tsv_files = sorted(data_path.glob("*.tsv"))
    
    print(f"Found {len(tsv_files)} repertoire files")
    
    # Load labels if provided
    labels = {}
    if labels_file and os.path.exists(labels_file):
        labels_df = pd.read_csv(labels_file, sep='\t')
        labels = dict(zip(labels_df.iloc[:, 0], labels_df.iloc[:, 1]))
        print(f"Loaded {len(labels)} labels")
    
    repertoire_data = []
    unique_sequences = set()
    sequence_counts = defaultdict(int)
    
    for tsv_file in tqdm(tsv_files, desc="Loading repertoires"):
        repertoire_id = tsv_file.stem
        
        # Read repertoire
        df = pd.read_csv(tsv_file, sep='\t')
        
        # Get CDR3 sequences (first column, usually 'junction_aa')
        cdr3_col = df.columns[0]
        sequences = df[cdr3_col].dropna().tolist()
        
        # Count frequencies within this repertoire
        seq_counts = pd.Series(sequences).value_counts()
        
        # Update global tracking
        for seq in sequences:
            unique_sequences.add(seq)
            sequence_counts[seq] += 1
        
        # Get label (default to -1 if unknown)
        # Try to infer from filename pattern or labels file
        label = labels.get(repertoire_id, -1)
        
        # Store repertoire data
        repertoire_data.append({
            'repertoire_id': repertoire_id,
            'label': int(label),
            'sequences': seq_counts.index.tolist(),
            'counts': seq_counts.values.tolist(),
            'total_sequences': len(sequences),
            'unique_sequences': len(seq_counts)
        })
    
    return repertoire_data, unique_sequences, sequence_counts


def save_mil_data(
    repertoire_data: list,
    unique_sequences: set,
    output_dir: str
):
    """
    Save MIL-ready data files.
    
    Creates:
    - repertoires.json: Repertoire data with sequences and frequencies
    - unique_sequences.parquet: All unique sequences for embedding extraction
    - metadata.json: Dataset statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable format with frequencies
    json_data = []
    for rep in tqdm(repertoire_data, desc="Preparing JSON"):
        total_count = sum(rep['counts'])
        frequencies = [c / total_count for c in rep['counts']]
        
        json_data.append({
            'repertoire_id': rep['repertoire_id'],
            'label': rep['label'],
            'sequences': rep['sequences'],
            'frequency': frequencies,
            'num_sequences': rep['unique_sequences'],
            'total_count': float(total_count)
        })
    
    # Save repertoires JSON
    json_path = output_path / 'repertoires.json'
    print(f"\nSaving repertoire data to {json_path}")
    with open(json_path, 'w') as f:
        json.dump(json_data, f)
    print(f"  Size: {os.path.getsize(json_path) / 1e6:.1f} MB")
    
    # Save unique sequences parquet
    sequences_df = pd.DataFrame({
        'sequence_id': list(unique_sequences)
    })
    parquet_path = output_path / 'unique_sequences.parquet'
    print(f"\nSaving unique sequences to {parquet_path}")
    sequences_df.to_parquet(parquet_path, index=False)
    print(f"  Sequences: {len(sequences_df):,}")
    print(f"  Size: {os.path.getsize(parquet_path) / 1e6:.1f} MB")
    
    # Save metadata
    labels = [r['label'] for r in repertoire_data]
    label_counts = pd.Series(labels).value_counts().to_dict()
    
    metadata = {
        'num_repertoires': len(repertoire_data),
        'num_unique_sequences': len(unique_sequences),
        'total_sequences': sum(r['total_sequences'] for r in repertoire_data),
        'label_distribution': {str(k): int(v) for k, v in label_counts.items()},
        'sequences_per_repertoire': {
            'min': min(r['unique_sequences'] for r in repertoire_data),
            'max': max(r['unique_sequences'] for r in repertoire_data),
            'mean': np.mean([r['unique_sequences'] for r in repertoire_data]),
        }
    }
    
    metadata_path = output_path / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDataset Statistics:")
    print(f"  Repertoires: {metadata['num_repertoires']}")
    print(f"  Unique sequences: {metadata['num_unique_sequences']:,}")
    print(f"  Total sequences: {metadata['total_sequences']:,}")
    print(f"  Overlap: {1 - metadata['num_unique_sequences']/metadata['total_sequences']:.1%}")
    print(f"  Labels: {metadata['label_distribution']}")
    
    return json_path, parquet_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MIL data from repertoire files"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing repertoire TSV files"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--labels_file", type=str, default=None,
        help="Optional TSV file with repertoire labels (repertoire_id, label)"
    )
    parser.add_argument(
        "--infer_labels", action="store_true",
        help="Try to infer labels from directory structure or filename patterns"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("PREPARE MIL DATA")
    print("="*80)
    print(f"Input: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    
    # Load repertoires
    repertoire_data, unique_sequences, sequence_counts = load_repertoires(
        args.data_dir, 
        args.labels_file
    )
    
    # Check if labels are missing
    labels = [r['label'] for r in repertoire_data]
    if all(l == -1 for l in labels):
        print("\n⚠️  WARNING: No labels found!")
        print("   Labels will be set to -1. You may need to provide a labels file.")
        print("   Use --labels_file with a TSV containing: repertoire_id<tab>label")
    
    # Save data
    json_path, parquet_path = save_mil_data(
        repertoire_data,
        unique_sequences,
        args.output_dir
    )
    
    print(f"\n{'='*80}")
    print("✅ DATA PREPARATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nNext step: Extract embeddings")
    print(f"  python scripts/mil/data/extract_embeddings_protbert.py \\")
    print(f"      --sequences_parquet {parquet_path} \\")
    print(f"      --output_parquet {args.output_dir}/embeddings.parquet \\")
    print(f"      --model_path checkpoints/protbert_cdr_dmlm_1M_balanced/best_model \\")
    print(f"      --batch_size 256 --fp16")


if __name__ == "__main__":
    main()
