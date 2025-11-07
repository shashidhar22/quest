#!/usr/bin/env python3
"""
prepare_repertoire_data.py
────────────────────────────────────────────────────────
Helper script to prepare TCR repertoire data for MIL analysis.

Converts various repertoire formats into the format expected by repertoire_mil.py
"""

import argparse
import json
import pandas as pd
from typing import List, Dict, Any
from collections import defaultdict


def prepare_from_csv(
    input_path: str,
    output_path: str,
    repertoire_col: str = "repertoire_id",
    sequence_col: str = "sequence",
    label_col: str = "label"
) -> None:
    """
    Prepare data from CSV with columns: [repertoire_id, sequence, label]
    
    Args:
        input_path: Path to input CSV
        output_path: Path to output JSON
        repertoire_col: Name of repertoire ID column
        sequence_col: Name of sequence column
        label_col: Name of label column
    """
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"Found columns: {list(df.columns)}")
    
    # Check required columns exist
    required = [repertoire_col, sequence_col, label_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Group by repertoire
    repertoire_data = []
    for rep_id, group in df.groupby(repertoire_col):
        sequences = group[sequence_col].dropna().unique().tolist()
        
        # Get label (should be same for all sequences in repertoire)
        labels = group[label_col].unique()
        if len(labels) > 1:
            print(f"Warning: Repertoire {rep_id} has multiple labels: {labels}. Using first.")
        label = labels[0]
        
        repertoire_data.append({
            'repertoire_id': str(rep_id),
            'sequences': sequences,
            'label': int(label) if isinstance(label, (int, float)) else str(label)
        })
    
    print(f"\nProcessed {len(repertoire_data)} repertoires")
    
    # Print statistics
    seq_counts = [len(r['sequences']) for r in repertoire_data]
    print(f"Sequences per repertoire:")
    print(f"  Min: {min(seq_counts)}")
    print(f"  Max: {max(seq_counts)}")
    print(f"  Mean: {sum(seq_counts)/len(seq_counts):.1f}")
    print(f"  Median: {sorted(seq_counts)[len(seq_counts)//2]}")
    
    label_counts = {}
    for r in repertoire_data:
        label = r['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    # Save
    print(f"\nSaving to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(repertoire_data, f, indent=2)
    
    print("✅ Done!")


def prepare_from_immuneaccess(
    input_path: str,
    output_path: str,
    label_mapping: Dict[str, int],
    chain_type: str = "TRB",
    min_sequences: int = 10
) -> None:
    """
    Prepare data from ImmuneACCESS format.
    
    Args:
        input_path: Path to ImmuneACCESS TSV file
        output_path: Path to output JSON
        label_mapping: Dict mapping sample names to labels
        chain_type: TCR chain type to extract (TRA/TRB)
        min_sequences: Minimum sequences per repertoire
    """
    print(f"Loading ImmuneACCESS data from: {input_path}")
    df = pd.read_csv(input_path, sep='\t')
    
    print(f"Columns: {list(df.columns)}")
    
    # Extract amino acid sequences
    if 'amino_acid' in df.columns:
        seq_col = 'amino_acid'
    elif 'aminoAcid' in df.columns:
        seq_col = 'aminoAcid'
    else:
        raise ValueError("Could not find amino acid sequence column")
    
    # Extract sample ID
    if 'sample_name' in df.columns:
        sample_col = 'sample_name'
    elif 'repertoire_id' in df.columns:
        sample_col = 'repertoire_id'
    else:
        raise ValueError("Could not find sample/repertoire ID column")
    
    # Filter by chain type if specified
    if 'locus' in df.columns:
        df = df[df['locus'] == chain_type]
        print(f"Filtered to {chain_type} chains: {len(df)} sequences")
    
    # Group by sample
    repertoire_data = []
    for sample_id, group in df.groupby(sample_col):
        sequences = group[seq_col].dropna().unique().tolist()
        
        if len(sequences) < min_sequences:
            continue
        
        # Get label
        if sample_id in label_mapping:
            label = label_mapping[sample_id]
        else:
            print(f"Warning: No label for sample {sample_id}, skipping")
            continue
        
        repertoire_data.append({
            'repertoire_id': str(sample_id),
            'sequences': sequences,
            'label': label
        })
    
    print(f"\nProcessed {len(repertoire_data)} repertoires")
    
    # Save
    print(f"Saving to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(repertoire_data, f, indent=2)
    
    print("✅ Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare TCR repertoire data for MIL analysis"
    )
    
    parser.add_argument("--input", type=str, required=True,
                        help="Input data file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file")
    parser.add_argument("--format", type=str, default="csv",
                        choices=["csv", "immuneaccess"],
                        help="Input format")
    
    # For CSV format
    parser.add_argument("--repertoire_col", type=str, default="repertoire_id",
                        help="Repertoire ID column name (for CSV)")
    parser.add_argument("--sequence_col", type=str, default="sequence",
                        help="Sequence column name (for CSV)")
    parser.add_argument("--label_col", type=str, default="label",
                        help="Label column name (for CSV)")
    
    # For ImmuneACCESS format
    parser.add_argument("--label_file", type=str, default=None,
                        help="JSON file with {sample_id: label} mapping (for ImmuneACCESS)")
    parser.add_argument("--chain_type", type=str, default="TRB",
                        help="Chain type to extract (for ImmuneACCESS)")
    parser.add_argument("--min_sequences", type=int, default=10,
                        help="Minimum sequences per repertoire")
    
    args = parser.parse_args()
    
    if args.format == "csv":
        prepare_from_csv(
            input_path=args.input,
            output_path=args.output,
            repertoire_col=args.repertoire_col,
            sequence_col=args.sequence_col,
            label_col=args.label_col
        )
    
    elif args.format == "immuneaccess":
        if not args.label_file:
            raise ValueError("--label_file required for ImmuneACCESS format")
        
        with open(args.label_file, 'r') as f:
            label_mapping = json.load(f)
        
        prepare_from_immuneaccess(
            input_path=args.input,
            output_path=args.output,
            label_mapping=label_mapping,
            chain_type=args.chain_type,
            min_sequences=args.min_sequences
        )


if __name__ == "__main__":
    main()
