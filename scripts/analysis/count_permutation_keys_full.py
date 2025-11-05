#!/usr/bin/env python3
"""
Script to count the number of examples for each permutation_key in parquet files.
Uses streaming to avoid loading all data into memory.
This version processes the FULL database.
"""

import pyarrow.parquet as pq
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import glob
import os

def count_permutation_keys_streaming(parquet_dir):
    """
    Stream through parquet files and count examples per permutation_key.
    
    Args:
        parquet_dir: Directory containing parquet files
        
    Returns:
        Dictionary with permutation_key counts
    """
    counts = defaultdict(int)
    total_examples = 0
    
    # Get all parquet files
    parquet_files = sorted(glob.glob(f"{parquet_dir}/*.parquet"))
    print(f"Found {len(parquet_files)} parquet files to process")
    
    # Process each file
    for parquet_file in tqdm(parquet_files, desc="Processing files"):
        # Open parquet file
        parquet_file_obj = pq.ParquetFile(parquet_file)
        
        # Stream through batches
        for batch in parquet_file_obj.iter_batches(batch_size=10000):
            # Convert batch to pandas for easier processing
            df = batch.to_pandas()
            
            # Count permutation_keys in this batch
            for key in df['permutation_key']:
                counts[key] += 1
                total_examples += 1
    
    return counts, total_examples


def write_results_to_markdown(counts, total_examples, output_file, dataset_path):
    """
    Write the counting results to a markdown file.
    
    Args:
        counts: Dictionary of permutation_key counts
        total_examples: Total number of examples
        output_file: Path to output markdown file
        dataset_path: Path to the dataset
    """
    # Sort by count (descending)
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    with open(output_file, 'w') as f:
        f.write("# Permutation Key Distribution Analysis (Full Database)\n\n")
        f.write(f"**Dataset Path:** `{dataset_path}`\n\n")
        f.write(f"**Total Examples:** {total_examples:,}\n\n")
        f.write(f"**Unique Permutation Keys:** {len(counts):,}\n\n")
        f.write("---\n\n")
        
        f.write("## Summary Statistics\n\n")
        if sorted_counts:
            max_count = sorted_counts[0][1]
            min_count = sorted_counts[-1][1]
            avg_count = total_examples / len(counts)
            
            f.write(f"- **Maximum count:** {max_count:,} (key: `{sorted_counts[0][0]}`)\n")
            f.write(f"- **Minimum count:** {min_count:,} (key: `{sorted_counts[-1][0]}`)\n")
            f.write(f"- **Average count per key:** {avg_count:,.2f}\n\n")
        
        f.write("---\n\n")
        f.write("## Detailed Counts by Permutation Key\n\n")
        f.write("| Rank | Permutation Key | Count | Percentage |\n")
        f.write("|------|----------------|-------|------------|\n")
        
        for idx, (key, count) in enumerate(sorted_counts, 1):
            percentage = (count / total_examples) * 100
            f.write(f"| {idx} | `{key}` | {count:,} | {percentage:.2f}% |\n")
    
    print(f"\nResults written to: {output_file}")


def main():
    # Get project root (2 levels up from this script)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    parquet_dir = "/mnt/ephemeral/deduplicated/full/database_mlm"
    output_file = project_root / "outputs" / "permutation_key_counts_full.md"
    
    print("Starting permutation_key counting (FULL DATABASE)...")
    print(f"Input directory: {parquet_dir}")
    print(f"Output file: {output_file}\n")
    
    # Count permutation keys
    counts, total_examples = count_permutation_keys_streaming(parquet_dir)
    
    print(f"\nProcessing complete!")
    print(f"Total examples processed: {total_examples:,}")
    print(f"Unique permutation keys: {len(counts):,}")
    
    # Write results to markdown
    write_results_to_markdown(counts, total_examples, output_file, parquet_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
