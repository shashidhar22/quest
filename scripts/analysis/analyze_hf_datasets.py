#!/usr/bin/env python3
"""
Analyze HuggingFace datasets to get data breakdown by permutation_key.
Streams data to avoid loading everything into memory.
"""

import argparse
from pathlib import Path
from collections import defaultdict
from datasets import load_from_disk
from tqdm import tqdm


def analyze_hf_dataset(dataset_path: Path) -> tuple:
    """
    Analyze a HuggingFace dataset and count examples per permutation_key.
    Streams data in batches to avoid loading all into memory.
    
    Args:
        dataset_path: Path to the HuggingFace dataset directory
        
    Returns:
        Tuple of (counts_dict, total_examples, dataset_info)
    """
    print(f"\n📊 Analyzing: {dataset_path}")
    
    # Load dataset (this just loads metadata, not all data)
    try:
        dataset = load_from_disk(str(dataset_path))
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return {}, 0, {}
    
    # Get dataset info
    dataset_info = {
        'num_rows': len(dataset),
        'features': list(dataset.features.keys()),
        'splits': list(dataset.keys()) if hasattr(dataset, 'keys') else ['train']
    }
    
    # Check if it's a DatasetDict or single Dataset
    if hasattr(dataset, 'keys'):
        # It's a DatasetDict, analyze all splits
        all_counts = defaultdict(int)
        total_examples = 0
        
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            print(f"   Processing split: {split_name} ({len(split_data):,} rows)")
            
            # Check if permutation_key exists
            if 'permutation_key' not in split_data.features:
                print(f"   ⚠️  No 'permutation_key' column found in {split_name}")
                continue
            
            # Stream through data in batches
            batch_size = 10000
            num_rows = len(split_data)
            
            with tqdm(total=num_rows, desc=f"   {split_name}", unit=" examples") as pbar:
                for i in range(0, num_rows, batch_size):
                    # Select only the permutation_key column to minimize memory
                    batch = split_data.select(range(i, min(i + batch_size, num_rows)))
                    
                    # Process batch
                    for example in batch:
                        perm_key = example.get('permutation_key', 'unknown')
                        all_counts[perm_key] += 1
                        total_examples += 1
                    
                    pbar.update(len(batch))
        
        return all_counts, total_examples, dataset_info
    else:
        # Single dataset
        print(f"   Processing dataset ({len(dataset):,} rows)")
        
        # Check if permutation_key exists
        if 'permutation_key' not in dataset.features:
            print(f"   ⚠️  No 'permutation_key' column found")
            return {}, 0, dataset_info
        
        counts = defaultdict(int)
        
        # Stream through data in batches
        batch_size = 10000
        num_rows = len(dataset)
        
        with tqdm(total=num_rows, desc="   Processing", unit=" examples") as pbar:
            for i in range(0, num_rows, batch_size):
                # Select only the permutation_key column to minimize memory
                batch = dataset.select(range(i, min(i + batch_size, num_rows)))
                
                # Process batch
                for example in batch:
                    perm_key = example.get('permutation_key', 'unknown')
                    counts[perm_key] += 1
                
                pbar.update(len(batch))
        
        return counts, num_rows, dataset_info


def write_analysis_to_markdown(dataset_analyses: dict, output_file: Path):
    """
    Write the analysis results to a markdown file.
    
    Args:
        dataset_analyses: Dict mapping dataset paths to (counts, total, info)
        output_file: Output markdown file path
    """
    with open(output_file, 'w') as f:
        f.write("# HuggingFace Dataset Analysis\n\n")
        f.write("**Analysis Date:** November 5, 2025\n\n")
        f.write("**Base Path:** `/mnt/ephemeral/tokenized/phase_zero/database/1M/`\n\n")
        f.write("---\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        f.write(f"**Total Datasets Analyzed:** {len(dataset_analyses)}\n\n")
        
        grand_total = sum(total for _, total, _ in dataset_analyses.values())
        f.write(f"**Grand Total Examples:** {grand_total:,}\n\n")
        f.write("---\n\n")
        
        # Detailed analysis for each dataset
        for dataset_path, (counts, total, info) in dataset_analyses.items():
            dataset_name = dataset_path.name
            relative_path = str(dataset_path).replace('/mnt/ephemeral/tokenized/phase_zero/database/1M/', '')
            
            f.write(f"## {dataset_name}\n\n")
            f.write(f"**Path:** `{relative_path}`\n\n")
            f.write(f"**Total Examples:** {total:,}\n\n")
            f.write(f"**Features:** {', '.join(f'`{feat}`' for feat in info.get('features', []))}\n\n")
            
            if counts:
                f.write(f"**Unique Permutation Keys:** {len(counts):,}\n\n")
                
                # Sort by count
                sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                
                # Statistics
                max_count = sorted_counts[0][1]
                min_count = sorted_counts[-1][1]
                avg_count = total / len(counts) if counts else 0
                
                f.write("### Statistics\n\n")
                f.write(f"- **Maximum count:** {max_count:,} (key: `{sorted_counts[0][0]}`)\n")
                f.write(f"- **Minimum count:** {min_count:,} (key: `{sorted_counts[-1][0]}`)\n")
                f.write(f"- **Average count per key:** {avg_count:,.2f}\n\n")
                
                # Detailed breakdown
                f.write("### Permutation Key Breakdown\n\n")
                f.write("| Rank | Permutation Key | Count | Percentage |\n")
                f.write("|------|----------------|-------|------------|\n")
                
                for idx, (key, count) in enumerate(sorted_counts, 1):
                    percentage = (count / total) * 100 if total > 0 else 0
                    f.write(f"| {idx} | `{key}` | {count:,} | {percentage:.2f}% |\n")
            else:
                f.write("⚠️ No permutation_key data found\n")
            
            f.write("\n---\n\n")
    
    print(f"\n✅ Results written to: {output_file}")


def main():
    # Get project root (2 levels up from this script)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    base_path = Path("/mnt/ephemeral/tokenized/phase_zero/database/1M/")
    output_file = project_root / "outputs" / "hf_dataset_analysis.md"
    
    print("🔍 Searching for HuggingFace datasets...")
    
    # Find all dataset directories (they typically have dataset_info.json)
    dataset_paths = []
    for path in base_path.rglob("*"):
        if path.is_dir():
            # Check if it's a HF dataset (has dataset_info.json or similar markers)
            if (path / "dataset_info.json").exists() or (path / "state.json").exists():
                dataset_paths.append(path)
    
    if not dataset_paths:
        print("❌ No HuggingFace datasets found!")
        return
    
    print(f"📦 Found {len(dataset_paths)} dataset(s):\n")
    for path in dataset_paths:
        print(f"   - {path}")
    
    # Analyze each dataset
    dataset_analyses = {}
    for dataset_path in dataset_paths:
        counts, total, info = analyze_hf_dataset(dataset_path)
        dataset_analyses[dataset_path] = (counts, total, info)
    
    # Write results
    write_analysis_to_markdown(dataset_analyses, output_file)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
