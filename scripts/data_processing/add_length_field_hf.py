import os
import argparse
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk
import numpy as np

def process_dataset_shard(input_path: Path, output_path: Path):
    """
    Loads a HF dataset from input_path, adds a 'length' field calculated from 
    'attention_mask', and saves it to output_path.
    """
    try:
        ds = load_from_disk(str(input_path))
        
        if 'attention_mask' not in ds.column_names:
            print(f"Warning: 'attention_mask' not found in {input_path}. Copying as is.")
            ds.save_to_disk(str(output_path))
            return

        # Add length field
        # Use map for efficiency, but for simple sum it might be faster to just add the column if it's small
        # however map handles the arrow backend well.
        ds_with_length = ds.map(
            lambda x: {"length": sum(x["attention_mask"])},
            num_proc=os.cpu_count() // 2 if os.cpu_count() > 1 else 1,
            desc=f"Adding length to {input_path.name}"
        )
        
        ds_with_length.save_to_disk(str(output_path))
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Add length field to HF sharded dataset.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to base dataset directory (e.g. foundation_100M)")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save processed dataset")
    args = parser.parse_args()

    base_input = Path(args.input_dir)
    base_output = Path(args.output_dir)

    # We expect splits like train, test, validation
    splits = [d for d in base_input.iterdir() if d.is_dir()]
    
    for split_dir in splits:
        split_name = split_dir.name
        print(f"Processing split: {split_name}")
        
        # We expect shard directories like shard_batch_XXXXXX
        shard_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith("shard_batch_")])
        
        if not shard_dirs:
            # Maybe the split_dir itself is a dataset?
            if (split_dir / "dataset_info.json").exists():
                print(f"  {split_name} is a direct dataset.")
                process_dataset_shard(split_dir, base_output / split_name)
            else:
                print(f"  No shards found in {split_dir}. Skipping.")
            continue

        for shard_dir in tqdm(shard_dirs, desc=f"Shards in {split_name}"):
            output_shard_path = base_output / split_name / shard_dir.name
            if output_shard_path.exists():
                continue
            process_dataset_shard(shard_dir, output_shard_path)

if __name__ == "__main__":
    main()
