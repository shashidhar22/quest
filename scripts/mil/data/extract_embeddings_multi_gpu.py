#!/usr/bin/env python3
"""
Multi-GPU embedding extraction for large-scale TCR datasets.

Uses DataParallel or DistributedDataParallel for efficient parallel processing
across multiple GPUs. Optimized for g5.12xlarge (4x A10G) or p4d.24xlarge (8x A100).
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
import gc
from pathlib import Path


class SequenceDataset(Dataset):
    """Dataset for batch processing sequences."""

    def __init__(self, sequences: list, tokenizer, max_length: int = 50):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Add spaces between amino acids for ESM tokenizer
        spaced_seq = ' '.join(list(seq))
        return spaced_seq, idx


def collate_fn(batch, tokenizer, max_length):
    """Custom collate function for tokenization."""
    sequences, indices = zip(*batch)
    encoded = tokenizer(
        list(sequences),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return encoded, list(indices)


class EmbeddingExtractor(nn.Module):
    """Wrapper for embedding extraction with mean pooling."""

    def __init__(self, model, pooling='mean'):
        super().__init__()
        self.model = model
        self.pooling = pooling

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        if self.pooling == 'mean':
            # Mean pooling over sequence length
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        elif self.pooling == 'cls':
            embeddings = hidden_states[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return embeddings


def extract_embeddings(
    sequences_parquet: str,
    output_parquet: str,
    model_path: str,
    batch_size: int = 512,
    num_workers: int = 4,
    max_length: int = 50,
    pooling: str = 'mean',
    fp16: bool = True,
    chunk_size: int = 1000000,
):
    """
    Extract embeddings for all sequences using multiple GPUs.

    Args:
        sequences_parquet: Path to parquet with sequence_id column
        output_parquet: Output path for embeddings
        model_path: Path to PEFT model checkpoint
        batch_size: Batch size per GPU
        num_workers: DataLoader workers
        max_length: Max sequence length
        pooling: Pooling method (mean/cls)
        fp16: Use FP16 inference
        chunk_size: Process in chunks for memory efficiency
    """
    print("="*80)
    print("MULTI-GPU EMBEDDING EXTRACTION")
    print("="*80)

    # Check GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available!")

    print(f"Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

    # Load sequences
    print(f"\nLoading sequences from {sequences_parquet}")
    sequences_df = pd.read_parquet(sequences_parquet)
    sequences = sequences_df['sequence_id'].tolist()
    print(f"Total sequences: {len(sequences):,}")

    # Load model
    print(f"\nLoading model from {model_path}")

    # Determine base model from config
    config_path = Path(model_path)
    if (config_path / 'adapter_config.json').exists():
        import json
        with open(config_path / 'adapter_config.json') as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get('base_model_name_or_path', 'facebook/esm2_t33_650M_UR50D')
    else:
        base_model_name = 'facebook/esm2_t33_650M_UR50D'

    print(f"Base model: {base_model_name}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModel.from_pretrained(base_model_name)

    # Load PEFT adapter if present
    if (config_path / 'adapter_config.json').exists():
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        print("Loaded and merged PEFT adapter")
    else:
        model = base_model

    # Wrap in extractor
    extractor = EmbeddingExtractor(model, pooling=pooling)

    # Convert to FP16 if requested
    if fp16:
        extractor = extractor.half()
        print("Using FP16 inference")

    # Use DataParallel for multi-GPU
    if num_gpus > 1:
        extractor = nn.DataParallel(extractor)
        print(f"Using DataParallel across {num_gpus} GPUs")
        effective_batch_size = batch_size * num_gpus
    else:
        effective_batch_size = batch_size

    extractor = extractor.cuda()
    extractor.eval()

    print(f"Effective batch size: {effective_batch_size}")

    # Process in chunks for memory efficiency
    num_chunks = (len(sequences) + chunk_size - 1) // chunk_size
    all_embeddings = []
    all_sequence_ids = []

    output_dir = Path(output_parquet).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # For incremental saving
    temp_files = []

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(sequences))
        chunk_sequences = sequences[start_idx:end_idx]

        print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks} "
              f"(sequences {start_idx:,} to {end_idx:,})")

        # Create dataset and dataloader
        dataset = SequenceDataset(chunk_sequences, tokenizer, max_length)
        dataloader = DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda b: collate_fn(b, tokenizer, max_length),
            pin_memory=True,
        )

        chunk_embeddings = []
        chunk_ids = []

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=fp16):
            for encoded, indices in tqdm(dataloader, desc=f"Chunk {chunk_idx + 1}"):
                input_ids = encoded['input_ids'].cuda()
                attention_mask = encoded['attention_mask'].cuda()

                embeddings = extractor(input_ids, attention_mask)

                # Convert to numpy
                embeddings_np = embeddings.float().cpu().numpy()
                chunk_embeddings.append(embeddings_np)

                # Get sequence IDs
                for idx in indices:
                    chunk_ids.append(chunk_sequences[idx])

        # Concatenate chunk results
        chunk_embeddings = np.vstack(chunk_embeddings)

        # Save chunk to temp file
        temp_path = output_dir / f'temp_chunk_{chunk_idx}.parquet'
        chunk_df = pd.DataFrame({
            'sequence_id': chunk_ids,
            'embedding': [emb.tolist() for emb in chunk_embeddings]
        })
        pq.write_table(pa.Table.from_pandas(chunk_df), temp_path)
        temp_files.append(temp_path)

        print(f"  Saved chunk to {temp_path}")
        print(f"  Embedding shape: {chunk_embeddings.shape}")

        # Clear memory
        del chunk_embeddings, chunk_ids, chunk_df
        gc.collect()
        torch.cuda.empty_cache()

    # Merge all chunks
    print(f"\nMerging {len(temp_files)} chunks...")
    tables = []
    for temp_path in tqdm(temp_files, desc="Loading chunks"):
        tables.append(pq.read_table(temp_path))

    merged_table = pa.concat_tables(tables)
    pq.write_table(merged_table, output_parquet, compression='snappy')

    # Clean up temp files
    for temp_path in temp_files:
        temp_path.unlink()

    # Final stats
    file_size = os.path.getsize(output_parquet) / 1e9
    print(f"\n{'='*80}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"Output: {output_parquet}")
    print(f"Size: {file_size:.2f} GB")
    print(f"Sequences: {len(sequences):,}")
    print(f"Embedding dim: {merged_table.column('embedding')[0].as_py().__len__()}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU embedding extraction for TCR sequences"
    )
    parser.add_argument("--sequences_parquet", type=str, required=True,
                       help="Parquet file with sequence_id column")
    parser.add_argument("--output_parquet", type=str, required=True,
                       help="Output parquet file for embeddings")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to PEFT model checkpoint")
    parser.add_argument("--batch_size", type=int, default=512,
                       help="Batch size per GPU (default: 512)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="DataLoader workers (default: 4)")
    parser.add_argument("--max_length", type=int, default=50,
                       help="Max sequence length (default: 50)")
    parser.add_argument("--pooling", type=str, default='mean',
                       choices=['mean', 'cls'],
                       help="Pooling method (default: mean)")
    parser.add_argument("--fp16", action='store_true',
                       help="Use FP16 inference")
    parser.add_argument("--chunk_size", type=int, default=1000000,
                       help="Process sequences in chunks (default: 1M)")

    args = parser.parse_args()

    extract_embeddings(
        args.sequences_parquet,
        args.output_parquet,
        args.model_path,
        args.batch_size,
        args.num_workers,
        args.max_length,
        args.pooling,
        args.fp16,
        args.chunk_size,
    )


if __name__ == "__main__":
    main()
