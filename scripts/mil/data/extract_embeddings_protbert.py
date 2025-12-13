#!/usr/bin/env python3
"""
Embedding extraction for ProtBERT models.

Supports both:
- Rostlab/prot_bert (original ProtBERT)
- Fine-tuned ProtBERT with PEFT/LoRA adapters

ProtBERT uses spaced amino acid tokenization (e.g., "C A S S L G G")
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
from transformers import AutoTokenizer, AutoModel, BertModel, BertTokenizer
from peft import PeftModel
import gc
from pathlib import Path
import json


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
        # ProtBERT expects spaces between amino acids
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
    """Wrapper for embedding extraction with multiple pooling options."""

    def __init__(self, model, pooling='mean'):
        super().__init__()
        self.model = model
        self.pooling = pooling

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        if self.pooling == 'mean':
            # Mean pooling over sequence length (excluding padding)
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        elif self.pooling == 'cls':
            # Use [CLS] token embedding (first token)
            embeddings = hidden_states[:, 0, :]
        elif self.pooling == 'max':
            # Max pooling over sequence length
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states[mask == 0] = float('-inf')
            embeddings = torch.max(hidden_states, dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return embeddings


def load_protbert_model(model_path: str, device: str = 'cuda'):
    """
    Load ProtBERT model, handling both base models and PEFT adapters.
    
    Args:
        model_path: Path to model checkpoint or PEFT adapter
        device: Device to load model on
        
    Returns:
        model, tokenizer
    """
    model_path = Path(model_path)
    
    # Check if this is a PEFT adapter
    adapter_config_path = model_path / 'adapter_config.json'
    
    if adapter_config_path.exists():
        print(f"Loading PEFT adapter from {model_path}")
        
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        
        base_model_name = adapter_config.get('base_model_name_or_path', 'Rostlab/prot_bert')
        print(f"Base model: {base_model_name}")
        
        # Load tokenizer - prefer local if available
        tokenizer_path = model_path if (model_path / 'tokenizer_config.json').exists() else base_model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, do_lower_case=False)
        
        # Load base model
        # For ProtBERT with MaskedLM, we need just the base encoder
        base_model = BertModel.from_pretrained(base_model_name)
        
        # Load and merge PEFT adapter
        # Note: The adapter was trained on BertForMaskedLM, but we can still load it
        # We need to handle this carefully
        try:
            model = PeftModel.from_pretrained(base_model, str(model_path))
            model = model.merge_and_unload()
            print("✓ Loaded and merged PEFT adapter")
        except Exception as e:
            print(f"Warning: Could not load PEFT adapter directly: {e}")
            print("Attempting to load with MaskedLM wrapper...")
            
            from transformers import BertForMaskedLM
            base_mlm = BertForMaskedLM.from_pretrained(base_model_name)
            peft_mlm = PeftModel.from_pretrained(base_mlm, str(model_path))
            peft_mlm = peft_mlm.merge_and_unload()
            # Extract just the BERT encoder
            model = peft_mlm.bert
            print("✓ Loaded adapter via MaskedLM wrapper")
    else:
        # Direct model path (not PEFT)
        print(f"Loading model directly from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=False)
        model = AutoModel.from_pretrained(model_path)
    
    return model, tokenizer


def extract_embeddings(
    sequences_parquet: str,
    output_parquet: str,
    model_path: str,
    batch_size: int = 256,
    num_workers: int = 4,
    max_length: int = 50,
    pooling: str = 'mean',
    fp16: bool = True,
    chunk_size: int = 500000,
):
    """
    Extract embeddings for all sequences using ProtBERT.

    Args:
        sequences_parquet: Path to parquet with sequence_id column
        output_parquet: Output path for embeddings
        model_path: Path to ProtBERT model or PEFT checkpoint
        batch_size: Batch size per GPU
        num_workers: DataLoader workers
        max_length: Max sequence length (in tokens, ~2x amino acid count due to spacing)
        pooling: Pooling method (mean/cls/max)
        fp16: Use FP16 inference
        chunk_size: Process in chunks for memory efficiency
    """
    print("="*80)
    print("PROTBERT EMBEDDING EXTRACTION")
    print("="*80)

    # Check GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("Warning: No GPUs available, using CPU (this will be slow)")
        device = 'cpu'
    else:
        device = 'cuda'
        print(f"Available GPUs: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

    # Load sequences
    print(f"\nLoading sequences from {sequences_parquet}")
    sequences_df = pd.read_parquet(sequences_parquet)
    
    # Handle both 'sequence_id' and 'sequence' column names
    if 'sequence_id' in sequences_df.columns:
        sequences = sequences_df['sequence_id'].tolist()
    elif 'sequence' in sequences_df.columns:
        sequences = sequences_df['sequence'].tolist()
    else:
        raise ValueError(f"Expected 'sequence_id' or 'sequence' column. Found: {sequences_df.columns.tolist()}")
    
    del sequences_df
    gc.collect()
    print(f"Total sequences: {len(sequences):,}")

    # Load model
    print(f"\nLoading model from {model_path}")
    model, tokenizer = load_protbert_model(model_path, device)
    
    # Get embedding dimension
    embedding_dim = model.config.hidden_size
    print(f"Embedding dimension: {embedding_dim}")

    # Wrap in extractor
    extractor = EmbeddingExtractor(model, pooling=pooling)

    # Convert to FP16 if requested and on GPU
    if fp16 and device == 'cuda':
        extractor = extractor.half()
        print("Using FP16 inference")

    # Use DataParallel for multi-GPU
    if num_gpus > 1:
        extractor = nn.DataParallel(extractor)
        print(f"Using DataParallel across {num_gpus} GPUs")
        effective_batch_size = batch_size * num_gpus
    else:
        effective_batch_size = batch_size

    extractor = extractor.to(device)
    extractor.eval()

    print(f"Effective batch size: {effective_batch_size}")

    # Process in chunks for memory efficiency
    num_chunks = (len(sequences) + chunk_size - 1) // chunk_size

    output_dir = Path(output_parquet).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stream directly to output file using ParquetWriter
    writer = None
    total_written = 0

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
            pin_memory=True if device == 'cuda' else False,
        )

        chunk_embeddings = []
        chunk_ids = []

        with torch.no_grad():
            if device == 'cuda' and fp16:
                context = torch.cuda.amp.autocast()
            else:
                context = torch.no_grad()  # dummy context
                
            with context:
                for encoded, indices in tqdm(dataloader, desc=f"Chunk {chunk_idx + 1}"):
                    input_ids = encoded['input_ids'].to(device)
                    attention_mask = encoded['attention_mask'].to(device)

                    embeddings = extractor(input_ids, attention_mask)

                    # Convert to numpy
                    embeddings_np = embeddings.float().cpu().numpy()
                    chunk_embeddings.append(embeddings_np)

                    # Get sequence IDs
                    for idx in indices:
                        chunk_ids.append(chunk_sequences[idx])

        # Concatenate chunk results
        chunk_embeddings = np.vstack(chunk_embeddings)

        # Create PyArrow table for this chunk
        flat_embeddings = chunk_embeddings.flatten()
        embedding_array = pa.FixedSizeListArray.from_arrays(
            pa.array(flat_embeddings, type=pa.float32()),
            embedding_dim
        )

        chunk_table = pa.table({
            'sequence_id': pa.array(chunk_ids, type=pa.string()),
            'embedding': embedding_array
        })

        # Initialize writer on first chunk
        if writer is None:
            writer = pq.ParquetWriter(
                output_parquet,
                chunk_table.schema,
                compression='snappy'
            )

        # Stream write this chunk
        writer.write_table(chunk_table)
        total_written += len(chunk_ids)

        print(f"  Streamed {len(chunk_ids):,} embeddings to file")
        print(f"  Embedding shape: {chunk_embeddings.shape}")

        # Clear memory
        del chunk_embeddings, chunk_ids, chunk_table, embedding_array, flat_embeddings
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

    # Close the writer
    if writer is not None:
        writer.close()

    # Final stats
    file_size = os.path.getsize(output_parquet) / 1e9
    print(f"\n{'='*80}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"Output: {output_parquet}")
    print(f"Size: {file_size:.2f} GB")
    print(f"Sequences: {total_written:,}")
    print(f"Embedding dim: {embedding_dim}")
    
    return output_parquet


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from ProtBERT models for MIL training"
    )
    parser.add_argument("--sequences_parquet", type=str, required=True,
                       help="Parquet file with sequence_id column")
    parser.add_argument("--output_parquet", type=str, required=True,
                       help="Output parquet file for embeddings")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to ProtBERT model or PEFT checkpoint")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size per GPU (default: 256, ProtBERT is larger than ESM2)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="DataLoader workers (default: 4)")
    parser.add_argument("--max_length", type=int, default=100,
                       help="Max sequence length in tokens (default: 100, ~50 amino acids)")
    parser.add_argument("--pooling", type=str, default='mean',
                       choices=['mean', 'cls', 'max'],
                       help="Pooling method (default: mean)")
    parser.add_argument("--fp16", action='store_true', default=True,
                       help="Use FP16 inference (default: True)")
    parser.add_argument("--no_fp16", action='store_false', dest='fp16',
                       help="Disable FP16 inference")
    parser.add_argument("--chunk_size", type=int, default=500000,
                       help="Process sequences in chunks (default: 500k)")

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
