#!/usr/bin/env python3
"""
export_embeddings.py
────────────────────────────────────────────────────────
Export embeddings for all sequences in the dataset to a file.
This is useful for downstream analysis, clustering, or similarity searches.
"""

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import PeftModel
import numpy as np
from datasets import load_from_disk
import argparse
from tqdm.auto import tqdm
import h5py
import json


def format_seq(seq):
    """Add spaces between amino acids for ProtBERT."""
    return " ".join(list(seq.replace(" ", "")))


def decode_sequence(input_ids, tokenizer):
    """Decode tokenized sequence back to amino acid string."""
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    # Remove special tokens and join
    aa_sequence = ''.join([t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]', '[MASK]']])
    return aa_sequence


def get_embeddings_batch(model, tokenizer, sequences, device, pooling="mean", batch_size=32):
    """
    Get embeddings for a batch of sequences.
    
    Args:
        model: PEFT model
        tokenizer: Tokenizer
        sequences: List of protein sequences
        device: Device for inference
        pooling: Pooling strategy ('mean', 'max', or 'cls')
        batch_size: Batch size for processing
    
    Returns:
        Numpy array of embeddings
    """
    all_embeddings = []
    
    for i in tqdm(range(0, len(sequences), batch_size), desc="Computing embeddings"):
        batch_seqs = sequences[i:i+batch_size]
        formatted_seqs = [format_seq(s) for s in batch_seqs]
        
        # Tokenize batch
        inputs = tokenizer(
            formatted_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model.base_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            
            if pooling == "mean":
                mask = inputs['attention_mask'].unsqueeze(-1)
                embeddings = (hidden_states * mask).sum(1) / mask.sum(1)
            elif pooling == "max":
                embeddings = hidden_states.max(1)[0]
            elif pooling == "cls":
                embeddings = hidden_states[:, 0]
        
        all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)


def main():
    parser = argparse.ArgumentParser(description="Export embeddings from PEFT model")
    parser.add_argument("--model_path", type=str, default="/mnt/ephemeral/protbert_01_specifcity",
                       help="Path to PEFT adapter")
    parser.add_argument("--dataset_path", type=str, default="/mnt/ephemeral/protbert_spec",
                       help="Path to dataset")
    parser.add_argument("--split", type=str, default="val",
                       choices=["train", "val", "test"],
                       help="Dataset split to use")
    parser.add_argument("--output", type=str, default="embeddings.h5",
                       help="Output file path (.h5 or .npz)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for inference")
    parser.add_argument("--pooling", type=str, default="mean",
                       choices=["mean", "max", "cls"],
                       help="Pooling strategy")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process (for testing)")
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    
    args = parser.parse_args()
    
    print("="*80)
    print("EMBEDDING EXPORT")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Split: {args.split}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Pooling: {args.pooling}")
    print()
    
    # Load model
    print("Loading model...")
    base_model = AutoModelForMaskedLM.from_pretrained("Rostlab/prot_bert_bfd")
    model = PeftModel.from_pretrained(base_model, args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = model.to(args.device)
    model.eval()
    print("✅ Model loaded!\n")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_from_disk(args.dataset_path)
    split_data = dataset[args.split]
    
    if args.max_samples:
        split_data = split_data.select(range(min(args.max_samples, len(split_data))))
    
    print(f"✅ Dataset loaded! ({len(split_data)} samples)\n")
    
    # Extract sequences
    print("Extracting sequences...")
    sequences = []
    metadata = []
    
    for i, sample in enumerate(tqdm(split_data)):
        # Decode sequence
        if 'input_ids' in sample:
            seq = decode_sequence(sample['input_ids'], tokenizer)
        elif 'sequence' in sample:
            seq = sample['sequence']
        elif 'text' in sample:
            seq = sample['text']
        else:
            seq = None
        
        if seq and len(seq) > 0:
            sequences.append(seq)
            
            # Store metadata
            meta = {'index': i}
            if 'combo_id' in sample:
                meta['combo_id'] = sample['combo_id']
            if 'combo_feats' in sample:
                meta['combo_feats'] = sample['combo_feats']
            metadata.append(meta)
    
    print(f"✅ Extracted {len(sequences)} sequences\n")
    
    # Compute embeddings
    print("Computing embeddings...")
    embeddings = get_embeddings_batch(
        model, tokenizer, sequences,
        args.device, args.pooling, args.batch_size
    )
    print(f"✅ Computed embeddings! Shape: {embeddings.shape}\n")
    
    # Save embeddings
    print(f"Saving to {args.output}...")
    
    if args.output.endswith('.h5'):
        # Save as HDF5
        with h5py.File(args.output, 'w') as f:
            f.create_dataset('embeddings', data=embeddings)
            f.create_dataset('sequences', data=np.array(sequences, dtype='S'))
            
            # Save metadata as JSON string
            f.attrs['metadata'] = json.dumps(metadata)
            f.attrs['pooling'] = args.pooling
            f.attrs['model_path'] = args.model_path
            f.attrs['dataset_path'] = args.dataset_path
            f.attrs['split'] = args.split
            
        print(f"✅ Saved {len(embeddings)} embeddings to {args.output} (HDF5 format)")
        
    elif args.output.endswith('.npz'):
        # Save as NPZ
        np.savez(
            args.output,
            embeddings=embeddings,
            sequences=np.array(sequences),
            metadata=json.dumps(metadata),
            pooling=args.pooling,
            model_path=args.model_path,
            dataset_path=args.dataset_path,
            split=args.split
        )
        print(f"✅ Saved {len(embeddings)} embeddings to {args.output} (NPZ format)")
        
    else:
        # Save as plain numpy array
        np.save(args.output, embeddings)
        
        # Save metadata separately
        metadata_path = args.output.replace('.npy', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'sequences': sequences,
                'metadata': metadata,
                'pooling': args.pooling,
                'model_path': args.model_path,
                'dataset_path': args.dataset_path,
                'split': args.split,
                'shape': embeddings.shape
            }, f, indent=2)
        
        print(f"✅ Saved {len(embeddings)} embeddings to {args.output}")
        print(f"✅ Saved metadata to {metadata_path}")
    
    print()
    print("="*80)
    print("EXPORT COMPLETE!")
    print("="*80)
    print("\nTo load the embeddings:")
    if args.output.endswith('.h5'):
        print(f"""
import h5py
import json

with h5py.File('{args.output}', 'r') as f:
    embeddings = f['embeddings'][:]
    sequences = [s.decode() for s in f['sequences'][:]]
    metadata = json.loads(f.attrs['metadata'])
""")
    elif args.output.endswith('.npz'):
        print(f"""
import numpy as np
import json

data = np.load('{args.output}', allow_pickle=True)
embeddings = data['embeddings']
sequences = data['sequences'].tolist()
metadata = json.loads(str(data['metadata']))
""")
    else:
        print(f"""
import numpy as np
import json

embeddings = np.load('{args.output}')
with open('{args.output.replace('.npy', '_metadata.json')}', 'r') as f:
    metadata = json.load(f)
sequences = metadata['sequences']
""")


if __name__ == "__main__":
    main()
