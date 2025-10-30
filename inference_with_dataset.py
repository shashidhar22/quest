#!/usr/bin/env python3
"""
inference_with_dataset.py
────────────────────────────────────────────────────────
Run inference on sequences from the protbert_spec dataset.
This script loads sequences from your dataset and runs various inference tasks.
"""

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import PeftModel
import numpy as np
from datasets import load_from_disk
import random


# ============================================================================
# Configuration
# ============================================================================
MODEL_PATH = "/mnt/ephemeral/protbert_01_specifcity"
BASE_MODEL = "Rostlab/prot_bert_bfd"
DATASET_PATH = "/mnt/ephemeral/protbert_spec"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Number of samples to use for examples
NUM_SAMPLES = 10


# ============================================================================
# Load Model
# ============================================================================
print("="*80)
print("LOADING MODEL")
print("="*80)
print(f"Base model: {BASE_MODEL}")
print(f"Adapter: {MODEL_PATH}")
print(f"Device: {DEVICE}")

base_model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = model.to(DEVICE)
model.eval()

print("✅ Model loaded!\n")


# ============================================================================
# Load Dataset
# ============================================================================
print("="*80)
print("LOADING DATASET")
print("="*80)
print(f"Dataset path: {DATASET_PATH}")

dataset = load_from_disk(DATASET_PATH)

print(f"Dataset splits: {list(dataset.keys())}")
print(f"Train size: {len(dataset['train'])}")
print(f"Val size: {len(dataset['val'])}")
print(f"Test size: {len(dataset['test'])}")
print(f"\nDataset columns: {dataset['train'].column_names}")
print(f"Dataset features: {dataset['train'].features}")
print("✅ Dataset loaded!\n")


# ============================================================================
# Helper Functions
# ============================================================================

def format_seq(seq):
    """Add spaces between amino acids for ProtBERT."""
    return " ".join(list(seq.replace(" ", "")))


def decode_sequence(input_ids, tokenizer):
    """Decode tokenized sequence back to amino acid string."""
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    # Remove special tokens and join
    aa_sequence = ''.join([t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]', '[MASK]']])
    return aa_sequence


def predict_masked(sequence, mask_positions, top_k=5):
    """
    Predict amino acids at masked positions.
    
    Args:
        sequence: Protein sequence (e.g., "CASSLAPGATNEKLFF")
        mask_positions: List of positions to mask (0-indexed)
        top_k: Number of top predictions to return
    """
    # Format and mask sequence
    seq_formatted = format_seq(sequence)
    seq_list = seq_formatted.split()
    
    original_aas = []
    for pos in mask_positions:
        if pos < len(seq_list):
            original_aas.append(seq_list[pos])
            seq_list[pos] = tokenizer.mask_token
    
    masked_seq = " ".join(seq_list)
    
    print(f"Original: {seq_formatted}")
    print(f"Masked:   {masked_seq}")
    print()
    
    # Tokenize and predict
    inputs = tokenizer(masked_seq, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
    
    # Get predictions for each masked position
    mask_token_id = tokenizer.mask_token_id
    mask_indices = (inputs['input_ids'] == mask_token_id).nonzero(as_tuple=True)[1]
    
    results = []
    for i, (pos, idx, original_aa) in enumerate(zip(mask_positions, mask_indices, original_aas)):
        print(f"Position {pos} (original: {original_aa}):")
        
        # Get probabilities
        probs = torch.softmax(predictions[0, idx], dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
        
        pred_list = []
        for rank, (prob, token_id) in enumerate(zip(top_k_probs, top_k_indices), 1):
            token = tokenizer.decode([token_id]).strip()
            pred_list.append({'token': token, 'prob': prob.item()})
            marker = "✓" if token == original_aa else " "
            print(f"  {rank}. {token:3s} - {prob.item()*100:5.2f}% {marker}")
        
        results.append({
            'position': pos,
            'original': original_aa,
            'predictions': pred_list
        })
        print()
    
    return results


def get_embedding(sequence, pooling="mean"):
    """
    Get sequence embedding.
    
    Args:
        sequence: Protein sequence
        pooling: 'mean', 'max', or 'cls'
    
    Returns:
        Embedding vector (numpy array)
    """
    seq_formatted = format_seq(sequence)
    
    inputs = tokenizer(seq_formatted, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.base_model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        
        if pooling == "mean":
            mask = inputs['attention_mask'].unsqueeze(-1)
            embedding = (hidden_states * mask).sum(1) / mask.sum(1)
        elif pooling == "max":
            embedding = hidden_states.max(1)[0]
        elif pooling == "cls":
            embedding = hidden_states[:, 0]
    
    return embedding.cpu().numpy()[0]


def compute_similarity(seq1, seq2):
    """Compute cosine similarity between two sequences."""
    emb1 = get_embedding(seq1)
    emb2 = get_embedding(seq2)
    
    # Cosine similarity
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity


# ============================================================================
# Get Sample Sequences from Dataset
# ============================================================================
print("="*80)
print("EXTRACTING SAMPLE SEQUENCES")
print("="*80)

# Get random samples from validation set
val_dataset = dataset['val']
sample_indices = random.sample(range(len(val_dataset)), min(NUM_SAMPLES, len(val_dataset)))

sample_sequences = []
for idx in sample_indices:
    sample = val_dataset[idx]
    
    # Try to extract the actual sequence
    if 'input_ids' in sample:
        # Decode from tokens
        sequence = decode_sequence(sample['input_ids'], tokenizer)
    elif 'sequence' in sample:
        sequence = sample['sequence']
    elif 'text' in sample:
        sequence = sample['text']
    else:
        # Try to decode from first available tensor
        for key in sample.keys():
            if isinstance(sample[key], (list, np.ndarray)) and len(sample[key]) > 0:
                try:
                    sequence = decode_sequence(sample[key], tokenizer)
                    break
                except:
                    continue
    
    if sequence:
        sample_sequences.append(sequence)

# Remove duplicates and empty sequences
sample_sequences = list(set([s for s in sample_sequences if s and len(s) > 0]))[:NUM_SAMPLES]

print(f"Extracted {len(sample_sequences)} unique sequences:")
for i, seq in enumerate(sample_sequences[:5], 1):
    print(f"  {i}. {seq[:50]}..." if len(seq) > 50 else f"  {i}. {seq}")
print(f"  ... and {len(sample_sequences) - 5} more\n")


# ============================================================================
# Example 1: Masked Prediction on Dataset Sequences
# ============================================================================
print("="*80)
print("EXAMPLE 1: Masked Amino Acid Prediction from Dataset")
print("="*80)
print()

for i, sequence in enumerate(sample_sequences[:3], 1):
    print(f"{'─'*80}")
    print(f"Sample {i}")
    print(f"{'─'*80}")
    
    # Mask a few positions (e.g., 25%, 50%, 75% through the sequence)
    seq_len = len(sequence)
    mask_positions = [seq_len // 4, seq_len // 2, 3 * seq_len // 4]
    mask_positions = [p for p in mask_positions if p < seq_len]
    
    if mask_positions:
        predict_masked(sequence, mask_positions, top_k=5)
    print()


# ============================================================================
# Example 2: Sequence Embeddings
# ============================================================================
print("="*80)
print("EXAMPLE 2: Sequence Embeddings from Dataset")
print("="*80)
print()

embeddings = []
for i, seq in enumerate(sample_sequences[:5], 1):
    embedding = get_embedding(seq, pooling="mean")
    embeddings.append(embedding)
    
    print(f"{i}. Sequence: {seq[:40]}..." if len(seq) > 40 else f"{i}. Sequence: {seq}")
    print(f"   Shape: {embedding.shape}")
    print(f"   Norm: {np.linalg.norm(embedding):.4f}")
    print()

embeddings = np.array(embeddings)


# ============================================================================
# Example 3: Sequence Similarity within Dataset
# ============================================================================
print("="*80)
print("EXAMPLE 3: Pairwise Sequence Similarity")
print("="*80)
print()

if len(sample_sequences) >= 3:
    print("Computing pairwise similarities for first 3 sequences:\n")
    
    for i in range(min(3, len(sample_sequences))):
        for j in range(i+1, min(3, len(sample_sequences))):
            seq1 = sample_sequences[i]
            seq2 = sample_sequences[j]
            similarity = compute_similarity(seq1, seq2)
            
            print(f"Sequence {i+1} <-> Sequence {j+1}")
            print(f"  {seq1[:40]}..." if len(seq1) > 40 else f"  {seq1}")
            print(f"  {seq2[:40]}..." if len(seq2) > 40 else f"  {seq2}")
            print(f"  Similarity: {similarity:.4f}\n")


# ============================================================================
# Example 4: Position-wise Prediction Confidence
# ============================================================================
print("="*80)
print("EXAMPLE 4: Position-wise Prediction Confidence")
print("="*80)
print()

if sample_sequences:
    sequence = sample_sequences[0]
    print(f"Analyzing sequence: {sequence}\n")
    print("Predicting each position when masked:")
    print("-" * 70)
    print("Pos | Original | Top Pred | Confidence | Match")
    print("-" * 70)
    
    for pos in range(min(len(sequence), 20)):  # Limit to first 20 positions
        seq_formatted = format_seq(sequence)
        seq_list = seq_formatted.split()
        
        if pos >= len(seq_list):
            continue
            
        original_aa = seq_list[pos]
        seq_list[pos] = tokenizer.mask_token
        masked_seq = " ".join(seq_list)
        
        inputs = tokenizer(masked_seq, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits
        
        mask_token_id = tokenizer.mask_token_id
        mask_idx = (inputs['input_ids'] == mask_token_id).nonzero(as_tuple=True)[1][0]
        
        probs = torch.softmax(predictions[0, mask_idx], dim=-1)
        top_prob, top_idx = torch.topk(probs, k=1)
        top_token = tokenizer.decode([top_idx[0]]).strip()
        
        match = "✓" if top_token == original_aa else "✗"
        print(f"{pos:3d} |    {original_aa:1s}     |    {top_token:1s}     |  {top_prob.item()*100:6.2f}%  |  {match}")
    
    print()


# ============================================================================
# Example 5: Batch Inference Statistics
# ============================================================================
print("="*80)
print("EXAMPLE 5: Dataset Statistics")
print("="*80)
print()

print(f"Number of sequences analyzed: {len(sample_sequences)}")
print(f"Sequence lengths:")
print(f"  Min: {min(len(s) for s in sample_sequences)}")
print(f"  Max: {max(len(s) for s in sample_sequences)}")
print(f"  Mean: {np.mean([len(s) for s in sample_sequences]):.1f}")
print(f"  Median: {np.median([len(s) for s in sample_sequences]):.1f}")

if len(embeddings) > 1:
    print(f"\nEmbedding statistics:")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    print(f"  Mean norm: {np.mean([np.linalg.norm(e) for e in embeddings]):.4f}")
    print(f"  Std norm: {np.std([np.linalg.norm(e) for e in embeddings]):.4f}")
    
    # Compute average pairwise similarity
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            similarities.append(sim)
    
    if similarities:
        print(f"\nPairwise similarity statistics:")
        print(f"  Mean: {np.mean(similarities):.4f}")
        print(f"  Std: {np.std(similarities):.4f}")
        print(f"  Min: {np.min(similarities):.4f}")
        print(f"  Max: {np.max(similarities):.4f}")

print()
print("="*80)
print("INFERENCE COMPLETE!")
print("="*80)
print("\nYou can modify this script to:")
print("  - Change NUM_SAMPLES to analyze more sequences")
print("  - Use 'train' or 'test' splits instead of 'val'")
print("  - Add custom analysis or filtering")
print("  - Export embeddings to file for downstream tasks")
