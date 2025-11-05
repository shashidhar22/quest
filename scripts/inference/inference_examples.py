#!/usr/bin/env python3
"""
inference_examples.py
────────────────────────────────────────────────────────
Simple examples for running inference with PEFT fine-tuned ProtBERT.
This script demonstrates various inference tasks you can perform.
"""

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import PeftModel
import numpy as np


# ============================================================================
# Configuration
# ============================================================================
MODEL_PATH = "/mnt/ephemeral/protbert_01_specifcity"
BASE_MODEL = "Rostlab/prot_bert_bfd"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Load Model
# ============================================================================
print("Loading model...")
print(f"  Base model: {BASE_MODEL}")
print(f"  Adapter: {MODEL_PATH}")
print(f"  Device: {DEVICE}")

base_model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = model.to(DEVICE)
model.eval()

print("✅ Model loaded!\n")


# ============================================================================
# Helper Functions
# ============================================================================

def format_seq(seq):
    """Add spaces between amino acids for ProtBERT."""
    return " ".join(list(seq.replace(" ", "")))


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
    
    for i, (pos, idx, original_aa) in enumerate(zip(mask_positions, mask_indices, original_aas)):
        print(f"Position {pos} (original: {original_aa}):")
        
        # Get probabilities
        probs = torch.softmax(predictions[0, idx], dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
        
        for rank, (prob, token_id) in enumerate(zip(top_k_probs, top_k_indices), 1):
            token = tokenizer.decode([token_id]).strip()
            print(f"  {rank}. {token:3s} - {prob.item()*100:5.2f}%")
        print()


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
# Example 1: Masked Prediction
# ============================================================================
print("="*80)
print("EXAMPLE 1: Masked Amino Acid Prediction")
print("="*80)
print()

# TCR sequence example
sequence = "CASSLAPGATNEKLFF"
mask_positions = [4, 8, 12]  # Mask L, G, E

predict_masked(sequence, mask_positions, top_k=5)


# ============================================================================
# Example 2: Multiple Sequences
# ============================================================================
print("="*80)
print("EXAMPLE 2: Batch Prediction on Multiple Sequences")
print("="*80)
print()

sequences = [
    "CASSLAPGATNEKLFF",
    "CASSLGQAYEQYF",
    "CASSLTGELF",
]

for i, seq in enumerate(sequences, 1):
    print(f"\nSequence {i}: {seq}")
    # Mask middle position
    mid_pos = len(seq) // 2
    predict_masked(seq, [mid_pos], top_k=3)


# ============================================================================
# Example 3: Sequence Embeddings
# ============================================================================
print("="*80)
print("EXAMPLE 3: Sequence Embeddings")
print("="*80)
print()

seq = "CASSLAPGATNEKLFF"
embedding = get_embedding(seq, pooling="mean")

print(f"Sequence: {seq}")
print(f"Embedding shape: {embedding.shape}")
print(f"Embedding (first 10 dimensions):")
print(embedding[:10])
print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
print()


# ============================================================================
# Example 4: Sequence Similarity
# ============================================================================
print("="*80)
print("EXAMPLE 4: Sequence Similarity")
print("="*80)
print()

seq1 = "CASSLAPGATNEKLFF"
seq2 = "CASSLGQAYEQYF"
seq3 = "CASSLTGELF"
seq4 = "MKTIIALSYIFCLVFA"  # Different sequence

print("Computing pairwise similarities:\n")

pairs = [
    (seq1, seq2),
    (seq1, seq3),
    (seq2, seq3),
    (seq1, seq4),
]

for s1, s2 in pairs:
    similarity = compute_similarity(s1, s2)
    print(f"{s1} <-> {s2}")
    print(f"  Similarity: {similarity:.4f}\n")


# ============================================================================
# Example 5: Generate Predictions for All Positions
# ============================================================================
print("="*80)
print("EXAMPLE 5: Scan All Positions")
print("="*80)
print()

sequence = "CASSLGELF"
print(f"Sequence: {sequence}\n")

print("Predicting each position when masked:")
print("-" * 60)

for pos in range(len(sequence)):
    seq_formatted = format_seq(sequence)
    seq_list = seq_formatted.split()
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
    print(f"Pos {pos:2d}: {original_aa} -> {top_token} ({top_prob.item()*100:5.2f}%) {match}")

print()
print("="*80)
print("Done!")
print("="*80)
