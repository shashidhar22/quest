#!/usr/bin/env python3
"""
quick_inference.py
────────────────────────────────────────────────────────
Quick and simple inference script - edit the sequences list and run!
"""

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import PeftModel
import numpy as np

# ============================================================================
# CONFIGURE YOUR SEQUENCES HERE
# ============================================================================

# Add your sequences here:
MY_SEQUENCES = [
    "CASSLAPGATNEKLFF",
    "CASSLGQAYEQYF", 
    "CASSLTGELF",
    # Add more sequences below:
    # "YOUR_SEQUENCE_HERE",
]

# Choose what to do:
TASK = "mask"  # Options: "mask", "embed", "similarity", "scan"

# For masking task, specify positions to mask (0-indexed):
MASK_POSITIONS = [4, 8, 12]  # Leave empty [] to not mask anything

# Number of top predictions to show:
TOP_K = 5

# ============================================================================
# LOAD MODEL (no need to edit below)
# ============================================================================

MODEL_PATH = "/mnt/ephemeral/protbert_01_specifcity"
BASE_MODEL = "Rostlab/prot_bert_bfd"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model on {DEVICE}...")
base_model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = model.to(DEVICE)
model.eval()
print("✅ Model loaded!\n")

def format_seq(seq):
    return " ".join(list(seq.replace(" ", "")))

# ============================================================================
# RUN INFERENCE
# ============================================================================

if TASK == "mask":
    print("="*80)
    print("MASKED PREDICTION")
    print("="*80)
    for i, seq in enumerate(MY_SEQUENCES, 1):
        print(f"\n{'─'*80}")
        print(f"Sequence {i}: {seq}")
        print(f"{'─'*80}")
        
        formatted = format_seq(seq)
        seq_list = formatted.split()
        
        if MASK_POSITIONS:
            mask_positions = [p for p in MASK_POSITIONS if p < len(seq_list)]
        else:
            mask_positions = []
        
        if not mask_positions:
            print("No positions masked. Set MASK_POSITIONS to mask specific positions.")
            continue
            
        original_aas = []
        for pos in mask_positions:
            original_aas.append(seq_list[pos])
            seq_list[pos] = tokenizer.mask_token
        
        masked_seq = " ".join(seq_list)
        print(f"Original: {formatted}")
        print(f"Masked:   {masked_seq}\n")
        
        inputs = tokenizer(masked_seq, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits
        
        mask_token_id = tokenizer.mask_token_id
        mask_indices = (inputs['input_ids'] == mask_token_id).nonzero(as_tuple=True)[1]
        
        for pos, idx, original_aa in zip(mask_positions, mask_indices, original_aas):
            print(f"Position {pos} (original: {original_aa}):")
            probs = torch.softmax(predictions[0, idx], dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k=TOP_K)
            
            for rank, (prob, token_id) in enumerate(zip(top_k_probs, top_k_indices), 1):
                token = tokenizer.decode([token_id]).strip()
                marker = "✓" if token == original_aa else " "
                print(f"  {rank}. {token:3s} - {prob.item()*100:5.2f}% {marker}")
            print()

elif TASK == "embed":
    print("="*80)
    print("SEQUENCE EMBEDDINGS")
    print("="*80)
    for i, seq in enumerate(MY_SEQUENCES, 1):
        formatted = format_seq(seq)
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.base_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            mask = inputs['attention_mask'].unsqueeze(-1)
            embedding = (hidden_states * mask).sum(1) / mask.sum(1)
        
        embedding = embedding.cpu().numpy()[0]
        print(f"\n{i}. {seq}")
        print(f"   Shape: {embedding.shape}")
        print(f"   Norm: {np.linalg.norm(embedding):.4f}")
        print(f"   First 5 dims: {embedding[:5]}")

elif TASK == "similarity":
    print("="*80)
    print("SEQUENCE SIMILARITY")
    print("="*80)
    
    # Get all embeddings
    embeddings = []
    for seq in MY_SEQUENCES:
        formatted = format_seq(seq)
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.base_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            mask = inputs['attention_mask'].unsqueeze(-1)
            embedding = (hidden_states * mask).sum(1) / mask.sum(1)
        
        embeddings.append(embedding.cpu().numpy()[0])
    
    embeddings = np.array(embeddings)
    
    # Compute pairwise similarities
    print("\nPairwise Cosine Similarities:\n")
    for i in range(len(MY_SEQUENCES)):
        for j in range(i+1, len(MY_SEQUENCES)):
            similarity = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            print(f"{i+1} <-> {j+1}: {similarity:.4f}")
            print(f"  {MY_SEQUENCES[i]}")
            print(f"  {MY_SEQUENCES[j]}")
            print()

elif TASK == "scan":
    print("="*80)
    print("SCAN ALL POSITIONS")
    print("="*80)
    
    for seq_idx, seq in enumerate(MY_SEQUENCES, 1):
        print(f"\n{'─'*80}")
        print(f"Sequence {seq_idx}: {seq}")
        print(f"{'─'*80}")
        
        results = []
        for pos in range(len(seq)):
            formatted = format_seq(seq)
            seq_list = formatted.split()
            original_aa = seq_list[pos]
            seq_list[pos] = tokenizer.mask_token
            masked = " ".join(seq_list)
            
            inputs = tokenizer(masked, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = outputs.logits
            
            mask_idx = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1][0]
            probs = torch.softmax(predictions[0, mask_idx], dim=-1)
            
            # Get top prediction
            top_prob, top_idx = torch.topk(probs, k=1)
            top_token = tokenizer.decode([top_idx[0]]).strip()
            
            # Get probability of original AA
            aa_token_id = tokenizer.encode(original_aa, add_special_tokens=False)[0]
            original_prob = probs[aa_token_id].item()
            
            match = "✓" if top_token == original_aa else "✗"
            results.append({
                'pos': pos,
                'original': original_aa,
                'predicted': top_token,
                'prob': top_prob.item(),
                'original_prob': original_prob,
                'match': match
            })
        
        print("\nPos | Original | Top Pred | Top Prob | Orig Prob | Match")
        print("-" * 60)
        for r in results:
            print(f"{r['pos']:3d} |    {r['original']:1s}     |    {r['predicted']:1s}     | {r['prob']*100:6.2f}% | {r['original_prob']*100:7.2f}% |  {r['match']}")

else:
    print(f"Unknown task: {TASK}")
    print("Available tasks: 'mask', 'embed', 'similarity', 'scan'")

print("\n" + "="*80)
print("DONE!")
print("="*80)
