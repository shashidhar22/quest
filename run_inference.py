#!/usr/bin/env python3
"""
run_inference.py
────────────────────────────────────────────────────────
Inference script for PEFT fine-tuned ProtBERT model.
Loads a LoRA adapter and runs inference on protein sequences.
"""

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import PeftModel
import argparse
from typing import List, Dict
import re


def load_peft_model(adapter_path: str, device: str = "cuda"):
    """
    Load a PEFT fine-tuned model.
    
    Args:
        adapter_path: Path to the PEFT adapter directory
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        model: Loaded PEFT model
        tokenizer: Tokenizer for the model
    """
    print(f"Loading base model from adapter config...")
    
    # The adapter config specifies the base model
    base_model_name = "Rostlab/prot_bert_bfd"
    
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForMaskedLM.from_pretrained(base_model_name)
    
    print(f"Loading PEFT adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully on {device}")
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer


def format_sequence(sequence: str) -> str:
    """
    Format protein sequence for ProtBERT (space-separated amino acids).
    
    Args:
        sequence: Protein sequence string
    
    Returns:
        Formatted sequence with spaces between amino acids
    """
    # Remove any existing spaces
    sequence = sequence.replace(" ", "")
    # Add spaces between amino acids
    return " ".join(list(sequence))


def predict_masked_tokens(
    model,
    tokenizer,
    sequence: str,
    mask_positions: List[int] = None,
    top_k: int = 5,
    device: str = "cuda"
) -> Dict:
    """
    Predict masked tokens in a protein sequence.
    
    Args:
        model: PEFT model
        tokenizer: Tokenizer
        sequence: Protein sequence (with or without spaces)
        mask_positions: List of positions to mask (0-indexed). If None, no masking is done.
        top_k: Number of top predictions to return
        device: Device for inference
    
    Returns:
        Dictionary with predictions
    """
    # Format sequence
    formatted_seq = format_sequence(sequence)
    
    # Mask specified positions
    if mask_positions:
        seq_list = formatted_seq.split()
        for pos in mask_positions:
            if 0 <= pos < len(seq_list):
                seq_list[pos] = tokenizer.mask_token
        masked_seq = " ".join(seq_list)
    else:
        masked_seq = formatted_seq
    
    print(f"\nInput sequence: {masked_seq[:100]}...")
    
    # Tokenize
    inputs = tokenizer(masked_seq, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
    
    results = {
        "original_sequence": sequence,
        "formatted_sequence": formatted_seq,
        "masked_sequence": masked_seq,
        "predictions": []
    }
    
    # Find mask token positions
    mask_token_id = tokenizer.mask_token_id
    mask_indices = (inputs['input_ids'] == mask_token_id).nonzero(as_tuple=True)[1]
    
    for idx in mask_indices:
        # Get top k predictions for this position
        top_k_tokens = torch.topk(predictions[0, idx], k=top_k)
        
        pred_info = {
            "position": idx.item(),
            "top_predictions": []
        }
        
        for score, token_id in zip(top_k_tokens.values, top_k_tokens.indices):
            token = tokenizer.decode([token_id]).strip()
            pred_info["top_predictions"].append({
                "token": token,
                "score": score.item(),
                "probability": torch.softmax(predictions[0, idx], dim=-1)[token_id].item()
            })
        
        results["predictions"].append(pred_info)
    
    return results


def compute_sequence_embeddings(
    model,
    tokenizer,
    sequence: str,
    device: str = "cuda",
    pooling: str = "mean"
) -> torch.Tensor:
    """
    Compute embeddings for a protein sequence.
    
    Args:
        model: PEFT model
        tokenizer: Tokenizer
        sequence: Protein sequence
        device: Device for inference
        pooling: Pooling strategy ('mean', 'max', or 'cls')
    
    Returns:
        Embedding tensor
    """
    formatted_seq = format_sequence(sequence)
    
    inputs = tokenizer(formatted_seq, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.base_model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Last layer
        
        if pooling == "mean":
            # Mean pooling (excluding padding)
            mask = inputs['attention_mask'].unsqueeze(-1)
            embedding = (hidden_states * mask).sum(1) / mask.sum(1)
        elif pooling == "max":
            # Max pooling
            embedding = hidden_states.max(1)[0]
        elif pooling == "cls":
            # CLS token
            embedding = hidden_states[:, 0]
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")
    
    return embedding.cpu()


def main():
    parser = argparse.ArgumentParser(description="Run inference on PEFT fine-tuned ProtBERT")
    parser.add_argument("--model_path", type=str, default="/mnt/ephemeral/protbert_01_specifcity",
                       help="Path to PEFT adapter directory")
    parser.add_argument("--sequence", type=str, default=None,
                       help="Protein sequence for inference")
    parser.add_argument("--mask_positions", type=int, nargs="+", default=None,
                       help="Positions to mask (0-indexed)")
    parser.add_argument("--top_k", type=int, default=5,
                       help="Number of top predictions to show")
    parser.add_argument("--task", type=str, choices=["mask", "embed"], default="mask",
                       help="Task to perform: 'mask' for masked prediction, 'embed' for embeddings")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_peft_model(args.model_path, device=args.device)
    
    # Example sequences if none provided
    example_sequences = [
        "CASSLAPGATNEKLFF",  # Example TCR sequence
        "CASSLGQAYEQYF",
        "CASSLTGELF",
    ]
    
    sequences = [args.sequence] if args.sequence else example_sequences
    
    print("\n" + "="*80)
    print("RUNNING INFERENCE")
    print("="*80)
    
    for i, seq in enumerate(sequences, 1):
        print(f"\n{'─'*80}")
        print(f"Example {i}")
        print(f"{'─'*80}")
        
        if args.task == "mask":
            # Masked language modeling
            results = predict_masked_tokens(
                model, tokenizer, seq,
                mask_positions=args.mask_positions,
                top_k=args.top_k,
                device=args.device
            )
            
            print(f"\nOriginal: {results['original_sequence']}")
            print(f"Masked:   {results['masked_sequence'][:100]}...")
            
            for pred in results["predictions"]:
                print(f"\nPosition {pred['position']} predictions:")
                for j, p in enumerate(pred["top_predictions"], 1):
                    print(f"  {j}. {p['token']:3s} (prob: {p['probability']:.4f}, score: {p['score']:.2f})")
        
        elif args.task == "embed":
            # Sequence embedding
            embedding = compute_sequence_embeddings(
                model, tokenizer, seq,
                device=args.device,
                pooling="mean"
            )
            print(f"\nSequence: {seq}")
            print(f"Embedding shape: {embedding.shape}")
            print(f"Embedding (first 10 dims): {embedding[0, :10].tolist()}")
            print(f"Embedding norm: {embedding.norm().item():.4f}")
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
