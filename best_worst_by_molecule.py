#!/usr/bin/env python3
"""
best_worst_by_molecule.py
────────────────────────────────────────────────────────
Analyze best and worst predictions broken down by molecule type.
Shows performance for TCR (TRA/TRB), peptide, and MHC sequences separately.
"""

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import PeftModel
import numpy as np
from datasets import load_from_disk
from tqdm.auto import tqdm
import argparse
from collections import defaultdict


def evaluate_sample(sample, model, tokenizer, device, top_k=5):
    """Evaluate predictions for a single pre-masked sample."""
    input_ids = torch.tensor(sample['input_ids']).unsqueeze(0).to(device)
    labels = torch.tensor(sample['labels']).unsqueeze(0).to(device)
    attention_mask = torch.tensor(sample['attention_mask']).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        loss = outputs.loss.item()
    
    masked_positions = (labels[0] != -100).nonzero(as_tuple=True)[0]
    
    results = {
        'loss': loss,
        'num_masked': len(masked_positions),
        'predictions': [],
        'correct_top1': 0,
        'correct_top5': 0,
    }
    
    for pos in masked_positions:
        true_token_id = labels[0, pos].item()
        true_token = tokenizer.decode([true_token_id]).strip()
        
        probs = torch.softmax(logits[0, pos], dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
        
        top_1_correct = top_k_indices[0].item() == true_token_id
        top_5_correct = true_token_id in top_k_indices.tolist()
        
        results['correct_top1'] += int(top_1_correct)
        results['correct_top5'] += int(top_5_correct)
        
        pred_info = {
            'position': pos.item(),
            'true_token': true_token,
            'top1_correct': top_1_correct,
            'top5_correct': top_5_correct,
            'top_predictions': []
        }
        
        for prob, token_id in zip(top_k_probs, top_k_indices):
            token = tokenizer.decode([token_id]).strip()
            pred_info['top_predictions'].append({
                'token': token,
                'probability': prob.item()
            })
        
        results['predictions'].append(pred_info)
    
    return results


def decode_sequence(input_ids, tokenizer):
    """Decode tokenized sequence back to amino acid string."""
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    aa_sequence = ''.join([t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]']])
    return aa_sequence


def get_molecule_regions(input_ids, tokenizer):
    """
    Identify which positions belong to which molecule.
    Returns dict with regions: {'tcr': [...], 'peptide': [...], 'mhc': [...]}
    """
    # Find [SEP] tokens to identify regions
    sep_token_id = tokenizer.sep_token_id
    sep_positions = []
    
    for i, token_id in enumerate(input_ids):
        if token_id == sep_token_id:
            sep_positions.append(i)
    
    regions = {}
    
    # First region (before first SEP) is TCR
    if len(sep_positions) >= 1:
        regions['tcr'] = list(range(1, sep_positions[0]))  # Skip [CLS]
    
    # Second region (between first and second SEP) is peptide
    if len(sep_positions) >= 2:
        regions['peptide'] = list(range(sep_positions[0] + 1, sep_positions[1]))
    
    # Third region (after second SEP) is MHC
    if len(sep_positions) >= 2:
        regions['mhc'] = list(range(sep_positions[1] + 1, len(input_ids)))
    
    return regions


def categorize_predictions_by_molecule(results, regions):
    """Split predictions by molecule type."""
    by_molecule = {
        'tcr': {'correct': 0, 'total': 0, 'predictions': []},
        'peptide': {'correct': 0, 'total': 0, 'predictions': []},
        'mhc': {'correct': 0, 'total': 0, 'predictions': []},
    }
    
    for pred in results['predictions']:
        pos = pred['position']
        
        # Determine which molecule this position belongs to
        if pos in regions.get('tcr', []):
            molecule = 'tcr'
        elif pos in regions.get('peptide', []):
            molecule = 'peptide'
        elif pos in regions.get('mhc', []):
            molecule = 'mhc'
        else:
            continue
        
        by_molecule[molecule]['total'] += 1
        if pred['top1_correct']:
            by_molecule[molecule]['correct'] += 1
        by_molecule[molecule]['predictions'].append(pred)
    
    return by_molecule


def main():
    parser = argparse.ArgumentParser(description="Analyze predictions by molecule type")
    parser.add_argument("--model_path", type=str, default="/mnt/ephemeral/protbert_01_specifcity")
    parser.add_argument("--dataset_path", type=str, default="/mnt/ephemeral/protbert_spec")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=200, help="Number of samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    print("Loading model...")
    base_model = AutoModelForMaskedLM.from_pretrained("Rostlab/prot_bert_bfd")
    model = PeftModel.from_pretrained(base_model, args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = model.to(args.device)
    model.eval()
    print("✅ Model loaded!\n")
    
    print("Loading dataset...")
    dataset = load_from_disk(args.dataset_path)
    split_data = dataset[args.split]
    print(f"✅ Dataset loaded! ({len(split_data)} samples in {args.split} split)\n")
    
    print(f"Evaluating {args.num_samples} samples...")
    
    # Store results by molecule type
    molecule_samples = {
        'tcr': [],
        'peptide': [],
        'mhc': []
    }
    
    overall_stats = {
        'tcr': {'correct': 0, 'total': 0},
        'peptide': {'correct': 0, 'total': 0},
        'mhc': {'correct': 0, 'total': 0}
    }
    
    for i in tqdm(range(min(args.num_samples, len(split_data)))):
        sample = split_data[i]
        results = evaluate_sample(sample, model, tokenizer, args.device, top_k=5)
        
        # Get molecule regions
        regions = get_molecule_regions(sample['input_ids'], tokenizer)
        
        # Categorize predictions by molecule
        by_molecule = categorize_predictions_by_molecule(results, regions)
        
        # Store sample info for each molecule type
        for mol_type in ['tcr', 'peptide', 'mhc']:
            if by_molecule[mol_type]['total'] > 0:
                accuracy = by_molecule[mol_type]['correct'] / by_molecule[mol_type]['total']
                molecule_samples[mol_type].append({
                    'index': i,
                    'sample': sample,
                    'results': results,
                    'molecule_results': by_molecule[mol_type],
                    'accuracy': accuracy,
                    'loss': results['loss']
                })
                
                # Update overall stats
                overall_stats[mol_type]['correct'] += by_molecule[mol_type]['correct']
                overall_stats[mol_type]['total'] += by_molecule[mol_type]['total']
    
    # Print overall statistics
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS BY MOLECULE TYPE")
    print(f"{'='*80}\n")
    
    for mol_type in ['tcr', 'peptide', 'mhc']:
        stats = overall_stats[mol_type]
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            print(f"{mol_type.upper():8s}: {accuracy*100:5.2f}% accuracy ({stats['correct']:5d}/{stats['total']:5d} tokens) from {len(molecule_samples[mol_type])} samples")
        else:
            print(f"{mol_type.upper():8s}: No data")
    
    # Show best and worst for each molecule type
    for mol_type in ['tcr', 'peptide', 'mhc']:
        samples = molecule_samples[mol_type]
        if len(samples) == 0:
            continue
        
        # Sort by accuracy
        samples.sort(key=lambda x: x['accuracy'], reverse=True)
        
        print(f"\n\n{'='*80}")
        print(f"🧬 {mol_type.upper()} ANALYSIS")
        print(f"{'='*80}")
        
        # Best prediction
        if len(samples) > 0:
            print(f"\n🏆 BEST {mol_type.upper()} PREDICTION")
            print("─"*80)
            
            best = samples[0]
            sample = best['sample']
            mol_res = best['molecule_results']
            
            print(f"Accuracy: {best['accuracy']*100:.2f}% ({mol_res['correct']}/{mol_res['total']} correct)")
            print(f"Loss: {best['loss']:.4f}")
            
            seq = decode_sequence(sample['input_ids'], tokenizer)
            print(f"Sequence: {seq[:100]}..." if len(seq) > 100 else f"Sequence: {seq}")
            
            if 'combo_feats' in sample and sample['combo_feats']:
                print(f"Features: {', '.join(sample['combo_feats'])}")
            
            print(f"\nSample predictions (first 5 {mol_type} positions):")
            for pred in mol_res['predictions'][:5]:
                marker = "✓" if pred['top1_correct'] else "✗"
                top = pred['top_predictions'][0]
                print(f"  Pos {pred['position']:3d}: {pred['true_token']:3s} → {top['token']:3s} ({top['probability']*100:5.2f}%) {marker}")
            
            if len(mol_res['predictions']) > 5:
                remaining = sum(1 for p in mol_res['predictions'][5:] if p['top1_correct'])
                print(f"  ... {len(mol_res['predictions']) - 5} more ({remaining} correct)")
        
        # Worst prediction
        if len(samples) > 0:
            print(f"\n⚠️  WORST {mol_type.upper()} PREDICTION")
            print("─"*80)
            
            worst = samples[-1]
            sample = worst['sample']
            mol_res = worst['molecule_results']
            
            print(f"Accuracy: {worst['accuracy']*100:.2f}% ({mol_res['correct']}/{mol_res['total']} correct)")
            print(f"Loss: {worst['loss']:.4f}")
            
            seq = decode_sequence(sample['input_ids'], tokenizer)
            print(f"Sequence: {seq[:100]}..." if len(seq) > 100 else f"Sequence: {seq}")
            
            if 'combo_feats' in sample and sample['combo_feats']:
                print(f"Features: {', '.join(sample['combo_feats'])}")
            
            print(f"\nSample predictions (first 5 {mol_type} positions):")
            for pred in mol_res['predictions'][:5]:
                marker = "✓" if pred['top1_correct'] else "✗"
                top = pred['top_predictions'][0]
                print(f"  Pos {pred['position']:3d}: {pred['true_token']:3s} → {top['token']:3s} ({top['probability']*100:5.2f}%) {marker}")
                
                if not pred['top1_correct']:
                    # Show where true token ranked
                    true_token = pred['true_token']
                    for rank, p in enumerate(pred['top_predictions'], 1):
                        if p['token'] == true_token:
                            print(f"           ↳ '{true_token}' was rank #{rank} ({p['probability']*100:.2f}%)")
                            break
                    else:
                        print(f"           ↳ '{true_token}' not in top-5")
            
            if len(mol_res['predictions']) > 5:
                remaining = sum(1 for p in mol_res['predictions'][5:] if not p['top1_correct'])
                print(f"  ... {len(mol_res['predictions']) - 5} more ({remaining} wrong)")
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    main()
