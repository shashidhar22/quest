#!/usr/bin/env python3
"""
calculate_perplexity.py
────────────────────────────────────────────────────────
Calculate perplexity of true sequences for each molecule type.
Perplexity measures how well the model predicts the true sequence.
Lower perplexity = better prediction quality.
"""

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import PeftModel
import numpy as np
from datasets import load_from_disk
from tqdm.auto import tqdm
import argparse
from collections import defaultdict
import math


def calculate_perplexity(sample, model, tokenizer, device):
    """
    Calculate perplexity for masked positions in a sample.
    Perplexity = exp(average negative log likelihood)
    """
    input_ids = torch.tensor(sample['input_ids']).unsqueeze(0).to(device)
    labels = torch.tensor(sample['labels']).unsqueeze(0).to(device)
    attention_mask = torch.tensor(sample['attention_mask']).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        loss = outputs.loss.item()  # This is the cross-entropy loss
    
    # Find masked positions
    masked_positions = (labels[0] != -100).nonzero(as_tuple=True)[0]
    
    if len(masked_positions) == 0:
        return None, []
    
    # Calculate perplexity from loss
    # Loss is average negative log likelihood, so perplexity = exp(loss)
    perplexity = math.exp(loss)
    
    # Also calculate per-position perplexities
    position_perplexities = []
    for pos in masked_positions:
        true_token_id = labels[0, pos].item()
        
        # Get probability of true token
        probs = torch.softmax(logits[0, pos], dim=-1)
        true_prob = probs[true_token_id].item()
        
        # Perplexity for this position = 1 / probability
        # Or equivalently: exp(-log(probability))
        if true_prob > 0:
            pos_perplexity = 1.0 / true_prob
        else:
            pos_perplexity = float('inf')
        
        position_perplexities.append({
            'position': pos.item(),
            'true_token': tokenizer.decode([true_token_id]).strip(),
            'true_prob': true_prob,
            'perplexity': pos_perplexity,
            'log_likelihood': math.log(true_prob) if true_prob > 0 else float('-inf')
        })
    
    return perplexity, position_perplexities


def decode_sequence(input_ids, tokenizer):
    """Decode tokenized sequence back to amino acid string."""
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    aa_sequence = ''.join([t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]']])
    return aa_sequence


def get_molecule_regions(input_ids, tokenizer):
    """Identify which positions belong to which molecule."""
    sep_token_id = tokenizer.sep_token_id
    sep_positions = []
    
    for i, token_id in enumerate(input_ids):
        if token_id == sep_token_id:
            sep_positions.append(i)
    
    regions = {}
    
    if len(sep_positions) >= 1:
        regions['tcr'] = list(range(1, sep_positions[0]))
    if len(sep_positions) >= 2:
        regions['peptide'] = list(range(sep_positions[0] + 1, sep_positions[1]))
    if len(sep_positions) >= 2:
        regions['mhc'] = list(range(sep_positions[1] + 1, len(input_ids)))
    
    return regions


def categorize_by_molecule(position_perplexities, regions):
    """Split perplexity results by molecule type."""
    by_molecule = {
        'tcr': [],
        'peptide': [],
        'mhc': []
    }
    
    for pos_data in position_perplexities:
        pos = pos_data['position']
        
        if pos in regions.get('tcr', []):
            by_molecule['tcr'].append(pos_data)
        elif pos in regions.get('peptide', []):
            by_molecule['peptide'].append(pos_data)
        elif pos in regions.get('mhc', []):
            by_molecule['mhc'].append(pos_data)
    
    return by_molecule


def main():
    parser = argparse.ArgumentParser(description="Calculate perplexity by molecule type")
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
    
    print(f"Calculating perplexity for {args.num_samples} samples...")
    
    # Store results by molecule type
    molecule_stats = {
        'tcr': {'perplexities': [], 'samples': []},
        'peptide': {'perplexities': [], 'samples': []},
        'mhc': {'perplexities': [], 'samples': []}
    }
    
    overall_perplexities = []
    
    for i in tqdm(range(min(args.num_samples, len(split_data)))):
        sample = split_data[i]
        
        # Calculate overall perplexity
        overall_ppl, position_ppls = calculate_perplexity(sample, model, tokenizer, args.device)
        
        if overall_ppl is None:
            continue
        
        overall_perplexities.append(overall_ppl)
        
        # Get molecule regions
        regions = get_molecule_regions(sample['input_ids'], tokenizer)
        
        # Categorize by molecule
        by_molecule = categorize_by_molecule(position_ppls, regions)
        
        # Calculate per-molecule perplexity
        for mol_type in ['tcr', 'peptide', 'mhc']:
            if len(by_molecule[mol_type]) > 0:
                # Calculate average log likelihood for this molecule
                log_likelihoods = [p['log_likelihood'] for p in by_molecule[mol_type] 
                                 if p['log_likelihood'] != float('-inf')]
                
                if log_likelihoods:
                    avg_log_likelihood = np.mean(log_likelihoods)
                    mol_perplexity = math.exp(-avg_log_likelihood)
                    
                    molecule_stats[mol_type]['perplexities'].append(mol_perplexity)
                    molecule_stats[mol_type]['samples'].append({
                        'index': i,
                        'sample': sample,
                        'perplexity': mol_perplexity,
                        'num_positions': len(by_molecule[mol_type]),
                        'position_data': by_molecule[mol_type]
                    })
    
    # Print results
    print(f"\n{'='*80}")
    print("PERPLEXITY ANALYSIS BY MOLECULE TYPE")
    print(f"{'='*80}\n")
    
    print("📊 OVERALL STATISTICS")
    print("─"*80)
    if overall_perplexities:
        avg_overall_ppl = np.mean(overall_perplexities)
        median_overall_ppl = np.median(overall_perplexities)
        std_overall_ppl = np.std(overall_perplexities)
        print(f"Overall Average Perplexity: {avg_overall_ppl:.4f}")
        print(f"Overall Median Perplexity:  {median_overall_ppl:.4f}")
        print(f"Overall Std Dev:            {std_overall_ppl:.4f}")
        print(f"Range: [{min(overall_perplexities):.4f}, {max(overall_perplexities):.4f}]")
    print()
    
    # Stats by molecule
    print("📈 BY MOLECULE TYPE")
    print("─"*80)
    print(f"{'Molecule':<10} {'Mean PPL':<12} {'Median PPL':<12} {'Std Dev':<12} {'Samples':<10}")
    print("─"*80)
    
    for mol_type in ['tcr', 'peptide', 'mhc']:
        ppls = molecule_stats[mol_type]['perplexities']
        if ppls:
            mean_ppl = np.mean(ppls)
            median_ppl = np.median(ppls)
            std_ppl = np.std(ppls)
            n_samples = len(ppls)
            print(f"{mol_type.upper():<10} {mean_ppl:<12.4f} {median_ppl:<12.4f} {std_ppl:<12.4f} {n_samples:<10}")
        else:
            print(f"{mol_type.upper():<10} {'N/A':<12} {'N/A':<12} {'N/A':<12} {0:<10}")
    
    print()
    print("💡 Note: Lower perplexity = better model confidence in predictions")
    print()
    
    # Show best (lowest perplexity) for each molecule
    for mol_type in ['tcr', 'peptide', 'mhc']:
        samples = molecule_stats[mol_type]['samples']
        if not samples:
            continue
        
        # Sort by perplexity (lower is better)
        samples.sort(key=lambda x: x['perplexity'])
        
        print(f"\n{'='*80}")
        print(f"🏆 {mol_type.upper()} - BEST (LOWEST PERPLEXITY)")
        print(f"{'='*80}")
        
        best = samples[0]
        sample = best['sample']
        
        print(f"Perplexity: {best['perplexity']:.4f} (lower is better)")
        print(f"Positions evaluated: {best['num_positions']}")
        
        seq = decode_sequence(sample['input_ids'], tokenizer)
        print(f"Sequence: {seq[:100]}..." if len(seq) > 100 else f"Sequence: {seq}")
        
        if 'combo_feats' in sample and sample['combo_feats']:
            print(f"Features: {', '.join(sample['combo_feats'])}")
        
        # Show positions with best probabilities
        pos_data = sorted(best['position_data'], key=lambda x: x['perplexity'])[:5]
        print(f"\nTop 5 most confident positions:")
        for data in pos_data:
            print(f"  Pos {data['position']:3d}: {data['true_token']:3s} - prob: {data['true_prob']*100:5.2f}% (ppl: {data['perplexity']:6.2f})")
        
        # Show worst prediction
        print(f"\n{'='*80}")
        print(f"⚠️  {mol_type.upper()} - WORST (HIGHEST PERPLEXITY)")
        print(f"{'='*80}")
        
        worst = samples[-1]
        sample = worst['sample']
        
        print(f"Perplexity: {worst['perplexity']:.4f} (higher is worse)")
        print(f"Positions evaluated: {worst['num_positions']}")
        
        seq = decode_sequence(sample['input_ids'], tokenizer)
        print(f"Sequence: {seq[:100]}..." if len(seq) > 100 else f"Sequence: {seq}")
        
        if 'combo_feats' in sample and sample['combo_feats']:
            print(f"Features: {', '.join(sample['combo_feats'])}")
        
        # Show positions with worst probabilities
        pos_data = sorted(worst['position_data'], key=lambda x: x['perplexity'], reverse=True)[:5]
        print(f"\nTop 5 least confident positions:")
        for data in pos_data:
            print(f"  Pos {data['position']:3d}: {data['true_token']:3s} - prob: {data['true_prob']*100:5.2f}% (ppl: {data['perplexity']:6.2f})")
    
    print("\n" + "="*80)
    print("INTERPRETATION GUIDE")
    print("="*80)
    print("""
Perplexity measures how 'surprised' the model is by the true sequence.
- Lower perplexity (closer to 1.0) = model assigns high probability to true tokens
- Higher perplexity = model is uncertain/surprised by true tokens
- Perplexity of N means the model is as uncertain as randomly picking from N options

Good perplexity values:
  < 2.0  : Excellent - model very confident in predictions
  2-5    : Good - model has learned patterns well
  5-10   : Fair - moderate uncertainty
  > 10   : Poor - high uncertainty, poor fit

Relationship to accuracy:
- Low perplexity typically correlates with high accuracy
- High perplexity indicates model struggles with the sequence
""")
    
    print("="*80)
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    main()
