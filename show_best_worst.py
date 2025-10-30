#!/usr/bin/env python3
"""
show_best_worst.py
────────────────────────────────────────────────────────
Quick script to show best and worst predictions from the model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import PeftModel
import numpy as np
from datasets import load_from_disk
from tqdm.auto import tqdm
import argparse


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
            'true_token_id': true_token_id,
            'top1_correct': top_1_correct,
            'top5_correct': top_5_correct,
            'top_predictions': []
        }
        
        for prob, token_id in zip(top_k_probs, top_k_indices):
            token = tokenizer.decode([token_id]).strip()
            pred_info['top_predictions'].append({
                'token': token,
                'token_id': token_id.item(),
                'probability': prob.item()
            })
        
        results['predictions'].append(pred_info)
    
    return results


def decode_sequence(input_ids, tokenizer):
    """Decode tokenized sequence back to amino acid string."""
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    aa_sequence = ''.join([t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]']])
    return aa_sequence


def main():
    parser = argparse.ArgumentParser(description="Show best and worst predictions")
    parser.add_argument("--model_path", type=str, default="/mnt/ephemeral/protbert_01_specifcity")
    parser.add_argument("--dataset_path", type=str, default="/mnt/ephemeral/protbert_spec")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--top_n", type=int, default=3, help="Show top N best and worst")
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
    sample_scores = []
    
    for i in tqdm(range(min(args.num_samples, len(split_data)))):
        sample = split_data[i]
        results = evaluate_sample(sample, model, tokenizer, args.device, top_k=5)
        
        accuracy = results['correct_top1'] / max(results['num_masked'], 1)
        sample_scores.append({
            'index': i,
            'sample': sample,
            'results': results,
            'accuracy': accuracy,
            'loss': results['loss']
        })
    
    # Sort by accuracy
    sample_scores.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # Calculate overall stats
    total_masked = sum(s['results']['num_masked'] for s in sample_scores)
    total_correct = sum(s['results']['correct_top1'] for s in sample_scores)
    avg_accuracy = total_correct / total_masked if total_masked > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"OVERALL: {avg_accuracy*100:.2f}% accuracy on {args.num_samples} samples ({total_correct}/{total_masked} tokens)")
    print(f"{'='*80}\n")
    
    # Show best
    print(f"🏆 TOP {args.top_n} BEST PREDICTIONS")
    print("="*80)
    
    for rank, item in enumerate(sample_scores[:args.top_n], 1):
        sample = item['sample']
        results = item['results']
        
        print(f"\n{'─'*80}")
        print(f"#{rank} - Accuracy: {item['accuracy']*100:.2f}% ({results['correct_top1']}/{results['num_masked']} correct) | Loss: {results['loss']:.4f}")
        print(f"{'─'*80}")
        
        masked_seq = decode_sequence(sample['input_ids'], tokenizer)
        print(f"Sequence: {masked_seq[:100]}..." if len(masked_seq) > 100 else f"Sequence: {masked_seq}")
        
        if 'combo_feats' in sample and sample['combo_feats']:
            print(f"Features: {', '.join(sample['combo_feats'])}")
        
        # Show sample predictions
        print(f"\nSample predictions:")
        shown = 0
        for pred in results['predictions']:
            if shown >= 5:
                break
            marker = "✓" if pred['top1_correct'] else "✗"
            top_pred = pred['top_predictions'][0]
            print(f"  Pos {pred['position']:3d}: {pred['true_token']:3s} → {top_pred['token']:3s} ({top_pred['probability']*100:5.2f}%) {marker}")
            shown += 1
        
        if len(results['predictions']) > 5:
            remaining_correct = sum(1 for p in results['predictions'][5:] if p['top1_correct'])
            print(f"  ... {len(results['predictions']) - 5} more ({remaining_correct} correct)")
    
    # Show worst
    print(f"\n\n⚠️  TOP {args.top_n} WORST PREDICTIONS")
    print("="*80)
    
    for rank, item in enumerate(sample_scores[-args.top_n:][::-1], 1):
        sample = item['sample']
        results = item['results']
        
        print(f"\n{'─'*80}")
        print(f"#{rank} (from bottom) - Accuracy: {item['accuracy']*100:.2f}% ({results['correct_top1']}/{results['num_masked']} correct) | Loss: {results['loss']:.4f}")
        print(f"{'─'*80}")
        
        masked_seq = decode_sequence(sample['input_ids'], tokenizer)
        print(f"Sequence: {masked_seq[:100]}..." if len(masked_seq) > 100 else f"Sequence: {masked_seq}")
        
        if 'combo_feats' in sample and sample['combo_feats']:
            print(f"Features: {', '.join(sample['combo_feats'])}")
        
        # Show sample predictions with what went wrong
        print(f"\nSample predictions (showing issues):")
        shown = 0
        for pred in results['predictions']:
            if shown >= 5:
                break
            marker = "✓" if pred['top1_correct'] else "✗"
            top_pred = pred['top_predictions'][0]
            print(f"  Pos {pred['position']:3d}: {pred['true_token']:3s} → {top_pred['token']:3s} ({top_pred['probability']*100:5.2f}%) {marker}")
            
            # Show where true token ranked if not correct
            if not pred['top1_correct']:
                true_token = pred['true_token']
                for i, p in enumerate(pred['top_predictions'], 1):
                    if p['token'] == true_token:
                        print(f"           ↳ '{true_token}' was rank #{i} ({p['probability']*100:.2f}%)")
                        break
                else:
                    print(f"           ↳ '{true_token}' not in top-5")
            shown += 1
        
        if len(results['predictions']) > 5:
            remaining_wrong = sum(1 for p in results['predictions'][5:] if not p['top1_correct'])
            print(f"  ... {len(results['predictions']) - 5} more ({remaining_wrong} wrong)")
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    main()
