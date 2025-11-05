#!/usr/bin/env python3
"""
evaluate_masked_predictions.py
────────────────────────────────────────────────────────
Evaluate model predictions on the pre-masked sequences in the dataset.
The dataset already contains masked sequences with labels, so we can directly
evaluate how well the model predicts the masked tokens.
"""

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import PeftModel
import numpy as np
from datasets import load_from_disk
from tqdm.auto import tqdm
import argparse
from collections import defaultdict


# ============================================================================
# Configuration
# ============================================================================
MODEL_PATH = "/mnt/ephemeral/protbert_01_specifcity"
BASE_MODEL = "Rostlab/prot_bert_bfd"
DATASET_PATH = "/mnt/ephemeral/protbert_spec"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

def evaluate_sample(sample, model, tokenizer, device, top_k=5):
    """
    Evaluate predictions for a single pre-masked sample.
    
    Args:
        sample: Dataset sample with input_ids and labels
        model: PEFT model
        tokenizer: Tokenizer
        device: Device for inference
        top_k: Number of top predictions to return
    
    Returns:
        Dictionary with evaluation metrics
    """
    input_ids = torch.tensor(sample['input_ids']).unsqueeze(0).to(device)
    labels = torch.tensor(sample['labels']).unsqueeze(0).to(device)
    attention_mask = torch.tensor(sample['attention_mask']).unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        logits = outputs.logits
        loss = outputs.loss.item()
    
    # Find masked positions (where labels != -100)
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
        
        # Get probabilities
        probs = torch.softmax(logits[0, pos], dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
        
        # Check if correct prediction is in top-k
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


# ============================================================================
# Evaluate on Sample Data
# ============================================================================
print("="*80)
print("EVALUATING PREDICTIONS ON PRE-MASKED SAMPLES")
print("="*80)
print()

# Use validation set
val_dataset = dataset['val']

# Evaluate a few samples in detail
num_detailed_samples = 5
print(f"Detailed evaluation of {num_detailed_samples} samples:\n")

for i in range(num_detailed_samples):
    sample = val_dataset[i]
    
    print(f"{'─'*80}")
    print(f"Sample {i+1}")
    print(f"{'─'*80}")
    
    # Decode the sequence (with masks)
    masked_seq = decode_sequence(sample['input_ids'], tokenizer)
    print(f"Masked sequence: {masked_seq[:80]}..." if len(masked_seq) > 80 else f"Masked sequence: {masked_seq}")
    
    if 'combo_id' in sample:
        print(f"Combo ID: {sample['combo_id']}")
    if 'combo_feats' in sample and sample['combo_feats']:
        print(f"Combo features: {sample['combo_feats']}")
    
    # Evaluate
    results = evaluate_sample(sample, model, tokenizer, DEVICE, top_k=5)
    
    print(f"\nLoss: {results['loss']:.4f}")
    print(f"Number of masked positions: {results['num_masked']}")
    print(f"Top-1 accuracy: {results['correct_top1']}/{results['num_masked']} ({100*results['correct_top1']/max(results['num_masked'], 1):.2f}%)")
    print(f"Top-5 accuracy: {results['correct_top5']}/{results['num_masked']} ({100*results['correct_top5']/max(results['num_masked'], 1):.2f}%)")
    
    print(f"\nDetailed predictions:")
    for pred in results['predictions'][:5]:  # Show first 5 masked positions
        print(f"\n  Position {pred['position']} - True: {pred['true_token']}")
        for rank, p in enumerate(pred['top_predictions'], 1):
            marker = "✓" if p['token'] == pred['true_token'] else " "
            print(f"    {rank}. {p['token']:3s} - {p['probability']*100:5.2f}% {marker}")
    
    if len(results['predictions']) > 5:
        print(f"  ... and {len(results['predictions']) - 5} more masked positions")
    
    print()


# ============================================================================
# Batch Evaluation Statistics
# ============================================================================
print("="*80)
print("BATCH EVALUATION STATISTICS")
print("="*80)
print()

num_eval_samples = 100
print(f"Evaluating {num_eval_samples} samples from validation set...\n")

all_results = {
    'total_masked': 0,
    'total_correct_top1': 0,
    'total_correct_top5': 0,
    'total_loss': 0,
    'per_position_accuracy': defaultdict(list),
}

# Track best and worst predictions
sample_scores = []

for i in tqdm(range(min(num_eval_samples, len(val_dataset)))):
    sample = val_dataset[i]
    results = evaluate_sample(sample, model, tokenizer, DEVICE, top_k=5)
    
    all_results['total_masked'] += results['num_masked']
    all_results['total_correct_top1'] += results['correct_top1']
    all_results['total_correct_top5'] += results['correct_top5']
    all_results['total_loss'] += results['loss']
    
    # Track per-position accuracy
    for pred in results['predictions']:
        all_results['per_position_accuracy'][pred['position']].append(pred['top1_correct'])
    
    # Store sample score for best/worst analysis
    accuracy = results['correct_top1'] / max(results['num_masked'], 1)
    sample_scores.append({
        'index': i,
        'sample': sample,
        'results': results,
        'accuracy': accuracy,
        'loss': results['loss']
    })

# Calculate overall metrics
avg_loss = all_results['total_loss'] / num_eval_samples
top1_accuracy = all_results['total_correct_top1'] / max(all_results['total_masked'], 1)
top5_accuracy = all_results['total_correct_top5'] / max(all_results['total_masked'], 1)

print(f"\n{'='*80}")
print("OVERALL RESULTS")
print(f"{'='*80}")
print(f"Samples evaluated: {num_eval_samples}")
print(f"Total masked tokens: {all_results['total_masked']}")
print(f"Average loss: {avg_loss:.4f}")
print(f"Top-1 accuracy: {top1_accuracy*100:.2f}%")
print(f"Top-5 accuracy: {top5_accuracy*100:.2f}%")
print()


# ============================================================================
# Per-Position Analysis
# ============================================================================
if all_results['per_position_accuracy']:
    print("="*80)
    print("PER-POSITION ACCURACY ANALYSIS")
    print("="*80)
    print()
    
    # Get positions with enough samples
    position_stats = []
    for pos, accuracies in all_results['per_position_accuracy'].items():
        if len(accuracies) >= 5:  # At least 5 samples
            position_stats.append({
                'position': pos,
                'accuracy': np.mean(accuracies),
                'count': len(accuracies)
            })
    
    # Sort by position
    position_stats.sort(key=lambda x: x['position'])
    
    if position_stats:
        print("Position | Accuracy | Sample Count")
        print("-" * 40)
        for stat in position_stats[:20]:  # Show first 20 positions
            print(f"{stat['position']:8d} | {stat['accuracy']*100:7.2f}% | {stat['count']:12d}")
        
        if len(position_stats) > 20:
            print(f"... and {len(position_stats) - 20} more positions")
        print()


# ============================================================================
# Best and Worst Predictions
# ============================================================================
print("="*80)
print("BEST AND WORST PREDICTIONS")
print("="*80)
print()

# Sort by accuracy
sample_scores.sort(key=lambda x: x['accuracy'], reverse=True)

# Get best predictions
best_samples = sample_scores[:3]
worst_samples = sample_scores[-3:]

print("🏆 TOP 3 BEST PREDICTIONS (Highest Accuracy)")
print("="*80)
for rank, item in enumerate(best_samples, 1):
    sample = item['sample']
    results = item['results']
    
    print(f"\n{'─'*80}")
    print(f"Rank #{rank} - Accuracy: {item['accuracy']*100:.2f}% ({results['correct_top1']}/{results['num_masked']} correct)")
    print(f"{'─'*80}")
    
    masked_seq = decode_sequence(sample['input_ids'], tokenizer)
    print(f"Sequence: {masked_seq[:80]}..." if len(masked_seq) > 80 else f"Sequence: {masked_seq}")
    
    if 'combo_id' in sample:
        print(f"Combo ID: {sample['combo_id']}")
    if 'combo_feats' in sample and sample['combo_feats']:
        print(f"Features: {sample['combo_feats']}")
    
    print(f"\nLoss: {results['loss']:.4f}")
    print(f"Masked positions: {results['num_masked']}")
    
    # Show some predictions
    print(f"\nSample predictions (first 5):")
    for pred in results['predictions'][:5]:
        marker = "✓" if pred['top1_correct'] else "✗"
        top_pred = pred['top_predictions'][0]
        print(f"  Pos {pred['position']:3d}: {pred['true_token']:3s} -> {top_pred['token']:3s} ({top_pred['probability']*100:5.2f}%) {marker}")
    
    if len(results['predictions']) > 5:
        print(f"  ... and {len(results['predictions']) - 5} more")

print("\n\n")
print("⚠️  TOP 3 WORST PREDICTIONS (Lowest Accuracy)")
print("="*80)
for rank, item in enumerate(worst_samples, 1):
    sample = item['sample']
    results = item['results']
    
    print(f"\n{'─'*80}")
    print(f"Rank #{rank} (from bottom) - Accuracy: {item['accuracy']*100:.2f}% ({results['correct_top1']}/{results['num_masked']} correct)")
    print(f"{'─'*80}")
    
    masked_seq = decode_sequence(sample['input_ids'], tokenizer)
    print(f"Sequence: {masked_seq[:80]}..." if len(masked_seq) > 80 else f"Sequence: {masked_seq}")
    
    if 'combo_id' in sample:
        print(f"Combo ID: {sample['combo_id']}")
    if 'combo_feats' in sample and sample['combo_feats']:
        print(f"Features: {sample['combo_feats']}")
    
    print(f"\nLoss: {results['loss']:.4f}")
    print(f"Masked positions: {results['num_masked']}")
    
    # Show some predictions with what went wrong
    print(f"\nSample predictions (first 5):")
    for pred in results['predictions'][:5]:
        marker = "✓" if pred['top1_correct'] else "✗"
        top_pred = pred['top_predictions'][0]
        print(f"  Pos {pred['position']:3d}: {pred['true_token']:3s} -> {top_pred['token']:3s} ({top_pred['probability']*100:5.2f}%) {marker}")
        
        # Show where true token ranked if not top-1
        if not pred['top1_correct']:
            true_token = pred['true_token']
            for i, p in enumerate(pred['top_predictions'], 1):
                if p['token'] == true_token:
                    print(f"           (true token '{true_token}' was rank #{i} at {p['probability']*100:.2f}%)")
                    break
            else:
                print(f"           (true token '{true_token}' not in top-5)")
    
    if len(results['predictions']) > 5:
        print(f"  ... and {len(results['predictions']) - 5} more")

print("\n" + "="*80)
print("EVALUATION COMPLETE!")
print("="*80)
print(f"""
Summary:
  - Evaluated {num_eval_samples} samples
  - Top-1 Accuracy: {top1_accuracy*100:.2f}%
  - Top-5 Accuracy: {top5_accuracy*100:.2f}%
  - Average Loss: {avg_loss:.4f}
  
This evaluation uses the pre-masked sequences from your dataset where
tokens are already masked and labels are provided.
""")
