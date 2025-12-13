#!/usr/bin/env python3
"""
evaluate_trb.py
────────────────────────────────────────────────────────────────────────────────
Specialized evaluation script for TRB/TRA CDR3 middle masking.

This script evaluates models trained with the TRB masking strategy, which masks
the middle N amino acids of the CDR3 region. It provides:

1. Overall accuracy and perplexity for the masked CDR3 positions
2. Position-wise accuracy within the masked region (position 1, 2, 3, 4, 5)
3. Per-amino-acid accuracy (how well does the model predict each amino acid type)
4. Confusion matrix for predictions vs ground truth
5. W&B logging for all metrics and visualizations

Usage:
    python scripts/inference/evaluate_trb.py \
        --model_path checkpoints/esm2_8M_cdr_dtrb_1M_balanced \
        --dataset_path data/tokenized/phase_two/database/1M/balanced/esm2_full_dtrb_1M_balanced/hf_dataset \
        --wandb_project quest-trb-eval \
        --batch_size 32 \
        --cdr3_mask_length 5
"""

import argparse
import json
import math
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

import matplotlib.pyplot as plt
import seaborn as sns

# Try to import Accelerate for distributed evaluation
try:
    from accelerate import Accelerator
    from accelerate.utils import gather_object
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    gather_object = None
    print("⚠️  Accelerate not available. Multi-GPU evaluation will not be supported.")

# Try to import PEFT for LoRA model support
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠️  PEFT not available. LoRA models will not be supported.")

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Standard amino acids
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(
    model_path: str,
    device: str = "cuda",
    use_accelerate: bool = False
) -> Tuple[Any, Any]:
    """
    Load model and tokenizer from HuggingFace or local path.
    Supports regular models and PEFT/LoRA adapters.
    """
    print(f"Loading model from: {model_path}")

    # Check if this is a PEFT adapter model
    is_peft_model = False
    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "adapter_config.json")):
        if not PEFT_AVAILABLE:
            raise ImportError(
                "This appears to be a PEFT/LoRA model but peft is not installed. "
                "Install with: pip install peft"
            )

        print(f"Detected PEFT adapter model at {model_path}")
        is_peft_model = True

        with open(os.path.join(model_path, "adapter_config.json"), 'r') as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", "facebook/esm2_t6_8M_UR50D")

        print(f"Loading base model: {base_model_name}")
        base_model = AutoModelForMaskedLM.from_pretrained(base_model_name)

        peft_model = PeftModel.from_pretrained(base_model, model_path)
        print("Merging PEFT weights into base model for faster inference...")
        model = peft_model.merge_and_unload()

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    else:
        model = AutoModelForMaskedLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.mask_token

    if not use_accelerate:
        model = model.to(device)

    model.eval()

    if not use_accelerate:
        print(f"✅ Model loaded successfully on {device}")
    else:
        print(f"✅ Model loaded successfully (will be distributed by Accelerate)")
    print(f"   Model type: {model.config.model_type}")
    print(f"   Vocab size: {len(tokenizer)}")

    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TRB-SPECIFIC MASKING COLLATOR
# ═══════════════════════════════════════════════════════════════════════════════

class TRBMaskingCollator:
    """
    Collator that applies TRB-style CDR3 middle masking for evaluation.
    Masks the middle N amino acids of the sequence deterministically.
    """
    
    def __init__(
        self,
        tokenizer: Any,
        cdr3_mask_length: int = 5,
        mask_all: bool = True,  # If True, mask all positions (for eval); if False, use MLM probability
    ):
        self.tokenizer = tokenizer
        self.cdr3_mask_length = cdr3_mask_length
        self.mask_all = mask_all
        
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.cls_token_id = tokenizer.cls_token_id if hasattr(tokenizer, 'cls_token_id') else None
        self.sep_token_id = tokenizer.sep_token_id if hasattr(tokenizer, 'sep_token_id') else None
        
    def _is_special_token(self, token_id: int) -> bool:
        """Check if token is a special token."""
        if token_id == self.pad_token_id:
            return True
        if self.cls_token_id is not None and token_id == self.cls_token_id:
            return True
        if self.sep_token_id is not None and token_id == self.sep_token_id:
            return True
        
        token = self.tokenizer.decode([token_id])
        if token.startswith('[') and token.endswith(']'):
            return True
        if token.startswith('<') and token.endswith('>'):
            return True
        
        return False
    
    def _find_sequence_boundaries(self, input_ids: List[int]) -> Tuple[int, int]:
        """Find start and end indices of actual sequence (excluding special tokens)."""
        seq_start = 0
        seq_end = len(input_ids)
        
        for i, token_id in enumerate(input_ids):
            if not self._is_special_token(token_id):
                seq_start = i
                break
        
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] != self.pad_token_id and not self._is_special_token(input_ids[i]):
                seq_end = i + 1
                break
        
        return seq_start, seq_end
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Apply TRB masking and create batch."""
        
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        batch_position_indices = []  # Track which positions (0-4) each masked token corresponds to
        batch_original_tokens = []   # Track original token IDs for analysis
        
        for f in features:
            input_ids = f["input_ids"]
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()
            else:
                input_ids = list(input_ids)
            
            attention_mask = f.get("attention_mask", [1] * len(input_ids))
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.tolist()
            
            label_ids = [-100] * len(input_ids)
            position_indices = [-1] * len(input_ids)  # -1 means not a masked position
            
            seq_start, seq_end = self._find_sequence_boundaries(input_ids)
            seq_length = seq_end - seq_start
            
            if seq_length > self.cdr3_mask_length:
                # Calculate middle region
                middle_start = seq_start + (seq_length - self.cdr3_mask_length) // 2
                middle_end = middle_start + self.cdr3_mask_length
                
                # Mask all positions in the middle region (for evaluation)
                for pos_idx, i in enumerate(range(middle_start, middle_end)):
                    if i < seq_end and not self._is_special_token(input_ids[i]):
                        label_ids[i] = input_ids[i]
                        position_indices[i] = pos_idx  # 0, 1, 2, 3, 4 for positions within masked region
                        input_ids[i] = self.mask_token_id
            
            batch_input_ids.append(input_ids)
            batch_labels.append(label_ids)
            batch_attention_mask.append(attention_mask)
            batch_position_indices.append(position_indices)
        
        # Pad to same length
        max_len = max(len(ids) for ids in batch_input_ids)
        
        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []
        padded_position_indices = []
        
        for input_ids, labels, attn_mask, pos_indices in zip(
            batch_input_ids, batch_labels, batch_attention_mask, batch_position_indices
        ):
            pad_len = max_len - len(input_ids)
            padded_input_ids.append(input_ids + [self.pad_token_id] * pad_len)
            padded_labels.append(labels + [-100] * pad_len)
            padded_attention_mask.append(attn_mask + [0] * pad_len)
            padded_position_indices.append(pos_indices + [-1] * pad_len)
        
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "position_indices": torch.tensor(padded_position_indices, dtype=torch.long),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. POSITION-WISE METRICS COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_positionwise_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    position_indices: torch.Tensor,
    tokenizer: Any,
    cdr3_mask_length: int = 5,
) -> Dict[str, Any]:
    """
    Compute position-wise accuracy and per-amino-acid metrics.
    
    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        labels: Ground truth labels [batch_size, seq_len] (-100 for non-masked)
        position_indices: Position within masked region [batch_size, seq_len] (-1 for non-masked)
        tokenizer: Tokenizer for decoding
        cdr3_mask_length: Number of positions in masked region
    
    Returns:
        Dictionary with position-wise and amino-acid-wise metrics
    """
    # Mask out -100 labels
    mask = labels != -100
    
    if mask.sum() == 0:
        return {
            "overall_accuracy": 0.0,
            "overall_loss": 0.0,
            "position_accuracy": {i: 0.0 for i in range(cdr3_mask_length)},
            "position_counts": {i: 0 for i in range(cdr3_mask_length)},
            "aa_accuracy": {aa: 0.0 for aa in AMINO_ACIDS},
            "aa_counts": {aa: 0 for aa in AMINO_ACIDS},
            "confusion_matrix": None,
            "total_masked": 0,
        }
    
    # Get predictions
    predictions = torch.argmax(logits, dim=-1)
    
    # Overall accuracy
    correct = (predictions[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    overall_accuracy = correct / total if total > 0 else 0.0
    
    # Calculate cross-entropy loss
    loss_fct = torch.nn.CrossEntropyLoss()
    overall_loss = loss_fct(
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    ).item()
    
    # Position-wise accuracy
    position_correct = defaultdict(int)
    position_total = defaultdict(int)
    
    # Amino acid accuracy (how well each AA type is predicted)
    aa_correct = defaultdict(int)
    aa_total = defaultdict(int)
    
    # Confusion matrix data
    confusion_data = []  # List of (true_aa, pred_aa) tuples
    
    # Create amino acid lookup
    aa_to_idx = {}
    for aa in AMINO_ACIDS:
        tokens = tokenizer.encode(aa, add_special_tokens=False)
        if tokens:
            aa_to_idx[tokens[0]] = aa
    
    # Process each position
    for batch_idx in range(labels.size(0)):
        for seq_idx in range(labels.size(1)):
            if labels[batch_idx, seq_idx] != -100:
                pos = position_indices[batch_idx, seq_idx].item()
                true_token = labels[batch_idx, seq_idx].item()
                pred_token = predictions[batch_idx, seq_idx].item()
                
                # Position-wise accuracy
                if 0 <= pos < cdr3_mask_length:
                    position_total[pos] += 1
                    if pred_token == true_token:
                        position_correct[pos] += 1
                
                # Amino acid accuracy
                true_aa = aa_to_idx.get(true_token, None)
                pred_aa = aa_to_idx.get(pred_token, None)
                
                if true_aa is not None:
                    aa_total[true_aa] += 1
                    if pred_token == true_token:
                        aa_correct[true_aa] += 1
                
                if true_aa is not None and pred_aa is not None:
                    confusion_data.append((true_aa, pred_aa))
    
    # Calculate position-wise accuracy
    position_accuracy = {}
    for pos in range(cdr3_mask_length):
        if position_total[pos] > 0:
            position_accuracy[pos] = position_correct[pos] / position_total[pos]
        else:
            position_accuracy[pos] = 0.0
    
    # Calculate amino acid accuracy
    aa_accuracy = {}
    for aa in AMINO_ACIDS:
        if aa_total[aa] > 0:
            aa_accuracy[aa] = aa_correct[aa] / aa_total[aa]
        else:
            aa_accuracy[aa] = 0.0
    
    return {
        "overall_accuracy": overall_accuracy,
        "overall_loss": overall_loss,
        "overall_perplexity": math.exp(overall_loss) if overall_loss < 100 else float('inf'),
        "position_accuracy": position_accuracy,
        "position_counts": dict(position_total),
        "position_correct": dict(position_correct),
        "aa_accuracy": aa_accuracy,
        "aa_counts": dict(aa_total),
        "aa_correct": dict(aa_correct),
        "confusion_data": confusion_data,
        "total_masked": total,
        "total_correct": correct,
    }


def run_trb_evaluation(
    model: Any,
    dataloader: DataLoader,
    tokenizer: Any,
    device: str = "cuda",
    cdr3_mask_length: int = 5,
    accelerator: Any = None,
) -> Dict[str, Any]:
    """
    Run TRB-specific evaluation and compute position-wise metrics.
    """
    model.eval()
    
    # Accumulators
    total_correct = 0
    total_masked = 0
    total_loss_weighted = 0.0
    
    position_correct = defaultdict(int)
    position_total = defaultdict(int)
    
    aa_correct = defaultdict(int)
    aa_total = defaultdict(int)
    
    all_confusion_data = []
    
    # Create amino acid lookup
    aa_to_idx = {}
    for aa in AMINO_ACIDS:
        tokens = tokenizer.encode(aa, add_special_tokens=False)
        if tokens:
            aa_to_idx[tokens[0]] = aa
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", 
                         disable=not accelerator.is_local_main_process if accelerator else False):
            
            if accelerator is None:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                position_indices = batch["position_indices"].to(device)
            else:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                position_indices = batch["position_indices"]
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            # Gather results if using distributed
            if accelerator is not None:
                logits = accelerator.gather_for_metrics(outputs.logits)
                labels_gathered = accelerator.gather_for_metrics(labels)
                position_indices_gathered = accelerator.gather_for_metrics(position_indices)
            else:
                logits = outputs.logits
                labels_gathered = labels
                position_indices_gathered = position_indices
            
            # Process only on main process
            if accelerator is None or accelerator.is_local_main_process:
                logits_cpu = logits.cpu()
                labels_cpu = labels_gathered.cpu()
                position_indices_cpu = position_indices_gathered.cpu()
                
                # Get predictions
                predictions = torch.argmax(logits_cpu, dim=-1)
                mask = labels_cpu != -100
                
                # Overall metrics
                batch_correct = (predictions[mask] == labels_cpu[mask]).sum().item()
                batch_total = mask.sum().item()
                total_correct += batch_correct
                total_masked += batch_total
                
                # Loss
                if batch_total > 0:
                    loss_fct = torch.nn.CrossEntropyLoss()
                    batch_loss = loss_fct(
                        logits_cpu.view(-1, logits_cpu.size(-1)),
                        labels_cpu.view(-1)
                    ).item()
                    total_loss_weighted += batch_loss * batch_total
                
                # Position-wise and AA-wise metrics
                for batch_idx in range(labels_cpu.size(0)):
                    for seq_idx in range(labels_cpu.size(1)):
                        if labels_cpu[batch_idx, seq_idx] != -100:
                            pos = position_indices_cpu[batch_idx, seq_idx].item()
                            true_token = labels_cpu[batch_idx, seq_idx].item()
                            pred_token = predictions[batch_idx, seq_idx].item()
                            
                            # Position-wise
                            if 0 <= pos < cdr3_mask_length:
                                position_total[pos] += 1
                                if pred_token == true_token:
                                    position_correct[pos] += 1
                            
                            # Amino acid-wise
                            true_aa = aa_to_idx.get(true_token, None)
                            pred_aa = aa_to_idx.get(pred_token, None)
                            
                            if true_aa is not None:
                                aa_total[true_aa] += 1
                                if pred_token == true_token:
                                    aa_correct[true_aa] += 1
                            
                            if true_aa is not None and pred_aa is not None:
                                all_confusion_data.append((true_aa, pred_aa))
                
                del logits_cpu, labels_cpu, position_indices_cpu
    
    # Compile results (only on main process)
    if accelerator is None or accelerator.is_local_main_process:
        # Overall metrics
        overall_accuracy = total_correct / total_masked if total_masked > 0 else 0.0
        overall_loss = total_loss_weighted / total_masked if total_masked > 0 else 0.0
        overall_perplexity = math.exp(overall_loss) if overall_loss < 100 else float('inf')
        
        # Position-wise accuracy
        pos_accuracy = {}
        for pos in range(cdr3_mask_length):
            if position_total[pos] > 0:
                pos_accuracy[pos] = position_correct[pos] / position_total[pos]
            else:
                pos_accuracy[pos] = 0.0
        
        # Amino acid accuracy
        aa_acc = {}
        for aa in AMINO_ACIDS:
            if aa_total[aa] > 0:
                aa_acc[aa] = aa_correct[aa] / aa_total[aa]
            else:
                aa_acc[aa] = 0.0
        
        return {
            "overall": {
                "accuracy": overall_accuracy,
                "loss": overall_loss,
                "perplexity": overall_perplexity,
                "total_masked": total_masked,
                "total_correct": total_correct,
            },
            "position_wise": {
                "accuracy": pos_accuracy,
                "counts": dict(position_total),
                "correct": dict(position_correct),
            },
            "amino_acid_wise": {
                "accuracy": aa_acc,
                "counts": dict(aa_total),
                "correct": dict(aa_correct),
            },
            "confusion_data": all_confusion_data,
        }
    else:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_position_accuracy_figure(
    position_accuracy: Dict[int, float],
    position_counts: Dict[int, float],
    cdr3_mask_length: int = 5,
) -> plt.Figure:
    """Create bar chart of position-wise accuracy."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    positions = list(range(cdr3_mask_length))
    accuracies = [position_accuracy.get(p, 0.0) for p in positions]
    counts = [position_counts.get(p, 0) for p in positions]
    
    # Create position labels (1-indexed for display)
    labels = [f"Pos {p+1}\n(n={counts[p]:,})" for p in positions]
    
    bars = ax.bar(labels, accuracies, color='steelblue', edgecolor='black')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=12)
    
    ax.set_xlabel('Position in Masked CDR3 Region', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Position-wise Prediction Accuracy in Masked CDR3 Region', fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=sum(accuracies)/len(accuracies), color='red', linestyle='--', 
               label=f'Mean: {sum(accuracies)/len(accuracies):.3f}')
    ax.legend()
    
    plt.tight_layout()
    return fig


def create_aa_accuracy_figure(
    aa_accuracy: Dict[str, float],
    aa_counts: Dict[str, int],
) -> plt.Figure:
    """Create bar chart of per-amino-acid accuracy."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Sort by accuracy
    sorted_aas = sorted(AMINO_ACIDS, key=lambda x: aa_accuracy.get(x, 0), reverse=True)
    accuracies = [aa_accuracy.get(aa, 0.0) for aa in sorted_aas]
    counts = [aa_counts.get(aa, 0) for aa in sorted_aas]
    
    labels = [f"{aa}\n(n={counts[i]:,})" for i, aa in enumerate(sorted_aas)]
    
    bars = ax.bar(labels, accuracies, color='forestgreen', edgecolor='black')
    
    ax.set_xlabel('Amino Acid', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Per-Amino-Acid Prediction Accuracy', fontsize=14)
    ax.set_ylim(0, 1.1)
    
    mean_acc = sum(accuracies) / len(accuracies) if accuracies else 0
    ax.axhline(y=mean_acc, color='red', linestyle='--', label=f'Mean: {mean_acc:.3f}')
    ax.legend()
    
    plt.xticks(rotation=0)
    plt.tight_layout()
    return fig


def create_confusion_matrix_figure(
    confusion_data: List[Tuple[str, str]],
) -> plt.Figure:
    """Create confusion matrix heatmap."""
    if not confusion_data:
        return None
    
    # Build confusion matrix
    confusion_matrix = np.zeros((len(AMINO_ACIDS), len(AMINO_ACIDS)))
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    
    for true_aa, pred_aa in confusion_data:
        if true_aa in aa_to_idx and pred_aa in aa_to_idx:
            confusion_matrix[aa_to_idx[true_aa], aa_to_idx[pred_aa]] += 1
    
    # Normalize by row (true label)
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    confusion_matrix_norm = confusion_matrix / row_sums
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    sns.heatmap(
        confusion_matrix_norm,
        xticklabels=AMINO_ACIDS,
        yticklabels=AMINO_ACIDS,
        cmap='Blues',
        annot=True,
        fmt='.2f',
        ax=ax,
        cbar_kws={'label': 'Probability'},
        square=True,
    )
    
    ax.set_xlabel('Predicted Amino Acid', fontsize=12)
    ax.set_ylabel('True Amino Acid', fontsize=12)
    ax.set_title('Confusion Matrix (Row-Normalized)', fontsize=14)
    
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 5. W&B LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def log_to_wandb(results: Dict[str, Any], figures: Dict[str, plt.Figure]) -> None:
    """Log all metrics and visualizations to W&B."""
    
    # Log overall metrics
    print("\n" + "="*80)
    print("OVERALL METRICS")
    print("="*80)
    for key, value in results["overall"].items():
        print(f"  {key}: {value}")
        wandb.log({f"overall/{key}": value})
    
    # Log position-wise metrics
    print("\n" + "="*80)
    print("POSITION-WISE ACCURACY")
    print("="*80)
    
    pos_data = []
    for pos in sorted(results["position_wise"]["accuracy"].keys()):
        acc = results["position_wise"]["accuracy"][pos]
        count = results["position_wise"]["counts"].get(pos, 0)
        correct = results["position_wise"]["correct"].get(pos, 0)
        print(f"  Position {pos+1}: {acc:.4f} ({correct:,}/{count:,})")
        wandb.log({f"position/{pos+1}/accuracy": acc})
        wandb.log({f"position/{pos+1}/count": count})
        pos_data.append([pos+1, acc, count, correct])
    
    # Create position table
    pos_table = wandb.Table(
        data=pos_data,
        columns=["Position", "Accuracy", "Total", "Correct"]
    )
    wandb.log({"position_wise/summary_table": pos_table})
    
    # Log amino acid metrics
    print("\n" + "="*80)
    print("AMINO ACID ACCURACY (Top 10)")
    print("="*80)
    
    aa_data = []
    sorted_aas = sorted(
        results["amino_acid_wise"]["accuracy"].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for i, (aa, acc) in enumerate(sorted_aas):
        count = results["amino_acid_wise"]["counts"].get(aa, 0)
        correct = results["amino_acid_wise"]["correct"].get(aa, 0)
        if i < 10:
            print(f"  {aa}: {acc:.4f} ({correct:,}/{count:,})")
        wandb.log({f"amino_acid/{aa}/accuracy": acc})
        wandb.log({f"amino_acid/{aa}/count": count})
        aa_data.append([aa, acc, count, correct])
    
    # Create AA table
    aa_table = wandb.Table(
        data=aa_data,
        columns=["Amino Acid", "Accuracy", "Total", "Correct"]
    )
    wandb.log({"amino_acid/summary_table": aa_table})
    
    # Log figures
    print("\n" + "="*80)
    print("VISUALIZATIONS")
    print("="*80)
    
    for name, fig in figures.items():
        if fig is not None:
            print(f"  Logging: {name}")
            wandb.log({f"visualizations/{name}": wandb.Image(fig)})
            plt.close(fig)
    
    print("\n✅ All results logged to W&B")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TRB/TRA CDR3 masking models with position-wise accuracy"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model (local directory or HuggingFace model name)"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to HuggingFace dataset directory"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="quest-trb-eval",
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--cdr3_mask_length",
        type=int,
        default=5,
        help="Number of positions to mask in middle of CDR3"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to evaluate (train, validation, test)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--use_multi_gpu",
        action="store_true",
        help="Use Accelerate for multi-GPU inference"
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Initialize Accelerator if requested
    accelerator = None
    if args.use_multi_gpu:
        if not ACCELERATE_AVAILABLE:
            raise ImportError("Multi-GPU requires accelerate. Install with: pip install accelerate")
        print("\n🚀 Initializing Accelerate for multi-GPU inference...")
        accelerator = Accelerator()
        args.device = accelerator.device
        print(f"   Number of processes: {accelerator.num_processes}")
        print(f"   Main process: {accelerator.is_main_process}")

    # Initialize W&B (only on main process)
    if accelerator is None or accelerator.is_main_process:
        wandb_config = vars(args).copy()
        wandb_config["device"] = str(wandb_config["device"])
        
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=wandb_config,
        )

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        args.device,
        use_accelerate=args.use_multi_gpu
    )

    # Load dataset
    if accelerator is None or accelerator.is_main_process:
        print(f"\nLoading dataset from: {args.dataset_path}")
    
    dataset = load_from_disk(args.dataset_path)
    
    if args.split not in dataset:
        available = list(dataset.keys())
        raise ValueError(f"Split '{args.split}' not found. Available: {available}")
    
    eval_dataset = dataset[args.split]
    
    if accelerator is None or accelerator.is_main_process:
        print(f"Evaluation dataset size: {len(eval_dataset):,} examples")
    
    if args.max_samples is not None and len(eval_dataset) > args.max_samples:
        if accelerator is None or accelerator.is_main_process:
            print(f"Limiting evaluation to {args.max_samples:,} samples")
        eval_dataset = eval_dataset.shuffle(seed=args.seed).select(range(args.max_samples))

    # Create collator and dataloader
    collator = TRBMaskingCollator(
        tokenizer=tokenizer,
        cdr3_mask_length=args.cdr3_mask_length,
    )
    
    dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
    )

    # Prepare with Accelerator if using multi-GPU
    if accelerator is not None:
        model, dataloader = accelerator.prepare(model, dataloader)

    # Run evaluation
    if accelerator is None or accelerator.is_main_process:
        print("\n" + "="*80)
        print(f"RUNNING TRB EVALUATION (mask_length={args.cdr3_mask_length})")
        print("="*80 + "\n")

    results = run_trb_evaluation(
        model=model,
        dataloader=dataloader,
        tokenizer=tokenizer,
        device=args.device,
        cdr3_mask_length=args.cdr3_mask_length,
        accelerator=accelerator,
    )

    # Process results (only on main process)
    if results is not None:
        # Create visualizations
        figures = {}
        
        figures["position_accuracy"] = create_position_accuracy_figure(
            results["position_wise"]["accuracy"],
            results["position_wise"]["counts"],
            args.cdr3_mask_length,
        )
        
        figures["aa_accuracy"] = create_aa_accuracy_figure(
            results["amino_acid_wise"]["accuracy"],
            results["amino_acid_wise"]["counts"],
        )
        
        figures["confusion_matrix"] = create_confusion_matrix_figure(
            results["confusion_data"]
        )
        
        # Log to W&B
        log_to_wandb(results, figures)
        
        wandb.finish()
        print("\n✅ TRB Evaluation complete!")
    else:
        if accelerator is not None:
            print("✅ Evaluation complete (worker process)")

    # Final synchronization
    if accelerator is not None:
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
