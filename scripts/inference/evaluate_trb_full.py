#!/usr/bin/env python3
"""
evaluate_trb_full.py
────────────────────────────────────────────────────────────────────────────────
Evaluation script for full TCR sequence position-wise accuracy.

This script evaluates MLM models on the ENTIRE TCR sequence, computing:
1. Overall accuracy and perplexity
2. Position-wise accuracy across the full sequence (0 to max_len)
3. Per-amino-acid accuracy
4. Region-wise accuracy (if ANARCI is available: FR1, CDR1, FR2, CDR2, FR3, CDR3, FR4)
5. Heatmap visualizations of position-wise accuracy

Usage:
    python scripts/inference/evaluate_trb_full.py \
        --model_path checkpoints/esm2_8M_full_dtrb_05Pd_1M_prop/best_model \
        --dataset_path data/tokenized/phase_two/database/1M/proportional/esm2_full_dtrb_1M_proportional/hf_dataset \
        --wandb_project quest-trb-eval \
        --batch_size 32
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

# Try to import Accelerate
try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

# Try to import PEFT
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(
    model_path: str,
    device: str = "cuda",
    use_accelerate: bool = False
) -> Tuple[Any, Any]:
    """Load model and tokenizer."""
    print(f"Loading model from: {model_path}")

    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "adapter_config.json")):
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT not installed")

        print(f"Detected PEFT adapter model")
        with open(os.path.join(model_path, "adapter_config.json"), 'r') as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", "facebook/esm2_t6_8M_UR50D")

        print(f"Loading base model: {base_model_name}")
        base_model = AutoModelForMaskedLM.from_pretrained(base_model_name)
        peft_model = PeftModel.from_pretrained(base_model, model_path)
        print("Merging PEFT weights...")
        model = peft_model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    else:
        model = AutoModelForMaskedLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.mask_token

    if not use_accelerate:
        model = model.to(device)
    model.eval()

    print(f"✅ Model loaded on {device}")
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FULL SEQUENCE MASKING COLLATOR
# ═══════════════════════════════════════════════════════════════════════════════

class FullSequenceMaskingCollator:
    """
    Collator that masks ALL positions in the sequence for evaluation.
    Each position gets its own label so we can compute position-wise accuracy.
    """
    
    def __init__(self, tokenizer: Any, mlm_probability: float = 0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
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
        if token.startswith('<') and token.endswith('>'):
            return True
        return False
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Mask all positions and create batch."""
        
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        batch_seq_positions = []  # Track the position within the actual sequence (0-indexed)
        
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
            seq_positions = [-1] * len(input_ids)  # -1 for special tokens
            
            # Find sequence boundaries (exclude special tokens)
            seq_pos = 0
            for i, token_id in enumerate(input_ids):
                if not self._is_special_token(token_id):
                    # Mask this position
                    label_ids[i] = input_ids[i]
                    seq_positions[i] = seq_pos
                    input_ids[i] = self.mask_token_id
                    seq_pos += 1
            
            batch_input_ids.append(input_ids)
            batch_labels.append(label_ids)
            batch_attention_mask.append(attention_mask)
            batch_seq_positions.append(seq_positions)
        
        # Pad to same length
        max_len = max(len(ids) for ids in batch_input_ids)
        
        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []
        padded_seq_positions = []
        
        for input_ids, labels, attn_mask, seq_pos in zip(
            batch_input_ids, batch_labels, batch_attention_mask, batch_seq_positions
        ):
            pad_len = max_len - len(input_ids)
            padded_input_ids.append(input_ids + [self.pad_token_id] * pad_len)
            padded_labels.append(labels + [-100] * pad_len)
            padded_attention_mask.append(attn_mask + [0] * pad_len)
            padded_seq_positions.append(seq_pos + [-1] * pad_len)
        
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "seq_positions": torch.tensor(padded_seq_positions, dtype=torch.long),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_sequence_evaluation(
    model: Any,
    dataloader: DataLoader,
    tokenizer: Any,
    device: str = "cuda",
    max_seq_len: int = 512,
    accelerator: Any = None,
) -> Dict[str, Any]:
    """Run full sequence evaluation and compute position-wise metrics."""
    model.eval()
    
    # Accumulators
    total_correct = 0
    total_masked = 0
    total_loss_weighted = 0.0
    
    # Position-wise (0 to max_seq_len)
    position_correct = defaultdict(int)
    position_total = defaultdict(int)
    
    # Amino acid-wise
    aa_correct = defaultdict(int)
    aa_total = defaultdict(int)
    
    # Confusion matrix
    all_confusion_data = []
    
    # Create amino acid lookup
    aa_to_idx = {}
    for aa in AMINO_ACIDS:
        tokens = tokenizer.encode(aa, add_special_tokens=False)
        if tokens:
            aa_to_idx[tokens[0]] = aa
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", 
                         disable=accelerator is not None and not accelerator.is_local_main_process):
            
            if accelerator is None:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                seq_positions = batch["seq_positions"].to(device)
            else:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                seq_positions = batch["seq_positions"]
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            if accelerator is not None:
                logits = accelerator.gather_for_metrics(outputs.logits)
                labels_gathered = accelerator.gather_for_metrics(labels)
                seq_positions_gathered = accelerator.gather_for_metrics(seq_positions)
            else:
                logits = outputs.logits
                labels_gathered = labels
                seq_positions_gathered = seq_positions
            
            if accelerator is None or accelerator.is_local_main_process:
                logits_cpu = logits.cpu()
                labels_cpu = labels_gathered.cpu()
                seq_positions_cpu = seq_positions_gathered.cpu()
                
                predictions = torch.argmax(logits_cpu, dim=-1)
                mask = labels_cpu != -100
                
                # Overall metrics
                batch_correct = (predictions[mask] == labels_cpu[mask]).sum().item()
                batch_total = mask.sum().item()
                total_correct += batch_correct
                total_masked += batch_total
                
                if batch_total > 0:
                    loss_fct = torch.nn.CrossEntropyLoss()
                    batch_loss = loss_fct(
                        logits_cpu.view(-1, logits_cpu.size(-1)),
                        labels_cpu.view(-1)
                    ).item()
                    total_loss_weighted += batch_loss * batch_total
                
                # Position-wise and AA-wise
                for batch_idx in range(labels_cpu.size(0)):
                    for seq_idx in range(labels_cpu.size(1)):
                        if labels_cpu[batch_idx, seq_idx] != -100:
                            pos = seq_positions_cpu[batch_idx, seq_idx].item()
                            true_token = labels_cpu[batch_idx, seq_idx].item()
                            pred_token = predictions[batch_idx, seq_idx].item()
                            
                            if 0 <= pos < max_seq_len:
                                position_total[pos] += 1
                                if pred_token == true_token:
                                    position_correct[pos] += 1
                            
                            true_aa = aa_to_idx.get(true_token, None)
                            pred_aa = aa_to_idx.get(pred_token, None)
                            
                            if true_aa is not None:
                                aa_total[true_aa] += 1
                                if pred_token == true_token:
                                    aa_correct[true_aa] += 1
                            
                            if true_aa is not None and pred_aa is not None:
                                all_confusion_data.append((true_aa, pred_aa))
                
                del logits_cpu, labels_cpu, seq_positions_cpu
    
    if accelerator is None or accelerator.is_local_main_process:
        overall_accuracy = total_correct / total_masked if total_masked > 0 else 0.0
        overall_loss = total_loss_weighted / total_masked if total_masked > 0 else 0.0
        overall_perplexity = math.exp(overall_loss) if overall_loss < 100 else float('inf')
        
        # Position-wise accuracy
        pos_accuracy = {}
        for pos in range(max_seq_len):
            if position_total[pos] > 0:
                pos_accuracy[pos] = position_correct[pos] / position_total[pos]
        
        # AA accuracy
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
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_position_accuracy_heatmap(
    position_accuracy: Dict[int, float],
    position_counts: Dict[int, int],
    max_positions: int = None,
) -> plt.Figure:
    """Create heatmap of position-wise accuracy across the full sequence."""
    
    if not position_accuracy:
        return None
    
    max_pos = max(position_accuracy.keys()) + 1
    if max_positions:
        max_pos = min(max_pos, max_positions)
    
    # Create accuracy array
    accuracies = np.zeros(max_pos)
    counts = np.zeros(max_pos)
    
    for pos in range(max_pos):
        accuracies[pos] = position_accuracy.get(pos, 0.0)
        counts[pos] = position_counts.get(pos, 0)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(20, 8), height_ratios=[1, 3])
    
    # Top: Line plot of accuracy across positions
    ax1 = axes[0]
    ax1.plot(range(max_pos), accuracies, 'b-', linewidth=1.5, alpha=0.8)
    ax1.fill_between(range(max_pos), accuracies, alpha=0.3)
    ax1.set_xlim(0, max_pos)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title(f'Position-wise Prediction Accuracy Across Full TCR Sequence (n={sum(counts):.0f} tokens)', fontsize=14)
    ax1.axhline(y=np.mean(accuracies[counts > 0]), color='red', linestyle='--', 
                label=f'Mean: {np.mean(accuracies[counts > 0]):.3f}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add region annotations (approximate for TRB)
    # Typical TRB structure: Leader(~20) + V(~100) + CDR3(~15) + J(~15) + C(~160)
    regions = [
        (0, 20, 'Leader', 'lightblue'),
        (20, 50, 'FR1', 'lightyellow'),
        (50, 58, 'CDR1', 'lightcoral'),
        (58, 75, 'FR2', 'lightyellow'),
        (75, 82, 'CDR2', 'lightcoral'),
        (82, 115, 'FR3', 'lightyellow'),
        (115, 130, 'CDR3', 'lightcoral'),
        (130, 145, 'FR4/J', 'lightyellow'),
        (145, 310, 'Constant', 'lightgreen'),
    ]
    
    for start, end, name, color in regions:
        if end <= max_pos:
            ax1.axvspan(start, min(end, max_pos), alpha=0.2, color=color)
            mid = (start + min(end, max_pos)) / 2
            ax1.text(mid, 0.95, name, ha='center', va='top', fontsize=8, rotation=90)
    
    # Bottom: Heatmap
    ax2 = axes[1]
    
    # Reshape to 2D for better visualization (e.g., 10 rows)
    n_rows = 10
    n_cols = (max_pos + n_rows - 1) // n_rows
    
    heatmap_data = np.zeros((n_rows, n_cols))
    for pos in range(max_pos):
        row = pos // n_cols
        col = pos % n_cols
        if row < n_rows:
            heatmap_data[row, col] = accuracies[pos]
    
    sns.heatmap(
        heatmap_data,
        cmap='RdYlGn',
        vmin=0, vmax=1,
        ax=ax2,
        cbar_kws={'label': 'Accuracy'},
        xticklabels=50,
        yticklabels=[f'{i*n_cols}-{(i+1)*n_cols-1}' for i in range(n_rows)],
    )
    ax2.set_xlabel('Position (mod)', fontsize=12)
    ax2.set_ylabel('Position Range', fontsize=12)
    ax2.set_title('Position-wise Accuracy Heatmap', fontsize=12)
    
    plt.tight_layout()
    return fig


def create_region_accuracy_figure(
    position_accuracy: Dict[int, float],
    position_counts: Dict[int, int],
) -> plt.Figure:
    """Create bar chart of accuracy by TCR region."""
    
    # Define approximate regions for TRB
    regions = {
        'Leader': (0, 20),
        'FR1': (20, 50),
        'CDR1': (50, 58),
        'FR2': (58, 75),
        'CDR2': (75, 82),
        'FR3': (82, 115),
        'CDR3': (115, 130),
        'FR4/J': (130, 145),
        'Constant': (145, 350),
    }
    
    region_accuracy = {}
    region_counts = {}
    
    for region_name, (start, end) in regions.items():
        correct = 0
        total = 0
        for pos in range(start, end):
            if pos in position_accuracy:
                correct += position_counts.get(pos, 0) * position_accuracy.get(pos, 0)
                total += position_counts.get(pos, 0)
        
        if total > 0:
            region_accuracy[region_name] = correct / total
            region_counts[region_name] = total
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = list(region_accuracy.keys())
    accs = [region_accuracy.get(n, 0) for n in names]
    counts = [region_counts.get(n, 0) for n in names]
    
    # Color CDR regions differently
    colors = ['lightcoral' if 'CDR' in n else 'steelblue' for n in names]
    
    bars = ax.bar(names, accs, color=colors, edgecolor='black')
    
    # Add count labels
    for bar, acc, count in zip(bars, accs, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.3f}\n(n={count:,})', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('TCR Region', fontsize=12)
    ax.set_title('Prediction Accuracy by TCR Region', fontsize=14)
    ax.set_ylim(0, 1.1)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightcoral', edgecolor='black', label='CDR regions'),
        Patch(facecolor='steelblue', edgecolor='black', label='Framework/Other'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def create_aa_accuracy_figure(
    aa_accuracy: Dict[str, float],
    aa_counts: Dict[str, int],
) -> plt.Figure:
    """Create bar chart of per-amino-acid accuracy."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    sorted_aas = sorted(AMINO_ACIDS, key=lambda x: aa_accuracy.get(x, 0), reverse=True)
    accuracies = [aa_accuracy.get(aa, 0.0) for aa in sorted_aas]
    counts = [aa_counts.get(aa, 0) for aa in sorted_aas]
    
    labels = [f"{aa}\n({counts[i]:,})" for i, aa in enumerate(sorted_aas)]
    
    bars = ax.bar(labels, accuracies, color='forestgreen', edgecolor='black')
    
    ax.set_xlabel('Amino Acid (count)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Per-Amino-Acid Prediction Accuracy', fontsize=14)
    ax.set_ylim(0, 1.1)
    
    mean_acc = sum(accuracies) / len(accuracies) if accuracies else 0
    ax.axhline(y=mean_acc, color='red', linestyle='--', label=f'Mean: {mean_acc:.3f}')
    ax.legend()
    
    plt.tight_layout()
    return fig


def create_confusion_matrix_figure(confusion_data: List[Tuple[str, str]]) -> plt.Figure:
    """Create confusion matrix heatmap."""
    if not confusion_data:
        return None
    
    confusion_matrix = np.zeros((len(AMINO_ACIDS), len(AMINO_ACIDS)))
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    
    for true_aa, pred_aa in confusion_data:
        if true_aa in aa_to_idx and pred_aa in aa_to_idx:
            confusion_matrix[aa_to_idx[true_aa], aa_to_idx[pred_aa]] += 1
    
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
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
    
    print("\n" + "="*80)
    print("OVERALL METRICS")
    print("="*80)
    for key, value in results["overall"].items():
        print(f"  {key}: {value}")
        wandb.log({f"overall/{key}": value})
    
    # Position-wise summary (binned)
    print("\n" + "="*80)
    print("POSITION-WISE ACCURACY (binned)")
    print("="*80)
    
    pos_acc = results["position_wise"]["accuracy"]
    pos_counts = results["position_wise"]["counts"]
    
    # Bin positions for summary
    bins = [(0, 20, 'Leader'), (20, 50, 'FR1'), (50, 58, 'CDR1'), 
            (58, 75, 'FR2'), (75, 82, 'CDR2'), (82, 115, 'FR3'),
            (115, 130, 'CDR3'), (130, 145, 'FR4'), (145, 350, 'Constant')]
    
    for start, end, name in bins:
        correct = sum(pos_counts.get(p, 0) * pos_acc.get(p, 0) for p in range(start, end))
        total = sum(pos_counts.get(p, 0) for p in range(start, end))
        if total > 0:
            acc = correct / total
            print(f"  {name} ({start}-{end}): {acc:.4f} (n={total:,})")
            wandb.log({f"region/{name}/accuracy": acc})
            wandb.log({f"region/{name}/count": total})
    
    # Log all position accuracies
    for pos, acc in pos_acc.items():
        wandb.log({f"position/{pos}/accuracy": acc})
    
    # Amino acid metrics
    print("\n" + "="*80)
    print("AMINO ACID ACCURACY")
    print("="*80)
    
    sorted_aas = sorted(
        results["amino_acid_wise"]["accuracy"].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for aa, acc in sorted_aas:
        count = results["amino_acid_wise"]["counts"].get(aa, 0)
        print(f"  {aa}: {acc:.4f} (n={count:,})")
        wandb.log({f"amino_acid/{aa}/accuracy": acc})
    
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
        description="Evaluate full TCR sequence position-wise accuracy"
    )
    
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, default="quest-trb-eval")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_multi_gpu", action="store_true")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    accelerator = None
    if args.use_multi_gpu:
        if not ACCELERATE_AVAILABLE:
            raise ImportError("Multi-GPU requires accelerate")
        accelerator = Accelerator()
        args.device = accelerator.device

    if accelerator is None or accelerator.is_main_process:
        wandb_config = vars(args).copy()
        wandb_config["device"] = str(wandb_config["device"])
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=wandb_config)

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device, args.use_multi_gpu)

    if accelerator is None or accelerator.is_main_process:
        print(f"\nLoading dataset from: {args.dataset_path}")
    
    dataset = load_from_disk(args.dataset_path)
    
    if args.split not in dataset:
        raise ValueError(f"Split '{args.split}' not found. Available: {list(dataset.keys())}")
    
    eval_dataset = dataset[args.split]
    
    if accelerator is None or accelerator.is_main_process:
        print(f"Evaluation dataset size: {len(eval_dataset):,}")
    
    if args.max_samples and len(eval_dataset) > args.max_samples:
        eval_dataset = eval_dataset.shuffle(seed=args.seed).select(range(args.max_samples))
        if accelerator is None or accelerator.is_main_process:
            print(f"Limited to {args.max_samples:,} samples")

    collator = FullSequenceMaskingCollator(tokenizer=tokenizer)
    
    dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
    )

    if accelerator is not None:
        model, dataloader = accelerator.prepare(model, dataloader)

    if accelerator is None or accelerator.is_main_process:
        print("\n" + "="*80)
        print("RUNNING FULL SEQUENCE EVALUATION")
        print("="*80 + "\n")

    results = run_full_sequence_evaluation(
        model=model,
        dataloader=dataloader,
        tokenizer=tokenizer,
        device=args.device,
        max_seq_len=args.max_seq_len,
        accelerator=accelerator,
    )

    if results is not None:
        figures = {}
        
        figures["position_accuracy_heatmap"] = create_position_accuracy_heatmap(
            results["position_wise"]["accuracy"],
            results["position_wise"]["counts"],
            max_positions=350,
        )
        
        figures["region_accuracy"] = create_region_accuracy_figure(
            results["position_wise"]["accuracy"],
            results["position_wise"]["counts"],
        )
        
        figures["aa_accuracy"] = create_aa_accuracy_figure(
            results["amino_acid_wise"]["accuracy"],
            results["amino_acid_wise"]["counts"],
        )
        
        figures["confusion_matrix"] = create_confusion_matrix_figure(
            results["confusion_data"]
        )
        
        log_to_wandb(results, figures)
        
        wandb.finish()
        print("\n✅ Full sequence evaluation complete!")
    
    if accelerator is not None:
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
