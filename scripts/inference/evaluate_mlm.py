#!/usr/bin/env python3
"""
evaluate_mlm.py
────────────────────────────────────────────────────────────────────────────────
Evaluation script for Masked Language Modeling on validation data.

Features:
1. Overall accuracy and perplexity metrics
2. Per-permutation key accuracy and perplexity
3. Attention map visualization for each permutation type
4. W&B logging for all metrics and visualizations

Usage:
    python scripts/inference/evaluate_mlm.py \
        --model_path Rostlab/prot_bert \
        --dataset_path data/masked/validation \
        --wandb_project quest-mlm-eval \
        --batch_size 32
"""

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import wandb
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorWithPadding,
)

import matplotlib.pyplot as plt
import seaborn as sns

# Try to import PEFT for LoRA model support
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠️  PEFT not available. LoRA models will not be supported.")

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(
    model_path: str,
    device: str = "cuda"
) -> Tuple[Any, Any]:
    """
    Load model and tokenizer from HuggingFace or local path.
    Supports regular models and PEFT/LoRA adapters.
    
    Args:
        model_path: Path to model (local or HuggingFace Hub)
        device: Device to load model on
    
    Returns:
        (model, tokenizer)
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
        
        # Load base model name from adapter config
        with open(os.path.join(model_path, "adapter_config.json"), 'r') as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", "Rostlab/prot_bert")
        
        print(f"Loading base model: {base_model_name}")
        base_model = AutoModelForMaskedLM.from_pretrained(base_model_name)
        
        # Load PEFT model and merge weights for faster inference
        peft_model = PeftModel.from_pretrained(base_model, model_path)
        print("Merging PEFT weights into base model for faster inference...")
        model = peft_model.merge_and_unload()
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    else:
        # Load regular HuggingFace model
        model = AutoModelForMaskedLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            print("⚠️  No pad token found, using mask token as pad token")
            tokenizer.pad_token = tokenizer.mask_token
    
    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully on {device}")
    print(f"   Model type: {model.config.model_type}")
    print(f"   Vocab size: {len(tokenizer)}")
    
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# 2. INFERENCE AND METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mlm_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute MLM metrics (accuracy, perplexity) for a batch.
    
    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        labels: Ground truth labels [batch_size, seq_len] (-100 for non-masked)
    
    Returns:
        Dictionary with accuracy, loss, and perplexity
    """
    # Mask out -100 labels
    mask = labels != -100
    
    if mask.sum() == 0:
        # No masked tokens in this batch
        return {
            "accuracy": 0.0,
            "loss": 0.0,
            "perplexity": float('inf'),
            "num_masked": 0,
        }
    
    # Calculate accuracy
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    accuracy = correct / total if total > 0 else 0.0
    
    # Calculate cross-entropy loss
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    ).item()
    
    # Calculate perplexity
    perplexity = math.exp(loss) if loss < 100 else float('inf')
    
    return {
        "accuracy": accuracy,
        "loss": loss,
        "perplexity": perplexity,
        "num_masked": total,
        "num_correct": correct,
    }


def run_inference(
    model: Any,
    dataloader: DataLoader,
    device: str = "cuda",
    return_attentions: bool = False,
) -> Dict[str, Any]:
    """
    Run inference on entire dataset and compute metrics.
    
    Args:
        model: Trained model
        dataloader: DataLoader with validation data
        device: Device to run inference on
        return_attentions: Whether to return attention weights
    
    Returns:
        Dictionary with metrics and optionally attention weights
    """
    model.eval()
    
    all_logits = []
    all_labels = []
    all_attentions = [] if return_attentions else None
    all_permutation_keys = []
    
    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_attentions=return_attentions,
            )
            
            # Store results
            all_logits.append(outputs.logits.cpu())
            all_labels.append(labels.cpu())
            
            if return_attentions and outputs.attentions is not None:
                # Store attention from last layer
                all_attentions.append(outputs.attentions[-1].cpu())
            
            # Store permutation keys if available
            if "permutation_key" in batch:
                all_permutation_keys.extend(batch["permutation_key"])
            
            # Accumulate metrics
            batch_metrics = compute_mlm_metrics(outputs.logits.cpu(), labels.cpu())
            total_correct += batch_metrics["num_correct"]
            total_masked += batch_metrics["num_masked"]
            total_loss += batch_metrics["loss"] * batch_metrics["num_masked"]
    
    # Concatenate all results
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute overall metrics
    overall_accuracy = total_correct / total_masked if total_masked > 0 else 0.0
    overall_loss = total_loss / total_masked if total_masked > 0 else 0.0
    overall_perplexity = math.exp(overall_loss) if overall_loss < 100 else float('inf')
    
    results = {
        "overall": {
            "accuracy": overall_accuracy,
            "loss": overall_loss,
            "perplexity": overall_perplexity,
            "total_masked_tokens": total_masked,
            "total_correct": total_correct,
        },
        "logits": all_logits,
        "labels": all_labels,
        "permutation_keys": all_permutation_keys if all_permutation_keys else None,
    }
    
    if return_attentions and all_attentions:
        results["attentions"] = torch.cat(all_attentions, dim=0)
    
    return results


def compute_per_permutation_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    permutation_keys: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics separately for each permutation key.
    
    Args:
        logits: Model predictions [num_examples, seq_len, vocab_size]
        labels: Ground truth labels [num_examples, seq_len]
        permutation_keys: List of permutation keys for each example
    
    Returns:
        Dictionary mapping permutation_key -> metrics
    """
    if not permutation_keys:
        return {}
    
    # Group examples by permutation key
    pkey_to_indices = defaultdict(list)
    for idx, pkey in enumerate(permutation_keys):
        pkey_to_indices[pkey].append(idx)
    
    per_pkey_metrics = {}
    
    for pkey, indices in pkey_to_indices.items():
        # Get logits and labels for this permutation key
        pkey_logits = logits[indices]
        pkey_labels = labels[indices]
        
        # Compute metrics
        metrics = compute_mlm_metrics(pkey_logits, pkey_labels)
        
        per_pkey_metrics[pkey] = {
            "accuracy": metrics["accuracy"],
            "loss": metrics["loss"],
            "perplexity": metrics["perplexity"],
            "num_examples": len(indices),
            "num_masked_tokens": metrics["num_masked"],
        }
    
    return per_pkey_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ATTENTION VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_attention_maps(
    attentions: torch.Tensor,
    permutation_keys: List[str],
    tokenizer: Any,
    input_ids: torch.Tensor,
    num_examples_per_pkey: int = 3,
    max_pkeys: int = 10,
) -> Dict[str, plt.Figure]:
    """
    Create attention map visualizations for each permutation key.
    
    Args:
        attentions: Attention weights [num_examples, num_heads, seq_len, seq_len]
        permutation_keys: List of permutation keys for each example
        tokenizer: Tokenizer for decoding sequences
        input_ids: Input token IDs for visualization
        num_examples_per_pkey: Number of examples to visualize per permutation key
        max_pkeys: Maximum number of permutation keys to visualize
    
    Returns:
        Dictionary mapping permutation_key -> matplotlib Figure
    """
    if not permutation_keys or attentions is None:
        return {}
    
    # Group examples by permutation key
    pkey_to_indices = defaultdict(list)
    for idx, pkey in enumerate(permutation_keys):
        pkey_to_indices[pkey].append(idx)
    
    # Sort by frequency (most common first)
    sorted_pkeys = sorted(
        pkey_to_indices.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:max_pkeys]
    
    attention_figures = {}
    
    for pkey, indices in sorted_pkeys:
        # Sample a few examples
        sample_indices = indices[:num_examples_per_pkey]
        
        # Create figure with subplots
        fig, axes = plt.subplots(
            len(sample_indices), 1,
            figsize=(12, 4 * len(sample_indices))
        )
        
        if len(sample_indices) == 1:
            axes = [axes]
        
        fig.suptitle(f"Attention Maps: {pkey}", fontsize=16, y=0.995)
        
        for plot_idx, example_idx in enumerate(sample_indices):
            # Get attention weights (average across heads)
            attn = attentions[example_idx].mean(dim=0)  # [seq_len, seq_len]
            
            # Get tokens for this example
            tokens = input_ids[example_idx]
            
            # Decode tokens
            token_strs = [tokenizer.decode([t]) for t in tokens]
            
            # Truncate to non-padding tokens
            seq_len = (tokens != tokenizer.pad_token_id).sum().item()
            attn = attn[:seq_len, :seq_len].numpy()
            token_strs = token_strs[:seq_len]
            
            # Plot attention heatmap
            ax = axes[plot_idx]
            sns.heatmap(
                attn,
                cmap="viridis",
                xticklabels=token_strs,
                yticklabels=token_strs,
                ax=ax,
                cbar_kws={"label": "Attention Weight"},
                square=True,
            )
            ax.set_title(f"Example {plot_idx + 1}", fontsize=12)
            ax.set_xlabel("Key Position")
            ax.set_ylabel("Query Position")
            
            # Rotate labels for readability
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        
        plt.tight_layout()
        attention_figures[pkey] = fig
    
    return attention_figures


# ═══════════════════════════════════════════════════════════════════════════════
# 4. W&B LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def log_to_wandb(
    overall_metrics: Dict[str, float],
    per_pkey_metrics: Dict[str, Dict[str, float]],
    attention_figures: Dict[str, plt.Figure],
    config: Dict[str, Any],
) -> None:
    """
    Log all metrics and visualizations to Weights & Biases.
    
    Args:
        overall_metrics: Overall evaluation metrics
        per_pkey_metrics: Per-permutation key metrics
        attention_figures: Attention map figures
        config: Evaluation configuration
    """
    # Log overall metrics
    print("\n" + "="*80)
    print("OVERALL METRICS")
    print("="*80)
    for key, value in overall_metrics.items():
        print(f"  {key}: {value}")
        wandb.log({f"overall/{key}": value})
    
    # Log per-permutation metrics
    print("\n" + "="*80)
    print("PER-PERMUTATION METRICS")
    print("="*80)
    
    if per_pkey_metrics:
        # Create comparison tables and charts
        pkey_data = []
        
        for pkey, metrics in sorted(
            per_pkey_metrics.items(),
            key=lambda x: x[1]["num_examples"],
            reverse=True
        ):
            print(f"\n{pkey}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value}")
                wandb.log({f"per_permutation/{pkey}/{metric_name}": value})
            
            # Add to comparison table
            pkey_data.append([
                pkey,
                metrics["accuracy"],
                metrics["perplexity"],
                metrics["num_examples"],
                metrics["num_masked_tokens"],
            ])
        
        # Create W&B table
        table = wandb.Table(
            data=pkey_data,
            columns=[
                "Permutation Key",
                "Accuracy",
                "Perplexity",
                "Num Examples",
                "Num Masked Tokens"
            ]
        )
        
        wandb.log({"per_permutation/summary_table": table})
        
        # Create bar charts for accuracy and perplexity
        accuracy_data = [[pkey, metrics["accuracy"]] for pkey, metrics in per_pkey_metrics.items()]
        accuracy_table = wandb.Table(data=accuracy_data, columns=["Permutation Key", "Accuracy"])
        
        perplexity_data = [[pkey, metrics["perplexity"]] for pkey, metrics in per_pkey_metrics.items()]
        perplexity_table = wandb.Table(data=perplexity_data, columns=["Permutation Key", "Perplexity"])
        
        wandb.log({
            "per_permutation/accuracy_comparison": wandb.plot.bar(
                accuracy_table,
                "Permutation Key",
                "Accuracy",
                title="Accuracy by Permutation Key"
            ),
            "per_permutation/perplexity_comparison": wandb.plot.bar(
                perplexity_table,
                "Permutation Key",
                "Perplexity",
                title="Perplexity by Permutation Key"
            ),
        })
    else:
        print("  No permutation key information available")
    
    # Log attention visualizations
    print("\n" + "="*80)
    print("ATTENTION VISUALIZATIONS")
    print("="*80)
    
    if attention_figures:
        for pkey, fig in attention_figures.items():
            print(f"  Logging attention map for: {pkey}")
            wandb.log({f"attention_maps/{pkey}": wandb.Image(fig)})
            plt.close(fig)  # Close to free memory
    else:
        print("  No attention visualizations generated")
    
    print("\n✅ All results logged to W&B")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN EVALUATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MLM models on masked validation data"
    )
    
    # Required arguments
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
        help="Path to HuggingFace dataset directory (must have 'validation' split)"
    )
    
    # W&B arguments
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="quest-mlm-eval",
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated)"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    
    # Visualization arguments
    parser.add_argument(
        "--visualize_attention",
        action="store_true",
        help="Generate attention map visualizations"
    )
    parser.add_argument(
        "--num_attention_examples",
        type=int,
        default=3,
        help="Number of examples to visualize per permutation key"
    )
    parser.add_argument(
        "--max_attention_pkeys",
        type=int,
        default=10,
        help="Maximum number of permutation keys to visualize"
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize W&B
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Load model and tokenizer
    # ─────────────────────────────────────────────────────────────────────────────
    
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Load dataset
    # ─────────────────────────────────────────────────────────────────────────────
    
    print(f"\nLoading dataset from: {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    
    # Get validation split
    if "validation" not in dataset:
        raise ValueError(f"Dataset must have 'validation' split. Available: {list(dataset.keys())}")
    
    val_dataset = dataset["validation"]
    print(f"Validation dataset size: {len(val_dataset):,} examples")
    
    # Limit samples if specified
    if args.max_samples is not None and len(val_dataset) > args.max_samples:
        print(f"Limiting evaluation to {args.max_samples:,} samples")
        val_dataset = val_dataset.shuffle(seed=args.seed).select(range(args.max_samples))
    
    # Check required columns
    required_columns = ["input_ids", "attention_mask", "labels"]
    missing_columns = [col for col in required_columns if col not in val_dataset.column_names]
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}\n"
            f"Available columns: {val_dataset.column_names}\n"
            f"Make sure the dataset has pre-computed masked labels."
        )
    
    print(f"Dataset columns: {val_dataset.column_names}")
    
    # Check if permutation_key is available
    has_permutation_keys = "permutation_key" in val_dataset.column_names
    if has_permutation_keys:
        print("✅ Permutation keys available - will compute per-permutation metrics")
    else:
        print("⚠️  No permutation keys found - skipping per-permutation analysis")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Create DataLoader
    # ─────────────────────────────────────────────────────────────────────────────
    
    # Custom collate function to preserve permutation_key
    def collate_fn(examples):
        batch = {
            "input_ids": torch.stack([torch.tensor(ex["input_ids"]) for ex in examples]),
            "attention_mask": torch.stack([torch.tensor(ex["attention_mask"]) for ex in examples]),
            "labels": torch.stack([torch.tensor(ex["labels"]) for ex in examples]),
        }
        
        if has_permutation_keys:
            batch["permutation_key"] = [ex["permutation_key"] for ex in examples]
        
        return batch
    
    dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Run inference
    # ─────────────────────────────────────────────────────────────────────────────
    
    print("\n" + "="*80)
    print("RUNNING INFERENCE")
    print("="*80 + "\n")
    
    results = run_inference(
        model=model,
        dataloader=dataloader,
        device=args.device,
        return_attentions=args.visualize_attention,
    )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Compute per-permutation metrics
    # ─────────────────────────────────────────────────────────────────────────────
    
    per_pkey_metrics = {}
    if results["permutation_keys"]:
        print("\nComputing per-permutation metrics...")
        per_pkey_metrics = compute_per_permutation_metrics(
            logits=results["logits"],
            labels=results["labels"],
            permutation_keys=results["permutation_keys"],
        )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Generate attention visualizations
    # ─────────────────────────────────────────────────────────────────────────────
    
    attention_figures = {}
    if args.visualize_attention and "attentions" in results:
        print("\nGenerating attention visualizations...")
        
        # Get input_ids for visualization
        input_ids = results["logits"].argmax(dim=-1)  # Use predictions for visualization
        
        attention_figures = visualize_attention_maps(
            attentions=results["attentions"],
            permutation_keys=results["permutation_keys"],
            tokenizer=tokenizer,
            input_ids=input_ids,
            num_examples_per_pkey=args.num_attention_examples,
            max_pkeys=args.max_attention_pkeys,
        )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Log to W&B
    # ─────────────────────────────────────────────────────────────────────────────
    
    log_to_wandb(
        overall_metrics=results["overall"],
        per_pkey_metrics=per_pkey_metrics,
        attention_figures=attention_figures,
        config=vars(args),
    )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Save results to file
    # ─────────────────────────────────────────────────────────────────────────────
    
    output_dir = f"evaluation_results/{wandb.run.name}"
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, "metrics.json")
    with open(results_file, 'w') as f:
        json.dump({
            "overall": results["overall"],
            "per_permutation": per_pkey_metrics,
            "config": vars(args),
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_file}")
    
    wandb.finish()
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
