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

# Try to import Accelerate for distributed evaluation
try:
    from accelerate import Accelerator
    from accelerate.utils import gather_object
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    gather_object = None
    print("⚠️  Accelerate not available. Multi-GPU evaluation will not be supported.")
    print("   Install with: pip install accelerate")

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
    device: str = "cuda",
    use_accelerate: bool = False
) -> Tuple[Any, Any]:
    """
    Load model and tokenizer from HuggingFace or local path.
    Supports regular models and PEFT/LoRA adapters.

    Args:
        model_path: Path to model (local or HuggingFace Hub)
        device: Device to load model on (ignored if use_accelerate=True)
        use_accelerate: Whether to use Accelerate for distributed inference

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

    # Move model to device and set to eval mode (if not using Accelerate)
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
    accelerator: Any = None,
    attention_sample_config: Dict[str, int] = None,
) -> Dict[str, Any]:
    """
    Run inference on entire dataset and compute metrics.
    Memory-efficient version that computes metrics on-the-fly.

    Args:
        model: Trained model
        dataloader: DataLoader with validation data
        device: Device to run inference on (ignored if accelerator is provided)
        return_attentions: Whether to return attention weights for sampled examples
        accelerator: Optional Accelerator instance for distributed inference
        attention_sample_config: Dict with 'examples_per_key' and 'max_keys' for sampling

    Returns:
        Dictionary with aggregated metrics and sampled attention data
    """
    model.eval()

    # Overall metrics accumulators
    total_loss = 0.0
    total_correct = 0
    total_masked = 0

    # Per-permutation metrics accumulators
    per_pkey_stats = defaultdict(lambda: {
        "loss_sum": 0.0,
        "correct": 0,
        "masked": 0,
        "num_examples": 0,
    })

    # For attention visualization: sample a few examples per permutation key
    # Only sample specific permutations of interest
    PERMUTATIONS_OF_INTEREST = {
        "tra", "trb", "peptide", "mhc_one", "mhc_two", "tra_trb",
        "peptide_mhc_one", "peptide_mhc_one_mhc_two", "tra_peptide_mhc_one",
        "trb_peptide_mhc_one", "tra_trb_peptide_mhc_one"
    }
    attention_samples = defaultdict(list) if return_attentions else None
    attention_sample_limit = attention_sample_config or {"examples_per_key": 3, "max_keys": 10}
    sampled_pkeys = set()

    example_idx = 0  # Track global example index
    layer_info_printed = False  # Flag to print layer info only once

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference", disable=not accelerator.is_local_main_process if accelerator else False):
            # Move batch to device (unless using accelerator which handles this)
            if accelerator is None:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
            else:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_attentions=return_attentions,
            )

            # Gather results from all processes if using distributed inference
            if accelerator is not None:
                logits = accelerator.gather_for_metrics(outputs.logits)
                labels_gathered = accelerator.gather_for_metrics(labels)
                input_ids_gathered = accelerator.gather_for_metrics(input_ids)

                if return_attentions and outputs.attentions is not None:
                    # Get middle 10 layers
                    num_layers = len(outputs.attentions)
                    middle_start = max(0, (num_layers - 10) // 2)
                    middle_end = min(num_layers, middle_start + 10)
                    middle_layer_indices = list(range(middle_start, middle_end))

                    # Print layer info once
                    if not layer_info_printed and (accelerator is None or accelerator.is_local_main_process):
                        print(f"\n📊 Capturing attention from middle 10 layers: {middle_layer_indices}")
                        print(f"   (Total layers in model: {num_layers})\n")
                        layer_info_printed = True

                    attentions_gathered = {
                        layer_idx: accelerator.gather_for_metrics(outputs.attentions[layer_idx])
                        for layer_idx in middle_layer_indices
                    }
                else:
                    attentions_gathered = None

                # Gather permutation keys if available (collective operation - all processes must call)
                if "permutation_key" in batch:
                    batch_pkeys = gather_object(batch["permutation_key"])
                else:
                    batch_pkeys = None
            else:
                logits = outputs.logits
                labels_gathered = labels
                input_ids_gathered = input_ids

                if return_attentions and outputs.attentions is not None:
                    # Get middle 10 layers
                    num_layers = len(outputs.attentions)
                    middle_start = max(0, (num_layers - 10) // 2)
                    middle_end = min(num_layers, middle_start + 10)
                    middle_layer_indices = list(range(middle_start, middle_end))

                    # Print layer info once
                    if not layer_info_printed:
                        print(f"\n📊 Capturing attention from middle 10 layers: {middle_layer_indices}")
                        print(f"   (Total layers in model: {num_layers})\n")
                        layer_info_printed = True

                    attentions_gathered = {
                        layer_idx: outputs.attentions[layer_idx]
                        for layer_idx in middle_layer_indices
                    }
                else:
                    attentions_gathered = None

                # No gathering needed for single process
                batch_pkeys = batch.get("permutation_key", None)

            # Process results (only on main process or if not using accelerator)
            if accelerator is None or accelerator.is_local_main_process:
                # Move to CPU for processing
                logits_cpu = logits.cpu()
                labels_cpu = labels_gathered.cpu()
                input_ids_cpu = input_ids_gathered.cpu()

                # Set batch_pkeys to None list if not available
                if batch_pkeys is None:
                    batch_pkeys = [None] * logits_cpu.size(0)

                # Process each example in the batch
                for i in range(logits_cpu.size(0)):
                    example_logits = logits_cpu[i:i+1]
                    example_labels = labels_cpu[i:i+1]
                    example_input_ids = input_ids_cpu[i:i+1]
                    pkey = batch_pkeys[i]

                    # Compute metrics for this example
                    metrics = compute_mlm_metrics(example_logits, example_labels)

                    # Skip if no masked tokens
                    if metrics["num_masked"] == 0:
                        example_idx += 1
                        continue

                    # Accumulate overall metrics
                    total_correct += metrics["num_correct"]
                    total_masked += metrics["num_masked"]
                    total_loss += metrics["loss"] * metrics["num_masked"]

                    # Accumulate per-permutation metrics
                    if pkey is not None:
                        per_pkey_stats[pkey]["loss_sum"] += metrics["loss"] * metrics["num_masked"]
                        per_pkey_stats[pkey]["correct"] += metrics["num_correct"]
                        per_pkey_stats[pkey]["masked"] += metrics["num_masked"]
                        per_pkey_stats[pkey]["num_examples"] += 1

                        # Sample attention weights for visualization (only for permutations of interest)
                        if (return_attentions and attentions_gathered is not None and
                            pkey in PERMUTATIONS_OF_INTEREST):

                            if len(attention_samples[pkey]) < attention_sample_limit["examples_per_key"]:
                                sampled_pkeys.add(pkey)
                                # Store attention from all middle layers
                                layer_attentions = {
                                    layer_idx: attn_tensor[i].cpu()
                                    for layer_idx, attn_tensor in attentions_gathered.items()
                                }
                                attention_samples[pkey].append({
                                    "attention": layer_attentions,
                                    "input_ids": example_input_ids[0],
                                    "example_idx": example_idx,
                                })

                    example_idx += 1

                # Free memory
                del logits_cpu, labels_cpu, input_ids_cpu
                if attentions_gathered is not None:
                    del attentions_gathered

    # Only process results on main process (or if not using accelerator)
    if accelerator is None or accelerator.is_local_main_process:
        # Compute overall metrics
        overall_accuracy = total_correct / total_masked if total_masked > 0 else 0.0
        overall_loss = total_loss / total_masked if total_masked > 0 else 0.0
        overall_perplexity = math.exp(overall_loss) if overall_loss < 100 else float('inf')

        # Compute per-permutation metrics from accumulated stats
        per_pkey_metrics = {}
        for pkey, stats in per_pkey_stats.items():
            if stats["masked"] > 0:
                per_pkey_metrics[pkey] = {
                    "accuracy": stats["correct"] / stats["masked"],
                    "loss": stats["loss_sum"] / stats["masked"],
                    "perplexity": math.exp(stats["loss_sum"] / stats["masked"]) if stats["loss_sum"] / stats["masked"] < 100 else float('inf'),
                    "num_examples": stats["num_examples"],
                    "num_masked_tokens": stats["masked"],
                }

        results = {
            "overall": {
                "accuracy": overall_accuracy,
                "loss": overall_loss,
                "perplexity": overall_perplexity,
                "total_masked_tokens": total_masked,
                "total_correct": total_correct,
            },
            "per_permutation": per_pkey_metrics,
            "attention_samples": dict(attention_samples) if attention_samples else None,
        }

        return results
    else:
        # Return empty results for non-main processes
        return None


# NOTE: compute_per_permutation_metrics has been removed.
# Metrics are now computed on-the-fly in run_inference() to save memory.


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ATTENTION VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_attention_maps(
    attention_samples: Dict[str, List[Dict[str, Any]]],
    tokenizer: Any,
) -> Dict[str, plt.Figure]:
    """
    Create attention map visualizations from sampled attention data.
    Averages attention across all sampled examples for each permutation and layer.

    Args:
        attention_samples: Dict mapping pkey -> list of {attention: {layer_idx: tensor}, input_ids, example_idx}
        tokenizer: Tokenizer for decoding sequences

    Returns:
        Dictionary mapping (permutation_key, layer_idx) -> matplotlib Figure
    """
    if not attention_samples:
        return {}

    attention_figures = {}

    for pkey, samples in attention_samples.items():
        if not samples:
            continue

        # Find the minimum sequence length across all samples (for proper averaging)
        min_seq_len = min(
            (sample["input_ids"] != tokenizer.pad_token_id).sum().item()
            for sample in samples
        )

        # Get all layer indices from the first sample
        layer_indices = sorted(samples[0]["attention"].keys())

        # Create visualizations for each layer
        for layer_idx in layer_indices:
            # Average attention across all samples for this layer
            # First, collect all attention matrices (averaged across heads and truncated)
            attention_matrices = []
            for sample in samples:
                # Get attention weights for this layer (average across heads)
                attn = sample["attention"][layer_idx].mean(dim=0)  # [seq_len, seq_len]
                # Truncate to minimum length for consistent averaging
                attn = attn[:min_seq_len, :min_seq_len]
                attention_matrices.append(attn)

            # Average across all samples
            avg_attention = torch.stack(attention_matrices).mean(dim=0).numpy()

            # Get tokens from the first sample (truncated to min length)
            tokens = samples[0]["input_ids"][:min_seq_len]
            token_strs = [tokenizer.decode([t]) for t in tokens]

            # Create figure with single subplot
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))

            # Plot averaged attention heatmap
            sns.heatmap(
                avg_attention,
                cmap="viridis",
                xticklabels=token_strs,
                yticklabels=token_strs,
                ax=ax,
                cbar_kws={"label": "Average Attention Weight"},
                square=True,
            )

            # Title with number of examples averaged and layer info
            ax.set_title(
                f"Average Attention Map: {pkey} - Layer {layer_idx}\n(Averaged across {len(samples)} examples)",
                fontsize=14
            )
            ax.set_xlabel("Key Position")
            ax.set_ylabel("Query Position")

            # Rotate labels for readability
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

            plt.tight_layout()
            # Use tuple key to store both pkey and layer
            attention_figures[f"{pkey}_layer_{layer_idx}"] = fig

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
        for fig_key, fig in attention_figures.items():
            print(f"  Logging attention map for: {fig_key}")
            wandb.log({f"attention_maps/{fig_key}": wandb.Image(fig)})
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
    parser.add_argument(
        "--use_multi_gpu",
        action="store_true",
        help="Use Accelerate for multi-GPU inference"
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ─────────────────────────────────────────────────────────────────────────────
    # Initialize Accelerator if requested
    # ─────────────────────────────────────────────────────────────────────────────

    accelerator = None
    if args.use_multi_gpu:
        if not ACCELERATE_AVAILABLE:
            raise ImportError(
                "Multi-GPU inference requires accelerate. Install with: pip install accelerate"
            )
        print("\n🚀 Initializing Accelerate for multi-GPU inference...")
        accelerator = Accelerator()
        args.device = accelerator.device
        print(f"   Number of processes: {accelerator.num_processes}")
        print(f"   Main process: {accelerator.is_main_process}")
        print(f"   Local main process: {accelerator.is_local_main_process}")

    # Initialize W&B (only on main process)
    if accelerator is None or accelerator.is_main_process:
        # Prepare config for W&B (convert device to string)
        wandb_config = vars(args).copy()
        wandb_config["device"] = str(wandb_config["device"])

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=wandb_config,
        )

    # ─────────────────────────────────────────────────────────────────────────────
    # Load model and tokenizer
    # ─────────────────────────────────────────────────────────────────────────────

    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        args.device,
        use_accelerate=args.use_multi_gpu
    )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Load dataset
    # ─────────────────────────────────────────────────────────────────────────────

    if accelerator is None or accelerator.is_main_process:
        print(f"\nLoading dataset from: {args.dataset_path}")

    dataset = load_from_disk(args.dataset_path)

    # Get validation split
    if "validation" not in dataset:
        raise ValueError(f"Dataset must have 'validation' split. Available: {list(dataset.keys())}")

    val_dataset = dataset["validation"]

    if accelerator is None or accelerator.is_main_process:
        print(f"Validation dataset size: {len(val_dataset):,} examples")

    # Limit samples if specified
    if args.max_samples is not None and len(val_dataset) > args.max_samples:
        if accelerator is None or accelerator.is_main_process:
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

    if accelerator is None or accelerator.is_main_process:
        print(f"Dataset columns: {val_dataset.column_names}")

    # Check if permutation_key is available
    has_permutation_keys = "permutation_key" in val_dataset.column_names
    if accelerator is None or accelerator.is_main_process:
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
    # Prepare model and dataloader with Accelerator if using multi-GPU
    # ─────────────────────────────────────────────────────────────────────────────

    if accelerator is not None:
        model, dataloader = accelerator.prepare(model, dataloader)

    # ─────────────────────────────────────────────────────────────────────────────
    # Run inference
    # ─────────────────────────────────────────────────────────────────────────────

    if accelerator is None or accelerator.is_main_process:
        print("\n" + "="*80)
        print("RUNNING INFERENCE")
        print("="*80 + "\n")

    # Configure attention sampling
    attention_sample_config = {
        "examples_per_key": args.num_attention_examples,
        "max_keys": args.max_attention_pkeys,
    }

    results = run_inference(
        model=model,
        dataloader=dataloader,
        device=args.device,
        return_attentions=args.visualize_attention,
        accelerator=accelerator,
        attention_sample_config=attention_sample_config,
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # Process results (only on main process)
    # ─────────────────────────────────────────────────────────────────────────────

    if results is not None:  # Only main process has results when using multi-GPU
        # Prepare config for logging (convert device to string for JSON serialization)
        config_dict = vars(args).copy()
        config_dict["device"] = str(config_dict["device"])

        # Per-permutation metrics are already computed in run_inference()
        # Filter to only the permutations of interest for logging/visualization
        PERMUTATIONS_OF_INTEREST = {
            "tra", "trb", "peptide", "mhc_one", "mhc_two", "tra_trb",
            "peptide_mhc_one", "peptide_mhc_one_mhc_two", "tra_peptide_mhc_one",
            "trb_peptide_mhc_one", "tra_trb_peptide_mhc_one"
        }
        all_per_pkey_metrics = results.get("per_permutation", {})
        per_pkey_metrics = {
            k: v for k, v in all_per_pkey_metrics.items()
            if k in PERMUTATIONS_OF_INTEREST
        }

        # ─────────────────────────────────────────────────────────────────────────────
        # Generate attention visualizations
        # ─────────────────────────────────────────────────────────────────────────────

        attention_figures = {}
        if args.visualize_attention and results.get("attention_samples"):
            print("\nGenerating attention visualizations...")

            attention_figures = visualize_attention_maps(
                attention_samples=results["attention_samples"],
                tokenizer=tokenizer,
            )

        # ─────────────────────────────────────────────────────────────────────────────
        # Log to W&B
        # ─────────────────────────────────────────────────────────────────────────────

        log_to_wandb(
            overall_metrics=results["overall"],
            per_pkey_metrics=per_pkey_metrics,
            attention_figures=attention_figures,
            config=config_dict,
        )

        # ─────────────────────────────────────────────────────────────────────────────
        # Finish W&B
        # ─────────────────────────────────────────────────────────────────────────────

        wandb.finish()
        print("\n✅ Evaluation complete!")
    else:
        # Non-main processes
        if accelerator is not None:
            print("✅ Evaluation complete (worker process)")

    # Final synchronization - all processes must wait before exit
    if accelerator is not None:
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
