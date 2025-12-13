#!/usr/bin/env python
"""
Position-wise TRB Evaluation with Single-Position Masking

This script evaluates a model's ability to predict each position in TRB sequences
by masking ONE position at a time, giving the model full context to make predictions.

This is the correct way to evaluate MLM models position-by-position since the model
was trained to predict ~15% of tokens while seeing the remaining ~85%.

Author: Quest
"""

import os
import sys
import argparse
import logging
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import PeftModel
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONSTANTS & REGIONS
# ═══════════════════════════════════════════════════════════════════════════════

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# TCR region definitions (1-indexed, converted to 0-indexed in code)
# Based on IMGT numbering scheme for TRB
TCR_REGIONS = {
    "Leader": (0, 20),
    "FR1": (20, 50),
    "CDR1": (50, 58),
    "FR2": (58, 75),
    "CDR2": (75, 82),
    "FR3": (82, 115),
    "CDR3": (115, 130),  # Variable, this is approximate
    "FR4": (130, 145),
    "Constant": (145, 350),
}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_path: str,
    device: str = "cuda",
    logger: Any = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model and tokenizer, handling PEFT adapters."""
    log = logger or logging.getLogger(__name__)
    
    log.info(f"Loading model from: {model_path}")
    
    # Check if this is a PEFT adapter
    adapter_config = Path(model_path) / "adapter_config.json"
    
    if adapter_config.exists():
        log.info("Detected PEFT adapter model")
        import json
        with open(adapter_config) as f:
            config = json.load(f)
        base_model_name = config.get("base_model_name_or_path", "facebook/esm2_t6_8M_UR50D")
        log.info(f"Loading base model: {base_model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        base_model = AutoModelForMaskedLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        
        log.info("Merging PEFT weights...")
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
    else:
        log.info("Loading standard model")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
    
    model = model.to(device)
    model.eval()
    
    log.info(f"✅ Model loaded on {device}")
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SINGLE-POSITION MASKING EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def get_sequence_positions(input_ids: torch.Tensor, tokenizer: Any) -> List[int]:
    """Get the indices of non-special tokens in the sequence."""
    special_tokens = {
        tokenizer.cls_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.mask_token_id,
    }
    special_tokens = {t for t in special_tokens if t is not None}
    
    positions = []
    for i, token_id in enumerate(input_ids.tolist()):
        if token_id not in special_tokens:
            positions.append(i)
    return positions


def evaluate_single_sequence_positionwise(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Evaluate a single sequence by masking each position one at a time.
    
    Returns:
        Dict with per-position predictions and ground truth
    """
    mask_token_id = tokenizer.mask_token_id
    
    # Get positions of actual sequence tokens (not special tokens)
    seq_positions = get_sequence_positions(input_ids[0], tokenizer)
    
    if len(seq_positions) == 0:
        return {"predictions": [], "ground_truth": [], "positions": []}
    
    results = {
        "predictions": [],
        "ground_truth": [],
        "positions": [],  # 0-indexed position within sequence
    }
    
    # Process positions in batches
    for batch_start in range(0, len(seq_positions), batch_size):
        batch_positions = seq_positions[batch_start:batch_start + batch_size]
        
        # Create batch with one masked position per sample
        batch_input_ids = []
        batch_labels = []
        
        for pos in batch_positions:
            # Clone and mask this position
            masked_ids = input_ids.clone()
            masked_ids[0, pos] = mask_token_id
            batch_input_ids.append(masked_ids)
            batch_labels.append(input_ids[0, pos].item())
        
        # Stack into batch
        batch_input_ids = torch.cat(batch_input_ids, dim=0).to(device)
        batch_attention_mask = attention_mask.expand(len(batch_positions), -1).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
            )
        
        # Get predictions for each masked position
        for i, (pos, true_token) in enumerate(zip(batch_positions, batch_labels)):
            # Get prediction for the masked position
            logits_at_pos = outputs.logits[i, pos]
            predicted_token = torch.argmax(logits_at_pos).item()
            
            # Map to sequence position (0-indexed within actual sequence)
            seq_pos = seq_positions.index(pos)
            
            results["predictions"].append(predicted_token)
            results["ground_truth"].append(true_token)
            results["positions"].append(seq_pos)
    
    return results


def get_region_for_position(pos: int) -> str:
    """Get the TCR region name for a given position."""
    for region_name, (start, end) in TCR_REGIONS.items():
        if start <= pos < end:
            return region_name
    return "Unknown"


def run_positionwise_evaluation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    device: str = "cuda",
    max_samples: int = 100,
    inner_batch_size: int = 32,
    logger: Any = None,
) -> Dict[str, Any]:
    """
    Run position-wise evaluation across multiple sequences.
    
    Args:
        model: The MLM model
        tokenizer: The tokenizer
        dataset: Dataset with tokenized sequences
        device: Device to use
        max_samples: Maximum number of sequences to evaluate
        inner_batch_size: Batch size for position-level batching within each sequence
        logger: Logger instance
    """
    log = logger or logging.getLogger(__name__)
    
    # Accumulators
    position_correct = defaultdict(int)
    position_total = defaultdict(int)
    
    region_correct = defaultdict(int)
    region_total = defaultdict(int)
    
    aa_correct = defaultdict(int)
    aa_total = defaultdict(int)
    
    total_correct = 0
    total_tokens = 0
    
    # Create token to AA mapping
    token_to_aa = {}
    for aa in AMINO_ACIDS:
        tokens = tokenizer.encode(aa, add_special_tokens=False)
        if tokens:
            token_to_aa[tokens[0]] = aa
    
    log.info(f"\nEvaluating {min(len(dataset), max_samples)} sequences with single-position masking...")
    
    for idx in tqdm(range(min(len(dataset), max_samples)), desc="Sequences"):
        sample = dataset[idx]
        
        input_ids = sample["input_ids"]
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_ids = input_ids.unsqueeze(0)  # [1, seq_len]
        
        attention_mask = sample.get("attention_mask", torch.ones_like(input_ids))
        if isinstance(attention_mask, list):
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        
        # Evaluate each position
        results = evaluate_single_sequence_positionwise(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            device=device,
            batch_size=inner_batch_size,
        )
        
        # Aggregate results
        for pred, true, pos in zip(results["predictions"], results["ground_truth"], results["positions"]):
            is_correct = pred == true
            
            # Position-wise
            position_total[pos] += 1
            if is_correct:
                position_correct[pos] += 1
                total_correct += 1
            total_tokens += 1
            
            # Region-wise
            region = get_region_for_position(pos)
            region_total[region] += 1
            if is_correct:
                region_correct[region] += 1
            
            # AA-wise
            true_aa = token_to_aa.get(true, None)
            pred_aa = token_to_aa.get(pred, None)
            if true_aa:
                aa_total[true_aa] += 1
                if is_correct:
                    aa_correct[true_aa] += 1
    
    # Compute accuracies
    overall_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    
    position_accuracy = {}
    for pos in sorted(position_total.keys()):
        acc = position_correct[pos] / position_total[pos] if position_total[pos] > 0 else 0.0
        position_accuracy[pos] = {
            "accuracy": acc,
            "correct": position_correct[pos],
            "total": position_total[pos],
        }
    
    region_accuracy = {}
    for region in TCR_REGIONS.keys():
        if region in region_total:
            acc = region_correct[region] / region_total[region] if region_total[region] > 0 else 0.0
            region_accuracy[region] = {
                "accuracy": acc,
                "correct": region_correct[region],
                "total": region_total[region],
            }
    
    aa_accuracy = {}
    for aa in AMINO_ACIDS:
        if aa in aa_total:
            acc = aa_correct[aa] / aa_total[aa] if aa_total[aa] > 0 else 0.0
            aa_accuracy[aa] = {
                "accuracy": acc,
                "correct": aa_correct[aa],
                "total": aa_total[aa],
            }
    
    return {
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_tokens": total_tokens,
        "position_accuracy": position_accuracy,
        "region_accuracy": region_accuracy,
        "aa_accuracy": aa_accuracy,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. VISUALIZATION & REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def print_results(results: Dict[str, Any], logger: Any = None):
    """Print evaluation results."""
    log = logger or logging.getLogger(__name__)
    
    log.info("\n" + "=" * 80)
    log.info("OVERALL METRICS")
    log.info("=" * 80)
    log.info(f"  Overall Accuracy: {results['overall_accuracy']:.4f} ({results['overall_accuracy']*100:.2f}%)")
    log.info(f"  Total Correct: {results['total_correct']:,}")
    log.info(f"  Total Tokens: {results['total_tokens']:,}")
    
    log.info("\n" + "=" * 80)
    log.info("REGION-WISE ACCURACY")
    log.info("=" * 80)
    for region, (start, end) in TCR_REGIONS.items():
        if region in results["region_accuracy"]:
            data = results["region_accuracy"][region]
            log.info(f"  {region} ({start}-{end}): {data['accuracy']:.4f} ({data['accuracy']*100:.2f}%) "
                    f"[n={data['total']:,}]")
    
    log.info("\n" + "=" * 80)
    log.info("AMINO ACID ACCURACY")
    log.info("=" * 80)
    for aa in AMINO_ACIDS:
        if aa in results["aa_accuracy"]:
            data = results["aa_accuracy"][aa]
            log.info(f"  {aa}: {data['accuracy']:.4f} ({data['accuracy']*100:.2f}%) [n={data['total']:,}]")
    
    # Print position-wise accuracy for first 30 positions as sample
    log.info("\n" + "=" * 80)
    log.info("POSITION-WISE ACCURACY (first 50 positions)")
    log.info("=" * 80)
    for pos in range(min(50, max(results["position_accuracy"].keys()) + 1)):
        if pos in results["position_accuracy"]:
            data = results["position_accuracy"][pos]
            bar = "█" * int(data["accuracy"] * 20)
            log.info(f"  Pos {pos:3d}: {data['accuracy']:.4f} {bar}")


def log_to_wandb(results: Dict[str, Any], args: argparse.Namespace):
    """Log results to Weights & Biases."""
    try:
        import wandb
        
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model_path": args.model_path,
                "dataset_path": args.dataset_path,
                "max_samples": args.max_samples,
                "inner_batch_size": args.inner_batch_size,
            }
        )
        
        # Log overall metrics
        wandb.log({
            "overall_accuracy": results["overall_accuracy"],
            "total_correct": results["total_correct"],
            "total_tokens": results["total_tokens"],
        })
        
        # Log region accuracy
        for region, data in results["region_accuracy"].items():
            wandb.log({
                f"region/{region}/accuracy": data["accuracy"],
                f"region/{region}/total": data["total"],
            })
        
        # Log AA accuracy
        for aa, data in results["aa_accuracy"].items():
            wandb.log({
                f"amino_acid/{aa}/accuracy": data["accuracy"],
                f"amino_acid/{aa}/total": data["total"],
            })
        
        # Create position accuracy table
        position_data = []
        for pos in sorted(results["position_accuracy"].keys()):
            data = results["position_accuracy"][pos]
            position_data.append([pos, data["accuracy"], data["total"]])
        
        position_table = wandb.Table(
            data=position_data,
            columns=["position", "accuracy", "count"]
        )
        wandb.log({"position_accuracy_table": position_table})
        
        # Create region bar chart
        region_data = [[region, data["accuracy"]] 
                       for region, data in results["region_accuracy"].items()]
        region_table = wandb.Table(data=region_data, columns=["region", "accuracy"])
        wandb.log({
            "region_accuracy_chart": wandb.plot.bar(
                region_table, "region", "accuracy", title="Accuracy by TCR Region"
            )
        })
        
        wandb.finish()
        print("✅ Results logged to W&B")
        
    except ImportError:
        print("⚠️ wandb not installed, skipping logging")
    except Exception as e:
        print(f"⚠️ Failed to log to W&B: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Position-wise TRB Evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model or PEFT adapter")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to HF dataset")
    parser.add_argument("--max_samples", type=int, default=100, help="Max sequences to evaluate")
    parser.add_argument("--inner_batch_size", type=int, default=32, help="Positions to batch together")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--wandb_project", type=str, default="quest-trb-eval", help="W&B project")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device, logger)
    
    # Load dataset
    logger.info(f"\nLoading dataset from: {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    
    # Handle DatasetDict
    if hasattr(dataset, 'keys'):
        if 'validation' in dataset:
            dataset = dataset['validation']
        elif 'test' in dataset:
            dataset = dataset['test']
        elif 'train' in dataset:
            dataset = dataset['train']
        else:
            dataset = dataset[list(dataset.keys())[0]]
    
    logger.info(f"Dataset size: {len(dataset):,}")
    
    # Run evaluation
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING POSITION-WISE EVALUATION (single-position masking)")
    logger.info("=" * 80)
    
    results = run_positionwise_evaluation(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=args.device,
        max_samples=args.max_samples,
        inner_batch_size=args.inner_batch_size,
        logger=logger,
    )
    
    # Print results
    print_results(results, logger)
    
    # Log to W&B
    if args.wandb_project:
        log_to_wandb(results, args)
    
    logger.info("\n✅ Position-wise evaluation complete!")


if __name__ == "__main__":
    main()
