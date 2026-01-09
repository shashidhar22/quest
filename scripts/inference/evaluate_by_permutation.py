#!/usr/bin/env python3
"""
evaluate_by_permutation.py
────────────────────────────────────────────────────────────────────────────────
Evaluates PEFT ESM2 MLM model on a stratified sample from foundation_permutations.
Samples N sequences per permutation_key and reports metrics + examples per permutation.

Usage:
    python scripts/inference/evaluate_by_permutation.py \
        --checkpoint_path data/model/esm2t33_foundation_100M/best_model.pt \
        --permutations_path data/deduplicated/full/foundation_permutations \
        --output_report permutation_eval_report.txt \
        --samples_per_key 325 \
        --examples_per_key 5
"""

import argparse
import glob
import hashlib
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load the PEFT model from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = checkpoint.get("config", {})
    model_name = config.get("model_name", "facebook/esm2_t33_650M_UR50D")

    print(f"Base model: {model_name}", flush=True)
    print(f"Best val loss from training: {checkpoint.get('best_val_loss', 'N/A'):.6f}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    try:
        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        print("Using Flash Attention 2", flush=True)
    except Exception:
        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )

    if config.get("use_lora", True):
        lora_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=0.0,  # No dropout for inference
            target_modules=["query", "key", "value", "dense"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)

    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, tokenizer, config


def sample_sequences_by_permutation(
    permutations_path: str,
    samples_per_key: int,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Sample sequences grouped by permutation_key."""
    np.random.seed(seed)

    parquet_files = sorted(glob.glob(os.path.join(permutations_path, "*.parquet")))
    print(f"Found {len(parquet_files)} parquet files", flush=True)

    sequences_by_key: Dict[str, List[str]] = defaultdict(list)
    keys_completed: set = set()

    # Shuffle for diversity
    np.random.shuffle(parquet_files)

    pbar = tqdm(parquet_files, desc="Sampling sequences", unit="file")
    for pq_file in pbar:
        pbar.set_postfix({"keys_done": f"{len(keys_completed)}/325"})

        # Read parquet file (only needed columns)
        df = pd.read_parquet(pq_file, columns=["permutation_key", "sequence"])

        # Filter out 'empty' and group by permutation_key
        df = df[df["permutation_key"] != "empty"]

        # Process each permutation key in this file (vectorized)
        for pkey, group in df.groupby("permutation_key"):
            if pkey in keys_completed:
                continue

            needed = samples_per_key - len(sequences_by_key[pkey])
            if needed <= 0:
                keys_completed.add(pkey)
                continue

            # Take up to 'needed' sequences from this group
            seqs = group["sequence"].head(needed).tolist()
            sequences_by_key[pkey].extend(seqs)

            if len(sequences_by_key[pkey]) >= samples_per_key:
                keys_completed.add(pkey)

        # Early exit if we have enough
        if len(keys_completed) >= 325:
            pbar.set_postfix({"keys_done": f"{len(keys_completed)}/325", "status": "complete"})
            break

    pbar.close()

    # Trim to exact count
    for pkey in sequences_by_key:
        if len(sequences_by_key[pkey]) > samples_per_key:
            sequences_by_key[pkey] = sequences_by_key[pkey][:samples_per_key]

    total = sum(len(seqs) for seqs in sequences_by_key.values())
    print(f"Collected {total:,} sequences across {len(sequences_by_key)} permutation keys")

    return dict(sequences_by_key)


def apply_mlm_masking(
    input_ids: torch.Tensor,
    tokenizer,
    mlm_probability: float = 0.15,
) -> tuple:
    """Apply MLM masking to input_ids."""
    labels = input_ids.clone()

    # Special tokens to exclude from masking
    special_ids = [tokenizer.pad_token_id, tokenizer.cls_token_id,
                   tokenizer.eos_token_id, tokenizer.sep_token_id,
                   tokenizer.unk_token_id]
    special_ids = [x for x in special_ids if x is not None]
    special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for sid in special_ids:
        special_mask |= (input_ids == sid)

    # Random masking
    rand = torch.rand(input_ids.shape)
    mask_indices = (rand < mlm_probability) & ~special_mask

    # Labels: -100 for non-masked positions
    labels[~mask_indices] = -100

    # Apply masking: 80% [MASK], 10% random, 10% unchanged
    masked_input = input_ids.clone()
    mask_type = torch.rand(input_ids.shape)

    # 80% -> [MASK]
    mask_token_indices = mask_indices & (mask_type < 0.8)
    masked_input[mask_token_indices] = tokenizer.mask_token_id

    # 10% -> random token
    random_token_indices = mask_indices & (mask_type >= 0.8) & (mask_type < 0.9)
    if random_token_indices.any():
        masked_input[random_token_indices] = torch.randint(
            len(tokenizer), (random_token_indices.sum(),), dtype=torch.long
        )

    return masked_input, labels, mask_indices


def evaluate_permutation(
    model,
    tokenizer,
    sequences: List[str],
    device: torch.device,
    mlm_probability: float = 0.15,
    num_examples: int = 5,
) -> Dict[str, Any]:
    """Evaluate model on sequences from a single permutation_key."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    examples = []

    with torch.no_grad():
        for seq_idx, sequence in enumerate(sequences):
            # Tokenize
            encoded = tokenizer(
                sequence,
                padding=False,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            # Apply masking
            masked_input, labels, mask_indices = apply_mlm_masking(
                input_ids, tokenizer, mlm_probability
            )

            # Move to device
            masked_input = masked_input.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Forward pass
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=masked_input,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            logits = outputs.logits.float()
            predictions = torch.argmax(logits, dim=-1)

            # Compute metrics for this sequence
            mask = labels[0] != -100
            if mask.sum() > 0:
                correct = (predictions[0][mask] == labels[0][mask]).sum().item()
                total = mask.sum().item()
                total_correct += correct
                total_masked += total

                # Loss
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                ).item()
                total_loss += loss * total

            # Collect examples
            if len(examples) < num_examples and mask.sum() > 0:
                # Decode sequences
                raw_seq = tokenizer.decode(input_ids[0], skip_special_tokens=True)

                # For masked sequence, keep special tokens to show <mask> tokens
                # but remove CLS/EOS at start/end
                masked_seq = tokenizer.decode(masked_input[0].cpu(), skip_special_tokens=False)
                # Remove CLS token at start and EOS token at end
                if tokenizer.cls_token and masked_seq.startswith(tokenizer.cls_token):
                    masked_seq = masked_seq[len(tokenizer.cls_token):]
                if tokenizer.eos_token and masked_seq.endswith(tokenizer.eos_token):
                    masked_seq = masked_seq[:-len(tokenizer.eos_token)]

                # Create predicted sequence (replace masked positions with predictions)
                pred_ids = input_ids[0].clone()
                for j in range(len(mask)):
                    if mask[j]:
                        pred_ids[j] = predictions[0][j].cpu()
                pred_seq = tokenizer.decode(pred_ids, skip_special_tokens=True)

                examples.append({
                    "raw_sequence": raw_seq.replace(" ", ""),
                    "masked_sequence": masked_seq.replace(" ", "").strip(),
                    "predicted_sequence": pred_seq.replace(" ", ""),
                    "num_masked": total,
                    "num_correct": correct,
                    "accuracy": correct / total if total > 0 else 0.0,
                })

    # Compute overall metrics
    if total_masked > 0:
        accuracy = total_correct / total_masked
        avg_loss = total_loss / total_masked
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    else:
        accuracy = 0.0
        avg_loss = 0.0
        perplexity = float('inf')

    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "perplexity": perplexity,
        "total_masked": total_masked,
        "total_correct": total_correct,
        "num_sequences": len(sequences),
        "examples": examples,
    }


def write_report(
    results_by_key: Dict[str, Dict],
    config: Dict,
    args: argparse.Namespace,
    output_path: str,
):
    """Write evaluation report to file."""
    # Compute overall metrics
    total_correct = sum(r["total_correct"] for r in results_by_key.values())
    total_masked = sum(r["total_masked"] for r in results_by_key.values())
    overall_accuracy = total_correct / total_masked if total_masked > 0 else 0.0

    total_loss = sum(r["loss"] * r["total_masked"] for r in results_by_key.values())
    overall_loss = total_loss / total_masked if total_masked > 0 else 0.0
    overall_perplexity = math.exp(overall_loss) if overall_loss < 100 else float('inf')

    with open(output_path, "w") as f:
        # Header
        f.write("=" * 100 + "\n")
        f.write("ESM2 MLM EVALUATION REPORT - BY PERMUTATION KEY\n")
        f.write("=" * 100 + "\n\n")

        # Metadata
        f.write("EVALUATION DETAILS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoint: {args.checkpoint_path}\n")
        f.write(f"Permutations Path: {args.permutations_path}\n")
        f.write(f"Samples per Key: {args.samples_per_key}\n")
        f.write(f"Examples per Key: {args.examples_per_key}\n")
        f.write(f"MLM Probability: {config.get('mlm_probability', 0.15)}\n")
        f.write("\n")

        # Overall metrics
        f.write("=" * 100 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Accuracy:           {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)\n")
        f.write(f"Loss:               {overall_loss:.6f}\n")
        f.write(f"Perplexity:         {overall_perplexity:.4f}\n")
        f.write(f"Total Masked Tokens: {total_masked:,}\n")
        f.write(f"Total Correct:      {total_correct:,}\n")
        f.write(f"Permutation Keys:   {len(results_by_key)}\n")
        f.write("\n")

        # Per-permutation summary table
        f.write("=" * 100 + "\n")
        f.write("PER-PERMUTATION SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"{'Permutation Key':<50} {'Accuracy':>10} {'Loss':>10} {'Perplexity':>12} {'Samples':>8}\n")
        f.write("-" * 90 + "\n")

        for pkey in sorted(results_by_key.keys()):
            r = results_by_key[pkey]
            f.write(f"{pkey:<50} {r['accuracy']:>10.4f} {r['loss']:>10.6f} {r['perplexity']:>12.4f} {r['num_sequences']:>8}\n")

        f.write("\n")

        # Per-permutation examples
        f.write("=" * 100 + "\n")
        f.write("EXAMPLES BY PERMUTATION KEY\n")
        f.write("=" * 100 + "\n\n")

        for pkey in sorted(results_by_key.keys()):
            r = results_by_key[pkey]
            f.write(f"\n{'='*80}\n")
            f.write(f"PERMUTATION: {pkey}\n")
            f.write(f"Accuracy: {r['accuracy']:.4f} | Loss: {r['loss']:.6f} | Perplexity: {r['perplexity']:.4f}\n")
            f.write(f"{'='*80}\n\n")

            for ex_idx, ex in enumerate(r["examples"]):
                f.write(f"--- Example {ex_idx + 1} ---\n")
                f.write(f"Accuracy: {ex['accuracy']:.2%} ({ex['num_correct']}/{ex['num_masked']} tokens)\n\n")
                f.write(f"RAW SEQUENCE:\n{ex['raw_sequence']}\n\n")
                f.write(f"MASKED SEQUENCE:\n{ex['masked_sequence']}\n\n")
                f.write(f"PREDICTED SEQUENCE:\n{ex['predicted_sequence']}\n\n")
                f.write("-" * 60 + "\n\n")

        f.write("=" * 100 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 100 + "\n")

    print(f"\nReport written to: {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PEFT ESM2 MLM by permutation key"
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/ubuntu/quest/data/model/esm2t33_foundation_100M/best_model.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--permutations_path",
        type=str,
        default="/home/ubuntu/quest/data/deduplicated/full/foundation_permutations",
        help="Path to foundation_permutations parquet files"
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default="permutation_eval_report.txt",
        help="Path to write evaluation report"
    )
    parser.add_argument(
        "--samples_per_key",
        type=int,
        default=325,
        help="Number of sequences per permutation_key"
    )
    parser.add_argument(
        "--examples_per_key",
        type=int,
        default=5,
        help="Number of examples to show per permutation_key in report"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # Load model
    print("\n" + "=" * 80, flush=True)
    print("Loading Model", flush=True)
    print("=" * 80, flush=True)

    model, tokenizer, config = load_model_from_checkpoint(args.checkpoint_path, device)
    mlm_probability = config.get("mlm_probability", 0.15)

    # Sample sequences
    print("\n" + "=" * 80, flush=True)
    print("Sampling Sequences", flush=True)
    print("=" * 80, flush=True)

    sequences_by_key = sample_sequences_by_permutation(
        args.permutations_path,
        args.samples_per_key,
        args.seed,
    )

    # Evaluate each permutation
    print("\n" + "=" * 80, flush=True)
    print("Evaluating by Permutation Key", flush=True)
    print("=" * 80, flush=True)

    results_by_key = {}
    pkeys = sorted(sequences_by_key.keys())

    pbar = tqdm(pkeys, desc="Evaluating permutations", unit="key")
    for pkey in pbar:
        pbar.set_postfix({"key": pkey[:30]})

        sequences = sequences_by_key[pkey]
        results = evaluate_permutation(
            model, tokenizer, sequences, device,
            mlm_probability, args.examples_per_key
        )
        results_by_key[pkey] = results

        pbar.set_postfix({"key": pkey[:20], "acc": f"{results['accuracy']:.3f}"})

    # Write report
    print("\n" + "=" * 80, flush=True)
    print("Writing Report", flush=True)
    print("=" * 80, flush=True)

    write_report(results_by_key, config, args, args.output_report)

    # Print summary
    total_correct = sum(r["total_correct"] for r in results_by_key.values())
    total_masked = sum(r["total_masked"] for r in results_by_key.values())
    overall_accuracy = total_correct / total_masked if total_masked > 0 else 0.0

    print("\n" + "=" * 80, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)", flush=True)
    print(f"Permutation Keys: {len(results_by_key)}", flush=True)
    print(f"Total Sequences: {sum(r['num_sequences'] for r in results_by_key.values()):,}", flush=True)
    print(f"Report: {args.output_report}", flush=True)


if __name__ == "__main__":
    main()
