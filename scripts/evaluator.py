#!/usr/bin/env python
"""
evaluate.py
────────────────────────────────────────────────────────
Evaluates a model on a pre-tokenized, "raw" dataset.
"""
import argparse
import math
import os
import copy
import torch
import json
import wandb
import random
import evaluate
from functools import partial
import numpy as np
from collections import Counter
from datasets import load_from_disk, Features
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    AutoTokenizer, AutoModelForMaskedLM,
)
from tqdm.auto import tqdm
from accelerate import Accelerator

os.environ["TOKENIZERS_PARALLELISM"] = "false"



# ---------------------------------------------------------------------
# 1. CLI Argument Parsing
# ---------------------------------------------------------------------
def cli():
    """Define the command-line arguments for the script."""
    p = argparse.ArgumentParser(description="Evaluate a pre-trained MLM.")
    p.add_argument("--raw_dataset_dir", required=True, help="Path to the RAW dataset")
    p.add_argument("--model_name", required=True)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--mlm_prob", type=float, default=0.15)
    p.add_argument("--schema_path", type=str, default=None, help="Optional: Path to a JSON file defining the dataset schema.")
    p.add_argument("--wandb_project", type=str, default=None, help="W&B project name to log metrics.")
    p.add_argument("--test", action="store_true", help="Run on a smaller subset of the validation data.")
    return p.parse_args()

# ---------------------------------------------------------------------
# 2. Helper Functions
# ---------------------------------------------------------------------
def custom_masking_collator(features, tokenizer, rules=None, mlm_probability=0.15):
    """Apply targeted masking based on rules defined in a schema, or random masking if no rules are provided."""
    import random
    for item in features:
        # Convert input_ids and attention_mask to lists to allow in-place modification
        item["input_ids"] = list(item["input_ids"])
        if "attention_mask" in item:
            item["attention_mask"] = list(item["attention_mask"])
        seq_len = len(item["input_ids"])
        labels = [-100] * seq_len  # Default to ignoring all labels
        if rules:  # Schema-based masking
            item_feats = tuple(sorted(item.get("combo_feats", [])))
            for rule in rules:
                if tuple(rule["if_combo_feats"]) == item_feats:
                    action = rule["action"]
                    params = rule["params"]

                    if action == "mask_middle":
                        num_tokens = params["num_tokens"]
                        start_idx = seq_len // 2 - num_tokens // 2
                        end_idx = start_idx + num_tokens
                    elif action == "mask_slice":
                        start_idx = params["start_index"]
                        end_idx   = params["end_index"]
                    else:  # "mask_all" or default
                        start_idx, end_idx = 0, seq_len

                    if 0 <= start_idx < end_idx <= seq_len:
                        labels[start_idx:end_idx] = item["input_ids"][start_idx:end_idx]  # copy token IDs into labels
                        for idx in range(start_idx, end_idx):
                            item["input_ids"][idx] = tokenizer.mask_token_id
                    break  # Stop after finding the first matching rule
        else:  # Random masking
            # Exclude special tokens from masking
            special_token_ids = set(tokenizer.all_special_ids)
            candidate_indices = [i for i, tid in enumerate(item["input_ids"]) if tid not in special_token_ids]
            num_to_mask = max(1, int(round(mlm_probability * len(candidate_indices))))
            mask_indices = random.sample(candidate_indices, min(num_to_mask, len(candidate_indices)))
            for idx in mask_indices:
                labels[idx] = item["input_ids"][idx]
                item["input_ids"][idx] = tokenizer.mask_token_id
        item["labels"] = labels

    
    # Truncate sequences that are too long before padding
    
    # Pad all fields to the same length before calling tokenizer.pad
    max_len = max(len(item["input_ids"]) for item in features)
    for item in features:
        if len(item["input_ids"]) < max_len:
            item["input_ids"] += [tokenizer.pad_token_id] * (max_len - len(item["input_ids"]))
        if len(item["labels"]) < max_len:
            item["labels"] += [-100] * (max_len - len(item["labels"]))
        if "attention_mask" in item and len(item["attention_mask"]) < max_len:
            item["attention_mask"] += [0] * (max_len - len(item["attention_mask"]))
    return tokenizer.pad(features, return_tensors="pt")

def log_predictions_to_wandb(raw_eval_ds, all_input_ids, all_labels, all_preds, tokenizer, num_examples=50):
    """
    Creates and logs a W&B Table with correctly reconstructed sequences.
    """
    print("Generating W&B prediction table...")
    accuracy_metric = evaluate.load("accuracy")
    
    flat_input_ids = [item for batch in all_input_ids for item in batch.tolist()]
    flat_labels = [item for batch in all_labels for item in batch.tolist()]
    flat_preds = [item for batch in all_preds for item in batch.tolist()]
    
    num_total_examples = len(flat_input_ids)
    #indices_to_log = random.sample(range(num_total_examples), min(num_examples, num_total_examples))
    
    table = wandb.Table(columns=["Combo ID", "True Sequence", "Masked Input", "Predicted Sequence", "Mask Accuracy"])

    for i in range(50): # Log first 50 examples
        # --- Get the full original example from the raw dataset ---
        # This ensures we have combo_id and combo_feats available.
        original_example = raw_eval_ds[i]
        combo_id = original_example.get("combo_id", "N/A")
        
        input_ids = flat_input_ids[i]
        labels = flat_labels[i]
        preds = flat_preds[i]

        # Reconstruct the original input sequence before masking
        reconstructed_ids = [
            input_ids[j] if labels[j] == -100 else labels[j]
            for j in range(len(input_ids))
        ]
        # Remove PAD tokens
        reconstructed_ids = [tid for tid in reconstructed_ids if tid != tokenizer.pad_token_id]
        true_sequence = tokenizer.decode(reconstructed_ids, skip_special_tokens=False)

        masked_ids = [tid for tid in input_ids if tid != tokenizer.pad_token_id]
        masked_sequence = tokenizer.decode(masked_ids, skip_special_tokens=False)
        # Reconstruct the predicted sequence (model's fill-in for masked tokens)
        predicted_ids = [
            input_ids[j] if labels[j] == -100 else preds[j]
            for j in range(len(input_ids))
        ]
        predicted_ids = [tid for tid in predicted_ids if tid != tokenizer.pad_token_id]
        predicted_sequence = tokenizer.decode(predicted_ids, skip_special_tokens=False)

        # Calculate accuracy for this single example
        masked_positions = np.array(labels) != -100
        accuracy = accuracy_metric.compute(
            predictions=np.array(preds)[masked_positions],
            references=np.array(labels)[masked_positions],
        )["accuracy"] if np.any(masked_positions) else 1.0

        # Breakpoint if masked_sequence has no [MASK] tokens
        if '[MASK]' not in masked_sequence:
            print("DEBUG: No [MASK] token in masked_sequence")
            print("combo_id:", combo_id)
            print("masked_sequence:", masked_sequence)
            print("true_sequence:", true_sequence)
            print("predicted_sequence:", predicted_sequence)
            print("labels:", labels)
            print("masked positions:", [j for j, l in enumerate(labels) if l != -100])
            print("preds at masked:", [preds[j] for j, l in enumerate(labels) if l != -100])
            print("labels at masked:", [labels[j] for j, l in enumerate(labels) if l != -100])
            breakpoint()

        table.add_data(combo_id, true_sequence, masked_sequence, predicted_sequence, f"{accuracy:.2%}")
        
    wandb.log({"prediction_samples": table})
    print("Prediction table logged to W&B.")

# ---------------------------------------------------------------------
# 3. Main Evaluation Function
# ---------------------------------------------------------------------
def main():
    args = cli()
    accelerator = Accelerator(mixed_precision="fp16")

    # --- 1. Load Model, Tokenizer, and Data ---
    accelerator.print(f"Loading model and tokenizer '{args.model_name}'...")
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = torch.compile(model)
    
    accelerator.print(f"Loading RAW dataset from {args.raw_dataset_dir} (validation split)")
    ds = load_from_disk(args.raw_dataset_dir) 
    if args.test:
        accelerator.print("Running in test mode: using a subset of the validation data.")
        eval_ds = ds["val"].shuffle(seed=42).select(range(100))  # Use a small subset for testing
    else:
        eval_ds = ds["val"]  # validation split only

    accelerator.print("Creating a copy for eval ds for debugging")
    eval_cp = copy.deepcopy(eval_ds)
    # Remove all columns except input_ids and attention_mask from eval_ds
    cols_to_remove = [c for c in eval_ds.column_names if c not in ["input_ids", "attention_mask"]]
    eval_ds = eval_ds.remove_columns(cols_to_remove)

    # --- 2. Setup Collator Based on Workflow ---
    masking_rules = []
    if args.schema_path:
        accelerator.print(f"Loading custom masking schema from {args.schema_path}...")
        with open(args.schema_path, 'r') as f:
            masking_rules = json.load(f).get("masking_rules", [])
    else:
        masking_rules = None

    accelerator.print("Using custom masking collator for evaluation (schema-based or random masking).")
    collator = partial(custom_masking_collator, tokenizer=tokenizer, rules=masking_rules, mlm_probability=args.mlm_prob)

        
    # --- Calculate Stats for Logging ---
    num_sequences = len(eval_cp)
    combo_feats_counts = Counter(tuple(sorted(feats)) for feats in eval_cp["combo_feats"])  # all of them

    # convert the dict keys (tuples) to strings for W&B's sake
    combo_feats_counts_str = {str('_'.join(key)): value for key, value in combo_feats_counts.items()}

    dataset_stats = {
        "total_test_sequences": num_sequences,
        "combo_feats_counts": combo_feats_counts_str,
    }
    accelerator.print(f"Dataset stats calculated: {num_sequences:,} sequences.")

    # --- Initialize W&B (if enabled)  ---
    if args.wandb_project and accelerator.is_main_process:
        run_config = vars(args)
        run_config.update(dataset_stats)  # add new stuff
        wandb.init(project=args.wandb_project, config=run_config)
    # --- Create DataLoader with On-the-Fly Masking Collator ---
    loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=max(1, os.cpu_count() // accelerator.num_processes - 1),
    )

    # Manual collator test for first 100 examples
    mask_token_id = tokenizer.mask_token_id
    for idx in range(min(100, len(eval_ds))):
        example = eval_ds[idx]
        batch = collator([example])
        input_ids = batch["input_ids"][0]
        labels = batch["labels"][0]
        masked_positions = [i for i, l in enumerate(labels) if l != -100]
        mismatch = any(input_ids[i] != mask_token_id for i in masked_positions)
        if masked_positions and mismatch:
            print(f"ERROR in example {idx}")
            print("input_ids:", input_ids)
            print("labels:", labels)
            print("tokenizer.mask_token_id:", mask_token_id)
            print("masked_positions:", masked_positions)
            print("input_ids at masked_positions:", [input_ids[i] for i in masked_positions])
            print("labels at masked_positions:", [labels[i] for i in masked_positions])
            breakpoint()


    model, loader = accelerator.prepare(model, loader)
    model.eval()
    
    # --- Evaluation Loop ---
    pbar = tqdm(range(len(loader)), disable=not accelerator.is_main_process, desc="Evaluating")
    total_loss, total_correct, total_masked = 0.0, 0, 0
    all_input_ids = []
    all_predictions = []
    all_labels = []
    for batch in loader:
        with torch.no_grad():
            outputs = model(**batch)  # ** is magic! calls the forward() method of the model

        total_loss += accelerator.gather_for_metrics(outputs.loss.detach()).sum().item()

        labels      = batch["labels"]
        mask        = labels != -100  # -100: ignore value, not part of MLM loss
        predictions = outputs.logits.argmax(dim=-1)  # token with highest prob
        correct     = (predictions[mask] == labels[mask]).sum()

        total_correct += accelerator.gather_for_metrics(correct).sum().item()
        total_masked  += accelerator.gather_for_metrics(mask.sum()).sum().item()

        # Save all raw ids, labels, preds so we can build the W&B table (if needed)
        all_input_ids.append(  accelerator.gather_for_metrics(batch["input_ids"]))
        all_predictions.append(accelerator.gather_for_metrics(outputs.logits.argmax(dim=-1)))
        all_labels.append(     accelerator.gather_for_metrics(batch["labels"]))
        
        pbar.update(1)

    # --- Calculate + Report Metrics ---
    avg_loss = total_loss / len(loader)
    perplexity  = math.exp(avg_loss)
    mask_accuracy = (total_correct / total_masked) * 100 if total_masked > 0 else 0  # prevent 0/0

    if accelerator.is_main_process:
        metrics = {
            "avg_loss": avg_loss, "perplexity": perplexity, "mask_accuracy": mask_accuracy,
        }

        
        
        # Log to terminal
        print("\n----- MLM Evaluation Results -----")
        for key, value in metrics.items():
            print(f"  {key.replace('_', ' ').capitalize():<18}: {value:.4f}")
        print("------------------------------------")

        # Log to W&B (if enabled) and log an example table
        if wandb.run:
            accelerator.print(f"Logging general metrics to W&B...")
            wandb.log(metrics)  # loss, perplexity, accuracy
            accelerator.print(f"Building W&B prediction table (example masked sequences)...")
            log_predictions_to_wandb(eval_cp, all_input_ids, all_labels, all_predictions, tokenizer)
            accelerator.print("All metrics and example predictions logged to W&B.")
            wandb.finish()
        else:
            accelerator.print("W&B logging skipped.")

if __name__ == "__main__":
    main()