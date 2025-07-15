#!/usr/bin/env python
"""
evaluate.py
────────────────────────────────────────────────────────
Evaluates a model on a pre-tokenized, "raw" dataset.
"""
import argparse
import math
import os
import torch
import json
import wandb
import random
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
    """Defines the command-line arguments for the script."""
    p = argparse.ArgumentParser(description="Evaluate a pre-trained MLM.")
    p.add_argument("--raw_dataset_dir", required=True, help="Path to the RAW dataset")
    p.add_argument("--model_name", required=True)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--mlm_prob", type=float, default=0.15)
    p.add_argument("--schema_path", type=str, default=None, help="Optional: Path to a JSON file defining the dataset schema.")
    p.add_argument("--wandb_project", type=str, default=None, help="W&B project name to log metrics.")
    return p.parse_args()

# ---------------------------------------------------------------------
# 2. Helper Functions
# ---------------------------------------------------------------------
def custom_masking_collator(features, tokenizer, rules):
    """
    Applies targeted masking based on rules defined in a schema,
    then pads the batch.
    """
    # Find the first matching rule for each example
    for item in features:
        item_feats = tuple(sorted(item["combo_feats"]))
        labels = [-100] * len(item["input_ids"])  # Default to ignoring all labels

        for rule in rules:
            if tuple(rule["if_combo_feats"]) == item_feats:
                action = rule["action"]
                params = rule["params"]
                
                if action == "mask_middle":
                    num_tokens = params["num_tokens"]
                    start_idx = len(item["input_ids"]) // 2 - num_tokens // 2
                    end_idx = start_idx + num_tokens
                elif action == "mask_slice":
                    start_idx = params["start_index"]
                    end_idx = params["end_index"]
                else: # "mask_all" or default
                    start_idx, end_idx = 0, len(item["input_ids"])

                if 0 <= start_idx < end_idx <= len(item["input_ids"]):
                    labels[start_idx:end_idx] = item["input_ids"][start_idx:end_idx]
                
                break # Stop after finding the first matching rule
        
        item["labels"] = labels
        
    # Use the tokenizer's robust padding method
    return tokenizer.pad(features, return_tensors="pt")

def log_predictions_to_wandb(all_input_ids, all_labels, all_preds, tokenizer, num_examples=50):
    """
    Creates and logs a W&B Table with sample predictions.
    """
    

    # Combine batches into single tensors
    flat_input_ids = [item for batch in all_input_ids for item in batch]
    flat_labels = [item for batch in all_labels for item in batch]
    flat_preds = [item for batch in all_preds for item in batch]
    
    # Randomly sample indices from the entire dataset
    num_total_examples = len(flat_input_ids)
    indices_to_log = random.sample(range(num_total_examples), min(num_examples, num_total_examples))

    # Create a new table with the desired columns
    table = wandb.Table(columns=["True Sequence", "Predicted Sequence"])

    for i in indices_to_log:
        # Get the data for one example
        input_ids = flat_input_ids[i]
        labels = flat_labels[i]
        preds = flat_preds[i]

        # Reconstruct the true sequence (ignoring padding)
        true_tokens = [tok for tok_id, tok in zip(input_ids, tokenizer.convert_ids_to_tokens(input_ids)) if tok_id != tokenizer.pad_token_id]

        # Reconstruct the predicted sequence
        predicted_tokens = []
        for input_id, label_id, pred_id in zip(input_ids, labels, preds):
            if input_id == tokenizer.pad_token_id:
                break # Stop at padding
            if label_id != -100:
                # This was a masked token; use the model's prediction
                predicted_tokens.append(tokenizer.convert_ids_to_tokens(pred_id))
            else:
                # This was not a masked token; use the original token
                predicted_tokens.append(tokenizer.convert_ids_to_tokens(input_id))
        
        table.add_data(" ".join(true_tokens), " ".join(predicted_tokens))

    # Log the table to W&B
    wandb.log({"prediction_samples": table})

# ---------------------------------------------------------------------
# 3. Main Evaluation Function
# ---------------------------------------------------------------------
def main():
    args = cli()
    accelerator = Accelerator(mixed_precision="fp16")
         
    # --- Load Model and Tokenizer ---
    accelerator.print(f"Loading model and tokenizer '{args.model_name}'...")
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = torch.compile(model)

    # --- Load Data and Optionally Apply Schema ---
    accelerator.print(f"Loading RAW dataset from {args.raw_dataset_dir}")
    test_ds = load_from_disk(args.raw_dataset_dir)["test"]
    masking_rules = []
    if args.schema_path:
        accelerator.print(f"Applying custom masking schema from {args.schema_path}...")
        with open(args.schema_path, 'r') as f:
            masking_rules = json.load(f).get("masking_rules", [])
        
    # --- Calculate Stats for Logging ---
    num_sequences = len(test_ds)
    combo_feats_counts = Counter(tuple(sorted(feats)) for feats in test_ds["combo_feats"])
    combo_feats_counts_str_keys = {str('_'.join(key)): value for key, value in combo_feats_counts.items()}

    dataset_characteristics = {
        "total_test_sequences": num_sequences,
        "combo_feats_counts": combo_feats_counts_str_keys
    }
    accelerator.print(f"Dataset stats calculated: {num_sequences:,} sequences.")

    # --- Initialize W&B with All Config ---
    if args.wandb_project and accelerator.is_main_process:
        run_config = vars(args)
        run_config.update(dataset_characteristics)
        wandb.init(project=args.wandb_project, config=run_config)
        accelerator.print(f"Logging to W&B project: {args.wandb_project}")

    # --- Create DataLoader with On-the-Fly Masking Collator ---
    accelerator.print(f"Loading model and tokenizer '{args.model_name}'...")
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = torch.compile(model)

    if masking_rules:
        # Use the custom collator if rules are provided
        from functools import partial
        collator = partial(custom_masking_collator, tokenizer=tokenizer, rules=masking_rules)
    else:
        # Default to standard on-the-fly MLM if no schema is given
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob
        )
    
    # Remove text columns right before creating the loader
    cols_to_remove = [c for c in test_ds.column_names if c not in ["input_ids", "attention_mask"]]
    eval_ds = test_ds.remove_columns(cols_to_remove)
    
    loader = DataLoader(
        eval_ds.with_format("torch"),
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=max(1, os.cpu_count() // accelerator.num_processes - 1),
    )

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
            outputs = model(**batch)
        
        total_loss += accelerator.gather_for_metrics(outputs.loss.detach()).sum().item()
        
        labels = batch["labels"]
        mask = labels != -100
        predictions = outputs.logits.argmax(dim=-1)
        
        total_correct += accelerator.gather_for_metrics((predictions[mask] == labels[mask]).sum()).sum().item()
        total_masked += accelerator.gather_for_metrics(mask.sum()).sum().item()
        # Collect the raw tensors for the W&B Table
        all_input_ids.append(accelerator.gather_for_metrics(batch["input_ids"]))
        all_predictions.append(accelerator.gather_for_metrics(outputs.logits.argmax(dim=-1)))
        all_labels.append(accelerator.gather_for_metrics(batch["labels"]))
        pbar.update(1)    
        
    # --- Report Metrics ---
    avg_loss = total_loss / len(loader)
    perplexity = math.exp(avg_loss)
    mask_accuracy = (total_correct / total_masked) * 100 if total_masked > 0 else 0
    
    # --- Log metrics to W&B ---
    if accelerator.is_main_process:
        metrics = {
            "avg_loss": avg_loss,
            "perplexity": perplexity,
            "mask_accuracy": mask_accuracy
        }

        
        
        # Log to terminal
        print("\n----- MLM Evaluation Results -----")
        for key, value in metrics.items():
            print(f"  {key.replace('_', ' ').capitalize():<18}: {value:.4f}")
        print("------------------------------------")

        # Log to W&B if enabled
        
        if wandb.run:
            wandb.log(metrics)
            accelerator.print(f"Generating W&B prediction table for random examples...")
            log_predictions_to_wandb(all_input_ids, all_labels, all_predictions, tokenizer)
            accelerator.print("Prediction table logged to W&B.")
            wandb.finish()
            print("Metrics reported to W&B.")

if __name__ == "__main__":
    main()