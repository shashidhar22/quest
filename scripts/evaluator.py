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
    Trainer, TrainingArguments,
)
from tqdm.auto import tqdm
from accelerate import Accelerator
import time

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
    """Efficiently apply targeted or random masking using tensor operations, minimizing memory usage."""
    # Convert input_ids and attention_mask to tensors
    input_ids = [torch.tensor(item["input_ids"]) for item in features]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in features if "attention_mask" in item]
    max_len = max(len(ids) for ids in input_ids)

    # Pad input_ids and attention_mask
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    if attention_mask:
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    else:
        attention_mask = None

    labels = torch.full_like(input_ids, -100)

    for i, item in enumerate(features):
        seq_len = len(item["input_ids"])
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
                        end_idx = params["end_index"]
                    else:  # "mask_all" or default
                        start_idx, end_idx = 0, seq_len

                    if 0 <= start_idx < end_idx <= seq_len:
                        labels[i, start_idx:end_idx] = input_ids[i, start_idx:end_idx]
                        input_ids[i, start_idx:end_idx] = tokenizer.mask_token_id
                    break  # Stop after finding the first matching rule
        else:  # Random masking
            special_token_ids = set(tokenizer.all_special_ids)
            candidate_indices = [j for j, tid in enumerate(item["input_ids"]) if tid not in special_token_ids]
            num_to_mask = max(1, int(round(mlm_probability * len(candidate_indices))))
            if len(candidate_indices) > 0:
                mask_indices = random.sample(candidate_indices, min(num_to_mask, len(candidate_indices)))
                # Ensure at least one token is masked
                if len(mask_indices) == 0:
                    mask_indices = [random.choice(candidate_indices)]
                mask_indices = torch.tensor(mask_indices, dtype=torch.long)
                labels[i, mask_indices] = input_ids[i, mask_indices]
                input_ids[i, mask_indices] = tokenizer.mask_token_id

    batch = {"input_ids": input_ids}
    if attention_mask is not None:
        batch["attention_mask"] = attention_mask
    batch["labels"] = labels
    return batch

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
    #model = torch.compile(model)
    
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
    
    # Prepare a batch of features for profiling
    features = [eval_ds[i] for i in range(128)]  # adjust batch size as needed

    # Profile custom collator
    start = time.time()
    batch_custom = custom_masking_collator(features, tokenizer, rules=masking_rules, mlm_probability=args.mlm_prob)
    print("Custom collator time:", time.time() - start)

    # Profile HF collator
    from transformers import DataCollatorForLanguageModeling
    hf_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=args.mlm_prob)
    start = time.time()
    batch_hf = hf_collator(features)
    print("HF collator time:", time.time() - start)

    # --- Trainer-based evaluation with W&B logging ---
    if args.wandb_project:
        training_args = TrainingArguments(
            output_dir="./results",
            per_device_eval_batch_size=args.batch_size,
            dataloader_num_workers=max(1, os.cpu_count() // accelerator.num_processes - 1),
            fp16=True,
            report_to=["wandb"],
            run_name=args.wandb_project,
            do_train=False,
            do_eval=True,
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=collator,
            eval_dataset=eval_ds,
        )
        eval_results = trainer.evaluate()
        print("Trainer evaluation results:", eval_results)
        if wandb.run:
            wandb.log(eval_results)


if __name__ == "__main__":
    main()