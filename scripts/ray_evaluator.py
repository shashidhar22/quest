#!/usr/bin/env python
"""
ray_evaluator.py
────────────────────────────────────────────────────────
Ray-native version of evaluator.py for distributed evaluation.

To resume after interruption, run:
    from ray.train.torch import TorchTrainer
    trainer = TorchTrainer.restore('/home/ubuntu/ray_results/<your_run_dir>')
    result = trainer.fit()
"""
import argparse
import os
import torch
import json
import wandb
import random
import evaluate  # type: ignore
from functools import partial
import numpy as np
from collections import Counter
from datasets import load_from_disk  # type: ignore
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM,  # type: ignore
)
from tqdm.auto import tqdm
import ray
import math
try:
    from ray.train.torch import TorchTrainer  # type: ignore
except ImportError:
    # Fallback for older Ray versions
    from ray.train import TorchTrainer  # type: ignore
from ray.train import ScalingConfig, RunConfig, FailureConfig, CheckpointConfig  # type: ignore
from ray.air import session  # type: ignore
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------
# 1. CLI Argument Parsing
# ---------------------------------------------------------------------
def cli() -> argparse.Namespace:
    """Define the command-line arguments for the script."""
    p = argparse.ArgumentParser(description="Evaluate a pre-trained MLM using Ray.")
    p.add_argument("--raw_dataset_dir", required=True, help="Path to the RAW dataset")
    p.add_argument("--model_name", required=True)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--mlm_prob", type=float, default=0.15)
    p.add_argument("--schema_path", type=str, default=None, help="Optional: Path to a JSON file defining the dataset schema.")
    p.add_argument("--wandb_project", type=str, default=None, help="W&B project name to log metrics.")
    p.add_argument("--test", action="store_true", help="Run on a smaller subset of the validation data.")
    return p.parse_args()

# ---------------------------------------------------------------------
# 2. Helper Functions (EXACTLY THE SAME AS ORIGINAL)
# ---------------------------------------------------------------------
def custom_masking_collator(
    features: list[dict[str, Any]],
    tokenizer: Any,  # Use Any due to missing stubs in transformers
    rules: list[dict[str, Any]] | None = None,
    mlm_probability: float = 0.15
) -> dict[str, Any]:
    """Efficiently apply targeted or random masking using tensor operations, minimizing memory usage."""
    # Convert input_ids and attention_mask to tensors
    input_ids = [torch.tensor(item["input_ids"]) for item in features]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in features if "attention_mask" in item]

    # Fallbacks for pad_token_id and mask_token_id for type checkers and runtime safety
    pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
    if pad_token_id is None:
        pad_token_id = 0
    mask_token_id = getattr(tokenizer, 'mask_token_id', 0)
    if mask_token_id is None:
        mask_token_id = 0

    # Pad input_ids and attention_mask
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
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
                        input_ids[i, start_idx:end_idx] = mask_token_id
                    break  # Stop after finding the first matching rule
        else:  # Random masking
            # Fallback for all_special_ids for type checker/runtime safety
            special_token_ids = set(getattr(tokenizer, 'all_special_ids', []) or [])
            candidate_indices = [j for j, tid in enumerate(item["input_ids"]) if tid not in special_token_ids]
            num_to_mask = max(1, int(round(mlm_probability * len(candidate_indices))))
            if len(candidate_indices) > 0:
                mask_indices = random.sample(candidate_indices, min(num_to_mask, len(candidate_indices)))
                # Ensure at least one token is masked
                if len(mask_indices) == 0:
                    mask_indices = [random.choice(candidate_indices)]
                mask_indices = torch.tensor(mask_indices, dtype=torch.long)
                labels[i, mask_indices] = input_ids[i, mask_indices]
                input_ids[i, mask_indices] = mask_token_id

    batch = {"input_ids": input_ids}
    if attention_mask is not None:
        batch["attention_mask"] = attention_mask
    batch["labels"] = labels
    return batch

def log_predictions_to_wandb(
    all_input_ids: list[list[int]],
    all_labels: list[list[int]],
    all_preds: list[list[int]],
    tokenizer: Any,  # Use Any due to missing stubs in transformers
    num_examples: int = 50
) -> None:
    """
    Creates and logs a W&B Table with correctly reconstructed sequences.
    """
    print("Generating W&B prediction table...")
    accuracy_metric = evaluate.load("accuracy")  # type: ignore
    flat_input_ids = [item for batch in all_input_ids for item in batch]
    flat_labels = [item for batch in all_labels for item in batch]
    flat_preds = [item for batch in all_preds for item in batch]
    table = wandb.Table(columns=["True Sequence", "Masked Input", "Predicted Sequence", "Mask Accuracy"])
    for i in range(min(num_examples, len(flat_input_ids))):
        # Already list[int], use as-is for type checker
        input_ids = flat_input_ids[i]
        labels = flat_labels[i]
        preds = flat_preds[i]
        reconstructed_ids = [input_ids[j] if labels[j] == -100 else labels[j] for j in range(len(input_ids))]  # type: ignore
        reconstructed_ids = [tid for tid in reconstructed_ids if tid != tokenizer.pad_token_id]  # type: ignore
        true_sequence = tokenizer.decode(reconstructed_ids, skip_special_tokens=False)  # type: ignore
        masked_ids = [tid for tid in input_ids if tid != tokenizer.pad_token_id]  # type: ignore
        masked_sequence = tokenizer.decode(masked_ids, skip_special_tokens=False)  # type: ignore
        predicted_ids = [input_ids[j] if labels[j] == -100 else preds[j] for j in range(len(input_ids))]  # type: ignore
        predicted_ids = [tid for tid in predicted_ids if tid != tokenizer.pad_token_id]  # type: ignore
        predicted_sequence = tokenizer.decode(predicted_ids, skip_special_tokens=False)  # type: ignore
        masked_positions = np.array(labels) != -100
        accuracy = accuracy_metric.compute(  # type: ignore
            predictions=np.array(preds)[masked_positions],
            references=np.array(labels)[masked_positions],
        )["accuracy"] if np.any(masked_positions) else 1.0  # type: ignore
        table.add_data(true_sequence, masked_sequence, predicted_sequence, f"{accuracy:.2%}")  # type: ignore
    wandb.log({"prediction_samples": table})
    print("Prediction table logged to W&B.")

# ---------------------------------------------------------------------
# 3. Ray Training Function
# ---------------------------------------------------------------------
def evaluate_func(config: dict[str, Any]) -> None:
    """Ray training function that runs on each worker."""
    
    # --- 1. Load Model, Tokenizer, and Data ---
    print(f"Loading model and tokenizer '{config['model_name']}'...")
    model = AutoModelForMaskedLM.from_pretrained(config["model_name"])  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])  # type: ignore
    
    print(f"Loading RAW dataset from {config['raw_dataset_dir']} (validation split)")
    ds = load_from_disk(config["raw_dataset_dir"]) 
    if config.get("test", False):
        print("Running in test mode: using a subset of the validation data.")
        eval_ds = ds["val"].shuffle(seed=42).select(range(8000))  # type: ignore
    else:
        eval_ds = ds["val"]  # type: ignore

    # Each worker now gets its own portion of the dataset
    world_size = session.get_world_size()  # type: ignore
    world_rank = session.get_world_rank()  # type: ignore

    total_size = len(eval_ds)  # type: ignore
    chunk_size = total_size // world_size  # type: ignore
    start_idx = world_rank * chunk_size  # type: ignore
    end_idx = start_idx + chunk_size if world_rank < world_size - 1 else total_size  # type: ignore

    # Get this worker's portion of the dataset
    eval_ds = eval_ds.select(range(start_idx, end_idx))  # type: ignore

    # Only keep a small sample for W&B
    sample_preds, sample_labels, sample_input_ids = [], [], []
    sample_limit = 50

    # --- 2. Setup Collator Based on Workflow ---
    masking_rules = []
    if config.get("schema_path"):
        print(f"Loading custom masking schema from {config['schema_path']}...")
        with open(config["schema_path"], 'r') as f:
            masking_rules = json.load(f).get("masking_rules", [])
    else:
        masking_rules = None

    print("Using custom masking collator for evaluation (schema-based or random masking).")
    collator = partial(custom_masking_collator, tokenizer=tokenizer, rules=masking_rules, mlm_probability=config["mlm_prob"])

    # --- Calculate Stats for Logging ---
    num_sequences = len(eval_ds)  # type: ignore
    print(f"Number of raw sequences in eval_ds: {num_sequences}")
    combo_feats_counts = Counter(tuple(sorted(feats)) for feats in eval_ds["combo_feats"])  # type: ignore
    # convert the dict keys (tuples) to strings for W&B's sake
    combo_feats_counts_str = {str('_'.join(key)): value for key, value in combo_feats_counts.items()}  # type: ignore
    # (Removed wandb.log here)

    dataset_stats = {  # type: ignore
        "total_test_sequences": num_sequences,
        "combo_feats_counts": combo_feats_counts_str,
        "worker_rank": world_rank,
        "worker_chunk_start": start_idx,
        "worker_chunk_end": end_idx,
    }  # type: ignore
    print(f"Dataset stats calculated: {num_sequences:,} sequences for worker {world_rank}.")

    # --- Initialize W&B (only on rank 0)  ---
    # Remove all wandb usage from here

    # --- Create DataLoader with On-the-Fly Masking Collator ---
    # For best performance, ensure your dataset is on local disk (not S3/EFS) on each node.
    # Set batch_size as high as possible (try 256, 384, 512) without OOM.
    loader = DataLoader(  # type: ignore
        eval_ds,  # type: ignore
        batch_size=config["batch_size"],
        collate_fn=collator,
        num_workers=4,  # Tune this (4 or 8 is usually optimal for AWS)
        pin_memory=True,
    )  # type: ignore

    # --- Move model to device and prepare for DDP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use mixed precision (FP16) for faster inference
    if torch.cuda.is_available():
        model = model.half()  # type: ignore
    model = model.to(device)  # type: ignore
    
    # Ray handles DDP automatically, but we can wrap if needed

    model.eval()  # type: ignore
    
    # Prepare a batch of features for profiling (use smaller batch for profiling)
    features = [eval_ds[i] for i in range(min(128, len(eval_ds)))]  # type: ignore  # adjust batch size as needed

    # --- Evaluation Loop ---
    correct = 0
    total = 0
    total_loss = 0.0
    total_loss_count = 0
    oom_count = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc=f"Evaluating (Worker {world_rank})")):  # type: ignore
            try:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device) if "attention_mask" in batch else None
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)  # type: ignore
                logits = outputs.logits  # type: ignore
                loss = outputs.loss  # type: ignore
                preds = torch.argmax(logits, dim=-1)  # type: ignore
                if i < sample_limit:
                    sample_preds.append(preds.cpu())  # type: ignore
                    sample_labels.append(labels.cpu())  # type: ignore
                    sample_input_ids.append(input_ids.cpu())  # type: ignore
                # Do NOT store all batches in memory!
                # Compute correct predictions for this batch (masked positions only)
                mask = (labels != -100)
                correct += (preds[mask] == labels[mask]).sum().item()
                total += mask.sum().item()
                if loss is not None:
                    total_loss += loss.item() * input_ids.size(0)  # type: ignore
                    total_loss_count += input_ids.size(0)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"[OOM] CUDA out of memory on worker {world_rank} at batch {i}. Consider reducing batch size or model size.")
                    oom_count += 1
                    torch.cuda.empty_cache()
                else:
                    raise
    if oom_count > 0:
        print(f"[WARNING] Worker {world_rank} encountered {oom_count} OOM errors during evaluation. Some batches were skipped.")
    # Only rank 0 will log the W&B table after aggregation (see main)
    if world_rank == 0:
        avg_loss = total_loss / total_loss_count if total_loss_count > 0 else 0.0  # type: ignore
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')  # type: ignore
        session.report({  # type: ignore
            "status": "evaluation_complete",
            "num_sequences": num_sequences,
            "worker_rank": world_rank,
            "worker_chunk_start": start_idx,
            "worker_chunk_end": end_idx,
            "correct": correct,
            "total": total,
            "avg_loss": avg_loss,
            "perplexity": perplexity,
            "sample_preds": [p.tolist() for p in sample_preds],  # type: ignore
            "sample_labels": [l.tolist() for l in sample_labels],  # type: ignore
            "sample_input_ids": [i.tolist() for i in sample_input_ids],  # type: ignore
            "eval_ds": eval_ds, # Pass the full dataset for logging
            "combo_feats_counts_str": combo_feats_counts_str,
        })  # type: ignore
    else:
        session.report({  # type: ignore
            "status": "evaluation_complete",
            "worker_rank": world_rank,
            "num_sequences": num_sequences,
            "combo_feats_counts_str": combo_feats_counts_str,
        })  # type: ignore
    # Return stats for aggregation
    return {
        "num_sequences": num_sequences,  # type: ignore
        "combo_feats_counts_str": combo_feats_counts_str,  # type: ignore
    }

# ---------------------------------------------------------------------
# 4. Main Function
# ---------------------------------------------------------------------
def main() -> None:
    args: argparse.Namespace = cli()
    
    # Initialize Ray (if not already initialized)
    if not ray.is_initialized():  # type: ignore
        ray.init()  # type: ignore
    
    # Create Ray Trainer with fault tolerance
    trainer = TorchTrainer(  # type: ignore
        evaluate_func,
        scaling_config=ScalingConfig(
            num_workers=8,  # 8 distributed workers (1 per g5.xlarge)
            use_gpu=True,
        ),
        run_config=RunConfig(
            failure_config=FailureConfig(max_failures=-1),  # Unlimited retries
            checkpoint_config=CheckpointConfig(num_to_keep=1),
        ),
        train_loop_config={
            "raw_dataset_dir": args.raw_dataset_dir,
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "mlm_prob": args.mlm_prob,
            "schema_path": args.schema_path,
            "wandb_project": args.wandb_project,
            "test": args.test,
        }
    )
    
    # Run the evaluation and collect metrics from all workers
    result = trainer.fit()  # type: ignore
    print("Evaluation completed!")
    v = result.metrics if hasattr(result, 'metrics') and isinstance(result.metrics, dict) else None  # type: ignore
    if v and 'correct' in v and 'total' in v:
        mask_accuracy = v['correct'] / v['total'] if v['total'] > 0 else 0.0  # type: ignore
        avg_loss = v.get('avg_loss', None)  # type: ignore
        perplexity = v.get('perplexity', None)  # type: ignore
        print(f"Mask accuracy (from rank 0): {mask_accuracy:.4f}")
        if avg_loss is not None:
            print(f"Average loss (from rank 0): {avg_loss:.4f}")
        if perplexity is not None:
            print(f"Perplexity (from rank 0): {perplexity:.4f}")
        if args.wandb_project and v.get('sample_preds') and v.get('sample_labels') and v.get('sample_input_ids') and v.get('eval_ds'):  # type: ignore
            print("Logging to W&B...")
            from transformers import AutoTokenizer  # type: ignore
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)  # type: ignore
            wandb.init(project=args.wandb_project, config=vars(args))
            # Reload the full validation split (or subset if test mode)
            from datasets import load_from_disk  # type: ignore
            ds = load_from_disk(args.raw_dataset_dir)
            if args.test:
                eval_ds = ds["val"].shuffle(seed=42).select(range(8000))  # type: ignore
            else:
                eval_ds = ds["val"]  # type: ignore
            num_sequences = len(eval_ds)  # type: ignore
            from collections import Counter  # type: ignore
            combo_feats_counts = Counter(tuple(sorted(feats)) for feats in eval_ds["combo_feats"])  # type: ignore
            combo_feats_counts_str = {str('_'.join(key)): value for key, value in combo_feats_counts.items()}  # type: ignore
            wandb.log({
                "mask_accuracy": mask_accuracy,
                "avg_loss": avg_loss,
                "perplexity": perplexity,
                "num_raw_eval_sequences": num_sequences,
            })
            if combo_feats_counts_str:
                table = wandb.Table(columns=["combo_feats", "count"])
                for k, val in combo_feats_counts_str.items():
                    table.add_data(k, val)  # type: ignore
                wandb.log({"combo_feats_counts": table})
            log_predictions_to_wandb(
                v['sample_input_ids'],  # type: ignore
                v['sample_labels'],  # type: ignore
                v['sample_preds'],  # type: ignore
                tokenizer
            )
            print("W&B logging complete. Exiting.")
    else:
        print("No predictions to aggregate.")

if __name__ == "__main__":
    main() 