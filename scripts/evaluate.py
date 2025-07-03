#!/usr/bin/env python
"""
evaluate.py
────────────────────────────────────────────────────────
Evaluates a model on a pre-masked, "processed" dataset.
"""
import argparse, math, os, torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm.auto import tqdm
from accelerate import Accelerator

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- A custom, robust data collator to replace the buggy one ---
def manual_padding_collator(features, pad_token_id=0, pad_label_id=-100):
    """
    A simple, manual data collator that correctly handles tensor or list inputs.
    """
    batch = {}
    
    # Find the longest sequence length in this specific batch
    max_length = max(len(f["input_ids"]) for f in features)

    for key in features[0].keys():
        padded_sequences = []
        padding_value = pad_label_id if key == "labels" else pad_token_id
        
        for item in features:
            sequence = item[key]
            # --- Convert tensor to list before padding ---
            # This handles the case where the dataset format is already "torch"
            if isinstance(sequence, torch.Tensor):
                sequence = sequence.tolist()
            
            padding_needed = max_length - len(sequence)
            padded_sequences.append(sequence + [padding_value] * padding_needed)

        # Convert the final list of padded sequences into a single PyTorch tensor
        batch[key] = torch.tensor(padded_sequences)
        
    return batch

def main():
    # --- 1. Setup & Arguments ---
    p = argparse.ArgumentParser(description="Evaluate a pre-trained MLM on a processed dataset.")
    p.add_argument("--processed_dataset_dir", required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--batch_size", type=int, default=128)
    args = p.parse_args()

    accelerator = Accelerator(mixed_precision="fp16")

    # --- 2. Load Model and Tokenizer ---
    accelerator.print(f"🔄  Loading model and tokenizer '{args.model_name}'...")
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = torch.compile(model)

    # --- 3. Load Pre-Processed Data ---
    accelerator.print(f"📂  Loading PROCESSED dataset from {args.processed_dataset_dir}")
    test_ds = load_from_disk(args.processed_dataset_dir)["test"]
    test_ds = test_ds.with_format("torch")

    # --- 4. Create DataLoader with the NEW Manual Collator ---
    accelerator.print("Setting up DataLoader with custom manual collator...")
    
    # Create a partial function to pass the tokenizer's pad_token_id
    from functools import partial
    custom_collator = partial(manual_padding_collator, pad_token_id=tokenizer.pad_token_id)
    
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        collate_fn=custom_collator, # <-- Use new manual collator
        num_workers=max(1, os.cpu_count() // accelerator.num_processes - 1),
    )

    model, loader = accelerator.prepare(model, loader)
    model.eval()

    # --- 5. Evaluation Loop ---
    pbar = tqdm(range(len(loader)), disable=not accelerator.is_main_process, desc="Evaluating")
    total_loss, total_correct, total_masked = 0.0, 0, 0

    for batch in loader:
        with torch.no_grad():
            outputs = model(**batch)

        total_loss += accelerator.gather_for_metrics(outputs.loss.detach()).sum().item()
        
        # Calculate accuracy on masked tokens
        labels = batch["labels"]
        mask = labels != -100
        predictions = outputs.logits.argmax(dim=-1)
        
        total_correct += accelerator.gather_for_metrics((predictions[mask] == labels[mask]).sum()).sum().item()
        total_masked += accelerator.gather_for_metrics(mask.sum()).sum().item()
        pbar.update(1)

    # --- 6. Report Metrics ---
    avg_loss = total_loss / len(loader)
    perplexity = math.exp(avg_loss)
    mask_accuracy = (total_correct / total_masked) * 100 if total_masked > 0 else 0

    if accelerator.is_main_process:
        print("\n----- MLM Evaluation Results -----")
        print(f"  Model            : {args.model_name}")
        print(f"  Dataset          : {args.processed_dataset_dir}")
        print(f"  Avg. Loss        : {avg_loss:.4f}")
        print(f"  Perplexity       : {perplexity:.2f}")
        print(f"  Mask Accuracy    : {mask_accuracy:.2f} %")
        print("------------------------------------")

if __name__ == "__main__":
    main()