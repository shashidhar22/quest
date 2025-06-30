#!/usr/bin/env python
"""
evaluate_mlm.py
────────────────────────────────────────────────────────
Evaluate an MLM checkpoint on the *test* split of a pre-tokenised
dataset saved with `save_to_disk`.

• Handles both single-split datasets and full DatasetDicts.
• Accelerate multi-GPU / BF16 ready.
• Reports loss, perplexity, and mask accuracy.
"""
import argparse, math, os, torch
from datasets import load_from_disk, Dataset
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    default_data_collator,
    DataCollatorWithPadding,
    AutoTokenizer, AutoModelForMaskedLM,
)
from tqdm.auto import tqdm
from accelerate import Accelerator, DataLoaderConfiguration

# ─── CLI ────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--processed_dataset_dir", required=True, help="`save_to_disk` folder")
p.add_argument("--model_name",    required=True)
p.add_argument("--batch_size",    type=int, default=8)
p.add_argument("--mlm_prob",      type=float, default=0.15)
args = p.parse_args()

# --- Accelerator and Model loading is the same ---
accelerator = Accelerator(mixed_precision="fp16")
accelerator.print(f"🔄  Loading model '{args.model_name}'")
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(args.model_name, trust_remote_code=True)
accelerator.print("✅ Model loaded. Compiling for performance...")
model = torch.compile(model)

# --- Data loading is now direct and simple ---
accelerator.print(f"📂  Loading PROCESSED dataset from {args.processed_dataset_dir}")
test_ds = load_from_disk(args.processed_dataset_dir)["test"]
test_ds = test_ds.with_format("torch")
accelerator.print(f"📝  test split rows: {len(test_ds):,}")

# --- DataLoader uses the default collator ---
# The data is already masked and contains the `labels` column.
accelerator.print("Setting up data collator with padding...")
collator = DataCollatorWithPadding(tokenizer=tokenizer)
loader = DataLoader(
    test_ds,
    batch_size=args.batch_size,
    num_workers=min(8, os.cpu_count() // accelerator.num_processes),
    persistent_workers=True,
    collate_fn=collator, # Use the padding collator
    drop_last=False,
    pin_memory=True,
)
model, loader = accelerator.prepare(model, loader)
model.eval()

# ─── 4️⃣  Evaluation loop ──────────────────────────────────────────────
num_batches = len(loader)
pbar = tqdm(range(num_batches),
            disable=not accelerator.is_main_process,
            desc="Evaluating")

tot_loss, tot_correct, tot_masked = 0.0, 0, 0
for batch in loader:
    with torch.no_grad():
        out = model(**batch)

    # gather loss (scalar) across processes
    tot_loss += accelerator.gather_for_metrics(out.loss.detach().reshape(1)).sum().item()

    # mask accuracy
    m     = batch["labels"] != -100
    preds = out.logits.argmax(dim=-1)
    tot_correct += accelerator.gather_for_metrics((preds[m] == batch["labels"][m]).sum()).sum().item()
    tot_masked  += accelerator.gather_for_metrics(m.sum()).sum().item()
    pbar.update(1)

avg_loss   = tot_loss / num_batches
perplexity = math.exp(avg_loss)
mask_acc   = 100 * tot_correct / tot_masked

if accelerator.is_main_process:
    print("\n----- MLM evaluation -----")
    print(f"model            : {args.model_name}")
    print(f"dataset (test)   : {args.tok_cache_dir}")
    print(f"avg loss         : {avg_loss:.4f}")
    print(f"perplexity       : {perplexity:.2f}")
    print(f"mask accuracy    : {mask_acc:.2f} %")
    print("--------------------------------")
