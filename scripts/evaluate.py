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
    AutoTokenizer, AutoModelForMaskedLM,
)
from tqdm.auto import tqdm
from accelerate import Accelerator, DataLoaderConfiguration

# ─── CLI ────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--tok_cache_dir", required=True, help="`save_to_disk` folder")
p.add_argument("--model_name",    required=True)
p.add_argument("--batch_size",    type=int, default=8)
p.add_argument("--mlm_prob",      type=float, default=0.15)
args = p.parse_args()

# ─── Accelerator setup ─────────────────────────────────────────────────
conf = DataLoaderConfiguration(dispatch_batches=True, split_batches=True)
accelerator = Accelerator(dataloader_config=conf, mixed_precision="bf16")

# ─── 1️⃣  Model & tokenizer ────────────────────────────────────────────
accelerator.print(f"🔄  Loading model '{args.model_name}'")
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
model     = AutoModelForMaskedLM.from_pretrained(args.model_name, trust_remote_code=True)

# ─── 2️⃣  Load *test* split only ───────────────────────────────────────
accelerator.print(f"📂  Loading cached dataset from {args.tok_cache_dir}")
raw = load_from_disk(args.tok_cache_dir)
test_ds = raw["test"] if not isinstance(raw, Dataset) else raw

if "labels" in test_ds.column_names:
    test_ds = test_ds.remove_columns("labels")
    
num_examples = len(test_ds)
accelerator.print(f"📝  test split rows: {num_examples:,}")

# keep only tensor columns
tensor_cols = {"input_ids", "attention_mask"}          # ← no "labels" here
test_ds = test_ds.remove_columns([c for c in test_ds.column_names
                                  if c not in tensor_cols]).with_format("torch")
# make sure tensors are PyTorch
test_ds = test_ds.with_format("torch")

# ─── 3️⃣  DataLoader & collator ────────────────────────────────────────
collator = DataCollatorForLanguageModeling(
    tokenizer,
    mlm=True,
    mlm_probability=args.mlm_prob,
    pad_to_multiple_of=8,
)

loader = DataLoader(
    test_ds,
    batch_size=args.batch_size,
    num_workers=min(8, os.cpu_count() // accelerator.num_processes),
    persistent_workers=True,
    collate_fn=collator,
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
    breakpoint()
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
