#!/usr/bin/env python
"""
Evaluate any Hugging Face *masked-language-model* checkpoint using Accelerate
to run on all available GPUs **with a pre-tokenised dataset**.
"""
import argparse, math, torch, os
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
)
from tqdm.auto import tqdm
from accelerate import Accelerator, DataLoaderConfiguration

# ──────────────── CLI ────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tok_cache_dir", required=True,
                   help="Folder that contains the *pre-tokenised* dataset.")
    p.add_argument("--model_name",     required=True,
                   help="Model card / Hub ID.")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Per-GPU batch size.")
    p.add_argument("--mlm_prob", type=float, default=0.15)
    return p.parse_args()

# ──────────────── main ───────────────
def main():
    conf = DataLoaderConfiguration(dispatch_batches=True, split_batches=True)
    accelerator = Accelerator(dataloader_config=conf, mixed_precision="bf16")

    args = parse_args()

    # 1️⃣  Model & tokenizer
    accelerator.print(f"🔄  Loading model and tokenizer for '{args.model_name}'…")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model     = AutoModelForMaskedLM.from_pretrained(args.model_name, trust_remote_code=True)

    # 2️⃣  Dataset (already tokenised → just load)
    accelerator.print(f"📂  Loading tokenised dataset from {args.tok_cache_dir}")
    tokenized_ds = load_from_disk(args.tok_cache_dir).with_format("torch")
    num_examples = len(tokenized_ds)

    # 3️⃣  Dataloader (per-GPU batch)
    collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=True, mlm_probability=args.mlm_prob, pad_to_multiple_of=8
    )
    loader = DataLoader(
        tokenized_ds,
        batch_size=args.batch_size,
        num_workers=12,              # keep the GPUs fed
        persistent_workers=True,
        collate_fn=collator,
        drop_last=True,
        pin_memory=True,
    )

    # Prepare for DDP / device placement
    model, loader = accelerator.prepare(model, loader)
    model.eval()

    # Fixed progress-bar length for all ranks
    num_batches = math.ceil(num_examples /
                            (args.batch_size * accelerator.num_processes))
    progress_bar = tqdm(range(num_batches),
                        disable=not accelerator.is_main_process,
                        desc="Evaluating")

    # 4️⃣  Evaluation loop (unchanged)
    all_losses, all_correct, all_masked = [], [], []
    for batch in loader:
        with torch.no_grad():
            out = model(**batch)

        m     = batch["labels"] != -100
        preds = out.logits.argmax(dim=-1)

        all_losses.append(accelerator.gather_for_metrics(out.loss.reshape(1, -1)))
        all_correct.append(accelerator.gather_for_metrics((preds[m] == batch["labels"][m]).sum()))
        all_masked.append(accelerator.gather_for_metrics(m.sum()))
        progress_bar.update(1)

    total_loss   = torch.cat([l.flatten() for l in all_losses]).mean().item()
    total_correct = torch.cat(all_correct).sum().item()
    total_masked  = torch.cat(all_masked).sum().item()

    if accelerator.is_main_process:
        print("\n----- MLM evaluation -----")
        print(f"model            : {args.model_name}")
        print(f"dataset (cached) : {args.tok_cache_dir}")
        print(f"avg loss         : {total_loss:.4f}")
        print(f"perplexity       : {math.exp(total_loss):.2f}")
        print(f"mask accuracy    : {100*total_correct/total_masked:.2f} %")
        print("--------------------------------")

if __name__ == "__main__":
    main()
