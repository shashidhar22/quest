#!/usr/bin/env python
"""
Evaluate any Hugging Face *masked-language-model* checkpoint on a dataset
of sequences stored with Dataset.save_to_disk().

Outputs:
  • average MLM loss
  • perplexity  (exp(loss))
  • masked-token top-1 accuracy
"""

import argparse, math, torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
)
from tqdm.auto import tqdm


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True,
                   help="Folder produced by dataset.save_to_disk(), must contain a 'sequence' column.")
    p.add_argument("--model_name",  required=True,
                   help="Model card / Hub ID, e.g. 'Rostlab/prot_bert' or 'bert-base-uncased'.")
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--mlm_prob",    type=float, default=0.15)
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------
def main():
    args   = parse_args()
    device = torch.device(args.device)

    # 1️⃣  Hub pull
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model     = AutoModelForMaskedLM.from_pretrained(args.model_name, trust_remote_code=True)
    model.to(device).eval()

    # 2️⃣  Dataset → tokenise twice (discover max_len, then pad)
    raw_ds  = load_from_disk(args.dataset_path)
    tmp_ds  = raw_ds.map(lambda b: tokenizer(b["sequence"], add_special_tokens=True),
                         batched=True)
    max_len = max(len(ids) for ids in tmp_ds["input_ids"])
    tokenizer.model_max_length = max_len
    print(f"📏  longest sequence = {max_len} tokens")

    ds = raw_ds.map(
        lambda b: tokenizer(b["sequence"], truncation=True,
                            padding="max_length", max_length=max_len),
        batched=True,
        remove_columns=raw_ds.column_names,   # drop raw text
    )

    # 3️⃣  Loader + collator
    collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=True, mlm_probability=args.mlm_prob
    )
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collator)

    # 4️⃣  Eval
    losses, correct, masked = [], 0, 0
    for batch in tqdm(loader, desc="eval"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch)
        losses.append(out.loss.item())

        m = batch["labels"] != -100
        preds = out.logits.argmax(dim=-1)
        correct += (preds[m] == batch["labels"][m]).sum().item()
        masked  += m.sum().item()

    avg_loss = sum(losses) / len(losses)
    print("\n----- MLM evaluation -----")
    print(f"model            : {args.model_name}")
    print(f"avg loss         : {avg_loss:.4f}")
    print(f"perplexity       : {math.exp(avg_loss):.2f}")
    print(f"mask accuracy    : {100*correct/masked:.2f} %")
    print("--------------------------------")


if __name__ == "__main__":
    main()
