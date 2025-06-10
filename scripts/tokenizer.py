#!/usr/bin/env python
"""
preprocess_tok.py
─────────────────
One-time script that:
  1. Downloads the raw Arrow shards from S3
  2. Tokenises every sequence with the specified model tokenizer
  3. Saves the tokenised dataset to /opt/dlami/nvme/tok_<model>_max<LEN>

Only rank-0 does the heavy lifting; other ranks just wait and load.
"""

import argparse, os, json
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from accelerate import Accelerator

# ─────────────────────── CLI ────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--model_name", required=True,
               help="HF checkpoint with a compatible tokenizer")
p.add_argument("--s3_root",    required=True,
               help="s3://… folder that contains test/*.arrow")
p.add_argument("--max_len",    type=int, required=True)
args = p.parse_args()

# ────────────────── Accelerator setup ───────────────
acc = Accelerator()
rank0 = acc.is_main_process
acc.print = acc.print if rank0 else (lambda *a, **k: None)

# ────────────────── Paths & cache dirs ──────────────
tok_cache = f"/opt/dlami/nvme/tok_{args.model_name.replace('/', '_')}_max{args.max_len}"
os.makedirs(tok_cache, exist_ok=True)   # harmless if already there

# ─────────────────── Rank-0 work ────────────────────
if rank0 and not os.listdir(tok_cache):
    acc.print(f"🔑  Loading tokenizer:  {args.model_name}")
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    acc.print("⬇️   Loading raw Arrow shards from S3 …")
    raw_ds = load_dataset(
        "arrow",
        data_files={"test": f"{args.s3_root.rstrip('/')}/test/*.arrow"},
        # non-streaming → we need real data to map over
    )["test"]

    def tok_fn(batch):
        return tok(batch["sequence"],
                   truncation=True,
                   max_length=args.max_len)

    acc.print("🛠️   Tokenising … (this can take a while)")
    (raw_ds
        .map(tok_fn,
             remove_columns=["sequence"],
             batched=True,            # speeds things up
             batch_size=1000,
             num_proc=32,             # g5.12xlarge → 96 vCPUs
             load_from_cache_file=False)
        .save_to_disk(tok_cache))

    acc.print(f"✅  Saved tokenised set → {tok_cache}")

# ───── All other ranks wait for the cache, then exit ─────
acc.wait_for_everyone()
if not rank0:
    acc.print(f"Using cached dataset at {tok_cache}")

# If you want to sanity-check it:
# ds = load_from_disk(tok_cache)
# print(ds[0])
