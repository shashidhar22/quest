#!/usr/bin/env python
"""
tokenise_saved_ds.py
────────────────────
Load an existing HF dataset from disk, tokenise every row with a chosen
tokeniser, and write a new DatasetDict.

• Handles single-split folders and full DatasetDict layouts.
• Uses writer_batch_size so memory stays small during the map.
• Multi-process friendly (--num_proc).
"""
import argparse, pathlib
from datasets import load_from_disk, DatasetDict, Features, Value, Sequence
from transformers import AutoTokenizer
import tqdm

# ── CLI ─────────────────────────────────────────────────────────────────
argp = argparse.ArgumentParser()
argp.add_argument("--model_name", required=True)
argp.add_argument("--in_dir",     required=True, help="Path written by save_to_disk")
argp.add_argument("--out_dir",    required=True, help="Where to save tokenised data")
argp.add_argument("--max_len",    type=int, default=1024)
argp.add_argument("--batch_size", type=int, default=2_000)
argp.add_argument("--num_proc",   type=int, default=8)
argp.add_argument("--writer_batch_size", type=int, default=1_000)
args = argp.parse_args()

in_dir  = pathlib.Path(args.in_dir).expanduser()
out_dir = pathlib.Path(args.out_dir).expanduser()
out_dir.mkdir(parents=True, exist_ok=True)

# ── Tokeniser ───────────────────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

def tok_fn(batch):
    return tok(batch["combo_id"], truncation=True, max_length=args.max_len)

# ── Load existing dataset ------------------------------------------------
raw = load_from_disk(in_dir)
if isinstance(raw, DatasetDict):
    splits = raw
else:                                # single-split folder
    splits = DatasetDict({"train": raw})

print("Loaded splits :", list(splits.keys()))

# ── Tokenise every split -------------------------------------------------
tokenised_splits = {}
for name, ds in splits.items():
    print(f"🛠️  Tokenising split '{name}' ({len(ds):,} rows)")
    processed = ds.map(
        tok_fn,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        writer_batch_size=args.writer_batch_size,
        remove_columns=[c for c in ds.column_names if c != "combo_id"],
        desc=f"{name} - tokenising",
    )
    tokenised_splits[name] = processed

tok_ds = DatasetDict(tokenised_splits)

# ── Save -----------------------------------------------------------------
tok_ds.save_to_disk(out_dir)
print(f"✅  Tokenised dataset saved to {out_dir}")
