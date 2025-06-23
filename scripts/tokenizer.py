#!/usr/bin/env python
"""
Fast multi-process tokenisation with 🤗 Accelerate.
g4dn.12xlarge → 4 GPU-backed processes × 12 CPU workers each.
"""
import argparse, os, shutil
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer
from accelerate import Accelerator

# ───────────── CLI ─────────────
p = argparse.ArgumentParser()
p.add_argument("--model_name", required=True)
p.add_argument("--s3_root",    required=True)
p.add_argument("--max_len",    type=int, required=True)
p.add_argument("--cache_dir",  default="/opt/dlami/nvme")
p.add_argument("--batch_size", type=int, default=2000)   # per proc
p.add_argument("--num_proc",   type=int, default=12)     # CPU workers / proc
args = p.parse_args()

# ────────── Accelerator ─────────
acc = Accelerator()
rank, world = acc.process_index, acc.num_processes
rank0       = rank == 0
barrier     = acc.wait_for_everyone

# ───────── Paths ────────────────
base_cache = f"{args.cache_dir}/tok_{args.model_name.replace('/','_')}_max{args.max_len}"
shard_dir  = f"{base_cache}/shard_{rank}"
final_dir  = f"{base_cache}/full"
os.makedirs(shard_dir, exist_ok=True)

# Stop early if the full cache already exists
if os.path.isdir(final_dir):
    if rank0:
        acc.print(f"✅  Using existing cache → {final_dir}")
    barrier(); exit()

# ───────── Tokeniser ────────────
if rank0:  acc.print(f"🔑  Loading tokenizer '{args.model_name}'")
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

# ───────── Load & SHARD dataset ─
if rank0:  acc.print("⬇️   Loading raw Arrow shards from S3 …")
raw = load_dataset("arrow",
                   data_files={"test": f"{args.s3_root.rstrip('/')}/test/*.arrow"},
                   split="test")                 # one Arrow dataset
shard = raw.shard(num_shards=world, index=rank, contiguous=True)

# ───────── Tokenise slice ───────
def tok_fn(batch):
    return tokenizer(batch["combo_id"],
                     truncation=True,
                     max_length=args.max_len)

acc.print(f"🛠️   Rank {rank}/{world-1}: tokenising {len(shard):,} rows …")
tok_ds: Dataset = (
    shard.map(tok_fn,
              batched=True,
              batch_size=args.batch_size,
              num_proc=args.num_proc,
              remove_columns=["combo_id"],
              load_from_cache_file=False)
)
tok_ds.save_to_disk(shard_dir)
acc.print(f"💾  Rank {rank}: wrote shard to {shard_dir}")

# ───────── Merge on rank-0 ───────
barrier()
if rank0:
    acc.print("🔗  Concatenating shards …")
    shards = [load_dataset("arrow", data_files={})
              for _ in range(world)]  # placeholder list
    # quicker: load_from_disk once per shard
    shards = [Dataset.load_from_disk(f"{base_cache}/shard_{i}") for i in range(world)]
    full   = concatenate_datasets(shards)
    full.save_to_disk(final_dir)
    acc.print(f"✅  All done → {final_dir}")

    # optional clean-up to reclaim NVMe space
    for i in range(world):
        shutil.rmtree(f"{base_cache}/shard_{i}", ignore_errors=True)

barrier()
if not rank0:
    acc.print(f"🟢  Rank {rank}: finished, cache ready at {final_dir}")
