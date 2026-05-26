#!/usr/bin/env python3
"""TCRBench v3 — Stage 3 baseline re-eval: produce ``canonical_v2_results.json``
for the Stage 2 Run 01 (CDR3 + pocket) checkpoint.

Stage 3's three-way comparison (Run 01 = pocket baseline, Run 04 = pocket+contact,
Run 05 = full MHC) needs a directly comparable bucketed eval JSON for all three
runs. Stage 2 Run 01's checkpoint dir lacks ``canonical_v2_results.json``
(the trainer logs region_restricted_eval results to wandb but doesn't persist
them to disk at end of training). This script loads the saved LoRA adapter,
runs ``region_restricted_eval`` exactly as the in-training callback does, and
writes the JSON.

Output:
  /home/ubuntu/quest/checkpoints/cpt_stage2_run01_cdr3/canonical_v2_results.json
  /home/ubuntu/quest/checkpoints/cpt_stage2_run01_cdr3/_stage3_baseline_done

Wall time: ~10 minutes on one L4. Single-GPU; no DDP needed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from datasets import load_from_disk
from peft import PeftModel

from cpt_training_lib import (
    get_tokenizer,
    load_base_model,
    region_restricted_eval,
)


BASE_MODEL = "esmc_300m"
RUN01_DIR = "/home/ubuntu/quest/checkpoints/cpt_stage2_run01_cdr3"
ADAPTER_DIR = os.path.join(RUN01_DIR, "final")
CANONICAL_EVAL = "/home/ubuntu/quest/data/cpt_canonical_eval_v2"
TCR_VARIANT = "cdr3"
MHC_VARIANT = "pocket"
EOS_STRATEGY = "raw"
MAX_LENGTH = 64           # matches Stage 2 Run 01 training
BATCH_SIZE = 16           # matches region_eval_batch_size used in training
MLM_PROBABILITY = 0.15


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", default=ADAPTER_DIR)
    ap.add_argument("--output_json", default=os.path.join(RUN01_DIR, "canonical_v2_results.json"))
    ap.add_argument("--canonical_eval_path", default=CANONICAL_EVAL)
    ap.add_argument("--base_model", default=BASE_MODEL)
    ap.add_argument("--tcr_variant", default=TCR_VARIANT)
    ap.add_argument("--mhc_variant", default=MHC_VARIANT)
    ap.add_argument("--eos_strategy", default=EOS_STRATEGY)
    ap.add_argument("--max_length", type=int, default=MAX_LENGTH)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    return ap.parse_args()


def _to_jsonable(obj):
    """Convert region_restricted_eval result to a JSON-safe nested structure.

    Bucket keys may be ints/strings; the result dict is otherwise plain. The
    inner ``rows`` field is a set of row indices (debug-only) — drop it for
    compactness.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "rows":
                continue
            out[str(k)] = _to_jsonable(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, set):
        return sorted(_to_jsonable(x) for x in obj)
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def main() -> int:
    args = parse_args()
    t0 = time.time()

    if not os.path.isdir(args.adapter_dir):
        print(f"[baseline] ERROR: adapter dir not found: {args.adapter_dir}")
        return 2
    if not os.path.isdir(args.canonical_eval_path):
        print(f"[baseline] ERROR: canonical eval dir not found: {args.canonical_eval_path}")
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[baseline] device={device}  base_model={args.base_model}")
    print(f"[baseline] adapter={args.adapter_dir}")

    print(f"[baseline] loading tokenizer + base model ...")
    tokenizer = get_tokenizer(args.base_model)
    base = load_base_model(args.base_model, attn_implementation="sdpa")
    base = base.to(device)

    print(f"[baseline] attaching LoRA adapter from {args.adapter_dir} ...")
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model = model.to(device)
    model.eval()

    print(f"[baseline] loading canonical eval from {args.canonical_eval_path}")
    ds = load_from_disk(args.canonical_eval_path)
    if hasattr(ds, "keys"):
        keys = list(ds.keys())
        split = "test" if "test" in keys else keys[0]
        print(f"[baseline] dataset is DatasetDict; using split={split}")
        ds = ds[split]
    print(f"[baseline] eval rows: {len(ds):,}")

    print(
        f"[baseline] running region_restricted_eval "
        f"(tcr={args.tcr_variant}, mhc={args.mhc_variant}, "
        f"eos={args.eos_strategy}, max_len={args.max_length}, bs={args.batch_size}) ..."
    )
    result = region_restricted_eval(
        model,
        ds,
        args.tcr_variant,
        args.mhc_variant,
        tokenizer,
        device=device,
        eos_strategy=args.eos_strategy,
        mlm_probability=MLM_PROBABILITY,
        max_length=args.max_length,
        batch_size=args.batch_size,
        verbose=True,
    )

    # Top-level summary print.
    overall = result["overall"]["overall"]["_"]
    cdr3 = result.get("cdr3", {}).get("overall", {}).get("_") or {"perplexity": float("nan")}
    pocket = result.get("pocket", {}).get("overall", {}).get("_") or {"perplexity": float("nan")}
    fw = result.get("framework", {}).get("overall", {}).get("_") or {"perplexity": float("nan")}
    print(
        f"\n[baseline] overall_ppl={overall['perplexity']:.3f}  "
        f"cdr3_ppl={cdr3['perplexity']:.3f}  "
        f"pocket_ppl={pocket['perplexity']:.3f}  "
        f"framework_ppl={fw['perplexity']:.3f}"
    )

    payload = {
        "tcr_variant": args.tcr_variant,
        "mhc_variant": args.mhc_variant,
        "eos_strategy": args.eos_strategy,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "mlm_probability": MLM_PROBABILITY,
        "adapter_dir": args.adapter_dir,
        "base_model": args.base_model,
        "canonical_eval_path": args.canonical_eval_path,
        "n_eval_rows": len(ds),
        "wall_seconds": round(time.time() - t0, 1),
        "regions": _to_jsonable(result),
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[baseline] wrote {args.output_json}")

    marker = os.path.join(os.path.dirname(args.output_json), "_stage3_baseline_done")
    with open(marker, "w") as f:
        f.write("OK\n")
    print(f"[baseline] wrote marker {marker}")
    print(f"[baseline] done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
