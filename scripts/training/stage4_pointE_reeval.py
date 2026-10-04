#!/usr/bin/env python3
"""Stage 4 Point E: re-evaluate the Stage 2 Run 03 (full TCR + pocket MHC)
checkpoint against canonical eval v2 with fresh Stage 4 timestamps.

The Stage 2 result file already exists at ``cpt_stage2_run03_full/final/
canonical_v2_results.json``, but Stage 4 re-runs region_restricted_eval for
cross-check and to attach Stage 4 ``_diagnostics`` metadata.

Pattern mirrors ``cpt_stage1_retroactive_eval.py`` but targets a single
ESM-C 300M checkpoint with eos_strategy='raw'.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpt_training_lib import get_tokenizer, region_restricted_eval, load_base_model
from datasets import load_from_disk

DEFAULT_BASE = "esmc_300m"
DEFAULT_CHECKPOINT = "/home/ubuntu/quest/checkpoints/cpt_stage2_run03_full/final"
DEFAULT_OUT = "/home/ubuntu/quest/checkpoints/cpt_stage4_pointE_run03_reeval"
DEFAULT_EVAL = "/home/ubuntu/quest/data/cpt_canonical_eval_v2"


def _log(msg: str) -> None:
    print(f"[stage4_pointE] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default=DEFAULT_BASE)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--canonical_eval", default=DEFAULT_EVAL)
    ap.add_argument("--tcr_variant", default="full", choices=["cdr3", "cdr123", "full"])
    ap.add_argument("--mhc_variant", default="pocket", choices=["pocket", "pocket_contact", "full"])
    ap.add_argument("--output_dir", default=DEFAULT_OUT)
    ap.add_argument("--eos_strategy", default="raw", choices=["raw", "fixb", "pipe_sub"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=1024)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "canonical_v2_results.json"
    done_file = out_dir / "_done"

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"Checkpoint missing: {args.checkpoint}")
    if not os.path.exists(args.canonical_eval):
        raise SystemExit(f"Canonical eval missing: {args.canonical_eval}")

    _log(f"Loading canonical eval v2 from {args.canonical_eval}")
    v2_ds = load_from_disk(args.canonical_eval)
    _log(f"  rows={len(v2_ds):,}  cols={len(v2_ds.column_names)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _log(f"Device: {device}")

    _log(f"Loading base {args.base_model} + LoRA adapter {args.checkpoint}")
    from peft import PeftModel
    # ESM-C is NOT a HuggingFace repo id — it must load via the esm package.
    # load_base_model() routes 'esmc_*' ids to esm.models.esmc.ESMC.from_pretrained
    # and attaches the PEFT config shim; PeftModel.from_pretrained then re-applies
    # the saved LoRA adapter. region_restricted_eval -> _model_forward_logits
    # handles the PEFT-wrapped ESM-C forward natively.
    base = load_base_model(args.base_model)
    model = PeftModel.from_pretrained(base, args.checkpoint)
    model = model.to(device).eval()

    tok = get_tokenizer(args.base_model)

    _log(f"Running region_restricted_eval (tcr={args.tcr_variant}, mhc={args.mhc_variant}, "
         f"eos={args.eos_strategy}, batch={args.batch_size}, max_len={args.max_length})")
    t0 = time.time()
    result = region_restricted_eval(
        model,
        v2_ds,
        args.tcr_variant,
        args.mhc_variant,
        tok,
        device=device,
        eos_strategy=args.eos_strategy,
        batch_size=args.batch_size,
        max_length=args.max_length,
        verbose=True,
    )
    elapsed = time.time() - t0
    _log(f"Eval wall time: {elapsed:.1f}s")

    overall = result["overall"]["overall"]["_"]["perplexity"]
    cdr3 = result.get("cdr3", {}).get("overall", {}).get("_", {}).get("perplexity")
    pocket = result.get("pocket", {}).get("overall", {}).get("_", {}).get("perplexity")
    _log(f"overall_ppl={overall:.4f}  cdr3_ppl={cdr3}  pocket_ppl={pocket}")

    payload = {
        "stage": "stage4_pointE_reeval",
        "config": {
            "base_model": args.base_model,
            "checkpoint": args.checkpoint,
            "tcr_variant": args.tcr_variant,
            "mhc_variant": args.mhc_variant,
            "eos_strategy": args.eos_strategy,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
        },
        "device": str(device),
        "eval_wall_seconds": elapsed,
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "result": result,
    }
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    _log(f"Saved: {out_file}")

    with open(done_file, "w") as f:
        f.write("STATUS=PASS\n")
        f.write(f"overall_ppl={overall:.4f}\n")
        if cdr3 is not None:
            f.write(f"cdr3_ppl={cdr3:.4f}\n")
        if pocket is not None:
            f.write(f"pocket_ppl={pocket:.4f}\n")
        f.write(f"eval_wall_seconds={elapsed:.1f}\n")
    _log(f"Wrote _done marker: {done_file}")


if __name__ == "__main__":
    main()
