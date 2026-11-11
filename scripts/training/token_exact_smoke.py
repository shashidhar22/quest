#!/usr/bin/env python3
"""A bounded real-device smoke of `token_exact` on a published corpus (new-cpt-plan gate 5).

Runs the REAL CPTTrainer for --steps optimizer steps and stops: no canonical eval, no final
save. Records per step the global target count, the loss (summed loss over that count), the
gradient norm before clipping, rows, tokens and seconds, then re-runs the corpus preflight to
show training wrote nothing into the corpus. Exit 0 only if every number is finite, every step
had targets, and the corpus still passes.

    python token_exact_smoke.py --dataset_path <version>/dataset --output_dir <dir> \
        --canonical_eval_path <dir> --steps 50 --out smoke.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import cpt_training_lib as L  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--canonical_eval_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--base_model", default="esmc_300m")
    ap.add_argument("--corpus_benchmark_sha256", default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=64)
    ap.add_argument("--token_budget", type=int, default=16384)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--eos_strategy", default="raw")
    ap.add_argument("--bucket_boundaries", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    ap.add_argument("--region_eval", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        ap.error(f"{a.out} exists")

    cfg = L.CPTConfig(
        base_model=a.base_model, dataset_path=a.dataset_path, canonical_eval_path=a.canonical_eval_path,
        output_dir=a.output_dir, lora_r=32, lora_alpha=64, lora_dropout=0.05, learning_rate=2e-4,
        num_epochs=1, batch_size=a.batch_size, gradient_accumulation_steps=a.gradient_accumulation_steps,
        eval_every_n_steps=10**9, save_every_n_steps=10**9, logging_steps=10**9, max_length=a.max_length,
        eos_strategy=a.eos_strategy, attn_implementation="sdpa", use_compile=False,
        token_budget=a.token_budget, bucket_boundaries=list(a.bucket_boundaries),
        dataloader_num_workers=a.num_workers, region_eval=bool(a.region_eval), tcr_variant="cdr3",
        mhc_variant="full", report_to="none", seed=42, early_stop_n_patience=0,
        loss_weighting="token_exact", corpus_preflight="required",
        corpus_benchmark_sha256=a.corpus_benchmark_sha256,
    )
    t_setup = time.perf_counter()
    tr = L.CPTTrainer(cfg)
    setup_s = time.perf_counter() - t_setup
    plan = tr.train_sampler.step_plan()
    steps = min(a.steps, len(plan))
    rec = [{"micro_batches": n, "rows": 0, "tokens": 0, "longest_row": 0, "loss_sum": 0.0} for n in plan[:steps]]
    cur = {"i": -1}
    orig_steps, orig_sum, orig_clip = L.exact_step_batches, L.summed_masked_loss, torch.nn.utils.clip_grad_norm_

    def bounded(batches, plan_, world_size):
        left = 0
        head = itertools.islice(batches, sum(plan_[:steps]))        # the loader holds the whole epoch
        for batch, scale in orig_steps(head, plan_[:steps], world_size):
            if left == 0:
                cur["i"] += 1
                left = plan_[cur["i"]]
                rec[cur["i"]].update(targets=round(world_size / scale), t0=time.perf_counter())
            left -= 1
            r = rec[cur["i"]]
            r["rows"] += int(batch["input_ids"].size(0))
            r["tokens"] += int(batch["attention_mask"].sum())
            r["longest_row"] = max(r["longest_row"], int(batch["input_ids"].size(1)))
            yield batch, scale

    def summed(logits, labels, *args, **kw):
        out = orig_sum(logits, labels, *args, **kw)
        rec[cur["i"]]["loss_sum"] += float(out.detach())
        rec[cur["i"]]["logits_dtype"] = str(logits.dtype)
        return out

    def clip(params, max_norm, *args, **kw):
        norm = orig_clip(params, max_norm, *args, **kw)
        r = rec[cur["i"]]
        r["grad_norm"] = float(norm)
        r["seconds"] = time.perf_counter() - r.pop("t0")
        r["loss"] = r["loss_sum"] / r["targets"]
        return norm

    L.exact_step_batches, L.summed_masked_loss, torch.nn.utils.clip_grad_norm_ = bounded, summed, clip
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        tr._train_one_epoch(0)
    finally:
        L.exact_step_batches, L.summed_masked_loss, torch.nn.utils.clip_grad_norm_ = orig_steps, orig_sum, orig_clip
    train_s = time.perf_counter() - t0

    after = L.check_corpus_and_objective(cfg)          # raises if training changed the corpus
    done = [r for r in rec if "loss" in r]
    warm = done[5:] or done                            # the first steps carry start-up cost
    tot = lambda k, rows=warm: sum(r[k] for r in rows)
    problems = []
    if len(done) != steps or tr.global_step != steps:
        problems.append(f"{len(done)} steps recorded, global_step {tr.global_step}, wanted {steps}")
    if not all(math.isfinite(r["loss"]) and math.isfinite(r["grad_norm"]) for r in done):
        problems.append("a non-finite loss or gradient norm")
    if any(r["targets"] <= 0 for r in done):
        problems.append("a step without targets")
    if after != tr._corpus_identity:
        problems.append("the corpus identity changed during training")
    report = {
        "status": "PASS" if not problems else "FAIL", "problems": problems,
        "corpus": after, "dataset_path": a.dataset_path, "device": str(tr.device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "world_size": tr.world_size, "train_rows": len(tr.train_ds), "steps_per_epoch": len(plan),
        "rows_per_step": cfg.batch_size * cfg.gradient_accumulation_steps, "token_budget": cfg.token_budget,
        "steps_run": len(done), "setup_seconds": round(setup_s, 1), "train_seconds": round(train_s, 1),
        "after_warmup": {
            "steps": len(warm), "seconds_per_step": tot("seconds") / max(1, len(warm)),
            "rows_per_second": tot("rows") / max(1e-9, tot("seconds")),
            "tokens_per_second": tot("tokens") / max(1e-9, tot("seconds")),
            "targets_per_second": tot("targets") / max(1e-9, tot("seconds")),
            "tokens_per_step": tot("tokens") / max(1, len(warm)),
            "targets_per_step": tot("targets") / max(1, len(warm)),
            "micro_batches_per_step": tot("micro_batches") / max(1, len(warm)),
        },
        "epoch_hours_at_this_rate": len(plan) * tot("seconds") / max(1, len(warm)) / 3600,
        "loss_first_last": [done[0]["loss"], done[-1]["loss"]] if done else None,
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
        "per_step": [{k: v for k, v in r.items() if k != "loss_sum"} for r in done],
    }
    with open(a.out, "x") as fh:
        json.dump(report, fh, indent=1)
        fh.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_step"}, indent=1))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
