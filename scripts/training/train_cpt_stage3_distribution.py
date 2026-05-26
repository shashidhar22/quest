#!/usr/bin/env python3
"""TCRBench v3 — CPT Stage 3 orchestrator: MHC-representation sweep.

Trains ESM-C 300M LoRA MLM on two 10M-row CPT datasets sequentially, holding
TCR rep at CDR3 (Stage 2's winner) and varying the MHC representation:

  - run04_cdr3_pocketcontact_proportional_10M  (MHC = pocket + TCR-contact)
  - run05_cdr3_fullmhc_proportional_10M        (MHC = full chain, ~270-320 AAs)

Stage 2 Run 01 (CDR3 + pocket) is reused as the pocket baseline — its bucketed
canonical-v2 eval JSON is produced separately by ``cpt_stage3_eval_baseline.py``.

Per-run batch settings (g6.12xlarge, 4× L4 24GB):
  Run 04: max_length=64,  batch=192, grad_accum=2, buckets=[16,24,32,48,64]
  Run 05: max_length=128, batch=128, grad_accum=2, buckets=[16,32,64,96,128]

Both runs:
  - early stop on ``canonical_v2/pocket_ppl_mhc_bearing`` (3-eval patience,
    1% rel improvement). Pocket tokens only exist on MHC-bearing rows, so
    ``result["pocket"]["overall"]["_"]["perplexity"]`` is intrinsically the
    MHC-bearing-only pocket perplexity.
  - LoRA r=32, α=64, dropout=0.05 (auto-detected ESM-C targets)
  - bf16 + SDPA + no torch.compile (matches Stage 2 verified config)
  - WandB project: ``tcrbench-v3-cpt-stage3``

Run order: 04 → 05 (Run 04 is the smaller delta; if it surfaces an integration
issue, we catch it before paying for Run 05). Each launches as a separate
torchrun invocation so DDP state is fully torn down between runs.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


CONFIGS = [
    {
        "name": "stage3-esmc_300m-run04-pocketcontact",
        "dataset": "/home/ubuntu/quest/data/cpt_datasets/run04_cdr3_pocketcontact_proportional_10M",
        "output_dir": "/home/ubuntu/quest/checkpoints/cpt_stage3_run04_pocketcontact",
        "tcr_variant": "cdr3",
        "mhc_variant": "pocket_contact",
        "max_length": 64,
        "batch_size": 192,
        "grad_accum": 2,
        "buckets": [16, 24, 32, 48, 64],
        "gradient_checkpointing": False,
    },
    {
        "name": "stage3-esmc_300m-run05-fullmhc",
        "dataset": "/home/ubuntu/quest/data/cpt_datasets/run05_cdr3_fullmhc_proportional_10M",
        "output_dir": "/home/ubuntu/quest/checkpoints/cpt_stage3_run05_fullmhc",
        "tcr_variant": "cdr3",
        "mhc_variant": "full",
        "max_length": 128,
        "batch_size": 128,
        "grad_accum": 2,
        "buckets": [16, 32, 64, 96, 128],
        "gradient_checkpointing": False,
    },
]

# Common arguments for ``train_cpt_stage2_single.py`` (stage-agnostic single-run
# trainer; only the orchestrator changes between stages).
COMMON_ARGS = [
    "--base_model", "esmc_300m",
    "--canonical_eval_path", "/home/ubuntu/quest/data/cpt_canonical_eval_v2",
    "--eos_strategy", "raw",
    "--num_epochs", "1",
    "--lr", "2e-4",
    "--lora_r", "32",
    "--lora_alpha", "64",
    "--lora_dropout", "0.05",
    "--eval_steps", "500",
    "--save_steps", "2000",
    "--logging_steps", "50",
    "--no_compile",
    "--attn_implementation", "sdpa",
    "--wandb_project", "tcrbench-v3-cpt-stage3",
    "--report_to", "wandb",
    "--early_stop_n_patience", "3",
    "--early_stop_min_rel", "0.01",
    # Stage 3 swap: monitor pocket_ppl (MHC-bearing-only by construction —
    # non-MHC rows contribute zero pocket tokens) instead of cdr3_ppl.
    "--early_stop_metric_path", "pocket", "overall", "_", "perplexity",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        choices=[c["name"] for c in CONFIGS],
        default=None,
        help="Run only this config (default: both, sequentially).",
    )
    ap.add_argument("--nproc_per_node", type=int, default=4)
    ap.add_argument("--master_port", type=int, default=29710)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    configs = [c for c in CONFIGS if args.only is None or c["name"] == args.only]
    if not configs:
        print(f"No configs match --only={args.only}")
        sys.exit(2)

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    train_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "train_cpt_stage2_single.py"
    )

    overall_t0 = time.time()
    for i, cfg in enumerate(configs, start=1):
        os.makedirs(cfg["output_dir"], exist_ok=True)
        log_path = os.path.join(cfg["output_dir"], "training.log")
        print(f"\n{'='*72}\n[stage3] {i}/{len(configs)}: {cfg['name']}\n{'='*72}", flush=True)
        print(f"  dataset:        {cfg['dataset']}")
        print(f"  output_dir:     {cfg['output_dir']}")
        print(f"  tcr_variant:    {cfg['tcr_variant']}")
        print(f"  mhc_variant:    {cfg['mhc_variant']}")
        print(f"  max_length:     {cfg['max_length']}")
        print(f"  batch x accum:  {cfg['batch_size']} x {cfg['grad_accum']} = {cfg['batch_size']*cfg['grad_accum']}")
        print(f"  log:            {log_path}")
        print(f"  wandb:          tcrbench-v3-cpt-stage3 / {cfg['name']}")
        t0 = time.time()

        cmd = [
            "torchrun",
            f"--nproc_per_node={args.nproc_per_node}",
            f"--master_port={args.master_port + i}",
            train_script,
            "--dataset_path", cfg["dataset"],
            "--output_dir", cfg["output_dir"],
            "--tcr_variant", cfg["tcr_variant"],
            "--mhc_variant", cfg["mhc_variant"],
            "--max_length", str(cfg["max_length"]),
            "--batch_size", str(cfg["batch_size"]),
            "--gradient_accumulation_steps", str(cfg["grad_accum"]),
            "--bucket_boundaries", *(str(b) for b in cfg["buckets"]),
            "--wandb_run_name", cfg["name"],
            *COMMON_ARGS,
        ]
        if cfg["gradient_checkpointing"]:
            cmd.append("--gradient_checkpointing")

        with open(log_path, "w") as logf:
            proc = subprocess.run(
                cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True
            )
        elapsed = time.time() - t0
        status = "PASS" if proc.returncode == 0 else f"FAIL (exit={proc.returncode})"
        print(f"  [done] {status}  wall={elapsed/60:.1f} min", flush=True)
        if proc.returncode != 0:
            print(f"  Aborting stage 3 — see {log_path} for details.", flush=True)
            sys.exit(proc.returncode)

    # Stage-level done marker on the last run's output dir.
    stage_marker = os.path.join(configs[-1]["output_dir"], "_stage_done")
    with open(stage_marker, "w") as f:
        f.write("OK\n")
    total = time.time() - overall_t0
    print(f"\n[stage3] All runs completed. Total wall: {total/60:.1f} min", flush=True)
    print(f"[stage3] wrote stage marker {stage_marker}", flush=True)


if __name__ == "__main__":
    main()
