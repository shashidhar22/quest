#!/usr/bin/env python3
"""TCRBench v3 — CPT Stage 2 per-run trainer (ESM-C 300M with region-restricted
canonical eval).

Wraps cpt_training_lib.CPTTrainer with Stage-2-specific config:
  - base_model is an ESM-C id (e.g. esmc_300m)
  - canonical_eval_path points at canonical_eval_v2 (full molecular columns)
  - region_eval=True → CPTTrainer uses region_restricted_eval and
    CdrPplPlateauCallback (cdr3_ppl early stop)
  - eos_strategy default 'raw' (ESM-C tokenizer handles <eos> literal natively;
    confirmed by Stage 2 verification)

Launch (single run):
  torchrun --nproc_per_node=4 scripts/training/train_cpt_stage2_single.py \\
    --base_model esmc_300m --tcr_variant cdr3 --mhc_variant pocket \\
    --dataset_path /home/ubuntu/quest/data/cpt_datasets/run01_cdr3_pocket_proportional_10M \\
    --canonical_eval_path /home/ubuntu/quest/data/cpt_canonical_eval_v2 \\
    --output_dir /home/ubuntu/quest/checkpoints/cpt_stage2_run01_cdr3 \\
    --max_length 64 --batch_size 192 --gradient_accumulation_steps 2 \\
    --wandb_run_name stage2-esmc_300m-run01-cdr3
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cpt_training_lib import CPTConfig, CPTTrainer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # Model
    ap.add_argument("--base_model", default="esmc_300m")
    # Data
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument(
        "--canonical_eval_path",
        default="/home/ubuntu/quest/data/cpt_canonical_eval_v2",
    )
    ap.add_argument("--tcr_variant", default="cdr3", choices=["cdr3", "cdr123", "full"])
    ap.add_argument("--mhc_variant", default="pocket", choices=["pocket", "pocket_contact", "full"])
    ap.add_argument(
        "--eos_strategy",
        default="raw",
        choices=["raw", "fixb", "pipe_sub"],
        help="ESM-C tokenizer handles <eos> natively; default 'raw'.",
    )
    ap.add_argument("--max_length", type=int, default=128)
    # Output
    ap.add_argument("--output_dir", required=True)
    # LoRA
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    # Training
    ap.add_argument("--num_epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--mlm_probability", type=float, default=0.15)
    ap.add_argument("--gradient_clip", type=float, default=1.0)
    # Eval / save cadence
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--save_steps", type=int, default=2000)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--region_eval_batch_size", type=int, default=16)
    # Length bucketing
    ap.add_argument(
        "--bucket_boundaries", type=int, nargs="+",
        default=[32, 48, 64, 96, 128],
    )
    ap.add_argument("--no_length_bucketing", action="store_true")
    # Runtime
    ap.add_argument(
        "--attn_implementation",
        default="sdpa",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="ESM-C internally manages attention; this is a no-op for it but kept for ESM-2 compat.",
    )
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--token_budget", type=int, default=0,
                    help="Stage 4.3: tokens per micro-batch for step-synchronised batching; "
                         "0 = off (fixed batch_size, the Stage 4.1 path)")
    ap.add_argument("--corpus_preflight", choices=("off", "required"), default="off",
                    help="'required': refuse unless the parent of --dataset is a published corpus "
                         "passing quest.data.corpus_preflight (needed by --loss_weighting token_exact)")
    ap.add_argument("--corpus_benchmark_sha256", default=None,
                    help="JSON {benchmark split file: sha256} the corpus's leakage audit must have read")
    ap.add_argument("--loss_weighting", choices=("row", "token", "token_exact"), default="row",
                    help="Stage 4.3: how a micro-batch's mean loss is weighted inside its step "
                         "when --token_budget > 0. 'row' = row share (the shipped v2 behaviour; "
                         "gives MHC-bearing tokens ~0.43x the gradient weight the Stage 4.1 path "
                         "gave them, which is why v2 split better-overall/worse-pocket). "
                         "'token' = token share, keeping per-token weight uniform and the "
                         "objective where Stage 4.1 had it. No effect when token_budget = 0.")
    ap.add_argument("--deterministic", action="store_true",
                    help="Constrain cuDNN/cuBLAS so two runs at the same seed are comparable. "
                         "Costs throughput; used by 4.3-noise to tell float non-determinism "
                         "from an unseeded component.")
    ap.add_argument("--resume_from", default=None,
                    help="Stage 4.3: a checkpoint_step_N dir to resume from, or 'auto' = the newest "
                         "resumable checkpoint_step_* under --output_dir (fresh start if none)")
    ap.add_argument("--resume_assume_objective", default=None,
                    help="Only for a checkpoint that records no objective semantics, e.g. "
                         "'token:input-length-share/v1'. Resume refuses such a checkpoint by "
                         "default rather than adopting the current objective and mixing two in "
                         "one run. The value is verified against what the checkpoint does "
                         "record and cannot license an actual change of objective.")
    # Early stop
    ap.add_argument("--early_stop_n_patience", type=int, default=3,
                    help="0 or less turns early stopping OFF. Required for equivalence and "
                         "single-configuration runs: an early-stopped run cannot be compared "
                         "with one that ran its full budget (quest-cpt/findings.md 9.5).")
    ap.add_argument("--early_stop_min_rel", type=float, default=0.01)
    ap.add_argument(
        "--early_stop_metric_path",
        nargs="+",
        default=["cdr3", "overall", "_", "perplexity"],
        help=(
            "Nested-dict path inside region_restricted_eval result for the "
            "early-stop callback. Stage 2 uses the default (cdr3 overall "
            "perplexity). Stage 3 swaps to pocket: "
            "'--early_stop_metric_path pocket overall _ perplexity'."
        ),
    )
    # Wandb
    ap.add_argument("--wandb_project", default="tcrbench-v3-cpt-stage2")
    ap.add_argument("--wandb_run_name", default=None)
    ap.add_argument("--report_to", default="wandb", choices=["wandb", "none"])
    # Reproducibility
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = CPTConfig(
        base_model=args.base_model,
        dataset_path=args.dataset_path,
        canonical_eval_path=args.canonical_eval_path,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_every_n_steps=args.eval_steps,
        save_every_n_steps=args.save_steps,
        logging_steps=args.logging_steps,
        max_length=args.max_length,
        mlm_probability=args.mlm_probability,
        eos_strategy=args.eos_strategy,
        attn_implementation=args.attn_implementation,
        use_compile=not args.no_compile,
        use_length_bucketing=not args.no_length_bucketing,
        bucket_boundaries=list(args.bucket_boundaries),
        gradient_checkpointing=args.gradient_checkpointing,
        token_budget=args.token_budget,
        loss_weighting=args.loss_weighting,
        corpus_preflight=args.corpus_preflight,
        corpus_benchmark_sha256=args.corpus_benchmark_sha256,
        resume_from=args.resume_from,
        resume_assume_objective=args.resume_assume_objective,
        region_eval=True,
        tcr_variant=args.tcr_variant,
        mhc_variant=args.mhc_variant,
        region_eval_batch_size=args.region_eval_batch_size,
        early_stop_n_patience=args.early_stop_n_patience,
        deterministic=args.deterministic,
        early_stop_min_rel=args.early_stop_min_rel,
        early_stop_metric_path=tuple(args.early_stop_metric_path),
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        report_to=args.report_to,
        seed=args.seed,
    )
    trainer = CPTTrainer(cfg)
    trainer.train()
    # Touch implementer-done marker for cross-agent verification.
    if int(os.environ.get("RANK", "0")) == 0:
        with open(os.path.join(args.output_dir, "_implementer_done"), "w") as f:
            f.write("OK\n")


if __name__ == "__main__":
    main()
