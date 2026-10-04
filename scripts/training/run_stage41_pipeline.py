#!/usr/bin/env python3
"""CPT Stage 4.1 — the Stage 4 representation sweep re-run at a FIXED,
non-truncating ``max_length``.

Why this stage exists
---------------------
Stage 4 varied ``max_length`` per point (64 / 256 / 1024 / 1024), and the window
always tracked the serialization being tested. Because the trainer passes its
``max_length`` straight into ``region_restricted_eval`` (cpt_training_lib.py
~1636), each point was also *scored* on its own truncated view of the canonical
eval set. Representation effects and context-window effects were therefore
inseparable. Stage 4.1 pins one window across every arm so the serialization is
the only thing that varies.

Arms (cost-ascending), all at max_length=1024, all at effective batch 1024:
  A. cdr3   + pocket    on run01  — Stage 4 Point A, which ran at 64 (TRUNCATED:
                                    run01's longest row is 162 tokens)
  D. cdr3   + full MHC  on run05  — Stage 4 Point D, which already ran at 1024,
                                    so this arm is a REPRODUCIBILITY CONTROL and
                                    should land near pocket_ppl 2.001
  F. cdr123 + full MHC  on run11  — NEW combination. Stage 4's Point B was
                                    cdr123 x pocket_contact, not x full.

Why 1024
--------
Measured maxima over every row of every split (scripts/training/measure_maxlen.py,
2026-10-02): run01 = 162 tokens, run05 = 578 tokens. 578 independently matches the
canonical eval set's longest cdr3-x-full row. cdr123 adds only CDR1+CDR2 per chain
(~30 tokens) over cdr3, so run11 sits well under 1024. The preflight re-measures and
REFUSES to launch if any dataset exceeds the window, so "no truncation" is enforced
rather than assumed.

Everything else is byte-identical to Stage 4: see COMMON_TRAIN_ARGS.

Per-point ``_done`` markers make the driver resumable; a point already PASS is
skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

QUEST_ROOT = Path("/home/ubuntu/quest")
HERE = Path(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = HERE / "train_cpt_stage2_single.py"
PHASE_MARKER = QUEST_ROOT / "checkpoints/_stage41_done"
DRIVER_LOG = QUEST_ROOT / "checkpoints/stage41_combined_DRIVER.log"
CANONICAL_EVAL = QUEST_ROOT / "data/cpt_canonical_eval_v2"

WANDB_PROJECT = "tcrbench-v3-cpt-stage41"

# ---------------------------------------------------------------------------
# The single knob this stage exists to fix.
# ---------------------------------------------------------------------------
FIXED_MAX_LENGTH = 1024
# One bucket list for every arm, so batch composition cannot differ by arm
# either. Rows above the last boundary land in an overflow bucket (np.digitize
# in quest/training/samplers.py), which at 1024 cannot happen here.
FIXED_BUCKETS = [64, 128, 256, 512, 1024]
# Held constant across arms: batch_size * grad_accum * n_gpus.
# Stage 4 did NOT hold this constant (Point A was 1536, B/C/D were 1024), which
# is a second confound. 1024 is the value 3 of the 4 original points used.
TARGET_EFFECTIVE_BATCH = 1024

POINTS = [
    {
        "name": "pointA_run01",
        "label": "cdr3 x pocket",
        "output_dir": str(QUEST_ROOT / "checkpoints/cpt_stage41_pointA_run01"),
        "dataset": str(QUEST_ROOT / "data/cpt_datasets/run01_cdr3_pocket_proportional_10M"),
        "tcr_variant": "cdr3",
        "mhc_variant": "pocket",
        # Rows top out at 162 tokens, so a large micro-batch is safe and keeps
        # this arm cheap. Dynamic padding + length bucketing mean compute tracks
        # real lengths, not the window, so 1024 costs little here.
        "batch_size": 128,
        "grad_accum": 2,
        "gradient_checkpointing": False,
        "wandb_run_name": "stage41-pointA-cdr3-pocket",
        "stage4_was": "max_length=64 (truncating; dataset max is 162)",
        "expected_wall_hr": 2,
    },
    {
        "name": "pointD_run05",
        "label": "cdr3 x full MHC",
        "output_dir": str(QUEST_ROOT / "checkpoints/cpt_stage41_pointD_run05"),
        "dataset": str(QUEST_ROOT / "data/cpt_datasets/run05_cdr3_fullmhc_proportional_10M"),
        "tcr_variant": "cdr3",
        "mhc_variant": "full",
        "batch_size": 16,
        "grad_accum": 16,
        "gradient_checkpointing": True,
        "wandb_run_name": "stage41-pointD-cdr3-fullmhc",
        "stage4_was": "max_length=1024 (already non-truncating) — reproducibility control",
        "expected_wall_hr": 6,
    },
    {
        "name": "pointF_run11",
        "label": "cdr123 x full MHC",
        "output_dir": str(QUEST_ROOT / "checkpoints/cpt_stage41_pointF_run11"),
        "dataset": str(QUEST_ROOT / "data/cpt_datasets/run11_cdr123_fullmhc_proportional_10M"),
        "tcr_variant": "cdr123",
        "mhc_variant": "full",
        "batch_size": 16,
        "grad_accum": 16,
        "gradient_checkpointing": True,
        "wandb_run_name": "stage41-pointF-cdr123-fullmhc",
        "stage4_was": "NEW — Stage 4 Point B was cdr123 x pocket_contact",
        "expected_wall_hr": 7,
    },
]

# Byte-identical to Stage 4's COMMON_TRAIN_ARGS apart from wandb project.
COMMON_TRAIN_ARGS = [
    "--base_model", "esmc_300m",
    "--canonical_eval_path", str(CANONICAL_EVAL),
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
    "--wandb_project", WANDB_PROJECT,
    "--report_to", "wandb",
    "--early_stop_n_patience", "3",
    "--early_stop_min_rel", "0.01",
    "--early_stop_metric_path", "pocket", "overall", "_", "perplexity",
]


def log_event(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    DRIVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DRIVER_LOG, "a") as f:
        f.write(line + "\n")


def _check_done(pt: dict) -> bool:
    marker = Path(pt["output_dir"]) / "_done"
    if not marker.exists():
        return False
    return "STATUS=PASS" in marker.read_text()


def _write_phase_marker(status: str, results: dict) -> None:
    PHASE_MARKER.parent.mkdir(parents=True, exist_ok=True)
    with open(PHASE_MARKER, "w") as f:
        f.write(f"STATUS={status}\n")
        f.write(json.dumps(results, indent=2))


def preflight(nproc: int) -> None:
    """Refuse to launch unless the no-truncation promise actually holds.

    A failure here looks like: a dataset whose measured max token length exceeds
    FIXED_MAX_LENGTH, a missing dataset, a missing eval set, or an arm whose
    effective batch differs from the others.
    """
    log_event("=== Stage 4.1 preflight ===")

    if not CANONICAL_EVAL.exists():
        raise SystemExit(f"preflight FAIL: canonical eval missing: {CANONICAL_EVAL}")
    log_event(f"  OK canonical eval present: {CANONICAL_EVAL}")

    for pt in POINTS:
        if not Path(pt["dataset"]).exists():
            raise SystemExit(f"preflight FAIL: dataset missing for {pt['name']}: {pt['dataset']}")
        log_event(f"  OK dataset present: {pt['name']} -> {Path(pt['dataset']).name}")

    # Effective batch must be identical across arms — that is half the point.
    for pt in POINTS:
        eff = pt["batch_size"] * pt["grad_accum"] * nproc
        if eff != TARGET_EFFECTIVE_BATCH:
            raise SystemExit(
                f"preflight FAIL: {pt['name']} effective batch {eff} "
                f"!= target {TARGET_EFFECTIVE_BATCH} "
                f"({pt['batch_size']} x {pt['grad_accum']} x {nproc} GPUs)"
            )
        log_event(f"  OK effective batch {eff} for {pt['name']}")

    # The load-bearing check: no dataset may exceed the fixed window.
    sys.path.insert(0, str(QUEST_ROOT))
    from datasets import load_from_disk  # noqa: E402
    from scripts.training.cpt_training_lib import get_tokenizer  # noqa: E402

    tok = get_tokenizer("esmc_300m")
    SPECIAL_SLACK = 2  # framing tokens add tokens but no characters
    TOP_N = 20_000
    for pt in POINTS:
        ds = load_from_disk(pt["dataset"])
        worst = 0
        for sp in sorted(ds.keys()):
            d = ds[sp]
            texts = d.data.column("input_text").to_pylist()
            char_len = [len(t) for t in texts]
            order = sorted(range(len(char_len)), key=lambda i: -char_len[i])[:TOP_N]
            cand = [texts[i] for i in order]
            mx = 0
            for i in range(0, len(cand), 512):
                enc = tok(cand[i : i + 512], add_special_tokens=True,
                          truncation=False, padding=False)
                mx = max(mx, max(len(x) for x in enc["input_ids"]))
            # Anything that could beat mx needs char_len >= mx - SPECIAL_SLACK.
            need = mx - SPECIAL_SLACK
            examined_floor = char_len[order[-1]] if order else 0
            n_cand = sum(1 for c in char_len if c >= need)
            if not (need > examined_floor or n_cand <= len(order)):
                raise SystemExit(
                    f"preflight FAIL: {pt['name']}/{sp} length scan incomplete "
                    f"({n_cand - len(order)} rows unexamined) — widen TOP_N"
                )
            worst = max(worst, mx)
        if worst > FIXED_MAX_LENGTH:
            raise SystemExit(
                f"preflight FAIL: {pt['name']} max token length {worst} exceeds "
                f"FIXED_MAX_LENGTH {FIXED_MAX_LENGTH} — rows WOULD be truncated, "
                f"which defeats the purpose of Stage 4.1"
            )
        log_event(f"  OK {pt['name']} max tokens {worst} <= {FIXED_MAX_LENGTH} (no truncation)")

    log_event("=== preflight PASS ===")


def run_point_train(pt: dict, env: dict, nproc_per_node: int, master_port: int) -> int:
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_port={master_port}",
        str(TRAIN_SCRIPT),
        "--dataset_path", pt["dataset"],
        "--output_dir", pt["output_dir"],
        "--tcr_variant", pt["tcr_variant"],
        "--mhc_variant", pt["mhc_variant"],
        "--max_length", str(FIXED_MAX_LENGTH),
        "--batch_size", str(pt["batch_size"]),
        "--gradient_accumulation_steps", str(pt["grad_accum"]),
        "--bucket_boundaries", *(str(b) for b in FIXED_BUCKETS),
        "--wandb_run_name", pt["wandb_run_name"],
        *COMMON_TRAIN_ARGS,
    ]
    if pt.get("gradient_checkpointing"):
        cmd.append("--gradient_checkpointing")
    log_event(f"  CMD: {' '.join(cmd)}")
    log_path = Path(pt["output_dir"]) / "training.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def _capture_finals(pt: dict) -> dict:
    result_file = Path(pt["output_dir"]) / "final" / "canonical_v2_results.json"
    if not result_file.exists():
        return {"result_file_missing": str(result_file)}
    d = json.load(open(result_file))
    r = d.get("result", {})

    def _get(region):
        return r.get(region, {}).get("overall", {}).get("_", {}).get("perplexity")

    return {
        "overall_ppl": _get("overall"),
        "cdr3_ppl": _get("cdr3"),
        "cdr12_ppl": _get("cdr12"),
        "pocket_ppl": _get("pocket"),
        "pocket_contact_only_ppl": _get("pocket_contact_only"),
        "peptide_ppl": _get("peptide"),
        "framework_ppl": _get("framework"),
        "global_step": d.get("global_step"),
        "result_file": str(result_file),
    }


def _write_done(pt: dict, elapsed_hr: float, metrics: dict) -> None:
    marker = Path(pt["output_dir"]) / "_done"
    marker.parent.mkdir(parents=True, exist_ok=True)
    with open(marker, "w") as f:
        f.write("STATUS=PASS\n")
        f.write(f"elapsed_hr={elapsed_hr:.2f}\n")
        f.write(f"max_length={FIXED_MAX_LENGTH}\n")
        f.write(f"effective_batch={TARGET_EFFECTIVE_BATCH}\n")
        for k in ("overall_ppl", "cdr3_ppl", "cdr12_ppl", "pocket_ppl",
                  "pocket_contact_only_ppl", "peptide_ppl", "framework_ppl",
                  "global_step", "result_file"):
            if metrics.get(k) is not None:
                f.write(f"{k}={metrics[k]}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc_per_node", type=int, default=4)
    ap.add_argument("--master_port", type=int, default=29750)
    ap.add_argument("--only", default=None, help="run a single point by name")
    ap.add_argument("--skip_preflight", action="store_true")
    args = ap.parse_args()

    log_event("=== Stage 4.1 pipeline start ===")
    log_event(f"FIXED_MAX_LENGTH={FIXED_MAX_LENGTH}  buckets={FIXED_BUCKETS}  "
              f"effective_batch={TARGET_EFFECTIVE_BATCH}  gpus={args.nproc_per_node}")

    if not args.skip_preflight:
        preflight(args.nproc_per_node)

    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(QUEST_ROOT))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    results: dict = {}
    port = args.master_port
    for pt in POINTS:
        if args.only and pt["name"] != args.only:
            continue
        if _check_done(pt):
            log_event(f"--- {pt['name']} already PASS — skipping")
            results[pt["name"]] = {"skipped": True, **_capture_finals(pt)}
            continue

        log_event(f"--- {pt['name']} ({pt['label']}) start  [Stage 4: {pt['stage4_was']}]")
        t0 = time.time()
        rc = run_point_train(pt, env, args.nproc_per_node, port)
        port += 1
        elapsed_hr = (time.time() - t0) / 3600.0

        if rc != 0:
            log_event(f"--- {pt['name']} FAILED rc={rc} after {elapsed_hr:.2f} hr")
            results[pt["name"]] = {"status": "FAIL", "rc": rc, "elapsed_hr": elapsed_hr}
            _write_phase_marker("FAIL", results)
            raise SystemExit(f"{pt['name']} failed with rc={rc}")

        metrics = _capture_finals(pt)
        _write_done(pt, elapsed_hr, metrics)
        results[pt["name"]] = {"status": "PASS", "elapsed_hr": elapsed_hr, **metrics}
        log_event(f"--- {pt['name']} DONE in {elapsed_hr:.2f} hr  metrics={metrics}")

    _write_phase_marker("PASS", results)
    log_event("=== Stage 4.1 pipeline complete ===")
    log_event(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
