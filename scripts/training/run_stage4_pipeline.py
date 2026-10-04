#!/usr/bin/env python3
"""Phase 3 of the Stage 4 combined pipeline: train the 5-point ESM-C 300M
coupled-representation sweep.

Points (executed in cost-ascending order):
  E. (eval-only) full TCR + pocket MHC — re-eval of cpt_stage2_run03_full
  A. cdr3   + pocket           on run01    (~3 hr)
  D. cdr3   + full MHC         on run05    (~5 hr)
  B. cdr123 + pocket_contact   on run10    (~4 hr)  -- needs Phase 2
  C. full   + full             on run09    (~28 hr) -- long pole

Per-point ``_done`` markers make the driver fully resumable; re-running
skips any point already marked PASS.

All training runs log to wandb project ``tcrbench-v3-cpt-stage4``.
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
POINT_E_SCRIPT = HERE / "stage4_pointE_reeval.py"
PHASE_MARKER = QUEST_ROOT / "checkpoints/_phase3_done"
PHASE2_MARKER = QUEST_ROOT / "checkpoints/_phase2_done"
DRIVER_LOG = QUEST_ROOT / "checkpoints/stage4_combined_DRIVER.log"

WANDB_PROJECT = "tcrbench-v3-cpt-stage4"

# Execution order: E, A, D, B, C (cheap → expensive)
POINTS = [
    {
        "name": "pointE_run03_reeval",
        "type": "reeval",
        "output_dir": str(QUEST_ROOT / "checkpoints/cpt_stage4_pointE_run03_reeval"),
        "checkpoint": str(QUEST_ROOT / "checkpoints/cpt_stage2_run03_full/final"),
        "tcr_variant": "full",
        "mhc_variant": "pocket",
        "max_length": 1024,
        "eval_batch_size": 16,
        "expected_wall_hr": 0.5,
    },
    {
        "name": "pointA_run01",
        "type": "train",
        "output_dir": str(QUEST_ROOT / "checkpoints/cpt_stage4_pointA_run01"),
        "dataset": str(QUEST_ROOT / "data/cpt_datasets/run01_cdr3_pocket_proportional_10M"),
        "tcr_variant": "cdr3",
        "mhc_variant": "pocket",
        "max_length": 64,
        "batch_size": 192,
        "grad_accum": 2,
        "buckets": [16, 24, 32, 48, 64],
        "gradient_checkpointing": False,
        "wandb_run_name": "stage4-pointA-cdr3-pocket",
        "expected_wall_hr": 3,
    },
    {
        "name": "pointD_run05",
        "type": "train",
        "output_dir": str(QUEST_ROOT / "checkpoints/cpt_stage4_pointD_run05"),
        "dataset": str(QUEST_ROOT / "data/cpt_datasets/run05_cdr3_fullmhc_proportional_10M"),
        "tcr_variant": "cdr3",
        "mhc_variant": "full",
        "max_length": 1024,
        "batch_size": 16,
        "grad_accum": 16,
        "buckets": [64, 128, 256, 512, 1024],
        "gradient_checkpointing": True,
        "wandb_run_name": "stage4-pointD-cdr3-fullmhc",
        "expected_wall_hr": 5,
    },
    {
        "name": "pointB_run10",
        "type": "train",
        "output_dir": str(QUEST_ROOT / "checkpoints/cpt_stage4_pointB_run10"),
        "dataset": str(QUEST_ROOT / "data/cpt_datasets/run10_cdr123_pocketcontact_proportional_10M"),
        "tcr_variant": "cdr123",
        "mhc_variant": "pocket_contact",
        "max_length": 256,
        "batch_size": 64,
        "grad_accum": 4,
        "buckets": [32, 64, 96, 128, 192, 256],
        "gradient_checkpointing": False,
        "wandb_run_name": "stage4-pointB-cdr123-pocketcontact",
        "expected_wall_hr": 4,
    },
    {
        "name": "pointC_run09",
        "type": "train",
        "output_dir": str(QUEST_ROOT / "checkpoints/cpt_stage4_pointC_run09"),
        "dataset": str(QUEST_ROOT / "data/cpt_datasets/run09_full_fullmhc_proportional_10M"),
        "tcr_variant": "full",
        "mhc_variant": "full",
        "max_length": 1024,
        "batch_size": 16,
        "grad_accum": 16,
        "buckets": [64, 128, 256, 512, 1024],
        "gradient_checkpointing": True,
        "wandb_run_name": "stage4-pointC-full-full",
        "expected_wall_hr": 28,
    },
]

COMMON_TRAIN_ARGS = [
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
    "--wandb_project", WANDB_PROJECT,
    "--report_to", "wandb",
    "--early_stop_n_patience", "3",
    "--early_stop_min_rel", "0.01",
    # Stage 4: optimize for pocket residues (MHC-bearing rows by construction)
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
    txt = marker.read_text()
    return "STATUS=PASS" in txt


def _write_phase_marker(status: str, results: dict) -> None:
    PHASE_MARKER.parent.mkdir(parents=True, exist_ok=True)
    with open(PHASE_MARKER, "w") as f:
        f.write(f"STATUS={status}\n")
        f.write(json.dumps(results, indent=2))


def run_point_reeval(pt: dict, env: dict) -> int:
    """Single-GPU re-evaluation of the Point E checkpoint."""
    cmd = [
        sys.executable, str(POINT_E_SCRIPT),
        "--base_model", "esmc_300m",
        "--checkpoint", pt["checkpoint"],
        "--canonical_eval", "/home/ubuntu/quest/data/cpt_canonical_eval_v2",
        "--tcr_variant", pt["tcr_variant"],
        "--mhc_variant", pt["mhc_variant"],
        "--output_dir", pt["output_dir"],
        "--eos_strategy", "raw",
        "--batch_size", str(pt["eval_batch_size"]),
        "--max_length", str(pt["max_length"]),
    ]
    log_event(f"  CMD: {' '.join(cmd)}")
    log_path = Path(pt["output_dir"]) / "reeval.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def run_point_train(pt: dict, env: dict, nproc_per_node: int, master_port: int) -> int:
    """Multi-GPU training launch via torchrun."""
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_port={master_port}",
        str(TRAIN_SCRIPT),
        "--dataset_path", pt["dataset"],
        "--output_dir", pt["output_dir"],
        "--tcr_variant", pt["tcr_variant"],
        "--mhc_variant", pt["mhc_variant"],
        "--max_length", str(pt["max_length"]),
        "--batch_size", str(pt["batch_size"]),
        "--gradient_accumulation_steps", str(pt["grad_accum"]),
        "--bucket_boundaries", *(str(b) for b in pt["buckets"]),
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
    """Read final canonical_v2_results.json if present, return summary metrics."""
    if pt["type"] == "reeval":
        result_file = Path(pt["output_dir"]) / "canonical_v2_results.json"
    else:
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
        "result_file": str(result_file),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc_per_node", type=int, default=4)
    ap.add_argument("--master_port", type=int, default=29710)
    ap.add_argument(
        "--only",
        choices=[p["name"] for p in POINTS],
        default=None,
        help="Run only one point",
    )
    ap.add_argument(
        "--skip_phase2_check",
        action="store_true",
        help="Allow running even if Phase 2 marker is missing (e.g. when only running Points E/A/D)",
    )
    args = ap.parse_args()

    points = [p for p in POINTS if args.only is None or p["name"] == args.only]
    # Phase 2 prerequisite — only required if Point B is in the slate
    needs_phase2 = any(p["name"] == "pointB_run10" for p in points)
    if needs_phase2 and not args.skip_phase2_check:
        if not PHASE2_MARKER.exists() or "STATUS=PASS" not in PHASE2_MARKER.read_text():
            raise SystemExit(
                "Phase 2 not complete — Point B requires Run 10. "
                "Use --skip_phase2_check to override, or run build_run10.py first."
            )

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    overall_t0 = time.time()
    results: dict[str, dict] = {}

    for i, pt in enumerate(points, start=1):
        log_event(f"=" * 72)
        log_event(f"Point {i}/{len(points)}: {pt['name']} (estimated {pt['expected_wall_hr']} hr)")

        if _check_done(pt):
            log_event(f"  SKIP — _done marker already PASS in {pt['output_dir']}")
            results[pt["name"]] = {"status": "skipped", **_capture_finals(pt)}
            continue

        os.makedirs(pt["output_dir"], exist_ok=True)
        t0 = time.time()

        if pt["type"] == "reeval":
            rc = run_point_reeval(pt, env)
        else:
            rc = run_point_train(pt, env, args.nproc_per_node, args.master_port + i)

        elapsed_hr = (time.time() - t0) / 3600
        if rc != 0:
            log_event(f"  FAIL {pt['name']} after {elapsed_hr:.1f} hr (exit={rc})")
            results[pt["name"]] = {"status": "fail", "exit_code": rc, "elapsed_hr": elapsed_hr}
            _write_phase_marker("FAIL", results)
            raise SystemExit(rc)

        # Verify expected output exists
        if pt["type"] == "reeval":
            expected = Path(pt["output_dir"]) / "canonical_v2_results.json"
        else:
            expected = Path(pt["output_dir"]) / "final" / "canonical_v2_results.json"
        if not expected.exists():
            log_event(f"  FAIL {pt['name']}: expected result file missing at {expected}")
            results[pt["name"]] = {"status": "fail_no_results", "elapsed_hr": elapsed_hr}
            _write_phase_marker("FAIL", results)
            raise SystemExit(1)

        finals = _capture_finals(pt)
        marker = Path(pt["output_dir"]) / "_done"
        with open(marker, "w") as f:
            f.write("STATUS=PASS\n")
            f.write(f"elapsed_hr={elapsed_hr:.2f}\n")
            for k, v in finals.items():
                f.write(f"{k}={v}\n")
        log_event(f"  DONE {pt['name']} in {elapsed_hr:.2f} hr  metrics={finals}")
        results[pt["name"]] = {"status": "pass", "elapsed_hr": elapsed_hr, **finals}

    total_hr = (time.time() - overall_t0) / 3600
    log_event(f"Phase 3 complete in {total_hr:.1f} hr total")
    _write_phase_marker("PASS", results)


if __name__ == "__main__":
    main()
