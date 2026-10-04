"""Phase 1 master orchestrator: 10 (ckpt × cube) runs, train then evaluate.

Sequential execution. Each run is resumable via per-run `_train_done` and
`_eval_done` markers.

Usage:
    python -m track_a.esm_sft_cubes.run_phase1 \
        --grid track_a/esm_sft_cubes/configs/_grid.yaml \
        [--only A,B] [--skip-train] [--skip-eval] [--smoke]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

_REPO_ROOT = "/home/ubuntu/quest"
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def materialize_config(run: dict, defaults: dict, output_root: str, out_path: Path) -> None:
    merged = {**defaults, **run, "output_dir": output_root}
    out_path.write_text(yaml.safe_dump(merged, sort_keys=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--grid", default="track_a/esm_sft_cubes/configs/_grid.yaml")
    p.add_argument("--only", default=None, help="Comma-separated checkpoint names to run")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    grid = yaml.safe_load(Path(args.grid).read_text())
    output_root = grid["output_root"]
    defaults = grid.get("defaults", {})
    runs = grid["runs"]
    only = set(args.only.split(",")) if args.only else None

    configs_dir = Path("/home/ubuntu/quest/track_a/esm_sft_cubes/configs")
    configs_dir.mkdir(parents=True, exist_ok=True)
    driver_log = Path(output_root) / "DRIVER.log"
    Path(output_root).mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
        sys.stdout.write(line); sys.stdout.flush()
        with open(driver_log, "a") as f:
            f.write(line)

    for run in runs:
        ckpt_name = run["ckpt_name"]; cube_name = run["cube_name"]
        if only is not None and ckpt_name not in only:
            continue
        cfg_path = configs_dir / f"{ckpt_name}_{cube_name}.yaml"
        materialize_config(run, defaults, output_root, cfg_path)
        log(f"=== run {ckpt_name} × {cube_name} ===")
        out_dir = Path(output_root) / f"{ckpt_name}__{cube_name}"

        if not args.skip_train and not (out_dir / "_train_done").exists():
            log(f"  train …")
            cmd = ["python", "-m", "track_a.esm_sft_cubes.train", "--config", str(cfg_path)]
            if args.smoke: cmd.append("--smoke")
            t0 = time.time()
            r = subprocess.run(cmd, cwd=_REPO_ROOT)
            log(f"  train rc={r.returncode}  ({(time.time()-t0)/60:.1f}m)")
            if r.returncode != 0:
                log(f"  train FAILED — stopping"); raise SystemExit(1)
        else:
            log(f"  train: skip ({(out_dir / '_train_done').exists() and 'done' or 'flag'})")

        if not args.skip_eval and not (out_dir / "_eval_done").exists():
            log(f"  eval …")
            cmd = ["python", "-m", "track_a.esm_sft_cubes.evaluate", "--config", str(cfg_path)]
            if args.smoke: cmd.append("--smoke")
            t0 = time.time()
            r = subprocess.run(cmd, cwd=_REPO_ROOT)
            log(f"  eval rc={r.returncode}  ({(time.time()-t0)/60:.1f}m)")
            if r.returncode != 0:
                log(f"  eval FAILED — continuing to next run")
                # don't raise; one bad eval shouldn't tank the whole orchestrator
        else:
            log(f"  eval: skip ({(out_dir / '_eval_done').exists() and 'done' or 'flag'})")

    log("ALL DONE")


if __name__ == "__main__":
    main()
