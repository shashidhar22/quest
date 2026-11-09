#!/usr/bin/env python3
"""Phase 2 of the Stage 4 combined pipeline: build Run 10 dataset
(cdr123 + pocket_contact + proportional + 10M).

Thin wrapper that invokes ``cpt_dataset_builder.py`` with the Run 10
parameters, verifies the output structure, and writes a phase marker.

Pre-requisite: Phase 1 must have populated
``/home/ubuntu/quest/data/molecule_clusters/``.
"""
from __future__ import annotations

import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from datasets import load_from_disk

QUEST_ROOT = Path("/home/ubuntu/quest")
BUILDER = QUEST_ROOT / "scripts/data_processing/cpt_dataset_builder.py"
OUT_DIR = QUEST_ROOT / "data/cpt_datasets/run10_cdr123_pocketcontact_proportional_10M"
PHASE_MARKER = QUEST_ROOT / "checkpoints/_phase2_done"
PHASE1_MARKER = QUEST_ROOT / "checkpoints/_phase1_done"
CLUSTERS_DIR = QUEST_ROOT / "data/molecule_clusters/clusters"

# The builder's DEFAULT_MEMORY_GB (850) was tuned for the decommissioned x2gd
# (~1 TB RAM). On this g6.12xlarge (181 GB, no swap), DuckDB must be capped well
# below physical RAM or it OOM-kills the box. Size memory_limit to 60% of
# detected RAM (DuckDB spills the remainder to --duckdb-tmp on the root disk),
# and hold threads below the core count since DuckDB peak memory scales with the
# number of concurrent operator pipelines.
_TOTAL_RAM_GB = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3))
SAFE_MEMORY_GB = max(16, int(_TOTAL_RAM_GB * 0.60))
SAFE_THREADS = max(4, min(24, (os.cpu_count() or 8) // 2))
DUCKDB_TMP = QUEST_ROOT / "data/duckdb_tmp_cpt"


def _log(msg: str) -> None:
    print(f"[build_run10] {msg}", flush=True)


def _require_phase1() -> None:
    if not PHASE1_MARKER.exists():
        raise SystemExit("Phase 1 marker missing — run recover_cluster_files.py first")
    content = PHASE1_MARKER.read_text()
    if "STATUS=PASS" not in content:
        raise SystemExit(f"Phase 1 did not PASS:\n{content}")
    # Sanity: clustering files exist
    expected = [
        "trb_cdr3_clusters.parquet",
        "tra_cdr3_clusters.parquet",
        "peptide_clusters.parquet",
        "mhc_one_pocket_clusters.parquet",
        "mhc_two_pocket_clusters.parquet",
    ]
    missing = [n for n in expected if not (CLUSTERS_DIR / n).exists()]
    if missing:
        raise SystemExit(f"Missing cluster files: {missing}")


def _write_marker(status: str, **extra: str) -> None:
    PHASE_MARKER.parent.mkdir(parents=True, exist_ok=True)
    with open(PHASE_MARKER, "w") as f:
        f.write(f"STATUS={status}\n")
        for k, v in extra.items():
            f.write(f"{k}={v}\n")


# Hard cgroup backstop: cap the builder's RSS so an overshoot of the DuckDB
# soft limit kills only the builder process group, never the whole box (no swap
# here, so a global OOM hangs the instance). Leaves ~30 GB for the OS.
CGROUP_HARD_CAP_GB = max(SAFE_MEMORY_GB + 24, _TOTAL_RAM_GB - 32)


def _wrap_with_memory_cap(cmd: list[str]) -> list[str]:
    """Prefix the command with a user-scope systemd cgroup memory cap if
    available; otherwise return the command unchanged (soft limit still applies).
    """
    if shutil.which("systemd-run") is None:
        _log("systemd-run unavailable — relying on DuckDB soft memory_limit only")
        return cmd
    probe = subprocess.run(
        ["systemd-run", "--user", "--scope", "-p", "MemoryMax=64M", "--quiet", "true"],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        _log("user-scope systemd-run not usable — relying on DuckDB soft limit only")
        return cmd
    cap = f"{CGROUP_HARD_CAP_GB}G"
    _log(f"Wrapping builder in cgroup hard cap MemoryMax={cap}, MemorySwapMax=0")
    return [
        "systemd-run", "--user", "--scope", "--quiet",
        "-p", f"MemoryMax={cap}",
        "-p", "MemorySwapMax=0",
        *cmd,
    ]


def main() -> None:
    start = time.time()
    _require_phase1()

    DUCKDB_TMP.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(BUILDER),
        "--tcr-variant", "cdr123",
        "--mhc-variant", "pocket_contact",
        "--distribution", "proportional",
        "--scale", "10000000",
        "--output-dir", str(OUT_DIR),
        "--run-id", "run10",
        # Reproduction of a past build, so the legacy eligibility policy is selected
        # deliberately -- including its 320-aa rejection, which discards most full-length
        # class-I heavy chains. A NEW corpus must pass reference-aware-v1 instead.
        "--eligibility-policy", "legacy-320-v0",
        # Likewise deliberate: past builds were not isolated from the benchmark.
        "--benchmark-exclusion", "none",
        "--seed", "42",
        # Memory safety on this 181 GB / no-swap g6 instance (see module header).
        "--memory-gb", str(SAFE_MEMORY_GB),
        "--threads", str(SAFE_THREADS),
        "--duckdb-tmp", str(DUCKDB_TMP),
    ]
    _log(f"System RAM detected: {_TOTAL_RAM_GB} GB  ->  DuckDB memory_limit={SAFE_MEMORY_GB} GB, "
         f"threads={SAFE_THREADS}, tmp={DUCKDB_TMP}")
    cmd = _wrap_with_memory_cap(cmd)
    _log("Launching builder: " + " ".join(cmd))

    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        _write_marker("FAIL", reason=f"builder_exit_{proc.returncode}")
        raise SystemExit(f"Builder exited {proc.returncode}")

    # Builder writes its own _implementer_done marker.
    impl_done = OUT_DIR / "_build_artifacts" / "_implementer_done"
    if not impl_done.exists():
        _write_marker("FAIL", reason="builder_no_implementer_done")
        raise SystemExit("Builder finished without _implementer_done marker")
    impl_text = impl_done.read_text()
    if "STATUS=PASS" not in impl_text:
        _write_marker("FAIL", reason="builder_implementer_not_pass")
        raise SystemExit(f"Builder _implementer_done not PASS:\n{impl_text}")

    # Verify dataset shape.
    _log("Verifying output dataset")
    ds = load_from_disk(str(OUT_DIR))
    splits = sorted(ds.keys())
    if splits != ["test", "train", "validation"]:
        _write_marker("FAIL", reason=f"unexpected_splits_{splits}")
        raise SystemExit(f"Expected train/validation/test splits, got {splits}")

    n_total = sum(len(ds[s]) for s in splits)
    _log(f"Total rows: {n_total:,}  (train={len(ds['train']):,}, "
         f"val={len(ds['validation']):,}, test={len(ds['test']):,})")

    if abs(n_total - 10_000_000) > 100_000:  # generous; builder may drop ~empty rows
        _write_marker("FAIL", reason=f"row_count_{n_total}")
        raise SystemExit(f"Row count {n_total:,} too far from 10M")

    # Length distribution sanity.
    n_sample = min(10_000, len(ds["train"]))
    sample = ds["train"].shuffle(seed=42).select(range(n_sample))
    lengths = [len(r["input_text"]) for r in sample]
    mean_len = statistics.mean(lengths)
    p95_len = statistics.quantiles(lengths, n=20)[18]
    p99_len = statistics.quantiles(lengths, n=100)[98]
    _log(f"input_text chars: mean={mean_len:.1f}  p95={p95_len:.0f}  p99={p99_len:.0f}")

    # cdr123 + pocket_contact: expect mean ~50-120 chars, p95 < 250.
    # Accept generously; this is a sanity smoke, not a strict gate.
    if not (10 < mean_len < 250):
        _write_marker("FAIL", reason=f"mean_length_{mean_len:.1f}")
        raise SystemExit(f"Mean input_text length {mean_len:.1f} outside expected range")

    elapsed = time.time() - start
    _write_marker(
        "PASS",
        total_rows=str(n_total),
        mean_chars=f"{mean_len:.1f}",
        p95_chars=f"{p95_len:.0f}",
        p99_chars=f"{p99_len:.0f}",
        elapsed_min=f"{elapsed / 60:.1f}",
    )
    _log(f"Phase 2 PASS in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
