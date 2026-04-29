"""Sync re-tokenized benchmark_v2 outputs to HuggingFace.

Pushes only the artifacts that changed during the manifest column-strip fix:

- `tokenized/foundation/foundation_10M_C3.parquet`
- `tokenized/foundation/foundation_100M_C3.parquet`
- `tokenized/foundation/foundation_500M_C3/` (only if `_COMPLETE` marker exists)
- `tokenized/mini/*.parquet`

Splits, clusters, foundation val/test, benchmark-scope tokenized outputs, and
audit artifacts are NOT re-uploaded.

Usage:
    huggingface-cli login   # or: export HF_TOKEN=hf_xxx
    python scripts/data_processing/sync_benchmark_v2_to_hf.py \\
        --repo-id myuser/benchmark_v2

Add `--dry-run` to print what would be pushed without sending.
Add `--include-500m-partial` to push the in-progress 500M shard dir even if
the `_COMPLETE` marker is missing (useful for early eyeballing).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

from huggingface_hub import HfApi

ROOT = Path("/home/ubuntu/quest/data/benchmark_v2")


@dataclass
class SyncTarget:
    local: Path
    repo_path: str
    kind: str  # "file" or "folder"
    commit_message: str


def build_targets(include_500m_partial: bool) -> List[SyncTarget]:
    targets: List[SyncTarget] = [
        SyncTarget(
            local=ROOT / "tokenized/foundation/foundation_10M_C3.parquet",
            repo_path="tokenized/foundation/foundation_10M_C3.parquet",
            kind="file",
            commit_message="Re-tokenize foundation_10M with full enriched cols (tra_full/trb_full/etc)",
        ),
        SyncTarget(
            local=ROOT / "tokenized/foundation/foundation_100M_C3.parquet",
            repo_path="tokenized/foundation/foundation_100M_C3.parquet",
            kind="file",
            commit_message="Re-tokenize foundation_100M with full enriched cols",
        ),
        SyncTarget(
            local=ROOT / "tokenized/mini",
            repo_path="tokenized/mini",
            kind="folder",
            commit_message="Re-tokenize mini ablations (M2/M3/O1/O3) with full enriched cols",
        ),
    ]

    shard_dir = ROOT / "tokenized/foundation/foundation_500M_C3"
    complete_marker = shard_dir / "_COMPLETE"
    if shard_dir.exists() and (complete_marker.exists() or include_500m_partial):
        targets.append(SyncTarget(
            local=shard_dir,
            repo_path="tokenized/foundation/foundation_500M_C3",
            kind="folder",
            commit_message="Re-tokenize foundation_500M with full enriched cols (70 shards)",
        ))
    return targets


def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def local_size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def push_target(api: HfApi, repo_id: str, t: SyncTarget, dry_run: bool) -> None:
    size = local_size(t.local) if t.local.exists() else 0
    print(f"\n[{t.kind}] {t.local} -> {repo_id}:{t.repo_path}  ({fmt_size(size)})", flush=True)
    if not t.local.exists():
        print(f"  SKIP: local path missing")
        return
    if dry_run:
        print(f"  DRY-RUN: would push with message: {t.commit_message!r}")
        return
    if t.kind == "file":
        api.upload_file(
            path_or_fileobj=str(t.local),
            path_in_repo=t.repo_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=t.commit_message,
        )
    else:
        # `upload_folder` is incremental against remote — files unchanged by
        # content hash are skipped server-side.
        api.upload_folder(
            folder_path=str(t.local),
            path_in_repo=t.repo_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=t.commit_message,
        )
    print(f"  OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="HF dataset repo, e.g. myuser/benchmark_v2")
    ap.add_argument("--dry-run", action="store_true", help="Print plan, don't push")
    ap.add_argument("--include-500m-partial", action="store_true",
                    help="Push 500M shard dir even if _COMPLETE marker is missing")
    args = ap.parse_args()

    api = HfApi()
    try:
        info = api.repo_info(repo_id=args.repo_id, repo_type="dataset")
        print(f"Target: {args.repo_id}  (last modified: {info.last_modified})")
    except Exception as e:
        print(f"ERROR: cannot reach repo {args.repo_id} — {e}", file=sys.stderr)
        return 1

    targets = build_targets(include_500m_partial=args.include_500m_partial)
    print(f"\nPlanning {len(targets)} sync targets:")
    total = 0
    for t in targets:
        if t.local.exists():
            total += local_size(t.local)
    print(f"  Total local size: {fmt_size(total)}")

    for t in targets:
        push_target(api, args.repo_id, t, dry_run=args.dry_run)

    print("\nDone." if not args.dry_run else "\nDry run complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
