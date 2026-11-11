#!/usr/bin/env python3
"""Publish one corpus version to S3 (corpus repair step 11).

Refuses unless the version directory passes `corpus_preflight.corpus_problems` and the
destination prefix is empty. Uploads everything, then PUBLISHED.json LAST; the marker lists the
size and sha256 of every other object, so an interrupted publish is recognizable as incomplete.
Then re-lists the prefix and compares keys and sizes with the marker.

Usage:
    python3 publish_corpus.py --version-dir <dir> --dest s3://bucket/prefix/<version>/ [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from quest.data.corpus_preflight import MARKER, corpus_files, corpus_problems  # noqa: E402


def s3_listing(dest: str) -> dict:
    """{key relative to dest: size}. An empty prefix gives {} (aws exits 1 on it)."""
    out = subprocess.run(["aws", "s3", "ls", "--recursive", dest], capture_output=True, text=True)
    if out.returncode not in (0, 1):
        raise RuntimeError(f"aws s3 ls failed: {out.stderr.strip()}")
    prefix = dest.split("/", 3)[3]
    listing = {}
    for line in out.stdout.splitlines():
        _date, _time, size, key = line.split(None, 3)
        listing[key[len(prefix):]] = int(size)
    return listing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version-dir", type=Path, required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--benchmark-sha256", type=Path, required=True,
                    help="JSON {split file name: sha256} the leakage audit must have read")
    ap.add_argument("--dry-run", action="store_true", help="check and write nothing, locally or remotely")
    a = ap.parse_args()
    if not (a.dest.startswith("s3://") and a.dest.endswith("/") and a.dest.count("/") >= 4):
        ap.error("--dest must be s3://bucket/prefix/.../ and end with a slash")
    if (a.version_dir / MARKER).exists():
        ap.error(f"{a.version_dir / MARKER} exists: this directory was already published")
    problems = corpus_problems(a.version_dir, require_marker=False,
                               benchmark_split_sha256=json.loads(a.benchmark_sha256.read_text()))
    if problems:
        print("REFUSED: the corpus does not pass preflight:\n  " + "\n  ".join(problems))
        return 1
    existing = s3_listing(a.dest)
    if existing:
        print(f"REFUSED: {a.dest} already holds {len(existing)} objects")
        return 1
    files = corpus_files(a.version_dir)
    manifest = json.loads((a.version_dir / "dataset/_build_artifacts/build_manifest.json").read_text())
    marker = {"published_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "destination": a.dest,
              "run_id": manifest.get("run_id"), "quest_git_sha": manifest.get("quest_git_sha"),
              "contract": manifest["sampling_contract"],
              "n_objects": len(files), "total_bytes": sum(s for s, _ in files.values()),
              "objects": {k: {"size": s, "sha256": h} for k, (s, h) in files.items()}}
    print(f"{len(files)} objects, {marker['total_bytes']:,} bytes -> {a.dest}")
    if a.dry_run:
        print("dry run: nothing written")
        return 0
    # The marker exists nowhere until every other object is up and listed: a failed upload
    # leaves neither a local nor a remote claim of completeness.
    subprocess.run(["aws", "s3", "cp", "--recursive", "--only-show-errors", "--exclude", MARKER,
                    "--exclude", "*.DS_Store", str(a.version_dir), a.dest], check=True)
    partial = s3_listing(a.dest)
    want = {k: s for k, (s, _) in files.items()}
    if partial != want:
        diff = sorted(k for k in set(partial) | set(want) if partial.get(k) != want.get(k))
        print(f"REFUSED to write the marker: {len(diff)} objects differ after upload, e.g. {diff[:3]}")
        return 1
    text = json.dumps(marker, indent=2) + "\n"
    subprocess.run(["aws", "s3", "cp", "--only-show-errors", "-", a.dest + MARKER], input=text.encode(), check=True)
    final = s3_listing(a.dest)
    if final != {**want, MARKER: len(text.encode())}:
        print("FAILED: the final listing is not the uploaded set plus the marker")
        return 1
    with (a.version_dir / MARKER).open("x") as fh:
        fh.write(text)
    print(f"PUBLISHED {len(final)} objects to {a.dest}; marker written last")
    return 0


if __name__ == "__main__":
    sys.exit(main())
