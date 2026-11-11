"""Preflight for a published CPT corpus (corpus repair step 11).

`corpus_problems(version_dir)` lists every reason a corpus directory may not be trained on. An
empty list is the only pass, and absence never counts as agreement: a missing file, a missing
field or a zero where evidence is expected is a problem.

Layout of a version directory, locally and on S3:

    dataset/                      the builder's output: train/ validation/ test/, dataset_dict.json
    dataset/_build_artifacts/     build_manifest.json, the builder's and verifier's markers, v1..v11
    leakage-audit.json            cpt_built_audit.py, on the built files
    coverage-by-split.csv         the same
    sampling-contract.json        the contract the build names, byte for byte
    PUBLISHED.json                written last: sha256 and size of every other file

What it is for: run05 was trained on with no isolation, 168 class-I rows and no record that
would have said so. A launcher calls this and refuses on a non-empty answer.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

MARKER = "PUBLISHED.json"
VERIFIER_CHECKS = tuple(f"v{i}" for i in range(1, 12))
CLASS_I_STRATA = ("s1_", "s2_", "s3_", "s4_")
DATASET_SPLITS = ("train", "validation", "test")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def corpus_files(version_dir) -> dict:
    """{relative path: (size, sha256)} of every file in the directory except the marker.
    Refuses a symbolic link: its target would escape the inventory."""
    root = Path(version_dir)
    links = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_symlink()]
    if links:
        raise ValueError(f"symbolic links in a corpus directory: {links[:3]}")
    return {str(p.relative_to(root)): (p.stat().st_size, sha256_file(p))
            for p in sorted(root.rglob("*")) if p.is_file() and p.name != MARKER and p.name != ".DS_Store"}


def _json(path, problems):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        problems.append(f"{path}: unreadable ({type(e).__name__})")
        return None


def corpus_problems(version_dir, require_marker: bool = True, benchmark_split_sha256: dict | None = None) -> list[str]:
    """Every reason this corpus must not be used. `require_marker=False` is for the publisher,
    which runs the same checks just before it writes the marker. `benchmark_split_sha256`
    ({file name: sha256}), when given, must be exactly what the leakage audit read."""
    root, problems = Path(version_dir), []
    art = root / "dataset" / "_build_artifacts"
    links = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_symlink()]
    if links:
        return [f"symbolic links in the corpus directory: {links[:3]}"]
    if not (root / "dataset" / "dataset_dict.json").is_file():
        problems.append("dataset/dataset_dict.json is missing")
    for split in DATASET_SPLITS:
        shards = list((root / "dataset" / split).glob("*.arrow"))
        if not shards or any(p.stat().st_size == 0 for p in shards):
            problems.append(f"dataset/{split} has no data files, or an empty one")
    for name, who in (("_implementer_done", "builder"), ("_verifier_done", "verifier")):
        try:
            if not (art / name).read_text().startswith("STATUS=PASS"):
                problems.append(f"{who} marker is not PASS")
        except OSError:
            problems.append(f"{who} marker is missing")

    manifest = _json(art / "build_manifest.json", problems) or {}
    contract = manifest.get("sampling_contract")
    if not contract:
        problems.append("the build names no sampling contract")
    else:
        try:
            if sha256_file(root / "sampling-contract.json") != contract.get("sha256"):
                problems.append("sampling-contract.json is not the contract the build names")
        except OSError:
            problems.append("sampling-contract.json is missing")
        if contract.get("kind") == "examples per stratum" and contract.get("background_universe") != "contract":
            problems.append(f"background drawn from {contract.get('background_universe')!r}: a test build")
    isolation = manifest.get("antigenic_specificity_exclusion")
    if not (isinstance(isolation, dict) and isolation.get("policy") and isolation.get("list_sha256")
            and isolation.get("n_peptides")):
        problems.append("the build was not isolated from the benchmark")
    run_id = manifest.get("run_id")
    if not run_id:
        problems.append("the build manifest has no run_id")

    for check in VERIFIER_CHECKS:
        found = sorted(art.glob(f"{check}_*.json"))
        result = _json(found[0], problems) if len(found) == 1 else None
        if result is None or result.get("status") != "PASS":
            problems.append(f"verifier check {check.upper()} is not PASS")
        elif check == "v11" and (result.get("contract") is None or not result.get("version")
                                 or result.get("sha256") != (contract or {}).get("sha256")
                                 or result.get("version") != (contract or {}).get("version")):
            problems.append("V11 did not check this build's contract and version")

    leak = _json(root / "leakage-audit.json", problems) or {}
    if leak.get("status") != "PASS" or leak.get("minimum_edit_distance_to_a_heldout_peptide") != "at least 2":
        problems.append("the leakage audit of the built files is not a clean PASS")
    elif not run_id or leak.get("run_id") != run_id or not leak.get("peptide_bearing_rows"):
        problems.append("the leakage audit is of another build, or checked no peptide rows")
    elif not (leak.get("task") and leak.get("heldout_parts") and leak.get("distinct_heldout_peptides")
              and leak.get("benchmark_split_sha256")) or leak.get("heldout_parts_without_a_peptide"):
        problems.append("the leakage audit does not say which benchmark it compared against, or compared against nothing")
    elif benchmark_split_sha256 is not None and leak.get("benchmark_split_sha256") != benchmark_split_sha256:
        problems.append("the leakage audit read other benchmark files than the ones expected")

    try:
        with (root / "coverage-by-split.csv").open() as fh:
            class_i = sum(int(r["rows"]) for r in csv.DictReader(fh)
                          if r["split"] == "train" and r["stratum"].startswith(CLASS_I_STRATA))
        if class_i <= 0:
            problems.append("no classical class-I rows in training")
    except (OSError, KeyError, ValueError):
        problems.append("coverage-by-split.csv is missing or unreadable")

    if require_marker:
        marker = _json(root / MARKER, problems) if (root / MARKER).exists() else None
        if marker is None:
            problems.append(f"{MARKER} is missing: the corpus was not published complete")
        else:
            listed = {k: (v["size"], v["sha256"]) for k, v in marker.get("objects", {}).items()}
            if not listed:
                problems.append(f"{MARKER} lists no objects")
            have = corpus_files(root)
            for name in sorted(set(listed) | set(have)):
                if listed.get(name) != have.get(name):
                    problems.append(f"{name}: differs from {MARKER}"
                                    if name in listed and name in have else
                                    f"{name}: {'missing' if name in listed else 'not listed in ' + MARKER}")
    return problems
