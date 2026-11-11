"""The training preflight for a published corpus: every missing or inconsistent piece refuses."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quest.data.corpus_preflight import MARKER, corpus_files, corpus_problems  # noqa: E402


BENCH = {"as_paired_i_val.parquet": "cd"}


def make(root: Path, publish: bool = True) -> Path:
    art = root / "dataset" / "_build_artifacts"
    art.mkdir(parents=True)
    (root / "dataset" / "dataset_dict.json").write_text('{"splits": ["train", "validation", "test"]}')
    for split in ("train", "validation", "test"):
        (root / "dataset" / split).mkdir()
        (root / "dataset" / split / "data-00000-of-00001.arrow").write_bytes(b"rows")
    (root / "sampling-contract.json").write_text('{"contract": "cpt-sampling-contract-v1"}')
    sha = hashlib.sha256((root / "sampling-contract.json").read_bytes()).hexdigest()
    (art / "_implementer_done").write_text("STATUS=PASS\n")
    (art / "_verifier_done").write_text("STATUS=PASS\n")
    (art / "build_manifest.json").write_text(json.dumps({
        "run_id": "r1", "antigenic_specificity_exclusion": {
            "policy": "heldout-peptides-v2", "list_sha256": "ab", "n_peptides": 3},
        "sampling_contract": {"sha256": sha, "version": "enriched", "kind": "examples per stratum", "background_universe": "contract"}}))
    for i in range(1, 12):
        body = {"status": "PASS"}
        if i == 11:
            body.update(contract="cpt-sampling-contract-v1", sha256=sha, version="enriched")
        (art / f"v{i}_check.json").write_text(json.dumps(body))
    (root / "leakage-audit.json").write_text(json.dumps({
        "status": "PASS", "minimum_edit_distance_to_a_heldout_peptide": "at least 2", "run_id": "r1",
        "peptide_bearing_rows": 10, "task": "as_paired_i", "heldout_parts": ["val"],
        "distinct_heldout_peptides": 4, "heldout_parts_without_a_peptide": [], "benchmark_split_sha256": BENCH}))
    (root / "coverage-by-split.csv").write_text(
        "split,stratum,rows\ntrain,s1_paired_tcr_peptide_classI,5\ntrain,s8_peptide_only,9\nval,s1_paired_tcr_peptide_classI,1\n")
    if publish:
        (root / MARKER).write_text(json.dumps({"objects": {k: {"size": s, "sha256": h}
                                                          for k, (s, h) in corpus_files(root).items()}}))
    return root


def edit_json(path: Path, **changes):
    body = json.loads(path.read_text())
    for k, v in changes.items():
        if v is None:
            body.pop(k, None)
        else:
            body[k] = v
    path.write_text(json.dumps(body))


def test_a_complete_published_corpus_has_no_problem(tmp_path):
    assert corpus_problems(make(tmp_path)) == []


def test_an_unpublished_corpus_passes_only_when_the_marker_is_not_required(tmp_path):
    root = make(tmp_path, publish=False)
    assert corpus_problems(root, require_marker=False) == []
    assert any(MARKER in p for p in corpus_problems(root))


ART = "dataset/_build_artifacts/"
BREAKS = {
    "builder failed": lambda r: (r / ART / "_implementer_done").write_text("STATUS=FAIL: refused\n"),
    "verifier missing": lambda r: (r / ART / "_verifier_done").unlink(),
    "no contract": lambda r: edit_json(r / ART / "build_manifest.json", sampling_contract=None),
    "other contract": lambda r: (r / "sampling-contract.json").write_text('{"contract": "edited"}'),
    "contract file missing": lambda r: (r / "sampling-contract.json").unlink(),
    "test build": lambda r: edit_json(r / ART / "build_manifest.json", sampling_contract={
        **json.loads((r / ART / "build_manifest.json").read_text())["sampling_contract"], "background_universe": {"subset": {}}}),
    "not isolated": lambda r: edit_json(r / ART / "build_manifest.json", antigenic_specificity_exclusion="NOT_APPLIED"),
    "a check failed": lambda r: edit_json(r / ART / "v5_check.json", status="FAIL"),
    "a check missing": lambda r: (r / ART / "v10_check.json").unlink(),
    "V11 of another contract": lambda r: edit_json(r / ART / "v11_check.json", sha256="0" * 64),
    "V11 not applied": lambda r: edit_json(r / ART / "v11_check.json", contract=None),
    "leak": lambda r: edit_json(r / "leakage-audit.json", status="FAIL"),
    "near peptide": lambda r: edit_json(r / "leakage-audit.json", minimum_edit_distance_to_a_heldout_peptide=1),
    "audit of another build": lambda r: edit_json(r / "leakage-audit.json", run_id="r0"),
    "audit checked nothing": lambda r: edit_json(r / "leakage-audit.json", peptide_bearing_rows=0),
    "audit missing": lambda r: (r / "leakage-audit.json").unlink(),
    "no class I in training": lambda r: (r / "coverage-by-split.csv").write_text(
        "split,stratum,rows\ntrain,s8_peptide_only,9\nval,s1_paired_tcr_peptide_classI,1\n"),
    "coverage missing": lambda r: (r / "coverage-by-split.csv").unlink(),
    # Review of 2026-11-11: absence or a foreign record passing as agreement.
    "isolation block empty": lambda r: edit_json(r / ART / "build_manifest.json", antigenic_specificity_exclusion={}),
    "no run id anywhere": lambda r: (edit_json(r / ART / "build_manifest.json", run_id=None),
                                     edit_json(r / "leakage-audit.json", run_id=None)),
    "V11 of another version": lambda r: edit_json(r / ART / "v11_check.json", version="proportional"),
    "V11 names no version": lambda r: edit_json(r / ART / "v11_check.json", version=None),
    "audit against no held-out peptide": lambda r: edit_json(r / "leakage-audit.json", distinct_heldout_peptides=0),
    "audit with an empty part": lambda r: edit_json(r / "leakage-audit.json", heldout_parts_without_a_peptide=["val"]),
    "audit names no benchmark files": lambda r: edit_json(r / "leakage-audit.json", benchmark_split_sha256=None),
    "a split has no data": lambda r: (r / "dataset/test/data-00000-of-00001.arrow").unlink(),
    "a data file is empty": lambda r: (r / "dataset/validation/data-00000-of-00001.arrow").write_bytes(b""),
    "dataset_dict missing": lambda r: (r / "dataset/dataset_dict.json").unlink(),
    "a symbolic link": lambda r: (r / "dataset/train/linked.arrow").symlink_to(r / "coverage-by-split.csv"),
}


def test_the_audit_must_have_read_the_expected_benchmark_files(tmp_path):
    root = make(tmp_path, publish=False)
    assert corpus_problems(root, require_marker=False, benchmark_split_sha256=BENCH) == []
    assert corpus_problems(root, require_marker=False, benchmark_split_sha256={"as_paired_i_val.parquet": "ee"})


@pytest.mark.parametrize("name", sorted(BREAKS))
def test_each_missing_or_inconsistent_piece_refuses(tmp_path, name):
    root = make(tmp_path, publish=False)
    BREAKS[name](root)
    assert corpus_problems(root, require_marker=False), name


def test_the_marker_pins_every_file(tmp_path):
    root = make(tmp_path)
    data = root / "dataset" / "train" / "data-00000-of-00001.arrow"
    data.write_bytes(b"rowz")                                     # same size, other content
    assert any("differs" in p for p in corpus_problems(root))
    data.write_bytes(b"rows")
    assert corpus_problems(root) == []
    (root / "dataset" / "train" / "extra.arrow").write_bytes(b"x")
    assert any("not listed" in p for p in corpus_problems(root))
    (root / "dataset" / "train" / "extra.arrow").unlink()
    data.unlink()
    assert any("missing" in p for p in corpus_problems(root))
