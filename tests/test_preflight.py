"""A preflight must prove usability, not existence.

Audit finding 5, 2026-11-06. The judge-first P0 tested `-f` and `-d`. It passed on a box with no
CUDA driver (the evaluator then fell back to the CPU silently), with peft absent, with an adapter
directory containing no adapter, and with the train/val splits missing -- which surfaced only
after the PAID base evaluation.

These tests run the real checks against real files. The GPU is absent on this laptop, which is
what makes the CUDA requirement falsifiable here: `torch.cuda.is_available()` is False, so a
preflight that demanded CUDA and passed would be broken.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PF = REPO / "scripts" / "training" / "v04_box" / "preflight.py"
sys.path.insert(0, str(PF.parent))
sys.path.insert(0, str(REPO))

import preflight as P  # noqa: E402

ARM_D = REPO / "checkpoints" / "cpt_stage41_pointD_run05" / "final"


def run(*argv, cwd=REPO, env=None):
    import os
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run([sys.executable, str(PF), *argv],
                          capture_output=True, text=True, cwd=cwd, env=e)


@pytest.fixture
def splits(tmp_path, monkeypatch):
    """A splits directory with every required file, which a test then damages."""
    d = tmp_path / "splits"
    d.mkdir()
    (d / "as_class_i_candidate_pool.tsv").write_text("peptide\nAAAAAAAAA\n")
    for c in ("train", "val", "test_level4", "test_novel_pep", "test_iid"):
        (d / f"as_paired_i_{c}.parquet").write_text(f"stand-in {c}\n")
    monkeypatch.setenv("QUEST_SPLITS_DIR", str(d))
    return d


# ----------------------------------------------------------------- the splits
def test_a_missing_split_fails_before_anything_is_paid_for(splits):
    (splits / "as_paired_i_train.parquet").unlink()
    r = run("--require", "", env={"QUEST_SPLITS_DIR": str(splits)})
    assert r.returncode == P.ERROR
    assert "train" in r.stderr and "split files missing" in r.stderr
    assert "after the base evaluation is paid for" in r.stderr, (
        "the refusal should say why this check belongs BEFORE the paid step")


def test_a_complete_splits_directory_passes(splits):
    r = run("--require", "", env={"QUEST_SPLITS_DIR": str(splits)})
    assert r.returncode == P.OK, r.stderr
    rep = json.loads(r.stdout)
    assert rep["splits"]["missing"] == []
    assert rep["pool"]["digest"] not in ("MISSING", None)
    assert set(rep["splits"]["digests"]) == {"train", "val", "test_level4",
                                            "test_novel_pep", "test_iid"}


def test_a_missing_pool_fails(splits):
    (splits / "as_class_i_candidate_pool.tsv").unlink()
    r = run("--require", "", env={"QUEST_SPLITS_DIR": str(splits)})
    assert r.returncode == P.ERROR
    assert "candidate pool" in r.stderr


# ----------------------------------------------------------------- CUDA
def test_requiring_cuda_fails_on_a_box_without_it(splits):
    """Falsifiable here precisely because this laptop has no CUDA."""
    r = run("--require", "", "--require-cuda", env={"QUEST_SPLITS_DIR": str(splits)})
    assert r.returncode == P.ERROR
    assert "CUDA" in r.stderr
    assert "nobody chose to buy" in r.stderr


def test_not_requiring_cuda_still_reports_what_it_found(splits):
    r = run("--require", "", env={"QUEST_SPLITS_DIR": str(splits)})
    assert r.returncode == P.OK
    cuda = json.loads(r.stdout)["cuda"]
    assert "available" in cuda
    assert cuda["available"] is False or "devices" in cuda
    if not cuda["available"]:
        assert cuda["why"], "an absent GPU must come with a reason, not just a false"


def test_the_device_choice_refuses_a_silent_cpu_fallback():
    from track_a.esm_sft.scoring import DeviceError, select_device

    assert select_device(True) == "cuda"
    assert select_device(False, allow_cpu=True) == "cpu"
    with pytest.raises(DeviceError, match="silent fallback"):
        select_device(False)


# ----------------------------------------------------------------- imports
def test_an_unimportable_requirement_is_named(splits):
    r = run("--require", "definitely_not_a_module_qqq",
            env={"QUEST_SPLITS_DIR": str(splits)})
    assert r.returncode == P.ERROR
    assert "definitely_not_a_module_qqq" in r.stderr


def test_an_importable_requirement_reports_its_version(splits):
    r = run("--require", "json,pathlib", env={"QUEST_SPLITS_DIR": str(splits)})
    assert r.returncode == P.OK, r.stderr
    assert json.loads(r.stdout)["imports"]["json"]


# ----------------------------------------------------------------- the adapter
def test_a_directory_that_merely_exists_is_not_an_adapter(tmp_path, splits):
    empty = tmp_path / "final"
    empty.mkdir()
    r = run("--require", "", "--adapter", str(empty), env={"QUEST_SPLITS_DIR": str(splits)})
    assert r.returncode == P.ERROR
    assert "adapter_model.safetensors" in r.stderr, (
        "an adapter directory with no adapter in it would load as a fresh LoRA and be "
        "evaluated as the base arm")


@pytest.mark.skipif(not ARM_D.is_dir(), reason="the legacy arm-D adapter is not in this checkout")
def test_the_real_arm_d_adapter_passes_and_reports_its_lineage(splits):
    r = run("--require", "", "--adapter", str(ARM_D), env={"QUEST_SPLITS_DIR": str(splits)})
    assert r.returncode == P.OK, r.stderr
    ad = json.loads(r.stdout)["adapter"]
    assert ad["ok"] and ad["missing"] == []
    assert ad["cpt_run_id"] == "cpt_stage41_pointD_run05"
    assert ad["cpt_objective"] == "fixed-batch-mean/v0"


def test_the_base_arm_needs_no_adapter(splits):
    r = run("--require", "", "--adapter", "none", env={"QUEST_SPLITS_DIR": str(splits)})
    assert r.returncode == P.OK, r.stderr
    assert json.loads(r.stdout)["adapter"]["adapter"] == "none"


# ----------------------------------------------------------------- the checkpoint
def test_an_empty_checkpoint_file_fails(tmp_path, splits):
    ck = tmp_path / "best.pt"
    ck.touch()
    r = run("--require", "", "--ckpt", str(ck), env={"QUEST_SPLITS_DIR": str(splits)})
    assert r.returncode == P.ERROR
    assert "zero bytes" in r.stderr


def test_a_missing_checkpoint_fails(tmp_path, splits):
    r = run("--require", "", "--ckpt", str(tmp_path / "nope.pt"),
            env={"QUEST_SPLITS_DIR": str(splits)})
    assert r.returncode == P.ERROR
    assert "missing or empty" in r.stderr


def test_a_checkpoint_with_no_trained_weights_fails(tmp_path, splits):
    """A real torch file that restores nothing. The old check saw a regular file and passed."""
    torch = pytest.importorskip("torch")
    ck = tmp_path / "best.pt"
    torch.save({"args": {"task": "as_paired_i"}, "trainable_state_dict": {}}, ck)
    r = run("--require", "", "--ckpt", str(ck), env={"QUEST_SPLITS_DIR": str(splits)})
    assert r.returncode == P.ERROR
    assert "no trainable_state_dict" in r.stderr


def test_a_real_checkpoint_reports_what_it_was_trained_with(tmp_path, splits):
    torch = pytest.importorskip("torch")
    ck = tmp_path / "best.pt"
    torch.save({"args": {"task": "as_paired_i", "tcr_variant": "cdr3", "mhc_variant": "full",
                         "adapter": "none", "seed": 42},
                "trainable_state_dict": {"pool.q.weight": torch.zeros(2, 2),
                                         "head.0.bias": torch.zeros(2)}}, ck)
    r = run("--require", "", "--ckpt", str(ck), env={"QUEST_SPLITS_DIR": str(splits)})
    assert r.returncode == P.OK, r.stderr
    rep = json.loads(r.stdout)["checkpoint"]
    assert rep["n_trainable_keys"] == 2
    assert rep["trainable_key_prefixes"] == ["head", "pool"]
    assert rep["recorded_args"]["task"] == "as_paired_i"
    assert rep["digest"] not in ("MISSING", None)


# ----------------------------------------------------------------- the report
def test_the_report_is_written_and_carries_the_commit(tmp_path, splits):
    out = tmp_path / "preflight.json"
    r = run("--require", "", "--out", str(out), env={"QUEST_SPLITS_DIR": str(splits)})
    assert r.returncode == P.OK, r.stderr
    rep = json.loads(out.read_text())
    assert rep["ok"] is True
    assert rep["commit"], "a result must carry the revision that produced it"
    assert rep["problems"] == []
