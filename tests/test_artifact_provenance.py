"""Provenance read from artifacts, not declared by whatever script is running.

Audit finding 1, 2026-11-06. Two separate defects shared one cause:

  - the launcher hashed `$POOL_FILE` while the evaluator read `load_candidate_pool()`, which
    resolved its own directory. A manifest could record the hash of a file nothing read.
  - both judge-first manifests were labelled `objective=legacy_row_share_cpt`. That is wrong
    twice: the base arm has no CPT at all, and arm D did not train under row share either.
    `checkpoints/cpt_stage41_pointD_run05/final/trainer_state.json` records `batch_size 16`,
    `gradient_accumulation_steps 16` and NO token budget, so Stage 4.1 ran the fixed-batch path,
    whose objective is the mean of equally weighted micro-batch means. Row share is the v2
    token-budget default, which that run never used.

The real adapter is the fixture wherever one is needed: it is the artifact the judge-first run
will actually read, and a synthetic stand-in would prove nothing about it.
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from track_a.shared import artifacts as A  # noqa: E402

ARM_D = REPO / "checkpoints" / "cpt_stage41_pointD_run05" / "final"


# --------------------------------------------------------------- one resolution rule
def test_the_loader_and_the_hasher_resolve_the_same_directory():
    """The defect: two copies of the rule meant the hash and the read could diverge."""
    from track_a.shared.load_splits import SPLITS_DIR
    assert str(SPLITS_DIR) == str(A.splits_dir())


def test_the_override_is_honoured(monkeypatch, tmp_path):
    d = tmp_path / "splits"
    d.mkdir()
    monkeypatch.setenv("QUEST_SPLITS_DIR", str(d))
    assert A.splits_dir() == d
    assert A.pool_path() == d / "as_class_i_candidate_pool.tsv"
    assert A.split_path("as_paired_i", "test_novel_pep") == d / "as_paired_i_test_novel_pep.parquet"


def test_a_missing_directory_still_names_a_usable_path(monkeypatch, tmp_path):
    monkeypatch.setenv("QUEST_SPLITS_DIR", str(tmp_path / "nope"))
    assert A.splits_dir() == tmp_path / "nope"


# --------------------------------------------------------------- digests
def test_a_file_digest_tracks_content(tmp_path):
    f = tmp_path / "pool.tsv"
    f.write_text("peptide\nAAA\n")
    d1 = A.file_digest(f)
    f.write_text("peptide\nAAB\n")
    assert A.file_digest(f) != d1, "a regenerated pool must not keep its hash"
    assert A.file_digest(tmp_path / "absent") == "MISSING"


def test_a_directory_digest_tracks_contents_names_and_additions(tmp_path):
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text('{"r": 32}')
    (d / "adapter_model.safetensors").write_bytes(b"weights")
    base = A.dir_digest(d)
    (d / "adapter_config.json").write_text('{"r": 64}')
    assert A.dir_digest(d) != base, "a changed LoRA shape must change the digest"
    (d / "adapter_config.json").write_text('{"r": 32}')
    assert A.dir_digest(d) == base
    (d / "extra.bin").write_bytes(b"x")
    assert A.dir_digest(d) != base, "an added file must change the digest"
    assert A.dir_digest(tmp_path / "absent") == "MISSING"


def test_split_digests_record_absence_rather_than_raising(monkeypatch, tmp_path):
    monkeypatch.setenv("QUEST_SPLITS_DIR", str(tmp_path))
    got = A.split_digests("as_paired_i", ("test_iid", "test_novel_pep"))
    assert got == {"test_iid": "MISSING", "test_novel_pep": "MISSING"}, (
        "a record of absence is still a record; the preflight decides what to do about it")


# --------------------------------------------------------------- lineage
def test_a_base_arm_has_no_cpt_lineage_at_all():
    """Not `legacy_row_share_cpt`. The base SFT checkpoint has no CPT history to label."""
    got = A.cpt_lineage("none")
    assert got["cpt_adapter"] == "none"
    assert got["cpt_objective"] == "none"
    for alias in (None, "", "base"):
        assert A.cpt_lineage(alias)["cpt_objective"] == "none"


@pytest.mark.skipif(not ARM_D.is_dir(), reason="the legacy arm-D adapter is not in this checkout")
def test_the_real_arm_d_adapter_reports_its_own_recorded_objective():
    got = A.cpt_lineage(ARM_D)
    assert got["cpt_run_id"] == "cpt_stage41_pointD_run05", (
        "`final` is a leaf, not a run id")
    assert got["cpt_adapter_digest"] not in ("none", "MISSING")
    # Read from trainer_state.json: fixed batch 16 x accumulation 16, no token budget.
    assert got["cpt_objective"] == "fixed-batch-mean/v0", got
    assert "row" not in got["cpt_objective"], (
        "arm D never trained under row share; that is the v2 token-budget default")
    assert got["cpt_steps"] == 7800, "the completed step count is part of what a run was"
    assert got["cpt_seed"] == 42
    assert got["cpt_token_budget"] == 0


def test_a_token_budget_run_is_named_by_its_weighting(tmp_path):
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "trainer_state.json").write_text(json.dumps({
        "global_step": 4000,
        "config": {"token_budget": 8192, "loss_weighting": "token", "seed": 7,
                   "batch_size": 16, "gradient_accumulation_steps": 16}}))
    got = A.cpt_lineage(d)
    assert got["cpt_objective"] == "input-length-share/v1"
    assert got["cpt_steps"] == 4000
    assert got["cpt_seed"] == 7
    (d / "trainer_state.json").write_text(json.dumps({
        "global_step": 10, "config": {"token_budget": 8192, "loss_weighting": "row"}}))
    assert A.cpt_lineage(d)["cpt_objective"] == "row-share/v1"


def test_a_silent_adapter_says_unrecorded_and_never_guesses(tmp_path):
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_model.safetensors").write_bytes(b"w")
    got = A.cpt_lineage(d)
    assert got["cpt_objective"] == "unrecorded"
    assert got["cpt_steps"] is None
    assert got["cpt_adapter_digest"] not in ("none", "MISSING")


def test_the_objective_names_agree_with_the_resume_guard():
    """Falsifiable anti-drift check: two tables naming the same objectives must not disagree.

    `resume_compat.LOSS_WEIGHTING_SEMANTICS` is what a resume refuses on. If this module named
    arm D's objective differently from the way a resume names it, a manifest and a resume could
    make opposite claims about the same checkpoint.
    """
    from quest.training.resume_compat import LOSS_WEIGHTING_SEMANTICS
    for weighting, semantics in LOSS_WEIGHTING_SEMANTICS.items():
        assert A.TOKEN_BUDGET_SEMANTICS[weighting] == semantics, weighting
    # and the fixed-batch path, which that table has no row for because it is not a weighting
    assert A.FIXED_BATCH_SEMANTICS == "fixed-batch-mean/v0"
    assert A.FIXED_BATCH_SEMANTICS not in LOSS_WEIGHTING_SEMANTICS.values()


def test_the_commit_is_recorded_with_its_dirty_state():
    got = A.code_commit(REPO)
    assert got == "unknown" or len(got.split(":")[-1]) == 12


# --------------------------------------------------------------- import boundary
def test_the_artifact_rules_import_without_pandas_or_torch():
    code = textwrap.dedent(
        r"""
        import importlib.abc
        import sys

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                for heavy in ("torch", "peft", "esm", "pandas", "numpy", "transformers"):
                    if fullname == heavy or fullname.startswith(heavy + "."):
                        raise ImportError(heavy + " is blocked for this test")
                return None

        sys.meta_path.insert(0, Block())
        from track_a.shared import artifacts
        assert artifacts.cpt_lineage("none")["cpt_objective"] == "none"
        assert artifacts.file_digest("setup.py") != "MISSING"
        print("OK")
        """
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, "the artifact rules reached the ML stack:\n" + r.stdout + r.stderr
