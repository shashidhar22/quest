"""A manifest must say what a run WAS, and absence must never read as agreement.

Audit finding 1, 2026-11-06. `check` compared `a.get(k) == b.get(k)`, so two manifests agreeing
on absence were "comparable": `check({}, {})` returned no offending keys. Combined with the
launcher overwriting a manifest before deciding whether to reuse an evaluation, that is how the
reviewer obtained a `comparison.json` from a planted stale eval, no checkpoint and no training.

The repair has three parts, all here:
  1. required fields, so a key that is absent on both sides refuses rather than matching;
  2. separate schemas, so a CPT run and an SFT judge cell are not described by one overloaded
     key set in which half the fields are the string "n/a";
  3. separate CPT / SFT / eval lineage, so the base arm is not labelled with a CPT objective it
     never had -- and neither is arm D, whose own artifact records the fixed-batch path.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROV = REPO / "scripts" / "training" / "v04_box" / "provenance.py"
sys.path.insert(0, str(PROV.parent))

import provenance as P  # noqa: E402


def sft(**over):
    """A complete SFT-cell manifest, which `over` then damages in one specific way."""
    m = {
        "manifest_kind": "sft_cell", "manifest_version": P.MANIFEST_VERSION,
        "task": "as_paired_i", "dataset": "as_paired_i", "draw": "length_matched",
        "pool_size": 100, "pool_digest": "aaaaaaaaaaaaaaaa",
        "split_digests": {"test_novel_pep": "bbbbbbbbbbbbbbbb"},
        "sft_ckpt_digest": "cccccccccccccccc", "sft_seed": 42, "sft_objective": "bce_pairs/v1",
        "cpt_adapter": "none", "cpt_adapter_digest": "none", "cpt_objective": "none",
        "cpt_run_id": "none", "cpt_steps": None, "cpt_seed": None, "cpt_token_budget": None,
        "device": "cuda", "code_commit": "abcdef123456", "cotenants": "none",
    }
    m.update(over)
    return m


def armd(**over):
    return sft(cpt_adapter="checkpoints/cpt_stage41_pointD_run05/final",
               cpt_adapter_digest="3afeaf40bc860fef", cpt_objective="fixed-batch-mean/v0",
               cpt_run_id="cpt_stage41_pointD_run05", cpt_steps=7800, cpt_seed=42,
               cpt_token_budget=0, sft_ckpt_digest="dddddddddddddddd", **over)


# ------------------------------------------------------------------ absence is not agreement
def test_two_empty_manifests_are_not_comparable():
    """The auditor's first probe line. It used to return ([], []) -- fully comparable."""
    bad, tolerated = P.check({}, {})
    assert bad, "two empty manifests were accepted as comparable"
    assert any("manifest_kind" in b for b in bad)


def test_a_field_missing_from_both_sides_refuses():
    a, b = sft(), sft()
    del a["pool_digest"], b["pool_digest"]
    bad, _ = P.check(a, b, "cells")
    assert any("pool_digest" in x and "missing" in x for x in bad), bad


def test_a_null_required_field_refuses():
    bad, _ = P.check(sft(pool_digest=None), sft(), "cells")
    assert any("pool_digest" in x for x in bad), bad
    bad, _ = P.check(sft(pool_digest=""), sft(), "cells")
    assert any("pool_digest" in x for x in bad), bad


def test_a_missing_pool_or_split_digest_refuses():
    """'MISSING' is what the hasher records for a file that is not there. It is a valid
    recording and an invalid basis for a verdict."""
    bad, _ = P.check(sft(pool_digest="MISSING"), sft(pool_digest="MISSING"), "cells")
    assert any("pool_digest" in x for x in bad), bad


def test_an_optional_field_may_be_null_because_null_is_a_fact():
    """A base arm has no CPT steps. Recording null is honest; demanding a number would force
    the invention the whole repair exists to stop.

    Compared under `strict`, i.e. against an identical manifest: `cells` would (rightly) refuse
    a base arm against itself as a self-comparison, which is a different rule, tested below.
    """
    assert sft()["cpt_steps"] is None
    bad, _ = P.check(sft(), sft(), "strict")
    assert bad == [], bad
    bad, _ = P.check(sft(), armd(), "cells")
    assert bad == [], bad


# ------------------------------------------------------------------ schemas
def test_a_cpt_run_cannot_be_compared_with_an_sft_cell():
    a = {"manifest_kind": "cpt_run", "manifest_version": P.MANIFEST_VERSION}
    bad, _ = P.check(a, sft(), "cells")
    assert any("manifest_kind" in x for x in bad), bad


def test_the_legacy_cpt_run_schema_still_validates():
    """run_v04.sh writes this shape; the repair must not break the CPT comparison path."""
    m = {k: "x" for k in P.SCHEMAS["cpt_run"]}
    m.update(manifest_kind="cpt_run", manifest_version=P.MANIFEST_VERSION)
    assert P.check(m, dict(m), "strict") == ([], [])
    gone = dict(m)
    del gone["steps"]
    bad, _ = P.check(gone, dict(m), "strict")
    assert any("steps" in x for x in bad), bad


# ------------------------------------------------------------------ treatment vs confound
def test_the_cpt_difference_is_tolerated_and_printed_not_silent():
    bad, tolerated = P.check(sft(), armd(), "cells")
    assert bad == [], bad
    assert any("cpt_adapter_digest" in t for t in tolerated)
    assert any("cpt_objective" in t for t in tolerated), (
        "the one difference the comparison is ABOUT must appear beside its result")


def test_a_changed_pool_is_not_tolerated():
    bad, _ = P.check(sft(), armd(pool_digest="ffffffffffffffff"), "cells")
    assert any("pool_digest" in x for x in bad), bad


def test_a_changed_split_is_not_tolerated():
    bad, _ = P.check(sft(), armd(split_digests={"test_novel_pep": "zzzzzzzzzzzzzzzz"}), "cells")
    assert any("split_digests" in x for x in bad), bad


def test_a_changed_draw_or_pool_size_is_not_tolerated():
    for over in ({"draw": "uniform"}, {"pool_size": 50}):
        bad, _ = P.check(sft(), armd(**over), "cells")
        assert bad, over


def test_a_changed_commit_is_not_tolerated_silently():
    bad, _ = P.check(sft(), armd(code_commit="999999999999"), "cells")
    assert any("code_commit" in x for x in bad), (
        "two arms scored by different code are not a paired comparison")


def test_comparing_an_arm_with_itself_refuses():
    """The stale-reuse shape: if neither the adapter nor the SFT checkpoint differs, the two
    'arms' are one model and any effect is noise. Nothing in the old check could see it."""
    bad, _ = P.check(armd(), armd(), "cells")
    assert any("same" in x.lower() for x in bad), bad
    # and the control: a real pair is accepted
    assert P.check(sft(), armd(), "cells")[0] == []


# ------------------------------------------------------------------ derived from the eval
def test_write_from_eval_takes_the_artifact_fields_from_the_eval_itself(tmp_path):
    ev = tmp_path / "eval.json"
    ev.write_text(json.dumps({
        "args": {"task": "as_paired_i", "distractor_draw": "length_matched",
                 "adapter": "none", "seed": 42},
        "pool_size": 100,
        "conditions": {"test_novel_pep": {"ranks": {"r1": 1}}},
        "provenance": {
            "pool_digest": "1111111111111111", "pool_path": "/x/pool.tsv",
            "split_digests": {"test_novel_pep": "2222222222222222"},
            "ckpt_digest": "3333333333333333", "device": "cuda",
            "code_commit": "abcdef123456", "cpt_adapter": "none",
            "cpt_adapter_digest": "none", "cpt_objective": "none", "cpt_run_id": "none",
            "cpt_steps": None, "cpt_seed": None, "cpt_token_budget": None,
        }}))
    out = tmp_path / "manifest.json"
    r = subprocess.run([sys.executable, str(PROV), "write", "--out", str(out),
                        "--from-eval", str(ev), "--cotenants", "none",
                        "--sft-objective", "bce_pairs/v1"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    m = json.loads(out.read_text())
    assert m["pool_digest"] == "1111111111111111", "the launcher's own idea of the pool won"
    assert m["sft_ckpt_digest"] == "3333333333333333"
    assert m["split_digests"] == {"test_novel_pep": "2222222222222222"}
    assert m["cpt_objective"] == "none", "a base arm must not be labelled with a CPT objective"
    assert m["manifest_kind"] == "sft_cell"
    assert P.check(m, dict(m), "strict") == ([], [])


def test_write_from_an_eval_with_no_provenance_block_refuses(tmp_path):
    """An evaluation produced before the evaluator recorded provenance cannot be certified.
    This is the stale-artifact case: the 2026-10-08 ranks file is exactly such a file."""
    ev = tmp_path / "eval.json"
    ev.write_text(json.dumps({"args": {"task": "as_paired_i"}, "pool_size": 100,
                              "conditions": {}}))
    r = subprocess.run([sys.executable, str(PROV), "write", "--out", str(tmp_path / "m.json"),
                        "--from-eval", str(ev)], capture_output=True, text=True)
    assert r.returncode != 0
    assert "provenance" in (r.stdout + r.stderr)
    assert not (tmp_path / "m.json").exists(), "a manifest was written for an uncertifiable eval"


def test_write_refuses_an_incomplete_manifest_at_write_time(tmp_path):
    """Catching it at check time is too late: the GPU work is already paid for."""
    out = tmp_path / "m.json"
    r = subprocess.run([sys.executable, str(PROV), "write", "--out", str(out),
                        "--manifest-kind", "sft_cell", "--task", "as_paired_i"],
                       capture_output=True, text=True)
    assert r.returncode != 0, r.stdout
    assert not out.exists()


# ------------------------------------------------------------------ the CLI
def test_the_check_cli_prints_every_tolerated_difference(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(sft()))
    b.write_text(json.dumps(armd()))
    r = subprocess.run([sys.executable, str(PROV), "check", "--a", str(a), "--b", str(b),
                        "--kind", "cells"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "TOLERATED" in r.stdout
    assert "cpt_objective" in r.stdout


def test_the_check_cli_refuses_two_empty_manifests(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text("{}")
    b.write_text("{}")
    r = subprocess.run([sys.executable, str(PROV), "check", "--a", str(a), "--b", str(b),
                        "--kind", "cells"], capture_output=True, text=True)
    assert r.returncode == 1, r.stdout
    assert "REFUSED" in r.stdout


def test_the_pool_fingerprint_follows_the_loader(monkeypatch, tmp_path):
    """Hashing a path the launcher names, while the evaluator reads a different one, is the
    disconnect the audit found."""
    monkeypatch.setenv("QUEST_SPLITS_DIR", str(tmp_path))
    (tmp_path / "as_class_i_candidate_pool.tsv").write_text("peptide\nAAA\n")
    r = subprocess.run([sys.executable, str(PROV), "fingerprint", "--pool"],
                       capture_output=True, text=True, cwd=REPO,
                       env={**dict(__import__("os").environ), "QUEST_SPLITS_DIR": str(tmp_path)})
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip() != "MISSING"
