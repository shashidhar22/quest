"""The Stage 4.2 wave-1 gate must tell 'broken' apart from 'underpowered'.

Fixtures are the two real wave-1 base cells:
  s42_eval_bf16_tied_20261007.json  — the bf16 failure: every candidate tied, recall 0 everywhere
  s42_eval_base_paired_20261008.json — the working judge: level4 0.337, iid 0.144, novel_pep 0.081

The 2026-10-08 gate read the second file and printed "the harness is broken", then skipped cells
2-4. These tests fail if that behaviour comes back.
"""
import json
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIX = pathlib.Path(__file__).parent / "fixtures"
sys.path.insert(0, str(ROOT / "scripts" / "training" / "stage42_box"))
import s42_check  # noqa: E402

WORKING = FIX / "s42_eval_base_paired_20261008.json"
TIED = FIX / "s42_eval_bf16_tied_20261007.json"


def run(path, condition="test_level4"):
    return s42_check.gate(types.SimpleNamespace(
        eval=str(path), condition=condition, pool_size=100))


def verdict(capsys):
    line = [l for l in capsys.readouterr().out.splitlines() if l.startswith("[gate] ")][0]
    return line.split("[gate] ")[1].split(":")[0]


def test_working_judge_passes_on_the_harness_condition(capsys):
    """level4 recall 0.337, CI [0.267, 0.406] — the gate must not stop the box."""
    rc = run(WORKING)
    assert verdict(capsys) == "PASS"
    assert rc == 0


def test_working_judge_is_underpowered_not_broken_on_test_iid(capsys):
    """The exact 2026-10-08 regression: iid 0.144 on 201 molecules is unresolvable, not broken."""
    rc = run(WORKING, condition="test_iid")
    assert verdict(capsys) == "UNDERPOWERED"
    assert rc == 0, "an underpowered condition must never stop the box"


def test_tied_scorer_is_broken(capsys):
    """The bf16 failure must still be caught, or the gate is worthless."""
    rc = run(TIED)
    assert verdict(capsys) == "BROKEN"
    assert rc == 1


def test_tie_is_caught_by_mrr_even_if_recall_is_nonzero(tmp_path, capsys):
    """Recall could be nonzero by luck; MRR == 1/pool is the tie's fingerprint."""
    d = json.loads(TIED.read_text())
    d["conditions"]["test_level4"].update(
        recall_at_10=0.11, recall_at_10_ci95=[0.105, 0.115], mrr=0.01)
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(d))
    rc = run(p)
    assert verdict(capsys) == "BROKEN"
    assert rc == 1


def test_gate_reports_the_other_conditions_for_context(capsys):
    """A reader of the log must see level4 beside iid, so the 10-08 misreading is visible."""
    run(WORKING)
    out = capsys.readouterr().out
    assert "context test_iid" in out and "context test_novel_pep" in out


def test_below_random_with_a_clear_ci_is_broken(tmp_path, capsys):
    """novel_pep sits below random by construction; on the harness condition that is broken."""
    d = json.loads(WORKING.read_text())
    d["conditions"]["test_level4"] = dict(
        d["conditions"]["test_novel_pep"], mrr=0.0449)
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(d))
    assert run(p) == 1
    assert verdict(capsys) == "BROKEN"
