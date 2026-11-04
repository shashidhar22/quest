"""The Stage 4.2 gate must separate three outcomes, and never pass on absent evidence.

Fixtures are the real wave-1 base cells:
  s42_eval_bf16_tied_20261007.json   — the bf16 failure: every candidate tied
  s42_eval_base_paired_20261008.json — the working judge, NO per-query ranks
  s42_eval_base_paired_ranks_20261008.json — the same cell with ranks

Two regressions these tests exist to stop:

1. The 2026-10-08 gate read the working cell, printed "the harness is broken", and skipped
   cells 2-4. Underpowered is not broken.
2. The 2026-11-04 audit found the opposite failure: with `--units` passed but no ranks in
   the eval, the readability gate printed SKIPPED and returned **0**, so the whole V04
   chain passed on absent evidence.

Hence three outcomes, not two (`s42_check.EXIT_*`):
  0 proceed · 1 scientific refusal, a spending decision · 2 technical error, a software or
  evidence fault that must stop the visit rather than read as a refusal.

The verdict rename is deliberate. A CI at or below chance used to be called BROKEN; a
working model can perform below chance, especially under a changed distractor draw, so
that is a scientific result. Only a direct software signature — every candidate scored
alike, so MRR is exactly 1/pool — is called a fault.
"""
import csv
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
RANKED = FIX / "s42_eval_base_paired_ranks_20261008.json"
TIED = FIX / "s42_eval_bf16_tied_20261007.json"
UNITS = FIX / "s42_as_paired_i_units.csv"


def run(path, condition="test_level4", pool_size=100):
    """Harness check only, as the 2026-10 gate was invoked."""
    return s42_check.gate(types.SimpleNamespace(
        eval=str(path), condition=condition, pool_size=pool_size))


def run_full(path, *, units=UNITS, condition="test_level4", pool_size=100, rho_floor=0.8,
             prereg_delta=0.05, primary_condition="test_novel_pep",
             primary_unit="peptide", primary_metric="norm_rank"):
    """The full locked read, as run_v04.sh invokes it."""
    return s42_check.gate(types.SimpleNamespace(
        eval=str(path), condition=condition, pool_size=pool_size,
        units=str(units) if units else None, prereg_delta=prereg_delta,
        rho_floor=rho_floor, primary_condition=primary_condition,
        primary_unit=primary_unit, primary_metric=primary_metric))


def verdict(capsys):
    line = [l for l in capsys.readouterr().out.splitlines() if l.startswith("[gate] ")][0]
    return line.split("[gate] ")[1].split(":")[0]


def edited(tmp_path, src, condition=None, top=None, **fields):
    """A fixture copy with fields overridden, for the cases no real cell provides."""
    d = json.loads(src.read_text())
    if fields:
        d["conditions"][condition or "test_level4"].update(fields)
    if top:
        d.update(top)
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(d))
    return p


# ------------------------------------------------------- the 2026-10-08 regression
def test_working_judge_passes_on_the_harness_condition(capsys):
    """level4 recall 0.337, CI [0.267, 0.406] — the gate must not stop the box."""
    rc = run(WORKING)
    assert verdict(capsys) == "PASS"
    assert rc == s42_check.EXIT_OK


def test_working_judge_is_underpowered_not_broken_on_test_iid(capsys):
    """The exact 2026-10-08 regression: iid 0.144 on 201 molecules is unresolvable."""
    rc = run(WORKING, condition="test_iid")
    assert verdict(capsys) == "UNDERPOWERED"
    assert rc == s42_check.EXIT_OK, "an underpowered condition must never stop the box"


def test_gate_reports_the_other_conditions_for_context(capsys):
    """A reader of the log must see level4 beside iid, so the 10-08 misreading is visible."""
    run(WORKING)
    out = capsys.readouterr().out
    assert "context test_iid" in out and "context test_novel_pep" in out


# ------------------------------------------------------- software faults are code 2
def test_tied_scorer_is_a_technical_error(capsys):
    """The bf16 failure must still be caught, and as a fault rather than a refusal."""
    rc = run(TIED)
    assert verdict(capsys) == "TIED"
    assert rc == s42_check.EXIT_ERROR


def test_tie_is_caught_by_mrr_even_if_recall_is_nonzero(tmp_path, capsys):
    """MRR == 1/pool is the tie's fingerprint; recall can be nonzero by luck."""
    p = edited(tmp_path, TIED, recall_at_10=0.11, recall_at_10_ci95=[0.105, 0.115], mrr=0.01)
    rc = run(p)
    assert verdict(capsys) == "TIED"
    assert rc == s42_check.EXIT_ERROR


def test_zero_recall_alone_is_not_called_a_tie(tmp_path, capsys):
    """Missing every top-ten target with varying scores is a result, not a software fault.

    The old check read recall == 0 as proof every candidate tied. A model can score every
    candidate differently and still never place the true peptide in the top ten.
    """
    p = edited(tmp_path, WORKING, recall_at_10=0.0, recall_at_10_ci95=[0.0, 0.02], mrr=0.031)
    rc = run(p)
    assert verdict(capsys) != "TIED", "zero recall was mistaken for a tie"
    assert rc != s42_check.EXIT_ERROR, "a scientific result was reported as a software fault"


def test_unreadable_eval_is_a_technical_error(tmp_path, capsys):
    rc = run(tmp_path / "does_not_exist.json")
    assert rc == s42_check.EXIT_ERROR
    assert "ERROR" in capsys.readouterr().out


def test_missing_recall_or_ci_is_a_technical_error(tmp_path, capsys):
    p = edited(tmp_path, WORKING, recall_at_10=None, recall_at_10_ci95=None)
    assert run(p) == s42_check.EXIT_ERROR
    assert "ERROR" in capsys.readouterr().out


def test_nonfinite_statistic_is_a_technical_error(tmp_path, capsys):
    p = edited(tmp_path, WORKING, recall_at_10=float("nan"))
    assert run(p) == s42_check.EXIT_ERROR
    assert "ERROR" in capsys.readouterr().out


def test_pool_size_disagreement_is_a_technical_error(tmp_path, capsys):
    """Random Recall@10 is k/pool, so the gate may not score against a pool the eval did not use."""
    assert run(WORKING, pool_size=50) == s42_check.EXIT_ERROR
    assert "pool" in capsys.readouterr().out.lower()


def test_eval_without_a_recorded_pool_size_is_a_technical_error(tmp_path, capsys):
    p = edited(tmp_path, WORKING, top={"pool_size": None})
    assert run(p) == s42_check.EXIT_ERROR
    assert "pool" in capsys.readouterr().out.lower()


# ------------------------------------------------------- below chance is science, not a fault
def test_below_random_is_a_scientific_refusal_not_a_software_fault(tmp_path, capsys):
    """novel_pep sits below random; on the harness condition that stops the chain.

    It still stops the chain, but as a refusal: a working model may sit below chance, so
    calling it BROKEN licensed a false conclusion about the software.
    """
    d = json.loads(WORKING.read_text())
    d["conditions"]["test_level4"] = dict(d["conditions"]["test_novel_pep"], mrr=0.0449)
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(d))
    rc = run(p)
    assert rc == s42_check.EXIT_REFUSED
    assert verdict(capsys) == "BELOW-CHANCE"


# ------------------------------------------------------- the 2026-11-04 fail-open
def test_requested_readability_without_ranks_is_a_technical_error(capsys):
    """THE audit finding: --units passed, no ranks in the eval, gate returned 0.

    A requested read that cannot be performed is missing evidence, so it must not read as
    an affordability refusal and must not pass.
    """
    rc = run_full(WORKING)
    out = capsys.readouterr().out
    assert rc == s42_check.EXIT_ERROR, "the gate passed with no ranks to read"
    assert "ERROR" in out


def test_affordable_locked_read_proceeds(capsys):
    """The real ranks at floor 0.80 clear the measured breakeven 0.765."""
    rc = run_full(RANKED, rho_floor=0.8)
    assert rc == s42_check.EXIT_OK, capsys.readouterr().out[-2000:]
    assert "READABILITY PASS" in capsys.readouterr().out


def test_unaffordable_locked_read_is_a_scientific_refusal(capsys):
    """Below the breakeven the read is a spending decision, not a fault."""
    rc = run_full(RANKED, rho_floor=0.7)
    assert rc == s42_check.EXIT_REFUSED
    assert "READABILITY FAIL" in capsys.readouterr().out


def test_incomplete_unit_map_is_a_technical_error(tmp_path, capsys):
    """Ranks absent from the unit map were dropped silently, shrinking the read's support."""
    rows = list(csv.DictReader(UNITS.open()))
    keep = [r for r in rows if r["condition"] != "test_novel_pep"] + \
           [r for r in rows if r["condition"] == "test_novel_pep"][:5]
    p = tmp_path / "units.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(keep)
    rc = run_full(RANKED, units=p)
    out = capsys.readouterr().out
    assert rc == s42_check.EXIT_ERROR, "an incomplete unit map passed silently"
    assert "coverage" in out.lower() or "unit map" in out.lower()


def test_the_three_exit_codes_are_distinct():
    """A caller has to be able to branch on them; the launcher does."""
    codes = {s42_check.EXIT_OK, s42_check.EXIT_REFUSED, s42_check.EXIT_ERROR}
    assert codes == {0, 1, 2}
