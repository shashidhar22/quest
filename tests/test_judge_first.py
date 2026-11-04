"""The judge-first run must stay the cheap half, and must refuse before the paid step.

Decision 2 of 2026-11-04. V04 as specified bought six CPT re-baselines and four judge cells
before any correct comparison had ever been produced. `judge_first.sh` runs only what answers
the first missing comparison: re-score the existing base SFT checkpoint under the new draw,
gate on it, then train and evaluate ONE paired legacy arm-D cell, then compare.

The property that makes it cheap is the order: the gate reads the cheap half and decides whether
to buy the expensive half. These tests pin that, the ceiling, and the scope boundary — no CPT
training may happen here under any branch.
"""
import json
import signal
import time

import pytest

from conftest import _alive


def test_the_happy_path_produces_a_comparison(jfbox):
    r = jfbox.run()
    assert r.returncode == 0, jfbox.run_log()[-3000:] + r.stderr[-1000:]
    out = jfbox.out / "comparison.json"
    assert out.exists(), "the run completed without producing a comparison"
    res = json.loads(out.read_text())
    assert res["primary"]["condition"] == "test_novel_pep"
    assert res["primary"]["unit"] == "peptide"
    assert res["primary"]["metric"] == "norm_rank"
    log = jfbox.run_log()
    assert "JUDGE-FIRST complete" in log
    assert "LEGACY Stage-4.1 arm-D adapter" in log, (
        "the scope caveat must travel with the result"
    )


def test_no_cpt_training_happens_under_any_branch(jfbox):
    """The whole point of decision 2: this run evaluates an existing adapter.

    If a CPT trainer ever appears here, the run has stopped being the cheap half.
    """
    jfbox.run()
    events = jfbox.event_lines()
    assert not [l for l in events if "armA_seed" in l or "armD_seed" in l], (
        f"a CPT re-baseline was started by the judge-first run: {events}"
    )
    assert not list((jfbox.root / "checkpoints").glob("v04_arm*_seed*")), (
        "the judge-first run created CPT checkpoint directories"
    )


def test_a_gate_refusal_stops_before_the_paid_step(jfbox):
    """The cheap half declining to buy the expensive half is the design, and exits 0."""
    r = jfbox.run(RHO_FLOOR="0.7")
    log = jfbox.run_log()
    assert r.returncode == 0, f"a pre-agreed refusal was reported as a failure: {log[-2000:]}"
    assert "G1 REFUSED" in log
    assert "NOT being trained" in log
    assert not (jfbox.out / "armD_paired" / "best.pt").exists(), (
        "the arm D cell was trained despite the refusal"
    )
    assert not (jfbox.out / "comparison.json").exists()
    assert (jfbox.out / "base_paired" / "eval.json").exists(), (
        "the base re-score is the one thing a refused run should still leave behind"
    )


def test_a_gate_technical_error_stops_before_the_paid_step(jfbox):
    """Missing evidence is not a refusal, and must not read as one."""
    ranksless = jfbox.root / "tests" / "fixtures" / "s42_eval_base_paired_20261008.json"
    r = jfbox.run(STUB_BASE_EVAL_FIXTURE=str(ranksless))
    log = jfbox.run_log()
    assert r.returncode != 0, "the run proceeded although the gate could not read its evidence"
    assert "TECHNICAL error" in log, log[-2000:]
    assert not (jfbox.out / "armD_paired" / "best.pt").exists()


def test_a_failed_probe_stops_everything(jfbox):
    """On 2026-10-07 every candidate tied and a cell burned 18 GPU-hours before anyone noticed."""
    r = jfbox.run(STUB_PROBE_RC="1")
    log = jfbox.run_log()
    assert r.returncode != 0
    assert "probe failed" in log
    assert not (jfbox.out / "base_paired" / "eval.json").exists(), (
        "the base re-score ran after the probe said the scorer ignores its input"
    )


def test_a_failed_sft_training_fails_the_run(jfbox):
    r = jfbox.run(STUB_SFT_TRAIN_RC="9")
    assert r.returncode != 0
    log = jfbox.run_log()
    assert "failed to train" in log
    assert "JUDGE-FIRST complete" not in log
    assert not (jfbox.out / "comparison.json").exists()


def test_a_missing_adapter_is_caught_by_preflight(jfbox):
    """Cheap checks happen before the GPU, not after it has been paid for."""
    r = jfbox.run(ADAPTER_D=str(jfbox.root / "checkpoints" / "no_such_adapter"))
    assert r.returncode == 2
    log = jfbox.run_log()
    assert "arm D adapter not found" in log
    assert "nothing was started" in log
    assert not jfbox.event_lines(), "a GPU step ran after a failed preflight"


def test_an_existing_cell_eval_is_not_paid_for_twice(jfbox):
    """A resumed run must not retrain a cell it already has."""
    cell = jfbox.out / "armD_paired"
    cell.mkdir(parents=True)
    ranks = (jfbox.root / "tests" / "fixtures" /
             "s42_eval_base_paired_ranks_20261008.json").read_text()
    (cell / "eval.json").write_text(ranks)
    r = jfbox.run()
    assert r.returncode == 0, jfbox.run_log()[-2000:]
    assert "already has an eval, skipping" in jfbox.run_log()
    assert not [l for l in jfbox.event_lines() if "sft_armD_paired" in l], (
        "the cell was retrained although its eval already existed"
    )


def test_the_hour_ceiling_binds_before_a_gpu_step(jfbox):
    """MAX_HOURS must stop the run between steps, not only at the end.

    MAX_HOURS=0 is the degenerate ceiling: the first check already exceeds it, so nothing
    should be spent at all.
    """
    r = jfbox.run(MAX_HOURS="0")
    log = jfbox.run_log()
    assert r.returncode != 0
    assert "ceiling reached" in log
    assert not jfbox.event_lines(), "a GPU step ran past the ceiling"


def test_sigterm_during_sft_training_leaves_no_live_child(jfbox):
    """Same lifecycle contract as run_v04.sh: the signal must reach the real process."""
    p, pids = jfbox.popen_until_started(1, STUB_SFT_TRAIN_SLEEP="60")
    p.send_signal(signal.SIGTERM)
    rc = p.wait(timeout=60)
    time.sleep(0.5)
    live = [q for q in pids if _alive(q)]
    assert rc != 0
    assert not live, f"these children outlived the signalled run: {live}"


def test_the_run_records_its_own_pid_for_the_watchdog(jfbox):
    jfbox.run()
    pid_file = jfbox.out / "judge_first.pid"
    assert pid_file.exists(), "the watchdog has no pid to wait on"
    assert pid_file.read_text().strip().isdigit()
