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
import subprocess
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


# --------------------------------------------------------------- audit 2026-11-06, finding 1
# Reuse. The old test here planted a bare eval.json and asserted that the run skipped training:
# it pinned exactly the contract the audit broke. With no checkpoint, no manifest and no record
# of what the eval read, the launcher overwrote the manifest, skipped the paid step and produced
# a `comparison.json`. Reuse is a saving only when it is the SAME result, so each refusal below
# is one of the four ways the planted scenario differs from one.
def _complete_cell(jfbox):
    """Run once to build a real, certified arm-D cell, then clear the event log."""
    r = jfbox.run()
    assert r.returncode == 0, jfbox.run_log()[-2000:]
    cell = jfbox.out / "armD_paired"
    assert (cell / "eval.json").exists() and (cell / "best.pt").exists()
    assert (cell / "manifest.json").exists()
    jfbox.events.write_text("")
    (jfbox.out / "run.log").unlink()
    return cell


def test_a_validated_cell_is_not_paid_for_twice(jfbox):
    """The saving the reuse path exists for, now conditional on the result being the same one."""
    _complete_cell(jfbox)
    r = jfbox.run()
    assert r.returncode == 0, jfbox.run_log()[-3000:]
    assert "reuse validated" in jfbox.run_log()
    assert not [l for l in jfbox.event_lines() if "sft_armD_paired" in l], (
        "the cell was retrained although a validated evaluation existed")
    assert (jfbox.out / "comparison.json").exists()


def test_an_eval_with_no_checkpoint_is_refused(jfbox):
    """The auditor's exact scenario: the planted rank fixture and nothing else."""
    cell = jfbox.out / "armD_paired"
    cell.mkdir(parents=True)
    ranks = (jfbox.root / "tests" / "fixtures" /
             "s42_eval_base_paired_ranks_20261008.json").read_text()
    (cell / "eval.json").write_text(ranks)
    (cell / "manifest.json").write_text(json.dumps(
        {"pool_id": "OLD_POOL", "seed": 999, "objective": "wrong"}))
    before = (cell / "manifest.json").read_text()
    r = jfbox.run()
    assert r.returncode != 0, jfbox.run_log()[-2000:]
    assert "Nothing may be certified from it" in jfbox.run_log(), (
        "the refusal must name the missing checkpoint as the reason")
    assert not (jfbox.out / "comparison.json").exists(), (
        "a comparison was certified from a planted evaluation")
    assert (cell / "manifest.json").read_text() == before, (
        "the stale manifest was overwritten, which is what made the planted result look current")


def test_an_eval_with_a_checkpoint_but_no_manifest_is_refused(jfbox):
    cell = _complete_cell(jfbox)
    (cell / "manifest.json").unlink()
    r = jfbox.run()
    assert r.returncode != 0
    assert "no manifest beside it" in jfbox.run_log()
    assert not [l for l in jfbox.event_lines() if "sft_armD_paired" in l]


def test_a_manifest_that_does_not_describe_its_eval_is_refused(jfbox):
    """And the manifest is not quietly replaced: which of the two is wrong is a human call."""
    cell = _complete_cell(jfbox)
    m = json.loads((cell / "manifest.json").read_text())
    m["pool_digest"] = "0000000000000000"
    (cell / "manifest.json").write_text(json.dumps(m))
    r = jfbox.run()
    assert r.returncode != 0
    assert "does not describe the" in jfbox.run_log()
    assert json.loads((cell / "manifest.json").read_text())["pool_digest"] == "0000000000000000"


def test_a_changed_pool_refuses_a_reused_eval(jfbox):
    """The inputs behind a stored result moved, so the result is no longer about them."""
    _complete_cell(jfbox)
    pool = jfbox.root / "benchmark_v2" / "splits" / "as_class_i_candidate_pool.tsv"
    pool.write_text(pool.read_text() + "PEPTIDE999\n")
    r = jfbox.run()
    assert r.returncode != 0
    assert "the candidate pool changed" in jfbox.run_log()
    assert not [l for l in jfbox.event_lines() if "sft_armD_paired" in l]


def test_a_changed_split_refuses_a_reused_eval(jfbox):
    _complete_cell(jfbox)
    jfbox.splits["test_novel_pep"].write_text("different split bytes\n")
    r = jfbox.run()
    assert r.returncode != 0
    assert "split test_novel_pep changed" in jfbox.run_log()


def test_an_eval_from_before_provenance_was_recorded_cannot_be_certified(jfbox):
    """The 2026-10-08 ranks are such a file. An honest answer is that it cannot be certified,
    not that it passes because nothing is known about it."""
    r = jfbox.run(STUB_EVAL_RAW="1")
    assert r.returncode != 0, jfbox.run_log()[-2000:]
    assert "no 'provenance' block" in jfbox.run_log()
    assert not (jfbox.out / "comparison.json").exists()


def test_the_manifests_no_longer_label_a_base_arm_with_a_cpt_objective(jfbox):
    """Both manifests used to say objective=legacy_row_share_cpt. The base arm has no CPT, and
    arm D's own trainer_state.json records the fixed-batch path, not row share."""
    r = jfbox.run()
    assert r.returncode == 0, jfbox.run_log()[-2000:]
    base = json.loads((jfbox.out / "base_paired" / "manifest.json").read_text())
    cell = json.loads((jfbox.out / "armD_paired" / "manifest.json").read_text())
    assert base["cpt_objective"] == "none", base
    assert base["cpt_adapter"] == "none"
    assert cell["cpt_run_id"] == "cpt_stage41_pointD_run05", cell
    assert cell["cpt_adapter_digest"] not in ("none", "MISSING")
    for m in (base, cell):
        assert m["sft_objective"] == "bce-pair-logits/v1"
        assert "legacy_row_share_cpt" not in json.dumps(m)
        assert m["manifest_kind"] == "sft_cell"
    log = jfbox.run_log()
    assert "TOLERATED by kind=cells: cpt_adapter_digest" in log, (
        "the difference the comparison is ABOUT must be disclosed beside the result")


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


# --------------------------------------------------------------- audit 2026-11-06, finding 3
def test_the_ceiling_terminates_a_child_that_overruns_it(jfbox):
    """The deadline must bind DURING a phase, not only between phases.

    Audit finding 3: `check_ceiling` ran between steps and `run_gpu` then blocked in `wait` on
    an unbounded child, so a phase that began at hour 7:59 could run for days without another
    check. The claim in the visit record that expiry terminates children from inside the
    launcher was simply untrue.

    Here the ceiling is ~9 seconds and the base evaluation sleeps 40. A pass requires the run
    to exit well before that sleep could end, to have killed the child, and to have written no
    evaluation artifact. If the deadline were still only checked between phases, the run would
    sit in `wait` for the full 40 seconds and this fails on the elapsed time alone.
    """
    t0 = time.time()
    r = jfbox.run(MAX_HOURS="0.0025", STUB_BASE_EVAL_SLEEP="40", POLL="1", timeout=120)
    elapsed = time.time() - t0
    log = jfbox.run_log()
    assert r.returncode != 0, log[-2000:]
    assert elapsed < 30, f"the launcher waited out its own child ({elapsed:.0f}s): {log[-2000:]}"
    assert "ceiling reached" in log, log[-2000:]
    pids = jfbox.eval_pids()
    assert pids, "the base evaluation never started, so nothing was tested"
    time.sleep(0.5)
    assert not [q for q in pids if _alive(q)], "the overrunning child outlived the ceiling"
    assert not (jfbox.out / "base_paired" / "eval.json").exists(), (
        "a phase killed at the ceiling left a success artifact behind")
    assert not (jfbox.out / "comparison.json").exists()


def test_a_child_that_finishes_inside_the_ceiling_is_left_alone(jfbox):
    """The control: a deadline that fires on a normal run would be worse than none."""
    r = jfbox.run(MAX_HOURS="8", STUB_BASE_EVAL_SLEEP="2", POLL="1")
    log = jfbox.run_log()
    assert r.returncode == 0, log[-2000:]
    assert "ceiling reached" not in log
    assert (jfbox.out / "comparison.json").exists()


def test_no_gpu_work_starts_without_a_verified_watchdog(jfbox):
    """REQUIRE_WATCHDOG=1 holds the run at preflight until the handshake exists.

    The watchdog was a line of prose in the visit record: launch, then arm it on the pid in
    judge_first.pid. Nothing checked that the second step happened, and an unwatched box is the
    2026-10-03 failure that idled a g6.24xlarge for 14.5 hours. Ordering it as a handshake makes
    the guarantee structural rather than remembered.
    """
    r = jfbox.run(REQUIRE_WATCHDOG="1", ARM_WAIT="2")
    log = jfbox.run_log()
    assert r.returncode == 2, log[-1500:]
    assert "no watchdog handshake" in log
    assert not jfbox.event_lines(), "a GPU step ran with no watchdog"


def test_the_handshake_releases_the_run(jfbox):
    """The control: the hold must be a hold, not a block."""
    jfbox.out.mkdir(parents=True, exist_ok=True)
    (jfbox.out / "watchdog.armed").touch()
    r = jfbox.run(REQUIRE_WATCHDOG="1", ARM_WAIT="2")
    assert r.returncode == 0, jfbox.run_log()[-1500:]
    assert "watchdog handshake present" in jfbox.run_log()
    assert (jfbox.out / "comparison.json").exists()


def test_the_launcher_refuses_to_leave_a_run_unwatched(jfbox):
    """launch_judge_first.sh must terminate the run when the watchdog cannot arm.

    Here it cannot: `nvidia-smi` is absent on this laptop, so the real watchdog refuses at its
    own self-test. A pass requires the wrapper to notice, say so, kill the run, and exit
    nonzero -- not to proceed with an unwatched GPU visit.
    """
    r = subprocess.run(
        ["bash", str(jfbox.root / "scripts/training/v04_box/launch_judge_first.sh")],
        capture_output=True, text=True, timeout=180,
        env=jfbox._env(ARM_TIMEOUT="5", WLOG=str(jfbox.out.parent / "autostop.log")),
    )
    assert r.returncode == 2, r.stdout[-2000:] + r.stderr[-1000:]
    assert "did not arm" in r.stdout, r.stdout[-2000:]
    assert not (jfbox.out / "comparison.json").exists(), (
        "the run produced a result although it was never watched")
    assert not (jfbox.out / "watchdog.armed").exists()
    time.sleep(1)
    pid_file = jfbox.out / "judge_first.pid"
    if pid_file.exists() and pid_file.read_text().strip().isdigit():
        assert not _alive(int(pid_file.read_text().strip())), "the unwatched run was left running"


def test_the_launcher_arms_the_watchdog_on_the_pid_the_run_recorded(jfbox):
    """With a mocked nvidia-smi the watchdog arms, and must do so on the run's OWN pid.

    Guard 1 of the watchdog keys on a pid, and the 2026-10-03 failure was a watcher keyed on the
    wrong process. `$!` is the wrapper's child, which is not necessarily the run that records
    itself, so the wrapper must read the pid from the file and the watchdog must confirm that
    exact number.
    """
    nvsmi = jfbox.bin / "nvidia-smi"
    nvsmi.write_text("#!/bin/sh\nprintf '0\\n'\n")
    nvsmi.chmod(0o755)
    wlog = jfbox.out.parent / "autostop.log"
    r = subprocess.run(
        ["bash", str(jfbox.root / "scripts/training/v04_box/launch_judge_first.sh")],
        capture_output=True, text=True, timeout=300,
        env=jfbox._env(ARM_TIMEOUT="30", WLOG=str(wlog), DRY_RUN="1", POLL="1", IDLE_MIN="60"),
    )
    assert r.returncode == 0, r.stdout[-3000:] + jfbox.run_log()[-2000:]
    assert "watchdog armed and verified" in r.stdout
    recorded = (jfbox.out / "judge_first.pid").read_text().strip()
    assert f"armed: pid={recorded} " in wlog.read_text(), (
        "the watchdog armed on a different pid than the run recorded")
    assert (jfbox.out / "comparison.json").exists()


# --------------------------------------------------------------- audit 2026-11-06, finding 5
def test_a_gpu_visit_refuses_to_start_on_the_cpu(jfbox):
    """Falsifiable here because this laptop has no CUDA: without ALLOW_CPU the run must stop.

    The evaluator chose `cuda if available else cpu`, so a box whose driver had not come up
    produced CPU numbers at a fraction of the throughput and said so only in a field nobody
    reads. The flag must actually reach the preflight, which is what this checks.
    """
    r = jfbox.run(ALLOW_CPU="")
    log = jfbox.run_log()
    assert r.returncode == 2, log[-2000:]
    assert "CUDA" in log
    assert not jfbox.event_lines(), "a GPU step ran on a box with no GPU"


def test_a_missing_train_split_stops_the_run_before_the_paid_step(jfbox):
    """The audit's sharpest version of this: train/val absence used to surface AFTER the base
    evaluation was paid for, at the one moment the money was already spent."""
    jfbox.splits["train"].unlink()
    r = jfbox.run()
    log = jfbox.run_log()
    assert r.returncode == 2, log[-2000:]
    assert "split files missing" in log
    assert not jfbox.event_lines(), "the base evaluation was paid for before this was noticed"


def test_an_adapter_directory_with_no_adapter_in_it_is_refused(jfbox):
    """It would have loaded as a fresh LoRA and been evaluated as the base arm."""
    (jfbox.root / "checkpoints/cpt_stage41_pointD_run05/final"
     / "adapter_model.safetensors").unlink()
    r = jfbox.run()
    assert r.returncode == 2
    assert "adapter_model.safetensors" in jfbox.run_log()
    assert not jfbox.event_lines()


def test_a_missing_package_stops_the_run(jfbox):
    r = jfbox.run(PREFLIGHT_REQUIRE="definitely_not_a_module_qqq")
    assert r.returncode == 2
    assert "definitely_not_a_module_qqq" in jfbox.run_log()
    assert not jfbox.event_lines()


def test_the_preflight_report_is_kept_beside_the_result(jfbox):
    r = jfbox.run()
    assert r.returncode == 0, jfbox.run_log()[-2000:]
    rep = json.loads((jfbox.out / "preflight.json").read_text())
    assert rep["ok"] is True
    assert rep["commit"]
    assert rep["adapter"]["cpt_run_id"] == "cpt_stage41_pointD_run05"
    assert rep["checkpoint"]["n_trainable_keys"] >= 1 or \
        "not checked" in str(rep["checkpoint"].get("loaded")), rep["checkpoint"]
