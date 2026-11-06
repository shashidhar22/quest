"""V04 launcher control-flow regression tests.

These pin the failures found by the 2026-11-04 external audit of `run_v04.sh` and
`dryrun.sh`. The sandbox and the stubs live in conftest.py, shared with judge_first.sh.
They are control-flow tests, not numerics: the trainer and the judge are stubs, so nothing
here says anything about GPU results.

Why each test exists, in the order the audit ranked them:

1. The dry run removed `$REPO/checkpoints/v04_arm*_seed*` and wrote fake training logs
   into those real paths. After a real visit those six directories hold the trained
   adapters the visit was bought to produce.
2. `train_arm` was called through command substitution, so the PID it echoed belonged to
   the substitution subshell and the driver could not `wait` on it. The driver therefore
   reached "V04 complete" and exited while six trainings were still running. The box
   watchdog waits on the driver PID, so driver exit means instance stop: the visit's own
   launcher would have killed it. Signalling it left them running too.
3. Every failing trainer and evaluator was followed by a successful `say`, so the visit
   exited 0 with no result.
4. The launcher invoked no comparison at all.

A scientific refusal by the G1 gate is NOT one of these failures and must stay exit 0.
"""
import json
import signal
import time

import pytest

from conftest import RANKS_FIXTURE, _alive


# --------------------------------------------------------------- audit finding 2
def test_driver_waits_for_every_delayed_trainer(box):
    """The driver must not exit while a trainer is still running.

    The box watchdog waits on the driver PID, so an early exit stops the instance and
    kills the training it was bought to run.
    """
    box.seed_cells()
    r = box.run(STUB_SLEEP="3")
    assert r.returncode == 0, f"visit failed: {r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    assert box.event_lines(), "no trainer ran at all"
    assert not box.unfinished_trainers(), (
        f"driver returned while these trainers were still running: "
        f"{box.unfinished_trainers()}"
    )
    dirs = sorted(p.name for p in (box.root / "checkpoints").glob("v04_arm*_seed*"))
    assert len(dirs) == 6, f"expected six arm directories, got {dirs}"
    for d in dirs:
        assert (box.root / "checkpoints" / d / ".stub_finished").exists(), (
            f"{d} had not finished when the driver exited"
        )


def test_driver_does_not_wait_on_a_pid_it_does_not_own(box):
    """`wait` must actually adopt the children, not print a 'not a child' error."""
    box.seed_cells()
    r = box.run(STUB_SLEEP="2")
    combined = r.stdout + r.stderr
    assert "is not a child of this shell" not in combined, (
        "driver tried to wait on a PID it does not own:\n"
        + "\n".join(l for l in combined.splitlines() if "not a child" in l)
    )


def test_phase_two_does_not_overlap_phase_one(box):
    """Arm D shares GPUs 0-2 with arm A, so P2 may not start before P1 finishes."""
    box.seed_cells()
    box.run(STUB_SLEEP="3")
    lines = box.event_lines()
    order = [(i, l.split()[0], l.split()[1]) for i, l in enumerate(lines)
             if l.startswith(("START", "END"))]
    last_a_end = max((i for i, kind, lbl in order
                      if kind == "END" and "armA" in lbl), default=None)
    first_d_start = min((i for i, kind, lbl in order
                         if kind == "START" and "armD" in lbl), default=None)
    assert last_a_end is not None, "arm A never finished"
    assert first_d_start is not None, "arm D never started"
    assert last_a_end < first_d_start, (
        "arm D started on GPUs 0-2 while arm A was still training there:\n"
        + "\n".join(lines)
    )


def test_sigterm_during_training_leaves_no_live_trainer(box):
    """TERM to the driver must take the training with it.

    Signalling the background function shell never reached the trainer, which was its
    child: the driver exited 130 while three trainers kept running on the GPUs. On the
    box that is the cost of the visit still being spent with nobody waiting for it.
    """
    box.seed_cells()
    p, pids = box.popen_until_started(3, STUB_SLEEP="30")
    p.send_signal(signal.SIGTERM)
    rc = p.wait(timeout=60)
    time.sleep(0.5)
    live = [q for q in pids if _alive(q)]
    assert rc != 0, "a terminated visit reported success"
    assert not live, f"these trainers outlived the signalled driver: {live}"


def test_sigterm_during_base_eval_leaves_no_live_child(box):
    """The same must hold when the signal lands while an evaluation is running.

    Bash defers a trap until the running foreground command finishes, so a foreground
    evaluation delayed cleanup by its own full duration. The trainer outlives the eval
    here on purpose: if cleanup waits for the eval, the trainers are still alive when
    the driver returns.
    """
    box.seed_cells()
    p, pids = box.popen_until_started(3, STUB_SLEEP="60", STUB_BASE_EVAL_SLEEP="8")
    p.send_signal(signal.SIGTERM)
    rc = p.wait(timeout=45)
    time.sleep(0.5)
    live = [q for q in pids if _alive(q)]
    assert rc != 0
    assert not live, f"these children outlived the signalled driver: {live}"


def test_sigterm_writes_no_success_artifact(box):
    """A killed visit must not leave anything a later reader takes for a result.

    Guard only: this already held before the lifecycle repair, so it is not evidence of
    it. It is here to catch a future change that announces completion on a killed run.
    """
    box.seed_cells()
    p, _ = box.popen_until_started(3, STUB_SLEEP="30")
    p.send_signal(signal.SIGTERM)
    p.wait(timeout=60)
    assert "V04 complete" not in box.run_log()


def test_abort_waits_for_live_children_instead_of_orphaning_them(box):
    """An abort must not leave training jobs running after the driver is gone."""
    r = box.run(STUB_BASE_EVAL_RC="7", STUB_SLEEP="3")
    assert r.returncode != 0, "the base re-score failed and the visit still succeeded"
    assert not box.unfinished_trainers(), (
        f"these trainers outlived the aborting driver: {box.unfinished_trainers()}"
    )


# --------------------------------------------------------------- audit finding 3
def test_trainer_failure_fails_the_visit(box):
    """A trainer exiting nonzero must not produce a successful visit."""
    r = box.run(STUB_TRAIN_RC="9", STUB_SLEEP="0")
    assert r.returncode != 0, "every trainer exited 9 and the visit still succeeded"
    assert "V04 complete" not in box.run_log(), (
        "the visit announced completion although every trainer failed"
    )


def test_comparison_eval_failure_fails_the_visit(box):
    """A comparison cell exiting nonzero must not produce a successful visit."""
    box.seed_cells()
    r = box.run(STUB_CELL_EVAL_RC="7")
    assert r.returncode != 0, "every comparison eval exited 7 and the visit still succeeded"
    assert "V04 complete" not in box.run_log()


def test_base_eval_failure_stops_before_the_gate(box):
    """The gate reads the base eval; if that eval failed there is nothing to gate on."""
    r = box.run(STUB_BASE_EVAL_RC="7")
    assert r.returncode != 0, "the base re-score failed and the visit still succeeded"
    assert "READABILITY PASS" not in box.run_log(), (
        "the gate claimed a pass although the eval it reads never ran"
    )


def test_unbuilt_comparison_cell_is_refused(box):
    """No launcher step trains the cell 2-4 checkpoints, so evaluating one is a fiction."""
    r = box.run()
    assert r.returncode != 0, "the visit succeeded with no trained comparison checkpoint"
    log = box.run_log()
    assert "REFUSED cell armD_paired: no trained comparison checkpoint" in log
    assert "V04 complete" not in log


def test_trb_cells_select_the_trb_task(box):
    """Both *_trb cells hard-coded --task as_paired_i, in the command and the manifest."""
    box.seed_cells()
    box.run()
    log = box.run_log()
    assert "judge base_trb rc=0 task=as_trb_i" in log, log[-2000:]
    assert "judge armD_trb rc=0 task=as_trb_i" in log, log[-2000:]
    m = json.loads((box.out / "base_trb" / "manifest.json").read_text())
    assert m["dataset"] == "as_trb_i", m


# --------------------------------------------------------------- audit finding 4
def test_the_visit_produces_a_paired_comparison(box):
    """A completed visit used to end with eval files and no comparison at all.

    `judge compare` was invoked nowhere in the launcher, so the question the visit exists
    to answer was never computed. The comparison is a required artifact now.
    """
    box.seed_cells()
    r = box.run()
    assert r.returncode == 0, box.run_log()[-3000:]
    out = box.out / "comparison.json"
    assert out.exists(), "the visit completed without producing a comparison"
    res = json.loads(out.read_text())
    assert res["primary"]["condition"] == "test_novel_pep"
    assert res["primary"]["unit"] == "peptide"
    assert res["primary"]["metric"] == "norm_rank"
    assert res["primary"]["n_units"] > 1, "the comparison resampled fewer than two peptides"
    assert res["cluster_sensitivity"]["unit"] == "pep_cluster"
    assert res["utility_target"]["established"] is False
    log = box.run_log()
    assert "PRIMARY" in log and "SECONDARY" in log
    assert "does not establish" in log.lower()


def test_a_refused_comparison_fails_the_visit(box):
    """An uncertified comparison is not a result, so it cannot leave a successful visit."""
    box.seed_cells()
    # The arm-D cell carries a UNIFORM-draw evaluation while the base cell is length_matched:
    # a stale result from another run. Declared through the stub rather than smuggled into a
    # fixture's args, because a real evaluation records the draw it was actually given.
    r = box.run(STUB_CELL_EVAL_DRAW="uniform")
    assert r.returncode != 0, "a comparison across a draw mismatch left a successful visit"
    assert not (box.out / "comparison.json").exists(), "a refused comparison wrote a result"
    log = box.run_log()
    assert "draw" in log, log[-2000:]
    assert "V04 complete" not in log


# --------------------------------------------------------------- audit finding 5
def test_absent_candidate_pool_aborts(box):
    """A missing pool became the string POOL_FILE_MISSING, which matched itself."""
    r = box.run(POOL_FILE=str(box.root / "no_such_pool.tsv"))
    assert r.returncode == 2, f"a missing candidate pool did not abort: rc={r.returncode}"
    assert "candidate pool not found" in box.run_log()


def test_tolerated_provenance_difference_is_disclosed(box):
    """A waiver must appear in the log beside the result it licensed."""
    box.seed_cells()
    box.run()
    assert "TOLERATED by kind=cells: cotenants" in box.run_log(), (
        "the base-vs-armD co-tenancy waiver was applied silently"
    )


def test_fewer_than_four_gpus_aborts(box):
    """The old fallback put concurrent jobs on GPU 0 and reused master_port 29800."""
    r = box.run(NGPU="2")
    assert r.returncode == 2, f"a 2-GPU box did not abort: rc={r.returncode}"
    assert "needs 4 GPUs" in box.run_log()


# --------------------------------------------------------------- audit finding 7
def test_gate_technical_error_stops_the_visit_before_arm_d(box):
    """A gate that cannot read its evidence must not be treated as a refusal.

    The ranks-less eval is the real 2026-10-08 cell. With `--units` given, the gate used
    to print SKIPPED and return 0, so the visit spent arm D and the comparison cells on a
    read that never happened. It now exits 2, and the launcher stops before P2.
    """
    box.seed_cells()
    ranksless = box.root / "tests" / "fixtures" / "s42_eval_base_paired_20261008.json"
    r = box.run(STUB_BASE_EVAL_FIXTURE=str(ranksless))
    log = box.run_log()
    assert r.returncode != 0, "the visit proceeded although the gate could not read its evidence"
    assert "TECHNICAL error" in log, log[-2000:]
    started = [l.split()[1] for l in box.event_lines() if l.startswith("START")]
    assert not [s for s in started if "armD" in s], (
        f"arm D was started after a technical gate error: {started}"
    )
    assert not box.unfinished_trainers(), "the abort orphaned a trainer"


# --------------------------------------------------------------- the refusal is not a bug
def test_gate_refusal_is_not_a_run_failure(box):
    """A pre-agreed scientific refusal must stay exit 0 and skip the cells.

    The audit asks for failed software and unaffordable science to stay distinguishable.
    """
    r = box.run(RHO_FLOOR="0.7", STUB_SLEEP="0")
    assert r.returncode == 0, (
        f"a scientific refusal was reported as a run failure: {r.stdout[-2000:]}"
    )
    log = box.run_log()
    assert "G1 FAILED: comparison cells 2-4 SKIPPED" in log
    assert "pre-agreed refusal" in log
    assert not (box.out / "armD_paired" / "eval.json").exists(), (
        "a comparison cell ran despite the refusal"
    )


# --------------------------------------------------------------- audit finding 1
def test_dryrun_preserves_checkpoints_it_did_not_create(box):
    """The dry run must not touch checkpoints outside its own temporary directory.

    After a real visit, `checkpoints/v04_arm*_seed*` holds the trained adapters. The dry
    run's cleanup glob matched exactly those.
    """
    keep = box.root / "checkpoints" / "v04_armD_seed99"
    keep.mkdir(parents=True)
    sentinel = keep / "adapter_model.safetensors"
    sentinel.write_bytes(b"a real trained adapter, bought with GPU hours")
    before = sentinel.read_bytes()

    r = box.run_dryrun()

    assert sentinel.exists(), (
        f"the dry run deleted a checkpoint it did not create (rc={r.returncode})\n"
        f"{r.stdout[-2000:]}"
    )
    assert sentinel.read_bytes() == before, "the dry run modified a real checkpoint"
    # Without this the test passed while the dry run failed, which is how the
    # false-failure below went unnoticed.
    assert r.returncode == 0, f"the dry run itself failed:\n{r.stdout[-3000:]}"


def test_dryrun_passes_with_pre_existing_checkpoints(box):
    """A checkout that already holds V04 checkpoints is the normal case after a visit.

    The repaired dry run writes nothing there, so their mere presence is not evidence
    that it did. Asserting on existence alone turned a safe run into a false failure.
    """
    expected = {}
    for arm in ("armA", "armD"):
        for seed in (1, 2, 3):
            d = box.root / "checkpoints" / f"v04_{arm}_seed{seed}"
            d.mkdir(parents=True)
            f = d / "adapter_model.safetensors"
            f.write_bytes(f"trained {arm} seed {seed}".encode())
            expected[f] = f.read_bytes()

    r = box.run_dryrun()

    assert r.returncode == 0, (
        f"the dry run failed merely because checkpoints existed:\n{r.stdout[-3000:]}"
    )
    for f, content in expected.items():
        assert f.exists(), f"{f.name} was removed"
        assert f.read_bytes() == content, f"{f.name} was modified"


def test_dryrun_writes_no_artifacts_into_the_checkpoint_tree(box):
    """Stub training logs and manifests in a real checkpoint path are a provenance trap.

    Snapshot-and-compare rather than "the tree is empty": the arm-D adapter's own files live
    there and are read by the preflight, so emptiness was only ever a proxy. What matters is
    that the dry run adds nothing and CHANGES nothing -- a stub manifest written over a real
    adapter's is the trap, and an emptiness check would not have seen it.
    """
    tree = box.root / "checkpoints"
    before = {str(p.relative_to(box.root)): p.read_bytes()
              for p in tree.rglob("*") if p.is_file()}
    box.run_dryrun()
    after = {str(p.relative_to(box.root)): p.read_bytes()
             for p in tree.rglob("*") if p.is_file()}
    assert sorted(after) == sorted(before), (
        f"the dry run changed the checkpoint tree's contents: "
        f"added {sorted(set(after) - set(before))}, "
        f"removed {sorted(set(before) - set(after))}")
    changed = [k for k in before if after[k] != before[k]]
    assert not changed, f"the dry run overwrote existing checkpoint artifacts: {changed}"


def test_dryrun_passes(box):
    """The dry run's own expectations must hold on the repaired launcher."""
    r = box.run_dryrun()
    assert r.returncode == 0, f"dryrun failed:\n{r.stdout[-4000:]}\n{r.stderr[-2000:]}"
    assert "overall: PASS" in r.stdout
