"""V04 launcher control-flow regression tests.

These pin the failures found by the 2026-11-04 external audit of `run_v04.sh` and
`dryrun.sh`. They are control-flow tests, not numerics: the trainer and the judge are
stubs, so nothing here says anything about GPU results.

Why each test exists, in the order the audit ranked them:

1. The dry run removed `$REPO/checkpoints/v04_arm*_seed*` and wrote fake training logs
   into those real paths. After a real visit those six directories hold the trained
   adapters the visit was bought to produce.
2. `train_arm` was called through command substitution, so the PID it echoed belonged to
   the substitution subshell and the driver could not `wait` on it. The driver therefore
   reached "V04 complete" and exited while six trainings were still running. The box
   watchdog waits on the driver PID, so driver exit means instance stop: the visit's own
   launcher would have killed it.
3. Every failing trainer and evaluator was followed by a successful `say`, so the visit
   exited 0 with no result.

A scientific refusal by the G1 gate is NOT one of these failures and must stay exit 0.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RANKS_FIXTURE = REPO / "tests" / "fixtures" / "s42_eval_base_paired_ranks_20261008.json"
CELLS = ("armD_paired", "base_trb", "armD_trb")

# A trainer stub that logs its own lifetime, so a test can see whether the driver waited
# and whether two phases overlapped. $EVENTS is the shared event log.
TORCHRUN_STUB = r"""#!/bin/bash
out=""; args=("$@")
for ((i=0;i<${#args[@]};i++)); do [ "${args[i]}" = --output_dir ] && out=${args[i+1]}; done
lbl=$(basename "$out")
mkdir -p "$out"
echo "START $lbl $$" >> "$EVENTS"
sleep "${STUB_SLEEP:-0}"
echo "stub trainer" > "$out/training.log"
echo "END $lbl $$" >> "$EVENTS"
touch "$out/.stub_finished"
exit "${STUB_TRAIN_RC:-0}"
"""

# Fakes only the judge; every other python call is the real interpreter, so the gate and
# the readability predicate are exercised rather than mocked.
PYTHON_STUB = r"""#!/bin/bash
if [ "$1" = "-m" ] && [ "$2" = "track_a.esm_sft.judge" ]; then
  out=""; args=("$@")
  for ((i=0;i<${#args[@]};i++)); do [ "${args[i]}" = --out ] && out=${args[i+1]}; done
  mkdir -p "$(dirname "$out")"
  case "$out" in
    *base_paired*) rc=${STUB_BASE_EVAL_RC:-0} ;;
    *)             rc=${STUB_CELL_EVAL_RC:-0} ;;
  esac
  if [ "$rc" -eq 0 ]; then cp "$RANKS_FIXTURE" "$out"; fi
  echo "judge $out rc=$rc" >> "$EVENTS"
  exit "$rc"
fi
exec "$PYREAL" "$@"
"""


@pytest.fixture
def box(tmp_path):
    """A disposable copy of just what the launcher reads, with its own checkpoints tree.

    The copy matters: `dryrun.sh` derives $REPO from its own location, so testing it in
    place is the very destruction under test.
    """
    root = tmp_path / "quest"
    for rel in ("scripts/training/v04_box", "scripts/training/stage42_box"):
        shutil.copytree(REPO / rel, root / rel)
    (root / "tests").mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO / "tests" / "fixtures", root / "tests" / "fixtures")
    (root / "checkpoints").mkdir()

    # The candidate pool at the launcher's default path. An absent pool must abort, so the
    # happy-path tests need a real one.
    pool = root / "benchmark_v2" / "splits" / "as_class_i_candidate_pool.tsv"
    pool.parent.mkdir(parents=True)
    pool.write_text("peptide\n" + "".join(f"PEPTIDE{i:03d}\n" for i in range(1, 121)))

    events = tmp_path / "events.log"
    events.touch()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "torchrun").write_text(TORCHRUN_STUB)
    (bin_dir / "python").write_text(PYTHON_STUB)
    for f in bin_dir.iterdir():
        f.chmod(0o755)

    class Box:
        def __init__(self):
            self.root = root
            self.events = events
            self.out = tmp_path / "out"

        def seed_cells(self):
            """Supply the comparison checkpoints no launcher step builds yet.

            Cells 2-4 refuse without them, so a test that wants the cell path exercised
            has to stand in for the unfinished train-then-evaluate wiring.
            """
            for c in CELLS:
                d = self.out / c
                d.mkdir(parents=True, exist_ok=True)
                (d / "best.pt").touch()

        def run(self, *, timeout=120, **env):
            e = dict(os.environ)
            e.update(
                Q=str(root),
                O=str(self.out),
                PY=str(bin_dir / "python"),
                TORCHRUN=str(bin_dir / "torchrun"),
                NGPU="4",
                SEEDS="1 2 3",
                RHO_FLOOR="0.8",
                BASE_CKPT="/dev/null",
                EVENTS=str(events),
                PYREAL="python3",
                RANKS_FIXTURE=str(root / "tests/fixtures" / RANKS_FIXTURE.name),
                PATH=f"{bin_dir}:{os.environ['PATH']}",
            )
            e.update({k: str(v) for k, v in env.items()})
            return subprocess.run(
                ["bash", str(root / "scripts/training/v04_box/run_v04.sh")],
                capture_output=True, text=True, timeout=timeout, env=e,
            )

        def run_dryrun(self, timeout=600):
            return subprocess.run(
                ["bash", str(root / "scripts/training/v04_box/dryrun.sh"), "python3"],
                capture_output=True, text=True, timeout=timeout,
            )

        def event_lines(self):
            return [l for l in self.events.read_text().splitlines() if l.strip()]

        def run_log(self):
            p = self.out / "run.log"
            return p.read_text() if p.exists() else ""

        def unfinished_trainers(self):
            started = {l.split()[1] for l in self.event_lines() if l.startswith("START")}
            ended = {l.split()[1] for l in self.event_lines() if l.startswith("END")}
            return sorted(started - ended)

    return Box()


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
    assert "pre-agreed refusal, not a run failure" in log
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


def test_dryrun_writes_no_artifacts_into_the_checkpoint_tree(box):
    """Stub training logs and manifests in a real checkpoint path are a provenance trap."""
    box.run_dryrun()
    leftovers = sorted(
        str(p.relative_to(box.root)) for p in (box.root / "checkpoints").rglob("*")
        if p.is_file()
    )
    assert not leftovers, f"the dry run left artifacts in the checkpoint tree: {leftovers}"


def test_dryrun_passes(box):
    """The dry run's own expectations must hold on the repaired launcher."""
    r = box.run_dryrun()
    assert r.returncode == 0, f"dryrun failed:\n{r.stdout[-4000:]}\n{r.stderr[-2000:]}"
    assert "overall: PASS" in r.stdout
