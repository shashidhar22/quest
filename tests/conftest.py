"""Shared sandbox and stubs for the box-launcher tests.

Both `run_v04.sh` and `judge_first.sh` read the same repo paths and call the same judge, so
the disposable copy and the stubs live here rather than in one test module.

The copy matters: `dryrun.sh` derives $REPO from its own location, so testing it in place is
the very destruction it was found to cause. Nothing here touches the real checkout.

These are control-flow stubs. Nothing in them says anything about GPU behaviour, throughput or
numerics, and a passing run is never evidence of any of those.
"""
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RANKS_FIXTURE = REPO / "tests" / "fixtures" / "s42_eval_base_paired_ranks_20261008.json"
V04_CELLS = ("armD_paired", "base_trb", "armD_trb")
ARM_D_ADAPTER = "checkpoints/cpt_stage41_pointD_run05/final"

# A CPT trainer stub that logs its own lifetime, so a test can see whether the driver waited and
# whether two phases overlapped. $EVENTS is the shared event log.
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

# Fakes only the judge, by subcommand. Every other python call is the real interpreter, so the
# gate, the readability predicate, provenance and the comparator are exercised, not mocked.
PYTHON_STUB = r"""#!/bin/bash
if [ "$1" = "-m" ] && [ "$2" = "track_a.esm_sft.judge" ]; then
  sub=$3; out=""; outdir=""; args=("$@")
  for ((i=0;i<${#args[@]};i++)); do
    [ "${args[i]}" = --out ] && out=${args[i+1]}
    [ "${args[i]}" = --out_dir ] && outdir=${args[i+1]}
  done
  case "$sub" in
    probe)
      rc=${STUB_PROBE_RC:-0}
      echo "[probe] $([ "$rc" -eq 0 ] && echo PASS || echo FAIL): stub scorer"
      echo "judge probe rc=$rc" >> "$EVENTS"
      exit "$rc" ;;
    train)
      rc=${STUB_SFT_TRAIN_RC:-0}
      mkdir -p "$outdir"
      echo "START sft_$(basename "$outdir") $$" >> "$EVENTS"
      sleep "${STUB_SFT_TRAIN_SLEEP:-0}"
      [ "$rc" -eq 0 ] && : > "$outdir/best.pt"
      echo "END sft_$(basename "$outdir") $$" >> "$EVENTS"
      echo "judge train $outdir rc=$rc" >> "$EVENTS"
      exit "$rc" ;;
    *)
      mkdir -p "$(dirname "$out")"
      src=$RANKS_FIXTURE
      case "$out" in
        *base_paired*)
          rc=${STUB_BASE_EVAL_RC:-0}
          src=${STUB_BASE_EVAL_FIXTURE:-$RANKS_FIXTURE}
          sleep "${STUB_BASE_EVAL_SLEEP:-0}" ;;
        *) rc=${STUB_CELL_EVAL_RC:-0}; src=${STUB_CELL_EVAL_FIXTURE:-$RANKS_FIXTURE} ;;
      esac
      if [ "$rc" -eq 0 ]; then cp "$src" "$out"; fi
      echo "judge $out rc=$rc" >> "$EVENTS"
      exit "$rc" ;;
  esac
fi
exec "$PYREAL" "$@"
"""


def _alive(pid):
    """True if the pid is still a live process. Zombies are reaped by init here."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class _Box:
    """A disposable copy of just what the launchers read, with its own checkpoint tree."""

    def __init__(self, tmp_path, script):
        self.root = root = tmp_path / "quest"
        self.script = script
        self.out = tmp_path / "out"
        self._live = []

        for rel in ("scripts/training/v04_box", "scripts/training/stage42_box"):
            shutil.copytree(REPO / rel, root / rel)
        (root / "tests").mkdir(parents=True, exist_ok=True)
        shutil.copytree(REPO / "tests" / "fixtures", root / "tests" / "fixtures")
        (root / "checkpoints").mkdir()
        # The legacy arm-D adapter the judge cells and judge_first.sh read.
        (root / ARM_D_ADAPTER).mkdir(parents=True)

        # The candidate pool at the launchers' default path. An absent pool must abort, so the
        # happy-path tests need a real one.
        pool = root / "benchmark_v2" / "splits" / "as_class_i_candidate_pool.tsv"
        pool.parent.mkdir(parents=True)
        pool.write_text("peptide\n" + "".join(f"PEPTIDE{i:03d}\n" for i in range(1, 121)))

        # Preflight requires these to exist before a GPU job starts. The trainer is stubbed, so
        # empty marker directories are the honest stand-in.
        for d in ("data/cpt_datasets/run01_cdr3_pocket_proportional_10M",
                  "data/cpt_datasets/run05_cdr3_fullmhc_proportional_10M",
                  "data/cpt_canonical_eval_v2"):
            (root / d).mkdir(parents=True, exist_ok=True)

        # A stand-in base SFT checkpoint. /dev/null will not do: preflight requires a regular
        # file, and a character device is not a checkpoint.
        self.base_stub = tmp_path / "base_best.pt"
        self.base_stub.touch()

        self.events = tmp_path / "events.log"
        self.events.touch()
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        (self.bin / "torchrun").write_text(TORCHRUN_STUB)
        (self.bin / "python").write_text(PYTHON_STUB)
        for f in self.bin.iterdir():
            f.chmod(0o755)

    # -- invocation ----------------------------------------------------------------
    def _env(self, **over):
        e = dict(os.environ)
        e.update(
            Q=str(self.root), O=str(self.out), PY=str(self.bin / "python"),
            TORCHRUN=str(self.bin / "torchrun"), NGPU="4", SEEDS="1 2 3",
            RHO_FLOOR="0.8", BASE_CKPT=str(self.base_stub), EVENTS=str(self.events),
            PYREAL="python3",
            RANKS_FIXTURE=str(self.root / "tests/fixtures" / RANKS_FIXTURE.name),
            PATH=f"{self.bin}:{os.environ['PATH']}",
        )
        e.update({k: str(v) for k, v in over.items()})
        return e

    def run(self, *, timeout=300, **env):
        return subprocess.run(
            ["bash", str(self.root / self.script)],
            capture_output=True, text=True, timeout=timeout, env=self._env(**env),
        )

    def popen_until_started(self, n_starts, settle=15, **env):
        """Launch in its own session and wait until n jobs report in.

        Returns (process, [stub pids]). The fixture kills the whole session afterwards
        whatever the test did.
        """
        self.out.mkdir(parents=True, exist_ok=True)
        log = self.out.parent / "driver.log"
        fh = log.open("w")
        p = subprocess.Popen(
            ["bash", str(self.root / self.script)],
            env=self._env(**env), stdout=fh, stderr=fh, start_new_session=True,
        )
        self._live.append(p)
        deadline = time.monotonic() + settle
        while time.monotonic() < deadline:
            if len([l for l in self.event_lines() if l.startswith("START")]) >= n_starts:
                break
            time.sleep(0.05)
        pids = [int(l.split()[2]) for l in self.event_lines() if l.startswith("START")]
        assert len(pids) >= n_starts, (
            f"only {len(pids)} jobs started within {settle}s: {log.read_text()[-2000:]}"
        )
        return p, pids

    def run_dryrun(self, name="dryrun.sh", timeout=900):
        return subprocess.run(
            ["bash", str(self.root / "scripts/training/v04_box" / name), "python3"],
            capture_output=True, text=True, timeout=timeout,
        )

    # -- observation ---------------------------------------------------------------
    def seed_cells(self, cells=V04_CELLS):
        """Supply the comparison checkpoints no launcher step builds yet.

        Cells 2-4 refuse without them, so a test that wants the cell path exercised has to
        stand in for the unfinished train-then-evaluate wiring.
        """
        for c in cells:
            d = self.out / c
            d.mkdir(parents=True, exist_ok=True)
            (d / "best.pt").touch()

    def event_lines(self):
        return [l for l in self.events.read_text().splitlines() if l.strip()]

    def run_log(self):
        p = self.out / "run.log"
        return p.read_text() if p.exists() else ""

    def unfinished(self):
        started = {l.split()[1] for l in self.event_lines() if l.startswith("START")}
        ended = {l.split()[1] for l in self.event_lines() if l.startswith("END")}
        return sorted(started - ended)

    # kept under the old name, which the V04 tests read as "trainers"
    unfinished_trainers = unfinished

    def kill_session(self):
        for p in self._live:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


@pytest.fixture
def box(tmp_path):
    """Sandbox wired to run_v04.sh."""
    b = _Box(tmp_path, "scripts/training/v04_box/run_v04.sh")
    yield b
    b.kill_session()


@pytest.fixture
def jfbox(tmp_path):
    """Sandbox wired to judge_first.sh."""
    b = _Box(tmp_path, "scripts/training/v04_box/judge_first.sh")
    yield b
    b.kill_session()
