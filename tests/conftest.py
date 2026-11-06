"""Shared sandbox and stubs for the box-launcher tests.

Both `run_v04.sh` and `judge_first.sh` read the same repo paths and call the same judge, so
the disposable copy and the stubs live here rather than in one test module.

The copy matters: `dryrun.sh` derives $REPO from its own location, so testing it in place is
the very destruction it was found to cause. Nothing here touches the real checkout.

These are control-flow stubs. Nothing in them says anything about GPU behaviour, throughput or
numerics, and a passing run is never evidence of any of those.
"""
import json
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
      echo "EVAL_START $(basename "$(dirname "$out")") $$" >> "$EVENTS"
      ckpt=""; adapter="none"; draw="length_matched"; task="as_paired_i"
      for ((i=0;i<${#args[@]};i++)); do
        [ "${args[i]}" = --ckpt ] && ckpt=${args[i+1]}
        [ "${args[i]}" = --adapter ] && adapter=${args[i+1]}
        [ "${args[i]}" = --distractor_draw ] && draw=${args[i+1]}
        [ "${args[i]}" = --task ] && task=${args[i+1]}
      done
      case "$out" in
        *base_paired*)
          rc=${STUB_BASE_EVAL_RC:-0}
          src=${STUB_BASE_EVAL_FIXTURE:-$RANKS_FIXTURE}
          sleep "${STUB_BASE_EVAL_SLEEP:-0}" ;;
        *) rc=${STUB_CELL_EVAL_RC:-0}; src=${STUB_CELL_EVAL_FIXTURE:-$RANKS_FIXTURE}
           # A cell evaluated under a DIFFERENT draw, i.e. a stale result from another run.
           # Explicit, because a real evaluation records the draw it was given and both arms
           # of one visit are given the same one.
           draw=${STUB_CELL_EVAL_DRAW:-$draw} ;;
      esac
      # A real evaluation records what it read; the launcher derives its manifest from that. So
      # the stub records it too, computed from the sandbox's own files. STUB_EVAL_RAW=1 writes
      # the bare recorded ranks instead, which is what a pre-repair evaluator produced.
      if [ "$rc" -eq 0 ]; then
        if [ "${STUB_EVAL_RAW:-0}" = 1 ]; then cp "$src" "$out"
        else "$PYREAL" "$QROOT/tests/fixtures/make_eval.py" --ranks "$src" --out "$out" \
               --ckpt "$ckpt" --adapter "$adapter" --draw "$draw" --task "$task" || exit 9
        fi
      fi
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
        # launch_judge_first.sh arms the real watchdog, so the sandbox needs it too.
        shutil.copy2(REPO / "scripts/training/autostop_when_done.sh",
                     root / "scripts/training/autostop_when_done.sh")
        # provenance.py reads artifact digests and lineage through track_a.shared.artifacts, and
        # the eval stand-in writes them, so the sandbox carries that stdlib-only module.
        (root / "track_a" / "shared").mkdir(parents=True)
        (root / "track_a" / "esm_sft").mkdir(parents=True)
        for rel in ("track_a/__init__.py", "track_a/shared/__init__.py",
                    "track_a/shared/artifacts.py", "track_a/esm_sft/__init__.py",
                    # scoring.py holds SFT_OBJECTIVE, sft_recipe and cotenants: the eval
                    # stand-in records provenance through the same code the real evaluator uses,
                    # so the stub cannot drift into being more complete than the real thing.
                    "track_a/esm_sft/scoring.py"):
            shutil.copy2(REPO / rel, root / rel)
        (root / "tests").mkdir(parents=True, exist_ok=True)
        shutil.copytree(REPO / "tests" / "fixtures", root / "tests" / "fixtures")
        (root / "checkpoints").mkdir()
        # The legacy arm-D adapter the judge cells and judge_first.sh read. The preflight now
        # requires an adapter DIRECTORY to contain an adapter, so the stand-in carries the two
        # files that decide what a LoRA is, plus the lineage a real one records.
        (root / ARM_D_ADAPTER).mkdir(parents=True)
        (root / ARM_D_ADAPTER / "adapter_config.json").write_text(
            '{"r": 32, "lora_alpha": 64, "lora_dropout": 0.05}')
        (root / ARM_D_ADAPTER / "adapter_model.safetensors").write_bytes(b"stand-in weights")
        (root / ARM_D_ADAPTER / "trainer_state.json").write_text(json.dumps({
            "global_step": 7800,
            "config": {"batch_size": 16, "gradient_accumulation_steps": 16, "seed": 42}}))

        # The candidate pool at the launchers' default path. An absent pool must abort, so the
        # happy-path tests need a real one.
        pool = root / "benchmark_v2" / "splits" / "as_class_i_candidate_pool.tsv"
        pool.parent.mkdir(parents=True)
        pool.write_text("peptide\n" + "".join(f"PEPTIDE{i:03d}\n" for i in range(1, 121)))
        # Real files at the real split paths. Not parquet -- nothing in the stubbed path parses
        # them -- but provenance hashes them, the preflight requires them, and a test can change
        # one to prove that a reused evaluation is refused.
        self.splits = {}
        for cond in ("train", "val", "test_iid", "test_novel_pep", "test_level4"):
            f = pool.parent / f"as_paired_i_{cond}.parquet"
            f.write_text(f"stand-in split {cond}\n")
            self.splits[cond] = f

        # Preflight requires these to exist before a GPU job starts. The trainer is stubbed, so
        # empty marker directories are the honest stand-in.
        for d in ("data/cpt_datasets/run01_cdr3_pocket_proportional_10M",
                  "data/cpt_datasets/run05_cdr3_fullmhc_proportional_10M",
                  "data/cpt_canonical_eval_v2"):
            (root / d).mkdir(parents=True, exist_ok=True)

        # A stand-in base SFT checkpoint. /dev/null will not do, and neither will an empty
        # file: the preflight now loads it and requires a trainable state dict, because a
        # checkpoint that restores nothing would otherwise be evaluated as freshly initialised
        # weights. So this is a REAL torch file wherever torch is installed.
        self.base_stub = tmp_path / "base_best.pt"
        try:
            import torch
            torch.save({"args": {"task": "as_paired_i", "tcr_variant": "cdr3",
                                 "mhc_variant": "full", "adapter": "none", "seed": 42},
                        "trainable_state_dict": {"pool.q.weight": torch.zeros(2, 2),
                                                 "head.0.bias": torch.zeros(2)}},
                       self.base_stub)
        except ImportError:
            # No torch here: the preflight reports the load as unchecked, which is honest.
            self.base_stub.write_bytes(b"stand-in checkpoint")

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
            PYREAL="python3", QROOT=str(self.root),
            # The sandbox has no GPU and no peft; everything else the preflight checks is real.
            ALLOW_CPU="1", PREFLIGHT_REQUIRE="",
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

    def eval_pids(self):
        """Pids of the stubbed evaluations, so a test can prove one was really terminated.

        A distinct prefix on purpose: `unfinished()` and `popen_until_started` key on "START",
        and an eval line counting as a START would let the SIGTERM test return before the phase
        it means to interrupt had begun.
        """
        return [int(l.split()[2]) for l in self.event_lines() if l.startswith("EVAL_START")]

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
