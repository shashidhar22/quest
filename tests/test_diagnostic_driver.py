"""The diagnostic driver: refuses before spending, runs six steps in order, records every exit.

Control-flow stubs only, as in test_judge_first.py: a stub `PY` records its argv and writes
the output file a real step would. Nothing here says anything about GPU behaviour or numerics.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import _alive

REPO = Path(__file__).resolve().parent.parent
DRIVER = REPO / "scripts/training/pilot12_box/run_diagnostic.sh"
WRAPPER = REPO / "scripts/training/v04_box/launch_judge_first.sh"
REQUIRED = ("Q", "O", "PY", "MANIFEST_A", "MANIFEST_A_SHA256", "MANIFEST_B", "MANIFEST_B_SHA256",
            "CKPT_BASE", "CKPT_D", "ADAPTER_D", "MAX_HOURS")

PY_STUB = """#!{py}
import json, os, sys, time
a = sys.argv[1:]
out = a[a.index('--out') + 1]
with open(os.environ['EVENTS'], 'a') as f: f.write(json.dumps({{'pid': os.getpid(), 'argv': a}}) + '\\n')
name = os.path.basename(out)
if name == os.environ.get('STUB_SLOW'): time.sleep(float(os.environ.get('STUB_SLEEP', '0')))
if name == os.environ.get('STUB_FAIL'): sys.exit(1)
if os.path.exists(out): sys.exit(3)
import hashlib
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
doc = {{'manifest_sha256': sha(a[a.index('--manifest') + 1])}}
if '--ckpt' in a:
    doc.update(protocol='judge-diagnostic-results-v1', checkpoint_sha256=sha(a[a.index('--ckpt') + 1]))
else:
    i = a.index('--results')
    doc['results'] = {{x.split('=')[0]: {{'checkpoint_sha256': json.load(open(x.split('=', 1)[1]))['checkpoint_sha256']}}
                      for x in a[i + 1:i + 3]}}
with open(out, 'x') as f: json.dump(doc, f, indent=2)
"""


class Box:
    def __init__(self, tmp_path):
        self.q = tmp_path / "quest"
        for m in ("diagnostic_eval.py", "diagnostic_analyze.py"):
            (self.q / "track_a/esm_sft").mkdir(parents=True, exist_ok=True)
            (self.q / "track_a/esm_sft" / m).write_text("")
        self.out = tmp_path / "out"
        self.events = tmp_path / "events.log"
        self.events.touch()
        self.py = tmp_path / "py"
        self.py.write_text(PY_STUB.format(py=sys.executable))
        self.py.chmod(0o755)
        files = tmp_path / "in"
        files.mkdir()
        self.inputs = {}
        for k in ("MANIFEST_A", "MANIFEST_B", "CKPT_BASE", "CKPT_D"):
            p = files / k.lower()
            p.write_text(f"stand-in {k}\n")
            self.inputs[k] = str(p)
        (files / "adapter").mkdir()
        self.inputs["ADAPTER_D"] = str(files / "adapter")
        for k in ("MANIFEST_A", "MANIFEST_B"):
            self.inputs[k + "_SHA256"] = hashlib.sha256(Path(self.inputs[k]).read_bytes()).hexdigest()

    def env(self, **over):
        e = {k: v for k, v in os.environ.items() if k not in REQUIRED}
        e.update(Q=str(self.q), O=str(self.out), PY=str(self.py), MAX_HOURS="1", ALLOW_CPU="1",
                 EVENTS=str(self.events), STEP_POLL="1", **self.inputs)
        e.update({k: str(v) for k, v in over.items() if v is not None})
        return {k: v for k, v in e.items() if k not in over or over[k] is not None}

    def run(self, timeout=120, **over):
        return subprocess.run(["bash", str(DRIVER)], env=self.env(**over), capture_output=True,
                              text=True, timeout=timeout)

    def calls(self):
        return [json.loads(l) for l in self.events.read_text().splitlines() if l.strip()]

    def status(self):
        return json.loads((self.out / "status.json").read_text())

    def log(self):
        p = self.out / "run.log"
        return p.read_text() if p.exists() else ""


@pytest.fixture
def dbox(tmp_path):
    return Box(tmp_path)


def _flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def test_unset_required_env_refuses_and_names_every_one(dbox):
    r = dbox.run(MANIFEST_B=None, CKPT_D=None, MAX_HOURS=None)
    assert r.returncode == 2
    text = r.stdout + r.stderr
    assert "required environment not set: MANIFEST_B CKPT_D MAX_HOURS\n" in text, text
    assert not dbox.calls()
    assert dbox.status()["outcome"] == "env_refused"


def test_a_wrong_manifest_sha_refuses_before_any_scoring(dbox):
    r = dbox.run(MANIFEST_B_SHA256="0" * 64)
    assert r.returncode == 2, dbox.log()
    assert "MANIFEST_B" in dbox.log()
    assert not dbox.calls()
    s = dbox.status()
    assert s["outcome"] == "preflight_failed" and s["stage"] == "preflight"


@pytest.mark.skipif(shutil.which("nvidia-smi") is not None, reason="this machine has a GPU")
def test_no_gpu_and_no_allow_cpu_refuses(dbox):
    r = dbox.run(ALLOW_CPU="")
    assert r.returncode == 2
    assert "nvidia-smi" in dbox.log()
    assert not dbox.calls()


def test_no_python_step_runs_without_the_watchdog_handshake(dbox):
    r = dbox.run(REQUIRE_WATCHDOG="1", ARM_WAIT="2")
    assert r.returncode == 2, dbox.log()
    assert "no watchdog handshake" in dbox.log()
    assert not dbox.calls()
    assert dbox.status()["outcome"] == "no_watchdog"


def test_the_handshake_releases_the_run(dbox):
    dbox.out.mkdir()
    (dbox.out / "watchdog.armed").touch()
    r = dbox.run(REQUIRE_WATCHDOG="1", ARM_WAIT="2")
    assert r.returncode == 0, dbox.log()
    assert len(dbox.calls()) == 6


def test_happy_path_runs_six_steps_in_order(dbox):
    r = dbox.run(RUN_TOKEN="run/launch")
    assert r.returncode == 0, dbox.log() + r.stderr
    calls = [c["argv"] for c in dbox.calls()]
    i = dbox.inputs
    want = [("A_base.json", "none", i["CKPT_BASE"], i["MANIFEST_A"]),
            ("A_armD.json", i["ADAPTER_D"], i["CKPT_D"], i["MANIFEST_A"]),
            ("A_analysis.json", None, None, i["MANIFEST_A"]),
            ("B_base.json", "none", i["CKPT_BASE"], i["MANIFEST_B"]),
            ("B_armD.json", i["ADAPTER_D"], i["CKPT_D"], i["MANIFEST_B"]),
            ("B_analysis.json", None, None, i["MANIFEST_B"])]
    assert [Path(_flag(a, "--out")).name for a in calls] == [w[0] for w in want]
    for argv, (out, adapter, ckpt, man) in zip(calls, want):
        assert _flag(argv, "--manifest") == man
        if adapter is None:
            assert argv[:2] == ["-m", "track_a.esm_sft.diagnostic_analyze"]
            stage = out[0]
            assert _flag(argv, "--results") == f"base={dbox.out}/{stage}_base.json"
            assert f"armD={dbox.out}/{stage}_armD.json" in argv
        else:
            assert argv[:2] == ["-m", "track_a.esm_sft.diagnostic_eval"]
            assert _flag(argv, "--adapter") == adapter and _flag(argv, "--ckpt") == ckpt
            for f, v in (("--task", "as_paired_i"), ("--tcr_variant", "cdr3"), ("--mhc_variant", "full"),
                         ("--max_len", "1024"), ("--batch-size", "100"), ("--precision", "native")):
                assert _flag(argv, f) == v
            assert _flag(argv, "--max-seconds") == ("600" if out[0] == "A" else "3000")
            assert "--allow-cpu" in argv
    s = dbox.status()
    assert s["outcome"] == "complete" and s["completed"] == "preflight A B"
    assert (dbox.out / "run.identity").read_text().split()[0] == "run/launch"


def test_a_failing_step_stops_the_run_and_names_its_stage(dbox):
    r = dbox.run(STUB_FAIL="B_base.json")
    assert r.returncode == 1, dbox.log()
    names = [Path(_flag(c["argv"], "--out")).name for c in dbox.calls()]
    assert names == ["A_base.json", "A_armD.json", "A_analysis.json", "B_base.json"]
    s = dbox.status()
    assert s["outcome"] == "failed" and s["stage"] == "B" and s["completed"] == "preflight A"


def test_an_expired_ceiling_stops_before_the_next_gpu_step(dbox):
    r = dbox.run(MAX_HOURS="0")
    assert r.returncode == 1, dbox.log()
    assert "ceiling reached" in dbox.log()
    assert not dbox.calls()
    assert dbox.status()["outcome"] == "failed"


def test_the_ceiling_kills_an_overrunning_step(dbox):
    t0 = time.time()
    r = dbox.run(MAX_HOURS="0.0025", STUB_SLOW="A_base.json", STUB_SLEEP="40")
    assert r.returncode == 1, dbox.log()
    assert time.time() - t0 < 30, "the driver waited out its own child"
    pids = [c["pid"] for c in dbox.calls()]
    assert len(pids) == 1
    time.sleep(0.5)
    assert not _alive(pids[0]), "the overrunning step outlived the ceiling"
    assert not (dbox.out / "A_base.json").exists()
    assert dbox.status()["stage"] == "A"


def test_an_existing_output_is_skipped_not_overwritten(dbox):
    dbox.out.mkdir()
    sha = lambda k: hashlib.sha256(Path(dbox.inputs[k]).read_bytes()).hexdigest()
    earlier = json.dumps({"protocol": "judge-diagnostic-results-v1", "manifest_sha256": sha("MANIFEST_A"),
                          "checkpoint_sha256": sha("CKPT_BASE"), "note": "EARLIER"})
    (dbox.out / "A_base.json").write_text(earlier)
    r = dbox.run()
    assert r.returncode == 0, dbox.log()
    names = [Path(_flag(c["argv"], "--out")).name for c in dbox.calls()]
    assert "A_base.json" not in names and len(names) == 5
    assert (dbox.out / "A_base.json").read_text() == earlier


def test_runs_through_the_launch_wrapper(dbox, tmp_path):
    """The wrapper must find this driver's identity and release it with the handshake."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "nvidia-smi").write_text("#!/bin/sh\necho 0\n")
    watch = tmp_path / "watch.sh"
    watch.write_text('#!/bin/bash\necho "armed: pid=$1 max_hours=$MAX_HOURS" >> "$LOG"\nsleep 20\n')
    for f in (bin_ / "nvidia-smi", watch):
        f.chmod(0o755)
    root = tmp_path / "root"
    r = subprocess.run(["bash", str(WRAPPER)], capture_output=True, text=True, timeout=120,
                       env=dbox.env(O=str(root), JF=str(DRIVER), AUTOSTOP=str(watch),
                                    WLOG=str(tmp_path / "watch.log"), ARM_TIMEOUT="10",
                                    PATH=f"{bin_}:{os.environ['PATH']}"))
    assert r.returncode == 0, r.stdout[-2000:]
    (run,) = sorted((root / "runs").iterdir())
    assert json.loads((run / "status.json").read_text())["outcome"] == "complete"
    assert (run / "B_analysis.json").exists()
    pid = (run / "run.identity").read_text().split()[1]
    assert f"armed: pid={pid} " in (tmp_path / "watch.log").read_text()


def test_each_step_is_wrapped_in_timeout_when_present(dbox, tmp_path):
    """The scorer's --max-seconds is a between-batch check; the external kill is coreutils."""
    bin_ = tmp_path / "tbin"
    bin_.mkdir()
    log = tmp_path / "timeout.log"
    (bin_ / "timeout").write_text(f'#!/bin/bash\necho "$1 $2 $3" >> "{log}"\nshift 3\nexec "$@"\n')
    (bin_ / "timeout").chmod(0o755)
    r = dbox.run(PATH=f"{bin_}:{os.environ['PATH']}")
    assert r.returncode == 0, dbox.log()
    assert log.read_text().splitlines() == ["-k 30 720", "-k 30 720", "-k 30 600",
                                            "-k 30 3120", "-k 30 3120", "-k 30 600"]


def test_a_step_ended_by_its_own_time_limit_is_not_reported_as_the_ceiling(dbox, tmp_path):
    """coreutils `timeout` exits 124, the status run_step also returns at MAX_HOURS. The record
    must say which of the two stopped the step."""
    bin_ = tmp_path / "tbin"
    bin_.mkdir()
    (bin_ / "timeout").write_text("#!/bin/bash\nexit 124\n")
    (bin_ / "timeout").chmod(0o755)
    r = dbox.run(PATH=f"{bin_}:{os.environ['PATH']}")
    assert r.returncode == 1, dbox.log()
    s = dbox.status()
    assert s["outcome"] == "failed" and s["stage"] == "A"
    assert "720 s time limit" in s["detail"] and "ceiling" not in s["detail"], s["detail"]


def test_a_ceiling_above_the_frozen_two_hours_refuses(dbox):
    """The wrapper defaults MAX_HOURS to 8; this visit's ceiling is frozen at 2 in the plan."""
    r = dbox.run(MAX_HOURS="8")
    assert r.returncode == 2 and "MAX_HOURS" in r.stderr and dbox.calls() == []
    assert json.loads((dbox.out / "status.json").read_text())["outcome"] == "env_refused"


def test_a_wrong_stage_a_manifest_sha_refuses_too(dbox):
    r = dbox.run(MANIFEST_A_SHA256="0" * 64)
    assert r.returncode == 2 and "MANIFEST_A" in dbox.log() and not dbox.calls()


def test_an_unwritable_status_refuses_before_any_scoring(dbox):
    dbox.out.mkdir()
    (dbox.out / "status.json").mkdir()
    r = dbox.run()
    assert r.returncode != 0 and not dbox.calls()
    assert "status.json" in r.stderr + dbox.log()


def test_a_launch_outside_the_wrapper_is_refused_on_a_gpu_visit(dbox, tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "nvidia-smi").write_text("#!/bin/sh\necho 0\n")
    (bin_ / "nvidia-smi").chmod(0o755)
    r = dbox.run(ALLOW_CPU="", PATH=f"{bin_}:{os.environ['PATH']}")
    assert r.returncode != 0 and not dbox.calls()
    assert "launch_judge_first.sh" in r.stderr + dbox.log()


@pytest.mark.parametrize("over", [{"MAX_SECONDS_B": "."}, {"KILL_SLACK": "."}, {"POLL": "x"}, {"POLL": "0.5"}, {"STEP_POLL": "x"},
                                  {"SHUTDOWN_GRACE": "x"}, {"MAX_HOURS": "."}])
def test_a_knob_that_is_not_a_number_is_refused(dbox, over):
    r = dbox.run(**over)
    assert r.returncode == 2 and not dbox.calls()
    assert dbox.status()["outcome"] == "env_refused" and f"{list(over)[0]}=" in r.stderr and "is not a" in r.stderr


def test_a_quote_in_a_path_is_refused_rather_than_written_into_status(dbox, tmp_path):
    r = dbox.run(O=str(tmp_path / 'o"ut'))
    assert r.returncode != 0 and not dbox.calls()


def test_a_signal_mid_step_is_recorded_as_terminated(dbox):
    proc = subprocess.Popen(["bash", str(DRIVER)], env=dbox.env(STUB_SLOW="A_base.json", STUB_SLEEP="60",
                                                               SHUTDOWN_GRACE="2"),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t0 = time.time()
    while not dbox.calls():
        assert time.time() - t0 < 30
        time.sleep(0.1)
    proc.terminate()
    proc.wait(60)
    assert proc.returncode == 130 and dbox.status()["outcome"] == "terminated"
    time.sleep(0.5)
    assert not _alive(dbox.calls()[0]["pid"])


def test_a_ceiling_kill_is_named_as_such_in_the_status(dbox):
    r = dbox.run(MAX_HOURS="0.0025", STUB_SLOW="A_base.json", STUB_SLEEP="40")
    # The exact phrase: the log path in a failure's detail holds this test's own name.
    assert r.returncode == 1 and "stopped at the ceiling" in dbox.status()["detail"]


# ------------------------------------------------------------------------------ after the Codex review
def test_a_stale_or_empty_output_is_refused_not_counted_as_scored(dbox):
    dbox.out.mkdir()
    (dbox.out / "A_base.json").write_text("{}")             # exists, says nothing about what it scored
    r = dbox.run()
    assert r.returncode != 0 and not dbox.calls()
    assert "A_base.json" in dbox.log() and dbox.status()["outcome"] != "complete"


def test_a_rerun_reuses_only_outputs_of_this_manifest_and_checkpoint(dbox):
    assert dbox.run().returncode == 0
    n = len(dbox.calls())
    assert dbox.run().returncode == 0 and len(dbox.calls()) == n       # everything reused
    Path(dbox.inputs["CKPT_D"]).write_text("another checkpoint\n")    # the arm D checkpoint changes
    r = dbox.run()
    assert r.returncode != 0 and len(dbox.calls()) == n
    assert "A_armD.json" in dbox.log()


def test_allow_cpu_does_not_excuse_the_watchdog_on_a_box_with_a_gpu(dbox, tmp_path):
    bin_ = tmp_path / "gpubin"
    bin_.mkdir()
    (bin_ / "nvidia-smi").write_text("#!/bin/sh\necho 0\n")
    (bin_ / "nvidia-smi").chmod(0o755)
    r = dbox.run(PATH=f"{bin_}:{os.environ['PATH']}")
    assert r.returncode != 0 and not dbox.calls()


def test_a_stale_analysis_is_refused_too(dbox):
    assert dbox.run().returncode == 0
    n = len(dbox.calls())
    (dbox.out / "A_analysis.json").write_text("{}")
    r = dbox.run()
    assert r.returncode != 0 and len(dbox.calls()) == n
    assert "A_analysis.json" in dbox.log() and dbox.status()["outcome"] == "failed"
