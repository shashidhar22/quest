"""The launch wrapper must know which job it started, and prove it.

Round-two review of 2026-11-06, findings 1, 2 and 5. The wrapper I added to close the previous
round's watchdog finding introduced a new instance of the same failure family: a STALE artifact
readable as a current one.

  1. It cleared `watchdog.armed` but not `judge_first.pid`, then read whatever pid file it found.
     With a live stale pid planted, it armed the watchdog on that pid and released the new
     driver -- so the watchdog was watching a process that was not the job, which is exactly the
     2026-10-03 failure it exists to prevent. Worse than the review stated: the arm-failure
     branch then does `kill -TERM` on that unverified pid, so a stale file makes the wrapper
     signal an unrelated process. With a dead stale pid it exited 2 and left the driver it had
     just started alive and unwatched.
  2. `set -m` makes every background job a process-group leader, and util-linux `setsid` returns
     EPERM from `setsid()` in that case, forks, and the parent exits 0 at once. So on any box
     that HAS setsid -- the training box -- `$RUN` was not the driver and the wrapper's exit
     status was not the job's. This laptop has no setsid, which is why the tests passed.
  3. One output directory served every visit, so a previous `comparison.json` survived a later
     refused run and would be published beside it as if it belonged to it.

The cure for all three is run identity: a fresh per-invocation directory, a pid published by
this invocation with a token this invocation generated, a supervisor whose exit status is the
job's, and an explicit status manifest.

The setsid cases use a shim implementing the published util-linux fork/setsid/exec semantics,
as the reviewer's probe did. It is a source-equivalent double, NOT a native Linux test; the
native check stays on the box list.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import _alive

WRAPPER = "scripts/training/v04_box/launch_judge_first.sh"

# util-linux setsid: setsid() fails with EPERM when the caller already leads its process group,
# and it then forks and the parent returns success unless --wait was given.
# sys-utils/setsid.c, fork/parent/wait logic.
SETSID_SHIM = """#!{py}
import os, sys
args = sys.argv[1:]
wait = False
while args and args[0] in ("--wait", "-w"):
    {wait_body}
    args.pop(0)
if os.getpgrp() == os.getpid():
    pid = os.fork()
    if pid:
        if not wait:
            os._exit(0)
        _, status = os.waitpid(pid, 0)
        os._exit(os.waitstatus_to_exitcode(status) if status >= 0 else 1)
    os.setsid()
else:
    os.setsid()
os.execvp(args[0], args)
"""


def _shim(path, supports_wait=True):
    body = "wait = True" if supports_wait else (
        'sys.stderr.write("setsid: unrecognized option\\n"); os._exit(1)')
    path.write_text(SETSID_SHIM.format(py=sys.executable, wait_body=body))
    path.chmod(0o755)


class Harness:
    """A mock driver and watchdog, so the wrapper's own logic is what is under test."""

    def __init__(self, tmp_path, repo, *, driver_sleep=3, driver_rc=9, record_identity=True,
                 watchdog_arms=True, setsid=None, startup_delay=1):
        self.p = tmp_path
        self.root = repo
        self.out = tmp_path / "out"
        self.out.mkdir(parents=True, exist_ok=True)
        self.job = tmp_path / "job.sh"
        ident = ('[ -n "${RUN_TOKEN:-}" ] && echo "$RUN_TOKEN $$" > "$O/run.identity"\n'
                 if record_identity else "")
        # The startup delay is what makes a stale pid file dangerous: without it the real pid
        # is in place before the wrapper looks, and the race the review found never happens.
        # The mock honours the same contract as the real launcher: under REQUIRE_WATCHDOG it
        # waits for the handshake before doing any work. The wait is long on purpose, so that
        # any early death is the wrapper's doing and the reaping tests cannot pass by the
        # driver giving up on its own.
        self.job.write_text(
            '#!/bin/bash\nset -u\n'
            f'sleep {startup_delay}\n'
            'echo $$ > "$O/judge_first.pid"\n'
            + ident +
            'if [ -n "${REQUIRE_WATCHDOG:-}" ]; then\n'
            '  w=0\n'
            '  while [ ! -f "$O/watchdog.armed" ] && [ "$w" -lt 120 ]; do sleep 1; w=$((w+1)); done\n'
            '  [ -f "$O/watchdog.armed" ] || { echo "no handshake"; exit 2; }\n'
            'fi\n'
            f'sleep {driver_sleep}\n'
            'echo "$$" > "$O/driver-finished"\n'
            f'exit {driver_rc}\n')
        self.wlog = tmp_path / "watch.log"
        self.watcher = tmp_path / "watch.sh"
        self.watcher.write_text(
            '#!/bin/bash\n' +
            ('echo "armed: pid=$1 max_hours=$MAX_HOURS" >> "$LOG"\nsleep 30\n'
             if watchdog_arms else 'echo "REFUSE: cannot arm" >> "$LOG"\nexit 2\n'))
        for f in (self.job, self.watcher):
            f.chmod(0o755)
        self.bin = tmp_path / "bin"
        self.bin.mkdir(exist_ok=True)
        # The launcher refuses a box with no nvidia-smi before anything starts, so the harness
        # supplies a mock one; the mock watchdog itself never calls it.
        nv = self.bin / "nvidia-smi"
        nv.write_text("#!/bin/sh\necho 0\n")
        nv.chmod(0o755)
        if setsid is not None:
            _shim(self.bin / "setsid", supports_wait=setsid)
        self.proc = None

    def env(self, **over):
        e = dict(os.environ)
        e.update(Q=str(self.root), O=str(self.out), JF=str(self.job),
                 AUTOSTOP=str(self.watcher), WLOG=str(self.wlog), ARM_TIMEOUT="8",
                 PATH=f"{self.bin}:{os.environ['PATH']}")
        e.update({k: str(v) for k, v in over.items()})
        return e

    def run(self, timeout=90, **over):
        return subprocess.run(["bash", str(self.root / WRAPPER)], env=self.env(**over),
                              capture_output=True, text=True, timeout=timeout)

    def popen(self, **over):
        self.proc = subprocess.Popen(["bash", str(self.root / WRAPPER)], env=self.env(**over),
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, start_new_session=True)
        return self.proc

    # -- observation ---------------------------------------------------------
    def run_dirs(self):
        d = self.out / "runs"
        return sorted(p for p in d.iterdir() if p.is_dir()) if d.is_dir() else []

    def armed_pids(self):
        if not self.wlog.exists():
            return []
        return [l.split("pid=")[1].split()[0] for l in self.wlog.read_text().splitlines()
                if "armed: pid=" in l]

    def kill(self):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), 9)
            except (ProcessLookupError, PermissionError):
                pass


@pytest.fixture
def harness(tmp_path):
    made = []

    def make(**kw):
        h = Harness(tmp_path, Path(__file__).resolve().parents[1], **kw)
        made.append(h)
        return h

    yield make
    for h in made:
        h.kill()


def _live_sleeper():
    """A process this test owns, standing in for whatever the stale pid file points at.

    Liveness is checked with `Popen.poll()`, never `os.kill(pid, 0)`: this is pytest's own
    unreaped child, so after a signal it becomes a zombie whose process-table entry still
    answers signal 0. `conftest._alive` would therefore report a terminated process as alive and
    the test would pass for exactly the wrong reason.
    """
    return subprocess.Popen(["sleep", "120"])


# --------------------------------------------------------------- finding 1: identity
def test_a_live_stale_pid_file_is_never_armed_on(harness):
    """The review's first case. The watchdog must be armed on the driver this launch started."""
    stale = _live_sleeper()
    try:
        h = harness()
        (h.out / "judge_first.pid").write_text(str(stale.pid))
        r = h.run()
        armed = h.armed_pids()
        assert armed, f"the watchdog was never armed: {r.stdout[-1500:]}"
        assert str(stale.pid) not in armed, (
            f"the watchdog was armed on the stale pid {stale.pid}: {armed}")
        dirs = h.run_dirs()
        assert len(dirs) == 1, f"expected one run directory, got {dirs}"
        real = (dirs[0] / "judge_first.pid").read_text().strip()
        assert armed == [real], f"armed {armed}, but this run's driver was {real}"
    finally:
        stale.kill()
        stale.wait()


def test_a_stale_pid_is_never_signalled_when_arming_fails(harness):
    """`kill -TERM $PID` on a pid read from a file this launch did not write is the worst form
    of the same defect: it terminates an unrelated process."""
    stale = _live_sleeper()
    try:
        h = harness(watchdog_arms=False)
        (h.out / "judge_first.pid").write_text(str(stale.pid))
        r = h.run()
        assert r.returncode == 2, r.stdout[-1500:]
        time.sleep(1)
        assert stale.poll() is None, (
            f"the wrapper signalled a process it did not start "
            f"(stale pid {stale.pid} exited with {stale.poll()})")
    finally:
        stale.kill()
        stale.wait()


def test_a_failure_to_arm_reaps_the_driver_this_launch_started(harness):
    """Nothing may be left running unwatched -- including by the branch that refuses."""
    h = harness(watchdog_arms=False, driver_sleep=30)
    r = h.run()
    assert r.returncode == 2, r.stdout[-1500:]
    dirs = h.run_dirs()
    assert len(dirs) == 1
    pid = int((dirs[0] / "judge_first.pid").read_text().strip())
    time.sleep(2)
    assert not _alive(pid), "the unwatched driver was left running"
    assert not (dirs[0] / "watchdog.armed").exists()


def test_a_driver_that_never_publishes_its_identity_is_reaped(harness):
    h = harness(record_identity=False, driver_sleep=30)
    r = h.run(ARM_TIMEOUT="3", IDENTITY_TIMEOUT="3")
    assert r.returncode == 2, r.stdout[-1500:]
    assert "identity" in r.stdout
    pid = int((h.run_dirs()[0] / "judge_first.pid").read_text().strip())
    time.sleep(2)
    assert not _alive(pid), "a driver with no verified identity was left running"


def test_two_launches_get_separate_run_directories(harness):
    h = harness(driver_sleep=1, driver_rc=0)
    assert h.run().returncode == 0
    assert h.run().returncode == 0
    dirs = h.run_dirs()
    assert len(dirs) == 2, f"both launches shared a directory: {dirs}"
    assert {d.name for d in dirs} == {d.name for d in dirs}
    for d in dirs:
        assert (d / "judge_first.pid").exists()


def test_a_concurrent_launch_in_the_same_output_root_is_refused(harness):
    """Two drivers in one output root would interleave their artifacts and their watchdogs."""
    h = harness(driver_sleep=25)
    first = h.popen()
    deadline = time.time() + 30
    while time.time() < deadline and not h.armed_pids():
        time.sleep(0.2)
    assert h.armed_pids(), "the first launch never armed, so the lock was not exercised"
    second = h.run(timeout=60)
    assert second.returncode == 2, second.stdout[-1500:]
    assert "already active" in second.stdout
    assert len(h.run_dirs()) == 1, "the refused launch still created a run directory"
    first.wait(timeout=60)


# --------------------------------------------------------------- finding 2: completion
@pytest.mark.parametrize("supports_wait", [True, False])
def test_the_wrapper_waits_for_the_driver_and_returns_its_status(harness, supports_wait):
    """util-linux setsid forks when already a process-group leader and its parent returns 0.

    With the shim present and --wait supported, the wrapper must use it; where --wait is
    unsupported it must not use setsid for the driver at all. Either way its exit status is the
    driver's and it does not return early. Before the repair the shimmed run returned 0 while
    the driver was still running, and the driver then exited 9.
    """
    h = harness(setsid=supports_wait, driver_sleep=4, driver_rc=9)
    r = h.run()
    d = h.run_dirs()[0]
    assert (d / "driver-finished").exists(), (
        f"the wrapper returned before the driver finished: {r.stdout[-1500:]}")
    assert r.returncode == 9, (
        f"the wrapper reported {r.returncode}, not the driver's 9: {r.stdout[-1500:]}")


def test_the_supervisor_choice_is_logged(harness):
    """Whether setsid is in use changes what the exit status MEANS, so it is not implicit.

    The assertion is anchored to a log line, not to the bare word: pytest names its temporary
    directory after the test, so `"supervisor" in r.stdout` matched this test's own tmp path and
    passed before the feature existed. A check that cannot fail proves nothing -- the same note
    this round made about the watchdog's pgrep case, and I had just written one myself.
    """
    h = harness(setsid=True, driver_sleep=1, driver_rc=0)
    r = h.run()
    lines = [l for l in r.stdout.splitlines() if "[launch] driver supervisor:" in l]
    assert len(lines) == 1, f"no supervisor line in:\n{r.stdout[-1200:]}"
    assert "setsid --wait" in lines[0], lines[0]

    h2 = harness(setsid=False, driver_sleep=1, driver_rc=0)
    r2 = h2.run()
    line = [l for l in r2.stdout.splitlines() if "[launch] driver supervisor:" in l]
    assert line and "setsid" not in line[0].replace("setsid --wait", ""), (
        f"a setsid without --wait must not be used for the driver: {line}")


# --------------------------------------------------------------- finding 5: retention
def test_a_previous_result_cannot_be_published_as_this_runs(harness):
    """The reviewer's last probe: a prior comparison survived a later refusal in one directory."""
    h = harness(driver_sleep=1, driver_rc=0)
    (h.out / "comparison.json").write_text('{"old_result": true}')
    assert h.run().returncode == 0
    d = h.run_dirs()[0]
    assert not (d / "comparison.json").exists(), (
        "a previous comparison appeared inside this run's own directory")
    assert (h.out / "comparison.json").read_text() == '{"old_result": true}', (
        "the previous result was destroyed rather than left where it was")


def test_the_launch_records_a_status_manifest(harness):
    h = harness(driver_sleep=1, driver_rc=0)
    r = h.run()
    assert r.returncode == 0
    st = json.loads((h.run_dirs()[0] / "launch_status.json").read_text())
    assert st["run_id"]
    assert st["driver_rc"] == 0
    assert st["watchdog_armed_pid"] == (h.run_dirs()[0] / "judge_first.pid").read_text().strip()
    assert st["outcome"] == "driver_exited_0"


def test_a_refused_launch_records_why(harness):
    h = harness(watchdog_arms=False, driver_sleep=5)
    h.run()
    st = json.loads((h.run_dirs()[0] / "launch_status.json").read_text())
    assert st["outcome"] == "watchdog_not_armed", st
    assert st["driver_rc"] is None or isinstance(st["driver_rc"], int)


# --------------------------------------------------------------- pre-launch code review
def test_a_resumed_run_reuses_its_directory_with_a_fresh_identity(harness):
    """Every launch made a fresh directory, so judge_first's validated reuse was unreachable
    through the only supported launch command and an interrupted visit paid again in full."""
    h = harness(watchdog_arms=False, driver_sleep=30)
    assert h.run().returncode == 2
    d = h.run_dirs()[0]
    old_token = (d / "run.identity").read_text().split()[0]
    h2 = harness(driver_sleep=1, driver_rc=0)
    r = h2.run(RESUME=d.name)
    assert r.returncode == 0, r.stdout[-1500:]
    assert h2.run_dirs() == [d], "a resume must not create a second directory"
    new_token, new_pid = (d / "run.identity").read_text().split()
    assert new_token != old_token, "a resumed launch reused the old identity token"
    assert h2.armed_pids() == [new_pid]


def test_a_stale_handshake_in_a_resumed_directory_cannot_release_the_driver(harness):
    """An old watchdog.armed in a reused directory would let the new driver start GPU work
    before this launch had armed anything."""
    h = harness(watchdog_arms=False, driver_sleep=30)
    h.run()
    d = h.run_dirs()[0]
    (d / "watchdog.armed").touch()                  # left by an earlier launch
    (d / "driver-finished").unlink(missing_ok=True)
    r = h.run(RESUME=d.name)
    assert r.returncode == 2, r.stdout[-1500:]
    time.sleep(1)
    assert not (d / "driver-finished").exists(), (
        "the driver ran on a stale handshake while the watchdog had not armed")


def test_reusing_a_run_id_for_a_new_run_is_refused(harness):
    h = harness(driver_sleep=1, driver_rc=0)
    assert h.run(RUN_ID="fixed").returncode == 0
    r = h.run(RUN_ID="fixed")
    assert r.returncode == 2
    assert "already exists" in r.stdout


def test_resuming_a_run_that_does_not_exist_is_refused(harness):
    h = harness(driver_sleep=1, driver_rc=0)
    r = h.run(RESUME="no-such-run")
    assert r.returncode == 2
    assert h.run_dirs() == []


def test_a_lock_held_by_a_live_driver_is_respected_after_its_launcher_died(harness):
    """An SSM session drop kills the launcher but not the driver or its watchdog. A lock that
    tracked only the launcher would then be cleared as stale and a second driver started on
    the same single GPU."""
    driver = _live_sleeper()
    try:
        h = harness(driver_sleep=1, driver_rc=0)
        (h.out / "launch.lock").write_text(f"999999999 {driver.pid}\n")
        r = h.run()
        assert r.returncode == 2, r.stdout[-1500:]
        assert "already active" in r.stdout
        assert h.run_dirs() == []
    finally:
        driver.kill()
        driver.wait()


def test_a_lock_whose_every_holder_is_dead_is_cleared(harness):
    h = harness(driver_sleep=1, driver_rc=0)
    (h.out / "launch.lock").write_text("999999998 999999999\n")
    r = h.run()
    assert r.returncode == 0, r.stdout[-1500:]
    assert "stale lock" in r.stdout


@pytest.mark.parametrize("var", ["MAX_HOURS", "WATCHDOG_HOURS"])
def test_a_non_numeric_ceiling_is_refused_by_the_launcher(harness, var):
    """`abc` evaluated to 0 inside awk, so it became a silent zero ceiling, not a refusal."""
    h = harness(driver_sleep=1, driver_rc=0)
    r = h.run(**{var: "abc"})
    assert r.returncode == 2, r.stdout[-1000:]
    assert "not a number" in r.stdout
    assert h.run_dirs() == []


def test_a_box_without_nvidia_smi_is_refused_before_anything_starts(harness):
    """The watchdog cannot arm without nvidia-smi. Finding that out after a driver has been
    started means a driver to reap; finding it first means nothing to undo."""
    h = harness(driver_sleep=1, driver_rc=0)
    (h.bin / "nvidia-smi").unlink()
    r = h.run(NVSMI="definitely-not-nvidia-smi")
    assert r.returncode == 2
    assert "not found" in r.stdout
    assert h.run_dirs() == []
