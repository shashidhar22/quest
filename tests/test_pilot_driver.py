"""The step-12 pilot driver: lock first, CPT P and E, the pair check, three SFT cells, read-out.

Plan step 5 (plans/step12-diagnostic-and-pilot.md). The driver is the cost boundary of the
pilot visit, so these pin its order and its refusals: nothing trains off the lock, no GPU step
starts unwatched or past a ceiling, a failed check stops every later stage, and a rerun reuses
only what still passes its check.

Control-flow tests only. The trainer, the judge and the comparator are stand-ins; the lock
module is the real one, run by the real interpreter, against a sandbox copy of the locked code.
Nothing here says anything about GPU behaviour or numerics.
"""
from __future__ import annotations

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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "training" / "pilot12_box"))
import experiment_lock as L  # noqa: E402

DRIVER = "scripts/training/pilot12_box/run_pilot.sh"
WRAPPER = "scripts/training/v04_box/launch_judge_first.sh"
SANDBOX_EXTRA = ("quest/__init__.py", "quest/data/__init__.py", "track_a/__init__.py",
                 "track_a/shared/__init__.py", "track_a/esm_sft/__init__.py",
                 "scripts/training/v04_box/launch_judge_first.sh")

PY_STUB = r"""#!/bin/bash
# Stands in for the trainer, the judge and the comparator; every other call is the real
# interpreter, so the lock module's checks are exercised, not mocked.
case "${1:-}" in
  */train_cpt_stage2_single.py) shift; exec "$PYREAL" "$STUBS/fake_trainer.py" "$@" ;;
  */pilot_compare.py) shift; exec "$PYREAL" "$STUBS/fake_compare.py" "$@" ;;
esac
if [ "${1:-}" = -m ] && [ "${2:-}" = track_a.esm_sft.judge ]; then
  shift 2; exec "$PYREAL" "$STUBS/fake_judge.py" "$@"
fi
exec "$PYREAL" "$@"
"""

COMMON = r'''
import json, os, sys, time
def flags(argv):
    out, key = {}, None
    for tok in argv:
        if tok.startswith("--"):
            key = tok[2:]; out[key] = []
        elif key is not None:
            out[key].append(tok)
    return {k: (v[0] if len(v) == 1 else v) for k, v in out.items()}
def event(kind, **kw):
    with open(os.environ["EVENTS"], "a") as fh:
        fh.write(json.dumps({"kind": kind, **kw}) + "\n")
def num(v):
    for t in (int, float):
        try:
            return t(v)
        except (TypeError, ValueError):
            pass
    return v
'''

FAKE_TRAINER = COMMON + r'''
import hashlib
argv = sys.argv[1:]
f = flags(argv)
out = f["output_dir"]; arm = os.path.basename(out).split("_")[-1]
event("cpt", arm=arm, argv=argv, pid=os.getpid())
os.makedirs(out, exist_ok=True)
time.sleep(float(os.environ.get(f"STUB_CPT_SLEEP_{arm}", "0")))
rc = int(os.environ.get(f"STUB_CPT_RC_{arm}", "0"))
if rc:
    sys.exit(rc)
names = {"lr": "learning_rate", "eval_steps": "eval_every_n_steps", "save_steps": "save_every_n_steps"}
cfg = {names.get(k, k): ([num(x) for x in v] if isinstance(v, list) else num(v))
       for k, v in f.items() if k != "no_compile"}
cfg["use_compile"] = "no_compile" not in f
cfg["bucket_boundaries"] = [num(x) for x in f["bucket_boundaries"]]
marker = os.path.join(os.path.dirname(f["dataset_path"]), "PUBLISHED.json")
m = json.load(open(marker))
ident = {"run_id": m["run_id"], "contract_sha256": m["contract"]["sha256"],
         "contract_version": m["contract"]["version"],
         "marker_sha256": hashlib.sha256(open(marker, "rb").read()).hexdigest(),
         "benchmark_checked": False}
os.makedirs(os.path.join(out, "final"), exist_ok=True)
state = {"global_step": int(os.environ.get(f"STUB_STEPS_{arm}", cfg["max_steps"])),
         "masked_targets_seen": int(os.environ.get(f"STUB_TARGETS_{arm}", "2300000")),
         "initial_lora_sha256": os.environ.get(f"STUB_INIT_{arm}", "ab" * 32),
         "corpus_identity": ident, "loss_weighting_semantics": "global-masked-target-mean/v2",
         "config": cfg}
json.dump(state, open(os.path.join(out, "final", "trainer_state.json"), "w"))
import struct
_data = struct.pack("<2f", 1.0, float(ord(arm)))
_head = json.dumps({"lora_A": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
open(os.path.join(out, "final", "adapter_model.safetensors"), "wb").write(struct.pack("<Q", len(_head)) + _head + _data)
open(os.path.join(out, "final", "adapter_config.json"), "w").write('{"r": 32}')
event("cpt_end", arm=arm)
'''

FAKE_JUDGE = COMMON + r'''
sys.path.insert(0, os.environ["QROOT"])
from track_a.shared import artifacts as A
from track_a.esm_sft.scoring import SFT_OBJECTIVE, sft_recipe
sub, argv = sys.argv[1], sys.argv[2:]
f = flags(argv)
if sub == "train":
    cell = os.path.basename(f["out_dir"]).split("_")[-1]
    event("sft_train", cell=cell, argv=argv)
    os.makedirs(f["out_dir"], exist_ok=True)
    rc = int(os.environ.get(f"STUB_SFT_RC_{cell}", "0"))
    args = {k: num(v) for k, v in f.items() if k not in ("allow_cpu", "smoke")}
    if rc:
        open(os.path.join(f["out_dir"], "last.pt"), "w").write("{}")
        sys.exit(rc)
    json.dump({"args": args, "data_digests": A.split_digests(args["task"], ("train", "val"))},
              open(os.path.join(f["out_dir"], "best.pt"), "w"))
    sys.exit(0)
cell = os.path.basename(os.path.dirname(f["out"])).split("_")[-1]
event("sft_eval", cell=cell, argv=argv)
ck = json.load(open(f["ckpt"]))
conds = f["conditions"] if isinstance(f["conditions"], list) else [f["conditions"]]
args = {k: num(v) for k, v in f.items() if k not in ("allow_cpu", "smoke")}
args["conditions"] = conds
args["smoke"] = "smoke" in f                    # the real judge records vars(a), smoke included
scored = conds + ([os.environ[f"STUB_EVAL_EXTRA_{cell}"]] if os.environ.get(f"STUB_EVAL_EXTRA_{cell}") else [])
doc = {"args": args, "pool_size": 100,
       "conditions": {c: {"ranks": {"r1": 3}, "true_peptides": {"r1": "AAAAAAAAA"}} for c in scored},
       "provenance": {"pool_digest": A.file_digest(A.pool_path()),
                      "split_digests": A.split_digests(args["task"], conds),
                      "sft_objective": SFT_OBJECTIVE, "sft_seed": ck["args"].get("seed"),
                      "sft_max_len": ck["args"].get("max_len"), "sft_recipe": sft_recipe(ck["args"]),
                      "sft_train_digests": ck["data_digests"], **A.cpt_lineage(args["adapter"])}}
json.dump(doc, open(f["out"], "w"))
'''

FAKE_COMPARE = COMMON + r'''
argv = sys.argv[1:]
event("compare", argv=argv)
f = flags(argv)
if os.path.exists(f["out"]):
    sys.exit(2)
json.dump({"stub": True, "cells": f["cells"]}, open(f["out"], "x"))
'''


class Pilot:
    """A sandbox quest root holding the locked code, two corpora, splits, a lock and stubs."""

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.root = root = tmp_path / "quest"
        for rel in (*L.code_files(REPO), *SANDBOX_EXTRA):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / rel, root / rel)
        self.corpora = {}
        for arm, version in L.ARM_CORPUS.items():
            d = tmp_path / f"corpus_{arm}"
            (d / "dataset").mkdir(parents=True)
            (d / "PUBLISHED.json").write_text(json.dumps(
                {"run_id": f"run-{arm}", "contract": {"sha256": "c" * 64, "version": version}}))
            self.corpora[arm] = d
        self.splits = tmp_path / "splits"
        self.splits.mkdir()
        for cond in L.BENCHMARK_CONDITIONS:
            (self.splits / f"as_paired_i_{cond}.parquet").write_text(f"stand-in {cond}\n")
        (self.splits / "as_class_i_candidate_pool.tsv").write_text("peptide\nAAAAAAAAA\n")
        self.lock = tmp_path / "experiment-lock.json"
        subprocess.run([sys.executable, str(root / "scripts/training/pilot12_box/experiment_lock.py"),
                        "write", "--out", str(self.lock), "--corpus", f"P={self.corpora['P']}",
                        "--corpus", f"E={self.corpora['E']}", "--splits-dir", str(self.splits),
                        "--repo", str(root)], check=True, capture_output=True)
        self.stubs = tmp_path / "stubs"
        self.stubs.mkdir()
        for name, src in (("fake_trainer.py", FAKE_TRAINER), ("fake_judge.py", FAKE_JUDGE),
                          ("fake_compare.py", FAKE_COMPARE)):
            (self.stubs / name).write_text(src)
        self.py = self.stubs / "python"
        self.py.write_text(PY_STUB)
        self.py.chmod(0o755)
        self.events = tmp_path / "events.jsonl"
        self.events.touch()
        self.out = tmp_path / "out"
        self._live = []

    def env(self, **over):
        e = dict(os.environ)
        e.update(Q=str(self.root), O=str(self.out), PY=str(self.py), EXPERIMENT_LOCK=str(self.lock),
                 CORPUS_P=str(self.corpora["P"]), CORPUS_E=str(self.corpora["E"]),
                 SPLITS_DIR=str(self.splits), CANONICAL_EVAL=str(self.tmp / "canonical"),
                 MAX_HOURS="1", ALLOW_CPU="1", STEP_POLL="0.1", PYREAL=sys.executable,
                 STUBS=str(self.stubs), EVENTS=str(self.events), QROOT=str(self.root))
        for k in [k for k, v in over.items() if v is None]:
            e.pop(k, None)
        e.update({k: str(v) for k, v in over.items() if v is not None})
        return e

    def run(self, timeout=120, **over):
        return subprocess.run(["bash", str(self.root / DRIVER)], env=self.env(**over),
                              capture_output=True, text=True, timeout=timeout)

    def ev(self, kind=None):
        rows = [json.loads(l) for l in self.events.read_text().splitlines() if l.strip()]
        return [r for r in rows if kind is None or r["kind"] == kind]

    def gpu(self):
        return [r for r in self.ev() if r["kind"] in ("cpt", "sft_train", "sft_eval")]

    def status(self):
        return json.loads((self.out / "status.json").read_text())

    def log(self):
        p = self.out / "run.log"
        return p.read_text() if p.exists() else ""


@pytest.fixture
def pilot(tmp_path):
    return Pilot(tmp_path)


def flagval(argv, flag):
    return argv[argv.index(flag) + 1]


# ------------------------------------------------------------------------------ refusals up front
def test_unset_environment_is_refused_naming_every_variable(pilot):
    r = pilot.run(EXPERIMENT_LOCK=None, CORPUS_E=None, CANONICAL_EVAL=None)
    assert r.returncode != 0
    for name in ("EXPERIMENT_LOCK", "CORPUS_E", "CANONICAL_EVAL"):
        assert name in r.stderr, r.stderr
    assert not pilot.gpu()


def test_a_lock_that_fails_verify_stops_before_any_training(pilot):
    judge = pilot.root / "track_a/esm_sft/judge.py"
    judge.write_text(judge.read_text() + "\n# edited on the box\n")
    r = pilot.run()
    assert r.returncode != 0
    assert "code_files.track_a/esm_sft/judge.py" in pilot.log()
    assert not pilot.gpu()
    st = pilot.status()
    assert st["outcome"] == "failed" and st["stages"]["verify"] == "failed"
    assert st["pilot"] is None


def test_max_hours_above_the_locks_ceiling_is_refused(pilot):
    r = pilot.run(MAX_HOURS="25")
    assert r.returncode != 0
    assert "visit_max_hours" in pilot.log()
    assert not pilot.gpu()


def test_cpt_max_hours_above_the_locks_ceiling_is_refused(pilot):
    r = pilot.run(CPT_MAX_HOURS="3")
    assert r.returncode != 0
    assert "cpt_stage_max_hours" in pilot.log()
    assert not pilot.gpu()


def test_no_gpu_step_without_the_watchdog_handshake(pilot):
    r = pilot.run(REQUIRE_WATCHDOG="1", ARM_WAIT="2")
    assert r.returncode != 0
    assert "no watchdog handshake" in pilot.log()
    assert not pilot.gpu()
    assert pilot.status()["outcome"] == "no_watchdog"


def test_the_handshake_releases_the_run(pilot):
    pilot.out.mkdir()
    (pilot.out / "watchdog.armed").touch()
    r = pilot.run(REQUIRE_WATCHDOG="1", ARM_WAIT="2")
    assert r.returncode == 0, pilot.log()[-2000:]


# ------------------------------------------------------------------------------ the happy path
def test_the_happy_path_runs_every_stage_in_order_from_the_lock(pilot):
    r = pilot.run()
    assert r.returncode == 0, pilot.log()[-3000:] + r.stderr[-2000:]
    order = [(e["kind"], e.get("arm") or e.get("cell")) for e in pilot.ev()
             if e["kind"] in ("cpt", "sft_train", "sft_eval", "compare")]
    assert order == [("cpt", "P"), ("cpt", "E"),
                     ("sft_train", "B"), ("sft_eval", "B"),
                     ("sft_train", "P"), ("sft_eval", "P"),
                     ("sft_train", "E"), ("sft_eval", "E"), ("compare", None)]
    cpt = {e["arm"]: e["argv"] for e in pilot.ev("cpt")}
    assert flagval(cpt["P"], "--gradient_accumulation_steps") == "64"
    assert flagval(cpt["E"], "--gradient_accumulation_steps") == "47"
    for arm, argv in cpt.items():
        assert flagval(argv, "--max_steps") == "780"
        assert flagval(argv, "--loss_weighting") == "token_exact"
        assert flagval(argv, "--dataset_path") == str(pilot.corpora[arm] / "dataset")
        assert flagval(argv, "--output_dir") == str(pilot.out / f"cpt_{arm}")
        # The leakage-audit check is always on, against the lock's own benchmark hashes.
        bench = pilot.out / "benchmark_split_sha256.json"
        assert flagval(argv, "--corpus_benchmark_sha256") == str(bench)
        assert json.loads(bench.read_text()) == json.loads(pilot.lock.read_text())["benchmark_split_sha256"]
        assert flagval(argv, "--eval_steps") == "781"          # no evaluation before the final one
    train = {e["cell"]: e["argv"] for e in pilot.ev("sft_train")}
    assert flagval(train["B"], "--adapter") == "none"
    assert flagval(train["P"], "--adapter") == str(pilot.out / "cpt_P" / "final")
    assert flagval(train["E"], "--adapter") == str(pilot.out / "cpt_E" / "final")
    for argv in train.values():
        assert flagval(argv, "--lr") == "0.0001" and flagval(argv, "--seed") == "42"
    for e in pilot.ev("sft_eval"):
        assert flagval(e["argv"], "--conditions") == "val"
        assert flagval(e["argv"], "--distractor_draw") == "length_matched"
        assert flagval(e["argv"], "--max_queries") == "5000"
    cmp_argv = pilot.ev("compare")[0]["argv"]
    i = cmp_argv.index("--cells")
    assert cmp_argv[i + 1:i + 4] == [f"{c}={pilot.out / f'sft_{c}' / 'eval.json'}" for c in "BPE"]
    i = cmp_argv.index("--contrasts")
    assert cmp_argv[i + 1:i + 4] == ["E-B", "P-B", "E-P"]
    assert flagval(cmp_argv, "--val-split") == str(pilot.splits / "as_paired_i_val.parquet")
    assert flagval(cmp_argv, "--n-boot") == "4000"
    assert (pilot.out / "pilot.json").exists()
    st = pilot.status()
    assert st["outcome"] == "complete" and st["pilot"] == str(pilot.out / "pilot.json")
    assert st["lock_sha256"] == hashlib.sha256(pilot.lock.read_bytes()).hexdigest()
    assert st["stages"] == {"verify": "passed", "cpt_P": "ran", "cpt_E": "ran", "pair": "passed",
                            "sft_B": "ran", "sft_P": "ran", "sft_E": "ran", "compare": "ran"}


# ------------------------------------------------------------------------------ stage failures
@pytest.mark.parametrize("over", [{"STUB_INIT_E": "cd" * 32}, {"STUB_TARGETS_E": "2240000"}])
def test_a_failing_pair_check_stops_before_any_sft(pilot, over):
    r = pilot.run(**over)
    assert r.returncode != 0
    assert len(pilot.ev("cpt")) == 2
    assert not pilot.ev("sft_train") and not pilot.ev("sft_eval")
    assert pilot.status()["stages"]["pair"] == "failed"


def test_a_failing_first_cpt_stops_before_the_second(pilot):
    r = pilot.run(STUB_CPT_RC_P="3")
    assert r.returncode != 0
    assert [e["arm"] for e in pilot.ev("cpt")] == ["P"]
    assert not pilot.ev("sft_train")


def test_a_failing_second_cpt_stops_before_any_sft(pilot):
    r = pilot.run(STUB_CPT_RC_E="3")
    assert r.returncode != 0
    assert [e["arm"] for e in pilot.ev("cpt")] == ["P", "E"]
    assert not pilot.ev("sft_train")
    assert pilot.status()["stages"]["cpt_E"] == "failed"


def test_a_cpt_off_the_lock_fails_check_cpt(pilot):
    r = pilot.run(STUB_STEPS_P="700")
    assert r.returncode != 0
    assert "global_step" in pilot.log()
    assert [e["arm"] for e in pilot.ev("cpt")] == ["P"]


def test_a_failing_sft_cell_stops_the_later_cells(pilot):
    r = pilot.run(STUB_SFT_RC_P="5")
    assert r.returncode != 0
    assert [e["cell"] for e in pilot.ev("sft_train")] == ["B", "P"]
    assert not pilot.ev("compare")
    assert not (pilot.out / "pilot.json").exists()


# ------------------------------------------------------------------------------ ceilings
def test_an_expired_deadline_stops_before_the_next_gpu_step(pilot):
    r = pilot.run(MAX_HOURS="0")
    assert r.returncode != 0
    assert "ceiling" in pilot.log()
    assert not pilot.gpu()


def test_a_cpt_ceiling_of_zero_stops_before_the_cpt(pilot):
    r = pilot.run(CPT_MAX_HOURS="0")
    assert r.returncode != 0
    assert not pilot.gpu()


def test_a_step_running_past_the_deadline_is_terminated(pilot):
    t0 = time.monotonic()
    r = pilot.run(MAX_HOURS="0.002", STUB_CPT_SLEEP_P="60", SHUTDOWN_GRACE="2")
    assert time.monotonic() - t0 < 40
    assert r.returncode != 0
    assert "WHILE this step was running" in pilot.log()
    assert "stopped at the ceiling" in pilot.status()["detail"]   # the cause, not just "failed"
    pid = pilot.ev("cpt")[0]["pid"]
    assert not _alive(pid), "the overrunning trainer was left running"
    assert not pilot.ev("cpt_end")
    assert [e["arm"] for e in pilot.ev("cpt")] == ["P"]


# ------------------------------------------------------------------------------ resume
def test_a_rerun_reuses_completed_stages_and_trains_nothing(pilot):
    assert pilot.run().returncode == 0, pilot.log()[-2000:]
    pilot.events.write_text("")
    r = pilot.run()
    assert r.returncode == 0, pilot.log()[-2000:]
    assert not pilot.gpu(), pilot.gpu()
    st = pilot.status()
    assert st["outcome"] == "complete"
    assert {k: v for k, v in st["stages"].items() if k not in ("verify", "pair")} == {
        "cpt_P": "reused", "cpt_E": "reused", "sft_B": "reused", "sft_P": "reused",
        "sft_E": "reused", "compare": "reused"}


def test_a_rerun_after_a_failed_cell_resumes_at_that_cell(pilot):
    assert pilot.run(STUB_SFT_RC_E="5").returncode != 0
    shutil.rmtree(pilot.out / "sft_E")
    pilot.events.write_text("")
    r = pilot.run()
    assert r.returncode == 0, pilot.log()[-2000:]
    assert [e["cell"] for e in pilot.ev("sft_train")] == ["E"]
    assert not pilot.ev("cpt")


def test_partial_cpt_output_is_refused_not_overwritten(pilot):
    part = pilot.out / "cpt_P" / "checkpoint_step_100"
    part.mkdir(parents=True)
    (part / "adapter_model.safetensors").write_bytes(b"partial")
    r = pilot.run()
    assert r.returncode != 0
    assert "partial" in pilot.log()
    assert not pilot.gpu()
    assert (part / "adapter_model.safetensors").read_bytes() == b"partial"


def test_partial_sft_output_is_refused_not_trained_on(pilot):
    assert pilot.run(STUB_SFT_RC_B="5").returncode != 0
    pilot.events.write_text("")
    r = pilot.run()
    assert r.returncode != 0
    assert "partial" in pilot.log()
    assert not pilot.ev("sft_train") and not pilot.ev("sft_eval")


def test_a_reused_cpt_that_no_longer_passes_is_refused(pilot):
    assert pilot.run().returncode == 0
    st = pilot.out / "cpt_E" / "final" / "trainer_state.json"
    doc = json.loads(st.read_text())
    doc["global_step"] = 10
    st.write_text(json.dumps(doc))
    pilot.events.write_text("")
    r = pilot.run()
    assert r.returncode != 0
    assert not pilot.gpu()
    assert json.loads(st.read_text())["global_step"] == 10
    # Refused AT the arm: the adapter digest in a later cell's check would also notice, but only
    # after the pair check had passed on a state that no longer matches the lock.
    assert pilot.status()["stages"].get("cpt_E") == "failed" and "pair" not in pilot.status()["stages"]


# ------------------------------------------------------------------------------ the wrapper
def test_it_runs_under_the_launch_wrapper(pilot, tmp_path):
    """The identity and handshake contract of launch_judge_first.sh, with a mock watchdog."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "nvidia-smi").write_text("#!/bin/sh\necho 0\n")
    watch = tmp_path / "watch.sh"
    watch.write_text('#!/bin/bash\necho "armed: pid=$1 max_hours=$MAX_HOURS" >> "$LOG"\n'
                     'while kill -0 "$1" 2>/dev/null; do sleep 0.2; done\n')
    for f in (bin_ / "nvidia-smi", watch):
        f.chmod(0o755)
    env = pilot.env(JF=str(pilot.root / DRIVER), AUTOSTOP=str(watch), ARM_TIMEOUT="10",
                    PATH=f"{bin_}:{os.environ['PATH']}")
    r = subprocess.run(["bash", str(pilot.root / WRAPPER)], env=env, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-3000:]
    run_dir = next((pilot.out / "runs").iterdir())
    assert (run_dir / "pilot.json").exists()
    token, pid = (run_dir / "run.identity").read_text().split()
    assert token.startswith(run_dir.name + "/")
    assert json.loads((run_dir / "status.json").read_text())["outcome"] == "complete"


# ------------------------------------------------------------------------------ after verification
def _start(pilot, **over):
    return subprocess.Popen(["bash", str(pilot.root / DRIVER)], env=pilot.env(**over),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _wait_for(pilot, kind, n=1, seconds=30):
    t0 = time.monotonic()
    while len(pilot.ev(kind)) < n:
        assert time.monotonic() - t0 < seconds, f"no {kind} event"
        time.sleep(0.1)


def _edit_lock(pilot):
    d = json.loads(pilot.lock.read_text())
    d["cpt"]["gradient_accumulation_steps"]["E"] = 48
    pilot.lock.write_text(json.dumps(d, indent=1))


def _edit_code(pilot):
    judge = pilot.root / "track_a/esm_sft/judge.py"
    judge.write_text(judge.read_text() + "\n# edited during the run\n")


@pytest.mark.parametrize("edit", [_edit_lock, _edit_code])
def test_the_lock_is_held_for_the_whole_run_not_only_at_the_start(pilot, edit):
    """Verified once and then re-read was not a lock: an edit during CPT P trained E off it."""
    proc = _start(pilot, STUB_CPT_SLEEP_P="3")
    _wait_for(pilot, "cpt")
    edit(pilot)
    proc.wait(120)
    assert proc.returncode != 0
    assert [e["arm"] for e in pilot.ev("cpt")] == ["P"], "an arm trained after the lock stopped matching"
    assert not pilot.ev("sft_train")
    assert pilot.status()["outcome"] == "failed"


def test_a_run_directory_started_under_one_lock_is_not_continued_under_another(pilot):
    assert pilot.run().returncode == 0
    d = json.loads(pilot.lock.read_text())
    d["created_utc"] = "2030-01-01T00:00:00Z"            # a regenerated lock: same values, other bytes
    pilot.lock.write_text(json.dumps(d, indent=1))
    pilot.events.write_text("")
    r = pilot.run()
    assert r.returncode != 0 and "another lock" in pilot.log()
    assert not pilot.gpu() and pilot.status()["outcome"] != "complete"


def test_a_read_out_older_than_a_cell_it_summarises_is_refused(pilot):
    assert pilot.run().returncode == 0
    old = (pilot.out / "pilot.json").read_bytes()
    shutil.rmtree(pilot.out / "sft_E")                    # the cell is re-run...
    pilot.events.write_text("")
    r = pilot.run(STUB_EVAL_EXTRA_NONE="")
    assert [e["cell"] for e in pilot.ev("sft_train")] == ["E"]
    ev = pilot.out / "sft_E" / "eval.json"
    doc = json.loads(ev.read_text())
    doc["conditions"]["val"]["ranks"]["r1"] = 9           # ...and scores differently
    if r.returncode == 0:                                  # identical bytes: nothing to refuse yet
        ev.write_text(json.dumps(doc))
        pilot.events.write_text("")
        r = pilot.run()
    assert r.returncode != 0, "a pilot.json that predates its cell was reported as the read-out"
    assert "pilot.json" in pilot.log() and pilot.status()["outcome"] == "failed"
    assert pilot.status()["pilot"] is None
    assert (pilot.out / "pilot.json").read_bytes() == old  # never overwritten either
    (pilot.out / "pilot.json").rename(pilot.out / "pilot.superseded.json")
    r = pilot.run()
    assert r.returncode == 0 and pilot.status()["stages"]["compare"] == "ran"


def test_an_unwritable_status_refuses_before_any_gpu_step(pilot):
    pilot.out.mkdir()
    (pilot.out / "status.json").mkdir()                   # cannot be written as a file
    r = pilot.run()
    assert r.returncode != 0 and not pilot.gpu()
    assert "status.json" in r.stderr + pilot.log()


def test_a_stale_read_only_status_cannot_outlive_a_new_launch(pilot):
    pilot.out.mkdir()
    st = pilot.out / "status.json"
    st.write_text('{"outcome": "complete"}')
    st.chmod(0o444)
    r = pilot.run(STUB_CPT_RC_P="3")
    st.chmod(0o644)
    assert r.returncode != 0 and not pilot.gpu(), "GPU work started with no way to record its outcome"


def test_a_launch_outside_the_wrapper_is_refused_on_a_gpu_visit(pilot):
    """REQUIRE_WATCHDOG is the wrapper's; without it only an explicit off-box dry run may proceed."""
    r = pilot.run(ALLOW_CPU=None)
    assert r.returncode != 0 and not pilot.gpu()
    assert "launch_judge_first.sh" in r.stderr + pilot.log()


def test_a_path_that_could_smuggle_trainer_flags_is_refused(pilot):
    r = pilot.run(CANONICAL_EVAL=str(pilot.tmp / "canonical") + "\n--resume_from\n/elsewhere")
    assert r.returncode != 0 and not pilot.gpu()
    r = pilot.run(O=str(pilot.tmp / 'o"ut'))
    assert r.returncode != 0 and not pilot.gpu()


def test_a_signal_mid_arm_is_recorded_as_terminated_and_leaves_no_trainer(pilot):
    proc = _start(pilot, STUB_CPT_SLEEP_P="60", SHUTDOWN_GRACE="2")
    _wait_for(pilot, "cpt")
    proc.terminate()
    proc.wait(60)
    assert proc.returncode == 130
    st = pilot.status()
    assert st["outcome"] == "terminated" and st["stages"]["cpt_P"] == "failed"
    time.sleep(0.5)
    assert not _alive(pilot.ev("cpt")[0]["pid"])


def test_a_checkpoint_without_the_completion_marker_is_not_evaluated(pilot):
    """`judge train` writes best.pt after the first improving epoch, so best.pt alone is not done."""
    d = pilot.out / "sft_B"
    d.mkdir(parents=True)
    (d / "best.pt").write_text("{}")
    assert pilot.run(STUB_CPT_RC_P="0").returncode != 0
    assert "partial" in pilot.log()
    assert not pilot.ev("sft_train") and not pilot.ev("sft_eval")


def test_an_evaluation_off_the_lock_stops_the_run_at_that_cell(pilot):
    r = pilot.run(STUB_EVAL_EXTRA_P="test_novel_pep")
    assert r.returncode != 0
    assert [e["cell"] for e in pilot.ev("sft_train")] == ["B", "P"]
    assert pilot.status()["stages"]["sft_P"] == "failed" and not (pilot.out / "pilot.json").exists()


def test_a_reused_evaluation_that_no_longer_passes_is_refused(pilot):
    assert pilot.run().returncode == 0
    ev = pilot.out / "sft_B" / "eval.json"
    doc = json.loads(ev.read_text())
    doc["conditions"]["test_iid"] = {"ranks": {"r1": 1}}
    ev.write_text(json.dumps(doc))
    pilot.events.write_text("")
    r = pilot.run()
    assert r.returncode != 0 and not pilot.gpu()
    assert pilot.status()["stages"]["sft_B"] == "failed"


# ------------------------------------------------------------------------------ after the Codex review
@pytest.mark.parametrize("over", [{"POLL": "0.1"}, {"POLL": "x"}, {"SHUTDOWN_GRACE": "x"},
                                  {"SHUTDOWN_GRACE": "100000"}, {"STEP_POLL": "x"}])
def test_a_knob_the_watchdog_shares_must_be_a_bounded_integer(pilot, over):
    """The watchdog adds POLL in shell arithmetic; a fraction there breaks it after it has armed."""
    r = pilot.run(**over)
    assert r.returncode != 0 and not pilot.gpu()
    assert "not" in r.stderr and list(over)[0] in r.stderr


def test_an_adapter_changed_after_the_pair_check_stops_the_next_gpu_step(pilot):
    assert pilot.run(STUB_SFT_RC_E="5").returncode != 0          # stops with sft_E still to train
    shutil.rmtree(pilot.out / "sft_E")
    w = pilot.out / "cpt_E" / "final" / "adapter_model.safetensors"
    b = bytearray(w.read_bytes())
    b[-1] ^= 0x01                                                # another valid adapter at the same path
    w.write_bytes(bytes(b))
    pilot.events.write_text("")
    r = pilot.run()
    assert r.returncode != 0 and not pilot.gpu()
    assert "adapter" in pilot.log() and pilot.status()["outcome"] == "failed"


def test_allow_cpu_does_not_excuse_the_watchdog_on_a_box_with_a_gpu(pilot, tmp_path):
    bin_ = tmp_path / "gpubin"
    bin_.mkdir()
    (bin_ / "nvidia-smi").write_text("#!/bin/sh\necho 0\n")
    (bin_ / "nvidia-smi").chmod(0o755)
    r = pilot.run(PATH=f"{bin_}:{os.environ['PATH']}")           # ALLOW_CPU=1, no wrapper, a GPU present
    assert r.returncode != 0 and not pilot.gpu()
    assert "launch_judge_first.sh" in r.stderr + pilot.log()
