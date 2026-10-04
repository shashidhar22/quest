# g6 Phase 2 monitor — operator README

## What it is

A read-only monitoring script (`scripts/monitor_g6.py`) that runs every 5 min via cron and writes:

- `track_a/monitor/status_<UTC>.json` — point-in-time snapshot
- `track_a/monitor/alerts.log` — append-only alert stream
- `track_a/monitor/state.json` — internal state for streak counters
- `track_a/monitor/cron.log` — cron's tee'd stdout

Exit code is **1 on any CRITICAL**, **0 otherwise**, so cron can be wired to mail/page on non-zero (we don't here, by spec).

## Reading status JSONs

Each file has these top-level keys:

| Key | What |
|---|---|
| `timestamp_utc` | Absolute time the snapshot was taken |
| `processes` | One entry per script in `expected_cells.json:expected_processes`. `alive` = bash with the script as its first non-flag arg is running. |
| `python_workers` | Python jobs spawned by the resume queues (`track_a.esm_sft.train`, etc.) for visibility. |
| `gpus` | `[{index, util_pct, mem_used_mb, mem_total_mb}, ...]` |
| `system` | mem_used/avail, swap, disk used % per `/home`, `/tmp`, `/var` |
| `cells` | `completed / expected / completion_pct / by_baseline` plus `_completed_cell_ids` (full list of `<baseline>/<task>/<condition>` strings) |
| `deltas_since_last_run` | `new_cells`, `elapsed_minutes`, `cells_per_hour` |
| `oom_events_since_last_run` | int, # of new OOM lines in dmesg |
| `alerts` | inline copy of alerts emitted this run (also appended to `alerts.log`) |

Quick one-liner peek:

```bash
LATEST=$(ls -t /home/ubuntu/quest/track_a/monitor/status_*.json | head -1)
python3 -c "import json; d=json.load(open('$LATEST')); print(d['cells']['completed'], '/', d['cells']['expected'], 'cells |', sum(1 for p in d['processes'].values() if p['alive']), '/', len(d['processes']), 'procs alive')"
```

## Alert types

### CRITICAL — investigate immediately, exit code 1

| Alert | What it means | First action |
|---|---|---|
| `nvidia-smi failed: ...` | Driver may be down; `nvidia-smi` not callable | `sudo systemctl status nvidia-persistenced`; `dmesg \| tail -50`; check `lsmod \| grep nvidia` |
| `Expected process X is not running` | A `resume_*.sh` died | Check `track_a/launch/X.log` for last error; if clean exit (queue done), update `expected_cells.json` to drop it; otherwise relaunch with `nohup setsid` |
| `N new OOM event(s) in dmesg` | Linux OOM-killed something. Capture which PID/cmdline | `sudo dmesg -T \| grep -i oom \| tail -20`; if a queue process was killed, free RAM and relaunch |
| `/home disk at >90%` | Disk filling — usually wandb/HF cache or score sidecars | `du -sh ~/.cache/huggingface /home/ubuntu/quest/wandb /home/ubuntu/quest/results/*/scores 2>/dev/null` |
| `All GPUs idle for N consecutive runs while resume processes alive` | Hung jobs, all GPUs cold but parents still parented | Tail each queue log; if CUDA error appears, kill+relaunch that queue |

### WARNING — monitor, decide whether to act

| Alert | What it means | First action |
|---|---|---|
| `GPU N util X% for K consecutive runs` | One GPU has been low for ≥3 runs (15+ min) while procs alive. Common during long data-loading phases (e.g., DeepTCR_SS init, negative-sampling on Paired-I). | Tail the queue mapped to that GPU; if loading data, normal — wait. If the python worker is gone and queue parent is alive, queue may have moved on. |
| `Memory used >90%` | Approaching OOM | Free large caches if needed (rare on g6.12xlarge w/ 181 GB) |
| `Swap used >1024 MB` | Swap-thrashing | Same as above |
| `Cell count unchanged for K consecutive runs` (≥30 min) | Possibly hung. Big cells can take 1–2h though. | Compare against expected wall-clock for the cell currently in-flight |
| `Baseline X at <50% expected cells AND no resume_* process alive` | Crashed before finishing | Relaunch the corresponding `resume_*.sh` (idempotent; skips already-done cells) |

### INFO — context, no action

| Alert | What it means |
|---|---|
| `Monitor baseline established at N/M cells; no deltas computed on first run.` | First run only |
| `New cell completed: <baseline>/<task>/<condition>` | Per-condition completion |
| `Throughput: X cells/hour over last Y min` | Progress rate |
| `Process X restarted (PID was A, now B)` | resume queue relaunched (you killed + relaunched it) |

## How to stop / restart the monitor

**Disable cron entry:**
```bash
crontab -l | grep -v monitor_g6.py | crontab -
```

**Re-enable:**
```bash
( crontab -l 2>/dev/null | grep -v "monitor_g6.py" ; \
  echo '*/5 * * * * /home/ubuntu/miniforge3/bin/python3 /home/ubuntu/quest/scripts/monitor_g6.py >> /home/ubuntu/quest/track_a/monitor/cron.log 2>&1' ) | crontab -
```

**Run manually (one cycle):**
```bash
/home/ubuntu/miniforge3/bin/python3 /home/ubuntu/quest/scripts/monitor_g6.py
```

**Reset state machine** (clear streak counters, no impact on results):
```bash
rm /home/ubuntu/quest/track_a/monitor/state.json
```

## Tuning

The thresholds at the top of `scripts/monitor_g6.py` are conservative on the noise-vs-coverage axis:

```
GPU_IDLE_PCT = 5.0
GPU_LOW_PCT = 10.0
ALL_GPU_IDLE_RUNS = 2
GPU_LOW_RUNS = 3
NO_PROGRESS_RUNS = 6
DISK_HIGH_PCT = 90.0
MEM_HIGH_PCT = 90.0
SWAP_HIGH_MB = 1024
```

To make alerts less chatty during long data-loading phases, raise `GPU_LOW_RUNS` (each unit = 5 min). For deeper hangs to alert sooner, lower `NO_PROGRESS_RUNS`.

## What the monitor will NOT do

- Will NOT kill or restart any process (read-only by spec)
- Will NOT call out to email/Slack
- Will NOT modify `results/`, training artifacts, or expected_cells.json
- Will NOT run if another instance holds `/tmp/monitor_g6.lock`
