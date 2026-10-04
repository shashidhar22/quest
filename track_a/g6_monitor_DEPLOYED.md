# g6 monitor — deployed

**Date**: 2026-05-04
**Operator**: Claude (Track A)

## Script

`/home/ubuntu/quest/scripts/monitor_g6.py` (stdlib + pandas only; read-only).

## Deployment

Cron entry installed; firing every 5 min:

```
*/5 * * * * /home/ubuntu/miniforge3/bin/python3 /home/ubuntu/quest/scripts/monitor_g6.py >> /home/ubuntu/quest/track_a/monitor/cron.log 2>&1
```

`cron.service` confirmed active (running since 2026-05-03 22:56 UTC). User linger already enabled (`Linger=yes`), so cron + the resume queues both survive SSH disconnect.

## Verification — first 2 invocations

| Invocation | Source | Status file | Notes |
|---|---|---|---|
| 1 | manual smoke | `status_20260504T153437Z.json` | First run baseline; INFO-only |
| 2 | manual smoke | `status_20260504T153929Z.json` | After fake-process injection / revert |
| 3 | manual smoke | `status_20260504T153550Z.json` | Match-logic regression test |
| 4 | **cron** `*/5` | `status_20260504T154001Z.json` | First cron auto-fire at 15:40 UTC ✅ |
| 5 | **cron** `*/5` | `status_20260504T154501Z.json` | Second cron auto-fire at 15:45 UTC ✅ |

Both cron-driven runs reported `procs alive=5/5  cells=68/132 (51.5%)  C=0 W=2 I=0` — the W=2 are GPU 3 idle (real, expected event during between-cell transition; not a false positive).

## Initial cell-count snapshot

```
Coverage: 68 / 132 expected cells (51.5%)
Baselines reporting: 7 of 11 expected
  deeptcr        (12/12 cells: as_trb_i, as_paired_i complete)
  esm2_35m-base  (5/12   cells: as_trb_i complete)
  esm2_8m-base   (5/12   cells: as_trb_i complete)
  lstm           (12/12 cells)
  mlp            (12/12 cells)
  tcrbert        (10/12 cells: as_trb_i, as_paired_i complete)
  tulip-original (12/12 cells incl. 2 class-II skipped)

Not yet reporting: esm2_8m-cpt, esm2_35m-cpt, esmc_300m-base, esmc_300m-cpt
```

## Process inventory at deployment

| script | PID | PPID | etimes | what it's doing |
|---|---|---|---|---|
| resume_gpu0.sh | 74413 | 1 (init) | 2859s | esm2_8m-base eval as_paired_i 600/2000 → continues to cpt × 4 |
| resume_gpu1.sh | 74412 | 1 (init) | 2859s | esm2_35m-base train as_paired_i → continues to cpt × 4 |
| resume_gpu2.sh | 80470 | 1 (init) | 1200s | esmc_300m-base train as_trb_i (2000 cap) |
| resume_gpu3.sh | 74411 | 1 (init) | 2859s | tcrbert train as_paired_ii → ESM-C 300M class-II next |
| resume_deeptcr.sh | 74410 | 1 (init) | 2859s | deeptcr training as_paired_ii on GPU 3 share |

All five PPIDs are 1, confirming detachment from any user terminal session (linger working).

## Alerts during deployment

`alerts.log` contents (after revert of fake-process test):

```
2026-05-04T15:34:37Z [INFO]     Monitor baseline established at 68/132 cells; no deltas computed on first run.
2026-05-04T15:35:09Z [WARNING]  GPU 3 util 0.0% for 3 consecutive runs.
2026-05-04T15:35:50Z [WARNING]  GPU 3 util 0.0% for 5 consecutive runs.
2026-05-04T15:39:29Z [CRITICAL] Expected process NONESCRIPT_xyz_999.sh is not running.   ← injected test
2026-05-04T15:39:29Z [WARNING]  GPU 3 util 0.0% for 6 consecutive runs.
```

The CRITICAL alert was injected on purpose (added a non-existent script to `expected_processes`) to verify the alert path. After revert, no false positives — the script correctly identifies the 5 real `resume_*.sh` processes via tokenized arg matching (an earlier substring match was too permissive; fixed during deployment).

The GPU 3 WARNING is a real, expected event: resume_gpu3 just transitioned from tcrbert TRB-II eval (now complete) to tcrbert Paired-II training, and is currently in negative-sampling for 553K positives. This is a normal cooldown period; the streak resets once GPU 3 picks up training. The threshold (3 runs at <10% = 15 min) is intentionally tight to catch real stalls, at the cost of these brief cooldowns reading as warnings.

## Bug fixes during deployment

1. **Process matching too permissive**: `if script in r["args"]` matched parent `bash -c` shells whose eval string mentioned the script name (Claude Code's shell-snapshot keeps such bashes alive). Replaced with tokenized-args matching: first non-flag token after `bash` must basename to the expected script, and `bash -c` invocations are explicitly rejected. Re-tested with both real and fake scripts — clean.

## Done criteria — checklist

- [x] `monitor_g6.py` written
- [x] expected_cells.json built and verified against actual conditions per task (Class II = 1, not 4 as in user's template)
- [x] Initial baseline snapshot taken: `cells_at_20260504T153243Z.json`
- [x] Cron set up (`crontab -l` shows the entry)
- [x] First cycle observed (manual smoke runs) without false positives
- [x] CRITICAL alert path tested by injecting a fake expected process
- [x] README written: `track_a/g6_monitor_README.md`
- [x] Setup notes written: `track_a/g6_monitor_setup.md`
- [x] Cron-driven first auto-fire confirmed: `status_20260504T154001Z.json` at 15:40 UTC, then `status_20260504T154501Z.json` at 15:45 UTC (5 min apart, on the `*/5` boundary)

## Output paths

```
scripts/monitor_g6.py                         # the script
track_a/expected_cells.json                   # goal config
track_a/g6_monitor_baseline.md                # state at deployment
track_a/g6_monitor_README.md                  # operator manual
track_a/g6_monitor_setup.md                   # cron / nohup install commands
track_a/g6_monitor_DEPLOYED.md                # this file
track_a/cells_at_<UTC>.json                   # baseline snapshot
track_a/monitor/status_<UTC>.json             # per-run snapshots (every 5 min)
track_a/monitor/alerts.log                    # append-only
track_a/monitor/state.json                    # streak state
track_a/monitor/cron.log                      # cron stdout
/tmp/monitor_g6.lock                          # flock for serialization
```
