# Stage 4.3 — V03 evidence, pulled from S3 2026-10-15

Source: `s3://fh-pi-warren-h-eco/quest/tcrbench2/checkpoints/stage43_check/`. Duplicate copies of
the training logs (published twice under two names) were removed and the rest gzipped.
Visit: `quest-cpt/box-jobs/visits/v03-2026-10-07.md`. Reading: `quest-cpt/findings.md` §9.5.

## Verdicts, in `s43_out/run.log`

```
EVAL_AB            PASS (exactly equal, 0 differing leaves)
TRAIN_EQUIV_V2     FAIL @2000, better: overall 1.838 vs 2.241 (17.98 %)
RESUME_LOG         PASS
RESUME_V2          FAIL @2000: overall 3.37 %, cdr3 0.04 %, pocket 6.36 %
V2_FINAL           DIVERGED-WORSE: v2 @step4000 vs ref @step7800
V2_DETERMINISM     @2000: fresh 1.862 vs run R 1.838 (1.31 %) — same seed and sampler
```

**Both headline verdicts are confounded.** `V2_FINAL` compares step 4000 with step 7800: the run
early-stopped at 4000 of 7800 on `pocket`, the one metric that had stalled, while overall and cdr3
were still falling. `V2_DETERMINISM` shows two identical runs differ 1.31 %, so every 2 % band used
here was tighter than the measurement.

The v2 curve, from `s43_equiv_v2_full_training.log.gz` (evals every 500 steps):

| step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 |
|---|---|---|---|---|---|---|---|---|
| overall | 2.704 | 1.978 | 1.890 | 1.862 | 1.913 | 1.803 | 1.830 | **1.795** |
| cdr3 | 5.699 | 5.148 | 4.918 | 4.719 | 4.528 | 4.291 | 4.247 | **4.190** |
| pocket (stop metric) | 3.099 | 2.219 | 2.284 | 2.124 | 2.092 | **2.081** | 2.223 | 2.103 |

Cause, found in code rather than in these logs: v2 ships row-share micro-batch loss weighting, which
gives MHC-bearing tokens 0.43× the per-token gradient weight the Stage 4.1 path gave them. Pinned in
`tests/test_micro_batch_loss_weighting.py`.

## Files

| Path | What |
|---|---|
| `s43_out/run.log` | **every verdict line**, all three launches (10-06 aborted, 10-06/07 v1, 10-07/08 v2) |
| `s43_out/eval_old.{json,log}`, `eval_new.{json,log}` | the `EVAL_AB` pair — the instrument did not move |
| `s43_equiv_v2_full_*` | the full-epoch v2 run: curve + `early_stop_history.json` (9 evals, best at eval 6) |
| `s43_equiv_v2_pointD_*`, `s43_equiv_pointD_*` | the 2,000-step equivalence runs, v2 and v1 |
| `s43_resume_v2_pointD_*`, `s43_resume_pointD_*` | the resume runs |
| `s43_out/judge_smoke/` | the 20-query judge smoke, rc 0, gate advisory |
| `s43_out/autostop.log`, `autostop.log` | the watchdog: it fired correctly, box stopped 04:08 UTC |

**Not here:** the script that computed `V2_FINAL` and `V2_DETERMINISM`. It was `/home/ubuntu/
v2_full.sh`, written on the box at 21:24 UTC 10-07 and never committed, so it survives only on the
stopped volume `vol-08f5626ccf18fdd4c`. Recover it on the next visit; its comparison logic needs
rewriting anyway, since it compares unmatched step budgets.
