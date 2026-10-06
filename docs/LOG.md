# LOG — quest (dated, append-only; newest last)

One entry per session: what changed, the numbers, incidents and their lessons. NEXT.md stays
short because history lives here. Facts that are still true belong in `~/Documents/ACS/quest-cpt/`
or the project memory, not in this log.

## 2026-10-02

- Stage 4.1 planned and launched on `i-0030c1a9c845b3714` (g6.24xlarge, only type with capacity):
  three arms at one fixed `max_length=1024` (measured global max 608) and effective batch 1024.
  `run11_cdr123_fullmhc_proportional_10M` built (9,998,544 rows, 56 min).
- `mhc_class=none` pocket tokens diagnosed: 157 class-II alpha chains in `mhc_one` mislabelled by
  `derive_mhc_class`; labels were always right, only the stratum; fixed; Point D re-score
  reproduced 2.001 exactly, proving the fix aggregate-neutral.
- KS RNA-Seq workspace sync to `archive/ks_manuscript_workspace/` launched from a helper
  (`xargs -P 16` per-file after single `aws s3 sync` managed 16–18 MB/s on the lazy-loading
  restored volume).

## 2026-10-03

- KS sync VERIFY PASS: 40,933 objects / 1,446,777,687,859 B exact. The `xargs` pass had died at
  entry 24,532 on an apostrophe (`unmatched single quote`); the fallback `aws s3 sync` pass
  finished the rest. Extras (`data_catalog/` 50 files, `kics/` 21 files) synced to
  `archive/ks_rnaseq_home/`, VERIFY PASS; the helper would not boot with the KS root clone
  attached (same AMI → same root label → booted the clone) — detach before start, attach after.
- `teardown.sh --execute`: helper + 3 TB temp volume + KS instance gone; snapshot kept.
- Stage 4.1 arms 2 and 3 PASS (5.98 h, 6.05 h). Results: `docs/cpt_stage4.1/README.md` §10.
  D reproduces Stage 4 Point D within 1 % on three regions; full MHC still −44.7 % vs pocket at
  a matched window; CDR1/2 hurts; class I still 7.6–17.7.
- **Incident:** the auto-stop watcher never fired (`pgrep -f` matched the tmux server's argv);
  the box idled 14.5 h, ≈ USD 100. Second bug: `build_stage41_summary.py` KeyError 'F' (Stage 4
  letter lookups). Both fixed; `scripts/training/autostop_when_done.sh` (PID + GPU-idle +
  MAX_HOURS guards) with `tests/test_autostop_when_done.sh` 8/8. Write-up §11.
- GPU-efficiency facts verified on the box: ESM-C already runs varlen flash attention;
  `--gradient_checkpointing` is a no-op; eval is ≈ 30 % of wall on rank 0. Fix order in
  `plans/stage4-3.md`.
- Stages 4.2–4.7 planned (`plans/`).

## 2026-10-04

- Seven commits on `yxu2/tcrbench2-continue` (recovered Stage 0–4 scripts, `track_a/`, the
  mhc_class fix, Stage 4.1, the watchdog, plans). Not pushed — origin is the collaborator's repo.
- Project memory consolidated to current state.

## 2026-10-05

- Cost review via Cost Explorer (permitted): idle run-rate ≈ USD 800/mo, 3/4 snapshots. On
  Yuexin's call: `neurips_two` terminated (root already in `snap-0aa5bccb08a3012e3` / AMI);
  data-loader (8.3 TB billed) and KS (3 TB) snapshots moved to the archive tier (90-day
  minimum, 24–72 h restore). Expected idle bill ≈ USD 300/mo.
- run11 backed up to S3 (35 files / 3,185,473,264 B, VERIFY PASS) by switching the box's
  instance type to m5.xlarge for 6 min — the g6.24xlarge start had failed on capacity.
  Practice: size the box to the job by switching the type of the same volume.
- System reorg: `~/Documents/ACS` became the private repo `acs-notes` (AWS runbooks, box
  records, `quest-cpt/` knowledge split from the 767-line checkpoints doc); this LOG.md
  created; NEXT.md cut to its contract; memory 10 → 4 files; `warren-gpu.sh` gained
  `inventory`, `cost`, `type`.

## 2026-10-05 (evening) — Stages 4.2 / 4.3 set up; Track A invalidated; GPU outage

- **Plans 4.2–4.7 corrected** before any run: 4.3's sampler would have desynced DDP; 4.4's
  token-matched screen confounded sampling with repetition (step 1 is now run06 vs 4.1 arm A,
  no build); 4.2 gated on a 4-cell wave; 4.5 judged downstream; 4.6 step 1 promoted.
- **4.3 code committed:** `TokenBudgetStepSampler` (10/10 tests; first version dropped the
  longest rows every epoch — the remainder came off the long end — caught by a test, fixed, and
  the fix proven by a mutant run); trainer wiring behind `--token_budget` (Stage 4.1 path
  unchanged); region eval caches tokenization and loops only over masked positions in the same
  order (should be bit-identical — NOT YET VERIFIED). Fixed a latent `n_examples` NameError.
  Equivalence criteria pre-registered in `plans/stage4-3.md`. Mid-run resume NOT built.
- **Track A downstream evidence invalidated.** Its ESM SFT never fed MHC to the model (loader
  omits the pocket column; 2,800/2,800 rows empty, checked on the box) and its base arms had no
  trainable LoRA. Old ESM-C crash (bf16 vs fp32 head) is already fixed in current `model.py`.
  `track_a/esm_sft/judge.py` fixes both defects, pairs arms by row_id-seeded distractors.
  Written, compiles, NOT yet run. Corrected in quest-cpt/findings.md, memory, plans/stage4-2.md.
- **GPU outage:** every g6 (xlarge…24xlarge) and g5.12xlarge refused in us-west-2a AND 2b,
  probed 19:50–20:39 UTC with immediately-terminated launches (no leaks). A CPU A/B of the eval
  rewrite was started on m6i.2xlarge then abandoned (the GPU job repeats it).
- **Launch kit** `scripts/training/stage43_box/`: laptop retry loop every 15 min for 24 h with
  `caffeinate -ims -w <pid>` (lid-close and SSO-session expiry still end it); on-box OS
  shutdown timer 240 min + PID watchdog (MAX_HOURS 3, idle 20 min, publishes to S3 first); the
  loop stops the box itself if SSM or launch verification fails. Worst case ≈ 4 h ≈ USD 18.
- Gotchas: modules loaded via `importlib.util.spec_from_file_location` must be put in
  `sys.modules` before `exec_module` or dataclasses fail; ESM-C on CPU needs
  `esm.models.esmc.is_flash_attn_available = False` before construction (the flash path makes
  CUDA calls); SSM lands as a user who cannot `cd` into `/home/ubuntu/quest`.

## 2026-10-06

- The first GPU retry loop ran attempts 1–4 (20:39–21:25 UTC, no capacity), then the laptop slept
  until 08:12. On wake `describe-instances` failed with AuthFailure, returned an empty state, and
  the loop aborted as if another user held the box (exit 4). Box never started — fail-safe, but
  for the wrong reason. Fixed: an empty result is now a transient API error, retried every 60 s,
  exit as SSO-expired after 5 in a row (commit `c5bf690`). Restarted 18:18 UTC (attempt 1: no
  capacity), caffeinate re-attached, SSO re-logged 18:19.
- Cost review written to acs-notes `AWSconnection/cost-review-2026-10.md`: idle bill
  USD 1,328/mo on 2026-08-01 (Cost Explorer actual, 16,600 GB gp3) → USD 973 on 09-30 → ≈ USD 295
  now (projection until archive tiering completes) — −78 %, ≈ USD 12,400/yr.

## 2026-10-06 (afternoon) — Stage 4.3 mid-run resume built; launch kit bug fixed before launch

- **Resume (change 4 of 4.3) built:** `checkpoint_step_N/training_state.pt` (optimizer, scheduler,
  counters, early-stop history, config, W&B id; written last, atomically) and
  `--resume_from <dir>|auto`. Exact data order (deterministic samplers + a skip wrapper) and exact
  optimizer/LR state; masking noise not replayed. `tests/test_cpt_resume.py` 8/8 on a tiny local
  ESM-2 (CPU, no network; scratch venv with transformers 5.19 / peft 0.21 / datasets 3.6).
  Independent `verifier` pass: 6/6 deliberate mutants caught (skip off-by-one, scheduler not
  restored, adapter not loaded, `rows_per_step` check dropped, name-ordered auto, legacy skip
  miscounted). Two defects it found by reading, both fixed and tested: `auto` into a dir of
  old-format checkpoints started fresh silently and the keep-3 cleanup then deleted the new run's
  own checkpoints (now refuses); a stray `checkpoint_step_10.bak` crashed rank 0 mid-save (now
  ignored). Also fixed: cleanup sorted checkpoints by name (`step_10000` < `step_8000`).
- **Launch-kit bug caught before any GPU time:** `run_s43.sh` waited for 4 eval lines in
  `$OUT/training.log`, a file only the Stage 4.1 driver writes — the equivalence run would never
  have been stopped at step 2000 (≈ 2 extra GPU-hours to the 3-h ceiling, no `TRAIN_EQUIV`
  verdict). Rewritten: torchrun logs into the checkpoint dir; the resume check (train to 1000,
  kill, `--resume_from auto`, stop after 2000) runs on GPUs 0/1 after the eval A/B while the
  equivalence run holds 2/3; `TRAIN_EQUIV` is written as soon as that run stops; new verdicts
  `RESUME_LOG` + `RESUME`. `dryrun.sh` exercises the whole flow on the laptop with stubs (passes).
  New `s43.tgz` + `s43box.tgz` uploaded 18:49 UTC (originals kept in the session scratchpad).
- **GPU outage continued** (attempts through 19:46 UTC, no capacity on any 4-GPU type). Checked
  with `describe-instance-types`: g5/g6/g6e sell **1, 4 or 8 GPUs — there is no 2-GPU shape**, so
  "smaller" means one GPU. The job now lays itself out by GPU count: 4 GPUs run the evals, the
  equivalence run R (2,000 steps, world size 2) and the resume run S in parallel (~2–4 h); 1 GPU
  runs the same phases sequentially at world size 1 (~5–10 h at ~1/5 the hourly rate). An optimizer
  step is 1024 rows either way (16 × 64/world × world). The box reads its own GPU count and sets
  the watchdog and OS timer from it (4 h/300 min vs 10 h/660 min). `dryrun.sh` now exercises both
  layouts (8/8 verdicts). S also changed: it resumes from a copy of R's own `checkpoint_step_1000`
  instead of re-training and killing at 1000 — same pre-registered claim, additionally tests resume
  across processes, and halves S's training. Retry list extended with
  `g6.4xlarge g6.2xlarge g5.4xlarge g5.2xlarge g6.xlarge g5.xlarge`; the 4-vCPU xlarge sizes last
  because tokenizing 10M rows is tight at 16 GB RAM. Kit re-uploaded 20:13 UTC (code unchanged).
- **Shortage is family-wide:** with the 1-GPU types added, attempt 1 of the 20:26 UTC loop was
  refused on **all nine** g5/g6 types in us-west-2a. Added the 4 L40S (`g6e.*`) types as tier 3 —
  a separate pool, offered in 2a, but several times the rate (`g6e.xlarge` spot USD 1.86/h vs
  `g6.xlarge` 0.46 on 2026-10-06; on-demand is higher, Pricing API denied to this role). The type
  list moved to `scripts/training/stage43_box/gpu_types.txt` and is **re-read every attempt**, so
  future list edits need no restart (this change itself needed one).
- **Retry loop stopped by Yuexin ≈18:55 UTC** (attempts 1–3 from 18:18, no capacity) so the next
  launch uses the repo's `launch_s43.sh` (watchdog 4 h, OS timer 300 min, publishes
  `resume_training.log`); `gpu_retry_loop.sh` now reads that file instead of a dead scratchpad
  copy. Not restarted by the assistant (safety classifier withheld the relaunch); the start
  command is NEXT.md item 1. Nothing is running on AWS; the box is stopped.
