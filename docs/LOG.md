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

## 2026-10-07 — the box visit ended; 4.3 fully measured; 4.2's gate caught a broken scorer

- **Visit:** 21:32 UTC 10-06 → 18:02 UTC 10-07 (20.5 h, ≈ USD 21 on g6.2xlarge). The watchdog
  fired DONE one minute after the job's last process exited and stopped the box; nothing idled.
  What LOOKED idle in the morning was wave-1 cell 1's evaluation (08:15–18:01), 9.8 h of
  scoring ~200 molecules 25× each.
- **4.3 final verdicts:** `EVAL_AB` PASS; `TRAIN_EQUIV` FAIL (v1 sampler, §9.3); `RESUME_LOG`
  PASS (resumed at step 1000); **`RESUME` marginal FAIL**: overall 2.881 vs 2.971 (3.03 % against
  the 2 % band), cdr3 0.86 %, pocket 1.84 % — the resumed run is better; a two-seed noise
  baseline does not exist, so this is reported as a narrow fail. **The v2 re-test did not run:**
  the wave-1 gate failure exits before the phase-3 hand-off (I had protected it against the
  ceiling, not against that path). Fix: a fresh launch under the v2 code IS the re-test
  (`s43_equiv_v2_pointD`, `TRAIN_EQUIV_V2`, `V2_VS_V1`, `RESUME_V2`); phase 3 retired.
- **4.2 cell 1 (base, as_paired_i): 8.0 h train + 9.8 h eval, then `GATE FAIL`** — Recall@10 =
  0.0 on all three conditions, every query rank 100 of 100, MRR = 0.01 and val AUC = 0.5 exactly,
  train loss = the label-prior entropy: every candidate scored identically. Data path cleared on
  the laptop (real rows serialize with the peptide and change with it). Cause (hypothesis fitting
  all symptoms): the earlier dtype fix ran pool + head under bf16 autocast, so logits were
  quantised to ~3 digits and candidates differing by 9 of ~400 tokens tied. **Fixed:** pool + head
  fp32 (`track_a/esm_sft/model.py`; `tests/test_judge_scorer_fp32.py` reproduces the collapse with
  a stub and shows the fix resolves it). **The smoke run had shown the same ranks 18 h earlier**;
  only its exit code was checked → new `judge probe` (scores must vary and be finite) runs first
  in the 4.3 smoke and as a hard pre-gate in wave 1 for both arms.
- **Benchmark splits are exploded across segment orderings:** as_paired_i test_iid 6,696 rows =
  201 `row_id`s (up to 120 orderings each), test_level4 4,872 = 202, test_novel_pep 49,776 =
  1,589, train 553,440 = 15,145; as_trb_i test_level4 83,178 = 9,734. The judge scored 5,000
  "queries" = ~200 molecules ×25, inflating n and narrowing the CI ~5×. **Fixed:** `load_split`
  de-duplicates on `row_id`; `min_queries` 500 → 150 (reported flag, not abort); a cell drops from
  18 GPU-h to ≈ 1–2.5 h. Paired class-I decisions rest on ~200 molecules — a benchmark property.
- Kit: `s43_compare.py`/`run_s43.sh` carry the sampler generation in dir names and labels (`SUF`);
  `wave1.sh` probe pre-gate; dry runs 8/8 (4.3) and 6/6 scenarios (4.2) incl. probe-abort.
- **Second visit launched 18:49 UTC** (g6.2xlarge after 4 attempts at the 5-min poll; the loop
  exited on launch). `sampler=v2` confirmed in the start line; on-box tests 21/21; `judge_probe`
  **PASS** — six candidates per query score 0.0159–0.0163, distinct in the 4th digit, exactly the
  spread bf16 could not hold; `judge_smoke` rc=0 with its 20-query advisory gate at recall 0.05
  (no longer 0.0; inconclusive at n=20 by design). Session closed with `eval_new` running and the
  v2 equivalence run next; verdicts land in `run.log` and S3 at the visit's end.

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
- **First launch, and a 4-minute failure:** the 20:26 UTC loop got a **g5.2xlarge** at 20:34 (the
  first start since the outage began). Every ESM-C call died with `cudaErrorNoKernelImageForDevice`
  in `flash_attn_varlen_forward`: the box's flash-attn is compiled for sm_89 (L4) only and the A10G
  is sm_86. The watchdog fired DONE at 20:40 and published; cost ≈ USD 0.30. **g5 removed from
  `gpu_types.txt`** (L40S is sm_89 and stays). The loop also used to label every refusal "no
  capacity"; it now logs the AWS error class per type and says "refused" when any is not
  InsufficientInstanceCapacity. Confirmed the shortage is real: last error text was
  InsufficientInstanceCapacity, no other GPU instance runs in the account, spot placement scores
  1/10 for 4-GPU types in all four zones and 4–5 for 1-GPU types in 2a/2b/2c.
- **Stage 4.2 wave 1 launcher built** (`scripts/training/stage42_box/wave1.sh` + `s42_check.py`):
  4 cells in decision order (base/arm D × as_paired_i first, then as_trb_i), both serialized
  cdr3 × full as arm D was trained; harness gate after cell 1 (base Recall@10 on test_iid must
  clear the random rate with its CI, else stop — the judge is broken); 5,000-query cap, n ≥ 500;
  paired bootstrap compare; publish after every cell; 3 cells in parallel on a 4-GPU box.
  Because the box had to be relaunched anyway, 4.3 and wave 1 now run as ONE job under one
  watchdog (`run_all.sh`), replacing a live-watchdog swap. Dry runs: 4.2 flow 5/5 scenarios, 4.3
  kit both layouts 8/8 verdicts. Kits re-uploaded. Not yet run on a GPU.
- 21:14 UTC loop attempt 1: all nine L4/L40S types refused, every one InsufficientInstanceCapacity.
  Spot discussed again and declined for now: it draws from the same spare pool (scores 1–5/10),
  is reclaimed first in a shortage, and needs an image + S3 checkpoint sync + interruption handler
  + a new launch path (~1–2 days) — worth building before 4.7, not for a USD 5–55 job. Instead the
  poll interval moved to `loop.conf` (POLL_SECONDS=300, MAX_HOURS=24), re-read every attempt like
  the type list, so short capacity windows are not missed.
- **21:26 UTC: a g6.2xlarge (1× L4) started; the Stage 4.3 job launched at 21:32.** On-box tests
  18/18. `eval_old` 403 s, `eval_new` 378 s, **`EVAL_AB` PASS (exactly equal)** — the eval rewrite's
  "bit-identical" claim is now verified. **`judge_smoke` rc=0** — the Stage 4.2 judge ran end to
  end for the first time. The equivalence run R did ≈ 1,400 steps/h on one L4 (Stage 4.1: 453 per
  GPU), 100 % utilisation, 18.5 of 23 GB. **`TRAIN_EQUIV` FAIL** at step 2000: overall 2.971 vs
  2.241 (+32.6 %), cdr3 4.176 vs 4.861 (−14.1 %), pocket 3.324 vs 2.569 (+29.4 %); the gap had the
  same sign at steps 500/1000/1500 and widened. Cause: the v1 sampler's length-homogeneous steps
  (4.12 micro-batches/step on average = almost every step pure short rows) concentrated the
  full-MHC rows into a few dozen steps — ordering, not data. Written up as `findings.md` §9.3.
- **Sampler v2 built the same evening** (global-permutation steps, longest-first round-robin
  dealing, rank-identical token-budget cuts, row-weighted loss; epoch remainder = globally
  shortest rows): 21 laptop tests (4 new) and a `verifier` pass — 6/6 mutants caught, invariants
  confirmed on two length profiles; it found the remainder fallback could drop long rows on data
  with few sub-64 rows (fixed) and restored the Stage 4.1 path's exact division. Not GPU-verified.
  **Phase 3** (`phase3_v2.sh`, handed off by `wave1.sh`) re-runs the equivalence test under v2 on
  the same box after wave 1 → `TRAIN_EQUIV_V2` and `V2_VS_V1`. Only not-yet-running files were
  swapped on the box (bash reads a running script incrementally; `run_s43.sh`/`run_all.sh` were
  verified untouched). S (resume) resumed correctly at step 1000 at 23:22 UTC.
- **Timeline and ceiling (23:50 UTC):** measured rates put the whole visit at 8–17 h (finish
  05:30–14:30 UTC 10-07) against a 22 h watchdog — adequate, and not re-armed (fixed at arm time;
  re-arming on a live box is the one risky operation). To protect the v2 verdict in the tail
  case, `wave1.sh` on one GPU now runs phase 3 between the paired cells and the trb cells; the
  reordered script was swapped onto the box at 23:46, before wave 1 started. Session closed with
  S in its step-2000 eval and wave 1 not yet started. Memory consolidated (both project files
  under 300 words).
- **Retry loop stopped by Yuexin ≈18:55 UTC** (attempts 1–3 from 18:18, no capacity) so the next
  launch uses the repo's `launch_s43.sh` (watchdog 4 h, OS timer 300 min, publishes
  `resume_training.log`); `gpu_retry_loop.sh` now reads that file instead of a dead scratchpad
  copy. Not restarted by the assistant (safety classifier withheld the relaunch); the start
  command is NEXT.md item 1. Nothing is running on AWS; the box is stopped.

## 2026-11-03/04 — Stage 4.3 objective decided; Stage 4.2 shown unreadable where it was gated; V04 launcher built

Laptop-only session. No GPU spend. Branch pushed for the first time since it was created.

- **`4.3-objective` DECIDED: `--loss_weighting token`.** `--token_budget 0` rejected. One-time costs
  of the two branches are comparable since both owe a noise band, but `--token_budget 0` pays ≈3.9×
  on every remaining 4.4–4.7 run — ≈USD 350–460, and at that factor neither `4.4-s2` nor `4.7-run`
  fits one visit — to stay continuous with a weighting that is an artifact of batch size 16, on a
  metric `4.6-s1` had already demoted. **Stage 4.3 CLOSED.** Re-baseline scoped to arms **A and D**;
  arm F controls nothing downstream (checked against `stage4-4/5/7.md` and `tests.md`). `4.3-tokw`
  and `4.3-noise` folded into it: arm D × 3 seeds, 7,800 steps, no early stop, which puts the band
  at the step comparisons are read at and doubles as `4.4-s2`'s control. Detail:
  `quest-cpt/box-jobs/README.md` §3.2.
- **Landmine fixed:** `--loss_weighting` defaults to `row`, the rejected objective, and **no launcher
  passed the flag** — `run_s43.sh` passed `--token_budget 16384` and nothing else, so it would have
  trained row-share silently. Now explicit and logged, verified via `stage43_box/dryrun.sh` in all
  three layouts.
- **Stage 4.2: the comparison was not readable where the gate read it.** Effect pre-registered first
  in its own commit (`quest-cpt/box-jobs/prereg-4.2-effect.md`), then computed from the V03 base
  cell's shipped per-query ranks. Smallest detectable effect at ρ=0.8 in Recall@10 points against a
  pre-registered **+0.050**: `test_level4`/peptide **+0.2574**, `test_level4`/pep_cluster **+0.3614**,
  `test_novel_pep`/norm_rank **+0.0170**. So cells 2–4 as specified would have returned an
  uninterpretable number, while the readable condition was **already in the same eval pass** — no
  task switch needed. `test_level4`'s 108 peptides collapse into **18 pep_clusters**, so the
  independence unit is not settled at the peptide. `as_trb_i` level4 is **335 peptides / 131
  clusters**, not the 48× its 9,734 molecules imply. Detail: `box-jobs/README.md` §3.3.
- **Primary read LOCKED** at `test_novel_pep` × `norm_rank` × peptide, chosen on pre-cell power with
  no CPT cell in existence, the other eleven pairings secondary. **Breakeven ρ = 0.765** against the
  0.8 assumed — this corrected an overclaim of mine that had called the margin threefold by
  converting a 1.6 % margin in λ into Recall@10 points. Nothing in the cell is readable unpaired.
  **§9.6's length finding needs restating per peptide:** every non-9 stratum holds 1–24 peptides, the
  12-mer row is one peptide carrying 48 molecules, and the homogeneous 9-mer stratum puts base
  **below chance** (0.0416). Direction of the length effect flips between conditions. §3.4.
- **New committed code.** `stage42_box/readability.py` (stdlib only: paired, peptide-clustered,
  metric-parameterised, breakeven ρ, length stratification) + 23 tests pinned to the real cell,
  independently reproducing §9.6's corrected intervals to within 0.01. `s42_check.py` emits two gate
  conditions separately and enforces the locked pairing. `v04_box/` — `run_v04.sh` (the gate chain),
  `provenance.py` (refuses a comparison across an objective/draw/pool-id/step/seed-set/batching
  mismatch), `dryrun.sh` (both gate branches on the real ranks, plus the row-share abort).
  `scripts/smoke_stdlib.sh` proves the no-dependency property on a fresh clone.
- **Seed-path audit (findings §9.8).** The data path is fully seeded — sampler
  `np.random.default_rng(seed+epoch)`, masking `torch.Generator(mask_seed)`, `manual_seed` before
  model load. The 1.31 % is **flash attention's backward**, which accumulates `dQ` with atomics,
  sits outside PyTorch's determinism registry and is active on every arm regardless of
  `--attn_implementation` (inert here). **The `--deterministic` comment had inverted the inference**
  — it said a residual difference means something is unseeded, which would have sent the next
  session hunting a non-existent bug. Consequence: the three-seed band is **not a seed band**.
- **Two latent repo faults found.** `*.csv` in `.gitignore` was swallowing test fixtures, so
  `test_judge_cluster_ci.py`'s 12 tests had been pinned only on this laptop; rule fixed, both
  fixtures tracked. And `test_micro_batch_loss_scale_reproduces_fixed_batch_weighting` was renamed —
  its assertions are sound but cover only equal micro-batches, while the name states the claim §9.5
  refuted.
- **Incidents, mine.** A regression comparison that silently did nothing because zsh does not
  word-split unquoted variables; a `local` statement deriving a path from variables assigned in the
  same statement, which tripped `set -u` and failed both launcher phases silently past `bash -n`;
  three-decimal rounding that put a pass and a fail on the same displayed value at the decision
  threshold; and a non-stdlib test file listed in the smoke subset. The dryrun and smoke targets
  caught the last two, which is what they are for.
- **Approach extracted:** `~/Documents/ACS/learnings/2026-11-04-can-this-measurement-answer-the-question.md`.
- **Cost:** idle spend measured at **USD 298/mo**, confirming the filed 295 rather than contradicting
  it; October's higher figure was a transition month. Both boxes stopped all session.

## 2026-11-06 — two audit rounds, a pre-launch review, and judge-first launched

- **Audit round one** (`quest-cpt/audits/2026-11-06/`). Five readiness defects, all reproduced
  before the code was read, all repaired. A stale eval with no checkpoint was certified, and its
  manifest overwritten. A NaN score ranked 1. The hour ceiling was checked only between phases,
  and the watchdog's publish had no time limit. `token_exact` could be reached through the
  fixed-batch path. The preflight checked that files existed, not that they worked. Provenance is
  now read from artifacts.
- **Found while fixing them:** arm D (Stage 4.1) trained **fixed-batch-mean/v0**, not row share
  (findings §9.10).
- **Audit round two** (`round-two/`). Four of the five defects were in that day's own repairs.
  - The launch wrapper armed the watchdog on a stale pid file, and could SIGTERM an unrelated
    process.
  - Under `set -m`, util-linux `setsid` forks, so the wrapper's status was not the job's.
  - The comparator used a weaker validator than the launcher.
  - The SFT seed was always null, because the eval parser has no `--seed`; the stub recorded
    42, which hid it.
  - A refused rerun could sit beside a stale comparison.

  Fixed by run identity, one validator, and the SFT recipe read from the checkpoint.
- **Pre-launch code review:** ten findings, all verified and fixed.
  - The watchdog pattern did not match the judge, so it would have stopped the box mid-run.
  - Resume was unreachable.
  - The lock tracked only the launcher.
  - The preflight never compared the base recipe with arm D's.
  - The `dirty:` commit identity was too weak.
  - A missing `token_budget` was read as fixed-batch without saying so.
  - `sft_train_digests` were never compared.
  - `POOL_FILE` was silently ignored.
  - `MAX_HOURS=abc` became a zero ceiling.
  - A box with no `nvidia-smi` was not refused.
- **Numbers:** tests 321 → 469 passing. The failure set is identical to 59cafcd: 33 failures plus
  4 collection errors, all from dependencies missing on the laptop.
- **Incidents and lessons:**
  - Three of my own tests passed for false reasons. One matched pytest's tmp path, which carries
    the test's name. One read a zombie as alive, because `kill -0` succeeds on an unreaped
    child. One indexed a key before an `or`.
  - The sandbox git repo exposed that hashing untracked files made the code identity drift
    mid-run.
- **Launch:** the box was started 22:33 UTC. Its checkout is a colleague's branch with 600 lines
  of uncommitted edits; that state was preserved to S3 (`jf_ops/preserve-20261106/`), and the
  reviewed code was deployed as a bundle into a worktree. The real preflight passed. Judge-first
  run `20261106T225852Z-9828` started 22:58 UTC; both probes pass, and the watchdog is armed on
  the verified pid.
- **Backup:** second remote `mine` → private `YuexinXu/quest`.

## 2026-11-09 — judge-first result read

- Run `20261106T225852Z-9828` completed 2026-11-07 03:52 UTC: 294 min, about USD 5.7. The
  watchdog's DONE guard fired, the results published, and the box stopped. `driver_rc` 0 under
  `setsid --wait`.
- The gate PASSED: the locked read's breakeven ρ is −0.933.
- **Primary (`test_novel_pep / peptide / norm_rank`):** arm D − base = **−0.023
  [−0.044, −0.002]**. At pep_cluster the interval spans zero. Measured ρ 0.735.
- **Secondary:** test_level4 Recall@10 0.401 → 0.228 (−0.173 [−0.356, −0.082]). test_iid shows
  no difference.
- **Caveats:**
  - One SFT seed per arm, and arm D's SFT validation AUC was lower (0.630 vs 0.702).
  - The base checkpoint was trained on 10-08 code and recorded no data hashes.
  - Only the legacy fixed-batch recipe was tested.
- **Unexplained:** both arms are below chance on test_novel_pep (0.457 and 0.434 against 0.5).
