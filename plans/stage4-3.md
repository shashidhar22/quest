# CPT Stage 4.3 — trainer efficiency + mid-run resume

**Status (2026-10-06 23:30 UTC):** first GPU run done on one L4. `EVAL_AB` **PASS** (bit-identical),
speed ≈ 3× per GPU, memory 18.5 GB; **`TRAIN_EQUIV` FAIL** — the v1 sampler's length-homogeneous
steps changed the optimisation (overall +33 %, framework +50 %, cdr3 −14 % at step 2000; cause and
numbers in `~/Documents/ACS/quest-cpt/findings.md` §9.3). Sampler v2 (mixed steps, row-weighted
loss) built, 21 tests, 6/6 verifier mutants; its GPU re-test `TRAIN_EQUIV_V2` runs as phase 3 of
the same box visit, after wave 1. `RESUME` pending (resumed correctly at step 1000) ·
**Depends on:** nothing (engineering) · **Knowledge:** `findings.md` §9.2 (efficiency) and §9.3.

## Built (2026-10-05/06) — what exists and what has been checked
- Changes 1–3 (cached eval, `TokenBudgetStepSampler` + `no_sync` + rank-plan check, rows/s and
  tokens/s log): committed 2026-10-05; sampler tests 10/10.
- Change 4, **mid-run resume** (2026-10-06): `checkpoint_step_N/training_state.pt` (optimizer,
  scheduler, counters, early-stop history, config, W&B run id; written last and atomically);
  `--resume_from <dir>|auto`. Resume restores the exact data order (the samplers are deterministic
  in seed+epoch; the interrupted epoch skips the consumed micro-batches) and the exact
  optimizer/LR state; masking noise is not replayed. `auto` refuses to start fresh over an
  old-format run. Token-budget runs may resume on a different GPU count (`rows_per_step` is the
  invariant). `tests/test_cpt_resume.py` 8/8 on CPU (tiny local ESM-2): adapter bitwise, optimizer
  moments, LR position, counters, data order == uninterrupted run's tail (token and Stage 4.1
  paths), config drift refused, auto picks newest by number, random-sampler path refused, re-save
  over an older checkpoint never leaves mixed state. Independent mutation check: 6/6 deliberate
  bugs caught. Also fixed: step-checkpoint cleanup sorted by name (would have deleted
  `step_10000` before `step_8000`).
- **On-box harness** `scripts/training/stage43_box/`: one job yields `EVAL_AB`, `TRAIN_EQUIV`,
  `RESUME_LOG`, `RESUME` in `run.log`. It lays itself out by GPU count (2026-10-06), so it runs on
  whatever type has capacity — 4 GPUs: evals + judge on 0/1, equivalence run R on 2,3, resume run S
  on 0,1, ~2–4 h; 1 GPU: the same phases sequentially at world size 1, ~5–10 h but ~1/5 the hourly
  rate. An optimizer step is 1024 rows either way (batch 16 × 64/world × world). `dryrun.sh`
  exercises **both** layouts on a laptop with stubs (8/8 verdicts).
  Two kit bugs fixed before any GPU launch: the run was never stopped after 4 evals (it watched a
  `training.log` only the Stage 4.1 driver writes), and the loop read its launch script from a dead
  scratchpad path.
- **Resume-check deviation from the Verification section below:** S resumes from a copy of R's own
  `checkpoint_step_1000` instead of re-running and killing at 1000. Same claim ("the resumed run's
  step-2000 eval matches an uninterrupted run's within 2 %"), same checkpoint (saves are atomic and
  complete before any kill), it additionally tests resume across processes — which is the real spot
  case — and it halves S's training.
- **Not yet checked on a GPU:** everything above (the retry loop has had no g6 capacity since
  2026-10-05 19:50 UTC).

## Goal
Cut a 10M full-MHC run from ≈ 5.7 h on 4× L4 to ≈ 2–2.5 h **without changing the optimization**,
and make runs resumable mid-arm so spot instances become usable.

## Measured cause (Stage 4.1 arm 2, read on the box)
- ESM-C already runs varlen flash attention, so padding is free. Sequence packing and "drop
  gradient checkpointing" are **not** fixes — the checkpointing flag has no consumer at all.
- ≈ 270 real tokens per forward, 16 micro-batches per optimizer step, a DDP all-reduce after
  every one of them (the loop never enters `no_sync`). CPU sat 95 % idle: launch-bound.
- `region_restricted_eval` ≈ 6.5 min per call on rank 0 while ranks 1–3 spin at the broadcast,
  every 500 steps — ≈ 30 % of wall.

## Changes, in order
1. **Eval**: tokenize + annotate the canonical eval once and cache it across calls; accumulate
   region losses with `index_add_` over region-id tensors instead of the Python loop; evaluate
   every 1000 steps. Expected ≈ 1 min per eval.
2. **Token-budget micro-batches + `no_sync`.** Per-bucket rows = `min(256, 16384 // bucket_max)`
   → ≤64: 256, ≤128: 128, ≤256: 64, ≤512: 32, ≤1024: 16 (today's proven worst case). Each rank
   steps after 256 rows, so an optimizer step is still 1024 rows on 4 GPUs.
   **Hard requirement — all ranks must be in the same bucket at the same step.** Today's
   `DistributedLengthBucketSampler` hands each rank a *stride* of one flat global index list
   (`samplers.py:201`), so with per-bucket micro-batch sizes two ranks could disagree on how many
   micro-batches make a step: DDP would then all-reduce different counts and hang or silently
   mis-average. So replace it with a **bucket-synchronised distributed batch sampler**: form
   per-bucket batches of `micro_rows × world_size` globally, shard each batch across ranks, and
   yield whole batches via `batch_sampler=`. Then the accumulation count is a property of the
   bucket and is identical on every rank, which is also what makes `no_sync()` on all but the
   last micro-batch safe.
3. **Throughput log**: rows/s and real tokens/s to W&B every logging step. Nothing records it
   today — the "4× slower" figure came from reading a progress bar.
4. **Mid-arm resume**: load `checkpoint_step_N` plus optimizer and scheduler state and skip
   N × 1024 rows. Today the driver resumes only per arm. This is what unlocks spot
   (g6.xlarge 0.42–0.62 spot vs 0.80 on-demand).

Raise the token budget above 16,384 only after a memory test.

## Cost
≈ 2–3 days engineering; ≈ USD 5 of GPU for the throughput and equivalence tests.

## Verification
- **Equivalence (the one that matters):** re-run arm 2's config for 2,000 steps under the new
  batching. **Pre-registered 2026-10-05, before the run:** PASS = `overall_ppl` and `cdr3_ppl` at
  step 2000 each within 2 % of Stage 4.1 arm 2's step-2000 eval; `pocket_ppl` is reported and
  flagged if > 5 % off, not gated at 2 %. Reason: pocket has the fewest tokens, and two runs of
  the *identical* config (Stage 4 Point D vs 4.1 arm D) already differed by 7 % on pocket while
  agreeing within 1 % on overall/cdr3 — a new sampler necessarily changes data order. This run
  doubles as 4.4's control.
- **Eval rewrite, A/B on the box:** old and new `region_restricted_eval` re-score the same
  adapter; every value must be **exactly** equal (the new loop sums in the same order).
- **No rank disagreement:** assert at every optimizer step that each rank's micro-batch count is
  equal (`dist.all_reduce` of the count, min == max); a desync fails loudly instead of hanging.
- **Eval rewrite is faithful within float tolerance:** every region's perplexity on a fixed
  adapter matches the old path to ≤ 1e-6 relative. Not bit-identical — vectorised accumulation
  changes the summation order, so demanding bit-equality would fail for the wrong reason.
- Peak memory < 20 GB per L4 on the ≤1024 bucket.
- **Resume:** kill at step 1000, resume, and the loss curve is continuous within noise; the
  resumed run's step-2000 eval matches an uninterrupted run's within 2 %.
