# CPT Stage 4.3 — trainer efficiency + mid-run resume

**Status:** planned · **Depends on:** nothing (engineering; can run in parallel with 4.2) · **Doc:** §9.2

## Goal
Cut a 10M full-MHC run from ≈ 5.7 h on 4× L4 to ≈ 2–2.5 h without changing the optimization,
and make runs resumable mid-arm so spot instances become usable.

## Measured cause (Stage 4.1 arm 2)
- ESM-C already runs varlen flash attention; padding is free. Packing and dropping gradient
  checkpointing are not fixes (the checkpointing flag has no consumer).
- ≈ 270 real tokens per forward, 16 micro-batches per optimizer step, all-reduce after each.
- `region_restricted_eval` ≈ 6.5 min per call on rank 0, ≈ 30 % of wall.

## Changes, in order
1. **Eval**: tokenize + annotate canonical eval once and cache; vectorize region-loss accumulation;
   eval every 1000 steps.
2. **Token-budget micro-batches**: per-bucket rows = `min(256, 16384 // bucket_max)`; accumulate
   until 256 rows per rank so every optimizer step is still 1024 rows; `no_sync()` on all but the
   last micro-batch; sampler yields index lists via `batch_sampler`.
3. **Throughput log**: rows/s and real tokens/s to W&B every logging step.
4. **Mid-arm resume**: load `checkpoint_step_N` + optimizer/scheduler state, skip N × 1024 rows.

## Cost
≈ 2–3 days engineering; ≈ USD 5 GPU for tests.

## Verification
- Equivalence: re-run Stage 4.1 arm 2 config for 2000 steps with new batching; `pocket_ppl` and
  `cdr3_ppl` at step 2000 within 2 % of arm 2's `checkpoint_step_2000` re-scored.
- Peak memory < 20 GB per L4 on the ≤1024 bucket.
- Eval result on a fixed adapter is bit-identical before and after the eval rewrite.
- Resume: kill at step 1000, resume, loss curve continuous within noise.
