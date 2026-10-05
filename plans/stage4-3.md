# CPT Stage 4.3 — trainer efficiency + mid-run resume

**Status:** planned, can run in parallel with 4.2 · **Depends on:** nothing (engineering)
· **Knowledge:** `~/Documents/ACS/quest-cpt/findings.md` (GPU-efficiency section)

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
  batching; `pocket_ppl` and `cdr3_ppl` at step 2000 within 2 % of arm 2's own
  `checkpoint_step_2000` re-scored. This run doubles as 4.4's control.
- **No rank disagreement:** assert at every optimizer step that each rank's micro-batch count is
  equal (`dist.all_reduce` of the count, min == max); a desync fails loudly instead of hanging.
- **Eval rewrite is faithful within float tolerance:** every region's perplexity on a fixed
  adapter matches the old path to ≤ 1e-6 relative. Not bit-identical — vectorised accumulation
  changes the summation order, so demanding bit-equality would fail for the wrong reason.
- Peak memory < 20 GB per L4 on the ≤1024 bucket.
- **Resume:** kill at step 1000, resume, and the loss curve is continuous within noise; the
  resumed run's step-2000 eval matches an uninterrupted run's within 2 %.
