# CPT Stage 4.4 — balanced vs proportional sampling

**Status:** planned · **Depends on:** 4.3 (equivalence run = the control); 4.2 for the downstream
read · **Knowledge:** `~/Documents/ACS/quest-cpt/plan.md` step 3

## Goal
Test whether balanced subset sampling fixes the class-I weakness, judged downstream — one
variable, matched rows, matched steps.

## Evidence
Stage 1 (matched, `max_length` 128 in both arms, **ESM2-150M not ESM-C**): class-I perplexity
**25.145 proportional → 4.091 balanced** on `canonical_buckets/mhc_class/I` at `checkpoint_best`,
**23 rows / 155 masked tokens**. The same comparison reads 24.094 → 4.693 and 24.887 → 3.594 on the
final checkpoint's region-restricted v2 eval; all three agree on a 5–6× fall (findings §9.9 names
each source). Every class-II stratum improved too. Never carried forward: Stages 2, 3, 4 and
4.1 all used `*_proportional_10M`. Class I remains the weak regime (Stage 4.1: 7.6–8.5 vs
1.3–2.5 for class II).

> **How thin that evidence is — measured 2026-10-15 (findings §9.7).** The Stage 4.1 class-I
> figures above are computed on the canonical eval's class-I stratum, which is **23 rows and
> ~130 masked tokens**, 11 distinct alleles, 11 of the 23 being HLA-B\*08:03, and
> **no HLA-A\*02:01 anywhere in the eval**. Bootstrapped, Stage 4.1 arm D's class-I figure is
> 7.66 with a 95 % CI of [5.60, 10.93]; Stage 4 Point D, the same recipe, gave 11.448. The class I
> vs class II *gap* is real and large, so the motivation for this stage stands. The *level* is not
> a stable number, and none of it transfers to an A\*02:01 benchmark. **Unchecked:** whether Stage
> 1's 25.1 → 4.1 rests on the same thin stratum — it ran at `max_length` 128 on an earlier eval and
> its class-I row count was not recovered in this session. Count it before quoting that shift as
> this stage's justification. This is a second, independent reason — beside 4.6's recall
> finding — that the decision below rests on the judge alone.

## Two steps, cheap one first

**Step 1 — one run, no build, fully matched.** `run06_cdr3_pocket_balanced_10M` already exists in
S3, and Stage 4.1 **arm A is `run01_cdr3_pocket_proportional_10M` at the final settings**
(`max_length` 1024, effective batch 1024, 7,800 steps). So training run06 at byte-identical
settings gives a matched pair that differs **only** in the sampling distribution — the same shape
as the trustworthy Stage 1 comparison, on the current model and window. One run, ≈ USD 6–8 after
4.3.

**Step 2 — runs alongside step 1 on the same box visit (decided 2026-10-07, Yuexin).** Originally
"only if step 1 wins"; changed because step 1 is a proxy on pocket serialization while the
production recipe is full MHC, and gating step 2 on it would cost a whole extra box visit plus the
GPU-capacity wait. Cost now at risk if step 1 loses: step 2's build and run, ≈30–47 L4-hours at
v2's measured throughput — more than the USD 10–20 estimated below. Build `run12_cdr3_fullmhc_balanced_10M`
(`cpt_dataset_builder.py --distribution balanced --scale 10000000 --seed 42`, run11's gates) and
compare against arm D. This is the one that matters for the production recipe, but it needs a
build and carries roughly 5–8× run05's tokens, because balanced quotas push rows into the 24 of
31 subset keys that carry an MHC chain (p50 rises from ~17 tokens to roughly 100).

**Rejected: the earlier "token-matched screen".** Matching on *tokens* meant ~1.5–2M balanced
rows against 10M proportional at a fixed step budget, i.e. ~4.7 epochs against ~0.8. That
confounds sampling with data repetition and breaks this stage's own one-variable rule. Match
rows and steps; pay the token cost.

Both steps: `max_length` 1024, effective batch 1024, **fixed step budget and no early stopping**.

## Cost
Step 1 ≈ USD 6–8 (one L4, after 4.3). Step 2: build ≈ 1 h CPU on a cheap box, run ≈ USD 10–20
after 4.3.

## Verification
- **Preflight:** measured max tokens ≤ 1024 (the driver aborts otherwise); row count within
  ±1 % of 10M; the arm's `batch_size × grad_accum × n_gpus` = 1024 exactly.
- **Control:** if 4.3 changed the batching, compare against 4.3's equivalence re-run of arm A /
  arm D, never against the pre-4.3 numbers.
- **Decision (revised 2026-10-07 after Stage 4.6 step 1):** the 4.2 judge's Recall@10 on
  `test_level4` improves outside the CI. `pocket_ppl` is reported but **does not vote**: 4.6 step 1
  showed pocket loss largely tracks how close the eval's MHC chain is to one seen in training
  (`docs/cpt_stage4.6/step1/REPORT.md`), so a class-I `pocket_ppl` gain from balanced sampling
  would be partly the memorization it adds, not better biochemistry. The original rule required
  both; requiring the perplexity half would now reward recall.
