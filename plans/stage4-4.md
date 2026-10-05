# CPT Stage 4.4 — balanced vs proportional sampling

**Status:** planned · **Depends on:** 4.3 (equivalence run = the control); 4.2 for the downstream
read · **Knowledge:** `~/Documents/ACS/quest-cpt/plan.md` step 3

## Goal
Test whether balanced subset sampling fixes the class-I weakness, judged downstream — one
variable, matched rows, matched steps.

## Evidence
Stage 1 (matched, `max_length` 128 in both arms): class-I perplexity **25.1 proportional → 4.1
balanced**, and every class-II stratum improved too. Never carried forward: Stages 2, 3, 4 and
4.1 all used `*_proportional_10M`. Class I remains the weak regime (Stage 4.1: 7.6–8.5 vs
1.3–2.5 for class II).

## Two steps, cheap one first

**Step 1 — one run, no build, fully matched.** `run06_cdr3_pocket_balanced_10M` already exists in
S3, and Stage 4.1 **arm A is `run01_cdr3_pocket_proportional_10M` at the final settings**
(`max_length` 1024, effective batch 1024, 7,800 steps). So training run06 at byte-identical
settings gives a matched pair that differs **only** in the sampling distribution — the same shape
as the trustworthy Stage 1 comparison, on the current model and window. One run, ≈ USD 6–8 after
4.3.

**Step 2 — only if step 1 wins.** Build `run12_cdr3_fullmhc_balanced_10M`
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
- **Decision:** class-I `pocket_ppl` improves at matched steps **and** the 4.2 judge's Recall@10
  on `test_level4` improves outside the CI. Perplexity alone does not carry the decision; if the
  two disagree, the judge wins and that disagreement gets written up.
