# CPT Stage 4.4 — balanced sampling at cdr3 × full MHC

**Status:** planned · **Depends on:** 4.2 (judge), 4.3 (throughput) · **Doc:** §9.1 step 3

## Goal
Test whether balanced subset sampling fixes the class-I weakness at the current best
serialization, judged downstream.

## Evidence
Stage 1 (matched, `max_length` 128 both arms): class-I perplexity 25.1 proportional → 4.1
balanced; every class-II stratum also improved. Never carried forward — Stages 2–4 all used
proportional data. Stage 4.1 arm 2 is the proportional baseline (`pocket_ppl` 1.862).

## Data
No balanced cdr3 × full set exists (run06/06b/06c are cdr3 × pocket). Build
`run12_cdr3_fullmhc_balanced_<N>` with `cpt_dataset_builder.py --distribution balanced`,
same seed and gates as run11. Balanced quotas push most rows into MHC-bearing subsets (24 of 31
subset keys), so rows become 180–540 tokens versus p50 17 now.

## Arms
1. **Token-matched screen** first: balanced rows chosen so total tokens ≈ run05's. ~1.5–2M rows.
2. **Row-matched** 10M run only if the screen wins on the judge.
Both: `max_length` 1024, effective batch 1024, fixed step budget, **no early stopping**.

## Cost
Build ≈ 1 h CPU. Screen ≈ USD 10–15. Row-matched ≈ USD 35–75 after 4.3.

## Verification
- Preflight: measured max tokens ≤ 1024; row count within ±1 % of target.
- Control: if 4.3 changed batching, the baseline is arm 2's config re-run under 4.3 (its
  equivalence run), not the original arm 2.
- Decision: class-I `pocket_ppl` improves **and** the 4.2 judge's Recall@10 on `test_level4`
  improves outside the CI versus arm 2.
