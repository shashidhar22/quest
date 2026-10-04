# CPT Stage 4.7 — data scaling (10M → 100M)

**Status:** planned · **Depends on:** 4.2 judge positive, 4.4/4.5 recipe fixed, 4.3 · **Doc:** §9.1 step 6

## Goal
Run the scaling curve the original Stage 4 was meant to be, at the final recipe, judged
downstream.

## Evidence
`foundation_100M` was used only by Gen C; `foundation_500M` never. All Gen D stages trained
on 10M. The planned 1M/10M/100M curve never ran.

## Approach
1. Build a 100M set at the winning serialization and sampling (run08 is cdr3 × pocket
   proportional, so a new build is needed). ≈ 9 h CPU at run11's rate, on a cheap CPU box.
2. Train one epoch, fixed step budget, same config as the 10M winner.
3. 500M only if 10M → 100M moves the judge.

## Cost
Run ≈ 10× a 10M run: USD 80–150 proportional after 4.3; several hundred if balanced at matched
rows. Needs 4.3's mid-run resume to use spot safely.

## Verification
Recall@10 on `test_level4` improves 10M → 100M outside the CI. Perplexity alone does not count.
Stages 7 (ESM-C 600M) and 8 (production build) stay deferred until this lands.
