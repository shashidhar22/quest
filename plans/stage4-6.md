# CPT Stage 4.6 — allele-frequency diagnostic (memorization check)

**Status:** planned — **step 1 costs nothing and should run alongside 4.2**
· **Depends on:** Stage 4.1 adapters and results (local in `docs/cpt_stage4.1/results/`)
· **Knowledge:** `~/Documents/ACS/quest-cpt/plan.md` step 5

## Goal
Answer Stage 3's unanswered Q2: does `pocket_ppl` measure pocket biochemistry, or recall of
frequently seen HLA chains? This governs how every `pocket_ppl` in the programme should be read,
which is why the free version runs early rather than last.

## Evidence
Stage 4.1 `pocket_ppl` tracks allele diversity rather than difficulty, in both full-MHC arms
(D / F): `II_partial_DP_beta` 1.350 / 1.316 < `II_partial_DQ_beta` 1.428 / 1.822 <
`II_complete` 1.800 / 2.006 < `II_partial_DR_beta` 2.131 / 2.465 < `II_partial_alpha`
2.314 / 2.487 ≪ **class I 7.555 / 8.538**. Class I is also the rarest stratum in the eval
(23 rows, ~0.5 % of pocket tokens). The canonical eval holds out no alleles at all.

## Approach
1. **Cheap form, 0 GPU.** Count each allele's row frequency in the run05 training set; bucket the
   canonical eval's pocket loss by that frequency (deciles) and plot perplexity against log
   frequency for arms A, D and F. Read the per-token losses out of
   `results/*/canonical_v2_results.json` and the training counts with DuckDB over the dataset.
2. **True hold-out, only if step 1 shows a steep slope.** Drop ~20 % of alleles from training,
   retrain arm D's config, and score the held-out alleles.

## Cost
Step 1: ≈ 1 h of CPU, no GPU. Step 2: ≈ USD 15–40 after 4.3.

## Verification
- A **steep** slope (say > 2× between the top and bottom frequency decile, with class I at the
  sparse end) means `pocket_ppl` is substantially allele recall. Then it stops being quoted as a
  biochemistry metric, and the 4.2 judge becomes the only decision metric.
- A **flat** slope supports the biochemical reading and step 2 is not worth buying.
- A failure of the method itself looks like fewer than ~5 populated deciles, or class I alone
  driving the slope — in that case the test is underpowered on 23 class-I rows and needs the
  hold-out instead.
