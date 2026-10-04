# CPT Stage 4.5 — pocket-upweighted masking

**Status:** planned · **Depends on:** 4.4 winner, 4.2, 4.3 · **Doc:** §9.1 step 4

## Goal
Test whether masking composition, not representation, is the lever: raise the mask rate on
MHC pocket positions while holding the overall rate near 15 %.

## Evidence
Stage 4 C vs D (both full MHC, both 1024): pocket tokens were 2.4 % of C's masked tokens vs
8.1 % of D's, and C's `pocket_ppl` was 7× worse. Uniform masking dilutes the pocket signal.

## Approach
1. **Builder**: emit per-row pocket spans (token indices) as a column. Training rows carry no
   region labels today; the eval annotates in Python per call, far too slow for millions of rows.
2. **Collator**: per-position mask probability — pocket p_hi, elsewhere p_lo, chosen so the
   expected overall rate stays 15 %. One setting first (e.g. pocket 3×), no sweep.
3. Train on the 4.4 winner's data and config. Sequential, not 2 × 2: three runs total across
   4.4 and 4.5.

## Cost
≈ 2 days engineering. One run ≈ USD 10–20 after 4.3.

## Verification
- Unit: on 10k rows, realized pocket mask rate ≈ p_hi and overall ≈ 15 % (±0.5 pp).
- Decision: `pocket_ppl` improves, `cdr3_ppl` within ±2 %, and the 4.2 judge does not regress.
