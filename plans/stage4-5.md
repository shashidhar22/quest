# CPT Stage 4.5 — pocket-upweighted masking

**Status:** planned · **Depends on:** 4.4 winner, 4.3, and 4.2 for the decision
· **Knowledge:** `~/Documents/ACS/quest-cpt/plan.md` step 4

## Goal
Test whether masking **composition**, not representation, is the lever: raise the mask rate on
MHC pocket positions while holding the overall rate at 15 %.

## Evidence
Stage 4 C vs D (both full MHC, both at 1024 — one of only two clean contrasts in the programme):
pocket tokens were 2.4 % of C's masked tokens against 8.1 % of D's, and C's `pocket_ppl` was 7×
worse. Stage 4.1 arm F adds the same signal: more TCR content dilutes pocket and CDR3 alike.
Under uniform 15 % masking, the pocket is a small, variable share of what the model is asked to
predict.

## Approach
1. **Builder**: emit per-row pocket spans (token indices) as a dataset column. Training rows
   carry no region labels today — the evaluator annotates regions in Python on every call, which
   is far too slow for millions of training rows.
2. **Collator**: per-position mask probability, `p_hi` on pocket positions and `p_lo` elsewhere,
   solved per row so the expected overall rate stays 15 %. One setting first (pocket ≈ 3×), no
   sweep.
3. Train on the 4.4 winner's data and config. Sequential, not a 2 × 2: three runs total across
   4.4 and 4.5.

**The eval is unaffected, verified.** `region_restricted_eval` masks a fixed 15 % of valid tokens
under a fixed `mask_seed=12345` (`cpt_training_lib.py:963,1089`), independent of the training
`mlm_probability`. So the measurement does not move with the treatment — the mistake that made
Stage 4 uninterpretable.

## Cost
≈ 2 days engineering. One run ≈ USD 10–20 after 4.3.

## Verification
- **Unit:** on 10k rows, the realised pocket mask rate ≈ `p_hi` and the overall rate ≈ 15 %
  (±0.5 pp). A failure looks like the overall rate drifting with the pocket share.
- **Sanity:** rows with no pocket tokens are masked exactly as before (unchanged byte-for-byte
  against the current collator on a fixed seed).
- **Decision: the 4.2 judge must improve outside its CI.** `pocket_ppl` improving is *expected*
  here — the model is trained on more pocket positions — so it is evidence the knob works, not
  evidence it helps. Secondary guard: `cdr3_ppl` within ±2 %, so the gain is not bought by
  degrading the TCR side.
