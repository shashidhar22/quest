# CPT Stage 4.6 — allele-frequency diagnostic (memorization check)

**Status:** planned · **Depends on:** Stage 4.1 adapters · **Doc:** §9.1 step 5

## Goal
Answer Stage 3's open Q2: does `pocket_ppl` reflect pocket biochemistry or recall of
frequently seen HLA chains?

## Evidence
Point D's pocket_ppl tracks allele diversity (DP 1.30 < DQ 1.47 < II_complete 1.95 < DR 2.28
<< class I 11.45), and the canonical eval has no allele hold-out.

## Approach
1. **Cheap form, 0 GPU**: count each allele's rows in run05 training; stratify canonical-eval
   pocket loss by that frequency (deciles); plot ppl vs log frequency for arms 1–3.
2. **True hold-out, only if step 1 shows a steep slope**: drop ~20 % of alleles from training,
   retrain the arm-2 config, score held-out alleles.

## Cost
Step 1: an hour of CPU. Step 2: USD 15–40 after 4.3.

## Verification
A steep frequency → ppl slope (e.g. > 2× between top and bottom decile) means recall, and
`pocket_ppl` should stop being quoted as a biochemistry metric. A flat slope supports it.
