# Diagnosing a nearly constant learned scorer

## Problem
A binding judge produced almost constant logits despite apparently informative validation AUROC.

## Approach
1. Preserve the original diagnostic and distinguish historical model behavior from unfinished code changes.
2. Test a normalized head and an unnormalized control on identical cached encoder features, using the real two-epoch budget.
3. Measure held-out BCE against the class prior and positive-versus-negative ranking within each molecule; global AUROC alone cannot exclude query offsets.
4. Follow the repair through serialization, batch collation, training, checkpoint save/load and diagnostic scoring with a real tokenizer and a tiny encoder.
5. Validate the full model: learning on cached features did not guarantee held-out BCE below the prior. Compare a receptor-free baseline and receptor substitutions before accepting global AUROC.
6. Audit perturbation controls before interpreting invariance: swapping in a receptor already known to bind the target is not a negative control. Exclude those pairs using the full reference, and report the reduced independent coverage.
7. Keep the candidate distribution separate from the known-positive reference: subsampling positives must not discard exclusion knowledge. Audit generated negatives against the full split before expensive fitting; never use held-out labels to filter training negatives.
8. Compare every tuned candidate with the constant prior inside training-only selection. Better held-out calibration is not a receptor-specificity verdict; a candidate that loses to the prior across training clusters is not validated by a favorable later point estimate. Check the folds before reading their loss: when negatives are other molecules' peptides, grouping by true peptide leaves each held-out peptide in the fit folds as a negative, and out-of-fold ranking falls far below chance (seen 2026-11-17, AUROC 0.06–0.12). Out-of-fold AUROC under 0.5 means the folds are broken, not that the prior won; draw negatives within each fold.
9. Review shared callers and batch boundaries. This exposed broken legacy compatibility, an obsolete fixture-generator API and zero-update checkpoints from drop_last.

## Judgment calls
Keep BCE and learning-rate defaults so the repair does not confound objective changes with representation changes. Preserve old Track A callers explicitly. Treat cached-feature success as evidence for head conditioning, not proof of receptor specificity or successful full-model training. Keep the paid pilot blocked until that separate experiment is complete.

## Reusable rule
When a learned scorer collapses, test feature learnability against a matched control, then verify every training-to-checkpoint boundary before promoting the repair to an experiment.
