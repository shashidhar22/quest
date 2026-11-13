# Diagnosing a nearly constant learned scorer

## Problem
A binding judge produced almost constant logits despite apparently informative validation AUROC.

## Approach
1. Preserve the original diagnostic and distinguish historical model behavior from unfinished code changes.
2. Test a normalized head and an unnormalized control on identical cached encoder features, using the real two-epoch budget.
3. Measure held-out BCE against the class prior and positive-versus-negative ranking within each molecule; global AUROC alone cannot exclude query offsets.
4. Follow the repair through serialization, batch collation, training, checkpoint save/load and diagnostic scoring with a real tokenizer and a tiny encoder.
5. Validate the full model: learning on cached features did not guarantee held-out BCE below the prior. Compare a receptor-free baseline and receptor substitutions before accepting global AUROC.
6. Review shared callers and batch boundaries. This exposed broken legacy compatibility, an obsolete fixture-generator API and zero-update checkpoints from drop_last.

## Judgment calls
Keep BCE and learning-rate defaults so the repair does not confound objective changes with representation changes. Preserve old Track A callers explicitly. Treat cached-feature success as evidence for head conditioning, not proof of receptor specificity or successful full-model training. Keep the paid pilot blocked until that separate experiment is complete.

## Reusable rule
When a learned scorer collapses, test feature learnability against a matched control, then verify every training-to-checkpoint boundary before promoting the repair to an experiment.
