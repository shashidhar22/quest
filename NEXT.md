# NEXT — quest (updated 2026-11-03) · history: `docs/LOG.md` · stage status: `plans/README.md` · knowledge: `~/Documents/ACS/quest-cpt/`

**Box jobs live in `~/Documents/ACS/quest-cpt/box-jobs/`:** `README.md` = visit registry, decision
gates (**D1 decided, §3.1**), parallel plan for 4.4–4.7; `tests.md` = every test by stage with its
pass criterion; `visits/v03-2026-10-07.md` = that visit, closed out.

**No box is running.** V03 stopped itself 2026-10-08 04:08 UTC, 12.6 h after launch, inside its
22 h watchdog. Nothing idled; the week's delay cost time, not money. Instance
`i-0030c1a9c845b3714` is `stopped`, volume `vol-08f5626ccf18fdd4c` intact.

## 1. What V03 actually said, read 2026-10-15

Both headline verdicts were **unreadable as written**, and both are now diagnosed. Evidence:
`quest-cpt/findings.md` §9.5 (trainer) and §9.6 (judge). Decisions: `box-jobs/README.md` §3.1.

- **Stage 4.3 — `V2_FINAL DIVERGED-WORSE` is an artifact.** The run early-stopped at step
  **4000/7800** on `pocket`, the one metric that had stalled, while overall and cdr3 were still
  falling monotonically; it was then scored against Stage 4.1 at step 7800. The stop threshold
  (1 % improvement) sits inside the metric's own ~7 % inter-eval wobble and the measured 1.31 %
  run-to-run noise. `plan.md`'s "no early stopping in single-configuration runs" was never applied.
- **The real difference is the objective, not the ordering.** v2 ships row-share micro-batch loss
  weighting. Measured against the real sampler, long:short per-token gradient weight is
  token-uniform 0.150 · Stage 4.1 **0.054** · v2 row **0.023** · v2 token **0.168**. So v2 starves
  MHC tokens to 0.43× of Stage 4.1, **and Stage 4.1 was never token-uniform either** — its
  weighting is an artifact of batch size 16. "A faster trainer that learns the same model" has no
  yes available; the programme must pick an objective. v2 as shipped is **rejected for production**.
- **Stage 4.2 — the judge works; the gate was wrong; and its CIs were 3–4x too narrow.** Base ×
  `as_paired_i`: val AUC **0.702**. Peptide-clustered recall@10 against random 0.10 —
  `test_level4` **0.337 [0.172, 0.633]** (108 peptides), `test_iid` 0.144 [0.027, 0.353] (53),
  `test_novel_pep` 0.081 [0.043, 0.147] (463). The gate keyed on `test_iid`, could not resolve it,
  reported "the harness is broken" and skipped cells 2–4. **There is still no base-vs-arm-D
  number.**
- **Two claims from the first read are retracted.** `test_novel_pep` is **not** below random once
  the CI clusters over peptides, and the peptide-familiarity explanation is **refuted**: measured
  from the splits, **no condition shares a peptide or peptide cluster with training**, so
  `test_iid` is not in-distribution and `test_level4` is not "both seen" (it shares no TCR
  either). What does drive recall is **peptide length** against a pool that is 76 % 9-mers — in
  test_level4, 9-mers score 0.553 and 14-mers 0.013. The epoch-2 val AUC of 0.465 still needs the
  same cluster-aware treatment before it counts as anti-signal.
- **Training is not deterministic** at a fixed seed (1.31 % at step 2000), which voids every 2 %
  band used in 4.3, including both marginal `RESUME` verdicts.

## 2. Landed on the laptop 2026-10-15 (no GPU used)

- **Gate rewritten** — `scripts/training/stage42_box/s42_check.py` now gates on `test_level4` with
  three outcomes (PASS · UNDERPOWERED · BROKEN); only BROKEN stops the box, and a tie is caught by
  `MRR == 1/pool`. Tests `tests/test_s42_gate.py` (6) use the two **real** wave-1 eval files as
  fixtures, so the 10-08 misreading cannot come back. `wave1.sh`'s gate messages follow.
- **Recall CIs now cluster-bootstrap over peptides** — `cluster_bootstrap_ci` in
  `track_a/esm_sft/judge.py`; `n_peptides` is reported and `--min_peptides` (default 100) replaces
  the query count as the power floor. Tests `tests/test_judge_cluster_ci.py` (12) pin it against
  the real V03 cell, including the `test_novel_pep` retraction and that `test_level4` survives.
- **`track_a/shared/negative_sampling.py` resolves inside the checkout** instead of only at
  `/home/ubuntu/...`, so `judge.py` imports and is testable off the box at all.
- **Token-share weighting implemented** — `--loss_weighting {row,token}`, default `row` so earlier
  runs stay reproducible. `TokenBudgetStepSampler.step_token_plan()` supplies each micro-batch's
  token share. Tests `tests/test_micro_batch_loss_weighting.py` (9) pin all four weightings,
  including the correction that token-share is a **third** objective and not an equivalence fix.
- **Stage 4.6 report regenerated** — the `_relabel_none_stratum` repair now applies (so `none`
  becomes `II_partial_alpha`, 157 rows), every judged ratio carries a bootstrap CI, and the header
  states the two limits. Arm F's `II_partial_DR_beta` honestly moves to *inconclusive*
  [0.92, 2.28]; all three `II_complete` verdicts hold on their lower bounds.
- **Assumption corrections** — class-I perplexity is a **23-row / ~130-token** statistic over 11
  alleles with **zero HLA-A\*02:01**; `state.md` §3 and `plan.md` §7 blocker 3 are corrected, and
  the IMMREP25 reading retracted.

### Landed 2026-11-03 (no GPU used)

- **Length-matched distractor draw** — `judge._draw_distractors`, with
  `--distractor_draw length_matched` the default and `uniform` kept so the 10-08 cell reproduces
  bit for bit. Matching is **partial at non-9 lengths** because the pool holds only tens of
  peptides there, so the draw fills outward from the nearest lengths and every condition now
  reports `frac_distractors_same_length` and `n_queries_fully_length_matched`. Tests
  `tests/test_length_matched_distractors.py` (10): one reproduces the long-peptide collapse from a
  length prior alone, one shows matching recovers a weak genuine signal the uniform draw buried,
  and one checks a signal-free scorer is still not flattered.
- **`track_a/shared/load_splits.py` resolves `SPLITS_DIR` inside the checkout** instead of only at
  `/home/ubuntu/...`, with a `QUEST_SPLITS_DIR` override. Same defect as `negative_sampling.py`
  had. Its `load_candidate_pool` docstring also claimed the eval ranks all 3,564 peptides; it
  ranks a 100-candidate shortlist, which is what makes random Recall@10 = 0.10.
- **`4.3-objective` DECIDED: `--loss_weighting token`.** `--token_budget 0` rejected; re-baseline
  scoped to **arms A and D**, arm F dropped. Arithmetic, scope check and consequences:
  `quest-cpt/box-jobs/README.md` §3.2. The two branches cost about the same once (both owe a noise
  band) but `--token_budget 0` pays ≈3.9× on every remaining 4.4–4.7 run — ≈USD 350–460, and
  neither `4.4-s2` nor `4.7-run` would then fit inside a single visit — to stay continuous with a
  weighting that is an artifact of batch size 16, on a metric `4.6-s1` already demoted. **4.3 is
  CLOSED** and nothing downstream is gated on the trainer any more.
- **`4.3-tokw` and `4.3-noise` are folded into the re-baseline.** Arm D under token-share at
  **three seeds, 7,800 steps, no early stop** puts the band at the step the comparisons are read at
  rather than at 2,000, and the 2,000-step points of those curves carry the mechanism check. One
  job family (≈18 L4-h) replaces three.
- **`run_s43.sh` now passes `--loss_weighting` explicitly**, `LOSS_WEIGHTING=token` by default and
  logged in `run.log`'s start line. The trainer flag's default is still `row` so earlier runs
  reproduce, which made it a landmine: the launcher passed `--token_budget 16384` and no weighting,
  so it would have trained the rejected objective silently. Verified with
  `scripts/training/stage43_box/dryrun.sh` — PASS in all three layouts (4-GPU parallel, 1-GPU
  sequential, R2), `loss_weighting=token` in every start line. The dryrun stubs `torchrun`, so it
  proves the launcher plumbing, not that the trainer honours the flag; that is pinned separately by
  `tests/test_micro_batch_loss_weighting.py` (9), which needs `peft` and so runs on the box.
- **`--min_queries` removed** from the judge's eval CLI. It had been dead since the power floor
  moved to peptides, and `wave1.sh` was still passing it, so a query count looked like it gated
  something. `wave1.sh` now passes `--min_peptides` and `--distractor_draw`.
- **`run_s43.sh` early stopping OFF** (`--early_stop_n_patience 0`), and the `pocket` early-stop
  metric is gone with it. `plan.md` already required this for control runs and it had never been
  applied; it is what produced the artifact `V2_FINAL DIVERGED-WORSE`.
- **`plans/stage4-2.md` protocol text corrected** — a measured table of what the splits actually
  contain (every condition is peptide-disjoint from train), the harness check keyed to
  `test_level4`, the power criterion in peptides, and the decision rule stating that an
  inconclusive wave 1 means `as_paired_i` cannot answer rather than that CPT failed.
- **Verified no regressions:** 239 pass. The 17 failures in `test_standardization.py`,
  `test_standardizers.py`, `test_duplication_tracking.py` and `test_reformatter_scripts.py` are
  pre-existing (missing `tidytcells`, missing fixtures) and identical with my changes stashed.

## 3. First moves next session

Everything local that 4.2 needed is now done, so the next moves are a decision and a box visit.

0. **`aws sso login --profile DataUserAccess-953915750371`.** The token is expired, so nothing
   touching the box or S3 runs until this happens. Instance `i-0030c1a9c845b3714` is `stopped`.
1. **~~Decide `4.3-objective`~~ — DONE 2026-11-03: token-share.** See §2 and
   `box-jobs/README.md` §3.2. Nothing in 4.4–4.7 is gated on the trainer any more. What this
   decision still owes V04 is a **launcher for the re-baseline**: `run_stage41_pipeline.py` holds
   arms A/D/F but hard-codes early stopping (patience 3, `min_rel` 0.01, metric `pocket` —
   precisely what produced the artifact verdict) and passes no `--token_budget`, while
   `run_s43.sh` is hardwired to arm D and caps R at 4 evals. **Neither can run the re-baseline as
   written, and this was not built here because it cannot be verified off the box** (no `peft`
   locally, so the trainer will not even import). Build it as the first V04 action, or extend the
   4.1 driver with pass-through knobs and leave its defaults alone.
1b. **DONE 2026-11-04 — the comparison is not readable where the gate reads it.**
   `box-jobs/README.md` §3.3, code `scripts/training/stage42_box/readability.py` (+11 tests),
   effect pre-registered first in `box-jobs/prereg-4.2-effect.md`. `test_level4` + Recall@10 needs
   a **+0.257** Recall@10 effect to be detectable at ρ=0.8 against a pre-registered **+0.050**, so
   cells 2–4 as specified could not have answered. `test_novel_pep` + norm_rank resolves **+0.017**
   and comes out of the **same eval pass**, so the fix is free and no task switch is needed. Read
   the comparison there; keep the harness check on `test_level4`. Also: `test_level4`'s 108 peptides
   are only **18 pep_clusters**, so the independence unit is still not settled; and `as_trb_i`
   level4 is 335 peptides / 131 clusters, not the 48× its 9,734 molecules imply.

2. **Re-score the base cell first, on the same visit (≈1.25 h, eval only, no retraining).**
   **Recompute the readability numbers from it** — everything in §3.3 is power under the *uniform*
   draw, and the length-matched draw changes the pool. Move
   `base_paired/eval.json` aside so `cell()` stops skipping it, then let `wave1.sh` re-evaluate the
   existing `best.pt` under the length-matched draw. Doing this before cells 2–4 is what keeps the
   base-vs-arm-D comparison on one footing; doing it after means re-running three trainings.
3. **V04, 4× L4 (g6.12xlarge):** the re-baseline under token-share — **arm D at three seeds,
   7,800 steps, no early stop** (≈18 L4-h, one seed per GPU ≈6 h wall) and **arm A at one seed**
   (≈2 L4-h). Those three arm-D curves are simultaneously the noise band, `4.3-tokw`'s mechanism
   check at their 2,000-step points, and `4.4-s2`'s control. `4.2-wave1` cells 2–4 (≈2.5 h each)
   fill the fourth GPU, and `4.4-s2-build` (run12, ≈1 h CPU) runs alongside — it needs no GPU and
   no gate.
4. **Recover `/home/ubuntu/v2_full.sh`** from `vol-08f5626ccf18fdd4c`. It computed `V2_FINAL` and
   `V2_DETERMINISM`, was never committed, and survives only on that volume. Its comparison logic
   needs rewriting anyway, so recover it to record what was actually run.
5. **Audit the seed path** for the 1.31 % non-determinism (dataloader worker order, cuDNN
   algorithm choice, RNG restore on resume). Until it is explained, `RESUME_V2`'s 3.37 % stays
   suggestive rather than settled, and no equivalence claim tighter than ~3 % is defensible.
6. **Optional, laptop:** `4.5-eng` (pocket-span column + per-position mask collator) is still
   unblocked and still ≈2 days off the critical path.

## 4. Open and unverified

- **The 1.31 % non-determinism** — cause unknown; blocks every tight equivalence claim.
- **ρ, the cross-arm peptide-level correlation** — unmeasurable from the base arm alone, so every
  readability number is reported against it. The first CPT cell to land measures it and pins the
  predicate.
- **The peptide-length confound is coded but unmeasured** — the length-matched draw landed
  2026-11-03 and no number has been produced under it. Which way the headline moves is an open
  empirical question: 9-mers were never helped by the uniform draw, while long peptides were
  actively buried by it, so `test_level4`'s 0.337 could move either way. Matching is also partial
  at non-9 lengths, so the residual confound has to be read off
  `frac_distractors_same_length`, not assumed away.
- **Power** — only `test_level4` resolves anything on `as_paired_i`, and its interval is 0.46
  wide. `as_trb_i` carries 9,734 molecules on level4 and wave 1 treats it as secondary; that
  ordering is backwards on power grounds.
- **Arm A has never been judge-scored**, and `4.4-s1`'s control (arm A under the adopted trainer)
  does not exist.
- **A launcher that can actually run the re-baseline** — see first move 1. The decision is made;
  the means to execute it is not written, and it is the one thing standing between V04 and the box.
- **Push decision, Yuexin's:** branch `yxu2/tcrbench2-continue` has never been pushed — this laptop
  is still the only copy of the CPT code. Committed locally 2026-11-03 (first commit on this branch
  since 10-07), so the work is at least in git history now, but still on one disk.
