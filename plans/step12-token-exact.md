# Corpus repair step 12, engineering half — `token_exact` in the loop, preflight in the trainer

Opened 2026-11-11. Not a stage. Step text: ACS `quest-cpt/audits/2026-11-09/corpus-repair-plan.md`
step 12 and its amendment; gates: ACS `quest-cpt/audits/2026-11-10/new-cpt-plan.md` gates 5-6.
Lightweight plan.

## Status 2026-11-11

Steps 1-5 done on the laptop, commits `fcfc2ed` and the review fixes after it. Independent
verifier: nine mutations all caught, real 2-process DDP (gloo) with uneven and zero-target
ranks matches a single-process reference to 1e-17, the historical `token`, `row` and
fixed-batch paths are bit-identical to the commit before. Codex: three findings fixed (cache
key now carries the vocabulary; the preflight is bound to `<version>/dataset`; the logged loss
is the global mean). Not done: a real-GPU smoke (NCCL, bf16, flash attention), pricing from
measured throughput.

## Goal

A trainer that can run the approved objective (`token_exact` = `global-masked-target-mean/v2`)
and refuses a corpus that does not pass `corpus_preflight.corpus_problems`. No training is
launched here; a GPU smoke and a priced pilot are separate decisions.

## Approach

- **Step buffering as a generator.** `loss_normalization.exact_step_batches(batches, plan,
  world_size)` reads one optimizer step's micro-batches, counts their supervised targets, does
  one all-reduce, and yields each micro-batch with the step's scale `world_size / N_global`. The
  loop's other paths are untouched and stay bit-identical.
- **In the loop**, `token_exact` takes the summed loss over float32 logits times that scale.
  It requires `token_budget > 0` (the sampler's step plan is the step boundary); asking for it
  with fixed batches refuses at construction.
- **Preflight.** `CPTConfig.corpus_preflight = "required"` runs `corpus_problems` on the parent
  of `dataset_path` before any model loads and refuses on a non-empty answer. `token_exact`
  refuses unless the preflight is required: the new objective is only for the new corpora.
- **Not in scope:** the comparison's frozen numbers, `experiment-lock.json`, the existing-
  checkpoint diagnostic, any box visit.

## Steps

1. `exact_step_batches` + tests: gradient equals one concatenated reference for variable
   lengths, uneven target counts, a zero-target micro-batch, uneven micro-batch counts per step
   and two simulated ranks; plan/loader mismatch and a zero-target step refuse.
2. Loop and CLI; a test that runs the REAL `CPTTrainer` on CPU for one step and compares the
   gradient it clips against the reference recomputed from the batches it consumed, with the
   `token` weighting as a control that must differ.
3. Preflight in the trainer + tests (refuses a failing corpus; passes a complete one).
4. Resume across a `token_exact` checkpoint continues the data order; objective drift refuses.
5. Review (verifier + Codex: this is the training objective), then price the GPU smoke and pilot.

## Verification

- New tests fail before each change and pass after; the existing loss, resume and sampler
  tests pass unchanged.
- Failure looks like: a gradient that depends on how the step was cut, a NaN from an empty
  micro-batch, a run that starts on an unpublished corpus, a step count that moves.
