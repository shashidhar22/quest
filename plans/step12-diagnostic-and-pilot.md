# Corpus repair step 12, spending half — checkpoint diagnostic and the 10% two-arm pilot

Opened 2026-11-11. Lightweight plan, nodded 2026-11-11. Gates: ACS
`quest-cpt/audits/2026-11-10/new-cpt-plan.md` gates 4, 6, 7. Preparation only: nothing here
starts EC2. Each visit is a separate go-ahead.

## Status 2026-11-11, after visit DG1

DG1's stage A ran (≈ USD 0.3); stage B was stopped and PL1 is on hold. Both V04a judges score
every candidate within 1e-04 of one logit, the training base rate; the SFT recipe this plan
freezes for the pilot is the one that produced them. Judge repair comes first (ACS
`quest-cpt/findings.md` 9.13). Also wrong in this plan: the diagnostic's price. It scores at
2.4 s per 100 candidates, not 0.7, so stage B needs about 63 minutes per checkpoint and a
ceiling near 3 h.

## Status 2026-11-11, before any box visit

Steps 1-8 done on the laptop; nothing has run on the box. Built: `diagnostic_analyze.py`,
`pilot12_box/run_diagnostic.sh`, `pilot12_box/experiment_lock.py` with `experiment-lock.json`
(sha256 `82b9e7fc…`, 57 code files pinned), `pilot12_box/run_pilot.sh`, the initial-LoRA hash in the trainer. Both
V04a `best.pt` files and the arm D adapter are on S3 (listed 2026-11-11). The 300-query
manifest rebuilds to the recorded hash. Seed 42 gives one initial LoRA hash in two processes
on the real ESM-C weights, another for seed 43.

Independent verifier: the spend boundary held under the wrapper; it found four real defects,
all fixed with tests that fail without the fix: a `pilot.json` reused after a cell was re-run,
a lock re-read on every call instead of held, a `status.json` that could be stale or missing,
and a direct launch outside the wrapper training unwatched.

Codex (Astra, tier hard): nine findings, all fixed with tests. The lock now pins every repo
module the trainer, the judge and the read-out import (computed, 57 files), where a hand-kept
list of 12 had left out the samplers, collators, SFT model and negative sampler. `check-cpt`
refuses an empty state file and an adapter that is truncated or not finite; `check-eval`
refuses a smoke evaluation. The drivers refuse a `POLL` or `SHUTDOWN_GRACE` the watchdog could
not use, hold both adapters to their digests after the pair check, and no longer excuse the
watchdog under `ALLOW_CPU` on a machine with a GPU. The diagnostic reuses an output only when
it names this manifest and checkpoint; its analysis takes clusters and donors from the
manifest; `pilot_compare.py` saves no positional verdict label.

42 protections were removed one at a time; each is caught by a test. The evaluation check was
also run on the two real V04a evaluation files: it flags the conditions, the CPT lineage and
the missing training digests of the old base, and nothing else.

Changed from the values first proposed: the trainer's benchmark-hash check is always on (from
the lock); the CPT evaluation cadence is one step past the cap, so the canonical evaluation
runs once per arm; `check-eval` binds each P / E cell to its own arm's adapter by digest.

Left open by the reviews: an SFT checkpoint is bound to its adapter at evaluation, not at
training; base-model weights, tokenizer and third-party packages are not pinned by digest;
the trainer still only warns when its final adapter save fails (`check-cpt` catches the
result). All three are in the lock's `limitations`.

Not established: anything on a GPU. The 2% target bounds rest on 45 smoke steps per corpus
(about 1% standard error per arm), so `check-pair` can stop the pilot after the CPT hour by
chance. The canonical evaluation and the validation evaluation are untimed. GNU `timeout`
and `setsid --wait` paths were not exercised on the laptop. Visit notes: ACS
`quest-cpt/box-jobs/visits/dg1-checkpoint-diagnostic.md`, `pl1-two-arm-pilot.md`.

## Goal

Both spends ready to launch with one command each, every open number frozen in
`experiment-lock.json`, and a laptop dry run of each driver passing.

## Proposed frozen values (the nod is on these)

Diagnostic (gate 4), exploratory, prints no pass/fail label:
- Checkpoints: V04a `base_paired/best.pt` and `armD_paired/best.pt`, `--precision native`
  (fp32 is not implemented; stated in the report, not built).
- Stage A: the saved 15-query manifest (`diagnostic-pilot.json`): load, repeat, reversed
  order, ties, seconds per scoring.
- Stage B: the 100-per-condition manifest, rebuilt and required to reproduce the hash in
  `diagnostic-coverage.json`. Replaceable queries 26 / 75 / 33 (iid / novel_pep / level4);
  the unmatched 74 / 25 / 67 are reported, not dropped silently.
- Read-out: Δ = original − replaced normalized rank (midrank), five donors averaged per
  query, peptide-cluster bootstrap, interval shown against ±0.02 as a description only.
- Ceiling: `MAX_HOURS=2`, about USD 2. Expected about 1 h (8 scorings per query, 0.7 s each
  from V04a; not measured on this scorer).

Pilot (gates 6-7):
- Full budget = the smaller supply: one epoch of the proportional corpus, 23.0 M masked
  targets. Pilot = 10% = 2.30 M targets.
- Same targets per step and same updates in both arms: P at 16 × 64 = 1,024 rows per step,
  E at 16 × 47 = 752 rows per step, both about 2,950 targets per step, both
  `--max_steps 780`, token budget 8192, lr 2e-4, warmup 0.05, cosine sized to 780.
  Realized targets are recorded; arms more than 2% apart refuse the SFT stage.
- CPT checkpoint: `final/` at step 780, no selection. Early stop off. Seed 42.
- Same initial LoRA state: the trainer records a sha256 of the initial LoRA tensors; unequal
  hashes refuse. Masking has no dedicated RNG stream; recorded as a limitation, not fixed.
- SFT: B, P, E fresh under the V04a recipe (lr 1e-4, batch 16, ≤ 2 epochs, patience 1, val
  AUROC selector, seed 42). Evaluation `--conditions val` only, length-matched, 5,000 queries.
- Read-out: `pilot_compare.py`, contrasts E-B (primary), P-B, E-P; triage only.
- Ceilings: CPT stage 2 h; whole visit `MAX_HOURS=24`, about USD 26. Expected about 19 h,
  about USD 20 (CPT 1.2 h; SFT 3 × 4.9 h from V04a; val evaluation not measured).

Open decision carried into the lock: gate 3 (evaluation-population freeze) is not done. The
pilot scores validation only with the current evaluator and keeps every test part closed.

## Steps

1. `track_a/esm_sft/diagnostic_analyze.py`: stability summary and Δ read-out; tests on fake results.
2. `pilot12_box/run_diagnostic.sh` under `launch_judge_first.sh` (identity, watchdog, ceiling, status on every exit); stub dry run.
3. Trainer records the initial LoRA hash; test that two corpora with one seed give one hash.
4. `pilot12_box/experiment_lock.py` (`write`, `verify`) and `experiment-lock.json`; verify refuses a changed corpus marker, commit, split digest or recipe.
5. `pilot12_box/run_pilot.sh`: verify lock → CPT P → CPT E → init-hash and target checks → SFT B/P/E → val evaluation → `pilot_compare.py`; resumable per cell; stub dry run.
6. Read-only S3 check that both V04a `best.pt` files and the arm D adapter exist.
7. One combined review, then `verifier` on both drivers (they are the cost boundary).
8. Draft the two visit notes and their tests in ACS `quest-cpt/box-jobs/`.

## Verification

- Each new test fails before its change and passes after; existing trainer, resume, judge and
  launcher tests pass unchanged.
- Dry runs: a stage that fails, a ceiling that expires and a lock that does not match each
  stop the driver with a non-zero status and no later stage started.
- Failure looks like: a GPU step without an armed watchdog, an arm trained off the lock, a
  test condition scored, a diagnostic that emits a verdict.
