# V04 audit repair — plan

Opened 2026-11-04 from the external Stage-4 audit, revised the same day after the reviewer's
response to the first repair. Verdicts and evidence:
`~/Documents/ACS/quest-cpt/audits/2026-11-04/triage.md`; the reviewer's follow-up and its probes:
`claude-handoff.md` and `triage-evidence/` beside it. Not a stage: this is the engineering debt
between the stage programme and a GPU visit.

## Goal

A V04 visit that either produces a certified comparison against the locked endpoint, or refuses
and says why. Today it can do neither: before the repair it exited 0 having done nothing, and it
still cannot compute the comparison it exists to compute.

## Constraint that shapes everything

The six CPT re-baselines in V04 *are* the new baselines for the adopted objective. So the
objective must be settled before the visit, not after. Spending ≈10 GPU hours under an objective
that then changes buys nothing. What does *not* wait on that decision: refusing a resume that
changes objective, and every repair below except step 5.

## Steps

1. **Launcher lifecycle. Done 2026-11-04**, in two passes. Normal completion: process ownership,
   failure propagation, orphan-free aborts, phase ordering. Abnormal completion: signal cleanup,
   so TERM to the driver takes the training with it rather than leaving trainers on the GPUs.
   Jobs lead their own process group under `set -m`, are `exec`ed so the tracked PID is the real
   process, evaluations are tracked rather than run in the foreground where a trap would wait
   behind them, and cleanup is bounded TERM → wait → KILL → reap with the driver alive throughout.
   Plus a P0 preflight of every path, checkpoint and config before a GPU job starts; dry-run
   isolation; the four-GPU requirement; per-cell task selection; a missing pool aborting;
   disclosed provenance waivers. `tests/test_v04_launcher.py`, 20 tests.
   **Not proven:** group cleanup of a real torchrun's workers. The tests use stubs; check it on
   Linux in the preflight.
2. **Gate fail-closed.** `s42_check.py` must refuse, not return 0, when a requested readability
   read has no ranks; validate rank coverage, counts and finiteness; validate the eval's own pool
   size instead of trusting the CLI; and use three distinct outcomes — valid proceed, scientific
   spending refusal, technical error — so a missing rank file can never read as a successful
   refusal. The launcher must stop on a technical error before P2. Replace the chance-CI and
   zero-recall inferences with direct finite-score, score-spread and tie-rate checks.
3. **One comparator for the locked endpoint.** Paired per-molecule differences, resampling whole
   peptide groups, preserving the molecule-weighted estimator, reporting the estimate, its
   interval, peptide counts, measured rho, and a cluster sensitivity analysis. Require complete
   paired membership or account for exclusions explicitly; require valid ranks and compatible
   pool, draw, task and serialization; call provenance inside the comparator and fail before
   reporting. Fixtures must include repeated peptides and mismatched query sets. Then invoke it
   from the launcher and require its output, since today nothing calls it.
   Put the statistics in a light module beside `readability.py`: they need NumPy, not torch, and
   the only reason they look GPU-bound is `judge.py`'s import chain. The judge CLI becomes a thin
   consumer. The assumed rho is a planning parameter, never post-result evidence.
4. **Provenance over real artifacts.** Required fields rather than absent-matches-absent; hash the
   pool, splits and checkpoints actually read, not the ones the launcher names; record the
   effective command and config version, backend and version, world size, completed steps, draw
   algorithm and seed, endpoint, and keep CPT lineage distinct from SFT and eval settings, so a
   legacy adapter is never labelled with the newly adopted objective. Allow intended treatment
   differences explicitly and reject the rest before a verdict.
5. **The objective itself, after the decision.** The resume guard is **done**: the objective is
   matched on resume, versioned so redefining `token` cannot pass under the same word, failing
   closed on a checkpoint recording no semantics with one verified override. Rule in
   `quest/training/resume_compat.py`, 13 tests, no torch needed. What waits on the decision is
   implementing the chosen contract: if it is an exact global masked-target mean, move to summed
   target loss with a globally reduced denominator, handle zero-target microbatches and rank
   imbalance, fix clipping order, and verify gradient equivalence across microbatch partitions and
   world sizes with fixed labels. Add the new semantics as a new version; never relabel historical
   `token` runs. The existing 15 %-tolerance test is replaced, not loosened.
6. **Run spec and plans.** Enforce the step budget and LR schedule or stop recording a step count
   as fact; immutable run IDs or validated resume so a restart cannot overwrite a completed run;
   record that `--gradient_checkpointing` is inert (a config field at `cpt_training_lib.py:1494`
   that nothing reads) rather than adding it and expecting a memory change; reconcile the stage
   index, the stage plans and the visit spec; recompute visit duration from the exact cell and
   seed list.
7. **Comparison cells.** Wire train-then-evaluate for the missing SFT cells, validate reusable
   checkpoint metadata, and scope failure per comparison: a failed beta-chain extension should not
   erase a valid paired primary result, while required work that fails still fails the visit.
8. **On-box preflight, then launch.** Preserve the uncommitted V03 scripts off the volume first.
   Then a bounded preflight of the exact command: backend and version actually loaded, input and
   mask hashes, a same-seed repeat, completed-step schedule, memory, throughput, and the Linux
   check on process-group cleanup from step 1. Only then the full visit.

## Verification

Every step lands with a test that failed first. Integration coverage runs from a fresh clone with
delayed jobs, missing checkpoints, trainer failures, mismatched manifests, missing ranks and
signals during each phase, and requires nonzero exits and no orphaned children.

Off-box coverage is wider than first assumed: the full statistical contract, metadata
compatibility, fixtures, gate exit codes and launcher invocation all test on a laptop once the
logic is lifted out of the torch import chain. `resume_compat.py` is the worked example. Only
model loading, trainer integration and CUDA/DDP numerics need the ML environment, and a stub run
is never numerical validation.

## Open decisions

Blocking, in order: the loss-normalization contract · the comparison-cell GPU budget · how the
locked primary endpoint relates to the declared Recall@10 utility criterion. The reviewer's
recommendations on all three, including a cheaper judge-first option that answers the first
missing comparison without buying six CPT baselines, are in `claude-handoff.md`. They are
recommendations; the spending and recipe calls are Yuexin's.
