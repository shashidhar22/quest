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
5. **The objective. Decision made 2026-11-04; contract proved, loop integration pending.**
   The approved objective is the **exact global masked-target mean**: the summed loss over every
   supervised target in the optimizer step, divided by their count across all ranks.
   Done: the contract lives in `quest/training/loss_normalization.py` with CPU gradient tests that
   prove the accumulated gradient is invariant to the micro-batch partition and to the world size,
   that an empty micro-batch contributes nothing and stays finite, and that the old input-length
   scale fails that same invariance (a control, so the check cannot pass vacuously). The resume
   guard is in, and the objective is named `token_exact` = `global-masked-target-mean/v2` rather
   than a new version of `token`, so a name never describes behaviour the loop lacks. Asking for
   it today raises, by design.
   **What remains, and it is a loop change:** the scale is `world_size / N_global`, and `N_global`
   must be known before the step's FIRST backward. The loop pulls one micro-batch at a time, so it
   cannot know it yet. Landing this means: buffer the step's micro-batches, count supervised
   targets per micro-batch, one all-reduce of the sum (`global_target_count`), then forward and
   backward over the buffer with the summed loss (`summed_masked_loss`, or `sum_from_mean` on the
   HuggingFace path). Clipping stays where it is, once per step after accumulation. Then switch
   `token_exact` from raising to returning, and point new baseline launchers at it.
   Historical `token` runs keep `input-length-share/v1` and stay reproducible under that name;
   never relabel them.

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

## Decisions, made 2026-11-04

All three were approved. They constrain recipe and spend; they did not block the software work.

1. **Objective:** the exact global masked-target mean, versioned, with historical input-length
   share preserved and never relabelled. Step 5 above.
2. **Comparison budget:** judge-first. Re-score the existing base SFT checkpoint, then train and
   evaluate ONE paired legacy arm-D cell, and produce the correct comparison. Beta-chain cells,
   the six CPT baselines and 100M scaling are separately priced follow-ons. Launcher:
   `scripts/training/v04_box/judge_first.sh`; itemized specification awaiting approval to spend:
   `~/Documents/ACS/quest-cpt/box-jobs/visits/v04a-judge-first.md`.
3. **Endpoint:** `test_novel_pep / peptide / norm_rank` stays the primary inferential endpoint,
   with Recall@10 and the other conditions reported as secondary and peptide-cluster sensitivity
   required. The +0.05 Recall@10 practical-utility target remains a SEPARATE claim on its own
   condition: a normalized-rank effect does not establish it, and the comparator says so in its
   own output. Whether a clear effect licenses 4.4-4.7 is a spending rule to state prospectively.

## Still open

Not a decision, a measurement: whether the judge-first comparison is worth acting on, and the
g6.2xlarge hourly rate, which is recorded nowhere in this repo and must come from Cost Explorer
before launch.
