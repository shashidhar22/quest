# V04 audit repair — plan

Opened 2026-11-04 from the external Stage-4 audit. Verdicts and evidence:
`~/Documents/ACS/quest-cpt/audits/2026-11-04/triage.md`. Not a stage: this is the engineering
debt that stands between the stage programme and a GPU visit.

## Goal

A V04 visit that either produces a certified comparison against the locked endpoint, or refuses
and says why. Today it can do neither: before the repair it exited 0 having done nothing, and it
still cannot compute the comparison it exists to compute.

## Constraint that shapes everything

The six CPT re-baselines in V04 *are* the new baselines for the adopted objective. So the
objective must be settled before the visit, not after. Spending ≈10 GPU hours under an objective
that then changes buys nothing.

## Steps

1. **Launcher repair.** Done 2026-11-04. Process ownership, failure propagation, orphan-free
   aborts, dry-run isolation, four-GPU requirement, per-cell task selection, missing-pool abort,
   disclosed provenance waivers. Verified by `tests/test_v04_launcher.py`, 16 tests; 7 of them
   failed before the repair, each for a reason the audit named.
2. **Gate fail-closed.** `s42_check.py` must refuse, not return 0, when a requested readability
   read has no ranks; validate rank coverage, counts and finiteness; validate the eval's own pool
   size instead of trusting the CLI; separate an affordability refusal from a software error with
   distinct exit codes, and replace the chance-CI and zero-recall inferences with direct
   finite-score, score-spread and tie-rate checks. The launcher's current note that a zero gate
   does not prove ranks existed comes out when this lands.
3. **One comparator for the locked endpoint.** Paired per-molecule differences, resampling whole
   peptide groups, preserving the molecule-weighted estimator, reporting the estimate, its
   interval, peptide counts, measured rho, and a cluster sensitivity analysis. Require complete
   paired membership or account for exclusions explicitly; require valid ranks and compatible
   pool, draw, task and serialization; call provenance inside the comparator and fail before
   reporting. Fixtures must include repeated peptides and mismatched query sets. Then invoke it
   from the launcher, which today calls no comparison at all.
4. **Provenance over real artifacts.** Required fields rather than absent-matches-absent; hash the
   pool the judge actually loads, not the one the launcher names; record checkpoint and data
   hashes, the effective command, backend and version, world size and completed steps; keep CPT
   lineage distinct from SFT and eval settings, so a legacy adapter is never labelled with the
   newly adopted objective.
5. **Objective, after the decision.** Add the objective to resume compatibility with a stated
   policy for older checkpoints that lack the field. If the decision is an exact global
   masked-target mean, move to summed target loss with a globally reduced denominator, handle
   zero-target microbatches, and verify gradient equivalence across microbatch partitions and
   world sizes with fixed labels. The existing 15 %-tolerance test is replaced, not loosened.
6. **Run spec and plans.** Enforce the step budget and LR schedule or stop recording a step count
   as fact; immutable run IDs or validated resume so a restart cannot overwrite a completed run;
   pass gradient checkpointing deliberately or record why not; reconcile the stage index, the stage
   plans and the visit spec onto one endpoint; recompute visit duration from the exact cell and
   seed list.
7. **On-box preflight, then launch.** Preserve the uncommitted V03 scripts off the volume first.
   Then a bounded preflight of the exact command: backend and version actually loaded, input and
   mask hashes, same-seed repeat, memory and throughput. Only then the full visit.

## Verification

Every step lands with a test that failed first. Integration coverage runs from a fresh clone with
delayed jobs, missing checkpoints, trainer failures, mismatched manifests and missing ranks, and
requires nonzero exits and no orphaned children. Steps 3 and 5 need an environment with torch and
peft; this machine has none, so they cannot be called verified from a laptop run.

## Open decisions

Blocking, in order: the loss-normalization contract · the comparison-cell GPU budget · which
endpoint 4.4 and 4.7 use. See the triage document's closing section.
