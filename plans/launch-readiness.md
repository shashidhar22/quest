# Launch readiness — repair plan

Opened 2026-11-06 from the second external audit, which reviewed the first repair round and the
judge-first run specification. Verdicts and my reproduction:
`~/Documents/ACS/quest-cpt/audits/2026-11-06/triage.md`. The reviewer's probe, which I ran before
reading any code, is `reproduce.py` beside it; all eight of its lines reproduce on `59cafcd`.

Not a stage. This is what stands between the approved judge-first run and launching it.

## Goal

A judge-first visit whose result cannot be a lie: not stale, not CPU, not a NaN read as a perfect
rank, not past its ceiling, and not labelled with provenance it did not earn.

## Constraint that shapes everything

The first repair round made the launcher *finish* correctly. This round makes its outputs
*trustworthy*. Those are different properties, and passing the first says nothing about the
second: the reviewer produced a `comparison.json` with no arm-D training, no checkpoint and a
planted stale manifest, and the launcher exited 0.

## Status, 2026-11-06

All five steps landed. The audit's own eight probe results now read as refusals:
`~/Documents/ACS/quest-cpt/audits/2026-11-06/repair-verification.txt`, reproducible with
`repair-check.py` beside it. What remains is on the box and is listed under "Still open".

## Steps

1. **Fail-closed objective naming. Done.** `micro_batch_loss_scale` validates `weighting` before any
   return, so `token_exact` cannot be obtained through the fixed-batch path. `sum_from_mean`
   returns a finite zero for a zero-target micro-batch instead of `NaN * 0`. The loop integration
   stays deferred and named (`v04-audit-repair.md` step 5); this only closes the bypass.
2. **No non-finite score may become a rank. Done.** Lift the rank computation and the checkpoint
   trainable-key check into a light module, reject non-finite logits per query before ranking,
   require the full expected trainable key set from a checkpoint rather than merging over fresh
   weights, record score-spread and tie diagnostics, and make the probe load the actual
   checkpoint it is vouching for.
3. **An active deadline, and a bounded publish. Done.** The launcher waits on a child with a deadline
   rather than blocking in `wait`, terminates it through its process group on expiry, and exits
   nonzero. The watchdog bounds its publish command and shuts down regardless of a stuck upload.
   One reproducible command launches both and verifies the watchdog armed on the right pid.
4. **Artifact-derived provenance, and validated reuse. Done.** Required fields rather than
   absent-matches-absent. The evaluator records what it actually resolved and read — splits
   directory, pool path and hash, per-condition split hashes, checkpoint hash, adapter hash,
   device, versions, commit — and the launcher and comparator validate those recorded values
   rather than the launcher's own declarations. Reuse of an existing eval requires its manifest
   and checkpoint to be present and to match; a manifest is never overwritten on a reused output.
   Base, CPT and SFT lineage are separate fields, so a base arm is not labelled with a CPT
   objective it never had.
5. **A preflight that proves usability, not existence. Done.** A CPU validation phase for every required
   artifact, split file and import, then a CUDA requirement before any GPU step, the deployed
   commit recorded, and the evaluator refusing a silent CPU fallback.

## Verification

Every step lands with a test that failed first. The reviewer's own `reproduce.py` becomes a
regression: after the repair its eight lines must read as refusals, which is a falsifiable check
on the repair itself rather than on my description of it.

Off-box limits, stated rather than papered over: stubs cannot prove Linux process-group cleanup
against a real torchrun, GPU memory, throughput, or that a real checkpoint loads. Those are box
steps and stay in the visit record.

## What landed, by file

- `quest/training/loss_normalization.py` — validation before every return; `sum_from_mean`
  returns a finite zero for a zero-target micro-batch and documents that the zero is detached,
  so a DDP loop needing a graph-connected zero must take the summed path.
- `track_a/esm_sft/scoring.py` (new, NumPy only) — `rank_of` refuses a non-finite score and
  says what rank it would otherwise have produced; `score_diagnostics` records spread and ties
  even for an unusable vector; `missing_trainable_keys` derives the expected set from the live
  model; `incompatible_checkpoint_args` refuses a checkpoint trained for another task,
  serialization or adapter; `select_device` refuses a silent CPU fallback.
- `track_a/shared/artifacts.py` (new, stdlib only) — the ONE splits-resolution rule, which
  `load_splits` now delegates to, plus file and directory digests, the commit, and
  `cpt_lineage`, which reads an adapter's own recorded config. Arm D's `trainer_state.json`
  records batch 16 x accumulation 16 and no token budget, so its objective is
  `fixed-batch-mean/v0`, not the row share both manifests used to claim.
- `scripts/training/v04_box/provenance.py` — two schemas (`cpt_run`, `sft_cell`) with required
  fields, so absence refuses; separate CPT / SFT / eval lineage; `write --from-eval` derives
  the artifact fields from the evaluation's own record; `verify-eval` re-hashes what an eval
  says it read; `check_eval_provenance` is the one rule the launcher and the comparator share;
  and a comparison whose treatment keys are all identical is refused as one model against itself.
- `scripts/training/v04_box/preflight.py` (new) — imports, CUDA, pool, every split including
  train and val, adapter contents, and a checkpoint that must load and carry trainable weights.
- `scripts/training/v04_box/judge_first.sh` — the environment preflight before any GPU step; an
  active deadline in `run_gpu`; the watchdog handshake; the base probe loads the real base
  checkpoint; manifests written after and from each evaluation; reuse validated four ways.
- `scripts/training/v04_box/launch_judge_first.sh` (new) — one command that starts the run,
  arms the watchdog on the pid the run recorded, verifies arming from the watchdog's own log,
  and only then releases the run.
- `scripts/training/autostop_when_done.sh` — bounded publish and bounded log copy, so a wedged
  upload cannot postpone the stop; `SHUTDOWN_CMD` makes the real stop path testable.
- `scripts/training/stage42_box/paired_compare.py` — metadata required rather than merely
  equal; it applies the provenance rule itself; the result carries both arms' provenance and
  the treatment differences.

## Round two, 2026-11-06

A review of the repair above found five more defects
(`~/Documents/ACS/quest-cpt/audits/2026-11-06/round-two/`). Done:

6. **Run identity.** Fresh `runs/<run_id>/` per launch, a token the driver echoes into
   `run.identity`, a lock per output root, a waitable supervisor (`setsid --wait` or none), reaping
   only the driver this launch started, `launch_status.json` and `status.json` on every path.
7. **One provenance validator.** The comparator derives manifests from each eval and runs the
   launcher's own validation, plus a split digest for every condition it reads.
8. **SFT recipe from the checkpoint.** Seed, trained max_len and recipe from the checkpoint's own
   args; `unrecorded` refused; `eval_max_len` compared; train/val digests hashed into new
   checkpoints; sft_seed no longer waived for cells.
9. **Two tests of mine that could not fail**, and a preflight that crashed on a raising finder.

## Still open, and it needs the box

1. Linux process-group cleanup against a real torchrun, not a stub; and `setsid --wait` on the
   box, with the wrapper's exit status equal to the run's.
2. That a real checkpoint loads and the probe passes on it.
3. GPU memory, throughput, and the deployed revision (the preflight records the commit; only a
   run on the box can say what that commit is there).
4. The exact-objective loop change, before any new CPT baseline.

## Deliberately not in this round

The exact-objective loop change (audit finding 4's full acceptance criteria). It needs a GPU to
validate, and the reviewer agrees it does not block legacy-adapter judge-first SFT. Step 1 closes
the naming bypass so nothing can be trained under a name the loop does not implement.
