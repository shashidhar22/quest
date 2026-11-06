# NEXT — quest (updated 2026-11-06, round two) · history: `docs/LOG.md` · stages: `plans/README.md` · knowledge: `~/Documents/ACS/quest-cpt/`

**V04 as specified is cancelled; the next run is JUDGE-FIRST and awaits approval to spend.**
Decisions 1-3 were approved 2026-11-04. A second external audit on 2026-11-06 found five
readiness defects in the first repair; all five are now repaired off-box. Audits, triage and the
before/after check: `~/Documents/ACS/quest-cpt/audits/2026-11-06/` (`launch-audit.md`,
`aws-checks.md`, `triage.md`, `repair-verification.txt`). Repair order:
`plans/launch-readiness.md`, then `plans/v04-audit-repair.md`.

**No box is running.** `i-0030c1a9c845b3714` is stopped, **g6.2xlarge · 1× L4**, us-west-2a,
instance-initiated shutdown behaviour `stop`, root volume `vol-08f5626ccf18fdd4c` (gp3, 1,000 GiB,
encrypted, DeleteOnTermination true — stopping preserves it, terminating would not). The
judge-first run needs **one GPU, so no resize**. The effective rate is now measured, not guessed:
**USD 1.068988 per GPU-hour** (Cost Explorer, October, usage type `USW2-BoxUsage:g6.2xlarge`,
non-estimated: USD 32.050349 / 29.981945 h). So ≈4.9 h ≈ **USD 5.24**, a gate refusal ≈1.3 h ≈
**USD 1.39**, the 8-hour ceiling ≈ **USD 8.55**, excluding storage, snapshots and transfer.

1. **`git push`** the local commits. Claude's sandbox refuses push as a public-surface action, so
   Yuexin runs it.
2. **Approve or amend the judge-first spend**, then launch it with ONE command:
   ```sh
   PUBLISH_DEST=s3://fh-pi-warren-h-eco/quest/tcrbench2/jf_out \
     bash scripts/training/v04_box/launch_judge_first.sh
   ```
   **If a visit is interrupted, resume it** rather than starting over: add `RESUME=<run_id>`
   (the directory name under `jf_out/runs/`). The base evaluation and a trained arm-D cell are
   then reused after validation, so a stop during the arm-D evaluation repays only that step.
   Each launch gets its own directory, `/home/ubuntu/jf_out/runs/<run_id>/` (`jf_out/latest`
   points at it), and publishes only that directory, to `$PUBLISH_DEST/<run_id>/`. The wrapper
   takes a lock on the output root, starts the run under a waitable supervisor, reads the pid
   from an identity file carrying a token it generated, arms the watchdog on that pid, verifies
   arming from the watchdog's own log, and only then releases the run. On any failure before
   release it terminates and reaps the driver it started and signals nothing else. Its exit
   status is the run's. Outcomes: `runs/<id>/status.json` (complete / refused / failed) and
   `launch_status.json`. Spec:
   `~/Documents/ACS/quest-cpt/box-jobs/visits/v04a-judge-first.md`. Pre-launch, in order:
   `aws sso login --profile DataUserAccess-953915750371`; a read-only hashed copy of
   `/home/ubuntu/v2_full.sh` and any surviving uncommitted `track_a/` off the volume, the only
   record of what computed the two failed V03 verdicts; then the command above.
3. **Verify on the box what the laptop cannot:** that `setsid --wait` exists there and the
   wrapper's exit status equals the run's (the laptop has no setsid; only a source-equivalent
   shim has exercised that path); that terminating the launcher kills a *real*
   torchrun's descendant workers, not only a stub; that the real base checkpoint loads and the
   probe passes on it (`judge probe --ckpt`); GPU memory and throughput; and what commit is
   actually deployed — the preflight records it, but only a run on the box can say which.
4. **Then, before any CPT baseline:** land the exact-normalization loop change (step 5 of
   `plans/v04-audit-repair.md`). The objective is decided and gradient-tested, but the loop still
   cannot supply the step's global target count before its first backward, so `token_exact`
   raises — now on every path, including the fixed-batch one that used to return 1/n_acc without
   looking at the name.

**Done 2026-11-06, round two.** A second review of my repair found five more defects, four in
code I had written that day; all reproduced before I read my own code, all repaired
(`~/Documents/ACS/quest-cpt/audits/2026-11-06/round-two/`: `review.md`, `triage.md`,
`repair-verification.txt`). The launch wrapper armed the watchdog on a stale pid file and could
SIGTERM an unrelated process; under `set -m`, util-linux `setsid` forks and returns 0 at once,
so on the box the wrapper's status was not the job's; the comparator used a weaker provenance
rule than the launcher and accepted evals with no model hashes; the SFT seed was a structural
null and max_len was compared nowhere; a refused rerun could sit beside an old comparison.
Fixed by run identity (fresh directory, token, lock, waitable supervisor, status manifests),
one validator for both entry points, and an SFT recipe read from the checkpoint's own args.
414 → 443 passing, failure set still byte-identical to `59cafcd`.

**Done 2026-11-06, round one.** The first audit's five findings, each reproduced before being read:

- **A cached result could be certified under invented provenance.** The launcher wrote the cell
  manifest *before* testing whether to reuse an existing evaluation, and `provenance.check`
  treated absence as agreement, so a planted stale eval with no checkpoint produced a
  `comparison.json`. Now: two schemas with required fields; the evaluator records the pool,
  split, checkpoint and adapter digests it actually resolved and read; manifests are DERIVED
  from that record after the evaluation succeeds; reuse requires the checkpoint, an unaltered
  manifest, and every recorded artifact still hashing the same; a manifest is never overwritten
  on a reused output; and two arms that differ in neither checkpoint nor adapter are refused as
  one model compared with itself. The comparator applies the same rule itself.
- **Lineage is read from artifacts, not declared.** Both manifests said
  `objective=legacy_row_share_cpt`. The base arm has no CPT at all, and arm D's own
  `trainer_state.json` records batch 16 × accumulation 16 with no token budget, i.e.
  **`fixed-batch-mean/v0`** — not row share, which is the v2 token-budget default. CPT, SFT and
  eval lineage are now separate fields.
- **A non-finite score became a perfect rank.** `1 + (sc[1:] >= sc[0]).sum()` is 1 for an
  all-NaN query, and 1 is a legal rank, so no downstream check could see it. `rank_of` refuses
  and names the rank it would have produced; score spread and tie counts are recorded beside
  every condition; the probe can now load the actual checkpoint, and an incomplete or
  wrong-task checkpoint refuses instead of evaluating partly fresh weights.
- **The ceiling binds during a phase.** `check_ceiling` ran between steps and `run_gpu` then
  blocked in `wait`, so a phase starting at hour 7:59 could run for days. The watchdog's publish
  command was unbounded immediately before the shutdown, so a wedged upload defeated the cost
  ceiling outright. Both are bounded now, and the watchdog is armed and verified by the launch
  command rather than by remembering a line of prose.
- **The preflight proves usability.** Imports, CUDA, the pool, every split *including train and
  val*, the adapter's actual contents and a checkpoint that must load and carry trainable
  weights — all on the CPU, before anything is bought. A missing train split used to surface
  after the base evaluation was paid for. The evaluator no longer falls back to the CPU silently.

93 new tests, no regressions: 321 → 414 passing, with the failure set byte-identical to the
baseline at `59cafcd` (33 failures + 4 collection errors, every one a dependency missing on this
laptop: peft, esm, neuronx_distributed, openpyxl).

**Open:** cells 2-4 still have no trained checkpoints of their own, so V04's comparison would read
the legacy arm-D adapter · `STEPS` is recorded in manifests and enforced nowhere, and the trainer
takes `--resume_from`, which no launcher passes · ρ is assumed 0.8 against a measured breakeven
**0.765**, and only a real comparison can replace that · every readability number is power under
the **uniform** draw until the gate recomputes it · **the 1.31 % is unapportioned**: training
masking uses the global and DataLoader-worker torch RNG rather than evaluation's dedicated
Generator, so that stream is seeded but not dedicated and resume does not restore it · arm A has
never been judge-scored · §9.6's length finding still needs restating per peptide in the findings
text · `run_v04.sh`'s base cell still carries the CPT-objective mislabel, which cannot be fixed in
isolation (see the note at its manifest write) and goes away when its cells move to the `sft_cell`
shape.

`--gradient_checkpointing` is **inert**: the trainer accepts it and never reads it
(`cpt_training_lib.py:1494`). A preflight of the real command is still needed for memory and
throughput; the flag is not the reason.
