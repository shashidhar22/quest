# NEXT — quest (updated 2026-11-04) · history: `docs/LOG.md` · stages: `plans/README.md` · knowledge: `~/Documents/ACS/quest-cpt/`

**V04 as specified is cancelled; the next run is JUDGE-FIRST and awaits approval to spend.**
Decisions 1-3 were approved 2026-11-04. Audit verdicts and evidence:
`~/Documents/ACS/quest-cpt/audits/2026-11-04/` (`stage4-audit.md`, `triage.md`, `claude-handoff.md`,
`decision-recommendations.md`). Remaining work in order: `plans/v04-audit-repair.md`.

**No box is running.** `i-0030c1a9c845b3714` is stopped and sized **g6.2xlarge · 1× L4**. The
judge-first run needs **one GPU, so no resize**. Volume `vol-08f5626ccf18fdd4c`.

1. **`git push`** the local commits. Claude's sandbox refuses push as a public-surface action, so
   Yuexin runs it.
2. **Approve or amend the judge-first spend**, then launch it:
   `~/Documents/ACS/quest-cpt/box-jobs/visits/v04a-judge-first.md` — ≈4.9 GPU-hours, ≈1.3 if the
   gate refuses, ≈USD 4-6 at the repo's recorded 0.80-1.15 per GPU-hour band. **Confirm the actual
   g6.2xlarge rate in Cost Explorer first: it is recorded nowhere in this repo.** Pre-launch, in
   order: `aws sso login --profile DataUserAccess-953915750371`; a read-only hashed copy of
   `/home/ubuntu/v2_full.sh` and any surviving uncommitted `track_a/` off the volume, the only
   record of what computed the two failed V03 verdicts; then
   `scripts/training/v04_box/judge_first.sh` with `autostop_when_done.sh` armed on the pid in
   `$O/judge_first.pid` and `MAX_HOURS` 8.
3. **Verify on the box what the laptop cannot:** that terminating the launcher kills a *real*
   torchrun's descendant workers, not only a stub. Process groups under `set -m` are right on
   Linux by construction, but only stubs have exercised it.
4. **Then, before any CPT baseline:** land the exact-normalization loop change (step 5 of the
   repair plan). The objective is decided and its contract is gradient-tested, but the training
   loop still cannot supply the step's global target count before its first backward, so
   `token_exact` raises on purpose. Buying baselines under `token` would buy the old objective.

**Done 2026-11-04.** Five commits, 125 new tests, no regressions: the collectible suite went
211 → 321 passing and its failure set from 34 to 33, all remaining failures being missing
dependencies on this laptop.

- **Lifecycle.** The driver could not `wait` on its own trainers (command substitution), and
  signalling it left them running (the tracked PID was a function shell). Either way the PID
  watchdog would have stopped the box on live training. Now: jobs lead their own process groups
  under `set -m`, are `exec`ed, evaluations are tracked rather than foreground, and cleanup is
  bounded TERM → wait → KILL → reap with the driver alive throughout.
- **Dry run.** No longer deletes `checkpoints/v04_arm*_seed*` or writes into the real tree, and its
  own check compares a before/after inventory instead of reading pre-existing checkpoints as proof
  it wrote them.
- **Preflight.** Every path, checkpoint and config is checked before a GPU job starts.
- **Gate.** Three outcomes (`s42_check.EXIT_OK/REFUSED/ERROR`): a requested readability read with
  no ranks is now a technical error rather than exit 0, unit-map coverage and rank validity are
  required, the eval's own pool size is authoritative, and a CI at or below chance is reported as
  science (`BELOW-CHANCE`) rather than broken software. Only MRR exactly 1/pool is called a fault.
- **The comparison exists.** `paired_compare.py` measures the locked endpoint, resamples peptides,
  keeps the molecule-weighted estimator, measures rho instead of assuming it, reports cluster
  sensitivity and secondary metrics, and states that a normalized-rank effect does not establish
  the +0.05 Recall@10 target. `judge compare` is a thin consumer; the launcher requires the result.
- **Resume** refuses an objective change, versioned, failing closed on a checkpoint with no
  recorded semantics (one verified override, `--resume_assume_objective`).
- **Off-box by construction.** `resume_compat.py`, `loss_normalization.py` and `paired_compare.py`
  hold the contracts; a subprocess test with torch blocked proves the import boundary. The lesson
  that cost a round: **a light module under a heavy package is still heavy** — `quest.training`
  re-exports are now lazy.

**Open:** cells 2-4 still have no trained checkpoints of their own, so V04's comparison would read
the legacy arm-D adapter · `STEPS` is recorded in manifests and enforced nowhere, and the trainer
takes `--resume_from`, which no launcher passes · ρ is assumed 0.8 against a measured breakeven
**0.765**, and only a real comparison can replace that · every readability number is power under
the **uniform** draw until the gate recomputes it · **the 1.31 % is unapportioned**: training
masking uses the global and DataLoader-worker torch RNG rather than evaluation's dedicated
Generator, so that stream is seeded but not dedicated and resume does not restore it; hash the
first batches, labels and masks at one seed before attributing it to anything · arm A has never
been judge-scored · §9.6's length finding still needs restating per peptide in the findings text.

`--gradient_checkpointing` is **inert**: the trainer accepts it and never reads it
(`cpt_training_lib.py:1494`). A preflight of the real command is still needed for memory and
throughput; the flag is not the reason.
