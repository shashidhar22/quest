# NEXT — quest (updated 2026-11-04) · history: `docs/LOG.md` · stages: `plans/README.md` · knowledge: `~/Documents/ACS/quest-cpt/`

**DO NOT LAUNCH V04 YET.** An external Stage-4 audit on 2026-11-04 found the launcher broken in a
way that would have destroyed the visit, and the repair is only partly done. Verdicts and evidence:
`~/Documents/ACS/quest-cpt/audits/2026-11-04/triage.md`. Remaining work and its order:
`plans/v04-audit-repair.md`.

Why the previous instruction here was dangerous, kept because it is the lesson: it said to launch
`run_v04.sh` and arm `autostop_when_done.sh` on the driver PID. Trainers were started through
command substitution, so every `wait` failed with "not a child of this shell", the driver printed
`V04 complete` within seconds, and the watchdog would have stopped the instance on top of six live
trainings. A dry run with an instant trainer stub could not see it.

**No box is running.** `i-0030c1a9c845b3714` is stopped and sized **g6.2xlarge**; V04 needs
**g6.12xlarge** on the same volume `vol-08f5626ccf18fdd4c`. Decision gates and all Stage 4.2/4.3
detail: `quest-cpt/box-jobs/README.md` §3.2–3.4.

1. **`git push`** the local commits. Claude's sandbox refuses push as a public-surface action, so
   Yuexin runs it.
2. **Decide the loss-normalization contract** — input-length approximation as shipped, or an exact
   global masked-target mean. This gates the GPU spend rather than following it: V04's six
   re-baselines *are* the new baselines, so buying them under an objective about to change wastes
   the visit. `step_token_plan()` sums input lengths while `raw_loss` is a mean over realized
   masked targets, and the normalization is per-rank with no all-reduce. The resume guard does not
   wait on this: refusing an A-to-B change is right either way and is already in. What waits is
   implementing the chosen contract and its gradient-invariance test.
3. **Finish the repair, off-box, in this order:** gate fail-closed, with distinct outcomes for
   proceed, scientific refusal and technical error · one comparator for the locked endpoint, then
   actually invoke it · provenance over real artifacts · wire train-then-evaluate for the missing
   SFT cells. The paired-comparison statistics need NumPy, not torch: keep them in a light module
   beside `readability.py` so the contract is testable off-box, with the judge CLI as a thin
   consumer. Only model loading, trainer integration and DDP numerics need the ML environment.
4. **Then, before any long visit:** `aws sso login --profile DataUserAccess-953915750371`, a
   read-only hashed copy of `/home/ubuntu/v2_full.sh` and any surviving uncommitted `track_a/` off
   `vol-08f5626ccf18fdd4c` (the only record of what computed the two failed V03 verdicts), resize,
   then a bounded preflight of the exact V04 command recording backend version, input and mask
   hashes, memory and throughput. Launch only after that reads clean.

**Done 2026-11-04 (33 tests: `tests/test_v04_launcher.py` 20, `tests/test_resume_objective_guard.py`
13):** process ownership and failure propagation · **signal cleanup**, so TERM to the driver takes
the training with it instead of leaving three trainers on the GPUs — jobs lead their own process
group under `set -m`, are `exec`ed so the tracked PID is the real process, evaluations are tracked
rather than run in the foreground where a trap waits behind them, and cleanup is bounded TERM →
wait → KILL → reap · orphan-free aborts · **P0 preflight**, so an absent path or checkpoint costs
nothing instead of being found after three GPU jobs are running · dry-run isolation, so it can no
longer delete `checkpoints/v04_arm*_seed*`, and its own check compares a before/after inventory
instead of reading pre-existing checkpoints as proof it wrote them · four-GPU requirement, removing
the GPU-0 port collision · per-cell task selection, so both `*_trb` cells stop running
`as_paired_i` · a missing candidate pool aborts instead of becoming a sentinel that matches itself ·
provenance waivers are printed, never silent · a comparison cell with no trained checkpoint refuses
instead of scoring an untrained head · **resume refuses an objective change**, versioned so
redefining `token` cannot pass under the same word, failing closed on a checkpoint that records no
semantics with one verified override (`--resume_assume_objective`). The rule lives in
`quest/training/resume_compat.py` and is importable with no ML stack present — which took a second
fix: the module was always stdlib-only, but `quest/training/__init__.py` eagerly re-exported
collators, so importing the light module executed the package and pulled in torch anyway. Those
re-exports are now lazy (PEP 562). **A light module under a heavy package is still heavy**, and the
proof is a subprocess test with torch blocked rather than a passing run on an interpreter that
happens to have it.

Group cleanup of a real torchrun's worker processes is **not** proven: the tests use stubs. Check it
on Linux during the preflight.

**Open:** the launcher still invokes **no comparison** — `judge compare` is called nowhere, and its
estimator is query-level Recall@10 with a row bootstrap, not the locked peptide-clustered
normalized-rank read the gate enforces · cells 2-4 have no trained checkpoints and training them is
GPU time the ≈10 h / ≈USD 46 estimate does not include · the gate still returns 0 when a requested
readability read has no ranks · `STEPS` is recorded in manifests and enforced nowhere, and the
trainer takes `--resume_from`, which the launcher never passes · ρ is assumed 0.8 against a
measured breakeven **0.765** · every readability number is power under the **uniform** draw until
G1 recomputes it · **the 1.31 % is unapportioned**: training masking uses the global and
DataLoader-worker torch RNG rather than evaluation's dedicated Generator, so that stream is seeded
but not dedicated and resume does not restore it, and fresh-run mask reproducibility has not been
measured here; the flash-attn attribution was elimination from a premise that was wrong, and
neither candidate has been measured · arm A has never been judge-scored · §9.6's length finding
still needs restating per peptide in the findings text itself.

`--gradient_checkpointing` is **inert** and deliberately absent from the V04 command: the trainer
accepts the flag and never reads it (`cpt_training_lib.py:1494`). A preflight of the real command
is still needed for memory and throughput; the flag is not the reason.
