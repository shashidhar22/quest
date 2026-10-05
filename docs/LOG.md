# LOG — quest (dated, append-only; newest last)

One entry per session: what changed, the numbers, incidents and their lessons. NEXT.md stays
short because history lives here. Facts that are still true belong in `~/Documents/ACS/quest-cpt/`
or the project memory, not in this log.

## 2026-10-02

- Stage 4.1 planned and launched on `i-0030c1a9c845b3714` (g6.24xlarge, only type with capacity):
  three arms at one fixed `max_length=1024` (measured global max 608) and effective batch 1024.
  `run11_cdr123_fullmhc_proportional_10M` built (9,998,544 rows, 56 min).
- `mhc_class=none` pocket tokens diagnosed: 157 class-II alpha chains in `mhc_one` mislabelled by
  `derive_mhc_class`; labels were always right, only the stratum; fixed; Point D re-score
  reproduced 2.001 exactly, proving the fix aggregate-neutral.
- KS RNA-Seq workspace sync to `archive/ks_manuscript_workspace/` launched from a helper
  (`xargs -P 16` per-file after single `aws s3 sync` managed 16–18 MB/s on the lazy-loading
  restored volume).

## 2026-10-03

- KS sync VERIFY PASS: 40,933 objects / 1,446,777,687,859 B exact. The `xargs` pass had died at
  entry 24,532 on an apostrophe (`unmatched single quote`); the fallback `aws s3 sync` pass
  finished the rest. Extras (`data_catalog/` 50 files, `kics/` 21 files) synced to
  `archive/ks_rnaseq_home/`, VERIFY PASS; the helper would not boot with the KS root clone
  attached (same AMI → same root label → booted the clone) — detach before start, attach after.
- `teardown.sh --execute`: helper + 3 TB temp volume + KS instance gone; snapshot kept.
- Stage 4.1 arms 2 and 3 PASS (5.98 h, 6.05 h). Results: `docs/cpt_stage4.1/README.md` §10.
  D reproduces Stage 4 Point D within 1 % on three regions; full MHC still −44.7 % vs pocket at
  a matched window; CDR1/2 hurts; class I still 7.6–17.7.
- **Incident:** the auto-stop watcher never fired (`pgrep -f` matched the tmux server's argv);
  the box idled 14.5 h, ≈ USD 100. Second bug: `build_stage41_summary.py` KeyError 'F' (Stage 4
  letter lookups). Both fixed; `scripts/training/autostop_when_done.sh` (PID + GPU-idle +
  MAX_HOURS guards) with `tests/test_autostop_when_done.sh` 8/8. Write-up §11.
- GPU-efficiency facts verified on the box: ESM-C already runs varlen flash attention;
  `--gradient_checkpointing` is a no-op; eval is ≈ 30 % of wall on rank 0. Fix order in
  `plans/stage4-3.md`.
- Stages 4.2–4.7 planned (`plans/`).

## 2026-10-04

- Seven commits on `yxu2/tcrbench2-continue` (recovered Stage 0–4 scripts, `track_a/`, the
  mhc_class fix, Stage 4.1, the watchdog, plans). Not pushed — origin is the collaborator's repo.
- Project memory consolidated to current state.

## 2026-10-05

- Cost review via Cost Explorer (permitted): idle run-rate ≈ USD 800/mo, 3/4 snapshots. On
  Yuexin's call: `neurips_two` terminated (root already in `snap-0aa5bccb08a3012e3` / AMI);
  data-loader (8.3 TB billed) and KS (3 TB) snapshots moved to the archive tier (90-day
  minimum, 24–72 h restore). Expected idle bill ≈ USD 300/mo.
- run11 backed up to S3 (35 files / 3,185,473,264 B, VERIFY PASS) by switching the box's
  instance type to m5.xlarge for 6 min — the g6.24xlarge start had failed on capacity.
  Practice: size the box to the job by switching the type of the same volume.
- System reorg: `~/Documents/ACS` became the private repo `acs-notes` (AWS runbooks, box
  records, `quest-cpt/` knowledge split from the 767-line checkpoints doc); this LOG.md
  created; NEXT.md cut to its contract; memory 10 → 4 files; `warren-gpu.sh` gained
  `inventory`, `cost`, `type`.
