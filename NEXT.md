# NEXT — quest

Updated 2026-10-04 00:30 UTC. First place to look when resuming.

## First move next time

**Nothing is running. Every box is stopped; KS RNA-Seq is decommissioned.** Stage 4.1 is
complete and pulled local: `docs/cpt_stage4.1/README.md` §10 (results) and §11 (the 14.5 h
idle incident). Start Stage 4.2 (`plans/stage4-2.md`) and 4.3 in parallel.

**Before launching ANY long job:** arm `scripts/training/autostop_when_done.sh <driver_pid>
'<publish cmd>'` with `MAX_HOURS` set to ~1.5× the expected run, in its own tmux session.
The old name-pattern watcher cost ~USD 100 on 2026-10-03.

**Committed but NOT pushed (no backup yet).** Branch `yxu2/tcrbench2-continue` is 7 commits
ahead of anything on `origin` (`shashidhar22/quest`, a collaborator's repo — push needs
Yuexin's go-ahead; the branch has never been pushed). Until it is, the recovered Stage 0–4
scripts, `track_a/`, Stage 4.1 results and the 4.2–4.7 plans exist only on this laptop. The
repo also lacks the backup standard (`scripts/claude-memory.sh`, `MIGRATE.md`).
`track_a/monitor/` (5,196 polling snapshots) is excluded locally via `.git/info/exclude`.

## Status at wrap-up

| Job | State |
|---|---|
| Stage 4.1 training | arm **2 of 3** (`pointD_run05`), step 2500, 0 errors, auto-stop armed |
| KS workspace → S3 | **828.6 GB of 1,446.8 GB (57%)**, 8,133 objects, 16 workers, **0 failures**, auto-stop armed |

**Checked 2026-10-03 02:45 UTC:**
- **KS sync DONE and VERIFIED** — `_sync_verification.log` at the destination says `VERIFY PASS`:
  40,933 objects / 1,446,777,687,859 bytes, exact, `rc=0`, 0 per-file failures. The helper stopped
  itself at 02:43 UTC. Deletion-checklist step 1 is satisfied; steps 2–5 remain, and the 3 TB temp
  volume `vol-00591ae4fb420af9a` is still billing. Lesson: the `xargs -P 16` pass died at entry
  24,532 of 40,478 on `Karthik's IM9 CRA/…` (unmatched single quote — plain `xargs` parses quotes;
  use `-0` or `-d '\n'`); the wrapper's fallback `aws s3 sync` pass uploaded the remaining ~16k
  files and is what made the total exact.
- **Update 04:37 UTC — arm 2 PASS** (5.98 h, step 7800, no early stop): `pocket` **1.862**,
  `overall` 1.726, `cdr3` 4.024, `framework` 1.275 vs Point D 2.001 / 1.746 / 4.002 / 1.283 —
  reproduces, pocket 7 % better. **Arm 3** (`pointF_run11`) started 03:26 UTC, at 19 % of its epoch,
  pocket 3.796 → 3.518 → 3.003 over its first 3 evals; ≈ 4 h 50 min left → summary + self-stop
  ≈ 09:30–10:00 UTC.
- **Stage 4.1 arm 2** (earlier check) at step ~7000 of ~7800, 5 h 14 min in. Evals every 500 steps: `pocket_ppl`
  3.516 → 1.993 (step 4000) → 2.070 → 1.945 → 1.900 → 1.888 → **1.867 (step 6500)**; at step 6500
  `overall` 1.735 / `cdr3` 4.034 / `framework` 1.283 vs Stage 4 Point D's final 1.746 / 4.002 /
  1.283. Three regions reproduce within 1 %; pocket is already 7 % *better* than the 2.001 target
  with ~1,300 steps to go — plausibly pocket noise (fewest tokens), but Stage 4 ran one seed so it
  cannot be checked. Arm 2 should finish ≈ 03:30 UTC; arm 3 then needs ≈ 6–7 h, so expect the
  summary + S3 sync + self-stop around 10:30 UTC on 2026-10-03.

### Arm 1 result (the first controlled number this programme has produced)

`pointA_run01`, cdr3 × pocket, 1.58 hr, global_step 7800:

| | Stage 4 Point A | Stage 4.1 Arm 1 |
|---|---:|---:|
| `max_length` | 64 (truncating; data max 162) | **1024** (no truncation) |
| `pocket_ppl` | 4.732 | **3.368** |
| `cdr3_ppl` | 3.968 | 3.849 |
| `overall_ppl` | 4.968 | 4.405 |
| steps | 5200 | 7800 |

Removing truncation improved `pocket_ppl` 29%. **Caveat: the step budget also grew 5200 → 7800**,
so part of that is extra training, not the window. Use the `save_steps 2000` checkpoints for a
matched-budget comparison *before* quoting 29% as a pure window effect.

Arm 2 (`pointD_run05`) is the **reproducibility control** — it should land near Stage 4 Point D's
`pocket_ppl` 2.001. **If it does not, treat the whole re-run as suspect and chase that first.**

## Automation left running (nothing needs a human to collect results)

- **GPU box `i-0030c1a9c845b3714`**, tmux `autostop` → waits for driver exit, runs
  `build_stage41_summary.py`, syncs all three arms + every log to
  `quest/tcrbench2/checkpoints/`, then `shutdown -h`. Verified
  `instanceInitiatedShutdownBehavior=stop`, so the volume survives.
- ~~Helper `i-0ab0e9f59429fc8eb`~~ — did its job (VERIFY PASS 02:41 UTC), **terminated
  05:19 UTC** with the temp volume and the KS box. Nothing KS-related is running or billing
  except snapshot storage.

## KS RNA-Seq decommission — checklist, in order

Yuexin approved deleting the instance and keeping the snapshot. Do NOT skip step 1.

1. Read `s3://fh-pi-warren-h-eco/archive/ks_manuscript_workspace/_sync_verification.log` — must say **PASS**.
2. ~~sync the 2.15 GB not covered~~ **DONE 2026-10-03 05:14 UTC** → `archive/ks_rnaseq_home/`
   (`data_catalog/` 50 files / 2,290,335,985 B, `kics/` 21 files / 12,481,507 B, both VERIFY PASS,
   log at that prefix). Gotcha hit on the way: the helper would not boot with the KS root clone
   attached (same AMI → same root label → booted the clone); detach before start, attach after.
3. ~~Delete temp volume~~ **DONE 2026-10-03 05:19 UTC** (`teardown.sh --execute`).
4. ~~Terminate KS RNA-Seq~~ **DONE 05:19 UTC** — helper terminated too. Both boxes gone.
5. **KEPT** `snap-05e5a7a6778f46f49`, still unlocked — now the only copy of the box's
   environment (data is in S3 twice). Lock command in `teardown.sh`'s header if wanted.

Rebuild caveat: the original used key pair `test`, whose private half we do not hold.

## GPU efficiency — measured 2026-10-03, corrected (do NOT change Stage 4.1 mid-run)

Read first-hand on the box: ESM-C already runs **varlen flash attention** (`_use_flash_attn=True`,
flash_attn 2.8.3, the forward unpads every batch), `--gradient_checkpointing` is a **no-op** in
`cpt_training_lib.py` (a config field with no consumer), and `--attn_implementation` is ignored by
ESM-C. So the earlier list here (packing / varlen / drop checkpointing) was wrong on three of four
items — packing even needs the non-flash SDPA path to be safe.

What actually costs time in arms 2–3: ≈ 270 real tokens per forward × 16 micro-batches per step
with a DDP all-reduce after each (no `no_sync`), plus `region_restricted_eval` every 500 steps at
≈ 6.5 min on rank 0 (re-tokenizes + re-annotates 19,708 rows per call, Python per-region
accumulation) while the other three GPUs spin — ≈ 30 % of wall. Fix order: cached + vectorized
eval → token-budget micro-batches with row-based accumulation and `no_sync` → throughput log →
mid-arm resume (unlocks spot). Full write-up, costs and the revised step order:
`AWSconnection/QUEST_CHECKPOINTS_AND_DATA.md` §9.1–9.2.

## Next CPT stages (planned 2026-10-03)

One-page plans in `plans/stage4-2.md` … `stage4-7.md`: 4.2 SFT judge → 4.3 trainer efficiency →
4.4 balanced sampling → 4.5 pocket masking → 4.6 allele diagnostic → 4.7 scaling. 4.2 and 4.3 can
run in parallel; nothing from 4.4 on trains until the 4.2 judge exists.

## Open threads

- **Nothing is committed.** 20 changed/untracked paths; the recovered scripts exist nowhere else.
  No migration standard here either (`scripts/claude-memory.sh`, `MIGRATE.md`) — worth installing
  via the `repo-backup` skill for exactly that reason.
- **The boot-landmine warning may name the wrong box.** `recreate-scratch-raid.service` is nowhere
  on KS RNA-Seq's root volume, yet it destroyed `data loader`. Real carrier unidentified, possibly
  still running. See the `warren-aws-gpu` memory.
- Only **8 of 24** Track A ESM SFT cells ever ran, and **Point D has never been SFT-tested** —
  still the real go/no-go for the CPT programme.
- Stage 4.1 keeps early stopping, so budgets may diverge across all three arms, not just arm 1.

## Uncommitted work (nothing else holds these)

Recovered from the `neurips_two` snapshot, previously recorded as lost:
`scripts/training/{build_stage4_summary,run_stage4_pipeline,stage4_pointE_reeval,_write_stage2_summary}.py`,
`scripts/training/{run_overnight_stage4,stage4_preflight}.sh`,
`scripts/data_processing/{build_run10,recover_cluster_files}.py`, `track_a/`, `docs/cpt_recovered/`.

New: `scripts/training/{run_stage41_pipeline,build_stage41_summary,measure_maxlen}.py`,
`scripts/data_processing/build_run11.py`, `docs/cpt_stage4.1/`, `plans/stage4-1.md`, this file.

Modified: `cpt_training_lib.py` (eval-time `mhc_class` repair),
`cpt_dataset_{verifier,builder}.py` (classifier fix), `docs/cpt_stage4/README.md`.
