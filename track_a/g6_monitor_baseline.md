# g6 monitor — baseline state at deployment

**Captured at**: 2026-05-04 15:27 UTC

## Hardware

`nvidia-smi`: 4× L4, driver 580.126.16, CUDA 13.0. All four GPUs hot:

| GPU | Util | Mem | Notes |
|---|---|---|---|
| 0 | 0% | 1258/23034 MiB | esm2_8m-base eval (between batches) |
| 1 | 60% | 616/23034 MiB | esm2_35m-base train |
| 2 | 28% | 2966/23034 MiB | esmc_300m-base train (just relaunched 15:15) |
| 3 | 95% | 540/23034 MiB | tcrbert eval |

## Detached resume processes (parents owned by `init` PID 1, surviving disconnect)

| PID | Cmd | Started | Notes |
|---|---|---|---|
| 74410 | `bash resume_deeptcr.sh` | 14:48 | spawns `deeptcr.run --task as_paired_ii` (PID 79183) |
| 74411 | `bash resume_gpu3.sh` | 14:48 | spawns tcrbert eval (PID 74456) |
| 74412 | `bash resume_gpu1.sh` | 14:48 | spawns esm2_35m-base train (PID 74460) |
| 74413 | `bash resume_gpu0.sh` | 14:48 | spawns esm2_8m-base eval (PID 74461) |
| 80470 | `bash resume_gpu2.sh` | 15:15 | spawns esmc_300m-base train (PID 80478) |

PPID for all `resume_*` is 1 (systemd init), confirming detachment.

## Results subdirectories

`/home/ubuntu/quest/results/` contains: `deeptcr`, `esm2_35m-base`, `esm2_8m-base`, `lstm`, `mlp`, `tcrbert`, `tulip-original`.

Expected dirs not yet present (queues still in flight): `esm2_8m-cpt`, `esm2_35m-cpt`, `esmc_300m-base`, `esmc_300m-cpt`.

## System

- RAM: 28 Gi used / 181 Gi total / 153 Gi available, 0 B swap
- Disk `/`: 206 G used / 968 G / 22% (763 G free)
- Linger: `Linger=yes` for ubuntu (verified)
- `user@1000.service`: active (running) since 14:30 UTC, peak memory 24.1 G
- `cron.service`: active (running) since Sun 2026-05-03 22:56 UTC — usable
- dmesg: no `oom_kill` / `Killed process` entries (clean baseline)

## Per-tool task matrix (from the original prompt)

| Tool | TRB-I | Paired-I | TRB-II | Paired-II | TRA-I | TRA-II |
|---|---|---|---|---|---|---|
| LSTM | yes | yes | yes | yes | if time | if time |
| MLP | yes | yes | yes | yes | if time | if time |
| DeepTCR | yes | yes | yes | yes | no | no |
| TCR-BERT (with peptide) | yes | yes | yes | yes | if time | if time |
| TULIP-original | yes | yes | (auto-skip class II) | (auto-skip class II) | no | no |

Plus the ESM 2×3 matrix (encoder ∈ {ESM2-8M, ESM2-35M, ESM-C 300M} × variant ∈ {base, CPT}) on all 4 main tasks.

**TRA tasks deferred** (appendix-only — only if all 4 main tasks complete with time to spare).
**TULIP-retrained deferred** (transformers-API incompatibilities; see `ANOMALIES.md`).

## Conditions per task

| Task family | Conditions per task |
|---|---|
| Class I (`as_trb_i`, `as_paired_i`, `as_tra_i`) | 5: `test_iid`, `test_novel_tcr`, `test_novel_pep`, `test_novel_allele`, `test_level4` |
| Class II (`as_trb_ii`, `as_paired_ii`, `as_tra_ii`) | 1: `eval` |

Verified empirically by reading `conditions` keys in completed `_summary.json` files. Note: user's example template had `"as_trb_ii": 4` which doesn't match the actual data — using the verified counts.

## Expected total (main tasks only — TRA deferred)

11 baselines × (5+5+1+1) cells = **132 cells** if every cell completes.

For TULIP-original specifically, class II tasks auto-emit a "skipped" result (1 condition with `skipped: true`) — these still satisfy the "summary JSON exists" check, so they count.
