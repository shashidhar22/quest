# Track A Status

**Last update**: 2026-05-04 ~07:25 UTC (T-71h to deadline)

## Phases

| Phase | State | Notes |
|---|---|---|
| 0 — Class II investigation | ✓ done | eval-only by design |
| 1a — `track_a/shared/` utilities | ✓ done | smoke-tested |
| 1b — Verification docs (5 baselines) | ✓ done | drafts written for batched review |
| 1c — Implementation + smoke tests | ✓ **6/6 baselines passing smoke** | |
| 2 — Full-scale launch | **IN PROGRESS** since 06:02 UTC | 4 GPUs + 2 CPU sessions |
| 3 — TRA appendix | pending | only if time remains |

## Phase 2 — running on 6 sessions

| Session | Cell | Task | Status | ETA |
|---|---|---|---|---|
| `g6_gpu0` | esm2_8m-base | as_trb_i | scoring test_novel_tcr 400/2000 | TRB-I cell ~3.5h total; queue ESM2-8M cpt × 4 + base × 3 more tasks after |
| `g6_gpu1` | esm2_35m-base | as_trb_i | scoring test_novel_tcr 550/2000 | similar; queue 35m cpt + 4 more tasks |
| `g6_gpu2` | lstm | as_paired_i | epoch 3/20, val_auc=0.75 (rising) | each Paired-I epoch is ~880s (large train); queue lstm × 4 → mlp × 4 |
| `g6_gpu3` | tcrbert | as_trb_i | scoring test_novel_tcr 1250/2000 | tcrbert × 4 → esmc_300m × 8 |
| `cpu_tulip` | tulip-original | as_trb_i | scoring test_iid 20/200 | TRB-I + Paired-I; class II auto-skipped |
| `cpu_deeptcr` | deeptcr | as_trb_i | training (60,282 rows, 818 classes) | × 4 main tasks |

## Sessions tmux state

```
cpu_deeptcr   running queue_cpu_deeptcr.sh
cpu_tulip     running queue_cpu_tulip.sh
g6_gpu0       running queue_gpu0.sh
g6_gpu1       running queue_gpu1.sh
g6_gpu2       running queue_gpu2.sh
g6_gpu3       running queue_gpu3.sh
neurips       USER's session — do not touch
```

## Results so far

| Baseline | Task | Status | Notes |
|---|---|---|---|
| lstm | as_trb_i | ✓ written | recall@10: iid 0.02 / novel_tcr 0.54 / novel_pep 0 / novel_allele 0 / level4 0 |

(More cells will land as queues progress.)

## Decisions taken (auto mode)

- Verification-doc batch review pushed to async (auto mode).
- TCR-BERT TRB-only uses empty-string TRA shim (ANOMALIES.md).
- DeepTCR uses class-prob lookup over training peptides → score over candidate pool. Peptides not in train get score 0.
- TULIP-original needs `track_a_tulip` env (transformers 4.30) — installed. Class II auto-skipped (TULIP-original wasn't pretrained on class II).
- Eval row cap = 2000 across all baselines.

## TULIP / DeepTCR setup notes

- `track_a_tulip` env: Python 3.10, transformers 4.30, tokenizers 0.13, torch 2.1.
- `track_a_deeptcr` env: Python 3.10, TF 2.12 CPU, DeepTCR (pip), pyarrow, numpy<2.
- Both run on CPU; doesn't compete with GPU training.

## Known gaps to revisit

1. `wandb_run_id` is null in results JSON because eval.py runs in a separate process from train.py. Could pipe via a sidecar file; deferred.
2. ESM-C 300M cells haven't started yet — queued behind TCR-BERT on GPU 3.
3. After ESM2-{8M,35M} cells finish, GPU 0/1 will idle. Plan: queue MORE ESM-C 300M cells across GPU 0/1 (faster than 1 GPU sequentially) OR queue TRA appendix.

## Next monitoring touchpoints

- **+30 min from now**: check first ESM2-8M and TCR-BERT cells complete; first DeepTCR result land.
- **+1.5h**: check Paired-I LSTM val_auc; consider early-stop if stalling.
- **+3h**: ESM2-8M TRB-I cell should be done; TCR-BERT TRB-I done; queue next.

## Files of record

- `PHASE0_CLASSII_INVESTIGATION.md`, `PRE_LAUNCH_VERIFICATION.md`, `ANOMALIES.md`
- `verification/{tulip,deeptcr,lstm_mlp,tcrbert,esm_sft}_verification.md`
- `shared/{__init__,load_splits,negative_sampling,allele_utils,retrieval_metrics,results_schema,task_aliases}.py`
- `lstm_mlp/{model,dataset,convert_data,train,evaluate}.py` (smoke ✓)
- `tcrbert/{model,dataset,convert_data,train,evaluate}.py` (smoke ✓)
- `esm_sft/{encoders,model,dataset,convert_data,train,evaluate}.py` (smoke ✓ for esm2_8m base+cpt and esmc_300m base+cpt)
- `tulip/score.py` + `tulip/repo/` (smoke ✓ in track_a_tulip env)
- `deeptcr/run.py` (smoke ✓ in track_a_deeptcr env)
- `launch/{run_baseline,queue_gpu0..3,queue_cpu_tulip,queue_cpu_deeptcr}.sh`
