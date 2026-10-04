# Track A Verification: DeepTCR

**Date**: 2026-05-04
**Authors**: Claude (Track A)
**Status**: Ready for review before Phase 1c implementation

## Scope

DeepTCR on AS-TRB-I, AS-Paired-I, AS-TRB-II, AS-Paired-II. **Per the user's per-tool matrix, DeepTCR is NOT run on TRA-only tasks.** Class II is eval-only.

## Upstream

- Repo: https://github.com/sidhomj/DeepTCR
- pip-installable: `pip install DeepTCR` (depends on TensorFlow 2.x)
- Reference: Sidhom et al., 2021. *DeepTCR is a deep learning framework for revealing sequence concepts within T-cell repertoires*.

The legacy script (`benchmark_deeptcr.py`) wraps `DeepTCR.DeepTCR_SS` (the supervised-sequence model) for binary binding classification.

## Architecture

DeepTCR_SS uses:
- Convolutional encoders over CDR3 amino-acid sequences
- Optional V/D/J gene one-hot embeddings (DeepTCR feeds them as auxiliary inputs)
- Optional MHC one-hot (HLA name vocabulary)
- 2-3 layer dense head → binary classifier

Hyperparameters: defaults from the upstream (`epochs_min=10`, `stop_criterion=0.001`, `hinge_loss_t=0.0`). The legacy script doesn't override these.

## Inputs

- TRB-only: `trb_cdr3` + V/D/J + `mhc_one_allele`
- Paired: `tra_cdr3, trb_cdr3` + V/D/J + `mhc_one_allele`
- For class II tasks: train on matching class-I `_train`; eval on `_eval`

**V/D/J gene columns are NOT in the AS-* parquet schema.** The 25-column TARGET_COLUMNS schema (per `quest/data/standardization.py`) has `trav_gene, trad_gene, traj_gene, trbv_gene, trbd_gene, trbj_gene` — but these were dropped at the enriched stage that feeds benchmark_v2/splits. To get V/D/J:

**Option A (preferred)**: Feed `'UNK'` placeholders for V/D/J (mirrors track_b's ERGO-II Phase 1 approach). DeepTCR treats unknown V/D/J as a no-op when the gene is `'UNK'`. The model loses the V/D/J signal but still trains; this is documented as a methodology limitation.

**Option B**: Run ANARCI on `tra_full` / `trb_full` to extract V/D/J. ~30 min for 60K rows; can run as a one-time cache to `track_a/deeptcr/cache/vj_calls.parquet`. Quality higher but slows the queue.

We start with **Option A** (UNK placeholders) for speed. If smoke-test AUC looks pathological, escalate to Option B.

## Negative sampling

`track_a/shared/negative_sampling.sample_negatives()` — overrides DeepTCR's native sampling.

## Conda env

DeepTCR ships TensorFlow 2.x as a hard dep. The `base` env has torch 2.11+cu130 but no TF. **Two options**:

1. `pip install DeepTCR tensorflow` into `base` — risk: TF/CUDA version conflict with torch (TF 2.x default CUDA may differ from CUDA 13).
2. Create sidecar env `track_a_deeptcr` with TF 2.x + DeepTCR.

We default to (2). Spec in `track_a/deeptcr/env.yml`. Smoke test runs in this env.

If TF refuses to use the L4 (sm_89 incompatibility for older TF builds), fall back to CPU. CPU runtime estimate: ~2h per task. Still acceptable.

## Smoke test

100 train rows + 50 val + 50 test_iid from `as_trb_i_*.parquet`. 5 epochs (DeepTCR's default minimum is 10; smoke test uses 5 to fit in 30s).
Pass criteria:
- DeepTCR imports successfully in the env
- TF detects the GPU (or fails over to CPU cleanly)
- Trains 5 epochs without NaN
- Test AUC > 0.5 (very weak; "model isn't dead")
- Per-condition results JSON written

Wall-clock target: ≤ 60s on L4 (or ≤ 5 min CPU fallback).

## Phase 2 wall-clock estimate

- AS-TRB-I (60K positives → 360K rows, default `epochs_min=10`): ~1h on L4 GPU
- AS-Paired-I (553K positives → 3.3M rows): ~6-8h on L4
- AS-TRB-II / Paired-II eval forward pass: ~10 min each
- Total GPU 1 queue: ~10-14h

If CPU fallback is required: multiply by 4-5x → may overflow time budget. Plan B: drop AS-Paired-I if CPU.

## WandB

- Project: `neurips`
- Group: `baselines`
- Run name: `deeptcr-{task}-{seed}`

DeepTCR has no native WandB hooks; we add manual logging in the wrapper.

## Output schema

`results/deeptcr/{task}_summary.json` per user-spec; score sidecar at `results/deeptcr/scores/{task}_{condition}.parquet`.

## Methodology caveats

- V/D/J = UNK placeholder. DeepTCR's V/D/J signal is removed; this is a deviation from the canonical DeepTCR usage. Document in ANOMALIES.md.
- TF 2.x in a sidecar env, not the canonical `base`. Document the env reproduction steps in `track_a/deeptcr/env.yml`.
- Class-II training uses class-I `_train` (no class-II train data by Phase 0).

## Risk: TF + L4 compatibility

TF 2.x compiled wheels typically support sm_70-86. sm_89 (L4) is supported by TF 2.13+ with CUDA 11.8+ at minimum, but the ambient CUDA on this g6 is 13.0. **Likely outcome: TF runs on CPU**, which is slow but functional. Plan A: try GPU first; Plan B: CPU.

## Files to be created in Phase 1c

```
track_a/deeptcr/
├── README.md
├── env.yml             (track_a_deeptcr conda env with TF 2.x + DeepTCR)
├── convert_data.py     (AS splits → DeepTCR's input arrays: beta_seq, alpha_seq, vdj genes (UNK), HLA, label)
├── train.py            (DeepTCR_SS init + Train_Sequences + 5x neg sampler integration)
├── evaluate.py         (Inference_Sequences for retrieval; emits per-condition JSON + score sidecar)
├── cache/              (gitignored — V/D/J cache if escalated to ANARCI)
└── results/            (gitignored)
```
