# Track A — Pre-Launch Verification

**Date**: 2026-05-04 (T-75h to NeurIPS 2026 ED deadline)
**Instance**: g6.12xlarge, 4× NVIDIA L4 (sm_89), CUDA 13.0
**Repo**: `/home/ubuntu/quest`, branch `tcrbench2`

## 1. Hardware

`nvidia-smi` confirmed at 04:45 UTC:
- 4× L4, 23 GB each, all idle (0 MiB used, no processes)
- Driver 580.126.16 / CUDA 13.0
- Persistence mode on, P8 idle state, ECC clean

## 2. Repo + data

- Working dir: `/home/ubuntu/quest` (verified)
- Branch: `tcrbench2` (clean working tree)
- Splits: `/home/ubuntu/quest/benchmark_v2/splits/` (NOT `data/benchmark_v2/...`; the `data/` dir is empty)
- All AS Class I 8-partition splits present (`train, val, test_iid, test_novel_tcr, test_novel_pep, test_novel_allele, test_level4, test_mixed`)
- AS Class II eval-only files present (`as_*_ii_eval.parquet`); see `PHASE0_CLASSII_INVESTIGATION.md` for the design rationale
- Candidate pool: `as_class_i_candidate_pool.tsv` (3,564 unique peptides for retrieval)
- Wiki inventory: `docs/wiki/benchmark_v2/inventory.json` (canonical row counts)

## 3. Conda envs (status)

| Env | Status | Used by |
|---|---|---|
| `base` (active) | ✓ torch 2.11+cu130, transformers 5.7.0, peft 0.19.1, esm 3.2.1, wandb 0.26.1, sklearn 1.8.0 | LSTM/MLP, TCR-BERT, ESM SFT |
| `track_a_tulip` | TBD — create in Phase 1c | TULIP (likely needs transformers 4.x) |
| `track_a_deeptcr` | TBD — create in Phase 1c | DeepTCR (TF 2.x sidecar) |
| `track_b_*` | present, not used here (g4dn baselines) | — |

## 4. Per-baseline verification doc references

Each baseline has its own verification doc in `verification/`:
- `verification/lstm_mlp_verification.md` — reuses architecture from legacy `benchmark_baseline.py`; new data loading.
- `verification/tcrbert_verification.md` — `wukevin/tcr-bert-mlm-only` + peptide-conditioning fix; **single-chain shim** for TRB-only tasks.
- `verification/esm_sft_verification.md` — 2×3 matrix (ESM2-8M, ESM2-35M, ESM-C 300M × {base, CPT}); InfoNCE retrieval head.
- `verification/tulip_verification.md` — original (zero-shot NLL) + retrained arms.
- `verification/deeptcr_verification.md` — TF 2.x sidecar; UNK V/D/J placeholder.

## 5. Track A shared utilities (`track_a/shared/`)

Smoke-tested 2026-05-04:

| Module | Purpose | Status |
|---|---|---|
| `load_splits.py` | AS Class I (8 conditions) + Class II (eval-only) loaders | ✓ verified — loaded `as_trb_i_test_iid` (1,044 rows) and `as_trb_ii_eval` (56,076 rows) |
| `negative_sampling.py` | Re-export of `track_b/shared/negative_sampling.py` (cross-track parity) | ✓ verified — NEG_SEED=43, NEG_RATIO=5 |
| `allele_utils.py` | Re-export of `track_b/shared/allele_utils.py` | ✓ verified — `is_hla_name`, `resolve_allele`, `filter_named_alleles` |
| `retrieval_metrics.py` | Wraps `quest.metrics.retrieval_metrics`; emits user-spec headline metrics | ✓ verified — sane numbers on random scores |
| `results_schema.py` | Per-task results JSON + per-(query, candidate) score sidecar | ✓ ready |
| `task_aliases.py` | AS-* ↔ ICML task-name mapping (forward-compat only) | ✓ ready |

## 6. WandB

- Project: `neurips`
- Group: `baselines` (5 simple-NN baselines) or `esm-sft` (ESM 2×3 matrix)
- Login confirmed (wandb 0.26.1 present in `base`)

## 7. Output structure

```
results/
├── lstm/{task}_summary.json + scores/{task}_{condition}.parquet
├── mlp/...
├── tulip-original/...
├── tulip-retrained/...
├── deeptcr/...
├── tcrbert/...
└── esm2_8m-base/, esm2_8m-cpt/, esm2_35m-base/, esm2_35m-cpt/, esmc_300m-base/, esmc_300m-cpt/...
```

JSON schema follows the user-required spec. Score sidecars enable post-hoc Normalized Recall@K.

## 8. Blockers (resolved)

| Blocker | Resolution |
|---|---|
| Track A scaffolding does not exist | Built from scratch this session, mirroring `track_b/` |
| Legacy `scripts/benchmark/*` is ICML-only | Not used; Track A reads `benchmark_v2/splits/` directly |
| Class II 4-condition split missing | Class II is eval-only by design (PHASE0); train on class I, eval on `_eval` |
| TCR-BERT paired-only | Single-chain shim (empty-TRA token sequence); flagged in ANOMALIES.md |
| DeepTCR not installed | Sidecar env `track_a_deeptcr` planned in Phase 1c |
| TULIP path missing | Clone canonical TULIP-TCR into `track_a/tulip/repo/` |
| ESM 2×3 matrix lacks driver | New `track_a/esm_sft/` driver per verification doc |

## 9. STOP-and-ask points (active)

If any of these arise during Phase 1c:
- DeepTCR pip install fails or TF refuses sm_89 → ask before falling back to CPU
- TULIP repo / weights not on GitHub → ask before searching alternative sources
- A baseline's smoke test fails 3+ times → ask before continuing
- Any methodology deviation not documented in this verification → log to ANOMALIES.md and notify

## 10. Next steps

Per the approved plan: proceed to Phase 1c (implement convert_data/train/evaluate per baseline + smoke tests). Verification-doc batch review can be done by user asynchronously; if any doc is invalidated, the corresponding code can be revised — none of the implementation is hard to reverse.
