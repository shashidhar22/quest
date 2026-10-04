# Track A Verification: TULIP-TCR (original + retrained)

**Date**: 2026-05-04
**Authors**: Claude (Track A)
**Status**: Ready for review before Phase 1c implementation

## Scope

TULIP on all 4 main tasks (AS-TRB-I, AS-Paired-I, AS-TRB-II, AS-Paired-II), in BOTH arms:
- **TULIP-original**: zero-shot inference with the published pretrained checkpoint
- **TULIP-retrained**: fine-tune from the published checkpoint on each AS task's train split

## Upstream

- Repo: https://github.com/barthelemymp/TULIP-TCR (canonical TULIP)
- Pretrained checkpoint: ships with the repo at `model_weights/pytorch_model.bin`
- Tokenizers: `aatok/` and `mhctok/` directories in the repo
- Reference: Meynard-Piganeau et al., 2024. *TULIP: A Transformer-based Unsupervised Language model for Interacting Peptides and T-cell receptors that generalizes to unseen epitopes*

The legacy scripts (`benchmark_tulip_pretrained.py`, `benchmark_tulip_train.py`) provide a working PyTorch wrapper. Track A's `tulip/{train,evaluate}.py` adapts them.

## Mode

TULIP scores binding via the **negative log-likelihood (NLL)** of generating the peptide given (TCR, MHC). Lower NLL → more likely binding. We convert NLL to `-NLL` for retrieval ranking.

Architecture (per upstream paper):
- Three BERT-style encoders (TRA, TRB, MHC) sharing tokens
- Three decoders (TRA, TRB, peptide) producing autoregressive sequence likelihoods
- Cross-attention between encoders and decoders
- Trained on a corpus of paired (TCR, peptide, MHC) interactions

For TRB-only tasks: TULIP supports a `<MIS>` (missing) token in the TRA chain. We feed `<MIS>` for all TRA inputs.

## Hyperparameters (retrained arm)

From the legacy script defaults:
- `batch_size=512` (smaller in practice for L4 — confirmed during smoke test)
- `lr=1e-4`, `weight_decay=0.0`
- `num_epochs=100`, with `eval_every=10`
- `neg_per_pos=5` — negatives via `track_a/shared/negative_sampling`
- `masking_proba=0.0` (no chain masking by default)
- `skip_mhc=False` — use MHC encoder

The legacy default `num_epochs=100` is high; for time-budget reasons we may cap at 50 unless the loss is still moving. Document in `STATUS.md`.

## Inputs

- TRA chain: `<MIS>` for TRB-only tasks; otherwise `tra_cdr3`
- TRB chain: `trb_cdr3`
- Peptide: `peptide`
- MHC: TULIP uses `mhc_one_id` for class I; for class II we'll feed `mhc_two_id` similarly
- Class II tasks: train on the matching class-I `_train.parquet`; eval on `as_*_ii_eval.parquet`

The legacy script's MHC formatting helper (`benchmark_tulip_pretrained.py:setup_tokenizer`) maps HLA names to TULIP's mhctok vocabulary. We reuse that logic.

## Negative sampling

`track_a/shared/negative_sampling.sample_negatives()` — overrides TULIP's native peptide-swap so Track A and Track B share a sampling distribution.

## Conda env

Likely runs in `base` — TULIP uses `transformers` (BertModel, BertConfig, AutoTokenizer). Transformers 5.7 may have API drift from TULIP's pinned version (likely 4.x). **Smoke test will catch this.**

If the upstream code breaks on transformers 5.7, options:
1. Pin a transformers 4.x sidecar env (`track_a_tulip`)
2. Patch the wrapper (NOT upstream code) to translate API differences
3. Vendor specific TULIP files into `track_a/tulip/repo/` with minimal patch annotations

We default to (1). Sidecar env spec in `track_a/tulip/env.yml`.

## Smoke test

100 train rows + 50 val + 50 test_iid from `as_trb_i_*.parquet`. 2 epochs of retrained arm.
Pass criteria:
- TULIP repo cloned to `track_a/tulip/repo/`, pretrained weights present
- Both tokenizers (`aatok/`, `mhctok/`) load
- Pretrained model loads, forward pass on 1 batch succeeds (verifies `<MIS>` token is in vocab)
- Trains 2 epochs without NaN
- Per-condition results JSON written

Wall-clock target: ≤ 90s on L4.

## Phase 2 wall-clock estimate

- TULIP-original (inference, ~360K query x candidate-pool eval per task): ~30 min per task → 2h for 4 tasks
- TULIP-retrained: ~3-4h per task on L4 (per legacy estimate); ~12-16h for 4 tasks
- Total GPU 0 queue: ~14-18h

## WandB

- Project: `neurips`
- Group: `baselines`
- Run name: `tulip-{original,retrained}-{task}-{seed}`

## Output schema

`results/tulip-{original,retrained}/{task}_summary.json` per user-spec; score sidecar at `results/tulip-{original,retrained}/scores/{task}_{condition}.parquet`.

## Methodology caveats

- TULIP uses native NLL-as-score, which is not a calibrated probability. Recall@K via NLL ranking is well-defined; per-peptide AUC may have an unusual scale. Report sidecar full metrics so reviewers can normalize.
- TRB-only tasks rely on the `<MIS>` placeholder token. The TULIP paper trained with these placeholders, so this is in-distribution for the model — not a methodology hack.
- Class II training uses class-I `_train` (no class-II train data exists by Phase 0).

## Risk: upstream repo availability

The legacy script's hardcoded path `/home/sravisha/tcrbench_tools/TULIP-TCR/` does not exist on this g6 instance. We need to clone the canonical TULIP-TCR repo into `track_a/tulip/repo/`. Pretrained weights are bundled in the GitHub repo (per upstream README); confirm during smoke test.

If the GitHub clone fails or the weights are not in-repo (sometimes uploaded to Zenodo), STOP and ask user.

## Files to be created in Phase 1c

```
track_a/tulip/
├── README.md
├── env.yml             (transformers 4.x sidecar env if needed)
├── convert_data.py     (AS splits → TULIP-format DataFrame: tra, trb, peptide, mhc_one_id columns)
├── train.py            (--mode {original, retrained})
├── evaluate.py         (NLL scoring against candidate pool; per-condition JSON + sidecar)
├── repo/               (cloned upstream; gitignored)
└── results/            (gitignored)
```
