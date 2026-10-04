# Track A Verification: ESM 2×3 Decoder-SFT Matrix

**Date**: 2026-05-04
**Authors**: Claude (Track A)
**Status**: Ready for review before Phase 1c implementation

## Scope

24 cells: encoder ∈ {ESM2-8M, ESM2-35M, ESM-C 300M} × variant ∈ {base, CPT} × task ∈ {AS-TRB-I, AS-Paired-I, AS-TRB-II, AS-Paired-II}. The ESM-2×3 driver SFTs a thin retrieval head on top of each frozen-or-LoRA-adapted encoder and reports `recall_at_10 / mrr / hit_at_1` per condition.

## Upstream

| Encoder | HF / source | Params |
|---|---|---|
| ESM2-8M base | `facebook/esm2_t6_8M_UR50D` | 8 M |
| ESM2-8M CPT | base + PEFT adapter `sravisha/esm2-8m-cpt-fnd100m` | 8 M + 1.13 M LoRA |
| ESM2-35M base | `facebook/esm2_t12_35M_UR50D` | 35 M |
| ESM2-35M CPT | base + PEFT adapter `sravisha/esm2-35m-cpt-fnd100m` | 35 M + 3.35 M LoRA |
| ESM-C 300M base | EvolutionaryScale `esmc_300m` (loaded via `esm` package) | 300 M |
| ESM-C 300M CPT | base + PEFT adapter `sravisha/esmc300m-foundation100m-c3-lora-r32` | 300 M + 14.7 M LoRA |

Loading patterns are documented and verified in `docs/CPT_HF_CHECKPOINTS.md` (cold-cache and warm-cache load times confirmed 2026-05-04). The ESM-C config-shim (no `.config` attribute on ESMC; PEFT requires one) is required and documented there.

## Architecture

```
   sequences (TRA, TRB, peptide, MHC pseudoseq)
                  │
       ┌──────────┴───────────┐
       │  Encoder forward     │   (ESM2 or ESM-C; LoRA-adapted if CPT variant)
       │  → token embeddings  │   ESM2 returns last-hidden-state via HF AutoModel
       │                      │   ESM-C returns sequence_logits via base_model.model(...)
       └──────────┬───────────┘
                  │
        ┌─────────┴─────────┐
        │  AttentionPooling │   from quest/models/attention_pooling.py
        │  → fixed vector   │
        └─────────┬─────────┘
                  │
   ┌──────────────┴──────────────┐
   │ DeepProjectionHead (2-layer │
   │  MLP, BatchNorm, ReLU)      │
   └──────────────┬──────────────┘
                  │
              tcr_emb,
              pep_emb (separate forward pass)
                  │
           cosine score(tcr_emb, pep_emb)
```

For the AS task: each row is `(TCR, MHC, peptide+, [peptide- ×5])`. The training loss is **InfoNCE on (tcr_emb, peptide_emb) pairs**, where positives are the (cognate TCR, peptide) pair and negatives are in-batch other TCRs' positives + the 5× peptide-swap negatives. This is contrastive retrieval, matching how Recall@K is measured at eval.

Why InfoNCE vs. binary BCE (used by LSTM/MLP/TCR-BERT)? At the 24-cell scale, contrastive training is the standard for ESM-style retrieval and is what the user prompt's "decoder SFT" implies. We reuse `quest/models/contrastive_loss.py:DistributedRobustInfoNCE` which is already in-tree.

## Encoder freezing strategy

- **Base variants**: Freeze the encoder entirely; train only the AttentionPooling + DeepProjectionHead.
- **CPT variants**: Freeze the base model; LoRA adapter weights remain trainable (loaded as PEFT — already trainable by default). AttentionPooling + DeepProjectionHead also trained.

This keeps GPU memory in budget on a single L4 (24 GB) for all three encoder sizes.

## Inputs

- TRB-only tasks: input `[TRB] {trb_cdr3} [ETRB] [PEP] {peptide} [EPEP] [MHO] {mhc_pseudoseq} [EMHO]`.
- Paired tasks: prepend `[TRA] {tra_cdr3} [ETRA]`.
- Class II uses `mhc_two_pocket_contact` instead of `mhc_one_pocket_contact` and `[MHT]/[EMHT]` delimiters.
- Special tokens are taken from `quest/constants.py`.

For class II eval, we use the matching class-I encoder weights as initialization (since class II is eval-only). For "retrained" cells, we additionally do one short SFT pass on class-I `_train` and evaluate on class-II `_eval`.

## Negative sampling

In-batch negatives + 5× peptide-swap explicit negatives (via `track_a/shared/negative_sampling`). The 5× negatives are useful to ensure pairs that look like in-distribution data (not random sequences) are negatives.

## Hyperparameters

- Optimizer: AdamW, `lr=2e-4` for LoRA-only training; `lr=1e-4` for head-only training (base variants).
- Schedule: cosine with 10% warmup.
- Batch size: tuned per encoder. ESM2-8M `bs=128`; ESM2-35M `bs=64`; ESM-C 300M `bs=16`.
- Epochs: 5 for ESM2 cells, 3 for ESM-C cells (larger model converges faster on retrieval).
- Mixed precision: bf16.

## Conda env

`base` — has torch 2.11+cu130, transformers 5.7.0, peft 0.19.1, esm 3.2.1 (EvolutionaryScale).

`docs/CPT_HF_CHECKPOINTS.md` confirmed all three CPT adapters are loadable from HF on this instance with the documented patterns.

## Smoke test

For each of the 6 (encoder × variant) cells: run a 100-row AS-TRB-I slice for 2 epochs at the smallest reasonable batch size (bs=8 for ESM-C, bs=32 for ESM2-*).
Pass criteria:
- Encoder loads from HF (cold or cached) without error
- LoRA attaches successfully (CPT variants)
- Forward pass produces finite logits, no NaN
- Loss decreases over 2 epochs
- Per-condition results JSON parseable

Wall-clock target: ≤ 90 s per cell on L4 (ESM-C 300M is slowest).

## Phase 2 wall-clock estimate (per cell, AS-TRB-I full train)

| Cell | Train rows | Epochs | Wall-clock |
|---|---|---|---|
| ESM2-8M base | 360K | 5 | ~30 min |
| ESM2-8M CPT | 360K | 5 | ~30 min |
| ESM2-35M base | 360K | 5 | ~45 min |
| ESM2-35M CPT | 360K | 5 | ~45 min |
| ESM-C 300M base | 360K | 3 | ~1.5h |
| ESM-C 300M CPT | 360K | 3 | ~1.5h |

Total for TRB-I column: ~5.5h on a single GPU. Run TRB-I column on GPU 2 (after MLP finishes) and GPU 3 (after TCR-BERT finishes) in parallel: ~3h wall-clock for the headline column.

Paired-I column will be ~10× rows but bs unchanged → ~50h wall-clock single-GPU. With 2-4 GPUs, ~12-25h. **The Paired-I column is the biggest risk to fitting Phase 2 in the time budget.** Plan B: run the full TRB-I column + 1-2 cells of Paired-I (e.g., ESM2-8M CPT only) and report the rest as "not run for time."

Class II cells: load the trained class-I encoder and evaluate on `_eval.parquet`. Forward pass only. ~5 min per cell.

## WandB

- Project: `neurips`
- Group: `esm-sft`
- Run name: `{esm2_8m,esm2_35m,esmc_300m}-{base,cpt}-{task}-{seed}`

## Output schema

`results/{encoder}-{variant}/{task}_summary.json` per user-spec; score sidecar at `results/{encoder}-{variant}/scores/{task}_{condition}.parquet`.

The encoder-variant naming convention:
- `esm2_8m-base`, `esm2_8m-cpt`
- `esm2_35m-base`, `esm2_35m-cpt`
- `esmc_300m-base`, `esmc_300m-cpt`

## Methodology caveats

- Decoder = a small AttentionPooling + 2-layer MLP scorer head, NOT a transformer decoder. The user-prompt phrasing "decoder SFT" refers to fine-tuning the prediction head on top of the (pretrained) encoder, not training a generative decoder. Document in ANOMALIES.md.
- ESM-C does not accept `input_ids` like an HF transformers model — its forward call is `model.base_model.model(sequence_tokens=..., sequence_id=...)`. The wrapper handles this branch.
- Tokenizer mismatch: ESM2 uses `EsmTokenizer` (HF AutoTokenizer); ESM-C uses `EsmSequenceTokenizer` from the `esm` package. Wrappers detect by `isinstance` and route correctly.
- Class II eval uses the class-I-trained encoder as-is. There is no class-II-trained variant because class II has no train split (Phase 0).

## Files to be created in Phase 1c

```
track_a/esm_sft/
├── README.md
├── env.yml           (uses base)
├── convert_data.py   (AS splits → tokenized retrieval pairs)
├── encoders.py       (load_esm2_base, load_esm2_cpt, load_esmc_base, load_esmc_cpt)
├── model.py          (RetrievalModel: encoder + AttentionPooling + DeepProjectionHead)
├── train.py          (InfoNCE training loop; --encoder {esm2_8m,esm2_35m,esmc_300m} --variant {base,cpt})
├── evaluate.py       (retrieval against candidate pool; emits per-condition JSON + score sidecar)
└── results/          (gitignored)
```
