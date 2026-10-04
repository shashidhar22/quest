# Track A Verification: TCR-BERT (with peptide conditioning)

**Date**: 2026-05-04
**Authors**: Claude (Track A)
**Status**: Ready for review before Phase 1c implementation

## Scope

TCR-BERT-with-peptide on all 4 main tasks: AS-TRB-I, AS-Paired-I, AS-TRB-II, AS-Paired-II. Single-chain (TRB-only) is supported via a documented shim — see "Methodology caveat" below and `ANOMALIES.md`.

## Upstream

- Repo: https://github.com/wukevin/tcr-bert
- HF model: `wukevin/tcr-bert-mlm-only` (BERT MLM-only checkpoint, used for fine-tuning)
- Inference-only model: `wukevin/tcr-bert` (has a classifier head; we don't use it)
- Reference paper: Wu et al., 2021. *TCR-BERT: learning the grammar of T-cell receptors for flexible antigen-binding analyses*.

The legacy script (`scripts/benchmark/benchmark_tcrbert_train.py`) is a working PyTorch wrapper that we will adapt. We do NOT modify the upstream HF model, only the wrapper and head.

## Mode

**Retrained with peptide-conditioning fix.** ICML-version-style runs encoded peptide + TCR separately and pooled at the end; this version includes peptide tokens in the same encoder pass so cross-attention sees the peptide. This is the "peptide-conditioning fix" referenced in the user prompt.

Architecture:
- Two separate encoder passes (or one shared encoder via `--shared_encoder`): one for TCR, one for peptide.
- Token-level outputs pooled via `--seq_pooling ∈ {cls, mean, max, pool}`. Default `cls`.
- Concat `(tcr_pool, peptide_pool, mhc_one_hot)` → linear classification head.
- Loss: BCE-with-logits. 5× negatives via shared sampler.
- Hyperparameters: `batch_size=128, lr=3e-5, num_epochs=25, weight_decay=0.0, dropout=0.2`. Preserves legacy defaults.

## Single-chain shim (for AS-TRB-I, AS-TRB-II)

The legacy script enforces paired tasks only by filtering rows with NaN TRA. AS-TRB-I and AS-TRB-II have **no `tra_cdr3` populated** (column is all NaN) — paired-only filtering would drop the entire dataset.

**Resolution**: skip the paired-only assertion. For TRB-only tasks, feed an empty-string TRA token sequence into the TRA encoder; the BERT cls-token output of an empty TRA string acts as a constant offset that the head learns to ignore. Functionally similar to TULIP's `<MIS>` placeholder.

**Methodology caveat (logged in `ANOMALIES.md`)**: TCR-BERT-with-peptide on TRB-only tasks uses an empty-TRA shim. This is a deviation from the upstream paper which presumed paired data. The shim is implementation-level (no upstream code changes); we report it transparently. An alternative is to skip TRB-only tasks for TCR-BERT — defer to user if smoke-test signal looks bad.

## Inputs

- TRB-only: `trb_cdr3` + `peptide` + MHC. TRA encoder fed an empty-string sequence.
- Paired: `tra_cdr3` + `trb_cdr3` + `peptide` + MHC.
- For class II tasks: train on the matching class-I `_train.parquet` (e.g. `as_paired_ii` trained on `as_paired_i_train.parquet`), evaluate on `as_*_ii_eval.parquet`.

MHC is one-hot over the allele vocabulary (same as LSTM/MLP baseline). No protein-sequence MHC features.

## Negative sampling

`track_a/shared/negative_sampling.sample_negatives()` — 5× peptide-swap, seed=43.

## Conda env

`base` — has torch 2.11+cu130, transformers 5.7.0, peft 0.19.1, wandb 0.26.1. The HF model loads via `BertModel.from_pretrained("wukevin/tcr-bert-mlm-only")`. **First-time download requires internet to `huggingface.co`** — confirm during smoke test.

If the model is gated (it shouldn't be — it's public), we'll log in to HF via `HF_TOKEN` env var.

## Smoke test

100 train rows + 50 val + 50 test_iid from `as_trb_i_*.parquet`. 2 epochs.
Pass criteria:
- Loads `wukevin/tcr-bert-mlm-only` (or fails fast with a clear error)
- Trains without NaN
- Val AUC > 0.5 after 2 epochs
- Per-condition results JSON written

Wall-clock target: ≤ 60s on L4 (BERT model has 12 layers, 768-dim — heavier than LSTM/MLP).

## Phase 2 wall-clock estimate

- AS-TRB-I (60K positives × 6 = 360K rows, 25 epochs, batch 128, L4): ~1.5h with TRA-shim
- AS-Paired-I (3.3M rows): ~6h
- AS-TRB-II / Paired-II eval forward pass: ~10 min each
- Total: ~8h GPU 3 queue

## WandB

- Project: `neurips`
- Group: `baselines`
- Run name: `tcrbert-{task}-{seed}`

## Output schema

`results/tcrbert/{task}_summary.json` per user-spec; score sidecar at `results/tcrbert/scores/{task}_{condition}.parquet`.

## Files to be created in Phase 1c

```
track_a/tcrbert/
├── README.md
├── env.yml           (uses base; included for parity)
├── convert_data.py   (AS splits → (tcr, peptide, mhc, label) tuples + 5x negatives)
├── train.py          (HF AutoTokenizer + AutoModel; concat-pool head; BCE loss)
├── evaluate.py       (retrieval against candidate pool)
├── single_chain_shim.py  (factor out the empty-TRA-string token construction)
└── results/          (gitignored)
```

## Notes on the legacy ICML script

`scripts/benchmark/benchmark_tcrbert_train.py` covers the basic architecture but has these blockers we'll fix in Track A's `train.py`:
1. Hardcoded ICML data layout — we read AS-* parquet directly via `track_a/shared/load_splits`.
2. `--task` accepts only ICML names — we accept `as_*` natively.
3. Paired-only filter — we replace with the single-chain shim above.
4. Output schema is `metrics.json` — we emit per-condition results JSON via `track_a.shared.results_schema`.
5. Native negative sampling — we override with `track_a.shared.negative_sampling`.

Architecture (encoders + head + pooling) is preserved unchanged.
