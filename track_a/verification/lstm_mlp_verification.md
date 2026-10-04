# Track A Verification: LSTM and MLP

**Date**: 2026-05-04
**Authors**: Claude (Track A)
**Status**: Ready for review before Phase 1c implementation

## Scope

Two simple-NN baselines on AS Class I (TRB-I, Paired-I) and AS Class II (TRB-II, Paired-II eval-only). Both speak `as_*` task names natively in Track A — no upstream third-party repo to clone. The legacy `scripts/benchmark/benchmark_baseline.py` already implements both architectures; Track A's `lstm_mlp/{train,evaluate}.py` will reuse the architectures verbatim while replacing the data-loading and metrics layers.

## Architecture

### LSTM

- Input encoding: amino-acid vocabulary (size 22 incl. PAD/UNK), embedded into `embed_dim=128`.
- Per-chain encoder: bidirectional LSTM, `hidden_dim=128`, `num_layers=2`, `dropout=0.3`.
- Final hidden states from forward+backward concat → fixed-size sequence representation.
- Concat `(tcr_repr, peptide_repr, mhc_one_hot)` → MLP head (2 hidden layers, 128 units, dropout 0.3, ReLU).
- Output: scalar binding logit; BCE-with-logits loss.

### MLP

- Input encoding: same AA vocab, average-pooled embeddings (no recurrence).
- Concat `(tcr_avg, peptide_avg, mhc_one_hot)` → MLP head identical to LSTM's.
- Otherwise same: scalar binding logit + BCE.

Hyperparameters (from legacy script defaults): `batch_size=64`, `lr=1e-3`, `num_epochs=50`, `patience=10` early-stopping on val AUC, `max_seq_len=512`. Will keep these for parity with the legacy ICML numbers.

## Inputs & data layout

- TRB-only tasks: `trb_cdr3` + `peptide` + MHC. TRA inputs are zero/empty.
- Paired tasks: `tra_cdr3` + `trb_cdr3` + `peptide` + MHC.
- MHC encoded as one-hot over the union of HLA alleles seen in `as_*_train.parquet` (allele list cached as `track_a/lstm_mlp/cache/allele_vocab.json`).
- For class II tasks: train on the matching class-I `_train.parquet`, evaluate on `as_*_ii_eval.parquet` (eval-only by Phase 0 verdict).

## Negative sampling

Use `track_a/shared/negative_sampling.sample_negatives()` directly. 5× negatives per positive, peptide-swap, seed=43, positive-set filtered. Cross-track parity with Track B is preserved.

## Metrics

Native: BCE loss, val AUC for early-stopping.
Reported: per-condition (test_iid, test_novel_tcr, test_novel_pep, test_novel_allele, test_level4 for class I; eval for class II) headline metrics `recall_at_10 / mrr / hit_at_1` via `track_a/shared/retrieval_metrics.headline_metrics`.

Retrieval evaluation: for each test row, score the (TCR, candidate_peptide) pair for every peptide in `as_class_i_candidate_pool.tsv` (3,564 candidates) using the trained binding head. Rank candidates by score; locate the true peptide's rank. Hit@1 / Recall@10 / MRR computed over those ranks.

## Conda env

`base` — has torch 2.11+cu130, sklearn 1.8.0, pandas, numpy. No additional packages.

## Smoke test

100 train rows + 50 val + 50 test_iid from `as_trb_i_*.parquet`, 2 epochs each architecture. Pass criteria:
- Trains without NaN or all-zero gradient
- Val AUC > 0.5 after 2 epochs (very weak threshold; serves only as "model isn't dead")
- Outputs the per-condition results JSON via `track_a.shared.results_schema.write_results`

Wall-clock target: ≤ 30s on L4.

## Phase 2 wall-clock estimate

- AS-TRB-I full train (60K positives → 360K rows with negatives, 50 epochs): LSTM ~30 min, MLP ~15 min on L4
- AS-Paired-I (553K positives → 3.3M rows): LSTM ~3-4h, MLP ~2h
- AS-TRB-II / Paired-II evaluation only: ~5 min each (forward pass)
- Total per architecture × 4 main tasks: LSTM ~5h, MLP ~3h

GPU 2 queue: LSTM all 4 main tasks → MLP all 4 main tasks. Total ~8h, fits within the budget for that GPU.

## WandB

- Project: `neurips`
- Group: `baselines`
- Run name: `{lstm,mlp}-{as_trb_i,as_paired_i,as_trb_ii,as_paired_ii}-{seed}`

## Output schema

Per-task `results/{lstm,mlp}/{task}_summary.json` matching the user-spec; per-(query, candidate) score sidecar at `results/{lstm,mlp}/scores/{task}_{condition}.parquet`.

## Methodology caveats

- LSTM/MLP have no notion of CDR1/2 or full-chain V/J context. They use CDR3 only. This is intentional — these are simple baselines, not state-of-the-art.
- MHC encoding is one-hot, NOT a protein-sequence embedding. Allele-novel test rows will have all-zero MHC vectors → predictions are TCR-peptide-driven only on `test_novel_allele`. Document in ANOMALIES.md.
- No CDR3 length filtering. Padding to `max_seq_len=64` (longer than max observed CDR3 lengths in benchmark_v2).

## Files to be created in Phase 1c

```
track_a/lstm_mlp/
├── README.md
├── env.yml          (no-op — uses base; included for parity)
├── convert_data.py  (pulls AS splits + injects negatives via shared)
├── train.py         (trains LSTM or MLP; --model {lstm, mlp} flag)
├── evaluate.py      (retrieval against candidate pool; emits per-condition JSON + score sidecar)
└── results/         (gitignored; per-task subdirs)
```
