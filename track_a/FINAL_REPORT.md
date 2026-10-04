# Track A — FINAL REPORT (in progress)

**Status**: Phase 2 in progress. This file is updated as cells complete.

## Coverage matrix

(See `track_a/launch/aggregate_results.py --md` for the live table; copied here at last update.)

```
$ python3 track_a/launch/aggregate_results.py --md
```

Last copied: <pending — run again when more cells land>

## Methodology summary

- All baselines train on `benchmark_v2/splits/{task}_train.parquet` (class I) and evaluate on `_test_<condition>.parquet` (class I) or `_eval.parquet` (class II).
- Negatives: 5× per positive, peptide-swap, seed=43, positive-set filtered. Cross-track parity with Track B preserved (re-export from `track_b/shared/negative_sampling.py`).
- Class II: train on the matching class-I task, evaluate on class II eval (eval-only by design — see `PHASE0_CLASSII_INVESTIGATION.md`).
- Candidate pool: published `as_class_i_candidate_pool.tsv` (3,563 peptides) for class I; class II uses `union(class-I pool, class-II eval peptides)` (~4,300 peptides). See `ANOMALIES.md`.
- Eval row cap: `--max_test_rows 2000` per condition for tractability. The score sidecar parquet captures all (query, candidate, score) triples.

## Per-baseline highlights

(Filled in as cells complete.)

### LSTM (placeholder; full table pending)
- AS-TRB-I: recall@10 — iid 0.023, novel_tcr 0.542, novel_pep 0.000, novel_allele 0.000, level4 0.000.
- Pattern: novel_tcr is much higher than iid because per the BENCHMARK_SPLITS spec the iid stratification deliberately holds out peptide clusters — so "iid" here is peptide-novel from the model's perspective, while novel_tcr is peptide-familiar. Worth flagging in the discussion.

### MLP
(Pending)

### DeepTCR
(Pending)

### TCR-BERT-with-peptide
(Pending)

### TULIP-original (zero-shot)
(Pending; class II skipped per upstream — it wasn't pretrained on class II MHC)

### TULIP-retrained
(Skipped — see `ANOMALIES.md`. May add if a sidecar env can be stood up post-deadline.)

### ESM 2×3 matrix
(Pending — 24 cells expected)

## Anomalies (full list in ANOMALIES.md)

1. TCR-BERT empty-TRA shim for TRB-only tasks.
2. DeepTCR uses class-prob lookup over training peptides → 0 for peptides not in train classes.
3. ESM "decoder SFT" interpreted as scorer-head SFT (not generative decoder).
4. Class-II candidate pool synthesized as union of class-I pool + class-II peptides.
5. Eval row cap 2000.
6. TULIP-original needed `track_a_tulip` sidecar env (transformers 4.30) due to upstream-API breakage in transformers 5.x.

## WandB

Project: `neurips`
Groups: `baselines` (LSTM, MLP, TCR-BERT) and `esm-sft` (ESM 2×3 matrix)
URL: https://wandb.ai/shashidhar-r-shankar-fred-hutchinson-cancer-center/neurips

## GPU-hours used (by Phase 2 end)

(Pending tally.)
