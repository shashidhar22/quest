# Benchmark Foundation MLM Datasets

Train / val / test split-by-construction for foundation MLM pretraining over
the QUEST corpus, plus three scaled training-sample manifests (10M / 100M /
500M) and several mini-ablation variants for sampling-weight (M2, M3) and
ordering (O1, O3) studies.

Outputs live at `data/benchmark_v2/foundation/` and are produced by
`scripts/data_processing/build_benchmark_foundation.py`.

---

## 1. Purpose & consumers

Foundation MLM pretraining needs:

- **Held-out val/test sets** that never appear in any benchmark test/eval
  partition, so MLM checkpoints can be evaluated cleanly.
- **A train pool** that excludes those held-outs and the benchmark exclusion
  list.
- **Sample manifests** at multiple scales so the same training pipeline can
  produce 10M / 100M / 500M-row streams without re-sampling.
- **Ablation variants** for sampling-weight (M1/M2/M3) and ordering
  (O1/O2/O3) studies.

Downstream consumers:
- Tokenization pipeline reads each manifest's `(source_file,
  source_row_index)` and pulls full sequences from the enriched parquet.
- Training script applies random masking on each forward pass.

---

## 2. Pipeline overview

```
Phase 1 clusters + Phase 2 splits/  ──┐
                                      │
data/deduplicated_again/exploded_     │
  deduped_enriched/  (1.44 B rows)    │
                                      ▼
                  ┌─────────────── exclusion (Step 1) ─────────────┐
                  │  splits/*test*.parquet + *eval*.parquet         │
                  │  → bio_hash set (~3.7 M unique)                 │
                  └──────────────────────┬──────────────────────────┘
                                         │
                                         ▼
        ┌──── partition_assignments (Step 3) ────┐
        │ scan source ⨝ allele lookup ⨯ exclusion │
        │ → bio_hash, partition (mod-20 of hash)  │
        │ → ~1.43 B rows (after exclusion drop)   │
        └────────────────────┬────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
      foundation_val   foundation_test   train pool (90 %)
       (~5 % / ~70 M)  (~5 % / ~70 M)
                                              │
                  ┌─── scaled (Step 5) ────────┤
                  │  M1 sqrt weights            │
                  │  10M / 100M / 500M          │
                  └────────────┬────────────────┘
                               │
                  ┌─── mini ablations (Step 6/7) ┐
                  │  M2 / M3 weights @ 10M       │
                  │  O1 / O3 orderings @ 10M     │
                  └──────────────────────────────┘
```

---

## 3. Inclusion / exclusion rules

**27 in-scope `subset_key` values** (every subset that has at least one TCR
chain OR has peptide+MHC):

```
1-mol:  tra, trb
2-mol:  tra_trb, tra_peptide, trb_peptide,
        tra_mhc_one, trb_mhc_one, tra_mhc_two, trb_mhc_two,
        peptide_mhc_one, peptide_mhc_two
3-mol:  tra_trb_peptide,
        tra_trb_mhc_one, tra_trb_mhc_two,
        tra_peptide_mhc_one, trb_peptide_mhc_one,
        tra_peptide_mhc_two, trb_peptide_mhc_two,
        peptide_mhc_one_mhc_two,
        tra_mhc_one_mhc_two, trb_mhc_one_mhc_two
4-mol:  tra_trb_peptide_mhc_one, tra_trb_peptide_mhc_two,
        tra_trb_mhc_one_mhc_two,
        tra_peptide_mhc_one_mhc_two, trb_peptide_mhc_one_mhc_two
5-mol:  tra_trb_peptide_mhc_one_mhc_two
```

**4 excluded** subset_keys:
- `peptide` (16.7M peptides only — computational predictions, no biological signal at row level)
- `mhc_one` (16.6K standalone MHC-I sequences)
- `mhc_two` (5.7K standalone MHC-II sequences)
- `mhc_one_mhc_two` (133 paired MHCs)

These contain no TCR and don't co-occur with peptide-MHC pairing — they're
already covered by the peptide/mhc combos in scope.

---

## 4. Biological-example hash

Identity = `(tra_cdr3, trb_cdr3, peptide, mhc_one_allele, mhc_two_allele)`.

Hash spec:

```sql
md5(
    coalesce(tra_cdr3,  '') || '|' ||
    coalesce(trb_cdr3,  '') || '|' ||
    coalesce(peptide,   '') || '|' ||
    coalesce(mhc_one_allele, '') || '|' ||
    coalesce(mhc_two_allele, '')
)
```

Returns 32-character hex MD5. Computed inside DuckDB SQL — the same
expression is used for the exclusion list (Step 1) and the partition
assignment (Step 3), guaranteeing identical hashes across phases.

**Why CDR3 (not full chain)?** The same biological TCR can appear with
different full-chain truncations across sources. CDR3 + V/J gene anchor is
the canonical specificity-determining region; using CDR3 collapses
truncation variants into one biological identity.

---

## 5. Source / allele reconstruction

The enriched stage doesn't carry `mhc_one_allele` / `mhc_two_allele`. We
left-join with `splits/lookup/sequence_to_source_allele.parquet` (built in
Phase 2 from the non-deduplicated `data/standardized_again/`):

```sql
COALESCE(lookup.mhc_one_allele, enriched.mhc_one) AS mhc_one_allele
COALESCE(lookup.mhc_two_allele, enriched.mhc_two) AS mhc_two_allele
```

When the lookup misses (rare), the protein sequence itself stands in as the
allele identifier — same convention used in Phase 2 splits. This guarantees
the bio_hash matches what the splits parquets already contain.

---

## 6. Partitioning rule

```python
partition_idx = duckdb_hash(bio_hash) % 20
0 → test
1 → val
2..19 → train
```

`duckdb_hash` is DuckDB's built-in `hash()` function (uint64). Uniform
distribution of MD5 prefixes guarantees the mod-20 partition is a clean
~5/5/90 split.

Expected proportions:
- **train**: 90% (≈ 1.29 B rows)
- **val**: 5% (≈ 72 M rows)
- **test**: 5% (≈ 72 M rows)

After excluding rows whose bio_hash appears in any benchmark test/eval:
total ≈ 1.44 B − ~ overlap ≈ ~1.43 B foundation rows.

The same biological example always lands in the same partition regardless
of `(subset_key, order_key)`. So an example used for foundation training in
its `tra_trb_peptide` form is also used in its `tra_trb_peptide_mhc_one`
form (with all order_key variants) — they all share the bio_hash.

---

## 7. Sampling weights & exposure cap

For each in-scope subset c with `N_c` unique training bio_hashes and `k_c`
molecules:

| Variant | W_c (raw, before normalization) |
|---------|---------------------------------|
| **M1** (foundation_*) | `1 / sqrt(min(N_c, 100M)) if c == 'trb' else 1 / sqrt(N_c)` |
| **M2** (mini_M2) | `M1 × min(k_c, 3)` |
| **M3** (mini_M3) | `M1 × k_c` |

After raw weights are normalized to sum to 1, per-subset target rows = `T ×
W_c`.

**Why the trb size cap (100 M)?** trb has ~860M unique examples (foundation
scope). Without the cap, trb's `1/sqrt(860M)` would drive its weight nearly
to zero and starve trb of representation in the sampled mix. The cap rescues
trb's share. Documented for transparency.

**Exposure cap**: no biological example can be sampled more than 20 times.
For each c, target is also capped at `min(k_c, 20) × N_c`. Overflow from
capped subsets is redistributed to uncapped ones in proportion to remaining
capacity (iterative, converges in ≤50 passes).

**Per-sample order_key**: independent uniform pick from c's k_c orderings,
with replacement. Two samples of the same example may or may not share an
order_key — random masking during training makes this benign.

**Random seed = 42** for every sampling step.

---

## 8. Ordering variants (O1 / O2 / O3)

| Variant | order_key | use_sep | Description |
|---------|-----------|---------|-------------|
| **O1** | canonical (tra → trb → peptide → mhc_one → mhc_two) | absent | Fixed deterministic ordering |
| **O2 = M1 default** | random uniform from k_c options | absent | The base sampling case |
| **O3** | random uniform from k_c options | `True` | Random order with separator-tokens flag |

O1 and O3 reuse M1's bio_hash selections — only the order_key (O1) or the
`use_sep` column (O3) changes. This makes them tightly comparable to M1.

**Canonical order**: `tra → trb → peptide → mhc_one → mhc_two`, skipping
absent components. So `tra_trb_peptide_mhc_one` → `tra_trb_peptide_mhc_one`,
`peptide_mhc_two` → `peptide_mhc_two`, `trb_peptide_mhc_one_mhc_two` →
`trb_peptide_mhc_one_mhc_two`.

---

## 9. Output schema

### Manifest files (compact, no full sequences)

`foundation_{10M,100M,500M}.parquet`, `mini_M2_10M.parquet`,
`mini_M3_10M.parquet`, `mini_O1_10M.parquet`:

| Column | Type | Notes |
|--------|------|-------|
| `bio_hash` | string(32) | MD5 hex of identity tuple |
| `order_key` | string | Permutation key from source partition path |
| `subset_key` | string | One of 27 in-scope subset_keys |
| `source_file` | string | Absolute path to source parquet file |
| `source_row_index` | int64 | 0-indexed row offset within source_file |

`mini_O3_10M.parquet` adds `use_sep: bool = True`.

### Full-data files

`foundation_val.parquet`, `foundation_test.parquet`:

All 20 columns from the enriched stage + `mhc_one_allele, mhc_two_allele,
bio_hash, partition, source_file, source_row_index`.

### Auxiliary files

- `downstream_test_exclusion.parquet` — `bio_hash, source_partitions[]`
- `downstream_test_exclusion.txt` — one hash per line
- `partition_assignments.parquet` — `bio_hash, subset_key, order_key,
  source_file, source_row_index, mhc_one_allele, mhc_two_allele, partition`
- `foundation_summary.json` — counts, weights, exposure, timings
- `timings.json` — per-task wall-clock seconds

---

## 10. Resolving manifests at tokenization time

Pipelines that consume a manifest read each row's `(source_file,
source_row_index)` and pull the full enriched row. Pattern:

```python
import pyarrow.parquet as pq
import duckdb

# Group manifest rows by source_file (rare — most rows share files)
manifest = pq.read_table("foundation/foundation_500M.parquet")
by_file = manifest.to_pandas().groupby("source_file")

con = duckdb.connect()
for source_file, group in by_file:
    indices = group["source_row_index"].tolist()
    rows = con.execute(f"""
        SELECT *, ROW_NUMBER() OVER () - 1 AS rid
        FROM read_parquet('{source_file}')
        WHERE rid IN {tuple(indices)}
    """).fetch_arrow_table()
    # tokenize + emit
```

For the val/test files there's no resolution step — they already carry full
row data.

---

## 11. Reproducibility

- **DuckDB**: 1.5.1
- **pyarrow**: repo env
- **numpy**: repo env
- **Python**: 3.13 (miniforge)
- **Random seed**: 42 — for every sampling, shuffle, partitioning step

Commands:

```bash
# Full pipeline
python scripts/data_processing/build_benchmark_foundation.py --task all

# Or per-step (each is independently re-runnable)
python scripts/data_processing/build_benchmark_foundation.py --task exclusion
python scripts/data_processing/build_benchmark_foundation.py --task partition
python scripts/data_processing/build_benchmark_foundation.py --task val_test
python scripts/data_processing/build_benchmark_foundation.py --task scaled
python scripts/data_processing/build_benchmark_foundation.py --task minis_weight
python scripts/data_processing/build_benchmark_foundation.py --task minis_order
python scripts/data_processing/build_benchmark_foundation.py --task summary
```

Re-running with existing outputs overwrites them; delete manually before
re-running if you want to preserve a prior generation.

---

## 12. Known deviations from spec

- **bio_hash uses MD5** (DuckDB-native, 32-char hex), not BLAKE2b-8 like
  Phase 2's `row_id`. The two have different purposes (per-row vs
  per-biological-identity); MD5 was chosen for speed at 1.4B-row scale and
  deterministic SQL evaluation.
- **`peptide_mhc_one_mhc_two`** is in scope (3 molecules: peptide + 2 MHCs)
  even though it has no TCR. The spec includes it in the 3-molecule list.
- **trb size cap of 100M** for the sqrt weighting only, applied to subset
  `trb` specifically. All other subsets use the full N_c.
