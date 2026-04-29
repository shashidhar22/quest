# TaDB (standardized)

## Identity
- **Output path**: `data/standardized_again/tadb/`
- **Standardizer**: [`scripts/data_processing/standardize/tadb.py`](../../../../scripts/data_processing/standardize/tadb.py) (`TadbStandardizer`)
- **Raw input**: see [raw-data manifest entry](../../raw_data/databases/tadb.md) for citation, fidelity, and source provenance
- **Generated**: 2026-03-26T19:12:12Z
- **Standardization wall time**: 0.5s

## What this standardizer does

TaDB (T cell Assay Database) ships a single CSV (`tadb.py:34-38`). The column map is minimal (`tadb.py:28-31`): `Epitope sequence` → `peptide`. The `HLA allele` column is split into `mhc_one` / `mhc_two` row-by-row via `split_mhc_to_alpha_beta` (`tadb.py:40-49`), then both columns are added to the column map before `standardize_dataframe` is called (`tadb.py:51-58`). No `binding` is set — TaDB does not carry positive/negative outcome columns at this layer of QUEST.

**Column populations** (which TARGET_COLUMNS this standardizer fills meaningfully):
- `peptide`
- `mhc_one`, `mhc_two`, `mhc_one_allele`, `mhc_two_allele`
- `source = "tadb"`, `study_id` empty
- No TCR fields, no `binding`, no `score`

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_standardized_manifest.py) -->

- **Path**: `data/standardized_again/tadb`
- **Standardized at**: 2026-03-26T19:12:12.917376+00:00
- **Rows (manifest)**: 1,147
- **Dropped**: 2 (0.17% drop rate)
- **Standardization elapsed**: 0.5s
- **Source files (checksummed)**: 1
- **Parquet**: 1 files, 32.6K (33,431 bytes), 1,147 rows (matches manifest)

**Top 8 most-populated columns:**

| Column | Populated | Coverage | Unique |
|--------|-----------|----------|--------|
| `peptide` | 1,147 | 100.00% | 909 |
| `source` | 1,147 | 100.00% | n/a |
| `mhc_one_allele` | 505 | 44.03% | 46 |
| `mhc_one` | 504 | 43.94% | n/a |
| `mhc_two_allele` | 162 | 14.12% | n/a |
| `mhc_two` | 161 | 14.04% | n/a |

**Bottom 5 least-populated columns:**

| Column | Populated | Coverage |
|--------|-----------|----------|
| `trb_cdr3` | 0 | 0.00% |
| `trb_full` | 0 | 0.00% |
| `binding` | 0 | 0.00% |
| `score` | 0 | 0.00% |
| `study_id` | 0 | 0.00% |

**Top 10 drop reasons:**

| Reason | Count | Share |
|--------|-------|-------|
| `mhc_unresolved` | 2 | 100.00% |

<!-- END: AUTO-INVENTORY -->

## Validation drops

2 of 1,149 input rows dropped — both `mhc_unresolved` (likely partially-typed alleles that didn't survive normalisation).

The Quantifier auto-inventory above shows the per-reason counts. The `dropped.tsv` file has columns `reason | source_file | row_index | field | raw_value` — useful for debugging upstream data-quality issues.

## Caveats specific to the standardized output

- 1,147 rows total — a tiny epitope-only reference, not a primary training corpus.
- `binding` is uniformly empty even though the records correspond to assayed epitopes (the binding outcome from the source CSV is not propagated). Treat as positives-by-curation if you need a label.
- Per-record provenance (`ACCESSION`, `Epitope type`) from the source CSV is not propagated — to trace back, join on raw `ACCESSION`.
- Heavy semantic overlap with `iedb_pmhc` (TaDB epitopes are largely subsets of IEDB-derived experimental data). Cross-source dedup runs in the next pipeline stage.

## Use in QUEST training

TaDB is curated peptide-MHC reference data with no TCR — per the integration framework it feeds peptide-MHC-only training (pMHC seq2seq, contrastive pMHC). It does not feed TCR interaction training. See [raw-data TaDB page](../../raw_data/databases/tadb.md) for the fidelity tier.
