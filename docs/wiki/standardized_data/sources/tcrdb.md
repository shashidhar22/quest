# TCRdb (standardized)

## Identity
- **Output path**: `data/standardized_again/tcrdb/`
- **Standardizer**: [`scripts/data_processing/standardize/tcrdb.py`](../../../../scripts/data_processing/standardize/tcrdb.py) (`TcrdbStandardizer`)
- **Raw input**: see [raw-data manifest entry](../../raw_data/databases/tcrdb.md) for citation, fidelity, and source provenance
- **Generated**: 2026-03-26T22:42:28Z
- **Standardization wall time**: 4,737.0s

## What this standardizer does

TCRdb is the third-largest source by row count (~316M). Files live under `tcrdb/{category}/{disease}/...` for six fixed categories (`cancer`, `autoimmunity`, `healthy`, `inflammation`, `viral`, `transplantation`; `tcrdb.py:25-33`). The standardizer is `streaming=True` with `parallel_run`. It enumerates `*.csv` and `*.tsv` in those category trees, excluding any file whose name contains `metadata` (`tcrdb.py:48-55`). Each file is read in 1M-row chunks (`tcrdb.py:64-69`).

Default column map is beta (`AASeq` → `trb`, `Vregion` → `trbv_gene`, etc., `tcrdb.py:40-46`). For chain detection: a `chain` column wins if present (`tcrdb.py:73-78`); otherwise the V-gene prefix is inspected — `TRA*` / `TCRA*` → alpha, anything else → beta (`tcrdb.py:124-129`). When alpha rows exist they are split out and re-mapped to alpha column targets (`tcrdb.py:90-104`). `study_id` is the file stem (typically a SRA / DDBJ run accession).

**Column populations** (which TARGET_COLUMNS this standardizer fills meaningfully):
- `trb`, `trbv_gene`, `trbd_gene`, `trbj_gene` (the dominant chain)
- `tra`, `trav_gene`, `trad_gene`, `traj_gene` for the small subset of files that flag alpha
- CDR1/CDR2 + `tra_full`/`trb_full` populated downstream
- `source = "tcrdb"`, `study_id` = file stem
- No peptide, MHC, binding, or score

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_standardized_manifest.py) -->

- **Path**: `data/standardized_again/tcrdb`
- **Standardized at**: 2026-03-26T22:42:28.618398+00:00
- **Rows (manifest)**: 316,284,375
- **Dropped**: 67,552,409 (17.60% drop rate)
- **Standardization elapsed**: 4,737.0s
- **Source files (checksummed)**: 9366
- **Parquet**: 317 files, 16.3G (17,523,860,842 bytes), 316,284,375 rows (matches manifest)

**Top 8 most-populated columns:**

| Column | Populated | Coverage | Unique |
|--------|-----------|----------|--------|
| `trb` | 316,284,375 | 100.00% | >1000000 |
| `trbj_gene` | 316,284,375 | 100.00% | n/a |
| `trb_cdr3` | 316,284,375 | 100.00% | n/a |
| `source` | 316,284,375 | 100.00% | n/a |
| `study_id` | 316,284,375 | 100.00% | 1059 (sampled from 30/317 shards) |
| `trbv_gene` | 316,284,343 | 100.00% | n/a |
| `trb_full` | 316,169,631 | 99.96% | n/a |
| `trb_cdr1` | 314,655,603 | 99.49% | n/a |

**Bottom 5 least-populated columns:**

| Column | Populated | Coverage |
|--------|-----------|----------|
| `mhc_two` | 0 | 0.00% |
| `mhc_one_allele` | 0 | 0.00% |
| `mhc_two_allele` | 0 | 0.00% |
| `binding` | 0 | 0.00% |
| `score` | 0 | 0.00% |

**Top 10 drop reasons:**

| Reason | Count | Share |
|--------|-------|-------|
| `invalid_trbd_gene` | 67,552,408 | 100.00% |
| `no_valid_field` | 1 | 0.00% |

<!-- END: AUTO-INVENTORY -->

## Validation drops

67.6M dropped of ~384M input rows (~17.6%). The 5M-row sample is dominated by `invalid_trbd_gene` (5M of 5M), suggesting that most TCRdb files use a non-IMGT D-gene format (likely Adaptive's TCRBD prefix or the literal `unresolved` / `unknown`). The actual drop reason mix in the full file is broader, but the D-gene path dominates the early shards.

The Quantifier auto-inventory above shows the per-reason counts. The `dropped.tsv` file has columns `reason | source_file | row_index | field | raw_value` — useful for debugging upstream data-quality issues.

## Caveats specific to the standardized output

- TCRdb is described in its docstring as "primarily beta-chain only". The alpha-detection branch is defensive; in practice almost all output rows are beta.
- 317 parquet shards.
- `study_id` granularity is per-file (SRA run); biological cohorts span many files. The TCRdb `tcrdb_all_metadata_combined.csv` (intentionally excluded by the metadata filter at `tcrdb.py:55`) maps run → study → disease but is not joined here.
- Drop rate is dominated by D-gene normalization, not CDR3 invalidation — most rows survive with a blank D-gene but a valid `trb`.
- BCR contamination is blanked (any `IG*` V/D/J gets the chain zeroed) rather than dropped; rows with no surviving valid field then fall into `no_valid_field`.

## Use in QUEST training

TCRdb is bulk repertoire — per the integration framework it feeds MLM and TCR-only training (`trb`, `trb_full` predominantly). It does **not** feed TCR–pMHC interaction training (no epitope labels). See [raw-data TCRdb page](../../raw_data/databases/tcrdb.md) for the fidelity tier.
