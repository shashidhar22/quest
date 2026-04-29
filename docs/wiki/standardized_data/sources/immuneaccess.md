# immuneACCESS (standardized)

## Identity
- **Output path**: `data/standardized_again/immuneaccess/`
- **Standardizer**: [`scripts/data_processing/standardize/immuneaccess.py`](../../../../scripts/data_processing/standardize/immuneaccess.py) (`ImmuneaccessStandardizer`)
- **Raw input**: see [raw-data manifest entry](../../raw_data/databases/immuneaccess.md) for citation, fidelity, and source provenance
- **Generated**: 2026-03-27T11:50:02Z
- **Standardization wall time**: 43,868.1s (~12.2 h, parallel multi-process)

## What this standardizer does

immuneACCESS is the second-largest source by row count (~3.5B raw rows). The standardizer is `streaming=True` with `parallel_run`. It walks `*.tsv` files recursively (`immuneaccess.py:41-42`), determines chain type from the filename / parent directory (`bulk_survey_tra` / `bulk_survey_trb` / `_TCRA` / `_TCRB`; `immuneaccess.py:121-135` — defaults to TRB with a warn-once log on ambiguous paths), then reads each file in 500K-row chunks (`immuneaccess.py:73-82`). Only the columns it needs are loaded via `usecols` (`immuneaccess.py:71-79`). Productive filtering uses `frame_type == "in"` if available, else `productive` boolean (`immuneaccess.py:83-92`). When `v_gene` / `d_gene` / `j_gene` are missing the `v_resolved` / `d_resolved` / `j_resolved` columns are used as a fallback (`immuneaccess.py:97-103`). `study_id` is the file stem.

**Column populations** (which TARGET_COLUMNS this standardizer fills meaningfully):
- `tra` or `trb` (`amino_acid` → CDR3)
- `trav_gene` / `trbv_gene`, `trad_gene` / `trbd_gene`, `traj_gene` / `trbj_gene`
- `source = "immuneaccess"`, `study_id` = file stem (e.g., `001-001-AH`)
- CDR1/CDR2 + `tra_full`/`trb_full` populated by downstream `enrich_cdr_columns()`
- No peptide, MHC, binding, or score

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_standardized_manifest.py) -->

- **Path**: `data/standardized_again/immuneaccess`
- **Standardized at**: 2026-03-27T11:50:02.333277+00:00
- **Rows (manifest)**: 2,768,339,213
- **Dropped**: 1,064,722,302 (27.78% drop rate)
- **Standardization elapsed**: 43,868.1s
- **Source files (checksummed)**: 29233
- **Parquet**: 2769 files, 139.6G (149,903,051,802 bytes), 2,768,339,213 rows (matches manifest)

**Top 8 most-populated columns:**

| Column | Populated | Coverage | Unique |
|--------|-----------|----------|--------|
| `source` | 2,768,339,213 | 100.00% | n/a |
| `study_id` | 2,768,339,213 | 100.00% | 419 (sampled from 30/2769 shards) |
| `trb` | 2,673,933,200 | 96.59% | >1000000 |
| `trb_cdr3` | 2,673,933,200 | 96.59% | n/a |
| `trbj_gene` | 2,670,345,396 | 96.46% | n/a |
| `trbv_gene` | 2,250,924,811 | 81.31% | n/a |
| `trb_full` | 2,246,186,899 | 81.14% | n/a |
| `trb_cdr1` | 2,229,275,138 | 80.53% | n/a |

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
| `invalid_trbd_gene` | 596,553,790 | 56.03% |
| `invalid_trbv_gene` | 422,435,461 | 39.68% |
| `invalid_trad_gene` | 33,734,824 | 3.17% |
| `no_valid_field` | 6,194,621 | 0.58% |
| `invalid_trbj_gene` | 4,054,692 | 0.38% |
| `invalid_traj_gene` | 1,381,060 | 0.13% |
| `invalid_trav_gene` | 185,990 | 0.02% |
| `invalid_tra` | 128,722 | 0.01% |
| `invalid_trb` | 53,142 | 0.00% |

<!-- END: AUTO-INVENTORY -->

## Validation drops

1.06B rows dropped (~38% of input). The dropped file is 58 GB (we sampled the first 5M rows). Top reasons in the sample: `invalid_trbd_gene` (~2.7M of 5M), `invalid_trbv_gene` (~1.7M), `no_valid_field` (~486K), `invalid_trad_gene`, `invalid_trbj_gene`, plus smaller `invalid_tra` / `invalid_trb` counts.

The high `invalid_trbd_gene` / `invalid_trbv_gene` rate reflects immunoSEQ's `_resolved` columns frequently containing `unresolved`, `TRBV20-1/TRBV20-2` ambiguity strings, or `TCRBV` (Adaptive's older nomenclature) that fail IMGT canonicalisation. Many rows still survive because only one chain field has to be valid for the row to pass `no_valid_field`.

The Quantifier auto-inventory above shows the per-reason counts. The `dropped.tsv` file has columns `reason | source_file | row_index | field | raw_value` — useful for debugging upstream data-quality issues.

## Caveats specific to the standardized output

- Per-row `study_id` granularity is the input filename, not the cohort/study. Downstream, multiple `study_id` values typically belong to one biological cohort (e.g., the `FHCRC-Warren-Updated_datasets/*` family).
- 2,769 parquet shards — glob the directory; reading individual files yields heterogeneous slices.
- The "warn once per parent dir if chain unknown" default-to-TRB path means a directory with non-conventional naming silently produces all-beta output. Spot-check `study_id` distributions if a particular study returned no alpha rows.
- The 58 GB `dropped.tsv` is too large for casual `awk`/`grep` — use `head -N` slices when investigating drop reasons.
- BCR contamination: any row with V/D/J starting `IG*` has its CDR3 + genes blanked at the standardization-core layer (`standardization.py:836-869`); the row is then dropped as `no_valid_field` if it had nothing else.

## Use in QUEST training

immuneACCESS is bulk repertoire data per the integration framework — it feeds MLM and TCR-only training (`tra`, `trb`, `tra_full`, `trb_full`). It is **not** used for TCR–pMHC interaction training (no epitope labels). See [raw-data immuneACCESS page](../../raw_data/databases/immuneaccess.md) for the fidelity tier.
