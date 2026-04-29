# OTS (standardized)

## Identity
- **Output path**: `data/standardized_again/ots/`
- **Standardizer**: [`scripts/data_processing/standardize/ots.py`](../../../../scripts/data_processing/standardize/ots.py) (`OtsStandardizer`)
- **Raw input**: see [raw-data manifest entry](../../raw_data/databases/ots.md) for citation, fidelity, and source provenance
- **Generated**: 2026-03-26T21:15:27Z
- **Standardization wall time**: 1140.8s

## What this standardizer does

OTS (Observed T cell Sequences) ships ~2,166 paired alpha-beta CSVs named `SRR*_1_Paired_All.csv` with a JSON metadata row at the top of each file. The standardizer reads with `skiprows=1` to skip the JSON header (`ots.py:60`), then chunks at 500K rows (`ots.py:55-69`). The column map is one-to-one (`ots.py:34-43`): `cdr3_aa_alpha` → `tra`, `v_call_alpha` → `trav_gene`, `j_call_alpha` → `traj_gene`, `cdr3_aa_beta` → `trb`, `v_call_beta` / `d_call_beta` / `j_call_beta` → corresponding beta gene columns. Every row is a paired alpha-beta record (no chain splitting needed). `study_id` is the SRR ID extracted from the filename stem (`ots.py:54`).

**Column populations** (which TARGET_COLUMNS this standardizer fills meaningfully):
- `tra`, `trav_gene`, `traj_gene` (note: no `trad_gene` — alpha has no D segment)
- `trb`, `trbv_gene`, `trbd_gene`, `trbj_gene`
- CDR1/CDR2 + `tra_full`/`trb_full` populated downstream
- `source = "ots"`, `study_id` = SRR ID
- No peptide, MHC, binding, or score — OTS is bulk paired repertoire only

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_standardized_manifest.py) -->

- **Path**: `data/standardized_again/ots`
- **Standardized at**: 2026-03-26T21:15:27.433240+00:00
- **Rows (manifest)**: 6,968,867
- **Dropped**: 384 (0.01% drop rate)
- **Standardization elapsed**: 1,140.8s
- **Source files (checksummed)**: 2166
- **Parquet**: 7 files, 676.0M (708,839,691 bytes), 6,968,867 rows (matches manifest)

**Top 8 most-populated columns:**

| Column | Populated | Coverage | Unique |
|--------|-----------|----------|--------|
| `trb` | 6,968,867 | 100.00% | >1000000 |
| `trb_cdr3` | 6,968,867 | 100.00% | n/a |
| `source` | 6,968,867 | 100.00% | n/a |
| `study_id` | 6,968,867 | 100.00% | 2,157 |
| `trbv_gene` | 6,968,865 | 100.00% | n/a |
| `trbj_gene` | 6,968,865 | 100.00% | n/a |
| `trav_gene` | 6,968,838 | 100.00% | n/a |
| `traj_gene` | 6,968,834 | 100.00% | n/a |

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
| `invalid_tra` | 384 | 100.00% |

<!-- END: AUTO-INVENTORY -->

## Validation drops

384 of ~6.97M rows dropped (~0.006%) — all `invalid_tra`, suggesting a small number of alpha-chain records with non-AA characters or sub-minimum length. Beta is uniformly clean.

The Quantifier auto-inventory above shows the per-reason counts. The `dropped.tsv` file has columns `reason | source_file | row_index | field | raw_value` — useful for debugging upstream data-quality issues.

## Caveats specific to the standardized output

- Every output row is paired (`tra` and `trb` both populated). This makes OTS an unusually clean source for paired-chain training and contrastive `tra↔trb` learning.
- 7 parquet shards (low for the row count — likely because the 1M-row streaming buffer flushes evenly).
- `study_id` granularity is per-SRR-ID (per-sample). Many SRRs belong to the same biological cohort or BioProject; consult the OTS metadata index for cohort grouping.
- The JSON metadata row (skipped) carries useful per-sample provenance (donor ID, tissue) that is not propagated here. Re-read the raw CSV's first line if you need that context.

## Use in QUEST training

OTS is bulk paired repertoire — per the integration framework it feeds MLM and TCR-only training (`tra`, `trb`, `tra_full`, `trb_full`, plus `tra_trb` paired modes). It does **not** feed pMHC interaction training. See [raw-data OTS page](../../raw_data/databases/ots.md) for the fidelity tier.
