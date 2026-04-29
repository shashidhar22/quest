# CEDAR (standardized)

## Identity
- **Output path**: `data/standardized_again/cedar/`
- **Standardizer**: [`scripts/data_processing/standardize/cedar.py`](../../../../scripts/data_processing/standardize/cedar.py) (`CedarStandardizer`)
- **Raw input**: see [raw-data manifest entry](../../raw_data/databases/cedar.md) for citation, fidelity, and source provenance
- **Generated**: 2026-03-26T19:12:35Z
- **Standardization wall time**: 4.2s

## What this standardizer does

CEDAR ships CSVs with two-row headers (`Group / Field`). The standardizer flattens these via `_flatten_header` (`cedar.py:30-55`) and reads the receptor file `receptor/tcr_full_v3.csv` (`cedar.py:108-115`). Column discovery is by case-insensitive substring matching on the flattened names (`cedar.py:122-149`): `chain 1` columns map to alpha (`tra`, `trav_gene`, …) and `chain 2` columns map to beta. Curated CDR3 fields are preferred; calculated CDR3 is the fallback (`cedar.py:142-149`). Epitope and MHC are pulled directly from the receptor row (no tcell join, since CEDAR receptor rows already carry epitope/MHC), and the MHC string is split via `split_mhc_to_alpha_beta` (`cedar.py:166-175`).

**Column populations** (which TARGET_COLUMNS this standardizer fills meaningfully):
- `tra`, `trav_gene`, `trad_gene`, `traj_gene`
- `trb`, `trbv_gene`, `trbd_gene`, `trbj_gene`
- `peptide`, `mhc_one`, `mhc_two`
- CDR1/CDR2 + `tra_full`/`trb_full` populated by downstream `enrich_cdr_columns()`
- `source = "cedar"`, `study_id` empty
- `binding` is **not** populated (CEDAR receptor file has no qualitative outcome column)

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_standardized_manifest.py) -->

- **Path**: `data/standardized_again/cedar`
- **Standardized at**: 2026-03-26T19:12:35.312511+00:00
- **Rows (manifest)**: 105,825
- **Dropped**: 385 (0.36% drop rate)
- **Standardization elapsed**: 4.2s
- **Source files (checksummed)**: 12
- **Parquet**: 1 files, 5.2M (5,479,789 bytes), 105,825 rows (matches manifest)

**Top 8 most-populated columns:**

| Column | Populated | Coverage | Unique |
|--------|-----------|----------|--------|
| `source` | 105,825 | 100.00% | n/a |
| `peptide` | 105,632 | 99.82% | 1,793 |
| `mhc_one` | 100,606 | 95.07% | n/a |
| `mhc_one_allele` | 100,606 | 95.07% | 56 |
| `trb` | 80,598 | 76.16% | 56,906 |
| `trb_cdr3` | 80,598 | 76.16% | n/a |
| `trbv_gene` | 59,711 | 56.42% | n/a |
| `trbj_gene` | 57,473 | 54.31% | n/a |

**Bottom 5 least-populated columns:**

| Column | Populated | Coverage |
|--------|-----------|----------|
| `mhc_two_allele` | 1,358 | 1.28% |
| `trad_gene` | 0 | 0.00% |
| `binding` | 0 | 0.00% |
| `score` | 0 | 0.00% |
| `study_id` | 0 | 0.00% |

**Top 10 drop reasons:**

| Reason | Count | Share |
|--------|-------|-------|
| `invalid_peptide` | 213 | 55.32% |
| `invalid_tra` | 80 | 20.78% |
| `invalid_trb` | 71 | 18.44% |
| `no_valid_field` | 20 | 5.19% |
| `invalid_trbv_gene` | 1 | 0.26% |

<!-- END: AUTO-INVENTORY -->

## Validation drops

385 of 106,210 rows dropped: 213 `invalid_peptide` (epitope strings with non-AA characters), 80 `invalid_tra` and 71 `invalid_trb` (CDR3 length < 4 or invalid AA), 20 `no_valid_field` (rows where everything except `source` is empty after normalization), and 1 `invalid_trbv_gene`. The drop rate (~0.36%) is consistent with a curated source.

The Quantifier auto-inventory above shows the per-reason counts. The `dropped.tsv` file has columns `reason | source_file | row_index | field | raw_value` — useful for debugging upstream data-quality issues.

## Caveats specific to the standardized output

- The `tcell/tcell_full_v3.csv` file is not joined for CEDAR — IDs do not overlap (per the docstring, `cedar.py:9-12`). Binding-outcome enrichment is therefore not performed.
- All emitted records leave `binding` empty even though CEDAR is a curated database. Downstream consumers that filter on `binding == "pos"` will drop CEDAR entirely; use `source == "cedar"` instead.
- CEDAR overlaps heavily with IEDB at the receptor-record level. Cross-source dedup happens in the next pipeline stage (`scripts/data_processing/deduplicate_streaming.py`), but here both buckets emit independently.
- Peptide-MHC-only ligand data from CEDAR lives in the sibling `cedar_pmhc` bucket, not here.

## Use in QUEST training

CEDAR is curated TCR–pMHC literature data; it feeds MLM, contrastive, cross-encoder and seq2seq interaction training. Because `binding` is empty, downstream loss formulations should treat CEDAR rows as implicit positives (the receptor-row epitope is the intended cognate epitope). See [raw-data CEDAR page](../../raw_data/databases/cedar.md) for the fidelity tier.
