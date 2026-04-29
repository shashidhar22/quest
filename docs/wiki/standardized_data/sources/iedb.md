# IEDB (standardized)

## Identity
- **Output path**: `data/standardized_again/iedb/`
- **Standardizer**: [`scripts/data_processing/standardize/iedb.py`](../../../../scripts/data_processing/standardize/iedb.py) (`IedbStandardizer`)
- **Raw input**: see [raw-data manifest entry](../../raw_data/databases/iedb.md) for citation, fidelity, and source provenance
- **Generated**: 2026-03-26T19:13:06Z
- **Standardization wall time**: 47.5s

## What this standardizer does

IEDB shares the multi-row-header CSV layout with CEDAR (`_read_cedar_csv` is reused, `iedb.py:27-30`). The receptor file `receptor/tcr_full_v3.csv` is parsed and filtered to drop non-TCR receptor types (`bcr`, `ig`, `immunoglobulin`; `iedb.py:65-74`). CDR3 + V/D/J columns are matched by `chain 1 / chain 2 ... curated` substrings, with calculated as fallback (`iedb.py:79-105`). Peptide and MHC are taken from the receptor row first, then the standardizer joins to `tcell/tcell_full_v3.csv` on numeric assay IDs to enrich `binding` (qualitative measurement) and back-fill missing peptide/MHC (`iedb.py:138-247`). The tcell side is filtered to positive outcomes (`iedb.py:168-172`) and human organism (`iedb.py:174-192`). MHC is split via `split_mhc_to_alpha_beta`.

**Column populations** (which TARGET_COLUMNS this standardizer fills meaningfully):
- `tra`, `trav_gene`, `trad_gene`, `traj_gene`
- `trb`, `trbv_gene`, `trbd_gene`, `trbj_gene`
- `peptide`, `mhc_one`, `mhc_two`, `mhc_one_allele`, `mhc_two_allele`
- `binding` (from joined tcell qualitative measurement)
- CDR1/CDR2 + `tra_full`/`trb_full` populated by downstream `enrich_cdr_columns()`
- `source = "iedb"`, `study_id` empty
- `score` empty (no numeric measurement propagated)

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_standardized_manifest.py) -->

- **Path**: `data/standardized_again/iedb`
- **Standardized at**: 2026-03-26T19:13:06.370114+00:00
- **Rows (manifest)**: 226,270
- **Dropped**: 2,270 (0.99% drop rate)
- **Standardization elapsed**: 47.5s
- **Source files (checksummed)**: 12
- **Parquet**: 1 files, 11.2M (11,742,431 bytes), 226,270 rows (matches manifest)

**Top 8 most-populated columns:**

| Column | Populated | Coverage | Unique |
|--------|-----------|----------|--------|
| `source` | 226,270 | 100.00% | n/a |
| `peptide` | 224,226 | 99.10% | 2,880 |
| `trb` | 193,419 | 85.48% | 156,820 |
| `trb_cdr3` | 193,419 | 85.48% | n/a |
| `trbv_gene` | 171,349 | 75.73% | n/a |
| `trbj_gene` | 168,621 | 74.52% | n/a |
| `trb_full` | 165,838 | 73.29% | n/a |
| `trb_cdr1` | 147,820 | 65.33% | n/a |

**Bottom 5 least-populated columns:**

| Column | Populated | Coverage |
|--------|-----------|----------|
| `mhc_two` | 5,530 | 2.44% |
| `mhc_two_allele` | 5,530 | 2.44% |
| `trad_gene` | 1 | 0.00% |
| `score` | 0 | 0.00% |
| `study_id` | 0 | 0.00% |

**Top 10 drop reasons:**

| Reason | Count | Share |
|--------|-------|-------|
| `invalid_peptide` | 2,076 | 91.45% |
| `invalid_tra` | 81 | 3.57% |
| `invalid_trb` | 71 | 3.13% |
| `no_valid_field` | 32 | 1.41% |
| `invalid_trbv_gene` | 6 | 0.26% |
| `invalid_trav_gene` | 1 | 0.04% |
| `invalid_traj_gene` | 1 | 0.04% |
| `invalid_trbd_gene` | 1 | 0.04% |
| `invalid_trbj_gene` | 1 | 0.04% |

<!-- END: AUTO-INVENTORY -->

## Validation drops

2,270 of 228,540 rows dropped (~1.0%): 2,076 `invalid_peptide` (epitope strings with non-AA characters / modifications), 81 `invalid_tra`, 71 `invalid_trb` (CDR3 length < 4 or invalid AA), 32 `no_valid_field`, plus a handful of invalid V/D/J gene names. No `mhc_unresolved` from this bucket — alleles are uniformly well-formed HLA strings.

The Quantifier auto-inventory above shows the per-reason counts. The `dropped.tsv` file has columns `reason | source_file | row_index | field | raw_value` — useful for debugging upstream data-quality issues.

## Caveats specific to the standardized output

- The receptor↔tcell join uses comma-split assay IDs in the receptor file; rows whose assay IDs don't match any tcell row keep the receptor-row epitope/MHC and end up with empty `binding` (`iedb.py:226-227`).
- Cross-source overlap with VDJdb, McPAS, and CEDAR is substantial (IEDB is upstream of all three). Dedup happens in the next pipeline stage.
- IEDB pMHC-only data (mhc_ligand + mhc_bind) is in the sibling `iedb_pmhc` bucket, not here.
- `binding` may be empty even on positive rows if the tcell join missed (e.g., the receptor-row assay IDs are not all in the tcell file post-positive filter).

## Use in QUEST training

IEDB is curated literature TCR–pMHC data; it feeds MLM and all interaction-training modes (contrastive, cross-encoder, seq2seq). Records with `binding == "pos"` are usable for binary classification; rows with empty `binding` are still usable as implicit positives (the receptor row carries the cognate epitope by design). See [raw-data IEDB page](../../raw_data/databases/iedb.md) for the fidelity tier.
