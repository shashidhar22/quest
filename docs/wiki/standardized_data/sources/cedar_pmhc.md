# CEDAR pMHC (standardized)

## Identity
- **Output path**: `data/standardized_again/cedar_pmhc/`
- **Standardizer**: [`scripts/data_processing/standardize/cedar_pmhc.py`](../../../../scripts/data_processing/standardize/cedar_pmhc.py) (`CedarPmhcStandardizer`)
- **Raw input**: see [raw-data manifest entry](../../raw_data/databases/cedar.md) for citation, fidelity, and source provenance
- **Generated**: 2026-03-26T19:18:18Z
- **Standardization wall time**: 347.3s

## What this standardizer does

Reads `mhc_ligand/mhc_ligand_full_v3.csv` (`cedar_pmhc.py:179`). The standardizer auto-detects two CSV layouts (`_is_api_format`, `cedar_pmhc.py:38-42`): the API-export form uses flat `__`-delimited headers (`epitope__name`, `mhc_restriction__name`, `host__name`, `assay__qualitative_measurement`, `epitope__object_type`); the bulk-download form uses two-row headers parsed via `_read_cedar_csv` from the sibling CEDAR module. In both branches the data is filtered to human host (`cedar_pmhc.py:67-74`) and to `linear peptide` epitope objects (`cedar_pmhc.py:76-83`). MHC is split into `mhc_one` / `mhc_two`. For API rows, `binding` is `pos`/`neg`/empty based on `assay__qualitative_measurement` (`cedar_pmhc.py:97-105`); for bulk rows (mass-spec elution) all records are positive (`cedar_pmhc.py:162`).

**Column populations** (which TARGET_COLUMNS this standardizer fills meaningfully):
- `peptide`
- `mhc_one`, `mhc_two`, plus `mhc_one_allele`/`mhc_two_allele` from the standardization core
- `binding` (`pos`/`neg`/empty per `assay__qualitative_measurement`; bulk elution = all `pos`)
- `source = "cedar_pmhc"`, `study_id` empty
- No TCR fields (this bucket is peptide-MHC only)

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_standardized_manifest.py) -->

- **Path**: `data/standardized_again/cedar_pmhc`
- **Standardized at**: 2026-03-26T19:18:18.390478+00:00
- **Rows (manifest)**: 4,316,254
- **Dropped**: 224,804 (4.95% drop rate)
- **Standardization elapsed**: 347.3s
- **Source files (checksummed)**: 12
- **Parquet**: 1 files, 31.5M (33,067,416 bytes), 4,316,254 rows (matches manifest)

**Top 8 most-populated columns:**

| Column | Populated | Coverage | Unique |
|--------|-----------|----------|--------|
| `binding` | 4,316,254 | 100.00% | n/a |
| `source` | 4,316,254 | 100.00% | n/a |
| `peptide` | 4,092,363 | 94.81% | >1000000 |
| `mhc_one_allele` | 1,549,451 | 35.90% | 230 |
| `mhc_one` | 1,548,538 | 35.88% | n/a |
| `mhc_two` | 393,619 | 9.12% | n/a |
| `mhc_two_allele` | 393,619 | 9.12% | n/a |

**Bottom 5 least-populated columns:**

| Column | Populated | Coverage |
|--------|-----------|----------|
| `trb_cdr2` | 0 | 0.00% |
| `trb_cdr3` | 0 | 0.00% |
| `trb_full` | 0 | 0.00% |
| `score` | 0 | 0.00% |
| `study_id` | 0 | 0.00% |

**Top 10 drop reasons:**

| Reason | Count | Share |
|--------|-------|-------|
| `invalid_peptide` | 223,891 | 99.59% |
| `mhc_unresolved` | 913 | 0.41% |

<!-- END: AUTO-INVENTORY -->

## Validation drops

Of ~4.54M ingested rows, 224,804 (5.0%) were dropped: 223,891 `invalid_peptide` (most likely non-linear / discontinuous epitopes that survived the `linear peptide` filter, or non-AA characters such as modification annotations), and 913 `mhc_unresolved` (MHC strings whose normalized allele could not be resolved to a sequence — typically non-human or partially-typed alleles).

The Quantifier auto-inventory above shows the per-reason counts. The `dropped.tsv` file has columns `reason | source_file | row_index | field | raw_value` — useful for debugging upstream data-quality issues.

## Caveats specific to the standardized output

- This bucket overlaps semantically with `iedb_pmhc`; CEDAR and IEDB share infrastructure and records cross-flow between them. Cross-source dedup happens in the next pipeline stage.
- For mass-spec eluted ligands all records get `binding = "pos"` regardless of any quantitative score — there is no IC50 / Kd field carried through.
- `score` is always empty (no quantitative measurement is propagated, even when present in the input).
- The detected-format branch is a one-shot heuristic at the top of the file; if a future CEDAR export changes header conventions both branches may silently produce the bulk fallback.

## Use in QUEST training

CEDAR pMHC is experimental peptide-MHC binding evidence with no TCR information — per the integration framework it feeds peptide-MHC-only training (pMHC seq2seq, contrastive pMHC). It does **not** feed TCR interaction training. See [raw-data CEDAR page](../../raw_data/databases/cedar.md) for the fidelity tier.
