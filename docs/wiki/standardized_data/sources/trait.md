# TRAIT (standardized)

## Identity
- **Output path**: `data/standardized_again/trait/`
- **Standardizer**: [`scripts/data_processing/standardize/trait.py`](../../../../scripts/data_processing/standardize/trait.py) (`TraitStandardizer`)
- **Raw input**: see [raw-data manifest entry](../../raw_data/databases/trait.md) for citation, fidelity, and source provenance
- **Generated**: 2026-03-26T19:24:12Z
- **Standardization wall time**: 67.7s

## What this standardizer does

TRAIT encodes per-epitope metadata in **filenames**, not columns. Each input is a ZIP whose name follows the pattern `{MHC}_{epitope}_{protein}_{disease}_binder_{pos|neg}.zip` (e.g., `A0201_GLCTLVAML_BMLF1_EBV_binder_pos.zip`). `_parse_trait_filename` (`trait.py:34-70`) splits on `_`, converts the compact MHC encoding (`A0201` → `HLA-A*02:01` via regex; `trait.py:50-54`) and reads `pos` / `neg` to set `binding`. There are two ZIP discovery paths: per-epitope ZIPs in `epitopes/binding/` and `epitopes/non_binding/` (`trait.py:144-166`), plus nested ZIPs in `main/Omics.zip` (`trait.py:169-186`).

Each inner file's columns are normalised to standard names by `_map_columns` (`trait.py:90-118`): `cdr3b` / `cdr3_beta` / `cdr3_b` → `trb`, `cdr3a` / `cdr3_alpha` → `tra`, V/J gene aliases (`vb`, `jb`, `va`, `ja`) → corresponding gene columns. A bare `cdr3` column defaults to `trb` (`trait.py:108-111`). The `peptide`, `mhc_one`, `binding` are then added to the row from the filename metadata (`trait.py:113-116`).

**Column populations** (which TARGET_COLUMNS this standardizer fills meaningfully):
- `tra`, `trav_gene`, `traj_gene` (when the inner file carries alpha columns)
- `trb`, `trbv_gene`, `trbj_gene`
- `peptide` (from filename), `mhc_one` (from filename), `mhc_one_allele`
- `binding = "pos"` or `"neg"` (from filename `_binder_pos` / `_binder_neg`)
- `mhc_two` always empty (TRAIT is class-I only)
- `source = "trait"`, `study_id` = ZIP stem (e.g., `A0101_VTEHDTLLY_IE-1_CMV_binder_pos`)
- No `score`. CDR1/CDR2 + `trb_full`/`tra_full` populated downstream when V/J resolve.

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_standardized_manifest.py) -->

- **Path**: `data/standardized_again/trait`
- **Standardized at**: 2026-03-26T19:24:12.325753+00:00
- **Rows (manifest)**: 3,967,323
- **Dropped**: 177 (0.00% drop rate)
- **Standardization elapsed**: 67.7s
- **Source files (checksummed)**: 36
- **Parquet**: 4 files, 335.3M (351,629,225 bytes), 3,967,323 rows (matches manifest)

**Top 8 most-populated columns:**

| Column | Populated | Coverage | Unique |
|--------|-----------|----------|--------|
| `trb` | 3,967,323 | 100.00% | 59,677 |
| `trbv_gene` | 3,967,323 | 100.00% | n/a |
| `trbj_gene` | 3,967,323 | 100.00% | n/a |
| `trb_cdr3` | 3,967,323 | 100.00% | n/a |
| `trb_full` | 3,967,323 | 100.00% | n/a |
| `peptide` | 3,967,323 | 100.00% | 50 |
| `mhc_one` | 3,967,323 | 100.00% | n/a |
| `mhc_one_allele` | 3,967,323 | 100.00% | 8 |

**Bottom 5 least-populated columns:**

| Column | Populated | Coverage |
|--------|-----------|----------|
| `trad_gene` | 0 | 0.00% |
| `trbd_gene` | 0 | 0.00% |
| `mhc_two` | 0 | 0.00% |
| `mhc_two_allele` | 0 | 0.00% |
| `score` | 0 | 0.00% |

**Top 10 drop reasons:**

| Reason | Count | Share |
|--------|-------|-------|
| `invalid_tra` | 177 | 100.00% |

<!-- END: AUTO-INVENTORY -->

## Validation drops

177 of ~3.97M rows dropped (~0.004%) — all `invalid_tra` (a small minority of inner files have alpha CDR3 columns with non-AA characters). Beta CDR3, peptide, and MHC are uniformly clean because peptide and MHC come from filename metadata.

The Quantifier auto-inventory above shows the per-reason counts. The `dropped.tsv` file has columns `reason | source_file | row_index | field | raw_value` — useful for debugging upstream data-quality issues.

## Caveats specific to the standardized output

- Filenames not matching the `MHC_epitope_protein_disease_binder_X` pattern are silently skipped (`trait.py:153-155`). If a future TRAIT release renames files, row counts can drop without an error.
- TRAIT is **class I only**; `mhc_two` is never populated. Class II would require a different filename schema.
- Negative records (`_binder_neg`) carry `binding = "neg"` and are usable for cross-encoder negative training. This is the only large-scale source in QUEST that emits explicit negatives.
- TRAIT aggregates many published studies. Per the integration framework, cross-source dedup against IEDB / VDJdb / McPAS in the next pipeline stage matters here — TRAIT's `study_id` (the ZIP stem) is not the underlying publication ID.
- `score` is empty (no per-record affinity is propagated even when the source CSV has one).
- 4 parquet shards.

## Use in QUEST training

TRAIT is multi-source aggregation with experimentally-validated TCR–pMHC binding (per the integration framework, "Multi-source aggregation with experimentally validated data"). It feeds MLM and all interaction-training modes (contrastive, cross-encoder, seq2seq). The presence of explicit `pos`/`neg` labels makes TRAIT especially valuable for binary cross-encoder training. See [raw-data TRAIT page](../../raw_data/databases/trait.md) for the fidelity tier.
