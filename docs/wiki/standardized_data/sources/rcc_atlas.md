# RCC_ATLAS (standardized)

## Identity
- **Output path**: `data/standardized_again/rcc_atlas/`
- **Standardizer**: [`scripts/data_processing/standardize/rcc_atlas.py`](../../../../scripts/data_processing/standardize/rcc_atlas.py) (`RccAtlasStandardizer`)
- **Raw input**: see [raw-data manifest entry](../../raw_data/databases/rcc_atlas.md) for citation, fidelity, and source provenance
- **Generated**: 2026-03-26T19:12:18Z
- **Standardization wall time**: 0.1s

## What this standardizer does

Reads the single CSV `rcc_tcells.csv` (`rcc_atlas.py:46`). The column map is direct (`rcc_atlas.py:32-42`): `CDR3A` → `tra`, `CDR3B` → `trb`, `TRAV` → `trav_gene`, `TRBV` → `trbv_gene`, `epitope` → `peptide`. The `MHC` column is split into `mhc_one` / `mhc_two` row-by-row via `split_mhc_to_alpha_beta` (`rcc_atlas.py:48-59`). Because every entry in this manually-curated literature index is a positive TCR–epitope association, `binding` is hard-coded to `"pos"` for all rows (`rcc_atlas.py:64-65`).

**Column populations** (which TARGET_COLUMNS this standardizer fills meaningfully):
- `tra`, `trb`
- `trav_gene`, `trbv_gene` (only V genes are in the source; `trad_gene`/`traj_gene`/`trbd_gene`/`trbj_gene` are empty)
- `peptide`, `mhc_one`, `mhc_two`, `mhc_one_allele`, `mhc_two_allele`
- `binding = "pos"` (uniformly)
- `source = "rcc_atlas"`, `study_id` empty
- No `score`. CDR1/CDR2 populated when V-gene resolves; `tra_full`/`trb_full` populated only if both V and J resolve — usually empty here because J genes are missing.

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_standardized_manifest.py) -->

- **Path**: `data/standardized_again/rcc_atlas`
- **Standardized at**: 2026-03-26T19:12:18.197106+00:00
- **Rows (manifest)**: 82
- **Dropped**: 6 (6.82% drop rate)
- **Standardization elapsed**: 0.1s
- **Source files (checksummed)**: 1
- **Parquet**: 1 files, 15.6K (15,948 bytes), 82 rows (matches manifest)

**Top 8 most-populated columns:**

| Column | Populated | Coverage | Unique |
|--------|-----------|----------|--------|
| `binding` | 82 | 100.00% | n/a |
| `source` | 82 | 100.00% | n/a |
| `peptide` | 76 | 92.68% | 27 |
| `mhc_one` | 75 | 91.46% | n/a |
| `mhc_one_allele` | 75 | 91.46% | 4 |
| `trb` | 67 | 81.71% | 45 |
| `trb_cdr3` | 67 | 81.71% | n/a |
| `tra` | 7 | 8.54% | 7 |

**Bottom 5 least-populated columns:**

| Column | Populated | Coverage |
|--------|-----------|----------|
| `trb_full` | 0 | 0.00% |
| `mhc_two` | 0 | 0.00% |
| `mhc_two_allele` | 0 | 0.00% |
| `score` | 0 | 0.00% |
| `study_id` | 0 | 0.00% |

**Top 10 drop reasons:**

| Reason | Count | Share |
|--------|-------|-------|
| `invalid_peptide` | 6 | 100.00% |

<!-- END: AUTO-INVENTORY -->

## Validation drops

6 of 88 rows dropped — all `invalid_peptide` (likely multi-token antigen names that aren't pure AA strings). Drop rate is ~7% but the absolute count is tiny.

The Quantifier auto-inventory above shows the per-reason counts. The `dropped.tsv` file has columns `reason | source_file | row_index | field | raw_value` — useful for debugging upstream data-quality issues.

## Caveats specific to the standardized output

- 82 emitted rows total — this is a **literature-curated** atlas, not a bulk dataset. Use it as a high-confidence positive-only seed set, not as a primary training corpus.
- J genes are absent; `tra_full` / `trb_full` will not reconstruct from V alone. CDR1 / CDR2 do populate from V-gene metadata.
- `binding = "pos"` is unconditionally set; there are no negative examples.
- `study_id` is empty — to trace a record's source publication, join on the raw CSV's `reference` / `note` / `clone_id` columns.

## Use in QUEST training

RCC_ATLAS is curated literature TCR–pMHC data with confirmed positive associations covering RCC tumor antigens (NY-ESO-1, WT1, PRAME) and viral antigens (CMV, EBV, Influenza, SARS-CoV-2). It feeds MLM and interaction-training modes (contrastive, cross-encoder, seq2seq) as small but high-fidelity positives. See [raw-data RCC_ATLAS page](../../raw_data/databases/rcc_atlas.md) for the fidelity tier.
