# BATMAN (standardized)

## Identity
- **Output path**: `data/standardized_again/batman/`
- **Standardizer**: [`scripts/data_processing/standardize/batman.py`](../../../../scripts/data_processing/standardize/batman.py) (`BatmanStandardizer`)
- **Raw input**: see [raw-data manifest entry](../../raw_data/databases/batman.md) for citation, fidelity, and source provenance
- **Generated**: 2026-03-26T19:12:19Z
- **Standardization wall time**: 6.7s

## What this standardizer does

BATMAN's two Excel files (`TCR_pMHCI_mutational_scan_database.xlsx`, `TCR_pMHCII_mutational_scan_database.xlsx`) are concatenated, tagged with `_mhc_class` and `_source_file` (`batman.py:79-93`), then filtered to `tcr_source_organism == "human"` (`batman.py:96-99`). Bare-numeric IMGT gene identifiers like `12-3` and `1*00(80)` are rewritten with the appropriate prefix (`TRAV12-3`, `TRBD1`) and parenthetical scores are stripped (`batman.py:42-60`, applied at `batman.py:101-106`). The MHC string is split into `mhc_one` / `mhc_two` via `split_mhc_to_alpha_beta` (`batman.py:108-122`). The continuous `peptide_activity` field is binarised into `binding`: >= 0.1 → `pos`, < 0.1 → `neg`, else empty (`batman.py:127-132`). `score` is set to the raw `peptide_activity` value.

**Column populations** (which TARGET_COLUMNS this standardizer fills meaningfully):
- `tra` (`cdr3a`), `trb` (`cdr3b`)
- `trav_gene`, `traj_gene`, `trbv_gene`, `trbd_gene`, `trbj_gene` (re-prefixed)
- `peptide`, `mhc_one`, `mhc_two`
- `binding` (derived from `peptide_activity`), `score = peptide_activity`
- `source = "batman"`, `study_id` empty

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_standardized_manifest.py) -->

- **Path**: `data/standardized_again/batman`
- **Standardized at**: 2026-03-26T19:12:19.092355+00:00
- **Rows (manifest)**: 12,731
- **Dropped**: 0 (0.00% drop rate)
- **Standardization elapsed**: 6.7s
- **Source files (checksummed)**: 3
- **Parquet**: 1 files, 190.9K (195,458 bytes), 12,731 rows (matches manifest)

**Top 8 most-populated columns:**

| Column | Populated | Coverage | Unique |
|--------|-----------|----------|--------|
| `peptide` | 12,731 | 100.00% | 3,884 |
| `binding` | 12,731 | 100.00% | n/a |
| `score` | 12,731 | 100.00% | n/a |
| `source` | 12,731 | 100.00% | n/a |
| `tra` | 11,451 | 89.95% | 56 |
| `trav_gene` | 11,451 | 89.95% | n/a |
| `tra_cdr3` | 11,451 | 89.95% | n/a |
| `trb` | 11,451 | 89.95% | 60 |

**Bottom 5 least-populated columns:**

| Column | Populated | Coverage |
|--------|-----------|----------|
| `trbd_gene` | 4,970 | 39.04% |
| `mhc_two` | 1,524 | 11.97% |
| `mhc_two_allele` | 1,524 | 11.97% |
| `trad_gene` | 0 | 0.00% |
| `study_id` | 0 | 0.00% |

<!-- END: AUTO-INVENTORY -->

## Validation drops

Zero rows dropped — BATMAN is small (12,731 rows after concat + human filter), and every row passes the CDR3-length, gene-prefix, and MHC normalization checks. This makes BATMAN a useful regression-test corpus: any future change to `quest/data/standardization.py` that breaks BATMAN ingestion is almost certainly a bug.

The Quantifier auto-inventory above shows the per-reason counts. The `dropped.tsv` file has columns `reason | source_file | row_index | field | raw_value` — useful for debugging upstream data-quality issues.

## Caveats specific to the standardized output

- BATMAN is dense mutational-scan data: many records share the same TCR with different peptide variants. Cross-source dedup against IEDB/VDJdb may collapse the wild-type rows but the mutant peptides remain unique.
- The `binding` threshold (0.1 on `peptide_activity`) is a project decision, not a BATMAN authorial threshold — re-tune in `batman.py:128-131` if you want a different positivity cutoff.
- Non-numeric `peptide_activity` values are silently left with empty `binding` and `score` (the `pd.to_numeric(..., errors="coerce")` path). These rows are kept for MLM but not for binding-classification training.

## Use in QUEST training

BATMAN is experimentally validated quantitative TCR–pMHC binding data — it enters at the high-fidelity tier and contributes to interaction training (contrastive, cross-encoder, seq2seq) for rows with non-empty `binding`. The `score` column carries the raw `peptide_activity` for any future regression-style training. See [raw-data BATMAN page](../../raw_data/databases/batman.md) for the fidelity tier.
