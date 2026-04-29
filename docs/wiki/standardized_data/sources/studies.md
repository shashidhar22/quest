# Studies (standardized)

## Identity
- **Output path**: `data/standardized_again/studies/`
- **Standardizer**: [`scripts/data_processing/standardize/studies.py`](../../../../scripts/data_processing/standardize/studies.py) (`StudiesStandardizer`)
- **Raw input**: see [raw-data manifest entry](../../raw_data/studies/README.md) for the per-study index (~70 GEO accessions, BGI/COBRA/CFAR pilots, ZEN preprints, etc.)
- **Generated**: 2026-04-08T19:45:23Z
- **Standardization wall time**: 2,613.2s

## What this standardizer does

The studies bucket consumes heterogeneous per-study TCR data dropped under `data/studies/{study_id}/`. The standardizer is `streaming=True`, parallel-capable, and **format-detecting**: each input file is auto-classified into one of ~10 schemas (10x, 10x_clonotype, AIRR, immunoSEQ v1/v2, MiTCR, MiXCR clone/full, VJCombo, BGI structure, generic) by inspecting column names (`_detect_format`, `studies.py:134-178`). Per format there's a templated column map (`FORMAT_COLUMN_MAPS`, `studies.py:28-92`) plus alpha-chain variants (`_ALPHA_COLUMN_MAPS`, `studies.py:95-131`). Chain inference (TRA vs TRB) walks: filename pattern → directory pattern → MiTCR header field (`studies.py:232-275`).

Special-case logic includes: 10x Genomics rows paired by cell barcode when a `barcode`/`cell_id` column is present (`_pair_chains_by_cell`, `studies.py:416-462`); 10x clonotype semicolon-encoded chains parsed by `_parse_10x_clonotype_chunk` (`studies.py:278-337`); MiXCR full-export gene-hit strings stripped of scores like `TRBV7-9*00(911.6)` (`studies.py:666-677`); productivity / locus filters with explicit drop tracking (`studies.py:683-735`). Files that match no recognised format are logged to `skipped_files.tsv` (`studies.py:535-545`). `study_id` is the top-level directory name (e.g., `GSE121810`, `BGI_pilot`, `ZEN14010377`).

**Column populations** (which TARGET_COLUMNS this standardizer fills meaningfully):
- `tra`, `trav_gene`, `trad_gene`, `traj_gene` (when alpha rows / paired data are present)
- `trb`, `trbv_gene`, `trbd_gene`, `trbj_gene`
- `peptide` for studies whose schema includes an `epitope`/`antigen`/`peptide` column (rare — most are bulk repertoire)
- CDR1/CDR2 + `tra_full`/`trb_full` populated downstream when V/J genes resolve
- `source = "studies"`, `study_id` = study directory name
- `binding`, `score`, `mhc_one`, `mhc_two` mostly empty

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_standardized_manifest.py) -->

- **Path**: `data/standardized_again/studies`
- **Standardized at**: 2026-04-08T19:45:23.470657+00:00
- **Rows (manifest)**: 92,142,845
- **Dropped**: 15,025,806 (14.02% drop rate)
- **Standardization elapsed**: 2,613.2s
- **Source files (checksummed)**: 6398
- **Parquet**: 93 files, 1.5G (1,629,973,872 bytes), 92,142,845 rows (matches manifest)

**Top 8 most-populated columns:**

| Column | Populated | Coverage | Unique |
|--------|-----------|----------|--------|
| `source` | 92,142,845 | 100.00% | n/a |
| `study_id` | 92,142,845 | 100.00% | 22 (sampled from 30/93 shards) |
| `trbj_gene` | 70,998,373 | 77.05% | n/a |
| `trbv_gene` | 67,473,216 | 73.23% | n/a |
| `trb_cdr1` | 67,102,562 | 72.82% | n/a |
| `trb_cdr2` | 67,102,477 | 72.82% | n/a |
| `trb` | 63,352,930 | 68.76% | >1000000 |
| `trb_cdr3` | 63,284,633 | 68.68% | n/a |

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
| `invalid_trb` | 5,460,052 | 36.34% |
| `invalid_tra` | 3,598,430 | 23.95% |
| `invalid_trbd_gene` | 2,967,028 | 19.75% |
| `invalid_trbv_gene` | 2,721,622 | 18.11% |
| `invalid_trad_gene` | 258,673 | 1.72% |
| `invalid_traj_gene` | 11,720 | 0.08% |
| `invalid_trbj_gene` | 6,066 | 0.04% |
| `invalid_trav_gene` | 1,192 | 0.01% |
| `non_productive` | 992 | 0.01% |
| `no_valid_field` | 31 | 0.00% |

<!-- END: AUTO-INVENTORY -->

## Validation drops

15.0M of ~107M input rows dropped (~14%). Top reasons in a 5M-row sample: `invalid_trb` (~4.07M), `invalid_tra` (~934K), and `non_productive` (~123 — these are the explicit `studies.py:680-735` drop-record entries for productivity filters). Heavy CDR3 invalidation reflects the format heterogeneity: many input files mix nucleotide and AA columns or carry placeholder strings like `unproductive` / `*` in the CDR3 column.

The Quantifier auto-inventory above shows the per-reason counts. The `dropped.tsv` file has columns `reason | source_file | row_index | field | raw_value` — useful for debugging upstream data-quality issues. `skipped_files.tsv` is **also** written to the bucket directory and lists files whose format could not be detected — these never reach the row-level drop pipeline.

## Caveats specific to the standardized output

- The `geo` subdirectory is **excluded** at the top of file enumeration (`studies.py:386-389` and `studies.py:553-555`). Studies that should live there must be moved to a top-level study directory, or their chain mapping won't be applied.
- `_pair_chains_by_cell` keeps only the first contig per cell per chain (`studies.py:440`); cells with multiple productive alpha or beta contigs lose all but one. Single-cell consumers may want to re-derive multiple chains downstream.
- `study_id` granularity is the top-level directory name. Some studies are split across many input files; cross-study dedup is not attempted here.
- `skipped_files.tsv` is the place to look when a study's row count is unexpectedly zero — it likely lacked a recognised CDR3 / V-gene column header.
- 93 parquet shards.

## Use in QUEST training

Studies is mixed-evidence — most files are bulk repertoire (used for MLM / TCR-only / paired training); a small subset carries epitope/peptide annotations (used for interaction training when present). Per the integration framework, route by labelled subset: filter `peptide != ""` for interaction training; otherwise treat as bulk. See [raw-data studies index](../../raw_data/studies/README.md) for per-study fidelity classification.
