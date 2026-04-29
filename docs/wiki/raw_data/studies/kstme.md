# kstme

## Identity
- **Source path**: `data/raw_data/studies/kstme/`
- **Accession**: internal cohort (KSTME — likely Kaposi Sarcoma Tumor Microenvironment study)
- **Primary citation**: unknown — needs investigation
- **Submission date**: unknown
- **License**: internal

## Data shape
- **Organism**: Homo sapiens (assumed)
- **Assay types present in directory**: bulk TRA + TRB survey, AIRR (`airr/`), 10x contigs (`contigs/*_contig_annotations.csv`), 10x clonotypes (`clonotypes/`)
- **Platform**: Mixed: Adaptive immunoSEQ (bulk) + 10x Genomics V(D)J (single-cell)
- **Sample N**: 849 files total (richest cohort by file count)
- **Disease / condition**: Kaposi sarcoma tumor microenvironment

## Raw inventory
<!-- BEGIN: AUTO-INVENTORY (build_raw_data_manifest.py) -->

- **Path**: `data/raw_data/studies/kstme`
- **Size**: 5.0G (5,329,082,305 bytes)
- **Files**: 849
- **Format breakdown**: `.tsv`=771, `.csv`=78
- **Records**: 5,099,803 (exact)
- **Counting method**: exact: sum of per-file record_count (after dedup)
- **Record definition**: tab/comma-delimited row excluding header

**Primary data files** (top 12 by size):

| File | Format | Size | Records | Method |
|------|--------|------|---------|--------|
| `tcr/bulk_survey_trb/008_264_V9_1_19_2022.tsv` | tsv | 70.7M | 66,788 | wc -l minus 1 (header) |
| `tcr/airr/008_216_V09_airr.tsv` | tsv | 63.4M | 30,877 | wc -l minus 1 (header) |
| `tcr/airr/008_216_V10_airr.tsv` | tsv | 59.8M | 29,177 | wc -l minus 1 (header) |
| `tcr/airr/008_217_V09_airr.tsv` | tsv | 57.3M | 25,584 | wc -l minus 1 (header) |
| `tcr/airr/008_223_scREP_B_airr.tsv` | tsv | 54.1M | 24,142 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/008_260_V1_11_17_20.tsv` | tsv | 51.8M | 46,065 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/008_259_V10_10_25_21.tsv` | tsv | 46.8M | 41,153 | wc -l minus 1 (header) |
| `tcr/airr/008_217_scREP_H_airr.tsv` | tsv | 45.1M | 20,147 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/008_260_12_7_20_PBMC.tsv` | tsv | 44.2M | 39,236 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/008_258_V10_2_2_2022.tsv` | tsv | 43.8M | 41,065 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/008_260_V9_7_21_21.tsv` | tsv | 43.1M | 38,503 | wc -l minus 1 (header) |
| `tcr/bulk_survey_tra/008_187_C.tsv` | tsv | 42.4M | 51,286 | wc -l minus 1 (header) |

<!-- END: AUTO-INVENTORY -->

## Study summary
- **Goal**: Multi-modal TCR profiling of Kaposi sarcoma tumor microenvironment.
- **Method**: Bulk + 10x single-cell V(D)J on KS tumor samples.
- **Findings**: N/A (in-progress / institutional cohort).

## Caveats / notes
- Largest custom cohort by file volume.
- Mixed bulk and single-cell — schema normalization needed.
- AIRR format folder suggests data already in standard format for portions.

## Fidelity assessment
- **Confidence (1-5)**: 3 — large mixed cohort with KS context; no public publication identified.
- **QUEST integration tier**: mixed (single-cell-paired + bulk-repertoire)
