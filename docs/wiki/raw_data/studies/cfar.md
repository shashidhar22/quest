# cfar

## Identity
- **Source path**: `data/raw_data/studies/cfar/`
- **Accession**: internal cohort (CFAR = Center for AIDS Research)
- **Primary citation**: unknown — needs investigation
- **Submission date**: unknown
- **License**: internal

## Data shape
- **Organism**: Homo sapiens (assumed)
- **Assay types present in directory**: bulk TRA + bulk TRB survey
- **Platform**: Adaptive immunoSEQ (TSV)
- **Sample N**: ~240 files (across TRA + TRB, sample IDs e.g. 015V12003376_CFAR.tsv)
- **Disease / condition**: HIV/AIDS-related (CFAR repository)

## Raw inventory
<!-- BEGIN: AUTO-INVENTORY (build_raw_data_manifest.py) -->

- **Path**: `data/raw_data/studies/cfar`
- **Size**: 2.3G (2,502,856,085 bytes)
- **Files**: 240
- **Format breakdown**: `.tsv`=240
- **Records**: 8,412,417 (exact)
- **Counting method**: exact: sum of per-file record_count (after dedup)
- **Record definition**: tab/comma-delimited row excluding header

**Primary data files** (top 12 by size):

| File | Format | Size | Records | Method |
|------|--------|------|---------|--------|
| `tcr/bulk_survey_trb/TRB-D001-001-PBMC-Unsorted.tsv` | tsv | 31.5M | 93,482 | wc -l minus 1 (header) |
| `tcr/bulk_survey_tra/015V12002658_CFAR.tsv` | tsv | 31.0M | 111,100 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/TRB-D001-023-PBMC-Unsorted.tsv` | tsv | 30.7M | 91,923 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/TRB-D001-035-PBMC-Unsorted.tsv` | tsv | 30.5M | 90,289 | wc -l minus 1 (header) |
| `tcr/bulk_survey_tra/015V14001394_CFAR.tsv` | tsv | 29.0M | 103,898 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/TRB-D002-031-PBMC-Unsorted.tsv` | tsv | 26.6M | 78,611 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/TRB-D001-027-PBMC-Unsorted.tsv` | tsv | 26.5M | 78,462 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/TRB-D001-015-PBMC-Unsorted.tsv` | tsv | 24.9M | 74,092 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/TRB-D002-011-PBMC-Unsorted.tsv` | tsv | 22.6M | 69,972 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/015V10004235_CFAR.tsv` | tsv | 21.7M | 78,495 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/TRB-D002-014-PBMC-Unsorted.tsv` | tsv | 20.2M | 62,434 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/TRB-D002-037-PBMC-Unsorted.tsv` | tsv | 19.7M | 61,256 | wc -l minus 1 (header) |

<!-- END: AUTO-INVENTORY -->

## Study summary
- **Goal**: TCR repertoire from CFAR (HIV-focused) cohort.
- **Method**: Bulk TRA + TRB CDR3 sequencing.
- **Findings**: N/A.

## Caveats / notes
- Both chains present; large cohort (~240 files); useful for paired-chain inference.

## Fidelity assessment
- **Confidence (1-5)**: 2 — bulk TCR (TRA+TRB) with HIV context, no epitope.
- **QUEST integration tier**: bulk-repertoire (potentially paired if sample IDs align)
