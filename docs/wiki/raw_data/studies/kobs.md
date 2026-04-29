# kobs

## Identity
- **Source path**: `data/raw_data/studies/kobs/`
- **Accession**: internal cohort (KOBS — likely Kenya/Kaposi/observational study cohort)
- **Primary citation**: unknown — needs investigation
- **Submission date**: unknown
- **License**: internal

## Data shape
- **Organism**: Homo sapiens (assumed)
- **Assay types present in directory**: bulk TRB survey
- **Platform**: Adaptive immunoSEQ (TSV)
- **Sample N**: ~32 files (KOBSC_* prefix)
- **Disease / condition**: unknown — likely Kaposi sarcoma observational study (KOBS-C)

## Raw inventory
<!-- BEGIN: AUTO-INVENTORY (build_raw_data_manifest.py) -->

- **Path**: `data/raw_data/studies/kobs`
- **Size**: 2.4G (2,604,494,904 bytes)
- **Files**: 32
- **Format breakdown**: `.tsv`=32
- **Records**: 2,219,979 (exact)
- **Counting method**: exact: sum of per-file record_count (after dedup)
- **Record definition**: tab/comma-delimited row excluding header

**Primary data files** (top 12 by size):

| File | Format | Size | Records | Method |
|------|--------|------|---------|--------|
| `tcr/bulk_survey_trb/KOBSC_54.tsv` | tsv | 144.6M | 130,453 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KOBSC_27.tsv` | tsv | 130.5M | 117,287 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KOBSC_9.tsv` | tsv | 119.8M | 106,357 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KOBSC_7.tsv` | tsv | 107.0M | 94,606 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KOBSC_4.tsv` | tsv | 105.5M | 95,130 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KOBSC_8.tsv` | tsv | 100.7M | 90,161 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KOBSC_16.tsv` | tsv | 100.2M | 89,335 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KOBSC_36.tsv` | tsv | 100.0M | 89,014 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KOBSC_35.tsv` | tsv | 98.5M | 87,604 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KOBSC_29.tsv` | tsv | 94.3M | 83,133 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KOBSC_3.tsv` | tsv | 89.3M | 79,032 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KOBSC_15.tsv` | tsv | 89.1M | 80,256 | wc -l minus 1 (header) |

<!-- END: AUTO-INVENTORY -->

## Study summary
- **Goal**: TCR repertoire profiling within KOBS observational cohort.
- **Method**: Bulk TRB CDR3 sequencing.
- **Findings**: N/A.

## Caveats / notes
- KOBSC sample-prefix naming pattern; check internal records for cohort definition.

## Fidelity assessment
- **Confidence (1-5)**: 2 — bulk TCR, no metadata available.
- **QUEST integration tier**: bulk-repertoire
