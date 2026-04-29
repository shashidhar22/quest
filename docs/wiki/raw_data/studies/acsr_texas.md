# acsr_texas

## Identity
- **Source path**: `data/raw_data/studies/acsr_texas/`
- **Accession**: internal cohort (ACSR Texas subset)
- **Primary citation**: unknown — needs investigation
- **Submission date**: unknown
- **License**: internal

## Data shape
- **Organism**: Homo sapiens (assumed)
- **Assay types present in directory**: bulk TRB survey
- **Platform**: Adaptive immunoSEQ
- **Sample N**: ~34 files (AL00*LA-*, P-* prefixes; some FFPE)
- **Disease / condition**: AIDS-related cancer (Texas regional subset of ACSR)

## Raw inventory
<!-- BEGIN: AUTO-INVENTORY (build_raw_data_manifest.py) -->

- **Path**: `data/raw_data/studies/acsr_texas`
- **Size**: 116.6M (122,235,484 bytes)
- **Files**: 34
- **Format breakdown**: `.tsv`=34
- **Records**: 421,643 (exact)
- **Counting method**: exact: sum of per-file record_count (after dedup)
- **Record definition**: tab/comma-delimited row excluding header

**Primary data files** (top 12 by size):

| File | Format | Size | Records | Method |
|------|--------|------|---------|--------|
| `tcr/bulk_survey_trb/AL0042LA-3.tsv` | tsv | 14.0M | 50,865 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/AL0071LA-5.tsv` | tsv | 12.7M | 46,178 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/AL0020LA-3.tsv` | tsv | 11.9M | 42,933 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/AL0006LA-4.tsv` | tsv | 8.7M | 31,582 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/AL0038LA-4.tsv` | tsv | 8.2M | 29,436 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/AL0024LA-3.tsv` | tsv | 6.7M | 24,393 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/AL0058LA-4.tsv` | tsv | 6.7M | 24,087 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/P-2001084315_FFPE.tsv` | tsv | 6.2M | 22,413 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/AL0076LA-2.tsv` | tsv | 6.0M | 21,560 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/AL0005LA-7.tsv` | tsv | 5.7M | 20,769 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/AL0048LA-2.tsv` | tsv | 5.1M | 18,530 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/AL0085LA-3.tsv` | tsv | 4.3M | 15,579 | wc -l minus 1 (header) |

<!-- END: AUTO-INVENTORY -->

## Study summary
- **Goal**: TCR repertoire from ACSR Texas regional cohort.
- **Method**: Bulk TRB CDR3 sequencing.
- **Findings**: N/A.

## Caveats / notes
- FFPE samples present.

## Fidelity assessment
- **Confidence (1-5)**: 2 — bulk TCR, no explicit labels.
- **QUEST integration tier**: bulk-repertoire
