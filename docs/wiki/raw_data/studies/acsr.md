# acsr

## Identity
- **Source path**: `data/raw_data/studies/acsr/`
- **Accession**: internal cohort (ACSR — likely AIDS/Cancer Specimen Resource or institutional ACSR cohort)
- **Primary citation**: unknown — needs investigation
- **Submission date**: unknown
- **License**: internal

## Data shape
- **Organism**: Homo sapiens (assumed)
- **Assay types present in directory**: bulk TRB CDR3 survey-depth (`tcr/bulk_survey_trb/*.tsv`)
- **Platform**: Adaptive immunoSEQ (TSV format suggests immunoSEQ output)
- **Sample N**: ~10 TSV files (samples include FFPE, e.g. KPA5092_FFPE.tsv)
- **Disease / condition**: KS / Kaposi sarcoma context (KPA prefix; AIDS-related cancer specimens)

## Raw inventory
<!-- BEGIN: AUTO-INVENTORY (build_raw_data_manifest.py) -->

- **Path**: `data/raw_data/studies/acsr`
- **Size**: 43.8M (45,953,020 bytes)
- **Files**: 10
- **Format breakdown**: `.tsv`=10
- **Records**: 158,609 (exact)
- **Counting method**: exact: sum of per-file record_count (after dedup)
- **Record definition**: tab/comma-delimited row excluding header

**Primary data files** (top 12 by size):

| File | Format | Size | Records | Method |
|------|--------|------|---------|--------|
| `tcr/bulk_survey_trb/KPA5394.tsv` | tsv | 11.5M | 41,696 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KPA5274.tsv` | tsv | 9.9M | 36,071 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KPA5092.tsv` | tsv | 8.3M | 30,088 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KPA5140.tsv` | tsv | 7.8M | 28,172 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KPA5210.tsv` | tsv | 6.2M | 22,339 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KPA5092_FFPE.tsv` | tsv | 31.4K | 108 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KPA5394_FFPE.tsv` | tsv | 13.5K | 46 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KPA5274_FFPE.tsv` | tsv | 9.4K | 31 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KPA5140_FFPE.tsv` | tsv | 9.1K | 30 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KPA5210_FFPE.tsv` | tsv | 8.5K | 28 | wc -l minus 1 (header) |

<!-- END: AUTO-INVENTORY -->

## Study summary
- **Goal**: Internal AIDS-related cancer cohort TCR repertoire profiling.
- **Method**: Bulk TRB CDR3 sequencing (immunoSEQ-style format).
- **Findings**: N/A (institutional dataset; no public manuscript identified).

## Caveats / notes
- No README, manifest, or filelist in directory.
- "ACSR" likely = AIDS and Cancer Specimen Resource (NIH-funded biorepository for HIV-related malignancies). Confirm via internal sources.
- FFPE samples present — beware of degraded TCR coverage.

## Fidelity assessment
- **Confidence (1-5)**: 2 — bulk TCR with implicit cancer/HIV context, no explicit metadata.
- **QUEST integration tier**: bulk-repertoire (MLM only)
