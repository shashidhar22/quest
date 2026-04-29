# acsr_southafrica

## Identity
- **Source path**: `data/raw_data/studies/acsr_southafrica/`
- **Accession**: internal cohort (ACSR South Africa subset)
- **Primary citation**: unknown — needs investigation
- **Submission date**: unknown
- **License**: internal

## Data shape
- **Organism**: Homo sapiens (assumed)
- **Assay types present in directory**: bulk TRA + bulk TRB survey (both alpha and beta!)
- **Platform**: Adaptive immunoSEQ
- **Sample N**: 60 files (split between TRA and TRB), A_* prefixed sample IDs, includes FFPE
- **Disease / condition**: AIDS-related cancer specimens (South African cohort)

## Raw inventory
<!-- BEGIN: AUTO-INVENTORY (build_raw_data_manifest.py) -->

- **Path**: `data/raw_data/studies/acsr_southafrica`
- **Size**: 96.4M (101,041,520 bytes)
- **Files**: 60
- **Format breakdown**: `.tsv`=60
- **Records**: 346,681 (exact)
- **Counting method**: exact: sum of per-file record_count (after dedup)
- **Record definition**: tab/comma-delimited row excluding header

**Primary data files** (top 12 by size):

| File | Format | Size | Records | Method |
|------|--------|------|---------|--------|
| `tcr/bulk_survey_tra/A_60085490.tsv` | tsv | 7.7M | 27,621 | wc -l minus 1 (header) |
| `tcr/bulk_survey_tra/A_600374710.tsv` | tsv | 5.9M | 21,016 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/A_600374710.tsv` | tsv | 5.9M | 21,181 | wc -l minus 1 (header) |
| `tcr/bulk_survey_tra/A_60097460.tsv` | tsv | 5.7M | 20,599 | wc -l minus 1 (header) |
| `tcr/bulk_survey_tra/A_600446250.tsv` | tsv | 5.2M | 18,816 | wc -l minus 1 (header) |
| `tcr/bulk_survey_tra/A_6006630.tsv` | tsv | 4.0M | 14,347 | wc -l minus 1 (header) |
| `tcr/bulk_survey_tra/A_600256610.tsv` | tsv | 3.9M | 14,190 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/A_600446250.tsv` | tsv | 3.6M | 12,929 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/A_60097460.tsv` | tsv | 3.4M | 12,328 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/A_6006630.tsv` | tsv | 3.4M | 12,182 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/A_600270700.tsv` | tsv | 3.3M | 11,931 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/A_600256610.tsv` | tsv | 3.0M | 10,754 | wc -l minus 1 (header) |

<!-- END: AUTO-INVENTORY -->

## Study summary
- **Goal**: Bulk TCR profiling from South African subset of ACSR (likely Kaposi sarcoma / HIV malignancies).
- **Method**: Bulk TRA + TRB CDR3 sequencing (notable: alpha sequencing present, unusual for survey-depth bulk).
- **Findings**: N/A.

## Caveats / notes
- Has BOTH alpha and beta — useful for paired chain inference if matched per-sample.
- FFPE samples present.

## Fidelity assessment
- **Confidence (1-5)**: 2 — bulk TCR (TRA+TRB) with regional context but no explicit clinical labels.
- **QUEST integration tier**: bulk-repertoire (potentially paired if TRA/TRB share sample IDs)
