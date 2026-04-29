# aku_breast_pilot

## Identity
- **Source path**: `data/raw_data/studies/aku_breast_pilot/`
- **Accession**: internal cohort (AKU = Aga Khan University, Karachi; breast cancer pilot)
- **Primary citation**: unknown — needs investigation
- **Submission date**: unknown
- **License**: internal

## Data shape
- **Organism**: Homo sapiens (assumed)
- **Assay types present in directory**: bulk TRB survey
- **Platform**: Adaptive immunoSEQ (TSV)
- **Sample N**: ~12 files (TME_* prefix — tumor microenvironment samples)
- **Disease / condition**: Breast cancer (pilot)

## Raw inventory
<!-- BEGIN: AUTO-INVENTORY (build_raw_data_manifest.py) -->

- **Path**: `data/raw_data/studies/aku_breast_pilot`
- **Size**: 1.5M (1,564,354 bytes)
- **Files**: 12
- **Format breakdown**: `.tsv`=12
- **Records**: 5,288 (exact)
- **Counting method**: exact: sum of per-file record_count (after dedup)
- **Record definition**: tab/comma-delimited row excluding header

**Primary data files** (top 12 by size):

| File | Format | Size | Records | Method |
|------|--------|------|---------|--------|
| `tcr/bulk_survey_trb/TME_6.tsv` | tsv | 819.6K | 2,849 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/TME_7.tsv` | tsv | 195.8K | 678 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/TME_2.tsv` | tsv | 156.5K | 543 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/TME_11.tsv` | tsv | 92.1K | 317 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/TME_1.tsv` | tsv | 57.5K | 197 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/TME_9.tsv` | tsv | 57.3K | 197 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/TME_5.tsv` | tsv | 46.2K | 159 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/TME_8.tsv` | tsv | 43.9K | 151 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/TME_4.tsv` | tsv | 20.8K | 70 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/TME_10.tsv` | tsv | 14.3K | 48 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/TME_3.tsv` | tsv | 13.5K | 45 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/TME_12.tsv` | tsv | 10.3K | 34 | wc -l minus 1 (header) |

<!-- END: AUTO-INVENTORY -->

## Study summary
- **Goal**: Pilot breast cancer TCR repertoire profiling, presumably South Asian cohort.
- **Method**: Bulk TRB CDR3 sequencing.
- **Findings**: N/A.

## Caveats / notes
- Pilot study; small N. No README.

## Fidelity assessment
- **Confidence (1-5)**: 2 — bulk TCR pilot, breast cancer context.
- **QUEST integration tier**: bulk-repertoire
