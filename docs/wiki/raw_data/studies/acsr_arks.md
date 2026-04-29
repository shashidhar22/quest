# acsr_arks

## Identity
- **Source path**: `data/raw_data/studies/acsr_arks/`
- **Accession**: internal cohort (ACSR ARKS — extension of ACSR cohort, "ARKS" likely sample-naming prefix)
- **Primary citation**: unknown — needs investigation
- **Submission date**: unknown
- **License**: internal

## Data shape
- **Organism**: Homo sapiens (assumed)
- **Assay types present in directory**: bulk TRB survey (`tcr/bulk_survey_trb/*.tsv`)
- **Platform**: Adaptive immunoSEQ (TSV)
- **Sample N**: ~14 files (KPA_* IDs)
- **Disease / condition**: AIDS-related malignancies (Kaposi sarcoma context, by KPA prefix)

## Raw inventory
<!-- BEGIN: AUTO-INVENTORY (build_raw_data_manifest.py) -->

- **Path**: `data/raw_data/studies/acsr_arks`
- **Size**: 118.3K (121,129 bytes)
- **Files**: 14
- **Format breakdown**: `.tsv`=14
- **Records**: 389 (exact)
- **Counting method**: exact: sum of per-file record_count (after dedup)
- **Record definition**: tab/comma-delimited row excluding header

**Primary data files** (top 12 by size):

| File | Format | Size | Records | Method |
|------|--------|------|---------|--------|
| `tcr/bulk_survey_trb/KPA_5092_ARKS.tsv` | tsv | 10.2K | 34 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KPA_5062_ARKS.tsv` | tsv | 9.4K | 31 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KPA_5662_ARKS.tsv` | tsv | 9.3K | 30 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KPA_5110_ARKS.tsv` | tsv | 9.3K | 31 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KPA_5050_ARKS.tsv` | tsv | 9.1K | 30 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KPA_5286_ARKS.tsv` | tsv | 9.0K | 30 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KPA_5273_ARKS.tsv` | tsv | 8.9K | 29 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KPA_5076_ARKS.tsv` | tsv | 8.7K | 29 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KPA_5140_ARKS.tsv` | tsv | 8.5K | 28 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KPA_5661_ARKS.tsv` | tsv | 8.2K | 27 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KPA_5394_ARKS.tsv` | tsv | 7.1K | 23 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/KPA_5210_ARKS.tsv` | tsv | 7.1K | 23 | wc -l minus 1 (header) |

<!-- END: AUTO-INVENTORY -->

## Study summary
- **Goal**: ARKS extension of ACSR — additional Kaposi-sarcoma-context bulk TRB.
- **Method**: Bulk TRB CDR3 sequencing.
- **Findings**: N/A.

## Caveats / notes
- No README/manifest; treat as ACSR sibling.
- Likely overlaps with `acsr/` and `acsr_arks_two/` — dedup carefully.

## Fidelity assessment
- **Confidence (1-5)**: 2 — bulk TCR, no metadata.
- **QUEST integration tier**: bulk-repertoire
