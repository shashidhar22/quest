# bl_uganda

## Identity
- **Source path**: `data/raw_data/studies/bl_uganda/`
- **Accession**: internal cohort (BL = Burkitt lymphoma, Uganda)
- **Primary citation**: unknown — needs investigation
- **Submission date**: unknown
- **License**: internal

## Data shape
- **Organism**: Homo sapiens (assumed)
- **Assay types present in directory**: bulk TRA + bulk TRB survey
- **Platform**: Adaptive immunoSEQ (TSV)
- **Sample N**: ~52 files (across TRA + TRB; H/MVQ/HU sample prefixes)
- **Disease / condition**: Burkitt lymphoma (Ugandan cohort, EBV-associated)

## Raw inventory
<!-- BEGIN: AUTO-INVENTORY (build_raw_data_manifest.py) -->

- **Path**: `data/raw_data/studies/bl_uganda`
- **Size**: 222.1M (232,880,210 bytes)
- **Files**: 52
- **Format breakdown**: `.tsv`=52
- **Records**: 803,616 (exact)
- **Counting method**: exact: sum of per-file record_count (after dedup)
- **Record definition**: tab/comma-delimited row excluding header

**Primary data files** (top 12 by size):

| File | Format | Size | Records | Method |
|------|--------|------|---------|--------|
| `tcr/bulk_survey_trb/H003840_O2_ARL.tsv` | tsv | 25.5M | 92,541 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/H031201_O1_ARL.tsv` | tsv | 23.8M | 85,881 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/H003840_01.tsv` | tsv | 21.1M | 76,344 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/H031201_O2_ARL.tsv` | tsv | 19.9M | 71,902 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/H003840_O1_ARL.tsv` | tsv | 18.0M | 65,349 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/H031201_02.tsv` | tsv | 16.1M | 58,138 | wc -l minus 1 (header) |
| `tcr/bulk_survey_tra/MVQ94865A.tsv` | tsv | 12.1M | 43,682 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/H003840_02.tsv` | tsv | 11.3M | 40,848 | wc -l minus 1 (header) |
| `tcr/bulk_survey_tra/MVQ91443A.tsv` | tsv | 8.0M | 28,810 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/H057520_O2_ARL.tsv` | tsv | 7.0M | 25,369 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/H057520_02.tsv` | tsv | 5.6M | 20,306 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/H057520_01.tsv` | tsv | 4.9M | 17,793 | wc -l minus 1 (header) |

<!-- END: AUTO-INVENTORY -->

## Study summary
- **Goal**: TCR repertoire of Burkitt lymphoma in a Ugandan cohort.
- **Method**: Bulk TRA + TRB CDR3 sequencing.
- **Findings**: N/A.

## Caveats / notes
- Has both TRA and TRB chains; sample-paired alpha/beta inference possible.
- EBV/malaria epidemiologic context — Burkitt lymphoma is endemic in Uganda.

## Fidelity assessment
- **Confidence (1-5)**: 2 — bulk TCR (TRA+TRB) with disease context, no epitope.
- **QUEST integration tier**: bulk-repertoire
