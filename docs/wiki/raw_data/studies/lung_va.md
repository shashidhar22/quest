# lung_va

## Identity
- **Source path**: `data/raw_data/studies/lung_va/`
- **Accession**: internal cohort (Lung VA — Veterans Affairs lung cancer cohort)
- **Primary citation**: unknown — needs investigation
- **Submission date**: unknown (sample timestamps span 2018-2020)
- **License**: internal

## Data shape
- **Organism**: Homo sapiens (assumed)
- **Assay types present in directory**: bulk TRB survey
- **Platform**: Adaptive immunoSEQ (TSV)
- **Sample N**: ~320 files (VA0* prefix; serial timepoints: PD/HO/FU encoded in filename)
- **Disease / condition**: Lung cancer in VA cohort (longitudinal sampling: progression, baseline, follow-up)

## Raw inventory
<!-- BEGIN: AUTO-INVENTORY (build_raw_data_manifest.py) -->

- **Path**: `data/raw_data/studies/lung_va`
- **Size**: 1.8G (1,923,782,381 bytes)
- **Files**: 320
- **Format breakdown**: `.tsv`=319, `.zip`=1
- **Records**: 11,351,019 (exact)
- **Counting method**: exact: sum of per-file record_count (after dedup)
- **Record definition**: sum of records in archived csv/tsv/fasta members; tab/comma-delimited row excluding header

**Primary data files** (top 12 by size):

| File | Format | Size | Records | Method |
|------|--------|------|---------|--------|
| `tcr/bulk_survey_trb/nsclc.zip` | zip | 228.3M | 6,104,178 | zip member inspection; sum(lines-1) over csv/tsv/txt + fasta entries |
| `tcr/bulk_survey_trb/VA098_9_15_21_PBMC.tsv` | tsv | 41.1M | 38,453 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/VA098_3_1_22_PBMC.tsv` | tsv | 40.7M | 37,987 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/VA086_10_13_21_PBMC.tsv` | tsv | 37.6M | 34,756 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/VA095_3_9_22_PBMC.tsv` | tsv | 26.1M | 24,470 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/VA058_6_1_22_PBMC.tsv` | tsv | 24.3M | 22,641 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/VA087_10_6_21_PBMC.tsv` | tsv | 18.6M | 17,286 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/VA024_HO_FU_4_11_18.tsv` | tsv | 18.4M | 66,608 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/VA012_9_14_16.tsv` | tsv | 16.0M | 57,915 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/VA047_HOFU_4_25_18.tsv` | tsv | 16.0M | 57,856 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/VA012_12_2_15.tsv` | tsv | 15.3M | 55,236 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/VA044_HO_FU_1_16_2019.tsv` | tsv | 15.3M | 55,205 | wc -l minus 1 (header) |

<!-- END: AUTO-INVENTORY -->

## Study summary
- **Goal**: Longitudinal TCR profiling of lung cancer patients in VA cohort across treatment timepoints.
- **Method**: Bulk TRB CDR3 sequencing at multiple timepoints per patient.
- **Findings**: N/A.

## Caveats / notes
- Longitudinal — same donor sampled multiple times; cluster by patient ID before splits.
- Filename encodes timepoint (e.g. `VA079_PD_9_15_20.tsv` = patient VA079, PD timepoint, date).

## Fidelity assessment
- **Confidence (1-5)**: 3 — bulk TCR with longitudinal disease/timepoint labels.
- **QUEST integration tier**: bulk-repertoire
