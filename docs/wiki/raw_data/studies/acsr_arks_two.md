# acsr_arks_two

## Identity
- **Source path**: `data/raw_data/studies/acsr_arks_two/`
- **Accession**: internal cohort (second ARKS batch within ACSR)
- **Primary citation**: unknown — needs investigation
- **Submission date**: unknown
- **License**: internal

## Data shape
- **Organism**: Homo sapiens (assumed)
- **Assay types present in directory**: bulk TRB survey (`tcr/bulk_survey_trb/*.tsv`)
- **Platform**: Adaptive immunoSEQ (TSV)
- **Sample N**: ~137 files
- **Disease / condition**: Likely KS / AIDS-related (sample naming uses 7-digit IDs)

## Raw inventory
<!-- BEGIN: AUTO-INVENTORY (build_raw_data_manifest.py) -->

- **Path**: `data/raw_data/studies/acsr_arks_two`
- **Size**: 738.8M (774,676,244 bytes)
- **Files**: 137
- **Format breakdown**: `.tsv`=137
- **Records**: 2,679,842 (exact)
- **Counting method**: exact: sum of per-file record_count (after dedup)
- **Record definition**: tab/comma-delimited row excluding header

**Primary data files** (top 12 by size):

| File | Format | Size | Records | Method |
|------|--------|------|---------|--------|
| `tcr/bulk_survey_trb/2351829_ARKS.tsv` | tsv | 16.4M | 59,376 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/2352111_ARKS.tsv` | tsv | 15.9M | 57,776 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/2135110_ARKS.tsv` | tsv | 13.1M | 47,321 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/231711_ARKS.tsv` | tsv | 12.7M | 46,209 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/1401139_ARKS.tsv` | tsv | 12.6M | 45,803 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/231628_ARKS.tsv` | tsv | 10.5M | 38,202 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/1399871_ARKS.tsv` | tsv | 10.2M | 37,306 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/1399953_ARKS.tsv` | tsv | 10.2M | 36,852 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/2134040_ARKS.tsv` | tsv | 10.1M | 36,867 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/2134639_ARKS.tsv` | tsv | 10.0M | 36,157 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/1400945_ARKS.tsv` | tsv | 10.0M | 36,143 | wc -l minus 1 (header) |
| `tcr/bulk_survey_trb/903062_ARKS.tsv` | tsv | 9.5M | 34,739 | wc -l minus 1 (header) |

<!-- END: AUTO-INVENTORY -->

## Study summary
- **Goal**: Larger ARKS batch within ACSR cohort family.
- **Method**: Bulk TRB CDR3 sequencing.
- **Findings**: N/A.

## Caveats / notes
- Larger N than `acsr_arks/` — cross-source dedup with `acsr/`, `acsr_arks/`.
- No README.

## Fidelity assessment
- **Confidence (1-5)**: 2 — bulk TCR, modest N, no explicit metadata.
- **QUEST integration tier**: bulk-repertoire
