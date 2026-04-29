# IMGTHLA (standardized)

## Identity
- **Output path**: `data/standardized_again/imgthla/`
- **Standardizer**: [`scripts/data_processing/standardize/imgthla.py`](../../../../scripts/data_processing/standardize/imgthla.py) (`ImgthlaStandardizer`)
- **Raw input**: see [raw-data manifest entry](../../raw_data/databases/imgthla.md) for citation, fidelity, and source provenance
- **Generated**: 2026-03-26T19:12:24Z
- **Standardization wall time**: 0.4s

## What this standardizer does

Reads the IPD-IMGT/HLA `hla_prot.fasta` and emits one row per unique 4-digit allele. The FASTA header regex `_ALLELE_RE` (`imgthla.py:36-38`) captures `gene*group:protein` (e.g., `A*01:01`); deeper-resolution alleles (6-digit, 8-digit) collapse to their 4-digit prefix and the longest sequence is kept (`imgthla.py:67-69`). Genes are bucketed by class: class I and class II alpha go into `mhc_one` (`imgthla.py:30-31`), class II beta goes into `mhc_two` (`imgthla.py:32`). Unknown genes default to `mhc_one`. The allele name (e.g., `HLA-A*01:01`) is stored in both the corresponding `mhc_one_allele` / `mhc_two_allele` field and in `study_id` (`imgthla.py:137-140`).

This bucket bypasses `standardize_dataframe` — it builds rows directly with all 25 TARGET_COLUMNS pre-initialized to `""` (`imgthla.py:134-141`). No HLA-resolution step is applied; the sequences are the source.

**Column populations** (which TARGET_COLUMNS this standardizer fills meaningfully):
- `mhc_one` (full HLA protein sequence, for class I and class II alpha) **or** `mhc_two` (for class II beta)
- `mhc_one_allele` / `mhc_two_allele` (HLA name, e.g., `HLA-A*01:01`)
- `study_id` = HLA allele name (used as a stable identifier for joining)
- `source = "imgthla"`
- No TCR fields, no peptide, no binding

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_standardized_manifest.py) -->

- **Path**: `data/standardized_again/imgthla`
- **Standardized at**: 2026-03-26T19:12:24.466970+00:00
- **Rows (manifest)**: 24,005
- **Dropped**: 0 (0.00% drop rate)
- **Standardization elapsed**: 0.4s
- **Source files (checksummed)**: 548
- **Parquet**: 1 files, 865.2K (886,007 bytes), 24,005 rows (matches manifest)

**Top 8 most-populated columns:**

| Column | Populated | Coverage | Unique |
|--------|-----------|----------|--------|
| `source` | 24,005 | 100.00% | n/a |
| `study_id` | 24,005 | 100.00% | 24,005 |
| `mhc_one` | 17,896 | 74.55% | n/a |
| `mhc_one_allele` | 17,896 | 74.55% | 17,896 |
| `mhc_two` | 6,109 | 25.45% | n/a |
| `mhc_two_allele` | 6,109 | 25.45% | n/a |

**Bottom 5 least-populated columns:**

| Column | Populated | Coverage |
|--------|-----------|----------|
| `trb_cdr3` | 0 | 0.00% |
| `trb_full` | 0 | 0.00% |
| `peptide` | 0 | 0.00% |
| `binding` | 0 | 0.00% |
| `score` | 0 | 0.00% |

<!-- END: AUTO-INVENTORY -->

## Validation drops

Zero rows dropped — IMGTHLA is the source-of-truth for HLA sequences, and the standardizer doesn't apply any AA-level validation (`dropped.tsv` is empty, `dropped_count = 0`).

The Quantifier auto-inventory above shows the per-reason counts. The `dropped.tsv` file has columns `reason | source_file | row_index | field | raw_value` — useful for debugging upstream data-quality issues.

## Caveats specific to the standardized output

- Resolution is collapsed to 4-digit. Higher-resolution alleles (e.g., `HLA-A*01:01:01:01`) are deduplicated to `HLA-A*01:01` and their per-allele protein sequence is the longest variant seen.
- This bucket is reference data, not training records. It is consumed by other standardizers via the `--hla-dir` argument: `standardize_dataframe(..., hla_dir=...)` calls `load_hla_sequences()` to resolve allele names to AA sequences and copies the name into `*_allele` columns (`quest/data/standardization.py:919-948`).
- 24,005 emitted rows is the number of distinct 4-digit alleles in `hla_prot.fasta` — substantially smaller than the raw FASTA's ~41K records (which include deeper-resolution duplicates).
- `binding`, `score`, `peptide`, all TCR fields, `tra_full`, `trb_full` are uniformly empty.

## Use in QUEST training

IMGTHLA is **reference / lookup data**, not training input. Other standardizers consume it via `hla_dir` so that `mhc_one`/`mhc_two` carry AA sequences (suitable for sequence-level encoding) while `mhc_one_allele`/`mhc_two_allele` carry the canonical names (suitable for allele-level joins). See [raw-data IMGTHLA page](../../raw_data/databases/imgthla.md) for citation and fidelity context.
