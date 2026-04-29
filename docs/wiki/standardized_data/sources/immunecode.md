# ImmuneCODE (standardized)

## Identity
- **Output path**: `data/standardized_again/immunecode/`
- **Standardizer**: [`scripts/data_processing/standardize/immunecode.py`](../../../../scripts/data_processing/standardize/immunecode.py) (`ImmunecodeStandardizer`)
- **Raw input**: see [raw-data manifest entry](../../raw_data/databases/immunecode.md) for citation, fidelity, and source provenance
- **Generated**: 2026-04-02T19:56:12Z
- **Standardization wall time**: 236.1s

## What this standardizer does

ImmuneCODE is COVID-19 TCR data from Adaptive. The standardizer handles two sub-formats with separate paths:

- **MIRA peptide-detail (`peptide-detail-ci.csv` only — Class I; `peptide-detail-cii.csv` is intentionally skipped because it contains minigene constructs, `immunecode.py:240-241`).** TCRs are stored as `BioIdentity` strings like `CASSAQGTGDRGYTF+TCRBV27-01+TCRBJ01-02`; these are vectorised-split via `_bio_identity_to_df` (`immunecode.py:53-60`). Comma-separated peptides in `Amino Acids` are exploded to one row per peptide (`immunecode.py:101-108`); peptides longer than 25 AA are filtered out as minigene constructs (`immunecode.py:120-122`). HLA typing per `Experiment` ID is loaded from `subject-metadata.csv` (`immunecode.py:205-236`) and the first Class I allele is mapped into `mhc_one`.
- **Review repertoire TSVs (`*_TCRB.tsv`).** Read in 500K-row chunks; filtered to `frame_type == "In"` (`immunecode.py:163-167`); `bio_identity` parsed into `trb`/`trbv_gene`/`trbj_gene`; `d_gene` populates `trbd_gene`. `study_id` is the file stem (e.g., `KHBR20-00164_TCRB`).

Beta-only throughout — ImmuneCODE does not carry alpha chains.

**Column populations** (which TARGET_COLUMNS this standardizer fills meaningfully):
- `trb`, `trbv_gene`, `trbd_gene`, `trbj_gene`
- `peptide` (only on MIRA rows; bulk Review rows leave it empty)
- `mhc_one` (Class I HLA from subject metadata for MIRA rows; empty for Review)
- `mhc_two` always empty (Class II MIRA file is skipped)
- `source = "immunecode"`, `study_id` = file stem
- CDR1/CDR2 + `trb_full` populated downstream
- No `tra`, no `binding`, no `score`

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_standardized_manifest.py) -->

- **Path**: `data/standardized_again/immunecode`
- **Standardized at**: 2026-04-02T19:56:12.268370+00:00
- **Rows (manifest)**: 251,274,283
- **Dropped**: 55,603,532 (18.12% drop rate)
- **Standardization elapsed**: 236.1s
- **Source files (checksummed)**: 1430
- **Parquet**: 1526 files, 14.9G (15,984,173,091 bytes), 251,274,283 rows (matches manifest)

**Top 8 most-populated columns:**

| Column | Populated | Coverage | Unique |
|--------|-----------|----------|--------|
| `trbj_gene` | 251,274,283 | 100.00% | n/a |
| `source` | 251,274,283 | 100.00% | n/a |
| `trbv_gene` | 251,066,967 | 99.92% | n/a |
| `trb` | 250,979,931 | 99.88% | >1000000 |
| `trb_cdr3` | 250,979,931 | 99.88% | n/a |
| `trb_full` | 250,629,953 | 99.74% | n/a |
| `study_id` | 223,219,759 | 88.84% | 29 (sampled from 30/1526 shards) |
| `trb_cdr1` | 214,965,488 | 85.55% | n/a |

**Bottom 5 least-populated columns:**

| Column | Populated | Coverage |
|--------|-----------|----------|
| `tra_full` | 0 | 0.00% |
| `mhc_two` | 0 | 0.00% |
| `mhc_two_allele` | 0 | 0.00% |
| `binding` | 0 | 0.00% |
| `score` | 0 | 0.00% |

**Top 10 drop reasons:**

| Reason | Count | Share |
|--------|-------|-------|
| `invalid_trbd_gene` | 55,384,869 | 99.61% |
| `invalid_trbv_gene` | 209,891 | 0.38% |
| `invalid_trb` | 8,772 | 0.02% |

<!-- END: AUTO-INVENTORY -->

## Validation drops

55.6M rows dropped from ~307M input rows (~18%). Top reasons in a 5M-row sample of the dropped file: `invalid_trbd_gene` (~5M, dominant — Adaptive's `TCRBD` prefix and `unresolved` values fail IMGT canonicalisation), with much smaller `invalid_trbv_gene` (~19K) and `invalid_trb` (~226). `no_valid_field` is rare because `trb` is usually valid even when D-gene fails.

The Quantifier auto-inventory above shows the per-reason counts. The `dropped.tsv` file has columns `reason | source_file | row_index | field | raw_value` — useful for debugging upstream data-quality issues.

## Caveats specific to the standardized output

- The Class II MIRA file (`peptide-detail-cii.csv`) is intentionally **excluded** because its peptides are minigene constructs (>25 AA, not native antigen peptides). If you need ImmuneCODE Class II epitope data you must rebuild this standardizer.
- HLA mapping is a "first allele" simplification (`immunecode.py:128-132`): each subject is heterozygous so this discards 1–5 alleles per row. Cross-reference `subject-metadata.csv` for the full typing.
- Review repertoire rows have `peptide` empty by design — they are bulk repertoire and only useful for MLM / TCR-only training.
- 1,526 parquet shards — glob the directory.
- Beta-only throughout: no `tra` rows are produced.

## Use in QUEST training

ImmuneCODE is mixed-evidence: MIRA rows are experimental TCR–pMHC binding (high fidelity, used for interaction training), Review rows are bulk repertoire (used only for MLM and TCR-only training). The integration framework treats this exactly as "Mixed source with labeled subsets" — split by label, route accordingly. See [raw-data ImmuneCODE page](../../raw_data/databases/immunecode.md) for the fidelity tier.
