# ADC (standardized)

## Identity
- **Output path**: `data/standardized_again/adc/`
- **Standardizer**: [`scripts/data_processing/standardize/adc.py`](../../../../scripts/data_processing/standardize/adc.py) (`AdcStandardizer`)
- **Raw input**: see [raw-data manifest entry](../../raw_data/databases/adc.md) for citation, fidelity, and source provenance
- **Generated**: 2026-03-29T02:20:54Z
- **Standardization wall time**: 134,432.5s (~37.3 h, parallel multi-process)

## What this standardizer does

ADC is the largest single source ingested by QUEST: ~9,300 AIRR rearrangement TSVs spread across multiple iReceptor repository directories under `data/raw_data/databases/adc/{repository}/rearrangements/`. The standardizer is run with `streaming=True` and `parallel_run` worker pool (`adc.py:33-41` for file enumeration, `adc.py:43-95` for per-file processing). Each TSV is read in 500K-row chunks (`adc.py:51-57`); rows are filtered to `productive == true/t/1` (`adc.py:60-65`), then split by the AIRR `locus` field — only `TRA` and `TRB` are kept (gamma-delta and other loci are silently dropped, `adc.py:73`). Per-locus column maps point `junction_aa` → `tra` or `trb` and `v_call`/`d_call`/`j_call` → the matching gene columns (`adc.py:102-119`). `study_id` is looked up from the per-repository `repertoires.json` so each row carries an iReceptor study identifier (`adc.py:121-138`).

**Column populations** (which TARGET_COLUMNS this standardizer fills meaningfully):
- `tra` or `trb` (junction_aa CDR3)
- `trav_gene` / `trbv_gene`, `trad_gene` / `trbd_gene`, `traj_gene` / `trbj_gene` (V/D/J gene calls)
- `source = "adc"`, `study_id` from `repertoires.json` (falls back to repository directory name)
- CDR1/CDR2 + `tra_full`/`trb_full` are populated downstream by `enrich_cdr_columns()` when V/J genes are recognised
- No peptide, MHC, binding, or score — ADC is bulk repertoire data with no epitope labels

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_standardized_manifest.py) -->

- **Path**: `data/standardized_again/adc`
- **Standardized at**: 2026-03-29T02:20:54.714720+00:00
- **Rows (manifest)**: 2,187,853,453
- **Dropped**: 126,625,882 (5.47% drop rate)
- **Standardization elapsed**: 134,432.5s
- **Source files (checksummed)**: 9296
- **Parquet**: 2188 files, 98.1G (105,281,699,204 bytes), 2,187,853,453 rows (matches manifest)

**Top 8 most-populated columns:**

| Column | Populated | Coverage | Unique |
|--------|-----------|----------|--------|
| `source` | 2,187,853,453 | 100.00% | n/a |
| `study_id` | 2,187,853,453 | 100.00% | 27 (sampled from 30/2188 shards) |
| `trbj_gene` | 2,112,561,158 | 96.56% | n/a |
| `trbv_gene` | 2,112,560,278 | 96.56% | n/a |
| `trb_cdr1` | 2,052,477,632 | 93.81% | n/a |
| `trb_cdr2` | 2,052,477,632 | 93.81% | n/a |
| `trb` | 1,989,720,397 | 90.94% | >1000000 |
| `trb_cdr3` | 1,989,720,397 | 90.94% | n/a |

**Bottom 5 least-populated columns:**

| Column | Populated | Coverage |
|--------|-----------|----------|
| `mhc_two` | 0 | 0.00% |
| `mhc_one_allele` | 0 | 0.00% |
| `mhc_two_allele` | 0 | 0.00% |
| `binding` | 0 | 0.00% |
| `score` | 0 | 0.00% |

**Top 10 drop reasons:**

| Reason | Count | Share |
|--------|-------|-------|
| `invalid_trb` | 108,314,570 | 85.54% |
| `invalid_tra` | 18,311,305 | 14.46% |
| `invalid_trbd_gene` | 7 | 0.00% |

<!-- END: AUTO-INVENTORY -->

## Validation drops

Of the 2.31B input rows, 126.6M (5.5%) were dropped. The two dominant reasons are `invalid_trb` (~108M) and `invalid_tra` (~18M) — these are CDR3 strings that fail the amino-acid character check or length floor (>= 4 AA) inside `quest/data/standardization.py:standardize_dataframe`. A handful of `invalid_trbd_gene` cases are AIRR rows where the D gene call could not be normalised. There is no `mhc_unresolved` because ADC carries no MHC.

The Quantifier auto-inventory above shows the per-reason counts. The `dropped.tsv` file has columns `reason | source_file | row_index | field | raw_value` — useful for debugging upstream data-quality issues.

## Caveats specific to the standardized output

- ADC is bulk repertoire only: every emitted row is either alpha-only or beta-only; no `tra`/`trb` pairing is reconstructed.
- `tra_full`/`trb_full` reconstruction relies on the AIRR `v_call`/`j_call` fields. Repositories with abbreviated or missing gene calls produce rows with empty `*_full` even when CDR3 is valid.
- 2,188 parquet shards (`part_NNNN.parquet`) — downstream consumers should glob the directory rather than reading a single file.
- `study_id` falls back to the repository directory name when `repertoires.json` is missing or unparseable; this means cross-source dedup against the per-study `studies` bucket may not catch overlap by ID.
- BCR contamination is blanked at the standardization-core layer (any row whose V/D/J contains an `IG*` gene has its CDR3 + genes zeroed; see `standardization.py:836-869`), but the row is not dropped — it just becomes empty for that chain.

## Use in QUEST training

ADC enters as bulk-repertoire evidence (per the raw-data fidelity rubric) and is consumed for masked-language-model training and TCR-only training modes (`tra`, `trb`, `tra_full`, `trb_full`). It is **not** used for TCR–pMHC interaction training because there are no epitope labels. See [raw-data ADC page](../../raw_data/databases/adc.md) for the fidelity tier.
