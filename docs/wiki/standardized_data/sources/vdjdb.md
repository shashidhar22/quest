# VDJdb (standardized)

## Identity
- **Output path**: `data/standardized_again/vdjdb/`
- **Standardizer**: [`scripts/data_processing/standardize/vdjdb.py`](../../../../scripts/data_processing/standardize/vdjdb.py) (`VdjdbStandardizer`)
- **Raw input**: see [raw-data manifest entry](../../raw_data/databases/vdjdb.md) for citation, fidelity, and source provenance
- **Generated**: 2026-03-26T19:12:20Z
- **Standardization wall time**: 5.6s

## What this standardizer does

Reads `vdjdb_full.txt` (wide-format paired alpha/beta), falling back to `vdjdb.txt` if absent (`vdjdb.py:69-75`). Filters to `species == "HomoSapiens"` (`vdjdb.py:79-83`). MHC fields are normalised: `mhc.a` → `mhc_one`, `mhc.b` → `mhc_two` (via `normalize_mhc_allele`; `vdjdb.py:85-91`).

The interesting logic is the **score-based split** (`vdjdb.py:93-137`). VDJdb assigns a `vdjdb.score` of 0–3 per record reflecting evidence quality. The standardizer uses three column maps:
- `COLUMN_MAP_FULL` (`vdjdb.py:29-42`): TCR + peptide + MHC + score + study_id, applied to score >= 1 rows.
- `COLUMN_MAP_TCR` (`vdjdb.py:45-55`): TCR + score + study_id only, applied to score-0 rows that have a CDR3.
- `COLUMN_MAP_PMHC` (`vdjdb.py:58-64`): peptide + MHC + score + study_id only, applied to score-0 rows that have peptide or MHC.

Score-0 rows therefore appear **twice** in the output: once as TCR-only (for MLM / TCR-only training) and once as pMHC-only (for pMHC modeling). Missing/empty scores are coerced to 1 (treated as trusted; `vdjdb.py:94-97`).

**Column populations** (which TARGET_COLUMNS this standardizer fills meaningfully):
- `tra`, `trav_gene`, `traj_gene` (no `trad_gene` — alpha has no D)
- `trb`, `trbv_gene`, `trbd_gene`, `trbj_gene`
- `peptide`, `mhc_one`, `mhc_two`, `mhc_one_allele`, `mhc_two_allele` (only on score >= 1 and pMHC-only score-0 rows)
- `score` = vdjdb.score
- `study_id` from `meta.study.id`
- CDR1/CDR2 + `tra_full`/`trb_full` populated downstream
- `source = "vdjdb"`. No `binding` column populated — every emitted record is implicitly positive (the cognate epitope is the labelled epitope).

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_standardized_manifest.py) -->

- **Path**: `data/standardized_again/vdjdb`
- **Standardized at**: 2026-03-26T19:12:20.043836+00:00
- **Rows (manifest)**: 237,578
- **Dropped**: 679 (0.28% drop rate)
- **Standardization elapsed**: 5.6s
- **Source files (checksummed)**: 19
- **Parquet**: 3 files, 9.5M (9,924,882 bytes), 237,578 rows (matches manifest)

**Top 8 most-populated columns:**

| Column | Populated | Coverage | Unique |
|--------|-----------|----------|--------|
| `score` | 237,578 | 100.00% | n/a |
| `source` | 237,578 | 100.00% | n/a |
| `peptide` | 126,137 | 53.09% | 1,705 |
| `trb` | 113,478 | 47.76% | 83,736 |
| `trb_cdr3` | 113,478 | 47.76% | n/a |
| `trbj_gene` | 113,475 | 47.76% | n/a |
| `trbv_gene` | 113,389 | 47.73% | n/a |
| `trb_full` | 113,358 | 47.71% | n/a |

**Bottom 5 least-populated columns:**

| Column | Populated | Coverage |
|--------|-----------|----------|
| `study_id` | 14,107 | 5.94% |
| `mhc_two_allele` | 2,441 | 1.03% |
| `mhc_two` | 1,924 | 0.81% |
| `trad_gene` | 0 | 0.00% |
| `binding` | 0 | 0.00% |

**Top 10 drop reasons:**

| Reason | Count | Share |
|--------|-------|-------|
| `mhc_unresolved` | 679 | 100.00% |

<!-- END: AUTO-INVENTORY -->

## Validation drops

679 of 238,257 rows dropped (~0.28%) — all `mhc_unresolved`. These are MHC strings that survived `normalize_mhc_allele` but couldn't be resolved against the IMGT/HLA reference (typically partial typing or non-standard allele names).

The Quantifier auto-inventory above shows the per-reason counts. The `dropped.tsv` file has columns `reason | source_file | row_index | field | raw_value` — useful for debugging upstream data-quality issues.

## Caveats specific to the standardized output

- The score-0 split-and-duplicate path is intentional but **inflates row count**: a score-0 record with both TCR and pMHC fields appears twice in this bucket (once as TCR-only, once as pMHC-only). Cross-source dedup or downstream consumers must be aware.
- `binding` is **never populated** — every emitted row is implicitly a positive TCR–pMHC association. Downstream filters that select `binding == "pos"` will drop VDJdb entirely; use `source == "vdjdb"` instead.
- `score` is the VDJdb-specific 0–3 quality score, not an affinity. It differs in semantics from `iedb_pmhc.score` (IC50/Kd) and `batman.score` (peptide_activity 0–1). Downstream code must read `score` jointly with `source`.
- VDJdb partially overlaps IEDB and McPAS. Cross-source dedup runs in the next pipeline stage.
- 3 parquet shards (the score-split branch produces 1–3 yields, each becoming its own part).

## Use in QUEST training

Per the integration framework, VDJdb is "Quality-scored with threshold". Score >= 1 rows feed MLM, contrastive, cross-encoder, and seq2seq interaction training. Score-0 rows are routed to MLM-only via the dual TCR-only / pMHC-only output (TCR-only rows are bulk-quality TCRs; pMHC-only rows are unattested binding evidence). See [raw-data VDJdb page](../../raw_data/databases/vdjdb.md) for the fidelity tier.
