# IEDB pMHC (standardized)

## Identity
- **Output path**: `data/standardized_again/iedb_pmhc/`
- **Standardizer**: [`scripts/data_processing/standardize/iedb_pmhc.py`](../../../../scripts/data_processing/standardize/iedb_pmhc.py) (`IedbPmhcStandardizer`)
- **Raw input**: see [raw-data manifest entry](../../raw_data/databases/iedb.md) for citation, fidelity, and source provenance
- **Generated**: 2026-03-26T19:23:04Z
- **Standardization wall time**: 645.6s

## What this standardizer does

Two complementary input sources are concatenated:
1. **Mass-spec eluted ligands** from `mhc_ligand/mhc_ligand_full_v3.csv`, parsed with the CEDAR multi-row-header reader and filtered to human organism (`iedb_pmhc.py:290-361`). Every ligand record is positive (`binding = "pos"`, `score = ""`).
2. **mhc_bind + mhc_elution** from the MySQL dump `full_database/iedb_public.sql.gz`. The standardizer line-streams the dump and parses `INSERT INTO ... VALUES (...)` for `mhc_bind`, `mhc_elution`, `curated_epitope`, and `epitope` tables via a hand-rolled MySQL VALUES tokenizer (`iedb_pmhc.py:51-104`, `_load_mhc_bind_from_mysql` at `iedb_pmhc.py:157-287`). `epitope.linear_peptide_seq` is preferred over `description` to avoid modification annotations like `SMYQTLLML + OX(M8)` (`iedb_pmhc.py:251-261`). For `mhc_bind`, `binding` is derived from the qualitative outcome (`Positive*` → `pos`, `Negative` → `neg`; `iedb_pmhc.py:39-48`, `_extract_records` at `iedb_pmhc.py:107-154`) and `score` is the numeric IC50/Kd. For `mhc_elution`, all rows are positive.

After concat, `mhc_allele` is split into `mhc_one` / `mhc_two` with a unique-then-map cache (`iedb_pmhc.py:396-400`).

**Column populations** (which TARGET_COLUMNS this standardizer fills meaningfully):
- `peptide`
- `mhc_one`, `mhc_two`, `mhc_one_allele`, `mhc_two_allele`
- `binding` (`pos`/`neg`/empty)
- `score` (numeric IC50/Kd from `mhc_bind`; empty for elution and ligand records)
- `source = "iedb_pmhc"`, `study_id` empty
- No TCR fields

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_standardized_manifest.py) -->

- **Path**: `data/standardized_again/iedb_pmhc`
- **Standardized at**: 2026-03-26T19:23:04.420085+00:00
- **Rows (manifest)**: 516,483
- **Dropped**: 6,176 (1.18% drop rate)
- **Standardization elapsed**: 645.6s
- **Source files (checksummed)**: 12
- **Parquet**: 1 files, 5.2M (5,413,265 bytes), 516,483 rows (matches manifest)

**Top 8 most-populated columns:**

| Column | Populated | Coverage | Unique |
|--------|-----------|----------|--------|
| `binding` | 516,483 | 100.00% | n/a |
| `source` | 516,483 | 100.00% | n/a |
| `peptide` | 513,958 | 99.51% | 303,190 |
| `mhc_one_allele` | 314,113 | 60.82% | 192 |
| `mhc_one` | 310,517 | 60.12% | n/a |
| `score` | 245,447 | 47.52% | n/a |
| `mhc_two_allele` | 108,312 | 20.97% | n/a |
| `mhc_two` | 108,257 | 20.96% | n/a |

**Bottom 5 least-populated columns:**

| Column | Populated | Coverage |
|--------|-----------|----------|
| `trb_cdr1` | 0 | 0.00% |
| `trb_cdr2` | 0 | 0.00% |
| `trb_cdr3` | 0 | 0.00% |
| `trb_full` | 0 | 0.00% |
| `study_id` | 0 | 0.00% |

**Top 10 drop reasons:**

| Reason | Count | Share |
|--------|-------|-------|
| `mhc_unresolved` | 3,651 | 59.12% |
| `invalid_peptide` | 2,525 | 40.88% |

<!-- END: AUTO-INVENTORY -->

## Validation drops

6,176 dropped of 522,659 rows (~1.2%): 3,651 `mhc_unresolved` (alleles that normalised to empty — e.g., partially-typed `HLA-A*01` with no second field, or non-human alleles surviving the upstream filter), 2,525 `invalid_peptide` (modification annotations not stripped, or non-standard AA characters).

The Quantifier auto-inventory above shows the per-reason counts. The `dropped.tsv` file has columns `reason | source_file | row_index | field | raw_value` — useful for debugging upstream data-quality issues.

## Caveats specific to the standardized output

- The MySQL parser is a custom regex+tokenizer pipeline; if a future IEDB dump renames columns or changes column ordering inside `mhc_bind`/`mhc_elution`, the hard-coded `allele_idx` (10 for bind, 36 for elution) silently breaks. Treat any sudden drop in `iedb_pmhc` row counts as a potential schema-shift signal.
- `score` semantics differ within the bucket: `mhc_bind` rows carry IC50/Kd as a string; `mhc_elution` and `mhc_ligand` rows have empty `score`. Downstream consumers must read `score` jointly with `binding`.
- Heavy overlap with `cedar_pmhc` and with `netmhcpan` (NetMHCpan was trained on IEDB-derived data). Cross-source dedup happens in the next pipeline stage.
- Multi-member gzip in the SQL dump triggers an `OSError`/`BadGzipFile` exit from the read loop (`iedb_pmhc.py:235-236`); rows that appear after the first gzip member would be lost — verify the dump is single-member.

## Use in QUEST training

Per the integration framework, IEDB pMHC is experimental peptide-MHC binding evidence with no TCR — it feeds peptide-MHC-only training (pMHC seq2seq, contrastive pMHC). It does **not** feed TCR interaction training. See [raw-data IEDB page](../../raw_data/databases/iedb.md) for the fidelity tier.
