# NetMHCpan (standardized)

## Identity
- **Output path**: `data/standardized_again/netmhcpan/`
- **Standardizer**: [`scripts/data_processing/standardize/netmhcpan.py`](../../../../scripts/data_processing/standardize/netmhcpan.py) (`NetmhcpanStandardizer`)
- **Raw input**: see [raw-data manifest entry](../../raw_data/databases/netmhcpan.md) for citation, fidelity, and source provenance
- **Generated**: 2026-04-07T17:09:29Z
- **Standardization wall time**: 439.9s

## What this standardizer does

Two parallel sub-pipelines:

- **MHC-I (`NetMHCpan_train/`)**: reads space-separated `c*_ba` (binding-affinity) and `c*_el` (eluted-ligand) files line by line (`netmhcpan.py:103-162`). The last token of each row is a sample ID resolved via `allelelist` (`netmhcpan.py:35-48`).
- **MHC-II (`NetMHCIIpan_train/`)**: reads tab-separated `train_*.txt` and `test_*.txt` files (`netmhcpan.py:164-227`); the third column is the raw allele.

Allele resolution is centralised in `_resolve_allele` (`netmhcpan.py:51-89`). It handles three cases: (1) single HLA names (direct or via `allele_map`); (2) heterodimer notation `HLA-DPA10103-DPB10201` (one alpha + one beta) — `split_mhc_to_alpha_beta` returns the right buckets; (3) **multi-allele samples** — comma-separated lists from elution experiments are *ambiguous* (we don't know which allele presented the peptide), so the peptide is kept but `mhc_one` and `mhc_two` are blanked. Non-human prefixes (BoLA, DLA, SLA, H-2, Mamu) are filtered out by requiring the resolved string to begin with `HLA-` (`netmhcpan.py:81-84`). Files are processed in 1M-row chunks (`netmhcpan.py:32`).

`study_id` is the source filename (e.g., `c000_ba`, `train_BA1.txt`).

**Column populations** (which TARGET_COLUMNS this standardizer fills meaningfully):
- `peptide`
- `mhc_one`, `mhc_two`, `mhc_one_allele`, `mhc_two_allele` (empty for multi-allele samples)
- `source = "netmhcpan"`, `study_id` = file name
- No TCR fields, no `binding`, no `score` — the per-row IC50 / `target` value is **not** propagated

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_standardized_manifest.py) -->

- **Path**: `data/standardized_again/netmhcpan`
- **Standardized at**: 2026-04-07T17:09:29.673052+00:00
- **Rows (manifest)**: 34,034,376
- **Dropped**: 27,332 (0.08% drop rate)
- **Standardization elapsed**: 439.9s
- **Source files (checksummed)**: 36
- **Parquet**: 35 files, 491.5M (515,357,992 bytes), 34,034,376 rows (matches manifest)

**Top 8 most-populated columns:**

| Column | Populated | Coverage | Unique |
|--------|-----------|----------|--------|
| `peptide` | 34,034,376 | 100.00% | >1000000 |
| `source` | 34,034,376 | 100.00% | n/a |
| `study_id` | 34,034,376 | 100.00% | 30 (sampled from 30/35 shards) |
| `mhc_one_allele` | 4,686,953 | 13.77% | 190 (sampled from 30/35 shards) |
| `mhc_one` | 4,655,097 | 13.68% | n/a |
| `mhc_two` | 3,598,452 | 10.57% | n/a |
| `mhc_two_allele` | 3,598,452 | 10.57% | n/a |

**Bottom 5 least-populated columns:**

| Column | Populated | Coverage |
|--------|-----------|----------|
| `trb_cdr2` | 0 | 0.00% |
| `trb_cdr3` | 0 | 0.00% |
| `trb_full` | 0 | 0.00% |
| `binding` | 0 | 0.00% |
| `score` | 0 | 0.00% |

**Top 10 drop reasons:**

| Reason | Count | Share |
|--------|-------|-------|
| `mhc_unresolved` | 27,332 | 100.00% |

<!-- END: AUTO-INVENTORY -->

## Validation drops

27,332 dropped of 34M rows (~0.08%) — the only reason is `mhc_unresolved`. These are rows where the allele key was either absent from `allelelist` or resolved to a non-human / partially-typed identifier that survived the filter.

The Quantifier auto-inventory above shows the per-reason counts. The `dropped.tsv` file has columns `reason | source_file | row_index | field | raw_value` — useful for debugging upstream data-quality issues.

## Caveats specific to the standardized output

- **No `binding` and no `score`**. The numeric column in the input files (target value / qualitative `1` for ligand) is not propagated. Downstream consumers cannot reconstruct an IC50 threshold from this bucket.
- Multi-allele eluted-ligand rows survive but with empty MHC. They are useful as peptide-only training data (peptide LM) but not as pMHC pairs. Filter `mhc_one != "" or mhc_two != ""` if you need resolved-allele rows.
- Heavy overlap with `iedb_pmhc` — NetMHCpan was largely trained on IEDB-derived data. Cross-source dedup happens in the next pipeline stage.
- 35 parquet shards.

## Use in QUEST training

NetMHCpan is experimental peptide-MHC binding-affinity / elution data with no TCR. Per the integration framework it feeds peptide-MHC-only modeling (pMHC seq2seq, contrastive pMHC). It does **not** feed TCR interaction training. Because `binding`/`score` are not propagated, downstream code must treat all rows as positive presentation evidence (or re-derive labels from the raw files). See [raw-data NetMHCpan page](../../raw_data/databases/netmhcpan.md) for the fidelity tier.
