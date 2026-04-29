# McPAS-TCR (standardized)

## Identity
- **Output path**: `data/standardized_again/mcpas/`
- **Standardizer**: [`scripts/data_processing/standardize/mcpas.py`](../../../../scripts/data_processing/standardize/mcpas.py) (`McpasStandardizer`)
- **Raw input**: see [raw-data manifest entry](../../raw_data/databases/mcpas.md) for citation, fidelity, and source provenance
- **Generated**: 2026-03-26T19:12:18Z
- **Standardization wall time**: 5.7s

## What this standardizer does

Reads the single CSV `McPAS-TCR.csv` with `latin-1` encoding (`mcpas.py:60`). Filters to `Species == "human"` (`mcpas.py:62-66`). Exploded `Epitope.peptide` on `/` so multi-peptide rows become separate records (`mcpas.py:69-75`). The `MHC` column is split into `mhc_one`/`mhc_two` row-by-row via `split_mhc_to_alpha_beta` (`mcpas.py:77-86`). The standardizer derives `binding = "pos"` only for rows whose `Antigen.identification.method` is in the curated whitelist `_VALIDATED_METHODS` (tetramer, dextramer, multimer, stimulation, ELISpot, ICS, cytotoxicity; `mcpas.py:30-41`, applied at `mcpas.py:92-97`). Other rows leave `binding` empty (treated as weaker evidence by downstream consumers).

**Column populations** (which TARGET_COLUMNS this standardizer fills meaningfully):
- `tra` (`CDR3.alpha.aa`), `trav_gene` (`TRAV`), `traj_gene` (`TRAJ`)
- `trb` (`CDR3.beta.aa`), `trbv_gene` (`TRBV`), `trbd_gene` (`TRBD`), `trbj_gene` (`TRBJ`)
- `peptide` (post-explode), `mhc_one`, `mhc_two`
- `binding = "pos"` for tetramer/stimulation-validated rows, empty otherwise
- CDR1/CDR2 + `tra_full`/`trb_full` populated downstream
- `source = "mcpas"`, `study_id` empty
- No `score`

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_standardized_manifest.py) -->

- **Path**: `data/standardized_again/mcpas`
- **Standardized at**: 2026-03-26T19:12:18.123710+00:00
- **Rows (manifest)**: 36,433
- **Dropped**: 244 (0.67% drop rate)
- **Standardization elapsed**: 5.7s
- **Source files (checksummed)**: 2
- **Parquet**: 1 files, 1.9M (1,980,089 bytes), 36,433 rows (matches manifest)

**Top 8 most-populated columns:**

| Column | Populated | Coverage | Unique |
|--------|-----------|----------|--------|
| `source` | 36,433 | 100.00% | n/a |
| `trb` | 34,129 | 93.68% | 29,156 |
| `trb_cdr3` | 34,129 | 93.68% | n/a |
| `trbv_gene` | 32,920 | 90.36% | n/a |
| `trb_cdr1` | 31,765 | 87.19% | n/a |
| `trb_cdr2` | 31,765 | 87.19% | n/a |
| `trbj_gene` | 20,593 | 56.52% | n/a |
| `trb_full` | 19,589 | 53.77% | n/a |

**Bottom 5 least-populated columns:**

| Column | Populated | Coverage |
|--------|-----------|----------|
| `mhc_two_allele` | 151 | 0.41% |
| `trad_gene` | 0 | 0.00% |
| `binding` | 0 | 0.00% |
| `score` | 0 | 0.00% |
| `study_id` | 0 | 0.00% |

**Top 10 drop reasons:**

| Reason | Count | Share |
|--------|-------|-------|
| `invalid_trb` | 160 | 65.57% |
| `no_valid_field` | 41 | 16.80% |
| `invalid_tra` | 23 | 9.43% |
| `invalid_trbd_gene` | 11 | 4.51% |
| `invalid_trav_gene` | 3 | 1.23% |
| `mhc_unresolved` | 3 | 1.23% |
| `invalid_trbj_gene` | 2 | 0.82% |
| `invalid_traj_gene` | 1 | 0.41% |

<!-- END: AUTO-INVENTORY -->

## Validation drops

244 of 36,677 rows dropped (~0.7%): 160 `invalid_trb`, 41 `no_valid_field`, 23 `invalid_tra`, 11 `invalid_trbd_gene`, plus a handful of other gene normalization failures and 3 `mhc_unresolved`. Drop rate is consistent with curated literature data.

The Quantifier auto-inventory above shows the per-reason counts. The `dropped.tsv` file has columns `reason | source_file | row_index | field | raw_value` — useful for debugging upstream data-quality issues.

## Caveats specific to the standardized output

- The `binding = "pos"` derivation is a project decision driven by `_VALIDATED_METHODS`. Rows reported via `Antigen.identification.method` values not in that whitelist (e.g., "predicted", "computational", "homology") are emitted but left with empty `binding`. Edit `mcpas.py:30-41` to change the whitelist.
- McPAS is a curated literature aggregation that overlaps with VDJdb and IEDB. Cross-source dedup happens in the next pipeline stage.
- The `/`-exploded peptide rows duplicate the rest of the record verbatim: a paired-chain TCR with two reported peptides becomes two output rows.
- `study_id` is empty — McPAS records carry a `PubMed.ID` field but it is not propagated. To trace back to the source publication, join on raw `PubMed.ID` from `data/raw_data/databases/McPAS-TCR/McPAS-TCR.csv`.

## Use in QUEST training

McPAS is curated literature TCR–pMHC data; rows with `binding == "pos"` (~tetramer/multimer/stimulation evidence) are high-fidelity and feed all interaction-training modes (contrastive, cross-encoder, seq2seq). Rows with empty `binding` are still usable for MLM and as implicit positives. See [raw-data McPAS page](../../raw_data/databases/mcpas.md) for the fidelity tier.
