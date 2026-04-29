# mhc_one

## Identity
- **Subset path (deduped)**: `data/deduplicated_again/exploded_deduped/subset_key=mhc_one/`
- **Subset path (enriched)**: `data/deduplicated_again/exploded_deduped_enriched/subset_key=mhc_one/`
- **Populated dedup columns**: `mhc_one`
- **Carried TARGET_COLUMNS**: mhc_one (MHC-I full sequence)
  (CDR1/2/3 columns are present whenever the parent chain is — see `tcrbench_dedup.py:46-50`)
- **Order_keys (orderings)**: 1 = 1!

## Counts
- **Unique records (`combination_counts.json`)**: n/a — this subset_key is a *projection* of one or more larger biological combinations; it appears in `permutation_counts.json` only.
- **Unique records (`permutation_counts.json`)**: 16,594
- **Total exploded rows** (`permutation_count × n!`): 16,594

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_deduplicated_manifest.py) -->

- **Populated columns** (1): `mhc_one`
- **Combination count**: _not present in `combination_counts.json`_ (combination_counts only tracks 11 of the 31 subsets — those exposing all `populated_columns` in the same form as a unique key)
- **Permutation row count** (rows per order_key, from `permutation_counts.json`): 16,594
- **Total exploded rows expected**: 16,594 × 1 (1! permutations of 1 columns) = 16,594
- **Order keys** (1): `mhc_one`

**Deduped exploded** (`exploded_deduped/subset_key=mhc_one/`):

- 1 order_keys · 1 parquet files · 554.9K (568,177 bytes) · 16,594 rows (matches expected)
- Schema: `tra_full`, `trb_full`, `peptide`, `mhc_one`, `mhc_two`, `tra_cdr1`, `tra_cdr2`, `tra_cdr3`, `trb_cdr1`, `trb_cdr2`, `trb_cdr3`, `sequence`

**Enriched exploded** (`exploded_deduped_enriched/subset_key=mhc_one/`):

- 1 order_keys · 1 parquet files · 747.9K (765,887 bytes) · 16,594 rows (matches expected) · enriched row count == deduped row count
- Schema columns added vs deduped: `mhc_one_contact`, `mhc_one_pocket`, `mhc_one_pocket_contact`, `mhc_two_contact`, `mhc_two_pocket`, `mhc_two_pocket_contact`, `order_key`, `subset_key`

<!-- END: AUTO-INVENTORY -->

## Training task this feeds

MLM pre-training on MHC class I full sequences. Small (16K unique HLA-I proteins) so usually combined with `peptide_mhc_one` for downstream pMHC training.

## Notes

- This subset_key is **not** in `combination_counts.json` (only in `permutation_counts.json`) — it represents a projection of larger biological combinations onto the 1 listed column(s). For example, every record in any combination that contains `mhc_one` as a strict subset of its populated columns also contributes one DISTINCT tuple to this subset_key.
- Sequence column for each `order_key=` is `CONCAT_WS(' ', <ordered_cols>)` — see `tcrbench_dedup.py:681-684`.
