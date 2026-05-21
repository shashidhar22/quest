# trb_mhc_two

## Identity
- **Subset path (deduped)**: `data/deduplicated_again/exploded_deduped/subset_key=trb_mhc_two/`
- **Subset path (enriched)**: `data/deduplicated_again/exploded_deduped_enriched/subset_key=trb_mhc_two/`
- **Populated dedup columns**: `trb, mhc_two`
- **Carried TARGET_COLUMNS**: trb (CDR3 fallback), trb_full, trb_cdr1, trb_cdr2, trb_cdr3, mhc_two (MHC-II full sequence)
  (CDR1/2/3 columns are present whenever the parent chain is — see `tcrbench_dedup.py:46-50`)
- **Order_keys (orderings)**: 2 = 2!

## Counts
- **Unique records (`combination_counts.json`)**: n/a — this subset_key is a *projection* of one or more larger biological combinations; it appears in `permutation_counts.json` only.
- **Unique records (`permutation_counts.json`)**: 4,156
- **Total exploded rows** (`permutation_count × n!`): 8,312

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_deduplicated_manifest.py) -->

- **Populated columns** (2): `trb`, `mhc_two`
- **Combination count**: _not present in `combination_counts.json`_ (combination_counts only tracks 11 of the 31 subsets — those exposing all `populated_columns` in the same form as a unique key)
- **Permutation row count** (rows per order_key, from `permutation_counts.json`): 4,156
- **Total exploded rows expected**: 4,156 × 2 (2! permutations of 2 columns) = 8,312
- **Order keys** (2): `mhc_two_trb`, `trb_mhc_two`

**Deduped exploded** (`exploded_deduped/subset_key=trb_mhc_two/`):

- 2 order_keys · 2 parquet files · 321.9K (329,639 bytes) · 8,312 rows (matches expected)
- Schema: `tra_full`, `trb_full`, `peptide`, `mhc_one`, `mhc_two`, `tra_cdr1`, `tra_cdr2`, `tra_cdr3`, `trb_cdr1`, `trb_cdr2`, `trb_cdr3`, `mhc_one_allele`, `mhc_two_allele`, `sequence`

**Enriched exploded** (`exploded_deduped_enriched/subset_key=trb_mhc_two/`):

- 2 order_keys · 2 parquet files · 385.4K (394,610 bytes) · 8,312 rows (matches expected) · enriched row count == deduped row count
- Schema columns added vs deduped: `mhc_one_contact`, `mhc_one_pocket`, `mhc_one_pocket_contact`, `mhc_two_contact`, `mhc_two_pocket`, `mhc_two_pocket_contact`, `order_key`, `subset_key`

<!-- END: AUTO-INVENTORY -->

## Training task this feeds

Beta + MHC-II context (no peptide). Small (4.2K).

## Notes

- This subset_key is **not** in `combination_counts.json` (only in `permutation_counts.json`) — it represents a projection of larger biological combinations onto the 2 listed column(s). For example, every record in any combination that contains `trb, mhc_two` as a strict subset of its populated columns also contributes one DISTINCT tuple to this subset_key.
- Sequence column for each `order_key=` is `CONCAT_WS(' ', <ordered_cols>)` — see `tcrbench_dedup.py:681-684`.
