# tra_mhc_one_mhc_two

## Identity
- **Subset path (deduped)**: `data/deduplicated_again/exploded_deduped/subset_key=tra_mhc_one_mhc_two/`
- **Subset path (enriched)**: `data/deduplicated_again/exploded_deduped_enriched/subset_key=tra_mhc_one_mhc_two/`
- **Populated dedup columns**: `tra, mhc_one, mhc_two`
- **Carried TARGET_COLUMNS**: tra (CDR3 fallback), tra_full, tra_cdr1, tra_cdr2, tra_cdr3, mhc_one (MHC-I full sequence), mhc_two (MHC-II full sequence)
  (CDR1/2/3 columns are present whenever the parent chain is — see `tcrbench_dedup.py:46-50`)
- **Order_keys (orderings)**: 6 = 3!

## Counts
- **Unique records (`combination_counts.json`)**: n/a — this subset_key is a *projection* of one or more larger biological combinations; it appears in `permutation_counts.json` only.
- **Unique records (`permutation_counts.json`)**: 1,394
- **Total exploded rows** (`permutation_count × n!`): 8,364

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_deduplicated_manifest.py) -->

- **Populated columns** (3): `tra`, `mhc_one`, `mhc_two`
- **Combination count**: _not present in `combination_counts.json`_ (combination_counts only tracks 11 of the 31 subsets — those exposing all `populated_columns` in the same form as a unique key)
- **Permutation row count** (rows per order_key, from `permutation_counts.json`): 1,394
- **Total exploded rows expected**: 1,394 × 6 (3! permutations of 3 columns) = 8,364
- **Order keys** (6): `mhc_one_mhc_two_tra`, `mhc_one_tra_mhc_two`, `mhc_two_mhc_one_tra`, `mhc_two_tra_mhc_one`, `tra_mhc_one_mhc_two`, `tra_mhc_two_mhc_one`

**Deduped exploded** (`exploded_deduped/subset_key=tra_mhc_one_mhc_two/`):

- 6 order_keys · 6 parquet files · 369.8K (378,675 bytes) · 8,364 rows (matches expected)
- Schema: `tra_full`, `trb_full`, `peptide`, `mhc_one`, `mhc_two`, `tra_cdr1`, `tra_cdr2`, `tra_cdr3`, `trb_cdr1`, `trb_cdr2`, `trb_cdr3`, `sequence`

**Enriched exploded** (`exploded_deduped_enriched/subset_key=tra_mhc_one_mhc_two/`):

- 6 order_keys · 6 parquet files · 458.0K (468,976 bytes) · 8,364 rows (matches expected) · enriched row count == deduped row count
- Schema columns added vs deduped: `mhc_one_contact`, `mhc_one_pocket`, `mhc_one_pocket_contact`, `mhc_two_contact`, `mhc_two_pocket`, `mhc_two_pocket_contact`, `order_key`, `subset_key`

<!-- END: AUTO-INVENTORY -->

## Training task this feeds

Alpha TCR + both MHC contexts but no peptide. Edge subset; primarily an MHC-restriction probe.

## Notes

- This subset_key is **not** in `combination_counts.json` (only in `permutation_counts.json`) — it represents a projection of larger biological combinations onto the 3 listed column(s). For example, every record in any combination that contains `tra, mhc_one, mhc_two` as a strict subset of its populated columns also contributes one DISTINCT tuple to this subset_key.
- Sequence column for each `order_key=` is `CONCAT_WS(' ', <ordered_cols>)` — see `tcrbench_dedup.py:681-684`.
