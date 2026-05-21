# mhc_two

## Identity
- **Subset path (deduped)**: `data/deduplicated_again/exploded_deduped/subset_key=mhc_two/`
- **Subset path (enriched)**: `data/deduplicated_again/exploded_deduped_enriched/subset_key=mhc_two/`
- **Populated dedup columns**: `mhc_two`
- **Carried TARGET_COLUMNS**: mhc_two (MHC-II full sequence)
  (CDR1/2/3 columns are present whenever the parent chain is — see `tcrbench_dedup.py:46-50`)
- **Order_keys (orderings)**: 1 = 1!

## Counts
- **Unique records (`combination_counts.json`)**: n/a — this subset_key is a *projection* of one or more larger biological combinations; it appears in `permutation_counts.json` only.
- **Unique records (`permutation_counts.json`)**: 5,721
- **Total exploded rows** (`permutation_count × n!`): 5,721

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_deduplicated_manifest.py) -->

- **Populated columns** (1): `mhc_two`
- **Combination count**: _not present in `combination_counts.json`_ (combination_counts only tracks 11 of the 31 subsets — those exposing all `populated_columns` in the same form as a unique key)
- **Permutation row count** (rows per order_key, from `permutation_counts.json`): 5,721
- **Total exploded rows expected**: 5,721 × 1 (1! permutations of 1 columns) = 5,721
- **Order keys** (1): `mhc_two`

**Deduped exploded** (`exploded_deduped/subset_key=mhc_two/`):

- 1 order_keys · 1 parquet files · 129.8K (132,926 bytes) · 5,721 rows (matches expected)
- Schema: `tra_full`, `trb_full`, `peptide`, `mhc_one`, `mhc_two`, `tra_cdr1`, `tra_cdr2`, `tra_cdr3`, `trb_cdr1`, `trb_cdr2`, `trb_cdr3`, `mhc_one_allele`, `mhc_two_allele`, `sequence`

**Enriched exploded** (`exploded_deduped_enriched/subset_key=mhc_two/`):

- 1 order_keys · 1 parquet files · 178.7K (182,938 bytes) · 5,721 rows (matches expected) · enriched row count == deduped row count
- Schema columns added vs deduped: `mhc_one_contact`, `mhc_one_pocket`, `mhc_one_pocket_contact`, `mhc_two_contact`, `mhc_two_pocket`, `mhc_two_pocket_contact`, `order_key`, `subset_key`

<!-- END: AUTO-INVENTORY -->

## Training task this feeds

MLM pre-training on MHC class II full sequences (α+β concatenated where applicable). Very small (5.7K) — primarily an auxiliary corpus.

## Notes

- This subset_key is **not** in `combination_counts.json` (only in `permutation_counts.json`) — it represents a projection of larger biological combinations onto the 1 listed column(s). For example, every record in any combination that contains `mhc_two` as a strict subset of its populated columns also contributes one DISTINCT tuple to this subset_key.
- Sequence column for each `order_key=` is `CONCAT_WS(' ', <ordered_cols>)` — see `tcrbench_dedup.py:681-684`.
