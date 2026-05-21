# tra

## Identity
- **Subset path (deduped)**: `data/deduplicated_again/exploded_deduped/subset_key=tra/`
- **Subset path (enriched)**: `data/deduplicated_again/exploded_deduped_enriched/subset_key=tra/`
- **Populated dedup columns**: `tra`
- **Carried TARGET_COLUMNS**: tra (CDR3 fallback), tra_full, tra_cdr1, tra_cdr2, tra_cdr3
  (CDR1/2/3 columns are present whenever the parent chain is — see `tcrbench_dedup.py:46-50`)
- **Order_keys (orderings)**: 1 = 1!

## Counts
- **Unique records (`combination_counts.json`)**: 39,057,858
- **Unique records (`permutation_counts.json`)**: 39,057,858
- **Total exploded rows** (`permutation_count × n!`): 39,057,858

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_deduplicated_manifest.py) -->

- **Populated columns** (1): `tra`
- **Combination count** (canonical, from `combination_counts.json`): 39,057,858
- **Permutation row count** (rows per order_key, from `permutation_counts.json`): 39,057,858
- **Total exploded rows expected**: 39,057,858 × 1 (1! permutations of 1 columns) = 39,057,858
- **Order keys** (1): `tra`

**Deduped exploded** (`exploded_deduped/subset_key=tra/`):

- 1 order_keys · 64 parquet files · 1.1G (1,180,092,925 bytes) · 39,057,858 rows (matches expected)
- Schema: `tra_full`, `trb_full`, `peptide`, `mhc_one`, `mhc_two`, `tra_cdr1`, `tra_cdr2`, `tra_cdr3`, `trb_cdr1`, `trb_cdr2`, `trb_cdr3`, `mhc_one_allele`, `mhc_two_allele`, `sequence`

**Enriched exploded** (`exploded_deduped_enriched/subset_key=tra/`):

- 1 order_keys · 64 parquet files · 1.1G (1,226,487,498 bytes) · 39,057,858 rows (matches expected) · enriched row count == deduped row count
- Schema columns added vs deduped: `mhc_one_contact`, `mhc_one_pocket`, `mhc_one_pocket_contact`, `mhc_two_contact`, `mhc_two_pocket`, `mhc_two_pocket_contact`, `order_key`, `subset_key`

<!-- END: AUTO-INVENTORY -->

## Training task this feeds

MLM pre-training of ESM2 / TCR encoders on alpha-chain CDR3+full-length repertoire. Primary trainer: [`scripts/training/esm_native_trainer.py`](../../../../scripts/training/esm_native_trainer.py). Also serves as the TRA tower input for [`tcr_robust_contrastive_trainer.py`](../../../../scripts/training/tcr_robust_contrastive_trainer.py) alpha-chain pre-training.

## Notes

- `combination_counts.json` and `permutation_counts.json` agree at 39,057,858 — meaning every deduped row whose populated columns match this set has *no* additional populated columns (no records with these 1 columns plus extras).
- Sequence column for each `order_key=` is `CONCAT_WS(' ', <ordered_cols>)` — see `tcrbench_dedup.py:681-684`.
