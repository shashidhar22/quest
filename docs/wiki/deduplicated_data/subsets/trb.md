# trb

## Identity
- **Subset path (deduped)**: `data/deduplicated_again/exploded_deduped/subset_key=trb/`
- **Subset path (enriched)**: `data/deduplicated_again/exploded_deduped_enriched/subset_key=trb/`
- **Populated dedup columns**: `trb`
- **Carried TARGET_COLUMNS**: trb (CDR3 fallback), trb_full, trb_cdr1, trb_cdr2, trb_cdr3
  (CDR1/2/3 columns are present whenever the parent chain is — see `tcrbench_dedup.py:46-50`)
- **Order_keys (orderings)**: 1 = 1!

## Counts
- **Unique records (`combination_counts.json`)**: 1,380,428,803
- **Unique records (`permutation_counts.json`)**: 1,380,428,803
- **Total exploded rows** (`permutation_count × n!`): 1,380,428,803

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_deduplicated_manifest.py) -->

- **Populated columns** (1): `trb`
- **Combination count** (canonical, from `combination_counts.json`): 1,380,428,803
- **Permutation row count** (rows per order_key, from `permutation_counts.json`): 1,380,428,803
- **Total exploded rows expected**: 1,380,428,803 × 1 (1! permutations of 1 columns) = 1,380,428,803
- **Order keys** (1): `trb`

**Deduped exploded** (`exploded_deduped/subset_key=trb/`):

- 1 order_keys · 64 parquet files · 43.0G (46,178,053,098 bytes) · 1,380,428,803 rows (matches expected)
- Schema: `tra_full`, `trb_full`, `peptide`, `mhc_one`, `mhc_two`, `tra_cdr1`, `tra_cdr2`, `tra_cdr3`, `trb_cdr1`, `trb_cdr2`, `trb_cdr3`, `sequence`

**Enriched exploded** (`exploded_deduped_enriched/subset_key=trb/`):

- 1 order_keys · 64 parquet files · 42.6G (45,700,817,886 bytes) · 1,380,428,803 rows (matches expected) · enriched row count == deduped row count
- Schema columns added vs deduped: `mhc_one_contact`, `mhc_one_pocket`, `mhc_one_pocket_contact`, `mhc_two_contact`, `mhc_two_pocket`, `mhc_two_pocket_contact`, `order_key`, `subset_key`

<!-- END: AUTO-INVENTORY -->

## Training task this feeds

MLM pre-training of ESM2 / TCR encoders on beta-chain CDR3+full-length repertoire — by far the largest single MLM corpus (1.38B unique TRBs). Primary trainer: [`scripts/training/esm_native_trainer.py`](../../../../scripts/training/esm_native_trainer.py). Also feeds the TRB tower of [`tcr_robust_contrastive_trainer.py`](../../../../scripts/training/tcr_robust_contrastive_trainer.py).

## Notes

- `combination_counts.json` and `permutation_counts.json` agree at 1,380,428,803 — meaning every deduped row whose populated columns match this set has *no* additional populated columns (no records with these 1 columns plus extras).
- Large exploded volume (1,380,428,803 rows). On disk this is the dominant component of `exploded_deduped/`.
- Sequence column for each `order_key=` is `CONCAT_WS(' ', <ordered_cols>)` — see `tcrbench_dedup.py:681-684`.
