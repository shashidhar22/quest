# tra_trb

## Identity
- **Subset path (deduped)**: `data/deduplicated_again/exploded_deduped/subset_key=tra_trb/`
- **Subset path (enriched)**: `data/deduplicated_again/exploded_deduped_enriched/subset_key=tra_trb/`
- **Populated dedup columns**: `tra, trb`
- **Carried TARGET_COLUMNS**: tra (CDR3 fallback), tra_full, tra_cdr1, tra_cdr2, tra_cdr3, trb (CDR3 fallback), trb_full, trb_cdr1, trb_cdr2, trb_cdr3
  (CDR1/2/3 columns are present whenever the parent chain is — see `tcrbench_dedup.py:46-50`)
- **Order_keys (orderings)**: 2 = 2!

## Counts
- **Unique records (`combination_counts.json`)**: 3,197,312
- **Unique records (`permutation_counts.json`)**: 3,197,312
- **Total exploded rows** (`permutation_count × n!`): 6,394,624

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_deduplicated_manifest.py) -->

- **Populated columns** (2): `tra`, `trb`
- **Combination count** (canonical, from `combination_counts.json`): 3,197,312
- **Permutation row count** (rows per order_key, from `permutation_counts.json`): 3,197,312
- **Total exploded rows expected**: 3,197,312 × 2 (2! permutations of 2 columns) = 6,394,624
- **Order keys** (2): `tra_trb`, `trb_tra`

**Deduped exploded** (`exploded_deduped/subset_key=tra_trb/`):

- 2 order_keys · 53 parquet files · 330.4M (346,427,416 bytes) · 6,394,624 rows (matches expected)
- Schema: `tra_full`, `trb_full`, `peptide`, `mhc_one`, `mhc_two`, `tra_cdr1`, `tra_cdr2`, `tra_cdr3`, `trb_cdr1`, `trb_cdr2`, `trb_cdr3`, `sequence`

**Enriched exploded** (`exploded_deduped_enriched/subset_key=tra_trb/`):

- 2 order_keys · 53 parquet files · 328.2M (344,095,429 bytes) · 6,394,624 rows (matches expected) · enriched row count == deduped row count
- Schema columns added vs deduped: `mhc_one_contact`, `mhc_one_pocket`, `mhc_one_pocket_contact`, `mhc_two_contact`, `mhc_two_pocket`, `mhc_two_pocket_contact`, `order_key`, `subset_key`

<!-- END: AUTO-INVENTORY -->

## Training task this feeds

Paired alpha+beta MLM (concatenated `[TRA] CDR3α [ETRA] [TRB] CDR3β [ETRB]` sequences) and TCR-pairing contrastive pre-training. Trainers: [`esm_native_trainer.py`](../../../../scripts/training/esm_native_trainer.py) (MLM mode) and [`tcr_robust_contrastive_trainer.py`](../../../../scripts/training/tcr_robust_contrastive_trainer.py) (TRA↔TRB dual encoder).

## Notes

- `combination_counts.json` and `permutation_counts.json` agree at 3,197,312 — meaning every deduped row whose populated columns match this set has *no* additional populated columns (no records with these 2 columns plus extras).
- Sequence column for each `order_key=` is `CONCAT_WS(' ', <ordered_cols>)` — see `tcrbench_dedup.py:681-684`.
