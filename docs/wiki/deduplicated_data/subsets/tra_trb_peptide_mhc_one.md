# tra_trb_peptide_mhc_one

## Identity
- **Subset path (deduped)**: `data/deduplicated_again/exploded_deduped/subset_key=tra_trb_peptide_mhc_one/`
- **Subset path (enriched)**: `data/deduplicated_again/exploded_deduped_enriched/subset_key=tra_trb_peptide_mhc_one/`
- **Populated dedup columns**: `tra, trb, peptide, mhc_one`
- **Carried TARGET_COLUMNS**: tra (CDR3 fallback), tra_full, tra_cdr1, tra_cdr2, tra_cdr3, trb (CDR3 fallback), trb_full, trb_cdr1, trb_cdr2, trb_cdr3, peptide, mhc_one (MHC-I full sequence)
  (CDR1/2/3 columns are present whenever the parent chain is — see `tcrbench_dedup.py:46-50`)
- **Order_keys (orderings)**: 24 = 4!

## Counts
- **Unique records (`combination_counts.json`)**: 57,079
- **Unique records (`permutation_counts.json`)**: 57,079
- **Total exploded rows** (`permutation_count × n!`): 1,369,896

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_deduplicated_manifest.py) -->

- **Populated columns** (4): `tra`, `trb`, `peptide`, `mhc_one`
- **Combination count** (canonical, from `combination_counts.json`): 57,079
- **Permutation row count** (rows per order_key, from `permutation_counts.json`): 57,079
- **Total exploded rows expected**: 57,079 × 24 (4! permutations of 4 columns) = 1,369,896
- **Order keys** (24): `mhc_one_peptide_tra_trb`, `mhc_one_peptide_trb_tra`, `mhc_one_tra_peptide_trb`, `mhc_one_tra_trb_peptide`, `mhc_one_trb_peptide_tra`, `mhc_one_trb_tra_peptide`, `peptide_mhc_one_tra_trb`, `peptide_mhc_one_trb_tra`, … (16 more)

**Deduped exploded** (`exploded_deduped/subset_key=tra_trb_peptide_mhc_one/`):

- 24 order_keys · 24 parquet files · 74.8M (78,479,363 bytes) · 1,369,896 rows (matches expected)
- Schema: `tra_full`, `trb_full`, `peptide`, `mhc_one`, `mhc_two`, `tra_cdr1`, `tra_cdr2`, `tra_cdr3`, `trb_cdr1`, `trb_cdr2`, `trb_cdr3`, `sequence`

**Enriched exploded** (`exploded_deduped_enriched/subset_key=tra_trb_peptide_mhc_one/`):

- 24 order_keys · 24 parquet files · 75.0M (78,650,708 bytes) · 1,369,896 rows (matches expected) · enriched row count == deduped row count
- Schema columns added vs deduped: `mhc_one_contact`, `mhc_one_pocket`, `mhc_one_pocket_contact`, `mhc_two_contact`, `mhc_two_pocket`, `mhc_two_pocket_contact`, `order_key`, `subset_key`

<!-- END: AUTO-INVENTORY -->

## Training task this feeds

Paired αβ TCR + pMHC-I — the classical full-context TCR-pMHC interaction record. 57K records. Primary input to the paired-chain class-I benchmark task `as_paired_i` (`build_benchmark_splits.py:478`) and the headline target for [`tcr_cross_encoder_trainer.py`](../../../../scripts/training/tcr_cross_encoder_trainer.py) full-paired mode.

## Notes

- `combination_counts.json` and `permutation_counts.json` agree at 57,079 — meaning every deduped row whose populated columns match this set has *no* additional populated columns (no records with these 4 columns plus extras).
- Sequence column for each `order_key=` is `CONCAT_WS(' ', <ordered_cols>)` — see `tcrbench_dedup.py:681-684`.
