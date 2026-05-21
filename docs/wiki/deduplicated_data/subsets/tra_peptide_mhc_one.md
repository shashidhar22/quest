# tra_peptide_mhc_one

## Identity
- **Subset path (deduped)**: `data/deduplicated_again/exploded_deduped/subset_key=tra_peptide_mhc_one/`
- **Subset path (enriched)**: `data/deduplicated_again/exploded_deduped_enriched/subset_key=tra_peptide_mhc_one/`
- **Populated dedup columns**: `tra, peptide, mhc_one`
- **Carried TARGET_COLUMNS**: tra (CDR3 fallback), tra_full, tra_cdr1, tra_cdr2, tra_cdr3, peptide, mhc_one (MHC-I full sequence)
  (CDR1/2/3 columns are present whenever the parent chain is — see `tcrbench_dedup.py:46-50`)
- **Order_keys (orderings)**: 6 = 3!

## Counts
- **Unique records (`combination_counts.json`)**: 76,804
- **Unique records (`permutation_counts.json`)**: 76,804
- **Total exploded rows** (`permutation_count × n!`): 460,824

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_deduplicated_manifest.py) -->

- **Populated columns** (3): `tra`, `peptide`, `mhc_one`
- **Combination count** (canonical, from `combination_counts.json`): 76,804
- **Permutation row count** (rows per order_key, from `permutation_counts.json`): 76,804
- **Total exploded rows expected**: 76,804 × 6 (3! permutations of 3 columns) = 460,824
- **Order keys** (6): `mhc_one_peptide_tra`, `mhc_one_tra_peptide`, `peptide_mhc_one_tra`, `peptide_tra_mhc_one`, `tra_mhc_one_peptide`, `tra_peptide_mhc_one`

**Deduped exploded** (`exploded_deduped/subset_key=tra_peptide_mhc_one/`):

- 6 order_keys · 6 parquet files · 14.8M (15,507,820 bytes) · 460,824 rows (matches expected)
- Schema: `tra_full`, `trb_full`, `peptide`, `mhc_one`, `mhc_two`, `tra_cdr1`, `tra_cdr2`, `tra_cdr3`, `trb_cdr1`, `trb_cdr2`, `trb_cdr3`, `mhc_one_allele`, `mhc_two_allele`, `sequence`

**Enriched exploded** (`exploded_deduped_enriched/subset_key=tra_peptide_mhc_one/`):

- 6 order_keys · 6 parquet files · 16.9M (17,679,179 bytes) · 460,824 rows (matches expected) · enriched row count == deduped row count
- Schema columns added vs deduped: `mhc_one_contact`, `mhc_one_pocket`, `mhc_one_pocket_contact`, `mhc_two_contact`, `mhc_two_pocket`, `mhc_two_pocket_contact`, `order_key`, `subset_key`

<!-- END: AUTO-INVENTORY -->

## Training task this feeds

Alpha TCR + pMHC-I full interaction triple. Primary input for cross-encoder and contrastive training of alpha-only TCR-pMHC-I models. Trainers: [`tcr_cross_encoder_trainer.py`](../../../../scripts/training/tcr_cross_encoder_trainer.py), [`tcr_robust_contrastive_trainer.py`](../../../../scripts/training/tcr_robust_contrastive_trainer.py), and the alpha-chain task `as_tra_i` in [`build_benchmark_splits.py`](../../../../scripts/data_processing/build_benchmark_splits.py).

## Notes

- `combination_counts.json` and `permutation_counts.json` agree at 76,804 — meaning every deduped row whose populated columns match this set has *no* additional populated columns (no records with these 3 columns plus extras).
- Sequence column for each `order_key=` is `CONCAT_WS(' ', <ordered_cols>)` — see `tcrbench_dedup.py:681-684`.
