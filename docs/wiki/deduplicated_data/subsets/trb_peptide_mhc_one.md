# trb_peptide_mhc_one

## Identity
- **Subset path (deduped)**: `data/deduplicated_again/exploded_deduped/subset_key=trb_peptide_mhc_one/`
- **Subset path (enriched)**: `data/deduplicated_again/exploded_deduped_enriched/subset_key=trb_peptide_mhc_one/`
- **Populated dedup columns**: `trb, peptide, mhc_one`
- **Carried TARGET_COLUMNS**: trb (CDR3 fallback), trb_full, trb_cdr1, trb_cdr2, trb_cdr3, peptide, mhc_one (MHC-I full sequence)
  (CDR1/2/3 columns are present whenever the parent chain is — see `tcrbench_dedup.py:46-50`)
- **Order_keys (orderings)**: 6 = 3!

## Counts
- **Unique records (`combination_counts.json`)**: 113,478
- **Unique records (`permutation_counts.json`)**: 113,478
- **Total exploded rows** (`permutation_count × n!`): 680,868

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_deduplicated_manifest.py) -->

- **Populated columns** (3): `trb`, `peptide`, `mhc_one`
- **Combination count** (canonical, from `combination_counts.json`): 113,478
- **Permutation row count** (rows per order_key, from `permutation_counts.json`): 113,478
- **Total exploded rows expected**: 113,478 × 6 (3! permutations of 3 columns) = 680,868
- **Order keys** (6): `mhc_one_peptide_trb`, `mhc_one_trb_peptide`, `peptide_mhc_one_trb`, `peptide_trb_mhc_one`, `trb_mhc_one_peptide`, `trb_peptide_mhc_one`

**Deduped exploded** (`exploded_deduped/subset_key=trb_peptide_mhc_one/`):

- 6 order_keys · 6 parquet files · 24.0M (25,135,161 bytes) · 680,868 rows (matches expected)
- Schema: `tra_full`, `trb_full`, `peptide`, `mhc_one`, `mhc_two`, `tra_cdr1`, `tra_cdr2`, `tra_cdr3`, `trb_cdr1`, `trb_cdr2`, `trb_cdr3`, `sequence`

**Enriched exploded** (`exploded_deduped_enriched/subset_key=trb_peptide_mhc_one/`):

- 6 order_keys · 6 parquet files · 26.2M (27,430,134 bytes) · 680,868 rows (matches expected) · enriched row count == deduped row count
- Schema columns added vs deduped: `mhc_one_contact`, `mhc_one_pocket`, `mhc_one_pocket_contact`, `mhc_two_contact`, `mhc_two_pocket`, `mhc_two_pocket_contact`, `order_key`, `subset_key`

<!-- END: AUTO-INVENTORY -->

## Training task this feeds

Beta TCR + pMHC-I interaction triple. Largest 3-molecule TCR-pMHC subset (113K). Primary input for [`tcr_cross_encoder_trainer.py`](../../../../scripts/training/tcr_cross_encoder_trainer.py) (β-only) and the `as_trb_i` benchmark task (`build_benchmark_splits.py:476`).

## Notes

- `combination_counts.json` and `permutation_counts.json` agree at 113,478 — meaning every deduped row whose populated columns match this set has *no* additional populated columns (no records with these 3 columns plus extras).
- Sequence column for each `order_key=` is `CONCAT_WS(' ', <ordered_cols>)` — see `tcrbench_dedup.py:681-684`.
