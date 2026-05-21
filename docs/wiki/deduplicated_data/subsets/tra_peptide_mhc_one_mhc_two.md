# tra_peptide_mhc_one_mhc_two

## Identity
- **Subset path (deduped)**: `data/deduplicated_again/exploded_deduped/subset_key=tra_peptide_mhc_one_mhc_two/`
- **Subset path (enriched)**: `data/deduplicated_again/exploded_deduped_enriched/subset_key=tra_peptide_mhc_one_mhc_two/`
- **Populated dedup columns**: `tra, peptide, mhc_one, mhc_two`
- **Carried TARGET_COLUMNS**: tra (CDR3 fallback), tra_full, tra_cdr1, tra_cdr2, tra_cdr3, peptide, mhc_one (MHC-I full sequence), mhc_two (MHC-II full sequence)
  (CDR1/2/3 columns are present whenever the parent chain is — see `tcrbench_dedup.py:46-50`)
- **Order_keys (orderings)**: 24 = 4!

## Counts
- **Unique records (`combination_counts.json`)**: 938
- **Unique records (`permutation_counts.json`)**: 938
- **Total exploded rows** (`permutation_count × n!`): 22,512

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_deduplicated_manifest.py) -->

- **Populated columns** (4): `tra`, `peptide`, `mhc_one`, `mhc_two`
- **Combination count** (canonical, from `combination_counts.json`): 938
- **Permutation row count** (rows per order_key, from `permutation_counts.json`): 938
- **Total exploded rows expected**: 938 × 24 (4! permutations of 4 columns) = 22,512
- **Order keys** (24): `mhc_one_mhc_two_peptide_tra`, `mhc_one_mhc_two_tra_peptide`, `mhc_one_peptide_mhc_two_tra`, `mhc_one_peptide_tra_mhc_two`, `mhc_one_tra_mhc_two_peptide`, `mhc_one_tra_peptide_mhc_two`, `mhc_two_mhc_one_peptide_tra`, `mhc_two_mhc_one_tra_peptide`, … (16 more)

**Deduped exploded** (`exploded_deduped/subset_key=tra_peptide_mhc_one_mhc_two/`):

- 24 order_keys · 24 parquet files · 1.2M (1,211,465 bytes) · 22,512 rows (matches expected)
- Schema: `tra_full`, `trb_full`, `peptide`, `mhc_one`, `mhc_two`, `tra_cdr1`, `tra_cdr2`, `tra_cdr3`, `trb_cdr1`, `trb_cdr2`, `trb_cdr3`, `mhc_one_allele`, `mhc_two_allele`, `sequence`

**Enriched exploded** (`exploded_deduped_enriched/subset_key=tra_peptide_mhc_one_mhc_two/`):

- 24 order_keys · 24 parquet files · 1.4M (1,481,230 bytes) · 22,512 rows (matches expected) · enriched row count == deduped row count
- Schema columns added vs deduped: `mhc_one_contact`, `mhc_one_pocket`, `mhc_one_pocket_contact`, `mhc_two_contact`, `mhc_two_pocket`, `mhc_two_pocket_contact`, `order_key`, `subset_key`

<!-- END: AUTO-INVENTORY -->

## Training task this feeds

Alpha TCR + peptide + both MHC classes. Very small (938 records) — used in benchmark task `as_tra_i` alongside `tra_peptide_mhc_one` (`build_benchmark_splits.py:477`).

## Notes

- `combination_counts.json` and `permutation_counts.json` agree at 938 — meaning every deduped row whose populated columns match this set has *no* additional populated columns (no records with these 4 columns plus extras).
- Very small N (938) — likely insufficient for standalone training; combined with related subsets in benchmark splits where applicable.
- Sequence column for each `order_key=` is `CONCAT_WS(' ', <ordered_cols>)` — see `tcrbench_dedup.py:681-684`.
