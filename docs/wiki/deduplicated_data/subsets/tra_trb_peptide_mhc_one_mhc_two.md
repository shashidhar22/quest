# tra_trb_peptide_mhc_one_mhc_two

## Identity
- **Subset path (deduped)**: `data/deduplicated_again/exploded_deduped/subset_key=tra_trb_peptide_mhc_one_mhc_two/`
- **Subset path (enriched)**: `data/deduplicated_again/exploded_deduped_enriched/subset_key=tra_trb_peptide_mhc_one_mhc_two/`
- **Populated dedup columns**: `tra, trb, peptide, mhc_one, mhc_two`
- **Carried TARGET_COLUMNS**: tra (CDR3 fallback), tra_full, tra_cdr1, tra_cdr2, tra_cdr3, trb (CDR3 fallback), trb_full, trb_cdr1, trb_cdr2, trb_cdr3, peptide, mhc_one (MHC-I full sequence), mhc_two (MHC-II full sequence)
  (CDR1/2/3 columns are present whenever the parent chain is — see `tcrbench_dedup.py:46-50`)
- **Order_keys (orderings)**: 120 = 5!

## Counts
- **Unique records (`combination_counts.json`)**: 565
- **Unique records (`permutation_counts.json`)**: 565
- **Total exploded rows** (`permutation_count × n!`): 67,800

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_deduplicated_manifest.py) -->

- **Populated columns** (5): `tra`, `trb`, `peptide`, `mhc_one`, `mhc_two`
- **Combination count** (canonical, from `combination_counts.json`): 565
- **Permutation row count** (rows per order_key, from `permutation_counts.json`): 565
- **Total exploded rows expected**: 565 × 120 (5! permutations of 5 columns) = 67,800
- **Order keys** (120): `mhc_one_mhc_two_peptide_tra_trb`, `mhc_one_mhc_two_peptide_trb_tra`, `mhc_one_mhc_two_tra_peptide_trb`, `mhc_one_mhc_two_tra_trb_peptide`, `mhc_one_mhc_two_trb_peptide_tra`, `mhc_one_mhc_two_trb_tra_peptide`, `mhc_one_peptide_mhc_two_tra_trb`, `mhc_one_peptide_mhc_two_trb_tra`, … (112 more)

**Deduped exploded** (`exploded_deduped/subset_key=tra_trb_peptide_mhc_one_mhc_two/`):

- 120 order_keys · 120 parquet files · 6.9M (7,205,467 bytes) · 67,800 rows (matches expected)
- Schema: `tra_full`, `trb_full`, `peptide`, `mhc_one`, `mhc_two`, `tra_cdr1`, `tra_cdr2`, `tra_cdr3`, `trb_cdr1`, `trb_cdr2`, `trb_cdr3`, `sequence`

**Enriched exploded** (`exploded_deduped_enriched/subset_key=tra_trb_peptide_mhc_one_mhc_two/`):

- 120 order_keys · 120 parquet files · 8.0M (8,365,236 bytes) · 67,800 rows (matches expected) · enriched row count == deduped row count
- Schema columns added vs deduped: `mhc_one_contact`, `mhc_one_pocket`, `mhc_one_pocket_contact`, `mhc_two_contact`, `mhc_two_pocket`, `mhc_two_pocket_contact`, `order_key`, `subset_key`

<!-- END: AUTO-INVENTORY -->

## Training task this feeds

All five molecules populated — paired αβ TCR + peptide + both MHC classes. Tiniest subset (565 records) but the most information-rich. Folded into `as_paired_i` alongside `tra_trb_peptide_mhc_one` (`build_benchmark_splits.py:478`).

## Notes

- `combination_counts.json` and `permutation_counts.json` agree at 565 — meaning every deduped row whose populated columns match this set has *no* additional populated columns (no records with these 5 columns plus extras).
- Very small N (565) — likely insufficient for standalone training; combined with related subsets in benchmark splits where applicable.
- Sequence column for each `order_key=` is `CONCAT_WS(' ', <ordered_cols>)` — see `tcrbench_dedup.py:681-684`.
