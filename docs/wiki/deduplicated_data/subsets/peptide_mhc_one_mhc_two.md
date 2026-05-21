# peptide_mhc_one_mhc_two

## Identity
- **Subset path (deduped)**: `data/deduplicated_again/exploded_deduped/subset_key=peptide_mhc_one_mhc_two/`
- **Subset path (enriched)**: `data/deduplicated_again/exploded_deduped_enriched/subset_key=peptide_mhc_one_mhc_two/`
- **Populated dedup columns**: `peptide, mhc_one, mhc_two`
- **Carried TARGET_COLUMNS**: peptide, mhc_one (MHC-I full sequence), mhc_two (MHC-II full sequence)
  (CDR1/2/3 columns are present whenever the parent chain is — see `tcrbench_dedup.py:46-50`)
- **Order_keys (orderings)**: 6 = 3!

## Counts
- **Unique records (`combination_counts.json`)**: 292,468
- **Unique records (`permutation_counts.json`)**: 292,468
- **Total exploded rows** (`permutation_count × n!`): 1,754,808

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_deduplicated_manifest.py) -->

- **Populated columns** (3): `peptide`, `mhc_one`, `mhc_two`
- **Combination count** (canonical, from `combination_counts.json`): 292,468
- **Permutation row count** (rows per order_key, from `permutation_counts.json`): 292,468
- **Total exploded rows expected**: 292,468 × 6 (3! permutations of 3 columns) = 1,754,808
- **Order keys** (6): `mhc_one_mhc_two_peptide`, `mhc_one_peptide_mhc_two`, `mhc_two_mhc_one_peptide`, `mhc_two_peptide_mhc_one`, `peptide_mhc_one_mhc_two`, `peptide_mhc_two_mhc_one`

**Deduped exploded** (`exploded_deduped/subset_key=peptide_mhc_one_mhc_two/`):

- 6 order_keys · 18 parquet files · 54.0M (56,655,050 bytes) · 1,754,808 rows (matches expected)
- Schema: `tra_full`, `trb_full`, `peptide`, `mhc_one`, `mhc_two`, `tra_cdr1`, `tra_cdr2`, `tra_cdr3`, `trb_cdr1`, `trb_cdr2`, `trb_cdr3`, `mhc_one_allele`, `mhc_two_allele`, `sequence`

**Enriched exploded** (`exploded_deduped_enriched/subset_key=peptide_mhc_one_mhc_two/`):

- 6 order_keys · 18 parquet files · 62.0M (65,020,444 bytes) · 1,754,808 rows (matches expected) · enriched row count == deduped row count
- Schema columns added vs deduped: `mhc_one_contact`, `mhc_one_pocket`, `mhc_one_pocket_contact`, `mhc_two_contact`, `mhc_two_pocket`, `mhc_two_pocket_contact`, `order_key`, `subset_key`

<!-- END: AUTO-INVENTORY -->

## Training task this feeds

Records with both class assignments — used as an auxiliary set for pMHC contrastive learning where the peptide must distinguish class I vs class II contexts.

## Notes

- `combination_counts.json` and `permutation_counts.json` agree at 292,468 — meaning every deduped row whose populated columns match this set has *no* additional populated columns (no records with these 3 columns plus extras).
- Sequence column for each `order_key=` is `CONCAT_WS(' ', <ordered_cols>)` — see `tcrbench_dedup.py:681-684`.
