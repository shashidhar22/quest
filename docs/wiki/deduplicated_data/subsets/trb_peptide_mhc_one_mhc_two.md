# trb_peptide_mhc_one_mhc_two

## Identity
- **Subset path (deduped)**: `data/deduplicated_again/exploded_deduped/subset_key=trb_peptide_mhc_one_mhc_two/`
- **Subset path (enriched)**: `data/deduplicated_again/exploded_deduped_enriched/subset_key=trb_peptide_mhc_one_mhc_two/`
- **Populated dedup columns**: `trb, peptide, mhc_one, mhc_two`
- **Carried TARGET_COLUMNS**: trb (CDR3 fallback), trb_full, trb_cdr1, trb_cdr2, trb_cdr3, peptide, mhc_one (MHC-I full sequence), mhc_two (MHC-II full sequence)
  (CDR1/2/3 columns are present whenever the parent chain is — see `tcrbench_dedup.py:46-50`)
- **Order_keys (orderings)**: 24 = 4!

## Counts
- **Unique records (`combination_counts.json`)**: 1,064
- **Unique records (`permutation_counts.json`)**: 1,064
- **Total exploded rows** (`permutation_count × n!`): 25,536

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_deduplicated_manifest.py) -->

- **Populated columns** (4): `trb`, `peptide`, `mhc_one`, `mhc_two`
- **Combination count** (canonical, from `combination_counts.json`): 1,064
- **Permutation row count** (rows per order_key, from `permutation_counts.json`): 1,064
- **Total exploded rows expected**: 1,064 × 24 (4! permutations of 4 columns) = 25,536
- **Order keys** (24): `mhc_one_mhc_two_peptide_trb`, `mhc_one_mhc_two_trb_peptide`, `mhc_one_peptide_mhc_two_trb`, `mhc_one_peptide_trb_mhc_two`, `mhc_one_trb_mhc_two_peptide`, `mhc_one_trb_peptide_mhc_two`, `mhc_two_mhc_one_peptide_trb`, `mhc_two_mhc_one_trb_peptide`, … (16 more)

**Deduped exploded** (`exploded_deduped/subset_key=trb_peptide_mhc_one_mhc_two/`):

- 24 order_keys · 24 parquet files · 1.3M (1,324,229 bytes) · 25,536 rows (matches expected)
- Schema: `tra_full`, `trb_full`, `peptide`, `mhc_one`, `mhc_two`, `tra_cdr1`, `tra_cdr2`, `tra_cdr3`, `trb_cdr1`, `trb_cdr2`, `trb_cdr3`, `mhc_one_allele`, `mhc_two_allele`, `sequence`

**Enriched exploded** (`exploded_deduped_enriched/subset_key=trb_peptide_mhc_one_mhc_two/`):

- 24 order_keys · 24 parquet files · 1.5M (1,613,694 bytes) · 25,536 rows (matches expected) · enriched row count == deduped row count
- Schema columns added vs deduped: `mhc_one_contact`, `mhc_one_pocket`, `mhc_one_pocket_contact`, `mhc_two_contact`, `mhc_two_pocket`, `mhc_two_pocket_contact`, `order_key`, `subset_key`

<!-- END: AUTO-INVENTORY -->

## Training task this feeds

Beta TCR + peptide + both MHC classes. 1064 records — folded into `as_trb_i` together with `trb_peptide_mhc_one` (`build_benchmark_splits.py:476`).

## Notes

- `combination_counts.json` and `permutation_counts.json` agree at 1,064 — meaning every deduped row whose populated columns match this set has *no* additional populated columns (no records with these 4 columns plus extras).
- Sequence column for each `order_key=` is `CONCAT_WS(' ', <ordered_cols>)` — see `tcrbench_dedup.py:681-684`.
