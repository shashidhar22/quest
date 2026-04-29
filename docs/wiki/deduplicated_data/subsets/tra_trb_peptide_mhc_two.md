# tra_trb_peptide_mhc_two

## Identity
- **Subset path (deduped)**: `data/deduplicated_again/exploded_deduped/subset_key=tra_trb_peptide_mhc_two/`
- **Subset path (enriched)**: `data/deduplicated_again/exploded_deduped_enriched/subset_key=tra_trb_peptide_mhc_two/`
- **Populated dedup columns**: `tra, trb, peptide, mhc_two`
- **Carried TARGET_COLUMNS**: tra (CDR3 fallback), tra_full, tra_cdr1, tra_cdr2, tra_cdr3, trb (CDR3 fallback), trb_full, trb_cdr1, trb_cdr2, trb_cdr3, peptide, mhc_two (MHC-II full sequence)
  (CDR1/2/3 columns are present whenever the parent chain is — see `tcrbench_dedup.py:46-50`)
- **Order_keys (orderings)**: 24 = 4!

## Counts
- **Unique records (`combination_counts.json`)**: n/a — this subset_key is a *projection* of one or more larger biological combinations; it appears in `permutation_counts.json` only.
- **Unique records (`permutation_counts.json`)**: 3,493
- **Total exploded rows** (`permutation_count × n!`): 83,832

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_deduplicated_manifest.py) -->

- **Populated columns** (4): `tra`, `trb`, `peptide`, `mhc_two`
- **Combination count**: _not present in `combination_counts.json`_ (combination_counts only tracks 11 of the 31 subsets — those exposing all `populated_columns` in the same form as a unique key)
- **Permutation row count** (rows per order_key, from `permutation_counts.json`): 3,493
- **Total exploded rows expected**: 3,493 × 24 (4! permutations of 4 columns) = 83,832
- **Order keys** (24): `mhc_two_peptide_tra_trb`, `mhc_two_peptide_trb_tra`, `mhc_two_tra_peptide_trb`, `mhc_two_tra_trb_peptide`, `mhc_two_trb_peptide_tra`, `mhc_two_trb_tra_peptide`, `peptide_mhc_two_tra_trb`, `peptide_mhc_two_trb_tra`, … (16 more)

**Deduped exploded** (`exploded_deduped/subset_key=tra_trb_peptide_mhc_two/`):

- 24 order_keys · 24 parquet files · 4.3M (4,475,746 bytes) · 83,832 rows (matches expected)
- Schema: `tra_full`, `trb_full`, `peptide`, `mhc_one`, `mhc_two`, `tra_cdr1`, `tra_cdr2`, `tra_cdr3`, `trb_cdr1`, `trb_cdr2`, `trb_cdr3`, `sequence`

**Enriched exploded** (`exploded_deduped_enriched/subset_key=tra_trb_peptide_mhc_two/`):

- 24 order_keys · 24 parquet files · 4.3M (4,553,221 bytes) · 83,832 rows (matches expected) · enriched row count == deduped row count
- Schema columns added vs deduped: `mhc_one_contact`, `mhc_one_pocket`, `mhc_one_pocket_contact`, `mhc_two_contact`, `mhc_two_pocket`, `mhc_two_pocket_contact`, `order_key`, `subset_key`

<!-- END: AUTO-INVENTORY -->

## Training task this feeds

Paired αβ TCR + pMHC-II. Drives benchmark task `as_paired_ii` (`build_benchmark_splits.py:483`).

## Notes

- This subset_key is **not** in `combination_counts.json` (only in `permutation_counts.json`) — it represents a projection of larger biological combinations onto the 4 listed column(s). For example, every record in any combination that contains `tra, trb, peptide, mhc_two` as a strict subset of its populated columns also contributes one DISTINCT tuple to this subset_key.
- Sequence column for each `order_key=` is `CONCAT_WS(' ', <ordered_cols>)` — see `tcrbench_dedup.py:681-684`.
