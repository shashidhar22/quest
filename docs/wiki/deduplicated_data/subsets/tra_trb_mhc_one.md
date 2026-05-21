# tra_trb_mhc_one

## Identity
- **Subset path (deduped)**: `data/deduplicated_again/exploded_deduped/subset_key=tra_trb_mhc_one/`
- **Subset path (enriched)**: `data/deduplicated_again/exploded_deduped_enriched/subset_key=tra_trb_mhc_one/`
- **Populated dedup columns**: `tra, trb, mhc_one`
- **Carried TARGET_COLUMNS**: tra (CDR3 fallback), tra_full, tra_cdr1, tra_cdr2, tra_cdr3, trb (CDR3 fallback), trb_full, trb_cdr1, trb_cdr2, trb_cdr3, mhc_one (MHC-I full sequence)
  (CDR1/2/3 columns are present whenever the parent chain is — see `tcrbench_dedup.py:46-50`)
- **Order_keys (orderings)**: 6 = 3!

## Counts
- **Unique records (`combination_counts.json`)**: n/a — this subset_key is a *projection* of one or more larger biological combinations; it appears in `permutation_counts.json` only.
- **Unique records (`permutation_counts.json`)**: 48,908
- **Total exploded rows** (`permutation_count × n!`): 293,448

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_deduplicated_manifest.py) -->

- **Populated columns** (3): `tra`, `trb`, `mhc_one`
- **Combination count**: _not present in `combination_counts.json`_ (combination_counts only tracks 11 of the 31 subsets — those exposing all `populated_columns` in the same form as a unique key)
- **Permutation row count** (rows per order_key, from `permutation_counts.json`): 48,908
- **Total exploded rows expected**: 48,908 × 6 (3! permutations of 3 columns) = 293,448
- **Order keys** (6): `mhc_one_tra_trb`, `mhc_one_trb_tra`, `tra_mhc_one_trb`, `tra_trb_mhc_one`, `trb_mhc_one_tra`, `trb_tra_mhc_one`

**Deduped exploded** (`exploded_deduped/subset_key=tra_trb_mhc_one/`):

- 6 order_keys · 6 parquet files · 16.1M (16,912,134 bytes) · 293,448 rows (matches expected)
- Schema: `tra_full`, `trb_full`, `peptide`, `mhc_one`, `mhc_two`, `tra_cdr1`, `tra_cdr2`, `tra_cdr3`, `trb_cdr1`, `trb_cdr2`, `trb_cdr3`, `mhc_one_allele`, `mhc_two_allele`, `sequence`

**Enriched exploded** (`exploded_deduped_enriched/subset_key=tra_trb_mhc_one/`):

- 6 order_keys · 6 parquet files · 17.6M (18,421,227 bytes) · 293,448 rows (matches expected) · enriched row count == deduped row count
- Schema columns added vs deduped: `mhc_one_contact`, `mhc_one_pocket`, `mhc_one_pocket_contact`, `mhc_two_contact`, `mhc_two_pocket`, `mhc_two_pocket_contact`, `order_key`, `subset_key`

<!-- END: AUTO-INVENTORY -->

## Training task this feeds

Paired αβ TCR + MHC-I (no peptide). Auxiliary subset; ~49K records.

## Notes

- This subset_key is **not** in `combination_counts.json` (only in `permutation_counts.json`) — it represents a projection of larger biological combinations onto the 3 listed column(s). For example, every record in any combination that contains `tra, trb, mhc_one` as a strict subset of its populated columns also contributes one DISTINCT tuple to this subset_key.
- Sequence column for each `order_key=` is `CONCAT_WS(' ', <ordered_cols>)` — see `tcrbench_dedup.py:681-684`.
