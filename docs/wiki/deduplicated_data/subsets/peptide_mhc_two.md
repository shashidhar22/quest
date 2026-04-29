# peptide_mhc_two

## Identity
- **Subset path (deduped)**: `data/deduplicated_again/exploded_deduped/subset_key=peptide_mhc_two/`
- **Subset path (enriched)**: `data/deduplicated_again/exploded_deduped_enriched/subset_key=peptide_mhc_two/`
- **Populated dedup columns**: `peptide, mhc_two`
- **Carried TARGET_COLUMNS**: peptide, mhc_two (MHC-II full sequence)
  (CDR1/2/3 columns are present whenever the parent chain is — see `tcrbench_dedup.py:46-50`)
- **Order_keys (orderings)**: 2 = 2!

## Counts
- **Unique records (`combination_counts.json`)**: n/a — this subset_key is a *projection* of one or more larger biological combinations; it appears in `permutation_counts.json` only.
- **Unique records (`permutation_counts.json`)**: 1,016,252
- **Total exploded rows** (`permutation_count × n!`): 2,032,504

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_deduplicated_manifest.py) -->

- **Populated columns** (2): `peptide`, `mhc_two`
- **Combination count**: _not present in `combination_counts.json`_ (combination_counts only tracks 11 of the 31 subsets — those exposing all `populated_columns` in the same form as a unique key)
- **Permutation row count** (rows per order_key, from `permutation_counts.json`): 1,016,252
- **Total exploded rows expected**: 1,016,252 × 2 (2! permutations of 2 columns) = 2,032,504
- **Order keys** (2): `mhc_two_peptide`, `peptide_mhc_two`

**Deduped exploded** (`exploded_deduped/subset_key=peptide_mhc_two/`):

- 2 order_keys · 18 parquet files · 57.0M (59,811,645 bytes) · 2,032,504 rows (matches expected)
- Schema: `tra_full`, `trb_full`, `peptide`, `mhc_one`, `mhc_two`, `tra_cdr1`, `tra_cdr2`, `tra_cdr3`, `trb_cdr1`, `trb_cdr2`, `trb_cdr3`, `sequence`

**Enriched exploded** (`exploded_deduped_enriched/subset_key=peptide_mhc_two/`):

- 2 order_keys · 18 parquet files · 62.8M (65,869,212 bytes) · 2,032,504 rows (matches expected) · enriched row count == deduped row count
- Schema columns added vs deduped: `mhc_one_contact`, `mhc_one_pocket`, `mhc_one_pocket_contact`, `mhc_two_contact`, `mhc_two_pocket`, `mhc_two_pocket_contact`, `order_key`, `subset_key`

<!-- END: AUTO-INVENTORY -->

## Training task this feeds

Peptide-MHC class II binding modeling. Feeds [`peptide_mhc_seq2seq_trainer.py`](../../../../scripts/training/peptide_mhc_seq2seq_trainer.py) MHC-II mode. ~1M records.

## Notes

- This subset_key is **not** in `combination_counts.json` (only in `permutation_counts.json`) — it represents a projection of larger biological combinations onto the 2 listed column(s). For example, every record in any combination that contains `peptide, mhc_two` as a strict subset of its populated columns also contributes one DISTINCT tuple to this subset_key.
- Sequence column for each `order_key=` is `CONCAT_WS(' ', <ordered_cols>)` — see `tcrbench_dedup.py:681-684`.
