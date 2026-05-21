# tra_peptide

## Identity
- **Subset path (deduped)**: `data/deduplicated_again/exploded_deduped/subset_key=tra_peptide/`
- **Subset path (enriched)**: `data/deduplicated_again/exploded_deduped_enriched/subset_key=tra_peptide/`
- **Populated dedup columns**: `tra, peptide`
- **Carried TARGET_COLUMNS**: tra (CDR3 fallback), tra_full, tra_cdr1, tra_cdr2, tra_cdr3, peptide
  (CDR1/2/3 columns are present whenever the parent chain is — see `tcrbench_dedup.py:46-50`)
- **Order_keys (orderings)**: 2 = 2!

## Counts
- **Unique records (`combination_counts.json`)**: n/a — this subset_key is a *projection* of one or more larger biological combinations; it appears in `permutation_counts.json` only.
- **Unique records (`permutation_counts.json`)**: 83,188
- **Total exploded rows** (`permutation_count × n!`): 166,376

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_deduplicated_manifest.py) -->

- **Populated columns** (2): `tra`, `peptide`
- **Combination count**: _not present in `combination_counts.json`_ (combination_counts only tracks 11 of the 31 subsets — those exposing all `populated_columns` in the same form as a unique key)
- **Permutation row count** (rows per order_key, from `permutation_counts.json`): 83,188
- **Total exploded rows expected**: 83,188 × 2 (2! permutations of 2 columns) = 166,376
- **Order keys** (2): `peptide_tra`, `tra_peptide`

**Deduped exploded** (`exploded_deduped/subset_key=tra_peptide/`):

- 2 order_keys · 2 parquet files · 4.7M (4,884,982 bytes) · 166,376 rows (matches expected)
- Schema: `tra_full`, `trb_full`, `peptide`, `mhc_one`, `mhc_two`, `tra_cdr1`, `tra_cdr2`, `tra_cdr3`, `trb_cdr1`, `trb_cdr2`, `trb_cdr3`, `mhc_one_allele`, `mhc_two_allele`, `sequence`

**Enriched exploded** (`exploded_deduped_enriched/subset_key=tra_peptide/`):

- 2 order_keys · 2 parquet files · 4.9M (5,177,207 bytes) · 166,376 rows (matches expected) · enriched row count == deduped row count
- Schema columns added vs deduped: `mhc_one_contact`, `mhc_one_pocket`, `mhc_one_pocket_contact`, `mhc_two_contact`, `mhc_two_pocket`, `mhc_two_pocket_contact`, `order_key`, `subset_key`

<!-- END: AUTO-INVENTORY -->

## Training task this feeds

Alpha-chain TCR-peptide interaction (no MHC). Feeds the alpha-chain branch of [`tcr_cross_encoder_trainer.py`](../../../../scripts/training/tcr_cross_encoder_trainer.py) and seq2seq tasks in [`tcr_seq2seq_trainer.py`](../../../../scripts/training/tcr_seq2seq_trainer.py).

## Notes

- This subset_key is **not** in `combination_counts.json` (only in `permutation_counts.json`) — it represents a projection of larger biological combinations onto the 2 listed column(s). For example, every record in any combination that contains `tra, peptide` as a strict subset of its populated columns also contributes one DISTINCT tuple to this subset_key.
- Sequence column for each `order_key=` is `CONCAT_WS(' ', <ordered_cols>)` — see `tcrbench_dedup.py:681-684`.
