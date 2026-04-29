# peptide

## Identity
- **Subset path (deduped)**: `data/deduplicated_again/exploded_deduped/subset_key=peptide/`
- **Subset path (enriched)**: `data/deduplicated_again/exploded_deduped_enriched/subset_key=peptide/`
- **Populated dedup columns**: `peptide`
- **Carried TARGET_COLUMNS**: peptide
  (CDR1/2/3 columns are present whenever the parent chain is — see `tcrbench_dedup.py:46-50`)
- **Order_keys (orderings)**: 1 = 1!

## Counts
- **Unique records (`combination_counts.json`)**: n/a — this subset_key is a *projection* of one or more larger biological combinations; it appears in `permutation_counts.json` only.
- **Unique records (`permutation_counts.json`)**: 16,714,242
- **Total exploded rows** (`permutation_count × n!`): 16,714,242

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_deduplicated_manifest.py) -->

- **Populated columns** (1): `peptide`
- **Combination count**: _not present in `combination_counts.json`_ (combination_counts only tracks 11 of the 31 subsets — those exposing all `populated_columns` in the same form as a unique key)
- **Permutation row count** (rows per order_key, from `permutation_counts.json`): 16,714,242
- **Total exploded rows expected**: 16,714,242 × 1 (1! permutations of 1 columns) = 16,714,242
- **Order keys** (1): `peptide`

**Deduped exploded** (`exploded_deduped/subset_key=peptide/`):

- 1 order_keys · 28 parquet files · 267.5M (280,468,969 bytes) · 16,714,242 rows (matches expected)
- Schema: `tra_full`, `trb_full`, `peptide`, `mhc_one`, `mhc_two`, `tra_cdr1`, `tra_cdr2`, `tra_cdr3`, `trb_cdr1`, `trb_cdr2`, `trb_cdr3`, `sequence`

**Enriched exploded** (`exploded_deduped_enriched/subset_key=peptide/`):

- 1 order_keys · 28 parquet files · 274.0M (287,263,548 bytes) · 16,714,242 rows (matches expected) · enriched row count == deduped row count
- Schema columns added vs deduped: `mhc_one_contact`, `mhc_one_pocket`, `mhc_one_pocket_contact`, `mhc_two_contact`, `mhc_two_pocket`, `mhc_two_pocket_contact`, `order_key`, `subset_key`

<!-- END: AUTO-INVENTORY -->

## Training task this feeds

MLM pre-training on peptides (8-30mers) — short-sequence corpus mostly drawn from NetMHCpan and IEDB pMHC sources. Feeds the peptide encoder of [`peptide_mhc_seq2seq_trainer.py`](../../../../scripts/training/peptide_mhc_seq2seq_trainer.py).

## Notes

- This subset_key is **not** in `combination_counts.json` (only in `permutation_counts.json`) — it represents a projection of larger biological combinations onto the 1 listed column(s). For example, every record in any combination that contains `peptide` as a strict subset of its populated columns also contributes one DISTINCT tuple to this subset_key.
- Sequence column for each `order_key=` is `CONCAT_WS(' ', <ordered_cols>)` — see `tcrbench_dedup.py:681-684`.
