# peptide_mhc_one

## Identity
- **Subset path (deduped)**: `data/deduplicated_again/exploded_deduped/subset_key=peptide_mhc_one/`
- **Subset path (enriched)**: `data/deduplicated_again/exploded_deduped_enriched/subset_key=peptide_mhc_one/`
- **Populated dedup columns**: `peptide, mhc_one`
- **Carried TARGET_COLUMNS**: peptide, mhc_one (MHC-I full sequence)
  (CDR1/2/3 columns are present whenever the parent chain is — see `tcrbench_dedup.py:46-50`)
- **Order_keys (orderings)**: 2 = 2!

## Counts
- **Unique records (`combination_counts.json`)**: 4,764,964
- **Unique records (`permutation_counts.json`)**: 4,764,964
- **Total exploded rows** (`permutation_count × n!`): 9,529,928

## Auto inventory
<!-- BEGIN: AUTO-INVENTORY (build_deduplicated_manifest.py) -->

- **Populated columns** (2): `peptide`, `mhc_one`
- **Combination count** (canonical, from `combination_counts.json`): 4,764,964
- **Permutation row count** (rows per order_key, from `permutation_counts.json`): 4,764,964
- **Total exploded rows expected**: 4,764,964 × 2 (2! permutations of 2 columns) = 9,529,928
- **Order keys** (2): `mhc_one_peptide`, `peptide_mhc_one`

**Deduped exploded** (`exploded_deduped/subset_key=peptide_mhc_one/`):

- 2 order_keys · 78 parquet files · 242.5M (254,314,377 bytes) · 9,529,928 rows (matches expected)
- Schema: `tra_full`, `trb_full`, `peptide`, `mhc_one`, `mhc_two`, `tra_cdr1`, `tra_cdr2`, `tra_cdr3`, `trb_cdr1`, `trb_cdr2`, `trb_cdr3`, `sequence`

**Enriched exploded** (`exploded_deduped_enriched/subset_key=peptide_mhc_one/`):

- 2 order_keys · 78 parquet files · 278.2M (291,702,723 bytes) · 9,529,928 rows (matches expected) · enriched row count == deduped row count
- Schema columns added vs deduped: `mhc_one_contact`, `mhc_one_pocket`, `mhc_one_pocket_contact`, `mhc_two_contact`, `mhc_two_pocket`, `mhc_two_pocket_contact`, `order_key`, `subset_key`

<!-- END: AUTO-INVENTORY -->

## Training task this feeds

Peptide-MHC class I binding modeling — primary input to [`peptide_mhc_seq2seq_trainer.py`](../../../../scripts/training/peptide_mhc_seq2seq_trainer.py) (MHC-I mode) and pMHC contrastive learning. 4.76M records, dominated by NetMHCpan + IEDB pMHC sources.

## Notes

- `combination_counts.json` and `permutation_counts.json` agree at 4,764,964 — meaning every deduped row whose populated columns match this set has *no* additional populated columns (no records with these 2 columns plus extras).
- Sequence column for each `order_key=` is `CONCAT_WS(' ', <ordered_cols>)` — see `tcrbench_dedup.py:681-684`.
