# Verifier Report — Deduplicated Data Manifest

**Verifier**: verifier agent
**Verified at**: 2026-04-27
**Verdict**: PASS

## Summary

All five verification tasks pass with zero discrepancies. Independent parquet metadata reads
reproduce every manifest count exactly (5/5 spot-checked subsets covering small, medium,
1.38B-row large, 120-order-key, and permutation-only cases). Schema deltas, pipeline-log
totals, combination-vs-permutation invariants, and MHC pseudosequence integrity all align.

## Task 1: Per-subset row counts

Sums computed independently via `pyarrow.parquet.ParquetFile(p).metadata.num_rows` across all
shards under `subset_key={key}/order_key=*/data_*.parquet` (deduped and enriched sides).
`expected = permutation_count × n_orders` where `n_orders = factorial(len(populated_columns))`.

| Subset | n_pop | n_orders | Manifest dedup | Independent dedup | perm × n! | Manifest enriched | Independent enriched | dedup==enriched | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| `mhc_one_mhc_two` (small) | 2 | 2 | 266 | 266 | 266 | 266 | 266 | yes | PASS |
| `tra_trb_peptide_mhc_one` (medium) | 4 | 24 | 1,369,896 | 1,369,896 | 1,369,896 | 1,369,896 | 1,369,896 | yes | PASS |
| `trb` (large) | 1 | 1 | 1,380,428,803 | 1,380,428,803 | 1,380,428,803 | 1,380,428,803 | 1,380,428,803 | yes | PASS |
| `tra_trb_peptide_mhc_one_mhc_two` (120 order keys) | 5 | 120 | 67,800 | 67,800 | 67,800 | 67,800 | 67,800 | yes | PASS |
| `peptide` (perm-only, not in combination_counts) | 1 | 1 | 16,714,242 | 16,714,242 | 16,714,242 | 16,714,242 | 16,714,242 | yes | PASS |

For the 120-order-key subset, the file system also showed exactly 120 distinct `order_key=` directories
on both deduped and enriched sides, matching `order_key_count: 120` in the manifest.

## Task 2: Schema diff verification

Expected delta (`set(enriched.schema_arrow.names) - set(deduped.schema_arrow.names)`):

```
{subset_key, order_key,
 mhc_one_pocket, mhc_one_contact, mhc_one_pocket_contact,
 mhc_two_pocket, mhc_two_contact, mhc_two_pocket_contact}
```

| Subset | dedup cols | enriched cols | delta size | Unexpected | Missing | Pass |
|---|---:|---:|---:|---|---|:---:|
| `peptide` (single-col) | 12 | 20 | 8 | none | none | PASS |
| `tra_trb` (multi-col) | 12 | 20 | 8 | none | none | PASS |
| `tra_trb_peptide_mhc_one_mhc_two` (5-col) | 12 | 20 | 8 | none | none | PASS |

Across all three, no columns appear only in the deduped side (i.e., enriched is a strict superset).

## Task 3: Pipeline-log consistency

| Check | Inventory value | Pipeline log value | Expected | Pass |
|---|---:|---:|---:|:---:|
| `pipeline_run.dedup_rows_out` | 1,439,165,664 | 1,439,165,664 | 1,439,165,664 | PASS |
| `pipeline_run.explosion_rows_out` | 1,461,360,146 | 1,461,360,146 | 1,461,360,146 | PASS |
| `top_level.deduped_parquet.row_count` == `pipeline_run.dedup_rows_out` | 1,439,165,664 | — | 1,439,165,664 | PASS |
| Σ subsets `deduped.row_count_from_parquet` == `pipeline_run.explosion_rows_out` | 1,461,360,146 | — | 1,461,360,146 | PASS |

Bonus: Σ subsets `enriched.row_count_from_parquet` = 1,461,360,146 (also matches; PASS).

## Task 4: Combination vs permutation

Eleven subsets have both `combination_count` and `permutation_count` populated. For every one,
`combination_count == permutation_count` (which is the expected invariant — explosion is a
permutation rotation, the per-key row count doesn't change relative to combination).

Across **all 31 subsets**, `permutation_count × factorial(len(populated_columns))` exactly
equals the per-subset parquet row count (deduped side). No mismatches.

| Subset | combo | perm | combo==perm |
|---|---:|---:|:---:|
| `peptide_mhc_one` | 4,764,964 | 4,764,964 | yes |
| `peptide_mhc_one_mhc_two` | 292,468 | 292,468 | yes |
| `tra` | 39,057,858 | 39,057,858 | yes |
| `tra_peptide_mhc_one` | 76,804 | 76,804 | yes |
| `tra_peptide_mhc_one_mhc_two` | 938 | 938 | yes |
| `tra_trb` | 3,197,312 | 3,197,312 | yes |
| `tra_trb_peptide_mhc_one` | 57,079 | 57,079 | yes |
| `tra_trb_peptide_mhc_one_mhc_two` | 565 | 565 | yes |
| `trb` | 1,380,428,803 | 1,380,428,803 | yes |
| `trb_peptide_mhc_one` | 113,478 | 113,478 | yes |
| `trb_peptide_mhc_one_mhc_two` | 1,064 | 1,064 | yes |

Spot-checks of `perm × n!` vs parquet rows (subset of 31, all 31 verified PASS):

| Subset | perm | n! | expected | parquet rows | eq |
|---|---:|---:|---:|---:|:---:|
| `peptide_mhc_one_mhc_two` | 292,468 | 6 | 1,754,808 | 1,754,808 | yes |
| `tra_peptide_mhc_one_mhc_two` | 938 | 24 | 22,512 | 22,512 | yes |
| `tra_trb_peptide_mhc_one_mhc_two` | 565 | 120 | 67,800 | 67,800 | yes |
| `trb` | 1,380,428,803 | 1 | 1,380,428,803 | 1,380,428,803 | yes |

## Task 5: MHC pseudosequence integrity

| Check | Manifest | Actual lookup | Pass |
|---|---:|---:|:---:|
| `mhc_pseudo_lookup.entry_count` | 22,315 | 22,315 | PASS |
| `entries_by_class.class_i` | 16,594 | 16,594 | PASS |
| `entries_by_class.class_ii` | 5,721 | 5,721 | PASS |

Sampled 5 random `class_i` alleles (full MHC-I sequences) and located each in
`subset_key=mhc_one/order_key=mhc_one/data_*.parquet`. For every match, the
`mhc_one_pocket`, `mhc_one_contact`, and `mhc_one_pocket_contact` column values
were byte-identical to `lookup['class_i'][allele]` entries 0/1/2.

| Sampled allele (first 40 chars) | Found | pocket== | contact== | pocket_contact== |
|---|:---:|:---:|:---:|:---:|
| `MLVMAPRTVLLLLSAALALTETWAGSHSMRYFYTSVSRPG…` | yes | True | True | True |
| `MRVMAPRTLLLLLSGALALTETWACSHSMRYFYTAVSRPG…` | yes | True | True | True |
| `SHSMRYFHTSVSRPGRGEPRFITVGYVDDTLFVRFDSDAT…` | yes | True | True | True |
| `MRVMAPRTLILLLSGALALTETWACSHSMRYFYTAVSRPG…` | yes | True | True | True |
| `SHSMRYFYTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAA…` | yes | True | True | True |

## Discrepancies

None.
