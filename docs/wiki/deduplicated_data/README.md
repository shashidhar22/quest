# QUEST Deduplicated Data Manifest

This is the catalog of **deduplicated output** under `data/deduplicated_again/` — Stage 2 of the [data curation pipeline](../../../CLAUDE.md#data-curation-pipeline-overview). Unlike the upstream [standardized-data manifest](../standardized_data/README.md), which is keyed by *source database*, the deduplicated layer is a **pipeline-stage output**: per-source provenance has been collapsed across all 18 standardized buckets into a single global table of unique (tra, trb, peptide, mhc_one, mhc_two) tuples, then exploded into 31 column-subset partitions and 325 ordered-permutation partitions.

The driver script is [`tcrbench_dedup.py`](../../../tcrbench_dedup.py) (DuckDB-based, ~158 GB working DB). A second pass, [`scripts/data_processing/add_mhc_pseudosequences.py`](../../../scripts/data_processing/add_mhc_pseudosequences.py), enriches the exploded output with MHC pocket / contact pseudosequences. A legacy streaming variant — [`scripts/data_processing/deduplicate_streaming.py`](../../../scripts/data_processing/deduplicate_streaming.py) — implements the same logic without DuckDB and is retained as a fallback; it was **not** the producer of the artifacts documented here (per `pipeline_run_log.json` the run came from the DuckDB pipeline on 2026-04-08).

## Output layout

```
data/deduplicated_again/
├── deduped.db                              # 158 GB DuckDB working store
├── deduped_parquet/                        # 1.44B unique deduped rows
│                                           # cols: tra_full, trb_full, peptide, mhc_one, mhc_two,
│                                           #       tra_cdr1..cdr3, trb_cdr1..cdr3
├── exploded_deduped/
│   └── subset_key=X/order_key=Y/*.parquet  # 1.46B exploded rows, 31 subsets × {n_pop!} order_keys
│                                           # adds: sequence (whitespace-joined per order_key)
├── exploded_deduped_enriched/
│   └── subset_key=X/order_key=Y/*.parquet  # same rows + 6 MHC pseudosequence cols + subset_key/order_key
│                                           # adds: mhc_one_pocket, mhc_one_contact, mhc_one_pocket_contact,
│                                           #       mhc_two_pocket, mhc_two_contact, mhc_two_pocket_contact
├── combination_counts.{json,csv}            # 11 biological combos (populated subsets only)
├── permutation_counts.{json,csv}            # 31 powerset subsets (unordered counts)
├── pipeline_run_log.json                    # timing + row counts per stage
└── mhc_pseudo_lookup.json                   # 8 MB: MHC sequence → pseudosequence cache
```

## How to read this manifest

- **Master tables below** are rendered from `combination_counts.json` and `permutation_counts.json`.
- **[`PIPELINE.md`](PIPELINE.md)** narrates the dedup → counts → explosion → enrichment flow with file:line references.
- **[Per-subset detail pages](subsets/)** (31 of them) cover what each `subset_key` contains, how many order_keys it has, and which trainer consumes it.

The 31 subset_keys are the non-empty subsets of the 5 dedup columns `{tra, trb, peptide, mhc_one, mhc_two}` (`tcrbench_dedup.py:41,100-111`). For a subset with `n` populated columns, the explosion step writes `n!` `order_key=` partitions inside that subset_key, one per ordering of the columns. Total ordered partitions across all 31 subsets = **325** (= 5 + 5·4 + 10·3! + 10·4! + 5! / aligned per powerset).

## Master table — combinations (11 biological, populated)

Source: `combination_counts.json`. These are the *biologically meaningful* column tuples — i.e. the column sets that actually have records after dedup. Many other powerset subsets (e.g. `peptide` alone, `mhc_one` alone) appear in `permutation_counts.json` because they are derivable from larger combinations during explosion, but they are not standalone "combinations" in the source data.

| Combination (k components)          | Unique deduped records |
|-------------------------------------|-----------------------:|
| `tra` (1)                           |             39,057,858 |
| `trb` (1)                           |          1,380,428,803 |
| `tra_trb` (2)                       |              3,197,312 |
| `peptide_mhc_one` (2)               |              4,764,964 |
| `peptide_mhc_one_mhc_two` (3)       |                292,468 |
| `tra_peptide_mhc_one` (3)           |                 76,804 |
| `trb_peptide_mhc_one` (3)           |                113,478 |
| `tra_trb_peptide_mhc_one` (4)       |                 57,079 |
| `tra_peptide_mhc_one_mhc_two` (4)   |                    938 |
| `trb_peptide_mhc_one_mhc_two` (4)   |                  1,064 |
| `tra_trb_peptide_mhc_one_mhc_two` (5) |                  565 |
| **Sum**                             |          **1,427,990,433** |

Sum across the 11 combinations is **1.428B**, which is below the `pipeline_run_log.json:steps.dedup.rows_out` figure of **1,439,165,664**. The ~11M gap is records whose populated columns fall into one of the 20 powerset subsets *not* listed in `COMBINATIONS` (`tcrbench_dedup.py:82-94`) — for example `peptide`-only or `tra_peptide`-only rows from sources like NetMHCpan and the assorted standalone TCR datasets. Those records still appear in `permutation_counts.json` (master table below) and in `exploded_deduped/`.

## Master table — permutations (all 31 powerset subsets)

Source: `permutation_counts.json`. Counts here are over the same deduped rows, but include subset_keys whose components are a *projection* of a larger biological combination (so `peptide` includes peptides from records that also have mhc_one, plus standalone peptides, etc.).

| subset_key                            | order_keys | unique rows |
|---------------------------------------|-----------:|------------:|
| `tra`                                 | 1   |             39,057,858 |
| `trb`                                 | 1   |          1,380,428,803 |
| `peptide`                             | 1   |             16,714,242 |
| `mhc_one`                             | 1   |                 16,594 |
| `mhc_two`                             | 1   |                  5,721 |
| `tra_trb`                             | 2   |              3,197,312 |
| `tra_peptide`                         | 2   |                 83,188 |
| `trb_peptide`                         | 2   |                709,192 |
| `tra_mhc_one`                         | 2   |                 68,417 |
| `trb_mhc_one`                         | 2   |                101,280 |
| `tra_mhc_two`                         | 2   |                  2,608 |
| `trb_mhc_two`                         | 2   |                  4,156 |
| `peptide_mhc_one`                     | 2   |              4,764,964 |
| `peptide_mhc_two`                     | 2   |              1,016,252 |
| `mhc_one_mhc_two`                     | 2   |                    133 |
| `tra_trb_peptide`                     | 6   |                 63,675 |
| `tra_trb_mhc_one`                     | 6   |                 48,908 |
| `tra_trb_mhc_two`                     | 6   |                  2,356 |
| `tra_peptide_mhc_one`                 | 6   |                 76,804 |
| `trb_peptide_mhc_one`                 | 6   |                113,478 |
| `tra_peptide_mhc_two`                 | 6   |                  3,049 |
| `trb_peptide_mhc_two`                 | 6   |                  5,090 |
| `tra_mhc_one_mhc_two`                 | 6   |                  1,394 |
| `trb_mhc_one_mhc_two`                 | 6   |                  1,480 |
| `peptide_mhc_one_mhc_two`             | 6   |                292,468 |
| `tra_trb_peptide_mhc_one`             | 24  |                 57,079 |
| `tra_trb_peptide_mhc_two`             | 24  |                  3,493 |
| `tra_trb_mhc_one_mhc_two`             | 24  |                    839 |
| `tra_peptide_mhc_one_mhc_two`         | 24  |                    938 |
| `trb_peptide_mhc_one_mhc_two`         | 24  |                  1,064 |
| `tra_trb_peptide_mhc_one_mhc_two`     | 120 |                    565 |

`order_keys` = `n_populated!` — total ordered partitions = 325. The `permutation_counts.json` value is the *unordered* unique-row count for the subset; the actual exploded row count for the subset is `unordered_count × order_keys` (e.g. `tra_trb_peptide_mhc_one_mhc_two` produces `565 × 120 = 67,800` exploded rows). Aggregate exploded row count across all 31 subsets = `pipeline_run_log.json:steps.explosion.rows_out` = **1,461,360,146**.

## Auto inventory (Quantifier)

<!-- BEGIN: AUTO-MASTER-TABLE (build_deduplicated_manifest.py) -->

**Last regenerated**: 2026-04-27T15:14:45Z · **Subsets**: 31

**Pipeline run**:

- Run timestamp: `2026-04-08T22:38:14`
- Dedup elapsed: 4,615.1s · output rows: 1,439,165,664
- Explosion elapsed: 466.9s · output rows: 1,461,360,146

**Top-level artifacts**:

- `deduped_parquet/`: 64 files, 28.0G (30,013,789,399 bytes), 1,439,165,664 rows
- `deduped.db` (DuckDB intermediate): 147.5G (158,349,668,352 bytes)
- `mhc_pseudo_lookup.json`: 7.8M, 22,315 entries

**Totals across 31 subsets**: deduped exploded rows = 1,461,360,146 · enriched exploded rows = 1,461,360,146 · deduped exploded size = 45.3G · enriched exploded size = 44.9G

| Subset | Populated cols | Order keys | Combination | Permutation rows | Total exploded rows | Deduped size | Enriched size |
|--------|---------------:|-----------:|------------:|-----------------:|--------------------:|-------------:|--------------:|
| [`mhc_one_mhc_two`](subsets/mhc_one_mhc_two.md) | 2 (mhc_one, mhc_two) | 2 | — | 133 | 266 | 22.4K | 36.4K |
| [`mhc_one`](subsets/mhc_one.md) | 1 (mhc_one) | 1 | — | 16,594 | 16,594 | 554.9K | 747.9K |
| [`mhc_two`](subsets/mhc_two.md) | 1 (mhc_two) | 1 | — | 5,721 | 5,721 | 135.1K | 175.9K |
| [`peptide_mhc_one_mhc_two`](subsets/peptide_mhc_one_mhc_two.md) | 3 (peptide, mhc_one, mhc_two) | 6 | 292,468 | 292,468 | 1,754,808 | 52.2M | 60.1M |
| [`peptide_mhc_one`](subsets/peptide_mhc_one.md) | 2 (peptide, mhc_one) | 2 | 4,764,964 | 4,764,964 | 9,529,928 | 242.5M | 278.2M |
| [`peptide_mhc_two`](subsets/peptide_mhc_two.md) | 2 (peptide, mhc_two) | 2 | — | 1,016,252 | 2,032,504 | 57.0M | 62.8M |
| [`peptide`](subsets/peptide.md) | 1 (peptide) | 1 | — | 16,714,242 | 16,714,242 | 267.5M | 274.0M |
| [`tra_mhc_one_mhc_two`](subsets/tra_mhc_one_mhc_two.md) | 3 (tra, mhc_one, mhc_two) | 6 | — | 1,394 | 8,364 | 369.8K | 458.0K |
| [`tra_mhc_one`](subsets/tra_mhc_one.md) | 2 (tra, mhc_one) | 2 | — | 68,417 | 136,834 | 4.2M | 4.6M |
| [`tra_mhc_two`](subsets/tra_mhc_two.md) | 2 (tra, mhc_two) | 2 | — | 2,608 | 5,216 | 189.0K | 221.8K |
| [`tra_peptide_mhc_one_mhc_two`](subsets/tra_peptide_mhc_one_mhc_two.md) | 4 (tra, peptide, mhc_one, mhc_two) | 24 | 938 | 938 | 22,512 | 1.2M | 1.5M |
| [`tra_peptide_mhc_one`](subsets/tra_peptide_mhc_one.md) | 3 (tra, peptide, mhc_one) | 6 | 76,804 | 76,804 | 460,824 | 14.6M | 15.9M |
| [`tra_peptide_mhc_two`](subsets/tra_peptide_mhc_two.md) | 3 (tra, peptide, mhc_two) | 6 | — | 3,049 | 18,294 | 664.6K | 721.0K |
| [`tra_peptide`](subsets/tra_peptide.md) | 2 (tra, peptide) | 2 | — | 83,188 | 166,376 | 4.7M | 4.7M |
| [`tra_trb_mhc_one_mhc_two`](subsets/tra_trb_mhc_one_mhc_two.md) | 4 (tra, trb, mhc_one, mhc_two) | 24 | — | 839 | 20,136 | 1.7M | 2.0M |
| [`tra_trb_mhc_one`](subsets/tra_trb_mhc_one.md) | 3 (tra, trb, mhc_one) | 6 | — | 48,908 | 293,448 | 16.0M | 16.0M |
| [`tra_trb_mhc_two`](subsets/tra_trb_mhc_two.md) | 3 (tra, trb, mhc_two) | 6 | — | 2,356 | 14,136 | 895.1K | 1011.9K |
| [`tra_trb_peptide_mhc_one_mhc_two`](subsets/tra_trb_peptide_mhc_one_mhc_two.md) | 5 (tra, trb, peptide, mhc_one, mhc_two) | 120 | 565 | 565 | 67,800 | 6.9M | 8.0M |
| [`tra_trb_peptide_mhc_one`](subsets/tra_trb_peptide_mhc_one.md) | 4 (tra, trb, peptide, mhc_one) | 24 | 57,079 | 57,079 | 1,369,896 | 74.8M | 75.0M |
| [`tra_trb_peptide_mhc_two`](subsets/tra_trb_peptide_mhc_two.md) | 4 (tra, trb, peptide, mhc_two) | 24 | — | 3,493 | 83,832 | 4.3M | 4.3M |
| [`tra_trb_peptide`](subsets/tra_trb_peptide.md) | 3 (tra, trb, peptide) | 6 | — | 63,675 | 382,050 | 19.0M | 18.1M |
| [`tra_trb`](subsets/tra_trb.md) | 2 (tra, trb) | 2 | 3,197,312 | 3,197,312 | 6,394,624 | 330.4M | 328.2M |
| [`tra`](subsets/tra.md) | 1 (tra) | 1 | 39,057,858 | 39,057,858 | 39,057,858 | 1.1G | 1.1G |
| [`trb_mhc_one_mhc_two`](subsets/trb_mhc_one_mhc_two.md) | 3 (trb, mhc_one, mhc_two) | 6 | — | 1,480 | 8,880 | 396.1K | 481.0K |
| [`trb_mhc_one`](subsets/trb_mhc_one.md) | 2 (trb, mhc_one) | 2 | — | 101,280 | 202,560 | 6.9M | 7.5M |
| [`trb_mhc_two`](subsets/trb_mhc_two.md) | 2 (trb, mhc_two) | 2 | — | 4,156 | 8,312 | 315.5K | 366.8K |
| [`trb_peptide_mhc_one_mhc_two`](subsets/trb_peptide_mhc_one_mhc_two.md) | 4 (trb, peptide, mhc_one, mhc_two) | 24 | 1,064 | 1,064 | 25,536 | 1.3M | 1.6M |
| [`trb_peptide_mhc_one`](subsets/trb_peptide_mhc_one.md) | 3 (trb, peptide, mhc_one) | 6 | 113,478 | 113,478 | 680,868 | 24.0M | 26.2M |
| [`trb_peptide_mhc_two`](subsets/trb_peptide_mhc_two.md) | 3 (trb, peptide, mhc_two) | 6 | — | 5,090 | 30,540 | 1.1M | 1.2M |
| [`trb_peptide`](subsets/trb_peptide.md) | 2 (trb, peptide) | 2 | — | 709,192 | 1,418,384 | 45.8M | 48.3M |
| [`trb`](subsets/trb.md) | 1 (trb) | 1 | 1,380,428,803 | 1,380,428,803 | 1,380,428,803 | 43.0G | 42.6G |

*Permutation rows* = canonical rows per order_key partition (= rows after dedup for that column subset). *Total exploded rows* = permutation rows × `n!` rotations. *Combination* counts come from `combination_counts.json` (only 11 of 31 subsets are tracked there — namely the ones whose populated columns include all of the chains that anchor a complete training record).

_All cross-checks passed: every subset's exploded row count matches `permutation_count × n!`, and every subset's enriched row count equals its deduped row count._

<!-- END: AUTO-MASTER-TABLE -->

## How to regenerate

```bash
# Stage 2a: dedup + counts + explosion (all in one DuckDB pipeline)
python tcrbench_dedup.py \
    --input "/home/ubuntu/quest/data/standardized_again/**/*.parquet" \
    --output_dir "/home/ubuntu/quest/data/deduplicated_again" \
    --tmp_dir "/scratch" \
    --memory_limit "800GB" \
    --threads 64 \
    --mode both

# Stage 2b: enrich exploded output with MHC pocket / contact pseudosequences
python scripts/data_processing/add_mhc_pseudosequences.py
```

The DuckDB working store at `deduped.db` (~158 GB) is intentionally retained after the run so re-runs of `step2_counts` / `step3_explode` can skip the (dominant) sanitize+dedup step. Pass `--force_recompute` to rebuild it.

## Links

- Upstream input: [standardized-data manifest](../standardized_data/README.md)
- Pipeline narrative: [`PIPELINE.md`](PIPELINE.md)
- Per-subset pages: [`subsets/`](subsets/)
- Source script: [`tcrbench_dedup.py`](../../../tcrbench_dedup.py)
- Enrichment script: [`scripts/data_processing/add_mhc_pseudosequences.py`](../../../scripts/data_processing/add_mhc_pseudosequences.py)
- Legacy streaming variant: [`scripts/data_processing/deduplicate_streaming.py`](../../../scripts/data_processing/deduplicate_streaming.py)
- Project instructions: [`CLAUDE.md`](../../../CLAUDE.md)
