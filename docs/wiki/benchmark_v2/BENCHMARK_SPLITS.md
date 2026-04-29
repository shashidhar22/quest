# Benchmark Splits

Train / val / test partition files for the four `benchmark_v2` tasks (AS, PM,
PAIR, MR). Built from the clustering artifacts in `data/benchmark_v2/clusters/`
(see `docs/BENCHMARK_CLUSTERING.md`) plus a sequence-tuple → source/allele
lookup reconstructed from `data/standardized_again/`.

Outputs live at `data/benchmark_v2/splits/` and are produced by
`scripts/data_processing/build_benchmark_splits.py`.

---

## 1. Purpose & consumers

Downstream training and evaluation pipelines load:

- `*_train.parquet` for training set
- `*_val.parquet` for validation
- `*_test_<condition>.parquet` for held-out evaluation under specific novelty
  conditions
- `*_eval.parquet` for evaluation-only benchmarks (Class II AS/MR, etc.)
- `*_candidate_pool.tsv` for retrieval candidate sets

The splits encode leakage-prevention via sequence/peptide/allele clustering so
generalization performance can be measured on truly held-out distributions.

---

## 2. Source dataset + cluster inputs

- **Enriched parquet**: `data/deduplicated_again/exploded_deduped_enriched/`
  (Hive-partitioned by `subset_key`, see clustering wiki §2)
- **Cluster TSVs** (read directly):
  - `clusters/trb_cdr3_interaction_clusters.tsv`
  - `clusters/trb_cdr3_restriction_clusters.tsv`
  - `clusters/tra_cdr3_interaction_clusters.tsv`
  - `clusters/tra_cdr3_restriction_clusters.tsv`
  - `clusters/peptide_clusters_interaction.tsv` (rapidfuzz Levenshtein single-linkage)
  - `clusters/peptide_clusters_pm.tsv` (MMseqs2 fallback at PM scale; see
    clustering wiki §5)
- **Source/allele lookup**: `splits/lookup/sequence_to_source_allele.parquet`
  (see §6)

The TRB foundation clustering (`clusters/trb_cdr3_all_clusters.tsv`) is
pretraining-scope and is **not** used by any splits task. It is allowed to
finish in the background.

---

## 3. Task taxonomy

| Task | Definition | Class I | Class II |
|------|------------|---------|----------|
| **AS** Antigenic Specificity | Given TCR + MHC, predict binding peptide (retrieval) | TRB-I, TRA-I, Paired-I — full splits | TRB-II, TRA-II, Paired-II — eval-only |
| **PM** Peptide-MHC Binding | Given MHC, predict bound peptides (no TCR) | PM-I — full splits | PM-II — full splits |
| **PAIR** TCR Pairing | Given one chain, predict cognate partner | (single benchmark) — full splits + negatives | — |
| **MR** TCR-MHC Restriction | Given TCR, predict restricting allele | TRB-I, TRA-I, Paired-I — full splits | TRB-II minimal split if eligible; TRA-II / Paired-II eval-only |

### Subset_key membership per task (predicates on `subset_key`)

- AS: `(tra in key OR trb in key) AND peptide in key AND (mhc_one in key OR mhc_two in key)`
- PM: `peptide in key AND (mhc_one in key OR mhc_two in key)`
- PAIR: `key starts with "tra_trb"` (i.e., contains both tra and trb)
- MR: `(tra in key OR trb in key) AND (mhc_one in key OR mhc_two in key)`

---

## 4. Split algorithm

Common scaffold for every benchmark:

1. **Load rows** for the matching subset_keys via DuckDB (returns Arrow table).
2. **Left-join** with the source/allele lookup (§6) to attach
   `source_db_primary, source_db_set, mhc_one_allele, mhc_two_allele`. Rows
   with no lookup hit fall back to `mhc_one`/`mhc_two` protein sequence as the
   allele identifier.
3. **Synthesize `row_id`** = BLAKE2b(8) over (tra_full, trb_full, peptide,
   mhc_one, mhc_two, subset_key).
4. **Dedup** when applicable (PM, PAIR, MR — see per-task sections). Tiebreak:
   experimental source > computational > other; then highest non-null column
   count; then lex-first row_id.
5. **Cluster lookup**: attach `tcr_cluster` (TCR CDR3 → cluster id from the
   relevant interaction or restriction TSV) and `pep_cluster` if applicable.
6. **Partition the cluster pool**:
   - AS: 70% TCR clusters + 70% peptide clusters → train pool; 20% of alleles
     with ≥50 examples → held out.
   - PM: 80% peptide clusters → train; 20% of eligible alleles held out.
   - PAIR: 80% TRA clusters + 80% TRB clusters → train pools.
   - MR: 70% TCR clusters → train; 20% of eligible alleles held out.
7. **Assign condition** per row from the novelty flags (see §5).
8. **Stratified split** of the IID condition by peptide (or TRB) cluster id —
   ratios are 85/7.5/7.5 for AS, 80/10/10 for PM and PAIR, 70/15/15 for MR.
9. **Write** one parquet per partition.

All sampling uses `random.Random(42)` and `numpy.random.seed(42)`.

---

## 5. Novelty conditions

Per row, define three boolean novelty flags, then map to a condition label:

- `tcr_novel`: the row's TCR cluster (TRB or TRA depending on benchmark) is
  NOT in the train cluster pool. For Paired benchmarks, novel if **either**
  TRA or TRB cluster is novel.
- `pep_novel`: the row's peptide cluster is NOT in the train cluster pool.
- `allele_novel`: the row's allele is in the held-out allele set.

Condition labels:

| Task | iid | novel_tcr | novel_pep | novel_allele | level4 | mixed |
|------|-----|-----------|-----------|--------------|--------|-------|
| AS   | none novel | only tcr | only pep | only allele | all three | exactly two of {tcr, pep, allele} |
| PM   | none novel | — | only pep | only allele | both pep and allele | — (collapsed into level4) |
| MR   | none novel | only tcr | — | only allele | both tcr and allele | — |

| Task | iid | novel_tra | novel_trb | novel_both |
|------|-----|-----------|-----------|------------|
| PAIR | both clusters in train | only tra novel | only trb novel | both novel |

---

## 6. Source / allele reconstruction (lookup)

The enriched parquet stage drops `source_db`, `mhc_one_allele`,
`mhc_two_allele`. The splits code reconstructs them by left-joining against a
lookup built once from the **non-deduplicated** `data/standardized_again/`
stage.

### How the lookup is built (`--task prep_lookup`)

DuckDB scans `standardized_again/**/*.parquet` (filtering bulk TCR-only rows
to avoid the 200+ GB immuneaccess noise), groups by the 5-tuple
`(tra_full, trb_full, peptide, mhc_one, mhc_two)`, and aggregates:

- `source_db_set` — distinct `source` values that contributed this tuple
- `source_db_primary` — first source from the **experimental** priority list
  `[iedb, cedar, vdjdb, mcpas, batman, trait, immunecode, ots]` if present;
  else first **computational** `[netmhcpan, cedar_pmhc]`; else lex-first.
- `mhc_one_allele`, `mhc_two_allele` — first allele lex-sorted from the
  distinct set seen for this tuple
- `mhc_one_allele_set`, `mhc_two_allele_set` — full distinct sets, retained
  for downstream auditing

Persisted at `splits/lookup/sequence_to_source_allele.parquet` with a sidecar
`splits/lookup/lookup_stats.json` reporting:

- Row count
- # of tuples with multi-source (>1 source contributed)
- # of tuples with multi-mhc1-allele or multi-mhc2-allele (rare; cedar_pmhc
  records up to 3 alleles per identical protein sequence)
- Distinct source values seen across the corpus

### Why `standardized_again` is not deduplicated

`standardized_again` is the input to the dedup pipeline. A given biological
record can appear once per source database; the splits code aggregates these
into a many-to-one lookup using deterministic rules (priority list + lex
sorting), so non-uniqueness in the upstream is expected and handled.

### `source_type` column

Derived from `source_db_primary` on PM outputs (per spec):
- `experimental` if in `[iedb, cedar, vdjdb, mcpas, batman, trait, immunecode, ots]`
- `computational` if in `[netmhcpan, cedar_pmhc]`
- `unknown` otherwise (or no lookup hit)

---

## 7. Other deviations from spec

- **Allele identifier as protein sequence**: The enriched stage stores
  `mhc_one`/`mhc_two` as full ~250 AA protein sequences, not HLA names. The
  lookup recovers HLA names where possible; `mhc_*_allele` falls back to the
  protein sequence when the lookup misses. Downstream code joining on
  `mhc_one_allele` works either way but should not assume name-format.
- **`row_id` synthesis**: There is no native unique-ID column. We synthesize
  `row_id` from `(tra_full, trb_full, peptide, mhc_one, mhc_two, subset_key)`
  via BLAKE2b-8 (8-byte hex) so each emitted row has a deterministic ID.
- **PM peptide clustering uses MMseqs2** (not edit distance). At PM scale
  (~4.7M unique peptides) all-pairs Levenshtein is infeasible; the clustering
  wiki documents the fallback.

---

## 8. Output artifacts

All parquet files share a common schema:

| Column | Notes |
|--------|-------|
| Original 20 enriched columns | `tra_full, trb_full, peptide, mhc_one, mhc_two, tra/trb_cdr1/2/3, sequence, subset_key, order_key, mhc_*_pocket/contact/pocket_contact` |
| `mhc_one_allele`, `mhc_two_allele` | From lookup or fallback to protein sequence |
| `source_db_primary`, `source_db_set` | From lookup; primary uses experimental priority |
| `source_type` | Derived from `source_db_primary`: `experimental`/`computational`/`unknown` |
| `row_id` | BLAKE2b(8) deterministic ID |
| `tcr_cluster` | Per-task: TRB cluster, TRA cluster, or `tra_cl|trb_cl` for paired |
| `pep_cluster` | AS, PM only |
| `condition` | One of the labels in §5 |
| `has_interaction` | PAIR only — true if subset_key contains peptide AND mhc |
| `label` | PAIR negatives only — 1 for positives, 0 for negatives |

### Files emitted

#### Task A (AS)

- `as_trb_i_{train,val,test_iid,test_novel_tcr,test_novel_pep,test_novel_allele,test_level4,test_mixed}.parquet`
- `as_tra_i_{...}.parquet` (same 8 partitions)
- `as_paired_i_{...}.parquet` (same 8 partitions)
- `as_trb_ii_eval.parquet`, `as_tra_ii_eval.parquet`, `as_paired_ii_eval.parquet`
- `as_class_i_candidate_pool.tsv` — unique peptides + cluster ids across
  all three Class I benchmarks

#### Task B (PM)

- `pm_i_{train,val,test_iid,test_novel_pep,test_novel_allele,test_level4}.parquet`
- `pm_ii_{...}.parquet`
- `pm_i_candidate_pool.tsv`, `pm_ii_candidate_pool.tsv` — unique peptides
  + cluster ids per benchmark

#### Task C (PAIR)

- `pair_{train,val,test_iid,test_novel_tra,test_novel_trb,test_novel_both}.parquet`
- `pair_negatives_train.parquet` — positives + 5×N random negatives with
  `label` column. Negatives are sampled deterministically from train-set
  TCRs and filtered against the positive set to avoid false negatives.

#### Task D (MR)

- `mr_trb_i_{train,val,test_iid,test_novel_tcr,test_novel_allele,test_level4}.parquet`
- `mr_tra_i_{...}.parquet`, `mr_paired_i_{...}.parquet` (same 6 partitions)
- `mr_trb_ii_{test_iid,test_novel_allele}.parquet` if eligible (≥5 alleles
  with ≥20 examples), else `mr_trb_ii_eval.parquet`
- `mr_tra_ii_eval.parquet`, `mr_paired_ii_eval.parquet`
- `mr_class_i_candidate_pool.tsv` — unique mhc_one_allele values

#### Auxiliary

- `splits/lookup/sequence_to_source_allele.parquet`
- `splits/lookup/lookup_stats.json`
- `splits/timings.json` — wall-clock per task
- `splits/leakage_validation.json` — see §9
- `splits/split_summary.json` — file inventory + leakage status + timings

---

## 9. Leakage validation (`--task leakage`)

For every `*_test_novel_*` and `*_test_level4` partition:

- **Novel-Pep**: pairwise rapidfuzz Levenshtein between train and test
  peptides. Threshold `ceil(0.2 * min(len_a, len_b))`. PASS iff every test
  peptide's distance to every train peptide exceeds threshold. Records up to
  20 violations and the closest pair.
- **Novel-TCR / Novel-TRA / Novel-TRB**: cluster-id set disjointness on
  `tcr_cluster`, `tra_cluster`, `trb_cluster`. PASS iff zero overlap.
- **Novel-Allele**: set disjointness on `mhc_one_allele` (or
  `mhc_two_allele` for Class II benchmarks).
- **Level-4**: all relevant disjointness checks simultaneously (AS: tcr +
  pep + allele; PM: pep + allele; MR: tcr + allele).

Output is one record per check in `leakage_validation.json` with `status:
PASS|FAIL`. The `summary` task aggregates and prints fail counts.

---

## 10. Candidate pools

Retrieval-task evaluation needs a candidate pool — the universe of peptides
or alleles a model is allowed to retrieve from.

| File | Pool | Used by |
|------|------|---------|
| `as_class_i_candidate_pool.tsv` | unique peptides across AS Class I benchmarks | AS retrieval at Class I |
| `pm_i_candidate_pool.tsv` | unique peptides in PM-I | PM-I peptide retrieval |
| `pm_ii_candidate_pool.tsv` | unique peptides in PM-II | PM-II peptide retrieval |
| `mr_class_i_candidate_pool.tsv` | unique mhc_one_allele values | MR-I allele retrieval |

---

## 11. Reproducibility & commands

Versions:

- DuckDB 1.5.1
- pyarrow (repo env)
- rapidfuzz 3.14.5
- Python 3.13 via miniforge
- Random seed: **42** for every sampling, shuffle, and stratified split.

Commands:

```bash
# Full pipeline
python scripts/data_processing/build_benchmark_splits.py --task all

# Or per-task
python scripts/data_processing/build_benchmark_splits.py --task prep_lookup
python scripts/data_processing/build_benchmark_splits.py --task as
python scripts/data_processing/build_benchmark_splits.py --task pm
python scripts/data_processing/build_benchmark_splits.py --task pair
python scripts/data_processing/build_benchmark_splits.py --task mr
python scripts/data_processing/build_benchmark_splits.py --task leakage
python scripts/data_processing/build_benchmark_splits.py --task summary
```

`prep_lookup` is the one-time scan that builds the source/allele lookup
parquet — re-running is a no-op if the lookup exists. Delete the lookup file
to force a rebuild.

---

## 12. Negative sampling for PAIR

The PAIR task is retrieval (given one chain, find its cognate partner). For
training a binary classification baseline ("is this a real pair?"), we
augment train with negatives:

- For each positive `(tra_X, trb_Y)` in `pair_train.parquet`, sample 5
  random `trb_Z` from train TRBs (Z ≠ Y) such that `(tra_X, trb_Z)` is not
  itself a positive pair (filtered against the positive set).
- Negatives are written to `pair_negatives_train.parquet` along with all
  positives, with a `label` column: 1 for positives, 0 for negatives.
- Random seed for negative sampling: 43 (= 42 + 1) so it's deterministic
  but separable from the partition-sampling seed.
- Only train has negatives — val and test are evaluation-only and use the
  retrieval candidate pool, not classification negatives.
