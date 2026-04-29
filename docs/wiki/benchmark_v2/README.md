# benchmark_v2 — TCR/pMHC Benchmark Corpus

Four-phase data pipeline that turns the raw TCR/pMHC corpus into model-ready
training/evaluation parquets for ESM-2 and ESM-C. All artifacts live under
`/home/ubuntu/quest/data/benchmark_v2/`.

## Phase index

| # | Phase | Output dir | Driver | Wiki |
|---|-------|-----------|--------|------|
| 1 | **Clustering** | `clusters/` | `scripts/data_processing/build_benchmark_clusters.py` | [BENCHMARK_CLUSTERING.md](BENCHMARK_CLUSTERING.md) |
| 2 | **Splits** | `splits/` | `scripts/data_processing/build_benchmark_splits.py` | [BENCHMARK_SPLITS.md](BENCHMARK_SPLITS.md) |
| 3 | **Foundation MLM** | `foundation/` | `scripts/data_processing/build_benchmark_foundation.py` | [BENCHMARK_FOUNDATION.md](BENCHMARK_FOUNDATION.md) |
| 4 | **Tokenization** | `tokenized/` | `scripts/data_processing/build_benchmark_tokenized.py` | [BENCHMARK_TOKENIZED.md](BENCHMARK_TOKENIZED.md) |

Source data: `/home/ubuntu/quest/data/deduplicated_again/exploded_deduped_enriched/`
(31 subset_keys, 639 parquet files, 45 GB, ~1.44B rows).

---

## Headline numbers

### Phase 1 — Clustering

- **TRB CDR3 foundation clusters**: 864,578,714 unique sequences → MMseqs2 clustering at 90% identity (28 GB TSV)
- **TRA CDR3 foundation clusters**: 29,081,162 unique → 20.3M clusters
- **Interaction-scope CDR3**: 208,982 TRB unique → 160,341 clusters; 51,041 TRA unique → 34,164 clusters
- **Peptide clusters (interaction)**: 5,497 unique → 2,067 clusters via rapidfuzz Levenshtein single-linkage
- **Peptide clusters (PM, MMseqs2 fallback)**: 4,736,475 unique → 4,533,801 clusters at 80% identity
- **Method comparison (interaction peptides)**: ED vs MMseqs2 = **99.19% pair agreement**, ED finds **40,518 extra similar pairs** that MMseqs2 misses (vs 12 the other way) — confirms reviewer concern about MMseqs2 missing short-peptide near-duplicates
- Wall-clock: ~17 min foreground + ~14 hours TRB foundation background

### Phase 2 — Splits

- **15 benchmarks across 4 tasks** (AS, PM, PAIR, MR)
- **67 partition parquet files** total
- **AS Class I** (full splits): TRB-I 706K rows, TRA-I 483K rows, Paired-I 1.44M rows × 5 format scopes downstream
- **PM-I** (deduped peptide-MHC class I): 4,764,964 unique pairs from 13.9M source rows
- **PM-II** (class II): 1,016,252 unique pairs
- **PAIR**: 3,063,337 unique (TRA, TRB) pairs + 7,958,640 random negatives (5× per positive)
- **MR-TRB-I**: 142,601 unique (TRB, MHC-I) pairs deduped; MR-TRA-I 96,930; MR-Paired-I 63,281
- **Source/allele lookup**: 24,399,183 sequence-tuples reconstructed from `standardized_again`
- **Leakage validation**: 28/30 PASS, 2 FAILs are expected (PM novel_pep MMseqs2-vs-ED gap)
- Wall-clock: ~1 hour total

### Phase 3 — Foundation MLM datasets

- **Exclusion list**: 3,677,196 unique biological hashes (CDR3+pep+allele MD5) collected from all benchmark test/eval partitions
- **Partition assignments**: 1,434,343,959 rows partitioned 90/5/5 train/val/test (deterministic mod-20 of hash)
  - Train: 1,290,967,150 rows
  - Val: 71,697,996 rows
  - Test: 71,678,813 rows
- **Foundation val/test** (full data): 71.7M rows each, 3.75 GB compressed each
- **Sample manifests** (M1 sqrt weights):
  - `foundation_10M.parquet` — 9,999,999 rows
  - `foundation_100M.parquet` — 100,000,000 rows
  - `foundation_500M/` — 500,000,000 rows in 70 shards
- **Mini-ablation manifests** (10M each):
  - `mini_M2_10M` (sqrt × min(k,3))
  - `mini_M3_10M` (sqrt × k)
  - `mini_O1_10M` (canonical order)
  - `mini_O3_10M` (random order with `<eos>` separator)
- Wall-clock: ~2 hours total

### Phase 4 — Tokenization

- **349 tokenized parquet files**, 835,495,585 rows, **44.79 GB**
- **Format variants**: C1–C5 (AS/MR), M1–M3 (PM), T1–T3 (PAIR), C3 only for foundation/mini
- **Tokenizer**: ESM-2 (`facebook/esm2_t33_650M_UR50D`); produces IDs identical to ESM-C since both share the AA vocabulary
- **Token-length distributions** (median):
  - C1=399, C2=81, C3=50, C4=57, C5=369
  - M1=286, M2=60, M3=34
  - T1=27, T2=50, T3=583
- **Foundation 500M sharded** into 70 parquets × ~7M rows
- Wall-clock: ~4 hours total

---

## Key design decisions (cross-phase)

1. **MHC = full ~250 AA protein sequence**, not allele names. The enriched
   stage drops `mhc_*_allele`; we reconstruct via lookup against
   `standardized_again` for splits/foundation but tokenize against the
   protein sequence (or its pocket/contact subsequences).
2. **Biological identity = MD5(tra_cdr3 || trb_cdr3 || peptide || mhc_one_allele || mhc_two_allele)**.
   Used for the exclusion list and partition assignment in Phase 3 (see
   [BENCHMARK_FOUNDATION.md](BENCHMARK_FOUNDATION.md) §4).
3. **No HLA leakage prevention is allele-name-based.** Allele names are
   reconstructed as a courtesy; the actual splitting uses MHC protein
   sequences.
4. **Peptide clustering**: edit distance for interaction scope (5K
   peptides, all-pairs feasible); MMseqs2 fallback for PM scope (4.7M,
   ED infeasible). Documented in [BENCHMARK_CLUSTERING.md](BENCHMARK_CLUSTERING.md) §5.
5. **Per-task molecule allowlists**: AS includes everything, PM drops
   TCR, PAIR drops peptide+MHC, MR drops peptide. See
   [BENCHMARK_TOKENIZED.md](BENCHMARK_TOKENIZED.md) §4.
6. **`<sep>` for O3 ablation = ESM `<eos>` token (id 2).** ESM-2 and
   ESM-C lack a dedicated separator. Training scripts must treat
   mid-sequence EOS as intra-example boundaries.
7. **Random seed 42** for every Phase 2/3/4 sampling/shuffle/split
   operation. Outputs are reproducible.

---

## Reproducibility

```bash
# Phase 1: clustering
python scripts/data_processing/build_benchmark_clusters.py --task all

# Phase 2: splits (depends on Phase 1)
python scripts/data_processing/build_benchmark_splits.py --task all

# Phase 3: foundation manifests + val/test (depends on Phase 2)
python scripts/data_processing/build_benchmark_foundation.py --task all

# Phase 4: tokenization (depends on Phases 2-3)
python scripts/data_processing/build_benchmark_tokenized.py --task all
```

Each phase is independently re-runnable; existing outputs are skipped.

---

## File inventories

See [`inventory.json`](inventory.json) for a machine-readable manifest of
every artifact across all four phases (paths, row counts, sizes).

---

## Known data quality gap: full TCR chains

Approximately **23% of TRA-CDR3-populated rows** and **14.5% of
TRB-CDR3-populated rows** across the benchmark/foundation corpus have
**null** `tra_full` / `trb_full`. The gap is upstream — inherited from
the enrichment stage's TCRStitcher, which can't reconstruct a full
chain when source records lack V/J gene calls.

Per-task breakdown:

| Task | TRA full coverage | TRB full coverage |
|------|-------------------|-------------------|
| AS / PM / MR | ~99% (high — interaction sources mostly stitched) | ~99% |
| PAIR | ~70-72% (bulk paired repertoires dominate, often lacking V/J) | ~70-72% |
| Foundation val/test | ~80% | ~86% |

Audit details:
- `data/benchmark_v2/audit/full_chain_coverage.json` — per-file stats
- `data/benchmark_v2/audit/full_chain_coverage_summary.md` — human
  rollup by task

**Impact on training**:
- **C5 / T3 formats** (full chain inputs) — affected rows produce
  `input_sequence` strings with the missing chain dropped (per the
  empty-molecule rule). Effective row counts for these formats are
  smaller than the partition row count.
- **C1–C4, M1–M3, T1–T2** — unaffected (they don't use full chains).

Closing the gap requires an upstream re-stitch pass; that's out of scope
for benchmark_v2 and tracked as a separate work item. See
[BENCHMARK_TOKENIZED.md §13](BENCHMARK_TOKENIZED.md) for full discussion.
