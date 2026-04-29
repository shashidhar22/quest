# Benchmark Tokenization

Pre-tokenized parquet files for ESM-2 / ESM-C training. Source data:
benchmark splits (`splits/`), foundation full-data + manifests
(`foundation/`). Output: model-ready `input_ids` arrays grouped by task,
benchmark, format variant, and partition.

Outputs live at `data/benchmark_v2/tokenized/` and are produced by
`scripts/data_processing/build_benchmark_tokenized.py`.

---

## 1. Purpose & consumers

Training scripts for ESM-2 and ESM-C MLM / fine-tuning jobs read the
pre-tokenized `input_ids` directly. Each row carries the token-id sequence
(with framing CLS + EOS), the original AA string for verification, the
format the row was tokenized at, and full provenance metadata.

---

## 2. ESM-2 / ESM-C tokenizer equivalence

ESM-2 (`facebook/esm2_t33_650M_UR50D`) and ESM-C
(`EvolutionaryScale/esmc-300m-2024-12`) share an identical AA vocabulary at
positions 0-30 (CLS, PAD, EOS, UNK, all 20 standard AAs, ambiguity codes
X/B/U/Z/O, gap `.`, terminal `-`). Position 31 differs (ESM-2 `<null_1>`
vs ESM-C `|`) but never appears in our data. Position 32 is `<mask>` in
both.

**The tokenized `input_ids` produced here work for both models.** No need
for separate per-model output trees. We use the ESM-2 `AutoTokenizer` for
the encoding step and document the equivalence here.

---

## 3. Format variants

Each format cell maps every molecule type to a list of source columns to
concatenate.

| Cell | TCR (tra/trb) | MHC (mhc_one/mhc_two) | peptide |
|------|---------------|------------------------|---------|
| C1 | `tra_cdr3` / `trb_cdr3` | `mhc_one` / `mhc_two` (full ~250 AA) | `peptide` |
| C2 | `tra_cdr3` / `trb_cdr3` | `mhc_one_pocket_contact` / `mhc_two_pocket_contact` | `peptide` |
| C3 | `tra_cdr1+cdr2+cdr3` / `trb_cdr1+cdr2+cdr3` | `mhc_one_pocket_contact` / `mhc_two_pocket_contact` | `peptide` |
| C4 | `tra_cdr1+cdr2+cdr3` / `trb_cdr1+cdr2+cdr3` | `mhc_one_contact` / `mhc_two_contact` | `peptide` |
| C5 | `tra_full` / `trb_full` (~300 AA) | `mhc_one_pocket_contact` / `mhc_two_pocket_contact` | `peptide` |
| M1 | (n/a — PM only) | `mhc_one` / `mhc_two` (full) | `peptide` |
| M2 | (n/a) | `mhc_one_pocket_contact` / `mhc_two_pocket_contact` | `peptide` |
| M3 | (n/a) | `mhc_one_contact` / `mhc_two_contact` | `peptide` |
| T1 | `tra_cdr3` / `trb_cdr3` | (n/a — PAIR only) | (n/a) |
| T2 | `tra_cdr1+cdr2+cdr3` / `trb_cdr1+cdr2+cdr3` | (n/a) | (n/a) |
| T3 | `tra_full` / `trb_full` | (n/a) | (n/a) |

Concatenation within a chain has **no separator** (e.g., for C3,
`tra_cdr1` + `tra_cdr2` + `tra_cdr3` is a single contiguous string).

---

## 4. Per-task molecule allowlists

Each task drops molecules outside its scope, even if they appear in the row's
`order_key`:

| Task | Emits | Drops |
|------|-------|-------|
| AS | tra, trb, peptide, mhc_one, mhc_two | — |
| PM | peptide, mhc_one, mhc_two | tra, trb |
| PAIR | tra, trb | peptide, mhc_one, mhc_two |
| MR | tra, trb, mhc_one, mhc_two | peptide |
| Foundation | tra, trb, peptide, mhc_one, mhc_two | — |

A row's `order_key` defines the order in which its molecules concatenate.
Molecules absent from the row (null/empty source columns) are skipped
silently — no separator, no placeholder.

---

## 5. order_key parsing

`order_key` is a `_`-delimited list. The molecule names `mhc_one` and
`mhc_two` contain underscores, so a naive split would produce
`['mhc', 'one', 'mhc', 'two']`. Parser handles this:

```python
def parse_order(order_key: str) -> List[str]:
    parts = order_key.split('_')
    out = []; i = 0
    while i < len(parts):
        if parts[i] == 'mhc' and i + 1 < len(parts):
            out.append(f'mhc_{parts[i+1]}'); i += 2
        else:
            out.append(parts[i]); i += 1
    return out
```

Examples:
- `tra_trb_peptide_mhc_one_mhc_two` → `['tra','trb','peptide','mhc_one','mhc_two']`
- `peptide_mhc_one_trb_mhc_two` → `['peptide','mhc_one','trb','mhc_two']`

---

## 6. Empty-molecule and `<sep>` semantics

- **Empty molecule**: if a molecule's concatenated source columns yield an
  empty string, that molecule is dropped from `input_sequence` entirely. No
  separator inserted.
- **`<sep>` for O3**: foundation/mini's O3 ablation uses ESM's `<eos>`
  token as the molecule separator (id 2 in both ESM-2 and ESM-C). So the
  AA string for O3 looks like `[CLS] AAA <eos> BBB <eos> CCC [EOS]`, with
  inner `<eos>` markers signaling intra-example molecule boundaries. The
  training script must treat mid-sequence EOS as separators, not
  end-of-sequence.

---

## 7. Output schema

Every output parquet has:

| Column | Type | Notes |
|--------|------|-------|
| `input_ids` | `list<int32>` | Token IDs including framing `<cls>` ... `<eos>` |
| `input_sequence` | string | The AA string we encoded (kept for verification) |
| `format_cell` | string | One of `C1..C5`, `M1..M3`, `T1..T3` |
| `order_key`, `subset_key` | string | Preserved |
| `source_db` | string | From Phase 2 lookup (renamed from `source_db_primary`) |
| Original molecule columns | string | Whatever the format references (e.g., `tra_cdr1`, `mhc_one_pocket_contact`) |
| Partition-specific columns | various | `condition`, `bio_hash`, `row_id`, `tcr_cluster`, `pep_cluster`, etc. — preserved when present in input |

For benchmark splits the input parquets already have most metadata
(`source_db_primary, source_type, mhc_one_allele, mhc_two_allele,
tcr_cluster, pep_cluster, condition, row_id`). For foundation outputs the
schema is sparser — see §10.

---

## 8. Output file inventory

```
tokenized/
├── as/
│   ├── as_trb_i_{C1..C5}_{train,val,test_iid,test_novel_tcr,test_novel_pep,
│   │                       test_novel_allele,test_level4,test_mixed}.parquet
│   ├── as_tra_i_*.parquet  (same partitions × C1..C5)
│   ├── as_paired_i_*.parquet
│   └── as_{trb,tra,paired}_ii_C3_eval.parquet
├── pm/
│   └── pm_{i,ii}_{M1,M2,M3}_{train,val,test_iid,test_novel_pep,
│                              test_novel_allele,test_level4}.parquet
├── pair/
│   └── pair_{T1,T2,T3}_{train,val,test_iid,test_novel_tra,
│                          test_novel_trb,test_novel_both}.parquet
├── mr/
│   ├── mr_{trb,tra,paired}_i_{C1..C5}_*.parquet
│   ├── mr_trb_ii_C3_{test_iid,test_novel_allele}.parquet
│   └── mr_{tra,paired}_ii_C3_eval.parquet
├── foundation/
│   ├── foundation_val_C3.parquet
│   ├── foundation_test_C3.parquet
│   ├── foundation_10M_C3.parquet
│   ├── foundation_100M_C3.parquet
│   └── foundation_500M_C3/
│       ├── shard_0000.parquet
│       ├── shard_0001.parquet
│       └── ...
└── mini/
    ├── mini_M2_10M_C3.parquet
    ├── mini_M3_10M_C3.parquet
    ├── mini_O1_10M_C3.parquet
    └── mini_O3_10M_C3_sep.parquet
```

Total: ~280 benchmark + 5 foundation + 4 mini ≈ 290 files (plus ~100
shards inside `foundation_500M_C3/`).

---

## 9. Sequence-length expectations

Single-chain example token counts (one `<cls>` + AAs + one `<eos>`):

| Format | Expected len | Note |
|--------|--------------|------|
| C1 | ~380 | CDR3 ~15 + full MHC ~365 |
| C2 | ~62 | CDR3 ~15 + pocket+contact ~47 |
| C3 | ~97 | CDR1+2+3 ~50 + pocket+contact ~47 |
| C4 | ~71 | CDR1+2+3 ~50 + contact ~21 |
| C5 | ~347 | full chain ~300 + pocket+contact ~47 |
| M1 | ~265 | peptide ~10 + full MHC ~250 |
| M2 | ~57 | peptide ~10 + pocket+contact ~47 |
| M3 | ~31 | peptide ~10 + contact ~21 |
| T1 | ~17 | CDR3 ~15 |
| T2 | ~52 | CDR1+2+3 ~50 |
| T3 | ~302 | full chain ~300 |

Paired examples (with both TRA + TRB) are roughly double the TCR portion.

`tokenization_summary.json` reports actual min/median/mean/p95/max per
format after tokenization.

---

## 10. Foundation manifests → source-row resolution

The foundation manifests (`foundation_10M.parquet`, `100M`, `500M`,
`mini_*.parquet`) carry only `(bio_hash, order_key, subset_key,
source_file, source_row_index)` — no sequence data. Tokenization joins
each row back to its source enriched parquet via
`(source_file, source_row_index)`.

Implementation: group manifest rows by `source_file`, then for each source
file in parallel, read only the columns the format needs, look up rows by
`source_row_index`, build the AA string, and tokenize. Writes to either
single parquet (10M, 100M) or sharded directory (500M).

Foundation outputs include only the format-referenced molecule columns
plus `bio_hash, source_file, source_row_index, order_key, subset_key,
input_sequence, input_ids, format_cell`. They do **not** include
condition, tcr_cluster, etc. (those are split-specific). They do **not**
include `source_db` for foundation manifests (would require an additional
lookup join — left as a future enhancement); foundation_val.parquet and
foundation_test.parquet inherit Phase 3's missing `source_db_primary`.

---

## 11. Reproducibility & commands

Versions:

- transformers 4.48.1 (downgraded from 5.x by `pip install esm`)
- esm 3.2.1.post1 (EvolutionaryScale package, provides ESM-C tokenizer
  for verification only — actual tokenization uses ESM-2 since vocabs are
  equivalent)
- pyarrow / duckdb (repo env)
- ESM-2 model: `facebook/esm2_t33_650M_UR50D` (cached on first run)

Commands:

```bash
# Full pipeline
python scripts/data_processing/build_benchmark_tokenized.py --task all

# Or per-step
python scripts/data_processing/build_benchmark_tokenized.py --task benchmark
python scripts/data_processing/build_benchmark_tokenized.py --task foundation_val_test
python scripts/data_processing/build_benchmark_tokenized.py --task foundation_manifests
python scripts/data_processing/build_benchmark_tokenized.py --task mini
python scripts/data_processing/build_benchmark_tokenized.py --task summary
```

Each step is independently re-runnable; existing outputs are skipped.

---

## 12. Known deviations from spec

- **Single tokenized output tree** instead of `tokenized/esm2/` and
  `tokenized/esmc/` separate trees, because the two tokenizers produce
  identical IDs for AA-only sequences. Documented above (§2).
- **`<sep>` token resolved as `<eos>`** (id 2). ESM-2/C have no dedicated
  separator. Training scripts must interpret mid-sequence EOS as
  intra-example boundaries.
- **`pair_negatives_train.parquet` is not tokenized** — it carries only
  `(tra_cdr3, trb_cdr3, label)` with no full sequence data. Generate
  negatives at training time from `pair_train_T*.parquet`.
- **Foundation manifests don't carry `source_db`** in the tokenized
  output. The benchmark splits do (renamed from `source_db_primary`).
- **`source_db_primary` column renamed to `source_db`** on output (the
  spec uses the latter name).

---

## 13. Upstream coverage gap: missing `tra_full` / `trb_full`

Approximately **23% of TRA-CDR3-populated rows** and **14.5% of
TRB-CDR3-populated rows** in the benchmark/foundation corpus have
`tra_full` / `trb_full` set to **null**. The full-chain values come from
the upstream enrichment stage (TCRStitcher), which fails to emit a full
chain when source records lack V/J gene calls. This is **inherited from
the source data**, not introduced by benchmark_v2 phases.

Coverage stats (from
`data/benchmark_v2/audit/full_chain_coverage.{json,md}`):

| Task | TRA cdr3 → full coverage | TRB cdr3 → full coverage |
|------|-------------------------|--------------------------|
| AS Class I/II | high (~99%) | high (~99%) |
| PM-I / PM-II | very high (~99.99%) | very high (~99.99%) |
| PAIR | ~70-72% (paired bulk repertoire dominates) | ~70-72% |
| MR | high | high |
| Foundation val/test | ~80% (TRB-only subsets dominate) | ~86% |

**Implications**:
- **C5 (TCR full chain) format outputs**: rows missing tra_full /
  trb_full will produce `input_sequence` strings with that molecule
  dropped (per the empty-molecule rule in §6). Models trained on C5 see
  fewer rows than the partition's row count suggests for paired
  examples missing one chain's full sequence.
- **T3 (PAIR full chain)** affected similarly.
- **C1-C4 / M1-M3 / T1-T2** are unaffected — they don't use full
  chains.

To fully close the gap, an upstream re-stitch pass would need to be
applied to source rows missing V/J gene calls. That is **out of scope
for benchmark_v2**; tracked separately.
