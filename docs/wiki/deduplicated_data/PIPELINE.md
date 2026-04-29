# Deduplication Pipeline

This page narrates the four phases that produce `data/deduplicated_again/`. The driver is [`tcrbench_dedup.py`](../../../tcrbench_dedup.py); MHC enrichment runs as a separate post-step in [`scripts/data_processing/add_mhc_pseudosequences.py`](../../../scripts/data_processing/add_mhc_pseudosequences.py).

`pipeline_run_log.json` from the production run shows the wall times:

| Phase                  | Wall time | Rows out          |
|------------------------|----------:|-------------------|
| Step 1: dedup          |  4615.1 s | 1,439,165,664     |
| Step 2: counts         |   380.2 s | (writes JSON/CSV) |
| Step 3: explosion      |   466.9 s | 1,461,360,146     |
| Step 4: MHC enrichment | (separate run, see `add_mhc_pseudosequences.py`) | unchanged |

## Step 0 — Preflight

`tcrbench_dedup.py:244-297`. Validates disk space (warns < 2 TB free), checks the file-descriptor limit (`ulimit -n` ≥ 65536 — required because step 3 opens many partition writers), prints the DuckDB version, and globs the input pattern. The production run pointed `--input` at `data/standardized_again/**/*.parquet`, picking up all 18 standardized buckets.

## Step 1 — Sanitize + dedup (3 stages, single DuckDB session)

`tcrbench_dedup.py:332-510` (`step1_sanitize_dedup`).

The five "primary columns" used for deduplication and explosion are **`tra_full, trb_full, peptide, mhc_one, mhc_two`** (`tcrbench_dedup.py:41`, `DEDUP_COLUMNS`). CDR1/CDR2/CDR3 columns for both chains are *carried alongside* (via `FIRST()` aggregation) but do not participate in the dedup key (`tcrbench_dedup.py:46-50`).

### Stage 1a — Sanitize (single materialized temp table)

`tcrbench_dedup.py:377-432`. One `CREATE TEMP TABLE sanitized AS SELECT ...` over `read_parquet('<input>/**.parquet', hive_partitioning=true)`. Per-column sanitization (`build_sanitize_expr`, `tcrbench_dedup.py:183-205`) applies, in order:

1. `TRIM` + `NULLIF('', '')` so empty strings become NULL.
2. **Coalesce** for TCR fields: `tra_full` falls back to `tra` (CDR3 only) when full-length is null, same for `trb_full` (`tcrbench_dedup.py:67-70`).
3. **AA-only regex check**: `^[ACDEFGHIKLMNPQRSTVWY]+$` — non-amino-acid characters cause the value to be NULLed (`tcrbench_dedup.py:56-57, 199-200`). Applies to `tra_full, trb_full, peptide, mhc_one, mhc_two` *and* the six CDR columns.
4. **Length rules** (`tcrbench_dedup.py:73-79`):
   - `tra_full`, `trb_full`: ≥ 4 (no upper bound)
   - `peptide`: 8–30
   - `mhc_one`, `mhc_two`: ≥ 4

Note that `mhc_one` / `mhc_two` here hold the **full MHC protein sequence** (the standardizer wrote sequences, not allele IDs, into these columns; the AA regex would otherwise reject HLA-style allele strings like `HLA-A*02:01`). The allele identifiers live in `mhc_one_allele` / `mhc_two_allele` upstream and are not carried into dedup.

The sanitization is materialized once into a temp table (commented "H1" optimization at `tcrbench_dedup.py:425-432`) so the salvage UNION ALL below can scan it three times without re-executing the parquet read + regex.

### Stage 1b — Salvage (split unreliable rows into two halves)

`tcrbench_dedup.py:404-457`. The `salvage_condition` is `(_binding = 'neg' OR (_source = 'vdjdb' AND _score = '0'))` (`tcrbench_dedup.py:423`). Rows matching it are *not* dropped — they are split into two virtual rows:

- **TCR-side**: keep `tra_full`, `trb_full`, `tra_cdr*`, `trb_cdr*`; null out `peptide`, `mhc_one`, `mhc_two`.
- **Antigen-side**: keep `peptide`, `mhc_one`, `mhc_two`; null out TCR + CDR columns.

This means a negative-binding (or vdjdb-score-0) row contributes its TCR sequence to the global TCR pool and its peptide-MHC tuple to the pMHC pool, but no row in the deduped output ever asserts an interaction we don't trust. (See [Input Data Curator role in `CLAUDE.md`](../../../CLAUDE.md#5-input-data-curator) for why this matters.)

### Stage 1c — Dedup (single GROUP BY)

`tcrbench_dedup.py:434-462`. The full dedup is one DuckDB query:

```sql
CREATE TABLE deduped AS
SELECT tra_full, trb_full, peptide, mhc_one, mhc_two,
       FIRST(tra_cdr1) AS tra_cdr1, ...  -- one FIRST() per CDR column
FROM (
    SELECT ... FROM sanitized WHERE NOT <salvage_condition>
    UNION ALL
    SELECT <tcr_side>     FROM sanitized WHERE <salvage_condition>
    UNION ALL
    SELECT <antigen_side> FROM sanitized WHERE <salvage_condition>
) combined
GROUP BY tra_full, trb_full, peptide, mhc_one, mhc_two
```

So "deduplication" here is **a single global GROUP BY on the 5 dedup columns**. There is no separate "exact hash → CDR3 near-dedup → cross-source" 3-stage flow — that pattern is in the legacy `deduplicate_streaming.py` (which uses sort + uniq on a tab-delimited dedup key, see `deduplicate_streaming.py:77-113, 523-556`). The DuckDB version collapses all three into the GROUP BY because (a) the standardizer already enforced the same length / AA / coalesce rules, and (b) cross-source deduplication is implicit in the global group-by — every (tra_full, trb_full, peptide, mhc_one, mhc_two) tuple is unique in the output regardless of which standardized bucket it came from.

`source` and `study_id` are **not** carried — once a row is deduped, its provenance is lost. To recover provenance, join back to `data/standardized_again/`.

### Stage 1d — Export to parquet + diagnostics

`tcrbench_dedup.py:464-509`. Writes `deduped_parquet/` via `COPY (SELECT * FROM deduped) TO '<dir>/' (FORMAT PARQUET, PER_THREAD_OUTPUT true, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)`. Logs per-column non-null fraction (`tcrbench_dedup.py:471-484`) and an empty-string sanity check (`tcrbench_dedup.py:489-495`).

The table is left resident in `deduped.db` (~158 GB) so steps 2 and 3 can re-use it without re-reading the input. Re-running with `--force_recompute` drops it.

## Step 2 — Counts

`tcrbench_dedup.py:526-595` (`step2_counts`).

Two count tables are written from the same `deduped` table:

- **`combination_counts.json`** — 11 explicit biological combinations (`COMBINATIONS`, `tcrbench_dedup.py:82-94`). These are the column-tuples that actually originate from biological records (e.g. the standardizer's pMHC-only rows produce `peptide_mhc_one`; the bulk repertoire data produces `trb` only; the TCR-pMHC interaction data produces `tra_trb_peptide_mhc_one`).
- **`permutation_counts.json`** — all 31 non-empty subsets of the 5 dedup columns (`SUBSETS`, `tcrbench_dedup.py:100-111, 135`). For a `subset_key`, the count is `SELECT COUNT(*) FROM (SELECT DISTINCT <cols> FROM deduped WHERE <cols IS NOT NULL>)` (`_count_unique`, `tcrbench_dedup.py:512-524`).

The two diverge because every powerset projection of a populated combination *also* counts as a subset_key. Example: `peptide_mhc_one_mhc_two` is in both files, but the `permutation_counts.json` value (292,468) includes records that originally also had a `tra` and/or `trb` populated — the projection drops those columns. The `combination_counts.json` value (also 292,468) for `peptide_mhc_one_mhc_two` is the count where `tra` and `trb` are *exactly* null. Where both files agree numerically (most rows of `combination_counts.json`), it's coincidence — there happen to be no records with the listed columns *and* extra populated columns. Where they disagree (e.g. `peptide` 16.7M only in `permutation_counts.json`), the subset_key represents a projection of larger combinations.

H3 optimization comment (`tcrbench_dedup.py:514-519`) — each label is its own query so DuckDB only holds one DISTINCT hash set in memory at a time; the previous version's `UNION ALL` of 31 DISTINCT branches was an OOM source.

## Step 3 — Explosion

`tcrbench_dedup.py:597-742` (`step3_explode`).

The 5-column powerset has 31 non-empty subsets ("masks"). For each subset of size `n`, the explosion writes `n!` ordered permutations as separate parquet partitions. Total = 5 + 5·4 + 10·3! + 10·4! + 5! = **325** ordered partitions (`PERMUTATIONS`, `tcrbench_dedup.py:114-138`).

The output layout is `exploded_deduped/subset_key=<unordered>/order_key=<ordered>/*.parquet`. Both partition labels are Hive-style.

The explosion logic is grouped by mask (the H2 optimization, commented `tcrbench_dedup.py:600-614`):

1. For each of the 31 masks, build a temp table `m_subset` containing the DISTINCT projection on just the in-mask columns (carrying CDRs via `FIRST()`):
   ```sql
   CREATE TEMP TABLE m_subset AS
   SELECT <in_mask_cols>, FIRST(<cdr_cols>) ...
   FROM deduped WHERE <every in_mask col IS NOT NULL>
   GROUP BY <in_mask_cols>
   ```
   The cardinality of `m_subset` is the `permutation_counts.json` value for that subset.
2. For each of the (≤ `n!`) ordered permutations sharing this mask, run a single `COPY` that emits the in-mask columns + the `n!` typed `CAST(NULL AS VARCHAR)` placeholders for out-of-mask columns + a `CONCAT_WS(' ', <ordered cols>)` joined `sequence` column. The order of `ordered_cols` is the permutation, so `sequence` is what a trainer reads when it consumes that `order_key=`.
3. `DROP TABLE m_subset` before moving on, so peak memory is bounded by the largest single subset.

Why explode rather than just store `m_subset` once and let the trainer permute at load time? Because the `sequence` column is pre-built for the requested order, the trainer can do a zero-copy parquet scan and feed straight into tokenization without per-batch concatenation. The cost is ~5× disk amplification (1.46B exploded rows vs 1.44B deduped rows on average; for the largest 5-column subset the multiplier is 120×, but it has only 565 records).

A **typed-NULL** detail at `tcrbench_dedup.py:668-676`: out-of-mask columns are written as `CAST(NULL AS VARCHAR)` rather than untyped NULL, so cross-partition reads (e.g. DuckDB scanning all 325 partitions at once) get a uniform schema. Without this, untyped NULL would default to INTEGER and the read would fail.

After all 31 masks are written, a validation query walks `subset_key=*/order_key=*` directories, verifies the 325 expected partitions are present, and re-counts each via `read_parquet('<exploded_dir>/**/*.parquet', hive_partitioning=true) GROUP BY subset_key, order_key` (`tcrbench_dedup.py:706-740`).

## Step 4 — MHC enrichment

`scripts/data_processing/add_mhc_pseudosequences.py` (entrypoint `main()`, `add_mhc_pseudosequences.py:342-433`).

Reads `exploded_deduped/`, writes `exploded_deduped_enriched/` with the same partition layout plus 6 added columns:

- `mhc_one_pocket` — 37 binding-pocket positions on the Class I α1/α2 helices (`MHCI_POCKET_POSITIONS`, `add_mhc_pseudosequences.py:34-38`).
- `mhc_one_contact` — 21 TCR-contact-loop positions (`MHCI_CONTACT_POSITIONS`, `add_mhc_pseudosequences.py:40-43`).
- `mhc_one_pocket_contact` — union of the two (47 positions).
- `mhc_two_pocket` / `_contact` / `_pocket_contact` — analogous α+β extractions for Class II (`add_mhc_pseudosequences.py:49-63`), with α+β concatenation handling for sequences ≥ 380 aa (`CLASS_II_CONCAT_THRESHOLD`, `add_mhc_pseudosequences.py:65, 141-181`).

Three phases (`add_mhc_pseudosequences.py:344-419`):

1. **Scan** — parallel scan of every parquet to collect the unique set of `mhc_one` and `mhc_two` strings (the pocket extraction is a function of the sequence, so we only need to compute it once per unique allele).
2. **Build lookup** — for each unique sequence, run `class_i_offset` (`add_mhc_pseudosequences.py:75-93`, motif-based detection of mature position 1 via `SHSMRYF`/`SMRYF`) or `class_ii_beta_offset` / `class_ii_alpha_offset` (`add_mhc_pseudosequences.py:96-112`, motif lists). The result is a 3-tuple per sequence. Lookup tables are persisted to `mhc_pseudo_lookup.json` (~8 MB) so the per-file workers can `mmap` it cheaply.
3. **Process** — `ProcessPoolExecutor` (60 workers) iterates each input parquet, appends 6 string columns via the lookup, and writes to the output partition with the same path under `exploded_deduped_enriched/`.

Validation uses the HLA-A*02:01 signature (M at mature pos 45, Y at pos 116) on a sample of Class I sequences as a sanity check that the offset detection is producing canonical Bjorkman numbering (`validate_class_i`, `add_mhc_pseudosequences.py:235-281`).

## Companion: legacy streaming variant

`scripts/data_processing/deduplicate_streaming.py` is the **older** dedup driver. It uses external Unix-style sort + uniq on tab-delimited dedup keys instead of DuckDB. Highlights:

- 3-stage pipeline: **molecule dedup** (sort + uniq on the dedup key, `deduplicate_streaming.py:77-113, 365-378, 523-556`) → **permutation generation** (parallel batched, `deduplicate_streaming.py:558-672`) → **permutation dedup** (sort + uniq again, `deduplicate_streaming.py:674-701`).
- DuckDB-free; usable on machines where 800 GB RAM isn't available.
- Same conceptual output (deduped molecules + exploded permutations) but writes a single CDR3-version parquet and a single full-length-version parquet without the `subset_key=/order_key=` partitioning. Also emits `permutation_key` and `sequence` columns directly.
- Was not the producer of the artifacts under `data/deduplicated_again/`. Retained as the fallback path documented in [`CLAUDE.md`](../../../CLAUDE.md#data-pipeline) for environments that can't run the DuckDB pipeline.

If you need to regenerate from raw streaming inputs without DuckDB, use that script. Otherwise, `tcrbench_dedup.py` is the preferred path.
