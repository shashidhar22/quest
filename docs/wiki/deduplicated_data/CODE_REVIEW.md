# Code Review: build_deduplicated_manifest.py

**Reviewer**: code-reviewer agent
**Reviewed at**: 2026-04-27
**Script**: `/home/ubuntu/quest/scripts/analysis/build_deduplicated_manifest.py` (1061 lines)
**Script SHA1**: `8907bee338b868384bafe5b8e8cd7189559d48a7`
**Verdict**: PASS

## Summary

The deduplicated-data manifest builder is the cleanest of the three manifest
scripts in this series. The previous footguns (raw-data C3, standardized-data
C1) are correctly closed (both `--out` AND `--csv` are guarded). The
cross-checks have correct semantics — the `expected_total = perm_count *
factorial(n_cols)` relationship is computed and matched against parquet rows,
the dedup-vs-enriched comparison is a strict equality, and the schema diff is
the right direction (`enriched − deduped`). On the live data the script
correctly reports 0 anomalies across 31 subsets / 325 ordered partitions / 639
parquet shards in well under one wall-second, using parquet metadata only.

I have no critical findings and no major findings. Five minor issues / latent
edge cases are noted; none affect the published numbers and most are dormant
(would only fire under future failure modes that the current data doesn't
exhibit).

Recommended action: ship as-is. The minor items are worth a follow-up pass
the next time someone touches the script but do not block.

## Findings

### Critical (must-fix)

_None._

### Major (should-fix)

_None._

### Minor (nice-to-have)

- **m1. Both `discover_subset_keys` and `permutation_counts.json` should be
  cross-checked, but only the former drives the loop.**
  - Location: `scripts/analysis/build_deduplicated_manifest.py:733-741`
    (`discover_subset_keys`) and `scripts/analysis/build_deduplicated_manifest.py:989-992`
    (the loop in `main`).
  - Observation: the script discovers subset_keys from on-disk
    `subset_key=` directories under `exploded_deduped/` and
    `exploded_deduped_enriched/`. It does **not** verify that the discovered
    set matches the keys present in `permutation_counts.json` (or vice
    versa). On the current data both sets contain exactly the same 31
    subset_keys, so it's silent. But: if a future run truncates partway
    through `step3_explode` (e.g. disk full), the script would report only
    the subsets that made it to disk and never warn that
    `permutation_counts.json` lists 31 but only N were inventoried.
  - Mitigation already in place: `inventory_subset` annotates per-subset
    `notes` when one of the two subset dirs is missing (L355, L367). What's
    missing is the symmetric check at the top level: keys present in
    `permutation_counts.json` but with no on-disk subset dir at all.
  - Suggestion: after `discover_subset_keys`, log
    `set(permutations) - set(keys)` and `set(keys) - set(permutations)` at
    WARNING level and surface the diff in `totals.buggy_subsets`. (Pure
    hygiene; doesn't impact today's published numbers.)

- **m2. Empty `order_key=` directories are silently treated as
  `row_count=0` and would only flag as an anomaly if other order_keys had
  non-zero counts.**
  - Location: `scripts/analysis/build_deduplicated_manifest.py:266-281`
    (the inner loop in `_stats_for_subset_dir`).
  - Behaviour: an empty `order_key=` directory contributes
    `file_count += 0`, `rows_here = 0`, and is recorded in `per_ok` with
    value `0`. The anomaly detector at L286-291 catches this only because
    `0 != expected_per_order_key` (when `expected_per_order_key` is
    populated from `permutation_counts.json`). If `permutation_counts.json`
    is missing the subset entirely (currently never the case), the
    fallback `most_common` mode-comparison at L294-302 would still flag a
    pure-zero rotation as an anomaly *only* if at least one other order_key
    had non-zero rows — i.e. an *all-empty* subset would silently report
    "all rotations agree, no anomalies" with `row_count_from_parquet=0`.
  - Suggestion: add an explicit check "if `expected_total is None and
    total_rows == 0` and `len(order_dirs) > 0`, mark the subset as
    `notes: 'all order_key dirs empty'`". Today no subsets are empty so
    this is purely defensive.

- **m3. Non-`data_*.parquet` files in `order_key=` directories are silently
  skipped — including legitimate `_metadata`, `_common_metadata`, `_SUCCESS`,
  or hand-staged shards with different naming.**
  - Location: `scripts/analysis/build_deduplicated_manifest.py:266`
    (`for pf_path in sorted(od.glob("data_*.parquet"))`) and
    `scripts/analysis/build_deduplicated_manifest.py:177`
    (the same restrictive glob in `inventory_top_level`).
  - Observation: today the `od.glob("data_*.parquet")` filter is correct
    because DuckDB's `COPY ... TO ... (FORMAT PARQUET, FILENAME_PATTERN
    'data_{i}')` writes exactly that naming. But other tools (pyarrow's
    `write_dataset` with default basename, Spark's `part-*.snappy.parquet`,
    or a hand-rerun of just one shard) would produce non-matching files
    that the inventory ignores without warning.
  - Verified: I ran
    `find data/deduplicated_again/exploded_deduped/ -mindepth 3 -maxdepth 3
    -type f -not -name "data_*.parquet"` — zero hits today.
  - Suggestion: glob `*.parquet` instead, and log a WARNING if any matched
    file does not match the expected `data_\d+\.parquet` pattern. Tighter
    safety net for almost no extra cost.

- **m4. The "schema captured from the first parquet shard encountered"
  pattern carries the same first-shard-bias caveat as the standardized-data
  script (M3 there).**
  - Location: `scripts/analysis/build_deduplicated_manifest.py:262-277` and
    `scripts/analysis/build_deduplicated_manifest.py:181-186`.
  - Behaviour: `schema_set` is flipped after the first successful parquet
    metadata read; subsequent shards' schemas are not compared. Today every
    subset has a uniform schema (verified — only one `schema_added` set
    appears across all 31 subsets), but a future schema-evolution bug would
    not be detected by this manifest. This is a deliberate trade-off for
    the 0.9s wall time and is consistent with the upstream scripts.
  - Suggestion (optional): compare each shard's schema to the first and
    record divergences. Cheap (still metadata-only) and catches schema
    drift early.

- **m5. `inventory_top_level` parses `mhc_pseudo_lookup.json` into memory
  for the `entry_count` and `entries_by_class` summary — the only place in
  the script that loads payload data rather than just metadata.**
  - Location: `scripts/analysis/build_deduplicated_manifest.py:212-233`.
  - Behaviour: the 7.8 MB JSON is small enough that this is fine, and the
    `try/except` correctly downgrades to `entry_count=None` if parse fails.
    Mentioning only because the script's docstring (`"All counts come from
    pyarrow.parquet.ParquetFile.metadata.num_rows — we never read parquet
    table data into memory."`) is slightly inaccurate for the non-parquet
    artifact.
  - Suggestion: tighten the docstring to "We never read **parquet**
    payload data; the small `mhc_pseudo_lookup.json` is parsed for an
    entry-count summary." Pure docs-hygiene.

### Confirmed-correct

The following items I specifically tested and confirmed work:

- **C1 / standardized-data C1 footgun closed (both axes).**
  Verified by reproduction:
  ```
  $ python scripts/analysis/build_deduplicated_manifest.py --subset tra --out /tmp/check.json
  build_deduplicated_manifest.py: error: --subset NAME without explicit --out
  AND --csv would overwrite the full inventory. Pass both, e.g. --out
  /tmp/check.json --csv /tmp/check.csv.
  ```
  The guard at `scripts/analysis/build_deduplicated_manifest.py:944-949`
  checks `args.out == DEFAULT_OUT_JSON OR args.csv == DEFAULT_OUT_CSV` (the
  symmetric check that was missing in standardized-data C1) and refuses
  cleanly. The `--subset mhc_one_mhc_two --out /tmp/x.json --csv /tmp/x.csv`
  invocation worked end-to-end on the smallest subset (133 rows × 2
  rotations, 2 files, 0.3s wall) with all cross-checks correctly populated
  including `combination_count: null` (the mhc_one_mhc_two subset is not in
  `combination_counts.json`, which is the documented expected behaviour).

- **Cross-check semantics — `parquet_rows == permutation_count × n!`.**
  Located at `scripts/analysis/build_deduplicated_manifest.py:331-334`
  (`expected_total = perm_count * fact if perm_count is not None else
  None`) and `scripts/analysis/build_deduplicated_manifest.py:304-308`
  (the `total_rows == expected_total` equality). Critically this is
  *strict* equality, not tolerance, and it correctly multiplies by the
  factorial — the README claim
  "all `permutation_count × n!` matches parquet rows" is what the code
  actually computes. I independently verified all 31 subsets against the
  inventory JSON: 31/31 deduped and 31/31 enriched match `permutation_count
  × factorial(n_columns)` exactly.

- **Cross-check semantics — `enriched_rows == deduped_rows`.**
  Located at `scripts/analysis/build_deduplicated_manifest.py:370-373`.
  Strict `drows != erows` test, no tolerance. Verified 31/31 subsets pass.

- **Schema-diff direction is correct.**
  Location: `scripts/analysis/build_deduplicated_manifest.py:376-382`.
  Computes `e_schema - d_schema` (added columns) — not the reverse. Then
  bidirectionally compares against `EXPECTED_ENRICHED_ADDED` (line 62-71)
  to populate both `schema_added_unexpected` (extra cols not expected) and
  `schema_added_missing` (expected cols absent). Both are surfaced into
  `totals.buggy_subsets` at L794-803 of `write_outputs`. Across all 31
  subsets the diff is exactly the expected 8-column set
  (`subset_key, order_key, mhc_{one,two}_{pocket,contact,pocket_contact}`)
  — confirmed via:
  ```
  $ jq -r '.subsets | map(.schema_added | sort | join(",")) | unique' inventory.json
  ```
  returns a single-element list.

- **AUTO-INVENTORY block markers are stable and idempotent.**
  Location: `scripts/analysis/build_deduplicated_manifest.py:392-403`
  (`AUTO_BLOCK_RE`, `MASTER_TABLE_RE`, `PLACEHOLDER_RE`) and
  `scripts/analysis/build_deduplicated_manifest.py:567-579`. The
  `lambda _m: block` substitution form correctly avoids re.sub
  backreference interpretation if the rendered block ever contains
  `\<digit>` sequences (it doesn't today, but the discipline is right).
  The replacement priority is sane: existing AUTO-block first, falling
  back to TODO placeholder for first-time embedding. Both the per-subset
  embedder and the master-table embedder follow the same pattern.

- **Master-table aggregation matches headline cross-checks.**
  Independently summed:
    - `sum(s.deduped.row_count_from_parquet)` over 31 subsets =
      `1,461,360,146` = `pipeline_run.explosion_rows_out` ✓
    - `top_level.deduped_parquet.row_count` = `1,439,165,664` =
      `pipeline_run.dedup_rows_out` ✓
  - Both reported correctly in the README master block.

- **Performance is parquet-metadata-only.**
  Verified by reading `inventory_top_level` (L181-188) and
  `_stats_for_subset_dir` (L266-281): both use
  `pyarrow.parquet.ParquetFile(...).metadata.num_rows` and
  `pf.schema_arrow.names`. No `pf.read()` / `read_table` / `iter_batches`
  calls. The 0.9s wall time across 639 parquet shards (= ~1.4 ms per shard
  including the `.stat()` call for size) is consistent with footer-only
  reads.

- **Per-order-key anomaly detection works in two modes.**
  Location: `scripts/analysis/build_deduplicated_manifest.py:285-302`. With
  `expected_per_order_key` from `permutation_counts.json`, every order_key's
  row count is compared against the canonical value. Without (fallback
  branch), it's compared against the mode of `per_ok.values()`. Both modes
  populate `order_key_row_anomalies` with `{order_key, rows, expected}`
  records and surface them via `totals.buggy_subsets`. None triggered on
  the current data (verified: no subset has anomalies in the JSON).

- **CSV schema is complete and matches JSON.**
  All 26 fields in the CSV header (L842-869) are populated for every row;
  spot-checked `mhc_one_mhc_two` matches the JSON. Tri-state
  `deduped_vs_enriched_match` is correctly serialized as `"True"` /
  `"False"` / `""` (L880-884).

## Reproduction notes

I ran (read-only):

1. `python scripts/analysis/build_deduplicated_manifest.py --subset tra --out
   /tmp/check.json` — confirmed argparse error path works.
2. `python scripts/analysis/build_deduplicated_manifest.py --subset
   mhc_one_mhc_two --out /tmp/check.json --csv /tmp/check.csv` — confirmed
   small-subset (133 × 2) inventorying works end-to-end in 0.3s with all
   cross-checks populated.
3. `find data/deduplicated_again/exploded_deduped/ -mindepth 2 -maxdepth 2
   -type d -name "subset_key=*" | wc -l` → 31 ✓
4. `find data/deduplicated_again/exploded_deduped -type d -name "order_key=*"
   | wc -l` → 325 ✓ (= sum of `n!` over 31 subsets)
5. `find data/deduplicated_again/exploded_deduped/ -mindepth 3 -maxdepth 3
   -name "*.parquet" | wc -l` → 639 ✓ (matches sum of CSV `deduped_files`)
6. `find ... -mindepth 3 -maxdepth 3 -type f -not -name "data_*.parquet"`
   → empty (no off-pattern files; m3 dormant)
7. `find ... -type d -empty` → empty (no empty `order_key=` dirs;
   m2 dormant)
8. JSON inspection confirmed all 31 subsets pass all three cross-checks
   (parquet vs `permutation_count × n!`, enriched vs deduped, expected
   schema diff).

I did **not** modify the script, the inventory JSON/CSV, the README, or any
of the per-subset markdown pages.

## Notes on style consistency with prior reviews

- Where the standardized-data review (M3) flagged "first-shard schema bias"
  as Major, I demoted the same pattern here to Minor (m4) because: (a) the
  current data has homogeneous schemas across all shards, (b) the dedup
  output is produced atomically by a single DuckDB `COPY` per partition
  (less prone to per-shard schema drift than the raw-data parquet writers
  upstream), and (c) the script correctly handles the case in the only
  place that matters — the schema diff is computed at the partition level
  (one schema per `subset_key`), not the shard level.
- The `human_size` helper at L120-129 mirrors the upstream scripts'
  formatter, including the same `"P"` final unit cap. Consistent with prior
  reviews; not re-flagged.
- The script is well-commented (especially the docstring, which spells out
  the `permutation_count × n!` relationship and notes that
  `permutation_counts.json` is a misnomer for what's actually a
  per-partition row count). This avoids the C1/C2 footguns that came from
  ambiguous semantics in the upstream scripts.
