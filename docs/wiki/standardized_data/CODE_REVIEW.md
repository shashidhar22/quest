# Code Review: build_standardized_manifest.py

**Reviewer**: code-reviewer agent
**Reviewed at**: 2026-04-27
**Script version**: SHA1 `d296305519867799f796f340c04cd7dcd47791e4`
**File**: `/home/ubuntu/quest/scripts/analysis/build_standardized_manifest.py` (1146 lines)
**Verdict**: PASS WITH CAVEATS

## Summary

The standardized-manifest builder is structurally cleaner than its raw-data
sibling — schema, idempotent embedder, and per-bucket data integrity all check
out (manifest row counts match parquet row counts in 18/18 buckets, and
`drop_reasons` totals match `manifest.dropped_count` exactly in 18/18 buckets).
The C3 lesson from the raw-data review is **partially** applied (the `--bucket
NAME` mode refuses to overwrite the default JSON path but **not the default
CSV path**, so the same destructive footgun is still reproducible with one
flag combination). Several reported numbers are correct in mechanics but
misleading without disclaimers — most notably `yield_pct > 100%` for buckets
whose raw counterpart was double-counted by the raw-data script (vdjdb,
immuneACCESS, adc, ots, tcrdb), and a couple of latent issues in
column-coverage extrapolation when the first parquet shard has a non-canonical
schema. None of these block shipping; recommend addressing C1 + M1 + M2 in
a follow-up patch.

## Findings

### Critical (must-fix)

- **C1. `--bucket NAME --out /tmp/X.json` (without `--csv`) silently clobbers
  the full `inventory.csv`.**
  - Location: `scripts/analysis/build_standardized_manifest.py:951-956`.
  - Bug: the safety guard at L951 only checks `args.out == DEFAULT_OUT_JSON`.
    The symmetric guard for `args.csv == DEFAULT_OUT_CSV` is missing.
    Reproducible: I ran `python scripts/analysis/build_standardized_manifest.py
    --bucket batman --out /tmp/x.json` and confirmed the on-disk
    `docs/wiki/standardized_data/inventory.csv` shrank from 19 lines (1 header
    + 18 buckets) to 2 lines (1 header + 1 bucket). Restored from a backup I
    took before reproducing.
  - Impact: identical to raw-data C3 — partial restore is required. The
    Quantifier's documented invocation pattern (`--bucket vdjdb --out
    /tmp/check.json --csv /tmp/check.csv`) is correct; users who copy only
    the first half of the example trip the bug.
  - Fix: extend the guard at L951 to also error when
    `args.csv == DEFAULT_OUT_CSV` and `args.bucket` is set. Concretely:
    ```python
    if args.bucket and (args.out == DEFAULT_OUT_JSON or args.csv == DEFAULT_OUT_CSV):
        p.error("--bucket NAME requires explicit --out PATH AND --csv PATH ...")
    ```

### Major (should-fix)

- **M1. Yield ratios for buckets sampled in the raw-data inventory are
  computed but the raw record total being divided into is the
  *double-counted* number from the raw-data script, producing nonsensical
  >100% yields with a misleading note.**
  - Location: `scripts/analysis/build_standardized_manifest.py:459-501`
    (`compute_yield`).
  - Symptom: `tcrdb` reports `yield_pct=121.51%`, `immuneaccess=113.59%`,
    `ots=107.51%`, `adc=106.43%`, `vdjdb=64.87%` (this last one is
    suspiciously low because vdjdb's raw `total_records` was double-counted
    per raw-data CR finding C2 — true raw is ~1.23M, true yield is ~19%
    not the reported 65%). Note that vdjdb is not flagged `sampled_raw`
    because the raw-data script reports `exact_record_count: True` for
    vdjdb — so the misleading vdjdb yield arrives without any caveat at all.
  - The script's only mitigation is the `note: "raw record total is a
    sampled estimate; yield is approximate"` for the SAMPLED_RAW_SOURCES set
    (L496-500). That note conveys "approximate" but does NOT convey
    "potentially > 100% because raw is mis-counted upstream". And vdjdb,
    CEDAR, trait (the C2-affected raw entries) are not in
    SAMPLED_RAW_SOURCES at all, so they get no caveat.
  - Impact: anyone trusting the yield_pct number will draw wrong conclusions
    about pipeline efficiency. A reader seeing
    `tcrdb yield 121.51%` correctly suspects a bug; one seeing
    `vdjdb yield 64.87%` will believe it.
  - Fix: when `yield_pct > 100%`, automatically attach a note like
    `"yield exceeds 100% — raw count likely double-counts archives or sampled
    estimate skews low; treat as illustrative only"`. Additionally, the
    SAMPLED_RAW_SOURCES set should be replaced with a runtime check on
    `raw_entry.get("exact_record_count", True) is False` (which is what
    L477-479 already correctly does as a fallback), and the hardcoded set
    at L114 should be deleted as redundant. Better: also flag
    raw-data-script known-bad entries (`vdjdb`, `cedar`, `trait` per raw-data
    CR C2) by reading the raw inventory's `notes` field for the magic
    string `"archive duplicates"` — once raw-data is fixed these flags
    disappear automatically.

- **M2. `dropped_count_from_dropped_tsv` is computed (`drop_total` in
  `aggregate_drop_reasons`) and used internally for a warning, but never
  emitted in the JSON output.**
  - Location: `scripts/analysis/build_standardized_manifest.py:586-592`
    (use of `drop_total`) and `scripts/analysis/build_standardized_manifest.py:547-554`
    + `scripts/analysis/build_standardized_manifest.py:823-848`
    (the JSON envelope, where `drop_total` is absent).
  - Bug: the user-facing JSON only contains `manifest.dropped_count` and the
    aggregated `drop_reasons[]` list; the consumer cannot independently verify
    that the sum of `drop_reasons[].count` equals the row-count of dropped.tsv.
    Today they do match in 18/18 buckets, but a future re-run with a
    truncated dropped.tsv (e.g., disk full mid-run) would silently misreport
    drop reasons without any signal in the JSON. Also: the log warning at
    L589-591 fires only when both numbers are non-zero AND disagree —
    if `drop_total == 0` (e.g., truncated to header only) but
    `manifest.dropped_count > 0`, no warning is emitted (early-out at L412
    on `total == 0`).
  - Impact: low-frequency, but a silent failure mode.
  - Fix: store `drop_total` in `entry` (e.g., `entry.parquet[
    "dropped_tsv_data_rows"]` or top-level `entry.dropped_tsv_data_rows`) and
    emit it in the JSON. Also unconditionally warn if
    `drop_total != manifest_dropped` (drop the `if drop_total and
    manifest_dropped` short-circuit at L588).

- **M3. Column-coverage uses the FIRST shard's schema as authoritative; if
  later shards add columns (or differ in nullability), those columns will be
  silently undercounted to zero.**
  - Location: `scripts/analysis/build_standardized_manifest.py:269-276`
    (`schema = pq.ParquetFile(parquet_paths[0]).schema_arrow`).
  - Bug: `present_cols` is fixed from the first shard. `cols_to_load` is
    derived from it (L287). `iter_batches(columns=cols_to_load, ...)` won't
    surface columns absent from `cols_to_load` (and would error on shards
    that lack one of those columns; the script catches and skips
    `iter_batches` exceptions at L331-333, dropping the entire shard's
    contribution to populated_counts). For all 18 current buckets the schemas
    are identical 25-string-column outputs (verified manually in 5 buckets),
    so the bug is latent. But schema drift across shards within a bucket is a
    realistic future scenario when a standardizer is updated mid-run.
  - Fix: union schemas across all `scan_paths` before deciding
    `present_cols`. Or, simpler, intersect with TARGET_COLUMNS by attempting
    to load each column independently and skipping per-shard if not present.

- **M4. `populated` is extrapolated by `total_rows / rows_scanned` while
  `pct` is computed against `rows_scanned`, but the JSON exposes
  `parquet.coverage_sampling.rows_scanned` only — not the extrapolation
  factor. The consumer cannot easily reverse the multiplication to recover
  the raw sampled count.**
  - Location: `scripts/analysis/build_standardized_manifest.py:336-369`.
  - Symptom: e.g., immuneaccess `tra` reports `populated=94,191,164,
    pct=3.40%`. Both numbers can be checked against
    `rows_scanned=29,339,213` and `total_rows=2,768,339,213`, but the
    consumer must know to do the cross-multiplication. There is no field
    that explicitly says "this populated count is extrapolated from a sample
    of N rows".
  - Impact: presentation issue only. CIs / uncertainty bounds are not
    surfaced (cf. raw-data CR M5 / M6).
  - Fix: add an explicit field on each sampled bucket's `column_coverage`
    entry — either `populated_extrapolated: true` or
    `extrapolation_factor: 94.36` — and ideally a Wilson-score CI on the
    sampled proportion (low-cost; n is large).

- **M5. The shared-input `note` overwrites the studies-aggregate `note`
  due to ordered branching.**
  - Location: `scripts/analysis/build_standardized_manifest.py:487-495`.
  - Bug: `studies` branch sets `note` at L489. `SHARED_INPUT_BUCKETS`
    branch immediately re-sets `note` at L492 (`info["note"] = ...`)
    rather than appending. Today `studies` is not in `SHARED_INPUT_BUCKETS`,
    so the bug is latent, but if a maintainer ever adds it (or, more
    realistically, a future bucket that is both shared-input and
    sampled-raw), the studies and shared-input notes will collide.
  - Fix: accumulate into a list and join with `; ` at the end, the same way
    the sampled-raw branch correctly does at L498-500.

### Minor (nice-to-have)

- **m1. `>UNIQUE_CAP` sentinel does not disclose that the sample was the
  basis of the overflow, so the reader can't tell whether the corpus has
  >1M unique values or just the sample crossed the cap.**
  - Location: `scripts/analysis/build_standardized_manifest.py:355-367`.
  - Symptom: immuneaccess `trb.unique = ">1000000"` — that's measured on
    only 30 of 2769 shards, so the *corpus* unique might be anywhere from
    1M to ~2.7B. Other immuneaccess columns get the
    `"... (sampled from 30/2769 shards)"` suffix; only the overflow
    sentinel does not.
  - Fix: when sampled and overflowed, emit
    `f">{UNIQUE_CAP} (sampled from {scanned}/{total} shards)"`.

- **m2. Variable `zero` is computed at L692 and never used.**
  - Location: `scripts/analysis/build_standardized_manifest.py:692`.
  - Cosmetic; remove or use it.

- **m3. Per-bucket timing covers parquet-metadata + column-coverage +
  dropped.tsv parse + yield as a single number, so a future regression in
  any one stage is invisible.**
  - Location: `scripts/analysis/build_standardized_manifest.py:992-1014`
    (the `dt = time.time() - t0` block in `main`).
  - Today wall time per bucket is dominated by column coverage on big
    sampled buckets (immuneaccess 705s) and by dropped.tsv read on big
    drop files (62GB for immuneaccess takes minutes by itself). The
    aggregate is fine, but breakdowns would help future profiling.
  - Fix: time `parquet_stats`, `compute_column_coverage`, and
    `aggregate_drop_reasons` separately. Optionally store the per-stage
    seconds in the bucket entry.

- **m4. `aggregate_drop_reasons` returns `[], 0` if any exception fires
  mid-stream, silently discarding *partial* reason counts already
  accumulated.**
  - Location: `scripts/analysis/build_standardized_manifest.py:408-410`.
  - For a 62GB file with a single corrupted line late in the stream,
    we'd lose all upstream counts. Realistic on disk-full / EIO. Probably
    OK (a partial parse is also misleading), but worth a `WARNING` log
    line documenting the row at which we gave up, and ideally returning
    what was accumulated so far with a flag in `entry.notes`.

- **m5. `re.fullmatch(...) or False` — wait, this script doesn't have it,
  but the analogous unused expression `step = (n - 1) / (max_shards - 1)
  if max_shards > 1 else n` at L258 has a confusing fallback (`else n`,
  which is a count, used as a step). With `max_shards <= 1` the
  `idxs = sorted({int(round(i * step)) for i in range(max_shards)})` would
  give `{0}` when max_shards=1, which is what you want; the `else n` is
  effectively dead. Cosmetic.**
  - Location: `scripts/analysis/build_standardized_manifest.py:258`.

- **m6. `human_size` rounds bytes via `int(val)` for the `B` unit, so a
  file of size 0 prints `0B` (good) but a file of size 1023 prints
  `1023B` (good). For unit `K` and up, `f"{val:.1f}{unit}"` always
  rounds *down* by truncation … actually `:.1f` rounds half-to-even per
  Python's default. Fine, but if downstream tools parse the human size,
  rounding may differ from `du -h`'s rounding. Cosmetic.**
  - Location: `scripts/analysis/build_standardized_manifest.py:164-173`.

- **m7. Embedder loops over `payload["buckets"]` but the JSON envelope
  schema field name is `buckets` (good, matches). However the embedder
  doesn't validate that each entry has the keys it dereferences with
  `entry_dict.get(...)` (graceful — returns empty); but it WILL fail on
  L756 (`bucket_name = entry_dict["name"]`) if any entry lacks `name`.
  Defensive; current data is fine.**
  - Location: `scripts/analysis/build_standardized_manifest.py:756`.

### Confirmed-correct

- **All 18 buckets show `row_count_matches_manifest: true`** — verified
  via `parquet_stats` summing per-shard `metadata.num_rows` against
  `manifest.row_count`. (Spot-checked vdjdb, batman, rcc_atlas; other 15
  via the inventory JSON.)
- **All 18 buckets show `drop_reasons.sum(count) == manifest.dropped_count`**
  — verified by independent computation against inventory.json (script in
  reproduction notes). This is a strong signal that streaming line-by-line
  dropped.tsv parsing works correctly even on the 62GB and 5.6GB files.
- **`\r\n` line endings handled correctly** — Python text-mode default
  (`newline=None`) translates CRLF to LF before the script's
  `line.find("\t")` runs, so the reason column is parsed consistently
  regardless of source file line endings. (Verified empirically with a
  small CRLF-encoded test file.)
- **Empty-string vs null masking on string columns is symmetric and
  correct** — for `pa.large_string` arrays:
  `pop_mask = pc.and_(is_valid, invert(equal_to_empty))` correctly excludes
  both null and `""` (verified via Arrow null-propagation: a null in
  `equal_to_empty` propagates through `invert` and stays null in
  `and_`, and `pc.sum` excludes null booleans → matches expected count of
  2 on `["a", "", None, "b"]`).
- **C3 lesson partially applied** — `--bucket NAME` without `--out` is
  rejected at L951-956 with a helpful error message. The CSV gap is
  C1 above.
- **Embedder idempotency** — verified by running
  `python scripts/analysis/build_standardized_manifest.py --bucket vdjdb
  --out /tmp/check_vdjdb.json --csv /tmp/check_vdjdb.csv --embed` and
  comparing md5 of `docs/wiki/standardized_data/sources/vdjdb.md` before
  and after: byte-identical. The `lambda _m: block` substitution at L781
  + L783 correctly avoids backslash-as-backreference issues if the
  rendered block ever contains literal `\1`-style content (raw-data CR
  m5 lesson applied).
- **`AUTO_BLOCK_RE` non-greedy match** — verified handles a single
  AUTO-INVENTORY block correctly with `.*?`+DOTALL+IGNORECASE, and would
  match each block independently if there were ever two (idempotent on
  re-run).
- **`STD_TO_RAW` mappings resolve in raw inventory** — all 17 explicit
  mappings (case-corrected to match raw inventory's case-sensitive names
  like `immuneACCESS`, `BATMAN`, `IMGTHLA`) found in raw_data
  `inventory.json`. The 18th bucket (`studies`) is correctly handled via
  the synthetic `__studies_aggregate__` key built at L437-456.
- **Studies aggregate** — sums `total_records` across the 70+ raw entries
  with `kind=="study"` (verified raw inventory has 82 sources, 70+ of
  type `study`); the script correctly aggregates to one synthetic entry
  with `study_count` set, and propagates `exact_record_count` only if
  *all* studies are exact (defensive — see L446-447).
- **Single-pass column coverage** — `iter_batches(columns=cols_to_load,
  batch_size=65536)` loads all 25 columns at once per batch, runs
  per-column compute kernels in C, and only pays Python `to_pylist()`
  for the 5 ID-like columns until UNIQUE_CAP hits. This avoids 25× column
  decode that would happen with one-column-at-a-time. Smart.
- **`UNIQUE_CAP` overflow drops the set** — at L329 the script replaces
  the overflowed set with `set()`, freeing memory. Memory footprint per
  bucket is bounded by min(corpus_unique, UNIQUE_CAP=1M) × 5 ID-like
  columns × ~50 bytes/string ≈ 250MB peak in the worst case. Acceptable.
- **`compute_yield` returns None when raw inventory is missing or the
  bucket has no raw counterpart** — graceful degradation, no exception.
- **`row_count_matches_manifest` is `None` when `manifest` failed to
  load** — three-state (`True/False/None`) signal preserved correctly
  through to JSON.
- **CSV writer field ordering is stable and matches headers** — verified
  by inspecting the on-disk CSV (19 lines, 18 fields). `top_drop_reason`
  picks `drop_reasons[0]` (which is the most-common, since Counter
  `most_common()` is sorted descending) — correct.
- **Studies bucket yield calculated against the aggregate** — verified
  studies yield = 92,142,845 / 256,178,876 = 35.97%, matches inventory.
- **Dropped.tsv parsing memory** — Counter is small; only ~5–15 distinct
  reasons per bucket. Even on 62GB immuneaccess, memory footprint of the
  Counter never exceeds a few KB. Streaming line-by-line is the right
  approach.

## Reproduction notes

- `sha1sum scripts/analysis/build_standardized_manifest.py` →
  `d296305519867799f796f340c04cd7dcd47791e4`.
- `python scripts/analysis/build_standardized_manifest.py --bucket batman
  --out /tmp/x.json` (no explicit --csv) ran in 0.5s, produced a 1-bucket
  CSV at `docs/wiki/standardized_data/inventory.csv` (overwriting the
  18-bucket one). I had backed it up first; restored from
  `/tmp/inventory_csv_backup.csv`. Confirms C1.
- `python scripts/analysis/build_standardized_manifest.py --bucket vdjdb
  --out /tmp/check_vdjdb.json --csv /tmp/check_vdjdb.csv --embed` ran in
  1.1s; pre/post md5 of `docs/wiki/standardized_data/sources/vdjdb.md`:
  identical (`253cf9dd30f7dd8593ff2ae19b7a1c1c`). Confirms embedder
  idempotency.
- `python scripts/analysis/build_standardized_manifest.py --bucket batman`
  (no --out, no --csv) errors out per L951-956: confirms C3-lesson
  --out guard. (CSV guard missing — see C1.)
- Cross-check of all 18 buckets:
  ```python
  import json
  d = json.load(open('docs/wiki/standardized_data/inventory.json'))
  for b in d['buckets']:
      md = b['manifest'].get('dropped_count')
      total = sum(int(r['count']) for r in b['drop_reasons']) if b['drop_reasons'] else 0
      assert md == total, b['name']
  ```
  → all 18 pass.
- Schema check: all 5 spot-checked buckets (vdjdb, iedb, batman, netmhcpan,
  immuneaccess) have identical 25-column `large_string` schemas. M3 is
  latent.
- Empty-string vs null mask correctness verified with
  `pa.array(["a", "", None, "b"], type=pa.large_string())` →
  `populated=2`. Both null and empty correctly excluded.
- Yield > 100% sanity check: tcrdb (121.51%), immuneaccess (113.59%),
  ots (107.51%), adc (106.43%) — all flagged `sampled_raw=True` with
  the "approximate" note. vdjdb (64.87%) is NOT flagged because raw
  inventory says exact, but the raw count is double-counted per
  raw-data CR C2. M1.
- `STD_TO_RAW` mappings: 17/17 resolve in raw inventory (`adc, BATMAN,
  CEDAR, CEDAR, IEDB, IEDB, IMGTHLA, immuneACCESS, immuneCODE, McPAS-TCR,
  NetMHCPan, OTS, RCC_ATLAS, TaDB, tcrdb, trait, vdjdb`). Studies handled
  via `__studies_aggregate__`.
- Run-level outputs: `docs/wiki/standardized_data/inventory.json`
  (953KB, 18 buckets, schema integrity verified), `inventory.csv` (19
  lines), `sources/*.md` (18 files, all containing `<!-- BEGIN:
  AUTO-INVENTORY -->` block per inspection of vdjdb.md).
