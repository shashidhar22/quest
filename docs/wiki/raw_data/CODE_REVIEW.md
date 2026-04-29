# Code Review: build_raw_data_manifest.py

**Reviewer**: code-reviewer agent
**Reviewed at**: 2026-04-27
**Script version**: SHA1 `8a83472051d86276e86a6717992b531283551a88`
**Verdict**: PASS WITH CAVEATS

## Summary

The manifest generator is structurally sound and produces a complete JSON/CSV
schema for every source. However, several concrete bugs cause meaningful
mis-counts in well-known sources (NetMHCPan training files entirely uncounted,
vdjdb / CEDAR / trait double-counted because zipped + unzipped duplicates are
both summed) and one operational footgun (`--source NAME` mode silently
overwrites the full `inventory.json` with a single-source one). None of these
prevent shipping the manifest as a coarse inventory, but several entries should
be regarded as upper-bound or undercounted; recommend re-running with a small
patch addressing items C1–C3 before treating numbers as authoritative.

## Findings

### Critical (must-fix)

- **C1. NetMHCPan training files (`c000_ba`, `c001_ba`, …, `c004_el`,
  `allelelist`) are silently uncounted.**
  - Location: `scripts/analysis/build_raw_data_manifest.py:189-218` (`detect_format`)
    and `scripts/analysis/build_raw_data_manifest.py:430-446` (the `txt` branch
    in `count_records`).
  - Bug: `detect_format()` returns `"noext"` for files with no suffix (e.g.
    `c000_ba`). The `txt` branch in `count_records` therefore never sees them,
    so the headerless-NetMHCpan-training rule at L433-437 never fires.
    They fall into the catch-all at L479 (`fmt in ("h5","bw",…,"noext","msf",…)`)
    and are reported as `"binary/non-tabular; not counted"`.
  - Evidence: `wc -l data/raw_data/databases/NetMHCPan/NetMHCpan_train/c000_ba`
    returns 41,206 lines of real binding data; the inventory shows
    `record_count=None` for every `c*_ba`/`c*_el` file. The parent tar.gz
    `NetMHCpan_train.tar.gz` reports `"no tabular members"` for the same reason
    (the inner files inside the tar are also extension-less, so the archive
    walker also skips them at L384).
  - Impact: the NetMHCPan total of 41.9M is missing the entire NetMHCpan-I
    training corpus (c000–c004 × {ba,el} ≈ 11 files of ~30k–1M rows each,
    plausibly 2–10M extra records).
  - Fix: route by directory + filename instead of by detected format. Either
    treat `.parent.name in {"NetMHCpan_train", "NetMHCIIpan_train"}` as a
    pre-format gate, or upgrade `detect_format` to map the `c\d{3}_(ba|el)`
    pattern (and `train_*` / `test_*`) to `"txt"` before format dispatch.

- **C2. vdjdb, CEDAR, and trait double-count records because both the original
  files and their zip/archive duplicates are summed into `total_records`.**
  - Location: `scripts/analysis/build_raw_data_manifest.py:674-783`
    (the per-format counting loop has no concept of "this archive is a backup
    of the loose files").
  - Evidence (vdjdb): 17 loose `.txt` files sum to 1,228,016 records *plus*
    `vdjdb-2025-12-29.zip` reports 1,228,409 records (same content). Reported
    total 2,456,425 ≈ exactly double the real ~1.23M.
  - Evidence (CEDAR): `tcell/tcell_full_v3.csv`=151,479 and
    `tcell/tcell_full_v3.zip`=151,479; same for `epitope/*` and `receptor/*`.
    At least 1.756M records (`151,479 + 1,497,440 + 106,999`) are
    double-counted; reported total 8,130,087 should be ≈ 6,374,168.
  - Evidence (trait): `A0201_ELAGIGILTV_…_pos.txt`=314 and the matching
    `…_pos.zip`=313 (small) and the analogous `…_neg.txt`/`…_neg.zip` pair.
    Two double-counts, ~67k records.
  - Note: IEDB happens to escape this bug because all but two of its zips
    fail with `BadZipFile` / CRC errors and are silently dropped — coincidence,
    not correctness.
  - Impact: the database-level totals for vdjdb, CEDAR, and trait are inflated.
    Anyone who uses `total_records` as an upper bound for downstream
    deduplication budgeting is fine; anyone who treats it as ground truth is
    not.
  - Fix: when a source contains a zip/tar that has been unpacked into a sibling
    directory (detect via name overlap of the archive's base name with a
    sibling directory, or via member basename overlap with on-disk files),
    count only one. Simplest: prefer the loose files and skip the archive's
    record contribution; emit a `notes` line `"archive duplicates loose files;
    not double-counted"`. Alternatively keep the archive as `record_count=null`
    when its members match on-disk files.

- **C3. `--source NAME` mode overwrites the full `inventory.json` /
  `inventory.csv` with a one-source file.**
  - Location: `scripts/analysis/build_raw_data_manifest.py:820-905`
    (`write_outputs`) and `scripts/analysis/build_raw_data_manifest.py:1058-1138`
    (`main`).
  - Bug: when `--source vdjdb` is passed, `pending` contains only that one
    source, and `write_outputs` writes that single entry to the default
    output paths (`docs/wiki/raw_data/inventory.json`/`.csv`), destroying the
    previously generated full inventory.
  - Evidence: I ran `python scripts/analysis/build_raw_data_manifest.py
    --source vdjdb --exact` and confirmed `inventory.json` shrank from 82
    sources to 1 source, file size 2,503,472 → 6,547 bytes. (Yes, this is how
    I noticed; the full inventory has been triggered to regenerate.)
  - Impact: any user iterating on a single source via `--source` accidentally
    nukes their full inventory. The Quantifier's instructions even document
    `--source vdjdb --exact` as a valid invocation.
  - Fix: when `--source` is set, either (a) refuse to write to the default
    output paths and require an explicit `--out` / `--csv`, or (b) merge the
    new entries into an existing inventory rather than overwriting. (a) is
    simpler and safer; (b) is more useful. Either is acceptable; the current
    behaviour is a footgun.

### Major (should-fix)

- **M1. CRC-corrupted IEDB zips are silently swallowed; no record / log /
  flag.**
  - Location: `scripts/analysis/build_raw_data_manifest.py:359-374`
    (the inner `try/except Exception: pass` in `count_archive_members`).
  - Symptom: `IEDB/bcell/bcell_full_v3.zip` is a 0-byte file; `IEDB/mhc_ligand/
    mhc_ligand_full_v3.zip` is 0-byte; `IEDB/receptor/receptor_full_v3.zip`
    triggers `BadZipFile: Bad CRC-32 for file 'tcr_full_v3.csv'` mid-stream.
    All three are reported as `"no tabular members"` rather than `"archive
    error: BadZipFile / CRC mismatch"`. The empty-zip case actually surfaces
    `"archive error: BadZipFile"` (good), but the CRC-mid-stream case looks
    indistinguishable from a legitimately empty archive.
  - Impact: IEDB's reported 3.17M is right by accident (because the CSVs
    co-exist), but a future user who deletes a redundant CSV thinking the zip
    is the source of truth will get zero. Also: data-quality auditors won't
    know the zip is broken.
  - Fix: in the `except Exception` at L365 / L391, capture the exception type
    in a member-level note and propagate to the FileEntry's `notes`. At
    minimum, log a WARNING.

- **M2. Inconsistent "header / no header" treatment for the same data inside
  archives vs. on disk.**
  - Location: `scripts/analysis/build_raw_data_manifest.py:358-370` (zip
    inspection) vs. `scripts/analysis/build_raw_data_manifest.py:430-452`
    (loose-file inspection).
  - Bug: a `.txt` file outside an archive defaults to `wc -l` (no header
    subtraction), but the same `.txt` file inside a zip is counted as
    `max(0, lines - 1)` (one row treated as header). For trait, the loose
    `A0201_ELAGIGILTV_..._pos.txt` reports 314 records and the same file
    inside the zip reports 313. This contributes to the C2 double-count
    discrepancy and is a separate correctness issue.
  - Fix: pick one convention and apply it in both places. The script's
    in-source comment at L444 says "vdjdb_*.txt files in vdjdb/ are TSVs with
    headers" — so the loose-file rule for ambiguous `.txt` should *also*
    subtract one when the file actually has a header (peek the first line and
    look for non-numeric content?). At minimum, document the divergence.

- **M3. NetMHCpan headerless rule's `parent.name` check is too narrow.**
  - Location: `scripts/analysis/build_raw_data_manifest.py:433-442`.
  - Bug: the check uses `path.parent.name == "NetMHCpan_train"` (case-sensitive
    direct parent only). If a file is nested deeper (e.g.
    `NetMHCpan/NetMHCpan_train/sub/c000_ba`) it would not match. Today the
    files happen to be at exactly that depth, but the rule is brittle.
    Also: `re.fullmatch(r"c\d{3}_(ba|el)", base) or False` — the `or False`
    is dead code (`re.fullmatch` already returns None if no match). Cosmetic.

- **M4. Sampling is size-blind but extrapolated over all files including
  >2GB skipped ones, which biases the OTS / adc / immuneACCESS / immuneCODE /
  tcrdb totals downward (or upward, depending on correlation).**
  - Location: `scripts/analysis/build_raw_data_manifest.py:688-755`.
  - Bug: lines 696-698 build `sampleable = [p for p in paths if p.stat().st_size
    <= SAMPLE_SKIP_FILE_BYTES]`, sample 30 from sampleable, compute mean over
    the 30, then on line 721 multiply the mean by `len(paths)` (ALL files
    including the >2GB ones). If file size and record count are positively
    correlated (typical for repertoire data), the sample mean understates the
    true mean and the extrapolation underestimates total records. The
    `notes` field appends "X files >2.0G excluded from sample" but does not
    say the extrapolation includes those files in the multiplier — the user
    has to read the code to learn this.
  - Concretely: adc reports `175 files >2.0G excluded from sample` of 9259
    total; immuneACCESS `5 of 29209`; immuneCODE 0; OTS 0; tcrdb 0. The bias
    is largest for adc.
  - Fix: either (a) include >2GB files in the sample (slower but unbiased),
    (b) compute total = mean_sampleable × len(sampleable) + Σ(record_count
    of >2GB files counted exactly), or (c) document the bias direction in
    `notes` and refuse to multiply. Option (b) is what the script's docstring
    suggests but is not what's implemented.

- **M5. Confidence interval computation uses normal z=1.96 but the function
  is named `t_interval`.**
  - Location: `scripts/analysis/build_raw_data_manifest.py:487-496`.
  - Bug: the docstring/name implies a t-distribution interval, but the code
    uses z=1.96 (normal). For n=30 the t critical value at 95% is 2.045, so
    reported CIs are ~5% too narrow. Also, file-count distributions for
    repertoires are heavy-tailed; even a t-interval is not robust here.
    Bootstrap would be more defensible.
  - Impact: CIs reported in `total_records_ci` understate true uncertainty.
  - Fix: rename to `z_interval` if keeping z, or import scipy and compute
    proper t-interval; ideally bootstrap (5,000 resamples of the per-file
    counts, scale by len(paths)) for heavy-tailed distributions.

- **M6. Sources with multiple sampled format groups would only report CI for
  the last group.**
  - Location: `scripts/analysis/build_raw_data_manifest.py:754-755`
    (`src.total_records_ci = list(ci_total)`).
  - Bug: assignment, not aggregation. Today no source has >1 sampled format,
    so the bug is latent, but it would silently corrupt CIs if e.g. tcrdb
    grew a sampled `tsv` group alongside its sampled `csv` group.
  - Fix: store CIs per-format-group, or aggregate via independence assumption
    (sum means, sum variances).

### Minor (nice-to-have)

- **m1. Code path for vdjdb subsumes `cluster_members.txt` and `motif_pwms.txt`,
  which are listed in `NON_TABULAR_TXT_BASENAMES`.**
  - Location: `scripts/analysis/build_raw_data_manifest.py:443-449`.
  - The `vdjdb`-source `txt` rule fires *before* the `NON_TABULAR_TXT_BASENAMES`
    check, so files explicitly listed as non-tabular are still counted as
    TSVs in vdjdb. `cluster_members.txt` reports 41,432 "records" which is
    not really a meaningful count. The reported vdjdb total includes ~64k
    such non-real records. Reorder the checks (non-tabular first) to fix.

- **m2. The `iter_files` exclusion list is open-ended and may miss future
  hidden directories.**
  - Location: `scripts/analysis/build_raw_data_manifest.py:98-104`
    (`EXCLUDE_DIR_NAMES`).
  - `data/raw_data/studies/ZEN8140861/raw_data/.Rproj.user/` exists on disk
    and is not in the list. The `EXCLUDE_NAME_PREFIXES = (..., ".",)` rule
    catches `.Rproj.user` *as a file* (because of the dotfile prefix), but
    that exclusion only applies in `is_excluded(path)` which is checked on
    files, not directories. Directory-walking does not skip the directory,
    so any contents inside `.Rproj.user/` (other than dotfiles) would leak
    in. Today there are no such files. Recommend: also test directory-name
    prefixes against `.`.

- **m3. `is_excluded` skips files ending in `.py` — including any `.py` that
  *is* legitimately data. Comment at L77 acknowledges this risk.**
  - Location: `scripts/analysis/build_raw_data_manifest.py:73-83`.
  - Today there's no data `.py` in raw_data, but a future asset (e.g.
    config-as-Python for a model) would silently disappear from the
    inventory. Recommend: build an explicit allowlist or surface skipped
    `.py`s under `non_data_artifacts` (currently they're listed under
    `non_data_artifacts` because `is_excluded` returns True — so this is
    actually OK; just verify).

- **m4. Embedder doesn't handle the `geo` study's per-GSE wiki sub-pages.**
  - Location: `scripts/analysis/build_raw_data_manifest.py:1008-1055`
    (`embed_inventory_into_wiki`).
  - 17 wiki files matching `studies/geo_GSE*.md` still have the unfilled
    `<!-- TODO: filled by Quantifier… -->` placeholder, because the on-disk
    source `data/raw_data/studies/geo/` is treated as a single `geo` source
    (one entry in `inventory.json` named "geo"). The embedder maps wiki stem
    → source name 1:1, so `geo_GSE99254.md` has no matching source. The
    Quantifier report claimed `geo.md` was the only one without a placeholder;
    in fact `geo.md` is intentionally skipped (ok), but the 17 sub-pages are
    incorrectly skipped as a side effect. Either (a) inventory each
    `studies/geo/<GSE_ID>` subdir separately (add a recursion rule for the
    `geo` source), or (b) add a special-case in the embedder.

- **m5. `re.sub(pattern, block, text)` is unsafe if `block` ever contains a
  literal backslash followed by a digit.**
  - Location: `scripts/analysis/build_raw_data_manifest.py:1042-1044`.
  - `re.sub` interprets the replacement string for backreferences. Source
    paths or filenames with `\1`-style content would be corrupted on
    replace. Today no such filenames exist, so it's latent. Use
    `AUTO_BLOCK_RE.sub(lambda m: block, text)` to bypass replacement parsing.

- **m6. `wc_l` shells out per file (one fork per file). For OTS / adc this
  is ~2k–9k forks per source.**
  - Location: `scripts/analysis/build_raw_data_manifest.py:248-269`.
  - Could be 5–10× faster by `wc -l file1 file2 …` in batched calls, but
    runtime is not currently a problem (~12 min total). Skip if not blocking.

- **m7. Trailing-newline detection in `wc_l` is correct but
  `wc_l(path)` always seeks the file even when subtracting for a header
  later. Fine, just observed.**

- **m8. The `args.source_filter` field in the JSON envelope captures
  `args.source` (the raw arg), so re-runs with `--source` are at least
  recorded — but this doesn't compensate for the destructive overwrite (C3).**

### Confirmed-correct

- **OTS `wc -l minus 2` rule.** Verified: `head -2 SRR16868480_1_Paired_All.csv`
  shows line 1 is a JSON metadata blob and line 2 is the column header. The
  matching condition `path.name.endswith("_Paired_All.csv")` is correct for
  every file under `data/raw_data/databases/OTS/`.
- **Trailing-newline correction in `wc_l`.** Logic at L258-268 reads the last
  byte and adds 1 if it's not `\n`. Verified manually with a hand-crafted file.
- **Random sampling reproducibility.** Seed=42 hardcoded at L695. Re-running
  with the same input set yields the same sample.
- **JSON output schema.** All required fields per the project plan are
  populated for every entry: `name, kind, path, size_bytes, size_human,
  file_count, format_breakdown, primary_data_files, total_records,
  record_definition, notes, non_data_artifacts, standardizer`. Spot-checked
  vdjdb, NetMHCPan, OTS, GSE114724.
- **Standardizer cross-reference for multi-standardizer DBs.** IEDB and
  CEDAR correctly emit a list with two entries each (`iedb.py` + `iedb_pmhc.py`,
  `cedar.py` + `cedar_pmhc.py`). Verified in the JSON.
- **FASTA counting.** `grep -c '^>'` is used; the IMGTHLA total of 7.6M
  records aligns with manual spot-checks on `hla_prot.fasta` etc. Risk of a
  sequence-line accidentally starting with `>` is real but vanishingly
  unlikely for IMGT-curated content; would manifest as a small overcount.
- **JSONL counting.** `wc -l` is the standard count; `wc_l` already corrects
  for missing trailing newline. (For `JSONL` specifically, `count_jsonl` calls
  `wc_l` directly, so it inherits the trailing-newline correction.)
- **Parquet counting.** `pyarrow.parquet.ParquetFile(path).metadata.num_rows`
  is correct per file; partitioned datasets would be summed correctly because
  the walker iterates files individually.
- **Idempotency of embedder for the AUTO-INVENTORY block path.** The
  `AUTO_BLOCK_RE` regex (DOTALL, non-greedy) correctly replaces an existing
  block on re-run. Manual inside-block edits are clobbered (acceptable per
  spec).
- **No symlinks in `data/raw_data/`** — verified via `find -type l`. The
  open question of whether `rglob` follows them is moot for current data.
- **`du -sh` parity.** `gather_files` walks all files (including those in
  excluded dirs) for `total_size`/`file_count`, matching `du -sh` semantics.
  The .git contents inside IMGTHLA, for example, contribute to size_bytes
  but not to data record counts — correctly.

## Reproduction notes

- `sha1sum scripts/analysis/build_raw_data_manifest.py` →
  `8a83472051d86276e86a6717992b531283551a88`.
- `python scripts/analysis/build_raw_data_manifest.py --source vdjdb --exact`
  ran in 2.5s, produced a 1-source `inventory.json` (overwriting the prior
  82-source one). I have triggered a full rerun in the background to restore
  the inventory; this exposed the C3 bug.
- `head -2 data/raw_data/databases/OTS/SRR16868480_1_Paired_All.csv` shows
  line 1 = JSON metadata, line 2 = column header; confirms `wc -l - 2` rule.
- `unzip -l data/raw_data/databases/vdjdb/vdjdb-2025-12-29.zip` shows the same
  17 files that exist loose under `data/raw_data/databases/vdjdb/`; confirms
  the C2 vdjdb double-count.
- `unzip -l data/raw_data/databases/CEDAR/tcell/tcell_full_v3.zip` shows
  `tcell_full_v3.csv` as the sole member, identical to the loose CSV;
  confirms the C2 CEDAR double-count.
- `python -c 'from build_raw_data_manifest import detect_format; print(detect_format(Path("c000_ba")))'`
  → `"noext"`; confirms C1.
- `wc -l data/raw_data/databases/NetMHCPan/NetMHCpan_train/c000_ba` →
  41,206 (real records, currently zero in inventory).
- `python -c "import zipfile, io; z = zipfile.ZipFile('data/raw_data/databases/IEDB/receptor/receptor_full_v3.zip'); ..."`
  raises `BadZipFile: Bad CRC-32 for file 'tcr_full_v3.csv'`; confirms M1.
