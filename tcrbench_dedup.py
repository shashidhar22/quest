#!/usr/bin/env python3
"""
TCRBench DuckDB Deduplication & Statistics Pipeline

Production-grade pipeline for deduplicating ~5 billion TCR-peptide-MHC records
using DuckDB on high-memory instances. Performs sanitization, deduplication,
combination counting, permutation counting, and powerset explosion.

Usage example:
    python tcrbench_dedup.py \
        --input "/path/to/raw/**/*.parquet" \
        --output_dir "/path/to/output/" \
        --tmp_dir "/path/to/nvme_tmp/" \
        --memory_limit "800GB" \
        --threads 64 \
        --mode both

Dependencies: duckdb>=1.1.0, pyarrow>=14.0
"""

import argparse
import glob
import json
import logging
import os
import resource
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import duckdb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Primary columns used for deduplication and explosion
DEDUP_COLUMNS = ["tra_full", "trb_full", "peptide", "mhc_one", "mhc_two"]
SHORT_NAMES = ["tra", "trb", "peptide", "mhc_one", "mhc_two"]
COLUMN_BITS = {col: 1 << i for i, col in enumerate(DEDUP_COLUMNS)}

# CDR columns carried alongside their parent chain (not used for dedup)
CDR_COLUMNS = {
    "tra_full": ["tra_cdr1", "tra_cdr2", "tra_cdr3"],
    "trb_full": ["trb_cdr1", "trb_cdr2", "trb_cdr3"],
}
ALL_CDR_COLUMNS = [c for cols in CDR_COLUMNS.values() for c in cols]

# All columns read from input (dedup keys + carried CDR columns)
ALL_COLUMNS = DEDUP_COLUMNS + ALL_CDR_COLUMNS

# Columns that must contain only valid amino acid characters
AA_ONLY_COLUMNS = {"tra_full", "trb_full", "peptide", "mhc_one", "mhc_two",
                   "tra_cdr1", "tra_cdr2", "tra_cdr3", "trb_cdr1", "trb_cdr2", "trb_cdr3"}

# TCR-side and antigen-side column groups for salvage logic
_TCR_DEDUP_COLS = {"tra_full", "trb_full"}
_ANTIGEN_DEDUP_COLS = {"peptide", "mhc_one", "mhc_two"}
_TCR_CDR_COLS = set(ALL_CDR_COLUMNS)
_ANTIGEN_CDR_COLS: set[str] = set()  # no CDR columns for antigen side

# Fallback columns: prefer full-length, fall back to CDR3
# Maps dedup column → raw source expression using COALESCE
COALESCE_EXPRS = {
    "tra_full": "COALESCE(NULLIF(TRIM(tra_full), ''), NULLIF(TRIM(tra), ''))",
    "trb_full": "COALESCE(NULLIF(TRIM(trb_full), ''), NULLIF(TRIM(trb), ''))",
}

# Length constraints per column: (min_len, max_len) or (min_len, None)
LENGTH_RULES = {
    "tra_full": (4, None),
    "trb_full": (4, None),
    "peptide": (8, 30),
    "mhc_one": (4, None),
    "mhc_two": (4, None),
}

# The 11 biologically meaningful combination definitions
COMBINATIONS = {
    "tra": ["tra_full"],
    "trb": ["trb_full"],
    "peptide_mhc_one": ["peptide", "mhc_one"],
    "peptide_mhc_one_mhc_two": ["peptide", "mhc_one", "mhc_two"],
    "tra_trb": ["tra_full", "trb_full"],
    "tra_peptide_mhc_one": ["tra_full", "peptide", "mhc_one"],
    "trb_peptide_mhc_one": ["trb_full", "peptide", "mhc_one"],
    "tra_trb_peptide_mhc_one": ["tra_full", "trb_full", "peptide", "mhc_one"],
    "tra_peptide_mhc_one_mhc_two": ["tra_full", "peptide", "mhc_one", "mhc_two"],
    "trb_peptide_mhc_one_mhc_two": ["trb_full", "peptide", "mhc_one", "mhc_two"],
    "tra_trb_peptide_mhc_one_mhc_two": ["tra_full", "trb_full", "peptide", "mhc_one", "mhc_two"],
}

# Short name → full column name mapping
SHORT_TO_COL = dict(zip(SHORT_NAMES, DEDUP_COLUMNS))


def build_all_subsets() -> dict[str, list[str]]:
    """Build all 31 non-empty subsets of columns (unordered), keyed by subset_key label."""
    result = {}
    for mask in range(1, 32):
        parts = []
        cols = []
        for i, (short, col) in enumerate(zip(SHORT_NAMES, DEDUP_COLUMNS)):
            if mask & (1 << i):
                parts.append(short)
                cols.append(col)
        result["_".join(parts)] = cols
    return result


def build_all_permutations() -> list[tuple[str, int, tuple[int, ...]]]:
    """Build all 325 ordered permutations of all non-empty subsets.

    Returns list of (subset_key, mask, col_indices) where:
    - subset_key: ordered short name label, e.g. "peptide_tra"
    - mask: bitmask of which columns are included
    - col_indices: tuple of column indices in order (into DEDUP_COLUMNS)
    """
    from itertools import combinations, permutations as iter_perms
    result = []
    n = len(DEDUP_COLUMNS)
    for size in range(1, n + 1):
        for subset in combinations(range(n), size):
            mask = sum(1 << i for i in subset)
            for perm in iter_perms(subset):
                key = "_".join(SHORT_NAMES[i] for i in perm)
                result.append((key, mask, perm))
    return result


# All 31 unordered subsets (for counting)
SUBSETS = build_all_subsets()

# All 325 ordered permutations (for explosion)
PERMUTATIONS = build_all_permutations()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

logger = logging.getLogger("tcrbench_dedup")


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-5s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.setLevel(level)
    logger.addHandler(handler)


def elapsed_str(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def build_subset_key_map() -> dict[int, str]:
    """Build mapping from bitmask (1..31) to human-readable subset_key string."""
    result = {}
    for mask in range(1, 32):
        parts = []
        for i, name in enumerate(SHORT_NAMES):
            if mask & (1 << i):
                parts.append(name)
        result[mask] = "_".join(parts)
    return result


def build_subset_key_case_expr(key_map: dict[int, str]) -> str:
    """Build a SQL CASE WHEN expression mapping subset_mask to subset_key."""
    clauses = [f"WHEN {mask} THEN '{label}'" for mask, label in sorted(key_map.items())]
    return "CASE s.mask " + " ".join(clauses) + " END"


def build_sanitize_expr(col: str, coalesce_overrides: dict[str, str] | None = None) -> str:
    """Build the CASE expression for sanitizing one column.

    For columns with COALESCE fallbacks (tra_full, trb_full), the source
    expression is COALESCE(full_length, cdr3) so that CDR3 is used when
    the full-length chain is missing.

    AA-only columns are validated with a regex check — values containing
    non-amino-acid characters are NULLed out.
    """
    coalesce_map = coalesce_overrides if coalesce_overrides is not None else COALESCE_EXPRS
    src = coalesce_map.get(col, f"NULLIF(TRIM({col}), '')")
    min_len, max_len = LENGTH_RULES[col]
    parts = [
        f"WHEN ({src}) IS NULL THEN NULL",
    ]
    if col in AA_ONLY_COLUMNS:
        parts.append(f"WHEN NOT regexp_matches(({src}), '^[ACDEFGHIKLMNPQRSTVWY]+$') THEN NULL")
    parts.append(f"WHEN LENGTH({src}) < {min_len} THEN NULL")
    if max_len is not None:
        parts.append(f"WHEN LENGTH({src}) > {max_len} THEN NULL")
    parts.append(f"ELSE ({src})")
    return "CASE " + " ".join(parts) + f" END AS {col}"


def print_ascii_table(headers: list[str], rows: list[list[str]], alignments: list[str] | None = None) -> None:
    """Print a simple ASCII table with box-drawing characters."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    if alignments is None:
        alignments = ["<"] * len(headers)

    def fmt_row(cells: list[str]) -> str:
        parts = []
        for cell, w, a in zip(cells, col_widths, alignments):
            if a == ">":
                parts.append(cell.rjust(w))
            else:
                parts.append(cell.ljust(w))
        return "│ " + " │ ".join(parts) + " │"

    top = "┌─" + "─┬─".join("─" * w for w in col_widths) + "─┐"
    mid = "├─" + "─┼─".join("─" * w for w in col_widths) + "─┤"
    bot = "└─" + "─┴─".join("─" * w for w in col_widths) + "─┘"

    print(top)
    print(fmt_row(headers))
    print(mid)
    for row in rows:
        print(fmt_row(row))
    print(bot)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def step0_preflight(args: argparse.Namespace) -> int:
    """Pre-flight validation. Returns number of input files found."""
    logger.info("STEP 0: Pre-flight validation")

    # 1. Disk space
    for label, path in [("output_dir", args.output_dir), ("tmp_dir", args.tmp_dir)]:
        try:
            usage = shutil.disk_usage(path)
            free_tb = usage.free / (1024**4)
            msg = f"  {label} ({path}): {free_tb:.2f} TB free"
            if free_tb < 2.0:
                logger.warning(msg + " — less than 2 TB free!")
            else:
                logger.info(msg)
        except OSError as e:
            logger.warning(f"  Could not check disk space for {label}: {e}")

    # 2. File descriptor limit
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < 65536:
        logger.warning(
            f"File descriptor limit is {soft}. For the explosion step, run:\n"
            "    ulimit -n 1048576\n"
            "before launching this script."
        )
    else:
        logger.info(f"  File descriptor limit: {soft} (OK)")

    # 3. DuckDB version
    ver = duckdb.__version__
    logger.info(f"  DuckDB version: {ver}")
    major_minor = tuple(int(x) for x in ver.split(".")[:2])
    if major_minor < (1, 1):
        logger.warning(f"DuckDB version {ver} is below 1.1.0 — some features may not work.")

    # 4. Input glob
    files = glob.glob(args.input, recursive=True)
    n_files = len(files)
    if n_files == 0:
        logger.error(f"No Parquet files found matching: {args.input}")
        sys.exit(1)
    logger.info(f"  Input files matched: {n_files}")

    # 5. Config summary
    logger.info("  Configuration:")
    logger.info(f"    input:        {args.input}")
    logger.info(f"    output_dir:   {args.output_dir}")
    logger.info(f"    tmp_dir:      {args.tmp_dir}")
    logger.info(f"    memory_limit: {args.memory_limit}")
    logger.info(f"    threads:      {args.threads}")
    logger.info(f"    mode:         {args.mode}")
    logger.info(f"    force:        {args.force_recompute}")

    return n_files


def configure_duckdb(con: duckdb.DuckDBPyConnection, args: argparse.Namespace) -> None:
    """Apply DuckDB settings."""
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET temp_directory='{args.tmp_dir}'")
    con.execute("SET preserve_insertion_order=false")
    # After H2, step 3 writes one partition at a time, so we don't need many
    # concurrent writers. Keeping this small reduces writer-side memory.
    con.execute("SET partitioned_write_max_open_files=8")
    # Give DuckDB an explicit large spill budget so it spills aggressively
    # rather than holding too much in RAM. Best-effort: ignored on older builds.
    try:
        usage = shutil.disk_usage(args.tmp_dir)
        # Reserve ~10% headroom on the spill volume.
        spill_bytes = int(usage.free * 0.9)
        con.execute(f"SET max_temp_directory_size='{spill_bytes}b'")
    except Exception as e:
        logger.debug(f"  Could not set max_temp_directory_size: {e}")
    # Warn if threads is high relative to CPU count — wide string hash
    # aggregates pay a per-thread overhead that often dominates RAM in step 3.
    try:
        cpu = os.cpu_count() or args.threads
        if args.threads > max(1, cpu // 2):
            logger.warning(
                f"  threads={args.threads} is > cpu_count/2 ({cpu // 2}); "
                "consider lowering for the explosion step to reduce per-thread "
                "hash-table memory."
            )
    except Exception:
        pass


def step1_sanitize_dedup(con: duckdb.DuckDBPyConnection, args: argparse.Namespace, n_files: int) -> dict:
    """Sanitize + deduplicate. Returns step metadata dict."""
    logger.info("STEP 1: Sanitize + Deduplicate — starting")
    logger.info(f"  Input: {args.input} ({n_files} files)")
    logger.info(f"  DuckDB: memory={args.memory_limit}, threads={args.threads}")

    t0 = time.time()

    # Check if table already exists
    tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    if "deduped" in tables:
        if args.force_recompute:
            logger.info("  Dropping existing deduped table (--force_recompute)")
            con.execute("DROP TABLE deduped")
        else:
            answer = input("Deduped table already exists. Reuse it? [y/N]: ").strip().lower()
            if answer == "y":
                row_count = con.execute("SELECT COUNT(*) FROM deduped").fetchone()[0]
                elapsed = time.time() - t0
                logger.info(f"  Reusing existing deduped table ({row_count:,} rows)")
                return {"status": "reused", "elapsed_seconds": round(elapsed, 1), "rows_out": row_count}
            else:
                logger.info("  Dropping existing deduped table")
                con.execute("DROP TABLE deduped")

    # Detect available columns in input to handle missing columns gracefully
    input_path = str(args.input_resolved)
    probe_con = duckdb.connect()
    available_cols = set(
        row[0] for row in probe_con.execute(
            f"SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{input_path}', hive_partitioning=true) LIMIT 0)"
        ).fetchall()
    )
    probe_con.close()
    logger.info(f"  Input columns detected: {sorted(available_cols)}")

    # Adjust COALESCE fallbacks: only reference tra/trb if they exist in input
    active_coalesce = {}
    for col, expr in COALESCE_EXPRS.items():
        fallback_col = "tra" if col == "tra_full" else "trb"
        if fallback_col in available_cols:
            active_coalesce[col] = expr
        else:
            active_coalesce[col] = f"NULLIF(TRIM({col}), '')"

    # Build sanitization SQL
    # Dedup columns get full sanitization (trim + length filter + AA validation)
    dedup_exprs = [build_sanitize_expr(col, active_coalesce) for col in DEDUP_COLUMNS]
    # CDR columns get trim + AA validation
    cdr_exprs = []
    for col in ALL_CDR_COLUMNS:
        if col in available_cols:
            cdr_exprs.append(
                f"CASE WHEN regexp_matches(NULLIF(TRIM({col}), ''), '^[ACDEFGHIKLMNPQRSTVWY]+$') "
                f"THEN NULLIF(TRIM({col}), '') ELSE NULL END AS {col}"
            )
        else:
            cdr_exprs.append(f"NULL AS {col}")

    # Carry binding/source/score for salvage logic (handle missing columns)
    meta_exprs = []
    for raw_col, alias in [("binding", "_binding"), ("source", "_source"), ("score", "_score")]:
        if raw_col in available_cols:
            if alias == "_score":
                meta_exprs.append(f"TRIM(COALESCE(CAST({raw_col} AS VARCHAR), '')) AS {alias}")
            else:
                meta_exprs.append(f"LOWER(TRIM(COALESCE(CAST({raw_col} AS VARCHAR), ''))) AS {alias}")
        else:
            meta_exprs.append(f"'' AS {alias}")

    inner_select = ",\n    ".join(dedup_exprs + cdr_exprs + meta_exprs)

    # Salvage logic: negative binders and VDJdb score-0 rows get split into
    # TCR-side and antigen-side sub-rows so each half can still be used for
    # MLM training even though the interaction is unreliable.
    #
    # Build column lists in canonical DEDUP_COLUMNS order for UNION ALL compatibility.
    tcr_side_cols = []  # TCR-side: keep TCR, null antigen
    antigen_side_cols = []  # antigen-side: null TCR, keep antigen
    for col in DEDUP_COLUMNS:
        if col in _TCR_DEDUP_COLS:
            tcr_side_cols.append(col)
            antigen_side_cols.append(f"NULL AS {col}")
        else:
            tcr_side_cols.append(f"NULL AS {col}")
            antigen_side_cols.append(col)

    tcr_side_cdr = ", ".join(ALL_CDR_COLUMNS)
    antigen_side_cdr = ", ".join(f"NULL AS {c}" for c in ALL_CDR_COLUMNS)
    cdr_cols = ", ".join(ALL_CDR_COLUMNS)

    salvage_condition = "(_binding = 'neg' OR (_source = 'vdjdb' AND _score = '0'))"

    # H1: materialize sanitization once into a temp table so the parquet scan
    # and per-row regex/length checks run a single time, instead of being
    # inlined three times across the salvage UNION ALL.
    sanitize_sql = f"""CREATE TEMP TABLE sanitized AS
SELECT {inner_select}
FROM read_parquet('{input_path}', hive_partitioning=true)"""
    logger.debug(f"  SQL:\n{sanitize_sql}")
    con.execute(sanitize_sql)

    sql = f"""CREATE TABLE deduped AS
SELECT
    {", ".join(DEDUP_COLUMNS)},
    {", ".join(f"FIRST({col}) AS {col}" for col in ALL_CDR_COLUMNS)}
FROM (
    -- Normal rows: keep all columns
    SELECT {", ".join(DEDUP_COLUMNS)}, {cdr_cols}
    FROM sanitized
    WHERE NOT {salvage_condition}

    UNION ALL

    -- Salvage: TCR-side of neg-binding / vdjdb-score-0 (null out antigen columns)
    SELECT {", ".join(tcr_side_cols)}, {tcr_side_cdr}
    FROM sanitized
    WHERE {salvage_condition}

    UNION ALL

    -- Salvage: antigen-side of neg-binding / vdjdb-score-0 (null out TCR + CDR columns)
    SELECT {", ".join(antigen_side_cols)}, {antigen_side_cdr}
    FROM sanitized
    WHERE {salvage_condition}
) combined
GROUP BY {", ".join(DEDUP_COLUMNS)}"""

    logger.debug(f"  SQL:\n{sql}")
    con.execute(sql)
    con.execute("DROP TABLE sanitized")

    elapsed = time.time() - t0

    # Post-dedup diagnostics
    row_count = con.execute("SELECT COUNT(*) FROM deduped").fetchone()[0]
    logger.info(f"STEP 1: complete in {elapsed_str(elapsed)}")
    logger.info(f"  Rows after dedup: {row_count:,}")

    # H5: collapse 11 separate COUNT() scans into a single pass over `deduped`.
    stat_cols = DEDUP_COLUMNS + ALL_CDR_COLUMNS
    stats_sql = "SELECT " + ", ".join(f"COUNT({c})" for c in stat_cols) + " FROM deduped"
    stats_row = con.execute(stats_sql).fetchone()
    stats = dict(zip(stat_cols, stats_row))

    parts = []
    for col in DEDUP_COLUMNS:
        nn = stats[col]
        frac = 1.0 - (nn / row_count) if row_count > 0 else 0
        short = col.replace("_full", "") if col.endswith("_full") else col
        parts.append(f"{short}={nn:,}")
        logger.info(f"  {col}: {nn:,} non-NULL ({frac:.1%} NULL)")
    logger.info(f"  Non-null: {' '.join(parts)}")

    cdr_parts = [f"{col}={stats[col]:,}" for col in ALL_CDR_COLUMNS]
    logger.info(f"  CDR non-null: {' '.join(cdr_parts)}")

    # Sanity check: no empty strings (cast to VARCHAR to handle NULL-typed columns)
    conditions = " OR ".join(f"CAST({col} AS VARCHAR) = ''" for col in ALL_COLUMNS)
    empty_count = con.execute(f"SELECT COUNT(*) FROM deduped WHERE {conditions}").fetchone()[0]
    if empty_count == 0:
        logger.info("  Empty string check: PASS")
    else:
        logger.warning(f"  Empty string check: FAIL — {empty_count:,} rows with empty strings remain")

    # Export to Parquet
    parquet_dir = str(args.output_dir_resolved / "deduped_parquet")
    os.makedirs(parquet_dir, exist_ok=True)
    logger.info(f"  Exporting deduped table to {parquet_dir}/")
    export_sql = (
        f"COPY (SELECT * FROM deduped) TO '{parquet_dir}/' "
        "(FORMAT PARQUET, PER_THREAD_OUTPUT true, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)"
    )
    logger.debug(f"  SQL:\n{export_sql}")
    con.execute(export_sql)
    logger.info("  Parquet export complete")

    return {"status": "complete", "elapsed_seconds": round(elapsed, 1), "rows_out": row_count}


def _count_unique(con: duckdb.DuckDBPyConnection, cols: list[str]) -> int:
    """Count distinct non-null tuples of the given columns in `deduped`.

    H3: each label is its own query so DuckDB only holds one DISTINCT hash
    set live at a time. The previous UNION ALL of 31 DISTINCT branches kept
    multiple large hash sets resident simultaneously and was a frequent
    secondary OOM source.
    """
    where = " AND ".join(f"{c} IS NOT NULL" for c in cols)
    distinct_cols = ", ".join(cols)
    sql = f"SELECT COUNT(*) FROM (SELECT DISTINCT {distinct_cols} FROM deduped WHERE {where})"
    return con.execute(sql).fetchone()[0]


def step2_counts(con: duckdb.DuckDBPyConnection, args: argparse.Namespace) -> dict:
    """Compute combination and permutation counts. Returns step metadata dict."""
    logger.info("STEP 2: Compute counts — starting")
    t0 = time.time()

    # --- 11 biologically meaningful combinations ---
    combo_counts: dict[str, int] = {}
    for label, cols in COMBINATIONS.items():
        combo_counts[label] = _count_unique(con, cols)
        logger.debug(f"  combo {label} = {combo_counts[label]:,}")

    # --- All 31 powerset subsets (unordered counts) ---
    perm_counts: dict[str, int] = {}
    for label, cols in SUBSETS.items():
        perm_counts[label] = _count_unique(con, cols)
        logger.debug(f"  subset {label} = {perm_counts[label]:,}")

    elapsed = time.time() - t0
    logger.info(f"STEP 2: complete in {elapsed_str(elapsed)}")

    # Write combination counts (11)
    json_path = args.output_dir_resolved / "combination_counts.json"
    with open(json_path, "w") as f:
        json.dump(combo_counts, f, indent=2)

    csv_path = args.output_dir_resolved / "combination_counts.csv"
    with open(csv_path, "w") as f:
        f.write("combo,unique_count\n")
        for label in COMBINATIONS:
            f.write(f"{label},{combo_counts.get(label, 0)}\n")
    logger.info(f"  Written: {json_path}, {csv_path}")

    # Write subset counts (all 31)
    perm_json_path = args.output_dir_resolved / "permutation_counts.json"
    with open(perm_json_path, "w") as f:
        json.dump(perm_counts, f, indent=2)

    perm_csv_path = args.output_dir_resolved / "permutation_counts.csv"
    with open(perm_csv_path, "w") as f:
        f.write("permutation,unique_count\n")
        for label in SUBSETS:
            f.write(f"{label},{perm_counts.get(label, 0)}\n")
    logger.info(f"  Written: {perm_json_path}, {perm_csv_path}")

    # Print combination table (11 biologically meaningful)
    print()
    combo_rows = [[label, f"{combo_counts.get(label, 0):,}"] for label in COMBINATIONS]
    print_ascii_table(["Combination (11 biological)", "Unique Count"], combo_rows, ["<", ">"])

    # Print subset table (all 31 subsets, non-zero only)
    print()
    perm_rows = [
        [label, f"{perm_counts.get(label, 0):,}"]
        for label in SUBSETS
        if perm_counts.get(label, 0) > 0
    ]
    if perm_rows:
        print_ascii_table(["Subset (all 31)", "Unique Count"], perm_rows, ["<", ">"])
    else:
        logger.info("  No non-zero permutation counts to display")
    print()

    return {"status": "complete", "elapsed_seconds": round(elapsed, 1)}


def step3_explode(con: duckdb.DuckDBPyConnection, args: argparse.Namespace) -> dict:
    """Powerset explosion with ordered permutations + dedup.

    H2: instead of one global ``SELECT DISTINCT`` over a CROSS JOIN of
    ``deduped`` × 325 permutations (which builds a single huge hash table on
    six wide string columns and is the dominant OOM source), we drive the
    explosion from Python over the 31 unique unordered subsets:

    1. For each mask, build a small temp table that contains the DISTINCT
       projection on just the in-mask columns. Cardinality is at most that
       of ``deduped`` and is typically much smaller for shorter masks.
    2. For each ordered permutation that shares this mask (~10 on average),
       emit a single ``COPY`` into ``subset_key=<key>/`` with ``CONCAT_WS``
       in the perm-specific order. No DISTINCT needed here — the ordering
       is a 1:1 function of the already-deduped tuples.
    3. Drop the temp table before moving to the next mask.

    Same output layout as before (``exploded_deduped/subset_key=<key>/*.parquet``).
    """
    from collections import defaultdict

    logger.info("STEP 3: Ordered permutation explosion + dedup — starting")
    logger.info(f"  Generating {len(PERMUTATIONS)} ordered permutations across {len(SUBSETS)} masks")
    t0 = time.time()

    exploded_dir = str(args.output_dir_resolved / "exploded_deduped")
    os.makedirs(exploded_dir, exist_ok=True)

    # Group ordered permutations by their (unordered) mask.
    perms_by_mask: dict[int, list[tuple[str, tuple[int, ...]]]] = defaultdict(list)
    for key, mask, col_indices in PERMUTATIONS:
        perms_by_mask[mask].append((key, col_indices))

    total_rows = 0
    rows_for_table: list[list[str]] = []

    try:
        for mask in sorted(perms_by_mask):
            perms = perms_by_mask[mask]
            cols_in_mask = [DEDUP_COLUMNS[i] for i in range(len(DEDUP_COLUMNS)) if mask & (1 << i)]
            cdr_cols_in_mask: list[str] = []
            for parent in cols_in_mask:
                cdr_cols_in_mask.extend(CDR_COLUMNS.get(parent, []))

            where = " AND ".join(f"{c} IS NOT NULL" for c in cols_in_mask)
            # Dedup on the in-mask columns; carry CDRs via FIRST() since CDRs
            # are functionally determined by their parent chain (already
            # collapsed in step 1's GROUP BY).
            select_parts = list(cols_in_mask) + [f"FIRST({c}) AS {c}" for c in cdr_cols_in_mask]
            temp_sql = (
                "CREATE TEMP TABLE m_subset AS "
                f"SELECT {', '.join(select_parts)} "
                f"FROM deduped WHERE {where} "
                f"GROUP BY {', '.join(cols_in_mask)}"
            )
            logger.debug(f"  SQL:\n{temp_sql}")
            con.execute(temp_sql)

            cnt = con.execute("SELECT COUNT(*) FROM m_subset").fetchone()[0]
            logger.info(
                f"  mask={mask:>2} cols=[{','.join(cols_in_mask)}] "
                f"dedup_rows={cnt:,} perms={len(perms)}"
            )

            # Canonical (unordered) subset name = column short-names in
            # DEDUP_COLUMNS order. All ordered permutations of this mask live
            # under the same outer subset_key directory; the specific ordering
            # is the inner order_key partition.
            subset_key = "_".join(
                SHORT_NAMES[i] for i in range(len(DEDUP_COLUMNS)) if mask & (1 << i)
            )

            for key, col_indices in perms:
                partition_dir = f"{exploded_dir}/subset_key={subset_key}/order_key={key}"
                os.makedirs(partition_dir, exist_ok=True)

                copy_select_parts: list[str] = []
                # NULLs must be typed VARCHAR so all partitions produce a
                # uniform parquet schema (otherwise cross-partition reads fail
                # because untyped NULL is inferred as INTEGER).
                for col in DEDUP_COLUMNS:
                    if col in cols_in_mask:
                        copy_select_parts.append(col)
                    else:
                        copy_select_parts.append(f"CAST(NULL AS VARCHAR) AS {col}")
                for cdr in ALL_CDR_COLUMNS:
                    if cdr in cdr_cols_in_mask:
                        copy_select_parts.append(cdr)
                    else:
                        copy_select_parts.append(f"CAST(NULL AS VARCHAR) AS {cdr}")
                ordered_cols = [DEDUP_COLUMNS[i] for i in col_indices]
                copy_select_parts.append(
                    f"CONCAT_WS(' ', {', '.join(ordered_cols)}) AS sequence"
                )

                copy_sql = (
                    "COPY (SELECT " + ", ".join(copy_select_parts) + " FROM m_subset) "
                    f"TO '{partition_dir}/' "
                    "(FORMAT PARQUET, PER_THREAD_OUTPUT true, "
                    "COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)"
                )
                logger.debug(f"  SQL:\n{copy_sql}")
                con.execute(copy_sql)

                total_rows += cnt
                rows_for_table.append([key, f"{cnt:,}"])

            con.execute("DROP TABLE m_subset")
    except KeyboardInterrupt:
        logger.warning(f"Interrupted — partial output may exist at {exploded_dir}/")
        raise

    elapsed = time.time() - t0
    logger.info(f"STEP 3: explosion complete in {elapsed_str(elapsed)}")

    # Post-explosion validation: walk subset_key=*/order_key=* dirs
    expected_order_keys = set(key for key, _, _ in PERMUTATIONS)
    found_order_keys = set()
    for subset_dir in Path(exploded_dir).iterdir():
        if not (subset_dir.is_dir() and subset_dir.name.startswith("subset_key=")):
            continue
        for order_dir in subset_dir.iterdir():
            if order_dir.is_dir() and order_dir.name.startswith("order_key="):
                found_order_keys.add(order_dir.name.split("=", 1)[1])

    missing = expected_order_keys - found_order_keys
    if missing:
        logger.warning(f"  Missing {len(missing)} expected partitions (may be empty subsets): {sorted(missing)[:10]}...")
    else:
        logger.info(f"  All {len(expected_order_keys)} order_key partitions present")

    # Per-partition row counts (grouped by ordered permutation)
    count_sql = (
        f"SELECT subset_key, order_key, COUNT(*) AS row_count "
        f"FROM read_parquet('{exploded_dir}/**/*.parquet', hive_partitioning=true) "
        f"GROUP BY subset_key, order_key ORDER BY row_count DESC"
    )
    logger.debug(f"  SQL:\n{count_sql}")
    partition_counts = con.execute(count_sql).fetchall()

    total_rows = 0
    rows_for_table = []
    for subset_key_val, order_key_val, cnt in partition_counts:
        total_rows += cnt
        rows_for_table.append([subset_key_val, order_key_val, f"{cnt:,}"])

    print()
    print_ascii_table(["subset_key", "order_key", "row_count"], rows_for_table, ["<", "<", ">"])
    print()
    logger.info(f"  Total exploded rows: {total_rows:,}")

    return {"status": "complete", "elapsed_seconds": round(elapsed, 1), "rows_out": total_rows}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TCRBench DuckDB Deduplication & Statistics Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python tcrbench_dedup.py \\\n"
            '    --input "/path/to/raw/**/*.parquet" \\\n'
            '    --output_dir "/path/to/output/" \\\n'
            '    --tmp_dir "/path/to/nvme_tmp/" \\\n'
            '    --memory_limit "800GB" \\\n'
            "    --threads 64 \\\n"
            "    --mode both\n"
        ),
    )
    parser.add_argument("--input", required=True, help="Glob pattern for input Parquet files (e.g. '/data/**/*.parquet')")
    parser.add_argument("--output_dir", required=True, help="Directory for all output files")
    parser.add_argument("--tmp_dir", required=True, help="Directory for DuckDB temp spill files (use fast NVMe)")
    parser.add_argument("--memory_limit", required=True, help="DuckDB memory limit (e.g. '800GB')")
    parser.add_argument("--threads", type=int, required=True, help="Number of DuckDB threads")
    parser.add_argument(
        "--mode",
        choices=["counts_only", "explode", "both"],
        default="both",
        help="Pipeline mode: counts_only, explode, or both (default: both)",
    )
    parser.add_argument("--force_recompute", action="store_true", help="Drop existing deduped table and recompute")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG-level logging (prints full SQL)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    # Resolve paths
    args.input_resolved = Path(args.input)  # kept as-is for glob; used as string in SQL
    args.output_dir_resolved = Path(args.output_dir).resolve()
    args.tmp_dir_resolved = Path(args.tmp_dir).resolve()

    os.makedirs(args.output_dir_resolved, exist_ok=True)
    os.makedirs(args.tmp_dir_resolved, exist_ok=True)

    run_log = {
        "run_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "args": {
            "input": args.input,
            "output_dir": str(args.output_dir_resolved),
            "tmp_dir": str(args.tmp_dir_resolved),
            "mode": args.mode,
            "memory_limit": args.memory_limit,
            "threads": args.threads,
            "force_recompute": args.force_recompute,
        },
        "steps": {},
    }

    # Step 0
    n_files = step0_preflight(args)

    # Connect to persistent DuckDB
    db_path = str(args.output_dir_resolved / "deduped.db")
    logger.info(f"Connecting to DuckDB: {db_path}")
    con = duckdb.connect(database=db_path)
    configure_duckdb(con, args)

    try:
        # Step 1: Sanitize + Dedup
        try:
            step1_meta = step1_sanitize_dedup(con, args, n_files)
            run_log["steps"]["dedup"] = step1_meta
        except KeyboardInterrupt:
            raise
        except Exception:
            logger.exception("STEP 1 failed")
            run_log["steps"]["dedup"] = {"status": "failed"}
            raise

        # Step 2: Counts (runs for counts_only and both)
        if args.mode in ("counts_only", "both"):
            try:
                step2_meta = step2_counts(con, args)
                run_log["steps"]["counts"] = step2_meta
            except KeyboardInterrupt:
                raise
            except Exception:
                logger.exception("STEP 2 failed")
                run_log["steps"]["counts"] = {"status": "failed"}
                raise

        # Step 3: Explosion (runs for explode and both)
        if args.mode in ("explode", "both"):
            try:
                step3_meta = step3_explode(con, args)
                run_log["steps"]["explosion"] = step3_meta
            except KeyboardInterrupt:
                logger.warning(f"Interrupted — partial output may exist at {args.output_dir_resolved / 'exploded_deduped'}/")
                run_log["steps"]["explosion"] = {"status": "interrupted"}
                raise
            except Exception:
                logger.exception("STEP 3 failed")
                run_log["steps"]["explosion"] = {"status": "failed"}
                raise

    finally:
        # Always write the run log
        log_path = args.output_dir_resolved / "pipeline_run_log.json"
        with open(log_path, "w") as f:
            json.dump(run_log, f, indent=2)
        logger.info(f"Run log written to {log_path}")

        con.close()

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
