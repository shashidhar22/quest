"""Build foundation MLM pretraining datasets for benchmark_v2 (Phase 3).

See ``docs/BENCHMARK_FOUNDATION.md`` for the spec, partition rules, sampling
weights, and downstream consumption notes.

Usage:
    python build_benchmark_foundation.py --task {exclusion,partition,val_test,scaled,minis_weight,minis_order,summary,all}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

SOURCE_ROOT = Path("/home/ubuntu/quest/data/deduplicated_again/exploded_deduped_enriched")
SPLITS_ROOT = Path("/home/ubuntu/quest/data/benchmark_v2/splits")
LOOKUP_PARQUET = SPLITS_ROOT / "lookup" / "sequence_to_source_allele.parquet"
OUT_ROOT = Path("/home/ubuntu/quest/data/benchmark_v2/foundation")
LOG_DIR = OUT_ROOT / "logs"
SCRATCH = Path("/scratch")
TIMINGS_PATH = OUT_ROOT / "timings.json"
SUMMARY_PATH = OUT_ROOT / "foundation_summary.json"
EXCLUSION_PARQUET = OUT_ROOT / "downstream_test_exclusion.parquet"
EXCLUSION_TXT = OUT_ROOT / "downstream_test_exclusion.txt"
PARTITION_PARQUET = OUT_ROOT / "partition_assignments.parquet"
VAL_PARQUET = OUT_ROOT / "foundation_val.parquet"
TEST_PARQUET = OUT_ROOT / "foundation_test.parquet"

THREADS = 64
SEED = 42

# 27 in-scope subset_keys for foundation training
IN_SCOPE_SUBSETS: List[str] = [
    # 1-molecule
    "tra", "trb",
    # 2-molecule
    "tra_trb", "tra_peptide", "trb_peptide",
    "tra_mhc_one", "trb_mhc_one", "tra_mhc_two", "trb_mhc_two",
    "peptide_mhc_one", "peptide_mhc_two",
    # 3-molecule
    "tra_trb_peptide",
    "tra_trb_mhc_one", "tra_trb_mhc_two",
    "tra_peptide_mhc_one", "trb_peptide_mhc_one",
    "tra_peptide_mhc_two", "trb_peptide_mhc_two",
    "peptide_mhc_one_mhc_two",
    "tra_mhc_one_mhc_two", "trb_mhc_one_mhc_two",
    # 4-molecule
    "tra_trb_peptide_mhc_one", "tra_trb_peptide_mhc_two",
    "tra_trb_mhc_one_mhc_two",
    "tra_peptide_mhc_one_mhc_two", "trb_peptide_mhc_one_mhc_two",
    # 5-molecule
    "tra_trb_peptide_mhc_one_mhc_two",
]

# Excluded (4): "peptide", "mhc_one", "mhc_two", "mhc_one_mhc_two"


def _k_of_subset(subset_key: str) -> int:
    """Number of distinct molecule components in a subset_key."""
    # Treat 'mhc_one' and 'mhc_two' as single components (special since they
    # contain the underscore-delimited prefix).
    parts = []
    s = subset_key
    for token in ["mhc_one_mhc_two", "mhc_one", "mhc_two", "tra", "trb", "peptide"]:
        if token in s:
            # Avoid double-counting: 'mhc_one' inside 'mhc_one_mhc_two'
            if token == "mhc_one" and "mhc_one_mhc_two" in subset_key:
                continue
            if token == "mhc_two" and "mhc_one_mhc_two" in subset_key:
                continue
            if token == "mhc_one_mhc_two":
                parts.extend(["mhc_one", "mhc_two"])
            else:
                parts.append(token)
            s = s.replace(token, "_")
    return len(parts)


CANONICAL_ORDER = ["tra", "trb", "peptide", "mhc_one", "mhc_two"]

# Enriched-stage data columns (matches Phase 2 list)
ENRICHED_DATA_COLS: List[str] = [
    "tra_full", "trb_full", "peptide", "mhc_one", "mhc_two",
    "tra_cdr1", "tra_cdr2", "tra_cdr3",
    "trb_cdr1", "trb_cdr2", "trb_cdr3",
    "sequence",
    "subset_key", "order_key",
    "mhc_one_pocket", "mhc_one_contact", "mhc_one_pocket_contact",
    "mhc_two_pocket", "mhc_two_contact", "mhc_two_pocket_contact",
]


def _canonical_order_key(subset_key: str) -> str:
    """Return the canonical 'tra → trb → peptide → mhc_one → mhc_two' ordering for a subset_key.

    Detects molecule presence via substring matching (mhc_one/mhc_two tokens are
    multi-word and need careful handling).
    """
    components = []
    if "tra" in subset_key.split("_") or subset_key.startswith("tra_") or subset_key == "tra":
        components.append("tra")
    if "trb" in subset_key.split("_") or "_trb" in subset_key or subset_key.startswith("trb"):
        components.append("trb")
    if "peptide" in subset_key:
        components.append("peptide")
    if "mhc_one" in subset_key:
        components.append("mhc_one")
    if "mhc_two" in subset_key:
        components.append("mhc_two")
    # canonical order is preserved by appending in the right sequence above
    return "_".join(components)


def _record_timing(task: str, seconds: float) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cur: Dict[str, float] = {}
    if TIMINGS_PATH.exists():
        try:
            cur = json.loads(TIMINGS_PATH.read_text())
        except json.JSONDecodeError:
            cur = {}
    cur[task] = round(seconds, 2)
    TIMINGS_PATH.write_text(json.dumps(cur, indent=2, sort_keys=True))


def _duckdb_conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET threads={THREADS}")
    con.execute("SET memory_limit='400GB'")
    con.execute(f"SET temp_directory='{SCRATCH}'")
    return con


# ---------------------------------------------------------------------------
# bio_hash SQL expression (used in multiple steps)
# ---------------------------------------------------------------------------

BIO_HASH_SQL = """
md5(
    coalesce(tra_cdr3, '') || '|' ||
    coalesce(trb_cdr3, '') || '|' ||
    coalesce(peptide, '') || '|' ||
    coalesce(mhc_one_allele, '') || '|' ||
    coalesce(mhc_two_allele, '')
)
""".strip()


# ---------------------------------------------------------------------------
# Step 1 — Exclusion list
# ---------------------------------------------------------------------------

def task_exclusion() -> None:
    print("[exclusion] scanning splits/*test*.parquet and *eval*.parquet ...", flush=True)
    t0 = time.time()
    con = _duckdb_conn()

    sql = f"""
    COPY (
        WITH all_test AS (
            SELECT
                tra_cdr3, trb_cdr3, peptide,
                mhc_one_allele, mhc_two_allele,
                regexp_extract(filename, '([^/]+\\.parquet)$', 1) AS source_partition
            FROM read_parquet('{SPLITS_ROOT}/*test*.parquet', filename=true)
            UNION ALL
            SELECT
                tra_cdr3, trb_cdr3, peptide,
                mhc_one_allele, mhc_two_allele,
                regexp_extract(filename, '([^/]+\\.parquet)$', 1) AS source_partition
            FROM read_parquet('{SPLITS_ROOT}/*eval*.parquet', filename=true)
        )
        SELECT
            {BIO_HASH_SQL} AS bio_hash,
            list_distinct(list(source_partition)) AS source_partitions
        FROM all_test
        GROUP BY bio_hash
    ) TO '{EXCLUSION_PARQUET}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(sql)

    # plain-text companion (one hash per line)
    con.execute(f"""
        COPY (
            SELECT bio_hash FROM read_parquet('{EXCLUSION_PARQUET}') ORDER BY bio_hash
        ) TO '{EXCLUSION_TXT}' (FORMAT CSV, HEADER false, QUOTE '', DELIMITER ',');
    """)

    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{EXCLUSION_PARQUET}')").fetchone()[0]
    breakdown = con.execute(f"""
        WITH unnested AS (
            SELECT unnest(source_partitions) AS p
            FROM read_parquet('{EXCLUSION_PARQUET}')
        )
        SELECT p, COUNT(*) AS n FROM unnested GROUP BY p ORDER BY n DESC LIMIT 20
    """).fetchall()
    con.close()

    elapsed = time.time() - t0
    _record_timing("exclusion", elapsed)
    print(f"[exclusion] {n:,} unique excluded biological examples ({elapsed:.1f}s)", flush=True)
    print("[exclusion] top contributing benchmark partitions:", flush=True)
    for partition, count in breakdown:
        print(f"  {partition}: {count:,}", flush=True)


# ---------------------------------------------------------------------------
# Step 3 — Partition assignments
# ---------------------------------------------------------------------------

def task_partition() -> None:
    print(f"[partition] scanning {len(IN_SCOPE_SUBSETS)} in-scope subset_keys ...", flush=True)
    t0 = time.time()
    con = _duckdb_conn()

    # Build a UNION ALL over the 27 subset paths
    union_parts = []
    for sk in IN_SCOPE_SUBSETS:
        path = SOURCE_ROOT / f"subset_key={sk}"
        if not path.exists():
            continue
        union_parts.append(f"""
            SELECT '{sk}' AS subset_key,
                   regexp_extract(filename, 'order_key=([^/]+)', 1) AS order_key,
                   filename AS source_file,
                   ROW_NUMBER() OVER (PARTITION BY filename ORDER BY (SELECT 0)) - 1 AS source_row_index,
                   tra_full, trb_full, peptide, mhc_one, mhc_two,
                   tra_cdr3, trb_cdr3
            FROM read_parquet('{path}/**/*.parquet', filename=true)
        """)
    union_sql = "\nUNION ALL\n".join(union_parts)

    sql = f"""
    COPY (
        WITH src AS ({union_sql}),
        joined AS (
            SELECT
                s.subset_key, s.order_key, s.source_file, s.source_row_index,
                s.tra_cdr3, s.trb_cdr3, s.peptide,
                COALESCE(l.mhc_one_allele, s.mhc_one) AS mhc_one_allele,
                COALESCE(l.mhc_two_allele, s.mhc_two) AS mhc_two_allele
            FROM src s
            LEFT JOIN read_parquet('{LOOKUP_PARQUET}') l
              ON COALESCE(s.tra_full, '') = COALESCE(l.tra_full, '')
             AND COALESCE(s.trb_full, '') = COALESCE(l.trb_full, '')
             AND COALESCE(s.peptide, '')  = COALESCE(l.peptide, '')
             AND COALESCE(s.mhc_one, '')  = COALESCE(l.mhc_one, '')
             AND COALESCE(s.mhc_two, '')  = COALESCE(l.mhc_two, '')
        ),
        hashed AS (
            SELECT
                {BIO_HASH_SQL} AS bio_hash,
                subset_key, order_key, source_file, source_row_index,
                mhc_one_allele, mhc_two_allele
            FROM joined
        ),
        excluded AS (
            SELECT bio_hash FROM read_parquet('{EXCLUSION_PARQUET}')
        )
        SELECT
            h.bio_hash,
            h.subset_key, h.order_key,
            h.source_file, h.source_row_index,
            h.mhc_one_allele, h.mhc_two_allele,
            CASE
                WHEN hash(h.bio_hash) % 20 = 0 THEN 'test'
                WHEN hash(h.bio_hash) % 20 = 1 THEN 'val'
                ELSE 'train'
            END AS partition
        FROM hashed h
        ANTI JOIN excluded e ON h.bio_hash = e.bio_hash
    ) TO '{PARTITION_PARQUET}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(sql)

    # Stats
    print("[partition] computing stats ...", flush=True)
    rows = con.execute(f"""
        SELECT partition, COUNT(*) AS n
        FROM read_parquet('{PARTITION_PARQUET}')
        GROUP BY partition ORDER BY partition
    """).fetchall()
    by_subset = con.execute(f"""
        SELECT subset_key, partition, COUNT(*) AS n
        FROM read_parquet('{PARTITION_PARQUET}')
        GROUP BY subset_key, partition
        ORDER BY subset_key, partition
    """).fetchall()
    con.close()

    elapsed = time.time() - t0
    _record_timing("partition", elapsed)
    print(f"[partition] done in {elapsed:.1f}s", flush=True)
    for partition, n in rows:
        print(f"  {partition}: {n:,}", flush=True)
    print("[partition] per subset_key x partition:", flush=True)
    for sk, p, n in by_subset:
        print(f"  {sk:<40s} {p:<6s} {n:,}", flush=True)


# ---------------------------------------------------------------------------
# Step 4 — Foundation val + test sets
# ---------------------------------------------------------------------------

def _val_test_per_file(args: Tuple[str, str, List[Tuple[int, str]]]) -> Optional[pa.Table]:
    """Process one source file: read it, attach bio_hash/partition/alleles, return rows for the requested partition.

    args = (source_file, partition_name, list_of_(source_row_index, bio_hash, mhc_one_allele, mhc_two_allele, subset_key, order_key))
    """
    source_file, partition_name, rows_info = args
    if not rows_info:
        return None
    indices = [r[0] for r in rows_info]
    # Build a small DuckDB query that reads the source file and selects rows by row_index
    con = duckdb.connect()
    con.execute("SET threads=4")
    indices_str = "(" + ",".join(str(i) for i in indices) + ")"
    rdr = con.execute(f"""
        SELECT *, (ROW_NUMBER() OVER () - 1) AS _ridx
        FROM read_parquet('{source_file}')
        QUALIFY _ridx IN {indices_str}
    """)
    df = rdr.fetch_arrow_table() if hasattr(rdr, "fetch_arrow_table") else rdr.arrow()
    if isinstance(df, pa.RecordBatchReader):
        df = df.read_all()
    con.close()
    # Build a hash from row_index → row_info to attach the augmenting columns
    info_by_idx = {r[0]: r for r in rows_info}
    n = df.num_rows
    bio_hash_col = []
    one_allele_col = []
    two_allele_col = []
    subset_col = []
    order_col = []
    src_idx_col = []
    df_ridx = df.column("_ridx").to_pylist()
    for ridx in df_ridx:
        info = info_by_idx[ridx]
        bio_hash_col.append(info[1])
        one_allele_col.append(info[2])
        two_allele_col.append(info[3])
        subset_col.append(info[4])
        order_col.append(info[5])
        src_idx_col.append(ridx)

    # Drop the helper column
    df = df.drop(["_ridx"])
    df = df.append_column("bio_hash", pa.array(bio_hash_col))
    df = df.append_column("mhc_one_allele", pa.array(one_allele_col))
    df = df.append_column("mhc_two_allele", pa.array(two_allele_col))
    df = df.append_column("source_file", pa.array([source_file] * n))
    df = df.append_column("source_row_index", pa.array(src_idx_col, type=pa.int64()))
    df = df.append_column("partition", pa.array([partition_name] * n))
    return df


def task_val_test() -> None:
    """Materialize val and test partitions per-file in parallel."""
    print("[val_test] materializing val and test partitions ...", flush=True)
    t0 = time.time()

    con = _duckdb_conn()
    for partition_name, out_path in [("val", VAL_PARQUET), ("test", TEST_PARQUET)]:
        print(f"[val_test] {partition_name}: indexing partition_assignments ...", flush=True)
        # Read all val/test rows from partition_assignments grouped by source_file
        rows = con.execute(f"""
            SELECT source_file, source_row_index, bio_hash, mhc_one_allele, mhc_two_allele,
                   subset_key, order_key
            FROM read_parquet('{PARTITION_PARQUET}')
            WHERE partition = '{partition_name}'
            ORDER BY source_file, source_row_index
        """).fetchall()
        print(f"[val_test]   {len(rows):,} rows across {len(set(r[0] for r in rows))} source files", flush=True)

        # Group by source_file
        by_file: Dict[str, List[Tuple]] = {}
        for r in rows:
            by_file.setdefault(r[0], []).append(r[1:])

        # Process files in parallel
        from multiprocessing import Pool
        args_list = [(f, partition_name, [(idx, bh, m1, m2, sk, ok) for (idx, bh, m1, m2, sk, ok) in info])
                     for f, info in by_file.items()]

        # Build a canonical schema (all enriched cols are string) so we can cast each chunk uniformly.
        canonical_fields = [
            *[(c, pa.string()) for c in ENRICHED_DATA_COLS],
            ("bio_hash", pa.string()),
            ("mhc_one_allele", pa.string()),
            ("mhc_two_allele", pa.string()),
            ("source_file", pa.string()),
            ("source_row_index", pa.int64()),
            ("partition", pa.string()),
        ]
        canonical_schema = pa.schema([(name, ty) for name, ty in canonical_fields])

        n_written = 0
        done = 0
        with pq.ParquetWriter(str(out_path), canonical_schema, compression="zstd") as writer:
            with Pool(processes=THREADS) as pool:
                for tbl in pool.imap_unordered(_val_test_per_file, args_list, chunksize=8):
                    done += 1
                    if tbl is None or tbl.num_rows == 0:
                        continue
                    # Force-cast each column to the canonical type
                    arrays = []
                    for fname, ftype in canonical_fields:
                        if fname in tbl.column_names:
                            col = tbl.column(fname)
                            if col.type != ftype:
                                col = col.cast(ftype, safe=False)
                            arrays.append(col)
                        else:
                            arrays.append(pa.nulls(tbl.num_rows, type=ftype))
                    cast_tbl = pa.table(arrays, names=[name for name, _ in canonical_fields])
                    writer.write_table(cast_tbl)
                    n_written += cast_tbl.num_rows
                    if done % 50 == 0:
                        print(f"[val_test]   {partition_name}: {done}/{len(args_list)} files, {n_written:,} rows", flush=True)
        print(f"[val_test]   {partition_name}: done — {n_written:,} rows -> {out_path.name}", flush=True)
    con.close()
    elapsed = time.time() - t0
    _record_timing("val_test", elapsed)
    print(f"[val_test] done in {elapsed:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Step 5/6/7 — Sample manifests
# ---------------------------------------------------------------------------

def _compute_subset_stats() -> Dict[str, Tuple[int, int]]:
    """Return {subset_key: (N_unique_bio_hashes, k_components)} for train partition."""
    con = _duckdb_conn()
    df = con.execute(f"""
        SELECT subset_key, COUNT(DISTINCT bio_hash) AS n_unique
        FROM read_parquet('{PARTITION_PARQUET}')
        WHERE partition = 'train'
        GROUP BY subset_key
    """).df()
    con.close()
    out: Dict[str, Tuple[int, int]] = {}
    for _, row in df.iterrows():
        sk = row["subset_key"]
        n = int(row["n_unique"])
        k = _k_of_subset(sk)
        out[sk] = (n, k)
    return out


def _compute_targets(
    stats: Dict[str, Tuple[int, int]],
    total: int,
    weight_fn,  # callable: (N_c, k_c, sk) -> raw weight
) -> Dict[str, int]:
    """Compute per-subset target T_c respecting the exposure cap.

    Iteratively redistribute overflow from capped subsets to uncapped ones.
    Recomputes from scratch each iteration so cumulative caps are honored.
    """
    raw_weights: Dict[str, float] = {}
    caps: Dict[str, int] = {}
    for sk, (n, k) in stats.items():
        if n == 0:
            continue
        raw_weights[sk] = weight_fn(n, k, sk)
        caps[sk] = min(k, 20) * n

    targets: Dict[str, int] = {sk: 0 for sk in raw_weights}
    capped: set = set()
    for _ in range(50):
        sum_capped_targets = sum(targets[sk] for sk in capped)
        remaining_total = max(0, total - sum_capped_targets)
        active = [sk for sk in raw_weights if sk not in capped]
        sw = sum(raw_weights[sk] for sk in active)
        if sw == 0 or not active:
            break
        any_overflow = False
        for sk in active:
            t = int(round(remaining_total * raw_weights[sk] / sw))
            if t > caps[sk]:
                targets[sk] = caps[sk]
                capped.add(sk)
                any_overflow = True
            else:
                targets[sk] = t
        if not any_overflow:
            break
    return {sk: t for sk, t in targets.items() if t > 0}


def _sample_subset_manifest(
    sk: str,
    target: int,
    k_c: int,
    rng: np.random.Generator,
) -> pa.Table:
    """Sample target rows from this subset's train partition.

    Returns Arrow table with: bio_hash, order_key, subset_key, source_file, source_row_index.

    Two paths:
      - **k_c == 1**: each unique example has exactly 1 row in partition_assignments,
        so just sample N=target row indices uniformly without ever building a
        hash -> rows dict. Avoids the 777M-entry dict for the trb subset.
      - **k_c > 1**: same example has up to k_c rows (one per ordering). Build a
        dict once, sample examples, pick a random ordering per sample.
    """
    con = _duckdb_conn()

    if k_c == 1:
        # Get total count, sample row indices, then fetch only those rows
        n_total = con.execute(f"""
            SELECT COUNT(*) FROM read_parquet('{PARTITION_PARQUET}')
            WHERE subset_key = '{sk}' AND partition = 'train'
        """).fetchone()[0]
        if n_total == 0:
            con.close()
            return pa.table({})
        if target >= n_total:
            chosen = np.arange(n_total, dtype=np.int64)
            extra = max(0, target - n_total)
            if extra:
                more = rng.integers(0, n_total, size=extra)
                chosen = np.concatenate([chosen, more])
        else:
            chosen = rng.choice(n_total, size=target, replace=False)
        # We need a SQL query that returns rows by row_offset; use ROW_NUMBER + filter
        # For efficiency, sort the chosen indices and use IN-list batched
        chosen_sorted = sorted(chosen.tolist())
        # Use a temporary table to hold indices for join
        idx_tbl = pa.table({"_idx": pa.array(chosen_sorted, type=pa.int64())})
        con.register("chosen_idx", idx_tbl)
        rdr = con.execute(f"""
            WITH ranked AS (
                SELECT bio_hash, order_key, source_file, source_row_index,
                       ROW_NUMBER() OVER () - 1 AS _row
                FROM read_parquet('{PARTITION_PARQUET}')
                WHERE subset_key = '{sk}' AND partition = 'train'
            )
            SELECT r.bio_hash, r.order_key, r.source_file, r.source_row_index
            FROM ranked r JOIN chosen_idx c ON r._row = c._idx
        """)
        df = rdr.fetch_arrow_table() if hasattr(rdr, "fetch_arrow_table") else rdr.arrow()
        if isinstance(df, pa.RecordBatchReader):
            df = df.read_all()
        con.close()
        return pa.table({
            "bio_hash": df.column("bio_hash"),
            "order_key": df.column("order_key"),
            "subset_key": pa.array([sk] * df.num_rows),
            "source_file": df.column("source_file"),
            "source_row_index": df.column("source_row_index"),
        })

    # k_c > 1: build hash -> rows dict
    rdr = con.execute(f"""
        SELECT bio_hash, order_key, source_file, source_row_index
        FROM read_parquet('{PARTITION_PARQUET}')
        WHERE subset_key = '{sk}' AND partition = 'train'
    """)
    df = rdr.fetch_arrow_table() if hasattr(rdr, "fetch_arrow_table") else rdr.arrow()
    if isinstance(df, pa.RecordBatchReader):
        df = df.read_all()
    con.close()

    bio_hashes = df.column("bio_hash").to_pylist()
    order_keys = df.column("order_key").to_pylist()
    source_files = df.column("source_file").to_pylist()
    row_indices = df.column("source_row_index").to_pylist()

    hash_to_rows: Dict[str, List[int]] = {}
    for i, h in enumerate(bio_hashes):
        hash_to_rows.setdefault(h, []).append(i)
    unique_hashes = list(hash_to_rows.keys())
    n_unique = len(unique_hashes)
    if n_unique == 0:
        return pa.table({})

    if target <= n_unique:
        chosen_idx = rng.choice(n_unique, size=target, replace=False)
        sampled_hashes = [unique_hashes[i] for i in chosen_idx]
    else:
        baseline = target // n_unique
        remainder = target - baseline * n_unique
        extra_idx = set(rng.choice(n_unique, size=remainder, replace=False))
        sampled_hashes = []
        for i, h in enumerate(unique_hashes):
            count = baseline + (1 if i in extra_idx else 0)
            for _ in range(count):
                sampled_hashes.append(h)

    out_bio, out_ord, out_file, out_idx = [], [], [], []
    for h in sampled_hashes:
        rows = hash_to_rows[h]
        pick = rows[rng.integers(0, len(rows))]
        out_bio.append(h)
        out_ord.append(order_keys[pick])
        out_file.append(source_files[pick])
        out_idx.append(row_indices[pick])

    return pa.table({
        "bio_hash": out_bio,
        "order_key": out_ord,
        "subset_key": [sk] * len(out_bio),
        "source_file": out_file,
        "source_row_index": out_idx,
    })


def _build_manifest(
    out_path: Path,
    total: int,
    weight_fn,
    label: str,
) -> Dict[str, int]:
    """Build a sample manifest of `total` rows using `weight_fn` over subsets.

    Streams output via parquet RecordBatchWriter so memory stays bounded.
    """
    print(f"[manifest:{label}] computing per-subset targets ...", flush=True)
    stats = _compute_subset_stats()
    targets = _compute_targets(stats, total, weight_fn)
    print(f"[manifest:{label}] target allocation:", flush=True)
    for sk, t in sorted(targets.items(), key=lambda x: -x[1])[:10]:
        n, k = stats[sk]
        print(f"  {sk:<40s} N={n:>12,d} k={k} -> T={t:,}", flush=True)

    rng = np.random.default_rng(SEED)
    schema = pa.schema([
        ("bio_hash", pa.string()),
        ("order_key", pa.string()),
        ("subset_key", pa.string()),
        ("source_file", pa.string()),
        ("source_row_index", pa.int64()),
    ])
    n_written = 0
    with pq.ParquetWriter(str(out_path), schema, compression="zstd") as writer:
        for sk, target in sorted(targets.items(), key=lambda x: -x[1]):
            t0 = time.time()
            tbl = _sample_subset_manifest(sk, target, _k_of_subset(sk), rng)
            if tbl.num_rows == 0:
                continue
            tbl = tbl.cast(schema)
            writer.write_table(tbl)
            n_written += tbl.num_rows
            print(f"[manifest:{label}]   {sk}: wrote {tbl.num_rows:,} ({time.time()-t0:.1f}s) — total={n_written:,}", flush=True)

    print(f"[manifest:{label}] done — {n_written:,} rows -> {out_path.name}", flush=True)
    return targets


def task_scaled() -> None:
    t0 = time.time()
    weight_fn = lambda n, k, sk: 1.0 / math.sqrt(min(n, 100_000_000) if sk == "trb" else n)
    _build_manifest(OUT_ROOT / "foundation_10M.parquet",  10_000_000,  weight_fn, "M1_10M")
    _build_manifest(OUT_ROOT / "foundation_100M.parquet", 100_000_000, weight_fn, "M1_100M")
    _build_manifest(OUT_ROOT / "foundation_500M.parquet", 500_000_000, weight_fn, "M1_500M")
    _record_timing("scaled", time.time() - t0)


def task_minis_weight() -> None:
    t0 = time.time()
    base = lambda n, k, sk: 1.0 / math.sqrt(min(n, 100_000_000) if sk == "trb" else n)
    weight_M2 = lambda n, k, sk: base(n, k, sk) * min(k, 3)
    weight_M3 = lambda n, k, sk: base(n, k, sk) * k
    _build_manifest(OUT_ROOT / "mini_M2_10M.parquet", 10_000_000, weight_M2, "M2_10M")
    _build_manifest(OUT_ROOT / "mini_M3_10M.parquet", 10_000_000, weight_M3, "M3_10M")
    _record_timing("minis_weight", time.time() - t0)


def task_minis_order() -> None:
    """O1 (canonical) and O3 (random + use_sep)."""
    t0 = time.time()
    m1_path = OUT_ROOT / "foundation_10M.parquet"
    if not m1_path.exists():
        raise FileNotFoundError(f"Need {m1_path} for ordering ablations; run --task scaled first")

    # O1: replace order_key with canonical; re-resolve source_file/source_row_index
    # by joining with partition_assignments on (bio_hash, subset_key, canonical_order_key).
    print("[order:O1] building canonical-order manifest ...", flush=True)
    con = _duckdb_conn()

    canonical_map_sql = " ".join(
        f"WHEN '{sk}' THEN '{_canonical_order_key(sk)}'" for sk in IN_SCOPE_SUBSETS
    )
    # Pre-dedupe partition_assignments to first row per (bio_hash, subset_key, order_key).
    # bio_hash is not strictly unique within that group (different full-chain truncations
    # may map to the same CDR3-based hash), so pick a canonical first occurrence.
    sql = f"""
    COPY (
        WITH base AS (
            SELECT bio_hash, subset_key,
                   CASE subset_key {canonical_map_sql} ELSE order_key END AS canonical_order_key
            FROM read_parquet('{m1_path}')
        ),
        partition_dedup AS (
            SELECT bio_hash, subset_key, order_key, source_file, source_row_index
            FROM (
                SELECT bio_hash, subset_key, order_key, source_file, source_row_index,
                       ROW_NUMBER() OVER (PARTITION BY bio_hash, subset_key, order_key
                                          ORDER BY source_file, source_row_index) AS _rn
                FROM read_parquet('{PARTITION_PARQUET}')
                WHERE partition = 'train'
            )
            WHERE _rn = 1
        ),
        resolved AS (
            SELECT b.bio_hash, p.order_key, b.subset_key, p.source_file, p.source_row_index
            FROM base b
            JOIN partition_dedup p
              ON b.bio_hash = p.bio_hash
             AND b.subset_key = p.subset_key
             AND b.canonical_order_key = p.order_key
        )
        SELECT * FROM resolved
    ) TO '{OUT_ROOT / "mini_O1_10M.parquet"}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(sql)
    n_o1 = con.execute(f"SELECT COUNT(*) FROM read_parquet('{OUT_ROOT}/mini_O1_10M.parquet')").fetchone()[0]
    print(f"[order:O1] {n_o1:,} rows", flush=True)

    # O3: same as M1 but add use_sep = True
    print("[order:O3] building use_sep manifest ...", flush=True)
    sql_o3 = f"""
    COPY (
        SELECT *, true AS use_sep FROM read_parquet('{m1_path}')
    ) TO '{OUT_ROOT / "mini_O3_10M.parquet"}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(sql_o3)
    n_o3 = con.execute(f"SELECT COUNT(*) FROM read_parquet('{OUT_ROOT}/mini_O3_10M.parquet')").fetchone()[0]
    print(f"[order:O3] {n_o3:,} rows", flush=True)

    con.close()
    _record_timing("minis_order", time.time() - t0)


# ---------------------------------------------------------------------------
# Step 8 — Summary
# ---------------------------------------------------------------------------

def task_summary() -> None:
    t0 = time.time()
    summary: Dict[str, object] = {}
    con = _duckdb_conn()

    # Exclusion stats
    if EXCLUSION_PARQUET.exists():
        n_excl = con.execute(f"SELECT COUNT(*) FROM read_parquet('{EXCLUSION_PARQUET}')").fetchone()[0]
        summary["exclusion_count"] = n_excl

    # Partition totals
    if PARTITION_PARQUET.exists():
        rows = con.execute(f"""
            SELECT partition, COUNT(*) AS n
            FROM read_parquet('{PARTITION_PARQUET}')
            GROUP BY partition
        """).fetchall()
        summary["partition_totals"] = {p: n for p, n in rows}

        by_subset = con.execute(f"""
            SELECT subset_key, partition, COUNT(*) AS n
            FROM read_parquet('{PARTITION_PARQUET}')
            GROUP BY subset_key, partition
        """).fetchall()
        bs: Dict[str, Dict[str, int]] = {}
        for sk, p, n in by_subset:
            bs.setdefault(sk, {})[p] = n
        summary["per_subset_partition"] = bs

    # Stats per scaled file
    summary["output_sizes"] = {}
    summary["exposure_per_dataset"] = {}
    for fname in [
        "foundation_10M.parquet", "foundation_100M.parquet", "foundation_500M.parquet",
        "mini_M2_10M.parquet", "mini_M3_10M.parquet",
        "mini_O1_10M.parquet", "mini_O3_10M.parquet",
        "foundation_val.parquet", "foundation_test.parquet",
    ]:
        p = OUT_ROOT / fname
        if not p.exists():
            continue
        n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{p}')").fetchone()[0]
        size_gb = p.stat().st_size / 1e9
        summary["output_sizes"][fname] = {"rows": n, "size_gb": round(size_gb, 2)}
        if "foundation_" in fname or fname.startswith("mini_"):
            try:
                exp = con.execute(f"""
                    WITH counts AS (
                        SELECT bio_hash, COUNT(*) AS c FROM read_parquet('{p}') GROUP BY bio_hash
                    )
                    SELECT MIN(c), MEDIAN(c), MAX(c) FROM counts
                """).fetchone()
                summary["exposure_per_dataset"][fname] = {
                    "min": int(exp[0] or 0), "median": float(exp[1] or 0), "max": int(exp[2] or 0)
                }
            except Exception:
                pass

    # Timings
    if TIMINGS_PATH.exists():
        summary["timings_seconds"] = json.loads(TIMINGS_PATH.read_text())

    con.close()
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[summary] wrote {SUMMARY_PATH}", flush=True)

    # Stdout table
    print()
    print(f"{'file':<40} {'rows':>15} {'size_gb':>10}")
    print("-" * 70)
    for fname, info in summary.get("output_sizes", {}).items():
        print(f"{fname:<40} {info['rows']:>15,} {info['size_gb']:>10.2f}")
    _record_timing("summary", time.time() - t0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", required=True,
        choices=["exclusion", "partition", "val_test", "scaled", "minis_weight", "minis_order", "summary", "all"],
    )
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.task == "exclusion":
        task_exclusion()
    elif args.task == "partition":
        task_partition()
    elif args.task == "val_test":
        task_val_test()
    elif args.task == "scaled":
        task_scaled()
    elif args.task == "minis_weight":
        task_minis_weight()
    elif args.task == "minis_order":
        task_minis_order()
    elif args.task == "summary":
        task_summary()
    elif args.task == "all":
        task_exclusion()
        task_partition()
        task_val_test()
        task_scaled()
        task_minis_weight()
        task_minis_order()
        task_summary()


if __name__ == "__main__":
    sys.exit(main())
