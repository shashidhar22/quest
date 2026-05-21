"""Re-emit deduplicated_again/{deduped_parquet,exploded_deduped} with the
allele columns truncated to 4-digit IMGT resolution (e.g. HLA-A*02:01:01:01
→ HLA-A*02:01).

Why a follow-up step: the in-flight tcrbench_dedup.py run loaded the old
build_allele_lookups() into memory and emitted parquet with the
full-resolution alleles already present upstream. The heavy GROUP BY work
is preserved in /data/deduplicated_again/deduped.db, so this script only
re-runs the cheap allele-JOIN and the parquet writes.

Strategy:
  1. Open the existing /data/deduplicated_again/deduped.db (1.44B-row deduped table preserved).
  2. Build 4-digit-truncated allele lookup tables from upstream standardized.
  3. Re-emit deduped_parquet/ with the truncated allele columns (LEFT JOIN at write).
  4. For each of the 31 subset_keys, rebuild the m_subset temp table and re-emit
     all order_key partitions with the truncated alleles + 'sequence' column.

Idempotent: writes to *_new directories first, then atomically renames.
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import duckdb

# Mirror tcrbench_dedup constants so we don't have to import the full module.
DEDUP_COLUMNS = ["tra_full", "trb_full", "peptide", "mhc_one", "mhc_two"]
SHORT_NAMES = ["tra", "trb", "peptide", "mhc_one", "mhc_two"]
ALL_CDR_COLUMNS = [
    "tra_cdr1", "tra_cdr2", "tra_cdr3",
    "trb_cdr1", "trb_cdr2", "trb_cdr3",
]
CDR_COLUMNS = {
    "tra_full": ["tra_cdr1", "tra_cdr2", "tra_cdr3"],
    "trb_full": ["trb_cdr1", "trb_cdr2", "trb_cdr3"],
}
DEDUP_DIR = "/data/deduplicated_again"
DB_PATH = f"{DEDUP_DIR}/deduped.db"
INPUT_GLOB = "/data/standardized_again/**/*.parquet"

TRUNCATE = (
    "COALESCE(NULLIF(regexp_extract({col}, '^([^:]+:[^:]+)', 1), ''), {col})"
)


def build_perms() -> list[tuple[str, int, tuple[int, ...]]]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory_limit", default="400GB")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--tmp_dir", default="/mnt/scratch_nvme")
    parser.add_argument(
        "--skip_explosion",
        action="store_true",
        help="Only re-emit deduped_parquet (skip the 325-partition explosion)",
    )
    args = parser.parse_args()

    print(f"connecting to {DB_PATH}")
    con = duckdb.connect(database=DB_PATH)
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET temp_directory='{args.tmp_dir}'")
    con.execute("SET preserve_insertion_order=false")

    # ---- 1. Verify the deduped table exists ----------------------------------
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    assert "deduped" in tables, (
        f"no `deduped` table in {DB_PATH} — run tcrbench_dedup.py first"
    )
    n = con.execute("SELECT COUNT(*) FROM deduped").fetchone()[0]
    print(f"  deduped table: {n:,} rows")

    # ---- 2. Build 4-digit-truncated lookups ---------------------------------
    print("building 4-digit allele lookups…")
    t0 = time.time()
    con.execute(f"""
        CREATE OR REPLACE TABLE mhc_one_allele_lookup AS
        SELECT mhc_one, MIN({TRUNCATE.format(col='mhc_one_allele')}) AS mhc_one_allele
        FROM read_parquet('{INPUT_GLOB}', hive_partitioning=true)
        WHERE mhc_one IS NOT NULL AND mhc_one != ''
          AND mhc_one_allele IS NOT NULL AND mhc_one_allele != ''
        GROUP BY mhc_one
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE mhc_two_allele_lookup AS
        SELECT mhc_two, MIN({TRUNCATE.format(col='mhc_two_allele')}) AS mhc_two_allele
        FROM read_parquet('{INPUT_GLOB}', hive_partitioning=true)
        WHERE mhc_two IS NOT NULL AND mhc_two != ''
          AND mhc_two_allele IS NOT NULL AND mhc_two_allele != ''
        GROUP BY mhc_two
    """)
    n1 = con.execute("SELECT COUNT(*) FROM mhc_one_allele_lookup").fetchone()[0]
    n2 = con.execute("SELECT COUNT(*) FROM mhc_two_allele_lookup").fetchone()[0]
    print(f"  lookups built in {time.time()-t0:.1f}s: {n1:,} mhc_one, {n2:,} mhc_two")

    # ---- 3. Re-emit deduped_parquet -----------------------------------------
    new_dir = f"{DEDUP_DIR}/deduped_parquet_new"
    old_dir = f"{DEDUP_DIR}/deduped_parquet"
    bak_dir = f"{DEDUP_DIR}/deduped_parquet_bak"
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.makedirs(new_dir)

    print(f"re-emitting deduped_parquet → {new_dir}")
    t0 = time.time()
    select = (
        ", ".join(f"d.{c}" for c in DEDUP_COLUMNS + ALL_CDR_COLUMNS)
        + ", l1.mhc_one_allele AS mhc_one_allele"
        + ", l2.mhc_two_allele AS mhc_two_allele"
    )
    con.execute(f"""
        COPY (
            SELECT {select} FROM deduped d
            LEFT JOIN mhc_one_allele_lookup l1 ON d.mhc_one = l1.mhc_one
            LEFT JOIN mhc_two_allele_lookup l2 ON d.mhc_two = l2.mhc_two
        ) TO '{new_dir}/'
        (FORMAT PARQUET, PER_THREAD_OUTPUT true, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)
    """)
    print(f"  emitted in {time.time()-t0:.1f}s")

    # Atomic swap
    if os.path.exists(bak_dir):
        shutil.rmtree(bak_dir)
    os.rename(old_dir, bak_dir)
    os.rename(new_dir, old_dir)
    print(f"  swapped: {old_dir} → {bak_dir}, new → {old_dir}")

    if args.skip_explosion:
        print("--skip_explosion: done")
        return

    # ---- 4. Re-emit exploded_deduped ----------------------------------------
    print("re-emitting exploded_deduped (325 partitions)…")
    perms = build_perms()
    perms_by_mask: dict[int, list[tuple[str, tuple[int, ...]]]] = defaultdict(list)
    for key, mask, col_indices in perms:
        perms_by_mask[mask].append((key, col_indices))

    new_explode = f"{DEDUP_DIR}/exploded_deduped_new"
    old_explode = f"{DEDUP_DIR}/exploded_deduped"
    bak_explode = f"{DEDUP_DIR}/exploded_deduped_bak"
    if os.path.exists(new_explode):
        shutil.rmtree(new_explode)
    os.makedirs(new_explode)

    t_total = time.time()
    for mask in sorted(perms_by_mask):
        cols_in_mask = [DEDUP_COLUMNS[i] for i in range(len(DEDUP_COLUMNS)) if mask & (1 << i)]
        cdr_cols_in_mask: list[str] = []
        for parent in cols_in_mask:
            cdr_cols_in_mask.extend(CDR_COLUMNS.get(parent, []))
        where = " AND ".join(f"{c} IS NOT NULL" for c in cols_in_mask)
        select_parts = list(cols_in_mask) + [
            f"FIRST({c}) AS {c}" for c in cdr_cols_in_mask
        ]
        con.execute("DROP TABLE IF EXISTS m_subset")
        con.execute(
            f"CREATE TEMP TABLE m_subset AS "
            f"SELECT {', '.join(select_parts)} "
            f"FROM deduped WHERE {where} "
            f"GROUP BY {', '.join(cols_in_mask)}"
        )
        cnt = con.execute("SELECT COUNT(*) FROM m_subset").fetchone()[0]
        subset_key = "_".join(
            SHORT_NAMES[i] for i in range(len(DEDUP_COLUMNS)) if mask & (1 << i)
        )
        print(f"  mask={mask:>2} subset_key={subset_key} rows={cnt:,} perms={len(perms_by_mask[mask])}")

        for key, col_indices in perms_by_mask[mask]:
            partition_dir = f"{new_explode}/subset_key={subset_key}/order_key={key}"
            os.makedirs(partition_dir, exist_ok=True)
            copy_parts: list[str] = []
            for col in DEDUP_COLUMNS:
                if col in cols_in_mask:
                    copy_parts.append(f"m.{col}")
                else:
                    copy_parts.append(f"CAST(NULL AS VARCHAR) AS {col}")
            for cdr in ALL_CDR_COLUMNS:
                if cdr in cdr_cols_in_mask:
                    copy_parts.append(f"m.{cdr}")
                else:
                    copy_parts.append(f"CAST(NULL AS VARCHAR) AS {cdr}")
            if "mhc_one" in cols_in_mask:
                copy_parts.append("l1.mhc_one_allele AS mhc_one_allele")
            else:
                copy_parts.append("CAST(NULL AS VARCHAR) AS mhc_one_allele")
            if "mhc_two" in cols_in_mask:
                copy_parts.append("l2.mhc_two_allele AS mhc_two_allele")
            else:
                copy_parts.append("CAST(NULL AS VARCHAR) AS mhc_two_allele")
            ordered_cols = [DEDUP_COLUMNS[i] for i in col_indices]
            copy_parts.append(
                f"CONCAT_WS(' ', {', '.join(f'm.{c}' for c in ordered_cols)}) AS sequence"
            )

            joins = []
            if "mhc_one" in cols_in_mask:
                joins.append("LEFT JOIN mhc_one_allele_lookup l1 ON m.mhc_one = l1.mhc_one")
            if "mhc_two" in cols_in_mask:
                joins.append("LEFT JOIN mhc_two_allele_lookup l2 ON m.mhc_two = l2.mhc_two")

            con.execute(
                "COPY (SELECT " + ", ".join(copy_parts)
                + " FROM m_subset m " + " ".join(joins) + ") "
                f"TO '{partition_dir}/' "
                "(FORMAT PARQUET, PER_THREAD_OUTPUT true, "
                "COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)"
            )
        con.execute("DROP TABLE m_subset")

    print(f"  exploded re-emit done in {time.time()-t_total:.1f}s")

    if os.path.exists(bak_explode):
        shutil.rmtree(bak_explode)
    os.rename(old_explode, bak_explode)
    os.rename(new_explode, old_explode)
    print(f"  swapped: {old_explode} → {bak_explode}, new → {old_explode}")
    print("done")


if __name__ == "__main__":
    main()
