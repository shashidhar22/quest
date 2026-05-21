"""Stages 3 + 5: cluster -> split assignment + master table.

Algorithm: largest-cluster-first into largest-remaining-budget (LPT bin-packing
heuristic). This handles heavy-tailed cluster-size distributions correctly
on heterogeneous axes (TRB has heavy tail; MHC pockets have tiny universes).

Per axis:
  - Compute cluster_record_count = sum(n_records) over molecules in cluster
  - Sort clusters by cluster_record_count DESC, hash(cluster_id + seed) tiebreak
  - Initialize budgets: train = 0.80*total, val = 0.10*total, test = 0.10*total
  - Walk clusters in sorted order; assign each to split with largest remaining
    budget (deterministic train>val>test preference on exact ties)
  - Write split_assignments/<axis>_split_assignment.parquet (cluster_id, split)

Then Stage 5 joins clusters + split_assignments per axis and concatenates
into master_cluster_assignments.parquet.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/home/ubuntu/quest/data/molecule_clusters")
CLUSTERS_DIR = ROOT / "clusters"
SPLIT_DIR = ROOT / "split_assignments"
ARTIFACT_DIR = ROOT / "_build_artifacts"
MASTER_PATH = ROOT / "master_cluster_assignments.parquet"
RECORD_COUNTS_PATH = ARTIFACT_DIR / "record_counts_per_molecule.parquet"

SEED = 42
AXES = [
    ("trb_cdr3", 0),
    ("tra_cdr3", 1),
    ("peptide", 2),
    ("mhc_one_pocket", 3),
    ("mhc_two_pocket", 4),
]

TARGET_FRACS = {"train": 0.80, "val": 0.10, "test": 0.10}


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{now_iso()}] {msg}", flush=True)


def assign_lpt(sizes: np.ndarray, total_rows: int,
               rng: np.random.Generator) -> np.ndarray:
    """LPT bin-packing assignment.

    sizes[i] = cluster_record_count for cluster at position i (in pre-sort order)
    Returns labels: array of bytes 't'/'v'/'r' (train/val/test) length n.

    Sort is performed inside this function using a deterministic random
    tiebreaker so ties (e.g. many singletons) get reproducibly distributed.
    """
    n = len(sizes)
    # Deterministic random tiebreak for same-size clusters
    jitter = rng.random(n)
    # Sort indices by (-size, jitter) ascending => largest first, then by jitter
    # We use lexsort: last key is the primary
    order = np.lexsort((jitter, -sizes.astype(np.int64)))

    budget_train = TARGET_FRACS["train"] * total_rows
    budget_val = TARGET_FRACS["val"] * total_rows
    budget_test = TARGET_FRACS["test"] * total_rows

    out = np.empty(n, dtype="U5")
    sizes_int = sizes.astype(np.int64)
    for k in range(n):
        i = order[k]
        s = sizes_int[i]
        # Compare budgets; ties broken by train > val > test
        if budget_train >= budget_val and budget_train >= budget_test:
            out[i] = "train"
            budget_train -= s
        elif budget_val >= budget_test:
            out[i] = "val"
            budget_val -= s
        else:
            out[i] = "test"
            budget_test -= s
    return out


def stage3_one_axis(con: duckdb.DuckDBPyConnection, axis: str, axis_idx: int) -> dict:
    log(f"=== {axis} (axis_idx={axis_idx}) ===")
    t0 = time.time()

    cl_path = str(CLUSTERS_DIR / f"{axis}_clusters.parquet")
    rc_path = str(RECORD_COUNTS_PATH)
    out_path = str(SPLIT_DIR / f"{axis}_split_assignment.parquet")
    axis_seed = SEED + axis_idx

    # Determine the molecule column name in record_counts
    rc_cols = con.execute(
        f"SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{rc_path}'))"
    ).fetchall()
    rc_col_names = [r[0] for r in rc_cols]
    if "molecule_string" in rc_col_names:
        rc_mol = "molecule_string"
    elif axis in rc_col_names:
        rc_mol = axis
    else:
        raise ValueError(f"can't find molecule col in record_counts: {rc_col_names}")

    # Step 1: get per-cluster (cluster_id, cluster_record_count) via DuckDB
    log("  computing per-cluster record counts...")
    t_sums = time.time()
    sums_df = con.execute(f"""
        SELECT c.cluster_id,
               SUM(COALESCE(r.n_records, 1))::BIGINT AS cluster_record_count
        FROM read_parquet('{cl_path}') c
        LEFT JOIN (
          SELECT {rc_mol} AS molecule, n_records
          FROM read_parquet('{rc_path}')
          WHERE axis = '{axis}'
        ) r ON c.{axis} = r.molecule
        GROUP BY c.cluster_id
        ORDER BY c.cluster_id
    """).fetch_arrow_table()
    log(f"    {len(sums_df):,} clusters, "
        f"sums computed in {time.time()-t_sums:.1f}s")

    cluster_ids = sums_df.column("cluster_id").to_numpy(zero_copy_only=False)
    sizes = sums_df.column("cluster_record_count").to_numpy(zero_copy_only=False)
    total_rows = int(sizes.sum())
    log(f"    total axis rows: {total_rows:,}")

    # Step 2: LPT assignment in numpy
    log("  running LPT assignment...")
    t_lpt = time.time()
    rng = np.random.default_rng(axis_seed)
    splits = assign_lpt(sizes, total_rows, rng)
    log(f"    LPT done in {time.time()-t_lpt:.1f}s")

    # Step 3: write parquet
    out_tbl = pa.table({
        "cluster_id": pa.array(cluster_ids, type=pa.int64()),
        "split": pa.array(splits, type=pa.large_string()),
    })
    pq.write_table(out_tbl, out_path, compression="zstd")
    log(f"  wrote {out_path}")

    # Stats
    n_clusters = len(cluster_ids)
    n_train = int((splits == "train").sum())
    n_val = int((splits == "val").sum())
    n_test = int((splits == "test").sum())
    rows_train = int(sizes[splits == "train"].sum())
    rows_val = int(sizes[splits == "val"].sum())
    rows_test = int(sizes[splits == "test"].sum())
    log(f"  clusters: train={n_train:,} ({n_train/n_clusters:.4f}), "
        f"val={n_val:,} ({n_val/n_clusters:.4f}), "
        f"test={n_test:,} ({n_test/n_clusters:.4f})")
    log(f"  rows:     train={rows_train:,} ({rows_train/total_rows:.4f}), "
        f"val={rows_val:,} ({rows_val/total_rows:.4f}), "
        f"test={rows_test:,} ({rows_test/total_rows:.4f})")

    return {
        "axis": axis,
        "axis_idx": axis_idx,
        "seed": axis_seed,
        "n_clusters": n_clusters,
        "cluster_split_counts": {"train": n_train, "val": n_val, "test": n_test},
        "row_split_counts_by_axis": {"train": rows_train, "val": rows_val, "test": rows_test},
        "total_axis_rows": total_rows,
        "elapsed_s": float(time.time() - t0),
        "output_path": out_path,
    }


def stage5_build_master(con: duckdb.DuckDBPyConnection) -> dict:
    log("=== Stage 5: building master_cluster_assignments.parquet ===")
    t0 = time.time()

    select_clauses = []
    for axis, _ in AXES:
        cl_path = str(CLUSTERS_DIR / f"{axis}_clusters.parquet")
        sp_path = str(SPLIT_DIR / f"{axis}_split_assignment.parquet")
        select_clauses.append(f"""
          SELECT '{axis}' AS axis,
                 c.{axis}::VARCHAR AS molecule_string,
                 c.cluster_id,
                 s.split
          FROM read_parquet('{cl_path}') c
          JOIN read_parquet('{sp_path}') s USING (cluster_id)
        """)
    union_sql = " UNION ALL ".join(select_clauses)
    sql = f"""
    COPY (
        {union_sql}
    ) TO '{MASTER_PATH}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(sql)
    elapsed = time.time() - t0
    n_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{MASTER_PATH}')"
    ).fetchone()[0]
    log(f"  wrote {MASTER_PATH} ({n_rows:,} rows) in {elapsed:.1f}s")
    return {"total_rows": int(n_rows), "elapsed_s": float(elapsed)}


def main():
    overall_t0 = time.time()
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(":memory:")
    con.execute("PRAGMA threads = 64;")
    con.execute("PRAGMA memory_limit = '700GB';")
    con.execute("PRAGMA temp_directory = '/home/ubuntu/quest/data/molecule_clusters/_build_artifacts/duckdb_tmp';")
    Path("/home/ubuntu/quest/data/molecule_clusters/_build_artifacts/duckdb_tmp").mkdir(exist_ok=True)

    stage3_stats = []
    for axis, axis_idx in AXES:
        stats = stage3_one_axis(con, axis, axis_idx)
        stage3_stats.append(stats)

    stage5_stats = stage5_build_master(con)

    manifest = {
        "stage": "stages_3_5",
        "seed_base": SEED,
        "algorithm": "Largest-cluster-first into largest-remaining-budget (LPT bin-packing). "
                     "Sort clusters by cluster_record_count DESC with deterministic per-axis "
                     "jitter tiebreaker. Walk sorted list; assign each cluster to whichever of "
                     "{train, val, test} has the largest remaining row budget; on exact ties "
                     "prefer train > val > test. Initial budgets = 0.80/0.10/0.10 of total "
                     "axis rows. Guarantees each split is within max_cluster_size of its target.",
        "target_row_fractions": TARGET_FRACS,
        "axes": stage3_stats,
        "master_table": stage5_stats,
        "elapsed_total_s": float(time.time() - overall_t0),
        "timestamp_utc": now_iso(),
    }
    manifest_path = ARTIFACT_DIR / "stages_3_5_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log(f"wrote {manifest_path}")
    log(f"DONE in {time.time()-overall_t0:.1f}s")


if __name__ == "__main__":
    main()
