"""Build train/val/test splits for benchmark_v2 from clustering artifacts.

See ``docs/BENCHMARK_SPLITS.md`` for the full specification, novelty condition
definitions, and downstream consumption notes.

Usage:
    python build_benchmark_splits.py --task {prep_lookup,as,pm,pair,mr,leakage,summary,all}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

SOURCE_ROOT = Path("/home/ubuntu/quest/data/deduplicated_again/exploded_deduped_enriched")
STANDARDIZED_ROOT = Path("/home/ubuntu/quest/data/standardized_again")
CLUSTERS_ROOT = Path("/home/ubuntu/quest/data/benchmark_v2/clusters")
OUT_ROOT = Path("/home/ubuntu/quest/data/benchmark_v2/splits")
LOOKUP_DIR = OUT_ROOT / "lookup"
LOG_DIR = OUT_ROOT / "logs"
SCRATCH = Path("/scratch")
TIMINGS_PATH = OUT_ROOT / "timings.json"
SUMMARY_PATH = OUT_ROOT / "split_summary.json"
LEAKAGE_PATH = OUT_ROOT / "leakage_validation.json"
LOOKUP_PARQUET = LOOKUP_DIR / "sequence_to_source_allele.parquet"
LOOKUP_STATS = LOOKUP_DIR / "lookup_stats.json"

THREADS = 64
SEED = 42

EXPERIMENTAL_SOURCES = (
    "iedb", "cedar", "vdjdb", "mcpas", "batman", "trait",
    "immunecode", "ots",
)
COMPUTATIONAL_SOURCES = ("netmhcpan", "cedar_pmhc")

PRIORITY_ORDER = (
    *EXPERIMENTAL_SOURCES,
    "iedb_pmhc", "adc", "immuneaccess", "rcc_atlas", "studies", "tadb", "tcrdb", "imgthla",
)

# All columns we want to retain on output rows. Pulled from enriched-stage schema.
ENRICHED_DATA_COLS: List[str] = [
    "tra_full", "trb_full", "peptide", "mhc_one", "mhc_two",
    "tra_cdr1", "tra_cdr2", "tra_cdr3",
    "trb_cdr1", "trb_cdr2", "trb_cdr3",
    "sequence",
    "subset_key", "order_key",
    "mhc_one_pocket", "mhc_one_contact", "mhc_one_pocket_contact",
    "mhc_two_pocket", "mhc_two_contact", "mhc_two_pocket_contact",
]


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

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


def _classify(subset_key: str) -> Dict[str, bool]:
    has_tra = "tra" in subset_key
    has_trb = "trb" in subset_key
    has_peptide = "peptide" in subset_key
    has_mhc_one = "mhc_one" in subset_key
    has_mhc_two = "mhc_two" in subset_key
    return {
        "has_tra": has_tra,
        "has_trb": has_trb,
        "has_peptide": has_peptide,
        "has_mhc_one": has_mhc_one,
        "has_mhc_two": has_mhc_two,
    }


def _list_subset_dirs() -> List[Path]:
    return sorted(p for p in SOURCE_ROOT.iterdir() if p.is_dir() and p.name.startswith("subset_key="))


def _subset_key_of(p: Path) -> str:
    return p.name.split("=", 1)[1]


def _files_for(subset_keys: Sequence[str]) -> List[str]:
    files: List[str] = []
    for skey in subset_keys:
        d = SOURCE_ROOT / f"subset_key={skey}"
        if not d.exists():
            continue
        files.extend(str(f) for f in d.rglob("*.parquet"))
    return files


def _row_id(*parts: Optional[str]) -> str:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Cluster TSV loading
# ---------------------------------------------------------------------------

def load_cluster_tsv(path: Path, seq_col: str = "sequence") -> Dict[str, str]:
    """Load a cluster TSV (header: sequence|peptide \\t cluster_id) into a dict.

    Streams the file (peptide TSVs use 'peptide' as the seq column).
    """
    mapping: Dict[str, str] = {}
    with open(path, "rb") as f:
        header = f.readline().rstrip(b"\n").split(b"\t")
        try:
            si = header.index(seq_col.encode())
        except ValueError:
            si = 0
        ci = 1 if si == 0 else 0
        for line in f:
            parts = line.rstrip(b"\n").split(b"\t")
            if len(parts) < 2:
                continue
            mapping[parts[si].decode()] = parts[ci].decode()
    return mapping


# ---------------------------------------------------------------------------
# DuckDB connection helper
# ---------------------------------------------------------------------------

def _duckdb_conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET threads={THREADS}")
    con.execute("SET memory_limit='400GB'")
    con.execute(f"SET temp_directory='{SCRATCH}'")
    return con


# ---------------------------------------------------------------------------
# Step 1a — prep_lookup
# ---------------------------------------------------------------------------

def task_prep_lookup() -> None:
    """One-time scan of standardized_again/* to build sequence-tuple -> (source, allele) lookup."""
    t0 = time.time()
    LOOKUP_DIR.mkdir(parents=True, exist_ok=True)
    if LOOKUP_PARQUET.exists():
        print(f"[prep_lookup] {LOOKUP_PARQUET} already exists; skipping (delete to rebuild)", flush=True)
        return

    con = _duckdb_conn()

    # Build the experimental-priority-aware aggregation in SQL.
    exp_list_sql = ",".join(f"'{s}'" for s in EXPERIMENTAL_SOURCES)
    comp_list_sql = ",".join(f"'{s}'" for s in COMPUTATIONAL_SOURCES)

    print(f"[prep_lookup] scanning {STANDARDIZED_ROOT}/**/*.parquet ...", flush=True)
    # Filter out bulk TCR-only rows (no peptide AND no MHC) — they don't contribute
    # to AS/PM/MR splits and dominate immuneaccess/adc by row count.
    sql = f"""
    COPY (
        WITH src AS (
            SELECT
                COALESCE(tra_full, '')        AS tra_full,
                COALESCE(trb_full, '')        AS trb_full,
                COALESCE(peptide, '')         AS peptide,
                COALESCE(mhc_one, '')         AS mhc_one,
                COALESCE(mhc_two, '')         AS mhc_two,
                NULLIF(source, '')            AS source,
                NULLIF(mhc_one_allele, '')    AS mhc_one_allele,
                NULLIF(mhc_two_allele, '')    AS mhc_two_allele
            FROM read_parquet('{STANDARDIZED_ROOT}/**/*.parquet', union_by_name=true)
            WHERE
                COALESCE(peptide, '') <> ''
                OR COALESCE(mhc_one, '') <> ''
                OR COALESCE(mhc_two, '') <> ''
                OR (COALESCE(tra_full, '') <> '' AND COALESCE(trb_full, '') <> '')
        ),
        agg AS (
            SELECT
                tra_full, trb_full, peptide, mhc_one, mhc_two,
                list_distinct(list(source) FILTER (source IS NOT NULL)) AS source_db_set,
                list_distinct(list(mhc_one_allele) FILTER (mhc_one_allele IS NOT NULL)) AS mhc_one_allele_set,
                list_distinct(list(mhc_two_allele) FILTER (mhc_two_allele IS NOT NULL)) AS mhc_two_allele_set
            FROM src
            GROUP BY tra_full, trb_full, peptide, mhc_one, mhc_two
        )
        SELECT
            tra_full, trb_full, peptide, mhc_one, mhc_two,
            source_db_set,
            COALESCE(
                list_filter(source_db_set, s -> s IN ({exp_list_sql}))[1],
                list_filter(source_db_set, s -> s IN ({comp_list_sql}))[1],
                source_db_set[1]
            ) AS source_db_primary,
            mhc_one_allele_set[1] AS mhc_one_allele,
            mhc_two_allele_set[1] AS mhc_two_allele,
            mhc_one_allele_set,
            mhc_two_allele_set
        FROM agg
    ) TO '{LOOKUP_PARQUET}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(sql)

    # Compute stats for the audit
    print("[prep_lookup] computing stats...", flush=True)
    stats = con.execute(f"""
        SELECT
            COUNT(*) AS n_rows,
            COUNT(*) FILTER (WHERE len(source_db_set) > 1) AS n_multi_source,
            COUNT(*) FILTER (WHERE len(mhc_one_allele_set) > 1) AS n_multi_mhc_one_allele,
            COUNT(*) FILTER (WHERE len(mhc_two_allele_set) > 1) AS n_multi_mhc_two_allele,
            COUNT(*) FILTER (WHERE source_db_primary IS NULL) AS n_no_source
        FROM read_parquet('{LOOKUP_PARQUET}')
    """).fetchone()
    distinct_sources = con.execute(f"""
        SELECT DISTINCT unnest(source_db_set) AS s
        FROM read_parquet('{LOOKUP_PARQUET}')
        ORDER BY s
    """).fetchall()
    con.close()

    n_rows, n_multi_src, n_multi_mhc1, n_multi_mhc2, n_no_src = stats
    LOOKUP_STATS.write_text(json.dumps({
        "n_rows": n_rows,
        "n_multi_source": n_multi_src,
        "n_multi_mhc_one_allele": n_multi_mhc1,
        "n_multi_mhc_two_allele": n_multi_mhc2,
        "n_no_source": n_no_src,
        "distinct_sources_seen": [s[0] for s in distinct_sources if s[0]],
        "experimental_sources": list(EXPERIMENTAL_SOURCES),
        "computational_sources": list(COMPUTATIONAL_SOURCES),
    }, indent=2))
    elapsed = time.time() - t0
    _record_timing("prep_lookup", elapsed)
    print(f"[prep_lookup] {n_rows:,} rows; {n_multi_src:,} multi-source, "
          f"{n_multi_mhc1:,} multi-mhc1-allele, {n_multi_mhc2:,} multi-mhc2-allele "
          f"({elapsed:.1f}s)", flush=True)


# ---------------------------------------------------------------------------
# Read enriched rows + attach lookup
# ---------------------------------------------------------------------------

def _read_enriched(con: duckdb.DuckDBPyConnection, files: List[str]) -> pa.Table:
    """Read enriched rows for given parquet files into Arrow Table.

    Joins with the source lookup to attach source_db_primary, mhc_one_allele, mhc_two_allele.
    Falls back to mhc_one/mhc_two protein sequence when allele lookup is null.
    """
    if not files:
        return pa.Table.from_arrays([pa.array([])], names=["empty"])

    files_sql = "['" + "','".join(files) + "']"
    cols = ", ".join(f"e.{c} AS {c}" for c in ENRICHED_DATA_COLS)
    sql = f"""
        SELECT
            {cols},
            l.source_db_primary AS source_db_primary,
            l.source_db_set AS source_db_set,
            COALESCE(l.mhc_one_allele, e.mhc_one) AS mhc_one_allele,
            COALESCE(l.mhc_two_allele, e.mhc_two) AS mhc_two_allele
        FROM read_parquet({files_sql}) e
        LEFT JOIN read_parquet('{LOOKUP_PARQUET}') l
        ON COALESCE(e.tra_full, '') = COALESCE(l.tra_full, '')
       AND COALESCE(e.trb_full, '') = COALESCE(l.trb_full, '')
       AND COALESCE(e.peptide, '')  = COALESCE(l.peptide, '')
       AND COALESCE(e.mhc_one, '')  = COALESCE(l.mhc_one, '')
       AND COALESCE(e.mhc_two, '')  = COALESCE(l.mhc_two, '')
    """
    return con.execute(sql).fetch_arrow_table()


def _add_row_id(table: pa.Table) -> pa.Table:
    """Add row_id, source_type, and non_null_count columns to an arrow Table."""
    n = table.num_rows
    if n == 0:
        return (
            table
            .append_column("row_id", pa.array([], type=pa.string()))
            .append_column("source_type", pa.array([], type=pa.string()))
            .append_column("non_null_count", pa.array([], type=pa.int32()))
        )

    cols_for_id = ["tra_full", "trb_full", "peptide", "mhc_one", "mhc_two", "subset_key"]
    py_arrs = [table.column(c).to_pylist() for c in cols_for_id]
    row_ids = [_row_id(*[a[i] for a in py_arrs]) for i in range(n)]

    src_primary = table.column("source_db_primary").to_pylist() if "source_db_primary" in table.column_names else [None] * n
    source_type = []
    for s in src_primary:
        if s is None:
            source_type.append("unknown")
        elif s in EXPERIMENTAL_SOURCES:
            source_type.append("experimental")
        elif s in COMPUTATIONAL_SOURCES:
            source_type.append("computational")
        else:
            source_type.append("unknown")

    # non_null_count: number of non-null fields per row (for dedup tiebreak)
    nn_acc = pc.cast(pc.is_valid(table.column(ENRICHED_DATA_COLS[0])), pa.int32())
    for col in ENRICHED_DATA_COLS[1:]:
        nn_acc = pc.add(nn_acc, pc.cast(pc.is_valid(table.column(col)), pa.int32()))

    return (
        table
        .append_column("row_id", pa.array(row_ids))
        .append_column("source_type", pa.array(source_type))
        .append_column("non_null_count", nn_acc)
    )


def _dedup_table_via_duckdb(
    table: pa.Table,
    key_cols: List[str],
    use_source_priority: bool = True,
) -> pa.Table:
    """Dedup an Arrow table by key_cols, keeping the best row per group.

    Best = (lowest source priority rank if use_source_priority,
            then highest non_null_count, then lex-first row_id).
    """
    if table.num_rows == 0:
        return table
    con = duckdb.connect()
    con.execute(f"SET threads={THREADS}")
    con.register("t", table)
    exp_list_sql = ",".join(f"'{s}'" for s in EXPERIMENTAL_SOURCES)
    comp_list_sql = ",".join(f"'{s}'" for s in COMPUTATIONAL_SOURCES)
    if use_source_priority:
        prio_expr = f"""
            CASE WHEN source_db_primary IN ({exp_list_sql}) THEN 0
                 WHEN source_db_primary IN ({comp_list_sql}) THEN 1
                 WHEN source_db_primary IS NOT NULL THEN 2
                 ELSE 3 END
        """
    else:
        prio_expr = "0"
    partition_by = ", ".join(key_cols)
    where_clause = " AND ".join(f"{c} IS NOT NULL AND {c} <> ''" for c in key_cols)
    sql = f"""
    WITH ranked AS (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY {partition_by}
                   ORDER BY {prio_expr} ASC,
                            non_null_count DESC,
                            row_id ASC
               ) AS _rn
        FROM t
        WHERE {where_clause}
    )
    SELECT * EXCLUDE (_rn) FROM ranked WHERE _rn = 1
    """
    rdr = con.execute(sql)
    out = rdr.fetch_arrow_table() if hasattr(rdr, "fetch_arrow_table") else rdr.arrow()
    if isinstance(out, pa.RecordBatchReader):
        out = out.read_all()
    con.close()
    return out


# ---------------------------------------------------------------------------
# Cluster + allele partitioning helpers
# ---------------------------------------------------------------------------

def _sample_clusters(clusters: Iterable[str], frac: float, rng: random.Random) -> Set[str]:
    """Return a random fraction of cluster IDs as the train pool."""
    cl = sorted(set(clusters))
    n_train = max(1, int(round(len(cl) * frac)))
    rng.shuffle(cl)
    return set(cl[:n_train])


def _holdout_alleles(allele_counts: Dict[str, int], min_count: int, holdout_frac: float, rng: random.Random) -> Set[str]:
    """Return a set of alleles to hold out (NOT in train).

    Eligible alleles have >= min_count examples; sample holdout_frac of those as held-out.
    Alleles with < min_count remain in train (too rare to evaluate on).
    """
    eligible = [a for a, c in allele_counts.items() if c >= min_count]
    eligible.sort()
    rng.shuffle(eligible)
    n_holdout = max(1, int(round(len(eligible) * holdout_frac))) if eligible else 0
    return set(eligible[:n_holdout])


def _stratified_split(
    df_indices: List[int],
    strat_keys: List[str],
    fractions: Tuple[float, float, float],
    rng: random.Random,
) -> Tuple[List[int], List[int], List[int]]:
    """Group-shuffle split: groups defined by strat_keys, fractions sum to 1.0.

    Indices in the same group always go to the same split (prevents leakage of
    same cluster across train/val/test).
    """
    groups: Dict[str, List[int]] = {}
    for idx, key in zip(df_indices, strat_keys):
        groups.setdefault(key, []).append(idx)
    group_keys = sorted(groups.keys())
    rng.shuffle(group_keys)

    f_train, f_val, _ = fractions
    n_groups = len(group_keys)
    n_train = int(round(n_groups * f_train))
    n_val = int(round(n_groups * f_val))
    train_groups = set(group_keys[:n_train])
    val_groups = set(group_keys[n_train:n_train + n_val])

    out_train, out_val, out_test = [], [], []
    for k, indices in groups.items():
        if k in train_groups:
            out_train.extend(indices)
        elif k in val_groups:
            out_val.extend(indices)
        else:
            out_test.extend(indices)
    return out_train, out_val, out_test


def _write_partition(table: pa.Table, indices: List[int], out_path: Path) -> int:
    if len(indices) == 0:
        # write empty table with schema for consistency
        empty = table.slice(0, 0)
        pq.write_table(empty, str(out_path), compression="zstd")
        return 0
    sub = table.take(pa.array(indices, type=pa.int64()))
    pq.write_table(sub, str(out_path), compression="zstd")
    return sub.num_rows


# ---------------------------------------------------------------------------
# Task A — AS
# ---------------------------------------------------------------------------

AS_CLASS_I_BENCHMARKS = [
    ("as_trb_i", ["trb_peptide_mhc_one", "trb_peptide_mhc_one_mhc_two"], "trb_cdr3", "trb_cdr3_interaction_clusters.tsv", "mhc_one"),
    ("as_tra_i", ["tra_peptide_mhc_one", "tra_peptide_mhc_one_mhc_two"], "tra_cdr3", "tra_cdr3_interaction_clusters.tsv", "mhc_one"),
    ("as_paired_i", ["tra_trb_peptide_mhc_one", "tra_trb_peptide_mhc_one_mhc_two"], None, None, "mhc_one"),
]
AS_CLASS_II_BENCHMARKS = [
    ("as_trb_ii", ["trb_peptide_mhc_two", "trb_peptide_mhc_one_mhc_two"]),
    ("as_tra_ii", ["tra_peptide_mhc_two", "tra_peptide_mhc_one_mhc_two"]),
    ("as_paired_ii", ["tra_trb_peptide_mhc_two", "tra_trb_peptide_mhc_one_mhc_two"]),
]


def _as_class_i_one(name: str, subset_keys: List[str], tcr_col: Optional[str], cluster_tsv: Optional[str], allele_col: str) -> Dict[str, int]:
    print(f"[as] === {name} ===", flush=True)
    rng = random.Random(SEED)
    con = _duckdb_conn()
    files = _files_for(subset_keys)
    print(f"[as] {name}: {len(files)} files", flush=True)
    table = _read_enriched(con, files)
    con.close()
    table = _add_row_id(table)
    n = table.num_rows
    print(f"[as] {name}: {n:,} rows loaded", flush=True)

    # Load clusters
    pep_clusters = load_cluster_tsv(CLUSTERS_ROOT / "peptide_clusters_interaction.tsv", seq_col="peptide")
    if name == "as_paired_i":
        trb_clusters = load_cluster_tsv(CLUSTERS_ROOT / "trb_cdr3_interaction_clusters.tsv")
        tra_clusters = load_cluster_tsv(CLUSTERS_ROOT / "tra_cdr3_interaction_clusters.tsv")
    else:
        single_clusters = load_cluster_tsv(CLUSTERS_ROOT / cluster_tsv)

    pep_col = table.column("peptide").to_pylist()
    allele_vals = table.column(allele_col + "_allele").to_pylist() if (allele_col + "_allele") in table.column_names else table.column(allele_col).to_pylist()

    if name == "as_paired_i":
        tra_seqs = table.column("tra_cdr3").to_pylist()
        trb_seqs = table.column("trb_cdr3").to_pylist()
        tra_cl = [tra_clusters.get(s or "", "_unk_tra") for s in tra_seqs]
        trb_cl = [trb_clusters.get(s or "", "_unk_trb") for s in trb_seqs]
        all_tra_cl = set(c for c in tra_cl if c != "_unk_tra")
        all_trb_cl = set(c for c in trb_cl if c != "_unk_trb")
        train_tra = _sample_clusters(all_tra_cl, 0.70, rng)
        train_trb = _sample_clusters(all_trb_cl, 0.70, rng)
    else:
        tcr_seqs = table.column(tcr_col).to_pylist()
        tcr_cl = [single_clusters.get(s or "", "_unk") for s in tcr_seqs]
        all_tcr_cl = set(c for c in tcr_cl if c != "_unk")
        train_tcr = _sample_clusters(all_tcr_cl, 0.70, rng)

    pep_cl = [pep_clusters.get(p or "", "_unk_pep") for p in pep_col]
    all_pep_cl = set(c for c in pep_cl if c != "_unk_pep")
    train_pep = _sample_clusters(all_pep_cl, 0.70, rng)

    # Allele holdout: ≥50 examples eligible, 20% held out
    allele_counts: Dict[str, int] = {}
    for a in allele_vals:
        if a:
            allele_counts[a] = allele_counts.get(a, 0) + 1
    heldout_alleles = _holdout_alleles(allele_counts, min_count=50, holdout_frac=0.20, rng=rng)

    # Conditions
    conditions: List[str] = []
    pep_clusters_per_row: List[str] = pep_cl
    tcr_clusters_per_row: List[str] = []
    for i in range(n):
        if name == "as_paired_i":
            tcr_novel = (tra_cl[i] not in train_tra) or (trb_cl[i] not in train_trb)
            tcr_clusters_per_row.append(f"{tra_cl[i]}|{trb_cl[i]}")
        else:
            tcr_novel = tcr_cl[i] not in train_tcr
            tcr_clusters_per_row.append(tcr_cl[i])
        pep_novel = pep_cl[i] not in train_pep
        a = allele_vals[i]
        allele_novel = (a is not None and a in heldout_alleles)
        if not tcr_novel and not pep_novel and not allele_novel:
            conditions.append("iid")
        elif tcr_novel and not pep_novel and not allele_novel:
            conditions.append("novel_tcr")
        elif not tcr_novel and pep_novel and not allele_novel:
            conditions.append("novel_pep")
        elif not tcr_novel and not pep_novel and allele_novel:
            conditions.append("novel_allele")
        elif tcr_novel and pep_novel and allele_novel:
            conditions.append("level4")
        else:
            conditions.append("mixed")

    # Annotate with extra columns
    table = table.append_column("tcr_cluster", pa.array(tcr_clusters_per_row))
    table = table.append_column("pep_cluster", pa.array(pep_clusters_per_row))
    table = table.append_column("condition", pa.array(conditions))

    # Stratified split of IID by pep_cluster (85/7.5/7.5)
    iid_indices = [i for i, c in enumerate(conditions) if c == "iid"]
    iid_strat = [pep_clusters_per_row[i] for i in iid_indices]
    train_idx, val_idx, test_iid_idx = _stratified_split(iid_indices, iid_strat, (0.85, 0.075, 0.075), rng)

    # Other partition indices
    parts: Dict[str, List[int]] = {
        "train": train_idx,
        "val": val_idx,
        "test_iid": test_iid_idx,
        "test_novel_tcr": [i for i, c in enumerate(conditions) if c == "novel_tcr"],
        "test_novel_pep": [i for i, c in enumerate(conditions) if c == "novel_pep"],
        "test_novel_allele": [i for i, c in enumerate(conditions) if c == "novel_allele"],
        "test_level4": [i for i, c in enumerate(conditions) if c == "level4"],
        "test_mixed": [i for i, c in enumerate(conditions) if c == "mixed"],
    }

    counts: Dict[str, int] = {}
    for part, idx in parts.items():
        out = OUT_ROOT / f"{name}_{part}.parquet"
        counts[f"{name}_{part}"] = _write_partition(table, idx, out)
        print(f"[as]   wrote {out.name}: {counts[f'{name}_{part}']:,} rows", flush=True)

    return counts


def _as_class_ii_one(name: str, subset_keys: List[str]) -> Dict[str, int]:
    print(f"[as] === {name} (eval-only) ===", flush=True)
    con = _duckdb_conn()
    files = _files_for(subset_keys)
    table = _read_enriched(con, files)
    con.close()
    table = _add_row_id(table)
    out = OUT_ROOT / f"{name}_eval.parquet"
    pq.write_table(table, str(out), compression="zstd")
    print(f"[as]   wrote {out.name}: {table.num_rows:,} rows", flush=True)
    return {f"{name}_eval": table.num_rows}


def task_as() -> Dict[str, int]:
    t0 = time.time()
    counts: Dict[str, int] = {}

    # Class I
    for name, subset_keys, tcr_col, cluster_tsv, allele_col in AS_CLASS_I_BENCHMARKS:
        counts.update(_as_class_i_one(name, subset_keys, tcr_col, cluster_tsv, allele_col))

    # Class II eval-only
    for name, subset_keys in AS_CLASS_II_BENCHMARKS:
        counts.update(_as_class_ii_one(name, subset_keys))

    # Candidate pool: union of Class I peptides
    print("[as] building Class I candidate pool...", flush=True)
    con = _duckdb_conn()
    union_files: List[str] = []
    for _, sks, _, _, _ in AS_CLASS_I_BENCHMARKS:
        union_files.extend(_files_for(sks))
    files_sql = "['" + "','".join(union_files) + "']"
    pep_clusters = load_cluster_tsv(CLUSTERS_ROOT / "peptide_clusters_interaction.tsv", seq_col="peptide")
    df = con.execute(f"""
        SELECT DISTINCT peptide
        FROM read_parquet({files_sql})
        WHERE peptide IS NOT NULL AND peptide <> ''
        ORDER BY peptide
    """).df()
    con.close()
    df["peptide_cluster_id"] = df["peptide"].map(lambda p: pep_clusters.get(p, "_unk_pep"))
    pool_path = OUT_ROOT / "as_class_i_candidate_pool.tsv"
    df.to_csv(pool_path, sep="\t", index=False)
    print(f"[as] candidate pool: {len(df):,} peptides -> {pool_path.name}", flush=True)

    _record_timing("as", time.time() - t0)
    return counts


# ---------------------------------------------------------------------------
# Task B — PM
# ---------------------------------------------------------------------------

def _pm_one(name: str, mhc_col: str) -> Dict[str, int]:
    print(f"[pm] === {name} ===", flush=True)
    rng = random.Random(SEED)
    # All subset_keys with peptide and mhc_X
    subset_keys = []
    for d in _list_subset_dirs():
        skey = _subset_key_of(d)
        cls = _classify(skey)
        if cls["has_peptide"] and cls[f"has_{mhc_col}"]:
            subset_keys.append(skey)
    print(f"[pm] {name}: {len(subset_keys)} subset_keys", flush=True)

    con = _duckdb_conn()
    files = _files_for(subset_keys)
    table = _read_enriched(con, files)
    con.close()
    table = _add_row_id(table)
    n_orig = table.num_rows
    print(f"[pm] {name}: {n_orig:,} rows pre-dedup", flush=True)

    # Dedup on (peptide, mhc_X) via DuckDB SQL ROW_NUMBER (kept in C++).
    table = _dedup_table_via_duckdb(table, ["peptide", mhc_col], use_source_priority=True)
    n = table.num_rows
    print(f"[pm] {name}: {n:,} rows after dedup", flush=True)

    # Cluster maps
    pep_clusters = load_cluster_tsv(CLUSTERS_ROOT / "peptide_clusters_pm.tsv", seq_col="peptide")
    pep_col = table.column("peptide").to_pylist()
    pep_cl = [pep_clusters.get(p or "", "_unk_pep") for p in pep_col]
    all_pep_cl = set(c for c in pep_cl if c != "_unk_pep")
    train_pep = _sample_clusters(all_pep_cl, 0.80, rng)

    allele_vals = table.column(mhc_col + "_allele").to_pylist() if (mhc_col + "_allele") in table.column_names else table.column(mhc_col).to_pylist()
    allele_counts: Dict[str, int] = {}
    for a in allele_vals:
        if a:
            allele_counts[a] = allele_counts.get(a, 0) + 1
    heldout = _holdout_alleles(allele_counts, min_count=50, holdout_frac=0.20, rng=rng)

    # Conditions
    conditions: List[str] = []
    for i in range(n):
        pep_novel = pep_cl[i] not in train_pep
        a = allele_vals[i]
        allele_novel = (a is not None and a in heldout)
        if not pep_novel and not allele_novel:
            conditions.append("iid")
        elif pep_novel and not allele_novel:
            conditions.append("novel_pep")
        elif not pep_novel and allele_novel:
            conditions.append("novel_allele")
        else:
            conditions.append("level4")

    table = table.append_column("pep_cluster", pa.array(pep_cl))
    table = table.append_column("condition", pa.array(conditions))

    iid_idx = [i for i, c in enumerate(conditions) if c == "iid"]
    iid_strat = [pep_cl[i] for i in iid_idx]
    train_idx, val_idx, test_iid_idx = _stratified_split(iid_idx, iid_strat, (0.80, 0.10, 0.10), rng)

    parts: Dict[str, List[int]] = {
        "train": train_idx,
        "val": val_idx,
        "test_iid": test_iid_idx,
        "test_novel_pep": [i for i, c in enumerate(conditions) if c == "novel_pep"],
        "test_novel_allele": [i for i, c in enumerate(conditions) if c == "novel_allele"],
        "test_level4": [i for i, c in enumerate(conditions) if c == "level4"],
    }
    counts: Dict[str, int] = {}
    for part, idx in parts.items():
        out = OUT_ROOT / f"{name}_{part}.parquet"
        counts[f"{name}_{part}"] = _write_partition(table, idx, out)
        print(f"[pm]   wrote {out.name}: {counts[f'{name}_{part}']:,} rows", flush=True)

    # Candidate pool
    pool = sorted(set(pep_col))
    pool_path = OUT_ROOT / f"{name}_candidate_pool.tsv"
    with open(pool_path, "w") as f:
        f.write("peptide\tpeptide_cluster_id\n")
        for p in pool:
            if p:
                f.write(f"{p}\t{pep_clusters.get(p, '_unk_pep')}\n")
    print(f"[pm] candidate pool: {len(pool):,} peptides -> {pool_path.name}", flush=True)

    return counts


def task_pm() -> Dict[str, int]:
    t0 = time.time()
    counts: Dict[str, int] = {}
    counts.update(_pm_one("pm_i", "mhc_one"))
    counts.update(_pm_one("pm_ii", "mhc_two"))
    _record_timing("pm", time.time() - t0)
    return counts


# ---------------------------------------------------------------------------
# Task C — PAIR
# ---------------------------------------------------------------------------

def task_pair() -> Dict[str, int]:
    print("[pair] === pair ===", flush=True)
    t0 = time.time()
    rng = random.Random(SEED)
    subset_keys = [_subset_key_of(d) for d in _list_subset_dirs() if _subset_key_of(d).startswith("tra_trb")]
    print(f"[pair] {len(subset_keys)} subset_keys", flush=True)

    con = _duckdb_conn()
    files = _files_for(subset_keys)
    table = _read_enriched(con, files)
    con.close()
    table = _add_row_id(table)
    n_orig = table.num_rows
    print(f"[pair] {n_orig:,} rows pre-dedup", flush=True)

    # has_interaction = True iff subset_key contains peptide AND mhc
    sk = table.column("subset_key").to_pylist()
    has_interaction = [("peptide" in s) and ("mhc_one" in s or "mhc_two" in s) for s in sk]
    table = table.append_column("has_interaction", pa.array(has_interaction))

    # Dedup on (tra_cdr3, trb_cdr3) via DuckDB SQL ROW_NUMBER.
    table = _dedup_table_via_duckdb(table, ["tra_cdr3", "trb_cdr3"], use_source_priority=False)
    n = table.num_rows
    print(f"[pair] {n:,} rows after dedup", flush=True)

    # PAIR uses the *_all cluster TSVs because paired-only data (e.g.
    # tra_trb without peptide/MHC) has TCRs that aren't in the interaction TSV.
    print("[pair] loading tra_cdr3_all clusters...", flush=True)
    tra_clusters = load_cluster_tsv(CLUSTERS_ROOT / "tra_cdr3_all_clusters.tsv")
    print(f"[pair]   {len(tra_clusters):,} TRA cluster entries", flush=True)
    print("[pair] loading trb_cdr3_all clusters...", flush=True)
    trb_clusters = load_cluster_tsv(CLUSTERS_ROOT / "trb_cdr3_all_clusters.tsv")
    print(f"[pair]   {len(trb_clusters):,} TRB cluster entries", flush=True)

    tra_arr = table.column("tra_cdr3").to_pylist()
    trb_arr = table.column("trb_cdr3").to_pylist()
    tra_cl = [tra_clusters.get(s or "", "_unk_tra") for s in tra_arr]
    trb_cl = [trb_clusters.get(s or "", "_unk_trb") for s in trb_arr]
    all_tra_cl = set(c for c in tra_cl if c != "_unk_tra")
    all_trb_cl = set(c for c in trb_cl if c != "_unk_trb")
    train_tra = _sample_clusters(all_tra_cl, 0.80, rng)
    train_trb = _sample_clusters(all_trb_cl, 0.80, rng)

    conditions: List[str] = []
    for i in range(n):
        tra_novel = tra_cl[i] not in train_tra
        trb_novel = trb_cl[i] not in train_trb
        if not tra_novel and not trb_novel:
            conditions.append("iid")
        elif tra_novel and not trb_novel:
            conditions.append("novel_tra")
        elif not tra_novel and trb_novel:
            conditions.append("novel_trb")
        else:
            conditions.append("novel_both")

    table = table.append_column("tra_cluster", pa.array(tra_cl))
    table = table.append_column("trb_cluster", pa.array(trb_cl))
    table = table.append_column("condition", pa.array(conditions))

    iid_idx = [i for i, c in enumerate(conditions) if c == "iid"]
    iid_strat = [trb_cl[i] for i in iid_idx]
    train_idx, val_idx, test_iid_idx = _stratified_split(iid_idx, iid_strat, (0.80, 0.10, 0.10), rng)

    parts: Dict[str, List[int]] = {
        "train": train_idx,
        "val": val_idx,
        "test_iid": test_iid_idx,
        "test_novel_tra": [i for i, c in enumerate(conditions) if c == "novel_tra"],
        "test_novel_trb": [i for i, c in enumerate(conditions) if c == "novel_trb"],
        "test_novel_both": [i for i, c in enumerate(conditions) if c == "novel_both"],
    }
    counts: Dict[str, int] = {}
    for part, idx in parts.items():
        out = OUT_ROOT / f"pair_{part}.parquet"
        counts[f"pair_{part}"] = _write_partition(table, idx, out)
        print(f"[pair]   wrote {out.name}: {counts[f'pair_{part}']:,} rows", flush=True)

    # Negatives: 5 per positive train pair, label=0; positives label=1
    train_table = table.take(pa.array(train_idx, type=pa.int64()))
    n_train = train_table.num_rows
    train_tra_seqs = train_table.column("tra_cdr3").to_pylist()
    train_trb_seqs = train_table.column("trb_cdr3").to_pylist()
    positive_set = set(zip(train_tra_seqs, train_trb_seqs))

    # Build negatives
    neg_rng = random.Random(SEED + 1)
    n_neg = n_train * 5
    if n_neg > 0:
        neg_tra: List[str] = []
        neg_trb: List[str] = []
        attempts = 0
        max_attempts = n_neg * 4
        while len(neg_tra) < n_neg and attempts < max_attempts:
            attempts += 1
            i = neg_rng.randrange(n_train)
            j = neg_rng.randrange(n_train)
            if i == j:
                continue
            cand = (train_tra_seqs[i], train_trb_seqs[j])
            if cand in positive_set:
                continue
            neg_tra.append(cand[0])
            neg_trb.append(cand[1])

        neg_tbl = pa.table({"tra_cdr3": neg_tra, "trb_cdr3": neg_trb,
                            "label": [0] * len(neg_tra)})
        # Add positives with label=1
        pos_tbl = pa.table({"tra_cdr3": train_tra_seqs, "trb_cdr3": train_trb_seqs,
                            "label": [1] * n_train})
        combined = pa.concat_tables([pos_tbl, neg_tbl])
        out_neg = OUT_ROOT / "pair_negatives_train.parquet"
        pq.write_table(combined, str(out_neg), compression="zstd")
        counts["pair_negatives_train"] = combined.num_rows
        print(f"[pair]   wrote pair_negatives_train.parquet: {n_train:,} pos + {len(neg_tra):,} neg = {combined.num_rows:,} rows", flush=True)

    _record_timing("pair", time.time() - t0)
    return counts


# ---------------------------------------------------------------------------
# Task D — MR
# ---------------------------------------------------------------------------

MR_CLASS_I_BENCHMARKS = [
    ("mr_trb_i", "trb_cdr3", "mhc_one", "trb_cdr3_restriction_clusters.tsv"),
    ("mr_tra_i", "tra_cdr3", "mhc_one", "tra_cdr3_restriction_clusters.tsv"),
    ("mr_paired_i", None, "mhc_one", None),
]
MR_CLASS_II_BENCHMARKS = [
    ("mr_trb_ii", "trb_cdr3", "mhc_two", "trb_cdr3_restriction_clusters.tsv"),
    ("mr_tra_ii", "tra_cdr3", "mhc_two", "tra_cdr3_restriction_clusters.tsv"),
    ("mr_paired_ii", None, "mhc_two", None),
]


def _mr_subset_keys(mhc_col: str) -> List[str]:
    """Subset keys with (tra or trb) AND the relevant MHC class."""
    out = []
    for d in _list_subset_dirs():
        skey = _subset_key_of(d)
        cls = _classify(skey)
        if (cls["has_tra"] or cls["has_trb"]) and cls[f"has_{mhc_col}"]:
            out.append(skey)
    return out


def _mr_one_class_i(name: str, tcr_col: Optional[str], mhc_col: str, cluster_tsv: Optional[str]) -> Dict[str, int]:
    print(f"[mr] === {name} ===", flush=True)
    rng = random.Random(SEED)
    subset_keys = _mr_subset_keys(mhc_col)
    if name.startswith("mr_paired"):
        subset_keys = [s for s in subset_keys if "tra" in s and "trb" in s]
    elif tcr_col == "tra_cdr3":
        subset_keys = [s for s in subset_keys if "tra" in s]
    elif tcr_col == "trb_cdr3":
        subset_keys = [s for s in subset_keys if "trb" in s]
    print(f"[mr] {name}: {len(subset_keys)} subset_keys", flush=True)

    con = _duckdb_conn()
    files = _files_for(subset_keys)
    table = _read_enriched(con, files)
    con.close()
    table = _add_row_id(table)
    n_orig = table.num_rows
    print(f"[mr] {name}: {n_orig:,} rows pre-dedup", flush=True)

    # Dedup via DuckDB SQL.
    if name.startswith("mr_paired"):
        key_cols = ["tra_cdr3", "trb_cdr3", mhc_col + "_allele"]
        # Fall back to mhc_col itself if mhc_col + '_allele' not present
        if (mhc_col + "_allele") not in table.column_names:
            key_cols = ["tra_cdr3", "trb_cdr3", mhc_col]
    elif tcr_col == "tra_cdr3":
        key_cols = ["tra_cdr3", mhc_col + "_allele"]
        if (mhc_col + "_allele") not in table.column_names:
            key_cols = ["tra_cdr3", mhc_col]
    else:
        key_cols = ["trb_cdr3", mhc_col + "_allele"]
        if (mhc_col + "_allele") not in table.column_names:
            key_cols = ["trb_cdr3", mhc_col]
    table = _dedup_table_via_duckdb(table, key_cols, use_source_priority=False)
    n = table.num_rows
    print(f"[mr] {name}: {n:,} rows after dedup", flush=True)

    # Clusters
    if name.startswith("mr_paired"):
        tra_clusters = load_cluster_tsv(CLUSTERS_ROOT / "tra_cdr3_restriction_clusters.tsv")
        trb_clusters = load_cluster_tsv(CLUSTERS_ROOT / "trb_cdr3_restriction_clusters.tsv")
        tra_seqs = table.column("tra_cdr3").to_pylist()
        trb_seqs = table.column("trb_cdr3").to_pylist()
        tra_cl = [tra_clusters.get(s or "", "_unk_tra") for s in tra_seqs]
        trb_cl = [trb_clusters.get(s or "", "_unk_trb") for s in trb_seqs]
        all_tra_cl = set(c for c in tra_cl if c != "_unk_tra")
        all_trb_cl = set(c for c in trb_cl if c != "_unk_trb")
        train_tra = _sample_clusters(all_tra_cl, 0.70, rng)
        train_trb = _sample_clusters(all_trb_cl, 0.70, rng)
    else:
        single_clusters = load_cluster_tsv(CLUSTERS_ROOT / cluster_tsv)
        tcr_seqs = table.column(tcr_col).to_pylist()
        tcr_cl = [single_clusters.get(s or "", "_unk") for s in tcr_seqs]
        all_tcr_cl = set(c for c in tcr_cl if c != "_unk")
        train_tcr = _sample_clusters(all_tcr_cl, 0.70, rng)

    allele_vals = (table.column(mhc_col + "_allele").to_pylist()
                   if (mhc_col + "_allele") in table.column_names
                   else table.column(mhc_col).to_pylist())
    allele_counts: Dict[str, int] = {}
    for a in allele_vals:
        if a:
            allele_counts[a] = allele_counts.get(a, 0) + 1
    heldout = _holdout_alleles(allele_counts, min_count=50, holdout_frac=0.20, rng=rng)

    conditions: List[str] = []
    tcr_clusters_per_row: List[str] = []
    for i in range(n):
        if name.startswith("mr_paired"):
            tcr_novel = (tra_cl[i] not in train_tra) or (trb_cl[i] not in train_trb)
            tcr_clusters_per_row.append(f"{tra_cl[i]}|{trb_cl[i]}")
        else:
            tcr_novel = tcr_cl[i] not in train_tcr
            tcr_clusters_per_row.append(tcr_cl[i])
        a = allele_vals[i]
        allele_novel = (a is not None and a in heldout)
        if not tcr_novel and not allele_novel:
            conditions.append("iid")
        elif tcr_novel and not allele_novel:
            conditions.append("novel_tcr")
        elif not tcr_novel and allele_novel:
            conditions.append("novel_allele")
        else:
            conditions.append("level4")

    table = table.append_column("tcr_cluster", pa.array(tcr_clusters_per_row))
    table = table.append_column("condition", pa.array(conditions))

    iid_idx = [i for i, c in enumerate(conditions) if c == "iid"]
    iid_strat = tcr_clusters_per_row if name.startswith("mr_paired") else (
        [tcr_cl[i] for i in iid_idx]
    )
    if name.startswith("mr_paired"):
        iid_strat = [tcr_clusters_per_row[i] for i in iid_idx]
    train_idx, val_idx, test_iid_idx = _stratified_split(iid_idx, iid_strat, (0.70, 0.15, 0.15), rng)

    parts = {
        "train": train_idx, "val": val_idx, "test_iid": test_iid_idx,
        "test_novel_tcr": [i for i, c in enumerate(conditions) if c == "novel_tcr"],
        "test_novel_allele": [i for i, c in enumerate(conditions) if c == "novel_allele"],
        "test_level4": [i for i, c in enumerate(conditions) if c == "level4"],
    }
    counts: Dict[str, int] = {}
    for part, idx in parts.items():
        out = OUT_ROOT / f"{name}_{part}.parquet"
        counts[f"{name}_{part}"] = _write_partition(table, idx, out)
        print(f"[mr]   wrote {out.name}: {counts[f'{name}_{part}']:,} rows", flush=True)
    return counts


def _mr_one_class_ii(name: str, tcr_col: Optional[str], mhc_col: str, cluster_tsv: Optional[str]) -> Dict[str, int]:
    print(f"[mr] === {name} ===", flush=True)
    subset_keys = _mr_subset_keys(mhc_col)
    if name.startswith("mr_paired"):
        subset_keys = [s for s in subset_keys if "tra" in s and "trb" in s]
    elif tcr_col == "tra_cdr3":
        subset_keys = [s for s in subset_keys if "tra" in s]
    elif tcr_col == "trb_cdr3":
        subset_keys = [s for s in subset_keys if "trb" in s]
    con = _duckdb_conn()
    files = _files_for(subset_keys)
    table = _read_enriched(con, files)
    con.close()
    table = _add_row_id(table)
    n = table.num_rows
    if n == 0:
        out = OUT_ROOT / f"{name}_eval.parquet"
        pq.write_table(table, str(out), compression="zstd")
        return {f"{name}_eval": 0}

    # For TRB-II: try IID + Novel-Allele if eligible
    if name == "mr_trb_ii":
        allele_vals = (table.column(mhc_col + "_allele").to_pylist()
                       if (mhc_col + "_allele") in table.column_names
                       else table.column(mhc_col).to_pylist())
        allele_counts: Dict[str, int] = {}
        for a in allele_vals:
            if a:
                allele_counts[a] = allele_counts.get(a, 0) + 1
        eligible_alleles = [a for a, c in allele_counts.items() if c >= 20]
        if len(eligible_alleles) >= 5:
            # Build minimal IID + novel_allele split
            rng = random.Random(SEED)
            eligible_alleles.sort()
            rng.shuffle(eligible_alleles)
            n_holdout = max(1, int(round(len(eligible_alleles) * 0.20)))
            heldout = set(eligible_alleles[:n_holdout])
            iid_idx = [i for i, a in enumerate(allele_vals) if a is not None and a not in heldout]
            novel_idx = [i for i, a in enumerate(allele_vals) if a is not None and a in heldout]
            counts: Dict[str, int] = {}
            for part, idx in [("test_iid", iid_idx), ("test_novel_allele", novel_idx)]:
                out = OUT_ROOT / f"{name}_{part}.parquet"
                counts[f"{name}_{part}"] = _write_partition(table, idx, out)
                print(f"[mr]   wrote {out.name}: {counts[f'{name}_{part}']:,} rows", flush=True)
            return counts

    out = OUT_ROOT / f"{name}_eval.parquet"
    pq.write_table(table, str(out), compression="zstd")
    print(f"[mr]   wrote {out.name}: {n:,} rows (eval-only)", flush=True)
    return {f"{name}_eval": n}


def task_mr() -> Dict[str, int]:
    t0 = time.time()
    counts: Dict[str, int] = {}
    for name, tcr_col, mhc_col, cluster_tsv in MR_CLASS_I_BENCHMARKS:
        counts.update(_mr_one_class_i(name, tcr_col, mhc_col, cluster_tsv))
    for name, tcr_col, mhc_col, cluster_tsv in MR_CLASS_II_BENCHMARKS:
        counts.update(_mr_one_class_ii(name, tcr_col, mhc_col, cluster_tsv))

    # Candidate pool: unique mhc_one_allele across Class I MR
    print("[mr] building Class I candidate pool...", flush=True)
    union_subset_keys = _mr_subset_keys("mhc_one")
    con = _duckdb_conn()
    files = _files_for(union_subset_keys)
    files_sql = "['" + "','".join(files) + "']"
    df = con.execute(f"""
        SELECT DISTINCT
            COALESCE(l.mhc_one_allele, e.mhc_one) AS allele
        FROM read_parquet({files_sql}) e
        LEFT JOIN read_parquet('{LOOKUP_PARQUET}') l
        ON COALESCE(e.tra_full, '')=COALESCE(l.tra_full, '')
       AND COALESCE(e.trb_full, '')=COALESCE(l.trb_full, '')
       AND COALESCE(e.peptide, '')=COALESCE(l.peptide, '')
       AND COALESCE(e.mhc_one, '')=COALESCE(l.mhc_one, '')
       AND COALESCE(e.mhc_two, '')=COALESCE(l.mhc_two, '')
        WHERE COALESCE(e.mhc_one, '') <> ''
        ORDER BY allele
    """).df()
    con.close()
    pool_path = OUT_ROOT / "mr_class_i_candidate_pool.tsv"
    df.to_csv(pool_path, sep="\t", index=False)
    print(f"[mr] candidate pool: {len(df):,} alleles -> {pool_path.name}", flush=True)

    _record_timing("mr", time.time() - t0)
    return counts


# ---------------------------------------------------------------------------
# Leakage validation
# ---------------------------------------------------------------------------

def _read_col(path: Path, col: str) -> List[str]:
    if not path.exists():
        return []
    t = pq.read_table(str(path), columns=[col])
    return [v for v in t.column(col).to_pylist() if v is not None]


def _novel_pep_check(train_path: Path, test_path: Path, max_pairs: int = 100_000_000) -> Dict[str, object]:
    """Pairwise rapidfuzz Levenshtein between test peptides and train peptides.

    For very large partitions (e.g. PM-I novel_pep with ~200K train × ~800K test
    = 160B comparisons), sample at most ``max_pairs`` (test, train) comparisons
    per partition to keep wall-clock under ~10 min while still catching gross
    violations. Sampling is deterministic via random.Random(SEED).
    """
    from rapidfuzz.distance import Levenshtein
    train_peps = sorted(set(_read_col(train_path, "peptide")))
    test_peps = sorted(set(_read_col(test_path, "peptide")))
    if not train_peps or not test_peps:
        return {"status": "PASS", "n_train": len(train_peps), "n_test": len(test_peps),
                "violations": [], "closest": None, "note": "empty side"}

    total_pairs = len(train_peps) * len(test_peps)
    n_test_total = len(test_peps)
    needs_sampling = total_pairs > max_pairs

    if needs_sampling:
        target_test_n = max(1, max_pairs // max(1, len(train_peps)))
        rng = random.Random(SEED)
        test_peps = sorted(rng.sample(test_peps, k=min(target_test_n, len(test_peps))))

    violations: List[Tuple[str, str, int, int]] = []

    if len(test_peps) <= 16:
        # Small case: in-process
        for t in test_peps:
            lt = len(t)
            for tr in train_peps:
                ltr = len(tr)
                min_len = min(lt, ltr)
                thr = math.ceil(0.2 * min_len)
                if abs(lt - ltr) > thr:
                    continue
                d = Levenshtein.distance(t, tr, score_cutoff=thr)
                if d <= thr:
                    violations.append((t, tr, d, thr))
    else:
        chunks: List[List[str]] = []
        chunk_size = max(1, len(test_peps) // THREADS + 1)
        for i in range(0, len(test_peps), chunk_size):
            chunks.append(test_peps[i:i + chunk_size])
        with Pool(processes=THREADS) as pool:
            for res in pool.imap_unordered(_LeakWorker(train_peps), chunks):
                violations.extend(res)

    closest = None
    if violations:
        for v in violations:
            if closest is None or v[2] < closest[2]:
                closest = v

    return {
        "status": "FAIL" if violations else "PASS",
        "n_train": len(train_peps),
        "n_test_total": n_test_total,
        "n_test_sampled": len(test_peps),
        "sampled": needs_sampling,
        "violations": [{"test": v[0], "train": v[1], "dist": v[2], "threshold": v[3]} for v in violations[:20]],
        "n_violations": len(violations),
        "closest": ({"test": closest[0], "train": closest[1], "dist": closest[2], "threshold": closest[3]} if closest else None),
    }


class _LeakWorker:
    def __init__(self, train_peps: List[str]):
        self.train_peps = train_peps

    def __call__(self, test_chunk: List[str]) -> List[Tuple[str, str, int, int]]:
        from rapidfuzz.distance import Levenshtein
        out = []
        for t in test_chunk:
            lt = len(t)
            for tr in self.train_peps:
                ltr = len(tr)
                min_len = min(lt, ltr)
                thr = math.ceil(0.2 * min_len)
                if abs(lt - ltr) > thr:
                    continue
                d = Levenshtein.distance(t, tr, score_cutoff=thr)
                if d <= thr:
                    out.append((t, tr, d, thr))
        return out


def _set_disjoint_check(train_path: Path, test_path: Path, col: str) -> Dict[str, object]:
    train = set(_read_col(train_path, col))
    test = set(_read_col(test_path, col))
    overlap = train & test
    return {
        "status": "PASS" if not overlap else "FAIL",
        "n_train": len(train),
        "n_test": len(test),
        "n_overlap": len(overlap),
        "sample_overlap": sorted(overlap)[:10],
    }


def _cluster_disjoint_check(train_path: Path, test_path: Path) -> Dict[str, object]:
    return _set_disjoint_check(train_path, test_path, "tcr_cluster")


def task_leakage() -> None:
    print("[leakage] running checks...", flush=True)
    t0 = time.time()
    results: List[Dict[str, object]] = []

    paths = sorted(OUT_ROOT.glob("*.parquet"))
    by_prefix: Dict[str, Dict[str, Path]] = {}
    for p in paths:
        # Strip .parquet, find the "_<partition>" suffix
        stem = p.stem
        for suffix in [
            "_train", "_val",
            "_test_iid", "_test_novel_pep", "_test_novel_tcr", "_test_novel_allele",
            "_test_novel_tra", "_test_novel_trb", "_test_novel_both",
            "_test_level4", "_test_mixed", "_eval",
        ]:
            if stem.endswith(suffix):
                prefix = stem[: -len(suffix)]
                by_prefix.setdefault(prefix, {})[suffix.lstrip("_")] = p
                break

    for prefix, parts in sorted(by_prefix.items()):
        if "train" not in parts:
            continue
        train_path = parts["train"]
        # novel_pep
        if "test_novel_pep" in parts:
            r = _novel_pep_check(train_path, parts["test_novel_pep"])
            r.update({"check": "novel_pep", "partition": f"{prefix}_test_novel_pep"})
            print(f"[leakage] {prefix}_novel_pep: {r['status']} ({r['n_violations']} violations)", flush=True)
            results.append(r)
        # novel_tcr (clusters disjoint)
        if "test_novel_tcr" in parts:
            r = _cluster_disjoint_check(train_path, parts["test_novel_tcr"])
            r.update({"check": "novel_tcr", "partition": f"{prefix}_test_novel_tcr"})
            print(f"[leakage] {prefix}_novel_tcr: {r['status']} (overlap={r['n_overlap']})", flush=True)
            results.append(r)
        # novel_tra / novel_trb (PAIR)
        for s, c in [("test_novel_tra", "tra_cluster"), ("test_novel_trb", "trb_cluster"), ("test_novel_both", None)]:
            if s in parts:
                if c is None:
                    # novel_both: both tra_cluster and trb_cluster disjoint
                    a = _set_disjoint_check(train_path, parts[s], "tra_cluster")
                    b = _set_disjoint_check(train_path, parts[s], "trb_cluster")
                    r = {
                        "check": "novel_both",
                        "partition": f"{prefix}_{s}",
                        "status": "PASS" if (a["status"] == "PASS" and b["status"] == "PASS") else "FAIL",
                        "tra_cluster_overlap": a["n_overlap"],
                        "trb_cluster_overlap": b["n_overlap"],
                    }
                else:
                    r = _set_disjoint_check(train_path, parts[s], c)
                    r.update({"check": s.replace("test_", ""), "partition": f"{prefix}_{s}"})
                print(f"[leakage] {prefix}_{s}: {r['status']}", flush=True)
                results.append(r)
        # novel_allele
        if "test_novel_allele" in parts:
            # Pick the right allele column from the benchmark name:
            # *_i / *_i_* benchmarks use mhc_one_allele; *_ii / *_ii_* use mhc_two_allele.
            if prefix.endswith("_ii") or "_ii_" in prefix or prefix == "pm_ii":
                check_col = "mhc_two_allele"
            else:
                check_col = "mhc_one_allele"
            r = _set_disjoint_check(train_path, parts["test_novel_allele"], check_col)
            r.update({"check": "novel_allele", "partition": f"{prefix}_test_novel_allele", "allele_col": check_col})
            print(f"[leakage] {prefix}_novel_allele: {r['status']} (overlap={r['n_overlap']})", flush=True)
            results.append(r)
        # level4: AS = all three; PM = pep+allele; MR = tcr+allele
        if "test_level4" in parts:
            tbl_schema = pq.read_schema(str(parts["test_level4"]))
            colnames = list(tbl_schema.names)
            sub_results: Dict[str, object] = {"check": "level4", "partition": f"{prefix}_test_level4"}
            if "tcr_cluster" in colnames:
                sub_results["tcr_disjoint"] = _set_disjoint_check(train_path, parts["test_level4"], "tcr_cluster")
            if "pep_cluster" in colnames and prefix.startswith(("as_", "pm_")):
                sub_results["pep_disjoint"] = _set_disjoint_check(train_path, parts["test_level4"], "pep_cluster")
            if prefix.endswith("_ii") or "_ii_" in prefix or prefix == "pm_ii":
                allele_col = "mhc_two_allele"
            else:
                allele_col = "mhc_one_allele"
            if allele_col in colnames:
                sub_results["allele_disjoint"] = _set_disjoint_check(train_path, parts["test_level4"], allele_col)
            ok = all((v.get("status") == "PASS") for k, v in sub_results.items() if isinstance(v, dict))
            sub_results["status"] = "PASS" if ok else "FAIL"
            print(f"[leakage] {prefix}_level4: {sub_results['status']}", flush=True)
            results.append(sub_results)

    LEAKAGE_PATH.write_text(json.dumps(results, indent=2))
    _record_timing("leakage", time.time() - t0)
    fails = [r for r in results if r.get("status") != "PASS"]
    print(f"[leakage] total: {len(results)} checks, {len(fails)} FAIL", flush=True)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def task_summary() -> None:
    print("[summary] generating...", flush=True)
    t0 = time.time()
    rows: List[Dict[str, object]] = []
    for p in sorted(OUT_ROOT.glob("*.parquet")):
        meta = pq.read_metadata(str(p))
        rows.append({"file": p.name, "rows": meta.num_rows})

    leakage = []
    if LEAKAGE_PATH.exists():
        leakage = json.loads(LEAKAGE_PATH.read_text())
    timings = {}
    if TIMINGS_PATH.exists():
        timings = json.loads(TIMINGS_PATH.read_text())
    lookup_stats = {}
    if LOOKUP_STATS.exists():
        lookup_stats = json.loads(LOOKUP_STATS.read_text())

    summary = {
        "partition_files": rows,
        "leakage_validation": leakage,
        "timings_seconds": timings,
        "lookup_stats": lookup_stats,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"[summary] wrote {SUMMARY_PATH}", flush=True)
    print()
    print(f"{'file':<55} {'rows':>12}")
    print("-" * 70)
    for r in rows:
        print(f"{r['file']:<55} {r['rows']:>12,}")
    _record_timing("summary", time.time() - t0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", required=True,
        choices=["prep_lookup", "as", "pm", "pair", "mr", "leakage", "summary", "all"],
    )
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOOKUP_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _set_seed()

    if args.task == "prep_lookup":
        task_prep_lookup()
    elif args.task == "as":
        task_as()
    elif args.task == "pm":
        task_pm()
    elif args.task == "pair":
        task_pair()
    elif args.task == "mr":
        task_mr()
    elif args.task == "leakage":
        task_leakage()
    elif args.task == "summary":
        task_summary()
    elif args.task == "all":
        task_prep_lookup()
        task_as()
        task_pm()
        task_pair()
        task_mr()
        task_leakage()
        task_summary()


if __name__ == "__main__":
    sys.exit(main())
