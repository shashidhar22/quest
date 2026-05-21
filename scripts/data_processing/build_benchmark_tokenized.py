"""Tokenize benchmark splits and foundation manifests for ESM-2 / ESM-C training.

ESM-2 and ESM-C share an identical AA vocabulary (positions 4-30) so a single
tokenized output works for both models. See `docs/BENCHMARK_TOKENIZED.md`.

Usage:
    python build_benchmark_tokenized.py --task {benchmark,foundation,mini,summary,all}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SOURCE_ROOT = Path("/home/ubuntu/quest/data/deduplicated_again/exploded_deduped_enriched")
SPLITS_ROOT = Path("/home/ubuntu/quest/data/benchmark_v2/splits")
LOOKUP_PARQUET = SPLITS_ROOT / "lookup" / "sequence_to_source_allele.parquet"
FOUNDATION_ROOT = Path("/home/ubuntu/quest/data/benchmark_v2/foundation")
PARTITION_ASSIGNMENTS = FOUNDATION_ROOT / "partition_assignments.parquet"
OUT_ROOT = Path("/home/ubuntu/quest/data/benchmark_v2/tokenized")
LOG_DIR = OUT_ROOT / "logs"
TIMINGS_PATH = OUT_ROOT / "timings.json"
SUMMARY_PATH = OUT_ROOT / "tokenization_summary.json"
SCRATCH = Path("/scratch")

THREADS = 64
SHARD_ROWS = 5_000_000  # rows per shard for foundation_500M

# Reduced settings for very large sharded manifests (e.g. foundation_500M).
# The parent process holds ~100 GB of Python objects just representing the
# manifest (`rows` list + `by_file` dict) plus in-flight Arrow tables from
# workers. With THREADS=64 and SHARD_ROWS=5M this OOM'd at ~354 GB RSS.
# Halving threads and shard size keeps peak parent memory well under 1 TB.
THREADS_LARGE = 16
SHARD_ROWS_LARGE = 2_000_000

# Full enriched-stage molecule columns (matches Phase 2/3 schema). These are
# preserved on every tokenized output, regardless of which subset the format
# happens to read for `input_sequence` construction. This list does NOT include
# `subset_key` / `order_key` (handled separately as metadata).
ALL_ENRICHED_COLS: List[str] = [
    "tra_full", "trb_full", "peptide", "mhc_one", "mhc_two",
    "tra_cdr1", "tra_cdr2", "tra_cdr3",
    "trb_cdr1", "trb_cdr2", "trb_cdr3",
    "sequence",
    "mhc_one_pocket", "mhc_one_contact", "mhc_one_pocket_contact",
    "mhc_two_pocket", "mhc_two_contact", "mhc_two_pocket_contact",
]

# ---------------------------------------------------------------------------
# Tokenizer (ESM-2; equivalent to ESM-C for AA-only sequences)
# ---------------------------------------------------------------------------

_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    return _TOKENIZER


# ---------------------------------------------------------------------------
# Format & order parsing
# ---------------------------------------------------------------------------

FORMAT_MOLECULE_COLS: Dict[str, Dict[str, List[str]]] = {
    "C1": {"tra": ["tra_cdr3"], "trb": ["trb_cdr3"], "peptide": ["peptide"],
           "mhc_one": ["mhc_one"], "mhc_two": ["mhc_two"]},
    "C2": {"tra": ["tra_cdr3"], "trb": ["trb_cdr3"], "peptide": ["peptide"],
           "mhc_one": ["mhc_one_pocket_contact"], "mhc_two": ["mhc_two_pocket_contact"]},
    "C3": {"tra": ["tra_cdr1", "tra_cdr2", "tra_cdr3"],
           "trb": ["trb_cdr1", "trb_cdr2", "trb_cdr3"],
           "peptide": ["peptide"],
           "mhc_one": ["mhc_one_pocket_contact"], "mhc_two": ["mhc_two_pocket_contact"]},
    "C4": {"tra": ["tra_cdr1", "tra_cdr2", "tra_cdr3"],
           "trb": ["trb_cdr1", "trb_cdr2", "trb_cdr3"],
           "peptide": ["peptide"],
           "mhc_one": ["mhc_one_contact"], "mhc_two": ["mhc_two_contact"]},
    "C5": {"tra": ["tra_full"], "trb": ["trb_full"], "peptide": ["peptide"],
           "mhc_one": ["mhc_one_pocket_contact"], "mhc_two": ["mhc_two_pocket_contact"]},
    "M1": {"peptide": ["peptide"], "mhc_one": ["mhc_one"], "mhc_two": ["mhc_two"]},
    "M2": {"peptide": ["peptide"], "mhc_one": ["mhc_one_pocket_contact"],
           "mhc_two": ["mhc_two_pocket_contact"]},
    "M3": {"peptide": ["peptide"], "mhc_one": ["mhc_one_contact"],
           "mhc_two": ["mhc_two_contact"]},
    "T1": {"tra": ["tra_cdr3"], "trb": ["trb_cdr3"]},
    "T2": {"tra": ["tra_cdr1", "tra_cdr2", "tra_cdr3"],
           "trb": ["trb_cdr1", "trb_cdr2", "trb_cdr3"]},
    "T3": {"tra": ["tra_full"], "trb": ["trb_full"]},
}


def parse_order(order_key: str) -> List[str]:
    """E.g. 'tra_trb_peptide_mhc_one_mhc_two' -> ['tra','trb','peptide','mhc_one','mhc_two']"""
    parts = order_key.split("_")
    out: List[str] = []
    i = 0
    while i < len(parts):
        if parts[i] == "mhc" and i + 1 < len(parts):
            out.append(f"mhc_{parts[i+1]}")
            i += 2
        else:
            out.append(parts[i])
            i += 1
    return out


def all_format_cols(format_cell: str) -> List[str]:
    """Union of all data columns referenced by this format."""
    cols: List[str] = []
    for v in FORMAT_MOLECULE_COLS[format_cell].values():
        cols.extend(v)
    return list(dict.fromkeys(cols))


def build_input_sequence(
    row_dict: Dict[str, Optional[str]],
    order_key: str,
    format_cell: str,
    sep: Optional[str] = None,
) -> str:
    """Build the AA string for a single row.

    `row_dict` is a column-name -> value dict for this row.
    `sep`: insert this string between non-empty molecule strings (None = direct concat).
    """
    fmt = FORMAT_MOLECULE_COLS[format_cell]
    parts: List[str] = []
    for mol in parse_order(order_key):
        cols = fmt.get(mol)
        if cols is None:
            continue  # molecule not in this format's allowlist (e.g., PM drops TCR)
        seq = "".join((row_dict.get(c) or "") for c in cols)
        if seq:
            parts.append(seq)
    if not parts:
        return ""
    if sep is not None:
        return sep.join(parts)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Job specification
# ---------------------------------------------------------------------------

AS_CLASS_I_PARTITIONS = ["train", "val", "test_iid", "test_novel_tcr",
                         "test_novel_pep", "test_novel_allele",
                         "test_level4", "test_mixed"]
AS_CLASS_I_BENCHMARKS = ["as_trb_i", "as_tra_i", "as_paired_i"]
AS_CLASS_I_FORMATS = ["C1", "C2", "C3", "C4", "C5"]
AS_CLASS_II_BENCHMARKS = ["as_trb_ii", "as_tra_ii", "as_paired_ii"]

PM_BENCHMARKS = ["pm_i", "pm_ii"]
PM_FORMATS = ["M1", "M2", "M3"]
PM_PARTITIONS = ["train", "val", "test_iid", "test_novel_pep",
                 "test_novel_allele", "test_level4"]

PAIR_FORMATS = ["T1", "T2", "T3"]
PAIR_PARTITIONS = ["train", "val", "test_iid", "test_novel_tra",
                   "test_novel_trb", "test_novel_both"]

MR_CLASS_I_BENCHMARKS = ["mr_trb_i", "mr_tra_i", "mr_paired_i"]
MR_CLASS_I_FORMATS = ["C1", "C2", "C3", "C4", "C5"]
MR_CLASS_I_PARTITIONS = ["train", "val", "test_iid", "test_novel_tcr",
                         "test_novel_allele", "test_level4"]


def benchmark_jobs() -> List[Tuple[str, str, str, str]]:
    """Return list of (task, benchmark, format, partition) tuples."""
    jobs: List[Tuple[str, str, str, str]] = []
    # AS Class I
    for bm in AS_CLASS_I_BENCHMARKS:
        for fmt in AS_CLASS_I_FORMATS:
            for part in AS_CLASS_I_PARTITIONS:
                jobs.append(("as", bm, fmt, part))
    # AS Class II eval
    for bm in AS_CLASS_II_BENCHMARKS:
        jobs.append(("as", bm, "C3", "eval"))
    # PM
    for bm in PM_BENCHMARKS:
        for fmt in PM_FORMATS:
            for part in PM_PARTITIONS:
                jobs.append(("pm", bm, fmt, part))
    # PAIR
    for fmt in PAIR_FORMATS:
        for part in PAIR_PARTITIONS:
            jobs.append(("pair", "pair", fmt, part))
    # MR Class I
    for bm in MR_CLASS_I_BENCHMARKS:
        for fmt in MR_CLASS_I_FORMATS:
            for part in MR_CLASS_I_PARTITIONS:
                jobs.append(("mr", bm, fmt, part))
    # MR Class II
    for part in ["test_iid", "test_novel_allele"]:
        jobs.append(("mr", "mr_trb_ii", "C3", part))
    for bm in ["mr_tra_ii", "mr_paired_ii"]:
        jobs.append(("mr", bm, "C3", "eval"))
    return jobs


# ---------------------------------------------------------------------------
# Shared helpers
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


def _row_dicts_from_table(table: pa.Table, cols: List[str]) -> List[Dict[str, Optional[str]]]:
    """Extract a list of column-name -> value dicts from an Arrow Table.

    Only includes the requested columns. Faster than building per-row dicts in pure Python.
    """
    arrays = {c: table.column(c).to_pylist() for c in cols if c in table.column_names}
    n = table.num_rows
    out = [{c: arrays[c][i] for c in arrays} for i in range(n)]
    return out


def _tokenize_sequences(sequences: List[str]) -> List[List[int]]:
    tok = _get_tokenizer()
    # batch_encode_plus is faster than per-row encode
    enc = tok(sequences, add_special_tokens=True, padding=False, truncation=False, return_attention_mask=False)
    return enc["input_ids"]


def _stream_writer(out_path: Path, schema: pa.Schema) -> pq.ParquetWriter:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return pq.ParquetWriter(str(out_path), schema, compression="zstd")


# ---------------------------------------------------------------------------
# Step 1 — Tokenize benchmark splits
# ---------------------------------------------------------------------------

def _tokenize_one_benchmark(args: Tuple[str, str, str, str]) -> Tuple[str, int, str]:
    """Tokenize one (task, benchmark, format, partition) job.

    Returns (output_path, n_rows, status).
    """
    task, bm, fmt, part = args
    in_path = SPLITS_ROOT / f"{bm}_{part}.parquet"
    if not in_path.exists():
        return (str(in_path), 0, "missing_input")
    out_path = OUT_ROOT / task / f"{bm}_{fmt}_{part}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 100:
        return (str(out_path), 0, "skipped_existing")

    table = pq.read_table(str(in_path))
    n = table.num_rows
    if n == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.append_column("input_sequence", pa.array([], type=pa.string()))
                            .append_column("input_ids", pa.array([], type=pa.list_(pa.int32())))
                            .append_column("format_cell", pa.array([], type=pa.string())),
                       str(out_path), compression="zstd")
        return (str(out_path), 0, "empty")

    data_cols_needed = list(set(all_format_cols(fmt) + ["order_key"]))
    row_dicts = _row_dicts_from_table(table, data_cols_needed)
    sequences = [build_input_sequence(d, d["order_key"], fmt) for d in row_dicts]
    input_ids = _tokenize_sequences(sequences)

    # Build output table = input table + input_sequence + input_ids + format_cell
    out_table = table
    out_table = out_table.append_column("input_sequence", pa.array(sequences, type=pa.string()))
    out_table = out_table.append_column("input_ids", pa.array(input_ids, type=pa.list_(pa.int32())))
    out_table = out_table.append_column("format_cell", pa.array([fmt] * n, type=pa.string()))
    # Rename source_db_primary -> source_db if present
    if "source_db_primary" in out_table.column_names and "source_db" not in out_table.column_names:
        idx = out_table.schema.get_field_index("source_db_primary")
        f = out_table.schema.field(idx)
        out_table = out_table.set_column(idx, pa.field("source_db", f.type), out_table.column(idx))

    pq.write_table(out_table, str(out_path), compression="zstd")
    return (str(out_path), n, "ok")


def task_benchmark() -> None:
    print("[benchmark] tokenizing benchmark splits...", flush=True)
    t0 = time.time()
    jobs = benchmark_jobs()
    print(f"[benchmark] {len(jobs)} jobs", flush=True)

    # Parallelize. Each worker loads its own tokenizer instance.
    completed = 0
    n_total = 0
    with ProcessPoolExecutor(max_workers=THREADS) as ex:
        futures = {ex.submit(_tokenize_one_benchmark, j): j for j in jobs}
        for fut in as_completed(futures):
            try:
                out_path, n, status = fut.result()
            except Exception as e:
                j = futures[fut]
                print(f"[benchmark] FAIL {j}: {e}", flush=True)
                continue
            completed += 1
            n_total += n
            if completed % 25 == 0 or completed == len(jobs):
                print(f"[benchmark]   {completed}/{len(jobs)} done, {n_total:,} rows total", flush=True)
            if status not in ("ok", "skipped_existing", "empty"):
                print(f"[benchmark]   note {status}: {out_path}", flush=True)

    _record_timing("benchmark", time.time() - t0)
    print(f"[benchmark] done in {time.time()-t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Step 2 — Tokenize foundation val/test (full data already present)
# ---------------------------------------------------------------------------

def _tokenize_full_batch_worker(args):
    """Worker: take (sequences, fmt) and return input_ids list."""
    sequences, _fmt = args
    tok = _get_tokenizer()
    enc = tok(sequences, add_special_tokens=True, padding=False, truncation=False, return_attention_mask=False)
    return enc["input_ids"]


def _tokenize_full_data_file(in_path: Path, out_path: Path, fmt: str = "C3", join_lookup: bool = True) -> int:
    """Tokenize a full-data parquet file with format `fmt`.

    Pipelines: read batches → build sequences (cheap) → tokenize batches in
    parallel via ProcessPool (expensive, parallelizable) → write output.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 100:
        meta = pq.read_metadata(str(out_path))
        return meta.num_rows

    pq_file = pq.ParquetFile(str(in_path))
    data_cols_needed = list(set(all_format_cols(fmt) + ["order_key", "subset_key"]))

    sample_batch = next(pq_file.iter_batches(batch_size=10))
    sample_table = pa.Table.from_batches([sample_batch])
    base_schema = sample_table.schema

    new_fields = [
        pa.field("input_sequence", pa.string()),
        pa.field("input_ids", pa.list_(pa.int32())),
        pa.field("format_cell", pa.string()),
    ]
    out_schema = pa.schema(list(base_schema) + new_fields)

    BATCH = 100_000
    n_total = 0
    t0 = time.time()
    with pq.ParquetWriter(str(out_path), out_schema, compression="zstd") as writer:
        # Stream batches; for each, build sequences then dispatch tokenization to a pool.
        # Use a pool with an initializer so workers preload the tokenizer once each.
        with Pool(processes=THREADS, initializer=_get_tokenizer) as pool:
            # Read all batches into a list of (table, sequences) pairs eagerly per chunk
            # to feed the pool. We process N batches at a time.
            batch_iter = pq_file.iter_batches(batch_size=BATCH)
            BATCHES_PER_ROUND = THREADS  # one batch per worker per round
            done = False
            while not done:
                round_batches = []
                round_sequences = []
                for _ in range(BATCHES_PER_ROUND):
                    try:
                        batch = next(batch_iter)
                    except StopIteration:
                        done = True
                        break
                    table = pa.Table.from_batches([batch])
                    row_dicts = _row_dicts_from_table(table, data_cols_needed)
                    sequences = [build_input_sequence(d, d["order_key"], fmt) for d in row_dicts]
                    round_batches.append(table)
                    round_sequences.append(sequences)
                if not round_batches:
                    break

                # Tokenize all batches in parallel
                tok_args = [(seqs, fmt) for seqs in round_sequences]
                tok_results = pool.map(_tokenize_full_batch_worker, tok_args)

                # Write each batch
                for table, sequences, input_ids in zip(round_batches, round_sequences, tok_results):
                    n = table.num_rows
                    arrays = list(table.columns) + [
                        pa.array(sequences, type=pa.string()),
                        pa.array(input_ids, type=pa.list_(pa.int32())),
                        pa.array([fmt] * n, type=pa.string()),
                    ]
                    out_table = pa.Table.from_arrays(arrays, names=[f.name for f in out_schema])
                    writer.write_table(out_table)
                    n_total += n

                if n_total > 0:
                    rate = n_total / max(0.1, time.time() - t0)
                    print(f"[foundation:{out_path.name}]   {n_total:,} rows ({rate:.0f}/s)", flush=True)
    return n_total


def task_foundation_val_test() -> None:
    """Step 3: tokenize foundation_val.parquet and foundation_test.parquet."""
    t0 = time.time()
    print("[foundation:val_test] tokenizing val + test...", flush=True)
    fmt = "C3"

    # Re-join with lookup to attach source_db. We do this via a DuckDB query
    # that reads the full-data file, joins lookup, and writes a temp file
    # with source_db_primary attached. Then tokenize from the temp file.
    # (Simpler: just attach source_db via post-read column; lookup parquet is small.)
    for in_name, out_name in [
        ("foundation_val.parquet", "foundation_val_C3.parquet"),
        ("foundation_test.parquet", "foundation_test_C3.parquet"),
    ]:
        in_path = FOUNDATION_ROOT / in_name
        out_path = OUT_ROOT / "foundation" / out_name
        if not in_path.exists():
            print(f"[foundation:val_test] missing {in_path}", flush=True)
            continue
        n = _tokenize_full_data_file(in_path, out_path, fmt=fmt)
        print(f"[foundation:val_test] {out_name}: {n:,} rows", flush=True)

    _record_timing("foundation_val_test", time.time() - t0)


# ---------------------------------------------------------------------------
# Step 3 — Tokenize foundation manifests (10M, 100M, 500M)
# ---------------------------------------------------------------------------

def _process_source_file_for_manifest(args: Tuple[str, List[Tuple[int, str, str]], str, bool]) -> pa.Table:
    """Process one source_file's worth of rows from a manifest.

    args = (source_file, list_of_(source_row_index, bio_hash, order_key), format, use_sep)

    Returns Arrow table with: input_sequence, input_ids, format_cell, order_key,
    subset_key, bio_hash, source_file, source_row_index. Plus the original
    molecule columns.
    """
    source_file, rows_info, fmt, use_sep = args
    if not rows_info:
        return None
    indices = sorted({r[0] for r in rows_info})

    # Read ALL enriched molecule columns from the source file, not just the
    # format-referenced ones. The format function still decides which columns
    # feed `input_sequence`, but the output table preserves every molecule
    # column for downstream consumers (e.g. switching format later, debugging).
    cols_for_fmt = all_format_cols(fmt)  # used by build_input_sequence
    cols_to_read = list(dict.fromkeys(ALL_ENRICHED_COLS + ["subset_key"]))
    con = duckdb.connect()
    con.execute("SET threads=4")
    indices_str = "(" + ",".join(str(i) for i in indices) + ")"
    cols_sql = ", ".join(cols_to_read)
    rdr = con.execute(f"""
        SELECT {cols_sql}, (ROW_NUMBER() OVER () - 1) AS _ridx
        FROM read_parquet('{source_file}')
        QUALIFY _ridx IN {indices_str}
    """)
    df = rdr.fetch_arrow_table() if hasattr(rdr, "fetch_arrow_table") else rdr.arrow()
    if isinstance(df, pa.RecordBatchReader):
        df = df.read_all()
    con.close()

    # Build a per-index lookup dict
    df_ridx = df.column("_ridx").to_pylist()
    by_idx: Dict[int, Dict[str, Optional[str]]] = {}
    cols_in_table = [c for c in cols_to_read if c in df.column_names]
    for i, ridx in enumerate(df_ridx):
        d = {c: df.column(c)[i].as_py() for c in cols_in_table}
        by_idx[ridx] = d

    # Build output rows. Preserve ALL_ENRICHED_COLS, not just format-referenced.
    out_seq, out_order, out_subset, out_bio, out_file, out_idx = [], [], [], [], [], []
    out_data: Dict[str, List] = {c: [] for c in ALL_ENRICHED_COLS}
    sep = "<eos>" if use_sep else None
    sequences_to_tokenize: List[str] = []
    for ridx, bio_hash, order_key in rows_info:
        d = by_idx.get(ridx)
        if d is None:
            continue
        seq = build_input_sequence(d, order_key, fmt, sep=sep)
        sequences_to_tokenize.append(seq)
        out_seq.append(seq)
        out_order.append(order_key)
        out_subset.append(d.get("subset_key"))
        out_bio.append(bio_hash)
        out_file.append(source_file)
        out_idx.append(ridx)
        for c in ALL_ENRICHED_COLS:
            out_data[c].append(d.get(c))

    if not out_seq:
        return None
    input_ids = _tokenize_sequences(sequences_to_tokenize)

    arrays = {
        "bio_hash": pa.array(out_bio),
        "order_key": pa.array(out_order),
        "subset_key": pa.array(out_subset),
        "source_file": pa.array(out_file),
        "source_row_index": pa.array(out_idx, type=pa.int64()),
        "input_sequence": pa.array(out_seq),
        "input_ids": pa.array(input_ids, type=pa.list_(pa.int32())),
        "format_cell": pa.array([fmt] * len(out_seq)),
    }
    for c in ALL_ENRICHED_COLS:
        arrays[c] = pa.array(out_data[c])
    return pa.table(arrays)


def _build_canonical_schema(fmt: str) -> pa.Schema:
    """Schema for foundation manifest tokenized outputs.

    Carries ALL enriched molecule columns (matching foundation_val/test
    schema), not just format-referenced ones, so downstream consumers can
    switch format at training time without re-resolving source rows.
    """
    fields = [
        ("bio_hash", pa.string()),
        ("order_key", pa.string()),
        ("subset_key", pa.string()),
        ("source_file", pa.string()),
        ("source_row_index", pa.int64()),
        ("input_sequence", pa.string()),
        ("input_ids", pa.list_(pa.int32())),
        ("format_cell", pa.string()),
    ]
    for c in ALL_ENRICHED_COLS:
        fields.append((c, pa.string()))
    return pa.schema(fields)


def _tokenize_manifest(
    manifest_path: Path,
    out_path: Path,
    fmt: str,
    use_sep: bool,
    sharded: bool = False,
    shard_dir: Optional[Path] = None,
    shard_size: int = SHARD_ROWS,
    threads: int = THREADS,
) -> int:
    """Tokenize a sample manifest by joining back to source parquet files."""
    t0 = time.time()

    # Shard-level resume: scan existing shards for source_files already done.
    # Each worker returns one source_file's worth of rows in one Arrow table,
    # so a source_file appearing in any completed shard is fully processed.
    completed_files: set = set()
    n_already_written = 0
    starting_shard_idx = 0
    if sharded:
        shard_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(shard_dir.glob("shard_*.parquet"))
        if existing:
            print(f"[manifest] found {len(existing)} existing shards — scanning for resume", flush=True)
            paths_sql = "[" + ",".join(f"'{p}'" for p in existing) + "]"
            con0 = duckdb.connect()
            done = con0.execute(
                f"SELECT DISTINCT source_file FROM read_parquet({paths_sql})"
            ).fetchall()
            completed_files.update(r[0] for r in done)
            n_already_written = con0.execute(
                f"SELECT COUNT(*) FROM read_parquet({paths_sql})"
            ).fetchone()[0]
            con0.close()
            last_idx = max(int(p.stem.removeprefix("shard_")) for p in existing)
            starting_shard_idx = last_idx + 1
            print(f"[manifest] resume: {len(completed_files)} source files done, "
                  f"{n_already_written:,} rows already written, "
                  f"next shard = shard_{starting_shard_idx:04d}", flush=True)

    print(f"[manifest] reading {manifest_path.name} ...", flush=True)
    con = duckdb.connect()
    con.execute("SET threads=64")
    rdr = con.execute(f"""
        SELECT source_file, source_row_index, bio_hash, order_key
        FROM read_parquet('{manifest_path}')
        ORDER BY source_file, source_row_index
    """)
    df = rdr.fetch_arrow_table() if hasattr(rdr, "fetch_arrow_table") else rdr.arrow()
    if isinstance(df, pa.RecordBatchReader):
        df = df.read_all()
    con.close()

    rows = list(zip(
        df.column("source_file").to_pylist(),
        df.column("source_row_index").to_pylist(),
        df.column("bio_hash").to_pylist(),
        df.column("order_key").to_pylist(),
    ))
    print(f"[manifest] {len(rows):,} rows across {len(set(r[0] for r in rows))} source files in {time.time()-t0:.1f}s", flush=True)

    by_file: Dict[str, List[Tuple[int, str, str]]] = {}
    for src, idx, bh, ok in rows:
        if src in completed_files:
            continue
        by_file.setdefault(src, []).append((idx, bh, ok))
    # Free the manifest tuple list — by_file owns what we still need.
    del rows, df

    args_list = [(f, info, fmt, use_sep) for f, info in by_file.items()]
    if completed_files:
        print(f"[manifest] {len(args_list)} source files remaining after resume filter", flush=True)
    schema = _build_canonical_schema(fmt)

    n_written = n_already_written
    if sharded:
        shard_idx = starting_shard_idx
        shard_buffer: List[pa.Table] = []
        shard_rows = 0
        with Pool(processes=threads) as pool:
            for tbl in pool.imap_unordered(_process_source_file_for_manifest, args_list, chunksize=1):
                if tbl is None or tbl.num_rows == 0:
                    continue
                tbl = _cast_to_schema(tbl, schema)
                shard_buffer.append(tbl)
                shard_rows += tbl.num_rows
                n_written += tbl.num_rows
                if shard_rows >= shard_size:
                    out_shard = shard_dir / f"shard_{shard_idx:04d}.parquet"
                    pq.write_table(pa.concat_tables(shard_buffer), str(out_shard), compression="zstd")
                    print(f"[manifest]   wrote {out_shard.name}: {shard_rows:,} rows (cumulative {n_written:,})", flush=True)
                    shard_idx += 1
                    shard_buffer = []
                    shard_rows = 0
        if shard_buffer:
            out_shard = shard_dir / f"shard_{shard_idx:04d}.parquet"
            pq.write_table(pa.concat_tables(shard_buffer), str(out_shard), compression="zstd")
            print(f"[manifest]   wrote {out_shard.name}: {shard_rows:,} rows (final, cumulative {n_written:,})", flush=True)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with pq.ParquetWriter(str(out_path), schema, compression="zstd") as writer:
            with Pool(processes=threads) as pool:
                for tbl in pool.imap_unordered(_process_source_file_for_manifest, args_list, chunksize=4):
                    if tbl is None or tbl.num_rows == 0:
                        continue
                    tbl = _cast_to_schema(tbl, schema)
                    writer.write_table(tbl)
                    n_written += tbl.num_rows
                    if n_written % 1_000_000 == 0:
                        print(f"[manifest]   {n_written:,} rows so far", flush=True)
    return n_written


def _cast_to_schema(tbl: pa.Table, schema: pa.Schema) -> pa.Table:
    arrays = []
    for f in schema:
        if f.name in tbl.column_names:
            col = tbl.column(f.name)
            if col.type != f.type:
                col = col.cast(f.type, safe=False)
            arrays.append(col)
        else:
            arrays.append(pa.nulls(tbl.num_rows, type=f.type))
    return pa.table(arrays, names=[f.name for f in schema])


def task_foundation_manifests() -> None:
    """Tokenize foundation_10M, foundation_100M, foundation_500M."""
    t0 = time.time()
    fmt = "C3"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for size, name, sharded in [
        ("10M", "foundation_10M.parquet", False),
        ("100M", "foundation_100M.parquet", False),
        ("500M", "foundation_500M.parquet", True),
    ]:
        manifest_path = FOUNDATION_ROOT / name
        if not manifest_path.exists():
            print(f"[foundation:{size}] missing {manifest_path}", flush=True)
            continue
        if sharded:
            shard_dir = OUT_ROOT / "foundation" / f"foundation_{size}_C3"
            done_marker = shard_dir / "_COMPLETE"
            if done_marker.exists():
                print(f"[foundation:{size}] skip — _COMPLETE marker present at {shard_dir}", flush=True)
                continue
            print(f"[foundation:{size}] starting (threads={THREADS_LARGE}, shard_size={SHARD_ROWS_LARGE:,})...", flush=True)
            n = _tokenize_manifest(manifest_path, None, fmt, use_sep=False,
                                   sharded=True, shard_dir=shard_dir,
                                   shard_size=SHARD_ROWS_LARGE,
                                   threads=THREADS_LARGE)
            done_marker.write_text(f"rows={n}\n")
        else:
            out_path = OUT_ROOT / "foundation" / f"foundation_{size}_C3.parquet"
            if out_path.exists() and out_path.stat().st_size > 100:
                print(f"[foundation:{size}] skip — {out_path.name} already exists", flush=True)
                continue
            print(f"[foundation:{size}] starting...", flush=True)
            n = _tokenize_manifest(manifest_path, out_path, fmt, use_sep=False)
        print(f"[foundation:{size}] done — {n:,} rows", flush=True)
    _record_timing("foundation_manifests", time.time() - t0)


# ---------------------------------------------------------------------------
# Step 4 — Mini ablations
# ---------------------------------------------------------------------------

def task_mini() -> None:
    """Tokenize mini_M2_10M, mini_M3_10M, mini_O1_10M, mini_O3_10M."""
    t0 = time.time()
    fmt = "C3"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name, out_name, use_sep in [
        ("mini_M2_10M.parquet", "mini_M2_10M_C3.parquet", False),
        ("mini_M3_10M.parquet", "mini_M3_10M_C3.parquet", False),
        ("mini_O1_10M.parquet", "mini_O1_10M_C3.parquet", False),
        ("mini_O3_10M.parquet", "mini_O3_10M_C3_sep.parquet", True),
    ]:
        manifest_path = FOUNDATION_ROOT / name
        if not manifest_path.exists():
            print(f"[mini] missing {manifest_path}", flush=True)
            continue
        out_path = OUT_ROOT / "mini" / out_name
        if out_path.exists() and out_path.stat().st_size > 100:
            print(f"[mini] skip — {out_name} already exists", flush=True)
            continue
        print(f"[mini] {out_name} ...", flush=True)
        n = _tokenize_manifest(manifest_path, out_path, fmt, use_sep=use_sep)
        print(f"[mini] {out_name}: {n:,} rows", flush=True)
    _record_timing("mini", time.time() - t0)


# ---------------------------------------------------------------------------
# Step 5 — Summary
# ---------------------------------------------------------------------------

def task_summary() -> None:
    t0 = time.time()
    summary: Dict[str, object] = {"tasks": {}, "format_lengths": {}}
    file_count = 0
    total_rows = 0
    total_size = 0

    files_by_task: Dict[str, List[Path]] = {}
    for p in OUT_ROOT.rglob("*.parquet"):
        rel = p.relative_to(OUT_ROOT)
        task_name = rel.parts[0] if len(rel.parts) > 1 else "root"
        files_by_task.setdefault(task_name, []).append(p)
        total_size += p.stat().st_size
        file_count += 1

    print(f"[summary] {file_count} files across {len(files_by_task)} tasks", flush=True)
    for task_name, files in sorted(files_by_task.items()):
        n_rows_task = 0
        formats: Dict[str, int] = {}
        for p in files:
            try:
                meta = pq.read_metadata(str(p))
                n_rows_task += meta.num_rows
            except Exception:
                continue
            # Format from filename
            for fmt_id in ["C1", "C2", "C3", "C4", "C5", "M1", "M2", "M3", "T1", "T2", "T3"]:
                if f"_{fmt_id}_" in p.name or p.name.endswith(f"_{fmt_id}.parquet") or f"_{fmt_id}.parquet" in p.name:
                    formats[fmt_id] = formats.get(fmt_id, 0) + 1
                    break
        summary["tasks"][task_name] = {"file_count": len(files), "rows": n_rows_task, "formats": formats}
        total_rows += n_rows_task

    # Format-level length distributions: sample one file per format
    for fmt_id in ["C1", "C2", "C3", "C4", "C5", "M1", "M2", "M3", "T1", "T2", "T3"]:
        for files in files_by_task.values():
            sample = next((p for p in files if f"_{fmt_id}_" in p.name), None)
            if sample:
                try:
                    tbl = pq.read_table(str(sample), columns=["input_ids"])
                    lens = np.array([len(x) for x in tbl.column("input_ids").to_pylist()])
                    if len(lens) > 0:
                        summary["format_lengths"][fmt_id] = {
                            "min": int(lens.min()),
                            "median": int(np.median(lens)),
                            "mean": float(lens.mean()),
                            "max": int(lens.max()),
                            "p95": int(np.percentile(lens, 95)),
                            "sample_file": str(sample),
                        }
                except Exception:
                    pass
                break

    summary["total_files"] = file_count
    summary["total_rows"] = total_rows
    summary["total_size_gb"] = round(total_size / 1e9, 2)
    if TIMINGS_PATH.exists():
        summary["timings_seconds"] = json.loads(TIMINGS_PATH.read_text())

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[summary] wrote {SUMMARY_PATH}", flush=True)
    print()
    print(f"{'task':<15} {'files':>8} {'rows':>15}  formats")
    print("-" * 80)
    for task_name, info in summary["tasks"].items():
        print(f"{task_name:<15} {info['file_count']:>8} {info['rows']:>15,}  {info['formats']}")
    print(f"\nTotal: {file_count} files, {total_rows:,} rows, {summary['total_size_gb']} GB")
    if "format_lengths" in summary:
        print()
        for fmt_id, stats in summary["format_lengths"].items():
            print(f"  {fmt_id}: min={stats['min']}, median={stats['median']}, mean={stats['mean']:.1f}, p95={stats['p95']}, max={stats['max']}")
    _record_timing("summary", time.time() - t0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", required=True,
        choices=["benchmark", "foundation_val_test", "foundation_manifests",
                 "mini", "summary", "all"],
    )
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.task == "benchmark":
        task_benchmark()
    elif args.task == "foundation_val_test":
        task_foundation_val_test()
    elif args.task == "foundation_manifests":
        task_foundation_manifests()
    elif args.task == "mini":
        task_mini()
    elif args.task == "summary":
        task_summary()
    elif args.task == "all":
        task_benchmark()
        task_foundation_val_test()
        task_foundation_manifests()
        task_mini()
        task_summary()


if __name__ == "__main__":
    sys.exit(main())
