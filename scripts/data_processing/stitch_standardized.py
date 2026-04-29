#!/usr/bin/env python3
"""Post-process standardized parquet files to add full-length TCR sequences.

Streaming pipeline optimized for very large databases (billions of rows):

  For each batch of parquet files:
    1. Scan batch files → find combos NOT yet in cache
    2. Parallel stitch only the new combos
    3. Apply growing cache → write output files

This streaming approach keeps memory bounded: only the cache dict (one entry
per globally unique combo) plus the current batch's data are in memory.

Usage:
    # Stitch all databases (60 workers, in-place)
    python scripts/data_processing/stitch_standardized.py

    # Stitch specific database(s)
    python scripts/data_processing/stitch_standardized.py --db batman vdjdb

    # Custom worker count, batch size, and output dir
    python scripts/data_processing/stitch_standardized.py --workers 32 \
        --file-batch-size 32 --output-dir /data/standardized_stitched

    # Dry run (report unique combo counts without stitching)
    python scripts/data_processing/stitch_standardized.py --dry-run
"""

import argparse
import logging
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)

# Columns needed for stitching
TRA_COLS = ["tra", "trav_gene", "traj_gene"]
TRB_COLS = ["trb", "trbv_gene", "trbj_gene"]
SCAN_COLS = TRA_COLS + TRB_COLS

# Module-level globals for apply workers (inherited via fork, copy-on-write)
_SHARED_TRA_CACHE: dict = {}
_SHARED_TRB_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Scan: extract unique combos from a single parquet file
# ---------------------------------------------------------------------------

def _scan_file(file_path: str) -> tuple[set, set]:
    """Worker: read one parquet file, return unique TRA and TRB combo sets."""
    table = pq.read_table(file_path, columns=SCAN_COLS)
    df = table.to_pandas()

    tra_set = set()
    trb_set = set()

    mask_a = (df["tra"] != "") & (df["trav_gene"] != "") & (df["traj_gene"] != "")
    if mask_a.any():
        tra_set = set(df.loc[mask_a, TRA_COLS].itertuples(index=False, name=None))

    mask_b = (df["trb"] != "") & (df["trbv_gene"] != "") & (df["trbj_gene"] != "")
    if mask_b.any():
        trb_set = set(df.loc[mask_b, TRB_COLS].itertuples(index=False, name=None))

    return tra_set, trb_set


def scan_batch(
    file_paths: list[str],
    existing_tra_keys: set,
    existing_trb_keys: set,
    workers: int,
) -> tuple[set, set]:
    """Scan a batch of files and return only NEW unique combos not in cache."""
    batch_tra: set = set()
    batch_trb: set = set()

    n_workers = min(workers, len(file_paths))

    if n_workers <= 1:
        for f in file_paths:
            tra_set, trb_set = _scan_file(f)
            batch_tra.update(tra_set)
            batch_trb.update(trb_set)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_scan_file, f) for f in file_paths]
            for future in as_completed(futures):
                tra_set, trb_set = future.result()
                batch_tra.update(tra_set)
                batch_trb.update(trb_set)

    # Filter to only new combos
    new_tra = batch_tra - existing_tra_keys
    new_trb = batch_trb - existing_trb_keys

    return new_tra, new_trb


# ---------------------------------------------------------------------------
# Stitch: parallel stitching of unique combos
# ---------------------------------------------------------------------------

def _stitch_batch(args: tuple) -> dict:
    """Worker: init own TCRStitcher, stitch a batch, return results dict."""
    combos, chain, species = args

    import logging as _logging
    _logging.getLogger("quest.parsers.tcr_stitcher").setLevel(_logging.WARNING)
    from quest.parsers.tcr_stitcher import TCRStitcher
    stitcher = TCRStitcher(species=species)

    results = {}
    for cdr3, v_gene, j_gene in combos:
        result = stitcher.stitch_tcr(cdr3, v_gene, j_gene, chain)
        results[(cdr3, v_gene, j_gene)] = result if result else ""

    return results


def parallel_stitch(
    unique_combos: set,
    chain: str,
    workers: int,
    species: str = "HUMAN",
) -> dict:
    """Stitch unique combos using ProcessPoolExecutor. Returns results dict."""
    if not unique_combos:
        return {}

    combo_list = list(unique_combos)
    n_total = len(combo_list)

    # At least 50 combos per worker to amortize process overhead
    n_workers = min(workers, max(1, n_total // 50))
    chunk_size = max(1, (n_total + n_workers - 1) // n_workers)
    chunks = []
    for i in range(0, n_total, chunk_size):
        chunks.append((combo_list[i : i + chunk_size], chain, species))

    cache: dict = {}
    t0 = time.time()

    if len(chunks) == 1:
        cache = _stitch_batch(chunks[0])
    else:
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            futures = [pool.submit(_stitch_batch, chunk) for chunk in chunks]
            for i, future in enumerate(as_completed(futures)):
                cache.update(future.result())
                if (i + 1) % 10 == 0 or (i + 1) == len(futures):
                    elapsed = time.time() - t0
                    rate = len(cache) / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"    {chain}: {len(cache):,}/{n_total:,} done "
                        f"({rate:,.0f}/sec)"
                    )

    elapsed = time.time() - t0
    n_success = sum(1 for v in cache.values() if v)
    rate = n_total / elapsed if elapsed > 0 else 0
    logger.info(
        f"    {chain}: {n_success:,}/{n_total:,} stitched "
        f"({rate:,.0f}/sec, {elapsed:.1f}s)"
    )

    # Free the combo list immediately
    del combo_list, unique_combos
    return cache


# ---------------------------------------------------------------------------
# Apply: look up cached results and write parquet files
# ---------------------------------------------------------------------------

def _apply_cache_to_file(args: tuple) -> dict:
    """Worker: read parquet, apply cache lookups from module globals, write output."""
    input_path, output_path = args

    table = pq.read_table(input_path)
    df = table.to_pandas()
    n_rows = len(df)
    tra_filled = 0
    trb_filled = 0

    if _SHARED_TRA_CACHE:
        mask_a = (df["tra"] != "") & (df["trav_gene"] != "") & (df["traj_gene"] != "")
        if mask_a.any():
            keys = list(zip(
                df.loc[mask_a, "tra"],
                df.loc[mask_a, "trav_gene"],
                df.loc[mask_a, "traj_gene"],
            ))
            lookups = [_SHARED_TRA_CACHE.get(k, "") for k in keys]
            df.loc[mask_a, "tra_full"] = lookups
            tra_filled = sum(1 for v in lookups if v)

    if _SHARED_TRB_CACHE:
        mask_b = (df["trb"] != "") & (df["trbv_gene"] != "") & (df["trbj_gene"] != "")
        if mask_b.any():
            keys = list(zip(
                df.loc[mask_b, "trb"],
                df.loc[mask_b, "trbv_gene"],
                df.loc[mask_b, "trbj_gene"],
            ))
            lookups = [_SHARED_TRB_CACHE.get(k, "") for k in keys]
            df.loc[mask_b, "trb_full"] = lookups
            trb_filled = sum(1 for v in lookups if v)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(out_table, output_path, compression="snappy")

    return {"rows": n_rows, "tra_filled": tra_filled, "trb_filled": trb_filled}


def apply_to_batch(
    file_tasks: list[tuple[str, str]],
    tra_cache: dict,
    trb_cache: dict,
    workers: int,
) -> dict:
    """Apply caches to a batch of files using fork-inherited globals."""
    global _SHARED_TRA_CACHE, _SHARED_TRB_CACHE
    _SHARED_TRA_CACHE = tra_cache
    _SHARED_TRB_CACHE = trb_cache

    stats = {"rows": 0, "tra_filled": 0, "trb_filled": 0, "files": 0}
    n_workers = min(workers, len(file_tasks), 16)

    if n_workers <= 1:
        for task in file_tasks:
            r = _apply_cache_to_file(task)
            stats["rows"] += r["rows"]
            stats["tra_filled"] += r["tra_filled"]
            stats["trb_filled"] += r["trb_filled"]
            stats["files"] += 1
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_apply_cache_to_file, t) for t in file_tasks]
            for future in as_completed(futures):
                r = future.result()
                stats["rows"] += r["rows"]
                stats["tra_filled"] += r["tra_filled"]
                stats["trb_filled"] += r["trb_filled"]
                stats["files"] += 1

    return stats


# ---------------------------------------------------------------------------
# Database-level orchestration — streaming batch approach
# ---------------------------------------------------------------------------

def process_database(
    db_name: str,
    input_dir: Path,
    output_dir: Path,
    workers: int,
    file_batch_size: int = 32,
    species: str = "HUMAN",
    dry_run: bool = False,
    in_place: bool = False,
):
    """Process one database using streaming batches.

    For each batch of file_batch_size parquet files:
      1. Scan → find NEW unique combos not yet cached
      2. Stitch only the new combos (parallel)
      3. Apply full cache to batch files → write output
    """
    db_input = input_dir / db_name
    db_output = db_input if in_place else output_dir / db_name

    parquet_files = sorted(db_input.glob("part_*.parquet"))
    if not parquet_files:
        logger.warning(f"No parquet files found in {db_input}")
        return None

    n_files = len(parquet_files)
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing {db_name}: {n_files} file(s)")
    logger.info(f"{'='*70}")
    t0 = time.time()

    # Caches grow across batches — only new combos are stitched each round
    tra_cache: dict = {}
    trb_cache: dict = {}

    total_stats = {
        "rows": 0, "tra_filled": 0, "trb_filled": 0, "files_done": 0,
    }

    for batch_start in range(0, n_files, file_batch_size):
        batch_files = parquet_files[batch_start : batch_start + file_batch_size]
        batch_num = batch_start // file_batch_size + 1
        n_batches = (n_files + file_batch_size - 1) // file_batch_size
        batch_file_strs = [str(f) for f in batch_files]

        logger.info(
            f"\n  Batch {batch_num}/{n_batches} "
            f"({len(batch_files)} files, "
            f"cache: TRA={len(tra_cache):,} TRB={len(trb_cache):,})"
        )

        # --- Step 1: Scan for new combos ---
        t1 = time.time()
        new_tra, new_trb = scan_batch(
            batch_file_strs,
            set(tra_cache.keys()),
            set(trb_cache.keys()),
            workers,
        )
        scan_time = time.time() - t1
        logger.info(
            f"    Scan ({scan_time:.1f}s): "
            f"{len(new_tra):,} new TRA, {len(new_trb):,} new TRB"
        )

        if dry_run:
            # In dry-run, still accumulate keys to track global unique count
            for k in new_tra:
                tra_cache[k] = None
            for k in new_trb:
                trb_cache[k] = None
            total_stats["files_done"] += len(batch_files)
            continue

        # --- Step 2: Stitch only new combos ---
        if new_tra or new_trb:
            t2 = time.time()
            if new_tra:
                new_tra_results = parallel_stitch(new_tra, "TRA", workers, species)
                tra_cache.update(new_tra_results)
                del new_tra_results
            del new_tra

            if new_trb:
                new_trb_results = parallel_stitch(new_trb, "TRB", workers, species)
                trb_cache.update(new_trb_results)
                del new_trb_results
            del new_trb

            stitch_time = time.time() - t2
            logger.info(f"    Stitch ({stitch_time:.1f}s)")
        else:
            logger.info("    No new combos to stitch")

        # --- Step 3: Apply cache to batch files ---
        t3 = time.time()
        file_tasks = []
        for pf in batch_files:
            out_path = str(pf) if in_place else str(db_output / pf.name)
            file_tasks.append((str(pf), out_path))

        batch_stats = apply_to_batch(file_tasks, tra_cache, trb_cache, workers)
        apply_time = time.time() - t3

        total_stats["rows"] += batch_stats["rows"]
        total_stats["tra_filled"] += batch_stats["tra_filled"]
        total_stats["trb_filled"] += batch_stats["trb_filled"]
        total_stats["files_done"] += batch_stats["files"]

        logger.info(
            f"    Apply ({apply_time:.1f}s): "
            f"{batch_stats['rows']:,} rows, "
            f"TRA={batch_stats['tra_filled']:,} TRB={batch_stats['trb_filled']:,} filled"
        )

    # Clear shared globals
    global _SHARED_TRA_CACHE, _SHARED_TRB_CACHE
    _SHARED_TRA_CACHE = {}
    _SHARED_TRB_CACHE = {}

    elapsed = time.time() - t0
    tra_success = sum(1 for v in tra_cache.values() if v) if not dry_run else 0
    trb_success = sum(1 for v in trb_cache.values() if v) if not dry_run else 0

    logger.info(
        f"\n  {db_name} SUMMARY ({elapsed:.1f}s):\n"
        f"    Files:      {total_stats['files_done']:>12,}\n"
        f"    Rows:       {total_stats['rows']:>12,}\n"
        f"    Cache TRA:  {len(tra_cache):>12,} unique "
        f"({tra_success:,} stitched)\n"
        f"    Cache TRB:  {len(trb_cache):>12,} unique "
        f"({trb_success:,} stitched)\n"
        f"    Filled TRA: {total_stats['tra_filled']:>12,} rows\n"
        f"    Filled TRB: {total_stats['trb_filled']:>12,} rows"
    )

    result = {
        "total_rows": total_stats["rows"],
        "unique_tra": len(tra_cache),
        "unique_trb": len(trb_cache),
        "tra_stitched": tra_success,
        "trb_stitched": trb_success,
        "tra_filled": total_stats["tra_filled"],
        "trb_filled": total_stats["trb_filled"],
    }

    # Free memory before next database
    del tra_cache, trb_cache

    return result


def main():
    multiprocessing.set_start_method("fork", force=True)

    parser = argparse.ArgumentParser(
        description="Stitch full-length TCR sequences in standardized parquet files"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("/data/standardized_again"),
        help="Input directory with per-database subdirectories of parquet files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: overwrite in place)",
    )
    parser.add_argument(
        "--db", nargs="+", default=None,
        help="Process only these database(s). Default: all.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report unique combo counts without stitching or writing.",
    )
    parser.add_argument(
        "--workers", type=int, default=60,
        help="Number of parallel workers (default: 60)",
    )
    parser.add_argument(
        "--file-batch-size", type=int, default=32,
        help="Number of parquet files to process per batch (default: 32). "
             "Lower values use less memory but may re-stitch fewer combos.",
    )
    parser.add_argument(
        "--species", default="HUMAN",
        help="Species for stitchr (default: HUMAN)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    in_place = args.output_dir is None
    output_dir = args.input_dir if in_place else args.output_dir

    if in_place and not args.dry_run:
        logger.info("Writing in place (overwriting input files)")

    if not args.dry_run:
        from quest.parsers.tcr_stitcher import TCRStitcher
        logger.info("Verifying TCR stitcher...")
        test_stitcher = TCRStitcher(species=args.species)
        if not test_stitcher.enabled:
            logger.error("Stitcher failed to initialize. Is stitchr installed?")
            sys.exit(1)
        del test_stitcher
        logger.info("Stitcher OK.")

    if args.db:
        databases = args.db
    else:
        databases = sorted(
            d.name for d in args.input_dir.iterdir() if d.is_dir()
        )

    logger.info(f"Databases: {databases}")
    logger.info(f"Workers: {args.workers}, File batch size: {args.file_batch_size}")

    grand = {
        "total_rows": 0, "tra_stitched": 0, "trb_stitched": 0,
        "tra_filled": 0, "trb_filled": 0,
    }
    t_start = time.time()

    for db in databases:
        db_stats = process_database(
            db, args.input_dir, output_dir, args.workers,
            args.file_batch_size, args.species, args.dry_run, in_place,
        )
        if db_stats:
            grand["total_rows"] += db_stats.get("total_rows", 0)
            grand["tra_stitched"] += db_stats.get("tra_stitched", 0)
            grand["trb_stitched"] += db_stats.get("trb_stitched", 0)
            grand["tra_filled"] += db_stats.get("tra_filled", 0)
            grand["trb_filled"] += db_stats.get("trb_filled", 0)

    elapsed = time.time() - t_start
    logger.info(f"\n{'='*70}")
    logger.info(f"ALL DATABASES COMPLETE ({elapsed:.1f}s)")
    logger.info(f"{'='*70}")
    logger.info(
        f"  Total rows:   {grand['total_rows']:>14,}\n"
        f"  TRA stitched: {grand['tra_stitched']:>14,} unique, "
        f"{grand['tra_filled']:>14,} rows filled\n"
        f"  TRB stitched: {grand['trb_stitched']:>14,} unique, "
        f"{grand['trb_filled']:>14,} rows filled"
    )


if __name__ == "__main__":
    main()
