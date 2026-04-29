"""Fix tra_full / trb_full columns where they wrongly contain just the CDR3.

If the full-length chain couldn't be constructed (V/J genes absent), the full
column should be empty — not the CDR3. This script rewrites enriched parquet
files in place, setting tra_full / trb_full to empty string when:

  1. It exactly equals the corresponding CDR3, OR
  2. It is shorter than 80 aa (can't possibly contain V + CDR3 + J regions).

A real mature TCR chain is ~240-270 aa; the data has a clean bimodal split
between <50 aa (just CDR3) and >=200 aa (real full chain).

Uses vectorised pyarrow.compute so large parquet files don't blow up memory
via .to_pylist(). Worker count is deliberately conservative because some
files contain hundreds of millions of rows and each worker materialises a
table.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

ROOT = Path("/home/ubuntu/quest/data/deduplicated_again/exploded_deduped_enriched")
MIN_FULL_LEN = 80
MAX_WORKERS = 16  # Big files + string columns; 60 workers OOM'd the 1 TB box.


def _clean_column(
    full_col: pa.ChunkedArray, cdr3_col: pa.ChunkedArray
) -> tuple[pa.ChunkedArray, int]:
    """Return chunked array where junk full sequences are replaced by ''.

    Uses only pyarrow.compute kernels (no Python materialisation).
    """
    # Mask of "should be replaced" = (full == cdr3) OR (length(full) < 80)
    eq_cdr3 = pc.equal(full_col, cdr3_col).fill_null(False)
    too_short = pc.less(pc.utf8_length(full_col), MIN_FULL_LEN).fill_null(False)
    junk_mask = pc.or_(eq_cdr3, too_short)

    # Count fixed cells
    fixed = pc.sum(junk_mask).as_py() or 0

    # Replace junk cells with "", preserve nulls otherwise
    empty = pa.scalar("", type=pa.string())
    cleaned = pc.if_else(junk_mask, empty, full_col)
    return cleaned, fixed


def _process_file(path_str: str) -> tuple[str, int, int, int, str]:
    path = Path(path_str)
    try:
        table = pq.read_table(str(path))
    except Exception as exc:  # pragma: no cover
        return path_str, 0, 0, 0, f"read_error: {exc}"

    n_rows = table.num_rows
    tra_fixed = 0
    trb_fixed = 0

    if "tra_full" in table.column_names and "tra_cdr3" in table.column_names:
        new_col, tra_fixed = _clean_column(table["tra_full"], table["tra_cdr3"])
        idx = table.schema.get_field_index("tra_full")
        table = table.set_column(idx, "tra_full", new_col)

    if "trb_full" in table.column_names and "trb_cdr3" in table.column_names:
        new_col, trb_fixed = _clean_column(table["trb_full"], table["trb_cdr3"])
        idx = table.schema.get_field_index("trb_full")
        table = table.set_column(idx, "trb_full", new_col)

    tmp = path.with_suffix(".tmp.parquet")
    try:
        pq.write_table(
            table,
            tmp,
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
        )
        os.replace(tmp, path)
    except Exception as exc:  # pragma: no cover
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        return path_str, n_rows, tra_fixed, trb_fixed, f"write_error: {exc}"
    return path_str, n_rows, tra_fixed, trb_fixed, "ok"


def _file_has_bug(path_str: str) -> bool:
    """Return True if tra_full == tra_cdr3 (or trb variant) for any row."""
    try:
        t = pq.read_table(
            path_str, columns=["tra_full", "tra_cdr3", "trb_full", "trb_cdr3"]
        )
    except Exception:
        return True  # re-process on error
    tra_eq = pc.sum(pc.equal(t["tra_full"], t["tra_cdr3"])).as_py() or 0
    trb_eq = pc.sum(pc.equal(t["trb_full"], t["trb_cdr3"])).as_py() or 0
    if tra_eq or trb_eq:
        return True
    tra_short = pc.sum(
        pc.and_(
            pc.is_valid(t["tra_full"]),
            pc.less(pc.utf8_length(t["tra_full"]), MIN_FULL_LEN),
        ).fill_null(False)
    ).as_py() or 0
    trb_short = pc.sum(
        pc.and_(
            pc.is_valid(t["trb_full"]),
            pc.less(pc.utf8_length(t["trb_full"]), MIN_FULL_LEN),
        ).fill_null(False)
    ).as_py() or 0
    return bool(tra_short) or bool(trb_short)


def main() -> None:
    t0 = time.time()
    all_paths = sorted(ROOT.rglob("*.parquet"))
    print(f"Scanning {len(all_paths)} parquet files in {ROOT}")

    # Only process files that still have the bug (resumable after OOM)
    print("Identifying files that still need fixing...")
    buggy: list[Path] = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_file_has_bug, str(p)): p for p in all_paths}
        for i, fut in enumerate(as_completed(futures), 1):
            if fut.result():
                buggy.append(futures[fut])
            if i % 100 == 0 or i == len(all_paths):
                print(f"  scan {i}/{len(all_paths)}  |  need_fix={len(buggy)}", flush=True)
    print(f"Files still needing fix: {len(buggy)} / {len(all_paths)}")

    if not buggy:
        print("Nothing to do.")
        return

    print(f"Rule: tra_full/trb_full -> '' if (equals CDR3) or (len < {MIN_FULL_LEN})")
    print(f"Workers: {MAX_WORKERS}")

    total_rows = 0
    total_tra_fixed = 0
    total_trb_fixed = 0
    done = 0
    errors: list[tuple[str, str]] = []
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_process_file, str(p)) for p in buggy]
        for fut in as_completed(futures):
            path_str, n_rows, tra_fixed, trb_fixed, status = fut.result()
            if status != "ok":
                errors.append((path_str, status))
            total_rows += n_rows
            total_tra_fixed += tra_fixed
            total_trb_fixed += trb_fixed
            done += 1
            if done % 25 == 0 or done == len(buggy):
                elapsed = time.time() - t1
                eta = elapsed / done * (len(buggy) - done) if done else 0
                print(
                    f"  {done}/{len(buggy)} files | {total_rows:,} rows | "
                    f"tra fixed={total_tra_fixed:,} trb fixed={total_trb_fixed:,} | "
                    f"{elapsed:.1f}s | ETA {eta:.1f}s",
                    flush=True,
                )

    print("\n== Summary ==")
    print(f"  Files rewritten     : {len(buggy)}")
    print(f"  Rows scanned        : {total_rows:,}")
    print(f"  tra_full cells fixed: {total_tra_fixed:,}")
    print(f"  trb_full cells fixed: {total_trb_fixed:,}")
    print(f"  Errors              : {len(errors)}")
    for p, s in errors[:10]:
        print(f"    {p}: {s}")
    print(f"  Elapsed             : {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
