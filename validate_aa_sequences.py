#!/usr/bin/env python3
"""Validate that all sequence columns in standardized_again contain valid amino acid sequences.

Reads first parquet file per database (up to 1M rows), samples for validation.
Distinguishes null, empty string, and actual sequence data.
"""

import re
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("/data/standardized_again")

SHORT_SEQ_COLS = ["tra", "trb", "tra_cdr1", "tra_cdr2", "tra_cdr3",
                  "trb_cdr1", "trb_cdr2", "trb_cdr3", "peptide"]
LONG_SEQ_COLS = ["tra_full", "trb_full", "mhc_one", "mhc_two"]
ALL_SEQ_COLS = SHORT_SEQ_COLS + LONG_SEQ_COLS

VALID_AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYXU]+$")
SAMPLE_SIZE = 50_000


def count_total_rows(db_dir):
    total = 0
    for f in sorted(db_dir.glob("part_*.parquet")):
        total += pq.read_metadata(f).num_rows
    return total


def validate_column(table, col_name):
    if col_name not in table.column_names:
        return {"has_col": False}

    col = table.column(col_name)
    n_total = col.length()

    # Count nulls
    n_null = pc.sum(pc.is_null(col, nan_is_null=True)).as_py()

    # Get non-null values
    non_null_vals = pc.filter(col, pc.invert(pc.is_null(col, nan_is_null=True))).to_pylist()

    # Count empty strings
    n_empty = sum(1 for v in non_null_vals if isinstance(v, str) and v == "")
    # Actual sequences (non-null, non-empty)
    actual_seqs = [v for v in non_null_vals if isinstance(v, str) and len(v) > 0]
    n_actual = len(actual_seqs)

    if n_actual == 0:
        return {"has_col": True, "n_total": n_total, "n_null": n_null,
                "n_empty": n_empty, "n_actual": 0}

    # Sample for validation
    if n_actual > SAMPLE_SIZE:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_actual, SAMPLE_SIZE, replace=False)
        sample = [actual_seqs[i] for i in idx]
    else:
        sample = actual_seqs

    n_valid = sum(1 for v in sample if VALID_AA_RE.match(v))
    n_invalid = len(sample) - n_valid
    pct_valid = 100.0 * n_valid / len(sample) if sample else None

    bad_chars = defaultdict(int)
    bad_examples = []
    for v in sample:
        if not VALID_AA_RE.match(v):
            for c in set(v) - set("ACDEFGHIKLMNPQRSTVWYXU"):
                bad_chars[c] += 1
            if len(bad_examples) < 5:
                bad_examples.append(v[:100])

    lengths = np.array([len(v) for v in sample])
    len_stats = {
        "min": int(lengths.min()), "q25": int(np.percentile(lengths, 25)),
        "median": int(np.median(lengths)), "q75": int(np.percentile(lengths, 75)),
        "max": int(lengths.max()),
    }

    return {
        "has_col": True, "n_total": n_total, "n_null": n_null,
        "n_empty": n_empty, "n_actual": n_actual,
        "pct_valid": pct_valid, "n_valid_sample": n_valid, "n_invalid_sample": n_invalid,
        "bad_chars": dict(bad_chars), "bad_examples": bad_examples,
        "lengths": len_stats,
    }


def fmt_count(n):
    if n >= 1e6: return f"{n/1e6:.1f}M"
    if n >= 1000: return f"{n/1e3:.0f}K"
    return str(n)


def main():
    databases = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    all_results = {}

    for db in databases:
        db_dir = DATA_DIR / db
        print(f"Processing {db}...", flush=True)
        total_rows = count_total_rows(db_dir)

        parquet_files = sorted(db_dir.glob("part_*.parquet"))
        if not parquet_files:
            continue

        table = pq.read_table(parquet_files[0])
        if table.num_rows > 1_000_000:
            table = table.slice(0, 1_000_000)

        db_res = {"total_rows": total_rows, "sampled_rows": table.num_rows, "columns": {}}
        for col in ALL_SEQ_COLS:
            db_res["columns"][col] = validate_column(table, col)
        all_results[db] = db_res

    total_all = sum(r["total_rows"] for r in all_results.values())

    # ==================== REPORT ====================
    print("\n" + "=" * 180)
    print("AMINO ACID SEQUENCE VALIDATION REPORT")
    print(f"Total databases: {len(all_results)} | Total rows: {total_all:,}")
    print("(Validated on first parquet file per database, up to 1M rows, sampled 50K for AA check)")
    print("=" * 180)

    # --- Table 1: Data availability per column ---
    print(f"\nDATA AVAILABILITY (actual sequences, excluding nulls and empty strings)")
    print(f"{'Database':<16} {'Total':>12} ", end="")
    for c in ALL_SEQ_COLS:
        print(f"| {c:>10} ", end="")
    print()
    print("-" * 200)

    for db, res in sorted(all_results.items()):
        line = f"{db:<16} {res['total_rows']:>12,} "
        for col in ALL_SEQ_COLS:
            info = res["columns"].get(col, {})
            n_actual = info.get("n_actual", 0)
            if not info.get("has_col"):
                line += f"| {'N/A':>10} "
            elif n_actual == 0:
                line += f"| {'—':>10} "
            else:
                line += f"| {fmt_count(n_actual):>10} "
        print(line)

    # --- Table 2: AA validity for columns with actual data ---
    print(f"\nAA VALIDITY (% of actual sequences that are valid amino acids)")
    print(f"{'Database':<16} ", end="")
    for c in ALL_SEQ_COLS:
        print(f"| {c:>10} ", end="")
    print()
    print("-" * 200)

    for db, res in sorted(all_results.items()):
        line = f"{db:<16} "
        for col in ALL_SEQ_COLS:
            info = res["columns"].get(col, {})
            n_actual = info.get("n_actual", 0)
            if n_actual == 0:
                line += f"| {'—':>10} "
            else:
                pct = info.get("pct_valid", 0)
                line += f"| {pct:>8.1f}% "
        print(line)

    # --- Table 3: Length distributions ---
    print(f"\nLENGTH DISTRIBUTIONS (median [min-max])")
    print(f"{'Database':<16}", end="")
    for col in ALL_SEQ_COLS:
        print(f" | {col:>16}", end="")
    print()
    print("-" * 220)

    for db, res in sorted(all_results.items()):
        line = f"{db:<16}"
        for col in ALL_SEQ_COLS:
            info = res["columns"].get(col, {})
            lens = info.get("lengths")
            if not lens:
                line += f" | {'—':>16}"
            else:
                line += f" | {lens['median']:>4} [{lens['min']:>3}-{lens['max']:>4}]"
        print(line)

    # --- Issues ---
    print(f"\nISSUES: COLUMNS WITH INVALID AMINO ACID CHARACTERS")
    print("-" * 100)
    found = False
    for db, res in sorted(all_results.items()):
        for col in ALL_SEQ_COLS:
            info = res["columns"].get(col, {})
            if info.get("n_invalid_sample", 0) > 0:
                found = True
                print(f"  {db}/{col}: {info['pct_valid']:.2f}% valid, {info['n_invalid_sample']} invalid in sample of {info.get('n_valid_sample',0)+info['n_invalid_sample']}")
                print(f"    Bad chars: {info['bad_chars']}")
                for ex in info.get("bad_examples", [])[:3]:
                    print(f"    Example: {ex}")
    if not found:
        print("  PASS - all sequence columns contain only valid amino acid characters!")

    # --- Summary: which databases have which data types ---
    print(f"\nDATABASE CAPABILITY MATRIX")
    print(f"{'Database':<16} {'TCR-a':>6} {'TCR-b':>6} {'Paired':>7} {'CDRs':>5} {'Full':>5} {'Pep':>5} {'MHC-I':>6} {'MHC-II':>7} {'Total Rows':>12}")
    print("-" * 85)

    for db, res in sorted(all_results.items()):
        c = res["columns"]
        has_tra = c.get("tra", {}).get("n_actual", 0) > 0
        has_trb = c.get("trb", {}).get("n_actual", 0) > 0
        has_paired = has_tra and has_trb
        has_cdrs = any(c.get(x, {}).get("n_actual", 0) > 0 for x in ["tra_cdr1", "trb_cdr1"])
        has_full = any(c.get(x, {}).get("n_actual", 0) > 0 for x in ["tra_full", "trb_full"])
        has_pep = c.get("peptide", {}).get("n_actual", 0) > 0
        has_mhc1 = c.get("mhc_one", {}).get("n_actual", 0) > 0
        has_mhc2 = c.get("mhc_two", {}).get("n_actual", 0) > 0

        def yn(b): return "Y" if b else ""

        print(f"{db:<16} {yn(has_tra):>6} {yn(has_trb):>6} {yn(has_paired):>7} {yn(has_cdrs):>5} {yn(has_full):>5} {yn(has_pep):>5} {yn(has_mhc1):>6} {yn(has_mhc2):>7} {res['total_rows']:>12,}")


if __name__ == "__main__":
    main()
