#!/usr/bin/env python3
"""
Functional validation of tcrbench_dedup.py with synthetic VDJdb-like data.

Creates a synthetic Parquet dataset with deliberate edge cases, runs the
pipeline in --mode both, and validates outputs.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


def create_synthetic_data(parquet_path: str) -> dict:
    """
    Create synthetic VDJdb-like Parquet data with edge cases.
    Returns metadata about expected outcomes.
    """
    rows = []

    # Fake MHC protein sequences (AA-only, >4 chars) for testing
    MHC_A0201 = "MAVMAPRTLVLLLSGALALTQTWAGSHSMRY"  # 30 AA, stands in for HLA-A*02:01
    MHC_A0301 = "MARMAPRTVLLLLLWGAVALTETWAGSHSMR"  # 30 AA, stands in for HLA-A*03:01
    MHC_B0702 = "MRVTAPRTVLLLLSEALALTQTWAGSHSLKY"  # 30 AA, stands in for HLA-B*07:02
    MHC_B0801 = "MLVMAPRTVLLLLSAALALTETWAGSHSMRY"  # 30 AA, stands in for HLA-B*08:01
    MHC_DRB1_0101 = "MVCLKFPGGSCMAALTVTLMVLSSPLAL"  # 28 AA, stands in for HLA-DRB1*01:01
    MHC_DRB1_0401 = "MVCLKLPGGSCMTALTVTLMVLSSPLAL"  # 28 AA, stands in for HLA-DRB1*04:01
    MHC_DRB1_1501 = "MVCLRLPGGSCMAVLTVTLMVLSSPLAL"  # 28 AA, stands in for HLA-DRB1*15:01

    # --- Group 1: Normal rows (should survive sanitization and dedup) ---
    # Row 0: fully populated
    rows.append({"tra_full": "CAVRNTGGFKTIF", "trb_full": "CASSLAPGATNEKLFF", "peptide": "GILGFVFTL", "mhc_one": MHC_A0201, "mhc_two": MHC_DRB1_0101})
    # Row 1: duplicate of row 0 (should be deduped)
    rows.append({"tra_full": "CAVRNTGGFKTIF", "trb_full": "CASSLAPGATNEKLFF", "peptide": "GILGFVFTL", "mhc_one": MHC_A0201, "mhc_two": MHC_DRB1_0101})
    # Row 2: another duplicate of row 0
    rows.append({"tra_full": "CAVRNTGGFKTIF", "trb_full": "CASSLAPGATNEKLFF", "peptide": "GILGFVFTL", "mhc_one": MHC_A0201, "mhc_two": MHC_DRB1_0101})
    # Row 3: different fully populated row
    rows.append({"tra_full": "CAVRDTGNQFYF", "trb_full": "CASSLGQAYEQYF", "peptide": "NLVPMVATV", "mhc_one": MHC_A0201, "mhc_two": MHC_DRB1_0401})
    # Row 4: tra-only
    rows.append({"tra_full": "CAVKDSNYQLIW", "trb_full": None, "peptide": None, "mhc_one": None, "mhc_two": None})
    # Row 5: trb-only
    rows.append({"tra_full": None, "trb_full": "CASSFSTCSANYGYTF", "peptide": None, "mhc_one": None, "mhc_two": None})
    # Row 6: tra+trb only (no peptide/mhc)
    rows.append({"tra_full": "CAENTGNQFYF", "trb_full": "CASSYSGGANTGELFF", "peptide": None, "mhc_one": None, "mhc_two": None})
    # Row 7: peptide+mhc_one only
    rows.append({"tra_full": None, "trb_full": None, "peptide": "GLCTLVAML", "mhc_one": MHC_A0201, "mhc_two": None})
    # Row 8: tra+peptide+mhc_one
    rows.append({"tra_full": "CAVRPLLDGGSQGNLIF", "trb_full": None, "peptide": "KLGGALQAK", "mhc_one": MHC_A0301, "mhc_two": None})

    # --- Group 2: Empty/whitespace strings (should become NULL) ---
    # Row 9: empty string tra, valid trb
    rows.append({"tra_full": "", "trb_full": "CASSIRSSYEQYF", "peptide": "TPRVTGGGAM", "mhc_one": MHC_B0702, "mhc_two": None})
    # Row 10: whitespace-only peptide
    rows.append({"tra_full": "CAVKDSNYQLIW", "trb_full": "CASSFSTCSANYGYTF", "peptide": "   ", "mhc_one": None, "mhc_two": None})
    # Row 11: all empty strings (should become all-NULL row)
    rows.append({"tra_full": "", "trb_full": "", "peptide": "", "mhc_one": "", "mhc_two": ""})
    # Row 12: another all-empty (duplicate all-NULL after sanitization, should dedup with row 11)
    rows.append({"tra_full": "  ", "trb_full": "  ", "peptide": " ", "mhc_one": "", "mhc_two": ""})

    # --- Group 3: Short sequences (below min length -> NULL) ---
    # Row 13: short tra (3 AA < 4 min), valid trb
    rows.append({"tra_full": "CAV", "trb_full": "CASSLAPGATNEKLFF", "peptide": "GILGFVFTL", "mhc_one": MHC_A0201, "mhc_two": None})
    # Row 14: short trb (2 AA < 4 min)
    rows.append({"tra_full": "CAVRNTGGFKTIF", "trb_full": "CA", "peptide": "NLVPMVATV", "mhc_one": MHC_A0201, "mhc_two": None})
    # Row 15: short peptide (7 AA < 8 min)
    rows.append({"tra_full": "CAVRNTGGFKTIF", "trb_full": "CASSLAPGATNEKLFF", "peptide": "GILGFVF", "mhc_one": MHC_A0201, "mhc_two": None})
    # Row 16: short mhc_one (3 chars < 4 min)
    rows.append({"tra_full": "CAVRNTGGFKTIF", "trb_full": "CASSLAPGATNEKLFF", "peptide": "GILGFVFTL", "mhc_one": "HLA", "mhc_two": None})
    # Row 17: short mhc_two (2 chars < 4 min)
    rows.append({"tra_full": "CAVRNTGGFKTIF", "trb_full": "CASSLAPGATNEKLFF", "peptide": "GILGFVFTL", "mhc_one": MHC_A0201, "mhc_two": "HL"})

    # --- Group 4: Long sequences (above max length -> NULL) ---
    # Row 18: long tra (no max for tra_full, so this is fine)
    rows.append({"tra_full": "A" * 31, "trb_full": "CASSLAPGATNEKLFF", "peptide": "GILGFVFTL", "mhc_one": MHC_A0201, "mhc_two": None})
    # Row 19: long peptide (31 AA > 30 max)
    rows.append({"tra_full": "CAVRNTGGFKTIF", "trb_full": "CASSLAPGATNEKLFF", "peptide": "A" * 31, "mhc_one": MHC_A0201, "mhc_two": None})
    # Row 20: long trb (no max for trb_full, but B is not a valid AA)
    rows.append({"tra_full": "CAVRNTGGFKTIF", "trb_full": "B" * 35, "peptide": "GILGFVFTL", "mhc_one": MHC_A0201, "mhc_two": None})

    # --- Group 5: Partial NULL rows (some columns present, some NULL) ---
    # Row 21: only mhc_one (no TCR, no peptide)
    rows.append({"tra_full": None, "trb_full": None, "peptide": None, "mhc_one": MHC_B0801, "mhc_two": None})
    # Row 22: only mhc_two
    rows.append({"tra_full": None, "trb_full": None, "peptide": None, "mhc_one": None, "mhc_two": MHC_DRB1_1501})
    # Row 23: all NULL (explicit NULLs, should dedup with rows 11/12)
    rows.append({"tra_full": None, "trb_full": None, "peptide": None, "mhc_one": None, "mhc_two": None})

    # --- Group 6: Duplicates that arise AFTER sanitization ---
    # Row 24: same as row 13 after sanitization (tra NULLed -> NULL, trb, pep, mhc_one same)
    # After sanitization row 13 becomes: (NULL, CASSLAPGATNEKLFF, GILGFVFTL, MHC_A0201, NULL)
    # This is the same as what we get, so it's a duplicate
    rows.append({"tra_full": "XX", "trb_full": "CASSLAPGATNEKLFF", "peptide": "GILGFVFTL", "mhc_one": MHC_A0201, "mhc_two": None})

    # Build pyarrow table
    tra_full = [r["tra_full"] for r in rows]
    trb_full = [r["trb_full"] for r in rows]
    peptide = [r["peptide"] for r in rows]
    mhc_one = [r["mhc_one"] for r in rows]
    mhc_two = [r["mhc_two"] for r in rows]

    table = pa.table({
        "tra_full": pa.array(tra_full, type=pa.string()),
        "trb_full": pa.array(trb_full, type=pa.string()),
        "peptide": pa.array(peptide, type=pa.string()),
        "mhc_one": pa.array(mhc_one, type=pa.string()),
        "mhc_two": pa.array(mhc_two, type=pa.string()),
    })

    pq.write_table(table, parquet_path)
    print(f"  Wrote {len(rows)} rows to {parquet_path}")

    # Now compute expected deduped rows manually:
    # Apply sanitization to each row and collect distinct tuples.
    import re
    _AA_RE = re.compile(r'^[ACDEFGHIKLMNPQRSTVWY]+$')

    def sanitize_val(val, col):
        from tcrbench_dedup import LENGTH_RULES, AA_ONLY_COLUMNS
        if val is None:
            return None
        trimmed = val.strip()
        if trimmed == "":
            return None
        if col in AA_ONLY_COLUMNS and not _AA_RE.match(trimmed):
            return None
        min_len, max_len = LENGTH_RULES[col]
        if len(trimmed) < min_len:
            return None
        if max_len is not None and len(trimmed) > max_len:
            return None
        return trimmed

    cols = ["tra_full", "trb_full", "peptide", "mhc_one", "mhc_two"]
    sanitized_set = set()
    for r in rows:
        tup = tuple(sanitize_val(r[c], c) for c in cols)
        sanitized_set.add(tup)

    print(f"  Expected distinct rows after sanitization+dedup: {len(sanitized_set)}")

    # Compute expected combination counts
    from tcrbench_dedup import COMBINATIONS
    expected_counts = {}
    for label, combo_cols in COMBINATIONS.items():
        distinct_tuples = set()
        for tup in sanitized_set:
            vals = tuple(tup[cols.index(c)] for c in combo_cols)
            if all(v is not None for v in vals):
                distinct_tuples.add(vals)
        expected_counts[label] = len(distinct_tuples)
    print(f"  Expected combination counts: {json.dumps(expected_counts, indent=2)}")

    # Compute expected explosion partitions using ordered permutations (325 total)
    from tcrbench_dedup import PERMUTATIONS, SHORT_NAMES
    expected_partitions = set()
    for tup in sanitized_set:
        presence = 0
        for i, v in enumerate(tup):
            if v is not None:
                presence |= (1 << i)
        if presence == 0:
            continue  # all-NULL rows don't generate any explosion rows
        for key, mask, col_indices in PERMUTATIONS:
            if mask & presence == mask:
                expected_partitions.add(key)

    # Compute expected exploded row counts per partition (ordered permutations)
    # For ordered permutations, rows with same columns but different orderings are
    # in different partitions. Within each partition, DISTINCT on column values applies.
    expected_exploded_counts = {}
    for key, mask, col_indices in PERMUTATIONS:
        projected = set()
        for tup in sanitized_set:
            presence = 0
            for i, v in enumerate(tup):
                if v is not None:
                    presence |= (1 << i)
            if mask & presence != mask:
                continue
            proj = tuple(tup[i] if (mask & (1 << i)) else None for i in range(5))
            projected.add(proj)
        if projected:
            expected_exploded_counts[key] = len(projected)

    return {
        "n_input_rows": len(rows),
        "expected_deduped_rows": len(sanitized_set),
        "sanitized_set": sanitized_set,
        "expected_counts": expected_counts,
        "expected_partitions": expected_partitions,
        "expected_exploded_counts": expected_exploded_counts,
    }


def run_pipeline(input_path: str, output_dir: str, tmp_dir: str) -> subprocess.CompletedProcess:
    """Run tcrbench_dedup.py as a subprocess."""
    cmd = [
        sys.executable, "/home/ubuntu/quest/tcrbench_dedup.py",
        "--input", input_path,
        "--output_dir", output_dir,
        "--tmp_dir", tmp_dir,
        "--memory_limit", "4GB",
        "--threads", "4",
        "--mode", "both",
        "--force_recompute",
        "--verbose",
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return result


def validate_deduped(output_dir: str, meta: dict) -> list[str]:
    """Validate deduped parquet output. Returns list of failure messages."""
    failures = []
    deduped_dir = os.path.join(output_dir, "deduped_parquet")

    if not os.path.isdir(deduped_dir):
        failures.append(f"Deduped parquet dir does not exist: {deduped_dir}")
        return failures

    # Read all parquet files in the directory
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{deduped_dir}/*.parquet')").fetchdf()
    con.close()

    actual_rows = len(df)
    expected_rows = meta["expected_deduped_rows"]
    if actual_rows != expected_rows:
        failures.append(f"Deduped row count: expected {expected_rows}, got {actual_rows}")

    # Check no empty strings remain
    cols = ["tra_full", "trb_full", "peptide", "mhc_one", "mhc_two"]
    for col in cols:
        empty_mask = df[col].apply(lambda x: isinstance(x, str) and x.strip() == "")
        n_empty = empty_mask.sum()
        if n_empty > 0:
            failures.append(f"Column {col} has {n_empty} empty/whitespace strings")

    # Check length constraints are respected for non-null values
    from tcrbench_dedup import LENGTH_RULES
    for col in cols:
        min_len, max_len = LENGTH_RULES[col]
        non_null = df[col].dropna()
        if len(non_null) > 0:
            too_short = non_null[non_null.str.len() < min_len]
            if len(too_short) > 0:
                failures.append(f"Column {col}: {len(too_short)} values shorter than min {min_len}")
            if max_len is not None:
                too_long = non_null[non_null.str.len() > max_len]
                if len(too_long) > 0:
                    failures.append(f"Column {col}: {len(too_long)} values longer than max {max_len}")

    return failures


def validate_counts(output_dir: str, meta: dict) -> list[str]:
    """Validate combination_counts.json. Returns list of failure messages."""
    failures = []

    json_path = os.path.join(output_dir, "combination_counts.json")
    if not os.path.isfile(json_path):
        failures.append(f"combination_counts.json not found at {json_path}")
        return failures

    with open(json_path) as f:
        actual_counts = json.load(f)

    expected_counts = meta["expected_counts"]
    for combo, expected in expected_counts.items():
        actual = actual_counts.get(combo)
        if actual != expected:
            failures.append(f"Count for '{combo}': expected {expected}, got {actual}")

    # Check no unexpected keys
    for combo in actual_counts:
        if combo not in expected_counts:
            failures.append(f"Unexpected combo in counts: '{combo}'")

    return failures


def validate_exploded(output_dir: str, meta: dict) -> list[str]:
    """Validate exploded_deduped output. Returns list of failure messages."""
    failures = []

    exploded_dir = os.path.join(output_dir, "exploded_deduped")
    if not os.path.isdir(exploded_dir):
        failures.append(f"Exploded dir does not exist: {exploded_dir}")
        return failures

    # Check which order_key partitions exist (nested under subset_key=*/order_key=*)
    found_partitions = set()
    for subset_dir in Path(exploded_dir).iterdir():
        if not (subset_dir.is_dir() and subset_dir.name.startswith("subset_key=")):
            continue
        for order_dir in subset_dir.iterdir():
            if order_dir.is_dir() and order_dir.name.startswith("order_key="):
                found_partitions.add(order_dir.name.split("=", 1)[1])

    expected_partitions = meta["expected_partitions"]
    missing = expected_partitions - found_partitions
    extra = found_partitions - expected_partitions
    if missing:
        failures.append(f"Missing expected partitions: {sorted(missing)}")
    if extra:
        failures.append(f"Unexpected extra partitions: {sorted(extra)}")

    # Check per-partition row counts (keyed by order_key, the ordered perm)
    con = duckdb.connect()
    try:
        partition_counts = con.execute(
            f"SELECT order_key, COUNT(*) AS cnt "
            f"FROM read_parquet('{exploded_dir}/**/*.parquet', hive_partitioning=true) "
            f"GROUP BY order_key"
        ).fetchall()
    except Exception as e:
        failures.append(f"Failed to read exploded parquet: {e}")
        con.close()
        return failures

    actual_partition_counts = {row[0]: row[1] for row in partition_counts}
    con.close()

    expected_exploded_counts = meta["expected_exploded_counts"]
    for key, expected in expected_exploded_counts.items():
        actual = actual_partition_counts.get(key)
        if actual != expected:
            failures.append(f"Exploded count for '{key}': expected {expected}, got {actual}")

    # Check no empty strings in exploded output
    con = duckdb.connect()
    df = con.execute(
        f"SELECT * FROM read_parquet('{exploded_dir}/**/*.parquet', hive_partitioning=true) LIMIT 10000"
    ).fetchdf()
    con.close()

    for col in ["tra_full", "trb_full", "peptide", "mhc_one", "mhc_two"]:
        if col in df.columns:
            empty_mask = df[col].apply(lambda x: isinstance(x, str) and x.strip() == "")
            n_empty = empty_mask.sum()
            if n_empty > 0:
                failures.append(f"Exploded column {col} has {n_empty} empty strings")

    # Check order_key values match expected labels (all 325 ordered permutations)
    if "order_key" in df.columns:
        from tcrbench_dedup import PERMUTATIONS
        valid_keys = set(key for key, _, _ in PERMUTATIONS)
        actual_keys = set(df["order_key"].unique())
        invalid_keys = actual_keys - valid_keys
        if invalid_keys:
            failures.append(f"Invalid order_key values found: {invalid_keys}")

    return failures


def main():
    print("=" * 70)
    print("TCRBench Dedup Functional Validation")
    print("=" * 70)

    tmpdir = tempfile.mkdtemp(prefix="tcrbench_test_")
    print(f"\nUsing temp directory: {tmpdir}")

    try:
        input_dir = os.path.join(tmpdir, "input")
        output_dir = os.path.join(tmpdir, "output")
        tmp_duckdb_dir = os.path.join(tmpdir, "duckdb_tmp")
        os.makedirs(input_dir)
        os.makedirs(output_dir)
        os.makedirs(tmp_duckdb_dir)

        parquet_path = os.path.join(input_dir, "synthetic_vdjdb.parquet")

        # Step 1: Create synthetic data
        print("\n--- Creating synthetic data ---")
        meta = create_synthetic_data(parquet_path)

        # Step 2: Run pipeline
        print("\n--- Running pipeline ---")
        result = run_pipeline(parquet_path, output_dir, tmp_duckdb_dir)

        if result.returncode != 0:
            print(f"\n  PIPELINE FAILED (exit code {result.returncode})")
            print(f"  STDOUT:\n{result.stdout}")
            print(f"  STDERR:\n{result.stderr}")
            sys.exit(1)
        else:
            print("  Pipeline completed successfully")
            if "--verbose" in str(result.args):
                # Print abbreviated output
                lines = result.stdout.strip().split("\n")
                for line in lines:
                    print(f"    {line}")

        # Step 3: Validate outputs
        all_failures = []

        print("\n--- Validating deduped output ---")
        failures = validate_deduped(output_dir, meta)
        all_failures.extend(failures)
        if failures:
            for f in failures:
                print(f"  FAIL: {f}")
        else:
            print("  PASS: Deduped output validated")

        print("\n--- Validating combination counts ---")
        failures = validate_counts(output_dir, meta)
        all_failures.extend(failures)
        if failures:
            for f in failures:
                print(f"  FAIL: {f}")
        else:
            print("  PASS: Combination counts validated")

        print("\n--- Validating exploded output ---")
        failures = validate_exploded(output_dir, meta)
        all_failures.extend(failures)
        if failures:
            for f in failures:
                print(f"  FAIL: {f}")
        else:
            print("  PASS: Exploded output validated")

        # Final summary
        print("\n" + "=" * 70)
        if all_failures:
            print(f"RESULT: {len(all_failures)} FAILURE(S)")
            for i, f in enumerate(all_failures, 1):
                print(f"  {i}. {f}")
            sys.exit(1)
        else:
            print("RESULT: ALL TESTS PASSED")
            sys.exit(0)

    finally:
        print(f"\nCleaning up temp directory: {tmpdir}")
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
