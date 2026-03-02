#!/usr/bin/env python3
"""
Verify the delta between mlm (CDR3) and mlm_full (full-length stitched) dedup outputs.

Scans standardized parquet files to classify each row's TCR chains as stitchable
(has CDR3 + V gene + J gene) vs missing-gene (has CDR3 but missing V or J gene).
Cross-references with DEDUP_SUMMARY.txt to reconcile the ~205M row delta.

Usage:
    python scripts/analysis/verify_dedup_delta.py \
        --standardized-dir data/standardized \
        --dedup-summary data/deduplicated/DEDUP_SUMMARY.txt \
        --standardization-summary data/standardized/STANDARDIZATION_SUMMARY.txt \
        --num-workers 32
"""

import argparse
import multiprocessing
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

# Columns needed for stitchability classification
CHAIN_SPECS = [
    ("tra", "tra", "trav_gene", "traj_gene"),
    ("trb", "trb", "trbv_gene", "trbj_gene"),
]

TCR_COLUMNS = ["tra", "trav_gene", "traj_gene", "trb", "trbv_gene", "trbj_gene"]


# ---------------------------------------------------------------------------
# Phase 1: Classify stitchability per parquet file (parallel)
# ---------------------------------------------------------------------------


def _classify_file(filepath: str) -> Dict[str, int]:
    """
    Classify TCR chain stitchability in a single parquet file.

    Returns dict with keys like:
        "db::tra_total", "db::tra_stitchable", "db::tra_missing_gene",
        "db::trb_total", "db::trb_stitchable", "db::trb_missing_gene",
        "db::total_rows"
    where db is extracted from the file path.
    """
    path = Path(filepath)
    # Extract database name: data/standardized/{db}/part_XXXX.parquet
    db_name = path.parent.name

    table = pq.read_table(filepath, columns=TCR_COLUMNS)
    n_rows = len(table)

    counts = {
        f"{db_name}::total_rows": n_rows,
    }

    for chain, cdr3_col, v_col, j_col in CHAIN_SPECS:
        cdr3 = table.column(cdr3_col)
        v_gene = table.column(v_col)
        j_gene = table.column(j_col)

        # Non-empty means has content (values are empty strings, not null)
        has_cdr3 = pc.not_equal(cdr3, "")
        has_v = pc.not_equal(v_gene, "")
        has_j = pc.not_equal(j_gene, "")

        # Total rows with CDR3 for this chain
        total_with_cdr3 = pc.sum(has_cdr3).as_py()

        # Stitchable: has CDR3 AND V gene AND J gene
        stitchable_mask = pc.and_(has_cdr3, pc.and_(has_v, has_j))
        stitchable = pc.sum(stitchable_mask).as_py()

        # Missing gene: has CDR3 but missing V or J
        missing_gene = total_with_cdr3 - stitchable

        # Breakdown: missing V only, missing J only, missing both
        has_cdr3_no_v = pc.and_(has_cdr3, pc.invert(has_v))
        has_cdr3_no_j = pc.and_(has_cdr3, pc.invert(has_j))
        has_cdr3_no_both = pc.and_(has_cdr3, pc.and_(pc.invert(has_v), pc.invert(has_j)))
        missing_v_only = pc.sum(has_cdr3_no_v).as_py() - pc.sum(has_cdr3_no_both).as_py()
        missing_j_only = pc.sum(has_cdr3_no_j).as_py() - pc.sum(has_cdr3_no_both).as_py()
        missing_both = pc.sum(has_cdr3_no_both).as_py()

        counts[f"{db_name}::{chain}_total"] = total_with_cdr3
        counts[f"{db_name}::{chain}_stitchable"] = stitchable
        counts[f"{db_name}::{chain}_missing_gene"] = missing_gene
        counts[f"{db_name}::{chain}_missing_v_only"] = missing_v_only
        counts[f"{db_name}::{chain}_missing_j_only"] = missing_j_only
        counts[f"{db_name}::{chain}_missing_both"] = missing_both

    return counts


def classify_all_files(
    standardized_dir: Path, num_workers: int
) -> Dict[str, Dict[str, int]]:
    """
    Scan all standardized parquet files and classify stitchability.

    Returns per-database aggregated counts.
    """
    # Collect all parquet files
    all_files = []
    for db_dir in sorted(standardized_dir.iterdir()):
        if not db_dir.is_dir():
            continue
        parquet_files = sorted(db_dir.glob("part_*.parquet"))
        all_files.extend([str(f) for f in parquet_files])

    if not all_files:
        print("No parquet files found!", file=sys.stderr)
        return {}

    print(f"Found {len(all_files):,} parquet files across databases")

    # Parallel classification
    per_db: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    with multiprocessing.Pool(num_workers) as pool:
        for counts in tqdm(
            pool.imap_unordered(_classify_file, all_files),
            total=len(all_files),
            desc="Classifying stitchability",
        ):
            for key, value in counts.items():
                db_name, metric = key.split("::", 1)
                per_db[db_name][metric] += value

    return dict(per_db)


# ---------------------------------------------------------------------------
# Phase 2: Parse DEDUP_SUMMARY.txt
# ---------------------------------------------------------------------------


def parse_dedup_summary(filepath: Path) -> Dict[str, Dict[str, int]]:
    """
    Parse DEDUP_SUMMARY.txt to extract per-mode row counts and unique molecules.

    Returns dict like:
        {"mlm": {"total_rows": N, "tra_unique": N, "trb_unique": N, ...},
         "mlm_full": {...}}
    """
    text = filepath.read_text()
    results = {}

    # Split on ={80} delimiter lines and find mode sections
    pattern = (
        r"DEDUPLICATION SUMMARY: (mlm(?:_full)?)\s+\(.*?\)\n=+\n"
        r"(.*?)(?=\n={10,}\nDEDUPLICATION|\n={10,}\nCOMPARISON|$)"
    )
    for match in re.finditer(pattern, text, re.DOTALL):
        mode = match.group(1)
        section = match.group(2)
        results[mode] = {}

        # Total rows
        total_match = re.search(r"Total rows:\s+([\d,]+)", section)
        if total_match:
            results[mode]["total_rows"] = int(
                total_match.group(1).replace(",", "")
            )

        # Unique molecule counts
        for mol in ["tra", "trb", "peptide", "mhc_one", "mhc_two"]:
            mol_match = re.search(
                rf"^\s+{mol}\s+([\d,]+)\s+unique values",
                section,
                re.MULTILINE,
            )
            if mol_match:
                results[mode][f"{mol}_unique"] = int(
                    mol_match.group(1).replace(",", "")
                )

        # Per-permutation-key row counts
        for line in section.splitlines():
            line = line.strip()
            perm_match = re.match(
                r"([\w_]+)\s+([\d,]+)\s+\(\s*[\d.]+%\)", line
            )
            if perm_match:
                perm_key = perm_match.group(1)
                if perm_key == "SUBTOTAL":
                    continue
                count = int(perm_match.group(2).replace(",", ""))
                results[mode][f"perm_{perm_key}"] = count

    return results


# ---------------------------------------------------------------------------
# Phase 3: Parse STANDARDIZATION_SUMMARY.txt
# ---------------------------------------------------------------------------


def parse_standardization_summary(
    filepath: Path,
) -> Tuple[int, Dict[str, int], Dict[str, int]]:
    """
    Parse STANDARDIZATION_SUMMARY.txt for manifest rows and dropped counts.

    Returns (total_manifest, per_db_manifest, per_db_dropped).
    """
    text = filepath.read_text()
    per_db_manifest: Dict[str, int] = {}
    per_db_dropped: Dict[str, int] = {}
    total = 0

    in_section2 = False
    past_header = False

    for line in text.splitlines():
        if "2. PER-DATABASE STATISTICS" in line:
            in_section2 = True
            continue

        if in_section2 and line.startswith("---"):
            if not past_header:
                past_header = True
                continue

        if in_section2 and past_header:
            if line.startswith("TOTAL"):
                parts = line.split()
                for part in parts[1:]:
                    cleaned = part.replace(",", "")
                    if cleaned.isdigit():
                        total = int(cleaned)
                        break
                break

            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) >= 3:
                db_name = parts[0]
                # Find manifest rows (first number) and dropped (second number)
                nums = []
                for part in parts[1:]:
                    cleaned = part.replace(",", "")
                    if cleaned.isdigit():
                        nums.append(int(cleaned))
                    elif cleaned.endswith("%"):
                        continue
                if len(nums) >= 2:
                    per_db_manifest[db_name] = nums[0]
                    per_db_dropped[db_name] = nums[1]
                elif len(nums) == 1:
                    per_db_manifest[db_name] = nums[0]

    return total, per_db_manifest, per_db_dropped


# ---------------------------------------------------------------------------
# Phase 4: Reconciliation report
# ---------------------------------------------------------------------------


def format_stitchability_table(per_db: Dict[str, Dict[str, int]]) -> str:
    """Format per-database stitchability classification table."""
    lines = []
    lines.append("=" * 120)
    lines.append("STITCHABILITY CLASSIFICATION (from standardized parquet files)")
    lines.append("=" * 120)
    lines.append("")

    # Header
    lines.append(
        f"{'Database':<16} {'Total Rows':>14}  "
        f"{'TRB CDR3':>12} {'TRB Stitch':>12} {'TRB Miss':>12} {'TRB %Miss':>9}  "
        f"{'TRA CDR3':>12} {'TRA Stitch':>12} {'TRA Miss':>12} {'TRA %Miss':>9}"
    )
    lines.append("-" * 120)

    grand = defaultdict(int)

    # Sort by total rows descending
    sorted_dbs = sorted(per_db.keys(), key=lambda d: per_db[d].get("total_rows", 0), reverse=True)

    for db in sorted_dbs:
        d = per_db[db]
        total = d.get("total_rows", 0)
        trb_total = d.get("trb_total", 0)
        trb_stitch = d.get("trb_stitchable", 0)
        trb_miss = d.get("trb_missing_gene", 0)
        trb_pct = (trb_miss / trb_total * 100) if trb_total > 0 else 0
        tra_total = d.get("tra_total", 0)
        tra_stitch = d.get("tra_stitchable", 0)
        tra_miss = d.get("tra_missing_gene", 0)
        tra_pct = (tra_miss / tra_total * 100) if tra_total > 0 else 0

        lines.append(
            f"{db:<16} {total:>14,}  "
            f"{trb_total:>12,} {trb_stitch:>12,} {trb_miss:>12,} {trb_pct:>8.1f}%  "
            f"{tra_total:>12,} {tra_stitch:>12,} {tra_miss:>12,} {tra_pct:>8.1f}%"
        )

        for key, val in d.items():
            grand[key] += val

    lines.append("-" * 120)

    # Grand totals
    trb_total = grand.get("trb_total", 0)
    trb_stitch = grand.get("trb_stitchable", 0)
    trb_miss = grand.get("trb_missing_gene", 0)
    trb_pct = (trb_miss / trb_total * 100) if trb_total > 0 else 0
    tra_total = grand.get("tra_total", 0)
    tra_stitch = grand.get("tra_stitchable", 0)
    tra_miss = grand.get("tra_missing_gene", 0)
    tra_pct = (tra_miss / tra_total * 100) if tra_total > 0 else 0

    lines.append(
        f"{'TOTAL':<16} {grand.get('total_rows', 0):>14,}  "
        f"{trb_total:>12,} {trb_stitch:>12,} {trb_miss:>12,} {trb_pct:>8.1f}%  "
        f"{tra_total:>12,} {tra_stitch:>12,} {tra_miss:>12,} {tra_pct:>8.1f}%"
    )
    lines.append("")

    return "\n".join(lines)


def format_missing_gene_breakdown(per_db: Dict[str, Dict[str, int]]) -> str:
    """Format breakdown of missing gene types per database."""
    lines = []
    lines.append("=" * 100)
    lines.append("MISSING GENE BREAKDOWN (rows with CDR3 but missing V and/or J gene)")
    lines.append("=" * 100)
    lines.append("")
    lines.append(
        f"{'Database':<16}  "
        f"{'TRB V-only':>12} {'TRB J-only':>12} {'TRB Both':>12}  "
        f"{'TRA V-only':>12} {'TRA J-only':>12} {'TRA Both':>12}"
    )
    lines.append("-" * 100)

    sorted_dbs = sorted(per_db.keys(), key=lambda d: per_db[d].get("total_rows", 0), reverse=True)
    grand = defaultdict(int)

    for db in sorted_dbs:
        d = per_db[db]
        trb_miss = d.get("trb_missing_gene", 0)
        tra_miss = d.get("tra_missing_gene", 0)
        if trb_miss == 0 and tra_miss == 0:
            continue

        lines.append(
            f"{db:<16}  "
            f"{d.get('trb_missing_v_only', 0):>12,} {d.get('trb_missing_j_only', 0):>12,} "
            f"{d.get('trb_missing_both', 0):>12,}  "
            f"{d.get('tra_missing_v_only', 0):>12,} {d.get('tra_missing_j_only', 0):>12,} "
            f"{d.get('tra_missing_both', 0):>12,}"
        )

        for key, val in d.items():
            grand[key] += val

    lines.append("-" * 100)
    lines.append(
        f"{'TOTAL':<16}  "
        f"{grand.get('trb_missing_v_only', 0):>12,} {grand.get('trb_missing_j_only', 0):>12,} "
        f"{grand.get('trb_missing_both', 0):>12,}  "
        f"{grand.get('tra_missing_v_only', 0):>12,} {grand.get('tra_missing_j_only', 0):>12,} "
        f"{grand.get('tra_missing_both', 0):>12,}"
    )
    lines.append("")

    return "\n".join(lines)


def format_reconciliation(
    per_db: Dict[str, Dict[str, int]],
    dedup: Dict[str, Dict[str, int]],
    std_total: int,
    std_dropped: Dict[str, int],
) -> str:
    """Format the delta reconciliation report."""
    lines = []
    lines.append("=" * 80)
    lines.append("DELTA RECONCILIATION: mlm vs mlm_full")
    lines.append("=" * 80)
    lines.append("")

    mlm = dedup.get("mlm", {})
    mlm_full = dedup.get("mlm_full", {})
    mlm_rows = mlm.get("total_rows", 0)
    mlm_full_rows = mlm_full.get("total_rows", 0)
    delta = mlm_rows - mlm_full_rows

    lines.append(f"mlm rows (CDR3):              {mlm_rows:>15,}")
    lines.append(f"mlm_full rows (stitched):      {mlm_full_rows:>15,}")
    lines.append(f"Delta:                         {delta:>15,}")
    lines.append("")

    # Unique molecule deltas
    lines.append("--- Unique molecule deltas ---")
    for mol in ["tra", "trb", "peptide", "mhc_one", "mhc_two"]:
        mlm_u = mlm.get(f"{mol}_unique", 0)
        full_u = mlm_full.get(f"{mol}_unique", 0)
        mol_delta = mlm_u - full_u
        if mlm_u > 0:
            pct = mol_delta / mlm_u * 100
            lines.append(
                f"  {mol:<12}  mlm={mlm_u:>14,}  mlm_full={full_u:>14,}  "
                f"delta={mol_delta:>14,} ({pct:.1f}% lost)"
            )
    lines.append("")

    # Per-permutation key deltas
    lines.append("--- Per-permutation-key row deltas ---")
    all_perm_keys = set()
    for mode_data in [mlm, mlm_full]:
        for key in mode_data:
            if key.startswith("perm_"):
                all_perm_keys.add(key[5:])  # strip "perm_" prefix

    perm_deltas = []
    for perm_key in sorted(all_perm_keys):
        mlm_count = mlm.get(f"perm_{perm_key}", 0)
        full_count = mlm_full.get(f"perm_{perm_key}", 0)
        d = mlm_count - full_count
        perm_deltas.append((perm_key, mlm_count, full_count, d))

    # Sort by delta descending
    perm_deltas.sort(key=lambda x: x[3], reverse=True)
    total_perm_delta = 0
    for perm_key, m, f, d in perm_deltas:
        if d != 0:
            lines.append(f"  {perm_key:<45} mlm={m:>14,}  full={f:>14,}  delta={d:>+14,}")
            total_perm_delta += d
    lines.append(f"  {'TOTAL delta from permutation keys':<45} {' ':>14}  {' ':>14}  delta={total_perm_delta:>+14,}")
    lines.append("")

    # Stitchability summary from standardized data
    grand = defaultdict(int)
    for db_data in per_db.values():
        for key, val in db_data.items():
            grand[key] += val

    trb_miss = grand.get("trb_missing_gene", 0)
    tra_miss = grand.get("tra_missing_gene", 0)
    trb_total = grand.get("trb_total", 0)
    tra_total = grand.get("tra_total", 0)
    trb_stitch = grand.get("trb_stitchable", 0)
    tra_stitch = grand.get("tra_stitchable", 0)

    lines.append("--- Pre-dedup stitchability (from standardized parquet files) ---")
    lines.append(f"  TRB chains with CDR3:           {trb_total:>15,}")
    lines.append(f"  TRB stitchable (CDR3+V+J):      {trb_stitch:>15,}")
    lines.append(f"  TRB missing gene (CDR3 only):   {trb_miss:>15,} ({trb_miss/trb_total*100:.1f}%)" if trb_total else "")
    lines.append(f"  TRA chains with CDR3:           {tra_total:>15,}")
    lines.append(f"  TRA stitchable (CDR3+V+J):      {tra_stitch:>15,}")
    lines.append(f"  TRA missing gene (CDR3 only):   {tra_miss:>15,} ({tra_miss/tra_total*100:.1f}%)" if tra_total else "")
    lines.append("")

    # Explanation
    lines.append("--- Interpretation ---")
    lines.append(
        "The delta between mlm and mlm_full is caused by TCR chains that have CDR3"
    )
    lines.append(
        "sequences but are missing V and/or J gene annotations, making them"
    )
    lines.append(
        "impossible to stitch into full-length sequences. These records appear in"
    )
    lines.append(
        "mlm (which only needs CDR3) but are excluded from mlm_full (which needs"
    )
    lines.append("the stitched full-length sequence).")
    lines.append("")
    lines.append(
        "Note: The pre-dedup missing-gene counts above are before dedup permutation"
    )
    lines.append(
        "expansion, so they won't match the post-dedup delta exactly. The dedup"
    )
    lines.append(
        "pipeline generates permutation rows (e.g., trb, trb_peptide,"
    )
    lines.append(
        "trb_peptide_mhc_one) from each source row. Rows with unstitchable TCR"
    )
    lines.append(
        "chains lose only the permutations involving that chain, not all permutations."
    )
    lines.append("")

    # Standardization drops (for reference)
    total_dropped = sum(std_dropped.values())
    lines.append("--- Standardization drops (for reference) ---")
    lines.append(f"  Total dropped during standardization:  {total_dropped:>15,}")
    lines.append(
        "  (These affect mlm and mlm_full equally and do NOT contribute to the delta)"
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Verify delta between mlm and mlm_full dedup outputs."
    )
    parser.add_argument(
        "--standardized-dir",
        type=Path,
        default=Path("data/standardized"),
        help="Path to standardized data directory",
    )
    parser.add_argument(
        "--dedup-summary",
        type=Path,
        default=Path("data/deduplicated/DEDUP_SUMMARY.txt"),
        help="Path to DEDUP_SUMMARY.txt",
    )
    parser.add_argument(
        "--standardization-summary",
        type=Path,
        default=Path("data/standardized/STANDARDIZATION_SUMMARY.txt"),
        help="Path to STANDARDIZATION_SUMMARY.txt",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, multiprocessing.cpu_count() // 2),
        help="Number of parallel workers (default: half of CPUs)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file path (default: print to stdout)",
    )
    args = parser.parse_args()

    # Phase 1: Classify stitchability
    print("Phase 1: Scanning standardized parquet files...")
    per_db = classify_all_files(args.standardized_dir, args.num_workers)

    # Phase 2: Parse dedup summary
    print("\nPhase 2: Parsing DEDUP_SUMMARY.txt...")
    dedup = parse_dedup_summary(args.dedup_summary)

    # Phase 3: Parse standardization summary
    print("\nPhase 3: Parsing STANDARDIZATION_SUMMARY.txt...")
    std_total, std_manifest, std_dropped = parse_standardization_summary(
        args.standardization_summary
    )

    # Phase 4: Generate report
    print("\nPhase 4: Generating reconciliation report...")
    report_parts = [
        format_stitchability_table(per_db),
        format_missing_gene_breakdown(per_db),
        format_reconciliation(per_db, dedup, std_total, std_dropped),
    ]
    full_report = "\n\n".join(report_parts) + "\n"

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(full_report)
        print(f"\nReport written to {args.output}")
    else:
        print("\n" + full_report)


if __name__ == "__main__":
    main()
