#!/usr/bin/env python3
"""Diagnostic audit of every file in data/studies/.

Read-only scan that classifies each file by extension, format detection result,
column map success, chain type, and whether it would be parsed by the studies
standardizer.  Produces a per-study and per-format summary to help identify
format gaps causing silent data loss.

Usage:
    python scripts/analysis/audit_studies_parsing.py \
        --source-dir data/studies \
        --output data/standardized/studies/PARSING_AUDIT.txt
"""

import argparse
import gzip
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# Supported extensions for discovery (mirrors studies.py + proposed additions)
DATA_EXTENSIONS = {
    ".csv", ".tsv", ".txt", ".csv.gz", ".tsv.gz", ".txt.gz", ".structure",
}


def _count_lines(fpath: Path) -> int:
    """Count lines in a file (handles .gz)."""
    try:
        opener = gzip.open if str(fpath).endswith(".gz") else open
        count = 0
        with opener(fpath, "rt", errors="replace") as f:
            for _ in f:
                count += 1
        return count
    except Exception:
        return -1


def _read_header(fpath: Path) -> tuple:
    """Read header from a data file.

    Returns (columns_list, separator, skip_rows, format_note).
    """
    suffix = fpath.suffix.lower()
    name_lower = fpath.name.lower()
    is_gz = name_lower.endswith(".gz")

    # Determine separator
    if is_gz:
        base_name = name_lower[:-3]  # strip .gz
        base_suffix = Path(base_name).suffix
    else:
        base_suffix = suffix

    sep = "\t" if base_suffix in (".tsv", ".txt", ".structure") else ","
    skip_rows = 0
    format_note = ""

    try:
        compression = "gzip" if is_gz else None

        # Check for MiTCR metadata header
        opener = gzip.open if is_gz else open
        with opener(fpath, "rt", errors="replace") as f:
            first_line = f.readline()

        if first_line.startswith("MiTCR"):
            skip_rows = 1
            format_note = "MiTCR metadata header detected"

        header_df = pd.read_csv(
            fpath, sep=sep, nrows=0, dtype=str,
            compression=compression, skiprows=skip_rows,
        )
        return header_df.columns.tolist(), sep, skip_rows, format_note
    except Exception as e:
        return [], sep, skip_rows, f"read_error: {e}"


def _detect_format_extended(columns: list) -> str:
    """Extended format detection matching studies.py + proposed new formats."""
    col_set = set(c.lower() for c in columns)

    # 10x with chain column
    if "barcode" in col_set and "chain" in col_set:
        return "10x"

    # AIRR
    if "junction_aa" in col_set and "v_call" in col_set:
        return "airr"

    # immunoSEQ v1 (snake_case)
    if "rearrangement" in col_set and "amino_acid" in col_set:
        return "immunoseq"

    # immunoSEQ v2 (camelCase)
    if "aminoacid" in col_set and "nucleotide" in col_set:
        return "immunoseq_v2"

    # MiTCR full export
    if any("cdr3 amino acid" in c.lower() for c in columns):
        return "mitcr"

    # MiXCR clone format
    if "cdr3aa" in col_set and "cdr3nt" in col_set:
        return "mixcr_clone"

    # BGI .structure format
    if "cdr3(aa)" in col_set and "v_ref" in col_set:
        return "bgi_structure"

    # 10X clonotype format
    if "cdr3s_aa" in col_set:
        return "10x_clonotype"

    # Generic fallback
    if "cdr3_aa" in col_set or "cdr3" in col_set:
        return "generic"

    return "unrecognized"


def _classify_chain(fpath: Path, columns: list) -> str:
    """Determine chain type from file path or column hints."""
    path_str = str(fpath).lower()

    # From directory name
    if "_tra/" in path_str or "_tra\\" in path_str or "bulk_survey_tra" in path_str:
        return "TRA"
    if "_trb/" in path_str or "_trb\\" in path_str or "bulk_survey_trb" in path_str:
        return "TRB"

    # From filename
    name_lower = fpath.stem.lower()
    if "clones_tra" in name_lower or "_alpha" in name_lower:
        return "TRA"
    if "clones_trb" in name_lower or "_beta" in name_lower:
        return "TRB"

    # From columns
    col_lower = set(c.lower() for c in columns)
    if "chain" in col_lower:
        return "TRA+TRB (chain col)"
    if "locus" in col_lower:
        return "mixed (locus col)"
    if "cdr3s_aa" in col_lower:
        return "TRA+TRB (paired)"

    return "unknown"


def _classify_content(fpath: Path, columns: list) -> str:
    """Classify file content type: TCR / BCR / GEX / metadata / other."""
    col_lower = set(c.lower() for c in columns)
    name_lower = fpath.name.lower()

    # GEX indicators
    gex_indicators = {"gene", "feature_name", "gene_symbols", "counts", "barcodes.tsv"}
    if any(g in name_lower for g in ("gene_expression", "gex", "rna", "features")):
        return "GEX"
    if col_lower & {"gene", "feature_name", "gene_ids"}:
        return "GEX"

    # Metadata
    if any(m in name_lower for m in ("metadata", "manifest", "readme", "log", "summary")):
        return "metadata"

    # BCR/IG indicators
    bcr_cols = {"igh", "igk", "igl", "ig_heavy", "ig_light"}
    if col_lower & bcr_cols:
        return "BCR"

    # TCR indicators
    tcr_cols = {
        "cdr3", "cdr3_aa", "junction_aa", "aminoacid", "cdr3aa",
        "cdr3 amino acid sequence", "cdr3s_aa", "v_call", "v_gene",
        "trav", "trbv", "vgenename", "v_ref",
    }
    if col_lower & tcr_cols:
        return "TCR"

    # Check for locus column that might contain both
    if "locus" in col_lower:
        return "TCR (locus-mixed)"

    return "other"


def audit_study(study_dir: Path) -> list:
    """Audit all files in a single study directory."""
    results = []

    # Collect all candidate data files
    data_files = []
    for f in sorted(study_dir.rglob("*")):
        if not f.is_file():
            continue
        name_lower = f.name.lower()
        suffix = f.suffix.lower()
        # Check multi-part extensions
        if name_lower.endswith((".csv.gz", ".tsv.gz", ".txt.gz")):
            data_files.append(f)
        elif suffix in (".csv", ".tsv", ".txt", ".structure"):
            data_files.append(f)
        # Also note non-data files
        elif suffix in (".parquet", ".h5", ".h5ad", ".mtx", ".bam", ".fastq"):
            results.append({
                "study": study_dir.name,
                "file": str(f.relative_to(study_dir)),
                "extension": suffix,
                "size_mb": round(f.stat().st_size / 1024**2, 2),
                "rows": -1,
                "format": "binary",
                "content_type": "binary/non-text",
                "chain": "N/A",
                "col_map_ok": False,
                "columns_sample": "",
                "note": "",
            })

    for fpath in data_files:
        size_mb = round(fpath.stat().st_size / 1024**2, 2)
        name_lower = fpath.name.lower()

        # Extension
        if name_lower.endswith(".gz"):
            ext = "." + ".".join(fpath.name.split(".")[-2:])
        else:
            ext = fpath.suffix

        # Row count (header included)
        row_count = _count_lines(fpath)

        # Read header
        columns, sep, skip_rows, note = _read_header(fpath)

        if not columns:
            results.append({
                "study": study_dir.name,
                "file": str(fpath.relative_to(study_dir)),
                "extension": ext,
                "size_mb": size_mb,
                "rows": row_count,
                "format": "unreadable",
                "content_type": "unknown",
                "chain": "unknown",
                "col_map_ok": False,
                "columns_sample": "",
                "note": note,
            })
            continue

        fmt = _detect_format_extended(columns)
        content_type = _classify_content(fpath, columns)
        chain = _classify_chain(fpath, columns)

        # Check if current standardizer can parse it
        # (simulates _build_column_map returning non-empty)
        from scripts.data_processing.standardize.studies import (
            _build_column_map,
            _detect_format,
        )
        current_fmt = _detect_format(columns)
        current_col_map = _build_column_map(columns, current_fmt)
        col_map_ok = bool(current_col_map)

        results.append({
            "study": study_dir.name,
            "file": str(fpath.relative_to(study_dir)),
            "extension": ext,
            "size_mb": size_mb,
            "rows": row_count,
            "format": fmt,
            "format_current": current_fmt,
            "content_type": content_type,
            "chain": chain,
            "col_map_ok": col_map_ok,
            "columns_sample": ", ".join(columns[:8]),
            "note": note,
        })

    return results


def format_report(all_results: list) -> str:
    """Format the full audit report."""
    lines = []
    lines.append("=" * 120)
    lines.append("STUDIES PARSING AUDIT")
    lines.append("=" * 120)
    lines.append("")

    # ---- Overall summary ----
    total_files = len(all_results)
    parseable = sum(1 for r in all_results if r["col_map_ok"])
    unparseable_tcr = sum(
        1 for r in all_results
        if not r["col_map_ok"] and r.get("content_type", "").startswith("TCR")
    )
    total_rows = sum(r["rows"] for r in all_results if r["rows"] > 0)
    parseable_rows = sum(r["rows"] for r in all_results if r["col_map_ok"] and r["rows"] > 0)
    unparseable_tcr_rows = sum(
        r["rows"] for r in all_results
        if not r["col_map_ok"] and r.get("content_type", "").startswith("TCR") and r["rows"] > 0
    )

    lines.append("1. OVERALL SUMMARY")
    lines.append("-" * 60)
    lines.append(f"  Total files scanned:          {total_files:>10,}")
    lines.append(f"  Currently parseable:          {parseable:>10,}")
    lines.append(f"  Unparseable TCR files:        {unparseable_tcr:>10,}")
    lines.append(f"  Total rows (all files):       {total_rows:>10,}")
    lines.append(f"  Parseable rows:               {parseable_rows:>10,}")
    lines.append(f"  Unparseable TCR rows:         {unparseable_tcr_rows:>10,}")
    lines.append("")

    # ---- Format breakdown ----
    fmt_counter: Counter = Counter()
    fmt_rows: Counter = Counter()
    for r in all_results:
        fmt = r.get("format", "unknown")
        fmt_counter[fmt] += 1
        if r["rows"] > 0:
            fmt_rows[fmt] += r["rows"]

    lines.append("2. FORMAT BREAKDOWN")
    lines.append("-" * 80)
    lines.append(f"  {'Format':<20} {'Files':>8} {'Rows':>15} {'Currently Parsed':>18}")
    lines.append("  " + "-" * 64)
    for fmt, count in fmt_counter.most_common():
        rows = fmt_rows.get(fmt, 0)
        # Check if current standardizer handles this format
        parsed = sum(
            1 for r in all_results
            if r.get("format") == fmt and r["col_map_ok"]
        )
        lines.append(f"  {fmt:<20} {count:>8,} {rows:>15,} {parsed:>8,}/{count:<8,}")
    lines.append("")

    # ---- Content type breakdown ----
    content_counter: Counter = Counter()
    content_rows: Counter = Counter()
    for r in all_results:
        ct = r.get("content_type", "unknown")
        content_counter[ct] += 1
        if r["rows"] > 0:
            content_rows[ct] += r["rows"]

    lines.append("3. CONTENT TYPE BREAKDOWN")
    lines.append("-" * 60)
    lines.append(f"  {'Content Type':<25} {'Files':>8} {'Rows':>15}")
    lines.append("  " + "-" * 50)
    for ct, count in content_counter.most_common():
        lines.append(f"  {ct:<25} {count:>8,} {content_rows.get(ct, 0):>15,}")
    lines.append("")

    # ---- Extension breakdown ----
    ext_counter: Counter = Counter()
    ext_rows: Counter = Counter()
    for r in all_results:
        ext = r.get("extension", "unknown")
        ext_counter[ext] += 1
        if r["rows"] > 0:
            ext_rows[ext] += r["rows"]

    lines.append("4. EXTENSION BREAKDOWN")
    lines.append("-" * 60)
    lines.append(f"  {'Extension':<15} {'Files':>8} {'Rows':>15}")
    lines.append("  " + "-" * 40)
    for ext, count in ext_counter.most_common():
        lines.append(f"  {ext:<15} {count:>8,} {ext_rows.get(ext, 0):>15,}")
    lines.append("")

    # ---- Unparseable TCR files (format gap) ----
    unparseable = [
        r for r in all_results
        if not r["col_map_ok"] and r.get("content_type", "").startswith("TCR")
    ]
    if unparseable:
        lines.append("5. UNPARSEABLE TCR FILES (FORMAT GAPS)")
        lines.append("-" * 120)
        lines.append(
            f"  {'Study':<20} {'File':<50} {'Format':<16} "
            f"{'Rows':>10} {'Columns (first 5)'}"
        )
        lines.append("  " + "-" * 115)

        # Sort by rows descending
        unparseable.sort(key=lambda r: r.get("rows", 0), reverse=True)
        for r in unparseable[:100]:  # Limit to top 100
            cols_sample = r.get("columns_sample", "")[:60]
            lines.append(
                f"  {r['study']:<20} {r['file']:<50} {r.get('format', '?'):<16} "
                f"{r.get('rows', 0):>10,} {cols_sample}"
            )
        if len(unparseable) > 100:
            lines.append(f"  ... and {len(unparseable) - 100} more files")
        lines.append("")

    # ---- Per-study summary ----
    study_stats: dict = defaultdict(lambda: {
        "files": 0, "parseable": 0, "rows": 0, "parseable_rows": 0,
    })
    for r in all_results:
        s = study_stats[r["study"]]
        s["files"] += 1
        if r["col_map_ok"]:
            s["parseable"] += 1
        if r["rows"] > 0:
            s["rows"] += r["rows"]
            if r["col_map_ok"]:
                s["parseable_rows"] += r["rows"]

    lines.append("6. PER-STUDY SUMMARY")
    lines.append("-" * 90)
    lines.append(
        f"  {'Study':<25} {'Files':>8} {'Parseable':>10} "
        f"{'Total Rows':>14} {'Parseable Rows':>16} {'Loss %':>8}"
    )
    lines.append("  " + "-" * 85)

    for study in sorted(study_stats.keys()):
        s = study_stats[study]
        loss = ((s["rows"] - s["parseable_rows"]) / s["rows"] * 100
                if s["rows"] > 0 else 0)
        lines.append(
            f"  {study:<25} {s['files']:>8,} {s['parseable']:>10,} "
            f"{s['rows']:>14,} {s['parseable_rows']:>16,} {loss:>7.1f}%"
        )
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Audit all files in data/studies/ for parsing coverage."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/studies"),
        help="Path to studies data directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/standardized/studies/PARSING_AUDIT.txt"),
        help="Output report file path",
    )
    args = parser.parse_args()

    if not args.source_dir.exists():
        print(f"Source directory not found: {args.source_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect all study directories
    study_dirs = sorted(
        d for d in args.source_dir.iterdir()
        if d.is_dir() and not d.name.startswith((".", "logs", "geo"))
    )

    print(f"Auditing {len(study_dirs)} study directories in {args.source_dir}")

    all_results = []
    for study_dir in tqdm(study_dirs, desc="Auditing studies"):
        results = audit_study(study_dir)
        all_results.extend(results)

    print(f"Scanned {len(all_results)} files total")

    # Generate report
    report = format_report(all_results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n")
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    from tqdm import tqdm
    main()
