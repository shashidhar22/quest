#!/usr/bin/env python3
"""Post-process standardized parquet files to fix gene names and re-annotate CDR1/CDR2.

Reads existing standardized parquet files, corrects V/D/J gene names against the
IMGT reference (from stitchr's FASTA data), then re-derives CDR1/CDR2 sequences
from the corrected V-genes using tidytcells.

Common fixes:
  - Dash-as-allele: TRBV27-1 → TRBV27*01 (TRBV27 has no subgroups)
  - *00 alleles: TRAV20*00 → TRAV20*01
  - CDR1/CDR2 re-annotation from corrected V-genes

Usage:
    # Dry run: report corrections without writing
    python scripts/data_processing/fix_gene_names.py --dry-run

    # Fix specific database(s)
    python scripts/data_processing/fix_gene_names.py --db immunecode vdjdb

    # Fix all databases, custom paths
    python scripts/data_processing/fix_gene_names.py \
        --input-dir /data/standardized_again \
        --output-dir /data/standardized_fixed

    # Fix in place (default)
    python scripts/data_processing/fix_gene_names.py --db immunecode
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quest.data.standardization import (
    _load_imgt_genes,
    _validate_against_imgt,
    _lookup_cdr_from_vgene,
)

logger = logging.getLogger(__name__)

GENE_COLS = ["trav_gene", "traj_gene", "trad_gene", "trbv_gene", "trbj_gene", "trbd_gene"]
V_GENE_CDR_MAP = {
    "trav_gene": ("tra_cdr1", "tra_cdr2"),
    "trbv_gene": ("trb_cdr1", "trb_cdr2"),
}

# Expected chain prefix for each gene column — genes that don't match are nulled out
CHAIN_PREFIX = {
    "trav_gene": "TRAV",
    "traj_gene": "TRAJ",
    "trad_gene": "TRAD",
    "trbv_gene": "TRBV",
    "trbj_gene": "TRBJ",
    "trbd_gene": "TRBD",
}
# When a V-gene is nulled, also null these related columns
V_GENE_RELATED_COLS = {
    "trav_gene": ("tra_cdr1", "tra_cdr2"),
    "trbv_gene": ("trb_cdr1", "trb_cdr2"),
}


def _build_correction_map(unique_values: list[str]) -> dict[str, str]:
    """Build a map of gene name corrections for a set of unique values."""
    corrections = {}
    for val in unique_values:
        if not val:
            continue
        corrected = _validate_against_imgt(val)
        if corrected != val:
            corrections[val] = corrected
    return corrections


def process_file(
    input_path: str,
    output_path: str,
    dry_run: bool = False,
) -> dict:
    """Process a single parquet file: fix gene names, re-annotate CDR1/CDR2.

    Returns stats dict with correction counts.
    """
    table = pq.read_table(input_path)
    df = table.to_pandas()
    n_rows = len(df)

    stats = {
        "file": Path(input_path).name,
        "rows": n_rows,
        "genes_corrected": 0,
        "cdr_updated": 0,
        "wrong_chain_nulled": 0,
        "corrections": {},  # {old: new} for reporting
        "wrong_chain": {},  # {col: {gene: count}} for reporting
    }

    # Ensure IMGT reference is loaded
    _load_imgt_genes()

    # Phase 0: Null out genes with wrong chain prefix (e.g. TRDV in trav_gene)
    for col, expected_prefix in CHAIN_PREFIX.items():
        if col not in df.columns:
            continue

        # Find non-empty values that don't start with the expected prefix
        mask = (df[col] != "") & ~df[col].str.startswith(expected_prefix)
        n_bad = int(mask.sum())
        if n_bad == 0:
            continue

        # Record what's being nulled for reporting
        bad_vals = df.loc[mask, col].value_counts().to_dict()
        stats["wrong_chain"][col] = bad_vals
        stats["wrong_chain_nulled"] += n_bad

        if not dry_run:
            df.loc[mask, col] = ""
            # Also null related CDR columns when a V-gene is nulled
            if col in V_GENE_RELATED_COLS:
                for related_col in V_GENE_RELATED_COLS[col]:
                    if related_col in df.columns:
                        df.loc[mask, related_col] = ""

    # Phase 1: Fix gene names
    v_genes_changed = set()  # track which V-gene columns had corrections

    for col in GENE_COLS:
        if col not in df.columns:
            continue

        unique_vals = df[col].unique().tolist()
        corrections = _build_correction_map(unique_vals)

        if corrections:
            stats["corrections"].update(corrections)
            # Count rows affected
            affected_mask = df[col].isin(corrections.keys())
            stats["genes_corrected"] += int(affected_mask.sum())

            if not dry_run:
                df[col] = df[col].replace(corrections)

            if col in V_GENE_CDR_MAP:
                v_genes_changed.add(col)

    # Phase 2: Re-annotate CDR1/CDR2 for corrected V-genes
    if not dry_run and v_genes_changed:
        for v_col in v_genes_changed:
            cdr1_col, cdr2_col = V_GENE_CDR_MAP[v_col]
            if cdr1_col not in df.columns or cdr2_col not in df.columns:
                continue

            unique_genes = df[v_col].unique()
            cdr1_map = {}
            cdr2_map = {}
            for gene in unique_genes:
                if gene:
                    cdrs = _lookup_cdr_from_vgene(gene)
                    cdr1_map[gene] = cdrs["cdr1"]
                    cdr2_map[gene] = cdrs["cdr2"]
                else:
                    cdr1_map[gene] = ""
                    cdr2_map[gene] = ""

            old_cdr1_empty = (df[cdr1_col] == "").sum()
            df[cdr1_col] = df[v_col].map(cdr1_map)
            df[cdr2_col] = df[v_col].map(cdr2_map)
            new_cdr1_empty = (df[cdr1_col] == "").sum()
            stats["cdr_updated"] += int(old_cdr1_empty - new_cdr1_empty)

    # Phase 3: Write output
    if not dry_run and (stats["genes_corrected"] > 0 or stats["wrong_chain_nulled"] > 0):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out_table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(out_table, output_path, compression="snappy")

    return stats


def process_database(
    db_name: str,
    input_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
    workers: int = 1,
) -> dict:
    """Process all parquet files for a single database."""
    db_input = input_dir / db_name
    db_output = output_dir / db_name

    parquet_files = sorted(db_input.glob("part_*.parquet"))
    if not parquet_files:
        logger.warning(f"No parquet files found in {db_input}")
        return {"db": db_name, "files": 0}

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {db_name}: {len(parquet_files)} file(s)")
    logger.info(f"{'='*60}")
    t0 = time.time()

    total = {
        "db": db_name,
        "files": len(parquet_files),
        "rows": 0,
        "genes_corrected": 0,
        "cdr_updated": 0,
        "wrong_chain_nulled": 0,
        "all_corrections": {},
        "all_wrong_chain": {},
    }

    file_tasks = []
    for pf in parquet_files:
        out_path = str(pf) if (input_dir == output_dir) else str(db_output / pf.name)
        file_tasks.append((str(pf), out_path, dry_run))

    if workers <= 1 or len(file_tasks) <= 1:
        for task in file_tasks:
            stats = process_file(*task)
            total["rows"] += stats["rows"]
            total["genes_corrected"] += stats["genes_corrected"]
            total["cdr_updated"] += stats["cdr_updated"]
            total["all_corrections"].update(stats["corrections"])
            total["wrong_chain_nulled"] += stats["wrong_chain_nulled"]
            for col, vals in stats["wrong_chain"].items():
                col_dict = total["all_wrong_chain"].setdefault(col, {})
                for gene, cnt in vals.items():
                    col_dict[gene] = col_dict.get(gene, 0) + cnt
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(file_tasks))) as pool:
            futures = [pool.submit(process_file, *t) for t in file_tasks]
            for future in as_completed(futures):
                stats = future.result()
                total["rows"] += stats["rows"]
                total["genes_corrected"] += stats["genes_corrected"]
                total["cdr_updated"] += stats["cdr_updated"]
                total["all_corrections"].update(stats["corrections"])
                total["wrong_chain_nulled"] += stats["wrong_chain_nulled"]
                for col, vals in stats["wrong_chain"].items():
                    col_dict = total["all_wrong_chain"].setdefault(col, {})
                    for gene, cnt in vals.items():
                        col_dict[gene] = col_dict.get(gene, 0) + cnt

    elapsed = time.time() - t0

    # Report wrong-chain genes nulled
    if total["all_wrong_chain"]:
        logger.info(f"\n  Wrong-chain genes nulled for {db_name}:")
        for col, vals in sorted(total["all_wrong_chain"].items()):
            for gene, cnt in sorted(vals.items(), key=lambda x: -x[1]):
                logger.info(f"    {col}: {gene} ({cnt:,} rows)")

    # Report corrections
    if total["all_corrections"]:
        logger.info(f"\n  Gene corrections for {db_name}:")
        for old, new in sorted(total["all_corrections"].items()):
            logger.info(f"    {old} -> {new}")

    logger.info(
        f"\n  {db_name} SUMMARY ({elapsed:.1f}s):\n"
        f"    Files:           {total['files']:>10,}\n"
        f"    Rows:            {total['rows']:>10,}\n"
        f"    Wrong-chain:     {total['wrong_chain_nulled']:>10,} (genes nulled)\n"
        f"    Genes corrected: {total['genes_corrected']:>10,} (cell values)\n"
        f"    CDR updated:     {total['cdr_updated']:>10,} (new CDR1/2 filled)"
    )

    return total


def main():
    parser = argparse.ArgumentParser(
        description="Fix gene names and re-annotate CDR1/CDR2 in standardized parquet files"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data/standardized"),
        help="Input directory with per-database subdirectories",
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
        help="Report corrections without writing files.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers per database (default: 1)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = args.input_dir if args.output_dir is None else args.output_dir
    in_place = args.output_dir is None

    if in_place and not args.dry_run:
        logger.info("Writing in place (overwriting input files)")

    # Ensure IMGT reference loads
    _load_imgt_genes()
    from quest.data.standardization import _IMGT_GENE_BASES
    if not _IMGT_GENE_BASES:
        logger.error("IMGT gene reference not available. Is stitchr installed?")
        sys.exit(1)
    logger.info(f"IMGT reference: {len(_IMGT_GENE_BASES)} gene bases loaded")

    if args.db:
        databases = args.db
    else:
        databases = sorted(
            d.name for d in args.input_dir.iterdir() if d.is_dir()
        )

    logger.info(f"Databases: {databases}")

    grand = {
        "rows": 0, "genes_corrected": 0, "cdr_updated": 0, "wrong_chain_nulled": 0,
    }
    t_start = time.time()

    for db in databases:
        db_stats = process_database(db, args.input_dir, output_dir, args.dry_run, args.workers)
        grand["rows"] += db_stats.get("rows", 0)
        grand["genes_corrected"] += db_stats.get("genes_corrected", 0)
        grand["cdr_updated"] += db_stats.get("cdr_updated", 0)
        grand["wrong_chain_nulled"] += db_stats.get("wrong_chain_nulled", 0)

    elapsed = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"ALL DATABASES COMPLETE ({elapsed:.1f}s)")
    logger.info(f"{'='*60}")
    logger.info(
        f"  Total rows:           {grand['rows']:>12,}\n"
        f"  Wrong-chain nulled:   {grand['wrong_chain_nulled']:>12,}\n"
        f"  Genes corrected:      {grand['genes_corrected']:>12,}\n"
        f"  CDR1/2 newly filled:  {grand['cdr_updated']:>12,}"
    )


if __name__ == "__main__":
    main()
