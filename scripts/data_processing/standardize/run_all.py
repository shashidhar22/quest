#!/usr/bin/env python3
"""Orchestrator for running all database standardizers.

Runs all standardizers in optimal order:
- Phase 1: Small DBs in parallel (all fit in memory)
- Phase 2: Large DBs sequentially (memory-intensive streaming)

Usage:
    python scripts/data_processing/standardize/run_all.py
    python scripts/data_processing/standardize/run_all.py --db batman
    python scripts/data_processing/standardize/run_all.py --skip immuneaccess adc
    python scripts/data_processing/standardize/run_all.py --force
    python scripts/data_processing/standardize/run_all.py --verify-only
"""

import argparse
import logging
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.data_processing.standardize.batman import BatmanStandardizer
from scripts.data_processing.standardize.tadb import TadbStandardizer
from scripts.data_processing.standardize.mcpas import McpasStandardizer
from scripts.data_processing.standardize.vdjdb import VdjdbStandardizer
from scripts.data_processing.standardize.cedar import CedarStandardizer
from scripts.data_processing.standardize.iedb import IedbStandardizer
from scripts.data_processing.standardize.iedb_pmhc import IedbPmhcStandardizer
from scripts.data_processing.standardize.cedar_pmhc import CedarPmhcStandardizer
from scripts.data_processing.standardize.immunecode import ImmunecodeStandardizer
from scripts.data_processing.standardize.trait import TraitStandardizer
from scripts.data_processing.standardize.ots import OtsStandardizer
from scripts.data_processing.standardize.netmhcpan import NetmhcpanStandardizer
from scripts.data_processing.standardize.tcrdb import TcrdbStandardizer
from scripts.data_processing.standardize.immuneaccess import ImmuneaccessStandardizer
from scripts.data_processing.standardize.adc import AdcStandardizer
from scripts.data_processing.standardize.studies import StudiesStandardizer
from scripts.data_processing.standardize.imgthla import ImgthlaStandardizer
from scripts.data_processing.standardize.rcc_atlas import RccAtlasStandardizer
from quest.data.standardization import clear_norm_cache
from scripts.data_processing.fix_gene_names import process_database as fix_genes_for_db

logger = logging.getLogger(__name__)

# Default base paths
DEFAULT_DB_DIR = "data/databases"
DEFAULT_STUDIES_DIR = "data/studies"
DEFAULT_OUTPUT_DIR = "data/standardized"

# Small databases (run in parallel)
SMALL_DBS = {
    "batman": (BatmanStandardizer, "BATMAN"),
    "tadb": (TadbStandardizer, "TaDB"),
    "mcpas": (McpasStandardizer, "McPAS-TCR"),
    "vdjdb": (VdjdbStandardizer, "vdjdb"),
    "cedar": (CedarStandardizer, "CEDAR"),
    "iedb": (IedbStandardizer, "IEDB"),
    "iedb_pmhc": (IedbPmhcStandardizer, "IEDB"),
    "cedar_pmhc": (CedarPmhcStandardizer, "CEDAR"),
    "imgthla": (ImgthlaStandardizer, "IMGTHLA"),
    "rcc_atlas": (RccAtlasStandardizer, "RCC_ATLAS"),
}

# Large databases (run sequentially — memory-intensive streaming)
LARGE_DBS = {
    "trait": (TraitStandardizer, "trait"),
    "immunecode": (ImmunecodeStandardizer, "immuneCODE"),
    "ots": (OtsStandardizer, "OTS"),
    "netmhcpan": (NetmhcpanStandardizer, "NetMHCPan"),
    "tcrdb": (TcrdbStandardizer, "tcrdb"),
    "immuneaccess": (ImmuneaccessStandardizer, "immuneACCESS"),
    "adc": (AdcStandardizer, "adc"),
    "studies": (StudiesStandardizer, None),  # Special path handling
}

ALL_DBS = {**SMALL_DBS, **LARGE_DBS}


def _run_single(db_name: str, db_dir: str, output_dir: str, force: bool, studies_dir: str = DEFAULT_STUDIES_DIR, stitch: bool = False, parallel_workers: int = 0, hla_dir: str = "") -> dict:
    """Run a single standardizer (used for parallel execution).

    Args:
        parallel_workers: If > 0, use parallel_run with this many workers
            for standardizers that support it (have get_file_list()).
    """
    warnings.filterwarnings("ignore")
    cls, subdir = ALL_DBS[db_name]

    if db_name == "studies":
        source_dir = Path(studies_dir)
    else:
        source_dir = Path(db_dir) / subdir

    out_dir = Path(output_dir) / db_name

    try:
        standardizer = cls(source_dir=source_dir, output_dir=out_dir, stitch=stitch, hla_dir=hla_dir)
        if parallel_workers > 0 and standardizer.get_file_list():
            return standardizer.parallel_run(workers=parallel_workers, force=force)
        return standardizer.run(force=force)
    except Exception as e:
        return {"status": "error", "name": db_name, "error": str(e)}


def _verify_single(db_name: str, output_dir: str, studies_dir: str = DEFAULT_STUDIES_DIR) -> dict:
    """Verify a single standardizer's output."""
    cls, subdir = ALL_DBS[db_name]
    out_dir = Path(output_dir) / db_name

    # Source dir doesn't matter for verify
    if db_name == "studies":
        source_dir = Path(studies_dir)
    else:
        source_dir = Path(DEFAULT_DB_DIR) / subdir

    try:
        standardizer = cls(source_dir=source_dir, output_dir=out_dir)
        return standardizer.verify()
    except Exception as e:
        return {"name": db_name, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Run all database standardizers"
    )
    parser.add_argument(
        "--db",
        nargs="+",
        help="Run only specific database(s)",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        help="Skip specific database(s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if source files haven't changed",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check existing output without re-running",
    )
    parser.add_argument(
        "--db-dir",
        default=DEFAULT_DB_DIR,
        help="Base directory for database sources",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Base directory for standardized output",
    )
    parser.add_argument(
        "--studies-dir",
        default=DEFAULT_STUDIES_DIR,
        help="Directory for study sources",
    )
    parser.add_argument(
        "--stitch",
        action="store_true",
        help="Generate full-length TCR sequences via stitchr",
    )
    parser.add_argument(
        "--hla-dir",
        default="",
        help="Path to IMGT/HLA fasta directory for MHC allele→sequence resolution",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Max parallel workers for small DBs (Phase 1)",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=0,
        help="Per-database file-level parallelism for large DBs. "
             "0 = serial (default). Recommended: 16-32 on high-core machines.",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Determine which DBs to run
    if args.db:
        dbs_to_run = [d for d in args.db if d in ALL_DBS]
        invalid = [d for d in args.db if d not in ALL_DBS]
        if invalid:
            print(f"Unknown databases: {invalid}")
            print(f"Available: {sorted(ALL_DBS.keys())}")
            sys.exit(1)
    else:
        dbs_to_run = [d for d in ALL_DBS if d not in args.skip]

    # Verify-only mode
    if args.verify_only:
        print(f"\n{'DB':<15s} {'Rows':>12s} {'Files':>6s} {'Status':<10s}")
        print("-" * 50)
        verify_results = []
        for db_name in dbs_to_run:
            result = _verify_single(db_name, args.output_dir)
            verify_results.append(result)
            if "error" in result:
                print(f"{db_name:<15s} {'':>12s} {'':>6s} {'ERROR':<10s}  {result['error']}")
            else:
                print(
                    f"{db_name:<15s} {result['total_rows']:>12,} "
                    f"{result['parquet_files']:>6d} {'OK':<10s}"
                )

        # Molecule combination catalog
        combo_keys = [
            'tra_only', 'trb_only', 'tra_trb', 'peptide_only',
            'pep_mhcI', 'pep_mhcII', 'tcr_pep_mhcI', 'tcr_pep_mhcII',
            'tcr_peptide', 'other',
        ]
        combo_labels = [
            'tra', 'trb', 'tra+trb', 'pep',
            'pep+mhcI', 'pep+mhcII', 'tcr+pep+mhcI', 'tcr+pep+mhcII',
            'tcr+pep', 'other',
        ]
        col_w = 14  # column width for combo values

        valid_results = [r for r in verify_results if "error" not in r and "combo_counts" in r]
        if valid_results:
            print(f"\nMolecule Combination Catalog")
            header = f"{'DB':<15s}" + "".join(f"{l:>{col_w}s}" for l in combo_labels) + f"{'total':>{col_w}s}"
            print(header)
            print("-" * len(header))

            totals = {k: 0 for k in combo_keys}
            grand_total = 0

            for r in sorted(valid_results, key=lambda x: x["name"]):
                combos = r["combo_counts"]
                row_total = sum(combos.values())
                line = f"{r['name']:<15s}"
                for k in combo_keys:
                    v = combos[k]
                    totals[k] += v
                    line += f"{v:>{col_w},}"
                line += f"{row_total:>{col_w},}"
                grand_total += row_total
                print(line)

            print("-" * len(header))
            line = f"{'TOTAL':<15s}"
            for k in combo_keys:
                line += f"{totals[k]:>{col_w},}"
            line += f"{grand_total:>{col_w},}"
            print(line)

            # Unique molecule counts
            unique_keys = ['tra', 'trb', 'peptide', 'mhc_one']
            unique_labels = ['tra', 'trb', 'peptide', 'mhc_one']
            print(f"\nUnique Molecule Counts")
            u_header = f"{'DB':<15s}" + "".join(f"{l:>{col_w}s}" for l in unique_labels)
            print(u_header)
            print("-" * len(u_header))
            for r in sorted(valid_results, key=lambda x: x["name"]):
                uc = r.get("unique_counts", {})
                line = f"{r['name']:<15s}"
                for k in unique_keys:
                    line += f"{uc.get(k, 0):>{col_w},}"
                print(line)

        return

    # Run standardizers
    print(f"\nStandardizing {len(dbs_to_run)} databases...")
    print(f"Output: {args.output_dir}/")
    print()

    results = []
    overall_start = time.time()

    # Overall progress bar across all databases
    overall_pbar = tqdm(
        total=len(dbs_to_run),
        desc="Overall",
        unit="db",
        position=0,
        dynamic_ncols=True,
    )

    # Phase 1: Small DBs in parallel
    small_dbs = [d for d in dbs_to_run if d in SMALL_DBS]
    if small_dbs:
        overall_pbar.set_postfix(phase="small DBs (parallel)")
        with ProcessPoolExecutor(max_workers=min(args.workers, len(small_dbs))) as pool:
            futures = {
                pool.submit(
                    _run_single, db_name, args.db_dir, args.output_dir, args.force, args.studies_dir, args.stitch, 0, args.hla_dir
                ): db_name
                for db_name in small_dbs
            }
            for future in as_completed(futures):
                db_name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    status = result.get("status", "unknown")
                    if status == "completed":
                        overall_pbar.write(
                            f"  {db_name}: {result['rows']:,} rows "
                            f"({result.get('dropped', 0):,} dropped) "
                            f"in {result.get('elapsed_s', 0):.1f}s"
                        )
                    elif status == "skipped":
                        overall_pbar.write(f"  {db_name}: up-to-date (skipped)")
                    else:
                        overall_pbar.write(
                            f"  {db_name}: {status} - {result.get('error', '')}"
                        )
                except Exception as e:
                    overall_pbar.write(f"  {db_name}: ERROR - {e}")
                    results.append({"status": "error", "name": db_name, "error": str(e)})
                overall_pbar.update(1)

    # Phase 2: Large DBs sequentially
    large_dbs = [d for d in dbs_to_run if d in LARGE_DBS]
    if large_dbs:
        overall_pbar.set_postfix(phase="large DBs (sequential)")
        for db_name in large_dbs:
            clear_norm_cache()  # Fresh cache per database to prevent cross-DB accumulation
            overall_pbar.set_postfix(phase=f"large DBs", current=db_name)
            result = _run_single(db_name, args.db_dir, args.output_dir, args.force, args.studies_dir, args.stitch, args.parallel_workers, args.hla_dir)
            results.append(result)
            status = result.get("status", "unknown")
            if status == "completed":
                overall_pbar.write(
                    f"  {db_name}: {result['rows']:,} rows "
                    f"({result.get('dropped', 0):,} dropped) "
                    f"in {result.get('elapsed_s', 0):.1f}s"
                )
            elif status == "skipped":
                overall_pbar.write(f"  {db_name}: up-to-date (skipped)")
            else:
                overall_pbar.write(
                    f"  {db_name}: {status} - {result.get('error', '')}"
                )
            overall_pbar.update(1)

    overall_pbar.close()

    # Phase 3: Fix gene names and re-annotate CDR1/CDR2
    completed_dbs = [
        r.get("name", "") for r in results if r.get("status") == "completed"
    ]
    if completed_dbs:
        print(f"\nPhase 3: Fixing gene names & re-annotating CDR1/CDR2...")
        output_path = Path(args.output_dir)
        for db_name in completed_dbs:
            try:
                fix_stats = fix_genes_for_db(
                    db_name,
                    input_dir=output_path,
                    output_dir=output_path,
                    dry_run=False,
                    workers=1,
                )
                wc = fix_stats.get("wrong_chain_nulled", 0)
                gc = fix_stats.get("genes_corrected", 0)
                cdr = fix_stats.get("cdr_updated", 0)
                if wc or gc or cdr:
                    print(
                        f"  {db_name}: {wc:,} wrong-chain nulled, "
                        f"{gc:,} genes corrected, {cdr:,} CDR filled"
                    )
            except Exception as e:
                print(f"  {db_name}: gene fix ERROR - {e}")

    # Summary
    overall_elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"{'DB':<15s} {'Status':<12s} {'Rows':>12s} {'Dropped':>10s} {'Time':>8s}")
    print(f"{'-'*60}")
    total_rows = 0
    total_dropped = 0
    for r in sorted(results, key=lambda x: x.get("name", "")):
        name = r.get("name", "?")
        status = r.get("status", "?")
        rows = r.get("rows", 0)
        dropped = r.get("dropped", 0)
        elapsed = r.get("elapsed_s", 0)
        total_rows += rows
        total_dropped += dropped

        if status == "completed":
            print(f"{name:<15s} {'done':<12s} {rows:>12,} {dropped:>10,} {elapsed:>7.1f}s")
        elif status == "skipped":
            print(f"{name:<15s} {'skipped':<12s} {'':>12s} {'':>10s} {'':>8s}")
        else:
            err = str(r.get("error", ""))[:30]
            print(f"{name:<15s} {'ERROR':<12s} {'':>12s} {'':>10s} {err}")

    print(f"{'-'*60}")
    print(
        f"{'TOTAL':<15s} {'':12s} {total_rows:>12,} {total_dropped:>10,} "
        f"{overall_elapsed:>7.1f}s"
    )
    print()


if __name__ == "__main__":
    main()
