#!/usr/bin/env python3
"""
Re-parse database files to fix MHC sequence conversion.

This script:
1. Removes old database parquet files (VDJdb, McPAS, TCRdb, IEDB, CEDAR)
2. Re-parses databases with fixed MHC transformation
3. Properly converts HLA alleles to sequences using IMGT HLA FASTA
"""

from parsers.airr_database_parser import DatabaseParser
from pathlib import Path
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    output_dir = Path('/mnt/ephemeral/parsed_output')
    mri_dir = output_dir / 'mri'
    seq_dir = output_dir / 'seq'
    
    # List of database files to remove
    databases = ['vdjdb', 'mcpas', 'tcrdb', 'iedb', 'cedar']
    
    logger.info("=" * 80)
    logger.info("STEP 1: Removing old database files")
    logger.info("=" * 80)
    
    removed_count = 0
    for db in databases:
        mri_file = mri_dir / f'{db}_mri.parquet'
        seq_file = seq_dir / f'{db}_seq.parquet'
        
        if mri_file.exists():
            logger.info(f"  Removing {mri_file}")
            mri_file.unlink()
            removed_count += 1
        
        if seq_file.exists():
            logger.info(f"  Removing {seq_file}")
            seq_file.unlink()
            removed_count += 1
    
    logger.info(f"\n✓ Removed {removed_count} old database files\n")
    
    # Re-parse databases
    logger.info("=" * 80)
    logger.info("STEP 2: Re-parsing databases with fixed MHC transformation")
    logger.info("=" * 80)
    logger.info("This will convert HLA alleles to sequences using IMGT HLA FASTA\n")
    
    # Get config path relative to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    config_path = project_root / 'config' / 'database_config.yaml'
    
    start_time = time.time()
    
    db_parser = DatabaseParser(
        config_path=str(config_path),
        test=False
    )
    
    db_stats = db_parser.parse()
    
    elapsed = time.time() - start_time
    
    logger.info("\n" + "=" * 80)
    logger.info("COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Databases parsed: {db_stats['databases_parsed']}")
    logger.info(f"MRI rows: {db_stats['total_mri_rows']:,}")
    logger.info(f"Seq rows: {db_stats['total_seq_rows']:,}")
    logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")
    logger.info("\nDatabase files now contain:")
    logger.info("  - mhc_one: Full MHC-I protein sequences")
    logger.info("  - mhc_two: Full MHC-II protein sequences")
    logger.info("  - mhc_one_id: HLA allele identifiers (e.g., A*02:01)")
    logger.info("  - mhc_two_id: HLA allele identifiers (e.g., DRB1*04:05)")

if __name__ == "__main__":
    main()
