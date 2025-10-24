#!/usr/bin/env python3
"""
Optimized parsing script for x2gd.16xlarge instance
- 64 vCPUs
- 1,024 GB RAM
- NVMe instance storage

This script uses aggressive parallelization for maximum throughput.

NEW: TCR Full-Length Stitching
- Uses stitchr to generate full-length TCR sequences from CDR3 + V/J genes
- Adds tra_full and trb_full columns to output
- Gene names are automatically normalized for compatibility

NEW: Database Integration
- Parses curated databases (VDJdb, McPAS, TCRdb, IEDB, CEDAR, iReceptor)
- Integrates with standard file parsing pipeline
- All outputs written to same directory structure
"""

from parsers.streaming_parser_complete import StreamingParserComplete
from parsers.airr_database_parser import DatabaseParser
import logging
import time
from datetime import datetime
import os

# Setup logging (reduced verbosity - progress shown via tqdm)
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    start_time = time.time()
    
    print("="*80)
    print("TCR Data Parser - Optimized for x2gd.16xlarge")
    print("="*80)
    print(f"Start time: {datetime.now()}")
    print()
    
    output_dir = '/mnt/ephemeral/parsed_output'
    
    # ========================================================================
    # PHASE 1: Parse Curated Databases
    # ========================================================================
    database_config = '/home/ubuntu/quest/config/database_config.yaml'
    db_stats = {'total_mri_rows': 0, 'total_seq_rows': 0, 'databases_parsed': 0}
    
    if os.path.exists(database_config):
        print("PHASE 1: Parsing Curated Databases")
        print("-" * 80)
        print("Sources: VDJdb, McPAS-TCR, TCRdb, IEDB, CEDAR, iReceptor")
        print()
        
        db_start = time.time()
        try:
            db_parser = DatabaseParser(
                config_path=database_config,
                test=False  # Full dataset
            )
            db_stats = db_parser.parse()
            db_elapsed = time.time() - db_start
            
            print()
            print(f"✓ Database parsing complete in {db_elapsed/60:.1f} minutes")
            print(f"  - Databases parsed: {db_stats['databases_parsed']}")
            print(f"  - MRI rows: {db_stats['total_mri_rows']:,}")
            print(f"  - Seq rows: {db_stats['total_seq_rows']:,}")
            print()
        except Exception as e:
            print(f"⚠ Database parsing failed: {e}")
            print("Continuing with file parsing...")
            print()
    else:
        print(f"⚠ Database config not found: {database_config}")
        print("Skipping database parsing...")
        print()
    
    # ========================================================================
    # PHASE 2: Parse Directory Files
    # ========================================================================
    print("PHASE 2: Parsing Data Directory")
    print("-" * 80)
    
    # Optimized settings for 64 vCPU instance
    parser = StreamingParserComplete(
        format_config='/home/ubuntu/quest/config/header_config.yaml',
        output_dir=output_dir,
        chunk_size=200_000,      # Larger chunks (you have the RAM)
        compression='snappy',     # Fast compression
        test_mode=False,          # Full dataset
        enable_stitching=True     # Enable TCR full-length stitching
    )
    
    print("Configuration:")
    print(f"  - Chunk size: 200,000 rows")
    print(f"  - Compression: snappy")
    print(f"  - TCR Stitching: ENABLED")
    print(f"  - Workers: 60 / 64 vCPUs")
    print()
    
    # Use high parallelization (leave some CPUs for system)
    # 60 workers out of 64 vCPUs
    file_stats = parser.parse_directory(
        input_dir='/mnt/ephemeral/data',
        pattern='**/*.tsv',
        max_workers=60
    )
    
    elapsed = time.time() - start_time
    
    # ========================================================================
    # FINAL REPORT
    # ========================================================================
    print()
    print("="*80)
    print("PARSING COMPLETE")
    print("="*80)
    
    # Combined statistics
    total_mri = db_stats['total_mri_rows'] + file_stats['total_mri_rows']
    total_seq = db_stats['total_seq_rows'] + file_stats['total_seq_rows']
    
    print("\n📊 Database Sources:")
    print(f"  - Databases parsed: {db_stats['databases_parsed']}")
    print(f"  - MRI rows: {db_stats['total_mri_rows']:,}")
    print(f"  - Seq rows: {db_stats['total_seq_rows']:,}")
    
    print("\n📁 File Sources:")
    print(f"  - Files processed: {file_stats['processed']:,}")
    print(f"  - Failed files: {file_stats['failed']}")
    print(f"  - MRI rows: {file_stats['total_mri_rows']:,}")
    print(f"  - Seq rows: {file_stats['total_seq_rows']:,}")
    
    print("\n📈 TOTALS:")
    print(f"  - Total MRI rows: {total_mri:,}")
    print(f"  - Total Seq rows: {total_seq:,}")
    print(f"  - Elapsed time: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
    
    if file_stats['processed'] > 0:
        print(f"  - File throughput: {file_stats['processed']/(elapsed/60):.1f} files/minute")
    
    if file_stats['failed'] > 0:
        print(f"\n⚠ Failed files (first 10): {file_stats['failed_files'][:10]}")

    print(f"\n📂 Output directory: {output_dir}/")
    print(f"  - MRI tables: {output_dir}/mri/")
    print(f"  - Seq tables: {output_dir}/seq/")
    print(f"\n📋 Columns in output:")
    print(f"  - Standard: tra, trb, trav_gene, traj_gene, trbv_gene, trbj_gene, peptide, mhc_one, ...")
    print(f"  - NEW: tra_full, trb_full (full-length stitched sequences)")
    print(f"  - Source tracking: source column indicates origin (vdjdb, mcpas, bulk_survey, single_cell, etc.)")

if __name__ == "__main__":
    main()
