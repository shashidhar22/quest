#!/usr/bin/env python3
"""
Optimized parsing script for x2gd.16xlarge instance
- 64 vCPUs
- 1,024 GB RAM
- NVMe instance storage

This script uses aggressive parallelization for maximum throughput.
"""

from parsers.streaming_parser_complete import StreamingParserComplete
import logging
import time
from datetime import datetime

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
    
    # Optimized settings for 64 vCPU instance
    parser = StreamingParserComplete(
        format_config='/home/ubuntu/quest/config/header_config.yaml',
        output_dir='/mnt/ephemeral/parsed_output',
        chunk_size=200_000,      # Larger chunks (you have the RAM)
        compression='snappy',     # Fast compression
        test_mode=False           # Full dataset
    )
    
    # Use high parallelization (leave some CPUs for system)
    # 60 workers out of 64 vCPUs
    stats = parser.parse_directory(
        input_dir='/mnt/ephemeral/data',
        pattern='**/*.tsv',
        max_workers=60
    )
    
    elapsed = time.time() - start_time
    
    print()
    print("="*80)
    print("PARSING COMPLETE")
    print("="*80)
    print(f"Total files processed: {stats['processed']:,}")
    print(f"Failed files: {stats['failed']}")
    print(f"Total MRI rows: {stats['total_mri_rows']:,}")
    print(f"Total Seq rows: {stats['total_seq_rows']:,}")
    print(f"Elapsed time: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
    print(f"Throughput: {stats['processed']/(elapsed/60):.1f} files/minute")
    
    if stats['failed'] > 0:
        print(f"\n⚠ Failed files (first 10): {stats['failed_files'][:10]}")

    print(f"\nOutput directory: /mnt/ephemeral/parsed_output/")
    print(f"  - MRI tables: /mnt/ephemeral/parsed_output/mri/")
    print(f"  - Seq tables: /mnt/ephemeral/parsed_output/seq/")

if __name__ == "__main__":
    main()
