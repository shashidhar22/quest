#!/usr/bin/env python3
"""
Test script for x2gd.16xlarge - processes 10% of data with high parallelization
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
    print("TCR Data Parser - TEST MODE (10% of data)")
    print("Instance: x2gd.16xlarge (64 vCPUs, 1TB RAM)")
    print("="*80)
    print(f"Start time: {datetime.now()}")
    print()
    
    # Get config path relative to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    format_config = project_root / 'config' / 'header_config.yaml'
    
    # Test settings - use 30 workers for testing
    parser = StreamingParserComplete(
        format_config=str(format_config),
        output_dir='/mnt/ephemeral/test_output',
        chunk_size=200_000,      # Large chunks (plenty of RAM)
        compression='snappy',
        test_mode=True           # Only 10% of each file
    )
    
    # Use 30 workers for test (half of production)
    stats = parser.parse_directory(
        input_dir='/mnt/ephemeral/data',
        pattern='**/*.tsv',
        max_workers=30
    )
    
    elapsed = time.time() - start_time
    
    print()
    print("="*80)
    print("TEST COMPLETE")
    print("="*80)
    print(f"Files processed: {stats['processed']:,}")
    print(f"Failed files: {stats['failed']}")
    print(f"MRI rows: {stats['total_mri_rows']:,}")
    print(f"Seq rows: {stats['total_seq_rows']:,}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Throughput: {stats['processed']/(elapsed/60):.1f} files/min")
    
    if stats['failed'] > 0:
        print(f"\n⚠ Failed files (first 10): {stats['failed_files'][:10]}")

    print(f"\nOutput: /mnt/ephemeral/test_output/")
    print("  Check sample files:")
    print("    ls /mnt/ephemeral/test_output/mri/ | head")
    print("    ls /mnt/ephemeral/test_output/seq/ | head")

if __name__ == "__main__":
    main()
