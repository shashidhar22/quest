#!/usr/bin/env python3
"""
Efficient data reformatter for large TCR datasets.

Usage:
    python reformat_data.py --input /mnt/ephemeral/data --output /mnt/ephemeral/output --workers 8

Features:
- Streams data in chunks (never loads full files)
- Parallel processing across multiple files
- Direct write to Parquet (efficient columnar format)
- Memory efficient for 2TB+ datasets
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml

from parsers.streaming_parser import StreamingParser, merge_parquet_partitions


def setup_logging(verbose=False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    parser = argparse.ArgumentParser(
        description='Reformat TCR datasets to standardized schema',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all TSV files with 8 parallel workers
  python reformat_data.py --input /mnt/ephemeral/data --output /mnt/ephemeral/output --workers 8
  
  # Test mode (process only 10% of each file)
  python reformat_data.py --input /mnt/ephemeral/data --output /mnt/ephemeral/output --test
  
  # Process only CSV files with custom chunk size
  python reformat_data.py --input /data --output /out --pattern "**/*.csv" --chunk-size 50000
  
  # Merge partitions after processing
  python reformat_data.py --merge-only --output /mnt/ephemeral/output
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input directory containing data files'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for reformatted Parquet files'
    )
    
    parser.add_argument(
        '--format-config',
        type=str,
        default='config/format_config.yaml',
        help='Path to format configuration YAML (default: config/format_config.yaml)'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='**/*.tsv',
        help='Glob pattern for input files (default: **/*.tsv)'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=100_000,
        help='Number of rows per chunk (default: 100,000)'
    )
    
    parser.add_argument(
        '--compression',
        type=str,
        default='snappy',
        choices=['snappy', 'gzip', 'brotli', 'zstd', 'lz4'],
        help='Parquet compression codec (default: snappy)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: process only 10%% of each file'
    )
    
    parser.add_argument(
        '--merge-only',
        action='store_true',
        help='Only merge existing partitions (skip parsing)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose logging'
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    if not args.merge_only and not args.input:
        parser.error("--input is required unless --merge-only is specified")
    
    output_path = Path(args.output)
    
    # Merge-only mode
    if args.merge_only:
        logger.info("=== Merge Mode ===")
        mri_dir = output_path / "mri"
        seq_dir = output_path / "seq"
        
        if mri_dir.exists():
            logger.info("Merging MRI partitions...")
            mri_rows = merge_parquet_partitions(
                mri_dir,
                output_path / "mri_combined.parquet",
                args.compression
            )
            logger.info(f"  ✓ MRI: {mri_rows:,} rows")
        
        if seq_dir.exists():
            logger.info("Merging sequence partitions...")
            seq_rows = merge_parquet_partitions(
                seq_dir,
                output_path / "seq_combined.parquet",
                args.compression
            )
            logger.info(f"  ✓ Seq: {seq_rows:,} rows")
        
        logger.info("✓ Merge complete")
        return 0
    
    # Parse mode
    logger.info("=== Reformatting TCR Dataset ===")
    logger.info(f"Input:       {args.input}")
    logger.info(f"Output:      {args.output}")
    logger.info(f"Pattern:     {args.pattern}")
    logger.info(f"Workers:     {args.workers}")
    logger.info(f"Chunk size:  {args.chunk_size:,}")
    logger.info(f"Test mode:   {args.test}")
    logger.info("")
    
    # Create streaming parser
    try:
        streaming_parser = StreamingParser(
            format_config=args.format_config,
            output_dir=args.output,
            chunk_size=args.chunk_size,
            compression=args.compression,
            test_mode=args.test
        )
    except FileNotFoundError as e:
        logger.error(f"Format config not found: {args.format_config}")
        logger.error("Create this file or specify --format-config")
        return 1
    
    # Parse directory
    try:
        stats = streaming_parser.parse_directory(
            input_dir=args.input,
            pattern=args.pattern,
            max_workers=args.workers
        )
        
        logger.info("")
        logger.info("=== Summary ===")
        logger.info(f"Files processed:  {stats['processed']}")
        logger.info(f"Files failed:     {stats['failed']}")
        logger.info(f"MRI rows:         {stats['total_mri_rows']:,}")
        logger.info(f"Sequence rows:    {stats['total_seq_rows']:,}")
        
        if stats['failed'] > 0:
            logger.warning(f"Failed files: {stats['failed_files']}")
        
        # Optionally merge partitions
        logger.info("")
        merge = input("Merge partitions into single files? [y/N]: ").strip().lower()
        if merge == 'y':
            logger.info("Merging MRI partitions...")
            mri_rows = merge_parquet_partitions(
                output_path / "mri",
                output_path / "mri_combined.parquet",
                args.compression
            )
            
            logger.info("Merging sequence partitions...")
            seq_rows = merge_parquet_partitions(
                output_path / "seq",
                output_path / "seq_combined.parquet",
                args.compression
            )
            
            logger.info("✓ Merge complete")
        
        logger.info("✓ All done!")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
