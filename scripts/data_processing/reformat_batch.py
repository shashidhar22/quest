#!/usr/bin/env python3
"""
Batch processor for reformatting 2TB TCR dataset
Processes all files in parallel with progress tracking
"""
import argparse
import logging
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quest.parsers.streaming_parser import StreamingParserComplete as StreamingParser


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_single_file(
    file_path: str,
    format_config: str,
    output_dir: str,
    chunk_size: int,
    compression: str,
    test_mode: bool
) -> Tuple[str, int, int, float]:
    """
    Process a single file (for parallel execution)
    
    Returns:
        (file_path, mri_rows, seq_rows, duration_seconds)
    """
    start = time.time()
    
    try:
        parser = StreamingParser(
            format_config=format_config,
            output_dir=output_dir,
            chunk_size=chunk_size,
            compression=compression,
            test_mode=test_mode
        )
        
        format_type, format_name, delimiter = parser.detect_format(file_path)
        mri_rows, seq_rows = parser.parse_file_streaming(
            file_path, format_type, format_name, delimiter
        )
        
        duration = time.time() - start
        return file_path, mri_rows, seq_rows, duration
    
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        duration = time.time() - start
        return file_path, 0, 0, duration


def find_files(
    input_dir: str,
    patterns: List[str] = None,
    recursive: bool = True
) -> List[Path]:
    """
    Find all data files in input directory
    
    Args:
        input_dir: Root directory to search
        patterns: List of glob patterns (default: ['*.tsv', '*.csv', '*.txt'])
        recursive: Search recursively
    
    Returns:
        List of file paths
    """
    if patterns is None:
        patterns = ['*.tsv', '*.csv', '*.txt', '*.tsv.gz', '*.csv.gz']
    
    input_path = Path(input_dir)
    files = []
    
    for pattern in patterns:
        if recursive:
            files.extend(input_path.rglob(pattern))
        else:
            files.extend(input_path.glob(pattern))
    
    return sorted(files)


def batch_process(
    input_dir: str,
    output_dir: str,
    format_config: str,
    max_workers: int = 4,
    chunk_size: int = 100_000,
    compression: str = 'snappy',
    test_mode: bool = False,
    file_patterns: List[str] = None
) -> Dict:
    """
    Process all files in directory using parallel workers
    
    Args:
        input_dir: Directory containing input files
        output_dir: Directory for output Parquet files
        format_config: Path to header_config.yaml
        max_workers: Number of parallel processes
        chunk_size: Rows per chunk
        compression: Parquet compression
        test_mode: Process only 10% of data per file
        file_patterns: List of glob patterns to match
    
    Returns:
        Dictionary with processing statistics
    """
    logger.info("="*80)
    logger.info("TCR Dataset Batch Reformatter")
    logger.info("="*80)
    
    # Find all files
    logger.info(f"Scanning {input_dir} for data files...")
    files = find_files(input_dir, patterns=file_patterns)
    
    if not files:
        logger.error(f"No files found in {input_dir}")
        return {}
    
    logger.info(f"Found {len(files):,} files to process")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Parallel workers: {max_workers}")
    logger.info(f"Chunk size: {chunk_size:,} rows")
    logger.info(f"Compression: {compression}")
    if test_mode:
        logger.warning("TEST MODE: Only processing 10% of each file")
    logger.info("="*80)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process files in parallel
    start_time = time.time()
    results = {}
    total_mri_rows = 0
    total_seq_rows = 0
    completed = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all files
        future_to_file = {
            executor.submit(
                process_single_file,
                str(f),
                format_config,
                output_dir,
                chunk_size,
                compression,
                test_mode
            ): f
            for f in files
        }
        
        # Process results as they complete
        for future in as_completed(future_to_file):
            file_path, mri_rows, seq_rows, duration = future.result()
            completed += 1
            
            if mri_rows > 0:
                total_mri_rows += mri_rows
                total_seq_rows += seq_rows
                results[file_path] = {
                    'mri_rows': mri_rows,
                    'seq_rows': seq_rows,
                    'duration': duration,
                    'status': 'success'
                }
                logger.info(
                    f"[{completed}/{len(files)}] ✓ {Path(file_path).name}: "
                    f"{mri_rows:,} MRI, {seq_rows:,} seq rows "
                    f"({duration:.1f}s, {mri_rows/duration:.0f} rows/sec)"
                )
            else:
                failed += 1
                results[file_path] = {
                    'mri_rows': 0,
                    'seq_rows': 0,
                    'duration': duration,
                    'status': 'failed'
                }
                logger.error(f"[{completed}/{len(files)}] ✗ {Path(file_path).name} FAILED")
    
    # Final summary
    total_time = time.time() - start_time
    logger.info("="*80)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*80)
    logger.info(f"Total files processed: {completed:,}")
    logger.info(f"Successful: {completed - failed:,}")
    logger.info(f"Failed: {failed:,}")
    logger.info(f"Total MRI rows: {total_mri_rows:,}")
    logger.info(f"Total sequence rows: {total_seq_rows:,}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Average throughput: {total_mri_rows/total_time:.0f} rows/sec")
    logger.info("="*80)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Batch reformat TCR datasets to unified schema',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files in /mnt/ephemeral/data with 8 workers
  python reformat_batch.py /mnt/ephemeral/data /mnt/ephemeral/output \\
      --workers 8 --chunk-size 100000

  # Test mode (10% of data per file)
  python reformat_batch.py /mnt/ephemeral/data /mnt/ephemeral/output \\
      --test --workers 4

  # Process only TSV files
  python reformat_batch.py /mnt/ephemeral/data /mnt/ephemeral/output \\
      --pattern "*.tsv" --workers 8

  # Use gzip compression for smaller output
  python reformat_batch.py /mnt/ephemeral/data /mnt/ephemeral/output \\
      --compression gzip --workers 8
        """
    )
    
    parser.add_argument(
        'input_dir',
        help='Input directory containing data files'
    )
    parser.add_argument(
        'output_dir',
        help='Output directory for Parquet files'
    )
    
    # Get default config path relative to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    default_config = str(project_root / 'config' / 'header_config.yaml')
    
    parser.add_argument(
        '--config',
        default=default_config,
        help='Path to header_config.yaml (default: %(default)s)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: %(default)s)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=100000,
        help='Rows per chunk (default: %(default)s)'
    )
    parser.add_argument(
        '--compression',
        choices=['snappy', 'gzip', 'none'],
        default='snappy',
        help='Parquet compression (default: %(default)s)'
    )
    parser.add_argument(
        '--pattern',
        action='append',
        help='File glob pattern(s) to match (can specify multiple times)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: process only 10%% of each file'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.input_dir).exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    if not Path(args.config).exists():
        logger.error(f"Config file does not exist: {args.config}")
        sys.exit(1)
    
    # Run batch processing
    batch_process(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        format_config=args.config,
        max_workers=args.workers,
        chunk_size=args.chunk_size,
        compression=args.compression,
        test_mode=args.test,
        file_patterns=args.pattern
    )


if __name__ == '__main__':
    main()
