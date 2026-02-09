#!/usr/bin/env python3
"""
Quick test script to validate the streaming parser on sample data.

Usage:
    python test_reformatter.py
"""

import pandas as pd
import tempfile
import shutil
from pathlib import Path
from quest.parsers.streaming_parser import StreamingParserComplete as StreamingParser
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data(output_dir: Path):
    """Create sample data files in different formats."""
    
    # Format 1: Bulk alpha chain
    format1_data = pd.DataFrame({
        'VGene': ['TRAV1-1', 'TRAV1-2', 'TRAV2-1'] * 100,
        'JGene': ['TRAJ12', 'TRAJ13', 'TRAJ14'] * 100,
        'aaCDR3': ['CAVRDSSYKLIF', 'CAVRGSNNRIFF', 'CAVGDGGSQGNLIF'] * 100,
        'VJCombo': ['TRAV1-1_TRAJ12'] * 300,
        'Copy': ['100', '200', '150'] * 100,
        'ntCDR3': ['TGTGCAGTAAGAGACAGCTCCTACAAACTA'] * 300,
        'NetInsertionLength': ['5'] * 300
    })
    
    sample1_file = output_dir / "sample1_alpha.tsv"
    format1_data.to_csv(sample1_file, sep='\t', index=False)
    logger.info(f"Created {sample1_file} with {len(format1_data)} rows")
    
    # Format 1: Bulk beta chain
    format1_beta = pd.DataFrame({
        'VGene': ['TRBV7-2', 'TRBV7-3', 'TRBV7-6'] * 100,
        'JGene': ['TRBJ2-1', 'TRBJ2-2', 'TRBJ2-3'] * 100,
        'aaCDR3': ['CASSLAGGTDTQYF', 'CASSLPGQGAYEQYF', 'CASSLQGQGGELFF'] * 100,
        'VJCombo': ['TRBV7-2_TRBJ2-1'] * 300,
        'Copy': ['500', '300', '250'] * 100,
        'ntCDR3': ['TGTGCCAGCAGCTTGGCAGGCGGGACTG'] * 300,
        'NetInsertionLength': ['8'] * 300
    })
    
    sample2_file = output_dir / "sample2_beta.tsv"
    format1_beta.to_csv(sample2_file, sep='\t', index=False)
    logger.info(f"Created {sample2_file} with {len(format1_beta)} rows")
    
    return [sample1_file, sample2_file]


def main():
    """Run test."""
    logger.info("=== Testing Streaming Parser ===\n")
    
    # Create temporary directories
    input_dir = Path(tempfile.mkdtemp(prefix='test_input_'))
    output_dir = Path(tempfile.mkdtemp(prefix='test_output_'))
    
    try:
        # Create sample data
        logger.info("Creating sample data...")
        sample_files = create_sample_data(input_dir)
        
        # Create parser
        logger.info("\nInitializing streaming parser...")
        parser = StreamingParser(
            format_config='config/format_config.yaml',
            output_dir=str(output_dir),
            chunk_size=100,  # Small chunks for demo
            compression='snappy',
            test_mode=False
        )
        
        # Parse files
        logger.info("\nParsing files...")
        stats = parser.parse_directory(
            input_dir=str(input_dir),
            pattern='*.tsv',
            max_workers=2
        )
        
        # Show results
        logger.info("\n=== Results ===")
        logger.info(f"Files processed:  {stats['processed']}")
        logger.info(f"Files failed:     {stats['failed']}")
        logger.info(f"MRI rows:         {stats['total_mri_rows']:,}")
        logger.info(f"Sequence rows:    {stats['total_seq_rows']:,}")
        
        # Read and display sample output
        logger.info("\n=== Sample MRI Output ===")
        mri_files = list((output_dir / "mri").glob("*.parquet"))
        if mri_files:
            mri_sample = pd.read_parquet(mri_files[0])
            logger.info(f"\nColumns: {list(mri_sample.columns)}")
            logger.info(f"\nFirst 3 rows:\n{mri_sample.head(3)}")
            logger.info(f"\nData types:\n{mri_sample.dtypes}")
        
        logger.info("\n=== Sample Sequence Output ===")
        seq_files = list((output_dir / "seq").glob("*.parquet"))
        if seq_files:
            seq_sample = pd.read_parquet(seq_files[0])
            logger.info(f"\nColumns: {list(seq_sample.columns)}")
            logger.info(f"\nFirst 3 rows:\n{seq_sample.head(3)}")
        
        logger.info(f"\n✓ Test completed successfully!")
        logger.info(f"\nOutput location: {output_dir}")
        logger.info("(Will be cleaned up on exit)")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1
    
    finally:
        # Cleanup
        logger.info("\nCleaning up temporary files...")
        shutil.rmtree(input_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
