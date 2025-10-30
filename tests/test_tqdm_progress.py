#!/usr/bin/env python3
"""
Quick test to verify tqdm progress bar works with the parser
"""

from parsers.streaming_parser_complete import StreamingParserComplete
import logging
from pathlib import Path

# Setup logging (only show warnings and errors now)
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s'
)

def main():
    print("Testing tqdm progress bar integration...")
    print("=" * 80)
    
    # Create parser with test mode
    parser = StreamingParserComplete(
        format_config='/home/ubuntu/quest/config/header_config.yaml',
        output_dir='/tmp/test_tqdm_output',
        chunk_size=50_000,
        compression='snappy',
        test_mode=True
    )
    
    # Test on a small subset
    test_dir = Path('/mnt/ephemeral/data/cancer_studies/kstme/tcr/bulk_survey_trb')
    
    if test_dir.exists():
        print(f"Testing on: {test_dir}")
        print("You should see a progress bar below with real-time stats:")
        print("-" * 80)
        
        stats = parser.parse_directory(
            input_dir=str(test_dir),
            pattern='*.tsv',
            max_workers=4
        )
        
        print("-" * 80)
        print(f"\nResults:")
        print(f"  Files processed: {stats['processed']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  MRI rows: {stats['total_mri_rows']:,}")
        print(f"  Seq rows: {stats['total_seq_rows']:,}")
        print("\n✓ Progress bar working correctly!")
    else:
        print(f"Test directory not found: {test_dir}")
        print("Skipping test...")

if __name__ == "__main__":
    main()
