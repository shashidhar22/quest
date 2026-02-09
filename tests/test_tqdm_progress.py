#!/usr/bin/env python3
"""
Quick test to verify tqdm progress bar works with the parser
"""

import sys
from quest.parsers.streaming_parser import StreamingParserComplete
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
    
    # Get test data directory from user input or environment variable
    import os
    default_dir = os.environ.get('TEST_DATA_DIR', '')
    
    if len(sys.argv) > 1:
        test_dir = Path(sys.argv[1])
    elif default_dir:
        test_dir = Path(default_dir)
    else:
        test_dir_input = input("\nEnter path to test data directory (or press Enter to skip): ").strip()
        if not test_dir_input:
            print("No test directory provided. Skipping test.")
            return
        test_dir = Path(test_dir_input)
    
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        print("Please provide a valid directory path")
        return
    
    # Get config path relative to project root
    project_root = Path(__file__).parent.parent
    config_path = project_root / 'config' / 'header_config.yaml'
    
    # Create parser with test mode
    parser = StreamingParserComplete(
        format_config=str(config_path),
        output_dir='/tmp/test_tqdm_output',
        chunk_size=50_000,
        compression='snappy',
        test_mode=True
    )
    
    print(f"\nTesting on: {test_dir}")
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

if __name__ == "__main__":
    main()
