#!/usr/bin/env python3
"""
Quick test script for TCR reformatter
Tests format detection and basic parsing
"""
import sys
from pathlib import Path
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parsers.streaming_parser import StreamingParser


def test_format_detection():
    """Test format detection on sample files"""
    print("="*80)
    print("Testing Format Detection")
    print("="*80)
    
    config_path = '/home/ubuntu/quest/config/header_config.yaml'
    parser = StreamingParser(
        format_config=config_path,
        output_dir='/tmp/test_output',
        chunk_size=1000,
        test_mode=True
    )
    
    # Test with sample data directory
    test_dir = Path('/mnt/ephemeral/data')
    
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        print("Please update test_dir in this script to point to your data")
        return False
    
    # Find first few files
    sample_files = list(test_dir.rglob('*.tsv'))[:5]
    sample_files.extend(list(test_dir.rglob('*.csv'))[:5])
    
    if not sample_files:
        print(f"❌ No data files found in {test_dir}")
        return False
    
    print(f"\nTesting format detection on {len(sample_files)} files...\n")
    
    detected = 0
    failed = 0
    
    for file_path in sample_files:
        try:
            format_type, format_name, delimiter = parser.detect_format(str(file_path))
            print(f"✓ {file_path.name}")
            print(f"  Type: {format_type}, Format: {format_name}, Delimiter: {repr(delimiter)}")
            detected += 1
        except Exception as e:
            print(f"✗ {file_path.name}")
            print(f"  Error: {e}")
            failed += 1
    
    print("\n" + "="*80)
    print(f"Format Detection Results: {detected} detected, {failed} failed")
    print("="*80)
    
    return failed == 0


def test_parsing():
    """Test actual parsing and output"""
    print("\n" + "="*80)
    print("Testing File Parsing")
    print("="*80)
    
    config_path = '/home/ubuntu/quest/config/header_config.yaml'
    output_dir = Path('/tmp/test_reformat_output')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    parser = StreamingParser(
        format_config=config_path,
        output_dir=str(output_dir),
        chunk_size=1000,
        test_mode=True  # Only process 10% of each file
    )
    
    # Find a sample file
    test_dir = Path('/mnt/ephemeral/data')
    sample_files = list(test_dir.rglob('*.tsv'))[:2]
    
    if not sample_files:
        print("❌ No sample files found")
        return False
    
    print(f"\nParsing {len(sample_files)} sample file(s) (10% each)...\n")
    
    for file_path in sample_files:
        try:
            print(f"Processing: {file_path.name}")
            
            format_type, format_name, delimiter = parser.detect_format(str(file_path))
            mri_rows, seq_rows = parser.parse_file_streaming(
                str(file_path),
                format_type,
                format_name,
                delimiter
            )
            
            print(f"  ✓ MRI rows: {mri_rows:,}")
            print(f"  ✓ SEQ rows: {seq_rows:,}")
            
            # Check output files exist
            repertoire_id = file_path.stem
            mri_file = output_dir / 'mri' / f'{repertoire_id}_mri.parquet'
            seq_file = output_dir / 'seq' / f'{repertoire_id}_seq.parquet'
            
            if mri_file.exists():
                mri_df = pd.read_parquet(mri_file)
                print(f"  ✓ MRI file: {len(mri_df)} rows, {len(mri_df.columns)} columns")
                print(f"    Columns: {', '.join(mri_df.columns[:5])}...")
                print(f"    Sample data:")
                for col in ['tra', 'trb', 'trav_gene', 'trbv_gene']:
                    if col in mri_df.columns:
                        non_empty = mri_df[col].astype(bool).sum()
                        print(f"      {col}: {non_empty} non-empty values")
            
            if seq_file.exists():
                seq_df = pd.read_parquet(seq_file)
                print(f"  ✓ SEQ file: {len(seq_df)} rows, {len(seq_df.columns)} columns")
            
            print()
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("="*80)
    print("Parsing Test Complete")
    print(f"Output files in: {output_dir}")
    print("="*80)
    
    return True


def test_output_schema():
    """Verify output matches expected schema"""
    print("\n" + "="*80)
    print("Testing Output Schema")
    print("="*80)
    
    output_dir = Path('/tmp/test_reformat_output')
    
    expected_columns = [
        'tra', 'trav_gene', 'trad_gene', 'traj_gene',
        'trb', 'trbv_gene', 'trbd_gene', 'trbj_gene',
        'peptide', 'mhc_one', 'mhc_two',
        'repertoire_id', 'study_id', 'source', 'host_organism'
    ]
    
    # Check MRI files
    mri_files = list((output_dir / 'mri').glob('*.parquet'))
    
    if not mri_files:
        print("❌ No MRI output files found")
        return False
    
    print(f"\nChecking schema for {len(mri_files)} MRI file(s)...\n")
    
    for mri_file in mri_files:
        df = pd.read_parquet(mri_file)
        print(f"File: {mri_file.name}")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        
        # Check for expected columns
        missing = set(expected_columns) - set(df.columns)
        if missing:
            print(f"  ⚠ Missing columns: {', '.join(missing)}")
        else:
            print(f"  ✓ All expected columns present")
        
        # Check data types
        print(f"  Column types: all {df.dtypes[0]}")
        
        # Check for TCR data
        has_tra = df['tra'].astype(bool).sum()
        has_trb = df['trb'].astype(bool).sum()
        print(f"  TCR alpha rows: {has_tra:,}")
        print(f"  TCR beta rows: {has_trb:,}")
        print()
    
    print("="*80)
    print("Schema Test Complete")
    print("="*80)
    
    return True


def main():
    """Run all tests"""
    print("\n" + "🧬 TCR Reformatter Test Suite 🧬\n")
    
    tests = [
        ("Format Detection", test_format_detection),
        ("File Parsing", test_parsing),
        ("Output Schema", test_output_schema)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {test_name} PASSED\n")
            else:
                failed += 1
                print(f"\n❌ {test_name} FAILED\n")
        except Exception as e:
            failed += 1
            print(f"\n❌ {test_name} FAILED with exception: {e}\n")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*80)
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
