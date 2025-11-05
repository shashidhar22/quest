#!/usr/bin/env python3
"""
Diagnostic test script for TCR stitching functionality.

This script comprehensively tests:
1. stitchr package installation and functionality
2. Reference data availability
3. Gene name normalization
4. Full-length sequence generation
5. Integration with the parser pipeline
6. Common failure modes

IMPORTANT: stitchr requires reference data to be downloaded first!
Run this command to download human TCR reference data:
    stitchrdl -s HUMAN

Run this to diagnose why tra_full/trb_full might be empty in production output.
"""

import sys
import os
import pandas as pd
import logging
import subprocess
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_stitchr_import():
    """Test if stitchr package is properly installed."""
    print("="*80)
    print("TEST 1: stitchr Package Import")
    print("="*80)
    
    try:
        from Stitchr import stitchrfunctions as fxn
        from Stitchr import stitchr as st
        print("✓ stitchr package imported successfully")
        print(f"  - stitchr module: {st.__file__}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import stitchr: {e}")
        print("\nTo install stitchr:")
        print("  pip install stitchr")
        return False

def test_reference_data():
    """Test if stitchr reference data is available."""
    print("\n" + "="*80)
    print("TEST 2: Reference Data Availability")
    print("="*80)
    
    try:
        # Find stitchr data directory
        result = subprocess.run(['stitchr', '-dd'], capture_output=True, text=True)
        data_dir = result.stdout.strip()
        
        print(f"Data directory: {data_dir}")
        
        # Check for HUMAN data
        human_dir = Path(data_dir) / "HUMAN"
        
        if not human_dir.exists():
            print(f"✗ HUMAN data directory not found: {human_dir}")
            print("\n" + "="*60)
            print("SETUP REQUIRED: Download reference data")
            print("="*60)
            print("\nRun this command to download human TCR reference data:")
            print("  stitchrdl -s HUMAN")
            print("\nThis will download IMGT gene references for human TCRs.")
            print("It only needs to be done once.")
            return False
        
        # Check for required files
        required_files = ['TRA.fasta', 'TRB.fasta', 'J-region-motifs.tsv', 'C-region-motifs.tsv']
        missing_files = []
        
        for fname in required_files:
            fpath = human_dir / fname
            if fpath.exists():
                print(f"  ✓ {fname}")
            else:
                print(f"  ✗ {fname} (missing)")
                missing_files.append(fname)
        
        if missing_files:
            print(f"\n✗ Missing {len(missing_files)} required file(s)")
            print("\nRe-download reference data with:")
            print("  stitchrdl -s HUMAN")
            return False
        
        print("\n✓ All required reference data files found")
        return True
        
    except Exception as e:
        print(f"✗ Error checking reference data: {e}")
        print("\nMake sure stitchr is installed: pip install stitchr")
        print("Then download reference data: stitchrdl -s HUMAN")
        return False

def test_stitcher_class():
    """Test TCRStitcher class initialization."""
    print("\n" + "="*80)
    print("TEST 2: TCRStitcher Class Initialization")
    print("="*80)
    
    try:
        from parsers.tcr_stitcher import TCRStitcher, STITCHR_AVAILABLE
        
        print(f"STITCHR_AVAILABLE flag: {STITCHR_AVAILABLE}")
        
        if not STITCHR_AVAILABLE:
            print("✗ stitchr marked as not available")
            return False
        
        stitcher = TCRStitcher(species="HUMAN")
        print(f"✓ TCRStitcher initialized")
        print(f"  - Enabled: {stitcher.enabled}")
        print(f"  - Species: {stitcher.species}")
        
        # Check if reference data loaded
        if hasattr(stitcher, 'tra_data') and hasattr(stitcher, 'trb_data'):
            print(f"  - TRA data loaded: {stitcher.tra_data is not None}")
            print(f"  - TRB data loaded: {stitcher.trb_data is not None}")
        
        return stitcher.enabled
    except Exception as e:
        print(f"✗ TCRStitcher initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gene_normalization():
    """Test gene name normalization with various formats."""
    print("\n" + "="*80)
    print("TEST 3: Gene Name Normalization")
    print("="*80)
    
    try:
        from parsers.tcr_stitcher import TCRStitcher
        
        stitcher = TCRStitcher()
        if not stitcher.enabled:
            print("⚠ Skipping (stitcher not enabled)")
            return False
        
        test_cases = [
            # (input, expected_pattern)
            ("TRAV12-2", "TRAV12-2"),
            ("TRAV12-2*01", "TRAV12-2"),  # Allele stripped
            ("TCRAV12-2", "TRAV12-2"),     # TCR prefix removed
            ("TRAV01-02", "TRAV1-2"),      # Leading zeros removed
            ("TRBV06-05", "TRBV6-5"),      # Leading zeros removed
            ("TRBV6-5*01", "TRBV6-5"),     # Allele stripped
            ("TRAJ33", "TRAJ33"),
            ("TRBJ1-4", "TRBJ1-4"),
            ("TRAV1-2,TRAV1-3", "TRAV1-2"),  # Multiple alleles - first taken
            ("", None),                     # Empty string
            (None, None),                   # None
        ]
        
        all_passed = True
        for input_gene, expected in test_cases:
            result = stitcher.normalize_gene_name(input_gene)
            
            # Check if result matches expected
            if expected is None:
                passed = result is None
            else:
                passed = result == expected
            
            status = "✓" if passed else "✗"
            print(f"  {status} '{input_gene}' -> '{result}' (expected: '{expected}')")
            
            if not passed:
                all_passed = False
        
        if all_passed:
            print("\n✓ All gene normalization tests passed")
        else:
            print("\n✗ Some gene normalization tests failed")
        
        return all_passed
    except Exception as e:
        print(f"✗ Gene normalization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_stitching():
    """Test basic TCR stitching with known good sequences."""
    print("\n" + "="*80)
    print("TEST 4: Basic TCR Stitching")
    print("="*80)
    
    try:
        from parsers.tcr_stitcher import TCRStitcher
        
        stitcher = TCRStitcher()
        if not stitcher.enabled:
            print("⚠ Skipping (stitcher not enabled)")
            return False
        
        # Test cases with known TCR sequences from literature
        test_cases = [
            {
                'name': 'Example TRA',
                'cdr3': 'CAVKDFNKFYF',
                'v_gene': 'TRAV12-2',
                'j_gene': 'TRAJ33',
                'chain': 'TRA'
            },
            {
                'name': 'Example TRB',
                'cdr3': 'CASSLAPGTTNEKLFF',
                'v_gene': 'TRBV6-5',
                'j_gene': 'TRBJ1-4',
                'chain': 'TRB'
            },
            {
                'name': 'Short TRA',
                'cdr3': 'CAASREGADRLTF',
                'v_gene': 'TRAV13-1',
                'j_gene': 'TRAJ5',
                'chain': 'TRA'
            },
            {
                'name': 'Short TRB',
                'cdr3': 'CASSLEETQYF',
                'v_gene': 'TRBV7-2',
                'j_gene': 'TRBJ2-5',
                'chain': 'TRB'
            }
        ]
        
        all_passed = True
        for test in test_cases:
            print(f"\n  Testing {test['name']}:")
            print(f"    CDR3: {test['cdr3']}")
            print(f"    V: {test['v_gene']}, J: {test['j_gene']}")
            
            full_seq = stitcher.stitch_tcr(
                cdr3=test['cdr3'],
                v_gene=test['v_gene'],
                j_gene=test['j_gene'],
                chain=test['chain']
            )
            
            if full_seq:
                print(f"    ✓ Generated: {full_seq[:50]}... (length: {len(full_seq)} aa)")
                
                # Verify CDR3 is in the full sequence
                if test['cdr3'] in full_seq:
                    print(f"    ✓ CDR3 found in full sequence")
                else:
                    print(f"    ✗ CDR3 NOT found in full sequence!")
                    all_passed = False
            else:
                print(f"    ✗ FAILED to generate sequence")
                all_passed = False
        
        if all_passed:
            print("\n✓ All basic stitching tests passed")
        else:
            print("\n✗ Some stitching tests failed")
        
        return all_passed
    except Exception as e:
        print(f"✗ Basic stitching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_paired_stitching():
    """Test stitching TCR alpha/beta pairs."""
    print("\n" + "="*80)
    print("TEST 5: Paired TCR Stitching")
    print("="*80)
    
    try:
        from parsers.tcr_stitcher import TCRStitcher
        
        stitcher = TCRStitcher()
        if not stitcher.enabled:
            print("⚠ Skipping (stitcher not enabled)")
            return False
        
        # Test TCR pair
        print("\n  Input TCR pair:")
        print(f"    TRA: CDR3=CAVKDFNKFYF, V=TRAV12-2, J=TRAJ33")
        print(f"    TRB: CDR3=CASSLAPGTTNEKLFF, V=TRBV6-5, J=TRBJ1-4")
        
        tra_full, trb_full = stitcher.process_tcr_pair(
            tra_cdr3='CAVKDFNKFYF',
            tra_v='TRAV12-2',
            tra_j='TRAJ33',
            trb_cdr3='CASSLAPGTTNEKLFF',
            trb_v='TRBV6-5',
            trb_j='TRBJ1-4'
        )
        
        print("\n  Output:")
        tra_ok = False
        trb_ok = False
        
        if tra_full:
            print(f"    ✓ TRA: {tra_full[:50]}... ({len(tra_full)} aa)")
            tra_ok = True
        else:
            print(f"    ✗ TRA: FAILED")
        
        if trb_full:
            print(f"    ✓ TRB: {trb_full[:50]}... ({len(trb_full)} aa)")
            trb_ok = True
        else:
            print(f"    ✗ TRB: FAILED")
        
        if tra_ok and trb_ok:
            print("\n✓ Paired stitching test passed")
            return True
        else:
            print("\n✗ Paired stitching test failed")
            return False
    except Exception as e:
        print(f"✗ Paired stitching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataframe_processing():
    """Test processing a DataFrame with TCR data."""
    print("\n" + "="*80)
    print("TEST 6: DataFrame Processing")
    print("="*80)
    
    try:
        from parsers.tcr_stitcher import TCRStitcher
        
        stitcher = TCRStitcher()
        if not stitcher.enabled:
            print("⚠ Skipping (stitcher not enabled)")
            return False
        
        # Create sample DataFrame matching parser output schema
        df = pd.DataFrame({
            'tra': ['CAVKDFNKFYF', 'CAASREGADRLTF', 'CAVRDSNYQLIW', '', 'CAVKDFNKFYF'],
            'trav_gene': ['TRAV12-2', 'TRAV13-1', 'TRAV21', '', 'TRAV12-2'],
            'traj_gene': ['TRAJ33', 'TRAJ5', 'TRAJ18', '', 'TRAJ33'],
            'trb': ['CASSLAPGTTNEKLFF', 'CASSLGQAYEQYF', 'CASSLEETQYF', 'CASSLEETQYF', ''],
            'trbv_gene': ['TRBV6-5', 'TRBV28', 'TRBV7-2', 'TRBV7-2', ''],
            'trbj_gene': ['TRBJ1-4', 'TRBJ2-7', 'TRBJ2-5', 'TRBJ2-5', '']
        })
        
        print(f"\n  Processing {len(df)} rows...")
        print(f"  Row types:")
        print(f"    - Both TRA & TRB: 3")
        print(f"    - TRB only: 1")
        print(f"    - TRA only: 1")
        
        result_df = stitcher.process_dataframe(df.copy())
        
        print(f"\n  Results:")
        print(f"    Rows: {len(result_df)}")
        
        # Count successes
        tra_success = (result_df['tra_full'] != '').sum()
        trb_success = (result_df['trb_full'] != '').sum()
        
        print(f"    TRA stitched: {tra_success}/{len(result_df)}")
        print(f"    TRB stitched: {trb_success}/{len(result_df)}")
        
        # Show details for each row
        print(f"\n  Row-by-row results:")
        for idx, row in result_df.iterrows():
            tra_status = "✓" if row['tra_full'] else "✗"
            trb_status = "✓" if row['trb_full'] else "✗"
            
            print(f"    Row {idx+1}:")
            print(f"      TRA {tra_status}: {row['tra'][:20] if row['tra'] else '(empty)':20s} -> {len(row['tra_full'])} aa")
            print(f"      TRB {trb_status}: {row['trb'][:20] if row['trb'] else '(empty)':20s} -> {len(row['trb_full'])} aa")
        
        # Expect at least some successes
        if tra_success >= 3 and trb_success >= 3:
            print("\n✓ DataFrame processing test passed")
            return True
        else:
            print(f"\n✗ DataFrame processing test failed (expected ≥3 TRA and ≥3 TRB)")
            return False
    except Exception as e:
        print(f"✗ DataFrame processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parser_integration():
    """Test integration with StreamingParserComplete."""
    print("\n" + "="*80)
    print("TEST 7: Parser Integration")
    print("="*80)
    
    try:
        from parsers.streaming_parser_complete import StreamingParserComplete
        
        # Get config path relative to project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / 'config' / 'header_config.yaml'
        
        if not os.path.exists(str(config_path)):
            print(f"⚠ Config not found: {config_path}")
            print("  Skipping parser integration test")
            return None
        
        print(f"\n  Creating parser with stitching enabled...")
        
        # Create parser with stitching enabled
        parser = StreamingParserComplete(
            format_config=str(config_path),
            output_dir='/tmp/test_stitcher_output',
            chunk_size=1000,
            test_mode=True,
            enable_stitching=True
        )
        
        print(f"    Parser created")
        print(f"    Stitching enabled: {parser.enable_stitching}")
        print(f"    Stitcher object: {parser.stitcher is not None}")
        
        if parser.stitcher:
            print(f"    Stitcher active: {parser.stitcher.enabled}")
        
        # Test the _apply_stitching method
        test_df = pd.DataFrame({
            'tra': ['CAVKDFNKFYF'],
            'trav_gene': ['TRAV12-2'],
            'traj_gene': ['TRAJ33'],
            'trb': ['CASSLAPGTTNEKLFF'],
            'trbv_gene': ['TRBV6-5'],
            'trbj_gene': ['TRBJ1-4']
        })
        
        print(f"\n  Testing _apply_stitching method...")
        result_df = parser._apply_stitching(test_df.copy())
        
        print(f"    Columns in result: {list(result_df.columns)}")
        print(f"    tra_full present: {'tra_full' in result_df.columns}")
        print(f"    trb_full present: {'trb_full' in result_df.columns}")
        
        if 'tra_full' in result_df.columns and 'trb_full' in result_df.columns:
            tra_val = result_df.iloc[0]['tra_full']
            trb_val = result_df.iloc[0]['trb_full']
            
            print(f"    tra_full value: {tra_val[:50] if tra_val else '(empty)'}{'...' if tra_val and len(tra_val) > 50 else ''}")
            print(f"    trb_full value: {trb_val[:50] if trb_val else '(empty)'}{'...' if trb_val and len(trb_val) > 50 else ''}")
            
            if tra_val and trb_val:
                print("\n✓ Parser integration test passed")
                return True
            else:
                print("\n✗ Parser integration test failed (empty sequences)")
                return False
        else:
            print("\n✗ Parser integration test failed (missing columns)")
            return False
        
    except Exception as e:
        print(f"✗ Parser integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """Test edge cases and common failure modes."""
    print("\n" + "="*80)
    print("TEST 8: Edge Cases")
    print("="*80)
    
    try:
        from parsers.tcr_stitcher import TCRStitcher
        
        stitcher = TCRStitcher()
        if not stitcher.enabled:
            print("⚠ Skipping (stitcher not enabled)")
            return False
        
        edge_cases = [
            {
                'name': 'Empty CDR3',
                'tra_cdr3': '',
                'tra_v': 'TRAV12-2',
                'tra_j': 'TRAJ33',
                'trb_cdr3': '',
                'trb_v': '',
                'trb_j': '',
                'expect_tra': False,
                'expect_trb': False
            },
            {
                'name': 'Missing V gene',
                'tra_cdr3': 'CAVKDFNKFYF',
                'tra_v': '',
                'tra_j': 'TRAJ33',
                'trb_cdr3': '',
                'trb_v': '',
                'trb_j': '',
                'expect_tra': False,
                'expect_trb': False
            },
            {
                'name': 'Missing J gene',
                'tra_cdr3': 'CAVKDFNKFYF',
                'tra_v': 'TRAV12-2',
                'tra_j': '',
                'trb_cdr3': '',
                'trb_v': '',
                'trb_j': '',
                'expect_tra': False,
                'expect_trb': False
            },
            {
                'name': 'Invalid gene name',
                'tra_cdr3': 'CAVKDFNKFYF',
                'tra_v': 'INVALID',
                'tra_j': 'TRAJ33',
                'trb_cdr3': '',
                'trb_v': '',
                'trb_j': '',
                'expect_tra': False,
                'expect_trb': False
            },
            {
                'name': 'Valid TRB only',
                'tra_cdr3': '',
                'tra_v': '',
                'tra_j': '',
                'trb_cdr3': 'CASSLAPGTTNEKLFF',
                'trb_v': 'TRBV6-5',
                'trb_j': 'TRBJ1-4',
                'expect_tra': False,
                'expect_trb': True
            }
        ]
        
        all_passed = True
        for test in edge_cases:
            print(f"\n  Testing: {test['name']}")
            
            tra_full, trb_full = stitcher.process_tcr_pair(
                tra_cdr3=test['tra_cdr3'],
                tra_v=test['tra_v'],
                tra_j=test['tra_j'],
                trb_cdr3=test['trb_cdr3'],
                trb_v=test['trb_v'],
                trb_j=test['trb_j']
            )
            
            tra_ok = (tra_full is not None and tra_full != '') == test['expect_tra']
            trb_ok = (trb_full is not None and trb_full != '') == test['expect_trb']
            
            tra_status = "✓" if tra_ok else "✗"
            trb_status = "✓" if trb_ok else "✗"
            
            print(f"    TRA {tra_status}: got {'sequence' if tra_full else 'None'}, expected {'sequence' if test['expect_tra'] else 'None'}")
            print(f"    TRB {trb_status}: got {'sequence' if trb_full else 'None'}, expected {'sequence' if test['expect_trb'] else 'None'}")
            
            if not (tra_ok and trb_ok):
                all_passed = False
        
        if all_passed:
            print("\n✓ All edge case tests passed")
        else:
            print("\n✗ Some edge case tests failed")
        
        return all_passed
    except Exception as e:
        print(f"✗ Edge case test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests."""
    print("\n" + "="*80)
    print("TCR STITCHER DIAGNOSTIC TEST SUITE")
    print("="*80)
    print("\nThis script will help diagnose why tra_full/trb_full might be empty.")
    print()
    
    results = []
    
    # Test 1: Import
    test1 = test_stitchr_import()
    results.append(("stitchr Import", test1))
    
    if not test1:
        print("\n" + "="*80)
        print("CRITICAL: stitchr not installed")
        print("="*80)
        print("\nInstall with: pip install stitchr")
        print("Then re-run this test.")
        return 1
    
    # Test 2: Reference data
    test2 = test_reference_data()
    results.append(("Reference Data", test2))
    
    if not test2:
        print("\n" + "="*80)
        print("CRITICAL: Reference data not available")
        print("="*80)
        print("\nSee instructions above to download reference data.")
        return 1
    
    # Test 3: Class initialization
    test3 = test_stitcher_class()
    results.append(("TCRStitcher Class", test3))
    
    if not test3:
        print("\n" + "="*80)
        print("CRITICAL: TCRStitcher initialization failed")
        print("="*80)
        print("\nCheck the error messages above for details.")
        return 1
    
    # Continue with other tests
    results.append(("Gene Normalization", test_gene_normalization()))
    results.append(("Basic Stitching", test_basic_stitching()))
    results.append(("Paired Stitching", test_paired_stitching()))
    results.append(("DataFrame Processing", test_dataframe_processing()))
    
    integration_result = test_parser_integration()
    if integration_result is not None:
        results.append(("Parser Integration", integration_result))
    
    results.append(("Edge Cases", test_edge_cases()))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:30s} {status}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    # Diagnostic recommendations
    print("\n" + "="*80)
    print("DIAGNOSTIC RECOMMENDATIONS")
    print("="*80)
    
    if passed_count == total_count:
        print("\n✓ All tests passed!")
        print("\nIf production parser still produces empty tra_full/trb_full:")
        print("  1. Check that enable_stitching=True in run_parser_production.py")
        print("  2. Verify input data has valid V and J gene annotations")
        print("  3. Check logs for stitching errors during parsing")
        print("  4. Run parser in test mode on small sample to inspect output")
    else:
        failed_tests = [name for name, passed in results if not passed]
        print(f"\n⚠ {total_count - passed_count} test(s) failed:")
        for test_name in failed_tests:
            print(f"  - {test_name}")
        
        print("\nTroubleshooting steps:")
        if not results[0][1]:
            print("  1. Install stitchr: pip install stitchr")
        if not results[1][1]:
            print("  2. Check stitchr installation and dependencies")
        if len(results) > 2 and not results[2][1]:
            print("  3. Gene normalization failing - check gene name formats")
        if len(results) > 3 and not results[3][1]:
            print("  4. Basic stitching failing - check stitchr reference data")
        
        print("\nFor detailed diagnostics, review the error messages above.")
    
    return 0 if passed_count == total_count else 1

if __name__ == "__main__":
    sys.exit(main())
