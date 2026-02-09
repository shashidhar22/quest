#!/usr/bin/env python3
"""
Test script for TCR stitching functionality.

Tests:
1. TCRStitcher initialization
2. Gene name standardization
3. Single TCR stitching
4. DataFrame processing
"""

import sys
import pandas as pd
from quest.parsers.tcr_stitcher import TCRStitcher, STITCHR_AVAILABLE

def test_initialization():
    """Test TCRStitcher initialization."""
    print("="*60)
    print("Test 1: TCRStitcher Initialization")
    print("="*60)
    
    print(f"stitchr available: {STITCHR_AVAILABLE}")
    
    stitcher = TCRStitcher(species="HUMAN")
    print(f"Stitcher enabled: {stitcher.enabled}")
    
    if not stitcher.enabled:
        print("⚠ Stitcher not enabled. Install dependency:")
        print("  pip install stitchr")
        return False
    
    print("✓ TCRStitcher initialized successfully")
    return True

def test_gene_standardization():
    """Test gene name normalization."""
    print("\n" + "="*60)
    print("Test 2: Gene Name Normalization")
    print("="*60)
    
    stitcher = TCRStitcher()
    if not stitcher.enabled:
        print("⚠ Skipping (stitcher not enabled)")
        return False
    
    test_genes = [
        "TRAV12-2",
        "TRAV12-2*01",
        "TCRAV12-2",
        "TRBV6-5",
        "TRBV06-05",
        "TRAJ33",
        "TRBJ1-4",
        "TRAV1-2,TRAV1-3",
    ]
    
    for gene in test_genes:
        normalized = stitcher.normalize_gene_name(gene)
        print(f"  {gene:20s} -> {normalized}")
    
    print("✓ Gene normalization test complete")
    return True

def test_single_stitching():
    """Test stitching a single TCR pair."""
    print("\n" + "="*60)
    print("Test 3: Single TCR Stitching")
    print("="*60)
    
    stitcher = TCRStitcher()
    if not stitcher.enabled:
        print("⚠ Skipping (stitcher not enabled)")
        return False
    
    # Example TCR from literature
    tra_full, trb_full = stitcher.process_tcr_pair(
        tra_cdr3="CAVKDFNKFYF",
        tra_v="TRAV12-2",
        tra_j="TRAJ33",
        trb_cdr3="CASSLAPGTTNEKLFF",
        trb_v="TRBV6-5",
        trb_j="TRBJ1-4"
    )
    
    print(f"\nInput:")
    print(f"  TRA CDR3: CAVKDFNKFYF")
    print(f"  TRA V: TRAV12-2")
    print(f"  TRA J: TRAJ33")
    print(f"  TRB CDR3: CASSLAPGTTNEKLFF")
    print(f"  TRB V: TRBV6-5")
    print(f"  TRB J: TRBJ1-4")
    
    print(f"\nOutput:")
    if tra_full:
        print(f"  TRA full: {tra_full[:50]}... (length: {len(tra_full)})")
    else:
        print(f"  TRA full: [FAILED]")
    
    if trb_full:
        print(f"  TRB full: {trb_full[:50]}... (length: {len(trb_full)})")
    else:
        print(f"  TRB full: [FAILED]")
    
    if tra_full or trb_full:
        print("✓ At least one chain stitched successfully")
        return True
    else:
        print("✗ Stitching failed for both chains")
        return False

def test_dataframe_processing():
    """Test processing a DataFrame."""
    print("\n" + "="*60)
    print("Test 4: DataFrame Processing")
    print("="*60)
    
    stitcher = TCRStitcher()
    if not stitcher.enabled:
        print("⚠ Skipping (stitcher not enabled)")
        return False
    
    # Sample data
    df = pd.DataFrame({
        'tra': ['CAVKDFNKFYF', 'CAASREGADRLTF', 'CAVRDSNYQLIW'],
        'trav_gene': ['TRAV12-2', 'TRAV13-1', 'TRAV21'],
        'traj_gene': ['TRAJ33', 'TRAJ5', 'TRAJ18'],
        'trb': ['CASSLAPGTTNEKLFF', 'CASSLGQAYEQYF', 'CASSLEETQYF'],
        'trbv_gene': ['TRBV6-5', 'TRBV28', 'TRBV7-2'],
        'trbj_gene': ['TRBJ1-4', 'TRBJ2-7', 'TRBJ2-5']
    })
    
    print(f"\nProcessing {len(df)} TCR pairs...")
    df_result = stitcher.process_dataframe(df)
    
    print(f"\nResults:")
    print(f"  Rows processed: {len(df_result)}")
    
    tra_success = (df_result['tra_full'] != '').sum()
    trb_success = (df_result['trb_full'] != '').sum()
    
    print(f"  TRA stitched: {tra_success}/{len(df_result)} ({100*tra_success/len(df_result):.1f}%)")
    print(f"  TRB stitched: {trb_success}/{len(df_result)} ({100*trb_success/len(df_result):.1f}%)")
    
    # Show sample output
    print(f"\nSample output:")
    for idx, row in df_result.head(3).iterrows():
        print(f"\n  Row {idx+1}:")
        print(f"    TRA CDR3: {row['tra']}")
        if row['tra_full']:
            print(f"    TRA full: {row['tra_full'][:40]}... ({len(row['tra_full'])} aa)")
        else:
            print(f"    TRA full: [not generated]")
        
        print(f"    TRB CDR3: {row['trb']}")
        if row['trb_full']:
            print(f"    TRB full: {row['trb_full'][:40]}... ({len(row['trb_full'])} aa)")
        else:
            print(f"    TRB full: [not generated]")
    
    if tra_success > 0 or trb_success > 0:
        print("\n✓ DataFrame processing successful")
        return True
    else:
        print("\n✗ No sequences stitched")
        return False

def main():
    """Run all tests."""
    print("\nTCR Stitching Functionality Tests")
    print("="*60)
    
    results = []
    
    # Test 1: Initialization
    results.append(("Initialization", test_initialization()))
    
    # Only run other tests if initialization succeeded
    if results[0][1]:
        results.append(("Gene Standardization", test_gene_standardization()))
        results.append(("Single TCR Stitching", test_single_stitching()))
        results.append(("DataFrame Processing", test_dataframe_processing()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:30s} {status}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total_count - passed_count} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
