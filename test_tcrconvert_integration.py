#!/usr/bin/env python3
"""
Test TCRStitcher with tcrconvert integration.
"""

import sys
sys.path.insert(0, '/home/ubuntu/quest')

from parsers.tcr_stitcher import TCRStitcher
import logging

# Setup logging to see debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_gene_normalization():
    """Test that gene normalization works with tcrconvert."""
    stitcher = TCRStitcher()
    
    print("=" * 80)
    print("Testing Gene Normalization with tcrconvert")
    print("=" * 80)
    
    # Test cases with problematic gene names
    test_cases = [
        ("TCRBV13-01*01", "TRB"),
        ("TCRBJ02-07*01", "TRB"),
        ("TCRAV1-2", "TRA"),
        ("TRBV06-05", "TRB"),
        ("TRAV1-2,TRAV1-3", "TRA"),  # Multiple alleles
    ]
    
    for gene, chain in test_cases:
        normalized = stitcher.normalize_gene_name(gene, chain)
        print(f"{gene:30} ({chain}) -> {normalized}")
    
    print()

def test_full_stitching():
    """Test full TCR stitching with real example."""
    stitcher = TCRStitcher()
    
    print("=" * 80)
    print("Testing Full TCR Stitching")
    print("=" * 80)
    
    # Test case that was failing before
    test_data = {
        'trb_cdr3': 'CASSPRDNAYEQYF',
        'trb_v': 'TCRBV13-01*01',
        'trb_j': 'TCRBJ02-07*01',
        'tra_cdr3': 'CAVRDSSYKLIF',
        'tra_v': 'TCRAV12-01*01',
        'tra_j': 'TCRAJ33*01',
    }
    
    print(f"\nTest Input:")
    print(f"  TRB: {test_data['trb_cdr3']} | V: {test_data['trb_v']} | J: {test_data['trb_j']}")
    print(f"  TRA: {test_data['tra_cdr3']} | V: {test_data['tra_v']} | J: {test_data['tra_j']}")
    
    tra_full, trb_full = stitcher.process_tcr_pair(
        tra_cdr3=test_data['tra_cdr3'],
        tra_v=test_data['tra_v'],
        tra_j=test_data['tra_j'],
        trb_cdr3=test_data['trb_cdr3'],
        trb_v=test_data['trb_v'],
        trb_j=test_data['trb_j']
    )
    
    print(f"\nResults:")
    if tra_full:
        print(f"  TRA Full-length: {tra_full[:60]}... (len={len(tra_full)})")
    else:
        print(f"  TRA Full-length: None (FAILED)")
    
    if trb_full:
        print(f"  TRB Full-length: {trb_full[:60]}... (len={len(trb_full)})")
    else:
        print(f"  TRB Full-length: None (FAILED)")
    
    # Verify CDR3 is in the full sequence
    if tra_full and test_data['tra_cdr3'] in tra_full:
        print(f"  ✓ TRA CDR3 found in full sequence")
    elif tra_full:
        print(f"  ✗ TRA CDR3 NOT found in full sequence")
    
    if trb_full and test_data['trb_cdr3'] in trb_full:
        print(f"  ✓ TRB CDR3 found in full sequence")
    elif trb_full:
        print(f"  ✗ TRB CDR3 NOT found in full sequence")
    
    print()
    
    return tra_full is not None or trb_full is not None

if __name__ == '__main__':
    print("\nStarting TCRStitcher Integration Tests\n")
    
    # Test gene normalization
    test_gene_normalization()
    
    # Test full stitching
    success = test_full_stitching()
    
    if success:
        print("=" * 80)
        print("✓ At least one chain successfully stitched!")
        print("=" * 80)
        sys.exit(0)
    else:
        print("=" * 80)
        print("✗ Stitching failed for both chains")
        print("=" * 80)
        sys.exit(1)
