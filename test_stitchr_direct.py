#!/usr/bin/env python3
"""
Test stitchr directly to understand expected gene name format.
"""

import stitchr as st
import logging

logging.basicConfig(level=logging.DEBUG)

# Initialize stitchr data
print("Initializing stitchr...")
tra_data = st.get_imgt_data('TRA', 'human')
trb_data = st.get_imgt_data('TRB', 'human')
print(f"TRA data keys: {tra_data.keys()}")
print(f"TRB data keys: {trb_data.keys()}")

# Test with different gene name formats
test_cases = [
    # Format 1: Full IMGT with TR prefix
    {
        'v': 'TRBV13-1*01',
        'j': 'TRBJ2-7*01',
        'cdr3': 'CASSPRDNAYEQYF',
        'description': 'Full IMGT format (TRBV...)'
    },
    # Format 2: Short IMGT without TR prefix
    {
        'v': 'BV13-1*01',
        'j': 'BJ2-7*01',
        'cdr3': 'CASSPRDNAYEQYF',
        'description': 'Short IMGT format (BV...)'
    },
    # Format 3: With leading zeros
    {
        'v': 'TRBV13-01*01',
        'j': 'TRBJ02-07*01',
        'cdr3': 'CASSPRDNAYEQYF',
        'description': 'With leading zeros'
    },
]

for i, test in enumerate(test_cases, 1):
    print(f"\n{'='*80}")
    print(f"Test {i}: {test['description']}")
    print(f"V: {test['v']}, J: {test['j']}, CDR3: {test['cdr3']}")
    print('='*80)
    
    tcr_bits = {
        'v': test['v'],
        'j': test['j'],
        'cdr3': test['cdr3'],
        'l': '',
        'c': 'TRBC1*01',
        'mode': '',
        'skip_c_checks': False,
        'skip_n_checks': False,
        'no_leader': True,
        'species': 'human',
        'seamless': False,
        '5_prime_seq': '',
        '3_prime_seq': '',
        'name': f'test-{i}'
    }
    
    try:
        result = st.stitch(
            tcr_bits,
            trb_data['tcr_dat'],
            trb_data['functionality'],
            trb_data['partial'],
            trb_data['codons'],
            3,
            '',
            trb_data['c_res'],
            trb_data['j_res'],
            trb_data['low_conf_js']
        )
        
        if result and 'seqs' in result and 'aa' in result['seqs']:
            aa_seq = result['seqs']['aa']
            print(f"✓ SUCCESS: Generated {len(aa_seq)} AA sequence")
            print(f"  Sequence: {aa_seq[:80]}...")
            print(f"  CDR3 in sequence: {test['cdr3'] in aa_seq}")
        else:
            print(f"✗ FAILED: No sequence generated")
            print(f"  Result: {result}")
    except Exception as e:
        print(f"✗ ERROR: {e}")

# Also check what V and J genes are available in the database
print(f"\n{'='*80}")
print("Checking available gene names in IMGT database...")
print('='*80)

# Get a few V and J genes to see the format
v_genes = list(trb_data['tcr_dat'].keys())[:10]
print(f"\nSample TRB V genes (first 10):")
for gene in v_genes:
    print(f"  {gene}")

# Check if our test genes are in the database
test_genes = ['TRBV13-1*01', 'BV13-1*01', 'TRBV13-01*01', 'TRBV13-1']
print(f"\nChecking if test gene names exist in database:")
for gene in test_genes:
    exists = gene in trb_data['tcr_dat']
    print(f"  {gene:20} {'✓' if exists else '✗'}")
