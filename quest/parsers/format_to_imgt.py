#!/usr/bin/env python3
"""
IMGT Gene Name Standardization

Converts various non-standard TCR gene name formats to IMGT standard format.

IMGT Standard Format:
    Locus (TRA/TRB/TRG/TRD) + Gene type (V/D/J/C) + Subgroup + optional(-Position) + optional(*Allele)
    Examples: TRAV1-2*01, TRBV6-5, TRAJ33*01, TRBD1

Handles:
    - TCR prefix removal: TCRBV13-01 -> TRBV13-1
    - Leading zeros removal: TRAV01-02 -> TRAV1-2
    - Colon to asterisk: TRAV3-3:01 -> TRAV3-3*01
    - Dual gene notation: TRAV29/DV5 -> TRAV29 (keep first, valid for stitchr)
    - OR alternatives: "TRBV20/OR9-2, or TRBV20-1" -> TRBV20/OR9-2
    - Concatenated genes: TRBV20-1V20/OR9-2 -> TRBV20-1 (take first valid)
    - Space-separated: "TRAJ23 60 0" -> TRAJ23
    - Pseudogene notation: TRBJ2-2P*01 -> TRBJ2-2*01

Reference:
    IMGT Nomenclature: https://www.imgt.org/IMGTrepertoire/index.php?section=LocusGenes
"""

import re
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Valid IMGT gene pattern (human TCR)
IMGT_PATTERN = re.compile(
    r'^TR[ABDG][VDJC]'  # Locus + gene type
    r'\d+'              # Subgroup number
    r'(-\d+)?'          # Optional position
    r'(\*\d+)?$'        # Optional allele
)

# Known dual genes - KEEP the dual notation as stitchr expects it
# These are legitimate dual TRA/TRD genes where both names are valid
# Stitchr data uses the full notation like TRAV29/DV5*01
DUAL_GENE_PATTERNS = [
    'TRAV29/DV5',
    'TRAV14/DV4', 
    'TRAV38-2/DV8',
    'TRAV38-1/DV8',  # Also valid
    'TRAV23/DV6',
    'TRAV36/DV7',
    'TRDV1',  # Some appear as TRAV but are TRD
    'TRDV2',
    'TRDV3',
]

# OR9-2 genes map to standard names
# These are orphan genes that map to functional genes
OR9_2_GENE_MAP = {
    'TRBV20/OR9-2': 'TRBV20-1',
    'TRBV20/OR9-2*01': 'TRBV20-1*01',
    'TRBV20/OR9-2*02': 'TRBV20-1*02',
    'TRBV24/OR9-2': 'TRBV24-1',
    'TRBV24-1/OR9-2': 'TRBV24-1',
    'TRBV26/OR9-2': 'TRBV26',
    'TRBV26/OR9-2*01': 'TRBV26*01',
    'TRBV29/OR9-2': 'TRBV29-1',
    'TRBV29/OR9-2*01': 'TRBV29-1*01',
}


def standardize_to_imgt(gene: str) -> Optional[str]:
    """
    Convert a TCR gene name to IMGT standard format.
    
    Args:
        gene: Gene name in any format
        
    Returns:
        Standardized IMGT gene name, or None if invalid/unrecognizable
    """
    if not gene or gene == '' or gene == 'nan' or gene == 'NA':
        return None
    
    original = str(gene).strip()
    gene = original
    
    # Handle empty/null
    if not gene:
        return None
    
    # Step 1: Handle ", or " alternatives - take the first one
    if ', or ' in gene:
        gene = gene.split(', or ')[0].strip()
    
    # Step 2: Handle semicolon-separated alternatives
    if ';' in gene:
        gene = gene.split(';')[0].strip()
    
    # Step 3: Handle comma-separated alternatives (but not in scores like "40.5")
    if ',' in gene and not re.search(r'\d+\.\d+', gene):
        gene = gene.split(',')[0].strip()
    
    # Step 4: Handle space-separated junk (e.g., "TRAJ23 60 0")
    if ' ' in gene:
        parts = gene.split()
        # Take only the gene part (first element that looks like a gene)
        for part in parts:
            if re.match(r'^(TR|TCR)[ABDG][VDJC]', part):
                gene = part
                break
    
    # Step 5: Check known mappings first
    # For dual genes, preserve the notation as stitchr expects it
    # e.g., TRAV29/DV5*01 is valid in stitchr data
    for pattern in DUAL_GENE_PATTERNS:
        if gene.startswith(pattern):
            # Already a dual gene, just standardize colon to asterisk and leading zeros
            gene = re.sub(r':(\d+)$', r'*\1', gene)
            # Remove leading zeros in allele
            gene = re.sub(r'\*0+(\d+)$', r'*\1', gene)
            return gene
    
    if gene in OR9_2_GENE_MAP:
        return OR9_2_GENE_MAP[gene]
    
    # Step 6: Handle dual gene notation with allele variations  
    # Keep the dual gene format, just normalize allele notation
    # TRAV29/DV5:01 -> TRAV29/DV5*01
    dual_match = re.match(r'^(TR[AD]V\d+(?:-\d+)?/D[VG]\d+)(?::(\d+)|\*(\d+))?$', gene)
    if dual_match:
        base = dual_match.group(1)
        allele = dual_match.group(2) or dual_match.group(3)
        if allele:
            return f"{base}*{allele}"
        return base
    
    # Step 7: Handle OR9-2 notation with allele variations
    or9_match = re.match(r'^(TRBV\d+(?:-\d+)?)/OR9-2(?:\*(\d+))?$', gene)
    if or9_match:
        base = or9_match.group(1)
        allele = or9_match.group(2)
        # Ensure we have the -1 suffix for standard form
        if not re.search(r'-\d+$', base):
            base = f"{base}-1"
        if allele:
            return f"{base}*{allele}"
        return base
    
    # Step 8: Remove TCR prefix (TCRBV -> TRBV)
    gene = re.sub(r'^TCR([ABDG][VDJC])', r'TR\1', gene)
    
    # Step 9: Handle concatenated genes (TRBV20-1V20/OR9-2 -> TRBV20-1)
    # Look for pattern like TRxVn-nVn or TRxVnVn
    concat_match = re.match(r'^(TR[AB][VJ]\d+(?:-\d+)?)V\d+', gene)
    if concat_match:
        gene = concat_match.group(1)
    
    # Also handle slash-concatenated (TRBV2-1/TRBV2-2/TRBV2-3 -> TRBV2-1)
    if '/' in gene and 'TRB' in gene:
        parts = gene.split('/')
        for part in parts:
            if re.match(r'^TR[AB][VJ]\d+(-\d+)?(\*\d+)?$', part):
                gene = part
                break
    
    # Step 10: Convert colon allele notation to asterisk
    # TRAV3-3:01 -> TRAV3-3*01
    gene = re.sub(r':(\d+)$', r'*\1', gene)
    
    # Step 11: Remove pseudogene notation (keep allele)
    # TRBJ2-2P*01 -> TRBJ2-2*01
    gene = re.sub(r'P(\*\d+)$', r'\1', gene)
    gene = re.sub(r'P$', '', gene)
    
    # Step 12: Remove leading zeros
    # TRBV01-02*01 -> TRBV1-2*01
    def remove_leading_zeros(m):
        locus = m.group(1)
        subgroup = str(int(m.group(2)))  # Remove leading zeros
        position = m.group(3)
        allele = m.group(4)
        
        result = f"{locus}{subgroup}"
        if position:
            pos_num = str(int(position.lstrip('-')))  # Remove leading zeros from position
            result = f"{result}-{pos_num}"
        if allele:
            allele_num = str(int(allele.lstrip('*')))  # Remove leading zeros from allele
            result = f"{result}*{allele_num}"
        return result
    
    gene = re.sub(
        r'^(TR[ABDG][VDJC])0*(\d+)(-0*\d+)?(\*0*\d+)?$',
        remove_leading_zeros,
        gene
    )
    
    # Step 13: Handle mouse gene prefixes
    if gene.startswith('m'):
        # Mouse genes - can't standardize for human, return None
        return None
    
    # Step 14: Validate final result
    if IMGT_PATTERN.match(gene):
        if gene != original:
            logger.debug(f"Standardized: {original} -> {gene}")
        return gene
    
    # Step 15: Try to extract a valid gene if we have a complex string
    # Last attempt: find any valid gene pattern in the string
    valid_match = re.search(r'(TR[ABDG][VDJC]\d+(?:-\d+)?(?:\*\d+)?)', gene)
    if valid_match:
        extracted = valid_match.group(1)
        if IMGT_PATTERN.match(extracted):
            logger.debug(f"Extracted: {original} -> {extracted}")
            return extracted
    
    # Could not standardize
    logger.debug(f"Could not standardize: {original}")
    return None


def is_imgt_standard(gene: str) -> bool:
    """
    Check if a gene name is already in IMGT standard format.
    
    Args:
        gene: Gene name to check
        
    Returns:
        True if gene is in IMGT standard format
    """
    if not gene or gene == '' or gene == 'nan':
        return False
    return bool(IMGT_PATTERN.match(str(gene).strip()))


def get_gene_info(gene: str) -> Optional[dict]:
    """
    Parse a standardized IMGT gene name into components.
    
    Args:
        gene: Gene name in IMGT format
        
    Returns:
        Dictionary with locus, gene_type, subgroup, position, allele
        or None if not valid
    """
    gene = standardize_to_imgt(gene)
    if not gene:
        return None
    
    match = re.match(
        r'^(TR[ABDG])([VDJC])(\d+)(?:-(\d+))?(?:\*(\d+))?$',
        gene
    )
    
    if match:
        return {
            'locus': match.group(1),
            'gene_type': match.group(2),
            'subgroup': int(match.group(3)),
            'position': int(match.group(4)) if match.group(4) else None,
            'allele': int(match.group(5)) if match.group(5) else None,
            'full_name': gene,
        }
    return None


# For testing
if __name__ == '__main__':
    test_cases = [
        # Standard format
        'TRBV5-1*01',
        'TRAJ33',
        'TRAV1-2',
        
        # TCR prefix
        'TCRBV01-01',
        'TCRBJ02-07',
        
        # Dual genes
        'TRAV29/DV5',
        'TRAV14/DV4*01',
        'TRAV38-2/DV8',
        'TRAV29/DV5:01',
        
        # OR9-2 genes
        'TRBV20/OR9-2*01',
        'TRBV29/OR9-2*01',
        
        # Colon notation
        'TRAV3-3:01',
        'TRBJ2-3:01',
        'TRBV5-1:01',
        
        # Concatenated
        'TRBV20-1V20/OR9-2',
        'TRBV6-2V6-3',
        'TRBV12-3V12-4',
        'TRBV2-1/TRBV2-2/TRBV2-3',
        
        # Alternatives
        'TRBV20/OR9-2, or TRBV20-1',
        'TRBJ2-5, or TRBJ1-5',
        
        # Space separated
        'TRAJ23 60 0',
        'TRAJ58 60 0',
        
        # Pseudogene
        'TRBJ2-2P*01',
        
        # Invalid
        'NA',
        'mTRAV14D-1',
        '',
    ]
    
    print('Gene Standardization Test')
    print('=' * 60)
    print(f'{"Input":<30} {"Output":<20} {"Valid"}')
    print('-' * 60)
    
    for gene in test_cases:
        result = standardize_to_imgt(gene)
        valid = is_imgt_standard(result) if result else False
        result_str = result if result else 'None'
        status = '✓' if valid else '✗'
        print(f'{gene:<30} {result_str:<20} {status}')
