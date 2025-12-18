"""
CDR Region Identifier

Uses tidytcells to identify CDR1, CDR2, and CDR3 regions in full-length TCR sequences.
This module provides utilities for locating Complementarity Determining Regions (CDRs)
within complete TCR alpha and beta chains.

Dependencies:
    - tidytcells: Gene standardization and CDR region extraction from V genes
"""

import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# Import tidytcells
try:
    import tidytcells.tr as tr
    TIDYTCELLS_AVAILABLE = True
except ImportError:
    TIDYTCELLS_AVAILABLE = False
    logger.warning("tidytcells not available. Install with: pip install tidytcells")


class CDRRegionIdentifier:
    """
    Identifies CDR1, CDR2, and CDR3 regions in full-length TCR sequences.

    Uses tidytcells to extract CDR1-IMGT and CDR2-IMGT from V genes, and searches
    for CDR3 in the full-length sequence. Caches results for performance.

    Example:
        >>> identifier = CDRRegionIdentifier()
        >>> regions = identifier.get_cdr_regions(
        ...     full_sequence="MGTRLL...CASSQDAGGTYEQYF...FGPGT",
        ...     cdr3_sequence="CASSQDAGGTYEQYF",
        ...     v_gene="TRBV7-9*01",
        ...     chain="TRB"
        ... )
        >>> print(regions)
        {'cdr1': (26, 31), 'cdr2': (49, 55), 'cdr3': (96, 111)}
    """

    def __init__(self):
        """Initialize with empty caches for performance."""
        self.gene_cache = {}  # {gene: {region: sequence}}
        self.normalization_cache = {}  # {(original_gene, chain): normalized_gene}
        self.enabled = TIDYTCELLS_AVAILABLE

        if not self.enabled:
            logger.warning("CDR region identification disabled. Missing: tidytcells")

    def get_cdr_regions(
        self,
        full_sequence: str,
        cdr3_sequence: str,
        v_gene: str,
        chain: str
    ) -> Dict[str, Tuple[int, int]]:
        """
        Identify CDR region boundaries in full-length TCR sequence.

        Args:
            full_sequence: Full-length TCR amino acid sequence
            cdr3_sequence: CDR3 amino acid sequence (for searching in full sequence)
            v_gene: V gene name (will be normalized to IMGT format)
            chain: 'TRA' or 'TRB'

        Returns:
            Dict with 'cdr1', 'cdr2', 'cdr3' keys, values are (start, end) tuples
            indicating 0-based positions in the full_sequence. Empty dict if regions
            cannot be identified.

        Example:
            >>> regions = identifier.get_cdr_regions(
            ...     "MGTRLL...CASSQDAGGTYEQYF...FGPGT",
            ...     "CASSQDAGGTYEQYF",
            ...     "TRBV7-9*01",
            ...     "TRB"
            ... )
            >>> regions['cdr3']
            (96, 111)  # CDR3 found at positions 96-111
        """
        if not self.enabled:
            logger.debug("tidytcells not available, cannot identify CDR regions")
            return {}

        if not full_sequence or not cdr3_sequence or not v_gene:
            logger.debug("Missing required data for CDR identification")
            return {}

        # Step 1: Normalize V gene to IMGT format
        norm_gene = self._normalize_gene(v_gene, chain)
        if not norm_gene:
            logger.debug(f"Could not normalize gene: {v_gene}")
            return {}

        # Step 2: Get CDR1/CDR2 sequences from V gene using tidytcells
        cdr_sequences = self._get_v_gene_cdr_sequences(norm_gene)
        if not cdr_sequences:
            logger.debug(f"Could not get CDR sequences for: {norm_gene}")
            return {}

        # Step 3: Find CDR positions in full-length sequence
        regions = {}

        # Find CDR1
        if cdr_sequences.get('cdr1'):
            pos = self._find_sequence_in_full(full_sequence, cdr_sequences['cdr1'])
            if pos:
                regions['cdr1'] = pos
                logger.debug(f"CDR1 found at positions {pos}")

        # Find CDR2
        if cdr_sequences.get('cdr2'):
            pos = self._find_sequence_in_full(full_sequence, cdr_sequences['cdr2'])
            if pos:
                regions['cdr2'] = pos
                logger.debug(f"CDR2 found at positions {pos}")

        # Find CDR3 (most critical)
        pos = self._find_sequence_in_full(full_sequence, cdr3_sequence)
        if pos:
            regions['cdr3'] = pos
            logger.debug(f"CDR3 found at positions {pos}")

        # Only return if we found at least CDR3
        if 'cdr3' not in regions:
            logger.debug(f"CDR3 '{cdr3_sequence}' not found in sequence")
            return {}

        logger.info(f"Identified {len(regions)} CDR regions for {v_gene}")
        return regions

    def _normalize_gene(self, gene: str, chain: str) -> Optional[str]:
        """
        Normalize gene name using tidytcells with caching.

        Args:
            gene: Gene name in any format (e.g., "TRBV7-9*01", "TCRBV07-09")
            chain: 'TRA' or 'TRB'

        Returns:
            Normalized gene name in IMGT format or None if normalization fails
        """
        cache_key = (gene, chain)
        if cache_key in self.normalization_cache:
            return self.normalization_cache[cache_key]

        try:
            # Try with allele precision first (most specific)
            normalized = tr.standardize(gene, precision='allele')
            self.normalization_cache[cache_key] = normalized
            logger.debug(f"Normalized {gene} → {normalized}")
            return normalized
        except Exception as e:
            logger.debug(f"tidytcells allele normalization failed for {gene}: {e}")

            # Try without allele (gene-level only)
            try:
                gene_base = gene.split('*')[0] if '*' in gene else gene
                normalized = tr.standardize(gene_base, precision='gene')

                # Re-add allele if original had one
                if '*' in gene and normalized:
                    allele = gene.split('*')[1]
                    normalized = f"{normalized}*{allele}"

                self.normalization_cache[cache_key] = normalized
                logger.debug(f"Normalized (gene-level) {gene} → {normalized}")
                return normalized
            except Exception as e2:
                logger.debug(f"tidytcells gene normalization failed for {gene}: {e2}")
                self.normalization_cache[cache_key] = None
                return None

    def _get_v_gene_cdr_sequences(self, v_gene: str) -> Dict[str, str]:
        """
        Get CDR1-IMGT and CDR2-IMGT sequences from V gene with caching.

        Args:
            v_gene: V gene name in IMGT format (e.g., "TRBV7-9*01")

        Returns:
            Dict with 'cdr1' and 'cdr2' keys, values are amino acid sequences.
            Empty dict if retrieval fails.
        """
        if v_gene in self.gene_cache:
            return self.gene_cache[v_gene]

        try:
            aa_sequences = tr.get_aa_sequence(v_gene)
            result = {
                'cdr1': aa_sequences.get('CDR1-IMGT', ''),
                'cdr2': aa_sequences.get('CDR2-IMGT', '')
            }
            self.gene_cache[v_gene] = result
            logger.debug(f"Retrieved CDR sequences for {v_gene}: CDR1({len(result['cdr1'])} aa), CDR2({len(result['cdr2'])} aa)")
            return result
        except Exception as e:
            logger.debug(f"Failed to get CDR sequences from tidytcells for {v_gene}: {e}")
            self.gene_cache[v_gene] = {}
            return {}

    def _find_sequence_in_full(
        self,
        full_seq: str,
        target_seq: str
    ) -> Optional[Tuple[int, int]]:
        """
        Find target sequence in full sequence, return (start, end) indices.

        Args:
            full_seq: Full-length sequence to search in
            target_seq: Subsequence to find

        Returns:
            Tuple of (start_position, end_position) if found, None otherwise.
            Positions are 0-based, end is exclusive (Python slice convention).
        """
        if not target_seq:
            return None

        pos = full_seq.find(target_seq)
        if pos == -1:
            return None

        return (pos, pos + len(target_seq))

    def clear_cache(self):
        """Clear all caches. Useful for testing or memory management."""
        self.gene_cache.clear()
        self.normalization_cache.clear()
        logger.debug("CDR identifier caches cleared")
