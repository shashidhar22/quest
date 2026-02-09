"""
CDR Region Analysis and Weighting

Provides CDR (Complementarity-Determining Region) annotation using ANARCI,
thread-safe caching of annotations, and position-weight construction
for CDR-weighted loss functions.

Extracted from scripts/training/tcr_seq2seq_trainer.py.
"""

import threading
from typing import Dict, Optional, Tuple

import torch


# =============================================================================
# CDR Region Weighting (ANARCI-based)
# =============================================================================


# Preset configurations for CDR region weighting
CDR_WEIGHT_PRESETS = {
    "uniform": {"cdr1": 1.0, "cdr2": 1.0, "cdr3": 1.0, "framework": 1.0},
    "cdr3_focused": {"cdr1": 2.0, "cdr2": 2.0, "cdr3": 4.0, "framework": 1.0},
    "all_cdr_equal": {"cdr1": 3.0, "cdr2": 3.0, "cdr3": 3.0, "framework": 1.0},
    "extreme_cdr3": {"cdr1": 1.5, "cdr2": 1.5, "cdr3": 8.0, "framework": 0.5},
}


# Optional: ANARCI for CDR annotation
try:
    from anarci import anarci as run_anarci
    ANARCI_AVAILABLE = True
except ImportError:
    ANARCI_AVAILABLE = False


def get_cdr_positions_anarci(sequence: str, chain_type: str = "B") -> Optional[Dict[str, Tuple[int, int]]]:
    """
    Get CDR region positions using ANARCI IMGT numbering.

    Args:
        sequence: Full TCR sequence
        chain_type: 'A' for alpha, 'B' for beta

    Returns:
        Dict mapping region names to (start, end) positions, or None if annotation fails
    """
    if not ANARCI_AVAILABLE:
        return None

    try:
        # Run ANARCI with IMGT scheme
        results = run_anarci([("seq", sequence)], scheme="imgt", allowed_species=["human"])

        if results[0][0] is None:
            return None

        # Extract numbering
        numbering = results[0][0][0][0]  # First hit, first domain

        # IMGT CDR definitions for TCRs:
        # CDR1: positions 27-38 (IMGT)
        # CDR2: positions 56-65 (IMGT)
        # CDR3: positions 105-117 (IMGT) + insertions

        cdr_positions = {
            'cdr1': None,
            'cdr2': None,
            'cdr3': None,
        }

        # Map IMGT positions to sequence positions
        seq_pos = 0
        cdr1_start, cdr1_end = None, None
        cdr2_start, cdr2_end = None, None
        cdr3_start, cdr3_end = None, None

        for (imgt_pos, insertion), aa in numbering:
            if aa == '-':
                continue

            # Track CDR1 (IMGT 27-38)
            if 27 <= imgt_pos <= 38:
                if cdr1_start is None:
                    cdr1_start = seq_pos
                cdr1_end = seq_pos + 1

            # Track CDR2 (IMGT 56-65)
            elif 56 <= imgt_pos <= 65:
                if cdr2_start is None:
                    cdr2_start = seq_pos
                cdr2_end = seq_pos + 1

            # Track CDR3 (IMGT 105-117 + insertions)
            elif imgt_pos >= 105 and imgt_pos <= 117:
                if cdr3_start is None:
                    cdr3_start = seq_pos
                cdr3_end = seq_pos + 1
            elif insertion and cdr3_start is not None:
                # CDR3 insertions
                cdr3_end = seq_pos + 1

            seq_pos += 1

        if cdr1_start is not None:
            cdr_positions['cdr1'] = (cdr1_start, cdr1_end)
        if cdr2_start is not None:
            cdr_positions['cdr2'] = (cdr2_start, cdr2_end)
        if cdr3_start is not None:
            cdr_positions['cdr3'] = (cdr3_start, cdr3_end)

        return cdr_positions

    except Exception:
        return None


class CDRAnnotationCache:
    """
    Thread-safe cache for CDR annotations to avoid re-running ANARCI.
    """

    def __init__(self, max_size: int = 10000):
        self._cache: Dict[str, Optional[Dict[str, Tuple[int, int]]]] = {}
        self._max_size = max_size
        self._lock = threading.Lock()

    def get_or_compute(self, sequence: str, chain_type: str = "B") -> Optional[Dict[str, Tuple[int, int]]]:
        """Get CDR positions from cache or compute if not cached."""
        cache_key = f"{chain_type}:{sequence[:50]}"  # Use prefix for key

        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Compute outside lock
        positions = get_cdr_positions_anarci(sequence, chain_type)

        with self._lock:
            if len(self._cache) < self._max_size:
                self._cache[cache_key] = positions

        return positions


# Global CDR annotation cache
_cdr_cache = CDRAnnotationCache()


def build_cdr_position_weights(
    seq_len: int,
    cdr_annotations: Optional[Dict[str, Tuple[int, int]]],
    cdr1_weight: float = 2.0,
    cdr2_weight: float = 2.0,
    cdr3_weight: float = 4.0,
    framework_weight: float = 1.0,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Build per-position weights based on CDR annotations.

    Args:
        seq_len: Sequence length
        cdr_annotations: Dict with 'cdr1', 'cdr2', 'cdr3' -> (start, end) tuples
        cdr1_weight: Weight for CDR1 region
        cdr2_weight: Weight for CDR2 region
        cdr3_weight: Weight for CDR3 region (most important!)
        framework_weight: Weight for framework regions
        device: Target device

    Returns:
        Tensor of shape (seq_len,) with per-position weights
    """
    weights = torch.full((seq_len,), framework_weight, device=device)

    if cdr_annotations is None:
        return weights

    # Apply CDR-specific weights
    for region, weight in [('cdr1', cdr1_weight), ('cdr2', cdr2_weight), ('cdr3', cdr3_weight)]:
        if region in cdr_annotations and cdr_annotations[region] is not None:
            start, end = cdr_annotations[region]
            if start is not None and end is not None and start < seq_len and end <= seq_len:
                weights[start:end] = weight

    return weights
