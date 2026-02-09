"""
Biophysical Property Metrics

Provides hydrophobicity profiling (Kyte-Doolittle), charge computation,
amino acid frequency distributions, and Jensen-Shannon divergence
for comparing generated and reference TCR sequences.

Extracted from scripts/training/tcr_seq2seq_trainer.py.
"""

from typing import Dict

import numpy as np


# =============================================================================
# Biophysical Constants
# =============================================================================

# Kyte-Doolittle hydrophobicity scale
# Positive = hydrophobic, Negative = hydrophilic
KYTE_DOOLITTLE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}

# Amino acid charge at physiological pH (~7.4)
AMINO_ACID_CHARGE = {
    'D': -1.0, 'E': -1.0,  # Acidic (negative)
    'K': 1.0, 'R': 1.0, 'H': 0.1,  # Basic (positive, H partial)
    'A': 0.0, 'N': 0.0, 'C': 0.0, 'Q': 0.0, 'G': 0.0,
    'I': 0.0, 'L': 0.0, 'M': 0.0, 'F': 0.0, 'P': 0.0,
    'S': 0.0, 'T': 0.0, 'W': 0.0, 'Y': 0.0, 'V': 0.0,
}

# Standard amino acids
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


# =============================================================================
# Amino Acid Composition Metrics
# =============================================================================


def compute_hydrophobicity_profile(seq: str) -> np.ndarray:
    """
    Compute per-position hydrophobicity using Kyte-Doolittle scale.

    Args:
        seq: Amino acid sequence

    Returns:
        Array of hydrophobicity values per position
    """
    profile = []
    for aa in seq.upper():
        profile.append(KYTE_DOOLITTLE.get(aa, 0.0))
    return np.array(profile)


def hydrophobicity_correlation(seq1: str, seq2: str) -> float:
    """
    Compute Pearson correlation between hydrophobicity profiles.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Correlation coefficient (-1 to 1), or 0 if cannot compute
    """
    if not seq1 or not seq2:
        return 0.0

    # Truncate to same length for comparison
    min_len = min(len(seq1), len(seq2))
    if min_len < 3:
        return 0.0

    profile1 = compute_hydrophobicity_profile(seq1[:min_len])
    profile2 = compute_hydrophobicity_profile(seq2[:min_len])

    # Compute Pearson correlation
    if np.std(profile1) < 1e-8 or np.std(profile2) < 1e-8:
        return 0.0

    corr = np.corrcoef(profile1, profile2)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def compute_net_charge(seq: str) -> float:
    """
    Compute net charge of a sequence at physiological pH.

    Args:
        seq: Amino acid sequence

    Returns:
        Net charge
    """
    return sum(AMINO_ACID_CHARGE.get(aa.upper(), 0.0) for aa in seq)


def aa_frequency_distribution(seq: str) -> Dict[str, float]:
    """
    Compute amino acid frequency distribution.

    Args:
        seq: Amino acid sequence

    Returns:
        Dict mapping amino acid to frequency (0-1)
    """
    if not seq:
        return {aa: 0.0 for aa in AMINO_ACIDS}

    counts = {aa: 0 for aa in AMINO_ACIDS}
    total = 0
    for aa in seq.upper():
        if aa in counts:
            counts[aa] += 1
            total += 1

    if total == 0:
        return {aa: 0.0 for aa in AMINO_ACIDS}

    return {aa: count / total for aa, count in counts.items()}


def jensen_shannon_divergence(dist1: Dict[str, float], dist2: Dict[str, float]) -> float:
    """
    Compute Jensen-Shannon divergence between two distributions.

    Args:
        dist1: First distribution (dict of probabilities)
        dist2: Second distribution (dict of probabilities)

    Returns:
        JS divergence (0 = identical, 1 = maximally different)
    """
    # Get all keys
    keys = set(dist1.keys()) | set(dist2.keys())

    p = np.array([dist1.get(k, 0.0) for k in keys])
    q = np.array([dist2.get(k, 0.0) for k in keys])

    # Normalize
    p = p / (p.sum() + 1e-10)
    q = q / (q.sum() + 1e-10)

    # Average distribution
    m = 0.5 * (p + q)

    # KL divergences
    def kl_div(a, b):
        mask = (a > 0) & (b > 0)
        return np.sum(a[mask] * np.log(a[mask] / b[mask]))

    js = 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)
    return float(js)
