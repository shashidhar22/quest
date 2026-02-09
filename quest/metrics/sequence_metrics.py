"""
Sequence Similarity Metrics

Provides BLOSUM62-based scoring, Levenshtein distance, sequence identity,
and batch similarity computation for TCR sequences.

Extracted from:
- scripts/training/tcr_seq2seq_trainer.py (BLOSUM62, levenshtein, identity, blosum functions)
- scripts/training/tcr_robust_contrastive_trainer.py (SequenceSimilarityCalculator)
"""

from functools import lru_cache
from typing import List

import torch


# =============================================================================
# Biological Constants for Sequence Metrics
# =============================================================================

# BLOSUM62 substitution matrix (symmetric, only store upper triangle + diagonal)
# Source: Henikoff & Henikoff (1992)
BLOSUM62 = {
    'A': {'A': 4, 'R': -1, 'N': -2, 'D': -2, 'C': 0, 'Q': -1, 'E': -1, 'G': 0, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 0, 'W': -3, 'Y': -2, 'V': 0},
    'R': {'A': -1, 'R': 5, 'N': 0, 'D': -2, 'C': -3, 'Q': 1, 'E': 0, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 2, 'M': -1, 'F': -3, 'P': -2, 'S': -1, 'T': -1, 'W': -3, 'Y': -2, 'V': -3},
    'N': {'A': -2, 'R': 0, 'N': 6, 'D': 1, 'C': -3, 'Q': 0, 'E': 0, 'G': 0, 'H': 1, 'I': -3, 'L': -3, 'K': 0, 'M': -2, 'F': -3, 'P': -2, 'S': 1, 'T': 0, 'W': -4, 'Y': -2, 'V': -3},
    'D': {'A': -2, 'R': -2, 'N': 1, 'D': 6, 'C': -3, 'Q': 0, 'E': 2, 'G': -1, 'H': -1, 'I': -3, 'L': -4, 'K': -1, 'M': -3, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -4, 'Y': -3, 'V': -3},
    'C': {'A': 0, 'R': -3, 'N': -3, 'D': -3, 'C': 9, 'Q': -3, 'E': -4, 'G': -3, 'H': -3, 'I': -1, 'L': -1, 'K': -3, 'M': -1, 'F': -2, 'P': -3, 'S': -1, 'T': -1, 'W': -2, 'Y': -2, 'V': -1},
    'Q': {'A': -1, 'R': 1, 'N': 0, 'D': 0, 'C': -3, 'Q': 5, 'E': 2, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 1, 'M': 0, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -2, 'Y': -1, 'V': -2},
    'E': {'A': -1, 'R': 0, 'N': 0, 'D': 2, 'C': -4, 'Q': 2, 'E': 5, 'G': -2, 'H': 0, 'I': -3, 'L': -3, 'K': 1, 'M': -2, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
    'G': {'A': 0, 'R': -2, 'N': 0, 'D': -1, 'C': -3, 'Q': -2, 'E': -2, 'G': 6, 'H': -2, 'I': -4, 'L': -4, 'K': -2, 'M': -3, 'F': -3, 'P': -2, 'S': 0, 'T': -2, 'W': -2, 'Y': -3, 'V': -3},
    'H': {'A': -2, 'R': 0, 'N': 1, 'D': -1, 'C': -3, 'Q': 0, 'E': 0, 'G': -2, 'H': 8, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -1, 'P': -2, 'S': -1, 'T': -2, 'W': -2, 'Y': 2, 'V': -3},
    'I': {'A': -1, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -3, 'E': -3, 'G': -4, 'H': -3, 'I': 4, 'L': 2, 'K': -3, 'M': 1, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -3, 'Y': -1, 'V': 3},
    'L': {'A': -1, 'R': -2, 'N': -3, 'D': -4, 'C': -1, 'Q': -2, 'E': -3, 'G': -4, 'H': -3, 'I': 2, 'L': 4, 'K': -2, 'M': 2, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -2, 'Y': -1, 'V': 1},
    'K': {'A': -1, 'R': 2, 'N': 0, 'D': -1, 'C': -3, 'Q': 1, 'E': 1, 'G': -2, 'H': -1, 'I': -3, 'L': -2, 'K': 5, 'M': -1, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
    'M': {'A': -1, 'R': -1, 'N': -2, 'D': -3, 'C': -1, 'Q': 0, 'E': -2, 'G': -3, 'H': -2, 'I': 1, 'L': 2, 'K': -1, 'M': 5, 'F': 0, 'P': -2, 'S': -1, 'T': -1, 'W': -1, 'Y': -1, 'V': 1},
    'F': {'A': -2, 'R': -3, 'N': -3, 'D': -3, 'C': -2, 'Q': -3, 'E': -3, 'G': -3, 'H': -1, 'I': 0, 'L': 0, 'K': -3, 'M': 0, 'F': 6, 'P': -4, 'S': -2, 'T': -2, 'W': 1, 'Y': 3, 'V': -1},
    'P': {'A': -1, 'R': -2, 'N': -2, 'D': -1, 'C': -3, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -4, 'P': 7, 'S': -1, 'T': -1, 'W': -4, 'Y': -3, 'V': -2},
    'S': {'A': 1, 'R': -1, 'N': 1, 'D': 0, 'C': -1, 'Q': 0, 'E': 0, 'G': 0, 'H': -1, 'I': -2, 'L': -2, 'K': 0, 'M': -1, 'F': -2, 'P': -1, 'S': 4, 'T': 1, 'W': -3, 'Y': -2, 'V': -2},
    'T': {'A': 0, 'R': -1, 'N': 0, 'D': -1, 'C': -1, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 5, 'W': -2, 'Y': -2, 'V': 0},
    'W': {'A': -3, 'R': -3, 'N': -4, 'D': -4, 'C': -2, 'Q': -2, 'E': -3, 'G': -2, 'H': -2, 'I': -3, 'L': -2, 'K': -3, 'M': -1, 'F': 1, 'P': -4, 'S': -3, 'T': -2, 'W': 11, 'Y': 2, 'V': -3},
    'Y': {'A': -2, 'R': -2, 'N': -2, 'D': -3, 'C': -2, 'Q': -1, 'E': -2, 'G': -3, 'H': 2, 'I': -1, 'L': -1, 'K': -2, 'M': -1, 'F': 3, 'P': -3, 'S': -2, 'T': -2, 'W': 2, 'Y': 7, 'V': -1},
    'V': {'A': 0, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -2, 'E': -2, 'G': -3, 'H': -3, 'I': 3, 'L': 1, 'K': -2, 'M': 1, 'F': -1, 'P': -2, 'S': -2, 'T': 0, 'W': -3, 'Y': -1, 'V': 4},
}

# Standard amino acids
STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


# =============================================================================
# Sequence Similarity Metrics
# =============================================================================


def levenshtein_distance(seq1: str, seq2: str) -> int:
    """
    Compute Levenshtein (edit) distance between two sequences.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Edit distance (insertions, deletions, substitutions)
    """
    if len(seq1) < len(seq2):
        return levenshtein_distance(seq2, seq1)

    if len(seq2) == 0:
        return len(seq1)

    previous_row = range(len(seq2) + 1)
    for i, c1 in enumerate(seq1):
        current_row = [i + 1]
        for j, c2 in enumerate(seq2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def sequence_identity(seq1: str, seq2: str) -> float:
    """
    Compute sequence identity (fraction of matching positions).

    Uses global alignment where shorter sequence is compared position-by-position.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Identity fraction (0.0 to 1.0)
    """
    if not seq1 or not seq2:
        return 0.0

    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))

    matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])

    # Normalize by max length to penalize length differences
    return matches / max_len


def blosum62_similarity(seq1: str, seq2: str) -> float:
    """
    Compute BLOSUM62 similarity score between two sequences.

    Compares aligned positions using BLOSUM62 substitution scores.
    Unaligned positions (length difference) are penalized with gap penalty.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Total BLOSUM62 score (can be negative)
    """
    if not seq1 or not seq2:
        return 0.0

    min_len = min(len(seq1), len(seq2))
    gap_penalty = -4  # Standard BLOSUM62 gap penalty

    score = 0.0
    for i in range(min_len):
        aa1 = seq1[i].upper()
        aa2 = seq2[i].upper()
        if aa1 in BLOSUM62 and aa2 in BLOSUM62[aa1]:
            score += BLOSUM62[aa1][aa2]
        else:
            # Unknown amino acid, use gap penalty
            score += gap_penalty

    # Penalize length differences
    len_diff = abs(len(seq1) - len(seq2))
    score += len_diff * gap_penalty

    return score


def blosum62_normalized(seq1: str, seq2: str) -> float:
    """
    Compute length-normalized BLOSUM62 similarity.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Normalized score (score per position)
    """
    if not seq1 or not seq2:
        return 0.0

    score = blosum62_similarity(seq1, seq2)
    max_len = max(len(seq1), len(seq2))
    return score / max_len


# =============================================================================
# Batch Sequence Similarity (from tcr_robust_contrastive_trainer.py)
# =============================================================================


class SequenceSimilarityCalculator:
    """
    Computes sequence-level similarity for hidden positive detection.

    Key insight: We use SEQUENCE similarity (Levenshtein), not EMBEDDING similarity.
    Embedding similarity is unreliable early in training, but sequence similarity
    is deterministic and biologically meaningful.

    This runs in CPU DataLoader workers, not on GPU.
    """

    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    @staticmethod
    @lru_cache(maxsize=50000)
    def levenshtein_ratio(s1: str, s2: str) -> float:
        """
        Compute normalized Levenshtein similarity (0-1).
        Cached to avoid recomputation for repeated pairs.
        """
        if s1 == s2:
            return 1.0

        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0

        # Standard DP for edit distance
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # deletion
                    dp[i][j-1] + 1,      # insertion
                    dp[i-1][j-1] + cost  # substitution
                )

        return 1.0 - dp[len1][len2] / max(len1, len2)

    def compute_batch_similarity_mask(self, sequences: List[str]) -> torch.Tensor:
        """
        Compute boolean mask indicating which sequence pairs are similar.

        Returns:
            mask: (batch, batch) tensor where mask[i,j] = True if sequences[i]
                  and sequences[j] have similarity >= threshold.
                  Diagonal is always False (self-similarity doesn't count).
        """
        batch_size = len(sequences)
        mask = torch.zeros(batch_size, batch_size, dtype=torch.bool)

        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                sim = self.levenshtein_ratio(sequences[i], sequences[j])
                if sim >= self.threshold:
                    mask[i, j] = True
                    mask[j, i] = True

        # Diagonal is False (handled by loop starting at j = i + 1)
        return mask
