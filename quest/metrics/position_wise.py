"""
Position-Wise Sequence Analysis

Provides per-position accuracy tracking, CDR3 conserved position analysis,
and substitution error pattern detection for generated vs. reference sequences.

Extracted from scripts/training/tcr_seq2seq_trainer.py.
"""

from typing import Dict, List, Tuple

import numpy as np


class PositionWiseAnalyzer:
    """
    Analyze per-position accuracy and confusion patterns.

    Tracks:
    - Accuracy at each sequence position
    - CDR3 conserved position accuracy (first 3 and last 2 positions)
    - Most common substitution errors
    """

    def __init__(self):
        self.position_correct: Dict[int, int] = {}
        self.position_total: Dict[int, int] = {}
        self.confusion: Dict[Tuple[str, str], int] = {}  # (ref, gen) -> count
        self.total_samples = 0

    def update(self, generated: str, reference: str) -> None:
        """
        Update statistics with a new generated/reference pair.

        Args:
            generated: Generated sequence
            reference: Reference sequence
        """
        self.total_samples += 1
        min_len = min(len(generated), len(reference))

        for i in range(min_len):
            ref_aa = reference[i].upper()
            gen_aa = generated[i].upper()

            # Update position accuracy
            if i not in self.position_total:
                self.position_total[i] = 0
                self.position_correct[i] = 0

            self.position_total[i] += 1
            if ref_aa == gen_aa:
                self.position_correct[i] += 1
            else:
                # Track confusion
                key = (ref_aa, gen_aa)
                self.confusion[key] = self.confusion.get(key, 0) + 1

    def get_position_accuracy(self) -> Dict[int, float]:
        """
        Get accuracy at each position.

        Returns:
            Dict mapping position to accuracy (0-1)
        """
        return {
            pos: self.position_correct[pos] / self.position_total[pos]
            for pos in sorted(self.position_total.keys())
            if self.position_total[pos] > 0
        }

    def get_mean_position_accuracy(self) -> float:
        """Get mean accuracy across all positions."""
        accuracies = self.get_position_accuracy()
        if not accuracies:
            return 0.0
        return float(np.mean(list(accuracies.values())))

    def get_cdr3_conserved_accuracy(self) -> Dict[str, float]:
        """
        Get accuracy for CDR3 conserved positions.

        CDR3 typically has conserved residues at:
        - First 3 positions (often Cys at position 0)
        - Last 2 positions (often Phe/Trp at -2, Gly at -1)

        Returns:
            Dict with 'first3' and 'last2' accuracy
        """
        pos_acc = self.get_position_accuracy()

        # First 3 positions
        first3_acc = []
        for i in range(3):
            if i in pos_acc:
                first3_acc.append(pos_acc[i])

        # Last 2 positions (need to find max position)
        if pos_acc:
            max_pos = max(pos_acc.keys())
            last2_acc = []
            for i in range(max(0, max_pos - 1), max_pos + 1):
                if i in pos_acc:
                    last2_acc.append(pos_acc[i])
        else:
            last2_acc = []

        return {
            'first3': float(np.mean(first3_acc)) if first3_acc else 0.0,
            'last2': float(np.mean(last2_acc)) if last2_acc else 0.0,
        }

    def get_top_substitution_errors(self, top_k: int = 10) -> List[Tuple[str, str, int]]:
        """
        Get most common substitution errors.

        Args:
            top_k: Number of top errors to return

        Returns:
            List of (reference_aa, generated_aa, count) tuples
        """
        sorted_errors = sorted(
            self.confusion.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [(ref, gen, count) for (ref, gen), count in sorted_errors[:top_k]]

    def reset(self) -> None:
        """Reset all accumulators."""
        self.position_correct.clear()
        self.position_total.clear()
        self.confusion.clear()
        self.total_samples = 0
