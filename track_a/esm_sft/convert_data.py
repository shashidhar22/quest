"""Load + prepare AS data for ESM SFT (re-uses LSTM/MLP loader)."""

from __future__ import annotations

# Reuse the splitting + negative-sampling logic from lstm_mlp; the data-loading
# patterns are identical (positives + 5x peptide-swap negatives, class-II →
# class-I-companion training, class-II candidate-pool synthesis).
from track_a.lstm_mlp.convert_data import (
    load_train_val,
    load_test_condition,
    load_candidates,
)

__all__ = ["load_train_val", "load_test_condition", "load_candidates"]
