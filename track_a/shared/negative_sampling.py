"""Negative sampling for binary baselines on AS — re-exported from track_b.

Cross-track parity: Track A and Track B MUST sample identical negatives
(seed=43, 5x peptide-swap, positive-set filtered) so their AUCs are
apples-to-apples in the paper.

This module re-exports track_b/shared/negative_sampling.py via path import.
We avoid copying so a future fix to negative-sampling propagates to both
tracks.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Resolve inside this checkout first, so judge.py imports (and is testable) off the training box.
# The absolute path is kept as a fallback for a box layout where track_b sits elsewhere.
_CANDIDATES = (
    Path(__file__).resolve().parents[2] / "track_b" / "shared" / "negative_sampling.py",
    Path("/home/ubuntu/quest/track_b/shared/negative_sampling.py"),
)
_TRACK_B_PATH = next((c for c in _CANDIDATES if c.exists()), None)

if _TRACK_B_PATH is None:
    raise ImportError(
        "track_b/shared/negative_sampling.py is required for cross-track "
        "negative-sample parity but was not found at any of: "
        + ", ".join(str(c) for c in _CANDIDATES)
    )

_spec = importlib.util.spec_from_file_location(
    "_track_b_negative_sampling", _TRACK_B_PATH
)
_module = importlib.util.module_from_spec(_spec)
sys.modules["_track_b_negative_sampling"] = _module
assert _spec.loader is not None
_spec.loader.exec_module(_module)

NEG_SEED = _module.NEG_SEED
NEG_RATIO = _module.NEG_RATIO
sample_negatives = _module.sample_negatives
attach_labels = _module.attach_labels

__all__ = ["NEG_SEED", "NEG_RATIO", "sample_negatives", "attach_labels"]
