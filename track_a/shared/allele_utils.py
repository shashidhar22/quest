"""MHC allele resolution — re-exported from track_b.

Same rationale as negative_sampling.py: avoid copy-pasted state.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_TRACK_B_PATH = Path("/home/ubuntu/quest/track_b/shared/allele_utils.py")

if not _TRACK_B_PATH.exists():
    raise ImportError(
        "track_b/shared/allele_utils.py is required but was not found "
        f"at {_TRACK_B_PATH}"
    )

_spec = importlib.util.spec_from_file_location(
    "_track_b_allele_utils", _TRACK_B_PATH
)
_module = importlib.util.module_from_spec(_spec)
sys.modules["_track_b_allele_utils"] = _module
assert _spec.loader is not None
_spec.loader.exec_module(_module)

is_hla_name = _module.is_hla_name
resolve_allele = _module.resolve_allele
filter_named_alleles = _module.filter_named_alleles

__all__ = ["is_hla_name", "resolve_allele", "filter_named_alleles"]
