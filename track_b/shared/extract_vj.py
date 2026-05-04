"""V/J gene extraction from full-length TCR chains.

For Phase-1 smoke tests, V/J calls are returned as 'UNK' (the ERGO-II / NetTCR
sentinel for missing genes). The model still runs; categorical embeddings for
V/J just collapse to the UNK padding-index embedding (effectively a constant).

For Phase 2 we need proper V/J extraction. Options:
- ANARCI (gold standard) — requires HMMER + germline DBs, not pip-installable on
  Python 3.7. The ergo_ii env is pinned to 3.7 because pytorch-lightning 0.7.6
  needs it. Workaround: build a separate Python-3.10 env just for V/J extraction
  and serialize the result to a parquet keyed by row_id.
- tidytcells germline match — tidytcells provides FR1-IMGT germline sequences
  via `tr.get_aa_sequence()`. Matching the first ~25 AA of trb_full against
  these gives a V-gene call. Less accurate than ANARCI but pip-clean.
- Reuse `quest.parsers.streaming_parser.StreamingParserComplete` if it has V/J
  inference (need to verify).

Cache: results keyed by row_id at track_b/shared/cache/vj_calls.parquet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE_PATH = Path("/home/ubuntu/quest/track_b/shared/cache/vj_calls.parquet")


def extract_vj_for_rows(df: pd.DataFrame, chain: str = "trb") -> pd.DataFrame:
    """Add `{chain}_v_gene` and `{chain}_j_gene` columns to df.

    Phase-1 implementation: returns 'UNK' for all rows. Phase 2 should swap
    this out for an ANARCI-based or tidytcells-germline-match implementation.

    Args:
        df: must have columns `row_id` and `{chain}_full` (the latter is unused
            in the Phase-1 stub but required for the Phase-2 implementation).
        chain: 'tra' or 'trb'.

    Returns:
        DataFrame with columns `{chain}_v_gene` and `{chain}_j_gene` added.
    """
    out = df.copy()
    out[f"{chain}_v_gene"] = "UNK"
    out[f"{chain}_j_gene"] = "UNK"
    return out
