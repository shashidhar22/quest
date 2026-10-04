"""Shared utilities for Track A baselines on benchmark_v2.

Mirrors track_b/shared/ but extends to:
- AS Class II (eval-only) loading
- Retrieval-metrics emission (Recall@K, MRR, Hit@1) per-condition
- User-spec results JSON + per-(query, candidate) score sidecar

Cross-track parity is enforced: negative_sampling.py and allele_utils.py
are imported from track_b/shared/ so Track A and Track B see identical
negatives and allele resolution.
"""
