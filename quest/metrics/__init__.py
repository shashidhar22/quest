"""
QUEST Metrics Package

Provides biological sequence metrics, biophysical property analysis,
retrieval metrics, position-wise analysis, and CDR region analysis
for TCR and MHC sequence evaluation.
"""

from quest.metrics.sequence_metrics import (
    BLOSUM62,
    STANDARD_AMINO_ACIDS,
    levenshtein_distance,
    sequence_identity,
    blosum62_similarity,
    blosum62_normalized,
    SequenceSimilarityCalculator,
)

from quest.metrics.biophysical import (
    KYTE_DOOLITTLE,
    AMINO_ACID_CHARGE,
    compute_hydrophobicity_profile,
    hydrophobicity_correlation,
    compute_net_charge,
    aa_frequency_distribution,
    jensen_shannon_divergence,
)

from quest.metrics.retrieval_metrics import (
    compute_retrieval_metrics_from_scores,
    compute_per_peptide_auc,
    compute_bootstrap_ci,
    compute_per_epitope_breakdown,
    compute_random_baselines,
    compute_lift_metrics,
    compute_all_unified_metrics,
)

from quest.metrics.position_wise import PositionWiseAnalyzer

from quest.metrics.cdr_analysis import (
    CDR_WEIGHT_PRESETS,
    get_cdr_positions_anarci,
    CDRAnnotationCache,
    build_cdr_position_weights,
)

__all__ = [
    # sequence_metrics
    "BLOSUM62",
    "STANDARD_AMINO_ACIDS",
    "levenshtein_distance",
    "sequence_identity",
    "blosum62_similarity",
    "blosum62_normalized",
    "SequenceSimilarityCalculator",
    # biophysical
    "KYTE_DOOLITTLE",
    "AMINO_ACID_CHARGE",
    "compute_hydrophobicity_profile",
    "hydrophobicity_correlation",
    "compute_net_charge",
    "aa_frequency_distribution",
    "jensen_shannon_divergence",
    # retrieval_metrics
    "compute_retrieval_metrics_from_scores",
    "compute_per_peptide_auc",
    "compute_bootstrap_ci",
    "compute_per_epitope_breakdown",
    "compute_random_baselines",
    "compute_lift_metrics",
    "compute_all_unified_metrics",
    # position_wise
    "PositionWiseAnalyzer",
    # cdr_analysis
    "CDR_WEIGHT_PRESETS",
    "get_cdr_positions_anarci",
    "CDRAnnotationCache",
    "build_cdr_position_weights",
]
