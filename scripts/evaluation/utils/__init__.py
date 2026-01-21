"""Utility modules for pMHC prediction tool evaluation."""

from .input_formatters import (
    format_for_netmhcpan,
    format_for_netmhciipan,
    format_for_mhcflurry,
    convert_allele_netmhcpan,
    convert_allele_netmhciipan,
    convert_allele_mhcflurry,
)
from .output_parsers import (
    parse_netmhcpan_output,
    parse_netmhciipan_output,
    parse_mhcflurry_output,
)
from .metrics import (
    compute_auc_roc,
    compute_auc_pr,
    compute_sensitivity_at_threshold,
    compute_ppv_at_topk,
    compute_all_metrics,
)

__all__ = [
    "format_for_netmhcpan",
    "format_for_netmhciipan",
    "format_for_mhcflurry",
    "convert_allele_netmhcpan",
    "convert_allele_netmhciipan",
    "convert_allele_mhcflurry",
    "parse_netmhcpan_output",
    "parse_netmhciipan_output",
    "parse_mhcflurry_output",
    "compute_auc_roc",
    "compute_auc_pr",
    "compute_sensitivity_at_threshold",
    "compute_ppv_at_topk",
    "compute_all_metrics",
]
