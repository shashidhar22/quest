"""Task-name aliasing for upstream baselines that pre-date benchmark_v2.

Some upstream baseline scripts (e.g. the legacy scripts/benchmark/*.py)
expect ICML-era task names like `trb_peptide_mhc_one`. Track A speaks
`as_trb_i` natively; this module lives here in case any baseline-specific
adapter needs to translate.

Currently empty — Track A wrappers all use AS-* names directly. If a
baseline ever needs the legacy form (e.g. when invoking pre-existing
internal helpers in the upstream wrapper), import from here.
"""

from __future__ import annotations

# AS-* (benchmark_v2) -> ICML-era equivalents (single-MHC class).
# Provided for forward-compat only. Class II splits do NOT have ICML
# equivalents; ICML lacked class II.
AS_TO_ICML = {
    "as_trb_i":    "trb_peptide_mhc_one",
    "as_tra_i":    "tra_peptide_mhc_one",
    "as_paired_i": "tra_trb_peptide_mhc_one",
    "as_trb_ii":   "trb_peptide_mhc_two",
    "as_tra_ii":   "tra_peptide_mhc_two",
    "as_paired_ii":"tra_trb_peptide_mhc_two",
}

ICML_TO_AS = {v: k for k, v in AS_TO_ICML.items()}


def to_icml(as_task: str) -> str:
    if as_task not in AS_TO_ICML:
        raise ValueError(f"unknown AS task: {as_task}")
    return AS_TO_ICML[as_task]


def to_as(icml_task: str) -> str:
    if icml_task not in ICML_TO_AS:
        raise ValueError(f"unknown ICML task: {icml_task}")
    return ICML_TO_AS[icml_task]
