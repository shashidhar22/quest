"""MHC allele handling: HLA name vs full protein sequence reconciliation.

The benchmark splits store mhc_one_allele as either:
- An HLA name (e.g., "HLA-B*08:01") when the upstream lookup hit, OR
- A full ~250 AA protein sequence when the lookup missed.

Most baselines expect HLA names. This module reverse-maps protein sequences
to HLA names via NetMHCpan's allele table.

NetMHCpan's MHC_pseudo.dat lookup file is the canonical source. We expect it
at one of:
- /home/ubuntu/quest/data/raw_data/netmhcpan/MHC_pseudo.dat
- /home/ubuntu/quest/track_b/shared/cache/MHC_pseudo.dat

If neither is available, resolve_allele() falls back to identity for HLA-named
inputs and None for protein-sequence inputs.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

HLA_NAME_RE = re.compile(r"^HLA-[A-Z]\*\d+:\d+")

PSEUDO_DAT_PATHS = [
    Path("/home/ubuntu/quest/data/raw_data/netmhcpan/MHC_pseudo.dat"),
    Path("/home/ubuntu/quest/track_b/shared/cache/MHC_pseudo.dat"),
]


def is_hla_name(value: str | None) -> bool:
    """True iff value looks like an HLA name (e.g., HLA-B*08:01)."""
    if value is None or not isinstance(value, str):
        return False
    return bool(HLA_NAME_RE.match(value))


@lru_cache(maxsize=1)
def _load_pseudo_table() -> dict[str, str]:
    """Build {full_protein_seq: hla_name} dict from NetMHCpan's MHC_pseudo.dat.

    The MHC_pseudo.dat file is whitespace-separated `<allele> <pseudo34>`. We
    do NOT have the full protein sequence in that file; we have the 34-AA
    pseudo-sequence. So this dict actually maps pseudo34 -> hla_name and is
    used in conjunction with quest's mhc_one_pocket_contact (47 AA).

    For the protein-sequence fallback we instead need a separate
    {full_protein_seq: hla_name} mapping. This is built from
    quest's enriched parquet lookup (see BENCHMARK_SPLITS.md sec 6) by joining
    mhc_one (full protein seq) with mhc_one_allele (HLA name) where the lookup
    succeeded. Built lazily by build_protein_to_hla_index().
    """
    for path in PSEUDO_DAT_PATHS:
        if path.exists():
            mapping = {}
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    allele, pseudo = parts[0], parts[1]
                    mapping[pseudo] = allele
            return mapping
    return {}


@lru_cache(maxsize=1)
def _load_protein_to_hla_index() -> dict[str, str]:
    """Build {mhc_one_protein_seq: hla_name} from the splits themselves.

    Strategy: scan all *_train.parquet files and harvest pairs where
    is_hla_name(mhc_one_allele) and mhc_one is non-empty. Many distinct rows
    share the same (protein_seq, hla_name) pair so the dict is small (~hundred
    HLA alleles). Cached on disk after first build.
    """
    cache_path = Path("/home/ubuntu/quest/track_b/shared/cache/protein_to_hla.json")
    if cache_path.exists():
        import json

        return json.loads(cache_path.read_text())

    import json

    import pandas as pd

    splits_dir = Path("/home/ubuntu/quest/benchmark_v2/splits")
    mapping: dict[str, str] = {}
    for path in sorted(splits_dir.glob("as_*_train.parquet")):
        df = pd.read_parquet(path, columns=["mhc_one", "mhc_one_allele"])
        for protein, allele in zip(df["mhc_one"], df["mhc_one_allele"]):
            if is_hla_name(allele) and isinstance(protein, str) and protein:
                if protein not in mapping:
                    mapping[protein] = allele
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(mapping))
    return mapping


def resolve_allele(value: str | None) -> str | None:
    """Return HLA name for value, or None if unresolvable.

    - If value is already an HLA name: passthrough.
    - If value is a protein sequence: look up via _load_protein_to_hla_index().
    - If value is None or unrecognized: None.
    """
    if value is None or not isinstance(value, str) or not value:
        return None
    if is_hla_name(value):
        return value
    return _load_protein_to_hla_index().get(value)


def filter_named_alleles(df, allele_col: str = "mhc_one_allele"):
    """Keep only rows whose `allele_col` resolves to an HLA name.

    Adds a resolved `mhc_hla_name` column with the canonical HLA name.
    Drops rows where resolution failed.
    """
    resolved = df[allele_col].map(resolve_allele)
    keep = resolved.notna()
    out = df.loc[keep].copy()
    out["mhc_hla_name"] = resolved[keep].values
    return out
