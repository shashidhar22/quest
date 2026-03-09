#!/usr/bin/env python3
"""
Create train/val/test splits and nested training subsets for mlm_full dedup output.

7-phase pipeline:
  Phase 0a: Resolve MHC — replace MHC allele IDs with full IMGT/HLA protein sequences (optional)
  Phase 0: Scan and index — classify rows, extract quintet/quartet indices
  Phase 1: Cluster — MMseqs2 clustering of TRB and peptide sequences
  Phase 2: Assign splits — cluster-based for quintets/quartets, hash-based for rest
  Phase 3: Build nested training subsets — proportional, stratified, balanced, ablation
  Phase 4: Binding probe eval sets — from BATMAN + TRAIT standardized data
  Phase 5: Validate and write manifest — comprehensive checks + checksums

Usage:
    python scripts/data_processing/create_mlm_splits.py \
        --input_dir data/deduplicated/mlm_full \
        --output_dir data/splits/mlm_full \
        --hla_dir data/databases/IMGTHLA/fasta \
        --seed 42 \
        --num_workers 32 \
        --trb_similarity 0.9 \
        --peptide_similarity 0.8 \
        --batman_dir data/standardized/batman \
        --trait_dir data/standardized/trait \
        --trait_neg_sample 18000 \
        --resume
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Imports from existing codebase
# ---------------------------------------------------------------------------
from scripts.analysis.summarize_dedup_output import parse_permutation_key
from scripts.data_processing.create_tcr_specificity_dataset import (
    cluster_peptides_mmseqs2,
    cluster_sequences_mmseqs2,
)
from quest.parsers.utils import parse_imgt_four_digit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MOLECULE_TYPES = ["tra", "trb", "peptide", "mhc_one", "mhc_two"]

CARDINALITY_LABELS = {
    1: "singles",
    2: "pairs",
    3: "triplets",
    4: "quartets",
    5: "quintets",
}

# Proportional subset sizes
PROPORTIONAL_SIZES = [1_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000, 100_000_000]

# Balanced subset sizes and per-group share
BALANCED_SIZES = {
    "balanced_1M": (1_000_000, 200_000),
    "balanced_5M": (5_000_000, 1_000_000),
    "balanced_25M": (25_000_000, 5_000_000),
}

# Stratified enriched target fractions by cardinality
# quintets=5%, quartets=24%, triplets=15%, pairs=20%, singles=50% (approximate)
STRATIFIED_FRACTIONS = {5: 0.05, 4: 0.24, 3: 0.15, 2: 0.20, 1: 0.50}
STRATIFIED_TOTAL = 3_650_000

# Oversampling ablation variants: (quintet_factor, quartet_factor, total_target)
OVERSAMPLING_VARIANTS = {
    "stratified_enriched_5M": (2, 1, 5_000_000),
    "stratified_enriched_8M": (5, 2, 8_000_000),
    "stratified_enriched_15M": (10, 5, 15_000_000),
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _hash_row(seed: int, batch_idx: int, row_idx: int) -> int:
    """Deterministic hash for a row, returns value in [0, 10000)."""
    h = hashlib.md5(f"{seed}_{batch_idx}_{row_idx}".encode()).hexdigest()
    return int(h, 16) % 10000


def _subset_score(seed: int, batch_idx: int, row_idx: int) -> float:
    """Deterministic score in [0, 1) for nested subset assignment."""
    h = hashlib.md5(f"{seed}_subset_{batch_idx}_{row_idx}".encode()).hexdigest()
    return float(int(h, 16) % (2**32)) / (2**32)


def _batch_idx_from_path(filepath: Path) -> int:
    """Extract batch index from filename like 'batch_000123.parquet'."""
    stem = filepath.stem
    # Try numeric suffix
    parts = stem.split("_")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    # Fallback: use sorted position
    return 0


def _get_cardinality(perm_key: str, cache: dict) -> int:
    """Get cardinality (number of fields) for a permutation key."""
    if perm_key not in cache:
        cache[perm_key] = len(parse_permutation_key(perm_key))
    return cache[perm_key]


def _extract_sequence_at_position(sequence: str, perm_key: str, target_field: str,
                                   field_cache: dict) -> Optional[str]:
    """Extract the subsequence for a specific field from a space-separated sequence."""
    if perm_key not in field_cache:
        field_cache[perm_key] = parse_permutation_key(perm_key)
    fields = field_cache[perm_key]
    if target_field not in fields:
        return None
    idx = fields.index(target_field)
    parts = sequence.split(" ")
    if idx < len(parts):
        return parts[idx]
    return None


def _sha256_file(filepath: Path) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _phase_complete(output_dir: Path, phase: int, checkpoint_files: List[str]) -> bool:
    """Check if all checkpoint files for a phase exist."""
    phase_dir = output_dir / f"phase{phase}_{'index' if phase == 0 else ['index','clusters','splits','subsets','binding_probes','manifest'][phase]}"
    return all((phase_dir / f).exists() for f in checkpoint_files)


# ---------------------------------------------------------------------------
# Phase 0a: Resolve MHC alleles → full protein sequences
# ---------------------------------------------------------------------------

_RESOLVE_HLA_DICT = None
_RESOLVE_FIELD_CACHE = None


def _init_resolve_worker(hla_dict: dict):
    """Initializer for resolve workers — shares hla_dict across tasks."""
    global _RESOLVE_HLA_DICT, _RESOLVE_FIELD_CACHE, _RESOLVE_PREFIX_CACHE
    _RESOLVE_HLA_DICT = hla_dict
    _RESOLVE_FIELD_CACHE = {}
    # Build prefix cache: "A*02" → "A*02:01" (first 4-digit match)
    # Same pattern as transform_mhc_restriction in quest/parsers/utils.py
    _RESOLVE_PREFIX_CACHE = {}
    for key in sorted(hla_dict.keys()):
        parts = key.split(":")
        for i in range(len(parts), 0, -1):
            prefix = ":".join(parts[:i])
            if prefix not in _RESOLVE_PREFIX_CACHE:
                _RESOLVE_PREFIX_CACHE[prefix] = key


def _resolve_mhc_in_sequence(
    sequence: str, perm_key: str, hla_dict: dict, field_cache: dict
) -> Tuple[str, int, int]:
    """Replace MHC allele IDs with full protein sequences in a space-separated sequence.

    Returns: (resolved_sequence, n_resolved, n_failed)
    """
    if perm_key not in field_cache:
        field_cache[perm_key] = parse_permutation_key(perm_key)
    fields = field_cache[perm_key]

    mhc_indices = [i for i, f in enumerate(fields) if f in ("mhc_one", "mhc_two")]
    if not mhc_indices:
        return sequence, 0, 0

    parts = sequence.split(" ")
    n_resolved = 0
    n_failed = 0

    for idx in mhc_indices:
        if idx >= len(parts):
            n_failed += 1
            continue
        allele = parts[idx]
        if not allele or allele in ("", "nan", "None"):
            continue
        # Strip HLA- prefix
        lookup_key = allele
        if lookup_key.startswith("HLA-"):
            lookup_key = lookup_key[4:]
        # Truncate to 4-digit resolution
        colon_parts = lookup_key.split(":")
        if len(colon_parts) > 2:
            lookup_key = ":".join(colon_parts[:2])

        full_seq = hla_dict.get(lookup_key)
        if not full_seq:
            # Fallback: 2-digit allele → first matching 4-digit allele
            resolved_key = _RESOLVE_PREFIX_CACHE.get(lookup_key)
            if resolved_key:
                full_seq = hla_dict.get(resolved_key)
        if full_seq:
            parts[idx] = full_seq
            n_resolved += 1
        else:
            n_failed += 1

    return " ".join(parts), n_resolved, n_failed


def _resolve_mhc_file(filepath: str) -> Dict[str, Any]:
    """Worker: resolve MHC alleles in one parquet file."""
    global _RESOLVE_HLA_DICT, _RESOLVE_FIELD_CACHE, _RESOLVE_PREFIX_CACHE

    table = pq.read_table(filepath, columns=["permutation_key", "sequence"])
    perm_keys = table.column("permutation_key").to_pylist()
    sequences = table.column("sequence").to_pylist()

    resolved_seqs = []
    total_resolved = 0
    total_failed = 0
    n_with_mhc = 0
    unresolved_alleles: Set[str] = set()
    length_records: List[Tuple[str, int]] = []

    for pk, seq in zip(perm_keys, sequences):
        new_seq, nr, nf = _resolve_mhc_in_sequence(
            seq, pk, _RESOLVE_HLA_DICT, _RESOLVE_FIELD_CACHE
        )
        resolved_seqs.append(new_seq)
        total_resolved += nr
        total_failed += nf
        if nr > 0 or nf > 0:
            n_with_mhc += 1

        # Track unresolved alleles
        if nf > 0:
            if pk not in _RESOLVE_FIELD_CACHE:
                _RESOLVE_FIELD_CACHE[pk] = parse_permutation_key(pk)
            fields = _RESOLVE_FIELD_CACHE[pk]
            parts = seq.split(" ")
            for i, f in enumerate(fields):
                if f in ("mhc_one", "mhc_two") and i < len(parts):
                    allele = parts[i]
                    if allele and allele not in ("", "nan", "None"):
                        lk = allele
                        if lk.startswith("HLA-"):
                            lk = lk[4:]
                        cp = lk.split(":")
                        if len(cp) > 2:
                            lk = ":".join(cp[:2])
                        if lk not in _RESOLVE_HLA_DICT and lk not in _RESOLVE_PREFIX_CACHE:
                            unresolved_alleles.add(allele)

        # Token length: space-separated parts + 2 special tokens (BOS/EOS)
        length_records.append((pk, len(new_seq.split(" ")) + 2))

    # Write resolved parquet with same schema
    new_table = pa.table({
        "permutation_key": perm_keys,
        "sequence": resolved_seqs,
    })
    return {
        "table": new_table,
        "total_rows": len(perm_keys),
        "n_with_mhc": n_with_mhc,
        "resolved": total_resolved,
        "failed": total_failed,
        "unresolved_alleles": list(unresolved_alleles),
        "length_records": length_records,
    }


def phase0a_resolve_mhc(
    input_dir: Path, output_dir: Path, hla_dir: Path, num_workers: int
) -> Path:
    """Phase 0a: Create resolved parquets with MHC allele IDs → full protein sequences."""
    resolved_dir = output_dir / "phase0a_resolved"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    stats_path = resolved_dir / "resolution_stats.json"

    source_files = sorted(input_dir.glob("*.parquet"))
    if not source_files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    # Checkpoint: skip if resolved dir has matching file count AND resolution succeeded
    existing = sorted(resolved_dir.glob("*.parquet"))
    if len(existing) == len(source_files) and stats_path.exists():
        with open(stats_path) as f:
            prev_stats = json.load(f)
        if prev_stats.get("successful_resolutions", 0) > 0:
            print("  Phase 0a: Loading from checkpoint...")
            return resolved_dir
        else:
            print("  Phase 0a: Previous run had 0 resolutions, re-running...")

    print("  Phase 0a: Loading IMGT/HLA dictionary...")
    hla_dict = parse_imgt_four_digit(str(hla_dir))
    if not hla_dict:
        raise ValueError(
            f"No HLA alleles loaded from {hla_dir}. "
            f"Check that directory contains *_prot.fasta files."
        )
    print(f"    Loaded {len(hla_dict):,} alleles from IMGT/HLA")

    print(f"  Phase 0a: Resolving MHC alleles in {len(source_files)} files...")

    # Aggregate stats
    total_rows = 0
    total_with_mhc = 0
    total_resolved = 0
    total_failed = 0
    all_unresolved: Set[str] = set()
    all_length_records: List[Tuple[str, int]] = []

    with multiprocessing.Pool(
        num_workers,
        initializer=_init_resolve_worker,
        initargs=(hla_dict,),
    ) as pool:
        results_iter = pool.imap(
            _resolve_mhc_file, [str(f) for f in source_files]
        )
        for src_file, result in tqdm(
            zip(source_files, results_iter),
            total=len(source_files),
            desc="    Resolving MHC",
        ):
            # Write resolved parquet with same filename
            out_path = resolved_dir / src_file.name
            pq.write_table(result["table"], out_path)

            total_rows += result["total_rows"]
            total_with_mhc += result["n_with_mhc"]
            total_resolved += result["resolved"]
            total_failed += result["failed"]
            all_unresolved.update(result["unresolved_alleles"])
            all_length_records.extend(result["length_records"])

    # --- Integrated length audit ---
    print("\n    === Sequence Length Audit (with resolved MHC) ===")
    length_by_cardinality: Dict[int, List[int]] = defaultdict(list)
    card_cache: Dict[str, int] = {}
    for pk, tok_len in all_length_records:
        card = _get_cardinality(pk, card_cache)
        length_by_cardinality[card].append(tok_len)

    all_lengths = []
    length_audit = {}
    for card in sorted(length_by_cardinality.keys()):
        lengths = np.array(length_by_cardinality[card])
        all_lengths.extend(lengths.tolist())
        label = CARDINALITY_LABELS.get(card, f"card_{card}")
        n_over_1024 = int(np.sum(lengths > 1024))
        n_over_2048 = int(np.sum(lengths > 2048))
        audit_entry = {
            "count": len(lengths),
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "p95": float(np.percentile(lengths, 95)),
            "p99": float(np.percentile(lengths, 99)),
            "max": int(np.max(lengths)),
            "pct_over_1024": round(n_over_1024 / len(lengths) * 100, 3),
            "pct_over_2048": round(n_over_2048 / len(lengths) * 100, 3),
        }
        length_audit[label] = audit_entry
        print(
            f"    {label:>10s}: n={len(lengths):>10,}  "
            f"mean={audit_entry['mean']:.0f}  median={audit_entry['median']:.0f}  "
            f"p95={audit_entry['p95']:.0f}  p99={audit_entry['p99']:.0f}  "
            f"max={audit_entry['max']}  "
            f">1024={audit_entry['pct_over_1024']:.1f}%  >2048={audit_entry['pct_over_2048']:.1f}%"
        )

    # Overall
    all_arr = np.array(all_lengths)
    n_over_1024 = int(np.sum(all_arr > 1024))
    n_over_2048 = int(np.sum(all_arr > 2048))
    length_audit["overall"] = {
        "count": len(all_arr),
        "mean": float(np.mean(all_arr)),
        "median": float(np.median(all_arr)),
        "p95": float(np.percentile(all_arr, 95)),
        "p99": float(np.percentile(all_arr, 99)),
        "max": int(np.max(all_arr)),
        "pct_over_1024": round(n_over_1024 / len(all_arr) * 100, 3),
        "pct_over_2048": round(n_over_2048 / len(all_arr) * 100, 3),
    }
    print(
        f"    {'overall':>10s}: n={len(all_arr):>10,}  "
        f"mean={length_audit['overall']['mean']:.0f}  "
        f"median={length_audit['overall']['median']:.0f}  "
        f"p95={length_audit['overall']['p95']:.0f}  "
        f"p99={length_audit['overall']['p99']:.0f}  "
        f"max={length_audit['overall']['max']}  "
        f">1024={length_audit['overall']['pct_over_1024']:.1f}%  "
        f">2048={length_audit['overall']['pct_over_2048']:.1f}%"
    )

    # Save stats
    resolution_rate = (
        total_resolved / (total_resolved + total_failed) * 100
        if (total_resolved + total_failed) > 0
        else 0.0
    )
    stats = {
        "total_rows": total_rows,
        "rows_with_mhc_fields": total_with_mhc,
        "successful_resolutions": total_resolved,
        "failed_resolutions": total_failed,
        "resolution_rate_pct": round(resolution_rate, 2),
        "unresolved_alleles": sorted(all_unresolved),
        "imgt_alleles_loaded": len(hla_dict),
        "length_audit": length_audit,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n    Resolution rate: {resolution_rate:.1f}%")
    print(f"    Resolved: {total_resolved:,}, Failed: {total_failed:,}")
    if all_unresolved:
        print(f"    Unresolved alleles ({len(all_unresolved)}): "
              f"{sorted(all_unresolved)[:20]}{'...' if len(all_unresolved) > 20 else ''}")
    print(f"    Stats saved to {stats_path}")

    return resolved_dir


# ---------------------------------------------------------------------------
# Phase 0: Scan and Index
# ---------------------------------------------------------------------------


def _index_file(args: Tuple[str, int]) -> Dict[str, Any]:
    """Worker: read one parquet file and extract quintet/quartet rows + counts."""
    filepath, batch_idx = args
    table = pq.read_table(filepath, columns=["permutation_key", "sequence"])
    perm_keys = table.column("permutation_key").to_pylist()
    sequences = table.column("sequence").to_pylist()

    field_cache: Dict[str, List[str]] = {}
    group_counts: Dict[str, int] = defaultdict(int)
    quintet_rows = []
    quartet_rows = []

    for row_idx, (pk, seq) in enumerate(zip(perm_keys, sequences)):
        if pk not in field_cache:
            field_cache[pk] = parse_permutation_key(pk)
        fields = field_cache[pk]
        cardinality = len(fields)
        group_counts[pk] += 1

        if cardinality == 5:
            quintet_rows.append((batch_idx, row_idx, pk, seq))
        elif cardinality == 4:
            quartet_rows.append((batch_idx, row_idx, pk, seq))

    return {
        "group_counts": dict(group_counts),
        "quintet_rows": quintet_rows,
        "quartet_rows": quartet_rows,
    }


def phase0_scan_and_index(
    input_dir: Path, output_dir: Path, num_workers: int
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Phase 0: Scan all files, classify rows, extract quintet/quartet indices."""
    phase_dir = output_dir / "phase0_index"
    phase_dir.mkdir(parents=True, exist_ok=True)

    gc_path = phase_dir / "group_counts.json"
    qi_path = phase_dir / "quintet_index.parquet"
    qr_path = phase_dir / "quartet_index.parquet"
    source_marker_path = phase_dir / "source_dir.txt"

    if gc_path.exists() and qi_path.exists() and qr_path.exists():
        # Invalidate checkpoint if source directory changed (e.g. resolved vs original)
        prev_source = source_marker_path.read_text().strip() if source_marker_path.exists() else None
        if prev_source is not None and prev_source != str(input_dir):
            print(f"  Phase 0: Source directory changed ({prev_source} → {input_dir}), re-indexing...")
        else:
            print("  Phase 0: Loading from checkpoint...")
            with open(gc_path) as f:
                group_counts = json.load(f)
            quintet_df = pd.read_parquet(qi_path)
            quartet_df = pd.read_parquet(qr_path)
            return quintet_df, quartet_df, group_counts

    print("  Phase 0: Scanning and indexing files...")
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    # Build (filepath, batch_idx) args
    file_args = [(str(f), _batch_idx_from_path(f)) for f in files]

    merged_counts: Dict[str, int] = defaultdict(int)
    all_quintet_rows = []
    all_quartet_rows = []

    with multiprocessing.Pool(num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_index_file, file_args),
            total=len(files),
            desc="    Indexing",
        ):
            for pk, cnt in result["group_counts"].items():
                merged_counts[pk] += cnt
            all_quintet_rows.extend(result["quintet_rows"])
            all_quartet_rows.extend(result["quartet_rows"])

    # Build DataFrames
    quintet_df = pd.DataFrame(
        all_quintet_rows,
        columns=["batch_idx", "row_idx", "permutation_key", "sequence"],
    )
    quartet_df = pd.DataFrame(
        all_quartet_rows,
        columns=["batch_idx", "row_idx", "permutation_key", "sequence"],
    )

    # Use compact dtypes
    quintet_df["batch_idx"] = quintet_df["batch_idx"].astype(np.uint16)
    quintet_df["row_idx"] = quintet_df["row_idx"].astype(np.uint32)
    quartet_df["batch_idx"] = quartet_df["batch_idx"].astype(np.uint16)
    quartet_df["row_idx"] = quartet_df["row_idx"].astype(np.uint32)

    # Compute per-cardinality totals
    card_cache: Dict[str, int] = {}
    cardinality_totals: Dict[str, int] = defaultdict(int)
    for pk, cnt in merged_counts.items():
        card = _get_cardinality(pk, card_cache)
        label = CARDINALITY_LABELS.get(card, f"card_{card}")
        cardinality_totals[label] += cnt

    total_rows = sum(merged_counts.values())

    output_counts = {
        "total_rows": total_rows,
        "per_permutation_key": dict(merged_counts),
        "per_cardinality": dict(cardinality_totals),
    }

    # Save checkpoints
    with open(gc_path, "w") as f:
        json.dump(output_counts, f, indent=2)
    quintet_df.to_parquet(qi_path, index=False)
    quartet_df.to_parquet(qr_path, index=False)
    source_marker_path.write_text(str(input_dir))

    print(f"    Total rows: {total_rows:,}")
    print(f"    Quintets: {len(quintet_df):,}, Quartets: {len(quartet_df):,}")
    for label, cnt in sorted(cardinality_totals.items()):
        print(f"    {label}: {cnt:,} ({cnt/total_rows*100:.2f}%)")

    return quintet_df, quartet_df, output_counts


# ---------------------------------------------------------------------------
# Phase 1: Cluster
# ---------------------------------------------------------------------------


def phase1_cluster(
    quintet_df: pd.DataFrame,
    quartet_df: pd.DataFrame,
    output_dir: Path,
    trb_similarity: float,
    peptide_similarity: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Phase 1: Cluster TRB and peptide sequences from quintets/quartets."""
    phase_dir = output_dir / "phase1_clusters"
    phase_dir.mkdir(parents=True, exist_ok=True)

    qc_path = phase_dir / "quintet_clustered.parquet"
    qrc_path = phase_dir / "quartet_clustered.parquet"

    if qc_path.exists() and qrc_path.exists():
        print("  Phase 1: Loading from checkpoint...")
        quintet_df = pd.read_parquet(qc_path)
        quartet_df = pd.read_parquet(qrc_path)
        return quintet_df, quartet_df

    print("  Phase 1: Clustering TRB and peptide sequences...")
    field_cache: Dict[str, List[str]] = {}

    # Extract TRB and peptide sequences from both quintets and quartets
    def extract_field(df: pd.DataFrame, field: str) -> List[Optional[str]]:
        perm_keys = df["permutation_key"].tolist()
        sequences = df["sequence"].tolist()
        results = []
        for pk, seq in zip(perm_keys, sequences):
            if pk not in field_cache:
                field_cache[pk] = parse_permutation_key(pk)
            fields = field_cache[pk]
            if field not in fields:
                results.append(None)
            else:
                idx = fields.index(field)
                parts = seq.split(" ")
                results.append(parts[idx] if idx < len(parts) else None)
        return results

    print("    Extracting TRB sequences...")
    quintet_trb = extract_field(quintet_df, "trb")
    quartet_trb = extract_field(quartet_df, "trb")

    print("    Extracting peptide sequences...")
    quintet_pep = extract_field(quintet_df, "peptide")
    quartet_pep = extract_field(quartet_df, "peptide")

    # Collect unique sequences
    all_trb = set()
    for s in quintet_trb + quartet_trb:
        if s:
            all_trb.add(s)

    all_peptides = set()
    for s in quintet_pep + quartet_pep:
        if s:
            all_peptides.add(s)

    print(f"    Unique TRB sequences: {len(all_trb):,}")
    print(f"    Unique peptide sequences: {len(all_peptides):,}")

    # Cluster TRB
    trb_to_cluster, trb_cluster_to_seqs = cluster_sequences_mmseqs2(
        list(all_trb),
        similarity_threshold=trb_similarity,
        coverage=0.8,
        seq_type="trb",
    )

    # Cluster peptides
    if len(all_peptides) > 0:
        pep_to_cluster, pep_cluster_to_seqs = cluster_peptides_mmseqs2(
            list(all_peptides),
            similarity_threshold=peptide_similarity,
            coverage=0.8,
        )
    else:
        pep_to_cluster, pep_cluster_to_seqs = {}, {}

    # Add cluster columns to DataFrames
    quintet_df = quintet_df.copy()
    quintet_df["trb_cluster"] = [trb_to_cluster.get(s, -1) if s else -1 for s in quintet_trb]
    quintet_df["peptide_cluster"] = [pep_to_cluster.get(s, -1) if s else -1 for s in quintet_pep]

    quartet_df = quartet_df.copy()
    quartet_df["trb_cluster"] = [trb_to_cluster.get(s, -1) if s else -1 for s in quartet_trb]
    quartet_df["peptide_cluster"] = [pep_to_cluster.get(s, -1) if s else -1 for s in quartet_pep]

    # Save cluster mappings as TSV
    trb_tsv = phase_dir / "trb_clusters.tsv"
    with open(trb_tsv, "w") as f:
        f.write("sequence\tcluster_id\tcluster_size\tis_representative\n")
        for cid, seqs in sorted(trb_cluster_to_seqs.items()):
            for i, seq in enumerate(seqs):
                f.write(f"{seq}\t{cid}\t{len(seqs)}\t{'true' if i == 0 else 'false'}\n")

    pep_tsv = phase_dir / "peptide_clusters.tsv"
    with open(pep_tsv, "w") as f:
        f.write("sequence\tcluster_id\tcluster_size\tis_representative\n")
        for cid, seqs in sorted(pep_cluster_to_seqs.items()):
            for i, seq in enumerate(seqs):
                f.write(f"{seq}\t{cid}\t{len(seqs)}\t{'true' if i == 0 else 'false'}\n")

    # Save clustered DataFrames
    quintet_df.to_parquet(qc_path, index=False)
    quartet_df.to_parquet(qrc_path, index=False)

    print(f"    TRB clusters: {len(trb_cluster_to_seqs):,}")
    print(f"    Peptide clusters: {len(pep_cluster_to_seqs):,}")

    return quintet_df, quartet_df


# ---------------------------------------------------------------------------
# Phase 2: Assign Splits
# ---------------------------------------------------------------------------


def _assign_cluster_splits(
    df: pd.DataFrame,
    seed: int,
    has_peptide: bool,
    test_frac: float = 0.15,
    val_frac: float = 0.15,
) -> pd.DataFrame:
    """
    Assign splits to rows based on TRB cluster grouping.

    For rows with peptide: group TRB clusters by dominant peptide_cluster,
    then assign entire TRB clusters within each peptide group.
    For rows without peptide: assign TRB clusters randomly.
    """
    rng = np.random.RandomState(seed)
    df = df.copy()
    df["split"] = ""

    if has_peptide:
        # Group TRB clusters by their dominant peptide cluster
        trb_dominant_pep = (
            df.groupby("trb_cluster")["peptide_cluster"]
            .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else -1)
        )

        # For each peptide cluster group, assign TRB clusters to splits
        all_trb_clusters = df["trb_cluster"].unique()
        cluster_split = {}

        for pep_cluster in df["peptide_cluster"].unique():
            # TRB clusters whose dominant peptide is this one
            trb_in_group = [
                tc for tc in all_trb_clusters
                if trb_dominant_pep.get(tc, -1) == pep_cluster
            ]
            rng.shuffle(trb_in_group)

            n_test = max(1, int(len(trb_in_group) * test_frac)) if len(trb_in_group) > 2 else 0
            n_val = max(1, int(len(trb_in_group) * val_frac)) if len(trb_in_group) > 2 else 0

            for tc in trb_in_group[:n_test]:
                cluster_split[tc] = "test"
            for tc in trb_in_group[n_test:n_test + n_val]:
                cluster_split[tc] = "val"
            for tc in trb_in_group[n_test + n_val:]:
                cluster_split[tc] = "train"

        # Any unassigned clusters (shouldn't happen, but safety)
        for tc in all_trb_clusters:
            if tc not in cluster_split:
                cluster_split[tc] = "train"

        df["split"] = df["trb_cluster"].map(cluster_split)
    else:
        # No peptide: split by TRB cluster alone
        all_trb_clusters = list(df["trb_cluster"].unique())
        rng.shuffle(all_trb_clusters)

        n_test = max(1, int(len(all_trb_clusters) * test_frac))
        n_val = max(1, int(len(all_trb_clusters) * val_frac))

        cluster_split = {}
        for tc in all_trb_clusters[:n_test]:
            cluster_split[tc] = "test"
        for tc in all_trb_clusters[n_test:n_test + n_val]:
            cluster_split[tc] = "val"
        for tc in all_trb_clusters[n_test + n_val:]:
            cluster_split[tc] = "train"

        df["split"] = df["trb_cluster"].map(cluster_split)

    return df


def phase2_assign_splits(
    quintet_df: pd.DataFrame,
    quartet_df: pd.DataFrame,
    group_counts: Dict[str, Any],
    output_dir: Path,
    seed: int,
) -> pd.DataFrame:
    """Phase 2: Assign every row to train/val/test."""
    phase_dir = output_dir / "phase2_splits"
    phase_dir.mkdir(parents=True, exist_ok=True)

    sa_path = phase_dir / "split_assignments.parquet"
    ss_path = phase_dir / "split_summary.json"

    if sa_path.exists() and ss_path.exists():
        print("  Phase 2: Loading from checkpoint...")
        split_assignments = pd.read_parquet(sa_path)
        return split_assignments

    print("  Phase 2: Assigning splits...")
    field_cache: Dict[str, int] = {}

    # --- Quintet splitting (cluster-based, epitope-stratified) ---
    print("    Splitting quintets (cluster-based, epitope-stratified)...")
    quintet_split = _assign_cluster_splits(quintet_df, seed, has_peptide=True)

    # --- Quartet splitting ---
    print("    Splitting quartets...")
    # Determine which quartets have peptide
    quartet_has_peptide = quartet_df["peptide_cluster"].apply(lambda x: x >= 0)
    quartet_with_pep = quartet_df[quartet_has_peptide].copy()
    quartet_without_pep = quartet_df[~quartet_has_peptide].copy()

    if len(quartet_with_pep) > 0:
        quartet_with_pep = _assign_cluster_splits(quartet_with_pep, seed + 1, has_peptide=True)
    else:
        quartet_with_pep["split"] = pd.Series(dtype=str)

    if len(quartet_without_pep) > 0:
        quartet_without_pep = _assign_cluster_splits(quartet_without_pep, seed + 2, has_peptide=False)
    else:
        quartet_without_pep["split"] = pd.Series(dtype=str)

    quartet_split = pd.concat([quartet_with_pep, quartet_without_pep], ignore_index=True)

    # --- Collect non-train assignments ---
    # For quintets/quartets: all rows get explicit entries
    assignments = []

    for _, row in quintet_split.iterrows():
        assignments.append({
            "batch_idx": row["batch_idx"],
            "row_idx": row["row_idx"],
            "split": row["split"],
        })

    for _, row in quartet_split.iterrows():
        assignments.append({
            "batch_idx": row["batch_idx"],
            "row_idx": row["row_idx"],
            "split": row["split"],
        })

    split_df = pd.DataFrame(assignments)
    split_df["batch_idx"] = split_df["batch_idx"].astype(np.uint16)
    split_df["row_idx"] = split_df["row_idx"].astype(np.uint32)

    # For triplets/pairs/singles, we DON'T store explicit entries for every row.
    # We only store test/val assignments (hash-based, 0.05%/0.05%/99.9%).
    # But we don't need to iterate all 735M rows here — Phase 3 will compute on-the-fly.
    # We store the seed and the rule in the summary.

    # Compute summary stats
    quin_counts = quintet_split["split"].value_counts().to_dict()
    quar_counts = quartet_split["split"].value_counts().to_dict()

    total_rows = group_counts.get("total_rows", 0)
    quintet_total = len(quintet_split)
    quartet_total = len(quartet_split)
    lower_total = total_rows - quintet_total - quartet_total

    # For lower cardinality: 0.05% test, 0.05% val, 99.9% train
    lower_test = int(lower_total * 0.0005)
    lower_val = int(lower_total * 0.0005)
    lower_train = lower_total - lower_test - lower_val

    summary = {
        "seed": seed,
        "total_rows": total_rows,
        "hash_rule": "md5(f'{seed}_{batch_idx}_{row_idx}') % 10000: <5=test, <10=val, else=train",
        "quintet_splits": quin_counts,
        "quartet_splits": quar_counts,
        "lower_cardinality": {
            "total": lower_total,
            "test_estimated": lower_test,
            "val_estimated": lower_val,
            "train_estimated": lower_train,
        },
        "overall": {
            "test": quin_counts.get("test", 0) + quar_counts.get("test", 0) + lower_test,
            "val": quin_counts.get("val", 0) + quar_counts.get("val", 0) + lower_val,
            "train": quin_counts.get("train", 0) + quar_counts.get("train", 0) + lower_train,
        },
    }

    # --- Verification ---
    print("    Verifying split integrity...")
    errors = []

    # No TRB cluster spans multiple splits (quintets)
    for name, sdf in [("quintet", quintet_split), ("quartet", quartet_split)]:
        if len(sdf) == 0:
            continue
        cluster_splits = sdf.groupby("trb_cluster")["split"].nunique()
        multi = cluster_splits[cluster_splits > 1]
        if len(multi) > 0:
            errors.append(f"{name}: {len(multi)} TRB clusters span multiple splits")

    # Test and val each contain >=10 distinct peptide clusters (quintets)
    for split_name in ["test", "val"]:
        quin_in_split = quintet_split[quintet_split["split"] == split_name]
        if len(quin_in_split) > 0:
            n_pep_clusters = quin_in_split["peptide_cluster"].nunique()
            if n_pep_clusters < 10:
                errors.append(
                    f"quintet {split_name} has only {n_pep_clusters} peptide clusters (want >=10)"
                )

    # No peptide cluster entirely absent from train
    if len(quintet_split) > 0:
        all_pep_clusters = set(quintet_split["peptide_cluster"].unique())
        train_pep_clusters = set(
            quintet_split[quintet_split["split"] == "train"]["peptide_cluster"].unique()
        )
        missing = all_pep_clusters - train_pep_clusters - {-1}
        if missing:
            errors.append(f"{len(missing)} peptide clusters absent from quintet train")

    if errors:
        for e in errors:
            print(f"    WARNING: {e}")
    else:
        print("    All split integrity checks passed")

    summary["verification_errors"] = errors

    # Save
    split_df.to_parquet(sa_path, index=False)
    with open(ss_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"    Split assignments: {len(split_df):,} explicit rows")
    print(f"    Quintet splits: {quin_counts}")
    print(f"    Quartet splits: {quar_counts}")

    return split_df


# ---------------------------------------------------------------------------
# Phase 3: Build Nested Training Subsets
# ---------------------------------------------------------------------------


def _get_row_split(
    batch_idx: int,
    row_idx: int,
    seed: int,
    explicit_lookup: Dict[Tuple[int, int], str],
) -> str:
    """Determine split for a row, checking explicit assignments first."""
    key = (int(batch_idx), int(row_idx))
    if key in explicit_lookup:
        return explicit_lookup[key]
    # Hash-based for lower cardinality
    h = _hash_row(seed, batch_idx, row_idx)
    if h < 5:
        return "test"
    elif h < 10:
        return "val"
    return "train"


def _scan_file_for_subsets(args: Tuple) -> Dict[str, Any]:
    """Worker: scan one file and collect train row info for subset assignment."""
    filepath, batch_idx, seed, explicit_lookup_items, proportional_thresholds = args

    # Rebuild lookup dict in worker
    explicit_lookup = dict(explicit_lookup_items)

    table = pq.read_table(filepath, columns=["permutation_key"])
    perm_keys = table.column("permutation_key").to_pylist()
    n_rows = len(perm_keys)

    field_cache: Dict[str, int] = {}
    train_count = 0
    # Per-cardinality train counts
    card_counts: Dict[int, int] = defaultdict(int)

    # For proportional subsets: collect (batch_idx, row_idx) for rows below each threshold
    subset_rows: Dict[str, List[Tuple[int, int]]] = {name: [] for name in proportional_thresholds}

    # For stratified: collect (batch_idx, row_idx, cardinality) for train rows
    stratified_by_card: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    for row_idx in range(n_rows):
        split = _get_row_split(batch_idx, row_idx, seed, explicit_lookup)
        if split != "train":
            continue

        train_count += 1
        pk = perm_keys[row_idx]
        card = _get_cardinality(pk, field_cache)
        card_counts[card] += 1

        score = _subset_score(seed, batch_idx, row_idx)

        for name, threshold in proportional_thresholds.items():
            if score < threshold:
                subset_rows[name].append((batch_idx, row_idx))

        stratified_by_card[card].append((batch_idx, row_idx))

    return {
        "train_count": train_count,
        "card_counts": dict(card_counts),
        "subset_rows": {k: v for k, v in subset_rows.items()},
        "stratified_by_card": dict(stratified_by_card),
    }


def phase3_build_subsets(
    split_assignments: pd.DataFrame,
    group_counts: Dict[str, Any],
    input_dir: Path,
    output_dir: Path,
    seed: int,
    num_workers: int,
) -> Dict[str, Any]:
    """Phase 3: Create proportional, stratified, balanced, and ablation subsets."""
    phase_dir = output_dir / "phase3_subsets"
    phase_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = phase_dir / "subset_manifest.json"
    if manifest_path.exists():
        print("  Phase 3: Loading from checkpoint...")
        with open(manifest_path) as f:
            return json.load(f)

    print("  Phase 3: Building nested training subsets...")

    # Build explicit lookup from split_assignments
    explicit_lookup = {}
    for _, row in split_assignments.iterrows():
        explicit_lookup[(int(row["batch_idx"]), int(row["row_idx"]))] = row["split"]

    total_rows = group_counts.get("total_rows", 0)

    # Estimate train size
    n_explicit_test = len(split_assignments[split_assignments["split"] == "test"])
    n_explicit_val = len(split_assignments[split_assignments["split"] == "val"])
    n_explicit_train = len(split_assignments[split_assignments["split"] == "train"])
    n_lower = total_rows - len(split_assignments)
    n_lower_test = int(n_lower * 0.0005)
    n_lower_val = int(n_lower * 0.0005)
    n_lower_train = n_lower - n_lower_test - n_lower_val
    estimated_train = n_explicit_train + n_lower_train

    print(f"    Estimated train size: {estimated_train:,}")

    # Compute proportional thresholds
    proportional_thresholds = {}
    for size in PROPORTIONAL_SIZES:
        if size < estimated_train:
            proportional_thresholds[f"proportional_{size // 1_000_000}M"] = size / estimated_train

    print(f"    Proportional subsets: {list(proportional_thresholds.keys())}")

    # Prepare worker args
    files = sorted(input_dir.glob("*.parquet"))
    # Serialize explicit_lookup as list of tuples for pickling
    explicit_items = list(explicit_lookup.items())

    file_args = [
        (str(f), _batch_idx_from_path(f), seed, explicit_items, proportional_thresholds)
        for f in files
    ]

    # Stream files and collect subset rows
    all_subset_rows: Dict[str, List[Tuple[int, int]]] = {
        name: [] for name in proportional_thresholds
    }
    all_stratified: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    actual_train = 0
    actual_card_counts: Dict[int, int] = defaultdict(int)

    print("    Scanning files for subset assignment...")
    with multiprocessing.Pool(num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_scan_file_for_subsets, file_args),
            total=len(files),
            desc="    Subsets",
        ):
            actual_train += result["train_count"]
            for card, cnt in result["card_counts"].items():
                actual_card_counts[card] += cnt
            for name, rows in result["subset_rows"].items():
                all_subset_rows[name].extend(rows)
            for card, rows in result["stratified_by_card"].items():
                all_stratified[card].extend(rows)

    print(f"    Actual train size: {actual_train:,}")
    print(f"    Card distribution: {dict(actual_card_counts)}")

    manifest = {
        "actual_train_size": actual_train,
        "cardinality_distribution": {str(k): v for k, v in sorted(actual_card_counts.items())},
        "subsets": {},
    }

    # --- Save proportional subsets ---
    print("    Saving proportional subsets...")
    prev_name = None
    for name in sorted(all_subset_rows.keys(), key=lambda n: len(all_subset_rows[n])):
        rows = all_subset_rows[name]
        arr = np.array(rows, dtype=np.uint32).reshape(-1, 2) if rows else np.empty((0, 2), dtype=np.uint32)
        npy_path = phase_dir / f"{name}.npy"
        np.save(npy_path, arr)
        manifest["subsets"][name] = {
            "size": len(rows),
            "type": "proportional",
            "nested": True,
        }
        print(f"      {name}: {len(rows):,} rows")
        prev_name = name

    # --- Stratified enriched subset ---
    print("    Building stratified enriched subset...")
    rng = np.random.RandomState(seed + 100)
    stratified_rows = []

    for card in sorted(actual_card_counts.keys(), reverse=True):
        available = all_stratified.get(card, [])
        frac = STRATIFIED_FRACTIONS.get(card, 0)
        target = int(STRATIFIED_TOTAL * frac)
        if target == 0:
            continue

        if len(available) <= target:
            # Use all available (no oversampling for stratified)
            stratified_rows.extend(available)
        else:
            indices = rng.choice(len(available), size=target, replace=False)
            stratified_rows.extend([available[i] for i in indices])

    strat_arr = np.array(stratified_rows, dtype=np.uint32).reshape(-1, 2) if stratified_rows else np.empty((0, 2), dtype=np.uint32)
    strat_path = phase_dir / "stratified_4M.npy"
    np.save(strat_path, strat_arr)
    manifest["subsets"]["stratified_4M"] = {
        "size": len(stratified_rows),
        "type": "stratified_enriched",
        "nested": False,
    }
    print(f"      stratified_4M: {len(stratified_rows):,} rows")

    # --- Balanced subsets (with oversampling) ---
    print("    Building balanced subsets...")
    for bal_name, (total_size, per_group) in BALANCED_SIZES.items():
        balanced_rows = []
        for card in sorted(actual_card_counts.keys()):
            available = all_stratified.get(card, [])
            if len(available) == 0:
                continue
            if len(available) >= per_group:
                indices = rng.choice(len(available), size=per_group, replace=False)
            else:
                # Oversample
                indices = rng.choice(len(available), size=per_group, replace=True)
            balanced_rows.extend([available[i] for i in indices])

        bal_arr = np.array(balanced_rows, dtype=np.uint32).reshape(-1, 2) if balanced_rows else np.empty((0, 2), dtype=np.uint32)
        bal_path = phase_dir / f"{bal_name}.npy"
        np.save(bal_path, bal_arr)
        manifest["subsets"][bal_name] = {
            "size": len(balanced_rows),
            "type": "balanced",
            "nested": False,
            "per_group_target": per_group,
        }
        print(f"      {bal_name}: {len(balanced_rows):,} rows")

    # --- Oversampling ablation variants ---
    print("    Building oversampling ablation variants...")
    for var_name, (quin_factor, quar_factor, total_target) in OVERSAMPLING_VARIANTS.items():
        variant_rows = []

        # Quintets oversampled
        quin_available = all_stratified.get(5, [])
        for _ in range(quin_factor):
            variant_rows.extend(quin_available)

        # Quartets oversampled
        quar_available = all_stratified.get(4, [])
        for _ in range(quar_factor):
            variant_rows.extend(quar_available)

        # Fill remaining with proportional lower-cardinality rows
        remaining = total_target - len(variant_rows)
        if remaining > 0:
            lower_rows = []
            for card in [3, 2, 1]:
                lower_rows.extend(all_stratified.get(card, []))
            if len(lower_rows) > 0:
                if len(lower_rows) >= remaining:
                    indices = rng.choice(len(lower_rows), size=remaining, replace=False)
                else:
                    indices = rng.choice(len(lower_rows), size=remaining, replace=True)
                variant_rows.extend([lower_rows[i] for i in indices])

        var_arr = np.array(variant_rows, dtype=np.uint32).reshape(-1, 2) if variant_rows else np.empty((0, 2), dtype=np.uint32)
        var_path = phase_dir / f"{var_name}.npy"
        np.save(var_path, var_arr)
        manifest["subsets"][var_name] = {
            "size": len(variant_rows),
            "type": "oversampling_ablation",
            "nested": False,
            "quintet_factor": quin_factor,
            "quartet_factor": quar_factor,
        }
        print(f"      {var_name}: {len(variant_rows):,} rows")

    # Save manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


# ---------------------------------------------------------------------------
# Phase 4: Binding Probe Eval Sets
# ---------------------------------------------------------------------------


def _load_standardized_parquets(data_dir: Path) -> pd.DataFrame:
    """Load all parquet files from a standardized data directory."""
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {data_dir}")
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


class _AhoCorasickPure:
    """Pure-Python Aho-Corasick automaton for multi-pattern substring search.

    Builds a trie with failure links for O(N + M + Z) searching where N is text
    length, M is total pattern length, and Z is number of matches.
    """

    def __init__(self):
        self._goto = [{}]       # state -> {char -> state}
        self._fail = [0]        # state -> failure state
        self._output = [[]]     # state -> list of pattern strings
        self._built = False

    def add_pattern(self, pattern: str) -> None:
        state = 0
        for ch in pattern:
            if ch not in self._goto[state]:
                new_state = len(self._goto)
                self._goto.append({})
                self._fail.append(0)
                self._output.append([])
                self._goto[state][ch] = new_state
            state = self._goto[state][ch]
        self._output[state].append(pattern)

    def build(self) -> None:
        from collections import deque
        queue = deque()
        # Initialize depth-1 states
        for ch, s in self._goto[0].items():
            self._fail[s] = 0
            queue.append(s)
        # BFS to build failure links
        while queue:
            r = queue.popleft()
            for ch, s in self._goto[r].items():
                queue.append(s)
                state = self._fail[r]
                while state != 0 and ch not in self._goto[state]:
                    state = self._fail[state]
                self._fail[s] = self._goto[state].get(ch, 0)
                if self._fail[s] == s:
                    self._fail[s] = 0
                self._output[s] = self._output[s] + self._output[self._fail[s]]
        self._built = True

    def search(self, text: str) -> Set[str]:
        """Return the set of patterns found anywhere in *text*."""
        if not self._built:
            raise RuntimeError("call build() before search()")
        found: Set[str] = set()
        state = 0
        goto = self._goto
        fail = self._fail
        output = self._output
        for ch in text:
            while state != 0 and ch not in goto[state]:
                state = fail[state]
            state = goto[state].get(ch, 0)
            if output[state]:
                found.update(output[state])
        return found


_P4_EXPLICIT_BY_BATCH: Dict[int, Dict[int, str]] = {}
_P4_AUTOMATON: Any = None
_P4_SEARCH_MODE: str = "none"  # "ahocorasick_c" or "ahocorasick_py"
_P4_SEED: int = 0


def _init_leakage_worker(explicit_by_batch, cdr3_patterns, seed):
    """Initializer for leakage-check worker processes.

    Builds the Aho-Corasick automaton and explicit lookup dict once per worker
    instead of once per task, avoiding redundant IPC.
    """
    global _P4_EXPLICIT_BY_BATCH, _P4_AUTOMATON, _P4_SEARCH_MODE, _P4_SEED
    _P4_EXPLICIT_BY_BATCH = explicit_by_batch
    _P4_SEED = seed
    try:
        import ahocorasick
        A = ahocorasick.Automaton()
        for pattern in cdr3_patterns:
            A.add_word(pattern, pattern)
        A.make_automaton()
        _P4_AUTOMATON = A
        _P4_SEARCH_MODE = "ahocorasick_c"
    except ImportError:
        A = _AhoCorasickPure()
        for pattern in cdr3_patterns:
            A.add_pattern(pattern)
        A.build()
        _P4_AUTOMATON = A
        _P4_SEARCH_MODE = "ahocorasick_py"


def _check_leakage_file(args: Tuple) -> Dict[str, Any]:
    """Worker: check one dedup file for CDR3 substring matches.

    Returns a compact result dict instead of a full match list:
      - matched_cdr3s: Set[str] of unique CDR3s found
      - total_match_count: int of (row, chain, cdr3) triples
      - sample_matches: List[Dict] of up to 100 example matches
    """
    filepath, batch_idx = args

    table = pq.read_table(filepath, columns=["permutation_key", "sequence"])

    # Classify which permutation keys contain TRA/TRB
    perm_col = table.column("permutation_key")
    unique_pks = perm_col.unique().to_pylist()
    field_cache: Dict[str, List[str]] = {}
    tcr_pks = set()
    for pk in unique_pks:
        fields = parse_permutation_key(pk)
        field_cache[pk] = fields
        if "tra" in fields or "trb" in fields:
            tcr_pks.add(pk)

    if not tcr_pks:
        return {"matched_cdr3s": set(), "total_match_count": 0, "sample_matches": []}

    # Pre-filter to TCR rows only using PyArrow
    tcr_filter = pc.is_in(perm_col, value_set=pa.array(list(tcr_pks)))
    tcr_table = table.filter(tcr_filter)
    # Map filtered indices back to original row_idx via boolean mask
    mask_list = tcr_filter.to_pylist()
    orig_row_indices = [i for i, m in enumerate(mask_list) if m]

    tcr_perm_keys = tcr_table.column("permutation_key").to_pylist()
    tcr_sequences = tcr_table.column("sequence").to_pylist()

    batch_explicit = _P4_EXPLICIT_BY_BATCH.get(batch_idx, {})
    use_c = _P4_SEARCH_MODE == "ahocorasick_c"

    matched_cdr3s: Set[str] = set()
    total_match_count = 0
    sample_matches: List[Dict[str, Any]] = []

    for i, (pk, seq) in enumerate(zip(tcr_perm_keys, tcr_sequences)):
        row_idx = orig_row_indices[i]

        # Determine split: O(1) batch lookup, then hash fallback
        if row_idx in batch_explicit:
            split = batch_explicit[row_idx]
        else:
            h = _hash_row(_P4_SEED, batch_idx, row_idx)
            if h < 5:
                split = "test"
            elif h < 10:
                split = "val"
            else:
                split = "train"

        if split != "train":
            continue

        fields = field_cache[pk]
        parts = seq.split(" ")

        for field_name in ["trb", "tra"]:
            if field_name in fields:
                idx = fields.index(field_name)
                if idx < len(parts):
                    full_seq = parts[idx]
                    if use_c:
                        # C ahocorasick iter() can return duplicates at
                        # different positions — deduplicate per (row, chain)
                        seen: Set[str] = set()
                        for _, matched_cdr3 in _P4_AUTOMATON.iter(full_seq):
                            if matched_cdr3 not in seen:
                                seen.add(matched_cdr3)
                                matched_cdr3s.add(matched_cdr3)
                                total_match_count += 1
                                if len(sample_matches) < 100:
                                    sample_matches.append({
                                        "batch_idx": batch_idx,
                                        "row_idx": row_idx,
                                        "chain": field_name,
                                        "matched_cdr3": matched_cdr3,
                                        "full_sequence": full_seq,
                                    })
                    else:
                        # Pure-Python AC returns a set directly
                        found = _P4_AUTOMATON.search(full_seq)
                        if found:
                            matched_cdr3s.update(found)
                            total_match_count += len(found)
                            if len(sample_matches) < 100:
                                for cdr3 in found:
                                    if len(sample_matches) >= 100:
                                        break
                                    sample_matches.append({
                                        "batch_idx": batch_idx,
                                        "row_idx": row_idx,
                                        "chain": field_name,
                                        "matched_cdr3": cdr3,
                                        "full_sequence": full_seq,
                                    })

    return {
        "matched_cdr3s": matched_cdr3s,
        "total_match_count": total_match_count,
        "sample_matches": sample_matches,
    }


def phase4_binding_probes(
    input_dir: Path,
    output_dir: Path,
    split_assignments: pd.DataFrame,
    seed: int,
    num_workers: int,
    batman_dir: Optional[Path],
    trait_dir: Optional[Path],
    trait_neg_sample: int = 18000,
) -> Dict[str, Any]:
    """Phase 4: Create binding probe eval sets from BATMAN + TRAIT."""
    phase_dir = output_dir / "phase4_binding_probes"
    phase_dir.mkdir(parents=True, exist_ok=True)

    stats_path = phase_dir / "binding_probe_stats.json"
    if stats_path.exists():
        print("  Phase 4: Loading from checkpoint...")
        with open(stats_path) as f:
            return json.load(f)

    if batman_dir is None or trait_dir is None:
        print("  Phase 4: Skipping (--batman_dir or --trait_dir not provided)")
        stats = {"skipped": True, "reason": "batman_dir or trait_dir not provided"}
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        return stats

    print("  Phase 4: Building binding probe eval sets...")

    # --- Step 4a: Load standardized data ---
    print("    Step 4a: Loading BATMAN and TRAIT data...")
    batman_df = _load_standardized_parquets(batman_dir)
    trait_df = _load_standardized_parquets(trait_dir)

    print(f"    BATMAN: {len(batman_df):,} rows")
    print(f"    TRAIT: {len(trait_df):,} rows")

    # Filter to rows with non-empty trb AND peptide
    keep_cols = ["tra", "trb", "peptide", "mhc_one", "mhc_two", "binding", "score", "source"]
    for col in keep_cols:
        if col not in batman_df.columns:
            batman_df[col] = None
        if col not in trait_df.columns:
            trait_df[col] = None

    batman_df = batman_df[keep_cols].copy()
    trait_df = trait_df[keep_cols].copy()

    # Filter: need trb AND peptide
    batman_df = batman_df[
        batman_df["trb"].notna() & (batman_df["trb"] != "") &
        batman_df["peptide"].notna() & (batman_df["peptide"] != "")
    ].reset_index(drop=True)

    trait_df = trait_df[
        trait_df["trb"].notna() & (trait_df["trb"] != "") &
        trait_df["peptide"].notna() & (trait_df["peptide"] != "")
    ].reset_index(drop=True)

    print(f"    BATMAN (filtered): {len(batman_df):,} rows")
    print(f"    TRAIT (filtered): {len(trait_df):,} rows")

    # --- Step 4b: Leakage check ---
    print("    Step 4b: Checking for CDR3 leakage against dedup training data...")

    # Collect unique CDR3s from batman + trait
    all_cdr3s = set()
    for col in ["trb", "tra"]:
        for df in [batman_df, trait_df]:
            vals = df[col].dropna().unique()
            all_cdr3s.update(v for v in vals if v and len(v) >= 4)

    print(f"    Unique CDR3 patterns to check: {len(all_cdr3s):,}")

    # Build explicit lookup indexed by batch for O(1) per-batch access
    explicit_by_batch: Dict[int, Dict[int, str]] = {}
    for _, row in split_assignments.iterrows():
        bi, ri = int(row["batch_idx"]), int(row["row_idx"])
        explicit_by_batch.setdefault(bi, {})[ri] = row["split"]

    cdr3_list = list(all_cdr3s)

    # Scan dedup files for leakage
    files = sorted(input_dir.glob("*.parquet"))
    file_args = [(str(f), _batch_idx_from_path(f)) for f in files]

    all_matched_cdr3s: Set[str] = set()
    total_match_count = 0
    all_sample_matches: List[Dict[str, Any]] = []
    with multiprocessing.Pool(
        num_workers,
        initializer=_init_leakage_worker,
        initargs=(explicit_by_batch, cdr3_list, seed),
    ) as pool:
        for result in tqdm(
            pool.imap_unordered(_check_leakage_file, file_args),
            total=len(files),
            desc="    Leakage check",
        ):
            all_matched_cdr3s.update(result["matched_cdr3s"])
            total_match_count += result["total_match_count"]
            if len(all_sample_matches) < 100:
                remaining = 100 - len(all_sample_matches)
                all_sample_matches.extend(result["sample_matches"][:remaining])

    leaked_cdr3s = all_matched_cdr3s
    print(f"    Leakage matches: {total_match_count:,} total, {len(leaked_cdr3s):,} unique CDR3s")

    leakage_report = {
        "total_matches": total_match_count,
        "unique_leaked_cdr3s": len(leaked_cdr3s),
        "leaked_cdr3_list": sorted(leaked_cdr3s),
        "sample_matches": all_sample_matches[:100],
    }
    with open(phase_dir / "leakage_report.json", "w") as f:
        json.dump(leakage_report, f, indent=2)

    # --- Step 4c: Build binding probe parquets ---
    print("    Step 4c: Building binding probe parquets...")

    # Classify batman rows
    batman_pos = batman_df[batman_df["binding"] == "pos"].copy()
    batman_neg = batman_df[batman_df["binding"] == "neg"].copy()

    batman_pos["label"] = "pos"
    batman_neg["label"] = "neg_true"

    print(f"    BATMAN positives: {len(batman_pos):,}")
    print(f"    BATMAN negatives: {len(batman_neg):,}")

    # Sample TRAIT negatives (soft negatives)
    trait_neg = trait_df[trait_df["binding"] == "neg"].copy()
    rng = np.random.RandomState(seed + 200)

    if len(trait_neg) > trait_neg_sample:
        # Stratified by peptide
        peptide_counts = trait_neg["peptide"].value_counts()
        n_peptides = len(peptide_counts)
        per_peptide = max(1, trait_neg_sample // n_peptides)

        sampled_indices = []
        for pep, count in peptide_counts.items():
            pep_rows = trait_neg[trait_neg["peptide"] == pep]
            n_sample = min(per_peptide, len(pep_rows))
            sampled_indices.extend(
                pep_rows.sample(n=n_sample, random_state=rng).index.tolist()
            )

        # If we haven't reached target, sample more randomly
        remaining = trait_neg_sample - len(sampled_indices)
        if remaining > 0:
            available = trait_neg.index.difference(sampled_indices)
            if len(available) > 0:
                extra = rng.choice(available, size=min(remaining, len(available)), replace=False)
                sampled_indices.extend(extra.tolist())

        trait_neg_sampled = trait_neg.loc[sampled_indices[:trait_neg_sample]].copy()
    else:
        trait_neg_sampled = trait_neg.copy()

    trait_neg_sampled["label"] = "neg_soft"
    print(f"    TRAIT soft negatives sampled: {len(trait_neg_sampled):,}")

    # --- Step 4d: Flag leaked sequences ---
    def flag_leakage(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        trb_leaked = df["trb"].isin(leaked_cdr3s) if "trb" in df.columns else pd.Series(False, index=df.index)
        tra_leaked = df["tra"].isin(leaked_cdr3s) if "tra" in df.columns else pd.Series(False, index=df.index)
        df["leakage_flag"] = trb_leaked | tra_leaked
        return df

    batman_pos = flag_leakage(batman_pos)
    batman_neg = flag_leakage(batman_neg)
    trait_neg_sampled = flag_leakage(trait_neg_sampled)

    # Combine batman rows
    batman_combined = pd.concat([batman_pos, batman_neg], ignore_index=True)

    # Split batman 50/50 into test/val
    batman_shuffled = batman_combined.sample(frac=1.0, random_state=seed + 300).reset_index(drop=True)
    n_half = len(batman_shuffled) // 2
    batman_test = batman_shuffled.iloc[:n_half].copy()
    batman_val = batman_shuffled.iloc[n_half:].copy()

    # Split trait neg 50/50 independently
    trait_shuffled = trait_neg_sampled.sample(frac=1.0, random_state=seed + 301).reset_index(drop=True)
    n_trait_half = len(trait_shuffled) // 2
    trait_test = trait_shuffled.iloc[:n_trait_half].copy()
    trait_val = trait_shuffled.iloc[n_trait_half:].copy()

    # Combine
    probe_test = pd.concat([batman_test, trait_test], ignore_index=True)
    probe_val = pd.concat([batman_val, trait_val], ignore_index=True)

    output_cols = ["tra", "trb", "peptide", "mhc_one", "mhc_two", "binding", "label", "score", "source", "leakage_flag"]
    probe_test = probe_test[[c for c in output_cols if c in probe_test.columns]]
    probe_val = probe_val[[c for c in output_cols if c in probe_val.columns]]

    # Save
    probe_test.to_parquet(phase_dir / "binding_probe_test.parquet", index=False)
    probe_val.to_parquet(phase_dir / "binding_probe_val.parquet", index=False)

    # Stats
    stats = {
        "batman_total": len(batman_df),
        "batman_positives": len(batman_pos),
        "batman_negatives": len(batman_neg),
        "trait_total": len(trait_df),
        "trait_neg_sampled": len(trait_neg_sampled),
        "probe_test_size": len(probe_test),
        "probe_val_size": len(probe_val),
        "probe_test_labels": probe_test["label"].value_counts().to_dict() if "label" in probe_test.columns else {},
        "probe_val_labels": probe_val["label"].value_counts().to_dict() if "label" in probe_val.columns else {},
        "leakage": {
            "total_matches": total_match_count,
            "unique_leaked_cdr3s": len(leaked_cdr3s),
            "test_leaked_rows": int(probe_test["leakage_flag"].sum()) if "leakage_flag" in probe_test.columns else 0,
            "val_leaked_rows": int(probe_val["leakage_flag"].sum()) if "leakage_flag" in probe_val.columns else 0,
        },
    }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"    Probe test: {len(probe_test):,} rows")
    print(f"    Probe val: {len(probe_val):,} rows")
    print(f"    Leakage-flagged rows (test): {stats['leakage']['test_leaked_rows']:,}")
    print(f"    Leakage-flagged rows (val): {stats['leakage']['val_leaked_rows']:,}")

    return stats


# ---------------------------------------------------------------------------
# Phase 5: Validate and Write Manifest
# ---------------------------------------------------------------------------


def phase5_validate_and_manifest(
    quintet_df: pd.DataFrame,
    quartet_df: pd.DataFrame,
    split_assignments: pd.DataFrame,
    group_counts: Dict[str, Any],
    subset_manifest: Dict[str, Any],
    binding_probe_stats: Dict[str, Any],
    output_dir: Path,
    seed: int,
    trb_similarity: float,
    peptide_similarity: float,
) -> Dict[str, Any]:
    """Phase 5: Comprehensive validation + final manifest with checksums."""
    phase_dir = output_dir / "phase5_manifest"
    phase_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = phase_dir / "manifest.json"
    if manifest_path.exists():
        print("  Phase 5: Loading from checkpoint...")
        with open(manifest_path) as f:
            return json.load(f)

    print("  Phase 5: Running validation checks...")
    checks = {}

    # 1. No TRB cluster leakage for quintets/quartets
    print("    Check 1: TRB cluster isolation...")
    for name, df in [("quintet", quintet_df), ("quartet", quartet_df)]:
        if len(df) == 0 or "trb_cluster" not in df.columns:
            checks[f"{name}_trb_cluster_isolated"] = True
            continue

        # Merge with split assignments
        merged = df.merge(
            split_assignments[["batch_idx", "row_idx", "split"]],
            on=["batch_idx", "row_idx"],
            how="left",
        )
        if "split" in merged.columns:
            cluster_splits = merged.groupby("trb_cluster")["split"].nunique()
            multi = cluster_splits[cluster_splits > 1]
            checks[f"{name}_trb_cluster_isolated"] = len(multi) == 0
            if len(multi) > 0:
                print(f"    FAIL: {name} has {len(multi)} TRB clusters spanning multiple splits")
            else:
                print(f"    PASS: {name} TRB clusters are isolated")
        else:
            checks[f"{name}_trb_cluster_isolated"] = True

    # 2. No peptide cluster monopolized
    print("    Check 2: Peptide cluster coverage...")
    if len(quintet_df) > 0 and "peptide_cluster" in quintet_df.columns:
        merged = quintet_df.merge(
            split_assignments[["batch_idx", "row_idx", "split"]],
            on=["batch_idx", "row_idx"],
            how="left",
        )
        if "split" in merged.columns:
            all_pep = set(merged["peptide_cluster"].unique()) - {-1}
            train_pep = set(merged[merged["split"] == "train"]["peptide_cluster"].unique()) - {-1}
            missing = all_pep - train_pep
            checks["no_peptide_cluster_monopolized"] = len(missing) == 0
            if missing:
                print(f"    FAIL: {len(missing)} peptide clusters absent from train")
            else:
                print(f"    PASS: All peptide clusters represented in train")
    else:
        checks["no_peptide_cluster_monopolized"] = True

    # 3. All rows accounted for
    print("    Check 3: Row accounting...")
    total_rows = group_counts.get("total_rows", 0)
    if total_rows > 0:
        # We can't easily count without re-scanning, so we verify the split summary
        split_summary_path = output_dir / "phase2_splits" / "split_summary.json"
        if split_summary_path.exists():
            with open(split_summary_path) as f:
                ss = json.load(f)
            overall = ss.get("overall", {})
            accounted = sum(overall.values())
            checks["all_rows_accounted"] = accounted == total_rows
            if accounted != total_rows:
                print(f"    FAIL: Accounted {accounted:,} != total {total_rows:,}")
            else:
                print(f"    PASS: All {total_rows:,} rows accounted for")
        else:
            checks["all_rows_accounted"] = "unable_to_verify"
    else:
        checks["all_rows_accounted"] = True

    # 4. Nested subset verification
    print("    Check 4: Nested subset verification...")
    phase3_dir = output_dir / "phase3_subsets"
    prev_set = None
    prev_name = None
    nested_ok = True
    for size in PROPORTIONAL_SIZES:
        name = f"proportional_{size // 1_000_000}M"
        npy_path = phase3_dir / f"{name}.npy"
        if npy_path.exists():
            arr = np.load(npy_path)
            current_set = set(map(tuple, arr.tolist()))
            if prev_set is not None:
                if not prev_set.issubset(current_set):
                    print(f"    FAIL: {prev_name} is NOT a subset of {name}")
                    nested_ok = False
            prev_set = current_set
            prev_name = name
    checks["nested_subsets_valid"] = nested_ok
    if nested_ok:
        print("    PASS: Nested subset property holds")

    # 5. Component distribution checks per subset
    print("    Check 5: Component distributions...")
    checks["subset_distributions"] = subset_manifest.get("cardinality_distribution", {})

    # 6. Binding probe checks
    print("    Check 6: Binding probe validation...")
    if not binding_probe_stats.get("skipped", False):
        bp_test_path = output_dir / "phase4_binding_probes" / "binding_probe_test.parquet"
        bp_val_path = output_dir / "phase4_binding_probes" / "binding_probe_val.parquet"
        if bp_test_path.exists() and bp_val_path.exists():
            bp_test = pd.read_parquet(bp_test_path)
            bp_val = pd.read_parquet(bp_val_path)
            test_labels = set(bp_test["label"].unique()) if "label" in bp_test.columns else set()
            val_labels = set(bp_val["label"].unique()) if "label" in bp_val.columns else set()
            expected_labels = {"pos", "neg_true", "neg_soft"}
            checks["binding_probe_labels_test"] = test_labels == expected_labels
            checks["binding_probe_labels_val"] = val_labels == expected_labels
            if test_labels != expected_labels:
                print(f"    FAIL: Test probe labels {test_labels} != {expected_labels}")
            else:
                print(f"    PASS: Test probe has all 3 label types")
            if val_labels != expected_labels:
                print(f"    FAIL: Val probe labels {val_labels} != {expected_labels}")
            else:
                print(f"    PASS: Val probe has all 3 label types")
        else:
            checks["binding_probe_labels_test"] = "files_not_found"
            checks["binding_probe_labels_val"] = "files_not_found"
    else:
        checks["binding_probe_skipped"] = True

    # 7. Val/test each have >=10 distinct peptide clusters
    print("    Check 7: Peptide cluster diversity in val/test...")
    if len(quintet_df) > 0 and "peptide_cluster" in quintet_df.columns:
        merged = quintet_df.merge(
            split_assignments[["batch_idx", "row_idx", "split"]],
            on=["batch_idx", "row_idx"],
            how="left",
        )
        for split_name in ["test", "val"]:
            if "split" in merged.columns:
                split_rows = merged[merged["split"] == split_name]
                n_clusters = split_rows["peptide_cluster"].nunique()
                checks[f"{split_name}_peptide_cluster_diversity"] = n_clusters >= 10
                if n_clusters < 10:
                    print(f"    FAIL: {split_name} has only {n_clusters} peptide clusters (want >=10)")
                else:
                    print(f"    PASS: {split_name} has {n_clusters} peptide clusters")
    else:
        checks["test_peptide_cluster_diversity"] = True
        checks["val_peptide_cluster_diversity"] = True

    # --- Compute checksums ---
    print("    Computing file checksums...")
    checksums = {}
    for phase_name in ["phase0_index", "phase1_clusters", "phase2_splits", "phase3_subsets", "phase4_binding_probes"]:
        phase_path = output_dir / phase_name
        if phase_path.exists():
            for fpath in sorted(phase_path.iterdir()):
                if fpath.is_file():
                    rel = str(fpath.relative_to(output_dir))
                    checksums[rel] = _sha256_file(fpath)

    # --- Build manifest ---
    manifest = {
        "creation_date": datetime.now().isoformat(),
        "seed": seed,
        "mmseqs2_parameters": {
            "trb_similarity": trb_similarity,
            "peptide_similarity": peptide_similarity,
            "coverage": 0.8,
        },
        "total_rows": group_counts.get("total_rows", 0),
        "split_summary": {},
        "subset_summary": subset_manifest.get("subsets", {}),
        "binding_probe_summary": binding_probe_stats,
        "validation_checks": checks,
        "checksums": checksums,
    }

    # Load split summary if available
    split_summary_path = output_dir / "phase2_splits" / "split_summary.json"
    if split_summary_path.exists():
        with open(split_summary_path) as f:
            manifest["split_summary"] = json.load(f)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Print summary
    all_passed = all(
        v is True for k, v in checks.items()
        if not k.endswith("_distributions") and v != "unable_to_verify"
        and not k.endswith("_skipped")
    )
    print(f"\n    Validation: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    print(f"    Manifest written to {manifest_path}")

    return manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits and nested training subsets for mlm_full dedup output."
    )
    parser.add_argument(
        "--input_dir", type=Path, required=True,
        help="Path to mlm_full dedup parquet directory",
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Output directory for splits and subsets",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num_workers", type=int,
        default=max(1, multiprocessing.cpu_count() // 2),
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--trb_similarity", type=float, default=0.9,
        help="TRB clustering similarity threshold",
    )
    parser.add_argument(
        "--peptide_similarity", type=float, default=0.8,
        help="Peptide clustering similarity threshold",
    )
    parser.add_argument(
        "--batman_dir", type=Path, default=None,
        help="Path to standardized BATMAN data directory",
    )
    parser.add_argument(
        "--trait_dir", type=Path, default=None,
        help="Path to standardized TRAIT data directory",
    )
    parser.add_argument(
        "--trait_neg_sample", type=int, default=18000,
        help="Number of TRAIT soft negatives to sample",
    )
    parser.add_argument(
        "--hla_dir", type=Path, default=None,
        help="Path to IMGT/HLA FASTA directory for MHC sequence resolution",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip completed phases (resume from checkpoints)",
    )
    args = parser.parse_args()

    # Validate --hla_dir early so the user gets a clear error
    if args.hla_dir:
        if not args.hla_dir.is_dir():
            parser.error(f"--hla_dir does not exist: {args.hla_dir}")
        fasta_files = list(args.hla_dir.glob("*_prot.fasta"))
        if not fasta_files:
            parser.error(f"--hla_dir contains no *_prot.fasta files: {args.hla_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Seed: {args.seed}")
    print(f"Workers: {args.num_workers}")
    if args.hla_dir:
        print(f"HLA dir: {args.hla_dir}")
    print()

    # Phase 0a (optional)
    if args.hla_dir:
        print("=" * 60)
        print("PHASE 0a: Resolve MHC Alleles")
        print("=" * 60)
        resolved_dir = phase0a_resolve_mhc(
            args.input_dir, args.output_dir, args.hla_dir, args.num_workers
        )
        print()
    else:
        resolved_dir = args.input_dir

    # Phase 0
    print("=" * 60)
    print("PHASE 0: Scan and Index")
    print("=" * 60)
    quintet_df, quartet_df, group_counts = phase0_scan_and_index(
        resolved_dir, args.output_dir, args.num_workers
    )
    print()

    # Phase 1
    print("=" * 60)
    print("PHASE 1: Cluster")
    print("=" * 60)
    quintet_df, quartet_df = phase1_cluster(
        quintet_df, quartet_df, args.output_dir,
        args.trb_similarity, args.peptide_similarity,
    )
    print()

    # Phase 2
    print("=" * 60)
    print("PHASE 2: Assign Splits")
    print("=" * 60)
    split_assignments = phase2_assign_splits(
        quintet_df, quartet_df, group_counts, args.output_dir, args.seed,
    )
    print()

    # Phase 3
    print("=" * 60)
    print("PHASE 3: Build Nested Training Subsets")
    print("=" * 60)
    subset_manifest = phase3_build_subsets(
        split_assignments, group_counts, resolved_dir, args.output_dir,
        args.seed, args.num_workers,
    )
    print()

    # Phase 4
    print("=" * 60)
    print("PHASE 4: Binding Probe Eval Sets")
    print("=" * 60)
    binding_probe_stats = phase4_binding_probes(
        resolved_dir, args.output_dir, split_assignments,
        args.seed, args.num_workers,
        args.batman_dir, args.trait_dir, args.trait_neg_sample,
    )
    print()

    # Phase 5
    print("=" * 60)
    print("PHASE 5: Validate and Write Manifest")
    print("=" * 60)
    manifest = phase5_validate_and_manifest(
        quintet_df, quartet_df, split_assignments,
        group_counts, subset_manifest, binding_probe_stats,
        args.output_dir, args.seed, args.trb_similarity, args.peptide_similarity,
    )
    print()

    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
