#!/usr/bin/env python3
"""
Create TCR antigenic specificity evaluation datasets with hierarchical stratified splits.

Creates evaluation datasets for training an encoder-decoder model that predicts peptides
given TCR (alpha/beta chain CDR3) and MHC context. Uses hierarchical stratified splitting
to prevent data leakage at multiple levels.

Model Architecture Target:
- Input: TCR sequences (alpha and/or beta chain CDR3) + MHC sequences
- Output: Peptide sequence (decoder target)
- Encoder: ESM2 for TCR/MHC encoding
- Decoder: Custom decoder for peptide generation

Data Leakage Considerations:
- TCR CDR3 similarity: Clustered at 90% identity
- Peptide similarity: Clustered at 80% identity
- MHC allele: Hold out alleles entirely
- Public epitopes: Stratify by epitope/peptide
- Exact TCR-peptide pairs: Deduplicate strictly

Test Set Difficulty Levels:
    | Test Set                      | Description                           | What it Tests                        |
    |-------------------------------|---------------------------------------|--------------------------------------|
    | test_seen_epitope             | Same peptides as training             | Baseline - can model learn mapping?  |
    | test_unseen_tcr_seen_epitope  | New TCR clusters, peptides seen       | Generalize to new TCRs for known pep |
    | test_unseen_epitope           | Peptide clusters not in training      | True generalization - novel peptides |
    | test_unseen_allele            | MHC alleles not in training           | Hardest - no MHC context shortcuts   |

Output Structure:
    output_dir/
    ├── dataset_statistics.md
    ├── tcr_clusters.tsv              # TCR CDR3 -> cluster mapping
    ├── peptide_clusters.tsv          # Peptide -> cluster mapping
    ├── split_assignments.tsv         # Full assignment table
    ├── class_one/
    │   ├── tra_peptide_mhc_one/
    │   │   ├── train.parquet
    │   │   ├── val.parquet
    │   │   ├── test_seen_epitope.parquet
    │   │   ├── test_unseen_tcr_seen_epitope.parquet
    │   │   ├── test_unseen_epitope.parquet
    │   │   └── test_unseen_allele.parquet
    │   ├── trb_peptide_mhc_one/
    │   │   └── ...
    │   └── tra_trb_peptide_mhc_one/
    │       └── ...
    └── class_two/
        └── ...

Dependencies:
    - tidytcells: For MHC and TCR gene standardization (pip install tidytcells)
    - MMseqs2: For sequence clustering (conda install -c bioconda mmseqs2)
    - tqdm: For progress bars

Usage:
    python scripts/data_processing/create_tcr_specificity_dataset.py \\
        --output_dir data/eval/tcr_specificity \\
        --tcr_similarity_threshold 0.9 \\
        --peptide_similarity_threshold 0.8 \\
        --holdout_allele_fraction 0.1 \\
        --holdout_epitope_fraction 0.1

Author: Shashidhar Ravishankar, Claude
"""

import argparse
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Try to import tidytcells for MHC and TCR gene standardization
try:
    import tidytcells as tt
    TIDYTCELLS_AVAILABLE = True
except ImportError:
    TIDYTCELLS_AVAILABLE = False

# Valid amino acids for sequence validation
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# MHC Class I patterns (single alpha chain)
MHC_CLASS_ONE_PATTERNS = ['HLA-A', 'HLA-B', 'HLA-C', 'H-2K', 'H-2D', 'H-2L']

# MHC Class II patterns (alpha and beta chains)
MHC_CLASS_TWO_PATTERNS = ['HLA-DR', 'HLA-DQ', 'HLA-DP', 'HLA-DM', 'HLA-DO',
                          'H-2-IA', 'H-2-IE', 'H2-IA', 'H2-IE']


def get_mhc_class(allele_id: str) -> Optional[str]:
    """
    Determine MHC class from allele ID by inspecting the allele name.

    MHC Class I molecules consist of a single alpha chain (HLA-A, HLA-B, HLA-C).
    MHC Class II molecules consist of an alpha + beta heterodimer:
    - Alpha chains: HLA-DRA, HLA-DQA, HLA-DPA
    - Beta chains: HLA-DRB, HLA-DQB, HLA-DPB

    Args:
        allele_id: The MHC allele identifier (e.g., "HLA-A*02:01", "HLA-DRB1*04:01")

    Returns:
        'class_one' for HLA-A, HLA-B, HLA-C
        'class_two' for HLA-DR, HLA-DQ, HLA-DP
        None if unable to determine
    """
    if not allele_id or pd.isna(allele_id):
        return None

    if isinstance(allele_id, float):
        return None

    allele_str = str(allele_id).upper().strip()
    if not allele_str or allele_str == "NAN":
        return None

    # Check Class I patterns first
    for pattern in MHC_CLASS_ONE_PATTERNS:
        if pattern in allele_str:
            return 'class_one'

    # Check Class II patterns
    for pattern in MHC_CLASS_TWO_PATTERNS:
        if pattern in allele_str:
            return 'class_two'

    return None

# Permutation key configurations for TCR specificity prediction
# Each config specifies which TCR chains and MHC class to use
#
# Note on MHC Classes:
# - MHC Class I: Single chain (mhc_one only) - uses mhc_one_id for allele ID
# - MHC Class II: Heterodimer with alpha + beta chains - uses BOTH mhc_one_id AND mhc_two_id
PERMUTATION_CONFIGS = {
    # ===== MHC CLASS I (single chain) =====
    "tra_peptide_mhc_one": {
        "description": "TCR alpha only + MHC-I -> peptide",
        "mhc_class": "class_one",
        "required_cols": ["tra", "peptide", "mhc_one_id", "mhc_one"],
        "tcr_cols": ["tra"],
        "mhc_id_cols": ["mhc_one_id"],  # Single allele for Class I
        "mhc_seq_cols": ["mhc_one"],
        "dedup_cols": ["tra", "peptide", "mhc_one_id"],
        "output_cols": [
            "tra", "trav_gene", "traj_gene",
            "peptide", "mhc_one_id", "mhc_one",
            "permutation_key", "source",
        ],
    },
    "trb_peptide_mhc_one": {
        "description": "TCR beta only + MHC-I -> peptide",
        "mhc_class": "class_one",
        "required_cols": ["trb", "peptide", "mhc_one_id", "mhc_one"],
        "tcr_cols": ["trb"],
        "mhc_id_cols": ["mhc_one_id"],
        "mhc_seq_cols": ["mhc_one"],
        "dedup_cols": ["trb", "peptide", "mhc_one_id"],
        "output_cols": [
            "trb", "trbv_gene", "trbd_gene", "trbj_gene",
            "peptide", "mhc_one_id", "mhc_one",
            "permutation_key", "source",
        ],
    },
    "tra_trb_peptide_mhc_one": {
        "description": "TCR alpha+beta paired + MHC-I -> peptide",
        "mhc_class": "class_one",
        "required_cols": ["tra", "trb", "peptide", "mhc_one_id", "mhc_one"],
        "tcr_cols": ["tra", "trb"],
        "mhc_id_cols": ["mhc_one_id"],
        "mhc_seq_cols": ["mhc_one"],
        "dedup_cols": ["tra", "trb", "peptide", "mhc_one_id"],
        "output_cols": [
            "tra", "trav_gene", "traj_gene",
            "trb", "trbv_gene", "trbd_gene", "trbj_gene",
            "peptide", "mhc_one_id", "mhc_one",
            "permutation_key", "source",
        ],
    },

    # ===== MHC CLASS II (alpha + beta heterodimer) =====
    "tra_peptide_mhc_two": {
        "description": "TCR alpha only + MHC-II (alpha+beta) -> peptide",
        "mhc_class": "class_two",
        "required_cols": ["tra", "peptide", "mhc_one_id", "mhc_two_id", "mhc_one", "mhc_two"],
        "tcr_cols": ["tra"],
        "mhc_id_cols": ["mhc_one_id", "mhc_two_id"],  # Both alleles for Class II
        "mhc_seq_cols": ["mhc_one", "mhc_two"],
        "dedup_cols": ["tra", "peptide", "mhc_one_id", "mhc_two_id"],
        "output_cols": [
            "tra", "trav_gene", "traj_gene",
            "peptide", "mhc_one_id", "mhc_two_id", "mhc_one", "mhc_two",
            "permutation_key", "source",
        ],
    },
    "trb_peptide_mhc_two": {
        "description": "TCR beta only + MHC-II (alpha+beta) -> peptide",
        "mhc_class": "class_two",
        "required_cols": ["trb", "peptide", "mhc_one_id", "mhc_two_id", "mhc_one", "mhc_two"],
        "tcr_cols": ["trb"],
        "mhc_id_cols": ["mhc_one_id", "mhc_two_id"],
        "mhc_seq_cols": ["mhc_one", "mhc_two"],
        "dedup_cols": ["trb", "peptide", "mhc_one_id", "mhc_two_id"],
        "output_cols": [
            "trb", "trbv_gene", "trbd_gene", "trbj_gene",
            "peptide", "mhc_one_id", "mhc_two_id", "mhc_one", "mhc_two",
            "permutation_key", "source",
        ],
    },
    "tra_trb_peptide_mhc_two": {
        "description": "TCR alpha+beta paired + MHC-II (alpha+beta) -> peptide",
        "mhc_class": "class_two",
        "required_cols": ["tra", "trb", "peptide", "mhc_one_id", "mhc_two_id", "mhc_one", "mhc_two"],
        "tcr_cols": ["tra", "trb"],
        "mhc_id_cols": ["mhc_one_id", "mhc_two_id"],
        "mhc_seq_cols": ["mhc_one", "mhc_two"],
        "dedup_cols": ["tra", "trb", "peptide", "mhc_one_id", "mhc_two_id"],
        "output_cols": [
            "tra", "trav_gene", "traj_gene",
            "trb", "trbv_gene", "trbd_gene", "trbj_gene",
            "peptide", "mhc_one_id", "mhc_two_id", "mhc_one", "mhc_two",
            "permutation_key", "source",
        ],
    },
}


def is_valid_sequence(seq: Optional[str]) -> bool:
    """Check if sequence is valid (non-null, non-empty, valid amino acids)."""
    if seq is None or pd.isna(seq) or seq == "" or seq == "nan":
        return False
    if isinstance(seq, float):
        return False
    return all(aa in VALID_AA for aa in str(seq).upper())


def is_valid_value(val) -> bool:
    """Check if a value is valid (non-null, non-empty)."""
    if val is None or pd.isna(val):
        return False
    if isinstance(val, float) and np.isnan(val):
        return False
    if isinstance(val, str) and (val.strip() == "" or val.lower() == "nan"):
        return False
    return True


def standardize_mhc_allele(allele: Optional[str], species: str = "homosapiens") -> Optional[str]:
    """
    Standardize MHC allele name using tidytcells.

    Converts various MHC naming formats to IMGT-compliant format.
    Examples:
        - "A1" -> "HLA-A*01"
        - "HLA-A*0101" -> "HLA-A*01:01"
        - "A*02:01" -> "HLA-A*02:01"

    Args:
        allele: The MHC allele name to standardize
        species: Species for standardization ("homosapiens" or "musmusculus")

    Returns:
        Standardized allele name, or original if standardization fails
    """
    if not TIDYTCELLS_AVAILABLE:
        return allele

    if allele is None or pd.isna(allele):
        return None

    if isinstance(allele, float):
        return None

    allele_str = str(allele).strip()
    if not allele_str or allele_str.lower() == "nan":
        return None

    try:
        standardized = tt.mh.standardize(
            allele_str,
            species=species,
            precision="allele",
            on_fail="keep",
            suppress_warnings=True,
        )
        return standardized if standardized else allele_str
    except Exception:
        return allele_str


def standardize_tcr_gene(gene: Optional[str], species: str = "homosapiens") -> Optional[str]:
    """
    Standardize TCR V/D/J gene name using tidytcells.tr module.

    Args:
        gene: The TCR gene name to standardize (e.g., "TRAV1-1", "TRBV5-1")
        species: Species for standardization

    Returns:
        Standardized gene name, or original if standardization fails
    """
    if not TIDYTCELLS_AVAILABLE:
        return gene

    if gene is None or pd.isna(gene):
        return None

    if isinstance(gene, float):
        return None

    gene_str = str(gene).strip()
    if not gene_str or gene_str.lower() == "nan":
        return None

    try:
        standardized = tt.tr.standardize(
            gene_str,
            species=species,
            precision="gene",
            on_fail="keep",
            suppress_warnings=True,
        )
        return standardized if standardized else gene_str
    except Exception:
        return gene_str


def standardize_mhc_columns(df: pd.DataFrame, mhc_id_cols: List[str]) -> pd.DataFrame:
    """
    Standardize MHC allele ID columns in a DataFrame.

    Args:
        df: DataFrame with MHC columns
        mhc_id_cols: List of column names to standardize

    Returns:
        DataFrame with standardized MHC columns
    """
    if not TIDYTCELLS_AVAILABLE:
        print("    Warning: tidytcells not available, skipping MHC standardization")
        return df

    df = df.copy()

    for col in mhc_id_cols:
        if col not in df.columns:
            continue

        print(f"    Standardizing {col} using tidytcells...")
        original_valid = df[col].apply(is_valid_value).sum()

        tqdm.pandas(desc=f"      Processing {col}", leave=False)
        df[col] = df[col].progress_apply(standardize_mhc_allele)

        new_valid = df[col].apply(is_valid_value).sum()
        print(f"      Valid values: {original_valid:,} -> {new_valid:,}")

    return df


def standardize_tcr_gene_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize TCR V/D/J gene columns using tidytcells.tr module.

    Columns to standardize:
    - trav_gene, traj_gene (alpha chain)
    - trbv_gene, trbd_gene, trbj_gene (beta chain)

    Args:
        df: DataFrame with TCR gene columns

    Returns:
        DataFrame with standardized TCR gene columns
    """
    if not TIDYTCELLS_AVAILABLE:
        print("    Warning: tidytcells not available, skipping TCR gene standardization")
        return df

    df = df.copy()

    gene_cols = ['trav_gene', 'traj_gene', 'trbv_gene', 'trbd_gene', 'trbj_gene']

    for col in gene_cols:
        if col not in df.columns:
            continue

        print(f"    Standardizing {col} using tidytcells.tr...")
        original_valid = df[col].apply(is_valid_value).sum()

        tqdm.pandas(desc=f"      Processing {col}", leave=False)
        df[col] = df[col].progress_apply(standardize_tcr_gene)

        new_valid = df[col].apply(is_valid_value).sum()
        print(f"      Valid values: {original_valid:,} -> {new_valid:,}")

    return df


def cluster_sequences_mmseqs2(
    sequences: List[str],
    similarity_threshold: float = 0.9,
    coverage: float = 0.8,
    temp_dir: Optional[Path] = None,
    seq_type: str = "tcr",
) -> Tuple[Dict[str, int], Dict[int, List[str]]]:
    """
    Cluster sequences using MMseqs2.

    Args:
        sequences: List of sequences to cluster
        similarity_threshold: Sequence identity threshold for clustering
        coverage: Alignment coverage threshold
        temp_dir: Optional temporary directory for MMseqs2 files
        seq_type: Type of sequences ("tcr" or "peptide") for logging

    Returns:
        seq_to_cluster: Dict mapping sequence -> cluster_id
        cluster_to_seqs: Dict mapping cluster_id -> list of sequences
    """
    if shutil.which("mmseqs") is None:
        raise RuntimeError(
            "MMseqs2 not found in PATH. Install with: conda install -c bioconda mmseqs2"
        )

    # Filter out invalid sequences
    valid_seqs = [s for s in sequences if s and isinstance(s, str) and len(s) > 0]
    valid_seqs = list(set(valid_seqs))  # Deduplicate

    if len(valid_seqs) == 0:
        return {}, {}

    print(f"    Clustering {len(valid_seqs):,} unique {seq_type} sequences at {similarity_threshold*100:.0f}% identity...")

    # Create temporary directory for MMseqs2 files
    cleanup_temp = False
    if temp_dir is None:
        temp_dir = Path(tempfile.mkdtemp(prefix=f"mmseqs2_{seq_type}_"))
        cleanup_temp = True
    else:
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Write sequences to FASTA file
        fasta_path = temp_dir / f"{seq_type}.fasta"
        with open(fasta_path, "w") as f:
            for i, seq in enumerate(valid_seqs):
                f.write(f">seq_{i}|{seq}\n{seq}\n")

        # MMseqs2 database and output paths
        db_path = temp_dir / f"{seq_type}_db"
        cluster_db = temp_dir / "cluster_db"
        cluster_tsv = temp_dir / "clusters.tsv"
        tmp_path = temp_dir / "tmp"
        tmp_path.mkdir(exist_ok=True)

        # Create MMseqs2 database
        cmd_createdb = ["mmseqs", "createdb", str(fasta_path), str(db_path)]
        subprocess.run(cmd_createdb, check=True, capture_output=True)

        # Run clustering
        cmd_cluster = [
            "mmseqs", "cluster",
            str(db_path), str(cluster_db), str(tmp_path),
            "--min-seq-id", str(similarity_threshold),
            "-c", str(coverage),
            "--cov-mode", "0",
            "-s", "7.5",
        ]
        subprocess.run(cmd_cluster, check=True, capture_output=True)

        # Convert cluster results to TSV
        cmd_tsv = [
            "mmseqs", "createtsv",
            str(db_path), str(db_path), str(cluster_db), str(cluster_tsv),
        ]
        subprocess.run(cmd_tsv, check=True, capture_output=True)

        # Parse cluster results
        seq_to_cluster: Dict[str, int] = {}
        cluster_to_seqs: Dict[int, List[str]] = {}

        cluster_id_map: Dict[str, int] = {}
        next_cluster_id = 0

        with open(cluster_tsv, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    rep_header, member_header = parts[0], parts[1]
                    rep_seq = rep_header.split("|")[-1] if "|" in rep_header else rep_header
                    member_seq = member_header.split("|")[-1] if "|" in member_header else member_header

                    if rep_seq not in cluster_id_map:
                        cluster_id_map[rep_seq] = next_cluster_id
                        next_cluster_id += 1

                    cluster_id = cluster_id_map[rep_seq]
                    seq_to_cluster[member_seq] = cluster_id

                    if cluster_id not in cluster_to_seqs:
                        cluster_to_seqs[cluster_id] = []
                    if member_seq not in cluster_to_seqs[cluster_id]:
                        cluster_to_seqs[cluster_id].append(member_seq)

        # Handle any sequences that might not appear in output
        for seq in valid_seqs:
            if seq not in seq_to_cluster:
                seq_to_cluster[seq] = next_cluster_id
                cluster_to_seqs[next_cluster_id] = [seq]
                next_cluster_id += 1

        print(f"    Created {len(cluster_to_seqs):,} clusters")
        return seq_to_cluster, cluster_to_seqs

    finally:
        if cleanup_temp and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def cluster_tcr_cdr3_mmseqs2(
    cdr3_sequences: List[str],
    similarity_threshold: float = 0.9,
    coverage: float = 0.8,
    temp_dir: Optional[Path] = None,
) -> Tuple[Dict[str, int], Dict[int, List[str]]]:
    """
    Cluster TCR CDR3 sequences using MMseqs2.

    Uses 90% identity threshold (conservative) because:
    - CDR3s are short (10-20 AA)
    - Small changes can significantly affect specificity

    Args:
        cdr3_sequences: List of TCR CDR3 sequences
        similarity_threshold: Sequence identity threshold (default: 0.9)
        coverage: Alignment coverage threshold (default: 0.8)
        temp_dir: Optional temporary directory

    Returns:
        cdr3_to_cluster: Dict mapping CDR3 -> cluster_id
        cluster_to_cdr3s: Dict mapping cluster_id -> list of CDR3s
    """
    return cluster_sequences_mmseqs2(
        cdr3_sequences,
        similarity_threshold=similarity_threshold,
        coverage=coverage,
        temp_dir=temp_dir,
        seq_type="tcr_cdr3",
    )


def cluster_peptides_mmseqs2(
    peptides: List[str],
    similarity_threshold: float = 0.8,
    coverage: float = 0.8,
    temp_dir: Optional[Path] = None,
) -> Tuple[Dict[str, int], Dict[int, List[str]]]:
    """
    Cluster peptides using MMseqs2.

    Args:
        peptides: List of peptide sequences
        similarity_threshold: Sequence identity threshold (default: 0.8)
        coverage: Alignment coverage threshold (default: 0.8)
        temp_dir: Optional temporary directory

    Returns:
        peptide_to_cluster: Dict mapping peptide -> cluster_id
        cluster_to_peptides: Dict mapping cluster_id -> list of peptides
    """
    return cluster_sequences_mmseqs2(
        peptides,
        similarity_threshold=similarity_threshold,
        coverage=coverage,
        temp_dir=temp_dir,
        seq_type="peptide",
    )


def load_databases(input_dir: Path) -> pd.DataFrame:
    """
    Load and combine TCR data from parser databases.

    Args:
        input_dir: Path to parser_databases directory

    Returns:
        Combined DataFrame with all databases
    """
    # Database files to load
    database_files = [
        "iedb.parquet",
        "vdjdb_seq.parquet",
        "mcpas_seq.parquet",
    ]

    dfs = []
    for db_file in database_files:
        db_path = input_dir / db_file
        if db_path.exists():
            print(f"  Loading {db_file}...")
            df = pq.read_table(db_path).to_pandas()

            # Standardize column names
            # McPAS uses 'mhc_restriction' instead of mhc_one_id
            if 'mhc_restriction' in df.columns and 'mhc_one_id' not in df.columns:
                df['mhc_one_id'] = df['mhc_restriction']

            # Ensure required columns exist
            required_cols = [
                'tra', 'trb', 'peptide',
                'trav_gene', 'traj_gene',
                'trbv_gene', 'trbd_gene', 'trbj_gene',
                'mhc_one_id', 'mhc_one',
                'mhc_two_id', 'mhc_two',
                'source',
            ]
            for col in required_cols:
                if col not in df.columns:
                    df[col] = None

            print(f"    Loaded {len(df):,} rows from {db_file}")
            dfs.append(df)
        else:
            print(f"  Warning: {db_file} not found at {db_path}")

    if not dfs:
        raise ValueError(f"No database files found in {input_dir}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total combined: {len(combined):,} rows")

    return combined


def filter_by_peptide_length(df: pd.DataFrame, min_len: int, max_len: int) -> pd.DataFrame:
    """Filter dataframe by peptide length."""
    df = df.copy()
    df['peptide'] = df['peptide'].astype(str)

    peptide_lengths = df['peptide'].str.len()
    mask = (peptide_lengths >= min_len) & (peptide_lengths <= max_len)
    filtered = df[mask].copy()

    print(f"    Filtered by peptide length ({min_len}-{max_len}): {len(df):,} -> {len(filtered):,}")
    return filtered


def process_permutation_key(
    df: pd.DataFrame,
    perm_key: str,
    config: Dict,
    min_peptide_len: int,
    max_peptide_len: int,
) -> pd.DataFrame:
    """
    Process data for a specific permutation key type.

    Args:
        df: Input DataFrame
        perm_key: Permutation key name
        config: Configuration dict for this permutation type
        min_peptide_len: Minimum peptide length
        max_peptide_len: Maximum peptide length

    Returns:
        Processed and deduplicated DataFrame
    """
    print(f"\n  Processing {perm_key}: {config['description']}")

    result = df.copy()

    # Filter by peptide length
    result = filter_by_peptide_length(result, min_peptide_len, max_peptide_len)

    # Filter rows with valid required columns
    required_cols = config["required_cols"]
    print(f"    Required columns: {required_cols}")

    for col in required_cols:
        if col not in result.columns:
            print(f"    Warning: Column {col} not found, skipping this permutation type")
            return pd.DataFrame()

        # Filter for valid values
        if col in ['peptide', 'tra', 'trb', 'mhc_one', 'mhc_two']:
            valid_mask = result[col].apply(is_valid_sequence)
        else:
            valid_mask = result[col].apply(is_valid_value)

        before = len(result)
        result = result[valid_mask].copy()
        print(f"    After filtering for valid {col}: {before:,} -> {len(result):,}")

    if len(result) == 0:
        print(f"    No valid rows for {perm_key}")
        return pd.DataFrame()

    # Filter by actual MHC class based on allele ID (not just column presence)
    # This ensures we correctly identify Class I vs Class II based on HLA gene name
    mhc_class = config["mhc_class"]
    mhc_id_cols = config.get("mhc_id_cols", ["mhc_one_id"])

    for mhc_col in mhc_id_cols:
        if mhc_col in result.columns:
            before = len(result)
            result['_mhc_class'] = result[mhc_col].apply(get_mhc_class)
            # For Class I, mhc_one_id should be class_one
            # For Class II, both mhc_one_id and mhc_two_id should be class_two
            result = result[result['_mhc_class'] == mhc_class]
            result = result.drop(columns=['_mhc_class'])
            print(f"    After filtering {mhc_col} for {mhc_class}: {before:,} -> {len(result):,}")

    if len(result) == 0:
        print(f"    No valid rows for {perm_key} after MHC class filtering")
        return pd.DataFrame()

    # Deduplicate
    dedup_cols = config["dedup_cols"]
    before_dedup = len(result)
    result = result.drop_duplicates(subset=dedup_cols, keep='first')
    print(f"    After deduplication on {dedup_cols}: {before_dedup:,} -> {len(result):,}")

    # Add permutation key column
    result['permutation_key'] = perm_key

    # Select output columns (only those available)
    output_cols = config["output_cols"]
    available_cols = [col for col in output_cols if col in result.columns]

    # Add any extra columns that exist
    extra_cols = ['tcr_cluster', 'peptide_cluster']
    for col in extra_cols:
        if col in result.columns and col not in available_cols:
            available_cols.append(col)

    result = result[available_cols].copy()

    return result


def hierarchical_tcr_split(
    df: pd.DataFrame,
    train_ratio: float,
    seed: int,
    tcr_similarity_threshold: float = 0.9,
    peptide_similarity_threshold: float = 0.8,
    holdout_allele_fraction: float = 0.1,
    holdout_epitope_fraction: float = 0.1,
    tcr_cols: List[str] = None,
    mhc_id_cols: List[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Create hierarchical splits with multiple difficulty levels for TCR specificity.

    Strategy:
    1. Cluster TCR CDR3 sequences at tcr_similarity_threshold identity
    2. Cluster peptides at peptide_similarity_threshold identity
    3. Hold out MHC alleles -> test_unseen_allele
    4. Hold out peptide clusters -> test_unseen_epitope
    5. Split remaining TCR clusters -> train/val
    6. From train peptide clusters, select unseen TCRs -> test_unseen_tcr_seen_epitope
    7. Sample from train distribution -> test_seen_epitope (baseline)

    Args:
        df: Input DataFrame with TCR, peptide, and MHC columns
        train_ratio: Fraction of data for training
        seed: Random seed
        tcr_similarity_threshold: TCR CDR3 clustering threshold (default: 0.9)
        peptide_similarity_threshold: Peptide clustering threshold (default: 0.8)
        holdout_allele_fraction: Fraction of MHC alleles to hold out (default: 0.1)
        holdout_epitope_fraction: Fraction of peptide clusters to hold out (default: 0.1)
        tcr_cols: TCR CDR3 column names (e.g., ['tra'], ['trb'], or ['tra', 'trb'])
        mhc_id_cols: Column names for MHC allele IDs (e.g., ['mhc_one_id'] for Class I,
                     ['mhc_one_id', 'mhc_two_id'] for Class II)

    Returns:
        Dict with splits and metadata
    """
    np.random.seed(seed)

    if tcr_cols is None:
        tcr_cols = ['tra', 'trb']

    if mhc_id_cols is None:
        mhc_id_cols = ['mhc_one_id']

    if len(df) == 0:
        return {
            'train': pd.DataFrame(),
            'val': pd.DataFrame(),
            'test_seen_epitope': pd.DataFrame(),
            'test_unseen_tcr_seen_epitope': pd.DataFrame(),
            'test_unseen_epitope': pd.DataFrame(),
            'test_unseen_allele': pd.DataFrame(),
        }

    # Step 1: Cluster TCR CDR3 sequences
    print(f"    Step 1: Clustering TCR CDR3 sequences...")
    all_tcr_seqs = []
    for col in tcr_cols:
        if col in df.columns:
            seqs = df[col].dropna().unique().tolist()
            all_tcr_seqs.extend(seqs)

    tcr_to_cluster, tcr_clusters = cluster_tcr_cdr3_mmseqs2(
        all_tcr_seqs,
        similarity_threshold=tcr_similarity_threshold,
        coverage=0.8,
    )

    # Create composite TCR cluster ID based on all TCR columns
    def get_tcr_cluster_id(row):
        cluster_ids = []
        for col in tcr_cols:
            if col in row.index and is_valid_value(row[col]):
                cid = tcr_to_cluster.get(row[col], -1)
                cluster_ids.append(cid)
        if cluster_ids:
            return min(cluster_ids)  # Use minimum cluster ID as representative
        return -1

    df = df.copy()
    df['tcr_cluster'] = df.apply(get_tcr_cluster_id, axis=1)

    # Step 2: Cluster peptides
    print(f"    Step 2: Clustering peptides...")
    unique_peptides = df['peptide'].dropna().unique().tolist()
    peptide_to_cluster, peptide_clusters = cluster_peptides_mmseqs2(
        unique_peptides,
        similarity_threshold=peptide_similarity_threshold,
        coverage=0.8,
    )
    df['peptide_cluster'] = df['peptide'].map(peptide_to_cluster)

    # Step 3: Hold out MHC alleles
    # For Class I: holdout based on single mhc_one_id
    # For Class II: holdout based on mhc_one_id + mhc_two_id combination
    print(f"    Step 3: Selecting MHC alleles to hold out ({holdout_allele_fraction*100:.0f}%)...")

    if len(mhc_id_cols) == 1:
        # Class I: single allele column
        mhc_col = mhc_id_cols[0]
        all_alleles = df[mhc_col].dropna().unique()
        num_holdout_alleles = max(1, int(len(all_alleles) * holdout_allele_fraction))

        # Prefer alleles with moderate representation
        allele_counts = df[mhc_col].value_counts()
        sorted_alleles = allele_counts.sort_values().index.tolist()
        middle_start = len(sorted_alleles) // 4
        middle_end = 3 * len(sorted_alleles) // 4
        candidate_alleles = sorted_alleles[middle_start:middle_end]

        if len(candidate_alleles) < num_holdout_alleles:
            candidate_alleles = sorted_alleles

        np.random.shuffle(candidate_alleles)
        holdout_alleles = set(candidate_alleles[:num_holdout_alleles])
        print(f"    Holding out {len(holdout_alleles):,} MHC alleles")

        test_unseen_allele_df = df[df[mhc_col].isin(holdout_alleles)].copy()
        remaining_df = df[~df[mhc_col].isin(holdout_alleles)].copy()
    else:
        # Class II: combine both allele columns for holdout
        # Create combined allele key
        df['_mhc_combined'] = df[mhc_id_cols[0]].astype(str) + '_' + df[mhc_id_cols[1]].astype(str)
        all_combos = df['_mhc_combined'].dropna().unique()
        num_holdout_combos = max(1, int(len(all_combos) * holdout_allele_fraction))

        # Prefer combinations with moderate representation
        combo_counts = df['_mhc_combined'].value_counts()
        sorted_combos = combo_counts.sort_values().index.tolist()
        middle_start = len(sorted_combos) // 4
        middle_end = 3 * len(sorted_combos) // 4
        candidate_combos = sorted_combos[middle_start:middle_end]

        if len(candidate_combos) < num_holdout_combos:
            candidate_combos = sorted_combos

        np.random.shuffle(candidate_combos)
        holdout_alleles = set(candidate_combos[:num_holdout_combos])
        print(f"    Holding out {len(holdout_alleles):,} MHC allele combinations (Class II)")

        test_unseen_allele_df = df[df['_mhc_combined'].isin(holdout_alleles)].copy()
        remaining_df = df[~df['_mhc_combined'].isin(holdout_alleles)].copy()

        # Clean up temporary column
        test_unseen_allele_df = test_unseen_allele_df.drop(columns=['_mhc_combined'])
        remaining_df = remaining_df.drop(columns=['_mhc_combined'])
    print(f"    test_unseen_allele: {len(test_unseen_allele_df):,} samples")
    print(f"    Remaining: {len(remaining_df):,} samples")

    # Step 4: Hold out peptide clusters
    print(f"    Step 4: Selecting peptide clusters to hold out ({holdout_epitope_fraction*100:.0f}%)...")
    remaining_peptide_clusters = remaining_df['peptide_cluster'].dropna().unique()
    num_holdout_peptide = max(1, int(len(remaining_peptide_clusters) * holdout_epitope_fraction))

    # Shuffle and select
    remaining_peptide_clusters = list(remaining_peptide_clusters)
    np.random.shuffle(remaining_peptide_clusters)
    holdout_peptide_clusters = set(remaining_peptide_clusters[:num_holdout_peptide])
    print(f"    Holding out {len(holdout_peptide_clusters):,} peptide clusters")

    test_unseen_epitope_df = remaining_df[remaining_df['peptide_cluster'].isin(holdout_peptide_clusters)].copy()
    remaining_df = remaining_df[~remaining_df['peptide_cluster'].isin(holdout_peptide_clusters)].copy()
    print(f"    test_unseen_epitope: {len(test_unseen_epitope_df):,} samples")
    print(f"    Remaining: {len(remaining_df):,} samples")

    # Step 5: Split by TCR clusters
    print(f"    Step 5: Splitting by TCR clusters...")
    remaining_tcr_clusters = remaining_df['tcr_cluster'].unique()
    remaining_tcr_clusters = list(remaining_tcr_clusters)
    np.random.shuffle(remaining_tcr_clusters)

    # Calculate split sizes
    val_test_ratio = 1.0 - train_ratio
    val_ratio = val_test_ratio / 2
    test_ratio = val_test_ratio / 2

    num_train_clusters = int(len(remaining_tcr_clusters) * train_ratio)
    num_val_clusters = int(len(remaining_tcr_clusters) * val_ratio)

    train_tcr_clusters = set(remaining_tcr_clusters[:num_train_clusters])
    val_tcr_clusters = set(remaining_tcr_clusters[num_train_clusters:num_train_clusters + num_val_clusters])
    test_tcr_clusters = set(remaining_tcr_clusters[num_train_clusters + num_val_clusters:])

    print(f"    TCR cluster split: train={len(train_tcr_clusters):,}, val={len(val_tcr_clusters):,}, test={len(test_tcr_clusters):,}")

    # Create base splits
    train_df = remaining_df[remaining_df['tcr_cluster'].isin(train_tcr_clusters)].copy()
    val_df = remaining_df[remaining_df['tcr_cluster'].isin(val_tcr_clusters)].copy()

    # Step 6: Create test_unseen_tcr_seen_epitope
    # TCR clusters not in train, but peptide clusters overlap with train
    print(f"    Step 6: Creating test_unseen_tcr_seen_epitope...")
    train_peptide_clusters = set(train_df['peptide_cluster'].unique())

    test_unseen_tcr_seen_epitope_df = remaining_df[
        (remaining_df['tcr_cluster'].isin(test_tcr_clusters)) &
        (remaining_df['peptide_cluster'].isin(train_peptide_clusters))
    ].copy()
    print(f"    test_unseen_tcr_seen_epitope: {len(test_unseen_tcr_seen_epitope_df):,} samples")

    # Step 7: Create test_seen_epitope (baseline)
    # Sample from validation to create baseline test set with overlap
    print(f"    Step 7: Creating test_seen_epitope (baseline)...")
    if len(val_df) > 0:
        sample_size = min(len(val_df) // 3, len(test_unseen_epitope_df) if len(test_unseen_epitope_df) > 0 else 100)
        sample_size = max(sample_size, 1)
        test_seen_epitope_df = val_df.sample(n=min(sample_size, len(val_df)), random_state=seed).copy()
    else:
        test_seen_epitope_df = pd.DataFrame()
    print(f"    test_seen_epitope: {len(test_seen_epitope_df):,} samples")

    # Print final statistics
    print(f"\n    Hierarchical split statistics:")
    print(f"      train: {len(train_df):,} samples, {train_df['peptide'].nunique():,} unique peptides")
    print(f"      val: {len(val_df):,} samples, {val_df['peptide'].nunique() if len(val_df) > 0 else 0:,} unique peptides")
    print(f"      test_seen_epitope: {len(test_seen_epitope_df):,} samples")
    print(f"      test_unseen_tcr_seen_epitope: {len(test_unseen_tcr_seen_epitope_df):,} samples")
    print(f"      test_unseen_epitope: {len(test_unseen_epitope_df):,} samples")
    print(f"      test_unseen_allele: {len(test_unseen_allele_df):,} samples")

    return {
        'train': train_df,
        'val': val_df,
        'test_seen_epitope': test_seen_epitope_df,
        'test_unseen_tcr_seen_epitope': test_unseen_tcr_seen_epitope_df,
        'test_unseen_epitope': test_unseen_epitope_df,
        'test_unseen_allele': test_unseen_allele_df,
        '_tcr_to_cluster': tcr_to_cluster,
        '_tcr_clusters': tcr_clusters,
        '_peptide_to_cluster': peptide_to_cluster,
        '_peptide_clusters': peptide_clusters,
        '_holdout_alleles': holdout_alleles,
        '_holdout_peptide_clusters': holdout_peptide_clusters,
        '_train_tcr_clusters': train_tcr_clusters,
        '_train_peptide_clusters': train_peptide_clusters,
    }


def verify_hierarchical_splits(splits: Dict, mhc_id_cols: List[str] = None) -> bool:
    """
    Verify that hierarchical splits have expected properties.

    Args:
        splits: Dict containing split DataFrames
        mhc_id_cols: List of MHC ID column names (e.g., ['mhc_one_id'] for Class I,
                     ['mhc_one_id', 'mhc_two_id'] for Class II)

    Returns:
        True if verification passes, False otherwise
    """
    if mhc_id_cols is None:
        mhc_id_cols = ['mhc_one_id']

    train_df = splits['train']
    if len(train_df) == 0:
        return True

    train_peptides = set(train_df['peptide'].unique())
    train_tcr_clusters = set(train_df['tcr_cluster'].unique()) if 'tcr_cluster' in train_df.columns else set()
    train_peptide_clusters = set(train_df['peptide_cluster'].unique()) if 'peptide_cluster' in train_df.columns else set()

    # Get train alleles based on Class I (single column) or Class II (combined columns)
    if len(mhc_id_cols) == 1:
        mhc_col = mhc_id_cols[0]
        train_alleles = set(train_df[mhc_col].dropna().unique()) if mhc_col in train_df.columns else set()
    else:
        # For Class II, combine the allele columns
        if all(col in train_df.columns for col in mhc_id_cols):
            train_alleles = set(
                train_df[mhc_id_cols[0]].astype(str) + '_' + train_df[mhc_id_cols[1]].astype(str)
            )
        else:
            train_alleles = set()

    errors = []

    # Check test_unseen_allele: no MHC allele overlap with train
    test_unseen_allele = splits.get('test_unseen_allele', pd.DataFrame())
    if len(test_unseen_allele) > 0:
        if len(mhc_id_cols) == 1:
            mhc_col = mhc_id_cols[0]
            if mhc_col in test_unseen_allele.columns:
                test_alleles = set(test_unseen_allele[mhc_col].dropna().unique())
                overlap = test_alleles & train_alleles
                if overlap:
                    errors.append(f"test_unseen_allele has {len(overlap)} alleles overlapping with train")
        else:
            # Class II: check combined allele overlap
            if all(col in test_unseen_allele.columns for col in mhc_id_cols):
                test_alleles = set(
                    test_unseen_allele[mhc_id_cols[0]].astype(str) + '_' + test_unseen_allele[mhc_id_cols[1]].astype(str)
                )
                overlap = test_alleles & train_alleles
                if overlap:
                    errors.append(f"test_unseen_allele has {len(overlap)} allele combinations overlapping with train")

    # Check test_unseen_epitope: no peptide cluster overlap with train
    test_unseen_epitope = splits.get('test_unseen_epitope', pd.DataFrame())
    if len(test_unseen_epitope) > 0 and 'peptide_cluster' in test_unseen_epitope.columns:
        test_clusters = set(test_unseen_epitope['peptide_cluster'].unique())
        overlap = test_clusters & train_peptide_clusters
        if overlap:
            errors.append(f"test_unseen_epitope has {len(overlap)} peptide clusters overlapping with train")

    # Check test_unseen_tcr_seen_epitope: no TCR cluster overlap with train
    test_unseen_tcr = splits.get('test_unseen_tcr_seen_epitope', pd.DataFrame())
    if len(test_unseen_tcr) > 0 and 'tcr_cluster' in test_unseen_tcr.columns:
        test_tcr_clusters = set(test_unseen_tcr['tcr_cluster'].unique())
        overlap = test_tcr_clusters & train_tcr_clusters
        if overlap:
            errors.append(f"test_unseen_tcr_seen_epitope has {len(overlap)} TCR clusters overlapping with train")

    # Check val: no TCR cluster overlap with train (peptide clusters may overlap, which is expected)
    val_df = splits.get('val', pd.DataFrame())
    if len(val_df) > 0 and 'tcr_cluster' in val_df.columns:
        val_tcr_clusters = set(val_df['tcr_cluster'].unique())
        overlap = val_tcr_clusters & train_tcr_clusters
        if overlap:
            errors.append(f"val has {len(overlap)} TCR clusters overlapping with train")

    if errors:
        for error in errors:
            print(f"    ERROR: {error}")
        return False

    print(f"    Verified: Hierarchical splits have expected properties")
    return True


def save_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """Save DataFrame to parquet file."""
    if len(df) == 0:
        print(f"    Skipping {output_path} (empty)")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove internal cluster columns before saving
    output_df = df.drop(columns=['tcr_cluster', 'peptide_cluster'], errors='ignore')

    table = pa.Table.from_pandas(output_df, preserve_index=False)
    pq.write_table(table, output_path)
    print(f"    Saved {len(df):,} rows to {output_path}")


def save_tcr_clusters(
    output_dir: Path,
    tcr_to_cluster: Dict[str, int],
    tcr_clusters: Dict[int, List[str]],
) -> None:
    """Save TCR CDR3 cluster assignments to TSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_path = output_dir / "tcr_clusters.tsv"

    with open(cluster_path, 'w') as f:
        f.write("cdr3\tcluster_id\tcluster_size\tis_representative\n")
        for cluster_id, cdr3s in sorted(tcr_clusters.items()):
            for i, cdr3 in enumerate(cdr3s):
                is_rep = "true" if i == 0 else "false"
                f.write(f"{cdr3}\t{cluster_id}\t{len(cdr3s)}\t{is_rep}\n")

    print(f"    Saved TCR clusters to {cluster_path}")


def save_peptide_clusters(
    output_dir: Path,
    peptide_to_cluster: Dict[str, int],
    peptide_clusters: Dict[int, List[str]],
) -> None:
    """Save peptide cluster assignments to TSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_path = output_dir / "peptide_clusters.tsv"

    with open(cluster_path, 'w') as f:
        f.write("peptide\tcluster_id\tcluster_size\tis_representative\n")
        for cluster_id, peptides in sorted(peptide_clusters.items()):
            for i, peptide in enumerate(peptides):
                is_rep = "true" if i == 0 else "false"
                f.write(f"{peptide}\t{cluster_id}\t{len(peptides)}\t{is_rep}\n")

    print(f"    Saved peptide clusters to {cluster_path}")


def save_split_assignments(
    output_dir: Path,
    perm_key: str,
    splits: Dict[str, pd.DataFrame],
    mhc_id_cols: List[str],
) -> None:
    """Save split assignments to TSV file for analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = output_dir / "split_assignments.tsv"

    all_rows = []
    split_names = ['train', 'val', 'test_seen_epitope', 'test_unseen_tcr_seen_epitope',
                   'test_unseen_epitope', 'test_unseen_allele']

    for split_name in split_names:
        df = splits.get(split_name, pd.DataFrame())
        if len(df) == 0:
            continue
        df_copy = df.copy()
        df_copy['split'] = split_name
        all_rows.append(df_copy)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        output_cols = ['peptide', 'split']
        if 'tcr_cluster' in combined.columns:
            output_cols.append('tcr_cluster')
        if 'peptide_cluster' in combined.columns:
            output_cols.append('peptide_cluster')
        # Add all MHC ID columns
        for mhc_col in mhc_id_cols:
            if mhc_col in combined.columns:
                output_cols.append(mhc_col)
        if 'tra' in combined.columns:
            output_cols.append('tra')
        if 'trb' in combined.columns:
            output_cols.append('trb')

        combined[output_cols].to_csv(assignment_path, sep='\t', index=False)
        print(f"    Saved split assignments to {assignment_path}")


def collect_hierarchical_statistics(
    perm_key: str,
    config: Dict,
    splits: Dict[str, pd.DataFrame],
    tcr_to_cluster: Dict[str, int],
    tcr_clusters: Dict[int, List[str]],
    peptide_to_cluster: Dict[str, int],
    peptide_clusters: Dict[int, List[str]],
    mhc_id_cols: List[str],
) -> Dict:
    """Collect detailed statistics for hierarchical splits."""
    stats = {
        'permutation_key': perm_key,
        'description': config['description'],
        'tcr_cols': config['tcr_cols'],
        'mhc_id_cols': mhc_id_cols,
        'hierarchical': True,
        'tcr_clustering': {
            'num_clusters': len(tcr_clusters),
            'num_sequences': len(tcr_to_cluster),
            'cluster_size_distribution': {},
        },
        'peptide_clustering': {
            'num_clusters': len(peptide_clusters),
            'num_sequences': len(peptide_to_cluster),
            'cluster_size_distribution': {},
        },
        'splits': {},
    }

    # TCR cluster size distribution
    tcr_cluster_sizes = [len(seqs) for seqs in tcr_clusters.values()]
    if tcr_cluster_sizes:
        stats['tcr_clustering']['cluster_size_distribution'] = {
            'min': min(tcr_cluster_sizes),
            'max': max(tcr_cluster_sizes),
            'mean': np.mean(tcr_cluster_sizes),
            'median': np.median(tcr_cluster_sizes),
            'singletons': sum(1 for s in tcr_cluster_sizes if s == 1),
        }

    # Peptide cluster size distribution
    peptide_cluster_sizes = [len(seqs) for seqs in peptide_clusters.values()]
    if peptide_cluster_sizes:
        stats['peptide_clustering']['cluster_size_distribution'] = {
            'min': min(peptide_cluster_sizes),
            'max': max(peptide_cluster_sizes),
            'mean': np.mean(peptide_cluster_sizes),
            'median': np.median(peptide_cluster_sizes),
            'singletons': sum(1 for s in peptide_cluster_sizes if s == 1),
        }

    # Get train info for overlap analysis
    train_df = splits.get('train', pd.DataFrame())
    train_peptides = set(train_df['peptide'].unique()) if len(train_df) > 0 else set()
    train_tcr_clusters = set(train_df['tcr_cluster'].unique()) if len(train_df) > 0 and 'tcr_cluster' in train_df.columns else set()
    train_peptide_clusters = set(train_df['peptide_cluster'].unique()) if len(train_df) > 0 and 'peptide_cluster' in train_df.columns else set()

    # Get train alleles based on Class I (single column) or Class II (combined columns)
    if len(mhc_id_cols) == 1:
        mhc_col = mhc_id_cols[0]
        train_alleles = set(train_df[mhc_col].dropna().unique()) if len(train_df) > 0 and mhc_col in train_df.columns else set()
    else:
        # For Class II, combine the allele columns
        if len(train_df) > 0 and all(col in train_df.columns for col in mhc_id_cols):
            train_alleles = set(
                train_df[mhc_id_cols[0]].astype(str) + '_' + train_df[mhc_id_cols[1]].astype(str)
            )
        else:
            train_alleles = set()

    split_names = ['train', 'val', 'test_seen_epitope', 'test_unseen_tcr_seen_epitope',
                   'test_unseen_epitope', 'test_unseen_allele']

    for split_name in split_names:
        df = splits.get(split_name, pd.DataFrame())
        split_stats = {
            'total_samples': len(df),
            'unique_peptides': df['peptide'].nunique() if len(df) > 0 else 0,
        }

        if len(df) > 0:
            # Peptide length distribution
            peptide_lengths = df['peptide'].str.len()
            split_stats['peptide_length'] = {
                'min': int(peptide_lengths.min()),
                'max': int(peptide_lengths.max()),
                'mean': float(peptide_lengths.mean()),
                'median': float(peptide_lengths.median()),
            }

            # TCR info
            for tcr_col in config['tcr_cols']:
                if tcr_col in df.columns:
                    unique_tcrs = df[tcr_col].nunique()
                    split_stats[f'unique_{tcr_col}'] = unique_tcrs

            # Cluster info
            if 'tcr_cluster' in df.columns:
                unique_tcr_clusters = df['tcr_cluster'].unique()
                split_stats['unique_tcr_clusters'] = len(unique_tcr_clusters)

                if split_name != 'train':
                    cluster_overlap = set(unique_tcr_clusters) & train_tcr_clusters
                    split_stats['tcr_cluster_overlap_with_train'] = len(cluster_overlap)
                    split_stats['tcr_cluster_overlap_pct'] = (
                        100 * len(cluster_overlap) / len(unique_tcr_clusters)
                        if len(unique_tcr_clusters) > 0 else 0
                    )

            if 'peptide_cluster' in df.columns:
                unique_pep_clusters = df['peptide_cluster'].unique()
                split_stats['unique_peptide_clusters'] = len(unique_pep_clusters)

                if split_name != 'train':
                    cluster_overlap = set(unique_pep_clusters) & train_peptide_clusters
                    split_stats['peptide_cluster_overlap_with_train'] = len(cluster_overlap)
                    split_stats['peptide_cluster_overlap_pct'] = (
                        100 * len(cluster_overlap) / len(unique_pep_clusters)
                        if len(unique_pep_clusters) > 0 else 0
                    )

            # Peptide overlap with train
            if split_name != 'train':
                split_peptides = set(df['peptide'].unique())
                peptide_overlap = split_peptides & train_peptides
                split_stats['peptide_overlap_with_train'] = len(peptide_overlap)
                split_stats['peptide_overlap_pct'] = (
                    100 * len(peptide_overlap) / len(split_peptides)
                    if len(split_peptides) > 0 else 0
                )

            # MHC allele distributions
            if len(mhc_id_cols) == 1:
                # Class I: single allele column
                mhc_col = mhc_id_cols[0]
                if mhc_col in df.columns:
                    mhc_counts = df[mhc_col].dropna().value_counts()
                    split_stats['mhc_alleles'] = {
                        'unique_count': len(mhc_counts),
                        'top_10': dict(mhc_counts.head(10)),
                    }

                    if split_name != 'train':
                        split_alleles = set(df[mhc_col].dropna().unique())
                        allele_overlap = split_alleles & train_alleles
                        split_stats['allele_overlap_with_train'] = len(allele_overlap)
                        split_stats['allele_overlap_pct'] = (
                            100 * len(allele_overlap) / len(split_alleles)
                            if len(split_alleles) > 0 else 0
                        )
            else:
                # Class II: combined allele columns
                if all(col in df.columns for col in mhc_id_cols):
                    combined_alleles = df[mhc_id_cols[0]].astype(str) + '_' + df[mhc_id_cols[1]].astype(str)
                    mhc_counts = combined_alleles.value_counts()
                    split_stats['mhc_alleles'] = {
                        'unique_count': len(mhc_counts),
                        'top_10': dict(mhc_counts.head(10)),
                    }
                    # Also add individual allele stats
                    for mhc_col in mhc_id_cols:
                        col_counts = df[mhc_col].dropna().value_counts()
                        split_stats[f'{mhc_col}_unique_count'] = len(col_counts)

                    if split_name != 'train':
                        split_alleles = set(combined_alleles)
                        allele_overlap = split_alleles & train_alleles
                        split_stats['allele_overlap_with_train'] = len(allele_overlap)
                        split_stats['allele_overlap_pct'] = (
                            100 * len(allele_overlap) / len(split_alleles)
                            if len(split_alleles) > 0 else 0
                        )

            # Source distribution
            if 'source' in df.columns:
                source_counts = df['source'].value_counts()
                split_stats['sources'] = dict(source_counts)

        stats['splits'][split_name] = split_stats

    return stats


def write_statistics_doc(
    output_dir: Path,
    all_detailed_stats: Dict,
    config_info: Dict,
) -> None:
    """Write dataset statistics to markdown documentation file."""
    doc_path = output_dir / "dataset_statistics.md"
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# TCR Antigenic Specificity Evaluation Dataset Statistics")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Configuration section
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Input Directory | `{config_info['input_dir']}` |")
    lines.append(f"| Output Directory | `{config_info['output_dir']}` |")
    lines.append(f"| Train Ratio | {config_info['train_ratio']} |")
    lines.append(f"| Random Seed | {config_info['seed']} |")
    lines.append(f"| Peptide Length Range | {config_info['min_peptide_len']}-{config_info['max_peptide_len']} |")
    lines.append(f"| TCR Similarity Threshold | {config_info['tcr_similarity_threshold']} |")
    lines.append(f"| Peptide Similarity Threshold | {config_info['peptide_similarity_threshold']} |")
    lines.append(f"| Holdout Allele Fraction | {config_info['holdout_allele_fraction']} |")
    lines.append(f"| Holdout Epitope Fraction | {config_info['holdout_epitope_fraction']} |")
    lines.append("")

    # Test set difficulty explanation
    lines.append("## Test Set Difficulty Levels")
    lines.append("")
    lines.append("| Test Set | Description | What it Tests |")
    lines.append("|----------|-------------|---------------|")
    lines.append("| `test_seen_epitope` | Same peptides as training | Baseline - can model learn TCR→peptide mapping? |")
    lines.append("| `test_unseen_tcr_seen_epitope` | New TCR clusters, peptides seen | Can model generalize to new TCRs for known peptides? |")
    lines.append("| `test_unseen_epitope` | Peptide clusters not in training | True generalization - novel peptide prediction |")
    lines.append("| `test_unseen_allele` | MHC alleles not in training | Hardest - no MHC context shortcuts |")
    lines.append("")

    # Detailed statistics for each permutation key
    for perm_key, stats in all_detailed_stats.items():
        lines.append(f"## {perm_key}")
        lines.append("")
        lines.append(f"**Description:** {stats['description']}")
        lines.append("")
        lines.append(f"**TCR Columns:** {', '.join(stats['tcr_cols'])}")
        lines.append("")

        # TCR clustering statistics
        if 'tcr_clustering' in stats:
            tcr_clust = stats['tcr_clustering']
            lines.append("### TCR CDR3 Clustering Statistics")
            lines.append("")
            lines.append(f"- **Total CDR3 sequences:** {tcr_clust['num_sequences']:,}")
            lines.append(f"- **Total clusters:** {tcr_clust['num_clusters']:,}")
            if 'cluster_size_distribution' in tcr_clust and tcr_clust['cluster_size_distribution']:
                dist = tcr_clust['cluster_size_distribution']
                lines.append(f"- **Cluster sizes:** min={dist['min']}, max={dist['max']}, "
                           f"mean={dist['mean']:.1f}, median={dist['median']:.1f}")
                lines.append(f"- **Singleton clusters:** {dist['singletons']:,}")
            lines.append("")

        # Peptide clustering statistics
        if 'peptide_clustering' in stats:
            pep_clust = stats['peptide_clustering']
            lines.append("### Peptide Clustering Statistics")
            lines.append("")
            lines.append(f"- **Total peptides:** {pep_clust['num_sequences']:,}")
            lines.append(f"- **Total clusters:** {pep_clust['num_clusters']:,}")
            if 'cluster_size_distribution' in pep_clust and pep_clust['cluster_size_distribution']:
                dist = pep_clust['cluster_size_distribution']
                lines.append(f"- **Cluster sizes:** min={dist['min']}, max={dist['max']}, "
                           f"mean={dist['mean']:.1f}, median={dist['median']:.1f}")
                lines.append(f"- **Singleton clusters:** {dist['singletons']:,}")
            lines.append("")

        # Summary table
        lines.append("### Split Summary")
        lines.append("")
        lines.append("| Split | Samples | Peptides | TCR Clusters | Pep Clusters | Peptide Overlap | TCR Cluster Overlap | Allele Overlap |")
        lines.append("|-------|---------|----------|--------------|--------------|-----------------|---------------------|----------------|")

        split_order = ['train', 'val', 'test_seen_epitope', 'test_unseen_tcr_seen_epitope',
                       'test_unseen_epitope', 'test_unseen_allele']

        for split_name in split_order:
            if split_name not in stats['splits']:
                continue
            split_stats = stats['splits'][split_name]
            samples = split_stats['total_samples']
            peptides = split_stats.get('unique_peptides', 0)
            tcr_clusters = split_stats.get('unique_tcr_clusters', '-')
            pep_clusters = split_stats.get('unique_peptide_clusters', '-')
            pep_overlap = f"{split_stats.get('peptide_overlap_pct', 0):.1f}%" if split_name != 'train' else '-'
            tcr_overlap = f"{split_stats.get('tcr_cluster_overlap_pct', 0):.1f}%" if split_name != 'train' else '-'
            allele_overlap = f"{split_stats.get('allele_overlap_pct', 0):.1f}%" if split_name != 'train' else '-'
            lines.append(f"| {split_name} | {samples:,} | {peptides:,} | {tcr_clusters} | {pep_clusters} | {pep_overlap} | {tcr_overlap} | {allele_overlap} |")

        lines.append("")

        # Detailed per-split statistics
        for split_name in split_order:
            if split_name not in stats['splits']:
                continue
            split_stats = stats['splits'][split_name]
            lines.append(f"### {split_name}")
            lines.append("")

            if split_stats['total_samples'] == 0:
                lines.append("*No data*")
                lines.append("")
                continue

            lines.append(f"- **Total samples:** {split_stats['total_samples']:,}")
            lines.append(f"- **Unique peptides:** {split_stats['unique_peptides']:,}")

            # TCR stats
            for tcr_col in stats['tcr_cols']:
                key = f'unique_{tcr_col}'
                if key in split_stats:
                    lines.append(f"- **Unique {tcr_col}:** {split_stats[key]:,}")

            if 'unique_tcr_clusters' in split_stats:
                lines.append(f"- **Unique TCR clusters:** {split_stats['unique_tcr_clusters']:,}")

            if 'unique_peptide_clusters' in split_stats:
                lines.append(f"- **Unique peptide clusters:** {split_stats['unique_peptide_clusters']:,}")

            if 'peptide_length' in split_stats:
                pl = split_stats['peptide_length']
                lines.append(f"- **Peptide length:** min={pl['min']}, max={pl['max']}, "
                           f"mean={pl['mean']:.1f}, median={pl['median']:.1f}")

            # Overlap stats
            if split_name != 'train':
                if 'peptide_overlap_with_train' in split_stats:
                    lines.append(f"- **Peptide overlap with train:** {split_stats['peptide_overlap_with_train']:,} "
                               f"({split_stats['peptide_overlap_pct']:.1f}%)")
                if 'tcr_cluster_overlap_with_train' in split_stats:
                    lines.append(f"- **TCR cluster overlap with train:** {split_stats['tcr_cluster_overlap_with_train']:,} "
                               f"({split_stats['tcr_cluster_overlap_pct']:.1f}%)")
                if 'peptide_cluster_overlap_with_train' in split_stats:
                    lines.append(f"- **Peptide cluster overlap with train:** {split_stats['peptide_cluster_overlap_with_train']:,} "
                               f"({split_stats['peptide_cluster_overlap_pct']:.1f}%)")
                if 'allele_overlap_with_train' in split_stats:
                    lines.append(f"- **Allele overlap with train:** {split_stats['allele_overlap_with_train']:,} "
                               f"({split_stats['allele_overlap_pct']:.1f}%)")

            lines.append("")

            # MHC allele distribution
            if 'mhc_alleles' in split_stats:
                mhc_stats = split_stats['mhc_alleles']
                lines.append(f"**MHC Alleles:** {mhc_stats['unique_count']:,} unique")
                lines.append("")
                lines.append("Top 10 alleles:")
                lines.append("")
                lines.append("| Allele | Count |")
                lines.append("|--------|-------|")
                for allele, count in list(mhc_stats['top_10'].items())[:10]:
                    lines.append(f"| {allele} | {count:,} |")
                lines.append("")

            # Source distribution
            if 'sources' in split_stats:
                lines.append("**Source Distribution:**")
                lines.append("")
                lines.append("| Source | Count |")
                lines.append("|--------|-------|")
                for source, count in split_stats['sources'].items():
                    lines.append(f"| {source} | {count:,} |")
                lines.append("")

    # Verification checklist
    lines.append("## Verification Checklist")
    lines.append("")
    lines.append("After running, verify each test set:")
    lines.append("")
    lines.append("1. **test_unseen_allele**:")
    lines.append("   - No MHC allele overlap with train")
    lines.append("   - May have TCR/peptide overlap (expected)")
    lines.append("")
    lines.append("2. **test_unseen_epitope**:")
    lines.append("   - No peptide cluster overlap with train")
    lines.append("   - MHC alleles may overlap (expected)")
    lines.append("")
    lines.append("3. **test_unseen_tcr_seen_epitope**:")
    lines.append("   - No TCR cluster overlap with train")
    lines.append("   - Peptide clusters overlap with train (required)")
    lines.append("")
    lines.append("4. **test_seen_epitope**:")
    lines.append("   - Baseline reference with overlap allowed")
    lines.append("")

    # Write to file
    with open(doc_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n  Statistics documentation written to: {doc_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create TCR antigenic specificity evaluation datasets with hierarchical stratified splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/parsed_output/parser_databases",
        help="Path to parser_databases directory (default: data/parsed_output/parser_databases)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for splits"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--tcr_similarity_threshold",
        type=float,
        default=0.9,
        help="TCR CDR3 clustering threshold (default: 0.9)"
    )
    parser.add_argument(
        "--peptide_similarity_threshold",
        type=float,
        default=0.8,
        help="Peptide clustering threshold (default: 0.8)"
    )
    parser.add_argument(
        "--holdout_allele_fraction",
        type=float,
        default=0.1,
        help="Fraction of MHC alleles to hold out for test_unseen_allele (default: 0.1)"
    )
    parser.add_argument(
        "--holdout_epitope_fraction",
        type=float,
        default=0.1,
        help="Fraction of peptide clusters to hold out for test_unseen_epitope (default: 0.1)"
    )
    parser.add_argument(
        "--min_peptide_len",
        type=int,
        default=8,
        help="Minimum peptide length (default: 8)"
    )
    parser.add_argument(
        "--max_peptide_len",
        type=int,
        default=15,
        help="Maximum peptide length (default: 15)"
    )
    parser.add_argument(
        "--permutation_keys",
        type=str,
        nargs='+',
        default=list(PERMUTATION_CONFIGS.keys()),
        choices=list(PERMUTATION_CONFIGS.keys()),
        help="Permutation keys to process (default: all)"
    )
    parser.add_argument(
        "--skip_mhc_standardization",
        action="store_true",
        help="Skip MHC allele standardization using tidytcells"
    )
    parser.add_argument(
        "--skip_tcr_gene_standardization",
        action="store_true",
        help="Skip TCR gene standardization using tidytcells"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("TCR Antigenic Specificity Evaluation Dataset Creation")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Input directory: {args.input_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Train ratio: {args.train_ratio}")
    print(f"  Random seed: {args.seed}")
    print(f"  TCR similarity threshold: {args.tcr_similarity_threshold}")
    print(f"  Peptide similarity threshold: {args.peptide_similarity_threshold}")
    print(f"  Holdout allele fraction: {args.holdout_allele_fraction}")
    print(f"  Holdout epitope fraction: {args.holdout_epitope_fraction}")
    print(f"  Peptide length: {args.min_peptide_len}-{args.max_peptide_len}")
    print(f"  Permutation keys: {args.permutation_keys}")
    if args.skip_mhc_standardization:
        print(f"  MHC standardization: DISABLED")
    else:
        print(f"  MHC standardization: {'ENABLED (tidytcells)' if TIDYTCELLS_AVAILABLE else 'UNAVAILABLE'}")
    if args.skip_tcr_gene_standardization:
        print(f"  TCR gene standardization: DISABLED")
    else:
        print(f"  TCR gene standardization: {'ENABLED (tidytcells)' if TIDYTCELLS_AVAILABLE else 'UNAVAILABLE'}")

    # Set random seed
    np.random.seed(args.seed)

    # Load data
    print("\n" + "=" * 70)
    print("Step 1: Loading databases")
    print("=" * 70)
    input_dir = Path(args.input_dir)
    combined_df = load_databases(input_dir)

    # Standardize MHC allele names
    if not args.skip_mhc_standardization:
        print("\n" + "=" * 70)
        print("Step 2: Standardizing MHC allele names")
        print("=" * 70)
        mhc_id_columns = ["mhc_one_id", "mhc_two_id"]
        combined_df = standardize_mhc_columns(combined_df, mhc_id_columns)

    # Standardize TCR gene names
    if not args.skip_tcr_gene_standardization:
        print("\n" + "=" * 70)
        print("Step 3: Standardizing TCR gene names")
        print("=" * 70)
        combined_df = standardize_tcr_gene_columns(combined_df)

    # Process each permutation key
    print("\n" + "=" * 70)
    print("Step 4: Processing permutation keys")
    print("=" * 70)

    output_dir = Path(args.output_dir)
    all_stats = {}
    all_detailed_stats = {}

    for perm_key in args.permutation_keys:
        config = PERMUTATION_CONFIGS[perm_key]

        # Process this permutation type
        processed_df = process_permutation_key(
            combined_df,
            perm_key,
            config,
            args.min_peptide_len,
            args.max_peptide_len,
        )

        if len(processed_df) == 0:
            print(f"    Skipping {perm_key} (no valid data)")
            continue

        # Organize output by MHC class
        mhc_class = config.get("mhc_class", "class_one")
        perm_output_dir = output_dir / mhc_class / perm_key
        mhc_id_cols = config.get("mhc_id_cols", ["mhc_one_id"])

        # Create hierarchical splits
        print(f"\n    Creating hierarchical splits for {perm_key}...")
        splits = hierarchical_tcr_split(
            processed_df,
            train_ratio=args.train_ratio,
            seed=args.seed,
            tcr_similarity_threshold=args.tcr_similarity_threshold,
            peptide_similarity_threshold=args.peptide_similarity_threshold,
            holdout_allele_fraction=args.holdout_allele_fraction,
            holdout_epitope_fraction=args.holdout_epitope_fraction,
            tcr_cols=config['tcr_cols'],
            mhc_id_cols=mhc_id_cols,
        )

        # Verify splits
        if not verify_hierarchical_splits(splits, mhc_id_cols):
            print(f"    WARNING: Hierarchical split verification failed for {perm_key}")

        # Save cluster mappings
        tcr_to_cluster = splits.get('_tcr_to_cluster', {})
        tcr_clusters = splits.get('_tcr_clusters', {})
        peptide_to_cluster = splits.get('_peptide_to_cluster', {})
        peptide_clusters = splits.get('_peptide_clusters', {})

        if tcr_to_cluster and tcr_clusters:
            save_tcr_clusters(perm_output_dir, tcr_to_cluster, tcr_clusters)
        if peptide_to_cluster and peptide_clusters:
            save_peptide_clusters(perm_output_dir, peptide_to_cluster, peptide_clusters)

        # Save split assignments
        save_split_assignments(perm_output_dir, perm_key, splits, mhc_id_cols)

        # Save to parquet files
        split_names = ['train', 'val', 'test_seen_epitope', 'test_unseen_tcr_seen_epitope',
                      'test_unseen_epitope', 'test_unseen_allele']
        for split_name in split_names:
            split_df = splits.get(split_name, pd.DataFrame())
            if len(split_df) > 0:
                save_parquet(split_df, perm_output_dir / f"{split_name}.parquet")

        # Store statistics
        all_stats[perm_key] = {
            split_name: len(splits.get(split_name, pd.DataFrame()))
            for split_name in split_names
        }

        # Collect detailed statistics
        all_detailed_stats[perm_key] = collect_hierarchical_statistics(
            perm_key, config, splits,
            tcr_to_cluster, tcr_clusters,
            peptide_to_cluster, peptide_clusters,
            mhc_id_cols,
        )

    # Write statistics documentation
    if all_detailed_stats:
        config_info = {
            'input_dir': args.input_dir,
            'output_dir': args.output_dir,
            'train_ratio': args.train_ratio,
            'seed': args.seed,
            'min_peptide_len': args.min_peptide_len,
            'max_peptide_len': args.max_peptide_len,
            'tcr_similarity_threshold': args.tcr_similarity_threshold,
            'peptide_similarity_threshold': args.peptide_similarity_threshold,
            'holdout_allele_fraction': args.holdout_allele_fraction,
            'holdout_epitope_fraction': args.holdout_epitope_fraction,
        }
        write_statistics_doc(output_dir, all_detailed_stats, config_info)

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"\nOutput directory: {output_dir}")
    for perm_key, stats in all_stats.items():
        total = sum(stats.values())
        print(f"\n  {perm_key}:")
        print(f"    Total: {total:,}")
        print(f"    Splits:")
        for split_name, count in stats.items():
            print(f"      {split_name}: {count:,}")

    print("\nDone!")


if __name__ == "__main__":
    main()
