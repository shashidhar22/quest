#!/usr/bin/env python3
"""
Create peptide-MHC evaluation datasets with train/val/test splits.

Extracts peptide-MHC binding data from parsed databases (IEDB, VDJDB, McPAS),
de-duplicates based on (peptide, MHC ID), and creates train/val/test splits
for evaluating peptide-MHC prediction tools.

Supports two splitting modes:
1. Simple peptide-level split (default): Ensures no peptide appears in multiple splits
2. Hierarchical stratified split: Creates multiple test sets of varying difficulty

Output Structure (Simple Split):
    output_dir/
    ├── dataset_statistics.md
    ├── class_one/
    │   └── peptide_mhc_one/           # Class I: peptide + single MHC chain
    │       ├── train.parquet
    │       ├── val.parquet
    │       └── test.parquet
    └── class_two/
        ├── peptide_mhc_one_class_two/ # Class II (alpha only): peptide + MHC alpha chain
        │   └── ...
        ├── peptide_mhc_two/           # Class II (beta only): peptide + MHC beta chain
        │   └── ...
        └── peptide_mhc_one_mhc_two/   # Class II (alpha+beta): peptide + both MHC chains
            └── ...

Output Structure (Hierarchical Split):
    output_dir/
    ├── dataset_statistics.md
    ├── class_one/
    │   └── peptide_mhc_one/
    │       ├── peptide_clusters.tsv              # Peptide -> cluster mapping
    │       ├── split_assignments.tsv             # Full assignment table
    │       ├── train.parquet
    │       ├── val.parquet
    │       ├── test_seen_motif.parquet           # Same binding motifs as training
    │       ├── test_unseen_peptide_seen_motif.parquet  # New peptides, but motifs seen
    │       ├── test_unseen_motif.parquet         # Binding motifs not in training
    │       └── test_unseen_allele.parquet        # MHC alleles not in training
    └── class_two/
        └── ...

Test Set Difficulty Levels (Hierarchical Split):
    | Test Set                      | Description                    | Expected Performance |
    |-------------------------------|--------------------------------|---------------------|
    | test_seen_motif               | Same binding motifs as train   | High (baseline)     |
    | test_unseen_peptide_seen_motif| New peptides, motifs seen      | Medium-high         |
    | test_unseen_motif             | Binding motifs not in train    | Lower (generalization)|
    | test_unseen_allele            | MHC alleles not in training    | Hardest             |

Usage:
    # Simple split (backward compatible)
    python scripts/data_processing/create_pmhc_eval_dataset.py \\
        --output_dir data/eval/pmhc

    # Hierarchical split with multiple difficulty levels
    python scripts/data_processing/create_pmhc_eval_dataset.py \\
        --output_dir data/eval/pmhc \\
        --use_hierarchical_split \\
        --similarity_threshold 0.8 \\
        --holdout_allele_fraction 0.1

Dependencies:
    - tidytcells: Optional, for MHC allele standardization (pip install tidytcells)
    - MMseqs2: Required for hierarchical split (conda install -c bioconda mmseqs2)

Author: Shashidhar Ravishankar, Claude
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Try to import tidytcells for MHC standardization
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

# Permutation key configurations
PERMUTATION_CONFIGS = {
    "peptide_mhc_one": {
        "description": "Class I: peptide + single MHC chain (alpha)",
        "mhc_class": "class_one",
        "required_cols": ["peptide", "mhc_one_id", "mhc_one"],
        "dedup_cols": ["peptide", "mhc_one_id"],
        "sequence_cols": ["peptide", "mhc_one"],
        "output_cols": ["peptide", "mhc_one_id", "mhc_one", "sequence", "permutation_key", "source"],
    },
    "peptide_mhc_one_class_two": {
        "description": "Class II (alpha only): peptide + MHC alpha chain",
        "mhc_class": "class_two",
        "required_cols": ["peptide", "mhc_one_id", "mhc_one"],
        "dedup_cols": ["peptide", "mhc_one_id"],
        "sequence_cols": ["peptide", "mhc_one"],
        "output_cols": ["peptide", "mhc_one_id", "mhc_one", "sequence", "permutation_key", "source"],
    },
    "peptide_mhc_two": {
        "description": "Class II (beta only): peptide + MHC beta chain",
        "mhc_class": "class_two",
        "required_cols": ["peptide", "mhc_two_id", "mhc_two"],
        "dedup_cols": ["peptide", "mhc_two_id"],
        "sequence_cols": ["peptide", "mhc_two"],
        "output_cols": ["peptide", "mhc_two_id", "mhc_two", "sequence", "permutation_key", "source"],
    },
    "peptide_mhc_one_mhc_two": {
        "description": "Class II (alpha+beta): peptide + both MHC chains",
        "mhc_class": "class_two",
        "required_cols": ["peptide", "mhc_one_id", "mhc_two_id", "mhc_one", "mhc_two"],
        "dedup_cols": ["peptide", "mhc_one_id", "mhc_two_id"],
        "sequence_cols": ["peptide", "mhc_one", "mhc_two"],
        "output_cols": ["peptide", "mhc_one_id", "mhc_two_id", "mhc_one", "mhc_two", "sequence", "permutation_key", "source"],
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
        # Try to standardize with tidytcells
        standardized = tt.mh.standardize(
            allele_str,
            species=species,
            precision="allele",  # Try to get allele-level precision
            on_fail="keep",  # Keep original if standardization fails
            suppress_warnings=True,
        )
        return standardized if standardized else allele_str
    except Exception:
        # If any error occurs, return the original
        return allele_str


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

        # Apply standardization with progress bar
        tqdm.pandas(desc=f"      Processing {col}", leave=False)
        df[col] = df[col].progress_apply(standardize_mhc_allele)

        # Count how many are now valid
        new_valid = df[col].apply(is_valid_value).sum()
        print(f"      Valid values: {original_valid:,} -> {new_valid:,} (recovered {new_valid - original_valid:,})")

    return df


def cluster_peptides_mmseqs2(
    peptides: List[str],
    similarity_threshold: float = 0.8,
    coverage: float = 0.8,
    temp_dir: Optional[Path] = None,
) -> Tuple[Dict[str, int], Dict[int, List[str]]]:
    """
    Cluster peptides using MMseqs2.

    Args:
        peptides: List of peptide sequences to cluster
        similarity_threshold: Sequence identity threshold for clustering (default: 0.8)
        coverage: Alignment coverage threshold (default: 0.8)
        temp_dir: Optional temporary directory for MMseqs2 files

    Returns:
        peptide_to_cluster: Dict mapping peptide -> cluster_id
        cluster_to_peptides: Dict mapping cluster_id -> list of peptides
    """
    # Check if MMseqs2 is available
    if shutil.which("mmseqs") is None:
        raise RuntimeError(
            "MMseqs2 not found in PATH. Install with: conda install -c bioconda mmseqs2"
        )

    # Filter out invalid peptides
    valid_peptides = [p for p in peptides if p and isinstance(p, str) and len(p) > 0]

    if len(valid_peptides) == 0:
        return {}, {}

    # Create temporary directory for MMseqs2 files
    cleanup_temp = False
    if temp_dir is None:
        temp_dir = Path(tempfile.mkdtemp(prefix="mmseqs2_"))
        cleanup_temp = True
    else:
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Write sequences to FASTA file
        fasta_path = temp_dir / "peptides.fasta"
        with open(fasta_path, "w") as f:
            for i, peptide in enumerate(valid_peptides):
                f.write(f">seq_{i}|{peptide}\n{peptide}\n")

        # MMseqs2 database and output paths
        db_path = temp_dir / "peptide_db"
        cluster_db = temp_dir / "cluster_db"
        cluster_tsv = temp_dir / "clusters.tsv"
        tmp_path = temp_dir / "tmp"
        tmp_path.mkdir(exist_ok=True)

        # Create MMseqs2 database
        cmd_createdb = ["mmseqs", "createdb", str(fasta_path), str(db_path)]
        subprocess.run(cmd_createdb, check=True, capture_output=True)

        # Run clustering
        # --min-seq-id: minimum sequence identity
        # -c: coverage threshold
        # --cov-mode: coverage mode (0=bidirectional, 2=target coverage)
        cmd_cluster = [
            "mmseqs", "cluster",
            str(db_path), str(cluster_db), str(tmp_path),
            "--min-seq-id", str(similarity_threshold),
            "-c", str(coverage),
            "--cov-mode", "0",
            "-s", "7.5",  # sensitivity
        ]
        subprocess.run(cmd_cluster, check=True, capture_output=True)

        # Convert cluster results to TSV
        cmd_tsv = [
            "mmseqs", "createtsv",
            str(db_path), str(db_path), str(cluster_db), str(cluster_tsv),
        ]
        subprocess.run(cmd_tsv, check=True, capture_output=True)

        # Parse cluster results
        peptide_to_cluster: Dict[str, int] = {}
        cluster_to_peptides: Dict[int, List[str]] = {}

        # Read cluster TSV: format is "representative\tmember"
        cluster_id_map: Dict[str, int] = {}  # Map representative seq to cluster ID
        next_cluster_id = 0

        with open(cluster_tsv, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    rep_header, member_header = parts[0], parts[1]
                    # Extract peptide sequence from header (format: seq_N|PEPTIDE)
                    rep_peptide = rep_header.split("|")[-1] if "|" in rep_header else rep_header
                    member_peptide = member_header.split("|")[-1] if "|" in member_header else member_header

                    # Assign cluster ID
                    if rep_peptide not in cluster_id_map:
                        cluster_id_map[rep_peptide] = next_cluster_id
                        next_cluster_id += 1

                    cluster_id = cluster_id_map[rep_peptide]
                    peptide_to_cluster[member_peptide] = cluster_id

                    if cluster_id not in cluster_to_peptides:
                        cluster_to_peptides[cluster_id] = []
                    if member_peptide not in cluster_to_peptides[cluster_id]:
                        cluster_to_peptides[cluster_id].append(member_peptide)

        # Handle any peptides that might not appear in the output
        for peptide in valid_peptides:
            if peptide not in peptide_to_cluster:
                # Assign to its own cluster
                peptide_to_cluster[peptide] = next_cluster_id
                cluster_to_peptides[next_cluster_id] = [peptide]
                next_cluster_id += 1

        return peptide_to_cluster, cluster_to_peptides

    finally:
        # Cleanup temporary directory
        if cleanup_temp and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def load_databases(input_dir: Path) -> pd.DataFrame:
    """
    Load and combine data from parser databases.

    Args:
        input_dir: Path to parser_databases directory

    Returns:
        Combined DataFrame with all databases
    """
    # Database files to load (relative to input_dir)
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

            # Standardize column names (some databases may have different column names)
            # McPAS uses 'mhc_restriction' instead of separate mhc_one_id/mhc_two_id
            if 'mhc_restriction' in df.columns and 'mhc_one_id' not in df.columns:
                # For McPAS, mhc_restriction contains the MHC allele ID
                # We'll treat it as mhc_one_id for Class I entries
                df['mhc_one_id'] = df['mhc_restriction']

            # Ensure required columns exist
            for col in ['peptide', 'mhc_one_id', 'mhc_one', 'mhc_two_id', 'mhc_two', 'source']:
                if col not in df.columns:
                    df[col] = None

            print(f"    Loaded {len(df):,} rows from {db_file}")
            dfs.append(df)
        else:
            print(f"  Warning: {db_file} not found at {db_path}")

    if not dfs:
        raise ValueError(f"No database files found in {input_dir}")

    # Combine all dataframes
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total combined: {len(combined):,} rows")

    return combined


def filter_by_peptide_length(df: pd.DataFrame, min_len: int, max_len: int) -> pd.DataFrame:
    """Filter dataframe by peptide length."""
    # Ensure peptide is string
    df = df.copy()
    df['peptide'] = df['peptide'].astype(str)

    # Calculate peptide lengths
    peptide_lengths = df['peptide'].str.len()

    # Filter by length
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
        perm_key: Permutation key name (e.g., 'peptide_mhc_one')
        config: Configuration dict for this permutation type
        min_peptide_len: Minimum peptide length
        max_peptide_len: Maximum peptide length

    Returns:
        Processed and deduplicated DataFrame
    """
    print(f"\n  Processing {perm_key}: {config['description']}")

    # Start with a copy
    result = df.copy()

    # Filter by peptide length first
    result = filter_by_peptide_length(result, min_peptide_len, max_peptide_len)

    # Filter rows with valid required columns
    required_cols = config["required_cols"]
    print(f"    Required columns: {required_cols}")

    for col in required_cols:
        if col not in result.columns:
            print(f"    Warning: Column {col} not found, skipping this permutation type")
            return pd.DataFrame()

        # Filter for valid values
        if col == 'peptide' or col.startswith('mhc_one') or col.startswith('mhc_two'):
            # For sequence columns, check if valid sequence
            if col in ['peptide', 'mhc_one', 'mhc_two']:
                valid_mask = result[col].apply(is_valid_sequence)
            else:
                # For ID columns, just check non-null
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

    if "mhc_one_id" in config["required_cols"]:
        # Check if mhc_one_id matches expected class
        before = len(result)
        result['_mhc_one_class'] = result['mhc_one_id'].apply(get_mhc_class)
        result = result[result['_mhc_one_class'] == mhc_class]
        result = result.drop(columns=['_mhc_one_class'])
        print(f"    After filtering mhc_one_id for {mhc_class}: {before:,} -> {len(result):,}")

    if "mhc_two_id" in config["required_cols"]:
        # mhc_two_id should always be Class II beta chain
        before = len(result)
        result['_mhc_two_class'] = result['mhc_two_id'].apply(get_mhc_class)
        result = result[result['_mhc_two_class'] == 'class_two']
        result = result.drop(columns=['_mhc_two_class'])
        print(f"    After filtering mhc_two_id for class_two: {before:,} -> {len(result):,}")

    if len(result) == 0:
        print(f"    No valid rows for {perm_key} after MHC class filtering")
        return pd.DataFrame()

    # Deduplicate based on dedup columns
    dedup_cols = config["dedup_cols"]
    before_dedup = len(result)
    result = result.drop_duplicates(subset=dedup_cols, keep='first')
    print(f"    After deduplication on {dedup_cols}: {before_dedup:,} -> {len(result):,}")

    # Create concatenated sequence (normalized to uppercase for consistency)
    sequence_cols = config["sequence_cols"]
    result['sequence'] = result[sequence_cols].apply(
        lambda row: " ".join(str(val).upper() for val in row if is_valid_value(val)),
        axis=1
    )

    # Add permutation key column
    result['permutation_key'] = perm_key

    # Select output columns
    output_cols = config["output_cols"]
    available_cols = [col for col in output_cols if col in result.columns]
    result = result[available_cols].copy()

    return result


def peptide_level_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data ensuring no peptide appears in multiple splits.

    This prevents data leakage where the model could see the same peptide
    in training and evaluation.

    Args:
        df: Input DataFrame with 'peptide' column
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if len(df) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Get unique peptides
    unique_peptides = df['peptide'].unique()
    print(f"    Unique peptides: {len(unique_peptides):,}")

    if len(unique_peptides) < 10:
        print(f"    Warning: Too few unique peptides ({len(unique_peptides)}) for reliable splitting")

    # Normalize ratios
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total

    # First split: train vs (val+test)
    val_test_ratio = val_ratio + test_ratio

    if len(unique_peptides) == 1:
        # If only one unique peptide, put all in training
        train_peptides = unique_peptides
        val_peptides = np.array([])
        test_peptides = np.array([])
    elif len(unique_peptides) == 2:
        # If only two unique peptides, put one in train, one in test
        train_peptides = unique_peptides[:1]
        val_peptides = np.array([])
        test_peptides = unique_peptides[1:]
    else:
        train_peptides, temp_peptides = train_test_split(
            unique_peptides,
            test_size=val_test_ratio,
            random_state=seed
        )

        # Second split: val vs test
        if len(temp_peptides) <= 1:
            val_peptides = temp_peptides
            test_peptides = np.array([])
        else:
            # Adjust ratio for second split
            test_ratio_adjusted = test_ratio / val_test_ratio
            val_peptides, test_peptides = train_test_split(
                temp_peptides,
                test_size=test_ratio_adjusted,
                random_state=seed
            )

    # Create dataframes based on peptide assignments
    train_set = set(train_peptides)
    val_set = set(val_peptides)
    test_set = set(test_peptides)

    train_df = df[df['peptide'].isin(train_set)].copy()
    val_df = df[df['peptide'].isin(val_set)].copy()
    test_df = df[df['peptide'].isin(test_set)].copy()

    print(f"    Split sizes: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")
    print(f"    Peptide counts: train={len(train_set):,}, val={len(val_set):,}, test={len(test_set):,}")

    return train_df, val_df, test_df


def verify_no_peptide_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> bool:
    """Verify that no peptide appears in multiple splits."""
    if len(train_df) == 0 and len(val_df) == 0 and len(test_df) == 0:
        return True

    train_peptides = set(train_df['peptide'].unique()) if len(train_df) > 0 else set()
    val_peptides = set(val_df['peptide'].unique()) if len(val_df) > 0 else set()
    test_peptides = set(test_df['peptide'].unique()) if len(test_df) > 0 else set()

    train_val_overlap = train_peptides & val_peptides
    train_test_overlap = train_peptides & test_peptides
    val_test_overlap = val_peptides & test_peptides

    if train_val_overlap:
        print(f"    ERROR: {len(train_val_overlap)} peptides in both train and val")
        return False
    if train_test_overlap:
        print(f"    ERROR: {len(train_test_overlap)} peptides in both train and test")
        return False
    if val_test_overlap:
        print(f"    ERROR: {len(val_test_overlap)} peptides in both val and test")
        return False

    print(f"    Verified: No peptide overlap between splits")
    return True


def hierarchical_stratified_split(
    df: pd.DataFrame,
    train_ratio: float,
    seed: int,
    similarity_threshold: float = 0.8,
    holdout_allele_fraction: float = 0.1,
    mhc_col: str = "mhc_one_id",
) -> Dict[str, pd.DataFrame]:
    """
    Create hierarchical train/val/test splits with multiple difficulty levels.

    Strategy:
    1. Cluster peptides by similarity (MMseqs2 at specified identity)
    2. Identify MHC alleles to hold out entirely (for test_unseen_allele)
    3. Split remaining data by peptide clusters
    4. Create difficulty-stratified test sets

    Args:
        df: Input DataFrame with 'peptide' and MHC columns
        train_ratio: Fraction of data for training (remaining split for val/test)
        seed: Random seed for reproducibility
        similarity_threshold: Peptide clustering threshold (default: 0.8)
        holdout_allele_fraction: Fraction of MHC alleles to hold out (default: 0.1)
        mhc_col: Column name for MHC allele ID (default: 'mhc_one_id')

    Returns:
        Dict with keys: 'train', 'val',
                       'test_seen_motif', 'test_unseen_peptide_seen_motif',
                       'test_unseen_motif', 'test_unseen_allele'
    """
    np.random.seed(seed)

    if len(df) == 0:
        return {
            'train': pd.DataFrame(),
            'val': pd.DataFrame(),
            'test_seen_motif': pd.DataFrame(),
            'test_unseen_peptide_seen_motif': pd.DataFrame(),
            'test_unseen_motif': pd.DataFrame(),
            'test_unseen_allele': pd.DataFrame(),
        }

    # Step 1: Cluster peptides by similarity
    print(f"    Clustering peptides at {similarity_threshold*100:.0f}% identity...")
    unique_peptides = df['peptide'].unique().tolist()
    peptide_to_cluster, cluster_to_peptides = cluster_peptides_mmseqs2(
        unique_peptides,
        similarity_threshold=similarity_threshold,
        coverage=0.8,
    )

    # Add cluster_id to dataframe
    df = df.copy()
    df['cluster_id'] = df['peptide'].map(peptide_to_cluster)

    num_clusters = len(cluster_to_peptides)
    print(f"    Created {num_clusters:,} clusters from {len(unique_peptides):,} unique peptides")

    # Step 2: Identify held-out MHC alleles
    print(f"    Selecting MHC alleles to hold out ({holdout_allele_fraction*100:.0f}%)...")
    all_alleles = df[mhc_col].dropna().unique()
    num_holdout = max(1, int(len(all_alleles) * holdout_allele_fraction))

    # Select alleles to hold out (prefer alleles with moderate representation)
    allele_counts = df[mhc_col].value_counts()
    # Sort by count and select from the middle range to avoid extreme cases
    sorted_alleles = allele_counts.sort_values().index.tolist()
    middle_start = len(sorted_alleles) // 4
    middle_end = 3 * len(sorted_alleles) // 4
    candidate_alleles = sorted_alleles[middle_start:middle_end]

    if len(candidate_alleles) < num_holdout:
        candidate_alleles = sorted_alleles

    np.random.shuffle(candidate_alleles)
    holdout_alleles = set(candidate_alleles[:num_holdout])
    print(f"    Holding out {len(holdout_alleles):,} MHC alleles for test_unseen_allele")

    # Step 3: Create test_unseen_allele set
    test_unseen_allele_df = df[df[mhc_col].isin(holdout_alleles)].copy()
    remaining_df = df[~df[mhc_col].isin(holdout_alleles)].copy()
    print(f"    test_unseen_allele: {len(test_unseen_allele_df):,} samples")
    print(f"    Remaining for train/val/test: {len(remaining_df):,} samples")

    # Step 4: Split clusters into train/val/test
    remaining_clusters = remaining_df['cluster_id'].unique()
    np.random.shuffle(remaining_clusters)

    # Calculate split sizes
    val_test_ratio = 1.0 - train_ratio
    val_ratio = val_test_ratio / 2
    test_ratio = val_test_ratio / 2

    num_train_clusters = int(len(remaining_clusters) * train_ratio)
    num_val_clusters = int(len(remaining_clusters) * val_ratio)

    train_clusters = set(remaining_clusters[:num_train_clusters])
    val_clusters = set(remaining_clusters[num_train_clusters:num_train_clusters + num_val_clusters])
    test_clusters = set(remaining_clusters[num_train_clusters + num_val_clusters:])

    print(f"    Cluster split: train={len(train_clusters):,}, val={len(val_clusters):,}, test={len(test_clusters):,}")

    # Step 5: Create base splits
    print(f"    Creating base splits...")
    train_df = remaining_df[remaining_df['cluster_id'].isin(train_clusters)].copy()
    val_df = remaining_df[remaining_df['cluster_id'].isin(val_clusters)].copy()
    test_unseen_motif_df = remaining_df[remaining_df['cluster_id'].isin(test_clusters)].copy()

    # Step 6: Create test_seen_motif from val (baseline with overlap)
    # Sample a portion of val set as test_seen_motif
    if len(val_df) > 0:
        sample_size = min(len(val_df) // 3, len(test_unseen_motif_df) if len(test_unseen_motif_df) > 0 else 100)
        sample_size = max(sample_size, 1)
        test_seen_motif_df = val_df.sample(n=min(sample_size, len(val_df)), random_state=seed).copy()
    else:
        test_seen_motif_df = pd.DataFrame()

    # Step 7: Create test_unseen_peptide_seen_motif
    # These are peptides from train clusters that aren't in the actual train set
    # Select some peptides from large train clusters to hold out
    print(f"    Finding unseen peptides from seen motifs...")
    train_peptides = set(train_df['peptide'].unique())
    test_unseen_peptide_candidates = []

    # Pre-compute cluster data for efficiency
    remaining_df_indexed = remaining_df.set_index('cluster_id')

    for cluster_id in tqdm(train_clusters, desc="    Scanning train clusters", leave=False):
        cluster_peptides = cluster_to_peptides.get(cluster_id, [])
        if len(cluster_peptides) > 1:
            try:
                cluster_data = remaining_df_indexed.loc[[cluster_id]].reset_index()
                cluster_data = cluster_data[~cluster_data['peptide'].isin(train_peptides)]
                if len(cluster_data) > 0:
                    test_unseen_peptide_candidates.append(cluster_data)
            except KeyError:
                continue

    # Also look for peptides in train clusters that have different MHC pairings not in train
    train_df_indexed = train_df.set_index('cluster_id')

    for cluster_id in tqdm(train_clusters, desc="    Finding novel peptides", leave=False):
        try:
            cluster_data = remaining_df_indexed.loc[[cluster_id]].reset_index()
        except KeyError:
            continue

        # Find peptides in this cluster that are not in train_df
        cluster_peptides = set(cluster_data['peptide'].unique())
        try:
            train_cluster_peptides = set(train_df_indexed.loc[[cluster_id]]['peptide'].unique())
        except KeyError:
            train_cluster_peptides = set()

        novel_peptides = cluster_peptides - train_cluster_peptides

        if novel_peptides:
            novel_data = cluster_data[cluster_data['peptide'].isin(novel_peptides)]
            if len(novel_data) > 0:
                test_unseen_peptide_candidates.append(novel_data)

    if test_unseen_peptide_candidates:
        test_unseen_peptide_df = pd.concat(test_unseen_peptide_candidates, ignore_index=True)
        test_unseen_peptide_df = test_unseen_peptide_df.drop_duplicates()
        # Remove any peptides that are actually in train
        test_unseen_peptide_df = test_unseen_peptide_df[
            ~test_unseen_peptide_df['peptide'].isin(train_peptides)
        ]
    else:
        # Fall back: sample from val_df ensuring no peptide overlap with train
        val_peptides = set(val_df['peptide'].unique())
        novel_val_peptides = val_peptides - train_peptides
        if novel_val_peptides and len(val_df) > 0:
            test_unseen_peptide_df = val_df[val_df['peptide'].isin(novel_val_peptides)].copy()
            # Limit size
            if len(test_unseen_peptide_df) > len(test_unseen_motif_df) * 2:
                test_unseen_peptide_df = test_unseen_peptide_df.sample(
                    n=len(test_unseen_motif_df), random_state=seed
                )
        else:
            test_unseen_peptide_df = pd.DataFrame()

    # Print statistics
    print(f"\n    Hierarchical split statistics:")
    print(f"      train: {len(train_df):,} samples, {train_df['peptide'].nunique():,} unique peptides")
    print(f"      val: {len(val_df):,} samples, {val_df['peptide'].nunique():,} unique peptides")
    print(f"      test_seen_motif: {len(test_seen_motif_df):,} samples")
    print(f"      test_unseen_peptide_seen_motif: {len(test_unseen_peptide_df):,} samples")
    print(f"      test_unseen_motif: {len(test_unseen_motif_df):,} samples")
    print(f"      test_unseen_allele: {len(test_unseen_allele_df):,} samples")

    return {
        'train': train_df,
        'val': val_df,
        'test_seen_motif': test_seen_motif_df,
        'test_unseen_peptide_seen_motif': test_unseen_peptide_df,
        'test_unseen_motif': test_unseen_motif_df,
        'test_unseen_allele': test_unseen_allele_df,
        '_peptide_to_cluster': peptide_to_cluster,
        '_cluster_to_peptides': cluster_to_peptides,
        '_holdout_alleles': holdout_alleles,
    }


def verify_hierarchical_splits(splits: Dict[str, pd.DataFrame]) -> bool:
    """Verify that hierarchical splits have expected properties."""
    train_df = splits['train']
    if len(train_df) == 0:
        return True

    train_peptides = set(train_df['peptide'].unique())
    train_clusters = set(train_df['cluster_id'].unique()) if 'cluster_id' in train_df.columns else set()
    holdout_alleles = splits.get('_holdout_alleles', set())
    mhc_col = 'mhc_one_id' if 'mhc_one_id' in train_df.columns else 'mhc_two_id'
    train_alleles = set(train_df[mhc_col].dropna().unique()) if mhc_col in train_df.columns else set()

    errors = []

    # Check test_unseen_allele: no MHC allele overlap with train
    test_unseen_allele = splits.get('test_unseen_allele', pd.DataFrame())
    if len(test_unseen_allele) > 0 and mhc_col in test_unseen_allele.columns:
        test_alleles = set(test_unseen_allele[mhc_col].dropna().unique())
        overlap = test_alleles & train_alleles
        if overlap:
            errors.append(f"test_unseen_allele has {len(overlap)} alleles overlapping with train")

    # Check test_unseen_motif: no peptide cluster overlap with train
    test_unseen_motif = splits.get('test_unseen_motif', pd.DataFrame())
    if len(test_unseen_motif) > 0 and 'cluster_id' in test_unseen_motif.columns:
        test_clusters = set(test_unseen_motif['cluster_id'].unique())
        overlap = test_clusters & train_clusters
        if overlap:
            errors.append(f"test_unseen_motif has {len(overlap)} clusters overlapping with train")

    # Check test_unseen_peptide_seen_motif: no peptide overlap but cluster overlap expected
    test_unseen_peptide = splits.get('test_unseen_peptide_seen_motif', pd.DataFrame())
    if len(test_unseen_peptide) > 0:
        test_peptides = set(test_unseen_peptide['peptide'].unique())
        peptide_overlap = test_peptides & train_peptides
        if peptide_overlap:
            errors.append(f"test_unseen_peptide_seen_motif has {len(peptide_overlap)} peptides overlapping with train")

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
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, output_path)
    print(f"    Saved {len(df):,} rows to {output_path}")


def print_statistics(
    perm_key: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Print statistics for a permutation type."""
    print(f"\n  Statistics for {perm_key}:")

    total = len(train_df) + len(val_df) + len(test_df)
    if total == 0:
        print(f"    No data available")
        return

    print(f"    Total samples: {total:,}")
    print(f"    Train: {len(train_df):,} ({100*len(train_df)/total:.1f}%)")
    print(f"    Val: {len(val_df):,} ({100*len(val_df)/total:.1f}%)")
    print(f"    Test: {len(test_df):,} ({100*len(test_df)/total:.1f}%)")

    # MHC allele distribution (for train set)
    if len(train_df) > 0:
        if 'mhc_one_id' in train_df.columns:
            mhc_one_counts = train_df['mhc_one_id'].value_counts()
            print(f"    Top 5 MHC-I alleles (train): {dict(mhc_one_counts.head(5))}")

        if 'mhc_two_id' in train_df.columns:
            mhc_two_counts = train_df['mhc_two_id'].value_counts()
            print(f"    Top 5 MHC-II alleles (train): {dict(mhc_two_counts.head(5))}")

        # Source distribution
        if 'source' in train_df.columns:
            source_counts = train_df['source'].value_counts()
            print(f"    Sources (train): {dict(source_counts)}")


def collect_detailed_statistics(
    perm_key: str,
    config: Dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict:
    """Collect detailed statistics for a permutation type."""
    stats = {
        'permutation_key': perm_key,
        'description': config['description'],
        'splits': {},
    }

    for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
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

            # MHC allele distributions
            if 'mhc_one_id' in df.columns:
                mhc_one_counts = df['mhc_one_id'].value_counts()
                split_stats['mhc_one_alleles'] = {
                    'unique_count': len(mhc_one_counts),
                    'top_10': dict(mhc_one_counts.head(10)),
                }

            if 'mhc_two_id' in df.columns:
                mhc_two_counts = df['mhc_two_id'].dropna().value_counts()
                if len(mhc_two_counts) > 0:
                    split_stats['mhc_two_alleles'] = {
                        'unique_count': len(mhc_two_counts),
                        'top_10': dict(mhc_two_counts.head(10)),
                    }

            # Source distribution
            if 'source' in df.columns:
                source_counts = df['source'].value_counts()
                split_stats['sources'] = dict(source_counts)

        stats['splits'][split_name] = split_stats

    return stats


def collect_hierarchical_statistics(
    perm_key: str,
    config: Dict,
    splits: Dict[str, pd.DataFrame],
    peptide_to_cluster: Dict[str, int],
    cluster_to_peptides: Dict[int, List[str]],
) -> Dict:
    """Collect detailed statistics for hierarchical splits."""
    stats = {
        'permutation_key': perm_key,
        'description': config['description'],
        'hierarchical': True,
        'clustering': {
            'num_clusters': len(cluster_to_peptides),
            'num_peptides': len(peptide_to_cluster),
            'cluster_size_distribution': {},
        },
        'splits': {},
    }

    # Cluster size distribution
    cluster_sizes = [len(peptides) for peptides in cluster_to_peptides.values()]
    if cluster_sizes:
        stats['clustering']['cluster_size_distribution'] = {
            'min': min(cluster_sizes),
            'max': max(cluster_sizes),
            'mean': np.mean(cluster_sizes),
            'median': np.median(cluster_sizes),
            'singletons': sum(1 for s in cluster_sizes if s == 1),
        }

    # Get train info for overlap analysis
    train_df = splits.get('train', pd.DataFrame())
    train_peptides = set(train_df['peptide'].unique()) if len(train_df) > 0 else set()
    train_clusters = set(train_df['cluster_id'].unique()) if len(train_df) > 0 and 'cluster_id' in train_df.columns else set()
    mhc_col = 'mhc_one_id' if 'mhc_one_id' in train_df.columns else 'mhc_two_id'
    train_alleles = set(train_df[mhc_col].dropna().unique()) if len(train_df) > 0 and mhc_col in train_df.columns else set()

    split_names = ['train', 'val', 'test_seen_motif', 'test_unseen_peptide_seen_motif',
                   'test_unseen_motif', 'test_unseen_allele']

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

            # Cluster info
            if 'cluster_id' in df.columns:
                unique_clusters = df['cluster_id'].unique()
                split_stats['unique_clusters'] = len(unique_clusters)

                # Overlap with train clusters
                if split_name != 'train':
                    cluster_overlap = set(unique_clusters) & train_clusters
                    split_stats['cluster_overlap_with_train'] = len(cluster_overlap)
                    split_stats['cluster_overlap_pct'] = (
                        100 * len(cluster_overlap) / len(unique_clusters)
                        if len(unique_clusters) > 0 else 0
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
            if mhc_col in df.columns:
                mhc_counts = df[mhc_col].dropna().value_counts()
                split_stats['mhc_alleles'] = {
                    'unique_count': len(mhc_counts),
                    'top_10': dict(mhc_counts.head(10)),
                }

                # Allele overlap with train
                if split_name != 'train':
                    split_alleles = set(df[mhc_col].dropna().unique())
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


def save_peptide_clusters(
    output_dir: Path,
    peptide_to_cluster: Dict[str, int],
    cluster_to_peptides: Dict[int, List[str]],
) -> None:
    """Save peptide cluster assignments to TSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_path = output_dir / "peptide_clusters.tsv"

    with open(cluster_path, 'w') as f:
        f.write("peptide\tcluster_id\tcluster_size\tis_representative\n")
        for cluster_id, peptides in sorted(cluster_to_peptides.items()):
            # First peptide is the representative
            for i, peptide in enumerate(peptides):
                is_rep = "true" if i == 0 else "false"
                f.write(f"{peptide}\t{cluster_id}\t{len(peptides)}\t{is_rep}\n")

    print(f"    Saved peptide clusters to {cluster_path}")


def save_split_assignments(
    output_dir: Path,
    perm_key: str,
    splits: Dict[str, pd.DataFrame],
) -> None:
    """Save split assignments to TSV file for analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = output_dir / "split_assignments.tsv"

    all_rows = []
    for split_name, df in splits.items():
        if split_name.startswith('_'):
            continue
        if len(df) == 0:
            continue
        df_copy = df.copy()
        df_copy['split'] = split_name
        all_rows.append(df_copy)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        # Select relevant columns
        output_cols = ['peptide', 'split']
        if 'cluster_id' in combined.columns:
            output_cols.append('cluster_id')
        if 'mhc_one_id' in combined.columns:
            output_cols.append('mhc_one_id')
        if 'mhc_two_id' in combined.columns:
            output_cols.append('mhc_two_id')

        combined[output_cols].to_csv(assignment_path, sep='\t', index=False)
        print(f"    Saved split assignments to {assignment_path}")


def write_hierarchical_statistics_doc(
    output_dir: Path,
    all_detailed_stats: Dict,
    config_info: Dict,
) -> None:
    """Write hierarchical dataset statistics to markdown documentation file."""
    from datetime import datetime

    doc_path = output_dir / "dataset_statistics.md"
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Peptide-MHC Evaluation Dataset Statistics (Hierarchical Split)")
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
    lines.append(f"| Similarity Threshold | {config_info.get('similarity_threshold', 0.8)} |")
    lines.append(f"| Holdout Allele Fraction | {config_info.get('holdout_allele_fraction', 0.1)} |")
    lines.append("")

    # Test set difficulty explanation
    lines.append("## Test Set Difficulty Levels")
    lines.append("")
    lines.append("| Test Set | Description | Expected Performance |")
    lines.append("|----------|-------------|---------------------|")
    lines.append("| `test_seen_motif` | Same binding motifs as training | High (baseline) |")
    lines.append("| `test_unseen_peptide_seen_motif` | New peptides, but motifs seen | Medium-high |")
    lines.append("| `test_unseen_motif` | Binding motifs not in training | Lower (true generalization) |")
    lines.append("| `test_unseen_allele` | MHC alleles not in training | Hardest |")
    lines.append("")

    # Detailed statistics for each permutation key
    for perm_key, stats in all_detailed_stats.items():
        lines.append(f"## {perm_key}")
        lines.append("")
        lines.append(f"**Description:** {stats['description']}")
        lines.append("")

        # Clustering statistics
        if 'clustering' in stats:
            clustering = stats['clustering']
            lines.append("### Clustering Statistics")
            lines.append("")
            lines.append(f"- **Total peptides:** {clustering['num_peptides']:,}")
            lines.append(f"- **Total clusters:** {clustering['num_clusters']:,}")
            if 'cluster_size_distribution' in clustering and clustering['cluster_size_distribution']:
                dist = clustering['cluster_size_distribution']
                lines.append(f"- **Cluster sizes:** min={dist['min']}, max={dist['max']}, "
                           f"mean={dist['mean']:.1f}, median={dist['median']:.1f}")
                lines.append(f"- **Singleton clusters:** {dist['singletons']:,}")
            lines.append("")

        # Summary table
        lines.append("### Split Summary")
        lines.append("")
        lines.append("| Split | Samples | Peptides | Clusters | Peptide Overlap | Cluster Overlap | Allele Overlap |")
        lines.append("|-------|---------|----------|----------|-----------------|-----------------|----------------|")

        split_order = ['train', 'val', 'test_seen_motif', 'test_unseen_peptide_seen_motif',
                       'test_unseen_motif', 'test_unseen_allele']

        for split_name in split_order:
            if split_name not in stats['splits']:
                continue
            split_stats = stats['splits'][split_name]
            samples = split_stats['total_samples']
            peptides = split_stats.get('unique_peptides', 0)
            clusters = split_stats.get('unique_clusters', '-')
            pep_overlap = f"{split_stats.get('peptide_overlap_pct', 0):.1f}%" if split_name != 'train' else '-'
            clust_overlap = f"{split_stats.get('cluster_overlap_pct', 0):.1f}%" if split_name != 'train' else '-'
            allele_overlap = f"{split_stats.get('allele_overlap_pct', 0):.1f}%" if split_name != 'train' else '-'
            lines.append(f"| {split_name} | {samples:,} | {peptides:,} | {clusters} | {pep_overlap} | {clust_overlap} | {allele_overlap} |")

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

            if 'unique_clusters' in split_stats:
                lines.append(f"- **Unique clusters:** {split_stats['unique_clusters']:,}")

            if 'peptide_length' in split_stats:
                pl = split_stats['peptide_length']
                lines.append(f"- **Peptide length:** min={pl['min']}, max={pl['max']}, "
                           f"mean={pl['mean']:.1f}, median={pl['median']:.1f}")

            # Overlap stats
            if split_name != 'train':
                if 'peptide_overlap_with_train' in split_stats:
                    lines.append(f"- **Peptide overlap with train:** {split_stats['peptide_overlap_with_train']:,} "
                               f"({split_stats['peptide_overlap_pct']:.1f}%)")
                if 'cluster_overlap_with_train' in split_stats:
                    lines.append(f"- **Cluster overlap with train:** {split_stats['cluster_overlap_with_train']:,} "
                               f"({split_stats['cluster_overlap_pct']:.1f}%)")
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

    # Write to file
    with open(doc_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n  Statistics documentation written to: {doc_path}")


def write_statistics_doc(
    output_dir: Path,
    all_detailed_stats: Dict,
    config_info: Dict,
) -> None:
    """Write dataset statistics to a markdown documentation file."""
    from datetime import datetime

    doc_path = output_dir / "dataset_statistics.md"
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Peptide-MHC Evaluation Dataset Statistics")
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
    lines.append(f"| Split Ratios | {config_info['train_ratio']}/{config_info['val_ratio']}/{config_info['test_ratio']} (train/val/test) |")
    lines.append(f"| Random Seed | {config_info['seed']} |")
    lines.append(f"| Peptide Length Range | {config_info['min_peptide_len']}-{config_info['max_peptide_len']} |")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Dataset | Total | Train | Val | Test | Unique Peptides (Train) |")
    lines.append("|---------|-------|-------|-----|------|-------------------------|")

    for perm_key, stats in all_detailed_stats.items():
        splits = stats['splits']
        total = sum(s['total_samples'] for s in splits.values())
        train_peptides = splits['train'].get('unique_peptides', 0)
        lines.append(
            f"| {perm_key} | {total:,} | {splits['train']['total_samples']:,} | "
            f"{splits['val']['total_samples']:,} | {splits['test']['total_samples']:,} | "
            f"{train_peptides:,} |"
        )
    lines.append("")

    # Detailed statistics for each permutation key
    for perm_key, stats in all_detailed_stats.items():
        lines.append(f"## {perm_key}")
        lines.append("")
        lines.append(f"**Description:** {stats['description']}")
        lines.append("")

        for split_name in ['train', 'val', 'test']:
            split_stats = stats['splits'][split_name]
            lines.append(f"### {split_name.capitalize()} Split")
            lines.append("")

            if split_stats['total_samples'] == 0:
                lines.append("*No data*")
                lines.append("")
                continue

            lines.append(f"- **Total samples:** {split_stats['total_samples']:,}")
            lines.append(f"- **Unique peptides:** {split_stats['unique_peptides']:,}")

            # Peptide length stats
            if 'peptide_length' in split_stats:
                pl = split_stats['peptide_length']
                lines.append(f"- **Peptide length:** min={pl['min']}, max={pl['max']}, "
                           f"mean={pl['mean']:.1f}, median={pl['median']:.1f}")

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

            # MHC-I allele distribution
            if 'mhc_one_alleles' in split_stats:
                mhc_stats = split_stats['mhc_one_alleles']
                lines.append(f"**MHC-I Alleles:** {mhc_stats['unique_count']:,} unique")
                lines.append("")
                lines.append("Top 10 alleles:")
                lines.append("")
                lines.append("| Allele | Count |")
                lines.append("|--------|-------|")
                for allele, count in list(mhc_stats['top_10'].items())[:10]:
                    lines.append(f"| {allele} | {count:,} |")
                lines.append("")

            # MHC-II allele distribution
            if 'mhc_two_alleles' in split_stats:
                mhc_stats = split_stats['mhc_two_alleles']
                lines.append(f"**MHC-II Alleles:** {mhc_stats['unique_count']:,} unique")
                lines.append("")
                lines.append("Top 10 alleles:")
                lines.append("")
                lines.append("| Allele | Count |")
                lines.append("|--------|-------|")
                for allele, count in list(mhc_stats['top_10'].items())[:10]:
                    lines.append(f"| {allele} | {count:,} |")
                lines.append("")

    # Write to file
    with open(doc_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n  Statistics documentation written to: {doc_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create peptide-MHC evaluation datasets with train/val/test splits",
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
        "--split_ratios",
        type=str,
        default="0.8,0.1,0.1",
        help="Train/val/test ratios (default: 0.8,0.1,0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
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
        default=25,
        help="Maximum peptide length (default: 25)"
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
        "--use_hierarchical_split",
        action="store_true",
        help="Enable hierarchical stratified splitting with multiple test sets"
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.8,
        help="Peptide clustering threshold for hierarchical split (default: 0.8)"
    )
    parser.add_argument(
        "--holdout_allele_fraction",
        type=float,
        default=0.1,
        help="Fraction of MHC alleles to hold out for test_unseen_allele (default: 0.1)"
    )
    parser.add_argument(
        "--skip_mhc_standardization",
        action="store_true",
        help="Skip MHC allele standardization using tidytcells"
    )

    args = parser.parse_args()

    # Parse split ratios
    try:
        ratios = [float(r) for r in args.split_ratios.split(',')]
        if len(ratios) != 3:
            raise ValueError("Must provide exactly 3 ratios")
        train_ratio, val_ratio, test_ratio = ratios
    except Exception as e:
        parser.error(f"Invalid split_ratios format: {e}")

    print("=" * 60)
    print("Peptide-MHC Evaluation Dataset Creation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Input directory: {args.input_dir}")
    print(f"  Output directory: {args.output_dir}")
    if args.use_hierarchical_split:
        print(f"  Split mode: HIERARCHICAL (multiple difficulty test sets)")
        print(f"  Train ratio: {train_ratio}")
        print(f"  Similarity threshold: {args.similarity_threshold}")
        print(f"  Holdout allele fraction: {args.holdout_allele_fraction}")
    else:
        print(f"  Split mode: Simple peptide-level")
        print(f"  Split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    print(f"  Random seed: {args.seed}")
    print(f"  Peptide length: {args.min_peptide_len}-{args.max_peptide_len}")
    print(f"  Permutation keys: {args.permutation_keys}")
    if args.skip_mhc_standardization:
        print(f"  MHC standardization: DISABLED")
    else:
        print(f"  MHC standardization: {'ENABLED (tidytcells)' if TIDYTCELLS_AVAILABLE else 'UNAVAILABLE (tidytcells not installed)'}")

    # Set random seed
    np.random.seed(args.seed)

    # Load data
    print("\n" + "=" * 60)
    print("Step 1: Loading databases")
    print("=" * 60)
    input_dir = Path(args.input_dir)
    combined_df = load_databases(input_dir)

    # Standardize MHC allele names using tidytcells
    if not args.skip_mhc_standardization:
        print("\n" + "=" * 60)
        print("Step 2: Standardizing MHC allele names")
        print("=" * 60)
        mhc_id_columns = ["mhc_one_id", "mhc_two_id"]
        combined_df = standardize_mhc_columns(combined_df, mhc_id_columns)

    # Process each permutation key
    print("\n" + "=" * 60)
    print("Step 3: Processing permutation keys")
    print("=" * 60)

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

        if args.use_hierarchical_split:
            # Hierarchical stratified split with multiple difficulty levels
            print(f"\n    Creating hierarchical splits for {perm_key}...")

            # Determine MHC column for this permutation key
            mhc_col = "mhc_one_id" if "mhc_one" in perm_key else "mhc_two_id"

            splits = hierarchical_stratified_split(
                processed_df,
                train_ratio=train_ratio,
                seed=args.seed,
                similarity_threshold=args.similarity_threshold,
                holdout_allele_fraction=args.holdout_allele_fraction,
                mhc_col=mhc_col,
            )

            # Verify splits
            if not verify_hierarchical_splits(splits):
                print(f"    ERROR: Hierarchical split verification failed for {perm_key}")
                continue

            # Save peptide clusters
            peptide_to_cluster = splits.get('_peptide_to_cluster', {})
            cluster_to_peptides = splits.get('_cluster_to_peptides', {})
            if peptide_to_cluster and cluster_to_peptides:
                save_peptide_clusters(perm_output_dir, peptide_to_cluster, cluster_to_peptides)

            # Save split assignments
            save_split_assignments(perm_output_dir, perm_key, splits)

            # Save to parquet files
            split_names = ['train', 'val', 'test_seen_motif', 'test_unseen_peptide_seen_motif',
                          'test_unseen_motif', 'test_unseen_allele']
            for split_name in split_names:
                split_df = splits.get(split_name, pd.DataFrame())
                if len(split_df) > 0:
                    # Remove internal columns before saving
                    output_df = split_df.drop(columns=['cluster_id'], errors='ignore')
                    save_parquet(output_df, perm_output_dir / f"{split_name}.parquet")

            # Store statistics
            all_stats[perm_key] = {
                split_name: len(splits.get(split_name, pd.DataFrame()))
                for split_name in split_names
            }

            # Collect detailed statistics
            all_detailed_stats[perm_key] = collect_hierarchical_statistics(
                perm_key, config, splits, peptide_to_cluster, cluster_to_peptides
            )

        else:
            # Simple peptide-level split
            print(f"\n    Creating train/val/test splits for {perm_key}...")
            train_df, val_df, test_df = peptide_level_split(
                processed_df,
                train_ratio,
                val_ratio,
                test_ratio,
                args.seed,
            )

            # Verify no overlap
            if not verify_no_peptide_overlap(train_df, val_df, test_df):
                print(f"    ERROR: Peptide overlap detected for {perm_key}")
                continue

            # Save to parquet files
            save_parquet(train_df, perm_output_dir / "train.parquet")
            save_parquet(val_df, perm_output_dir / "val.parquet")
            save_parquet(test_df, perm_output_dir / "test.parquet")

            # Store statistics
            all_stats[perm_key] = {
                'train': len(train_df),
                'val': len(val_df),
                'test': len(test_df),
            }

            # Collect detailed statistics
            all_detailed_stats[perm_key] = collect_detailed_statistics(
                perm_key, config, train_df, val_df, test_df
            )

            # Print statistics
            print_statistics(perm_key, train_df, val_df, test_df)

    # Write statistics documentation
    if all_detailed_stats:
        config_info = {
            'input_dir': args.input_dir,
            'output_dir': args.output_dir,
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'seed': args.seed,
            'min_peptide_len': args.min_peptide_len,
            'max_peptide_len': args.max_peptide_len,
            'similarity_threshold': args.similarity_threshold,
            'holdout_allele_fraction': args.holdout_allele_fraction,
        }
        if args.use_hierarchical_split:
            write_hierarchical_statistics_doc(output_dir, all_detailed_stats, config_info)
        else:
            write_statistics_doc(output_dir, all_detailed_stats, config_info)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print(f"\nOutput directory: {output_dir}")
    for perm_key, stats in all_stats.items():
        total = sum(stats.values())
        print(f"\n  {perm_key}:")
        print(f"    Total: {total:,}")
        if args.use_hierarchical_split:
            print(f"    Splits:")
            for split_name, count in stats.items():
                print(f"      {split_name}: {count:,}")
        else:
            print(f"    Train/Val/Test: {stats['train']:,}/{stats['val']:,}/{stats['test']:,}")

    print("\nDone!")


if __name__ == "__main__":
    main()
