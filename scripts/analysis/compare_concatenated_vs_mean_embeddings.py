#!/usr/bin/env python3
"""
Compare concatenated vs. mean molecule embeddings.

This script analyzes how embeddings of concatenated molecule combinations differ
from the mean of individual molecule embeddings.

Usage:
    python scripts/analysis/compare_concatenated_vs_mean_embeddings.py \
        --deduplicated_data data/deduplicated/ \
        --combinations tra_trb peptide_mhc_one \
        --model_name facebook/esm2_t6_8M_UR50D \
        --output_dir results/embedding_comparison \
        --sample 500

This script:
1. Loads deduplicated parquet files with permutation_key and sequence columns
2. Extracts embeddings for concatenated sequences and individual molecules
3. Computes mean of individual molecule embeddings
4. Compares concat vs. mean using multiple metrics
5. Performs statistical tests
6. Creates comprehensive visualizations
"""

import argparse
import gc
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
import torch
import torch.nn as nn
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ========================================================================
# REUSED CLASSES FROM visualize_embeddings_umap.py
# ========================================================================

class SequenceDataset(Dataset):
    """Dataset for batch processing sequences."""

    def __init__(self, sequences: List[str], tokenizer, max_length: int = 512):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return seq, idx


def collate_fn(batch, tokenizer, max_length):
    """Custom collate function for tokenization."""
    sequences, indices = zip(*batch)

    # ESM models expect sequences without spaces
    encoded = tokenizer(
        list(sequences),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
        add_special_tokens=True
    )
    return encoded, list(indices)


class EmbeddingExtractor(nn.Module):
    """Wrapper for embedding extraction with mean pooling."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # Mean pooling over sequence length (excluding padding)
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask

        return embeddings


def extract_embeddings(
    sequences: List[str],
    model_name: str,
    batch_size: int = 32,
    max_length: int = 512,
    device: str = 'cuda',
    model=None,
    tokenizer=None,
    extractor=None
) -> np.ndarray:
    """
    Extract embeddings using ESM model.

    Args:
        sequences: List of amino acid sequences
        model_name: HuggingFace model name (e.g., facebook/esm2_t6_8M_UR50D)
        batch_size: Batch size for inference
        max_length: Maximum sequence length
        device: Device to run on
        model: Pre-loaded model (optional, for reuse)
        tokenizer: Pre-loaded tokenizer (optional, for reuse)
        extractor: Pre-loaded extractor (optional, for reuse)

    Returns:
        numpy array of shape (num_sequences, embedding_dim)
    """
    # Check device
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = 'cpu'

    # Load model and tokenizer if not provided
    load_model = model is None
    if load_model:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        extractor = EmbeddingExtractor(model)
        extractor = extractor.to(device)
        extractor.eval()

    # Create dataset and dataloader
    dataset = SequenceDataset(sequences, tokenizer, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length),
        pin_memory=True if device == 'cuda' else False,
    )

    # Extract embeddings
    all_embeddings = []

    with torch.no_grad():
        for encoded, indices in dataloader:
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            embeddings = extractor(input_ids, attention_mask)
            embeddings_np = embeddings.cpu().numpy()
            all_embeddings.append(embeddings_np)

    # Concatenate all embeddings
    embeddings_array = np.vstack(all_embeddings)

    # Clear GPU memory if we loaded the model
    if load_model:
        del model, extractor
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

    return embeddings_array


# ========================================================================
# MOLECULE EXTRACTION FUNCTIONS
# ========================================================================

def parse_permutation_key(pkey: str) -> List[str]:
    """
    Parse permutation key into molecule types.

    Properly handles multi-word components like mhc_one and mhc_two.

    Args:
        pkey: Permutation key (e.g., 'tra_trb', 'peptide_mhc_one')

    Returns:
        List of molecule types (e.g., ['tra', 'trb'], ['peptide', 'mhc_one'])
    """
    # Define valid components in order of specificity (longer first)
    valid_components = ['mhc_one', 'mhc_two', 'peptide', 'tra', 'trb']

    # Extract components from the key
    components = []
    remaining = pkey

    while remaining:
        matched = False
        for component in valid_components:
            if remaining.startswith(component):
                components.append(component)
                # Remove matched component and any following underscore
                remaining = remaining[len(component):]
                if remaining.startswith('_'):
                    remaining = remaining[1:]
                matched = True
                break

        if not matched:
            # If no component matched, we have an unexpected format
            print(f"Warning: Could not fully parse permutation key '{pkey}', using fallback")
            # Fallback: just split by underscore
            return pkey.split('_')

    return components


def extract_molecules_from_row(row: Dict) -> Dict[str, str]:
    """
    Extract individual molecules from concatenated sequence.

    Logic:
    1. Parse permutation_key: 'tra_trb' → ['tra', 'trb']
    2. Split sequence by space: 'CASSLG GILGFV' → ['CASSLG', 'GILGFV']
    3. Map in order: {'tra': 'CASSLG', 'trb': 'GILGFV'}

    Fallback: Use individual columns (tra, trb, peptide, etc.) if available

    Args:
        row: Dictionary representing a data row

    Returns:
        Dictionary mapping molecule type to sequence
    """
    if 'sequence' in row and 'permutation_key' in row and row['sequence']:
        # New format: extract from concatenated sequence
        pkey = row['permutation_key']
        sequence = row['sequence']

        # Parse permutation key to get molecule types
        mol_types = parse_permutation_key(pkey)

        # Split sequence by space
        molecules = sequence.split()

        # Check for mismatch
        if len(molecules) != len(mol_types):
            print(f"Warning: Mismatch in row - {len(molecules)} molecules, {len(mol_types)} types")
            print(f"  permutation_key: {pkey}")
            print(f"  sequence: {sequence}")
            # Try to handle gracefully
            min_len = min(len(molecules), len(mol_types))
            mol_types = mol_types[:min_len]
            molecules = molecules[:min_len]

        # Map molecule types to sequences
        return dict(zip(mol_types, molecules))
    else:
        # Old format: use individual columns
        result = {}
        for col in ['tra', 'trb', 'peptide', 'mhc_one', 'mhc_two']:
            if col in row and row[col] and str(row[col]) != 'nan':
                result[col] = str(row[col])
        return result


# ========================================================================
# DATA LOADING
# ========================================================================

def load_data_for_combinations(
    data_path: str,
    permutation_keys: List[str],
    sample_per_key: int = 100,
    max_files: int = None
) -> pd.DataFrame:
    """
    Load deduplicated parquet files and filter by user-specified permutation keys.

    Args:
        data_path: Path to directory containing parquet files
        permutation_keys: List of permutation keys to analyze (e.g., ['tra_trb', 'peptide_mhc_one'])
        sample_per_key: Number of sequences to sample per permutation key
        max_files: Maximum number of parquet files to read (for testing)

    Returns:
        DataFrame with columns: permutation_key, sequence, and extracted molecules
    """
    print("=" * 80)
    print("LOADING AND SAMPLING DATA")
    print("=" * 80)

    data_path = Path(data_path)
    parquet_files = sorted(data_path.glob("**/*.parquet"))

    if len(parquet_files) == 0:
        raise ValueError(f"No parquet files found in {data_path}")

    if max_files:
        parquet_files = parquet_files[:max_files]
        print(f"Limiting to first {max_files} files for testing")

    print(f"Found {len(parquet_files)} parquet files")
    print(f"Target permutation keys: {', '.join(permutation_keys)}")

    # Read parquet files and filter by permutation keys
    dfs = []
    for pf in tqdm(parquet_files, desc="Reading parquet files"):
        df = pd.read_parquet(pf)
        # Filter to only include specified permutation keys
        df = df[df['permutation_key'].isin(permutation_keys)]
        if len(df) > 0:
            dfs.append(df)

    if len(dfs) == 0:
        raise ValueError(f"No data found for permutation keys: {permutation_keys}")

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows matching permutation keys: {len(full_df):,}")

    # Count sequences per permutation key
    key_counts = full_df['permutation_key'].value_counts()
    print(f"\nPermutation key distribution:")
    for key, count in key_counts.items():
        print(f"  {key}: {count:,}")

    # Sample per permutation key
    print(f"\nSampling up to {sample_per_key} sequences per permutation key...")
    sampled_dfs = []
    for key in permutation_keys:
        key_df = full_df[full_df['permutation_key'] == key]
        if len(key_df) == 0:
            print(f"Warning: No data found for permutation key '{key}'")
            continue

        if len(key_df) > sample_per_key:
            key_df = key_df.sample(n=sample_per_key, random_state=42)
        sampled_dfs.append(key_df)

    if len(sampled_dfs) == 0:
        raise ValueError("No data after sampling")

    sampled_df = pd.concat(sampled_dfs, ignore_index=True)
    print(f"\nTotal sampled sequences: {len(sampled_df):,}")

    # Show final distribution
    final_counts = sampled_df['permutation_key'].value_counts()
    print(f"\nFinal distribution:")
    for key, count in final_counts.items():
        print(f"  {key}: {count:,}")

    return sampled_df


# ========================================================================
# EMBEDDING EXTRACTION
# ========================================================================

def extract_individual_embeddings_efficient(
    df: pd.DataFrame,
    model_name: str,
    batch_size: int = 32,
    max_length: int = 512,
    device: str = 'cuda'
) -> Dict[int, List[Tuple[str, np.ndarray]]]:
    """
    Extract individual molecule embeddings with deduplication optimization.

    Strategy:
    1. Collect all unique molecules across all rows
    2. Extract embeddings once per unique molecule
    3. Map embeddings back to rows

    Args:
        df: DataFrame with permutation_key and sequence columns
        model_name: HuggingFace model name
        batch_size: Batch size for inference
        max_length: Maximum sequence length
        device: Device to run on

    Returns:
        {row_idx: [('tra', embedding), ('trb', embedding)]}
    """
    print("\n" + "=" * 80)
    print("EXTRACTING INDIVIDUAL MOLECULE EMBEDDINGS")
    print("=" * 80)

    # Step 1: Collect all molecules and track which rows they belong to
    all_molecules = {}  # {sequence: [(row_idx, molecule_type)]}
    rows_to_molecules = {}  # {row_idx: [molecule_types]}

    print("Collecting unique molecules...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing molecules"):
        molecules_dict = extract_molecules_from_row(row.to_dict())
        rows_to_molecules[idx] = list(molecules_dict.keys())

        for mol_type, mol_seq in molecules_dict.items():
            if mol_seq:
                if mol_seq not in all_molecules:
                    all_molecules[mol_seq] = []
                all_molecules[mol_seq].append((idx, mol_type))

    print(f"Found {len(all_molecules)} unique molecules across {len(df)} rows")

    # Step 2: Extract embeddings for unique molecules
    unique_sequences = list(all_molecules.keys())
    print(f"Extracting embeddings for {len(unique_sequences)} unique sequences...")

    embeddings = extract_embeddings(
        unique_sequences,
        model_name,
        batch_size,
        max_length,
        device
    )

    sequence_to_embedding = dict(zip(unique_sequences, embeddings))

    # Step 3: Map embeddings back to rows
    print("Mapping embeddings back to rows...")
    row_embeddings = {}
    for seq, row_mol_pairs in all_molecules.items():
        emb = sequence_to_embedding[seq]
        for row_idx, mol_type in row_mol_pairs:
            if row_idx not in row_embeddings:
                row_embeddings[row_idx] = []
            row_embeddings[row_idx].append((mol_type, emb))

    print(f"Mapped embeddings to {len(row_embeddings)} rows")

    return row_embeddings


def extract_dual_embeddings(
    df: pd.DataFrame,
    model_name: str,
    batch_size: int = 32,
    max_length: int = 512,
    device: str = 'cuda'
) -> Tuple[np.ndarray, Dict, Dict]:
    """
    Extract both concatenated and individual embeddings.

    Args:
        df: DataFrame with permutation_key and sequence columns
        model_name: HuggingFace model name
        batch_size: Batch size for inference
        max_length: Maximum sequence length
        device: Device to run on

    Returns:
        - concat_embeddings: (N, D) array
        - individual_embeddings: dict {row_idx → [(mol_name, embedding)]}
        - metadata: dict with embedding_dim, model_name, etc.
    """
    # Extract concatenated embeddings
    print("\n" + "=" * 80)
    print("EXTRACTING CONCATENATED EMBEDDINGS")
    print("=" * 80)

    concat_sequences = df['sequence'].tolist()
    print(f"Extracting embeddings for {len(concat_sequences)} concatenated sequences...")

    concat_embeddings = extract_embeddings(
        concat_sequences,
        model_name,
        batch_size,
        max_length,
        device
    )

    print(f"Concatenated embeddings shape: {concat_embeddings.shape}")

    # Extract individual molecule embeddings
    individual_embeddings = extract_individual_embeddings_efficient(
        df,
        model_name,
        batch_size,
        max_length,
        device
    )

    # Create metadata
    metadata = {
        'model_name': model_name,
        'embedding_dim': concat_embeddings.shape[1],
        'n_samples': len(df),
        'device': device,
        'batch_size': batch_size,
        'max_length': max_length
    }

    return concat_embeddings, individual_embeddings, metadata


# ========================================================================
# METRICS COMPUTATION
# ========================================================================

def compute_mean_embeddings(
    individual_embeddings_dict: Dict[int, List[Tuple[str, np.ndarray]]]
) -> np.ndarray:
    """
    Compute mean of individual molecule embeddings for each row.

    Args:
        individual_embeddings_dict: {row_idx: [('tra', embedding), ('trb', embedding)]}

    Returns:
        mean_embeddings: (N, D) array
    """
    print("\n" + "=" * 80)
    print("COMPUTING MEAN EMBEDDINGS")
    print("=" * 80)

    # Sort by row index to ensure consistent ordering
    sorted_indices = sorted(individual_embeddings_dict.keys())

    mean_embeddings = []
    for row_idx in sorted_indices:
        mol_embeddings = [emb for _, emb in individual_embeddings_dict[row_idx]]
        # Compute mean across molecules
        mean_emb = np.mean(mol_embeddings, axis=0)
        mean_embeddings.append(mean_emb)

    mean_embeddings_array = np.array(mean_embeddings)
    print(f"Mean embeddings shape: {mean_embeddings_array.shape}")

    return mean_embeddings_array


def compute_all_metrics(
    concat_embeddings: np.ndarray,
    mean_embeddings: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Calculate comprehensive metrics.

    Args:
        concat_embeddings: (N, D) array of concatenated embeddings
        mean_embeddings: (N, D) array of mean embeddings

    Returns:
        Dictionary containing:
        - cosine_similarities: (N,) array
        - euclidean_distances: (N,) array
        - l2_norms_concat: (N,) array
        - l2_norms_mean: (N,) array
        - embedding_variance_concat: scalar
        - embedding_variance_mean: scalar
    """
    print("\n" + "=" * 80)
    print("COMPUTING METRICS")
    print("=" * 80)

    n_samples = concat_embeddings.shape[0]

    # Cosine similarities (row-wise)
    print("Computing cosine similarities...")
    cosine_sims = np.array([
        cosine_similarity([concat_embeddings[i]], [mean_embeddings[i]])[0, 0]
        for i in range(n_samples)
    ])

    # Euclidean distances (row-wise)
    print("Computing Euclidean distances...")
    euclidean_dists = np.array([
        euclidean_distances([concat_embeddings[i]], [mean_embeddings[i]])[0, 0]
        for i in range(n_samples)
    ])

    # L2 norms
    print("Computing L2 norms...")
    l2_norms_concat = np.linalg.norm(concat_embeddings, axis=1)
    l2_norms_mean = np.linalg.norm(mean_embeddings, axis=1)

    # Embedding variances
    print("Computing embedding variances...")
    embedding_variance_concat = np.var(concat_embeddings)
    embedding_variance_mean = np.var(mean_embeddings)

    metrics = {
        'cosine_similarities': cosine_sims,
        'euclidean_distances': euclidean_dists,
        'l2_norms_concat': l2_norms_concat,
        'l2_norms_mean': l2_norms_mean,
        'embedding_variance_concat': embedding_variance_concat,
        'embedding_variance_mean': embedding_variance_mean
    }

    # Print summary statistics
    print(f"\nMetrics Summary:")
    print(f"  Cosine similarity: {cosine_sims.mean():.4f} ± {cosine_sims.std():.4f}")
    print(f"  Euclidean distance: {euclidean_dists.mean():.4f} ± {euclidean_dists.std():.4f}")
    print(f"  L2 norm (concat): {l2_norms_concat.mean():.4f} ± {l2_norms_concat.std():.4f}")
    print(f"  L2 norm (mean): {l2_norms_mean.mean():.4f} ± {l2_norms_mean.std():.4f}")
    print(f"  Variance (concat): {embedding_variance_concat:.4f}")
    print(f"  Variance (mean): {embedding_variance_mean:.4f}")

    return metrics


# ========================================================================
# STATISTICAL ANALYSIS
# ========================================================================

def perform_statistical_tests(
    concat_embeddings: np.ndarray,
    mean_embeddings: np.ndarray,
    permutation_keys: List[str],
    metrics: Dict
) -> Dict[str, Any]:
    """
    Perform statistical tests.

    Tests:
    1. Paired t-test on L2 norms (concat vs mean)
    2. Paired t-test on cosine similarities
    3. ANOVA across permutation groups (if multiple combinations)
    4. Wilcoxon signed-rank test (non-parametric)
    5. Effect size calculations (Cohen's d)

    Args:
        concat_embeddings: (N, D) array
        mean_embeddings: (N, D) array
        permutation_keys: List of permutation keys for each row
        metrics: Dict from compute_all_metrics()

    Returns:
        Dict with test results, p-values, effect sizes
    """
    print("\n" + "=" * 80)
    print("PERFORMING STATISTICAL TESTS")
    print("=" * 80)

    results = {}

    # 1. Paired t-test on L2 norms
    print("\n1. Paired t-test: L2 norms (concat vs mean)")
    t_stat, p_value = stats.ttest_rel(
        metrics['l2_norms_concat'],
        metrics['l2_norms_mean']
    )
    results['l2_norms_ttest'] = {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05
    }
    print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.4e}")
    print(f"  Significant: {p_value < 0.05}")

    # 2. Wilcoxon signed-rank test on L2 norms (non-parametric)
    print("\n2. Wilcoxon signed-rank test: L2 norms")
    w_stat, p_value_w = stats.wilcoxon(
        metrics['l2_norms_concat'],
        metrics['l2_norms_mean']
    )
    results['l2_norms_wilcoxon'] = {
        'w_statistic': float(w_stat),
        'p_value': float(p_value_w),
        'significant': p_value_w < 0.05
    }
    print(f"  W-statistic: {w_stat:.4f}, p-value: {p_value_w:.4e}")

    # 3. Effect size (Cohen's d) for L2 norms
    print("\n3. Effect size (Cohen's d): L2 norms")
    diff = metrics['l2_norms_concat'] - metrics['l2_norms_mean']
    cohens_d = diff.mean() / diff.std()
    results['l2_norms_cohens_d'] = float(cohens_d)
    print(f"  Cohen's d: {cohens_d:.4f}")

    # 4. One-sample t-test: Are cosine similarities significantly different from 1?
    print("\n4. One-sample t-test: Cosine similarities vs 1.0")
    t_stat_cos, p_value_cos = stats.ttest_1samp(
        metrics['cosine_similarities'],
        1.0
    )
    results['cosine_similarity_ttest'] = {
        't_statistic': float(t_stat_cos),
        'p_value': float(p_value_cos),
        'significant': p_value_cos < 0.05
    }
    print(f"  t-statistic: {t_stat_cos:.4f}, p-value: {p_value_cos:.4e}")
    print(f"  Mean cosine similarity: {metrics['cosine_similarities'].mean():.4f}")

    # 5. Correlation between concat and mean embeddings
    print("\n5. Correlation between concat and mean embeddings")
    correlations = []
    for i in range(concat_embeddings.shape[0]):
        corr = np.corrcoef(concat_embeddings[i], mean_embeddings[i])[0, 1]
        correlations.append(corr)
    correlations = np.array(correlations)
    results['embedding_correlations'] = {
        'mean': float(correlations.mean()),
        'std': float(correlations.std()),
        'min': float(correlations.min()),
        'max': float(correlations.max())
    }
    print(f"  Mean correlation: {correlations.mean():.4f} ± {correlations.std():.4f}")

    # 6. ANOVA across permutation groups (if multiple)
    unique_keys = list(set(permutation_keys))
    if len(unique_keys) > 1:
        print(f"\n6. ANOVA across {len(unique_keys)} permutation groups")

        # Group cosine similarities by permutation key
        groups_cosine = [
            metrics['cosine_similarities'][np.array(permutation_keys) == key]
            for key in unique_keys
        ]

        f_stat, p_value_anova = stats.f_oneway(*groups_cosine)
        results['cosine_anova'] = {
            'f_statistic': float(f_stat),
            'p_value': float(p_value_anova),
            'significant': p_value_anova < 0.05,
            'groups': unique_keys
        }
        print(f"  F-statistic: {f_stat:.4f}, p-value: {p_value_anova:.4e}")
        print(f"  Significant differences across groups: {p_value_anova < 0.05}")

        # Group Euclidean distances by permutation key
        groups_euclidean = [
            metrics['euclidean_distances'][np.array(permutation_keys) == key]
            for key in unique_keys
        ]

        f_stat_euc, p_value_euc = stats.f_oneway(*groups_euclidean)
        results['euclidean_anova'] = {
            'f_statistic': float(f_stat_euc),
            'p_value': float(p_value_euc),
            'significant': p_value_euc < 0.05,
            'groups': unique_keys
        }
        print(f"  Euclidean ANOVA F-statistic: {f_stat_euc:.4f}, p-value: {p_value_euc:.4e}")

    return results


# ========================================================================
# VISUALIZATION FUNCTIONS
# ========================================================================

def create_summary_table(
    metrics: Dict,
    permutation_keys: List[str],
    output_path: Path
):
    """
    Create summary CSV with statistics per permutation key.

    Args:
        metrics: Dict from compute_all_metrics()
        permutation_keys: List of permutation keys for each row
        output_path: Path to save CSV
    """
    print("\n" + "=" * 80)
    print("CREATING SUMMARY TABLE")
    print("=" * 80)

    # Group by permutation key
    unique_keys = sorted(set(permutation_keys))
    permutation_keys_array = np.array(permutation_keys)

    summary_data = []
    for key in unique_keys:
        mask = permutation_keys_array == key
        n_samples = mask.sum()

        row = {
            'permutation_key': key,
            'n_samples': n_samples,
            'mean_cosine_sim': metrics['cosine_similarities'][mask].mean(),
            'std_cosine_sim': metrics['cosine_similarities'][mask].std(),
            'min_cosine_sim': metrics['cosine_similarities'][mask].min(),
            'max_cosine_sim': metrics['cosine_similarities'][mask].max(),
            'mean_euclidean_dist': metrics['euclidean_distances'][mask].mean(),
            'std_euclidean_dist': metrics['euclidean_distances'][mask].std(),
            'mean_l2_norm_concat': metrics['l2_norms_concat'][mask].mean(),
            'std_l2_norm_concat': metrics['l2_norms_concat'][mask].std(),
            'mean_l2_norm_mean': metrics['l2_norms_mean'][mask].mean(),
            'std_l2_norm_mean': metrics['l2_norms_mean'][mask].std(),
        }
        summary_data.append(row)

    # Add overall statistics
    summary_data.append({
        'permutation_key': 'OVERALL',
        'n_samples': len(permutation_keys),
        'mean_cosine_sim': metrics['cosine_similarities'].mean(),
        'std_cosine_sim': metrics['cosine_similarities'].std(),
        'min_cosine_sim': metrics['cosine_similarities'].min(),
        'max_cosine_sim': metrics['cosine_similarities'].max(),
        'mean_euclidean_dist': metrics['euclidean_distances'].mean(),
        'std_euclidean_dist': metrics['euclidean_distances'].std(),
        'mean_l2_norm_concat': metrics['l2_norms_concat'].mean(),
        'std_l2_norm_concat': metrics['l2_norms_concat'].std(),
        'mean_l2_norm_mean': metrics['l2_norms_mean'].mean(),
        'std_l2_norm_mean': metrics['l2_norms_mean'].std(),
    })

    # Create DataFrame and save
    summary_df = pd.DataFrame(summary_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False, float_format='%.6f')

    print(f"Saved summary table to: {output_path}")
    print(f"\n{summary_df.to_string(index=False)}")


def plot_metric_distributions(
    metrics: Dict,
    permutation_keys: List[str],
    output_dir: Path
):
    """
    Create distribution plots for metrics.

    Args:
        metrics: Dict from compute_all_metrics()
        permutation_keys: List of permutation keys for each row
        output_dir: Output directory for plots
    """
    print("\n" + "=" * 80)
    print("CREATING DISTRIBUTION PLOTS")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    unique_keys = sorted(set(permutation_keys))

    # 1. Cosine similarity histogram
    print("Creating cosine similarity histogram...")
    fig, ax = plt.subplots(figsize=(10, 6))
    for key in unique_keys:
        mask = np.array(permutation_keys) == key
        values = metrics['cosine_similarities'][mask]
        ax.hist(values, bins=30, alpha=0.6, label=key, edgecolor='black')

    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Cosine Similarities (Concat vs Mean)', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()

    path = output_dir / 'cosine_similarity_hist.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved to: {path}")

    # 2. Euclidean distance histogram
    print("Creating Euclidean distance histogram...")
    fig, ax = plt.subplots(figsize=(10, 6))
    for key in unique_keys:
        mask = np.array(permutation_keys) == key
        values = metrics['euclidean_distances'][mask]
        ax.hist(values, bins=30, alpha=0.6, label=key, edgecolor='black')

    ax.set_xlabel('Euclidean Distance', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Euclidean Distances (Concat vs Mean)', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()

    path = output_dir / 'euclidean_distance_hist.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved to: {path}")

    # 3. Violin plot: Cosine similarity by permutation key
    print("Creating cosine similarity violin plot...")
    fig, ax = plt.subplots(figsize=(12, 6))

    data_for_plot = []
    for key in unique_keys:
        mask = np.array(permutation_keys) == key
        values = metrics['cosine_similarities'][mask]
        for val in values:
            data_for_plot.append({'Permutation Key': key, 'Cosine Similarity': val})

    plot_df = pd.DataFrame(data_for_plot)
    sns.violinplot(data=plot_df, x='Permutation Key', y='Cosine Similarity', ax=ax)
    ax.set_title('Cosine Similarity by Permutation Key', fontsize=14, fontweight='bold')
    ax.set_xlabel('Permutation Key', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    path = output_dir / 'cosine_violin.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved to: {path}")

    # 4. Violin plot: Euclidean distance by permutation key
    print("Creating Euclidean distance violin plot...")
    fig, ax = plt.subplots(figsize=(12, 6))

    data_for_plot = []
    for key in unique_keys:
        mask = np.array(permutation_keys) == key
        values = metrics['euclidean_distances'][mask]
        for val in values:
            data_for_plot.append({'Permutation Key': key, 'Euclidean Distance': val})

    plot_df = pd.DataFrame(data_for_plot)
    sns.violinplot(data=plot_df, x='Permutation Key', y='Euclidean Distance', ax=ax)
    ax.set_title('Euclidean Distance by Permutation Key', fontsize=14, fontweight='bold')
    ax.set_xlabel('Permutation Key', fontsize=12)
    ax.set_ylabel('Euclidean Distance', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    path = output_dir / 'euclidean_violin.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved to: {path}")

    # 5. L2 norms comparison (box plot)
    print("Creating L2 norms comparison plot...")
    fig, ax = plt.subplots(figsize=(10, 6))

    data_for_plot = []
    for i, key in enumerate(permutation_keys):
        data_for_plot.append({
            'Permutation Key': key,
            'Type': 'Concatenated',
            'L2 Norm': metrics['l2_norms_concat'][i]
        })
        data_for_plot.append({
            'Permutation Key': key,
            'Type': 'Mean',
            'L2 Norm': metrics['l2_norms_mean'][i]
        })

    plot_df = pd.DataFrame(data_for_plot)
    sns.boxplot(data=plot_df, x='Type', y='L2 Norm', hue='Permutation Key', ax=ax)
    ax.set_title('L2 Norms: Concatenated vs Mean Embeddings', fontsize=14, fontweight='bold')
    ax.set_xlabel('Embedding Type', fontsize=12)
    ax.set_ylabel('L2 Norm', fontsize=12)
    plt.tight_layout()

    path = output_dir / 'l2_norms_comparison.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved to: {path}")


def compute_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'cosine',
    random_state: int = 42
) -> np.ndarray:
    """
    Compute UMAP projection.

    Args:
        embeddings: Array of shape (num_sequences, embedding_dim)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: Distance metric
        random_state: Random seed

    Returns:
        Array of shape (num_sequences, 2) with UMAP coordinates
    """
    try:
        import umap
    except ImportError:
        print("Warning: umap-learn not installed. Skipping UMAP.")
        return None

    print(f"Fitting UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric})...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=False
    )

    umap_coords = reducer.fit_transform(embeddings)
    return umap_coords


def plot_joint_embedding_space(
    concat_embeddings: np.ndarray,
    mean_embeddings: np.ndarray,
    permutation_keys: List[str],
    output_dir: Path,
    skip_umap: bool = False
):
    """
    Create UMAP and PCA visualizations of joint embedding space.

    Args:
        concat_embeddings: (N, D) array
        mean_embeddings: (N, D) array
        permutation_keys: List of permutation keys
        output_dir: Output directory
        skip_umap: Skip UMAP (use only PCA)
    """
    print("\n" + "=" * 80)
    print("CREATING JOINT EMBEDDING SPACE VISUALIZATIONS")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Stack embeddings and create labels
    stacked_embeddings = np.vstack([concat_embeddings, mean_embeddings])

    # Create labels for each point
    labels_type = ['Concatenated'] * len(concat_embeddings) + ['Mean'] * len(mean_embeddings)
    labels_pkey = permutation_keys + permutation_keys

    # PCA
    print("\nComputing PCA...")
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(stacked_embeddings)

    print(f"PCA explained variance: {pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f}")

    # Plot PCA
    print("Creating PCA plot...")
    fig, ax = plt.subplots(figsize=(12, 10))

    unique_keys = sorted(set(permutation_keys))
    markers = {'Concatenated': 'o', 'Mean': '^'}
    colors = sns.color_palette("husl", n_colors=len(unique_keys))
    color_map = {key: colors[i] for i, key in enumerate(unique_keys)}

    for pkey in unique_keys:
        for emb_type in ['Concatenated', 'Mean']:
            mask = (np.array(labels_pkey) == pkey) & (np.array(labels_type) == emb_type)
            ax.scatter(
                pca_coords[mask, 0],
                pca_coords[mask, 1],
                c=[color_map[pkey]],
                marker=markers[emb_type],
                label=f'{pkey} ({emb_type})',
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidth=0.5
            )

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('PCA: Concatenated vs Mean Embeddings', fontsize=14, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, fontsize=9)
    plt.tight_layout()

    path = output_dir / 'pca_joint_space.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved PCA plot to: {path}")

    # UMAP
    if not skip_umap:
        print("\nComputing UMAP...")
        umap_coords = compute_umap(stacked_embeddings)

        if umap_coords is not None:
            print("Creating UMAP plot...")
            fig, ax = plt.subplots(figsize=(12, 10))

            for pkey in unique_keys:
                for emb_type in ['Concatenated', 'Mean']:
                    mask = (np.array(labels_pkey) == pkey) & (np.array(labels_type) == emb_type)
                    ax.scatter(
                        umap_coords[mask, 0],
                        umap_coords[mask, 1],
                        c=[color_map[pkey]],
                        marker=markers[emb_type],
                        label=f'{pkey} ({emb_type})',
                        alpha=0.6,
                        s=50,
                        edgecolors='black',
                        linewidth=0.5
                    )

            ax.set_xlabel('UMAP 1', fontsize=12)
            ax.set_ylabel('UMAP 2', fontsize=12)
            ax.set_title('UMAP: Concatenated vs Mean Embeddings', fontsize=14, fontweight='bold')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, fontsize=9)
            plt.tight_layout()

            path = output_dir / 'umap_joint_space.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
            plt.close()
            print(f"Saved UMAP plot to: {path}")


def plot_similarity_heatmaps(
    concat_embeddings: np.ndarray,
    mean_embeddings: np.ndarray,
    permutation_keys: List[str],
    output_dir: Path
):
    """
    Create similarity heatmaps.

    Args:
        concat_embeddings: (N, D) array
        mean_embeddings: (N, D) array
        permutation_keys: List of permutation keys
        output_dir: Output directory
    """
    print("\n" + "=" * 80)
    print("CREATING SIMILARITY HEATMAPS")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pairwise cosine similarity matrix (sampled if too large)
    n_samples = concat_embeddings.shape[0]
    max_samples_for_heatmap = 100

    if n_samples > max_samples_for_heatmap:
        print(f"Sampling {max_samples_for_heatmap} points for similarity matrix (dataset has {n_samples} samples)...")
        sample_indices = np.random.choice(n_samples, max_samples_for_heatmap, replace=False)
        concat_sample = concat_embeddings[sample_indices]
        mean_sample = mean_embeddings[sample_indices]
        pkeys_sample = [permutation_keys[i] for i in sample_indices]
    else:
        concat_sample = concat_embeddings
        mean_sample = mean_embeddings
        pkeys_sample = permutation_keys

    print("Computing pairwise similarity matrix...")
    similarity_matrix = cosine_similarity(concat_sample, mean_sample)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        similarity_matrix,
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        center=0.5,
        square=True,
        cbar_kws={'label': 'Cosine Similarity'},
        ax=ax
    )
    ax.set_xlabel('Mean Embeddings (sample index)', fontsize=12)
    ax.set_ylabel('Concatenated Embeddings (sample index)', fontsize=12)
    ax.set_title('Pairwise Cosine Similarity: Concat vs Mean', fontsize=14, fontweight='bold')
    plt.tight_layout()

    path = output_dir / 'similarity_matrix.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved similarity matrix to: {path}")

    # 2. Per-permutation average similarity
    unique_keys = sorted(set(permutation_keys))
    if len(unique_keys) > 1:
        print("Computing per-permutation similarity matrix...")

        perm_similarity = np.zeros((len(unique_keys), len(unique_keys)))
        for i, key1 in enumerate(unique_keys):
            mask1 = np.array(permutation_keys) == key1
            concat_subset1 = concat_embeddings[mask1]

            for j, key2 in enumerate(unique_keys):
                mask2 = np.array(permutation_keys) == key2
                mean_subset2 = mean_embeddings[mask2]

                # Average similarity between all pairs
                sim_matrix = cosine_similarity(concat_subset1, mean_subset2)
                perm_similarity[i, j] = sim_matrix.mean()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            perm_similarity,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            center=0.5,
            xticklabels=unique_keys,
            yticklabels=unique_keys,
            square=True,
            cbar_kws={'label': 'Avg Cosine Similarity'},
            ax=ax
        )
        ax.set_xlabel('Mean Embeddings (permutation key)', fontsize=12)
        ax.set_ylabel('Concatenated Embeddings (permutation key)', fontsize=12)
        ax.set_title('Average Similarity by Permutation Key', fontsize=14, fontweight='bold')
        plt.tight_layout()

        path = output_dir / 'per_permutation_similarity.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved per-permutation similarity to: {path}")


# ========================================================================
# OUTPUT SAVING
# ========================================================================

def save_results(
    concat_embeddings: np.ndarray,
    mean_embeddings: np.ndarray,
    individual_embeddings: Dict,
    metrics: Dict,
    statistical_results: Dict,
    metadata: Dict,
    output_dir: Path
):
    """
    Save raw data and metadata to disk.

    Args:
        concat_embeddings: (N, D) array
        mean_embeddings: (N, D) array
        individual_embeddings: Dict from extract_individual_embeddings_efficient()
        metrics: Dict from compute_all_metrics()
        statistical_results: Dict from perform_statistical_tests()
        metadata: Dict with model info
        output_dir: Output directory
    """
    print("\n" + "=" * 80)
    print("SAVING RAW DATA")
    print("=" * 80)

    raw_data_dir = output_dir / 'raw_data'
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    np.save(raw_data_dir / 'concatenated_embeddings.npy', concat_embeddings)
    print(f"Saved concatenated embeddings: {raw_data_dir / 'concatenated_embeddings.npy'}")

    np.save(raw_data_dir / 'mean_embeddings.npy', mean_embeddings)
    print(f"Saved mean embeddings: {raw_data_dir / 'mean_embeddings.npy'}")

    # Save individual embeddings
    with open(raw_data_dir / 'individual_embeddings.pkl', 'wb') as f:
        pickle.dump(individual_embeddings, f)
    print(f"Saved individual embeddings: {raw_data_dir / 'individual_embeddings.pkl'}")

    # Save metrics
    with open(raw_data_dir / 'metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    print(f"Saved metrics: {raw_data_dir / 'metrics.pkl'}")

    # Save statistical results
    summary_dir = output_dir / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)

    with open(summary_dir / 'statistical_tests.json', 'w') as f:
        json.dump(statistical_results, f, indent=2)
    print(f"Saved statistical tests: {summary_dir / 'statistical_tests.json'}")

    # Save metadata
    with open(summary_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {summary_dir / 'metadata.json'}")


# ========================================================================
# MAIN CLI
# ========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare concatenated vs. mean molecule embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare TRA-TRB permutation
  python scripts/analysis/compare_concatenated_vs_mean_embeddings.py \\
      --deduplicated_data data/deduplicated/ \\
      --permutation_keys tra_trb \\
      --model_name facebook/esm2_t6_8M_UR50D \\
      --output_dir results/concat_vs_mean_tra_trb \\
      --sample 500

  # Compare multiple permutation keys
  python scripts/analysis/compare_concatenated_vs_mean_embeddings.py \\
      --deduplicated_data data/deduplicated/ \\
      --permutation_keys tra_trb peptide_mhc_one \\
      --model_name facebook/esm2_t33_650M_UR50D \\
      --output_dir results/multi_key_analysis \\
      --sample 1000

  # Test mode with small dataset
  python scripts/analysis/compare_concatenated_vs_mean_embeddings.py \\
      --deduplicated_data data/deduplicated/ \\
      --permutation_keys tra_trb \\
      --sample 10 \\
      --max_files 1 \\
      --skip_umap
        """
    )

    parser.add_argument(
        "--deduplicated_data",
        type=str,
        required=True,
        help="Path to deduplicated parquet directory"
    )
    parser.add_argument(
        "--permutation_keys",
        type=str,
        nargs='+',
        required=True,
        help="Permutation keys to analyze (e.g., tra_trb peptide_mhc_one)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/esm2_t6_8M_UR50D",
        help="HuggingFace model name (default: facebook/esm2_t6_8M_UR50D)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/embedding_comparison",
        help="Output directory (default: results/embedding_comparison)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=100,
        help="Number of sequences to sample per permutation key (default: 100)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding extraction (default: 32)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on (default: cuda)"
    )
    parser.add_argument(
        "--skip_umap",
        action="store_true",
        help="Skip UMAP computation (faster, only PCA)"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of parquet files to read (for testing)"
    )

    args = parser.parse_args()

    # Convert paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EMBEDDING COMPARISON ANALYSIS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data path: {args.deduplicated_data}")
    print(f"  Permutation keys: {', '.join(args.permutation_keys)}")
    print(f"  Model: {args.model_name}")
    print(f"  Output directory: {output_dir}")
    print(f"  Sample per permutation key: {args.sample}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {args.device}")
    print(f"  Skip UMAP: {args.skip_umap}")

    # Load data
    df = load_data_for_combinations(
        args.deduplicated_data,
        args.permutation_keys,
        sample_per_key=args.sample,
        max_files=args.max_files
    )

    # Extract embeddings
    concat_embeddings, individual_embeddings, metadata = extract_dual_embeddings(
        df,
        args.model_name,
        args.batch_size,
        args.max_length,
        args.device
    )

    # Compute mean embeddings
    mean_embeddings = compute_mean_embeddings(individual_embeddings)

    # Compute metrics
    metrics = compute_all_metrics(concat_embeddings, mean_embeddings)

    # Perform statistical tests
    permutation_keys = df['permutation_key'].tolist()
    statistical_results = perform_statistical_tests(
        concat_embeddings,
        mean_embeddings,
        permutation_keys,
        metrics
    )

    # Create visualizations
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    # Summary table
    create_summary_table(
        metrics,
        permutation_keys,
        output_dir / 'summary' / 'summary_statistics.csv'
    )

    # Distribution plots
    plot_metric_distributions(
        metrics,
        permutation_keys,
        output_dir / 'distributions'
    )

    # Joint embedding space (PCA/UMAP)
    plot_joint_embedding_space(
        concat_embeddings,
        mean_embeddings,
        permutation_keys,
        output_dir / 'embeddings',
        skip_umap=args.skip_umap
    )

    # Similarity heatmaps
    plot_similarity_heatmaps(
        concat_embeddings,
        mean_embeddings,
        permutation_keys,
        output_dir / 'heatmaps'
    )

    # Save raw data
    save_results(
        concat_embeddings,
        mean_embeddings,
        individual_embeddings,
        metrics,
        statistical_results,
        metadata,
        output_dir
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nResults:")
    print(f"  Summary statistics: {output_dir / 'summary' / 'summary_statistics.csv'}")
    print(f"  Statistical tests: {output_dir / 'summary' / 'statistical_tests.json'}")
    print(f"  Distribution plots: {output_dir / 'distributions'}")
    print(f"  Embedding visualizations: {output_dir / 'embeddings'}")
    print(f"  Heatmaps: {output_dir / 'heatmaps'}")
    print(f"  Raw data: {output_dir / 'raw_data'}")


if __name__ == "__main__":
    main()
