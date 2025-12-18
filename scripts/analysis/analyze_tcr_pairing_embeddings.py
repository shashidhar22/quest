#!/usr/bin/env python3
"""
Analyze TRA/TRB individual vs concatenated embeddings.

This script analyzes the relationship between individual TRA and TRB sequence
embeddings and their paired concatenated sequence embeddings using ESM2 models.

Usage:
    python scripts/analysis/analyze_tcr_pairing_embeddings.py \
        --deduplicated_data data/deduplicated/ \
        --model_name facebook/esm2_t6_8M_UR50D \
        --output_dir results/tcr_pairing_analysis \
        --sample_per_key 1000

This script:
1. Loads deduplicated parquet files with tra_trb and trb_tra permutation keys
2. Extracts ESM embeddings for concatenated and individual TRA/TRB sequences
3. Computes comprehensive similarity metrics
4. Analyzes ordering effects (tra_trb vs trb_tra)
5. Performs vector decomposition analysis
6. Creates comprehensive visualizations
7. Generates statistical test results and analysis report
"""

import argparse
import gc
import hashlib
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
# REUSED CLASSES FROM EXISTING SCRIPTS
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


# ========================================================================
# DATA LOADING & PARSING
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
            # If no component matched, fallback to split by underscore
            print(f"Warning: Could not fully parse permutation key '{pkey}', using fallback")
            return pkey.split('_')

    return components


def parse_tcr_pair(permutation_key: str, sequence: str) -> Tuple[str, str]:
    """
    Extract individual TRA and TRB sequences from concatenated sequence.

    Always returns in canonical order (TRA first, TRB second) regardless of
    permutation_key ordering.

    Args:
        permutation_key: Permutation key ('tra_trb' or 'trb_tra')
        sequence: Concatenated sequence (space-separated)

    Returns:
        (tra_sequence, trb_sequence) in canonical order

    Examples:
        parse_tcr_pair('tra_trb', 'CASSLG GILGFVFTL') → ('CASSLG', 'GILGFVFTL')
        parse_tcr_pair('trb_tra', 'GILGFVFTL CASSLG') → ('CASSLG', 'GILGFVFTL')
    """
    mol_types = parse_permutation_key(permutation_key)
    molecules = sequence.split()

    if len(molecules) != len(mol_types):
        raise ValueError(
            f"Mismatch: {len(molecules)} molecules but {len(mol_types)} types. "
            f"permutation_key={permutation_key}, sequence={sequence}"
        )

    # Create mapping
    mol_dict = dict(zip(mol_types, molecules))

    # Extract in canonical order
    tra_seq = mol_dict.get('tra', '')
    trb_seq = mol_dict.get('trb', '')

    return tra_seq, trb_seq


def create_pair_id(tra_seq: str, trb_seq: str) -> str:
    """
    Create unique identifier for canonical TRA/TRB pair.

    Args:
        tra_seq: TRA sequence
        trb_seq: TRB sequence

    Returns:
        MD5 hash of canonical pair
    """
    pair_str = f"{tra_seq}|{trb_seq}"
    return hashlib.md5(pair_str.encode()).hexdigest()[:16]


def load_tcr_pairing_data(
    data_path: str,
    sample_per_key: int = 1000,
    max_files: int = None
) -> pd.DataFrame:
    """
    Load deduplicated parquet files and filter for tra_trb and trb_tra.

    Args:
        data_path: Path to directory containing parquet files
        sample_per_key: Number of sequences to sample per permutation key
        max_files: Maximum number of parquet files to read (for testing)

    Returns:
        DataFrame with columns: permutation_key, sequence, tra_seq, trb_seq, pair_id
    """
    print("=" * 80)
    print("LOADING TCR PAIRING DATA")
    print("=" * 80)

    data_path = Path(data_path)
    parquet_files = sorted(data_path.glob("**/*.parquet"))

    if len(parquet_files) == 0:
        raise ValueError(f"No parquet files found in {data_path}")

    if max_files:
        parquet_files = parquet_files[:max_files]
        print(f"Limiting to first {max_files} files for testing")

    print(f"Found {len(parquet_files)} parquet files")
    print("Target permutation keys: tra_trb, trb_tra")

    # Read parquet files and filter
    dfs = []
    for pf in tqdm(parquet_files, desc="Reading parquet files"):
        df = pd.read_parquet(pf)
        # Filter to only include tra_trb and trb_tra
        df = df[df['permutation_key'].isin(['tra_trb', 'trb_tra'])]
        if len(df) > 0:
            dfs.append(df)

    if len(dfs) == 0:
        raise ValueError("No data found for permutation keys: tra_trb, trb_tra")

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
    for key in ['tra_trb', 'trb_tra']:
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

    # Parse TCR pairs
    print("\nParsing TCR pairs...")
    tra_seqs = []
    trb_seqs = []
    pair_ids = []

    for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Parsing"):
        tra_seq, trb_seq = parse_tcr_pair(row['permutation_key'], row['sequence'])
        tra_seqs.append(tra_seq)
        trb_seqs.append(trb_seq)
        pair_ids.append(create_pair_id(tra_seq, trb_seq))

    sampled_df['tra_seq'] = tra_seqs
    sampled_df['trb_seq'] = trb_seqs
    sampled_df['pair_id'] = pair_ids

    # Show final distribution
    final_counts = sampled_df['permutation_key'].value_counts()
    print(f"\nFinal distribution:")
    for key, count in final_counts.items():
        print(f"  {key}: {count:,}")

    # Check for pairs appearing in both orderings
    pair_counts = sampled_df['pair_id'].value_counts()
    pairs_in_both = (pair_counts > 1).sum()
    print(f"\nPairs appearing in both orderings: {pairs_in_both:,}")

    return sampled_df


# ========================================================================
# EMBEDDING EXTRACTION
# ========================================================================

def extract_embeddings_batch(
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
        model_name: HuggingFace model name
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


def extract_unique_embeddings(
    df: pd.DataFrame,
    model_name: str = 'facebook/esm2_t6_8M_UR50D',
    batch_size: int = 32,
    max_length: int = 512,
    device: str = 'cuda'
) -> Dict[str, np.ndarray]:
    """
    Extract embeddings with deduplication optimization.

    Strategy:
    1. Collect all unique concatenated sequences
    2. Collect all unique TRA sequences
    3. Collect all unique TRB sequences
    4. Extract embeddings once per unique sequence
    5. Store in dictionaries: {sequence_str: embedding_array}

    Args:
        df: DataFrame with permutation_key, sequence, tra_seq, trb_seq columns
        model_name: HuggingFace model name
        batch_size: Batch size for inference
        max_length: Maximum sequence length
        device: Device to run on

    Returns:
        {
            'concat': {sequence: embedding},
            'tra': {sequence: embedding},
            'trb': {sequence: embedding},
            'metadata': {...}
        }
    """
    print("\n" + "=" * 80)
    print("EXTRACTING UNIQUE EMBEDDINGS")
    print("=" * 80)

    # Collect unique sequences
    unique_concat = df['sequence'].unique().tolist()
    unique_tra = df['tra_seq'].unique().tolist()
    unique_trb = df['trb_seq'].unique().tolist()

    print(f"Unique concatenated sequences: {len(unique_concat):,}")
    print(f"Unique TRA sequences: {len(unique_tra):,}")
    print(f"Unique TRB sequences: {len(unique_trb):,}")

    total_unique = len(unique_concat) + len(unique_tra) + len(unique_trb)
    print(f"Total unique sequences to embed: {total_unique:,}")

    # Load model once for all extractions
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    embedding_dim = model.config.hidden_size
    print(f"Embedding dimension: {embedding_dim}")

    extractor = EmbeddingExtractor(model)
    extractor = extractor.to(device)
    extractor.eval()

    # Extract concatenated embeddings
    print(f"\nExtracting concatenated embeddings ({len(unique_concat):,} sequences)...")
    concat_embeddings = extract_embeddings_batch(
        unique_concat,
        model_name,
        batch_size,
        max_length,
        device,
        model=model,
        tokenizer=tokenizer,
        extractor=extractor
    )
    concat_dict = dict(zip(unique_concat, concat_embeddings))
    print(f"Extracted shape: {concat_embeddings.shape}")

    # Extract TRA embeddings
    print(f"\nExtracting TRA embeddings ({len(unique_tra):,} sequences)...")
    tra_embeddings = extract_embeddings_batch(
        unique_tra,
        model_name,
        batch_size,
        max_length,
        device,
        model=model,
        tokenizer=tokenizer,
        extractor=extractor
    )
    tra_dict = dict(zip(unique_tra, tra_embeddings))
    print(f"Extracted shape: {tra_embeddings.shape}")

    # Extract TRB embeddings
    print(f"\nExtracting TRB embeddings ({len(unique_trb):,} sequences)...")
    trb_embeddings = extract_embeddings_batch(
        unique_trb,
        model_name,
        batch_size,
        max_length,
        device,
        model=model,
        tokenizer=tokenizer,
        extractor=extractor
    )
    trb_dict = dict(zip(unique_trb, trb_embeddings))
    print(f"Extracted shape: {trb_embeddings.shape}")

    # Clear GPU memory
    del model, extractor
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    metadata = {
        'model_name': model_name,
        'embedding_dim': embedding_dim,
        'n_unique_concat': len(unique_concat),
        'n_unique_tra': len(unique_tra),
        'n_unique_trb': len(unique_trb),
        'device': device,
        'batch_size': batch_size,
        'max_length': max_length
    }

    return {
        'concat': concat_dict,
        'tra': tra_dict,
        'trb': trb_dict,
        'metadata': metadata
    }


def create_pairing_dataset(
    df: pd.DataFrame,
    unique_embeddings: Dict
) -> pd.DataFrame:
    """
    Map unique embeddings back to paired data structure.

    Args:
        df: DataFrame with permutation_key, sequence, tra_seq, trb_seq, pair_id columns
        unique_embeddings: Dict from extract_unique_embeddings()

    Returns:
        DataFrame with added columns:
        - concat_embedding: embedding of concatenated sequence
        - tra_embedding: embedding of TRA sequence
        - trb_embedding: embedding of TRB sequence
    """
    print("\n" + "=" * 80)
    print("MAPPING EMBEDDINGS TO PAIRS")
    print("=" * 80)

    concat_dict = unique_embeddings['concat']
    tra_dict = unique_embeddings['tra']
    trb_dict = unique_embeddings['trb']

    # Map embeddings
    concat_embeddings = []
    tra_embeddings = []
    trb_embeddings = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Mapping"):
        concat_embeddings.append(concat_dict[row['sequence']])
        tra_embeddings.append(tra_dict[row['tra_seq']])
        trb_embeddings.append(trb_dict[row['trb_seq']])

    # Add to dataframe
    df = df.copy()
    df['concat_embedding'] = concat_embeddings
    df['tra_embedding'] = tra_embeddings
    df['trb_embedding'] = trb_embeddings

    print(f"Mapped embeddings to {len(df):,} pairs")

    return df


# ========================================================================
# METRIC COMPUTATION
# ========================================================================

def compute_pairing_metrics(pairing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute comprehensive metrics for each pair.

    Args:
        pairing_df: DataFrame with concat_embedding, tra_embedding, trb_embedding columns

    Returns:
        DataFrame with added metric columns
    """
    print("\n" + "=" * 80)
    print("COMPUTING PAIRING METRICS")
    print("=" * 80)

    df = pairing_df.copy()
    n_samples = len(df)

    # Initialize metric lists
    metrics = {
        'cosine_concat_tra': [],
        'cosine_concat_trb': [],
        'cosine_tra_trb': [],
        'cosine_concat_mean': [],
        'euclidean_concat_mean': [],
        'l2_norm_concat': [],
        'l2_norm_tra': [],
        'l2_norm_trb': [],
        'l2_norm_mean': [],
    }

    print(f"Computing metrics for {n_samples:,} pairs...")

    for idx, row in tqdm(df.iterrows(), total=n_samples, desc="Computing"):
        concat_emb = row['concat_embedding']
        tra_emb = row['tra_embedding']
        trb_emb = row['trb_embedding']

        # Mean embedding
        mean_emb = (tra_emb + trb_emb) / 2

        # Cosine similarities
        metrics['cosine_concat_tra'].append(
            cosine_similarity([concat_emb], [tra_emb])[0, 0]
        )
        metrics['cosine_concat_trb'].append(
            cosine_similarity([concat_emb], [trb_emb])[0, 0]
        )
        metrics['cosine_tra_trb'].append(
            cosine_similarity([tra_emb], [trb_emb])[0, 0]
        )
        metrics['cosine_concat_mean'].append(
            cosine_similarity([concat_emb], [mean_emb])[0, 0]
        )

        # Euclidean distance
        metrics['euclidean_concat_mean'].append(
            euclidean_distances([concat_emb], [mean_emb])[0, 0]
        )

        # L2 norms
        metrics['l2_norm_concat'].append(np.linalg.norm(concat_emb))
        metrics['l2_norm_tra'].append(np.linalg.norm(tra_emb))
        metrics['l2_norm_trb'].append(np.linalg.norm(trb_emb))
        metrics['l2_norm_mean'].append(np.linalg.norm(mean_emb))

    # Add all metrics to dataframe
    for key, values in metrics.items():
        df[key] = values

    # Print summary
    print(f"\nMetric Summary:")
    print(f"  cosine(concat, TRA): {np.mean(metrics['cosine_concat_tra']):.4f} ± {np.std(metrics['cosine_concat_tra']):.4f}")
    print(f"  cosine(concat, TRB): {np.mean(metrics['cosine_concat_trb']):.4f} ± {np.std(metrics['cosine_concat_trb']):.4f}")
    print(f"  cosine(TRA, TRB): {np.mean(metrics['cosine_tra_trb']):.4f} ± {np.std(metrics['cosine_tra_trb']):.4f}")
    print(f"  cosine(concat, mean): {np.mean(metrics['cosine_concat_mean']):.4f} ± {np.std(metrics['cosine_concat_mean']):.4f}")
    print(f"  euclidean(concat, mean): {np.mean(metrics['euclidean_concat_mean']):.4f} ± {np.std(metrics['euclidean_concat_mean']):.4f}")

    return df


# ========================================================================
# ORDERING EFFECT ANALYSIS
# ========================================================================

def analyze_ordering_effect(pairing_df: pd.DataFrame) -> Dict:
    """
    Analyze the effect of ordering (tra_trb vs trb_tra) on embeddings.

    For pairs appearing in both orderings, compute similarity between
    the two concatenated embeddings.

    Args:
        pairing_df: DataFrame with pair_id, permutation_key, concat_embedding columns

    Returns:
        Dictionary with ordering analysis results
    """
    print("\n" + "=" * 80)
    print("ANALYZING ORDERING EFFECT")
    print("=" * 80)

    # Find pairs appearing in both orderings
    pair_counts = pairing_df['pair_id'].value_counts()
    pairs_in_both = pair_counts[pair_counts > 1].index.tolist()

    print(f"Pairs appearing in both orderings: {len(pairs_in_both):,}")

    if len(pairs_in_both) == 0:
        print("No pairs found in both orderings. Skipping analysis.")
        return {'n_pairs': 0, 'cosine_similarities': np.array([])}

    # For each pair, get embeddings from both orderings
    cosine_sims = []
    tra_trb_embeddings = []
    trb_tra_embeddings = []

    for pair_id in tqdm(pairs_in_both, desc="Analyzing pairs"):
        pair_rows = pairing_df[pairing_df['pair_id'] == pair_id]

        tra_trb_row = pair_rows[pair_rows['permutation_key'] == 'tra_trb']
        trb_tra_row = pair_rows[pair_rows['permutation_key'] == 'trb_tra']

        if len(tra_trb_row) == 0 or len(trb_tra_row) == 0:
            continue

        emb1 = tra_trb_row.iloc[0]['concat_embedding']
        emb2 = trb_tra_row.iloc[0]['concat_embedding']

        tra_trb_embeddings.append(emb1)
        trb_tra_embeddings.append(emb2)

        cos_sim = cosine_similarity([emb1], [emb2])[0, 0]
        cosine_sims.append(cos_sim)

    cosine_sims = np.array(cosine_sims)

    print(f"\nOrdering Effect Results:")
    print(f"  Pairs analyzed: {len(cosine_sims):,}")
    print(f"  Mean cosine similarity: {cosine_sims.mean():.4f} ± {cosine_sims.std():.4f}")
    print(f"  Min: {cosine_sims.min():.4f}")
    print(f"  Max: {cosine_sims.max():.4f}")

    return {
        'n_pairs': len(cosine_sims),
        'cosine_similarities': cosine_sims,
        'mean_similarity': float(cosine_sims.mean()) if len(cosine_sims) > 0 else 0.0,
        'std_similarity': float(cosine_sims.std()) if len(cosine_sims) > 0 else 0.0,
        'tra_trb_embeddings': np.array(tra_trb_embeddings) if tra_trb_embeddings else np.array([]),
        'trb_tra_embeddings': np.array(trb_tra_embeddings) if trb_tra_embeddings else np.array([])
    }


# ========================================================================
# VECTOR DECOMPOSITION ANALYSIS
# ========================================================================

def analyze_vector_decomposition(pairing_df: pd.DataFrame) -> Dict:
    """
    Analyze how concatenated embeddings relate to component embeddings.

    Tests hypothesis: concat ≈ α*TRA + β*TRB

    Args:
        pairing_df: DataFrame with concat_embedding, tra_embedding, trb_embedding columns

    Returns:
        Dictionary with decomposition analysis results
    """
    print("\n" + "=" * 80)
    print("ANALYZING VECTOR DECOMPOSITION")
    print("=" * 80)

    n_samples = len(pairing_df)

    # Extract embeddings as arrays
    concat_embeddings = np.vstack(pairing_df['concat_embedding'].values)
    tra_embeddings = np.vstack(pairing_df['tra_embedding'].values)
    trb_embeddings = np.vstack(pairing_df['trb_embedding'].values)

    # Test Hypothesis 1: concat ≈ mean(TRA, TRB)
    print("\nHypothesis 1: concat ≈ mean(TRA, TRB)")
    mean_embeddings = (tra_embeddings + trb_embeddings) / 2

    cosine_sims_mean = np.array([
        cosine_similarity([concat_embeddings[i]], [mean_embeddings[i]])[0, 0]
        for i in range(n_samples)
    ])
    euclidean_dists_mean = np.array([
        euclidean_distances([concat_embeddings[i]], [mean_embeddings[i]])[0, 0]
        for i in range(n_samples)
    ])

    print(f"  Cosine similarity: {cosine_sims_mean.mean():.4f} ± {cosine_sims_mean.std():.4f}")
    print(f"  Euclidean distance: {euclidean_dists_mean.mean():.4f} ± {euclidean_dists_mean.std():.4f}")

    # Test Hypothesis 2: concat ≈ α*TRA + β*TRB (weighted combination)
    print("\nHypothesis 2: concat ≈ α*TRA + β*TRB (solving for optimal α, β)")

    alphas = []
    betas = []
    reconstruction_errors = []

    for i in tqdm(range(n_samples), desc="Solving"):
        concat = concat_embeddings[i]
        tra = tra_embeddings[i]
        trb = trb_embeddings[i]

        # Solve least squares: concat = α*tra + β*trb
        # Stack as: [tra, trb] @ [α, β]^T = concat
        A = np.column_stack([tra, trb])
        try:
            weights, residuals, rank, s = np.linalg.lstsq(A, concat, rcond=None)
            alpha, beta = weights
            predicted = alpha * tra + beta * trb
            error = np.linalg.norm(concat - predicted)

            alphas.append(alpha)
            betas.append(beta)
            reconstruction_errors.append(error)
        except np.linalg.LinAlgError:
            alphas.append(0.5)
            betas.append(0.5)
            reconstruction_errors.append(np.inf)

    alphas = np.array(alphas)
    betas = np.array(betas)
    reconstruction_errors = np.array(reconstruction_errors)

    print(f"  Mean α: {alphas.mean():.4f} ± {alphas.std():.4f}")
    print(f"  Mean β: {betas.mean():.4f} ± {betas.std():.4f}")
    print(f"  Mean reconstruction error: {reconstruction_errors.mean():.4f} ± {reconstruction_errors.std():.4f}")

    return {
        'mean_hypothesis': {
            'cosine_similarities': cosine_sims_mean,
            'euclidean_distances': euclidean_dists_mean,
            'mean_cosine': float(cosine_sims_mean.mean()),
            'mean_euclidean': float(euclidean_dists_mean.mean())
        },
        'weighted_hypothesis': {
            'alphas': alphas,
            'betas': betas,
            'reconstruction_errors': reconstruction_errors,
            'mean_alpha': float(alphas.mean()),
            'mean_beta': float(betas.mean()),
            'mean_error': float(reconstruction_errors.mean())
        }
    }


# ========================================================================
# VISUALIZATION FUNCTIONS
# ========================================================================

def compute_umap_wrapper(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'cosine',
    random_state: int = 42
) -> np.ndarray:
    """Compute UMAP projection."""
    try:
        import umap
    except ImportError:
        print("Warning: umap-learn not installed. Skipping UMAP.")
        return None

    print(f"Computing UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric})...")
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
    pairing_df: pd.DataFrame,
    output_dir: Path,
    skip_umap: bool = False
):
    """
    Create UMAP and PCA visualizations showing TRA, TRB, and Concat together.

    Args:
        pairing_df: DataFrame with embeddings
        output_dir: Output directory
        skip_umap: Skip UMAP (use only PCA)
    """
    print("\n" + "=" * 80)
    print("CREATING JOINT EMBEDDING SPACE VISUALIZATIONS")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Stack all embeddings
    concat_embeddings = np.vstack(pairing_df['concat_embedding'].values)
    tra_embeddings = np.vstack(pairing_df['tra_embedding'].values)
    trb_embeddings = np.vstack(pairing_df['trb_embedding'].values)

    stacked_embeddings = np.vstack([concat_embeddings, tra_embeddings, trb_embeddings])

    n_samples = len(pairing_df)
    labels_type = ['Concat'] * n_samples + ['TRA'] * n_samples + ['TRB'] * n_samples
    labels_pkey = (pairing_df['permutation_key'].tolist() * 3)

    # PCA
    print("\nComputing PCA...")
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(stacked_embeddings)

    print(f"PCA explained variance: {pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f}")

    # Plot PCA
    print("Creating PCA plot...")
    fig, ax = plt.subplots(figsize=(14, 10))

    markers = {'Concat': 's', 'TRA': 'o', 'TRB': '^'}
    colors = {'tra_trb': '#1f77b4', 'trb_tra': '#ff7f0e'}

    for pkey in ['tra_trb', 'trb_tra']:
        for emb_type in ['Concat', 'TRA', 'TRB']:
            mask = (np.array(labels_pkey) == pkey) & (np.array(labels_type) == emb_type)
            ax.scatter(
                pca_coords[mask, 0],
                pca_coords[mask, 1],
                c=colors[pkey],
                marker=markers[emb_type],
                label=f'{pkey} - {emb_type}',
                alpha=0.5,
                s=30,
                edgecolors='black',
                linewidth=0.3
            )

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('PCA: TRA, TRB, and Concatenated Embeddings', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = output_dir / 'joint_space_pca.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved PCA plot to: {path}")

    # UMAP
    if not skip_umap:
        print("\nComputing UMAP...")
        umap_coords = compute_umap_wrapper(stacked_embeddings)

        if umap_coords is not None:
            print("Creating UMAP plot...")
            fig, ax = plt.subplots(figsize=(14, 10))

            for pkey in ['tra_trb', 'trb_tra']:
                for emb_type in ['Concat', 'TRA', 'TRB']:
                    mask = (np.array(labels_pkey) == pkey) & (np.array(labels_type) == emb_type)
                    ax.scatter(
                        umap_coords[mask, 0],
                        umap_coords[mask, 1],
                        c=colors[pkey],
                        marker=markers[emb_type],
                        label=f'{pkey} - {emb_type}',
                        alpha=0.5,
                        s=30,
                        edgecolors='black',
                        linewidth=0.3
                    )

            ax.set_xlabel('UMAP 1', fontsize=12)
            ax.set_ylabel('UMAP 2', fontsize=12)
            ax.set_title('UMAP: TRA, TRB, and Concatenated Embeddings', fontsize=14, fontweight='bold')
            ax.legend(loc='best', frameon=True, fontsize=9, ncol=2)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            path = output_dir / 'joint_space_umap.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
            plt.close()
            print(f"Saved UMAP plot to: {path}")


def plot_metric_distributions(
    pairing_df: pd.DataFrame,
    output_dir: Path
):
    """Create distribution plots for metrics."""
    print("\n" + "=" * 80)
    print("CREATING METRIC DISTRIBUTION PLOTS")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Multi-panel figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    metrics_to_plot = [
        ('cosine_concat_tra', 'Cosine Similarity: Concat vs TRA'),
        ('cosine_concat_trb', 'Cosine Similarity: Concat vs TRB'),
        ('cosine_tra_trb', 'Cosine Similarity: TRA vs TRB'),
        ('cosine_concat_mean', 'Cosine Similarity: Concat vs Mean(TRA, TRB)'),
        ('euclidean_concat_mean', 'Euclidean Distance: Concat vs Mean'),
        ('l2_norm_concat', 'L2 Norm: Concatenated Embedding')
    ]

    for idx, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[idx]

        for pkey in ['tra_trb', 'trb_tra']:
            data = pairing_df[pairing_df['permutation_key'] == pkey][metric]
            ax.hist(data, bins=30, alpha=0.6, label=pkey, edgecolor='black')

        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / 'metric_distributions.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved metric distributions to: {path}")


def plot_ordering_analysis(
    ordering_results: Dict,
    output_dir: Path
):
    """Plot ordering effect analysis."""
    print("\n" + "=" * 80)
    print("CREATING ORDERING EFFECT PLOTS")
    print("=" * 80)

    if ordering_results['n_pairs'] == 0:
        print("No pairs in both orderings. Skipping ordering plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    cosine_sims = ordering_results['cosine_similarities']

    # Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(cosine_sims, bins=30, edgecolor='black', alpha=0.7, color='#2ca02c')
    ax.axvline(cosine_sims.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {cosine_sims.mean():.4f}')
    ax.set_xlabel('Cosine Similarity (tra_trb vs trb_tra)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Ordering Effect: Similarity Between tra_trb and trb_tra\n(n={len(cosine_sims):,} pairs)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = output_dir / 'ordering_similarity_hist.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ordering histogram to: {path}")


def plot_decomposition_analysis(
    decomposition_results: Dict,
    output_dir: Path
):
    """Plot vector decomposition analysis."""
    print("\n" + "=" * 80)
    print("CREATING DECOMPOSITION PLOTS")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Weight distributions
    alphas = decomposition_results['weighted_hypothesis']['alphas']
    betas = decomposition_results['weighted_hypothesis']['betas']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(alphas, bins=30, edgecolor='black', alpha=0.7, color='#1f77b4')
    axes[0].axvline(alphas.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {alphas.mean():.3f}')
    axes[0].set_xlabel('α (TRA weight)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of α Weights', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(betas, bins=30, edgecolor='black', alpha=0.7, color='#ff7f0e')
    axes[1].axvline(betas.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {betas.mean():.3f}')
    axes[1].set_xlabel('β (TRB weight)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of β Weights', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / 'weight_distributions.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved weight distributions to: {path}")


# ========================================================================
# STATISTICAL TESTS
# ========================================================================

def perform_statistical_tests(
    pairing_df: pd.DataFrame,
    ordering_results: Dict,
    decomposition_results: Dict
) -> Dict:
    """
    Perform comprehensive statistical testing.

    Args:
        pairing_df: DataFrame with metrics
        ordering_results: Results from analyze_ordering_effect()
        decomposition_results: Results from analyze_vector_decomposition()

    Returns:
        Dictionary with all test results
    """
    print("\n" + "=" * 80)
    print("PERFORMING STATISTICAL TESTS")
    print("=" * 80)

    results = {}

    # Test 1: Is concat close to mean(TRA, TRB)?
    print("\n1. One-sample t-test: cosine(concat, mean) vs 1.0")
    cosine_concat_mean = pairing_df['cosine_concat_mean'].values
    t_stat, p_value = stats.ttest_1samp(cosine_concat_mean, 1.0)
    results['concat_vs_mean_ttest'] = {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'mean_similarity': float(cosine_concat_mean.mean())
    }
    print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.4e}")
    print(f"  Mean similarity: {cosine_concat_mean.mean():.4f}")

    # Test 2: Does ordering affect embeddings?
    if ordering_results['n_pairs'] > 0:
        print("\n2. Ordering effect test")
        ordering_sims = ordering_results['cosine_similarities']
        t_stat_order, p_value_order = stats.ttest_1samp(ordering_sims, 1.0)
        results['ordering_effect_ttest'] = {
            't_statistic': float(t_stat_order),
            'p_value': float(p_value_order),
            'significant': p_value_order < 0.05,
            'mean_similarity': float(ordering_sims.mean()),
            'n_pairs': int(ordering_results['n_pairs'])
        }
        print(f"  t-statistic: {t_stat_order:.4f}, p-value: {p_value_order:.4e}")
        print(f"  Mean similarity (tra_trb vs trb_tra): {ordering_sims.mean():.4f}")

    # Test 3: Correlation between concat and individual embeddings
    print("\n3. Correlation: concat vs TRA")
    cosine_concat_tra = pairing_df['cosine_concat_tra'].values
    print(f"  Mean: {cosine_concat_tra.mean():.4f} ± {cosine_concat_tra.std():.4f}")

    print("\n4. Correlation: concat vs TRB")
    cosine_concat_trb = pairing_df['cosine_concat_trb'].values
    print(f"  Mean: {cosine_concat_trb.mean():.4f} ± {cosine_concat_trb.std():.4f}")

    results['correlations'] = {
        'concat_tra': {
            'mean': float(cosine_concat_tra.mean()),
            'std': float(cosine_concat_tra.std())
        },
        'concat_trb': {
            'mean': float(cosine_concat_trb.mean()),
            'std': float(cosine_concat_trb.std())
        }
    }

    return results


# ========================================================================
# REPORT GENERATION
# ========================================================================

def generate_analysis_report(
    pairing_df: pd.DataFrame,
    ordering_results: Dict,
    decomposition_results: Dict,
    statistical_results: Dict,
    metadata: Dict,
    output_path: Path
):
    """Generate comprehensive analysis report."""
    print("\n" + "=" * 80)
    print("GENERATING ANALYSIS REPORT")
    print("=" * 80)

    report = f"""# TCR Pairing Embedding Analysis Report

## Executive Summary

- **Total pairs analyzed:** {len(pairing_df):,}
- **Unique TRA sequences:** {metadata.get('n_unique_tra', 'N/A'):,}
- **Unique TRB sequences:** {metadata.get('n_unique_trb', 'N/A'):,}
- **Permutation keys:** tra_trb ({len(pairing_df[pairing_df['permutation_key'] == 'tra_trb']):,}), trb_tra ({len(pairing_df[pairing_df['permutation_key'] == 'trb_tra']):,})
- **Model:** {metadata.get('model_name', 'N/A')}
- **Embedding dimension:** {metadata.get('embedding_dim', 'N/A')}

## Key Findings

### 1. Relationship Between Individual and Concatenated Embeddings

**Mean hypothesis: concat ≈ mean(TRA, TRB)**
- Mean cosine similarity: {decomposition_results['mean_hypothesis']['mean_cosine']:.4f}
- Mean Euclidean distance: {decomposition_results['mean_hypothesis']['mean_euclidean']:.4f}
- Statistical test: p-value = {statistical_results['concat_vs_mean_ttest']['p_value']:.4e}
- **Interpretation:** {'High' if decomposition_results['mean_hypothesis']['mean_cosine'] > 0.9 else 'Moderate' if decomposition_results['mean_hypothesis']['mean_cosine'] > 0.7 else 'Low'} similarity indicates that concatenated embeddings {'closely resemble' if decomposition_results['mean_hypothesis']['mean_cosine'] > 0.9 else 'partially resemble' if decomposition_results['mean_hypothesis']['mean_cosine'] > 0.7 else 'differ from'} the mean of individual chain embeddings.

**Similarity to individual chains:**
- cosine(concat, TRA): {statistical_results['correlations']['concat_tra']['mean']:.4f} ± {statistical_results['correlations']['concat_tra']['std']:.4f}
- cosine(concat, TRB): {statistical_results['correlations']['concat_trb']['mean']:.4f} ± {statistical_results['correlations']['concat_trb']['std']:.4f}

### 2. Effect of Ordering (tra_trb vs trb_tra)

{f'''- Pairs appearing in both orders: {ordering_results['n_pairs']:,}
- Mean cosine similarity between orderings: {ordering_results['mean_similarity']:.4f} ± {ordering_results['std_similarity']:.4f}
- Statistical test: p-value = {statistical_results.get('ordering_effect_ttest', {}).get('p_value', 'N/A')}
- **Conclusion:** {'Order does NOT significantly affect embeddings (high similarity)' if ordering_results['mean_similarity'] > 0.95 else 'Order DOES affect embeddings (moderate to low similarity)'}
''' if ordering_results['n_pairs'] > 0 else '- No pairs found in both orderings'}

### 3. Vector Decomposition Analysis

**Weighted combination: concat ≈ α·TRA + β·TRB**
- Mean α (TRA weight): {decomposition_results['weighted_hypothesis']['mean_alpha']:.4f}
- Mean β (TRB weight): {decomposition_results['weighted_hypothesis']['mean_beta']:.4f}
- Mean reconstruction error: {decomposition_results['weighted_hypothesis']['mean_error']:.4f}
- **Interpretation:** {'Both chains contribute roughly equally' if abs(decomposition_results['weighted_hypothesis']['mean_alpha'] - decomposition_results['weighted_hypothesis']['mean_beta']) < 0.2 else f"TRA contributes more" if decomposition_results['weighted_hypothesis']['mean_alpha'] > decomposition_results['weighted_hypothesis']['mean_beta'] else "TRB contributes more"} to the concatenated embedding.

## Statistical Tests Summary

1. **Concat vs Mean(TRA, TRB):** t-test p-value = {statistical_results['concat_vs_mean_ttest']['p_value']:.4e} ({'significant' if statistical_results['concat_vs_mean_ttest']['significant'] else 'not significant'})
{f"2. **Ordering Effect:** t-test p-value = {statistical_results.get('ordering_effect_ttest', {}).get('p_value', 'N/A')} ({('significant' if statistical_results.get('ordering_effect_ttest', {}).get('significant', False) else 'not significant')})" if ordering_results['n_pairs'] > 0 else "2. **Ordering Effect:** Not enough data (no pairs in both orderings)"}

## Visualizations Generated

- Joint embedding space (UMAP/PCA)
- Metric distributions
- Ordering effect analysis
- Vector decomposition (weight distributions)

## Recommendations

Based on this analysis:
- {"Concatenated embeddings can be reasonably approximated by averaging individual chain embeddings." if decomposition_results['mean_hypothesis']['mean_cosine'] > 0.9 else "Concatenated embeddings capture additional pairing information beyond simple averaging of individual chains."}
- {"Order of concatenation (TRA-TRB vs TRB-TRA) has minimal impact on embeddings, suggesting order-invariant representations." if ordering_results.get('mean_similarity', 0) > 0.95 else "Order of concatenation affects embeddings, suggesting the model encodes positional information."}
- {"Both TRA and TRB contribute roughly equally to the concatenated embedding." if abs(decomposition_results['weighted_hypothesis']['mean_alpha'] - decomposition_results['weighted_hypothesis']['mean_beta']) < 0.2 else f"{'TRA' if decomposition_results['weighted_hypothesis']['mean_alpha'] > decomposition_results['weighted_hypothesis']['mean_beta'] else 'TRB'} has stronger influence on the concatenated embedding."}

---
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Saved analysis report to: {output_path}")


# ========================================================================
# RESULT SAVING
# ========================================================================

def save_results(
    pairing_df: pd.DataFrame,
    unique_embeddings: Dict,
    ordering_results: Dict,
    decomposition_results: Dict,
    statistical_results: Dict,
    metadata: Dict,
    output_dir: Path
):
    """Save all raw data and results."""
    print("\n" + "=" * 80)
    print("SAVING RAW DATA AND RESULTS")
    print("=" * 80)

    # Save pairing dataframe
    raw_data_dir = output_dir / 'raw_data'
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    # Save as parquet (without embedding columns for efficiency)
    df_to_save = pairing_df.drop(columns=['concat_embedding', 'tra_embedding', 'trb_embedding'])
    df_to_save.to_parquet(raw_data_dir / 'pairing_df.parquet')
    print(f"Saved pairing dataframe (without embeddings): {raw_data_dir / 'pairing_df.parquet'}")

    # Save embeddings separately
    with open(raw_data_dir / 'unique_embeddings.pkl', 'wb') as f:
        pickle.dump(unique_embeddings, f)
    print(f"Saved unique embeddings: {raw_data_dir / 'unique_embeddings.pkl'}")

    # Save analysis results
    summary_dir = output_dir / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)

    with open(summary_dir / 'ordering_results.json', 'w') as f:
        json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in ordering_results.items()}, f, indent=2)

    with open(summary_dir / 'decomposition_results.json', 'w') as f:
        json.dump({k: {k2: v2.tolist() if isinstance(v2, np.ndarray) else v2 for k2, v2 in v.items()} if isinstance(v, dict) else v for k, v in decomposition_results.items()}, f, indent=2)

    # Convert numpy/pandas types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (bool, int, float, str, type(None))):
            return obj
        else:
            return str(obj)

    with open(summary_dir / 'statistical_tests.json', 'w') as f:
        json.dump(convert_to_serializable(statistical_results), f, indent=2)

    with open(summary_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved all results to: {summary_dir}")

    # Save metrics summary CSV
    metrics_cols = [c for c in pairing_df.columns if c.startswith('cosine_') or c.startswith('euclidean_') or c.startswith('l2_norm_')]
    summary_stats = pairing_df.groupby('permutation_key')[metrics_cols].agg(['mean', 'std', 'min', 'max'])
    summary_stats.to_csv(summary_dir / 'metrics_summary.csv')
    print(f"Saved metrics summary: {summary_dir / 'metrics_summary.csv'}")


# ========================================================================
# MAIN CLI
# ========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze TRA/TRB individual vs concatenated embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--deduplicated_data",
        type=str,
        required=True,
        help="Path to deduplicated parquet directory"
    )

    # Optional arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/esm2_t6_8M_UR50D",
        help="ESM model name (default: esm2_t6_8M_UR50D)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/tcr_pairing_analysis",
        help="Output directory (default: results/tcr_pairing_analysis)"
    )

    parser.add_argument(
        "--sample_per_key",
        type=int,
        default=1000,
        help="Sequences to sample per permutation key (default: 1000)"
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
        help="Device (default: cuda)"
    )

    parser.add_argument(
        "--skip_umap",
        action="store_true",
        help="Skip UMAP (use only PCA for faster analysis)"
    )

    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Limit parquet files for testing (default: all)"
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # Convert paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TCR PAIRING EMBEDDING ANALYSIS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data path: {args.deduplicated_data}")
    print(f"  Model: {args.model_name}")
    print(f"  Output directory: {output_dir}")
    print(f"  Sample per permutation key: {args.sample_per_key}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {args.device}")
    print(f"  Skip UMAP: {args.skip_umap}")

    # Load data
    pairing_df = load_tcr_pairing_data(
        args.deduplicated_data,
        sample_per_key=args.sample_per_key,
        max_files=args.max_files
    )

    # Extract embeddings
    unique_embeddings = extract_unique_embeddings(
        pairing_df,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device
    )

    # Map embeddings to pairs
    pairing_df = create_pairing_dataset(pairing_df, unique_embeddings)

    # Compute metrics
    pairing_df = compute_pairing_metrics(pairing_df)

    # Analyze ordering effect
    ordering_results = analyze_ordering_effect(pairing_df)

    # Analyze vector decomposition
    decomposition_results = analyze_vector_decomposition(pairing_df)

    # Perform statistical tests
    statistical_results = perform_statistical_tests(
        pairing_df,
        ordering_results,
        decomposition_results
    )

    # Create visualizations
    plot_joint_embedding_space(
        pairing_df,
        output_dir / 'embeddings',
        skip_umap=args.skip_umap
    )

    plot_metric_distributions(
        pairing_df,
        output_dir / 'distributions'
    )

    plot_ordering_analysis(
        ordering_results,
        output_dir / 'ordering'
    )

    plot_decomposition_analysis(
        decomposition_results,
        output_dir / 'decomposition'
    )

    # Generate report
    generate_analysis_report(
        pairing_df,
        ordering_results,
        decomposition_results,
        statistical_results,
        unique_embeddings['metadata'],
        output_dir / 'summary' / 'analysis_report.md'
    )

    # Save results
    save_results(
        pairing_df,
        unique_embeddings,
        ordering_results,
        decomposition_results,
        statistical_results,
        unique_embeddings['metadata'],
        output_dir
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nResults:")
    print(f"  Analysis report: {output_dir / 'summary' / 'analysis_report.md'}")
    print(f"  Metrics summary: {output_dir / 'summary' / 'metrics_summary.csv'}")
    print(f"  Visualizations:")
    print(f"    - Joint embedding space: {output_dir / 'embeddings'}")
    print(f"    - Metric distributions: {output_dir / 'distributions'}")
    print(f"    - Ordering analysis: {output_dir / 'ordering'}")
    print(f"    - Decomposition: {output_dir / 'decomposition'}")
    print(f"  Raw data: {output_dir / 'raw_data'}")


if __name__ == "__main__":
    main()
