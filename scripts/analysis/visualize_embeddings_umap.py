#!/usr/bin/env python3
"""
Extract sequence embeddings from ESM2/ESM3 models and visualize with UMAP.

Usage:
    python scripts/analysis/visualize_embeddings_umap.py \
        --deduplicated_data data/deduplicated/ \
        --model_name facebook/esm2_t6_8M_UR50D \
        --output_dir results/umap_analysis \
        --sample 100

This script:
1. Loads deduplicated parquet files with permutation_key and sequence columns
2. Samples N sequences per permutation key (default: 100)
3. Extracts embeddings using specified ESM model
4. Performs UMAP dimensionality reduction
5. Creates scatter plot colored by permutation key
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import gc

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import umap

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


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


def collapse_permutation_key(key: str) -> str:
    """
    Collapse permutation keys so that tra_trb and trb_tra become (tra, trb).

    Properly handles multi-word components like mhc_one and mhc_two.

    Args:
        key: Original permutation key (e.g., 'tra_trb', 'trb_tra', 'peptide_mhc_one')

    Returns:
        Collapsed key as sorted tuple string (e.g., '(tra, trb)', '(mhc_one, peptide)')
    """
    # Define valid components in order of specificity (longer first to avoid partial matches)
    valid_components = ['mhc_one', 'mhc_two', 'peptide', 'tra', 'trb']

    # Extract components from the key
    components = []
    remaining = key

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
            # Fall back to just returning the original key
            print(f"Warning: Could not parse permutation key '{key}', using as-is")
            return key

    # Sort components alphabetically and return as tuple string
    sorted_components = sorted(components)
    return f"({', '.join(sorted_components)})"


def filter_by_components(key: str, required_components: List[str]) -> bool:
    """
    Check if a collapsed key contains all required components.

    Args:
        key: Collapsed key string (e.g., '(tra, trb, peptide)')
        required_components: List of required components (e.g., ['tra', 'trb', 'peptide'])

    Returns:
        True if key contains all required components, False otherwise
    """
    # Extract components from the key (remove parentheses and split by comma)
    key_components = set(c.strip() for c in key.strip('()').split(','))
    required_set = set(required_components)
    return required_set.issubset(key_components)


def load_and_sample_data(
    data_path: str,
    sample_per_key: int = 100,
    max_files: int = None,
    collapse_keys: bool = True,
    component_filter: List[str] = None
) -> pd.DataFrame:
    """
    Load deduplicated parquet files and sample sequences per permutation key.

    Args:
        data_path: Path to directory containing parquet files
        sample_per_key: Number of sequences to sample per permutation key
        max_files: Maximum number of parquet files to read (for testing)
        collapse_keys: Whether to collapse permutation keys (e.g., tra_trb and trb_tra -> (tra, trb))
        component_filter: List of components that must be present (e.g., ['tra', 'trb', 'peptide'])

    Returns:
        DataFrame with columns: permutation_key, collapsed_key (if collapse_keys=True), sequence
    """
    print("="*80)
    print("LOADING AND SAMPLING DATA")
    print("="*80)

    data_path = Path(data_path)
    parquet_files = sorted(data_path.glob("**/*.parquet"))

    if len(parquet_files) == 0:
        raise ValueError(f"No parquet files found in {data_path}")

    if max_files:
        parquet_files = parquet_files[:max_files]
        print(f"Limiting to first {max_files} files for testing")

    print(f"Found {len(parquet_files)} parquet files")

    # Read all parquet files
    dfs = []
    for pf in tqdm(parquet_files, desc="Reading parquet files"):
        df = pd.read_parquet(pf)
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(full_df):,}")

    # Collapse permutation keys if requested
    if collapse_keys:
        print("\nCollapsing permutation keys...")
        full_df['collapsed_key'] = full_df['permutation_key'].apply(collapse_permutation_key)

        # Show mapping
        mapping = full_df[['permutation_key', 'collapsed_key']].drop_duplicates().sort_values('collapsed_key')
        print("\nPermutation key mapping:")
        for _, row in mapping.iterrows():
            print(f"  {row['permutation_key']} -> {row['collapsed_key']}")

        grouping_key = 'collapsed_key'
    else:
        grouping_key = 'permutation_key'

    # Filter by components if requested
    if component_filter:
        print(f"\nFiltering combinations that contain: {', '.join(component_filter)}")
        initial_count = len(full_df)

        if collapse_keys:
            # Filter on collapsed keys
            mask = full_df['collapsed_key'].apply(
                lambda k: filter_by_components(k, component_filter)
            )
        else:
            # Filter on original permutation keys - need to collapse temporarily for filtering
            mask = full_df['permutation_key'].apply(
                lambda k: filter_by_components(collapse_permutation_key(k), component_filter)
            )

        full_df = full_df[mask].copy()
        filtered_count = len(full_df)
        print(f"Filtered {initial_count:,} -> {filtered_count:,} sequences "
              f"({filtered_count/initial_count*100:.1f}% retained)")

        if filtered_count == 0:
            raise ValueError(
                f"No sequences found with components: {component_filter}. "
                f"Available components: tra, trb, peptide, mhc_one, mhc_two"
            )

    # Count sequences per key
    key_counts = full_df[grouping_key].value_counts()
    print(f"\n{grouping_key} distribution:")
    for key, count in key_counts.items():
        print(f"  {key}: {count:,}")

    # Sample per key
    print(f"\nSampling {sample_per_key} sequences per {grouping_key}...")
    sampled_dfs = []
    for key in key_counts.index:
        key_df = full_df[full_df[grouping_key] == key]
        if len(key_df) > sample_per_key:
            key_df = key_df.sample(n=sample_per_key, random_state=42)
        sampled_dfs.append(key_df)

    sampled_df = pd.concat(sampled_dfs, ignore_index=True)
    print(f"\nTotal sampled sequences: {len(sampled_df):,}")

    # Show final distribution
    final_counts = sampled_df[grouping_key].value_counts()
    print(f"\nFinal distribution:")
    for key, count in final_counts.items():
        print(f"  {key}: {count:,}")

    return sampled_df


def extract_embeddings(
    sequences: List[str],
    model_name: str,
    batch_size: int = 32,
    max_length: int = 512,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Extract embeddings using ESM model.

    Args:
        sequences: List of amino acid sequences
        model_name: HuggingFace model name (e.g., facebook/esm2_t6_8M_UR50D)
        batch_size: Batch size for inference
        max_length: Maximum sequence length
        device: Device to run on

    Returns:
        numpy array of shape (num_sequences, embedding_dim)
    """
    print("\n" + "="*80)
    print("EXTRACTING EMBEDDINGS")
    print("="*80)

    # Check device
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = 'cpu'

    if device == 'cuda':
        num_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

    # Load model and tokenizer
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    embedding_dim = model.config.hidden_size
    print(f"Embedding dimension: {embedding_dim}")

    # Wrap in extractor
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

    print(f"\nExtracting embeddings (batch_size={batch_size})...")
    with torch.no_grad():
        for encoded, indices in tqdm(dataloader, desc="Extracting"):
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            embeddings = extractor(input_ids, attention_mask)
            embeddings_np = embeddings.cpu().numpy()
            all_embeddings.append(embeddings_np)

    # Concatenate all embeddings
    embeddings_array = np.vstack(all_embeddings)
    print(f"Extracted embeddings shape: {embeddings_array.shape}")

    # Clear GPU memory
    del model, extractor
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    return embeddings_array


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
    print("\n" + "="*80)
    print("COMPUTING UMAP")
    print("="*80)

    print(f"Parameters:")
    print(f"  n_neighbors: {n_neighbors}")
    print(f"  min_dist: {min_dist}")
    print(f"  metric: {metric}")

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=True
    )

    print("\nFitting UMAP...")
    umap_coords = reducer.fit_transform(embeddings)
    print(f"UMAP coordinates shape: {umap_coords.shape}")

    return umap_coords


def plot_umap(
    umap_coords: np.ndarray,
    labels: List[str],
    output_path: str,
    title: str = "UMAP Visualization of Sequence Embeddings",
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Create UMAP scatter plot colored by permutation key.

    Args:
        umap_coords: Array of shape (num_sequences, 2)
        labels: List of permutation keys (can be original or collapsed)
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
    """
    print("\n" + "="*80)
    print("CREATING VISUALIZATION")
    print("="*80)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique labels for color mapping
    unique_labels = sorted(set(labels))
    print(f"Number of unique keys: {len(unique_labels)}")

    # Create color palette
    if len(unique_labels) <= 10:
        palette = sns.color_palette("tab10", n_colors=len(unique_labels))
    else:
        palette = sns.color_palette("husl", n_colors=len(unique_labels))

    color_map = {label: palette[i] for i, label in enumerate(unique_labels)}

    # Plot each permutation key separately for legend
    for label in unique_labels:
        mask = np.array(labels) == label
        ax.scatter(
            umap_coords[mask, 0],
            umap_coords[mask, 1],
            c=[color_map[label]],
            label=label,
            alpha=0.6,
            s=20,
            edgecolors='none'
        )

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Create legend
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        frameon=True,
        title="Combination Key",
        title_fontsize=11,
        fontsize=10
    )

    plt.tight_layout()

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to: {output_path}")

    # Also save as PDF for publication quality
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved PDF to: {pdf_path}")

    plt.close()


def save_results(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    umap_coords: np.ndarray,
    output_dir: str
):
    """
    Save embeddings and UMAP coordinates to files.

    Args:
        df: DataFrame with sequence data
        embeddings: Embeddings array
        umap_coords: UMAP coordinates
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    embeddings_path = output_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings to: {embeddings_path}")

    # Save UMAP coordinates
    umap_path = output_dir / "umap_coords.npy"
    np.save(umap_path, umap_coords)
    print(f"Saved UMAP coordinates to: {umap_path}")

    # Save combined CSV
    results_df = df.copy()
    results_df['umap_1'] = umap_coords[:, 0]
    results_df['umap_2'] = umap_coords[:, 1]

    csv_path = output_dir / "results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved results CSV to: {csv_path}")

    # Save summary statistics
    summary_path = output_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("UMAP Embedding Analysis Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total sequences: {len(df)}\n")
        f.write(f"Embedding dimension: {embeddings.shape[1]}\n\n")

        # Show both original and collapsed key distributions if available
        if 'collapsed_key' in df.columns:
            f.write("Collapsed key distribution:\n")
            for key, count in df['collapsed_key'].value_counts().items():
                f.write(f"  {key}: {count}\n")
            f.write("\n")

        f.write("Original permutation key distribution:\n")
        for key, count in df['permutation_key'].value_counts().items():
            f.write(f"  {key}: {count}\n")

    print(f"Saved summary to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings and visualize with UMAP"
    )
    parser.add_argument(
        "--deduplicated_data",
        type=str,
        required=True,
        help="Path to deduplicated parquet directory"
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
        default="results/umap_analysis",
        help="Output directory for results"
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
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of parquet files to read (for testing)"
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter (default: 15)"
    )
    parser.add_argument(
        "--min_dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter (default: 0.1)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        help="UMAP distance metric (default: cosine)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on (default: cuda)"
    )
    parser.add_argument(
        "--no_collapse_keys",
        action="store_true",
        help="Don't collapse permutation keys (e.g., keep tra_trb and trb_tra separate)"
    )
    parser.add_argument(
        "--filter_components",
        type=str,
        nargs='+',
        default=None,
        help="Filter to only include combinations with these components (e.g., --filter_components tra trb peptide)"
    )

    args = parser.parse_args()

    # Validate filter components if provided
    if args.filter_components:
        valid_components = {'tra', 'trb', 'peptide', 'mhc_one', 'mhc_two'}
        invalid = set(args.filter_components) - valid_components
        if invalid:
            parser.error(
                f"Invalid components: {invalid}. "
                f"Valid components are: {', '.join(sorted(valid_components))}"
            )

    # Load and sample data
    df = load_and_sample_data(
        args.deduplicated_data,
        sample_per_key=args.sample,
        max_files=args.max_files,
        collapse_keys=not args.no_collapse_keys,
        component_filter=args.filter_components
    )

    # Extract embeddings
    embeddings = extract_embeddings(
        sequences=df['sequence'].tolist(),
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device
    )

    # Compute UMAP
    umap_coords = compute_umap(
        embeddings,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric
    )

    # Create visualization
    model_short_name = args.model_name.split('/')[-1]
    plot_path = Path(args.output_dir) / f"umap_{model_short_name}.png"

    # Use collapsed keys if available, otherwise use permutation_key
    label_column = 'collapsed_key' if 'collapsed_key' in df.columns else 'permutation_key'

    plot_umap(
        umap_coords,
        labels=df[label_column].tolist(),
        output_path=str(plot_path),
        title=f"UMAP Visualization - {model_short_name}"
    )

    # Save results
    save_results(df, embeddings, umap_coords, args.output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
