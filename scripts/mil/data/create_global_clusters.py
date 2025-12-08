#!/usr/bin/env python3
"""
Global Clustering for MIL Pipeline (CPU-Optimized)

Performs global clustering across all unique sequences in a dataset.
Optimized for large CPU instances (e.g., x2gd.16xlarge with 64 vCPUs, 512GB RAM).

Supports:
- K-means: Uses sklearn MiniBatchKMeans (CPU) or cuML KMeans (GPU if available)
- Leiden: Uses leidenalg with parallel k-NN graph construction

Output:
- Cluster centroids
- Sequence-to-cluster mapping
- Cluster quality metrics
"""

import argparse
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
import gc
import warnings
warnings.filterwarnings('ignore')

# For parallel processing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

# Clustering libraries
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

# Check for GPU support
try:
    import cupy as cp
    from cuml.cluster import KMeans as cuKMeans
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Leiden clustering
try:
    import igraph as ig
    import leidenalg
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False


def load_embeddings_chunked(embeddings_path: str, chunk_size: int = 100000):
    """
    Load embeddings in chunks for memory efficiency.
    
    Returns:
        sequences: list of sequence IDs
        embeddings: numpy array of shape (n_sequences, embedding_dim)
    """
    print(f"Loading embeddings from: {embeddings_path}")
    
    pf = pq.ParquetFile(embeddings_path)
    total_rows = pf.metadata.num_rows
    print(f"  Total sequences: {total_rows:,}")
    
    # Read in chunks
    all_sequences = []
    all_embeddings = []
    
    for batch in tqdm(pf.iter_batches(batch_size=chunk_size), 
                      total=(total_rows + chunk_size - 1) // chunk_size,
                      desc="Loading embeddings"):
        df = batch.to_pandas()
        all_sequences.extend(df['sequence_id'].tolist())
        
        # Handle embedding column (can be list or fixed-size array)
        if isinstance(df['embedding'].iloc[0], (list, np.ndarray)):
            embeddings = np.vstack(df['embedding'].values)
        else:
            embeddings = np.vstack([np.array(e) for e in df['embedding']])
        all_embeddings.append(embeddings)
        
        del df
        gc.collect()
    
    embeddings = np.vstack(all_embeddings)
    del all_embeddings
    gc.collect()
    
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Memory: {embeddings.nbytes / 1e9:.2f} GB")
    
    return all_sequences, embeddings


def load_counts_for_sequences(dataset_dir: str, sequences: list, dataset_num: int) -> np.ndarray:
    """
    Load count/frequency information for sequences.
    
    For datasets 7-8: Uses 'templates' column
    For datasets 1-6: Returns uniform weights (1.0)
    """
    # Check if this dataset has counts
    if dataset_num not in [7, 8]:
        print(f"  Dataset {dataset_num} has no count column, using uniform weights")
        return np.ones(len(sequences), dtype=np.float32)
    
    print(f"  Loading counts for dataset {dataset_num} (has 'templates' column)")
    
    # Build sequence to count mapping from TSV files
    sequence_counts = {}
    tsv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.tsv')]
    
    for tsv_file in tqdm(tsv_files, desc="Loading counts from TSVs"):
        df = pd.read_csv(os.path.join(dataset_dir, tsv_file), sep='\t')
        if 'templates' in df.columns and 'junction_aa' in df.columns:
            for _, row in df.iterrows():
                seq = row['junction_aa']
                count = row['templates']
                if seq in sequence_counts:
                    sequence_counts[seq] += count
                else:
                    sequence_counts[seq] = count
        del df
    
    # Map to our sequence list
    counts = np.array([sequence_counts.get(seq, 1.0) for seq in sequences], dtype=np.float32)
    print(f"  Loaded counts for {len(sequence_counts):,} sequences")
    print(f"  Count range: [{counts.min():.0f}, {counts.max():.0f}], mean: {counts.mean():.1f}")
    
    return counts


def run_kmeans_clustering(
    embeddings: np.ndarray,
    n_clusters: int = 100,
    sample_weights: np.ndarray = None,
    batch_size: int = 10000,
    max_iter: int = 100,
    n_init: int = 3,
    random_state: int = 42,
    use_gpu: bool = False,
) -> tuple:
    """
    Run K-means clustering with optional sample weighting.
    
    Returns:
        labels: cluster assignments
        centroids: cluster centers
        metrics: dict with quality metrics
    """
    print(f"\nRunning K-means clustering (K={n_clusters})")
    
    # Normalize embeddings
    print("  Normalizing embeddings...")
    embeddings_norm = normalize(embeddings, norm='l2')
    
    if use_gpu and GPU_AVAILABLE:
        print("  Using GPU (cuML KMeans)")
        import cupy as cp
        from cuml.cluster import KMeans as cuKMeans
        
        embeddings_gpu = cp.asarray(embeddings_norm)
        weights_gpu = cp.asarray(sample_weights) if sample_weights is not None else None
        
        kmeans = cuKMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
        )
        
        if weights_gpu is not None:
            kmeans.fit(embeddings_gpu, sample_weight=weights_gpu)
        else:
            kmeans.fit(embeddings_gpu)
        
        labels = cp.asnumpy(kmeans.labels_)
        centroids = cp.asnumpy(kmeans.cluster_centers_)
        
        del embeddings_gpu, weights_gpu
        cp.get_default_memory_pool().free_all_blocks()
    else:
        print(f"  Using CPU (MiniBatchKMeans, batch_size={batch_size})")
        
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
            verbose=1,
        )
        
        # Note: sklearn MiniBatchKMeans doesn't support sample_weight in fit()
        # We'll use it for centroid refinement if provided
        kmeans.fit(embeddings_norm)
        
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        
        # Refine centroids with weights if provided
        if sample_weights is not None:
            print("  Refining centroids with sample weights...")
            weighted_centroids = np.zeros_like(centroids)
            for k in range(n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    weights_k = sample_weights[mask]
                    embeddings_k = embeddings_norm[mask]
                    weighted_centroids[k] = np.average(embeddings_k, weights=weights_k, axis=0)
                else:
                    weighted_centroids[k] = centroids[k]
            centroids = normalize(weighted_centroids, norm='l2')
    
    # Compute quality metrics on sample
    print("  Computing cluster quality metrics...")
    sample_size = min(50000, len(embeddings_norm))
    sample_idx = np.random.choice(len(embeddings_norm), sample_size, replace=False)
    
    try:
        sil_score = silhouette_score(
            embeddings_norm[sample_idx], 
            labels[sample_idx], 
            metric='cosine',
            sample_size=min(10000, sample_size)
        )
    except Exception as e:
        print(f"  Warning: Could not compute silhouette score: {e}")
        sil_score = -1.0
    
    try:
        db_score = davies_bouldin_score(embeddings_norm[sample_idx], labels[sample_idx])
    except Exception as e:
        print(f"  Warning: Could not compute Davies-Bouldin score: {e}")
        db_score = -1.0
    
    # Cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    
    metrics = {
        'method': 'kmeans',
        'n_clusters': n_clusters,
        'silhouette_score': float(sil_score),
        'davies_bouldin_score': float(db_score),
        'cluster_sizes_mean': float(counts.mean()),
        'cluster_sizes_std': float(counts.std()),
        'cluster_sizes_min': int(counts.min()),
        'cluster_sizes_max': int(counts.max()),
        'inertia': float(kmeans.inertia_) if hasattr(kmeans, 'inertia_') else -1.0,
    }
    
    print(f"  Results: {n_clusters} clusters")
    print(f"    Silhouette score: {sil_score:.4f}")
    print(f"    Davies-Bouldin score: {db_score:.4f}")
    print(f"    Cluster sizes: min={counts.min()}, max={counts.max()}, mean={counts.mean():.0f}")
    
    del embeddings_norm
    gc.collect()
    
    return labels, centroids, metrics


def build_knn_graph_parallel(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    metric: str = 'cosine',
    n_jobs: int = -1,
    chunk_size: int = 50000,
) -> tuple:
    """
    Build k-NN graph in parallel chunks for large datasets.
    
    Returns:
        sources: edge source indices
        targets: edge target indices  
        weights: edge weights (similarity)
    """
    n_samples = len(embeddings)
    
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    print(f"  Building k-NN graph (k={n_neighbors}, metric={metric}, jobs={n_jobs})")
    
    # Normalize for cosine similarity
    embeddings_norm = normalize(embeddings, norm='l2')
    
    # Fit NearestNeighbors on full data
    print(f"  Fitting NearestNeighbors on {n_samples:,} points...")
    nn = NearestNeighbors(
        n_neighbors=n_neighbors + 1,  # +1 because point is its own neighbor
        metric='euclidean' if metric == 'cosine' else metric,  # cosine on normalized = euclidean
        algorithm='auto',
        n_jobs=n_jobs,
    )
    nn.fit(embeddings_norm)
    
    # Query in chunks
    all_sources = []
    all_targets = []
    all_weights = []
    
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    
    for chunk_idx in tqdm(range(n_chunks), desc="Building k-NN graph"):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_samples)
        
        distances, indices = nn.kneighbors(embeddings_norm[start_idx:end_idx])
        
        # Convert to edges (excluding self-loops)
        for local_i, (dists, nbrs) in enumerate(zip(distances, indices)):
            global_i = start_idx + local_i
            for d, j in zip(dists[1:], nbrs[1:]):  # Skip first (self)
                # Weight = similarity (1 - normalized_distance for cosine)
                if metric == 'cosine':
                    weight = 1.0 - (d ** 2) / 2.0  # Convert euclidean on unit sphere to cosine sim
                else:
                    weight = 1.0 / (1.0 + d)
                
                all_sources.append(global_i)
                all_targets.append(j)
                all_weights.append(max(0, weight))
    
    del embeddings_norm, nn
    gc.collect()
    
    return np.array(all_sources), np.array(all_targets), np.array(all_weights)


def run_leiden_clustering(
    embeddings: np.ndarray,
    sample_weights: np.ndarray = None,
    n_neighbors: int = 15,
    resolution: float = 1.0,
    metric: str = 'cosine',
    n_jobs: int = -1,
    random_state: int = 42,
) -> tuple:
    """
    Run Leiden clustering on embeddings.
    
    Returns:
        labels: cluster assignments
        centroids: cluster centers (weighted by sample_weights if provided)
        metrics: dict with quality metrics
    """
    if not LEIDEN_AVAILABLE:
        raise ImportError("Leiden clustering requires: pip install igraph leidenalg")
    
    print(f"\nRunning Leiden clustering (resolution={resolution})")
    
    # Build k-NN graph
    sources, targets, weights = build_knn_graph_parallel(
        embeddings, n_neighbors, metric, n_jobs
    )
    
    print(f"  Graph edges: {len(sources):,}")
    
    # Create igraph
    print("  Creating igraph...")
    n_vertices = len(embeddings)
    edges = list(zip(sources.tolist(), targets.tolist()))
    
    g = ig.Graph(n=n_vertices, edges=edges, directed=False)
    g.es['weight'] = weights.tolist()
    
    # Incorporate sample weights into edge weights if provided
    if sample_weights is not None:
        print("  Incorporating sample weights into edges...")
        new_weights = []
        for e in g.es:
            src, tgt = e.source, e.target
            # Multiply edge weight by geometric mean of endpoint weights
            w = e['weight'] * np.sqrt(sample_weights[src] * sample_weights[tgt])
            new_weights.append(w)
        g.es['weight'] = new_weights
    
    del sources, targets, weights, edges
    gc.collect()
    
    # Run Leiden
    print(f"  Running Leiden algorithm...")
    np.random.seed(random_state)
    
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter=resolution,
        seed=random_state,
    )
    
    labels = np.array(partition.membership)
    n_clusters = len(set(labels))
    modularity = partition.modularity
    
    print(f"  Found {n_clusters} clusters, modularity={modularity:.4f}")
    
    del g, partition
    gc.collect()
    
    # Compute centroids
    print("  Computing cluster centroids...")
    embeddings_norm = normalize(embeddings, norm='l2')
    centroids = np.zeros((n_clusters, embeddings.shape[1]), dtype=np.float32)
    
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() > 0:
            if sample_weights is not None:
                weights_k = sample_weights[mask]
                centroids[k] = np.average(embeddings_norm[mask], weights=weights_k, axis=0)
            else:
                centroids[k] = embeddings_norm[mask].mean(axis=0)
    
    centroids = normalize(centroids, norm='l2')
    
    # Compute quality metrics
    print("  Computing cluster quality metrics...")
    sample_size = min(50000, len(embeddings_norm))
    sample_idx = np.random.choice(len(embeddings_norm), sample_size, replace=False)
    
    try:
        sil_score = silhouette_score(
            embeddings_norm[sample_idx],
            labels[sample_idx],
            metric='cosine',
            sample_size=min(10000, sample_size)
        )
    except Exception as e:
        print(f"  Warning: Could not compute silhouette score: {e}")
        sil_score = -1.0
    
    try:
        db_score = davies_bouldin_score(embeddings_norm[sample_idx], labels[sample_idx])
    except Exception as e:
        print(f"  Warning: Could not compute Davies-Bouldin score: {e}")
        db_score = -1.0
    
    # Cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    
    metrics = {
        'method': 'leiden',
        'n_clusters': n_clusters,
        'resolution': resolution,
        'n_neighbors': n_neighbors,
        'modularity': float(modularity),
        'silhouette_score': float(sil_score),
        'davies_bouldin_score': float(db_score),
        'cluster_sizes_mean': float(counts.mean()),
        'cluster_sizes_std': float(counts.std()),
        'cluster_sizes_min': int(counts.min()),
        'cluster_sizes_max': int(counts.max()),
    }
    
    print(f"  Results: {n_clusters} clusters")
    print(f"    Modularity: {modularity:.4f}")
    print(f"    Silhouette score: {sil_score:.4f}")
    print(f"    Davies-Bouldin score: {db_score:.4f}")
    print(f"    Cluster sizes: min={counts.min()}, max={counts.max()}, mean={counts.mean():.0f}")
    
    del embeddings_norm
    gc.collect()
    
    return labels, centroids, metrics


def save_clustering_results(
    output_dir: str,
    method: str,
    sequences: list,
    labels: np.ndarray,
    centroids: np.ndarray,
    metrics: dict,
    sample_weights: np.ndarray = None,
):
    """Save clustering results to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save sequence-to-cluster mapping
    mapping_file = output_path / f"{method}_sequence_clusters.parquet"
    mapping_df = pd.DataFrame({
        'sequence_id': sequences,
        'cluster_id': labels,
    })
    if sample_weights is not None:
        mapping_df['weight'] = sample_weights
    mapping_df.to_parquet(mapping_file, index=False)
    print(f"  Saved mapping: {mapping_file}")
    
    # Save centroids
    centroids_file = output_path / f"{method}_centroids.npy"
    np.save(centroids_file, centroids)
    print(f"  Saved centroids: {centroids_file}")
    
    # Save metrics
    metrics_file = output_path / f"{method}_metrics.json"
    metrics['timestamp'] = datetime.now().isoformat()
    metrics['num_sequences'] = len(sequences)
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics: {metrics_file}")
    
    return mapping_file, centroids_file, metrics_file


def main():
    parser = argparse.ArgumentParser(
        description="Global clustering for MIL pipeline (CPU-optimized)"
    )
    parser.add_argument("--embeddings_path", type=str, required=True,
                       help="Path to embeddings parquet file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for clustering results")
    parser.add_argument("--dataset_dir", type=str, default=None,
                       help="Path to dataset TSV files (for loading counts)")
    parser.add_argument("--dataset_num", type=int, default=1,
                       help="Dataset number (1-8), used for count column detection")
    parser.add_argument("--method", type=str, default="both",
                       choices=['kmeans', 'leiden', 'both'],
                       help="Clustering method (default: both)")
    
    # K-means options
    parser.add_argument("--n_clusters", type=int, default=100,
                       help="Number of clusters for K-means (default: 100)")
    parser.add_argument("--kmeans_batch_size", type=int, default=10000,
                       help="Batch size for MiniBatchKMeans (default: 10000)")
    
    # Leiden options
    parser.add_argument("--resolution", type=float, default=1.0,
                       help="Resolution for Leiden (default: 1.0)")
    parser.add_argument("--n_neighbors", type=int, default=15,
                       help="Number of neighbors for k-NN graph (default: 15)")
    
    # General options
    parser.add_argument("--n_jobs", type=int, default=-1,
                       help="Number of parallel jobs (-1 for all CPUs)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--use_gpu", action='store_true',
                       help="Use GPU for K-means if available")
    parser.add_argument("--chunk_size", type=int, default=100000,
                       help="Chunk size for loading embeddings (default: 100000)")
    
    args = parser.parse_args()
    
    print("="*80)
    print("GLOBAL CLUSTERING FOR MIL PIPELINE")
    print("="*80)
    print(f"Embeddings: {args.embeddings_path}")
    print(f"Output: {args.output_dir}")
    print(f"Method: {args.method}")
    print(f"Dataset: {args.dataset_num}")
    if args.n_jobs == -1:
        print(f"CPUs: {mp.cpu_count()}")
    else:
        print(f"CPUs: {args.n_jobs}")
    if args.use_gpu:
        print(f"GPU available: {GPU_AVAILABLE}")
    print()
    
    # Load embeddings
    sequences, embeddings = load_embeddings_chunked(args.embeddings_path, args.chunk_size)
    
    # Load counts/weights
    if args.dataset_dir:
        sample_weights = load_counts_for_sequences(args.dataset_dir, sequences, args.dataset_num)
    else:
        sample_weights = np.ones(len(sequences), dtype=np.float32)
    
    # Run K-means
    if args.method in ['kmeans', 'both']:
        print("\n" + "="*80)
        print("K-MEANS CLUSTERING")
        print("="*80)
        
        labels_km, centroids_km, metrics_km = run_kmeans_clustering(
            embeddings,
            n_clusters=args.n_clusters,
            sample_weights=sample_weights,
            batch_size=args.kmeans_batch_size,
            random_state=args.seed,
            use_gpu=args.use_gpu and GPU_AVAILABLE,
        )
        
        save_clustering_results(
            args.output_dir, 'kmeans',
            sequences, labels_km, centroids_km, metrics_km, sample_weights
        )
        
        del labels_km, centroids_km
        gc.collect()
    
    # Run Leiden
    if args.method in ['leiden', 'both']:
        print("\n" + "="*80)
        print("LEIDEN CLUSTERING")
        print("="*80)
        
        labels_ld, centroids_ld, metrics_ld = run_leiden_clustering(
            embeddings,
            sample_weights=sample_weights,
            n_neighbors=args.n_neighbors,
            resolution=args.resolution,
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )
        
        save_clustering_results(
            args.output_dir, 'leiden',
            sequences, labels_ld, centroids_ld, metrics_ld, sample_weights
        )
        
        del labels_ld, centroids_ld
        gc.collect()
    
    print("\n" + "="*80)
    print("CLUSTERING COMPLETE")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
