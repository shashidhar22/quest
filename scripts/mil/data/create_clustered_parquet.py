#!/usr/bin/env python3
"""
Create clustered MIL dataset from Parquet files with count-weighted clustering.

Reads repertoires and embeddings from Parquet format, clusters each repertoire
using sequence counts as weights, and outputs a pickle file for MIL training.
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm
import os
import gc

# GPU clustering
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
    from sklearn.neighbors import NearestNeighbors
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False


def cluster_kmeans_weighted(embeddings, counts, n_clusters=100):
    """
    KMeans clustering with count-weighted centroids.

    Args:
        embeddings: (N, D) array of embeddings
        counts: (N,) array of sequence counts
        n_clusters: Number of clusters

    Returns:
        cluster_embeddings: (K, D) weighted centroids
        cluster_weights: (K,) total counts per cluster
    """
    n_clusters = min(n_clusters, len(embeddings))

    # Cluster
    if GPU_AVAILABLE and len(embeddings) > 500:
        try:
            embeddings_gpu = cp.asarray(embeddings.astype(np.float32))
            kmeans = cuKMeans(n_clusters=n_clusters, random_state=42, max_iter=100)
            labels = kmeans.fit_predict(embeddings_gpu).get()
            del embeddings_gpu, kmeans
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as e:
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024, n_init=3)
            labels = kmeans.fit_predict(embeddings)
    else:
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024, n_init=3)
        labels = kmeans.fit_predict(embeddings)

    # Compute count-weighted centroids
    cluster_embs = []
    cluster_weights = []

    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() == 0:
            continue

        cluster_seqs = embeddings[mask]
        cluster_counts = counts[mask]

        # Weight by counts
        weights = cluster_counts / cluster_counts.sum()
        centroid = np.average(cluster_seqs, axis=0, weights=weights)

        cluster_embs.append(centroid)
        cluster_weights.append(cluster_counts.sum())

    return np.array(cluster_embs, dtype=np.float32), np.array(cluster_weights, dtype=np.float32)


def cluster_leiden_weighted(embeddings, counts, resolution=1.0, n_neighbors=15):
    """
    Leiden community detection with count-weighted centroids.
    """
    if not LEIDEN_AVAILABLE:
        raise ImportError("Leiden not available")

    # Build kNN graph
    n_neighbors = min(n_neighbors, len(embeddings) - 1)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', n_jobs=-1)
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    # Create graph
    edges = []
    edge_weights = []
    for i in range(len(embeddings)):
        for j, d in zip(indices[i], distances[i]):
            if i != j:
                edges.append((i, j))
                edge_weights.append(1.0 - d)

    g = ig.Graph(n=len(embeddings), edges=edges, directed=False)
    g.es['weight'] = edge_weights

    # Leiden clustering
    partition = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution, weights='weight'
    )
    labels = np.array(partition.membership)

    # Compute count-weighted centroids
    cluster_embs = []
    cluster_weights = []

    for k in range(max(labels) + 1):
        mask = labels == k
        if mask.sum() == 0:
            continue

        cluster_seqs = embeddings[mask]
        cluster_counts = counts[mask]

        weights = cluster_counts / cluster_counts.sum()
        centroid = np.average(cluster_seqs, axis=0, weights=weights)

        cluster_embs.append(centroid)
        cluster_weights.append(cluster_counts.sum())

    return np.array(cluster_embs, dtype=np.float32), np.array(cluster_weights, dtype=np.float32)


def create_clustered_dataset(
    repertoires_parquet: str,
    embeddings_parquet: str,
    output_pkl: str,
    method: str = 'kmeans',
    n_clusters: int = 100,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    seed: int = 42,
):
    """
    Create clustered MIL dataset from Parquet files.

    Args:
        repertoires_parquet: Parquet with repertoire_id, sequence_id, count, label
        embeddings_parquet: Parquet with sequence_id, embedding
        output_pkl: Output pickle file
        method: Clustering method (kmeans/leiden)
        n_clusters: Number of clusters for kmeans
        resolution: Resolution for leiden
        n_neighbors: Neighbors for leiden graph
        seed: Random seed
    """
    np.random.seed(seed)

    print("="*80)
    print(f"CREATING CLUSTERED DATASET ({method.upper()})")
    print("="*80)

    # Load repertoire data
    print(f"\nLoading repertoires from {repertoires_parquet}")
    repertoires_df = pd.read_parquet(repertoires_parquet)
    print(f"Total rows: {len(repertoires_df):,}")

    # Get unique repertoire IDs
    repertoire_ids = repertoires_df['repertoire_id'].unique()
    print(f"Unique repertoires: {len(repertoire_ids)}")

    # Load embeddings
    print(f"\nLoading embeddings from {embeddings_parquet}")
    embeddings_df = pd.read_parquet(embeddings_parquet)
    print(f"Total embeddings: {len(embeddings_df):,}")

    # Create embedding lookup
    print("Building embedding index...")
    seq_to_emb = {}
    for _, row in tqdm(embeddings_df.iterrows(), total=len(embeddings_df), desc="Indexing"):
        seq_to_emb[row['sequence_id']] = np.array(row['embedding'], dtype=np.float32)

    del embeddings_df
    gc.collect()

    # Process each repertoire
    print(f"\nClustering repertoires...")
    results = []

    for rep_id in tqdm(repertoire_ids, desc="Processing"):
        # Get repertoire sequences
        rep_data = repertoires_df[repertoires_df['repertoire_id'] == rep_id]
        label = rep_data['label'].iloc[0]

        # Build arrays
        embeddings = []
        counts = []
        sequence_ids = []

        for _, row in rep_data.iterrows():
            seq_id = row['sequence_id']
            if seq_id in seq_to_emb:
                embeddings.append(seq_to_emb[seq_id])
                counts.append(row['count'])
                sequence_ids.append(seq_id)

        if len(embeddings) < 10:
            print(f"  Skipping {rep_id}: only {len(embeddings)} sequences with embeddings")
            continue

        embeddings = np.array(embeddings, dtype=np.float32)
        counts = np.array(counts, dtype=np.float32)

        # Cluster with count weights
        if method == 'kmeans':
            instances, instance_weights = cluster_kmeans_weighted(embeddings, counts, n_clusters)
        else:
            instances, instance_weights = cluster_leiden_weighted(embeddings, counts, resolution, n_neighbors)

        results.append({
            'repertoire_id': rep_id,
            'label': int(label),
            'instances': instances,
            'instance_weights': instance_weights,
            'num_sequences': len(embeddings),
            'total_count': counts.sum(),
        })

        # Cleanup
        del embeddings, counts
        gc.collect()

    print(f"\nProcessed {len(results)} repertoires")

    # Normalize embeddings
    print("\nNormalizing cluster embeddings...")
    all_embs = np.vstack([r['instances'] for r in results])
    mean = all_embs.mean(axis=0)
    std = all_embs.std(axis=0) + 1e-8

    for r in results:
        r['instances'] = ((r['instances'] - mean) / std).astype(np.float32)

    # Save
    print(f"\nSaving to {output_pkl}")
    os.makedirs(os.path.dirname(output_pkl) or '.', exist_ok=True)
    with open(output_pkl, 'wb') as f:
        pickle.dump(results, f)

    # Stats
    labels = [r['label'] for r in results]
    n_clusters_actual = [len(r['instances']) for r in results]
    total_counts = [r['total_count'] for r in results]

    print(f"\nDataset statistics:")
    print(f"  Repertoires: {len(results)}")
    print(f"  Label 0: {sum(l==0 for l in labels)}, Label 1: {sum(l==1 for l in labels)}")
    print(f"  Clusters per repertoire: min={min(n_clusters_actual)}, max={max(n_clusters_actual)}, mean={np.mean(n_clusters_actual):.1f}")
    print(f"  Total counts: min={min(total_counts):.0f}, max={max(total_counts):.0f}, mean={np.mean(total_counts):.0f}")

    file_size = os.path.getsize(output_pkl) / 1e6
    print(f"  Output size: {file_size:.1f} MB")

    print(f"\n{'='*80}")
    print("Done!")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Create clustered MIL dataset from Parquet with count weights"
    )
    parser.add_argument("--repertoires_parquet", type=str, required=True,
                       help="Parquet with repertoire_id, sequence_id, count, label")
    parser.add_argument("--embeddings_parquet", type=str, required=True,
                       help="Parquet with sequence_id, embedding")
    parser.add_argument("--output_pkl", type=str, required=True,
                       help="Output pickle file")
    parser.add_argument("--method", choices=['kmeans', 'leiden'], default='kmeans',
                       help="Clustering method (default: kmeans)")
    parser.add_argument("--n_clusters", type=int, default=100,
                       help="Number of clusters for kmeans (default: 100)")
    parser.add_argument("--resolution", type=float, default=1.0,
                       help="Resolution for leiden (default: 1.0)")
    parser.add_argument("--n_neighbors", type=int, default=15,
                       help="Neighbors for leiden graph (default: 15)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")

    args = parser.parse_args()

    create_clustered_dataset(
        args.repertoires_parquet,
        args.embeddings_parquet,
        args.output_pkl,
        args.method,
        args.n_clusters,
        args.resolution,
        args.n_neighbors,
        args.seed,
    )


if __name__ == "__main__":
    main()
