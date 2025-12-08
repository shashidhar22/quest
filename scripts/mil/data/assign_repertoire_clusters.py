#!/usr/bin/env python3
"""
Assign Global Clusters to Repertoires

Takes global clustering results and creates per-repertoire cluster representations
suitable for MIL training.

For each repertoire:
1. Look up cluster assignments for each sequence
2. Aggregate cluster memberships (sum weights per cluster)
3. Create cluster centroids as "instances" for the repertoire
4. Save in format compatible with MIL training scripts

Output format (clustered.pkl):
[
    {
        'repertoire_id': str,
        'label': int,
        'instances': np.array (n_clusters, embedding_dim) - cluster centroids present in this repertoire
        'instance_weights': np.array (n_clusters,) - aggregated weights per cluster
        'num_sequences': int - total sequences in repertoire
        'total_count': float - total count (sum of templates)
        'num_clusters': int - number of clusters with sequences
        'cluster_ids': list - which global cluster IDs are represented
    },
    ...
]
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
from collections import defaultdict
import gc


def load_global_clustering(clustering_dir: str, method: str = 'kmeans'):
    """
    Load global clustering results.
    
    Returns:
        sequence_to_cluster: dict mapping sequence_id -> cluster_id
        sequence_to_weight: dict mapping sequence_id -> weight
        centroids: np.array of cluster centroids
    """
    clustering_path = Path(clustering_dir)
    
    # Load sequence-cluster mapping
    mapping_file = clustering_path / f"{method}_sequence_clusters.parquet"
    if not mapping_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
    
    print(f"Loading {method} clustering from: {clustering_dir}")
    mapping_df = pd.read_parquet(mapping_file)
    
    sequence_to_cluster = dict(zip(mapping_df['sequence_id'], mapping_df['cluster_id']))
    
    if 'weight' in mapping_df.columns:
        sequence_to_weight = dict(zip(mapping_df['sequence_id'], mapping_df['weight']))
    else:
        sequence_to_weight = {seq: 1.0 for seq in mapping_df['sequence_id']}
    
    print(f"  Loaded {len(sequence_to_cluster):,} sequence-cluster mappings")
    
    # Load centroids
    centroids_file = clustering_path / f"{method}_centroids.npy"
    centroids = np.load(centroids_file)
    print(f"  Centroids shape: {centroids.shape}")
    
    # Load metrics
    metrics_file = clustering_path / f"{method}_metrics.json"
    with open(metrics_file) as f:
        metrics = json.load(f)
    print(f"  Clusters: {metrics.get('n_clusters', 'unknown')}")
    
    del mapping_df
    gc.collect()
    
    return sequence_to_cluster, sequence_to_weight, centroids, metrics


def load_repertoire_data(processed_dir: str):
    """
    Load repertoire metadata and sequence information.
    
    Returns:
        repertoires: list of dicts with repertoire_id, label, sequences
    """
    processed_path = Path(processed_dir)
    
    # Try loading from repertoires.json first
    json_file = processed_path / "repertoires.json"
    if json_file.exists():
        print(f"Loading repertoires from: {json_file}")
        with open(json_file) as f:
            repertoires = json.load(f)
        print(f"  Loaded {len(repertoires)} repertoires")
        return repertoires
    
    # Fallback: load from parquet files
    meta_file = processed_path / "metadata.parquet"
    rep_file = processed_path / "repertoires.parquet"
    
    if not meta_file.exists() or not rep_file.exists():
        raise FileNotFoundError(f"Could not find repertoire data in {processed_dir}")
    
    print(f"Loading repertoires from parquet files...")
    
    # Load metadata
    meta_df = pd.read_parquet(meta_file)
    rep_id_to_label = dict(zip(meta_df['repertoire_id'], meta_df['label']))
    
    # Load repertoire sequences
    rep_df = pd.read_parquet(rep_file)
    
    # Group by repertoire
    repertoires = []
    for rep_id, group in tqdm(rep_df.groupby('repertoire_id'), desc="Loading repertoires"):
        sequences = group['junction_aa'].tolist() if 'junction_aa' in group.columns else group['sequence_id'].tolist()
        
        # Get counts if available
        if 'templates' in group.columns:
            counts = group['templates'].tolist()
        elif 'frequency' in group.columns:
            counts = group['frequency'].tolist()
        else:
            counts = [1.0] * len(sequences)
        
        repertoires.append({
            'repertoire_id': rep_id,
            'label': rep_id_to_label.get(rep_id, 0),
            'sequences': sequences,
            'counts': counts,
        })
    
    print(f"  Loaded {len(repertoires)} repertoires")
    
    del meta_df, rep_df
    gc.collect()
    
    return repertoires


def assign_clusters_to_repertoires(
    repertoires: list,
    sequence_to_cluster: dict,
    sequence_to_weight: dict,
    centroids: np.ndarray,
    use_counts: bool = True,
) -> list:
    """
    Assign global clusters to each repertoire.
    
    For each repertoire:
    - Map sequences to clusters
    - Aggregate weights per cluster (sum of sequence counts)
    - Create instance representation using cluster centroids
    
    Returns:
        clustered_repertoires: list of dicts with cluster-based representation
    """
    n_clusters, embedding_dim = centroids.shape
    clustered_repertoires = []
    
    unmapped_total = 0
    mapped_total = 0
    
    for rep in tqdm(repertoires, desc="Assigning clusters to repertoires"):
        rep_id = rep['repertoire_id']
        label = rep['label']
        sequences = rep.get('sequences', [])
        counts = rep.get('counts', [1.0] * len(sequences))
        
        if not use_counts:
            counts = [1.0] * len(sequences)
        
        # Aggregate weights per cluster
        cluster_weights = defaultdict(float)
        cluster_sequence_counts = defaultdict(int)
        
        for seq, count in zip(sequences, counts):
            if seq in sequence_to_cluster:
                cluster_id = sequence_to_cluster[seq]
                weight = count * sequence_to_weight.get(seq, 1.0)
                cluster_weights[cluster_id] += weight
                cluster_sequence_counts[cluster_id] += 1
                mapped_total += 1
            else:
                unmapped_total += 1
        
        # Get clusters present in this repertoire
        present_clusters = sorted(cluster_weights.keys())
        
        if len(present_clusters) == 0:
            # No clusters found - use all centroids with zero weight
            # This shouldn't happen in practice
            print(f"  Warning: No clusters found for {rep_id}")
            instances = centroids
            instance_weights = np.zeros(n_clusters, dtype=np.float32)
            cluster_ids = list(range(n_clusters))
        else:
            # Get centroids and weights for present clusters
            instances = centroids[present_clusters]
            instance_weights = np.array([cluster_weights[c] for c in present_clusters], dtype=np.float32)
            cluster_ids = present_clusters
        
        # Normalize weights
        if instance_weights.sum() > 0:
            instance_weights_norm = instance_weights / instance_weights.sum()
        else:
            instance_weights_norm = np.ones(len(instance_weights), dtype=np.float32) / len(instance_weights)
        
        clustered_repertoires.append({
            'repertoire_id': rep_id,
            'label': label,
            'instances': instances.astype(np.float32),
            'instance_weights': instance_weights_norm,
            'num_sequences': len(sequences),
            'total_count': sum(counts),
            'num_clusters': len(present_clusters),
            'cluster_ids': cluster_ids,
            'metrics': {
                'sequences_mapped': sum(cluster_sequence_counts.values()),
                'sequences_unmapped': len(sequences) - sum(cluster_sequence_counts.values()),
                'cluster_coverage': len(present_clusters) / n_clusters,
            }
        })
    
    print(f"\nCluster assignment summary:")
    print(f"  Total sequences mapped: {mapped_total:,}")
    print(f"  Total sequences unmapped: {unmapped_total:,}")
    print(f"  Mapping rate: {100*mapped_total/(mapped_total+unmapped_total):.1f}%")
    
    return clustered_repertoires


def save_clustered_repertoires(
    clustered_repertoires: list,
    output_path: str,
    method: str,
    metrics: dict,
):
    """Save clustered repertoires to both pickle and JSON formats."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save pickle (with numpy arrays for cluster centroids)
    with open(output_path, 'wb') as f:
        pickle.dump(clustered_repertoires, f)
    
    file_size = os.path.getsize(output_path) / 1e6
    print(f"Saved: {output_path} ({file_size:.1f} MB)")
    
    # Save JSON format for compatibility with existing training scripts
    # Convert numpy arrays to lists for JSON serialization
    json_repertoires = []
    for rep in clustered_repertoires:
        json_rep = {
            'repertoire_id': rep['repertoire_id'],
            'label': int(rep['label']),
            'num_sequences': rep['num_sequences'],
            'num_clusters': rep['num_clusters'],
            'cluster_ids': rep['cluster_ids'],
            # Store cluster weights as "frequency" for MIL training compatibility
            'sequences': [f"cluster_{cid}" for cid in rep['cluster_ids']],
            'frequency': rep['instance_weights'].tolist() if isinstance(rep['instance_weights'], np.ndarray) else rep['instance_weights'],
        }
        # Add cluster_embeddings separately (can be loaded for direct MIL training)
        if 'instances' in rep:
            json_rep['cluster_embeddings'] = rep['instances'].tolist() if isinstance(rep['instances'], np.ndarray) else rep['instances']
        json_repertoires.append(json_rep)
    
    json_path = output_path.parent / f"{output_path.stem}.json"
    with open(json_path, 'w') as f:
        json.dump(json_repertoires, f)
    
    json_size = os.path.getsize(json_path) / 1e6
    print(f"Saved: {json_path} ({json_size:.1f} MB)")
    
    # Save summary metrics
    summary = {
        'method': method,
        'num_repertoires': len(clustered_repertoires),
        'global_clustering_metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'repertoire_stats': {
            'mean_clusters': np.mean([r['num_clusters'] for r in clustered_repertoires]),
            'mean_sequences': np.mean([r['num_sequences'] for r in clustered_repertoires]),
            'label_distribution': {
                '0': sum(1 for r in clustered_repertoires if r['label'] == 0),
                '1': sum(1 for r in clustered_repertoires if r['label'] == 1),
            }
        }
    }
    
    summary_path = output_path.parent / f"{output_path.stem}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Assign global clusters to repertoires for MIL training"
    )
    parser.add_argument("--clustering_dir", type=str, required=True,
                       help="Directory containing global clustering results")
    parser.add_argument("--processed_dir", type=str, required=True,
                       help="Directory containing processed repertoire data")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output path for clustered.pkl file")
    parser.add_argument("--method", type=str, default="kmeans",
                       choices=['kmeans', 'leiden'],
                       help="Clustering method to use (default: kmeans)")
    parser.add_argument("--use_counts", action='store_true', default=True,
                       help="Use sequence counts for weighting (default: True)")
    parser.add_argument("--no_counts", action='store_false', dest='use_counts',
                       help="Ignore sequence counts, use uniform weights")
    
    args = parser.parse_args()
    
    print("="*80)
    print("ASSIGN GLOBAL CLUSTERS TO REPERTOIRES")
    print("="*80)
    print(f"Clustering: {args.clustering_dir}")
    print(f"Repertoires: {args.processed_dir}")
    print(f"Method: {args.method}")
    print(f"Use counts: {args.use_counts}")
    print()
    
    # Load global clustering
    seq_to_cluster, seq_to_weight, centroids, metrics = load_global_clustering(
        args.clustering_dir, args.method
    )
    
    # Load repertoire data
    repertoires = load_repertoire_data(args.processed_dir)
    
    # Assign clusters
    clustered_repertoires = assign_clusters_to_repertoires(
        repertoires, seq_to_cluster, seq_to_weight, centroids, args.use_counts
    )
    
    # Save results
    save_clustered_repertoires(
        clustered_repertoires, args.output_path, args.method, metrics
    )
    
    print("\n" + "="*80)
    print("CLUSTER ASSIGNMENT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
