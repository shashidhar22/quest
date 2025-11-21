#!/usr/bin/env python3
"""
repertoire_mil_clustering.py
────────────────────────────────────────────────────────
Clustering-Based Multiple Instance Learning for TCR Repertoire Classification

Key Innovation:
Instead of using individual TCR embeddings, we:
1. Cluster all TCRs based on embedding similarity
2. For each repertoire, compute cluster-level features:
   - Number of unique TCRs in each cluster
   - Total count of TCRs in each cluster (with frequency)
   - Cluster density/tightness
3. Use these cluster features for MIL classification

Biological Rationale:
Disease-specific immune responses cause expansion of specific TCR clonotypes.
These expanded clonotypes cluster together in embedding space.
By looking at cluster enrichment patterns, we can identify disease signatures.
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.cluster import MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from tqdm.auto import tqdm
import h5py

sys.path.append('/home/ubuntu/quest')
from scripts.inference.repertoire_mil_adaptive import StreamingEmbeddingStorage

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TCR CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════

class TCRClusterer:
    """
    Cluster TCR sequences based on embedding similarity.
    """
    
    def __init__(
        self,
        method: str = 'kmeans',
        n_clusters: int = 1000,
        random_state: int = 42,
        **kwargs
    ):
        """
        Args:
            method: Clustering method ('kmeans', 'dbscan', 'hierarchical')
            n_clusters: Number of clusters (for kmeans/hierarchical)
            random_state: Random seed
            **kwargs: Additional parameters for clustering algorithm
        """
        self.method = method
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kwargs = kwargs
        self.clusterer = None
        self.cluster_centers = None
        self.cluster_assignments = None
        
    def fit(self, embeddings: np.ndarray, sequences: List[str] = None):
        """
        Fit clustering model on TCR embeddings.
        
        Args:
            embeddings: Array of shape (n_sequences, embedding_dim)
            sequences: Optional list of sequence strings
        """
        print(f"\nClustering {len(embeddings)} TCR sequences using {self.method}...")
        
        if self.method == 'kmeans':
            self.clusterer = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                batch_size=1024,
                max_iter=100,
                n_init=3,
                **self.kwargs
            )
            self.cluster_assignments = self.clusterer.fit_predict(embeddings)
            self.cluster_centers = self.clusterer.cluster_centers_
            
        elif self.method == 'hierarchical':
            self.clusterer = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage='ward',
                **self.kwargs
            )
            self.cluster_assignments = self.clusterer.fit_predict(embeddings)
            # Compute cluster centers
            self.cluster_centers = np.zeros((self.n_clusters, embeddings.shape[1]))
            for i in range(self.n_clusters):
                mask = self.cluster_assignments == i
                if mask.sum() > 0:
                    self.cluster_centers[i] = embeddings[mask].mean(axis=0)
                    
        elif self.method == 'dbscan':
            eps = self.kwargs.get('eps', 0.5)
            min_samples = self.kwargs.get('min_samples', 5)
            self.clusterer = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            self.cluster_assignments = self.clusterer.fit_predict(embeddings)
            # Get actual number of clusters (excluding noise points labeled -1)
            unique_clusters = np.unique(self.cluster_assignments)
            unique_clusters = unique_clusters[unique_clusters >= 0]
            self.n_clusters = len(unique_clusters)
            # Compute cluster centers
            self.cluster_centers = np.zeros((self.n_clusters, embeddings.shape[1]))
            for i, cluster_id in enumerate(unique_clusters):
                mask = self.cluster_assignments == cluster_id
                if mask.sum() > 0:
                    self.cluster_centers[i] = embeddings[mask].mean(axis=0)
        
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        # Print statistics
        unique_clusters, counts = np.unique(self.cluster_assignments, return_counts=True)
        print(f"Created {self.n_clusters} clusters")
        print(f"Cluster sizes: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}, median={np.median(counts):.1f}")
        
        # Check for noise in DBSCAN
        if self.method == 'dbscan' and -1 in unique_clusters:
            noise_count = (self.cluster_assignments == -1).sum()
            print(f"Noise points (cluster -1): {noise_count} ({noise_count/len(embeddings)*100:.1f}%)")
        
        return self
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Assign new embeddings to existing clusters.
        """
        if self.method == 'kmeans':
            return self.clusterer.predict(embeddings)
        else:
            # For hierarchical/DBSCAN, assign to nearest cluster center
            if self.cluster_centers is None:
                raise ValueError("Model not fitted yet")
            distances = pairwise_distances(embeddings, self.cluster_centers, metric='euclidean')
            return np.argmin(distances, axis=1)
    
    def get_cluster_statistics(self, embeddings: np.ndarray, cluster_ids: np.ndarray) -> Dict[int, Dict]:
        """
        Compute statistics for each cluster.
        
        Returns:
            Dict mapping cluster_id to statistics dict
        """
        stats = {}
        
        for cluster_id in np.unique(cluster_ids):
            if cluster_id == -1:  # Skip noise in DBSCAN
                continue
                
            mask = cluster_ids == cluster_id
            cluster_embeddings = embeddings[mask]
            
            if len(cluster_embeddings) == 0:
                continue
            
            # Compute cluster center
            center = cluster_embeddings.mean(axis=0)
            
            # Compute distances from center
            distances = np.linalg.norm(cluster_embeddings - center, axis=1)
            
            stats[cluster_id] = {
                'size': len(cluster_embeddings),
                'center': center,
                'mean_distance': distances.mean(),
                'std_distance': distances.std(),
                'max_distance': distances.max(),
                'density': len(cluster_embeddings) / (distances.mean() + 1e-10),  # Higher = tighter cluster
                'compactness': 1.0 / (distances.std() + 1e-10)  # Higher = more compact
            }
        
        return stats


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CLUSTER-BASED REPERTOIRE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def extract_cluster_features(
    sequences: List[str],
    embeddings_storage: StreamingEmbeddingStorage,
    clusterer: TCRClusterer,
    frequencies: Optional[List[float]] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Extract cluster-based features for a repertoire.
    
    Features per cluster:
    1. Number of unique TCRs in cluster
    2. Total frequency (count) of TCRs in cluster
    3. Average cluster density (tightness)
    
    Args:
        sequences: List of TCR sequences in repertoire
        embeddings_storage: HDF5 storage with embeddings
        clusterer: Fitted TCR clusterer
        frequencies: Optional frequency of each sequence
        normalize: Whether to normalize features
    
    Returns:
        Feature vector of shape (n_clusters * 3,)
    """
    if frequencies is None:
        frequencies = [1.0] * len(sequences)
    
    # Get embeddings and cluster assignments
    embeddings = []
    valid_sequences = []
    valid_frequencies = []
    
    for seq, freq in zip(sequences, frequencies):
        emb = embeddings_storage.get(seq)
        if emb is not None:
            embeddings.append(emb)
            valid_sequences.append(seq)
            valid_frequencies.append(freq)
    
    if len(embeddings) == 0:
        # Return zero features if no valid embeddings
        return np.zeros(clusterer.n_clusters * 3, dtype=np.float32)
    
    embeddings = np.array(embeddings)
    valid_frequencies = np.array(valid_frequencies)
    
    # Assign to clusters
    cluster_ids = clusterer.predict(embeddings)
    
    # Initialize features
    unique_counts = np.zeros(clusterer.n_clusters, dtype=np.float32)
    total_counts = np.zeros(clusterer.n_clusters, dtype=np.float32)
    avg_distances = np.zeros(clusterer.n_clusters, dtype=np.float32)
    
    # Compute features for each cluster
    for cluster_id in range(clusterer.n_clusters):
        mask = cluster_ids == cluster_id
        
        if mask.sum() > 0:
            # Unique count
            unique_counts[cluster_id] = mask.sum()
            
            # Total count (weighted by frequency)
            total_counts[cluster_id] = valid_frequencies[mask].sum()
            
            # Average distance to cluster center (inverse of density)
            cluster_embeddings = embeddings[mask]
            center = clusterer.cluster_centers[cluster_id]
            distances = np.linalg.norm(cluster_embeddings - center, axis=1)
            avg_distances[cluster_id] = distances.mean() if len(distances) > 0 else 0
    
    # Combine features
    features = np.concatenate([
        unique_counts,
        total_counts,
        avg_distances
    ])
    
    # Normalize if requested
    if normalize:
        # LOG-TRANSFORM COUNTS (this is key for variance!)
        # Log(x+1) prevents log(0) and compresses large values
        unique_counts_log = np.log1p(unique_counts)
        total_counts_log = np.log1p(total_counts)
        
        # Combine features with log-transformed counts
        features = np.concatenate([
            unique_counts_log,
            total_counts_log,
            avg_distances
        ])
    
    return features


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class ClusterRepertoireDataset(Dataset):
    """Dataset using cluster-based features for repertoires."""
    
    def __init__(
        self,
        repertoire_data: List[Dict[str, Any]],
        embeddings_storage: StreamingEmbeddingStorage,
        clusterer: TCRClusterer,
        normalize: bool = True
    ):
        self.repertoire_data = repertoire_data
        self.embeddings_storage = embeddings_storage
        self.clusterer = clusterer
        self.normalize = normalize
    
    def __len__(self) -> int:
        return len(self.repertoire_data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.repertoire_data[idx]
        
        # Check if pre-scaled features exist
        if '_scaled_features' in item:
            features = item['_scaled_features']
        else:
            # Extract cluster features on-the-fly
            frequencies = item.get('frequency', None)
            features = extract_cluster_features(
                item['sequences'],
                self.embeddings_storage,
                self.clusterer,
                frequencies=frequencies,
                normalize=self.normalize
            )
        
        return {
            'repertoire_id': item['repertoire_id'],
            'features': features,
            'label': item['label']
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3B. PRE-COMPUTED CLUSTER DATASET (for attention-based MIL)
# ═══════════════════════════════════════════════════════════════════════════════

class PrecomputedClusterDataset(Dataset):
    """Dataset using pre-computed cluster embeddings and weights."""

    def __init__(self, repertoire_data: List[Dict[str, Any]]):
        """
        Args:
            repertoire_data: List of dicts with keys:
                - repertoire_id
                - label
                - gmm_means: (K, D) array of cluster centroids in PCA space
                - cluster_sizes: (K,) array of number of sequences per cluster
        """
        self.repertoire_data = repertoire_data

    def __len__(self) -> int:
        return len(self.repertoire_data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.repertoire_data[idx]

        # Support multiple formats:
        # 1. Direct instances (from normalized embeddings)
        # 2. cluster_means (from KMeans/HDBSCAN/Leiden clustering)
        # 3. gmm_means (from GMM clustering with PCA)
        if 'instances' in item:
            cluster_embs = item['instances'].astype(np.float32)
            cluster_weights = item.get('instance_weights', np.ones(len(cluster_embs), dtype=np.float32))
        elif 'cluster_means' in item:
            cluster_embs = item['cluster_means'].astype(np.float32)
            # Use cluster_weights if available, otherwise cluster_sizes
            if 'cluster_weights' in item and len(item['cluster_weights']) == len(cluster_embs):
                cluster_weights = item['cluster_weights'].astype(np.float32)
            else:
                cluster_weights = item['cluster_sizes'].astype(np.float32)
        elif 'gmm_means' in item:
            cluster_embs = item['gmm_means'].astype(np.float32)
            cluster_weights = item['cluster_sizes'].astype(np.float32)
        else:
            raise KeyError("No embeddings found (expected 'instances', 'cluster_means', or 'gmm_means')")

        # Normalize weights
        if cluster_weights.sum() > 0:
            cluster_weights = cluster_weights / cluster_weights.sum()

        return {
            'repertoire_id': item['repertoire_id'],
            'label': item['label'],
            'instances': torch.from_numpy(cluster_embs),  # (K, D) where D=embedding_dim
            'instance_weights': torch.from_numpy(cluster_weights),  # (K,)
            'bag_size': len(cluster_embs)
        }


def collate_mil_batch(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for variable-length bags."""

    # Get max bag size
    max_bag_size = max(b['bag_size'] for b in batch)
    embedding_dim = batch[0]['instances'].shape[1]

    # Pad instances
    instances = torch.zeros(len(batch), max_bag_size, embedding_dim)
    instance_weights = torch.zeros(len(batch), max_bag_size)
    masks = torch.zeros(len(batch), max_bag_size, dtype=torch.bool)

    for i, b in enumerate(batch):
        size = b['bag_size']
        instances[i, :size] = b['instances']
        instance_weights[i, :size] = b['instance_weights']
        masks[i, :size] = True

    return {
        'instances': instances,
        'instance_weights': instance_weights,
        'masks': masks,
        'labels': torch.tensor([b['label'] for b in batch]),
        'repertoire_ids': [b['repertoire_id'] for b in batch]
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class AttentionMIL(nn.Module):
    """
    Attention-based MIL model for pre-computed cluster embeddings.

    Uses gated attention mechanism to aggregate cluster representations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
        use_instance_weights: bool = True
    ):
        super().__init__()

        self.use_instance_weights = use_instance_weights

        # Instance-level feature extraction
        self.instance_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Gated attention
        self.attention_V = nn.Linear(hidden_dim, hidden_dim)
        self.attention_U = nn.Linear(hidden_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1)

        # Bag-level classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(
        self,
        instances: torch.Tensor,
        masks: torch.Tensor,
        instance_weights: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            instances: (B, N, D) instance embeddings
            masks: (B, N) boolean mask for valid instances
            instance_weights: (B, N) optional instance weights

        Returns:
            logits: (B, num_classes)
            attention: (B, N) attention weights
        """
        # Encode instances
        H = self.instance_encoder(instances)  # (B, N, hidden_dim)

        # Gated attention
        A_V = torch.tanh(self.attention_V(H))  # (B, N, hidden_dim)
        A_U = torch.sigmoid(self.attention_U(H))  # (B, N, hidden_dim)
        A = self.attention_w(A_V * A_U)  # (B, N, 1)
        A = A.squeeze(-1)  # (B, N)

        # Apply mask
        A = A.masked_fill(~masks, float('-inf'))

        # Combine with instance weights if provided
        if self.use_instance_weights and instance_weights is not None:
            # Normalize instance weights
            weights_norm = instance_weights.masked_fill(~masks, 0)
            weights_norm = weights_norm / (weights_norm.sum(dim=1, keepdim=True) + 1e-8)

            # Combine attention and weights (multiplicative)
            A = A + torch.log(weights_norm + 1e-8)

        # Softmax attention
        A = torch.softmax(A, dim=1)  # (B, N)

        # Aggregate
        M = torch.bmm(A.unsqueeze(1), H).squeeze(1)  # (B, hidden_dim)

        # Classify
        logits = self.classifier(M)  # (B, num_classes)

        return logits, A


class ClusterMIL(nn.Module):
    """
    MIL classifier using cluster-based repertoire features.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256],
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape (batch_size, input_dim)
        
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        return self.classifier(features)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_attention_mil(
    model: nn.Module,
    train_dataset: PrecomputedClusterDataset,
    val_dataset: PrecomputedClusterDataset,
    num_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    device: str = "cuda",
    patience: int = 10,
    output_dir: str = "./output",
    use_wandb: bool = False
) -> Dict[str, List[float]]:
    """Train attention-based MIL model with pre-computed clusters."""

    os.makedirs(output_dir, exist_ok=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_mil_batch, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_mil_batch, num_workers=0
    )

    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\nStarting training...")
    print(f"  Train batches per epoch: {len(train_loader)}")
    print(f"  Val batches per epoch: {len(val_loader)}")

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for batch in train_pbar:
            instances = batch['instances'].to(device)
            masks = batch['masks'].to(device)
            instance_weights = batch['instance_weights'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits, attention = model(instances, masks, instance_weights)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_correct/train_total:.4f}'
            })

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False):
                instances = batch['instances'].to(device)
                masks = batch['masks'].to(device)
                instance_weights = batch['instance_weights'].to(device)
                labels = batch['labels'].to(device)

                logits, attention = model(instances, masks, instance_weights)
                loss = criterion(logits, labels)

                val_loss += loss.item() * len(labels)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_loss /= val_total
        val_acc = val_correct / val_total

        epoch_time = time.time() - epoch_start

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"\nEpoch {epoch+1}/{num_epochs} (time: {epoch_time:.1f}s):")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'train/accuracy': train_acc,
                'val/loss': val_loss,
                'val/accuracy': val_acc,
            })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            print(f"  ✓ New best model! (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt")))
    return history


def evaluate_attention_mil(
    model: nn.Module,
    test_dataset: PrecomputedClusterDataset,
    device: str = "cuda",
    batch_size: int = 16
) -> Dict[str, Any]:
    """Evaluate attention-based MIL model."""

    model.eval()
    model.to(device)

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_mil_batch, num_workers=0
    )

    all_preds = []
    all_labels = []
    all_probs = []
    repertoire_ids = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            instances = batch['instances'].to(device)
            masks = batch['masks'].to(device)
            instance_weights = batch['instance_weights'].to(device)
            labels = batch['labels'].to(device)

            logits, attention = model(instances, masks, instance_weights)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            repertoire_ids.extend(batch['repertoire_ids'])

    # Metrics
    all_probs = np.array(all_probs)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    if all_probs.shape[1] == 2:
        try:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        except:
            auc = None
    else:
        auc = None

    cm = confusion_matrix(all_labels, all_preds)

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist(),
        'predictions': {
            rep_id: {'predicted': int(pred), 'true': int(true), 'probs': probs.tolist()}
            for rep_id, pred, true, probs in zip(repertoire_ids, all_preds, all_labels, all_probs)
        }
    }

    return results


def train_cluster_mil(
    model: nn.Module,
    train_dataset: ClusterRepertoireDataset,
    val_dataset: ClusterRepertoireDataset,
    num_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    device: str = "cuda",
    patience: int = 10,
    output_dir: str = "./cluster_mil_output",
    use_wandb: bool = False
) -> Dict[str, List[float]]:
    """Train cluster-based MIL model."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    best_val_acc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                labels = batch['label'].to(device)
                
                logits = model(features)
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * len(labels)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Log
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            patience_counter = 0
            print(f"✓ New best model saved (val_acc = {val_acc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt")))
    
    return history


def evaluate_cluster_mil(
    model: nn.Module,
    test_dataset: ClusterRepertoireDataset,
    device: str = "cuda",
    batch_size: int = 32
) -> Dict[str, Any]:
    """Evaluate cluster-based MIL model."""
    
    model.eval()
    model.to(device)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_labels = []
    all_probs = []
    repertoire_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            features = batch['features'].to(device)
            labels = batch['label'].numpy()
            
            logits = model(features)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs.cpu().numpy())
            repertoire_ids.extend(batch['repertoire_id'])
    
    # Metrics
    all_probs = np.array(all_probs)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    if all_probs.shape[1] == 2:
        try:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        except:
            auc = None
    else:
        auc = None
    
    cm = confusion_matrix(all_labels, all_preds)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist(),
        'predictions': {
            rep_id: {'predicted': int(pred), 'true': int(true), 'probs': probs.tolist()}
            for rep_id, pred, true, probs in zip(repertoire_ids, all_preds, all_labels, all_probs)
        }
    }
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Clustering-Based MIL for TCR Repertoire Classification"
    )

    # Data
    parser.add_argument("--data_json", type=str, default=None,
                        help="Path to JSON file with repertoire data (for computing clusters)")
    parser.add_argument("--embeddings_cache", type=str, default=None,
                        help="Path to HDF5 embeddings cache (for computing clusters)")
    parser.add_argument("--precomputed_clusters", type=str, default=None,
                        help="Path to pickle file with pre-computed clustered repertoires")
    
    # Clustering
    parser.add_argument("--clustering_method", type=str, default="kmeans",
                        choices=["kmeans", "hierarchical", "dbscan"])
    parser.add_argument("--n_clusters", type=int, default=1000)
    parser.add_argument("--dbscan_eps", type=float, default=0.5)
    parser.add_argument("--dbscan_min_samples", type=int, default=5)
    
    # Model
    parser.add_argument("--hidden_dims", type=int, nargs='+', default=[512, 256])
    parser.add_argument("--dropout", type=float, default=0.3)
    
    # Training
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    
    # Other
    parser.add_argument("--output_dir", type=str, default="./cluster_mil_output")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    
    args = parser.parse_args()

    # Validate arguments
    if args.precomputed_clusters is None and (args.data_json is None or args.embeddings_cache is None):
        parser.error("Either --precomputed_clusters OR both --data_json and --embeddings_cache are required")

    # Initialize W&B
    use_wandb = args.wandb_project is not None and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    if args.precomputed_clusters:
        print("MIL TRAINING WITH PRE-COMPUTED CLUSTERS")
    else:
        print("MIL TRAINING WITH ON-THE-FLY CLUSTERING")
    print("="*80)

    # Branch based on mode
    if args.precomputed_clusters:
        # ============== PRE-COMPUTED CLUSTERS MODE ==============
        import pickle

        print(f"\nLoading pre-computed clustered data from: {args.precomputed_clusters}")
        with open(args.precomputed_clusters, 'rb') as f:
            repertoire_data = pickle.load(f)
        print(f"Loaded {len(repertoire_data)} repertoires")

        # Print statistics - support all formats
        if 'instances' in repertoire_data[0]:
            cluster_sizes = [len(r['instances']) for r in repertoire_data]
            embedding_dim = repertoire_data[0]['instances'].shape[1]
        elif 'cluster_means' in repertoire_data[0]:
            cluster_sizes = [len(r['cluster_means']) for r in repertoire_data]
            embedding_dim = repertoire_data[0]['cluster_means'].shape[1]
        else:
            cluster_sizes = [len(r['gmm_means']) for r in repertoire_data]
            embedding_dim = repertoire_data[0]['gmm_means'].shape[1]

        print(f"\nInstance/cluster statistics per repertoire:")
        print(f"  Min: {min(cluster_sizes)}")
        print(f"  Max: {max(cluster_sizes)}")
        print(f"  Mean: {np.mean(cluster_sizes):.1f}")
        print(f"  Median: {np.median(cluster_sizes):.1f}")

        # Get embedding dimension and number of classes
        print(f"  Embedding dimension: {embedding_dim}")
        labels = [r['label'] for r in repertoire_data]
        num_classes = len(set(labels))

        # Split data
        train_val_data, test_data = train_test_split(
            repertoire_data, test_size=args.test_size, stratify=labels, random_state=args.seed
        )
        train_labels = [r['label'] for r in train_val_data]
        train_data, val_data = train_test_split(
            train_val_data, test_size=args.val_size, stratify=train_labels, random_state=args.seed
        )

        print(f"\nDataset splits:")
        print(f"  Train: {len(train_data)}")
        print(f"  Val: {len(val_data)}")
        print(f"  Test: {len(test_data)}")

        # Create datasets
        train_dataset = PrecomputedClusterDataset(train_data)
        val_dataset = PrecomputedClusterDataset(val_data)
        test_dataset = PrecomputedClusterDataset(test_data)

        # Create attention-based MIL model
        model = AttentionMIL(
            input_dim=embedding_dim,
            hidden_dim=args.hidden_dims[0] if args.hidden_dims else 256,
            num_classes=num_classes,
            dropout=args.dropout,
            use_instance_weights=True
        )

        print(f"\nModel: AttentionMIL")
        print(f"  Input dim: {embedding_dim}")
        print(f"  Hidden dim: {args.hidden_dims[0] if args.hidden_dims else 256}")
        print(f"  Num classes: {num_classes}")

        # Train
        print("\n" + "="*80)
        print("TRAINING")
        print("="*80)

        history = train_attention_mil(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_epochs=args.num_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            device=args.device,
            patience=args.patience,
            output_dir=args.output_dir,
            use_wandb=use_wandb
        )

        # Evaluate
        print("\n" + "="*80)
        print("EVALUATION")
        print("="*80)

        results = evaluate_attention_mil(
            model=model,
            test_dataset=test_dataset,
            device=args.device,
            batch_size=args.batch_size
        )

    else:
        # ============== ON-THE-FLY CLUSTERING MODE ==============

        # Load data
        print(f"\nLoading data from {args.data_json}")
        with open(args.data_json, 'r') as f:
            repertoire_data = json.load(f)

        print(f"Loaded {len(repertoire_data)} repertoires")

        # Load embeddings
        print(f"\nLoading embeddings from {args.embeddings_cache}")
        embeddings_storage = StreamingEmbeddingStorage(args.embeddings_cache, mode='r')

        # Collect all unique sequences and embeddings for clustering
        print("\nCollecting all unique sequences...")
        all_sequences = set()
        for rep in repertoire_data:
            all_sequences.update(rep['sequences'])
        all_sequences = sorted(list(all_sequences))
    
        print(f"Total unique sequences: {len(all_sequences)}")
    
        # Get embeddings for clustering
        print("Loading embeddings for clustering...")
        all_embeddings = []
        valid_sequences = []
    
        for seq in tqdm(all_sequences, desc="Loading embeddings"):
            emb = embeddings_storage.get(seq)
            if emb is not None:
                all_embeddings.append(emb)
                valid_sequences.append(seq)
    
        all_embeddings = np.array(all_embeddings)
        print(f"Loaded {len(all_embeddings)} embeddings with shape {all_embeddings.shape}")
    
        # Cluster TCRs
        print("\n" + "="*80)
        print("CLUSTERING TCR SEQUENCES")
        print("="*80)
    
        clustering_kwargs = {}
        if args.clustering_method == 'dbscan':
            clustering_kwargs = {'eps': args.dbscan_eps, 'min_samples': args.dbscan_min_samples}
    
        clusterer = TCRClusterer(
            method=args.clustering_method,
            n_clusters=args.n_clusters,
            random_state=args.seed,
            **clustering_kwargs
        )
        clusterer.fit(all_embeddings, valid_sequences)
    
        # Save clusterer
        import pickle
        with open(os.path.join(args.output_dir, 'clusterer.pkl'), 'wb') as f:
            pickle.dump(clusterer, f)
    
        # Split data
        labels = [r['label'] for r in repertoire_data]
        train_val_data, test_data = train_test_split(
            repertoire_data, test_size=args.test_size, stratify=labels, random_state=args.seed
        )
    
        train_labels = [r['label'] for r in train_val_data]
        train_data, val_data = train_test_split(
            train_val_data, test_size=args.val_size, stratify=train_labels, random_state=args.seed
        )
    
        print(f"\nDataset splits:")
        print(f"  Train: {len(train_data)} repertoires")
        print(f"  Val: {len(val_data)} repertoires")
        print(f"  Test: {len(test_data)} repertoires")
    
        # Create datasets
        print("\nCreating cluster-based datasets...")
        train_dataset = ClusterRepertoireDataset(train_data, embeddings_storage, clusterer)
        val_dataset = ClusterRepertoireDataset(val_data, embeddings_storage, clusterer)
        test_dataset = ClusterRepertoireDataset(test_data, embeddings_storage, clusterer)
    
        # Get feature dimension
        sample_features = train_dataset[0]['features']
        feature_dim = len(sample_features)
        print(f"Feature dimension: {feature_dim}")
    
        # FIT STANDARDSCALER ON TRAINING DATA
        print("\nFitting StandardScaler on training features...")
        from sklearn.preprocessing import StandardScaler
    
        train_features = np.array([train_dataset[i]['features'] for i in range(len(train_dataset))])
        scaler = StandardScaler()
        scaler.fit(train_features)
    
        print(f"Feature statistics BEFORE scaling:")
        print(f"  Mean: {train_features.mean():.6f}, Std: {train_features.std():.6f}")
        print(f"  Min: {train_features.min():.6f}, Max: {train_features.max():.6f}")
    
        # Apply scaling to all datasets
        for i in range(len(train_dataset)):
            train_dataset.repertoire_data[i]['_scaled_features'] = scaler.transform(
                train_dataset[i]['features'].reshape(1, -1)
            )[0]
        for i in range(len(val_dataset)):
            val_dataset.repertoire_data[i]['_scaled_features'] = scaler.transform(
                val_dataset[i]['features'].reshape(1, -1)
            )[0]
        for i in range(len(test_dataset)):
            test_dataset.repertoire_data[i]['_scaled_features'] = scaler.transform(
                test_dataset[i]['features'].reshape(1, -1)
            )[0]
    
        # Save scaler
        with open(os.path.join(args.output_dir, 'feature_scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
    
        print(f"✓ Features standardized and scaler saved")
    
        # Initialize model
        num_classes = len(set(labels))
        model = ClusterMIL(
            input_dim=feature_dim,
            hidden_dims=args.hidden_dims,
            num_classes=num_classes,
            dropout=args.dropout
        )
    
        print(f"\nModel architecture:")
        print(model)
    
        # Train
        print("\n" + "="*80)
        print("TRAINING")
        print("="*80)
    
        history = train_cluster_mil(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_epochs=args.num_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            device=args.device,
            patience=args.patience,
            output_dir=args.output_dir,
            use_wandb=use_wandb
        )

        # Evaluate
        print("\n" + "="*80)
        print("EVALUATION")
        print("="*80)
    
        results = evaluate_cluster_mil(
            model=model,
            test_dataset=test_dataset,
            device=args.device,
            batch_size=args.batch_size
        )

        embeddings_storage.close()

    # ============== COMMON: PLOT AND SAVE RESULTS ==============
    # (This works for both pre-computed and on-the-fly modes)

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'), dpi=150)
    print(f"Saved training curves to {args.output_dir}/training_curves.png")

    # Print test results
    print(f"\nTest Results:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1: {results['f1']:.4f}")
    if results['auc'] is not None:
        print(f"  AUC: {results['auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(np.array(results['confusion_matrix']))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f"Confusion Matrix (Acc={results['accuracy']:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'), dpi=150)
    print(f"Saved confusion matrix to {args.output_dir}/confusion_matrix.png")

    # Save results
    summary_results = {k: v for k, v in results.items() if k != 'predictions'}
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump(summary_results, f, indent=2)

    if 'predictions' in results:
        with open(os.path.join(args.output_dir, 'predictions.json'), 'w') as f:
            json.dump(results['predictions'], f, indent=2)

    print(f"\n✅ Results saved to: {args.output_dir}")

    # W&B logging
    if use_wandb:
        wandb.log({
            'test_accuracy': results['accuracy'],
            'test_f1': results['f1'],
            'test_auc': results['auc']
        })
        wandb.finish()


if __name__ == "__main__":
    main()
