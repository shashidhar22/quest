#!/usr/bin/env python3
"""
Multi-Scale Attention MIL

Hybrid approach combining:
1. Cluster-level attention: Efficient attention over cluster centroids
2. Within-cluster attention: Fine-grained attention on top-attended clusters
3. Cross-scale fusion: Combine both scales for final prediction

This provides both efficiency (from clustering) and granularity (from sequences).
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from tqdm.auto import tqdm
import pickle
import h5py
import matplotlib.pyplot as plt
import seaborn as sns

# Import optimized embedding loader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding_loaders import get_embedding_loader


class MultiScaleMILDataset(Dataset):
    """
    Dataset that provides both cluster centroids and sequences within clusters.
    """

    def __init__(
        self,
        clustered_data: list,
        repertoire_data: list,
        embedding_loader,
        top_clusters: int = 10,
        seqs_per_cluster: int = 20
    ):
        """
        Args:
            clustered_data: List of dicts with cluster centroids
            repertoire_data: List of dicts with original sequences
            embedding_loader: Loader for sequence embeddings
            top_clusters: Number of top-attended clusters to expand
            seqs_per_cluster: Number of sequences to sample per cluster
        """
        self.clustered_data = clustered_data
        self.embedding_loader = embedding_loader
        self.top_clusters = top_clusters
        self.seqs_per_cluster = seqs_per_cluster

        # Create lookup from repertoire_id to repertoire data
        self.repertoire_lookup = {r['repertoire_id']: r for r in repertoire_data}

    def __len__(self):
        return len(self.clustered_data)

    def __getitem__(self, idx: int):
        cluster_item = self.clustered_data[idx]
        rep_id = cluster_item['repertoire_id']

        # Get cluster centroids (coarse scale)
        cluster_centroids = cluster_item['instances']  # (K, D)
        cluster_weights = cluster_item['instance_weights']  # (K,)

        # Get original sequences for fine-grained attention
        if rep_id in self.repertoire_lookup:
            rep_data = self.repertoire_lookup[rep_id]
            sequences = rep_data['sequences']
            frequencies = rep_data.get('frequency', [1.0] * len(sequences))

            # Sample sequences (take top frequent + random)
            n_sample = min(self.top_clusters * self.seqs_per_cluster, len(sequences))
            sorted_indices = np.argsort(frequencies)[::-1]

            # Take top sequences
            sampled_indices = sorted_indices[:n_sample]
            sampled_seqs = [sequences[i] for i in sampled_indices]
            sampled_freqs = [frequencies[i] for i in sampled_indices]

            # Get embeddings
            sequence_embeddings = self.embedding_loader.get_batch(sampled_seqs)
            sequence_weights = np.array(sampled_freqs, dtype=np.float32)
            sequence_weights = sequence_weights / (sequence_weights.sum() + 1e-8)
        else:
            # Fallback: use cluster centroids
            sequence_embeddings = cluster_centroids
            sequence_weights = cluster_weights / (cluster_weights.sum() + 1e-8)

        return {
            'repertoire_id': rep_id,
            'label': cluster_item['label'],
            # Coarse scale
            'cluster_centroids': torch.from_numpy(cluster_centroids.astype(np.float32)),
            'cluster_weights': torch.from_numpy(cluster_weights.astype(np.float32)),
            'num_clusters': len(cluster_centroids),
            # Fine scale
            'sequence_embeddings': torch.from_numpy(sequence_embeddings.astype(np.float32)),
            'sequence_weights': torch.from_numpy(sequence_weights),
            'num_sequences': len(sequence_embeddings),
        }


def collate_multiscale_batch(batch: list):
    """Collate function for multi-scale data."""
    max_clusters = max(b['num_clusters'] for b in batch)
    max_sequences = max(b['num_sequences'] for b in batch)
    embedding_dim = batch[0]['cluster_centroids'].shape[1]

    # Cluster-level
    cluster_centroids = torch.zeros(len(batch), max_clusters, embedding_dim)
    cluster_weights = torch.zeros(len(batch), max_clusters)
    cluster_masks = torch.zeros(len(batch), max_clusters, dtype=torch.bool)

    # Sequence-level
    sequence_embeddings = torch.zeros(len(batch), max_sequences, embedding_dim)
    sequence_weights = torch.zeros(len(batch), max_sequences)
    sequence_masks = torch.zeros(len(batch), max_sequences, dtype=torch.bool)

    for i, b in enumerate(batch):
        nc = b['num_clusters']
        ns = b['num_sequences']

        cluster_centroids[i, :nc] = b['cluster_centroids']
        cluster_weights[i, :nc] = b['cluster_weights']
        cluster_masks[i, :nc] = True

        sequence_embeddings[i, :ns] = b['sequence_embeddings']
        sequence_weights[i, :ns] = b['sequence_weights']
        sequence_masks[i, :ns] = True

    return {
        'cluster_centroids': cluster_centroids,
        'cluster_weights': cluster_weights,
        'cluster_masks': cluster_masks,
        'sequence_embeddings': sequence_embeddings,
        'sequence_weights': sequence_weights,
        'sequence_masks': sequence_masks,
        'labels': torch.tensor([b['label'] for b in batch]),
        'repertoire_ids': [b['repertoire_id'] for b in batch]
    }


class AttentionModule(nn.Module):
    """Gated attention module."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.attention_V = nn.Linear(hidden_dim, hidden_dim)
        self.attention_U = nn.Linear(hidden_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, instances, masks, instance_weights=None):
        """
        Args:
            instances: (B, N, D)
            masks: (B, N)
            instance_weights: (B, N) optional

        Returns:
            aggregated: (B, hidden_dim)
            attention: (B, N)
        """
        H = self.encoder(instances)  # (B, N, hidden_dim)

        # Gated attention
        A_V = torch.tanh(self.attention_V(H))
        A_U = torch.sigmoid(self.attention_U(H))
        A = self.attention_w(A_V * A_U).squeeze(-1)  # (B, N)

        # Apply mask
        A = A.masked_fill(~masks, float('-inf'))

        # Combine with instance weights
        if instance_weights is not None:
            weights_norm = instance_weights.masked_fill(~masks, 0)
            weights_norm = weights_norm / (weights_norm.sum(dim=1, keepdim=True) + 1e-8)
            A = A + torch.log(weights_norm + 1e-8)

        # Softmax
        A = torch.softmax(A, dim=1)

        # Aggregate
        aggregated = torch.bmm(A.unsqueeze(1), H).squeeze(1)  # (B, hidden_dim)

        return aggregated, A


class MultiScaleAttentionMIL(nn.Module):
    """
    Multi-scale attention MIL model.

    Architecture:
    1. Cluster-level attention on centroids
    2. Sequence-level attention on raw sequences
    3. Fusion of both scales
    4. Classification
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
        fusion_type: str = 'concat'  # 'concat', 'add', 'weighted'
    ):
        super().__init__()

        self.fusion_type = fusion_type

        # Cluster-level attention
        self.cluster_attention = AttentionModule(input_dim, hidden_dim, dropout)

        # Sequence-level attention
        self.sequence_attention = AttentionModule(input_dim, hidden_dim, dropout)

        # Fusion
        if fusion_type == 'concat':
            fusion_dim = hidden_dim * 2
        elif fusion_type == 'add':
            fusion_dim = hidden_dim
        elif fusion_type == 'weighted':
            fusion_dim = hidden_dim
            self.fusion_weight = nn.Parameter(torch.tensor([0.5, 0.5]))
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, batch):
        """
        Args:
            batch: Dict with cluster and sequence data

        Returns:
            logits: (B, num_classes)
            attention_dict: Dict with attention weights
        """
        # Cluster-level attention
        cluster_agg, cluster_att = self.cluster_attention(
            batch['cluster_centroids'],
            batch['cluster_masks'],
            batch['cluster_weights']
        )

        # Sequence-level attention
        sequence_agg, sequence_att = self.sequence_attention(
            batch['sequence_embeddings'],
            batch['sequence_masks'],
            batch['sequence_weights']
        )

        # Fusion
        if self.fusion_type == 'concat':
            fused = torch.cat([cluster_agg, sequence_agg], dim=1)
        elif self.fusion_type == 'add':
            fused = cluster_agg + sequence_agg
        elif self.fusion_type == 'weighted':
            weights = torch.softmax(self.fusion_weight, dim=0)
            fused = weights[0] * cluster_agg + weights[1] * sequence_agg

        # Classify
        logits = self.classifier(fused)

        attention_dict = {
            'cluster_attention': cluster_att,
            'sequence_attention': sequence_att
        }

        return logits, attention_dict


def train_model(model, train_loader, val_loader, num_epochs=50, lr=1e-3,
                weight_decay=1e-4, device='cuda', patience=10, output_dir='./output'):
    """Train multi-scale MIL model."""

    os.makedirs(output_dir, exist_ok=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels = batch['labels']

            optimizer.zero_grad()
            logits, _ = model(batch)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Val
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                labels = batch['labels']

                logits, _ = model(batch)
                loss = criterion(logits, labels)

                val_loss += loss.item() * len(labels)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_loss /= val_total
        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
            print(f"  ✓ Best model (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pt')))

    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return history


def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model."""
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels = batch['labels']

            logits, _ = model(batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    auc = None
    if all_probs.shape[1] == 2:
        try:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        except:
            pass

    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Scale Attention MIL")
    parser.add_argument('--clustered_pkl', required=True, help='Path to clustered data pickle')
    parser.add_argument('--repertoire_json', required=True, help='Path to original repertoire JSON')
    parser.add_argument('--embeddings_path', required=True, help='Path to HDF5 embeddings')
    parser.add_argument('--top_clusters', type=int, default=10)
    parser.add_argument('--seqs_per_cluster', type=int, default=20)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--fusion_type', choices=['concat', 'add', 'weighted'], default='concat')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--test_size', type=float, default=0.1, help='Test set fraction (default: 0.1 for 80/10/10 split)')
    parser.add_argument('--val_size', type=float, default=0.111, help='Validation fraction from train_val (default: 0.111 for ~10% of total)')
    parser.add_argument('--output_dir', type=str, default='./results/multiscale_mil')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("MULTI-SCALE ATTENTION MIL")
    print("="*80)

    # Load data
    print(f"\nLoading clustered data from: {args.clustered_pkl}")
    with open(args.clustered_pkl, 'rb') as f:
        clustered_data = pickle.load(f)
    print(f"Loaded {len(clustered_data)} repertoires")

    print(f"\nLoading repertoire data from: {args.repertoire_json}")
    with open(args.repertoire_json, 'r') as f:
        repertoire_data = json.load(f)

    print(f"\nLoading embeddings from: {args.embeddings_path}")
    embedding_loader = get_embedding_loader(args.embeddings_path)

    # Split
    labels = [r['label'] for r in clustered_data]
    train_val_data, test_data = train_test_split(
        clustered_data, test_size=args.test_size, stratify=labels, random_state=args.seed
    )
    train_labels = [r['label'] for r in train_val_data]
    train_data, val_data = train_test_split(
        train_val_data, test_size=args.val_size, stratify=train_labels, random_state=args.seed
    )

    print(f"\nSplits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    # Collect all unique sequences for prefetching from repertoire_data
    # (The clustered data only has centroids, not raw sequences)
    print("\nCollecting sequences for prefetching from repertoire data...")
    
    # Build lookup from repertoire_id to sequences
    repertoire_lookup = {r['repertoire_id']: r for r in repertoire_data}
    
    # Get repertoire IDs in our splits
    split_rep_ids = set()
    for data in [train_data, val_data, test_data]:
        for item in data:
            split_rep_ids.add(item['repertoire_id'])
    
    # Collect sequences from matching repertoires
    all_sequences = set()
    for rep_id in split_rep_ids:
        if rep_id in repertoire_lookup:
            rep_data = repertoire_lookup[rep_id]
            sequences = rep_data.get('sequences', [])
            
            # Only sample top sequences by frequency to reduce memory usage
            frequencies = rep_data.get('frequency', [1.0] * len(sequences))
            max_seqs_per_rep = args.top_clusters * args.seqs_per_cluster
            
            if len(sequences) > max_seqs_per_rep:
                sorted_indices = np.argsort(frequencies)[::-1][:max_seqs_per_rep]
                sampled_seqs = [sequences[i] for i in sorted_indices]
                all_sequences.update(sampled_seqs)
            else:
                all_sequences.update(sequences)
    
    print(f"Found {len(all_sequences):,} unique sequences across {len(split_rep_ids)} repertoires")

    # Prefetch embeddings into cache
    embedding_loader.prefetch_sequences(all_sequences)

    # Create datasets
    train_dataset = MultiScaleMILDataset(
        train_data, repertoire_data, embedding_loader, args.top_clusters, args.seqs_per_cluster
    )
    val_dataset = MultiScaleMILDataset(
        val_data, repertoire_data, embedding_loader, args.top_clusters, args.seqs_per_cluster
    )
    test_dataset = MultiScaleMILDataset(
        test_data, repertoire_data, embedding_loader, args.top_clusters, args.seqs_per_cluster
    )

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_multiscale_batch, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_multiscale_batch, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_multiscale_batch, num_workers=0)

    # Create model
    num_classes = len(set(labels))
    model = MultiScaleAttentionMIL(
        input_dim=embedding_loader.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        dropout=args.dropout,
        fusion_type=args.fusion_type
    )

    print(f"\nModel: MultiScaleAttentionMIL")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    history = train_model(model, train_loader, val_loader, args.num_epochs, args.lr,
                           args.weight_decay, args.device, args.patience, args.output_dir)

    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    results = evaluate_model(model, test_loader, args.device)

    print(f"\nTest Results:")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1:        {results['f1']:.4f}")
    if results['auc']:
        print(f"  AUC:       {results['auc']:.4f}")

    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {args.output_dir}")

    embedding_loader.close()


if __name__ == '__main__':
    main()
