#!/usr/bin/env python3
"""
train_clustering_mil.py
────────────────────────────────────────────────────────
Clustering-based Multiple Instance Learning for TCR repertoires.

Approach:
1. Load pre-computed embeddings for all unique TCR sequences
2. For each repertoire:
   - Cluster TCR embeddings (K-means, HDBSCAN, etc.)
   - Create cluster-level representations (mean, weighted by frequency)
3. Use attention-based MIL to classify repertoires based on cluster representations

This reduces the number of instances per bag while preserving diversity.
"""

import os
import json
import argparse
import numpy as np
from typing import Dict, List, Any, Tuple
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.cluster import MiniBatchKMeans
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class StreamingEmbeddingLoader:
    """Memory-efficient embedding loader."""
    
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self.h5file = None
        self.seq_to_idx = {}
        self.embedding_dim = None
        self._load_index()
    
    def _load_index(self):
        """Load sequence index."""
        index_path = self.h5_path + '.idx.pkl'
        
        if os.path.exists(index_path):
            import pickle
            with open(index_path, 'rb') as f:
                self.seq_to_idx = pickle.load(f)
            
            with h5py.File(self.h5_path, 'r') as f:
                self.embedding_dim = f['embeddings'].shape[1]
            
            print(f"Loaded index: {len(self.seq_to_idx):,} sequences, dim={self.embedding_dim}")
        else:
            raise FileNotFoundError(f"Index not found: {index_path}")
    
    def get_batch(self, sequences: List[str]) -> np.ndarray:
        """Get embeddings for a batch of sequences."""
        if self.h5file is None:
            self.h5file = h5py.File(self.h5_path, 'r')
        
        embeddings = []
        for seq in sequences:
            if seq in self.seq_to_idx:
                idx = self.seq_to_idx[seq]
                embeddings.append(self.h5file['embeddings'][idx])
        
        if not embeddings:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        
        return np.array(embeddings, dtype=np.float32)
    
    def close(self):
        if self.h5file is not None:
            self.h5file.close()


def cluster_embeddings(
    embeddings: np.ndarray,
    frequencies: np.ndarray,
    n_clusters: int,
    method: str = 'kmeans'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster embeddings and create cluster-level representations.
    
    Args:
        embeddings: (N, D) array of embeddings
        frequencies: (N,) array of sequence frequencies
        n_clusters: Number of clusters
        method: Clustering method ('kmeans')
    
    Returns:
        cluster_embeddings: (K, D) cluster centroids
        cluster_weights: (K,) cluster weights (sum of frequencies)
    """
    if embeddings.shape[0] == 0:
        return np.zeros((0, embeddings.shape[1]), dtype=np.float32), np.zeros(0, dtype=np.float32)
    
    # Adjust n_clusters if we have fewer sequences
    n_clusters = min(n_clusters, embeddings.shape[0])
    
    if n_clusters == 1 or embeddings.shape[0] <= n_clusters:
        # Too few sequences, just use weighted mean
        weights = frequencies / frequencies.sum()
        cluster_emb = np.average(embeddings, axis=0, weights=weights).reshape(1, -1)
        cluster_weights = np.array([frequencies.sum()], dtype=np.float32)
        return cluster_emb, cluster_weights
    
    # Cluster
    if method == 'kmeans':
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024)
        labels = kmeans.fit_predict(embeddings)
        
        # Compute cluster representations weighted by frequency
        cluster_embeddings = []
        cluster_weights = []
        
        for k in range(n_clusters):
            mask = labels == k
            if mask.sum() == 0:
                continue
            
            cluster_seqs = embeddings[mask]
            cluster_freqs = frequencies[mask]
            
            # Weighted mean
            weights = cluster_freqs / cluster_freqs.sum()
            cluster_emb = np.average(cluster_seqs, axis=0, weights=weights)
            cluster_weight = cluster_freqs.sum()
            
            cluster_embeddings.append(cluster_emb)
            cluster_weights.append(cluster_weight)
        
        return np.array(cluster_embeddings, dtype=np.float32), np.array(cluster_weights, dtype=np.float32)
    
    else:
        raise ValueError(f"Unknown clustering method: {method}")


class ClusteringMILDataset(Dataset):
    """Dataset that creates cluster-level bags for MIL."""
    
    def __init__(
        self,
        repertoire_data: List[Dict[str, Any]],
        embedding_loader: StreamingEmbeddingLoader,
        n_clusters: int = 50,
        clustering_method: str = 'kmeans',
        normalize_weights: bool = True
    ):
        self.repertoire_data = repertoire_data
        self.embedding_loader = embedding_loader
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.normalize_weights = normalize_weights
        
        print(f"\nClustering configuration:")
        print(f"  Method: {clustering_method}")
        print(f"  Clusters per repertoire: {n_clusters}")
        print(f"  Normalize weights: {normalize_weights}")
    
    def __len__(self) -> int:
        return len(self.repertoire_data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.repertoire_data[idx]
        
        # Get embeddings
        embeddings = self.embedding_loader.get_batch(item['sequences'])
        
        # Get frequencies
        if 'frequency' in item and len(item['frequency']) > 0:
            frequencies = np.array(item['frequency'][:embeddings.shape[0]], dtype=np.float32)
        else:
            frequencies = np.ones(embeddings.shape[0], dtype=np.float32)
        
        # Cluster (this is slow - happens every time)
        if embeddings.shape[0] > 0:
            cluster_embs, cluster_weights = cluster_embeddings(
                embeddings, frequencies, self.n_clusters, self.clustering_method
            )
            
            # Normalize weights
            if self.normalize_weights and cluster_weights.sum() > 0:
                cluster_weights = cluster_weights / cluster_weights.sum()
        else:
            cluster_embs = np.zeros((1, self.embedding_loader.embedding_dim), dtype=np.float32)
            cluster_weights = np.ones(1, dtype=np.float32)
        
        return {
            'repertoire_id': item['repertoire_id'],
            'label': item['label'],
            'instances': torch.from_numpy(cluster_embs),  # (K, D)
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


class AttentionMIL(nn.Module):
    """
    Attention-based MIL model.
    
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


def train_model(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    num_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    device: str = "cuda",
    patience: int = 10,
    output_dir: str = "./output",
    use_wandb: bool = False
) -> Dict[str, List[float]]:
    """Train MIL model."""
    
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
        
        if use_wandb and WANDB_AVAILABLE:
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


def evaluate_model(
    model: nn.Module,
    test_dataset: Dataset,
    batch_size: int = 16,
    device: str = "cuda"
) -> Dict[str, Any]:
    """Evaluate MIL model."""
    
    model.eval()
    model.to(device)
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_mil_batch, num_workers=0
    )
    
    all_preds = []
    all_labels = []
    all_probs = []
    
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
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description="Clustering-based MIL classification")
    
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--embeddings_path", type=str, required=True)
    parser.add_argument("--n_clusters", type=int, default=50,
                        help="Number of clusters per repertoire")
    parser.add_argument("--clustering_method", type=str, default='kmeans',
                        choices=['kmeans'])
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use_instance_weights", action='store_true', default=True,
                        help="Use frequency-based instance weights in attention")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="./results/clustering_mil")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Initialize W&B
    use_wandb = args.wandb_project is not None and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    
    print("="*80)
    print("CLUSTERING-BASED MIL TRAINING")
    print("="*80)
    
    # Load data
    print(f"\nLoading data from: {args.data_path}")
    with open(args.data_path, 'r') as f:
        repertoire_data = json.load(f)
    print(f"Loaded {len(repertoire_data)} repertoires")
    
    # Load embedding loader
    embedding_loader = StreamingEmbeddingLoader(args.embeddings_path)
    
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
    print(f"  Train: {len(train_data)}")
    print(f"  Val: {len(val_data)}")
    print(f"  Test: {len(test_data)}")
    
    # Create datasets
    print(f"\nCreating MIL datasets...")
    train_dataset = ClusteringMILDataset(
        train_data, embedding_loader, args.n_clusters, args.clustering_method
    )
    val_dataset = ClusteringMILDataset(
        val_data, embedding_loader, args.n_clusters, args.clustering_method
    )
    test_dataset = ClusteringMILDataset(
        test_data, embedding_loader, args.n_clusters, args.clustering_method
    )
    
    # Create model
    num_classes = len(set(labels))
    model = AttentionMIL(
        input_dim=embedding_loader.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        dropout=args.dropout,
        use_instance_weights=args.use_instance_weights
    )
    
    print(f"\nModel architecture:")
    print(f"  Input dim: {embedding_loader.embedding_dim}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Num classes: {num_classes}")
    print(f"  Use instance weights: {args.use_instance_weights}")
    
    # Train
    history = train_model(
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
    
    # Plot training curves
    os.makedirs(args.output_dir, exist_ok=True)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'), dpi=150)
    
    # Evaluate
    print("\nEvaluating on test set...")
    results = evaluate_model(model, test_dataset, batch_size=args.batch_size, device=args.device)
    
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1:        {results['f1']:.4f}")
    if results['auc'] is not None:
        print(f"AUC:       {results['auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(np.array(results['confusion_matrix']))
    
    # Save results
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'), dpi=150)
    
    if use_wandb:
        wandb.log({
            'test/accuracy': results['accuracy'],
            'test/precision': results['precision'],
            'test/recall': results['recall'],
            'test/f1': results['f1'],
        })
        if results['auc'] is not None:
            wandb.log({'test/auc': results['auc']})
        wandb.finish()
    
    print(f"\n✅ Results saved to: {args.output_dir}")
    print("="*80)
    
    # Close embedding loader
    embedding_loader.close()


if __name__ == "__main__":
    main()
