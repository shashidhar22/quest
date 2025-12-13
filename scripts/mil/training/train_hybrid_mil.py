#!/usr/bin/env python3
"""
Hybrid MIL: Top-K + Random Sampling

Combines two sampling strategies:
1. Top-K most frequent sequences (captures clonal expansion)
2. Random sampling from rare sequences (captures diversity)

This approach ensures both common and rare clonotypes are represented,
providing better classification accuracy and interpretability.
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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# Add parent directory to path to import embedding_loaders
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding_loaders import get_embedding_loader


class StreamingEmbeddingLoader:
    """Load embeddings from HDF5 with caching."""

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
            with open(index_path, 'rb') as f:
                self.seq_to_idx = pickle.load(f)

            with h5py.File(self.h5_path, 'r') as f:
                self.embedding_dim = f['embeddings'].shape[1]

            print(f"Loaded index: {len(self.seq_to_idx):,} sequences, dim={self.embedding_dim}")
        else:
            # Build index
            print("Building index...")
            with h5py.File(self.h5_path, 'r') as f:
                sequences = f['sequences'][:]
                for idx, seq in enumerate(tqdm(sequences)):
                    if isinstance(seq, bytes):
                        seq = seq.decode('utf-8')
                    self.seq_to_idx[seq] = idx
                self.embedding_dim = f['embeddings'].shape[1]

            # Save index
            with open(index_path, 'wb') as f:
                pickle.dump(self.seq_to_idx, f)
            print(f"Built and saved index: {len(self.seq_to_idx):,} sequences")

    def get_batch(self, sequences: list) -> np.ndarray:
        """Get embeddings for sequences."""
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


class HybridMILDataset(Dataset):
    """MIL dataset with hybrid Top-K + Random sampling."""

    def __init__(
        self,
        repertoire_data: list,
        embedding_loader: StreamingEmbeddingLoader,
        top_k: int = 100,
        random_k: int = 100,
        use_frequency_weights: bool = True,
        return_sequences: bool = False
    ):
        """
        Args:
            top_k: Number of most frequent sequences to take
            random_k: Number of random rare sequences to sample
            use_frequency_weights: Whether to weight instances by frequency
            return_sequences: Whether to return sequence strings (for attention extraction)
        """
        self.repertoire_data = repertoire_data
        self.embedding_loader = embedding_loader
        self.top_k = top_k
        self.random_k = random_k
        self.use_frequency_weights = use_frequency_weights
        self.return_sequences = return_sequences

    def __len__(self):
        return len(self.repertoire_data)

    def __getitem__(self, idx: int):
        item = self.repertoire_data[idx]

        sequences = item['sequences']
        frequencies = item.get('frequency', [1.0] * len(sequences))

        # Sort by frequency
        sorted_indices = np.argsort(frequencies)[::-1]

        # Take top-K most frequent
        top_k_count = min(self.top_k, len(sequences))
        top_k_indices = sorted_indices[:top_k_count]
        top_k_seqs = [sequences[i] for i in top_k_indices]
        top_k_freqs = [frequencies[i] for i in top_k_indices]

        # Sample random from remaining sequences
        remaining_indices = sorted_indices[top_k_count:]
        if len(remaining_indices) > 0 and self.random_k > 0:
            random_k_count = min(self.random_k, len(remaining_indices))
            random_sample_indices = np.random.choice(
                remaining_indices, random_k_count, replace=False
            )
            random_k_seqs = [sequences[i] for i in random_sample_indices]
            random_k_freqs = [frequencies[i] for i in random_sample_indices]
        else:
            random_k_seqs = []
            random_k_freqs = []

        # Combine
        sampled_seqs = top_k_seqs + random_k_seqs
        sampled_freqs = top_k_freqs + random_k_freqs

        # Get embeddings
        embeddings = self.embedding_loader.get_batch(sampled_seqs)

        # Compute weights
        if self.use_frequency_weights and len(sampled_freqs) > 0:
            weights = np.array(sampled_freqs, dtype=np.float32)
            weights = weights / (weights.sum() + 1e-8)
        else:
            weights = np.ones(len(embeddings), dtype=np.float32) / (len(embeddings) + 1e-8)

        result = {
            'repertoire_id': item['repertoire_id'],
            'label': item['label'],
            'instances': torch.from_numpy(embeddings),
            'instance_weights': torch.from_numpy(weights),
            'bag_size': len(embeddings)
        }

        if self.return_sequences:
            result['sequences'] = sampled_seqs
            result['frequencies'] = sampled_freqs

        return result


def collate_mil_batch(batch: list):
    """Collate function for variable-length bags."""
    max_bag_size = max(b['bag_size'] for b in batch)
    embedding_dim = batch[0]['instances'].shape[1]

    instances = torch.zeros(len(batch), max_bag_size, embedding_dim)
    instance_weights = torch.zeros(len(batch), max_bag_size)
    masks = torch.zeros(len(batch), max_bag_size, dtype=torch.bool)

    for i, b in enumerate(batch):
        size = b['bag_size']
        instances[i, :size] = b['instances']
        instance_weights[i, :size] = b['instance_weights']
        masks[i, :size] = True

    result = {
        'instances': instances,
        'instance_weights': instance_weights,
        'masks': masks,
        'labels': torch.tensor([b['label'] for b in batch]),
        'repertoire_ids': [b['repertoire_id'] for b in batch]
    }

    # Include sequences if present
    if 'sequences' in batch[0]:
        result['sequences'] = [b['sequences'] for b in batch]
        result['frequencies'] = [b['frequencies'] for b in batch]

    return result


class AttentionMIL(nn.Module):
    """Attention-based MIL model with gated attention mechanism."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 2,
                 dropout: float = 0.3, use_instance_weights: bool = True):
        super().__init__()

        self.use_instance_weights = use_instance_weights

        # Instance encoder
        self.instance_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Gated attention
        self.attention_V = nn.Linear(hidden_dim, hidden_dim)
        self.attention_U = nn.Linear(hidden_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, instances, masks, instance_weights=None):
        """
        Args:
            instances: (B, N, D) instance embeddings
            masks: (B, N) boolean mask for valid instances
            instance_weights: (B, N) frequency weights

        Returns:
            logits: (B, num_classes) classification logits
            attention: (B, N) attention weights
        """
        H = self.instance_encoder(instances)  # (B, N, hidden_dim)

        # Gated attention
        A_V = torch.tanh(self.attention_V(H))
        A_U = torch.sigmoid(self.attention_U(H))
        A = self.attention_w(A_V * A_U).squeeze(-1)  # (B, N)

        # Apply mask
        A = A.masked_fill(~masks, float('-inf'))

        # Combine with instance weights
        if self.use_instance_weights and instance_weights is not None:
            weights_norm = instance_weights.masked_fill(~masks, 0)
            weights_norm = weights_norm / (weights_norm.sum(dim=1, keepdim=True) + 1e-8)
            A = A + torch.log(weights_norm + 1e-8)

        # Softmax
        A = torch.softmax(A, dim=1)

        # Aggregate
        M = torch.bmm(A.unsqueeze(1), H).squeeze(1)

        # Classify
        logits = self.classifier(M)

        return logits, A


def train_model(model, train_loader, val_loader, num_epochs=50, lr=1e-3,
                weight_decay=1e-4, device='cuda', patience=10, output_dir='./output'):
    """Train MIL model with early stopping."""

    os.makedirs(output_dir, exist_ok=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
            instances = batch['instances'].to(device)
            masks = batch['masks'].to(device)
            instance_weights = batch['instance_weights'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits, _ = model(instances, masks, instance_weights)
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
                instances = batch['instances'].to(device)
                masks = batch['masks'].to(device)
                instance_weights = batch['instance_weights'].to(device)
                labels = batch['labels'].to(device)

                logits, _ = model(instances, masks, instance_weights)
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
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
            print(f"  ✓ Best model (val_acc={val_acc:.4f}, val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pt')))

    # Save history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return history


def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model and compute metrics."""
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            instances = batch['instances'].to(device)
            masks = batch['masks'].to(device)
            instance_weights = batch['instance_weights'].to(device)
            labels = batch['labels'].to(device)

            logits, _ = model(instances, masks, instance_weights)
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
        'confusion_matrix': cm.tolist(),
        'all_labels': all_labels,
        'all_preds': all_preds,
        'all_probs': all_probs.tolist()
    }


def extract_attention_weights(model, dataset, device='cuda', top_n=10):
    """
    Extract attention weights for all repertoires.
    Identifies top-N most important TCRs per repertoire.
    """
    model.eval()
    model.to(device)

    attention_data = []

    for idx in tqdm(range(len(dataset)), desc="Extracting attention weights"):
        sample = dataset[idx]

        instances = sample['instances'].unsqueeze(0).to(device)  # (1, N, D)
        masks = torch.ones(1, sample['bag_size'], dtype=torch.bool).to(device)
        instance_weights = sample['instance_weights'].unsqueeze(0).to(device)

        with torch.no_grad():
            logits, attention = model(instances, masks, instance_weights)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1).item()

        attention = attention.squeeze(0).cpu().numpy()  # (N,)
        sequences = sample['sequences']
        frequencies = sample['frequencies']

        # Get top-N attended sequences
        top_indices = np.argsort(attention)[::-1][:top_n]

        top_sequences = []
        for i in top_indices:
            top_sequences.append({
                'sequence': sequences[i],
                'attention': float(attention[i]),
                'frequency': float(frequencies[i]),
                'rank': int(i) + 1
            })

        attention_data.append({
            'repertoire_id': sample['repertoire_id'],
            'label': sample['label'],
            'prediction': pred,
            'probability': probs.squeeze(0).cpu().numpy().tolist(),
            'top_sequences': top_sequences,
            'mean_attention': float(attention.mean()),
            'max_attention': float(attention.max()),
            'entropy': float(-np.sum(attention * np.log(attention + 1e-10)))
        })

    return attention_data


def plot_results(history, results, output_dir):
    """Plot training curves and confusion matrix."""

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train', linewidth=2)
    axes[1].plot(history['val_acc'], label='Validation', linewidth=2)
    axes[1].axhline(y=0.5, color='r', linestyle='--', label='Random', linewidth=1.5)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Confusion matrix
    cm = np.array(results['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Hybrid MIL: Top-K + Random Sampling")
    parser.add_argument('--data_path', required=True, help='Path to repertoire JSON data')
    parser.add_argument('--embeddings_path', required=True, help='Path to HDF5 embeddings')
    parser.add_argument('--top_k', type=int, default=100, help='Number of top frequent sequences')
    parser.add_argument('--random_k', type=int, default=100, help='Number of random rare sequences')
    parser.add_argument('--use_frequency_weights', action='store_true', default=True)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--test_size', type=float, default=0.1, help='Test set fraction (default: 0.1 for 80/10/10 split)')
    parser.add_argument('--val_size', type=float, default=0.111, help='Validation fraction from train_val (default: 0.111 for ~10% of total)')
    parser.add_argument('--output_dir', type=str, default='./results/hybrid_mil')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--extract_attention', action='store_true', help='Extract attention weights after training')
    parser.add_argument('--top_n_sequences', type=int, default=10, help='Top N sequences to extract per repertoire')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Save args
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("="*80)
    print(f"HYBRID MIL: Top-{args.top_k} + Random-{args.random_k}")
    print("="*80)

    # Load data
    print(f"\nLoading repertoire data from: {args.data_path}")
    with open(args.data_path, 'r') as f:
        repertoire_data = json.load(f)
    print(f"Loaded {len(repertoire_data)} repertoires")

    # Load embeddings (supports both .h5 and .parquet)
    print(f"\nLoading embeddings from: {args.embeddings_path}")
    embedding_loader = get_embedding_loader(args.embeddings_path)

    # Split
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
    print(f"  Val:   {len(val_data)} repertoires")
    print(f"  Test:  {len(test_data)} repertoires")

    # Collect all unique sequences for prefetching
    print("\nCollecting sequences for prefetching...")
    all_sequences = set()
    for data in [train_data, val_data, test_data]:
        for item in data:
            all_sequences.update(item['sequences'])
    print(f"Found {len(all_sequences):,} unique sequences across all splits")

    # Prefetch embeddings into cache (much faster than loading during training)
    embedding_loader.prefetch_sequences(all_sequences)

    # Create datasets
    train_dataset = HybridMILDataset(
        train_data, embedding_loader, args.top_k, args.random_k,
        args.use_frequency_weights, return_sequences=False
    )
    val_dataset = HybridMILDataset(
        val_data, embedding_loader, args.top_k, args.random_k,
        args.use_frequency_weights, return_sequences=False
    )
    test_dataset = HybridMILDataset(
        test_data, embedding_loader, args.top_k, args.random_k,
        args.use_frequency_weights, return_sequences=False
    )

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_mil_batch, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_mil_batch, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_mil_batch, num_workers=0)

    # Create model
    num_classes = len(set(labels))
    model = AttentionMIL(
        input_dim=embedding_loader.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        dropout=args.dropout,
        use_instance_weights=args.use_frequency_weights
    )

    print(f"\nModel: AttentionMIL")
    print(f"  Input dim:  {embedding_loader.embedding_dim}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Num classes: {num_classes}")
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

    print("\nTest Results:")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1:        {results['f1']:.4f}")
    if results['auc'] is not None:
        print(f"  AUC:       {results['auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(np.array(results['confusion_matrix']))

    # Save results
    results_to_save = {k: v for k, v in results.items() if k not in ['all_labels', 'all_preds']}
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump(results_to_save, f, indent=2)

    # Plot
    plot_results(history, results, args.output_dir)

    # Extract attention weights
    if args.extract_attention:
        print("\n" + "="*80)
        print("EXTRACTING ATTENTION WEIGHTS")
        print("="*80)

        # Create dataset with sequences for attention extraction
        test_dataset_with_seqs = HybridMILDataset(
            test_data, embedding_loader, args.top_k, args.random_k,
            args.use_frequency_weights, return_sequences=True
        )

        attention_data = extract_attention_weights(
            model, test_dataset_with_seqs, args.device, args.top_n_sequences
        )

        # Save attention data
        with open(os.path.join(args.output_dir, 'attention_weights.json'), 'w') as f:
            json.dump(attention_data, f, indent=2)

        print(f"\nExtracted attention weights for {len(attention_data)} repertoires")
        print(f"Top {args.top_n_sequences} sequences per repertoire saved")

    print(f"\n✅ Results saved to: {args.output_dir}")

    embedding_loader.close()


if __name__ == '__main__':
    main()
