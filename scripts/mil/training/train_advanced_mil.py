#!/usr/bin/env python3
"""
Advanced MIL Architectures

Implements state-of-the-art MIL methods:
1. DSMIL (Dual-Stream MIL): Li et al., CVPR 2021
   - Combines max pooling and attention pooling
   - Two parallel streams: instance-level and bag-level

2. TransMIL (Transformer MIL): Shao et al., NeurIPS 2021
   - Full self-attention between all instances
   - Positional encoding for sequence order
   - Hierarchical transformer blocks

3. ABMIL+ (Attention-Based MIL with Multi-Task Learning)
   - Classification + frequency prediction
   - Regularizes attention to find discriminative TCRs
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from tqdm.auto import tqdm
import pickle
import h5py
import matplotlib.pyplot as plt

# Import optimized embedding loader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding_loaders import get_embedding_loader


class SimpleMILDataset(Dataset):
    """Dataset for advanced MIL models."""

    def __init__(self, repertoire_data: list, embedding_loader,
                 sampling_strategy: str = 'hybrid', k: int = 200):
        self.repertoire_data = repertoire_data
        self.embedding_loader = embedding_loader
        self.sampling_strategy = sampling_strategy
        self.k = k

    def __len__(self):
        return len(self.repertoire_data)

    def __getitem__(self, idx: int):
        item = self.repertoire_data[idx]
        sequences = item['sequences']
        frequencies = item.get('frequency', [1.0] * len(sequences))

        # Hybrid sampling: top-K/2 + random K/2
        if self.sampling_strategy == 'hybrid':
            k_top = self.k // 2
            k_rand = self.k - k_top

            sorted_indices = np.argsort(frequencies)[::-1]
            top_indices = sorted_indices[:min(k_top, len(sequences))]
            remaining_indices = sorted_indices[len(top_indices):]

            if len(remaining_indices) > 0:
                rand_indices = np.random.choice(remaining_indices, min(k_rand, len(remaining_indices)), replace=False)
                sampled_indices = np.concatenate([top_indices, rand_indices])
            else:
                sampled_indices = top_indices

            sampled_seqs = [sequences[i] for i in sampled_indices]
            sampled_freqs = [frequencies[i] for i in sampled_indices]
        else:
            sampled_seqs = sequences[:self.k]
            sampled_freqs = frequencies[:self.k]

        embeddings = self.embedding_loader.get_batch(sampled_seqs)
        weights = np.array(sampled_freqs, dtype=np.float32)
        weights = weights / (weights.sum() + 1e-8)

        return {
            'repertoire_id': item['repertoire_id'],
            'label': item['label'],
            'instances': torch.from_numpy(embeddings),
            'instance_weights': torch.from_numpy(weights),
            'bag_size': len(embeddings)
        }


def collate_fn(batch: list):
    """Collate variable-length bags."""
    max_size = max(b['bag_size'] for b in batch)
    dim = batch[0]['instances'].shape[1]

    instances = torch.zeros(len(batch), max_size, dim)
    weights = torch.zeros(len(batch), max_size)
    masks = torch.zeros(len(batch), max_size, dtype=torch.bool)

    for i, b in enumerate(batch):
        size = b['bag_size']
        instances[i, :size] = b['instances']
        weights[i, :size] = b['instance_weights']
        masks[i, :size] = True

    return {
        'instances': instances,
        'instance_weights': weights,
        'masks': masks,
        'labels': torch.tensor([b['label'] for b in batch]),
        'repertoire_ids': [b['repertoire_id'] for b in batch]
    }


# ============================================================================
# DSMIL: Dual-Stream MIL
# ============================================================================

class DSMIL(nn.Module):
    """
    Dual-Stream Multiple Instance Learning (DSMIL).
    Paper: Li et al., "Dual-stream Multiple Instance Learning Network for Whole Slide Image Classification with Self-supervised Contrastive Learning", CVPR 2021.

    Combines two streams:
    1. Instance-level: Max pooling + critical instance selection
    2. Bag-level: Attention pooling

    Architecture helps capture both the most discriminative instance and the overall bag structure.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        # Instance-level stream (critical instance)
        self.instance_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.instance_classifier = nn.Sequential(
            nn.Linear(hidden_dim, num_classes)
        )

        # Bag-level stream (attention pooling)
        self.bag_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.bag_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, instances, masks, instance_weights=None):
        """
        Args:
            instances: (B, N, D)
            masks: (B, N)
            instance_weights: (B, N) optional

        Returns:
            bag_logits: (B, num_classes) primary output
            instance_logits: (B, num_classes) auxiliary output from critical instance
            attention: (B, N) attention weights
        """
        # Instance-level stream: find critical instance via max pooling
        H_instance = self.instance_encoder(instances)  # (B, N, hidden_dim)

        # Max pooling over instances
        H_instance_masked = H_instance.clone()
        H_instance_masked[~masks.unsqueeze(-1).expand_as(H_instance)] = float('-inf')
        critical_instance = torch.max(H_instance_masked, dim=1)[0]  # (B, hidden_dim)

        instance_logits = self.instance_classifier(critical_instance)  # (B, num_classes)

        # Bag-level stream: attention pooling
        H_bag = self.bag_encoder(instances)  # (B, N, hidden_dim)

        A = self.attention(H_bag).squeeze(-1)  # (B, N)
        A = A.masked_fill(~masks, float('-inf'))
        A = torch.softmax(A, dim=1)

        # Aggregate
        bag_representation = torch.bmm(A.unsqueeze(1), H_bag).squeeze(1)  # (B, hidden_dim)
        bag_logits = self.bag_classifier(bag_representation)  # (B, num_classes)

        return bag_logits, instance_logits, A


# ============================================================================
# TransMIL: Transformer-based MIL
# ============================================================================

class TransMIL(nn.Module):
    """
    Transformer-based Multiple Instance Learning (TransMIL).
    Paper: Shao et al., "TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification", NeurIPS 2021.

    Uses self-attention to model relationships between all instances.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 2,
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, 500, hidden_dim) * 0.02)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

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
            instances: (B, N, D)
            masks: (B, N)

        Returns:
            logits: (B, num_classes)
            attention: (B, N)
        """
        B, N, D = instances.shape

        # Project to hidden dim
        H = self.input_projection(instances)  # (B, N, hidden_dim)

        # Add positional encoding
        H = H + self.pos_encoding[:, :N, :]

        # Transformer
        # Create attention mask for transformer (inverted: True = ignore)
        src_key_padding_mask = ~masks  # (B, N)
        H = self.transformer(H, src_key_padding_mask=src_key_padding_mask)  # (B, N, hidden_dim)

        # Attention pooling
        A = self.attention(H).squeeze(-1)  # (B, N)
        A = A.masked_fill(~masks, float('-inf'))
        A = torch.softmax(A, dim=1)

        # Aggregate
        M = torch.bmm(A.unsqueeze(1), H).squeeze(1)  # (B, hidden_dim)

        # Classify
        logits = self.classifier(M)

        return logits, A


# ============================================================================
# ABMIL+ with Multi-Task Learning
# ============================================================================

class ABMILPlus(nn.Module):
    """
    Attention-Based MIL with Multi-Task Learning.

    Joint training on:
    1. Classification (primary task)
    2. Frequency prediction (auxiliary task)

    The auxiliary task helps regularize attention weights.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Attention
        self.attention_V = nn.Linear(hidden_dim, hidden_dim)
        self.attention_U = nn.Linear(hidden_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Frequency prediction head (auxiliary)
        self.frequency_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, instances, masks, instance_weights=None):
        """
        Returns:
            logits: (B, num_classes)
            freq_pred: (B, N) predicted frequencies
            attention: (B, N)
        """
        H = self.encoder(instances)  # (B, N, hidden_dim)

        # Gated attention
        A_V = torch.tanh(self.attention_V(H))
        A_U = torch.sigmoid(self.attention_U(H))
        A = self.attention_w(A_V * A_U).squeeze(-1)  # (B, N)

        A = A.masked_fill(~masks, float('-inf'))
        A = torch.softmax(A, dim=1)

        # Aggregate for classification
        M = torch.bmm(A.unsqueeze(1), H).squeeze(1)
        logits = self.classifier(M)

        # Predict frequencies (auxiliary task)
        freq_pred = self.frequency_predictor(H).squeeze(-1)  # (B, N)
        freq_pred = freq_pred.masked_fill(~masks, 0)

        return logits, freq_pred, A


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_model(model, train_loader, val_loader, model_type='dsmil',
                num_epochs=50, lr=1e-3, weight_decay=1e-4, device='cuda',
                patience=10, output_dir='./output'):
    """Train advanced MIL model."""

    os.makedirs(output_dir, exist_ok=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()

    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False):
            instances = batch['instances'].to(device)
            masks = batch['masks'].to(device)
            weights = batch['instance_weights'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            if model_type == 'dsmil':
                bag_logits, instance_logits, _ = model(instances, masks, weights)
                loss = 0.5 * criterion(bag_logits, labels) + 0.5 * criterion(instance_logits, labels)
                logits = bag_logits
            elif model_type == 'abmil_plus':
                logits, freq_pred, _ = model(instances, masks, weights)
                cls_loss = criterion(logits, labels)
                freq_loss = mse_criterion(freq_pred, weights)
                loss = cls_loss + 0.1 * freq_loss
            else:  # transmil
                logits, _ = model(instances, masks, weights)
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
                weights = batch['instance_weights'].to(device)
                labels = batch['labels'].to(device)

                if model_type == 'dsmil':
                    bag_logits, instance_logits, _ = model(instances, masks, weights)
                    loss = 0.5 * criterion(bag_logits, labels) + 0.5 * criterion(instance_logits, labels)
                    logits = bag_logits
                elif model_type == 'abmil_plus':
                    logits, freq_pred, _ = model(instances, masks, weights)
                    cls_loss = criterion(logits, labels)
                    freq_loss = mse_criterion(freq_pred, weights)
                    loss = cls_loss + 0.1 * freq_loss
                else:
                    logits, _ = model(instances, masks, weights)
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

        print(f"Epoch {epoch+1}: Train={train_acc:.4f}, Val={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping")
                break

    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pt')))
    return history


def evaluate_model(model, test_loader, model_type='dsmil', device='cuda'):
    """Evaluate model."""
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            instances = batch['instances'].to(device)
            masks = batch['masks'].to(device)
            weights = batch['instance_weights'].to(device)
            labels = batch['labels'].to(device)

            if model_type == 'dsmil':
                bag_logits, _, _ = model(instances, masks, weights)
                logits = bag_logits
            elif model_type == 'abmil_plus':
                logits, _, _ = model(instances, masks, weights)
            else:
                logits, _ = model(instances, masks, weights)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    auc = roc_auc_score(all_labels, all_probs[:, 1]) if all_probs.shape[1] == 2 else None
    cm = confusion_matrix(all_labels, all_preds)

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc, 'confusion_matrix': cm.tolist()}


def main():
    parser = argparse.ArgumentParser(description="Advanced MIL architectures")
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--embeddings_path', required=True)
    parser.add_argument('--model_type', choices=['dsmil', 'transmil', 'abmil_plus'], default='dsmil')
    parser.add_argument('--k', type=int, default=200, help='Instances per bag')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4, help='TransMIL attention heads')
    parser.add_argument('--num_layers', type=int, default=2, help='TransMIL layers')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='./results/advanced_mil')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_size', type=float, default=0.1, help='Test set fraction (default: 0.1 for 80/10/10 split)')
    parser.add_argument('--val_size', type=float, default=0.111, help='Validation fraction from train_val (default: 0.111 for ~10% of total)')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("="*80)
    print(f"ADVANCED MIL: {args.model_type.upper()}")
    print("="*80)

    # Load data
    with open(args.data_path, 'r') as f:
        repertoire_data = json.load(f)

    embedding_loader = get_embedding_loader(args.embeddings_path)

    # Split
    labels = [r['label'] for r in repertoire_data]
    train_val, test_data = train_test_split(repertoire_data, test_size=args.test_size, stratify=labels, random_state=args.seed)
    train_labels = [r['label'] for r in train_val]
    train_data, val_data = train_test_split(train_val, test_size=args.val_size, stratify=train_labels, random_state=args.seed)

    # Collect all unique sequences for prefetching
    print("\nCollecting sequences for prefetching...")
    all_sequences = set()
    for data in [train_data, val_data, test_data]:
        for item in data:
            all_sequences.update(item['sequences'])
    print(f"Found {len(all_sequences):,} unique sequences across all splits")

    # Prefetch embeddings into cache
    embedding_loader.prefetch_sequences(all_sequences)

    # Datasets
    train_dataset = SimpleMILDataset(train_data, embedding_loader, 'hybrid', args.k)
    val_dataset = SimpleMILDataset(val_data, embedding_loader, 'hybrid', args.k)
    test_dataset = SimpleMILDataset(test_data, embedding_loader, 'hybrid', args.k)

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    num_classes = len(set(labels))
    if args.model_type == 'dsmil':
        model = DSMIL(embedding_loader.embedding_dim, args.hidden_dim, num_classes, args.dropout)
    elif args.model_type == 'transmil':
        model = TransMIL(embedding_loader.embedding_dim, args.hidden_dim, num_classes,
                         args.num_heads, args.num_layers, args.dropout)
    else:  # abmil_plus
        model = ABMILPlus(embedding_loader.embedding_dim, args.hidden_dim, num_classes, args.dropout)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    history = train_model(model, train_loader, val_loader, args.model_type,
                           args.num_epochs, args.lr, args.weight_decay,
                           args.device, args.patience, args.output_dir)

    # Evaluate
    results = evaluate_model(model, test_loader, args.model_type, args.device)

    print(f"\nTest: Acc={results['accuracy']:.4f}, F1={results['f1']:.4f}, AUC={results['auc']:.4f}")

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    embedding_loader.close()


if __name__ == '__main__':
    main()
