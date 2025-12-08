#!/usr/bin/env python3
"""
train_global_cluster_mil.py
────────────────────────────────────────────────────────
MIL training using pre-computed global cluster centroids.

This script works with output from the global clustering pipeline:
- Each repertoire has cluster centroids as "instances" 
- Weights represent aggregate counts per cluster
- No embedding lookup needed - uses pre-computed centroids

Supports multiple MIL architectures:
- hybrid: Hybrid attention MIL
- transmil: Transformer MIL  
- dsmil: Dual-stream MIL
- abmil: Attention-based MIL

Usage:
    python train_global_cluster_mil.py \
        --data_path data/mil/clustered/train_dataset_1/kmeans/clustered.pkl \
        --model hybrid \
        --output_dir results/train_dataset_1_kmeans_hybrid
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, confusion_matrix
)
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')


# ============================================================================
# Dataset
# ============================================================================

class GlobalClusterDataset(Dataset):
    """Dataset using pre-computed global cluster centroids."""
    
    def __init__(
        self,
        repertoire_data: List[Dict],
        max_clusters: int = 100,
    ):
        """
        Args:
            repertoire_data: List of dicts with 'instances' (cluster centroids),
                            'instance_weights', and 'label'
            max_clusters: Maximum clusters to use per repertoire
        """
        self.repertoire_data = repertoire_data
        self.max_clusters = max_clusters
        
        # Get embedding dimension from first non-empty repertoire
        self.embedding_dim = None
        for r in repertoire_data:
            if 'instances' in r and len(r['instances']) > 0:
                instances = np.array(r['instances'])
                self.embedding_dim = instances.shape[1]
                break
        
        if self.embedding_dim is None:
            raise ValueError("Could not determine embedding dimension from data")
    
    def __len__(self):
        return len(self.repertoire_data)
    
    def __getitem__(self, idx: int):
        item = self.repertoire_data[idx]
        
        # Get cluster centroids and weights
        instances = np.array(item['instances'], dtype=np.float32)
        weights = np.array(item['instance_weights'], dtype=np.float32)
        label = int(item['label'])
        
        n_clusters = len(instances)
        
        if n_clusters == 0:
            # Empty repertoire - return zeros
            instances = np.zeros((1, self.embedding_dim), dtype=np.float32)
            weights = np.array([1.0], dtype=np.float32)
        elif n_clusters > self.max_clusters:
            # Sample top clusters by weight
            top_indices = np.argsort(weights)[::-1][:self.max_clusters]
            instances = instances[top_indices]
            weights = weights[top_indices]
        
        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights), dtype=np.float32) / len(weights)
        
        return {
            'instances': torch.tensor(instances),
            'weights': torch.tensor(weights),
            'label': torch.tensor(label, dtype=torch.long),
            'repertoire_id': item.get('repertoire_id', str(idx)),
            'n_clusters': len(instances),
        }


def collate_fn(batch):
    """Custom collate for variable-length bags."""
    max_len = max(b['instances'].shape[0] for b in batch)
    dim = batch[0]['instances'].shape[1]
    
    instances_padded = torch.zeros(len(batch), max_len, dim)
    weights_padded = torch.zeros(len(batch), max_len)
    masks = torch.zeros(len(batch), max_len, dtype=torch.bool)
    labels = torch.zeros(len(batch), dtype=torch.long)
    
    for i, b in enumerate(batch):
        n = b['instances'].shape[0]
        instances_padded[i, :n] = b['instances']
        weights_padded[i, :n] = b['weights']
        masks[i, :n] = True
        labels[i] = b['label']
    
    return {
        'instances': instances_padded,
        'weights': weights_padded,
        'mask': masks,
        'labels': labels,
    }


# ============================================================================
# Models
# ============================================================================

class AttentionMIL(nn.Module):
    """Standard attention-based MIL."""
    
    def __init__(self, input_dim, hidden_dim=256, num_classes=2, dropout=0.3):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, instances, weights=None, mask=None):
        # instances: (B, N, D)
        # weights: (B, N)
        # mask: (B, N)
        
        B, N, D = instances.shape
        
        # Feature extraction
        features = self.feature_extractor(instances)  # (B, N, H)
        
        # Attention scores
        attention_scores = self.attention(features).squeeze(-1)  # (B, N)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Weight by instance weights if provided
        if weights is not None:
            attention_weights = attention_weights * weights
            attention_weights = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Aggregate
        bag_repr = torch.sum(features * attention_weights.unsqueeze(-1), dim=1)  # (B, H)
        
        # Classify
        logits = self.classifier(bag_repr)  # (B, C)
        
        return logits, attention_weights


class HybridMIL(nn.Module):
    """Hybrid attention MIL with both global and local attention."""
    
    def __init__(self, input_dim, hidden_dim=256, num_classes=2, dropout=0.3):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Local attention (per-instance)
        self.local_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Global attention (cross-instance)
        self.global_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.global_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        
        # Classifier with both local and global representations
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, instances, weights=None, mask=None):
        B, N, D = instances.shape
        
        features = self.feature_extractor(instances)  # (B, N, H)
        
        # Local attention
        local_scores = self.local_attention(features).squeeze(-1)  # (B, N)
        if mask is not None:
            local_scores = local_scores.masked_fill(~mask, float('-inf'))
        local_weights = F.softmax(local_scores, dim=-1)
        
        if weights is not None:
            local_weights = local_weights * weights
            local_weights = local_weights / (local_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        local_repr = torch.sum(features * local_weights.unsqueeze(-1), dim=1)  # (B, H)
        
        # Global attention
        query = self.global_query.expand(B, -1, -1)  # (B, 1, H)
        key_padding_mask = ~mask if mask is not None else None
        global_repr, _ = self.global_attn(query, features, features, key_padding_mask=key_padding_mask)
        global_repr = global_repr.squeeze(1)  # (B, H)
        
        # Combine
        combined = torch.cat([local_repr, global_repr], dim=-1)  # (B, 2H)
        logits = self.classifier(combined)
        
        return logits, local_weights


class TransMIL(nn.Module):
    """Transformer-based MIL with positional encoding."""
    
    def __init__(self, input_dim, hidden_dim=256, num_classes=2, 
                 num_heads=4, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, instances, weights=None, mask=None):
        B, N, D = instances.shape
        
        features = self.input_proj(instances)  # (B, N, H)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, H)
        features = torch.cat([cls_tokens, features], dim=1)  # (B, N+1, H)
        
        # Update mask for CLS token
        if mask is not None:
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)
        
        # Transformer
        src_key_padding_mask = ~mask if mask is not None else None
        transformed = self.transformer(features, src_key_padding_mask=src_key_padding_mask)
        
        # Use CLS token representation
        cls_repr = transformed[:, 0]  # (B, H)
        
        logits = self.classifier(cls_repr)
        
        return logits, None


class DSMIL(nn.Module):
    """Dual-Stream MIL with instance and bag-level branches."""
    
    def __init__(self, input_dim, hidden_dim=256, num_classes=2, dropout=0.3):
        super().__init__()
        
        # Instance-level branch
        self.instance_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.instance_classifier = nn.Linear(hidden_dim, num_classes)
        
        # Bag-level branch with attention
        self.bag_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self.bag_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, instances, weights=None, mask=None):
        B, N, D = instances.shape
        
        # Instance encoding
        inst_features = self.instance_encoder(instances)  # (B, N, H)
        
        # Instance predictions
        inst_logits = self.instance_classifier(inst_features)  # (B, N, C)
        
        # Get max instance for supervision
        with torch.no_grad():
            inst_probs = F.softmax(inst_logits, dim=-1)
            # Pick instance with highest probability for positive class
            if inst_probs.shape[-1] > 1:
                max_inst_scores = inst_probs[:, :, 1]
            else:
                max_inst_scores = inst_probs.squeeze(-1)
            
            if mask is not None:
                max_inst_scores = max_inst_scores.masked_fill(~mask, float('-inf'))
            
            max_inst_idx = max_inst_scores.argmax(dim=1)  # (B,)
        
        # Bag attention
        attn_scores = self.bag_attention(inst_features).squeeze(-1)  # (B, N)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Aggregate
        bag_repr = torch.sum(inst_features * attn_weights.unsqueeze(-1), dim=1)  # (B, H)
        
        # Bag prediction
        bag_logits = self.bag_classifier(bag_repr)
        
        # Combine instance and bag predictions
        batch_idx = torch.arange(B, device=instances.device)
        max_inst_logits = inst_logits[batch_idx, max_inst_idx]  # (B, C)
        
        logits = 0.5 * bag_logits + 0.5 * max_inst_logits
        
        return logits, attn_weights


def get_model(model_type: str, input_dim: int, hidden_dim: int, 
              num_classes: int, dropout: float, **kwargs) -> nn.Module:
    """Factory function for models."""
    if model_type == 'hybrid':
        return HybridMIL(input_dim, hidden_dim, num_classes, dropout)
    elif model_type == 'transmil':
        return TransMIL(input_dim, hidden_dim, num_classes, 
                        kwargs.get('num_heads', 4), kwargs.get('num_layers', 2), dropout)
    elif model_type == 'dsmil':
        return DSMIL(input_dim, hidden_dim, num_classes, dropout)
    elif model_type in ['abmil', 'attention']:
        return AttentionMIL(input_dim, hidden_dim, num_classes, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(loader, desc="Training", leave=False):
        instances = batch['instances'].to(device)
        weights = batch['weights'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        logits, _ = model(instances, weights, mask)
        loss = F.cross_entropy(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(loader), all_preds, all_labels


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            instances = batch['instances'].to(device)
            weights = batch['weights'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits, _ = model(instances, weights, mask)
            loss = F.cross_entropy(logits, labels)
            
            probs = F.softmax(logits, dim=-1)
            
            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
    
    return {
        'loss': total_loss / len(loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels,
    }


def train_model(
    model, train_loader, val_loader, 
    num_epochs, lr, weight_decay, device, 
    patience=10, output_dir=None
):
    """Train with early stopping."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_auc = 0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        train_loss, train_preds, train_labels = train_epoch(model, train_loader, optimizer, device)
        train_acc = accuracy_score(train_labels, train_preds)
        
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
              f"Val Loss={val_metrics['loss']:.4f}, Val AUC={val_metrics['auc']:.4f}, Val Acc={val_metrics['accuracy']:.4f}")
        
        # Early stopping
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch
            patience_counter = 0
            
            if output_dir:
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"\nBest validation AUC: {best_val_auc:.4f} at epoch {best_epoch+1}")
    
    # Load best model
    if output_dir and os.path.exists(os.path.join(output_dir, 'best_model.pt')):
        model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pt')))
    
    return history, best_val_auc, best_epoch


def main():
    parser = argparse.ArgumentParser(description="Train MIL on global cluster centroids")
    parser.add_argument('--data_path', required=True, 
                        help='Path to clustered.pkl file')
    parser.add_argument('--model', choices=['hybrid', 'transmil', 'dsmil', 'abmil'], 
                        default='hybrid', help='MIL architecture')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--max_clusters', type=int, default=100,
                        help='Maximum clusters per repertoire')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--test_size', type=float, default=0.1,
                        help='Test set fraction (default: 0.1 for 80/10/10 split)')
    parser.add_argument('--val_size', type=float, default=0.111,
                        help='Validation fraction from train_val (default: 0.111 for ~10% of total)')
    parser.add_argument('--output_dir', type=str, default='./results/global_cluster_mil')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print(f"GLOBAL CLUSTER MIL: {args.model.upper()}")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading data from: {args.data_path}")
    with open(args.data_path, 'rb') as f:
        repertoire_data = pickle.load(f)
    print(f"Loaded {len(repertoire_data)} repertoires")
    
    # Check data format
    if 'instances' not in repertoire_data[0]:
        raise ValueError("Data does not contain 'instances' field. "
                        "Use assign_repertoire_clusters.py to generate proper format.")
    
    # Split data
    labels = [r['label'] for r in repertoire_data]
    train_val, test_data = train_test_split(
        repertoire_data, test_size=args.test_size, stratify=labels, random_state=args.seed
    )
    train_labels = [r['label'] for r in train_val]
    train_data, val_data = train_test_split(
        train_val, test_size=args.val_size, stratify=train_labels, random_state=args.seed
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_data)} ({sum(1 for r in train_data if r['label']==1)} positive)")
    print(f"  Val: {len(val_data)} ({sum(1 for r in val_data if r['label']==1)} positive)")
    print(f"  Test: {len(test_data)} ({sum(1 for r in test_data if r['label']==1)} positive)")
    
    # Create datasets
    train_dataset = GlobalClusterDataset(train_data, args.max_clusters)
    val_dataset = GlobalClusterDataset(val_data, args.max_clusters)
    test_dataset = GlobalClusterDataset(test_data, args.max_clusters)
    
    embedding_dim = train_dataset.embedding_dim
    print(f"\nEmbedding dimension: {embedding_dim}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    
    # Create model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = get_model(
        args.model, embedding_dim, args.hidden_dim, 
        num_classes=2, dropout=args.dropout
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Train
    print(f"\nTraining for up to {args.num_epochs} epochs...")
    history, best_val_auc, best_epoch = train_model(
        model, train_loader, val_loader,
        args.num_epochs, args.lr, args.weight_decay,
        device, args.patience, args.output_dir
    )
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(test_metrics['labels'], test_metrics['predictions'])
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
    
    # Save results
    results = {
        'model': args.model,
        'data_path': args.data_path,
        'test_accuracy': float(test_metrics['accuracy']),
        'test_auc': float(test_metrics['auc']),
        'test_precision': float(test_metrics['precision']),
        'test_recall': float(test_metrics['recall']),
        'test_f1': float(test_metrics['f1']),
        'best_val_auc': float(best_val_auc),
        'best_epoch': int(best_epoch),
        'num_epochs_trained': len(history['train_loss']),
        'confusion_matrix': cm.tolist(),
        'args': vars(args),
        'timestamp': datetime.now().isoformat(),
    }
    
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Save history
    history_path = os.path.join(args.output_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
