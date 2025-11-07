#!/usr/bin/env python3
"""
repertoire_mil.py
────────────────────────────────────────────────────────
Multiple Instance Learning (MIL) for TCR Repertoire Classification

Given a repertoire (bag) of TCRs (instances), predicts:
1. The repertoire-level label (e.g., disease status, response, etc.)
2. Top 100 most important TCRs (features) contributing to the prediction

Uses embeddings from a fine-tuned ProtBERT model as TCR representations.
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import PeftModel

# Optional: W&B for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# 1. EMBEDDING EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

class TCREmbeddingExtractor:
    """Extract embeddings from fine-tuned ProtBERT model."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 32,
        pooling: str = "mean"
    ):
        """
        Args:
            model_path: Path to fine-tuned PEFT model
            device: Device to run model on
            batch_size: Batch size for embedding extraction
            pooling: Pooling strategy ('mean', 'cls', 'max')
        """
        self.device = device
        self.batch_size = batch_size
        self.pooling = pooling
        
        print(f"Loading fine-tuned model from: {model_path}")
        
        # Load adapter config to get base model name
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path", "Rostlab/prot_bert")
        else:
            base_model_name = "Rostlab/prot_bert"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Load base model
        base_model = AutoModelForMaskedLM.from_pretrained(base_model_name)
        
        # Load PEFT adapters and merge
        peft_model = PeftModel.from_pretrained(base_model, model_path)
        print("Merging PEFT adapters with base model...")
        self.model = peft_model.merge_and_unload()
        
        # Get the encoder (BERT model without MLM head)
        self.encoder = self.model.bert  # For BERT-based models
        
        self.model.to(device)
        self.model.eval()
        
        print(f"Model loaded successfully. Pooling strategy: {pooling}")
    
    def extract_embeddings(self, sequences: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Extract embeddings for a list of TCR sequences.
        
        Args:
            sequences: List of TCR amino acid sequences
            show_progress: Show progress bar
        
        Returns:
            Embeddings array of shape (n_sequences, embedding_dim)
        """
        embeddings = []
        
        # Create dataloader
        dataloader = DataLoader(
            sequences,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda x: x
        )
        
        with torch.no_grad():
            iterator = tqdm(dataloader, desc="Extracting embeddings") if show_progress else dataloader
            
            for batch_sequences in iterator:
                # Tokenize
                encoded = self.tokenizer(
                    batch_sequences,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512
                )
                
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                
                # Get embeddings from encoder
                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                
                # Last hidden state: (batch_size, seq_len, hidden_dim)
                last_hidden_state = outputs.last_hidden_state
                
                # Apply pooling
                if self.pooling == "mean":
                    # Mean pooling (excluding padding tokens)
                    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
                    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                    batch_embeddings = sum_embeddings / sum_mask
                
                elif self.pooling == "cls":
                    # Use [CLS] token embedding (first token)
                    batch_embeddings = last_hidden_state[:, 0, :]
                
                elif self.pooling == "max":
                    # Max pooling
                    batch_embeddings = torch.max(last_hidden_state, dim=1)[0]
                
                else:
                    raise ValueError(f"Unknown pooling strategy: {self.pooling}")
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        embeddings = np.vstack(embeddings)
        print(f"Extracted embeddings: shape={embeddings.shape}")
        
        return embeddings


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

class RepertoireDataset(Dataset):
    """Dataset for TCR repertoires (bags of TCR sequences)."""
    
    def __init__(
        self,
        repertoire_data: List[Dict[str, Any]],
        embeddings_dict: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Args:
            repertoire_data: List of dicts with keys:
                - 'repertoire_id': Unique identifier for repertoire
                - 'sequences': List of TCR sequences
                - 'label': Repertoire-level label (int or str)
            embeddings_dict: Optional pre-computed embeddings {sequence: embedding}
        """
        self.repertoire_data = repertoire_data
        self.embeddings_dict = embeddings_dict
    
    def __len__(self) -> int:
        return len(self.repertoire_data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.repertoire_data[idx]
        
        result = {
            'repertoire_id': item['repertoire_id'],
            'sequences': item['sequences'],
            'label': item['label']
        }
        
        # Add embeddings if available
        if self.embeddings_dict is not None:
            embeddings = np.array([
                self.embeddings_dict[seq] 
                for seq in item['sequences']
                if seq in self.embeddings_dict
            ])
            result['embeddings'] = embeddings
        
        return result


def load_repertoire_data(
    data_path: str,
    file_format: str = "csv"
) -> List[Dict[str, Any]]:
    """
    Load repertoire data from file.
    
    Expected format:
    - CSV: columns = [repertoire_id, sequence, label]
    - JSON: list of dicts with keys [repertoire_id, sequences, label]
    - TSV: same as CSV with tab separator
    
    Args:
        data_path: Path to data file
        file_format: File format ('csv', 'tsv', 'json')
    
    Returns:
        List of repertoire dicts
    """
    print(f"Loading repertoire data from: {data_path}")
    
    if file_format == "json":
        with open(data_path, 'r') as f:
            repertoire_data = json.load(f)
    
    elif file_format in ["csv", "tsv"]:
        sep = '\t' if file_format == "tsv" else ','
        df = pd.read_csv(data_path, sep=sep)
        
        # Group by repertoire_id
        repertoire_data = []
        for rep_id, group in df.groupby('repertoire_id'):
            repertoire_data.append({
                'repertoire_id': rep_id,
                'sequences': group['sequence'].tolist(),
                'label': group['label'].iloc[0]  # Assume same label for all sequences in repertoire
            })
    
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
    print(f"Loaded {len(repertoire_data)} repertoires")
    
    # Print statistics
    seq_counts = [len(r['sequences']) for r in repertoire_data]
    print(f"  Sequences per repertoire: min={min(seq_counts)}, max={max(seq_counts)}, mean={np.mean(seq_counts):.1f}")
    
    label_counts = pd.Series([r['label'] for r in repertoire_data]).value_counts()
    print(f"  Label distribution:")
    for label, count in label_counts.items():
        print(f"    {label}: {count}")
    
    return repertoire_data


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MULTIPLE INSTANCE LEARNING MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class AttentionMIL(nn.Module):
    """
    Attention-based Multiple Instance Learning.
    
    Uses attention mechanism to identify important instances (TCRs) in each bag (repertoire).
    Reference: "Attention-based Deep Multiple Instance Learning" (Ilse et al., 2018)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(
        self,
        instances: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            instances: Tensor of shape (n_instances, input_dim)
            return_attention: Whether to return attention weights
        
        Returns:
            logits: Tensor of shape (num_classes,)
            attention_weights: Optional tensor of shape (n_instances,)
        """
        # Compute attention weights
        attention_logits = self.attention(instances)  # (n_instances, 1)
        attention_weights = torch.softmax(attention_logits, dim=0)  # (n_instances, 1)
        
        # Weighted sum of instances
        bag_representation = torch.sum(instances * attention_weights, dim=0)  # (input_dim,)
        
        # Classify
        logits = self.classifier(bag_representation)  # (num_classes,)
        
        if return_attention:
            return logits, attention_weights.squeeze()
        else:
            return logits, None


class MaxPoolingMIL(nn.Module):
    """Simple max-pooling MIL baseline."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(
        self,
        instances: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Max pooling across instances
        bag_representation = torch.max(instances, dim=0)[0]
        logits = self.classifier(bag_representation)
        return logits, None


class MeanPoolingMIL(nn.Module):
    """Simple mean-pooling MIL baseline."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(
        self,
        instances: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Mean pooling across instances
        bag_representation = torch.mean(instances, dim=0)
        logits = self.classifier(bag_representation)
        return logits, None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING AND EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def train_mil_model(
    model: nn.Module,
    train_dataset: RepertoireDataset,
    val_dataset: RepertoireDataset,
    num_epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    device: str = "cuda",
    patience: int = 10,
    output_dir: str = "./mil_checkpoints"
) -> Dict[str, List[float]]:
    """Train MIL model."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
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
        
        for item in tqdm(train_dataset, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            embeddings = torch.tensor(item['embeddings'], dtype=torch.float32).to(device)
            label = torch.tensor(item['label'], dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            logits, _ = model(embeddings)
            loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = torch.argmax(logits)
            train_correct += (pred == label).item()
            train_total += 1
        
        train_loss /= len(train_dataset)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for item in tqdm(val_dataset, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                embeddings = torch.tensor(item['embeddings'], dtype=torch.float32).to(device)
                label = torch.tensor(item['label'], dtype=torch.long).to(device)
                
                logits, _ = model(embeddings)
                loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
                
                val_loss += loss.item()
                pred = torch.argmax(logits)
                val_correct += (pred == label).item()
                val_total += 1
        
        val_loss /= len(val_dataset)
        val_acc = val_correct / val_total
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            print(f"  ✓ New best model saved (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt")))
    
    return history


def evaluate_mil_model(
    model: nn.Module,
    test_dataset: RepertoireDataset,
    device: str = "cuda",
    top_k: int = 100
) -> Dict[str, Any]:
    """
    Evaluate MIL model and identify top-k important instances.
    
    Returns:
        results: Dict with metrics and top instances per repertoire
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    all_probs = []
    repertoire_top_instances = {}
    
    with torch.no_grad():
        for item in tqdm(test_dataset, desc="Evaluating"):
            embeddings = torch.tensor(item['embeddings'], dtype=torch.float32).to(device)
            label = item['label']
            sequences = item['sequences']
            rep_id = item['repertoire_id']
            
            logits, attention_weights = model(embeddings, return_attention=True)
            probs = torch.softmax(logits, dim=0)
            pred = torch.argmax(logits).item()
            
            all_preds.append(pred)
            all_labels.append(label)
            all_probs.append(probs.cpu().numpy())
            
            # Get top-k instances based on attention weights
            if attention_weights is not None:
                attention_scores = attention_weights.cpu().numpy()
                top_k_indices = np.argsort(attention_scores)[-top_k:][::-1]
                
                top_instances = [
                    {
                        'sequence': sequences[i],
                        'attention_score': float(attention_scores[i]),
                        'rank': rank + 1
                    }
                    for rank, i in enumerate(top_k_indices)
                ]
            else:
                # For non-attention models, use embedding norms as proxy
                norms = np.linalg.norm(embeddings.cpu().numpy(), axis=1)
                top_k_indices = np.argsort(norms)[-top_k:][::-1]
                
                top_instances = [
                    {
                        'sequence': sequences[i],
                        'importance_score': float(norms[i]),
                        'rank': rank + 1
                    }
                    for rank, i in enumerate(top_k_indices)
                ]
            
            repertoire_top_instances[rep_id] = {
                'predicted_label': pred,
                'true_label': label,
                'probabilities': probs.cpu().numpy().tolist(),
                'top_instances': top_instances
            }
    
    # Calculate metrics
    all_probs = np.array(all_probs)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # ROC AUC (for binary classification)
    if all_probs.shape[1] == 2:
        try:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        except:
            auc = None
    else:
        auc = None
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist(),
        'repertoire_predictions': repertoire_top_instances
    }
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Multiple Instance Learning for TCR Repertoire Classification"
    )
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to repertoire data file")
    parser.add_argument("--file_format", type=str, default="csv",
                        choices=["csv", "tsv", "json"],
                        help="Input file format")
    
    # Model arguments
    parser.add_argument("--peft_model_path", type=str, required=True,
                        help="Path to fine-tuned PEFT model")
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "cls", "max"],
                        help="Pooling strategy for embeddings")
    parser.add_argument("--mil_model", type=str, default="attention",
                        choices=["attention", "max_pool", "mean_pool"],
                        help="MIL model architecture")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for embedding extraction")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension for MIL model")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./mil_output",
                        help="Output directory for results")
    parser.add_argument("--embeddings_cache", type=str, default=None,
                        help="Path to save/load cached embeddings")
    parser.add_argument("--top_k", type=int, default=100,
                        help="Number of top instances to identify per repertoire")
    
    # Other arguments
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test set fraction")
    parser.add_argument("--val_size", type=float, default=0.1,
                        help="Validation set fraction (from training set)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 1. Load repertoire data
    # ─────────────────────────────────────────────────────────────────────────────
    
    repertoire_data = load_repertoire_data(args.data_path, args.file_format)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 2. Extract or load embeddings
    # ─────────────────────────────────────────────────────────────────────────────
    
    if args.embeddings_cache and os.path.exists(args.embeddings_cache):
        print(f"Loading cached embeddings from: {args.embeddings_cache}")
        with open(args.embeddings_cache, 'rb') as f:
            embeddings_dict = pickle.load(f)
    else:
        print("Extracting embeddings from fine-tuned model...")
        extractor = TCREmbeddingExtractor(
            model_path=args.peft_model_path,
            device=args.device,
            batch_size=args.batch_size,
            pooling=args.pooling
        )
        
        # Get all unique sequences
        all_sequences = set()
        for rep in repertoire_data:
            all_sequences.update(rep['sequences'])
        all_sequences = list(all_sequences)
        
        print(f"Total unique sequences: {len(all_sequences)}")
        
        # Extract embeddings
        embeddings = extractor.extract_embeddings(all_sequences)
        
        # Create dict
        embeddings_dict = {seq: emb for seq, emb in zip(all_sequences, embeddings)}
        
        # Cache embeddings
        if args.embeddings_cache:
            print(f"Caching embeddings to: {args.embeddings_cache}")
            os.makedirs(os.path.dirname(args.embeddings_cache) or '.', exist_ok=True)
            with open(args.embeddings_cache, 'wb') as f:
                pickle.dump(embeddings_dict, f)
    
    embedding_dim = next(iter(embeddings_dict.values())).shape[0]
    print(f"Embedding dimension: {embedding_dim}")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 3. Split data
    # ─────────────────────────────────────────────────────────────────────────────
    
    labels = [r['label'] for r in repertoire_data]
    
    # Train/test split
    train_val_data, test_data = train_test_split(
        repertoire_data,
        test_size=args.test_size,
        stratify=labels,
        random_state=args.seed
    )
    
    # Train/val split
    train_labels = [r['label'] for r in train_val_data]
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=args.val_size,
        stratify=train_labels,
        random_state=args.seed
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_data)} repertoires")
    print(f"  Val: {len(val_data)} repertoires")
    print(f"  Test: {len(test_data)} repertoires")
    
    # Create datasets
    train_dataset = RepertoireDataset(train_data, embeddings_dict)
    val_dataset = RepertoireDataset(val_data, embeddings_dict)
    test_dataset = RepertoireDataset(test_data, embeddings_dict)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 4. Initialize and train MIL model
    # ─────────────────────────────────────────────────────────────────────────────
    
    num_classes = len(set(labels))
    print(f"\nNumber of classes: {num_classes}")
    
    if args.mil_model == "attention":
        model = AttentionMIL(
            input_dim=embedding_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            dropout=args.dropout
        )
    elif args.mil_model == "max_pool":
        model = MaxPoolingMIL(
            input_dim=embedding_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            dropout=args.dropout
        )
    else:  # mean_pool
        model = MeanPoolingMIL(
            input_dim=embedding_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            dropout=args.dropout
        )
    
    print(f"\nTraining {args.mil_model} MIL model...")
    history = train_mil_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        patience=args.patience,
        output_dir=args.output_dir
    )
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'), dpi=150)
    print(f"Training curves saved to: {os.path.join(args.output_dir, 'training_curves.png')}")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 5. Evaluate on test set
    # ─────────────────────────────────────────────────────────────────────────────
    
    print("\nEvaluating on test set...")
    results = evaluate_mil_model(
        model=model,
        test_dataset=test_dataset,
        device=args.device,
        top_k=args.top_k
    )
    
    print("\nTest Results:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1 Score: {results['f1']:.4f}")
    if results['auc'] is not None:
        print(f"  ROC AUC: {results['auc']:.4f}")
    
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'), dpi=150)
    print(f"Confusion matrix saved to: {os.path.join(args.output_dir, 'confusion_matrix.png')}")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 6. Save results
    # ─────────────────────────────────────────────────────────────────────────────
    
    # Save detailed results (excluding repertoire predictions to keep file size manageable)
    summary_results = {k: v for k, v in results.items() if k != 'repertoire_predictions'}
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump(summary_results, f, indent=2)
    
    # Save repertoire predictions and top instances
    with open(os.path.join(args.output_dir, 'repertoire_predictions.json'), 'w') as f:
        json.dump(results['repertoire_predictions'], f, indent=2)
    
    print(f"\n✅ Results saved to: {args.output_dir}")
    print(f"   - test_results.json: Overall metrics")
    print(f"   - repertoire_predictions.json: Per-repertoire predictions and top-{args.top_k} instances")
    print(f"   - best_model.pt: Trained MIL model")


if __name__ == "__main__":
    main()
