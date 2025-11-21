#!/usr/bin/env python3
"""
repertoire_mil_adaptive.py
────────────────────────────────────────────────────────
Multiple Instance Learning (MIL) for TCR Repertoire Classification
Adapted for the Adaptive Immune Profiling Challenge 2025

Key modifications:
- Direct integration with ESM2 fine-tuned models
- Parquet file support for efficient data loading
- Streaming embedding extraction with HDF5 caching
- Support for additional features (V/J genes, frequency)
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from tqdm.auto import tqdm

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

class StreamingEmbeddingStorage:
    """
    Memory-efficient storage for embeddings using HDF5.
    Allows streaming writes during extraction and lazy reads during training.
    """
    
    def __init__(self, cache_path: str, mode: str = 'r', embedding_dim: Optional[int] = None):
        """
        Args:
            cache_path: Path to HDF5 file
            mode: 'r' for read, 'w' for write
            embedding_dim: Required for write mode
        """
        self.cache_path = cache_path
        self.mode = mode
        self.embedding_dim = embedding_dim
        self.h5file = None
        self.seq_to_idx = {}
        
        if mode == 'w':
            # Create new HDF5 file
            self.h5file = h5py.File(cache_path, 'w')
            
            # Create resizable datasets
            self.embeddings_dataset = self.h5file.create_dataset(
                'embeddings',
                shape=(0, embedding_dim),
                maxshape=(None, embedding_dim),
                dtype=np.float32,
                compression='gzip',
                compression_opts=4
            )
            
            self.sequences_dataset = self.h5file.create_dataset(
                'sequences',
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.string_dtype(encoding='utf-8')
            )
            
            self.current_idx = 0
            
        elif mode == 'r':
            # Open existing HDF5 file
            self.h5file = h5py.File(cache_path, 'r')
            self.embeddings_dataset = self.h5file['embeddings']
            self.sequences_dataset = self.h5file['sequences']
            
            # Build sequence index
            print(f"Building sequence index from {len(self.sequences_dataset)} sequences...")
            for idx, seq in enumerate(tqdm(self.sequences_dataset, desc="Indexing sequences")):
                if isinstance(seq, bytes):
                    seq = seq.decode('utf-8')
                self.seq_to_idx[seq] = idx
            
            self.embedding_dim = self.embeddings_dataset.shape[1]
    
    def add_batch(self, sequences: List[str], embeddings: np.ndarray):
        """Add a batch of sequences and embeddings (write mode only)."""
        if self.mode != 'w':
            raise ValueError("Can only add in write mode")
        
        batch_size = len(sequences)
        # Resize datasets
        new_size = self.current_idx + batch_size
        self.embeddings_dataset.resize((new_size, self.embedding_dim))
        self.sequences_dataset.resize((new_size,))
        
        # Write data
        self.embeddings_dataset[self.current_idx:new_size] = embeddings
        self.sequences_dataset[self.current_idx:new_size] = sequences
        
        # Update index
        for i, seq in enumerate(sequences):
            self.seq_to_idx[seq] = self.current_idx + i
        
        self.current_idx = new_size
    
    def get(self, sequence: str) -> Optional[np.ndarray]:
        """Get embedding for a sequence (read mode)."""
        if self.mode != 'r':
            raise ValueError("Can only get in read mode")
        
        idx = self.seq_to_idx.get(sequence)
        if idx is None:
            return None
        return self.embeddings_dataset[idx]
    
    def get_batch(self, sequences: List[str]) -> np.ndarray:
        """Get embeddings for multiple sequences (read mode)."""
        if self.mode != 'r':
            raise ValueError("Can only get in read mode")
        
        # Get indices for sequences that exist in storage
        indices = []
        for seq in sequences:
            idx = self.seq_to_idx.get(seq)
            if idx is not None:
                indices.append(idx)
        
        if not indices:
            # Return empty array with correct shape
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        
        # Convert to numpy array for efficient indexing
        indices_array = np.array(indices, dtype=np.int64)
        
        # For small batches, use individual reads
        if len(indices_array) < 100:
            embeddings = np.array([self.embeddings_dataset[idx] for idx in indices_array])
        else:
            # For larger batches, use fancy indexing
            embeddings = self.embeddings_dataset[indices_array]
        
        return embeddings
    
    def __contains__(self, sequence: str) -> bool:
        """Check if sequence exists in storage."""
        return sequence in self.seq_to_idx
    
    def __len__(self) -> int:
        """Number of sequences in storage."""
        return len(self.seq_to_idx)
    
    def close(self):
        """Close HDF5 file."""
        if self.h5file is not None:
            self.h5file.close()
            self.h5file = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class TCREmbeddingExtractor:
    """Extract embeddings from fine-tuned ESM2 model."""
    
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
                base_model_name = adapter_config.get('base_model_name_or_path', 'facebook/esm2_t33_650M_UR50D')
        else:
            base_model_name = 'facebook/esm2_t33_650M_UR50D'
        
        print(f"Base model: {base_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Load base model
        base_model = AutoModelForMaskedLM.from_pretrained(base_model_name)
        
        # Load PEFT adapters and merge
        peft_model = PeftModel.from_pretrained(base_model, model_path)
        print("Merging PEFT adapters with base model...")
        self.model = peft_model.merge_and_unload()
        
        # Get the encoder (ESM2 model without MLM head)
        if hasattr(self.model, 'esm'):
            self.encoder = self.model.esm  # ESM2 models
        elif hasattr(self.model, 'bert'):
            self.encoder = self.model.bert  # BERT-based models
        else:
            raise ValueError("Could not find encoder in model")
        
        self.model.to(device)
        self.model.eval()
        
        print(f"Model loaded successfully. Pooling strategy: {pooling}")
    
    def extract_embeddings(
        self, 
        sequences: List[str], 
        show_progress: bool = True,
        storage: Optional[StreamingEmbeddingStorage] = None
    ) -> Optional[np.ndarray]:
        """
        Extract embeddings for a list of TCR sequences.
        
        Args:
            sequences: List of TCR amino acid sequences
            show_progress: Show progress bar
            storage: Optional StreamingEmbeddingStorage to write to (for streaming mode)
        
        Returns:
            Embeddings array of shape (n_sequences, embedding_dim) if storage is None,
            otherwise None (embeddings written to storage)
        """
        return_embeddings = storage is None
        all_embeddings = [] if return_embeddings else None
        
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
                inputs = self.tokenizer(
                    batch_sequences,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Get embeddings
                outputs = self.encoder(**inputs)
                hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
                
                # Pool embeddings
                if self.pooling == "mean":
                    # Mean pooling (excluding padding)
                    attention_mask = inputs['attention_mask'].unsqueeze(-1)
                    masked_embeddings = hidden_states * attention_mask
                    sum_embeddings = masked_embeddings.sum(dim=1)
                    sum_mask = attention_mask.sum(dim=1)
                    embeddings = sum_embeddings / sum_mask
                    
                elif self.pooling == "cls":
                    # Use [CLS] token (first token)
                    embeddings = hidden_states[:, 0, :]
                    
                elif self.pooling == "max":
                    # Max pooling
                    embeddings = hidden_states.max(dim=1)[0]
                
                else:
                    raise ValueError(f"Unknown pooling strategy: {self.pooling}")
                
                # Convert to numpy
                embeddings_np = embeddings.cpu().numpy().astype(np.float32)
                
                # Store or accumulate
                if storage is not None:
                    storage.add_batch(batch_sequences, embeddings_np)
                else:
                    all_embeddings.append(embeddings_np)
        
        if return_embeddings:
            return np.concatenate(all_embeddings, axis=0)
        else:
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

class RepertoireDataset(Dataset):
    """Dataset for TCR repertoires (bags of TCR sequences) with lazy loading."""
    
    def __init__(
        self,
        repertoire_data: List[Dict[str, Any]],
        embeddings_storage: Optional[StreamingEmbeddingStorage] = None,
        use_additional_features: bool = True,
        v_gene_encoder: Optional[Dict[str, int]] = None,
        j_gene_encoder: Optional[Dict[str, int]] = None
    ):
        """
        Args:
            repertoire_data: List of dicts with keys:
                - 'repertoire_id': Unique identifier for repertoire
                - 'sequences': List of TCR sequences
                - 'label': Repertoire-level label (int)
                - 'frequency': Optional list of sequence frequencies
                - 'v_gene': Optional list of V gene names
                - 'j_gene': Optional list of J gene names
                - 'cdr3_length': Optional list of CDR3 lengths
            embeddings_storage: HDF5 storage for lazy loading
            use_additional_features: Whether to use V/J genes, frequency, etc.
            v_gene_encoder: Dict mapping V gene names to indices
            j_gene_encoder: Dict mapping J gene names to indices
        """
        self.repertoire_data = repertoire_data
        self.embeddings_storage = embeddings_storage
        self.use_additional_features = use_additional_features
        
        # Build gene encoders if not provided
        if use_additional_features:
            if v_gene_encoder is None:
                print("Building V gene encoder...")
                self.v_gene_encoder = self._build_gene_encoder('v_gene')
            else:
                self.v_gene_encoder = v_gene_encoder
            
            if j_gene_encoder is None:
                print("Building J gene encoder...")
                self.j_gene_encoder = self._build_gene_encoder('j_gene')
            else:
                self.j_gene_encoder = j_gene_encoder
        else:
            self.v_gene_encoder = None
            self.j_gene_encoder = None
    
    def _build_gene_encoder(self, gene_key: str) -> Dict[str, int]:
        """Build encoder for V or J genes."""
        all_genes = set()
        for item in self.repertoire_data:
            if gene_key in item:
                all_genes.update(item[gene_key])
        
        # Sort for consistency
        sorted_genes = sorted(all_genes)
        encoder = {gene: idx for idx, gene in enumerate(sorted_genes)}
        # Add unknown token
        encoder['UNK'] = len(encoder)
        
        print(f"  Built {gene_key} encoder with {len(encoder)} unique genes")
        return encoder
    
    def __len__(self) -> int:
        return len(self.repertoire_data)
    
    def _encode_additional_features(self, item: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Encode additional features for each sequence.
        
        Returns:
            Array of shape (n_sequences, n_features) or None
        """
        if not self.use_additional_features:
            return None
        
        n_sequences = len(item['sequences'])
        features_list = []
        
        # Frequency (log-scaled)
        if 'frequency' in item:
            freq = np.array(item['frequency'], dtype=np.float32).reshape(-1, 1)
            # Log-scale (add small epsilon to avoid log(0))
            freq_log = np.log(freq + 1e-10)
            features_list.append(freq_log)
        
        # V gene (one-hot or index)
        if 'v_gene' in item and self.v_gene_encoder is not None:
            v_indices = np.array([
                self.v_gene_encoder.get(gene, self.v_gene_encoder['UNK']) 
                for gene in item['v_gene']
            ], dtype=np.float32).reshape(-1, 1)
            features_list.append(v_indices)
        
        # J gene (one-hot or index)
        if 'j_gene' in item and self.j_gene_encoder is not None:
            j_indices = np.array([
                self.j_gene_encoder.get(gene, self.j_gene_encoder['UNK']) 
                for gene in item['j_gene']
            ], dtype=np.float32).reshape(-1, 1)
            features_list.append(j_indices)
        
        # CDR3 length
        if 'cdr3_length' in item:
            cdr3_len = np.array(item['cdr3_length'], dtype=np.float32).reshape(-1, 1)
            # Normalize by dividing by typical max length
            cdr3_len_norm = cdr3_len / 30.0
            features_list.append(cdr3_len_norm)
        
        if len(features_list) == 0:
            return None
        
        # Concatenate all features
        additional_features = np.concatenate(features_list, axis=1)
        return additional_features
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.repertoire_data[idx]
        
        result = {
            'repertoire_id': item['repertoire_id'],
            'sequences': item['sequences'],
            'label': item['label']
        }
        
        # Track which sequences have embeddings
        valid_indices = []
        
        # Add embeddings from HDF5 storage
        if self.embeddings_storage is not None:
            embeddings_list = []
            for i, seq in enumerate(item['sequences']):
                emb = self.embeddings_storage.get(seq)
                if emb is not None:
                    embeddings_list.append(emb)
                    valid_indices.append(i)
            
            if embeddings_list:
                result['embeddings'] = np.stack(embeddings_list, axis=0)
            else:
                result['embeddings'] = np.zeros((0, self.embeddings_storage.embedding_dim), dtype=np.float32)
        else:
            raise ValueError("embeddings_storage is required")
        
        # Add additional features - ONLY for sequences with embeddings
        additional_features = self._encode_additional_features(item)
        if additional_features is not None and len(valid_indices) > 0:
            # Filter to only valid sequences
            result['additional_features'] = additional_features[valid_indices]
        elif additional_features is not None:
            result['additional_features'] = np.zeros((0, additional_features.shape[1]), dtype=np.float32)
        
        return result


def collate_repertoires(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Custom collate function for DataLoader.
    Returns list of dicts as-is (since repertoires have variable sizes).
    """
    return batch


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MULTIPLE INSTANCE LEARNING MODEL
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
        dropout: float = 0.3,
        additional_features_dim: int = 0
    ):
        super().__init__()
        
        self.input_dim = input_dim + additional_features_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(
        self,
        instances: torch.Tensor,
        additional_features: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            instances: Tensor of shape (n_instances, input_dim)
            additional_features: Optional tensor of shape (n_instances, additional_features_dim)
            return_attention: Whether to return attention weights
        
        Returns:
            logits: Tensor of shape (num_classes,)
            attention_weights: Optional tensor of shape (n_instances,) if return_attention=True
        """
        # Concatenate additional features if provided
        if additional_features is not None:
            instances = torch.cat([instances, additional_features], dim=-1)
        
        # Compute attention weights
        attention_logits = self.attention(instances)  # (n_instances, 1)
        attention_weights = torch.softmax(attention_logits, dim=0)  # (n_instances, 1)
        
        # Weighted sum of instances
        bag_representation = torch.sum(instances * attention_weights, dim=0)  # (input_dim,)
        
        # Classification
        logits = self.classifier(bag_representation)  # (num_classes,)
        
        if return_attention:
            return logits, attention_weights.squeeze(-1)
        else:
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
    output_dir: str = "./mil_checkpoints",
    use_wandb: bool = False,
    accumulation_steps: int = 8,
    use_amp: bool = True,
    num_workers: int = 2
) -> Dict[str, List[float]]:
    """
    Train MIL model with gradient accumulation, mixed precision, and prefetching.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda') if use_amp and device == "cuda" else None
    
    # Create DataLoaders with prefetching
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # Process one repertoire at a time
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_repertoires,
        pin_memory=True if device == "cuda" else False,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_repertoires,
        pin_memory=True if device == "cuda" else False,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    
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
        
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            repertoire = batch[0]  # Single repertoire
            
            # Get instances and label
            instances = torch.tensor(repertoire['embeddings'], dtype=torch.float32).to(device)
            label = torch.tensor(repertoire['label'], dtype=torch.long).to(device)
            
            # Get additional features if available
            additional_features = None
            if 'additional_features' in repertoire:
                additional_features = torch.tensor(repertoire['additional_features'], dtype=torch.float32).to(device)
            
            # Skip if no instances
            if len(instances) == 0:
                continue
            
            # Forward pass with mixed precision
            if use_amp and scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits, _ = model(instances, additional_features)
                    loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()
            else:
                logits, _ = model(instances, additional_features)
                loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
                loss = loss / accumulation_steps
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                if use_amp and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            # Metrics
            train_loss += loss.item() * accumulation_steps
            pred = torch.argmax(logits)
            train_correct += (pred == label).item()
            train_total += 1
            
            pbar.set_postfix({'loss': train_loss / train_total, 'acc': train_correct / train_total})
        
        # Final optimizer step if there are leftover gradients
        if train_total % accumulation_steps != 0:
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                repertoire = batch[0]
                
                instances = torch.tensor(repertoire['embeddings'], dtype=torch.float32).to(device)
                label = torch.tensor(repertoire['label'], dtype=torch.long).to(device)
                
                additional_features = None
                if 'additional_features' in repertoire:
                    additional_features = torch.tensor(repertoire['additional_features'], dtype=torch.float32).to(device)
                
                if len(instances) == 0:
                    continue
                
                logits, _ = model(instances, additional_features)
                loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
                
                val_loss += loss.item()
                pred = torch.argmax(logits)
                val_correct += (pred == label).item()
                val_total += 1
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Log metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, "
              f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
        
        # W&B logging
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
            print(f"✗ No improvement ({patience_counter}/{patience})")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
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
        for idx in tqdm(range(len(test_dataset)), desc="Evaluating"):
            repertoire = test_dataset[idx]
            
            instances = torch.tensor(repertoire['embeddings'], dtype=torch.float32).to(device)
            label = repertoire['label']
            
            additional_features = None
            if 'additional_features' in repertoire:
                additional_features = torch.tensor(repertoire['additional_features'], dtype=torch.float32).to(device)
            
            if len(instances) == 0:
                # Skip empty repertoires
                continue
            
            # Get predictions and attention weights
            logits, attention_weights = model(instances, additional_features, return_attention=True)
            probs = torch.softmax(logits, dim=0)
            pred = torch.argmax(logits).item()
            
            all_preds.append(pred)
            all_labels.append(label)
            all_probs.append(probs.cpu().numpy())
            
            # Get top-k instances by attention weight
            if attention_weights is not None:
                top_k_indices = torch.topk(attention_weights, min(top_k, len(attention_weights))).indices.cpu().numpy()
                top_k_sequences = [repertoire['sequences'][i] for i in top_k_indices]
                top_k_weights = attention_weights[top_k_indices].cpu().numpy().tolist()
                
                repertoire_top_instances[repertoire['repertoire_id']] = {
                    'predicted_label': pred,
                    'true_label': label,
                    'prediction_probs': probs.cpu().numpy().tolist(),
                    'top_sequences': top_k_sequences,
                    'top_weights': top_k_weights
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
        description="Multiple Instance Learning for TCR Repertoire Classification (Adaptive Challenge)"
    )
    
    # Data arguments
    parser.add_argument("--data_json", type=str, required=True,
                        help="Path to repertoire data JSON file")
    
    # Model arguments
    parser.add_argument("--peft_model_path", type=str, required=True,
                        help="Path to fine-tuned PEFT model (ESM2)")
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "cls", "max"],
                        help="Pooling strategy for embeddings")
    
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
    parser.add_argument("--embeddings_cache", type=str, required=True,
                        help="Path to save/load cached embeddings (HDF5)")
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
    parser.add_argument("--accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="Use automatic mixed precision")
    parser.add_argument("--no_amp", action="store_false", dest="use_amp",
                        help="Disable automatic mixed precision")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of data loading workers")
    parser.add_argument("--use_additional_features", action="store_true", default=False,
                        help="Use additional features (frequency, V/J genes)")
    
    # W&B arguments
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name (enables W&B logging)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name")
    
    args = parser.parse_args()
    
    # Initialize W&B if requested
    use_wandb = args.wandb_project is not None and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
    elif args.wandb_project and not WANDB_AVAILABLE:
        print("Warning: W&B logging requested but wandb not installed")
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 1. Load repertoire data
    # ─────────────────────────────────────────────────────────────────────────────
    
    print(f"Loading repertoire data from: {args.data_json}")
    with open(args.data_json, 'r') as f:
        repertoire_data = json.load(f)
    
    print(f"Loaded {len(repertoire_data)} repertoires")
    
    # Print statistics
    seq_counts = [len(r['sequences']) for r in repertoire_data]
    print(f"  Sequences per repertoire: min={min(seq_counts)}, max={max(seq_counts)}, mean={np.mean(seq_counts):.1f}")
    
    labels = [r['label'] for r in repertoire_data]
    unique, counts = np.unique(labels, return_counts=True)
    print(f"  Label distribution:")
    for label, count in zip(unique, counts):
        print(f"    Label {label}: {count} ({count/len(labels)*100:.1f}%)")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 2. Extract or load embeddings (STREAMING MODE)
    # ─────────────────────────────────────────────────────────────────────────────
    
    if os.path.exists(args.embeddings_cache):
        print(f"\nLoading embeddings from cache: {args.embeddings_cache}")
        embeddings_storage = StreamingEmbeddingStorage(args.embeddings_cache, mode='r')
        embedding_dim = embeddings_storage.embedding_dim
        print(f"Loaded {len(embeddings_storage)} cached embeddings")
    else:
        print(f"\nExtracting embeddings and caching to: {args.embeddings_cache}")
        
        # Collect all unique sequences
        all_sequences = set()
        for rep in repertoire_data:
            all_sequences.update(rep['sequences'])
        all_sequences = sorted(list(all_sequences))
        print(f"Total unique sequences: {len(all_sequences)}")
        
        # Initialize extractor
        extractor = TCREmbeddingExtractor(
            model_path=args.peft_model_path,
            device=args.device,
            batch_size=args.batch_size,
            pooling=args.pooling
        )
        
        # Get embedding dimension
        sample_emb = extractor.extract_embeddings([all_sequences[0]], show_progress=False)
        embedding_dim = sample_emb.shape[1]
        print(f"Embedding dimension: {embedding_dim}")
        
        # Create storage and extract embeddings
        with StreamingEmbeddingStorage(args.embeddings_cache, mode='w', embedding_dim=embedding_dim) as storage:
            extractor.extract_embeddings(all_sequences, show_progress=True, storage=storage)
        
        print(f"✅ Embeddings cached to: {args.embeddings_cache}")
        
        # Reopen in read mode
        embeddings_storage = StreamingEmbeddingStorage(args.embeddings_cache, mode='r')
    
    print(f"Embedding dimension: {embedding_dim}")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 3. Split data
    # ─────────────────────────────────────────────────────────────────────────────
    
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
    train_dataset = RepertoireDataset(
        train_data, 
        embeddings_storage,
        use_additional_features=args.use_additional_features
    )
    val_dataset = RepertoireDataset(
        val_data, 
        embeddings_storage,
        use_additional_features=args.use_additional_features,
        v_gene_encoder=train_dataset.v_gene_encoder if args.use_additional_features else None,
        j_gene_encoder=train_dataset.j_gene_encoder if args.use_additional_features else None
    )
    test_dataset = RepertoireDataset(
        test_data, 
        embeddings_storage,
        use_additional_features=args.use_additional_features,
        v_gene_encoder=train_dataset.v_gene_encoder if args.use_additional_features else None,
        j_gene_encoder=train_dataset.j_gene_encoder if args.use_additional_features else None
    )
    
    # Determine additional features dimension
    additional_features_dim = 0
    if args.use_additional_features:
        # Get a sample to determine feature dimension
        sample = train_dataset[0]
        if 'additional_features' in sample:
            additional_features_dim = sample['additional_features'].shape[1]
            print(f"Additional features dimension: {additional_features_dim}")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 4. Initialize and train MIL model
    # ─────────────────────────────────────────────────────────────────────────────
    
    num_classes = len(set(labels))
    print(f"\nNumber of classes: {num_classes}")
    
    model = AttentionMIL(
        input_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        dropout=args.dropout,
        additional_features_dim=additional_features_dim
    )
    
    print(f"\nTraining Attention MIL model...")
    
    history = train_mil_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        patience=args.patience,
        output_dir=args.output_dir,
        use_wandb=use_wandb,
        accumulation_steps=args.accumulation_steps,
        use_amp=args.use_amp,
        num_workers=args.num_workers
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
    training_curves_path = os.path.join(args.output_dir, 'training_curves.png')
    plt.savefig(training_curves_path, dpi=150)
    print(f"Training curves saved to: {training_curves_path}")
    
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
        print(f"  AUC: {results['auc']:.4f}")
    
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    confusion_matrix_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path, dpi=150)
    print(f"Confusion matrix saved to: {confusion_matrix_path}")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 6. Save results
    # ─────────────────────────────────────────────────────────────────────────────
    
    # Save detailed results
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
    
    # Close embeddings storage
    embeddings_storage.close()
    
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
