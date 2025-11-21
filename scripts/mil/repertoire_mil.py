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
import h5py

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
            if embedding_dim is None:
                raise ValueError("embedding_dim required for write mode")
            # Create new HDF5 file
            os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
            self.h5file = h5py.File(cache_path, 'w')
            # Create resizable dataset for embeddings
            self.embeddings_dataset = self.h5file.create_dataset(
                'embeddings',
                shape=(0, embedding_dim),
                maxshape=(None, embedding_dim),
                dtype='float32',
                chunks=(1000, embedding_dim)
            )
            # Create dataset for sequence strings (variable length)
            dt = h5py.string_dtype(encoding='utf-8')
            self.sequences_dataset = self.h5file.create_dataset(
                'sequences',
                shape=(0,),
                maxshape=(None,),
                dtype=dt,
                chunks=(1000,)
            )
            self.current_idx = 0
        
        elif mode == 'r':
            if not os.path.exists(cache_path):
                raise FileNotFoundError(f"Cache file not found: {cache_path}")
            self.h5file = h5py.File(cache_path, 'r')
            self.embeddings_dataset = self.h5file['embeddings']
            self.sequences_dataset = self.h5file['sequences']
            self.embedding_dim = self.embeddings_dataset.shape[1]
            
            # Check for cached index
            index_cache_path = cache_path + '.idx.pkl'
            if os.path.exists(index_cache_path):
                print(f"Loading cached sequence index from: {index_cache_path}")
                with open(index_cache_path, 'rb') as f:
                    self.seq_to_idx = pickle.load(f)
                print(f"  ✓ Loaded index for {len(self.seq_to_idx)} sequences")
            else:
                # Build sequence index - ensure proper string decoding
                print(f"Building sequence index from {len(self.sequences_dataset)} entries...")
                print("  (This will be cached for future use)")
                for idx in tqdm(range(len(self.sequences_dataset)), desc="Indexing"):
                    seq = self.sequences_dataset[idx]
                    # Decode bytes to string if needed
                    if isinstance(seq, bytes):
                        seq = seq.decode('utf-8')
                    self.seq_to_idx[seq] = idx
                
                # Cache the index
                print(f"Caching sequence index to: {index_cache_path}")
                with open(index_cache_path, 'wb') as f:
                    pickle.dump(self.seq_to_idx, f)
                print("  ✓ Index cached for future use")
    
    def add_batch(self, sequences: List[str], embeddings: np.ndarray):
        """Add a batch of sequences and embeddings (write mode only)."""
        if self.mode != 'w':
            raise ValueError("Can only add to storage in write mode")
        
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
            raise ValueError("Can only read from storage in read mode")
        
        idx = self.seq_to_idx.get(sequence)
        if idx is None:
            return None
        return self.embeddings_dataset[idx]
    
    def get_batch(self, sequences: List[str]) -> np.ndarray:
        """Get embeddings for multiple sequences (read mode)."""
        if self.mode != 'r':
            raise ValueError("Can only read from storage in read mode")
        
        # Get indices for sequences that exist in storage
        indices = []
        for seq in sequences:
            if seq in self.seq_to_idx:
                indices.append(self.seq_to_idx[seq])
        
        if not indices:
            # Return empty array with correct shape (0, embedding_dim)
            return np.zeros((0, self.embedding_dim), dtype='float32')
        
        # Convert to numpy array for efficient indexing
        indices_array = np.array(indices, dtype=np.int64)
        
        # For small batches, use individual reads (HDF5 fancy indexing is problematic)
        # For larger batches, we could optimize further with chunked reads
        if len(indices_array) < 100:
            # Individual reads for small batches
            embeddings_list = []
            for idx in indices_array:
                embeddings_list.append(self.embeddings_dataset[int(idx)])
            embeddings = np.array(embeddings_list)
        else:
            # For larger batches, try to use slicing where possible
            # Sort indices to enable potential slice optimization
            sort_order = np.argsort(indices_array)
            sorted_indices = indices_array[sort_order]
            
            # Read using sorted indices
            embeddings_list = []
            for idx in sorted_indices:
                embeddings_list.append(self.embeddings_dataset[int(idx)])
            
            embeddings_sorted = np.array(embeddings_list)
            
            # Restore original order
            unsort_order = np.argsort(sort_order)
            embeddings = embeddings_sorted[unsort_order]
        
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
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


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
                
                batch_embeddings_np = batch_embeddings.cpu().numpy()
                
                # Stream to storage or accumulate
                if storage is not None:
                    storage.add_batch(batch_sequences, batch_embeddings_np)
                else:
                    all_embeddings.append(batch_embeddings_np)
        
        if return_embeddings:
            embeddings = np.vstack(all_embeddings)
            print(f"Extracted embeddings: shape={embeddings.shape}")
            return embeddings
        else:
            print(f"Streamed {len(storage)} embeddings to storage")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

class RepertoireDataset(Dataset):
    """Dataset for TCR repertoires (bags of TCR sequences) with lazy loading."""
    
    def __init__(
        self,
        repertoire_data: List[Dict[str, Any]],
        embeddings_dict: Optional[Dict[str, np.ndarray]] = None,
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
                - 'label': Repertoire-level label (int or str)
                - 'frequency': Optional list of sequence frequencies
                - 'v_gene': Optional list of V gene names
                - 'j_gene': Optional list of J gene names
                - 'cdr3_length': Optional list of CDR3 lengths
            embeddings_dict: Optional pre-computed embeddings dict (LEGACY - uses more RAM)
            embeddings_storage: Optional HDF5 storage for lazy loading (RECOMMENDED)
            use_additional_features: Whether to use V/J genes, frequency, etc.
            v_gene_encoder: Dict mapping V gene names to indices
            j_gene_encoder: Dict mapping J gene names to indices
        """
        self.repertoire_data = repertoire_data
        self.embeddings_dict = embeddings_dict
        self.embeddings_storage = embeddings_storage
        self.use_additional_features = use_additional_features
        
        # Build gene encoders if not provided
        if use_additional_features:
            if v_gene_encoder is None:
                self.v_gene_encoder = self._build_gene_encoder('v_gene')
            else:
                self.v_gene_encoder = v_gene_encoder
                
            if j_gene_encoder is None:
                self.j_gene_encoder = self._build_gene_encoder('j_gene')
            else:
                self.j_gene_encoder = j_gene_encoder
        else:
            self.v_gene_encoder = {}
            self.j_gene_encoder = {}
    
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
            freq = np.array(item['frequency'], dtype=np.float32)
            # Log-scale frequency (add 1 to avoid log(0))
            freq_feature = np.log1p(freq).reshape(-1, 1)
            features_list.append(freq_feature)
        
        # V gene (one-hot or index)
        if 'v_gene' in item:
            v_indices = np.array([
                self.v_gene_encoder.get(gene, self.v_gene_encoder['UNK'])
                for gene in item['v_gene']
            ], dtype=np.float32).reshape(-1, 1)
            features_list.append(v_indices)
        
        # J gene (one-hot or index)
        if 'j_gene' in item:
            j_indices = np.array([
                self.j_gene_encoder.get(gene, self.j_gene_encoder['UNK'])
                for gene in item['j_gene']
            ], dtype=np.float32).reshape(-1, 1)
            features_list.append(j_indices)
        
        # CDR3 length
        if 'cdr3_length' in item:
            lengths = np.array(item['cdr3_length'], dtype=np.float32).reshape(-1, 1)
            # Normalize length
            lengths = (lengths - 12) / 5.0  # Typical TCR CDR3 ~12-22 AA
            features_list.append(lengths)
        
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
        
        # Add embeddings - prioritize HDF5 storage (lazy loading)
        if self.embeddings_storage is not None:
            # Pre-filter sequences that exist in storage (faster lookup)
            valid_sequences = []
            for i, seq in enumerate(item['sequences']):
                if seq in self.embeddings_storage:
                    valid_sequences.append(seq)
                    valid_indices.append(i)
            
            # Batch load embeddings for all valid sequences at once
            if valid_sequences:
                embeddings = self.embeddings_storage.get_batch(valid_sequences)
            else:
                embeddings = np.zeros((0, self.embeddings_storage.embedding_dim), dtype='float32')
            
            result['embeddings'] = embeddings
            
        elif self.embeddings_dict is not None:
            # Fallback to dict (uses more RAM)
            valid_sequences = []
            for i, seq in enumerate(item['sequences']):
                if seq in self.embeddings_dict:
                    valid_sequences.append(seq)
                    valid_indices.append(i)
            
            if valid_sequences:
                # Vectorized lookup is faster than list comprehension
                embeddings = np.array([self.embeddings_dict[seq] for seq in valid_sequences])
            else:
                embeddings = np.zeros((0, len(next(iter(self.embeddings_dict.values())))), dtype='float32')
            
            result['embeddings'] = embeddings
        else:
            valid_indices = list(range(len(item['sequences'])))
        
        # Add additional features - ONLY for sequences with embeddings
        additional_features = self._encode_additional_features(item)
        if additional_features is not None and len(valid_indices) > 0:
            # Align additional features with valid embeddings
            result['additional_features'] = additional_features[valid_indices]
        elif additional_features is not None:
            # No valid embeddings, return empty features
            result['additional_features'] = np.zeros((0, additional_features.shape[1]), dtype='float32')
        
        return result


def collate_repertoires(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Custom collate function for DataLoader.
    Returns list of dicts as-is (since repertoires have variable sizes).
    """
    return batch


def load_repertoire_data(
    data_path: str,
    file_format: str = "auto"
) -> List[Dict[str, Any]]:
    """
    Load repertoire data from file.
    
    Expected format:
    - CSV: columns = [repertoire_id, sequence, label]
    - JSON: list of dicts with keys [repertoire_id, sequences, label]
    - TSV: same as CSV with tab separator
    
    Args:
        data_path: Path to data file
        file_format: File format ('csv', 'tsv', 'json', 'auto'). 
                     If 'auto', detects from file extension.
    
    Returns:
        List of repertoire dicts
    """
    print(f"Loading repertoire data from: {data_path}")
    
    # Auto-detect format from file extension
    if file_format == "auto":
        if data_path.endswith('.json'):
            file_format = "json"
        elif data_path.endswith('.tsv'):
            file_format = "tsv"
        elif data_path.endswith('.csv'):
            file_format = "csv"
        else:
            raise ValueError(f"Cannot auto-detect file format from: {data_path}. Please specify --file_format")
    
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
        dropout: float = 0.3,
        additional_features_dim: int = 0
    ):
        super().__init__()
        
        # Total input dimension includes embeddings + additional features
        total_input_dim = input_dim + additional_features_dim
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(
        self,
        instances: torch.Tensor,
        additional_features: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            instances: Tensor of shape (n_instances, input_dim) - embeddings
            additional_features: Optional tensor of shape (n_instances, additional_features_dim)
            return_attention: Whether to return attention weights
        
        Returns:
            logits: Tensor of shape (num_classes,)
            attention_weights: Optional tensor of shape (n_instances,)
        """
        # Concatenate embeddings with additional features
        if additional_features is not None:
            instances = torch.cat([instances, additional_features], dim=1)
        
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
        dropout: float = 0.3,
        additional_features_dim: int = 0
    ):
        super().__init__()
        
        total_input_dim = input_dim + additional_features_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(
        self,
        instances: torch.Tensor,
        additional_features: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Concatenate features
        if additional_features is not None:
            instances = torch.cat([instances, additional_features], dim=1)
        
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
        dropout: float = 0.3,
        additional_features_dim: int = 0
    ):
        super().__init__()
        
        total_input_dim = input_dim + additional_features_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(
        self,
        instances: torch.Tensor,
        additional_features: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Concatenate features
        if additional_features is not None:
            instances = torch.cat([instances, additional_features], dim=1)
        
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
    output_dir: str = "./mil_checkpoints",
    use_wandb: bool = False,
    accumulation_steps: int = 8,
    use_amp: bool = True,
    num_workers: int = 4
) -> Dict[str, List[float]]:
    """
    Train MIL model with gradient accumulation, mixed precision, and prefetching.
    
    Args:
        accumulation_steps: Number of repertoires to accumulate gradients over before updating.
        use_amp: Use automatic mixed precision (FP16) for faster training.
        num_workers: Number of workers for data loading (prefetching).
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
        batch_size=1,  # One repertoire at a time (they have variable sizes)
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_repertoires,
        pin_memory=True if device == "cuda" else False,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False  # Keep workers alive between epochs
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
        
        # Track skipped repertoires
        skipped_count = 0
        
        for idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")):
            item = batch[0]  # DataLoader returns list of items
            
            # Skip repertoires with no embeddings early
            if item['embeddings'].shape[0] == 0:
                skipped_count += 1
                continue
            
            embeddings = torch.tensor(item['embeddings'], dtype=torch.float32).to(device, non_blocking=True)
            label = torch.tensor(item['label'], dtype=torch.long).to(device, non_blocking=True)
            
            # Get additional features if available
            additional_features = None
            if 'additional_features' in item:
                if item['additional_features'].shape[0] > 0 and item['additional_features'].shape[0] == embeddings.shape[0]:
                    # Only use if shape matches embeddings
                    additional_features = torch.tensor(
                        item['additional_features'], 
                        dtype=torch.float32
                    ).to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            if use_amp and scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits, _ = model(embeddings, additional_features)
                    loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()
            else:
                logits, _ = model(embeddings, additional_features)
                loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
                loss = loss / accumulation_steps
                loss.backward()
            
            # Update weights every accumulation_steps
            if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(train_loader):
                if use_amp and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
            pred = torch.argmax(logits)
            train_correct += (pred == label).item()
            train_total += 1
        
        # Report skipped repertoires only once per epoch
        if skipped_count > 0:
            print(f"  Skipped {skipped_count} repertoires with no embeddings")
        
        train_loss /= len(train_dataset)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_skipped = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                item = batch[0]
                
                # Skip repertoires with no embeddings early
                if item['embeddings'].shape[0] == 0:
                    val_skipped += 1
                    continue
                
                embeddings = torch.tensor(item['embeddings'], dtype=torch.float32).to(device, non_blocking=True)
                label = torch.tensor(item['label'], dtype=torch.long).to(device, non_blocking=True)
                
                # Get additional features if available
                additional_features = None
                if 'additional_features' in item:
                    if item['additional_features'].shape[0] > 0 and item['additional_features'].shape[0] == embeddings.shape[0]:
                        # Only use if shape matches embeddings
                        additional_features = torch.tensor(
                            item['additional_features'], 
                            dtype=torch.float32
                        ).to(device, non_blocking=True)
                
                # Forward pass with mixed precision
                if use_amp and scaler is not None:
                    with torch.amp.autocast('cuda'):
                        logits, _ = model(embeddings, additional_features)
                        loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
                else:
                    logits, _ = model(embeddings, additional_features)
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
        
        # Log to W&B
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'train/accuracy': train_acc,
                'val/loss': val_loss,
                'val/accuracy': val_acc,
            })
        
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
            
            # Get additional features if available
            additional_features = None
            if 'additional_features' in item:
                if item['additional_features'].shape[0] > 0 and item['additional_features'].shape[0] == embeddings.shape[0]:
                    additional_features = torch.tensor(
                        item['additional_features'], 
                        dtype=torch.float32
                    ).to(device)
            
            logits, attention_weights = model(embeddings, additional_features, return_attention=True)
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
    parser.add_argument("--file_format", type=str, default="auto",
                        choices=["csv", "tsv", "json", "auto"],
                        help="Input file format (default: auto-detect from extension)")
    
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
    parser.add_argument("--accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps (effective batch size for better GPU utilization)")
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="Use automatic mixed precision (FP16) for faster training")
    parser.add_argument("--no_amp", action="store_false", dest="use_amp",
                        help="Disable automatic mixed precision")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of data loading workers for prefetching (0 = no prefetching, 2-4 recommended for low RAM)")
    parser.add_argument("--use_additional_features", action="store_true", default=False,
                        help="Use additional features (frequency, V/J genes) from enriched dataset")
    parser.add_argument("--compile", action="store_true", default=False,
                        help="Use torch.compile for faster training (PyTorch 2.0+ only, may be slower on first epoch)")
    
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
            config={
                'peft_model_path': args.peft_model_path,
                'pooling': args.pooling,
                'mil_model': args.mil_model,
                'num_epochs': args.num_epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'hidden_dim': args.hidden_dim,
                'dropout': args.dropout,
                'patience': args.patience,
                'test_size': args.test_size,
                'val_size': args.val_size,
                'seed': args.seed,
                'top_k': args.top_k,
                'accumulation_steps': args.accumulation_steps,
                'use_amp': args.use_amp,
            }
        )
        print(f"✅ W&B logging enabled: {args.wandb_project}/{args.wandb_run_name or 'auto'}")
    elif args.wandb_project and not WANDB_AVAILABLE:
        print("⚠️ W&B requested but not available. Install with: pip install wandb")
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 1. Load repertoire data
    # ─────────────────────────────────────────────────────────────────────────────
    
    repertoire_data = load_repertoire_data(args.data_path, args.file_format)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 2. Extract or load embeddings (STREAMING MODE for low RAM usage)
    # ─────────────────────────────────────────────────────────────────────────────
    
    embeddings_storage = None
    embeddings_dict = None
    
    # Determine cache format (HDF5 recommended for low RAM)
    if args.embeddings_cache:
        # Use HDF5 if file extension is .h5 or .hdf5, otherwise pickle (legacy)
        use_hdf5 = args.embeddings_cache.endswith(('.h5', '.hdf5'))
        
        if use_hdf5:
            if os.path.exists(args.embeddings_cache):
                print(f"Loading embeddings from HDF5: {args.embeddings_cache}")
                embeddings_storage = StreamingEmbeddingStorage(args.embeddings_cache, mode='r')
                embedding_dim = embeddings_storage.embedding_dim
                print(f"  ✓ Loaded {len(embeddings_storage)} embeddings (lazy loading enabled)")
            else:
                print("Extracting embeddings to HDF5 (streaming mode - low RAM usage)...")
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
                
                # Get embedding dimension first
                sample_emb = extractor.extract_embeddings([all_sequences[0]], show_progress=False)
                embedding_dim = sample_emb.shape[1]
                
                # Create streaming storage and extract embeddings directly to HDF5
                with StreamingEmbeddingStorage(args.embeddings_cache, mode='w', embedding_dim=embedding_dim) as storage:
                    extractor.extract_embeddings(all_sequences, storage=storage)
                
                # Reopen for reading
                embeddings_storage = StreamingEmbeddingStorage(args.embeddings_cache, mode='r')
                print(f"  ✓ Cached {len(embeddings_storage)} embeddings to HDF5")
        
        else:
            # Legacy pickle mode (uses more RAM)
            print("⚠️ Using pickle format (high RAM usage). Consider using .h5 extension for HDF5 format.")
            if os.path.exists(args.embeddings_cache):
                print(f"Loading cached embeddings from: {args.embeddings_cache}")
                with open(args.embeddings_cache, 'rb') as f:
                    embeddings_dict = pickle.load(f)
                embedding_dim = next(iter(embeddings_dict.values())).shape[0]
            else:
                print("Extracting embeddings (non-streaming mode)...")
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
                
                # Extract embeddings (all in RAM)
                embeddings = extractor.extract_embeddings(all_sequences)
                embedding_dim = embeddings.shape[1]
                
                # Create dict
                embeddings_dict = {seq: emb for seq, emb in zip(all_sequences, embeddings)}
                
                # Cache embeddings
                print(f"Caching embeddings to: {args.embeddings_cache}")
                os.makedirs(os.path.dirname(args.embeddings_cache) or '.', exist_ok=True)
                with open(args.embeddings_cache, 'wb') as f:
                    pickle.dump(embeddings_dict, f)
    
    else:
        # No caching - extract on the fly (not recommended for repeated runs)
        print("⚠️ No embedding cache specified. Extracting embeddings without caching...")
        extractor = TCREmbeddingExtractor(
            model_path=args.peft_model_path,
            device=args.device,
            batch_size=args.batch_size,
            pooling=args.pooling
        )
        
        all_sequences = set()
        for rep in repertoire_data:
            all_sequences.update(rep['sequences'])
        all_sequences = list(all_sequences)
        
        print(f"Total unique sequences: {len(all_sequences)}")
        embeddings = extractor.extract_embeddings(all_sequences)
        embedding_dim = embeddings.shape[1]
        embeddings_dict = {seq: emb for seq, emb in zip(all_sequences, embeddings)}
    
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
    
    # Create datasets with lazy loading if using HDF5
    train_dataset = RepertoireDataset(
        train_data, 
        embeddings_dict, 
        embeddings_storage,
        use_additional_features=args.use_additional_features
    )
    val_dataset = RepertoireDataset(
        val_data, 
        embeddings_dict, 
        embeddings_storage,
        use_additional_features=args.use_additional_features,
        v_gene_encoder=train_dataset.v_gene_encoder if args.use_additional_features else None,
        j_gene_encoder=train_dataset.j_gene_encoder if args.use_additional_features else None
    )
    test_dataset = RepertoireDataset(
        test_data, 
        embeddings_dict, 
        embeddings_storage,
        use_additional_features=args.use_additional_features,
        v_gene_encoder=train_dataset.v_gene_encoder if args.use_additional_features else None,
        j_gene_encoder=train_dataset.j_gene_encoder if args.use_additional_features else None
    )
    
    # Determine additional features dimension
    additional_features_dim = 0
    if args.use_additional_features:
        # Get a sample with valid embeddings to determine feature dimension
        sample = None
        for i in range(len(train_dataset)):
            candidate = train_dataset[i]
            if 'additional_features' in candidate and candidate['additional_features'].shape[0] > 0:
                sample = candidate
                break
        
        if sample is not None and 'additional_features' in sample:
            additional_features_dim = sample['additional_features'].shape[1]
            print(f"\nAdditional features dimension: {additional_features_dim}")
            print(f"  V gene vocabulary: {len(train_dataset.v_gene_encoder)} genes")
            print(f"  J gene vocabulary: {len(train_dataset.j_gene_encoder)} genes")
        else:
            print(f"\n⚠️ Warning: No valid samples found with embeddings. Disabling additional features.")
            args.use_additional_features = False
    
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
            dropout=args.dropout,
            additional_features_dim=additional_features_dim
        )
    elif args.mil_model == "max_pool":
        model = MaxPoolingMIL(
            input_dim=embedding_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            dropout=args.dropout,
            additional_features_dim=additional_features_dim
        )
    else:  # mean_pool
        model = MeanPoolingMIL(
            input_dim=embedding_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            dropout=args.dropout,
            additional_features_dim=additional_features_dim
        )
    
    print(f"\nTraining {args.mil_model} MIL model...")
    
    # Optionally compile model for faster training (PyTorch 2.0+)
    if args.compile:
        try:
            print("Compiling model with torch.compile (may take a moment on first epoch)...")
            print("⚠️  Note: torch.compile requires consistent input shapes. Disabling if additional features are used.")
            if additional_features_dim > 0:
                print("  Additional features detected - skipping compilation for compatibility")
                args.compile = False
            else:
                model = torch.compile(model)
        except Exception as e:
            print(f"⚠️ torch.compile failed: {e}. Continuing without compilation.")
            args.compile = False
    
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
    
    # Log training curves to W&B
    if use_wandb:
        wandb.log({"training_curves": wandb.Image(training_curves_path)})
    
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
    
    # Log test results to W&B
    if use_wandb:
        test_metrics = {
            'test/accuracy': results['accuracy'],
            'test/precision': results['precision'],
            'test/recall': results['recall'],
            'test/f1': results['f1'],
        }
        if results['auc'] is not None:
            test_metrics['test/auc'] = results['auc']
        wandb.log(test_metrics)
    
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
    
    # Log confusion matrix to W&B
    if use_wandb:
        wandb.log({"confusion_matrix": wandb.Image(confusion_matrix_path)})
    
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
    
    # Log model artifact to W&B
    if use_wandb:
        model_path = os.path.join(args.output_dir, 'best_model.pt')
        wandb.save(model_path)
        print(f"   - Model logged to W&B")
        wandb.finish()


if __name__ == "__main__":
    main()
