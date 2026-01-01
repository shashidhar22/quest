#!/usr/bin/env python3
"""
Tokenize deduplicated/permuted sequences for foundation model training.
Supports: ProtBERT, BERT, ESM-2, ESM-3, and custom BPE for LSTM/Transformer.
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm
import numpy as np

# Try to import transformers
try:
    from transformers import (
        BertTokenizer,
        EsmTokenizer,
        AutoTokenizer,
    )
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
    from datasets import Dataset, DatasetDict
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed. Install with: pip install transformers tokenizers datasets")

# Try to import ESM3 from evolutionaryscale/esm
try:
    from esm.models.esm3 import ESM3
    from esm.tokenization import get_esm3_model_tokenizers
    HAS_ESM3 = True
except ImportError:
    HAS_ESM3 = False
    # ESM3 not available - will fallback to ESM2


class SequenceTokenizerBase:
    """Base class for all tokenizers."""
    
    def __init__(self, max_length: int = 512):
        self.max_length = max_length
        self.vocab = {}
        self.tokenizer = None
    
    def tokenize_batch(self, sequences: List[str]) -> Dict[str, List[List[int]]]:
        """Tokenize a batch of sequences. Returns dict with input_ids and attention_mask."""
        raise NotImplementedError
    
    def save(self, output_dir: Path):
        """Save tokenizer to disk."""
        raise NotImplementedError
    
    def load(self, output_dir: Path):
        """Load tokenizer from disk."""
        raise NotImplementedError


class ProtBERTTokenizer(SequenceTokenizerBase):
    """
    ProtBERT tokenizer - amino acids separated by spaces.
    Uses BERT's vocabulary with special tokens.
    """

    def __init__(self, max_length: int = 512, use_fast: bool = True):
        super().__init__(max_length)
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers required for ProtBERT")

        # Try to load fast tokenizer first, fallback to slow if unavailable
        try:
            self.tokenizer = BertTokenizer.from_pretrained(
                "Rostlab/prot_bert",
                do_lower_case=False,
                use_fast=use_fast
            )
            is_fast = getattr(self.tokenizer, 'is_fast', False)
            tokenizer_type = "Fast (Rust)" if is_fast else "Slow (Python)"
            print(f"✓ Loaded ProtBERT tokenizer ({tokenizer_type}, vocab size: {len(self.tokenizer)})")
        except Exception as e:
            if use_fast:
                print(f"⚠️  Fast tokenizer unavailable, falling back to slow: {e}")
                self.tokenizer = BertTokenizer.from_pretrained(
                    "Rostlab/prot_bert",
                    do_lower_case=False,
                    use_fast=False
                )
                print(f"✓ Loaded ProtBERT tokenizer (Slow/Python, vocab size: {len(self.tokenizer)})")
            else:
                raise
    
    def _add_spaces(self, seq: str) -> str:
        """Add spaces between amino acids for ProtBERT."""
        return " ".join(list(seq))
    
    def tokenize_batch(self, sequences: List[str]) -> Dict[str, List[List[int]]]:
        """Tokenize sequences with spaces between amino acids."""
        
        
        # Tokenize - BertTokenizer automatically adds [CLS] and [SEP]
        encoded = self.tokenizer(
            sequences,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors=None
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
    
    def save(self, output_dir: Path):
        """Save tokenizer."""
        output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(str(output_dir))
    
    def load(self, output_dir: Path):
        """Load tokenizer."""
        self.tokenizer = BertTokenizer.from_pretrained(str(output_dir))


class StandardBERTTokenizer(SequenceTokenizerBase):
    """
    Standard BERT tokenizer (e.g., google-bert/bert-base-uncased) for protein sequences.
    Uses character-level tokenization with spaces between amino acids.
    """

    def __init__(self, max_length: int = 512, model_name: str = "google-bert/bert-base-uncased", use_fast: bool = True):
        super().__init__(max_length)
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers required for BERT")

        # Try to load fast tokenizer first, fallback to slow if unavailable
        try:
            self.tokenizer = BertTokenizer.from_pretrained(
                model_name,
                do_lower_case=False,  # Keep amino acids case-sensitive
                use_fast=use_fast
            )
            is_fast = getattr(self.tokenizer, 'is_fast', False)
            tokenizer_type = "Fast (Rust)" if is_fast else "Slow (Python)"
            print(f"✓ Loaded Standard BERT tokenizer from {model_name} ({tokenizer_type}, vocab size: {len(self.tokenizer)})")
        except Exception as e:
            if use_fast:
                print(f"⚠️  Fast tokenizer unavailable, falling back to slow: {e}")
                self.tokenizer = BertTokenizer.from_pretrained(
                    model_name,
                    do_lower_case=False,
                    use_fast=False
                )
                print(f"✓ Loaded Standard BERT tokenizer (Slow/Python, vocab size: {len(self.tokenizer)})")
            else:
                raise
    
    def _add_spaces(self, seq: str) -> str:
        """Add spaces between amino acids."""
        return " ".join(list(seq))
    
    def tokenize_batch(self, sequences: List[str]) -> Dict[str, List[List[int]]]:
        """Tokenize sequences with spaces between amino acids."""
        
        # Tokenize - BertTokenizer automatically adds [CLS] and [SEP]
        encoded = self.tokenizer(
            sequences,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors=None
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
    
    def save(self, output_dir: Path):
        """Save tokenizer."""
        output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(str(output_dir))
    
    def load(self, output_dir: Path):
        """Load tokenizer."""
        self.tokenizer = BertTokenizer.from_pretrained(str(output_dir))


class ESM2Tokenizer(SequenceTokenizerBase):
    """
    ESM-2 tokenizer - no spaces, direct amino acid encoding.
    """

    def __init__(self, max_length: int = 512, model_name: str = "facebook/esm2_t6_8M_UR50D", use_fast: bool = True):
        super().__init__(max_length)
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers required for ESM-2")

        # Note: ESM tokenizers do not have fast implementations as of 2025
        # The use_fast parameter is accepted for API consistency but ignored
        if use_fast:
            print(f"ℹ️  Note: ESM2 tokenizer does not have a fast implementation (use_fast ignored)")

        # Use ESM-2 tokenizer
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        print(f"✓ Loaded ESM-2 tokenizer from {model_name} (Python only, vocab size: {len(self.tokenizer)})")
    
    def tokenize_batch(self, sequences: List[str]) -> Dict[str, List[List[int]]]:
        """
        Tokenize sequences using standard ESM tokenization.
        
        Molecules are separated by dash (-) which is a native ESM token (ID 30).
        The ESM tokenizer's encode() method preserves dashes correctly.
        
        Format: "CASSLG-GILGFVFTL" -> [<cls>, C, A, S, S, L, G, -, G, I, L, G, F, V, F, T, L, <eos>]
        """
        # ESM tokenizer correctly handles dash as a native token
        encoded = self.tokenizer(
            sequences,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors=None
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
    
    def save(self, output_dir: Path):
        """Save tokenizer."""
        output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(str(output_dir))
    
    def load(self, output_dir: Path):
        """Load tokenizer."""
        self.tokenizer = EsmTokenizer.from_pretrained(str(output_dir))


class ESM3Tokenizer(SequenceTokenizerBase):
    """
    ESM-3 tokenizer from evolutionaryscale/esm.
    Uses the official ESM3 tokenization from the esm package.
    """
    
    def __init__(self, max_length: int = 512, model_name: str = "esm3_sm_open_v1"):
        super().__init__(max_length)
        
        if not HAS_ESM3:
            print(f"⚠️  ESM-3 package not available. Install with: pip install esm")
            print(f"   Falling back to ESM-2 tokenizer")
            if not HAS_TRANSFORMERS:
                raise ImportError("Neither ESM-3 nor transformers available")
            self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
            self.is_esm3 = False
            print(f"✓ Loaded ESM-2 tokenizer as fallback (vocab size: {len(self.tokenizer)})")
        else:
            # Load ESM3 tokenizer
            try:
                # Get the tokenizers for ESM3
                # ESM3 uses sequence tokenizer and structure tokenizer
                self.tokenizers = get_esm3_model_tokenizers(model_name)
                self.tokenizer = self.tokenizers.sequence  # Use sequence tokenizer
                self.is_esm3 = True
                print(f"✓ Loaded ESM-3 tokenizer from {model_name}")
            except Exception as e:
                print(f"⚠️  Failed to load ESM-3 tokenizer: {e}")
                print(f"   Falling back to ESM-2 tokenizer")
                if not HAS_TRANSFORMERS:
                    raise ImportError("Failed to load ESM-3 and transformers not available")
                self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
                self.is_esm3 = False
                print(f"✓ Loaded ESM-2 tokenizer as fallback (vocab size: {len(self.tokenizer)})")
    
    def tokenize_batch(self, sequences: List[str]) -> Dict[str, List[List[int]]]:
        """Tokenize sequences."""
        if self.is_esm3:
            # ESM3 tokenization
            input_ids = []
            attention_mask = []
            
            for seq in sequences:
                # ESM3 tokenizer encode method
                tokens = self.tokenizer.encode(seq)
                
                # Truncate or pad
                if len(tokens) > self.max_length:
                    tokens = tokens[:self.max_length]
                
                mask = [1] * len(tokens)
                
                # Pad if needed
                if len(tokens) < self.max_length:
                    padding_length = self.max_length - len(tokens)
                    # Use pad token id (usually 1 for ESM3)
                    pad_id = getattr(self.tokenizer, 'pad_token_id', 1)
                    tokens.extend([pad_id] * padding_length)
                    mask.extend([0] * padding_length)
                
                input_ids.append(tokens)
                attention_mask.append(mask)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        else:
            # Use ESM-2 tokenizer - dash is handled natively
            # Molecules are separated by dash (-) which is token ID 30
            encoded = self.tokenizer(
                sequences,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors=None
            )
            
            return {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask']
            }
    
    def save(self, output_dir: Path):
        """Save tokenizer."""
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.is_esm3:
            # Save ESM3 tokenizer info
            import json
            config = {
                'tokenizer_type': 'esm3',
                'max_length': self.max_length
            }
            with open(output_dir / 'tokenizer_config.json', 'w') as f:
                json.dump(config, f)
            print("Note: ESM3 tokenizer is loaded from model, config saved")
        else:
            # Save ESM-2 tokenizer
            self.tokenizer.save_pretrained(str(output_dir))
    
    def load(self, output_dir: Path):
        """Load tokenizer."""
        config_file = output_dir / 'tokenizer_config.json'
        if config_file.exists():
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
            if config.get('tokenizer_type') == 'esm3':
                print("ESM3 tokenizer - reinitializing from model")
                # Reinitialize ESM3
                if HAS_ESM3:
                    self.tokenizers = get_esm3_model_tokenizers("esm3_sm_open_v1")
                    self.tokenizer = self.tokenizers.sequence
                    self.is_esm3 = True
                else:
                    raise ImportError("ESM-3 not available")
        else:
            # Load as ESM-2
            self.tokenizer = EsmTokenizer.from_pretrained(str(output_dir))
            self.is_esm3 = False


class BPETokenizer(SequenceTokenizerBase):
    """
    Custom BPE tokenizer for LSTM/Transformer models.
    Includes special tokens for molecule boundaries: [TRA], [TRB], [PEP], [MHC1], [MHC2]
    """
    
    def __init__(self, max_length: int = 512, vocab_size: int = 1000):
        super().__init__(max_length)
        self.vocab_size = vocab_size
        
        # Special tokens for molecule boundaries
        self.special_tokens = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
            "[TRA]", "[ETRA]",    # TRA start/end
            "[TRB]", "[ETRB]",    # TRB start/end
            "[PEP]", "[EPEP]",    # Peptide start/end
            "[MHC1]", "[EMHC1]",  # MHC class I start/end
            "[MHC2]", "[EMHC2]",  # MHC class II start/end
        ]
        
        if HAS_TRANSFORMERS:
            # Initialize BPE tokenizer
            self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
            self.tokenizer.normalizer = normalizers.Sequence([])
            self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.Split(" ", behavior="isolated")
            ])
        else:
            print("⚠️  tokenizers not available, BPE will have limited functionality")
            self.tokenizer = None
    
    def train_on_sequences(self, sequences: List[str], output_dir: Path):
        """Train BPE tokenizer on sequences."""
        if not HAS_TRANSFORMERS:
            raise ImportError("tokenizers required for BPE training")
        
        print(f"\n📊 Training BPE tokenizer on {len(sequences):,} sequences...")
        
        # Prepare training data - add character-level splits
        training_data = []
        for seq in tqdm(sequences, desc="Preparing training data"):
            # Split into characters with spaces
            training_data.append(" ".join(list(seq)))
        
        # Train BPE
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            show_progress=True,
            min_frequency=2
        )
        
        self.tokenizer.train_from_iterator(training_data, trainer=trainer)
        
        # Save
        output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(output_dir / "bpe_tokenizer.json"))
        
        # Save vocab for reference
        vocab_file = output_dir / "vocab.json"
        with open(vocab_file, 'w') as f:
            json.dump(self.tokenizer.get_vocab(), f, indent=2)
        
        print(f"✓ BPE tokenizer trained (vocab size: {self.tokenizer.get_vocab_size()})")
        print(f"✓ Saved to {output_dir}")
    
    def tokenize_batch(self, sequences: List[str]) -> Dict[str, List[List[int]]]:
        """Tokenize sequences with BPE."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Train or load a tokenizer first.")
        
        # Add character-level splits
        spaced_seqs = [" ".join(list(seq)) for seq in sequences]
        
        # Tokenize
        encodings = [self.tokenizer.encode(seq) for seq in spaced_seqs]
        
        # Pad/truncate to max_length
        input_ids = []
        attention_mask = []
        
        pad_id = self.tokenizer.token_to_id("[PAD]")
        
        for enc in encodings:
            ids = enc.ids[:self.max_length]
            mask = [1] * len(ids)
            
            # Pad if needed
            if len(ids) < self.max_length:
                padding_length = self.max_length - len(ids)
                ids.extend([pad_id] * padding_length)
                mask.extend([0] * padding_length)
            
            input_ids.append(ids)
            attention_mask.append(mask)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def save(self, output_dir: Path):
        """Save tokenizer."""
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer to save")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(output_dir / "bpe_tokenizer.json"))
    
    def load(self, output_dir: Path):
        """Load tokenizer."""
        if not HAS_TRANSFORMERS:
            raise ImportError("tokenizers required for BPE")
        self.tokenizer = Tokenizer.from_file(str(output_dir / "bpe_tokenizer.json"))


def get_permutation_signature(row: Dict) -> str:
    """
    Get the permutation signature for a row.
    For new format: reads directly from 'permutation_key' column
    For old format: infers from which molecules are present
    """
    # New format: permutation_key column exists
    if 'permutation_key' in row:
        perm_key = row['permutation_key']
        if perm_key and str(perm_key) != 'nan' and str(perm_key) != '':
            return str(perm_key)
    
    # Old format: infer from present fields
    present = []
    for field in ['tra', 'trb', 'peptide', 'mhc_one', 'mhc_two']:
        val = row.get(field, '')
        if val and str(val) != 'nan' and str(val) != '':
            present.append(field)
    return "_".join(present) if present else "empty"


def matches_mode(pkey: str, mode: Optional[str]) -> bool:
    """
    Check if permutation key matches the specified mode for filtering.
    
    Filtering logic (same as pre_mask_dataset.py):
    - None/mlm: keep all keys
    - tra: keep only keys that are exactly "tra"
    - trb: keep only keys that are exactly "trb"
    - tra_trb_pairing: keep only keys "tra_trb" or "trb_tra"
    - tcr_mhc: keys with at least one TCR chain AND at least one MHC chain, but NO peptide
    - peptide_mhc: keys with peptide AND at least one MHC chain, but NO TCR chains
    - specificity: keys with at least one TCR chain, peptide, AND at least one MHC chain
    
    Args:
        pkey: Permutation key string (e.g., 'tra', 'trb_peptide_mhc_one')
        mode: Filtering mode (None means keep all)
        
    Returns:
        True if the key matches the mode, False otherwise
    """
    if mode is None or mode == "mlm":
        return True
    
    if not pkey:
        return False
    
    pkey_lower = pkey.lower()
    
    # Check for presence of each molecule type using word matching
    has_tra = 'tra' in pkey_lower.replace('_', ' ').split()
    has_trb = 'trb' in pkey_lower.replace('_', ' ').split()
    has_tcr = has_tra or has_trb
    has_peptide = 'peptide' in pkey_lower
    has_mhc = 'mhc_one' in pkey_lower or 'mhc_two' in pkey_lower or 'mhcone' in pkey_lower or 'mhctwo' in pkey_lower
    
    if mode == "tra":
        return pkey_lower == "tra"
    
    elif mode == "trb":
        return pkey_lower == "trb"
    
    elif mode == "tra_trb_pairing":
        return pkey_lower in ["tra_trb", "trb_tra"]
    
    elif mode == "tcr_mhc":
        return has_tcr and has_mhc and not has_peptide
    
    elif mode == "peptide_mhc":
        return has_peptide and has_mhc and not has_tcr
    
    elif mode == "specificity":
        return has_tcr and has_peptide and has_mhc
    
    else:
        # Unknown mode - keep all
        return True


def detect_existing_output(output_dir: Path) -> Tuple[Dict[str, int], int]:
    """
    Detect existing tokenized output to enable resume functionality.

    Returns:
        Tuple of (split_counts dict, total_rows_processed)
    """
    split_counts = {'train': 0, 'validation': 0, 'test': 0}
    total_rows = 0

    for split_name in ['train', 'validation', 'test']:
        split_dir = output_dir / split_name
        if split_dir.exists():
            # Count existing parquet files
            parquet_files = sorted(split_dir.glob("chunk_*.parquet"))
            for pf in parquet_files:
                try:
                    table = pq.read_table(pf)
                    num_rows = len(table)
                    split_counts[split_name] += num_rows
                    total_rows += num_rows
                except Exception as e:
                    print(f"⚠️  Warning: Could not read {pf}: {e}")

    return split_counts, total_rows


def stratified_split_rows(
    chunk_rows: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> np.ndarray:
    """
    Perform stratified split on chunk rows to preserve class balance.
    
    Groups rows by permutation_key (or computes it from molecule fields),
    then splits each class proportionally into train/val/test.
    
    Args:
        chunk_rows: List of row dictionaries
        train_ratio: Fraction for training set (default 0.8)
        val_ratio: Fraction for validation set (default 0.1)
        test_ratio: Fraction for test set (default 0.1)
        seed: Random seed for reproducibility
        
    Returns:
        numpy array of split assignments ('train', 'validation', 'test')
    """
    np.random.seed(seed)
    
    # Group indices by permutation key
    pkey_to_indices: Dict[str, List[int]] = {}
    for i, row in enumerate(chunk_rows):
        # Get permutation key - either from column or compute it
        if 'permutation_key' in row:
            pkey = row['permutation_key']
        else:
            pkey = get_permutation_signature(row)
        
        if pkey not in pkey_to_indices:
            pkey_to_indices[pkey] = []
        pkey_to_indices[pkey].append(i)
    
    # Initialize split assignments
    split_assignments = np.empty(len(chunk_rows), dtype=object)
    
    # Split each class proportionally
    for pkey, indices in pkey_to_indices.items():
        n = len(indices)
        
        # Shuffle indices for this class
        shuffled_indices = np.array(indices)
        np.random.shuffle(shuffled_indices)
        
        # Calculate split boundaries
        n_train = max(1, int(round(n * train_ratio))) if n >= 3 else n
        n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 0
        n_test = n - n_train - n_val
        
        # Handle edge cases for very small classes
        if n == 1:
            # Single sample goes to train
            n_train, n_val, n_test = 1, 0, 0
        elif n == 2:
            # Two samples: one train, one val
            n_train, n_val, n_test = 1, 1, 0
        elif n_test < 0:
            # Adjust if we overallocated
            n_test = 0
            n_val = n - n_train
        
        # Assign splits
        split_assignments[shuffled_indices[:n_train]] = 'train'
        split_assignments[shuffled_indices[n_train:n_train + n_val]] = 'validation'
        split_assignments[shuffled_indices[n_train + n_val:]] = 'test'
    
    return split_assignments


def random_split_rows(
    chunk_rows: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> np.ndarray:
    """
    Perform random split on chunk rows without stratification.

    Args:
        chunk_rows: List of row dictionaries
        train_ratio: Fraction for training set (default 0.8)
        val_ratio: Fraction for validation set (default 0.1)
        test_ratio: Fraction for test set (default 0.1)
        seed: Random seed for reproducibility

    Returns:
        numpy array of split assignments ('train', 'validation', 'test')
    """
    np.random.seed(seed)
    n = len(chunk_rows)

    # Create shuffled indices
    indices = np.arange(n)
    np.random.shuffle(indices)

    # Calculate split boundaries
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    # Initialize split assignments
    split_assignments = np.empty(n, dtype=object)

    # Assign splits
    split_assignments[indices[:n_train]] = 'train'
    split_assignments[indices[n_train:n_train + n_val]] = 'validation'
    split_assignments[indices[n_train + n_val:]] = 'test'

    return split_assignments


def _count_permutations_in_file(args) -> tuple[Dict[str, int], int]:
    """
    Count permutations in a single parquet file.
    Used for parallel processing.
    
    Args:
        args: Tuple of (file_path, mode) where mode can be None
    """
    file_path, mode = args
    permutation_counts = {}
    total_rows = 0
    
    # Read parquet file
    table = pq.read_table(file_path)
    # Convert PyArrow table directly to dict (faster than pandas intermediate step)
    rows_dict = table.to_pydict()
    # Convert column-oriented dict to row-oriented list of dicts
    rows = [
        {col: rows_dict[col][i] for col in rows_dict.keys()}
        for i in range(len(table))
    ]
    
    for row in rows:
        perm = get_permutation_signature(row)
        # Filter by mode if specified
        if not matches_mode(perm, mode):
            continue
        permutation_counts[perm] = permutation_counts.get(perm, 0) + 1
        total_rows += 1
    
    return permutation_counts, total_rows


def analyze_dataset_distribution(input_dir: Path, chunk_size: int = 10_000_000, num_workers: int = 16, mode: Optional[str] = None):
    """
    Analyze the distribution of permutations in the dataset by streaming through ALL data.
    Uses parallel processing to speed up analysis.
    Returns dict of permutation -> count and total rows.
    
    Args:
        input_dir: Directory containing parquet files
        chunk_size: Not used (kept for API compatibility)
        num_workers: Number of parallel workers
        mode: Optional mode to filter permutation keys
    """
    mode_str = f" (filtering for mode: {mode})" if mode else ""
    print(f"\n📊 Analyzing full dataset distribution{mode_str} (parallel processing with {num_workers} workers)...")
    
    parquet_files = sorted(input_dir.glob("*.parquet"))
    print(f"   Found {len(parquet_files)} parquet files to analyze")
    
    # Process files in parallel - pass mode to each worker
    from multiprocessing import Pool
    
    # Create args tuples for each file
    worker_args = [(pf, mode) for pf in parquet_files]
    
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(_count_permutations_in_file, worker_args),
            total=len(parquet_files),
            desc="Scanning files"
        ))
    
    # Merge results from all workers
    print(f"   Merging results from {len(results)} files...")
    permutation_counts = {}
    total_rows = 0
    
    for file_counts, file_rows in results:
        total_rows += file_rows
        for perm, count in file_counts.items():
            permutation_counts[perm] = permutation_counts.get(perm, 0) + count
    
    print(f"\n✓ Analyzed {total_rows:,} rows across {len(parquet_files)} files")
    print(f"✓ Found {len(permutation_counts)} unique permutations:")
    for perm, count in sorted(permutation_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_rows
        print(f"   {perm:30s}: {count:10,} ({pct:5.2f}%)")
    
    return permutation_counts, total_rows


def stream_parquet_files(input_dir: Path, sample: Optional[int] = None, chunk_size: int = 10_000_000, 
                         sample_mode: Optional[str] = None, num_workers: int = 16, oversample: bool = False,
                         mode: Optional[str] = None):
    """
    Stream parquet files in chunks to avoid loading everything into memory.
    
    Args:
        input_dir: Directory containing parquet files
        sample: If set, only yield first N rows total
        chunk_size: Number of rows per chunk (default 10M = ~15GB RAM per chunk)
        sample_mode: Sampling strategy - 'proportional' or 'balanced' or None
        num_workers: Number of parallel workers for distribution analysis (default: 16)
        oversample: If True and sample_mode='balanced', duplicate underrepresented samples to reach target
        mode: Filter mode - only include permutation keys matching this mode (mlm, tra, trb, etc.)
    
    Yields:
        (chunk_rows, total_rows_seen) tuples
    """
    parquet_files = sorted(input_dir.glob("*.parquet"))
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {input_dir}")
    
    print(f"📁 Found {len(parquet_files)} parquet files")
    print(f"📦 Chunk size: {chunk_size:,} rows (~{chunk_size * 150 / 1e9:.1f}GB per chunk)")
    
    # If sampling with a specific mode, analyze distribution first
    permutation_targets = None
    permutation_counts = {}
    
    if sample and sample_mode:
        print(f"\n🎯 Sampling mode: {sample_mode}")
        if mode:
            print(f"🔍 Filtering for mode: {mode}")
        # Use same num_workers for distribution analysis (passed from outer scope)
        dist_counts, total_analyzed = analyze_dataset_distribution(
            input_dir, chunk_size, num_workers=min(num_workers, len(list(input_dir.glob("*.parquet")))), mode=mode
        )
        
        if sample_mode == 'proportional':
            # Maintain original distribution
            print(f"\n📐 Proportional sampling: maintaining original distribution")
            permutation_targets = {
                perm: int(sample * count / total_analyzed)
                for perm, count in dist_counts.items()
            }
        elif sample_mode == 'balanced':
            # Equal samples per permutation, but capped by available data
            print(f"\n⚖️  Balanced sampling: target equal samples per permutation")
            n_permutations = len(dist_counts)
            ideal_per_perm = sample // n_permutations
            print(f"   Ideal target: {ideal_per_perm:,} samples per permutation ({n_permutations} classes)")
            
            # Calculate targets: min(ideal, available) for each class
            # This ensures we sample all available from small classes
            permutation_targets = {}
            total_expected = 0
            capped_classes = []
            
            for perm, available in dist_counts.items():
                if available >= ideal_per_perm:
                    permutation_targets[perm] = ideal_per_perm
                    total_expected += ideal_per_perm
                else:
                    # Cap at available amount
                    permutation_targets[perm] = available
                    total_expected += available
                    capped_classes.append((perm, available, ideal_per_perm))
            
            print(f"   Expected total samples: {total_expected:,}")
            if capped_classes:
                print(f"   ⚠️  {len(capped_classes)} classes have fewer than ideal ({ideal_per_perm:,}):")
                for perm, avail, ideal in sorted(capped_classes, key=lambda x: x[1]):
                    print(f"      {perm:30s}: {avail:,} available (capped)")
            
            if oversample:
                print(f"   🔄 Oversampling enabled: capped classes will be duplicated to reach {ideal_per_perm:,}")
                # With oversampling, set targets back to ideal
                permutation_targets = {perm: ideal_per_perm for perm in dist_counts.keys()}
        
        print(f"\n🎯 Sampling targets:")
        for perm, target in sorted(permutation_targets.items(), key=lambda x: -x[1]):
            available = dist_counts.get(perm, 0)
            status = "" if available >= target else f" (⚠️ only {available:,} available)"
            print(f"   {perm:30s}: {target:10,}{status}")
        
        # Initialize counters
        permutation_counts = {perm: 0 for perm in permutation_targets.keys()}
    
    total_rows = 0
    chunk_buffer = []
    
    # For oversampling: store samples by permutation for later duplication
    oversample_pool = {} if (oversample and sample_mode == 'balanced') else None
    
    for pf in tqdm(parquet_files, desc="Processing parquet files"):
        table = pq.read_table(pf)
        # Convert PyArrow table directly to dict (faster than pandas intermediate step)
        rows_dict = table.to_pydict()
        # Convert column-oriented dict to row-oriented list of dicts
        rows = [
            {col: rows_dict[col][i] for col in rows_dict.keys()}
            for i in range(len(table))
        ]
        
        for row in rows:
            # Check if we should include this row based on mode filter
            perm = get_permutation_signature(row)
            
            # Filter by mode if specified
            if mode and not matches_mode(perm, mode):
                continue
            
            # Check if we should include this row based on sampling mode
            should_include = True
            
            if sample and sample_mode and permutation_targets:
                # perm already computed above for mode filtering
                if perm in permutation_counts:
                    if permutation_counts[perm] >= permutation_targets[perm]:
                        should_include = False
                    else:
                        permutation_counts[perm] += 1
                        # Store sample for potential oversampling
                        if oversample_pool is not None:
                            if perm not in oversample_pool:
                                oversample_pool[perm] = []
                            oversample_pool[perm].append(row)
                else:
                    # Unknown permutation, skip it
                    should_include = False
            
            if should_include:
                chunk_buffer.append(row)
                total_rows += 1
                
                # Yield chunk when buffer is full
                if len(chunk_buffer) >= chunk_size:
                    yield chunk_buffer, total_rows
                    chunk_buffer = []
                
                # Stop if sample limit reached (for non-mode sampling)
                if sample and not sample_mode and total_rows >= sample:
                    if chunk_buffer:
                        yield chunk_buffer, total_rows
                    return
            
            # For mode-based sampling, stop when all targets met
            if sample and sample_mode and permutation_targets:
                if all(permutation_counts[p] >= permutation_targets[p] for p in permutation_targets):
                    if chunk_buffer:
                        yield chunk_buffer, total_rows
                    print(f"\n✓ All sampling targets met!")
                    return
    
    # Yield remaining rows
    if chunk_buffer:
        yield chunk_buffer, total_rows
        chunk_buffer = []
    
    # Oversampling: duplicate underrepresented samples to reach target
    if oversample_pool is not None and permutation_targets:
        print(f"\n🔄 Oversampling underrepresented permutations...")
        oversampled_rows = []
        
        for perm in sorted(permutation_counts.keys()):
            achieved = permutation_counts[perm]
            target = permutation_targets[perm]
            
            if achieved < target and perm in oversample_pool and len(oversample_pool[perm]) > 0:
                needed = target - achieved
                available_samples = oversample_pool[perm]
                
                # Duplicate samples cyclically to reach target
                duplicates_needed = needed
                idx = 0
                while duplicates_needed > 0:
                    oversampled_rows.append(available_samples[idx % len(available_samples)].copy())
                    idx += 1
                    duplicates_needed -= 1
                    permutation_counts[perm] += 1
                
                print(f"   {perm:30s}: duplicated {needed:,} samples ({len(available_samples):,} unique → {target:,} total)")
        
        # Yield oversampled rows in chunks
        if oversampled_rows:
            total_rows += len(oversampled_rows)
            print(f"   Total oversampled: {len(oversampled_rows):,} rows")
            
            # Shuffle oversampled rows to mix duplicates
            np.random.shuffle(oversampled_rows)
            
            for i in range(0, len(oversampled_rows), chunk_size):
                chunk = oversampled_rows[i:i + chunk_size]
                yield chunk, total_rows
    
    # Print final sampling stats
    if sample and sample_mode and permutation_targets:
        print(f"\n📊 Final sampling statistics:")
        under_target = []
        for perm in sorted(permutation_counts.keys()):
            achieved = permutation_counts[perm]
            target = permutation_targets[perm]
            pct = 100 * achieved / target if target > 0 else 0
            status = "✓" if achieved >= target else "⚠️"
            print(f"   {perm:30s}: {achieved:10,} / {target:10,} ({pct:5.1f}%) {status}")
            if achieved < target:
                under_target.append((perm, achieved, target))
        
        if under_target and not oversample_pool:
            print(f"\n⚠️  {len(under_target)} permutations could not reach target!")
            print(f"   Consider using --oversample to duplicate underrepresented samples.")


def concatenate_molecule_sequences(row: Dict, model_type: str = None) -> str:
    """
    Concatenate molecule sequences from row.
    
    For new format (has 'sequence' column):
        - Reads pre-concatenated sequence from 'sequence' column (molecules separated by single space)
        - Example: "CASSLGQAYEQYF GILGFVFTL KLGEEHLHL" (3 molecules)
        - Splits by space to get individual molecules
        - Re-formats based on model_type
    
    For old format (has individual molecule columns):
        - Reads from tra, trb, peptide, mhc_one, mhc_two columns
        - Concatenates based on model_type
    
    Tokenization formats by model type:
        - ProtBERT/BERT: Space between each amino acid, [SEP] between molecules
          Example: "C A S S L G Q A Y E Q Y F [SEP] G I L G F V F T L"
        - ESM2/ESM3: Contiguous amino acids within molecules, dash (-) between molecules
          Example: "CASSLGQAYEQYF-GILGFVFTL" 
          Note: Dash is a native ESM token (ID 30) used for gaps, serves as molecule boundary
        - BPE/LSTM/Transformer: Contiguous amino acids, dash between molecules
          Example: "CASSLGQAYEQYF-GILGFVFTL"
    """
    # New format: has 'sequence' column with molecules separated by single space
    if 'sequence' in row and row.get('sequence'):
        sequence_str = str(row['sequence'])
        if sequence_str and sequence_str != 'nan' and sequence_str != '':
            # Split by space to get individual molecules
            # Each molecule is a contiguous amino acid string (e.g., "CASSLGQAYEQYF")
            molecules = sequence_str.split()
            # Filter out NA and na values
            molecules = [mol for mol in molecules if mol.upper() != 'NA']
            
            if model_type in ['protbert', 'bert']:
                # ProtBERT/BERT: Add spaces between amino acids for each molecule
                # Add [SEP] between molecules
                # BertTokenizer will add [CLS] at start
                spaced_molecules_with_sep = []
                for mol in molecules:
                    # Add spaces between amino acids
                    spaced = " ".join(list(mol)).strip()
                    spaced_molecules_with_sep.append(spaced)
                # Join all molecules with [SEP]
                return " [SEP] ".join(spaced_molecules_with_sep)
            elif model_type in ['esm2', 'esm3']:
                # ESM2/ESM3: Contiguous amino acids within molecules, dash between molecules
                # Dash is a native ESM token (ID 30) used for gaps in alignments
                # Example: "CASSLGQAYEQYF-GILGFVFTL" -> [<cls>, C, A, S, ..., F, -, G, I, L, ..., L, <eos>]
                return "-".join(molecules)
            else:
                # BPE/LSTM/Transformer: Contiguous amino acids, dash between molecules
                return "-".join(molecules)
    
    # Old format: individual molecule columns
    else:
        sequences = []
        for field in ['tra', 'trb', 'peptide', 'mhc_one', 'mhc_two']:
            val = row.get(field, '')
            # Filter out empty, nan, NA, and na values
            if val and str(val) != 'nan' and str(val) != '' and str(val).upper() != 'NA':
                sequences.append(str(val))
    
        # Concatenate based on model type
        if model_type in ['protbert', 'bert']:
            # ProtBERT/BERT: Add spaces between amino acids for each molecule
            # Join molecules with [SEP]
            spaced_molecules = [" ".join(list(seq)) for seq in sequences]
            return " [SEP] ".join(spaced_molecules)
        elif model_type in ['esm2', 'esm3']:
            # ESM2/ESM3: Contiguous amino acids within molecules, dash between molecules
            # Dash is a native ESM token (ID 30) used for gaps in alignments
            return "-".join(sequences)
        else:
            # BPE/LSTM/Transformer: Contiguous amino acids, dash between molecules
            return "-".join(sequences)


<<<<<<< HEAD
def add_full_tcr_and_cdr_positions(chunk_rows: List[Dict]) -> Tuple[List[Dict], Dict]:
=======
def _concatenate_batch_worker(args):
>>>>>>> 2a62664 (Updated to incrementally write toeknized shards and implement stratified data splits)
    """
    Worker function for parallel sequence concatenation.
    Processes a batch of rows and returns concatenated sequences.

<<<<<<< HEAD
    For each row with TRA/TRB data:
    1. Standardize gene names using tidytcells (via TCRStitcher)
    2. Stitch full-length sequences using TCRStitcher
    3. Identify CDR1, CDR2, CDR3 positions using CDRRegionIdentifier
    4. Add columns:
       - Gene names: trav_gene_std, traj_gene_std, trbv_gene_std, trbj_gene_std (standardized)
       - Full sequences: tra_full, trb_full
       - CDR positions: tra_cdr1_pos, tra_cdr2_pos, tra_cdr3_pos, trb_cdr1_pos, trb_cdr2_pos, trb_cdr3_pos
=======
    This function is designed to be called by multiprocessing.Pool.map()
    to parallelize the sequence concatenation across multiple CPU cores.

    Args:
        args: Tuple of (row_batch, model_type)
            row_batch: List of row dictionaries to process
            model_type: Model type for formatting (esm2, protbert, etc.)

    Returns:
        List of concatenated sequences (one per row in the batch)
    """
    row_batch, model_type = args
    return [concatenate_molecule_sequences(row, model_type) for row in row_batch]


def concatenate_sequences_arrow_esm2(chunk_rows: List[Dict], model_type: str) -> List[str]:
    """
    Ultra-fast concatenation using PyArrow vectorized operations for ESM-2.

    This function uses PyArrow's compute functions to perform string operations
    in a vectorized manner, which is significantly faster than Python loops.

    Works best for new format data (has 'sequence' column) where concatenation
    is a simple string replacement operation.
>>>>>>> 2a62664 (Updated to incrementally write toeknized shards and implement stratified data splits)

    Args:
        chunk_rows: List of row dictionaries
        model_type: Model type (esm2, esm3, protbert, etc.)

    Returns:
<<<<<<< HEAD
        Tuple of (updated list of row dictionaries, statistics dictionary)
    """
    stats = {
        'tra_attempted': 0,
        'tra_gene_std': 0,
        'tra_stitched': 0,
        'tra_cdr': 0,
        'trb_attempted': 0,
        'trb_gene_std': 0,
        'trb_stitched': 0,
        'trb_cdr': 0,
        'by_pkey': {}  # Per-permutation-key breakdown
    }

    if not HAS_TCR_TOOLS:
        # Add empty columns if tools not available
        for row in chunk_rows:
            row['trav_gene_std'] = ''
            row['traj_gene_std'] = ''
            row['trbv_gene_std'] = ''
            row['trbj_gene_std'] = ''
            row['tra_full'] = ''
            row['trb_full'] = ''
            row['tra_cdr1_pos'] = None
            row['tra_cdr2_pos'] = None
            row['tra_cdr3_pos'] = None
            row['trb_cdr1_pos'] = None
            row['trb_cdr2_pos'] = None
            row['trb_cdr3_pos'] = None
        return chunk_rows, stats
=======
        List of concatenated sequences
    """
    import pyarrow.compute as pc
>>>>>>> 2a62664 (Updated to incrementally write toeknized shards and implement stratified data splits)

    # Check if we have the new format (sequence column)
    if chunk_rows and 'sequence' in chunk_rows[0]:
        # New format: convert to PyArrow array for vectorized operations
        sequences = [row.get('sequence', '') for row in chunk_rows]
        arrow_array = pa.array(sequences, type=pa.string())

<<<<<<< HEAD
    print(f"   🧬 Standardizing gene names (tidytcells), stitching full TCR sequences, and identifying CDR regions...")

    for row in tqdm(chunk_rows, desc="   Processing TCRs", leave=False):
        # Initialize columns
        row['trav_gene_std'] = ''
        row['traj_gene_std'] = ''
        row['trbv_gene_std'] = ''
        row['trbj_gene_std'] = ''
        row['tra_full'] = ''
        row['trb_full'] = ''
        row['tra_cdr1_pos'] = None
        row['tra_cdr2_pos'] = None
        row['tra_cdr3_pos'] = None
        row['trb_cdr1_pos'] = None
        row['trb_cdr2_pos'] = None
        row['trb_cdr3_pos'] = None

        # Get permutation key for statistics
        pkey = row.get('permutation_key', get_permutation_signature(row))
        if pkey not in stats['by_pkey']:
            stats['by_pkey'][pkey] = {
                'tra_attempted': 0, 'tra_gene_std': 0, 'tra_stitched': 0, 'tra_cdr': 0,
                'trb_attempted': 0, 'trb_gene_std': 0, 'trb_stitched': 0, 'trb_cdr': 0
            }

        # Process TRA
        tra_cdr3 = row.get('tra', '')
        trav_gene = row.get('trav_gene', '')
        traj_gene = row.get('traj_gene', '')

        if tra_cdr3 and trav_gene and traj_gene:
            stats['tra_attempted'] += 1
            stats['by_pkey'][pkey]['tra_attempted'] += 1

            # Standardize gene names using tidytcells (via TCRStitcher.normalize_gene_name)
            trav_gene_std = stitcher.normalize_gene_name(trav_gene, 'TRA')
            traj_gene_std = stitcher.normalize_gene_name(traj_gene, 'TRA')

            # Store standardized gene names
            if trav_gene_std:
                row['trav_gene_std'] = trav_gene_std
            if traj_gene_std:
                row['traj_gene_std'] = traj_gene_std

            if trav_gene_std and traj_gene_std:
                stats['tra_gene_std'] += 1
                stats['by_pkey'][pkey]['tra_gene_std'] += 1

                # Stitch full TRA sequence using standardized gene names
                tra_full = stitcher.stitch_tcr(
                    cdr3=tra_cdr3,
                    v_gene=trav_gene_std,
                    j_gene=traj_gene_std,
                    chain='TRA'
                )

                if tra_full:
                    row['tra_full'] = tra_full
                    stats['tra_stitched'] += 1
                    stats['by_pkey'][pkey]['tra_stitched'] += 1

                    # Identify CDR regions using standardized gene name
                    cdr_regions = cdr_identifier.get_cdr_regions(
                        full_sequence=tra_full,
                        cdr3_sequence=tra_cdr3,
                        v_gene=trav_gene_std,
                        chain='TRA'
                    )

                    if cdr_regions:
                        # Store positions as tuples (start, end)
                        row['tra_cdr1_pos'] = cdr_regions.get('cdr1')
                        row['tra_cdr2_pos'] = cdr_regions.get('cdr2')
                        row['tra_cdr3_pos'] = cdr_regions.get('cdr3')
                        stats['tra_cdr'] += 1
                        stats['by_pkey'][pkey]['tra_cdr'] += 1

        # Process TRB
        trb_cdr3 = row.get('trb', '')
        trbv_gene = row.get('trbv_gene', '')
        trbj_gene = row.get('trbj_gene', '')

        if trb_cdr3 and trbv_gene and trbj_gene:
            stats['trb_attempted'] += 1
            stats['by_pkey'][pkey]['trb_attempted'] += 1

            # Standardize gene names using tidytcells (via TCRStitcher.normalize_gene_name)
            trbv_gene_std = stitcher.normalize_gene_name(trbv_gene, 'TRB')
            trbj_gene_std = stitcher.normalize_gene_name(trbj_gene, 'TRB')

            # Store standardized gene names
            if trbv_gene_std:
                row['trbv_gene_std'] = trbv_gene_std
            if trbj_gene_std:
                row['trbj_gene_std'] = trbj_gene_std

            if trbv_gene_std and trbj_gene_std:
                stats['trb_gene_std'] += 1
                stats['by_pkey'][pkey]['trb_gene_std'] += 1

                # Stitch full TRB sequence using standardized gene names
                trb_full = stitcher.stitch_tcr(
                    cdr3=trb_cdr3,
                    v_gene=trbv_gene_std,
                    j_gene=trbj_gene_std,
                    chain='TRB'
                )

                if trb_full:
                    row['trb_full'] = trb_full
                    stats['trb_stitched'] += 1
                    stats['by_pkey'][pkey]['trb_stitched'] += 1

                    # Identify CDR regions using standardized gene name
                    cdr_regions = cdr_identifier.get_cdr_regions(
                        full_sequence=trb_full,
                        cdr3_sequence=trb_cdr3,
                        v_gene=trbv_gene_std,
                        chain='TRB'
                    )

                    if cdr_regions:
                        row['trb_cdr1_pos'] = cdr_regions.get('cdr1')
                        row['trb_cdr2_pos'] = cdr_regions.get('cdr2')
                        row['trb_cdr3_pos'] = cdr_regions.get('cdr3')
                        stats['trb_cdr'] += 1
                        stats['by_pkey'][pkey]['trb_cdr'] += 1

    print(f"   ✓ Gene names standardized: {stats['tra_gene_std']:,} TRA, {stats['trb_gene_std']:,} TRB")
    print(f"   ✓ Stitched: {stats['tra_stitched']:,} TRA, {stats['trb_stitched']:,} TRB")
    print(f"   ✓ CDR positions: {stats['tra_cdr']:,} TRA, {stats['trb_cdr']:,} TRB")

    return chunk_rows, stats
=======
        if model_type in ['esm2', 'esm3']:
            # ESM-2/ESM-3: "MOL1 MOL2 MOL3" → "MOL1-MOL2-MOL3"
            # Single vectorized operation: replace all spaces with dashes
            result_array = pc.replace_substring(arrow_array, ' ', '-')
            return result_array.to_pylist()

        elif model_type in ['protbert', 'bert']:
            # ProtBERT: Need to add spaces between amino acids and use [SEP]
            # This is more complex and better handled by parallel Python
            return None

        else:
            # BPE/LSTM/Transformer: same as ESM (dash separator)
            result_array = pc.replace_substring(arrow_array, ' ', '-')
            return result_array.to_pylist()

    # Old format or complex case: return None to trigger fallback
    return None
>>>>>>> 2a62664 (Updated to incrementally write toeknized shards and implement stratified data splits)


def create_tokenize_function(tokenizer, model_type: str):
    """
    Create a tokenization function for HuggingFace Dataset.map().
    This function will be called by the Dataset's parallel processing.

    Handles both new format (permutation_key, sequence) and old format (tra, trb, peptide, mhc_one, mhc_two).
    """
    def tokenize_function(examples):
        """Tokenize a batch of examples from HuggingFace Dataset."""
        # Concatenate sequences for each example
        sequences = []

        # Check if new format (has 'permutation_key' and 'sequence' columns)
        if 'permutation_key' in examples and 'sequence' in examples:
            # New format: use sequence column directly
            for i in range(len(examples['permutation_key'])):
                row = {
                    'permutation_key': examples['permutation_key'][i],
                    'sequence': examples['sequence'][i],
                }
                sequences.append(concatenate_molecule_sequences(row, model_type))
        else:
            # Old format: use individual molecule columns
            for i in range(len(examples['tra'])):
                row = {
                    'tra': examples['tra'][i],
                    'trb': examples['trb'][i],
                    'peptide': examples['peptide'][i],
                    'mhc_one': examples['mhc_one'][i],
                    'mhc_two': examples['mhc_two'][i],
                }
                sequences.append(concatenate_molecule_sequences(row, model_type))

        # Tokenize entire batch
        encoded = tokenizer.tokenize_batch(sequences)

        # Add sequences to output
        encoded['sequence'] = sequences

        return encoded

    return tokenize_function


def create_tokenize_function_fast(tokenizer):
    """
    Create a FAST tokenization function that uses pre-computed concatenated sequences.
    This avoids calling concatenate_molecule_sequences for every row during map().
    """
    def tokenize_function(examples):
        # Just tokenize the pre-computed 'concatenated_sequence' field
        sequences = examples['concatenated_sequence']

        # Tokenize entire batch
        encoded = tokenizer.tokenize_batch(sequences)

        # Add sequences to output
        encoded['sequence'] = sequences

        return encoded

    return tokenize_function


def detect_existing_split_chunks(output_dir: Path) -> Dict:
    """
    Detect existing train/val/test chunks and return resume state.

    Returns dict with:
        - chunks_exist: bool
        - train_chunk_files: List[Path] of existing train chunk directories
        - val_chunk_files: List[Path] of existing val chunk directories
        - test_chunk_files: List[Path] of existing test chunk directories
        - completed_chunk_nums: List[int] of completed chunk numbers (chunks that exist in all 3 splits)
        - metadata: Dict from resume_metadata.json (if exists)
        - last_chunk_num: int, highest chunk number found
    """
    train_chunks_dir = output_dir / "train_chunks"
    val_chunks_dir = output_dir / "val_chunks"
    test_chunks_dir = output_dir / "test_chunks"

    if not any(d.exists() for d in [train_chunks_dir, val_chunks_dir, test_chunks_dir]):
        return {
            'chunks_exist': False,
            'train_chunk_files': [],
            'val_chunk_files': [],
            'test_chunk_files': [],
            'completed_chunk_nums': [],
            'metadata': None,
            'last_chunk_num': 0
        }

    # Find chunks for each split
    train_chunks = sorted([p for p in train_chunks_dir.glob("chunk_*") if p.is_dir()]) if train_chunks_dir.exists() else []
    val_chunks = sorted([p for p in val_chunks_dir.glob("chunk_*") if p.is_dir()]) if val_chunks_dir.exists() else []
    test_chunks = sorted([p for p in test_chunks_dir.glob("chunk_*") if p.is_dir()]) if test_chunks_dir.exists() else []

    # Extract chunk numbers for each split
    def extract_chunk_nums(chunk_list):
        nums = []
        for chunk_path in chunk_list:
            try:
                chunk_num = int(chunk_path.name.split('_')[1])
                nums.append(chunk_num)
            except (IndexError, ValueError):
                print(f"⚠️  Warning: Malformed chunk directory name: {chunk_path.name}")
                continue
        return set(nums)

    train_nums = extract_chunk_nums(train_chunks)
    val_nums = extract_chunk_nums(val_chunks)
    test_nums = extract_chunk_nums(test_chunks)

    # Find intersection (all three splits must exist for a chunk to be considered complete)
    completed_chunk_nums = sorted(train_nums & val_nums & test_nums)

    # Load metadata
    metadata_file = output_dir / "resume_metadata.json"
    metadata = None
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Failed to load resume metadata: {e}")

    last_chunk_num = max(completed_chunk_nums) if completed_chunk_nums else 0

    return {
        'chunks_exist': bool(completed_chunk_nums),
        'train_chunk_files': train_chunks,
        'val_chunk_files': val_chunks,
        'test_chunk_files': test_chunks,
        'completed_chunk_nums': completed_chunk_nums,
        'metadata': metadata,
        'last_chunk_num': last_chunk_num
    }


def validate_chunk_compatibility(metadata: Dict, current_params: Dict) -> Tuple[bool, List[str]]:
    """
    Validate that existing chunks are compatible with current run parameters.

    Args:
        metadata: Resume metadata from detect_existing_chunks()
        current_params: Dict of current run parameters

    Returns:
        (is_compatible, list_of_issues)
    """
    if metadata is None:
        # No metadata file - assume incompatible for safety
        return False, ["No resume metadata file found (chunks may be from old version)"]

    issues = []
    critical_params = ['model_type', 'max_length', 'chunk_size']
    warning_params = ['batch_size', 'num_workers', 'use_fast_tokenizer']

    # Check critical parameters (must match exactly)
    for param in critical_params:
        if param in metadata and param in current_params:
            if metadata[param] != current_params[param]:
                issues.append(f"CRITICAL: {param} mismatch (saved: {metadata[param]}, current: {current_params[param]})")

    # Check warning parameters (log but allow)
    for param in warning_params:
        if param in metadata and param in current_params:
            if metadata[param] != current_params[param]:
                print(f"⚠️  Warning: {param} changed (saved: {metadata[param]}, current: {current_params[param]})")

    # Sampling parameter changes
    if 'sample' in metadata and 'sample' in current_params:
        if metadata['sample'] != current_params['sample']:
            issues.append(f"CRITICAL: sample size changed (saved: {metadata['sample']}, current: {current_params['sample']})")

    if 'sample_mode' in metadata and 'sample_mode' in current_params:
        if metadata['sample_mode'] != current_params['sample_mode']:
            issues.append(f"CRITICAL: sample_mode changed (saved: {metadata['sample_mode']}, current: {current_params['sample_mode']})")

    is_compatible = not any(issue.startswith("CRITICAL") for issue in issues)
    return is_compatible, issues


def save_resume_metadata_with_splits(
    output_dir: Path,
    params: Dict,
    completed_chunks: List[int],
    rows_per_split: Dict[str, int]
):
    """
    Save resume metadata for split-during-tokenization approach.

    Args:
        output_dir: Output directory
        params: Dict of run parameters
        completed_chunks: List of completed chunk numbers
        rows_per_split: Dict with keys 'train', 'validation', 'test' and their row counts
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = output_dir / "resume_metadata.json"

    from datetime import datetime

    # Accumulate total rows if metadata exists
    total_rows_per_split = {'train': 0, 'validation': 0, 'test': 0}
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                old_metadata = json.load(f)
                if 'total_rows_per_split' in old_metadata:
                    total_rows_per_split = old_metadata['total_rows_per_split']
        except Exception:
            pass

    # Add current chunk's rows
    for split, count in rows_per_split.items():
        total_rows_per_split[split] = total_rows_per_split.get(split, 0) + count

    metadata = {
        'version': '2.0',  # New version for split-during-tokenization
        'splitting_approach': 'split_during_tokenization',
        'model_type': params.get('model_type'),
        'max_length': params.get('max_length'),
        'batch_size': params.get('batch_size'),
        'num_workers': params.get('num_workers'),
        'chunk_size': params.get('chunk_size'),
        'sample': params.get('sample'),
        'sample_mode': params.get('sample_mode'),
        'mode': params.get('mode'),
        'stratified_split': params.get('stratified_split'),
        'test_split': params.get('test_split'),
        'val_split': params.get('val_split'),
        'use_fast_tokenizer': params.get('use_fast_tokenizer'),
        'completed_chunks': sorted(completed_chunks),
        'total_chunks_expected': params.get('total_chunks_expected', len(completed_chunks)),
        'last_updated': datetime.utcnow().isoformat() + 'Z',
        'total_rows_per_split': total_rows_per_split,
        'last_chunk_rows_per_split': rows_per_split
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def concatenate_chunks_in_batches(
    chunk_files: List[Path],
    batch_size: int = 5,
    split_name: str = "dataset"
) -> Dataset:
    """
    Concatenate chunks in batches to avoid OOM.

    Args:
        chunk_files: List of chunk directory paths
        batch_size: Number of chunks to load at once (default: 5)
        split_name: Name of the split for logging (train/val/test)

    Returns:
        Concatenated Dataset
    """
    from datasets import concatenate_datasets, load_from_disk
    import gc

    if not chunk_files:
        raise ValueError(f"No chunk files provided for {split_name}")

    accumulated_dataset = None
    num_batches = (len(chunk_files) + batch_size - 1) // batch_size

    for i in range(0, len(chunk_files), batch_size):
        batch = chunk_files[i:i+batch_size]
        batch_num = i//batch_size + 1
        print(f"   Loading {split_name} batch {batch_num}/{num_batches} ({len(batch)} chunks)...")

        # Load batch
        batch_datasets = [load_from_disk(str(p)) for p in batch]
        batch_concat = concatenate_datasets(batch_datasets)
        del batch_datasets
        gc.collect()

        # Accumulate
        if accumulated_dataset is None:
            accumulated_dataset = batch_concat
        else:
            print(f"   Merging batch {batch_num} with accumulated dataset...")
            accumulated_dataset = concatenate_datasets([accumulated_dataset, batch_concat])
            del batch_concat
            gc.collect()

    return accumulated_dataset


def validate_split_balance(
    pkey_train_counts: Dict[str, int],
    pkey_val_counts: Dict[str, int],
    pkey_test_counts: Dict[str, int],
    expected_train_ratio: float,
    expected_val_ratio: float,
    expected_test_ratio: float,
    tolerance: float = 0.05
) -> Tuple[bool, List[str]]:
    """
    Validate that splits maintain expected class balance.

    Args:
        pkey_train_counts: Count of each permutation_key in train split
        pkey_val_counts: Count of each permutation_key in val split
        pkey_test_counts: Count of each permutation_key in test split
        expected_train_ratio: Expected train fraction (e.g., 0.8)
        expected_val_ratio: Expected validation fraction (e.g., 0.1)
        expected_test_ratio: Expected test fraction (e.g., 0.1)
        tolerance: Tolerance for balance check (default: 0.05 = ±5%)

    Returns:
        Tuple of (is_balanced, list_of_warnings)
    """
    warnings = []
    all_balanced = True

    all_pkeys = set(pkey_train_counts.keys()) | set(pkey_val_counts.keys()) | set(pkey_test_counts.keys())

    print(f"\n📊 Split Balance Validation (tolerance: ±{tolerance*100:.1f}%):")
    print(f"{'Permutation Key':<30} {'Train %':>10} {'Val %':>10} {'Test %':>10} {'Status':>10}")
    print("-" * 72)

    for pkey in sorted(all_pkeys):
        train_count = pkey_train_counts.get(pkey, 0)
        val_count = pkey_val_counts.get(pkey, 0)
        test_count = pkey_test_counts.get(pkey, 0)
        total = train_count + val_count + test_count

        if total == 0:
            continue

        train_ratio = train_count / total
        val_ratio = val_count / total
        test_ratio = test_count / total

        train_ok = abs(train_ratio - expected_train_ratio) <= tolerance
        val_ok = abs(val_ratio - expected_val_ratio) <= tolerance
        test_ok = abs(test_ratio - expected_test_ratio) <= tolerance

        status = "✓" if (train_ok and val_ok and test_ok) else "⚠️"
        if not (train_ok and val_ok and test_ok):
            all_balanced = False
            warnings.append(f"{pkey}: Train {train_ratio:.1%}, Val {val_ratio:.1%}, Test {test_ratio:.1%}")

        print(f"{pkey:<30} {train_ratio:>9.1%} {val_ratio:>9.1%} {test_ratio:>9.1%} {status:>10}")

    if not all_balanced:
        print(f"\n⚠️  Warning: {len(warnings)} classes not balanced within tolerance")
    else:
        print(f"\n✓ All classes balanced within tolerance")

    return all_balanced, warnings


def tokenize_dataset(
    input_dir: Path,
    output_dir: Path,
    model_type: str,
    max_length: int = 512,
    vocab_size: int = 1000,
    sample: Optional[int] = None,
    sample_mode: Optional[str] = None,
    train_bpe: bool = False,
    test_split: float = 0.1,
    val_split: float = 0.1,
    chunk_size: int = 50_000_000,
    batch_size: int = 10000,
    num_workers: int = 40,
    precompute_workers: int = 60,
    use_legacy_pipeline: bool = False,
    use_incremental_write: bool = True,
    oversample: bool = False,
    mode: Optional[str] = None,
<<<<<<< HEAD
    resume: bool = False,
    skip_tcr_stitching: bool = False,
):
    """
    Main tokenization function with streaming support.
=======
    use_fast_tokenizer: bool = True,
    stratified_split: bool = True,
    resume: bool = False,
):
    """
    Main tokenization function with streaming support (OPTIMIZED).
>>>>>>> 2a62664 (Updated to incrementally write toeknized shards and implement stratified data splits)

    Args:
        chunk_size: Rows per chunk (default 50M = ~7.5GB RAM)
        batch_size: Batch size for Dataset.map() tokenization (default 10000)
        num_workers: Parallel workers for Dataset.map() (default 40)
        precompute_workers: Parallel workers for sequence pre-computation (default 60, 0=disable parallelization)
        sample: Number of sequences to sample (optional)
        sample_mode: Sampling strategy - 'proportional' (maintain distribution) or 'balanced' (equal per permutation)
        oversample: If True and sample_mode='balanced', duplicate underrepresented samples to reach target
        mode: Filter mode - only include permutation keys matching this mode (mlm, tra, trb, tra_trb_pairing, tcr_mhc, peptide_mhc, specificity)
<<<<<<< HEAD
        resume: If True, resume from existing output (skip already processed rows)
        skip_tcr_stitching: If True, skip expensive TCR stitching/CDR identification (much faster, but no full-length TCR sequences)
=======
        stratified_split: If True, use stratified split to maintain class balance; if False, use random split
>>>>>>> 2a62664 (Updated to incrementally write toeknized shards and implement stratified data splits)
    """
    print(f"\n{'='*60}")
    print(f"TOKENIZATION: {model_type.upper()} (SPLIT-DURING-TOKENIZATION)")
    print(f"{'='*60}\n")
    print(f"⚙️  Chunk size: {chunk_size:,} rows")
    print(f"⚙️  Batch size: {batch_size:,} sequences per tokenization call")
    print(f"⚙️  Workers: {num_workers}")
    print(f"⚙️  Max RAM usage: ~{chunk_size * 150 / 1e9:.0f}GB per chunk")
    print(f"⚙️  Split strategy: {'Stratified' if stratified_split else 'Random'} ({test_split:.0%} test, {val_split:.0%} val, {1-test_split-val_split:.0%} train)")
    if sample:
        oversample_str = ", with oversampling" if oversample else ""
        print(f"⚙️  Sampling: {sample:,} sequences (mode: {sample_mode or 'first-N'}{oversample_str})")
    if resume:
        print(f"⚙️  Resume mode: enabled")
    if skip_tcr_stitching:
        print(f"⚡ TCR stitching: DISABLED (fast mode)")
    print()

<<<<<<< HEAD
    # Check for existing output and calculate resume point
    skip_rows = 0
    existing_split_counts = {'train': 0, 'validation': 0, 'test': 0}

    if resume:
        print(f"🔍 Checking for existing output...")
        existing_split_counts, existing_total = detect_existing_output(output_dir)

        if existing_total > 0:
            skip_rows = existing_total
            print(f"✓ Found existing output: {existing_total:,} rows already processed")
            print(f"   Train: {existing_split_counts['train']:,}, Val: {existing_split_counts['validation']:,}, Test: {existing_split_counts['test']:,}")
            print(f"   Resuming from row {skip_rows:,}...")

            # If sampling, check if we've already reached the sample size
            if sample and existing_total >= sample:
                print(f"⚠️  Sample size ({sample:,}) already reached. Nothing to do.")
                return
        else:
            print(f"   No existing output found. Starting from beginning.")
=======
    # Resume detection
    resume_state = None
    if resume:
        print(f"\n{'='*60}")
        print(f"🔄 RESUME MODE ENABLED")
        print(f"{'='*60}")

        resume_state = detect_existing_split_chunks(output_dir)

        if resume_state['chunks_exist']:
            print(f"✓ Found {len(resume_state['completed_chunk_nums'])} existing split chunks")
            print(f"   Chunk numbers: {resume_state['completed_chunk_nums']}")
            print(f"   Train chunks: {len(resume_state['train_chunk_files'])}")
            print(f"   Val chunks: {len(resume_state['val_chunk_files'])}")
            print(f"   Test chunks: {len(resume_state['test_chunk_files'])}")
            if resume_state['last_chunk_num'] > 0:
                print(f"   Last completed chunk: {resume_state['last_chunk_num']}")

            # Validate compatibility
            current_params = {
                'model_type': model_type,
                'max_length': max_length,
                'batch_size': batch_size,
                'num_workers': num_workers,
                'chunk_size': chunk_size,
                'sample': sample,
                'sample_mode': sample_mode,
                'mode': mode,
                'stratified_split': stratified_split,
                'test_split': test_split,
                'val_split': val_split,
                'use_fast_tokenizer': use_fast_tokenizer
            }

            is_compatible, issues = validate_chunk_compatibility(resume_state['metadata'], current_params)

            if not is_compatible:
                print(f"\n❌ INCOMPATIBLE CHUNKS DETECTED:")
                for issue in issues:
                    print(f"   - {issue}")
                print(f"\n💡 Options:")
                print(f"   1. Delete split chunk directories and restart from scratch")
                print(f"   2. Use same parameters as original run")
                raise ValueError("Cannot resume: incompatible chunk parameters")

            print(f"✓ Chunk compatibility validated")
            print(f"   Will resume from chunk {resume_state['last_chunk_num'] + 1}")
        else:
            print(f"ℹ️  No existing split chunks found, starting from scratch")
            resume_state = None
>>>>>>> 2a62664 (Updated to incrementally write toeknized shards and implement stratified data splits)

    # Create tokenizer
    if model_type == "protbert":
        tokenizer = ProtBERTTokenizer(max_length, use_fast=use_fast_tokenizer)
    elif model_type == "bert":
        tokenizer = StandardBERTTokenizer(max_length, model_name="google-bert/bert-base-cased", use_fast=use_fast_tokenizer)
    elif model_type == "esm2":
        tokenizer = ESM2Tokenizer(max_length, use_fast=use_fast_tokenizer)
    elif model_type == "esm3":
        tokenizer = ESM3Tokenizer(max_length)  # ESM3 uses custom tokenizer, no use_fast parameter
    elif model_type in ["bpe", "lstm", "transformer"]:
        tokenizer = BPETokenizer(max_length, vocab_size)

        # Train BPE if requested
        if train_bpe:
            print("\n🔧 Training BPE tokenizer (streaming mode)...")
            all_sequences = []
            sequence_count = 0
            max_training_sequences = 1_000_000  # Limit for BPE training

            for chunk_rows, _ in stream_parquet_files(input_dir, sample, chunk_size, sample_mode, num_workers, mode=mode):
                for row in chunk_rows:
                    for field in ['tra', 'trb', 'peptide', 'mhc_one', 'mhc_two']:
                        val = row.get(field, '')
                        if val and str(val) != 'nan' and str(val) != '':
                            all_sequences.append(str(val))
                            sequence_count += 1
                            if sequence_count >= max_training_sequences:
                                break
                    if sequence_count >= max_training_sequences:
                        break
                if sequence_count >= max_training_sequences:
                    break

            print(f"Training BPE on {len(all_sequences):,} sequences...")
            tokenizer.train_on_sequences(all_sequences, output_dir / "tokenizer")
        else:
            print("Loading pre-trained BPE tokenizer...")
            tokenizer.load(output_dir / "tokenizer")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
<<<<<<< HEAD
    for split in ['train', 'validation', 'test']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    # Initialize counters (use existing counts if resuming)
    split_writers = {
        'train': [],
        'validation': [],
        'test': []
    }
    split_counts = existing_split_counts.copy() if resume else {'train': 0, 'validation': 0, 'test': 0}

    # Calculate starting chunk number based on existing files
    starting_chunk_nums = {'train': 0, 'validation': 0, 'test': 0}
    if resume:
        for split_name in ['train', 'validation', 'test']:
            split_dir = output_dir / split_name
            if split_dir.exists():
                existing_chunks = list(split_dir.glob("chunk_*.parquet"))
                if existing_chunks:
                    # Get the highest chunk number
                    chunk_nums = [int(f.stem.split('_')[1]) for f in existing_chunks]
                    starting_chunk_nums[split_name] = max(chunk_nums) + 1
    
    # Track per-permutation split distribution for validation
    pkey_split_totals: Dict[str, Dict[str, int]] = {}

    # Track cumulative TCR stitching statistics
    cumulative_tcr_stats = {
        'tra_attempted': 0,
        'tra_gene_std': 0,
        'tra_stitched': 0,
        'tra_cdr': 0,
        'trb_attempted': 0,
        'trb_gene_std': 0,
        'trb_stitched': 0,
        'trb_cdr': 0,
        'by_pkey': {}
    }
=======

    # Setup for split-during-tokenization chunk processing
    if use_incremental_write:
        # Incremental mode: save split chunks to disk immediately
        train_chunks_dir = output_dir / "train_chunks"
        val_chunks_dir = output_dir / "val_chunks"
        test_chunks_dir = output_dir / "test_chunks"

        for split_dir in [train_chunks_dir, val_chunks_dir, test_chunks_dir]:
            split_dir.mkdir(parents=True, exist_ok=True)

        train_chunk_files = []
        val_chunk_files = []
        test_chunk_files = []

        print(f"💾 Incremental write mode enabled - split chunks will be saved to:")
        print(f"   Train: {train_chunks_dir}")
        print(f"   Val:   {val_chunks_dir}")
        print(f"   Test:  {test_chunks_dir}")
    else:
        # Legacy mode: accumulate all chunks in memory (may cause OOM!)
        all_tokenized_chunks = []
        print(f"⚠️  Legacy mode: accumulating chunks in memory (may cause OOM on large datasets)")
        print(f"⚠️  Note: Split-during-tokenization requires incremental write mode")

    # Track per-permutation distribution for each split
    pkey_train_counts: Dict[str, int] = {}
    pkey_val_counts: Dict[str, int] = {}
    pkey_test_counts: Dict[str, int] = {}
>>>>>>> 2a62664 (Updated to incrementally write toeknized shards and implement stratified data splits)

    # Process in streaming chunks
    print(f"\n📦 Processing dataset in chunks...")
    np.random.seed(42)

<<<<<<< HEAD
    chunk_num = 0
    processing_chunk_num = 0  # Tracks chunks actually processed (for display)
    rows_skipped = 0

    for chunk_rows, total_rows in stream_parquet_files(input_dir, sample, chunk_size, sample_mode, num_workers, oversample, mode):
        chunk_num += 1  # Always increment for consistent seed generation

        # Skip chunks if resuming
        if resume and rows_skipped < skip_rows:
            rows_to_skip_in_chunk = min(len(chunk_rows), skip_rows - rows_skipped)

            if rows_to_skip_in_chunk >= len(chunk_rows):
                # Skip entire chunk
                rows_skipped += len(chunk_rows)
                continue
            else:
                # Skip partial chunk
                chunk_rows = chunk_rows[rows_to_skip_in_chunk:]
                rows_skipped += rows_to_skip_in_chunk

        processing_chunk_num += 1
        print(f"\n📦 Processing chunk {processing_chunk_num} (stream chunk {chunk_num}) ({len(chunk_rows):,} rows, total: {total_rows:,})")

        # Stratified split assignment for this chunk (preserves class balance)
        train_ratio = 1 - test_split - val_split
        chunk_splits = stratified_split_rows(
            chunk_rows,
            train_ratio=train_ratio,
            val_ratio=val_split,
            test_ratio=test_split,
            seed=42 + chunk_num  # Vary seed per chunk for variety
        )

        # Log split distribution for first chunk to validate stratification
        if processing_chunk_num == 1:
            from collections import Counter
            split_counts_preview = Counter(chunk_splits)
            print(f"   📊 Chunk 1 split preview: {dict(split_counts_preview)}")

            # Show per-class distribution for validation
            pkey_split_counts: Dict[str, Dict[str, int]] = {}
            for i, row in enumerate(chunk_rows):
                pkey = row.get('permutation_key', get_permutation_signature(row))
                if pkey not in pkey_split_counts:
                    pkey_split_counts[pkey] = {'train': 0, 'validation': 0, 'test': 0}
                pkey_split_counts[pkey][chunk_splits[i]] += 1

            print(f"   📊 Stratified split by class (first 5):")
            for pkey in list(pkey_split_counts.keys())[:5]:
                counts = pkey_split_counts[pkey]
                total = sum(counts.values())
                print(f"      {pkey}: {counts} (total: {total})")

        # Add full TCR sequences and CDR positions (NEW) - skip if disabled
        if not skip_tcr_stitching:
            chunk_rows, chunk_tcr_stats = add_full_tcr_and_cdr_positions(chunk_rows)
        else:
            # Skip TCR stitching - create empty stats
            chunk_tcr_stats = {
                'tra_attempted': 0, 'tra_gene_std': 0, 'tra_stitched': 0, 'tra_cdr': 0,
                'trb_attempted': 0, 'trb_gene_std': 0, 'trb_stitched': 0, 'trb_cdr': 0,
                'by_pkey': {}
            }
            # Add empty columns for consistency
            for row in chunk_rows:
                row['trav_gene_std'] = ''
                row['traj_gene_std'] = ''
                row['trbv_gene_std'] = ''
                row['trbj_gene_std'] = ''
                row['tra_full'] = ''
                row['trb_full'] = ''
                row['tra_cdr1_pos'] = None
                row['tra_cdr2_pos'] = None
                row['tra_cdr3_pos'] = None
                row['trb_cdr1_pos'] = None
                row['trb_cdr2_pos'] = None
                row['trb_cdr3_pos'] = None

        # Accumulate statistics
        cumulative_tcr_stats['tra_attempted'] += chunk_tcr_stats['tra_attempted']
        cumulative_tcr_stats['tra_gene_std'] += chunk_tcr_stats['tra_gene_std']
        cumulative_tcr_stats['tra_stitched'] += chunk_tcr_stats['tra_stitched']
        cumulative_tcr_stats['tra_cdr'] += chunk_tcr_stats['tra_cdr']
        cumulative_tcr_stats['trb_attempted'] += chunk_tcr_stats['trb_attempted']
        cumulative_tcr_stats['trb_gene_std'] += chunk_tcr_stats['trb_gene_std']
        cumulative_tcr_stats['trb_stitched'] += chunk_tcr_stats['trb_stitched']
        cumulative_tcr_stats['trb_cdr'] += chunk_tcr_stats['trb_cdr']

        # Merge per-permutation-key stats
        for pkey, pkey_stats in chunk_tcr_stats['by_pkey'].items():
            if pkey not in cumulative_tcr_stats['by_pkey']:
                cumulative_tcr_stats['by_pkey'][pkey] = {
                    'tra_attempted': 0, 'tra_gene_std': 0, 'tra_stitched': 0, 'tra_cdr': 0,
                    'trb_attempted': 0, 'trb_gene_std': 0, 'trb_stitched': 0, 'trb_cdr': 0
                }
            for key in pkey_stats:
                cumulative_tcr_stats['by_pkey'][pkey][key] += pkey_stats[key]

        # Display cumulative statistics
        tra_success_rate = (cumulative_tcr_stats['tra_stitched'] / cumulative_tcr_stats['tra_attempted'] * 100) if cumulative_tcr_stats['tra_attempted'] > 0 else 0
        trb_success_rate = (cumulative_tcr_stats['trb_stitched'] / cumulative_tcr_stats['trb_attempted'] * 100) if cumulative_tcr_stats['trb_attempted'] > 0 else 0
        print(f"   📊 Cumulative: TRA {cumulative_tcr_stats['tra_stitched']:,}/{cumulative_tcr_stats['tra_attempted']:,} ({tra_success_rate:.1f}%), "
              f"TRB {cumulative_tcr_stats['trb_stitched']:,}/{cumulative_tcr_stats['trb_attempted']:,} ({trb_success_rate:.1f}%)")
=======
    # Initialize chunk counter (resume from last chunk if needed)
    if resume_state and resume_state['chunks_exist']:
        chunk_num = resume_state['last_chunk_num']
        print(f"\n📦 Resuming from chunk {chunk_num + 1} (skipping {chunk_num} completed chunks)")
        chunks_to_skip = set(resume_state['completed_chunk_nums'])
    else:
        chunk_num = 0
        chunks_to_skip = set()

    total_rows_processed = 0

    for chunk_rows, total_rows in stream_parquet_files(input_dir, sample, chunk_size, sample_mode, num_workers, oversample, mode):
        chunk_num += 1

        # Skip this chunk if already completed
        if chunk_num in chunks_to_skip:
            print(f"\n📦 Skipping chunk {chunk_num} (already completed)")
            continue

        chunk_start_time = time.time()
        print(f"\n📦 Processing chunk {chunk_num} ({len(chunk_rows):,} rows, total: {total_rows:,})")

        # Track permutation distribution
        if chunk_num == 1:
            from collections import Counter
            pkey_counts = Counter([row.get('permutation_key', get_permutation_signature(row)) for row in chunk_rows])
            print(f"   📊 Chunk 1 permutation distribution (first 5):")
            for pkey, count in list(pkey_counts.items())[:5]:
                print(f"      {pkey}: {count:,}")

        # PRE-COMPUTE concatenated sequences to avoid per-row function calls during tokenization
        precompute_start = time.time()
>>>>>>> 2a62664 (Updated to incrementally write toeknized shards and implement stratified data splits)

        # Try PyArrow vectorization first (fastest for ESM-2/ESM-3 with new format)
        # Skip if legacy pipeline is requested
        if not use_legacy_pipeline:
            concatenated_sequences = concatenate_sequences_arrow_esm2(chunk_rows, model_type)
        else:
            concatenated_sequences = None

        if concatenated_sequences is not None:
            # PyArrow vectorization succeeded (ESM-2/ESM-3 with new format)
            print(f"   ⚡ Pre-computing concatenated sequences (PyArrow vectorized)...")
            precompute_time = time.time() - precompute_start
            print(f"      ⏱️  Pre-compute time: {precompute_time:.2f}s ({len(chunk_rows)/precompute_time:.0f} rows/sec)")

        elif precompute_workers > 0 and len(chunk_rows) > 10000:
            # Fallback: Parallel mode for complex cases or old format
            print(f"   ⚡ Pre-computing concatenated sequences with {precompute_workers} workers...")
            batch_size_worker = max(1000, len(chunk_rows) // precompute_workers)
            batches = [
                (chunk_rows[i:i+batch_size_worker], model_type)
                for i in range(0, len(chunk_rows), batch_size_worker)
            ]

            with Pool(precompute_workers) as pool:
                results = pool.map(_concatenate_batch_worker, batches)

            # Flatten results from all workers
            concatenated_sequences = [seq for batch_result in results for seq in batch_result]
            precompute_time = time.time() - precompute_start
            print(f"      ⏱️  Pre-compute time: {precompute_time:.2f}s ({len(chunk_rows)/precompute_time:.0f} rows/sec)")

        else:
            # Sequential mode: fallback for small chunks or when parallelization is disabled
            print(f"   ⚡ Pre-computing concatenated sequences (sequential)...")
            concatenated_sequences = [
                concatenate_molecule_sequences(row, model_type)
                for row in chunk_rows
            ]
            precompute_time = time.time() - precompute_start
            print(f"      ⏱️  Pre-compute time: {precompute_time:.2f}s ({len(chunk_rows)/precompute_time:.0f} rows/sec)")

        # Create HuggingFace Dataset with concatenated sequences (optimized zero-copy path)
        print(f"   ⚡ Creating Dataset with concatenated sequences...")
        dataset_start = time.time()

        # Convert chunk_rows to PyArrow table
        table = pa.Table.from_pylist(chunk_rows)

        # Append concatenated sequences as a new column (zero-copy operation)
        concat_array = pa.array(concatenated_sequences, type=pa.string())
        table = table.append_column('concatenated_sequence', concat_array)

        # Create Dataset from PyArrow table (zero-copy!)
        chunk_dataset = Dataset(arrow_table=table)

        dataset_time = time.time() - dataset_start
        print(f"      ⏱️  Dataset creation time: {dataset_time:.2f}s")

        # Note: Row mutation step eliminated! (previously ~5-7 seconds)

        # Tokenize using Dataset.map with multiprocessing (FAST version)
        # Uses pre-computed 'concatenated_sequence' field to avoid per-row overhead
        print(f"   ⚡ Tokenizing with {min(40, num_workers)} workers (batch_size={batch_size:,})...")
        tokenize_start = time.time()
        tokenize_fn_fast = create_tokenize_function_fast(tokenizer)
        tokenized_dataset = chunk_dataset.map(
            tokenize_fn_fast,
            batched=True,
            batch_size=batch_size,
            num_proc=min(40, num_workers),
            desc="   Tokenizing"
        )
        tokenize_time = time.time() - tokenize_start
        print(f"      ⏱️  Tokenization time: {tokenize_time:.2f}s")

<<<<<<< HEAD
        # Add back TCR-specific columns (NEW - for full_tra/full_trb modes)
        tcr_columns = [
            'tra_full', 'trb_full',
            'trav_gene', 'traj_gene', 'trbv_gene', 'trbj_gene',
            'trav_gene_std', 'traj_gene_std', 'trbv_gene_std', 'trbj_gene_std',  # Standardized gene names
            'tra_cdr1_pos', 'tra_cdr2_pos', 'tra_cdr3_pos',
            'trb_cdr1_pos', 'trb_cdr2_pos', 'trb_cdr3_pos'
        ]
        for col in tcr_columns:
            if col in chunk_dataset.column_names:
                tokenized_dataset = tokenized_dataset.add_column(col, chunk_dataset[col])
        
        # Convert to list for splitting
        tokenized_rows = tokenized_dataset.to_pandas().to_dict('records')
        
        # Track per-permutation split distribution
        for i, row in enumerate(chunk_rows):
            pkey = row.get('permutation_key', get_permutation_signature(row))
            if pkey not in pkey_split_totals:
                pkey_split_totals[pkey] = {'train': 0, 'validation': 0, 'test': 0}
            pkey_split_totals[pkey][chunk_splits[i]] += 1
        
        # Split and write
        for split_name in ['train', 'validation', 'test']:
            split_mask = chunk_splits == split_name
            split_data = [tokenized_rows[i] for i in range(len(tokenized_rows)) if split_mask[i]]

            if split_data:
                # Write chunk to parquet (use starting_chunk_nums for proper numbering when resuming)
                df = pa.Table.from_pylist(split_data)
                chunk_file = output_dir / split_name / f"chunk_{starting_chunk_nums[split_name]:06d}.parquet"
                pq.write_table(df, chunk_file)

                split_counts[split_name] += len(split_data)
                split_writers[split_name].append(chunk_file)
                starting_chunk_nums[split_name] += 1
        
        print(f"   ✓ Chunk {processing_chunk_num} complete")
        print(f"   📊 Cumulative: Train={split_counts['train']:,}, "
              f"Val={split_counts['validation']:,}, Test={split_counts['test']:,}")
    
    # Print stratified split validation summary
=======
        # ===== NEW: Split the tokenized dataset into train/val/test =====
        print(f"   🎯 Splitting chunk into train/val/test...")
        split_start = time.time()

        # Fast column access - get all permutation keys at once (avoid slow indexed access)
        if 'permutation_key' in tokenized_dataset.column_names:
            pkeys = tokenized_dataset['permutation_key']
        else:
            pkeys = [''] * len(tokenized_dataset)

        chunk_rows_for_split = [{'permutation_key': pkey} for pkey in pkeys]

        # Get split assignments using stratified or random split
        if stratified_split:
            split_assignments = stratified_split_rows(
                chunk_rows_for_split,
                train_ratio=1.0 - test_split - val_split,
                val_ratio=val_split,
                test_ratio=test_split,
                seed=42
            )
        else:
            split_assignments = random_split_rows(
                chunk_rows_for_split,
                train_ratio=1.0 - test_split - val_split,
                val_ratio=val_split,
                test_ratio=test_split,
                seed=42
            )

        # Create indices for each split
        train_indices = np.where(split_assignments == 'train')[0].tolist()
        val_indices = np.where(split_assignments == 'validation')[0].tolist()
        test_indices = np.where(split_assignments == 'test')[0].tolist()

        # Select rows for each split (zero-copy operation)
        train_dataset = tokenized_dataset.select(train_indices)
        val_dataset = tokenized_dataset.select(val_indices)
        test_dataset = tokenized_dataset.select(test_indices)

        split_time = time.time() - split_start
        print(f"      Train: {len(train_dataset):,} ({len(train_dataset)/len(tokenized_dataset)*100:.1f}%)")
        print(f"      Val:   {len(val_dataset):,} ({len(val_dataset)/len(tokenized_dataset)*100:.1f}%)")
        print(f"      Test:  {len(test_dataset):,} ({len(test_dataset)/len(tokenized_dataset)*100:.1f}%)")
        print(f"      ⏱️  Split time: {split_time:.2f}s")

        # Track permutation distribution for each split
        if 'permutation_key' in tokenized_dataset.column_names:
            for pkey in train_dataset['permutation_key']:
                pkey_train_counts[pkey] = pkey_train_counts.get(pkey, 0) + 1
            for pkey in val_dataset['permutation_key']:
                pkey_val_counts[pkey] = pkey_val_counts.get(pkey, 0) + 1
            for pkey in test_dataset['permutation_key']:
                pkey_test_counts[pkey] = pkey_test_counts.get(pkey, 0) + 1

        # Save split chunks or accumulate
        chunk_size_rows = len(tokenized_dataset)  # Save before potentially deleting
        if use_incremental_write:
            # Incremental mode: save three split chunks to disk immediately
            train_path = train_chunks_dir / f"chunk_{chunk_num:04d}"
            val_path = val_chunks_dir / f"chunk_{chunk_num:04d}"
            test_path = test_chunks_dir / f"chunk_{chunk_num:04d}"

            print(f"   💾 Saving split chunks...")
            save_start = time.time()

            train_dataset.save_to_disk(str(train_path))
            val_dataset.save_to_disk(str(val_path))
            test_dataset.save_to_disk(str(test_path))

            save_time = time.time() - save_start
            print(f"      ⏱️  Save time: {save_time:.2f}s")

            train_chunk_files.append(train_path)
            val_chunk_files.append(val_path)
            test_chunk_files.append(test_path)

            # Update resume metadata after each successful chunk
            if resume:
                completed_chunks = resume_state['completed_chunk_nums'] if resume_state and resume_state['chunks_exist'] else []
                completed_chunks.append(chunk_num)

                current_params = {
                    'model_type': model_type,
                    'max_length': max_length,
                    'batch_size': batch_size,
                    'num_workers': num_workers,
                    'chunk_size': chunk_size,
                    'sample': sample,
                    'sample_mode': sample_mode,
                    'mode': mode,
                    'stratified_split': stratified_split,
                    'test_split': test_split,
                    'val_split': val_split,
                    'use_fast_tokenizer': use_fast_tokenizer
                }

                rows_per_split = {
                    'train': len(train_dataset),
                    'validation': len(val_dataset),
                    'test': len(test_dataset)
                }

                save_resume_metadata_with_splits(output_dir, current_params, completed_chunks, rows_per_split)

                # Update resume_state for next iteration
                if resume_state is None:
                    resume_state = {'completed_chunk_nums': []}
                resume_state['completed_chunk_nums'] = completed_chunks

            # Clear from memory
            del tokenized_dataset, train_dataset, val_dataset, test_dataset, chunk_dataset
            import gc
            gc.collect()
        else:
            # Legacy mode: accumulate in memory (may cause OOM!)
            all_tokenized_chunks.append(tokenized_dataset)
            print(f"⚠️  Warning: Legacy mode doesn't support split-during-tokenization")

        total_rows_processed += chunk_size_rows

        chunk_total_time = time.time() - chunk_start_time
        print(f"\n   ✓ Chunk {chunk_num} complete ({chunk_size_rows:,} rows)")
        print(f"   ⏱️  TOTAL CHUNK TIME: {chunk_total_time:.2f}s ({len(chunk_rows)/chunk_total_time:.0f} rows/sec)")
        print(f"   📊 Breakdown:")
        print(f"      - Pre-compute:          {precompute_time:6.2f}s ({precompute_time/chunk_total_time*100:5.1f}%)")
        print(f"      - Dataset creation:     {dataset_time:6.2f}s ({dataset_time/chunk_total_time*100:5.1f}%)")
        print(f"      - Tokenization:         {tokenize_time:6.2f}s ({tokenize_time/chunk_total_time*100:5.1f}%)")
        print(f"      - Splitting:            {split_time:6.2f}s ({split_time/chunk_total_time*100:5.1f}%)")
        other_time = chunk_total_time - precompute_time - dataset_time - tokenize_time - split_time
        print(f"      - Other (I/O, etc):     {other_time:6.2f}s ({other_time/chunk_total_time*100:5.1f}%)")
        print(f"   📊 Cumulative total: {total_rows_processed:,} rows")

    # Concatenate split chunks separately (batched approach to avoid OOM)
>>>>>>> 2a62664 (Updated to incrementally write toeknized shards and implement stratified data splits)
    print(f"\n{'='*60}")
    print(f"🔗 CONCATENATING SPLIT CHUNKS")
    print(f"{'='*60}")
<<<<<<< HEAD
    print(f"\nPer-permutation split distribution:")
    
    # Sort by total count to show smallest classes first (most important for balanced sampling)
    sorted_pkeys = sorted(pkey_split_totals.keys(), key=lambda k: sum(pkey_split_totals[k].values()))
    
    for pkey in sorted_pkeys:
        counts = pkey_split_totals[pkey]
        total = sum(counts.values())
        train_pct = counts['train'] / total * 100 if total > 0 else 0
        val_pct = counts['validation'] / total * 100 if total > 0 else 0
        test_pct = counts['test'] / total * 100 if total > 0 else 0
        print(f"  {pkey:40s}: train={counts['train']:6,} ({train_pct:5.1f}%), "
              f"val={counts['validation']:6,} ({val_pct:5.1f}%), "
              f"test={counts['test']:6,} ({test_pct:5.1f}%) [total: {total:,}]")
    
    # Show summary for smallest class
    if sorted_pkeys:
        smallest_pkey = sorted_pkeys[0]
        smallest_counts = pkey_split_totals[smallest_pkey]
        print(f"\n✓ Smallest class '{smallest_pkey}' preserved with stratified split:")
        print(f"  Train: {smallest_counts['train']:,}, Val: {smallest_counts['validation']:,}, Test: {smallest_counts['test']:,}")

    # Print TCR stitching summary (only if not skipped)
    if not skip_tcr_stitching:
        print(f"\n{'='*60}")
        print(f"🧬 TCR STITCHING SUMMARY")
        print(f"{'='*60}")

        # Overall statistics
        print(f"\n📊 Overall Statistics:")
        print(f"  TRA:")
        print(f"    Attempted:          {cumulative_tcr_stats['tra_attempted']:10,}")
        print(f"    Gene standardized:  {cumulative_tcr_stats['tra_gene_std']:10,} ({100*cumulative_tcr_stats['tra_gene_std']/cumulative_tcr_stats['tra_attempted']:.1f}%)" if cumulative_tcr_stats['tra_attempted'] > 0 else "    Gene standardized:  0")
        print(f"    Successfully stitched: {cumulative_tcr_stats['tra_stitched']:7,} ({100*cumulative_tcr_stats['tra_stitched']/cumulative_tcr_stats['tra_attempted']:.1f}%)" if cumulative_tcr_stats['tra_attempted'] > 0 else "    Successfully stitched: 0")
        print(f"    CDR positions found: {cumulative_tcr_stats['tra_cdr']:9,} ({100*cumulative_tcr_stats['tra_cdr']/cumulative_tcr_stats['tra_stitched']:.1f}%)" if cumulative_tcr_stats['tra_stitched'] > 0 else "    CDR positions found: 0")

        print(f"\n  TRB:")
        print(f"    Attempted:          {cumulative_tcr_stats['trb_attempted']:10,}")
        print(f"    Gene standardized:  {cumulative_tcr_stats['trb_gene_std']:10,} ({100*cumulative_tcr_stats['trb_gene_std']/cumulative_tcr_stats['trb_attempted']:.1f}%)" if cumulative_tcr_stats['trb_attempted'] > 0 else "    Gene standardized:  0")
        print(f"    Successfully stitched: {cumulative_tcr_stats['trb_stitched']:7,} ({100*cumulative_tcr_stats['trb_stitched']/cumulative_tcr_stats['trb_attempted']:.1f}%)" if cumulative_tcr_stats['trb_attempted'] > 0 else "    Successfully stitched: 0")
        print(f"    CDR positions found: {cumulative_tcr_stats['trb_cdr']:9,} ({100*cumulative_tcr_stats['trb_cdr']/cumulative_tcr_stats['trb_stitched']:.1f}%)" if cumulative_tcr_stats['trb_stitched'] > 0 else "    CDR positions found: 0")

        # Per-permutation-key breakdown
        if cumulative_tcr_stats['by_pkey']:
            print(f"\n📊 Per-Permutation-Key Breakdown:")
            print(f"{'Permutation Key':<30} {'TRA Att.':>10} {'TRA Stitch':>12} {'Success %':>10} {'TRB Att.':>10} {'TRB Stitch':>12} {'Success %':>10}")
            print(f"{'-'*110}")

            # Sort by permutation key name for readability
            for pkey in sorted(cumulative_tcr_stats['by_pkey'].keys()):
                pkey_stats = cumulative_tcr_stats['by_pkey'][pkey]

                tra_att = pkey_stats['tra_attempted']
                tra_stitch = pkey_stats['tra_stitched']
                tra_success = (100 * tra_stitch / tra_att) if tra_att > 0 else 0

                trb_att = pkey_stats['trb_attempted']
                trb_stitch = pkey_stats['trb_stitched']
                trb_success = (100 * trb_stitch / trb_att) if trb_att > 0 else 0

                # Only show if there were any TCR attempts
                if tra_att > 0 or trb_att > 0:
                    print(f"{pkey:<30} {tra_att:>10,} {tra_stitch:>12,} {tra_success:>9.1f}% {trb_att:>10,} {trb_stitch:>12,} {trb_success:>9.1f}%")

            print(f"{'-'*110}")

            # Summary stats
            total_tra_att = sum(s['tra_attempted'] for s in cumulative_tcr_stats['by_pkey'].values())
            total_tra_stitch = sum(s['tra_stitched'] for s in cumulative_tcr_stats['by_pkey'].values())
            total_trb_att = sum(s['trb_attempted'] for s in cumulative_tcr_stats['by_pkey'].values())
            total_trb_stitch = sum(s['trb_stitched'] for s in cumulative_tcr_stats['by_pkey'].values())

            print(f"{'TOTAL':<30} {total_tra_att:>10,} {total_tra_stitch:>12,} {100*total_tra_stitch/total_tra_att if total_tra_att > 0 else 0:>9.1f}% {total_trb_att:>10,} {total_trb_stitch:>12,} {100*total_trb_stitch/total_trb_att if total_trb_att > 0 else 0:>9.1f}%")

    # Create HuggingFace datasets from parquet chunks
    print(f"\n💾 Creating HuggingFace DatasetDict...")
    datasets_dict = {}
    
    for split_name in ['train', 'validation', 'test']:
        if split_counts[split_name] > 0:
            print(f"   Loading {split_name}: {split_counts[split_name]:,} rows from {len(split_writers[split_name])} chunks")
            # Load all chunks for this split
            split_dataset = Dataset.from_parquet([str(f) for f in split_writers[split_name]])
            datasets_dict[split_name] = split_dataset
    
    dataset_dict_obj = DatasetDict(datasets_dict)
    
=======

    from datasets import DatasetDict, concatenate_datasets, load_from_disk
    import gc

    if use_incremental_write:
        # Memory-efficient batched concatenation for each split

        # Collect all chunk files (both resumed and new)
        all_train_files = []
        all_val_files = []
        all_test_files = []

        # Add resumed chunks if exist
        if resume_state and resume_state['chunks_exist']:
            all_train_files.extend(resume_state['train_chunk_files'])
            all_val_files.extend(resume_state['val_chunk_files'])
            all_test_files.extend(resume_state['test_chunk_files'])
            print(f"📦 Including resumed chunks:")
            print(f"   Train: {len(resume_state['train_chunk_files'])}")
            print(f"   Val:   {len(resume_state['val_chunk_files'])}")
            print(f"   Test:  {len(resume_state['test_chunk_files'])}")

        # Add newly created chunks
        all_train_files.extend(train_chunk_files)
        all_val_files.extend(val_chunk_files)
        all_test_files.extend(test_chunk_files)

        print(f"\n📦 Total chunks to concatenate:")
        print(f"   Train: {len(all_train_files)}")
        print(f"   Val:   {len(all_val_files)}")
        print(f"   Test:  {len(all_test_files)}")

        # Concatenate train chunks (use batching due to size)
        print(f"\n🔗 Concatenating train chunks...")
        concat_start = time.time()
        train_dataset = concatenate_chunks_in_batches(all_train_files, batch_size=5, split_name="train")
        train_time = time.time() - concat_start
        print(f"   ✓ Train dataset: {len(train_dataset):,} rows ({train_time:.2f}s)")
        gc.collect()

        # Concatenate val chunks (can do all at once - smaller)
        print(f"\n🔗 Concatenating val chunks...")
        concat_start = time.time()
        val_datasets = [load_from_disk(str(p)) for p in all_val_files]
        val_dataset = concatenate_datasets(val_datasets)
        val_time = time.time() - concat_start
        print(f"   ✓ Val dataset: {len(val_dataset):,} rows ({val_time:.2f}s)")
        del val_datasets
        gc.collect()

        # Concatenate test chunks (can do all at once - smaller)
        print(f"\n🔗 Concatenating test chunks...")
        concat_start = time.time()
        test_datasets = [load_from_disk(str(p)) for p in all_test_files]
        test_dataset = concatenate_datasets(test_datasets)
        test_time = time.time() - concat_start
        print(f"   ✓ Test dataset: {len(test_dataset):,} rows ({test_time:.2f}s)")
        del test_datasets
        gc.collect()

        # Create final DatasetDict
        dataset_dict_obj = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })

        total_rows = len(train_dataset) + len(val_dataset) + len(test_dataset)
        print(f"\n✓ Split sizes:")
        print(f"  Train:      {len(train_dataset):,} ({len(train_dataset)/total_rows*100:.1f}%)")
        print(f"  Validation: {len(val_dataset):,} ({len(val_dataset)/total_rows*100:.1f}%)")
        print(f"  Test:       {len(test_dataset):,} ({len(test_dataset)/total_rows*100:.1f}%)")
        print(f"  Total:      {total_rows:,}")

        # Validate split balance
        if pkey_train_counts or pkey_val_counts or pkey_test_counts:
            validate_split_balance(
                pkey_train_counts,
                pkey_val_counts,
                pkey_test_counts,
                expected_train_ratio=1.0 - test_split - val_split,
                expected_val_ratio=val_split,
                expected_test_ratio=test_split,
                tolerance=0.05
            )

        # Clean up temporary split chunks
        print(f"\n🧹 Cleaning up temporary split chunk files...")
        import shutil
        for split_dir in [train_chunks_dir, val_chunks_dir, test_chunks_dir]:
            if split_dir.exists():
                for chunk_path in split_dir.iterdir():
                    if chunk_path.is_dir():
                        shutil.rmtree(chunk_path, ignore_errors=True)
                shutil.rmtree(split_dir, ignore_errors=True)
                print(f"   ✓ Cleaned {split_dir.name}")

        # Force garbage collection
        gc.collect()
    else:
        # Legacy mode: direct concatenation (high memory usage)
        print(f"⚠️  Warning: Legacy mode doesn't support split-during-tokenization")
        print(f"Concatenating {len(all_tokenized_chunks)} chunks into single dataset...")
        concat_start = time.time()
        full_dataset = concatenate_datasets(all_tokenized_chunks)
        concat_time = time.time() - concat_start
        print(f"   ⏱️  Concatenation time: {concat_time:.2f}s")
        print(f"✓ Concatenated dataset: {len(full_dataset):,} rows")

        # Perform splits the old way
        train_val_test_split = full_dataset.train_test_split(test_size=test_split, seed=42)
        val_ratio_adjusted = val_split / (1 - test_split)
        train_val_split = train_val_test_split['train'].train_test_split(test_size=val_ratio_adjusted, seed=42)

        dataset_dict_obj = DatasetDict({
            'train': train_val_split['train'],
            'validation': train_val_split['test'],
            'test': train_val_test_split['test']
        })

        # Clear chunks from memory
        del all_tokenized_chunks, full_dataset
        gc.collect()

>>>>>>> 2a62664 (Updated to incrementally write toeknized shards and implement stratified data splits)
    # Save HuggingFace dataset
    print(f"\n{'='*60}")
    print(f"💾 SAVING DATASET")
    print(f"{'='*60}")
    hf_output_dir = output_dir / "hf_dataset"
    print(f"Saving to {hf_output_dir}...")
    dataset_dict_obj.save_to_disk(str(hf_output_dir))
    print(f"✓ Saved HuggingFace dataset to {hf_output_dir}")

    # Save tokenizer
    print(f"\n💾 Saving tokenizer...")
    tokenizer.save(output_dir / "tokenizer")
    print(f"✓ Saved tokenizer to {output_dir / 'tokenizer'}")

    print(f"\n{'='*60}")
    print(f"✅ TOKENIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"HuggingFace dataset: {hf_output_dir}")
    print(f"Tokenizer: {output_dir / 'tokenizer'}")


def main():
    parser = argparse.ArgumentParser(description="Tokenize sequences for foundation model training")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory with parquet files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for tokenized data")
    parser.add_argument("--model-type", type=str, required=True,
                       choices=['protbert', 'bert', 'esm2', 'esm3', 'bpe', 'lstm', 'transformer'],
                       help="Model type for tokenization")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--vocab-size", type=int, default=1000, help="Vocabulary size for BPE")
    parser.add_argument("--sample", type=int, help="Sample N rows (optional)")
    parser.add_argument("--sample-mode", type=str, choices=['proportional', 'balanced'], 
                       help="Sampling mode: 'proportional' (maintain distribution) or 'balanced' (equal per permutation)")
    parser.add_argument("--oversample", action="store_true", 
                       help="When using balanced sampling, duplicate underrepresented samples to reach target count")
    parser.add_argument("--mode", type=str, 
                       choices=['mlm', 'tra', 'trb', 'tra_trb_pairing', 'tcr_mhc', 'peptide_mhc', 'specificity'],
                       help="Filter mode: only include permutation keys matching this mode")
    parser.add_argument("--train-bpe", action="store_true", help="Train BPE tokenizer (required for first run)")
    parser.add_argument("--test-split", type=float, default=0.1, help="Test split fraction")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split fraction")
<<<<<<< HEAD
    parser.add_argument("--chunk-size", type=int, default=10_000_000, help="Number of rows per chunk (default: 10M = ~15GB RAM)")
    parser.add_argument("--num-workers", type=int, default=16, help="Number of parallel workers for tokenization (default: 16, recommended: 8-24)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output (skip already processed rows)")
    parser.add_argument("--skip-tcr-stitching", action="store_true", help="Skip expensive TCR stitching/CDR identification (much faster)")
=======
    parser.add_argument("--chunk-size", type=int, default=50_000_000, help="Number of rows per chunk (default: 50M = ~7.5GB RAM)")
    parser.add_argument("--batch-size", type=int, default=50000, help="Batch size for tokenization (default: 50000, larger = faster but more RAM)")
    parser.add_argument("--num-workers", type=int, default=16, help="Number of parallel workers for tokenization (default: 16, reduced from 40 to prevent shared memory issues)")
    parser.add_argument("--precompute-workers", type=int, default=32, help="Number of parallel workers for sequence pre-computation (default: 32, reduced from 60 to prevent overhead)")
    parser.add_argument("--use-legacy-pipeline", action="store_true", help="Use legacy pipeline without optimizations (for debugging/comparison)")
    parser.add_argument("--incremental-write", action="store_true", default=True, help="Write chunks incrementally to avoid OOM (default: True)")
    parser.add_argument("--no-incremental-write", action="store_false", dest="incremental_write", help="Disable incremental writing (may cause OOM on large datasets)")
    parser.add_argument("--use-fast-tokenizer", action="store_true", default=True,
                       help="Use fast Rust-based tokenizers where available (default: True, 2-10x faster for BERT/ProtBERT)")
    parser.add_argument("--no-fast-tokenizer", action="store_false", dest="use_fast_tokenizer",
                       help="Disable fast tokenizers (use Python implementation)")
    parser.add_argument("--stratified-split", action="store_true", default=True,
                       help="Use stratified split to maintain class balance (default: True)")
    parser.add_argument("--random-split", action="store_false", dest="stratified_split",
                       help="Use random split instead of stratified split")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from existing split chunk directories (skip already-completed chunks)")
>>>>>>> 2a62664 (Updated to incrementally write toeknized shards and implement stratified data splits)

    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)
    
    tokenize_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        model_type=args.model_type,
        max_length=args.max_length,
        vocab_size=args.vocab_size,
        sample=args.sample,
        sample_mode=args.sample_mode,
        train_bpe=args.train_bpe,
        test_split=args.test_split,
        val_split=args.val_split,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        precompute_workers=0 if args.use_legacy_pipeline else args.precompute_workers,
        use_legacy_pipeline=args.use_legacy_pipeline,
        use_incremental_write=args.incremental_write,
        oversample=args.oversample,
        mode=args.mode,
<<<<<<< HEAD
        resume=args.resume,
        skip_tcr_stitching=args.skip_tcr_stitching,
=======
        use_fast_tokenizer=args.use_fast_tokenizer,
        stratified_split=args.stratified_split,
        resume=args.resume,
>>>>>>> 2a62664 (Updated to incrementally write toeknized shards and implement stratified data splits)
    )


if __name__ == "__main__":
    main()
