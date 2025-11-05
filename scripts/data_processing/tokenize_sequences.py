#!/usr/bin/env python3
"""
Tokenize deduplicated/permuted sequences for foundation model training.
Supports: ProtBERT, BERT, ESM-2, ESM-3, and custom BPE for LSTM/Transformer.
"""

import argparse
import json
import os
import sys
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
    from esm.tokenization import get_model_tokenizers
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
    
    def __init__(self, max_length: int = 512):
        super().__init__(max_length)
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers required for ProtBERT")
        
        # Use pre-trained ProtBERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "Rostlab/prot_bert",
            do_lower_case=False
        )
        print(f"✓ Loaded ProtBERT tokenizer (vocab size: {len(self.tokenizer)})")
    
    def _add_spaces(self, seq: str) -> str:
        """Add spaces between amino acids for ProtBERT."""
        return " ".join(list(seq))
    
    def tokenize_batch(self, sequences: List[str]) -> Dict[str, List[List[int]]]:
        """Tokenize sequences with spaces between amino acids."""
        # Check if sequences already have [CLS] token (already spaced)
        # If so, don't add spaces again
        spaced_seqs = []
        for seq in sequences:
            if seq.startswith('[CLS]'):
                # Already formatted with [CLS] and [SEP] tokens and spaces
                spaced_seqs.append(seq)
            else:
                # Add spaces between amino acids
                spaced_seqs.append(self._add_spaces(seq))
        
        # Tokenize
        encoded = self.tokenizer(
            spaced_seqs,
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
    
    def __init__(self, max_length: int = 512, model_name: str = "facebook/esm2_t6_8M_UR50D"):
        super().__init__(max_length)
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers required for ESM-2")
        
        # Use ESM-2 tokenizer
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        print(f"✓ Loaded ESM-2 tokenizer from {model_name} (vocab size: {len(self.tokenizer)})")
    
    def tokenize_batch(self, sequences: List[str]) -> Dict[str, List[List[int]]]:
        """Tokenize sequences directly (no spaces)."""
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
                self.tokenizers = get_model_tokenizers(model_name)
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
            # Use ESM-2 tokenizer (HuggingFace format)
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
                    self.tokenizers = get_model_tokenizers("esm3_sm_open_v1")
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


def _count_permutations_in_file(file_path: Path) -> tuple[Dict[str, int], int]:
    """
    Count permutations in a single parquet file.
    Used for parallel processing.
    """
    permutation_counts = {}
    total_rows = 0
    
    # Read parquet file
    table = pq.read_table(file_path)
    df = table.to_pandas()
    rows = df.to_dict('records')
    
    for row in rows:
        perm = get_permutation_signature(row)
        permutation_counts[perm] = permutation_counts.get(perm, 0) + 1
        total_rows += 1
    
    return permutation_counts, total_rows


def analyze_dataset_distribution(input_dir: Path, chunk_size: int = 10_000_000, num_workers: int = 16):
    """
    Analyze the distribution of permutations in the dataset by streaming through ALL data.
    Uses parallel processing to speed up analysis.
    Returns dict of permutation -> count and total rows.
    """
    print(f"\n📊 Analyzing full dataset distribution (parallel processing with {num_workers} workers)...")
    
    parquet_files = sorted(input_dir.glob("*.parquet"))
    print(f"   Found {len(parquet_files)} parquet files to analyze")
    
    # Process files in parallel
    from multiprocessing import Pool
    
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(_count_permutations_in_file, parquet_files),
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
                         sample_mode: Optional[str] = None, num_workers: int = 16):
    """
    Stream parquet files in chunks to avoid loading everything into memory.
    
    Args:
        input_dir: Directory containing parquet files
        sample: If set, only yield first N rows total
        chunk_size: Number of rows per chunk (default 10M = ~15GB RAM per chunk)
        sample_mode: Sampling strategy - 'proportional' or 'balanced' or None
        num_workers: Number of parallel workers for distribution analysis (default: 16)
    
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
        # Use same num_workers for distribution analysis (passed from outer scope)
        dist_counts, total_analyzed = analyze_dataset_distribution(
            input_dir, chunk_size, num_workers=min(num_workers, len(list(input_dir.glob("*.parquet"))))
        )
        
        if sample_mode == 'proportional':
            # Maintain original distribution
            print(f"\n📐 Proportional sampling: maintaining original distribution")
            permutation_targets = {
                perm: int(sample * count / total_analyzed)
                for perm, count in dist_counts.items()
            }
        elif sample_mode == 'balanced':
            # Equal samples per permutation
            print(f"\n⚖️  Balanced sampling: equal samples per permutation")
            n_permutations = len(dist_counts)
            per_perm = sample // n_permutations
            permutation_targets = {perm: per_perm for perm in dist_counts.keys()}
            print(f"   Target: {per_perm:,} samples per permutation")
        
        print(f"\n🎯 Sampling targets:")
        for perm, target in sorted(permutation_targets.items(), key=lambda x: -x[1]):
            print(f"   {perm:30s}: {target:10,}")
        
        # Initialize counters
        permutation_counts = {perm: 0 for perm in permutation_targets.keys()}
    
    total_rows = 0
    chunk_buffer = []
    
    for pf in tqdm(parquet_files, desc="Processing parquet files"):
        table = pq.read_table(pf)
        df = table.to_pandas()
        rows = df.to_dict('records')
        
        for row in rows:
            # Check if we should include this row based on sampling mode
            should_include = True
            
            if sample and sample_mode and permutation_targets:
                perm = get_permutation_signature(row)
                if perm in permutation_counts:
                    if permutation_counts[perm] >= permutation_targets[perm]:
                        should_include = False
                    else:
                        permutation_counts[perm] += 1
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
    
    # Print final sampling stats
    if sample and sample_mode and permutation_targets:
        print(f"\n📊 Final sampling statistics:")
        for perm in sorted(permutation_counts.keys()):
            achieved = permutation_counts[perm]
            target = permutation_targets[perm]
            pct = 100 * achieved / target if target > 0 else 0
            print(f"   {perm:30s}: {achieved:10,} / {target:10,} ({pct:5.1f}%)")


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
    
    For ProtBERT/BERT: adds [CLS] at start and [SEP] between molecules
    For other models: uses space separator between molecules
    """
    # New format: has 'sequence' column with molecules separated by single space
    if 'sequence' in row and row.get('sequence'):
        sequence_str = str(row['sequence'])
        if sequence_str and sequence_str != 'nan' and sequence_str != '':
            # Split by space to get individual molecules
            # Each molecule is a contiguous amino acid string (e.g., "CASSLGQAYEQYF")
            molecules = sequence_str.split()
            
            if model_type in ['protbert', 'bert']:
                # ProtBERT/BERT: [CLS] at start, [SEP] between molecules
                # Add spaces between amino acids for each molecule
                spaced_molecules = [" ".join(list(mol)) for mol in molecules]
                # Join with [SEP] token
                return "[CLS] " + " [SEP] ".join(spaced_molecules)
            else:
                # Other models: keep molecules without internal spacing
                # Just join with spaces
                return " ".join(molecules)
    
    # Old format: individual molecule columns
    sequences = []
    for field in ['tra', 'trb', 'peptide', 'mhc_one', 'mhc_two']:
        val = row.get(field, '')
        if val and str(val) != 'nan' and str(val) != '':
            sequences.append(str(val))
    
    # Concatenate based on model type
    if model_type in ['protbert', 'bert']:
        # ProtBERT/BERT: [CLS] at start, [SEP] between molecules
        # Add spaces between amino acids for each molecule
        spaced_molecules = [" ".join(list(seq)) for seq in sequences]
        # Join with [SEP] token
        return "[CLS] " + " [SEP] ".join(spaced_molecules)
    else:
        # Other models: simple space separator
        return " ".join(sequences)


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
    chunk_size: int = 10_000_000,
    num_workers: int = 50,
):
    """
    Main tokenization function with streaming support.
    
    Args:
        chunk_size: Rows per chunk (default 10M = ~15GB RAM). Max ~40M for 600GB buffer.
        num_workers: Parallel workers for Dataset.map()
        sample: Number of sequences to sample (optional)
        sample_mode: Sampling strategy - 'proportional' (maintain distribution) or 'balanced' (equal per permutation)
    """
    print(f"\n{'='*60}")
    print(f"TOKENIZATION: {model_type.upper()}")
    print(f"{'='*60}\n")
    print(f"⚙️  Chunk size: {chunk_size:,} rows")
    print(f"⚙️  Batch size: 1000 sequences per tokenization call")
    print(f"⚙️  Workers: {num_workers}")
    print(f"⚙️  Max RAM usage: ~{chunk_size * 150 / 1e9:.0f}GB per chunk")
    if sample:
        print(f"⚙️  Sampling: {sample:,} sequences (mode: {sample_mode or 'first-N'})")
    print()
    
    # Create tokenizer
    if model_type == "protbert":
        tokenizer = ProtBERTTokenizer(max_length)
    elif model_type == "bert":
        tokenizer = ProtBERTTokenizer(max_length)
    elif model_type == "esm2":
        tokenizer = ESM2Tokenizer(max_length)
    elif model_type == "esm3":
        tokenizer = ESM3Tokenizer(max_length)
    elif model_type in ["bpe", "lstm", "transformer"]:
        tokenizer = BPETokenizer(max_length, vocab_size)
        
        # Train BPE if requested
        if train_bpe:
            print("\n🔧 Training BPE tokenizer (streaming mode)...")
            all_sequences = []
            sequence_count = 0
            max_training_sequences = 1_000_000  # Limit for BPE training
            
            for chunk_rows, _ in stream_parquet_files(input_dir, sample, chunk_size, sample_mode, num_workers):
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
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'validation', 'test']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Initialize counters
    split_writers = {
        'train': [],
        'validation': [],
        'test': []
    }
    split_counts = {'train': 0, 'validation': 0, 'test': 0}
    
    # Process in streaming chunks
    print(f"\n📦 Processing dataset in chunks...")
    np.random.seed(42)
    
    chunk_num = 0
    for chunk_rows, total_rows in stream_parquet_files(input_dir, sample, chunk_size, sample_mode, num_workers):
        chunk_num += 1
        print(f"\n📦 Processing chunk {chunk_num} ({len(chunk_rows):,} rows, total: {total_rows:,})")
        
        # Random split assignment for this chunk
        chunk_splits = np.random.choice(
            ['train', 'validation', 'test'],
            size=len(chunk_rows),
            p=[1 - test_split - val_split, val_split, test_split]
        )
        
        # Convert to HuggingFace Dataset for parallel tokenization
        print(f"   ⚡ Creating Dataset...")
        chunk_dataset = Dataset.from_list(chunk_rows)
        
        # Tokenize using Dataset.map with multiprocessing
        print(f"   ⚡ Tokenizing with {num_workers} workers...")
        tokenize_fn = create_tokenize_function(tokenizer, model_type)
        tokenized_dataset = chunk_dataset.map(
            tokenize_fn,
            batched=True,
            batch_size=1000,
            num_proc=num_workers,
            remove_columns=chunk_dataset.column_names,
            desc="   Tokenizing"
        )
        
        # Add back original columns (handle both new and old formats)
        if 'permutation_key' in chunk_dataset.column_names:
            # New format: add back permutation_key
            tokenized_dataset = tokenized_dataset.add_column('permutation_key', chunk_dataset['permutation_key'])
        else:
            # Old format: add back individual molecule columns
            for col in ['tra', 'trb', 'peptide', 'mhc_one', 'mhc_two']:
                if col in chunk_dataset.column_names:
                    tokenized_dataset = tokenized_dataset.add_column(col, chunk_dataset[col])
        
        # Convert to list for splitting
        tokenized_rows = tokenized_dataset.to_pandas().to_dict('records')
        
        # Split and write
        for split_name in ['train', 'validation', 'test']:
            split_mask = chunk_splits == split_name
            split_data = [tokenized_rows[i] for i in range(len(tokenized_rows)) if split_mask[i]]
            
            if split_data:
                # Write chunk to parquet
                df = pa.Table.from_pylist(split_data)
                chunk_file = output_dir / split_name / f"chunk_{split_counts[split_name]:06d}.parquet"
                pq.write_table(df, chunk_file)
                
                split_counts[split_name] += len(split_data)
                split_writers[split_name].append(chunk_file)
        
        print(f"   ✓ Chunk {chunk_num} complete")
        print(f"   📊 Cumulative: Train={split_counts['train']:,}, "
              f"Val={split_counts['validation']:,}, Test={split_counts['test']:,}")
    
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
    
    # Save HuggingFace dataset
    hf_output_dir = output_dir / "hf_dataset"
    dataset_dict_obj.save_to_disk(str(hf_output_dir))
    print(f"✓ Saved HuggingFace dataset to {hf_output_dir}")
    
    # Save tokenizer
    print(f"\n💾 Saving tokenizer...")
    tokenizer.save(output_dir / "tokenizer")
    
    print(f"\n{'='*60}")
    print(f"✅ TOKENIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Tokenizer saved to: {output_dir / 'tokenizer'}")


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
    parser.add_argument("--train-bpe", action="store_true", help="Train BPE tokenizer (required for first run)")
    parser.add_argument("--test-split", type=float, default=0.1, help="Test split fraction")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--chunk-size", type=int, default=10_000_000, help="Number of rows per chunk (default: 10M = ~15GB RAM)")
    parser.add_argument("--num-workers", type=int, default=16, help="Number of parallel workers for tokenization (default: 16, recommended: 8-24)")
    
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
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
