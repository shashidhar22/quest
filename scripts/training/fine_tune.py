#!/usr/bin/env python3
"""
fine_tune.py
────────────────────────────────────────────────────────
Unified fine-tuning script with task-specific data filtering and masking strategies.
Uses HuggingFace Accelerate for distributed training and PEFT for LoRA.

DATA LOADING MODES:

1. Pre-tokenized (original):
   Uses HuggingFace dataset prepared by tokenize_sequences.py
   Usage: --dataset_path /path/to/hf_dataset
   Pros: Fastest training (tokenization already done)
   Cons: High disk usage (stores raw + tokenized data)

2. On-the-fly tokenization (NEW):
   Loads raw parquet files and tokenizes during training
   Usage: --raw_data_dir /path/to/parquet --tokenizer_type esm2
   Pros:
   - Memory efficient: No duplicate storage
   - 48% disk space savings
   - First epoch tokenizes and caches to disk
   - Subsequent epochs read from cache (fast)
   - Supports CDR identification for full_tra/full_trb modes
   Cons:
   - First epoch ~10-30min slower (one-time cost)
   - NO TCR stitching (use tokenize_sequences.py if needed)

   Example:
   python fine_tune.py \
       --raw_data_dir /data/raw_parquet \
       --tokenizer_type esm2 \
       --model_path facebook/esm2_t12_35M_UR50D \
       --mode mlm \
       --batch_size 32 \
       --fp16

FINE-TUNING MODES:

Supports multiple fine-tuning modes:
1. MLM: Standard masked language modeling (15% random masking)
2. TRA/TRB: Mask middle 5 amino acids of CDR3 region
3. TRA-TRB Pairing: Mask entire first chain
4. TCR-MHC: Mask first molecule (or first two if both TCR chains)
5. Peptide-MHC: Mask first molecule (or both MHC chains if first two)
6. Specificity: Mask first molecule in complete TCR complexes
"""

import argparse
import json
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import wandb
import evaluate
from collections import Counter
from datasets import load_from_disk, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from transformers.modeling_outputs import MaskedLMOutput
from peft import get_peft_model, LoraConfig, PeftModel
from accelerate import Accelerator
from typing import Any, Dict, List, Optional, Tuple
from tqdm.auto import tqdm

# Try to import ESM3 package
try:
    from esm.pretrained import (
        ESM3_sm_open_v0,
        ESM3_structure_encoder_v0,
        ESM3_structure_decoder_v0,
    )
    from esm.sdk.api import ESMProtein, ESM3InferenceClient
    ESM3_AVAILABLE = True
except ImportError:
    ESM3_AVAILABLE = False
    print("⚠️  ESM3 package not available. Install with: pip install esm")

# Try to import CDR region identifier
try:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from parsers.cdr_region_identifier import CDRRegionIdentifier
    CDR_IDENTIFIER_AVAILABLE = True
except ImportError:
    CDR_IDENTIFIER_AVAILABLE = False
    print("⚠️  CDRRegionIdentifier not available")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ═══════════════════════════════════════════════════════════════════════════════
# EARLY CACHE CONFIGURATION FOR SAGEMAKER
# ═══════════════════════════════════════════════════════════════════════════════
# Set HuggingFace cache to EBS volume if in SageMaker (before any HF operations!)
# This prevents the 30GB root filesystem from filling up
if os.environ.get('SM_MODEL_DIR'):  # SageMaker environment
    sagemaker_cache = "/opt/ml/input/data/training/.huggingface_cache"
    sagemaker_tmp = "/opt/ml/input/data/training/.tmp"

    # Create directories
    os.makedirs(sagemaker_cache, exist_ok=True)
    os.makedirs(sagemaker_tmp, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════════
    # FORCE override ALL cache and temporary file locations
    # ═══════════════════════════════════════════════════════════════════════

    # HuggingFace cache locations
    os.environ['HF_HOME'] = sagemaker_cache
    os.environ['HF_DATASETS_CACHE'] = os.path.join(sagemaker_cache, 'datasets')
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(sagemaker_cache, 'transformers')
    os.environ['HF_HUB_CACHE'] = os.path.join(sagemaker_cache, 'hub')
    os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(sagemaker_cache, 'hub')

    # PyTorch cache locations
    os.environ['PYTORCH_KERNEL_CACHE_PATH'] = os.path.join(sagemaker_cache, 'torch_kernels')
    os.environ['TORCH_HOME'] = os.path.join(sagemaker_cache, 'torch')

    # ═══════════════════════════════════════════════════════════════════════
    # CRITICAL: Redirect ALL temporary files to EBS volume
    # ═══════════════════════════════════════════════════════════════════════
    # PyArrow and HuggingFace create temporary files during parquet processing
    # By default these go to /tmp (on root filesystem) which fills up quickly
    os.environ['TMPDIR'] = sagemaker_tmp
    os.environ['TEMP'] = sagemaker_tmp
    os.environ['TMP'] = sagemaker_tmp
    os.environ['TEMPDIR'] = sagemaker_tmp

    # Arrow-specific temp directory
    os.environ['ARROW_TMPDIR'] = sagemaker_tmp

    # Python tempfile module will use TMPDIR
    import tempfile
    tempfile.tempdir = sagemaker_tmp

    print(f"🔧 FORCED cache override for SageMaker:")
    print(f"   HF_HOME: {os.environ['HF_HOME']}")
    print(f"   HF_DATASETS_CACHE: {os.environ['HF_DATASETS_CACHE']}")
    print(f"   TRANSFORMERS_CACHE: {os.environ['TRANSFORMERS_CACHE']}")
    print(f"   TMPDIR: {os.environ['TMPDIR']} ⚠️  CRITICAL for Arrow/Parquet")

    # Create all cache subdirectories
    for subdir in ['datasets', 'transformers', 'hub', 'torch_kernels', 'torch']:
        os.makedirs(os.path.join(sagemaker_cache, subdir), exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SAGEMAKER ENVIRONMENT DETECTION AND PATH RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

def is_sagemaker_environment() -> bool:
    """
    Detect if running in AWS SageMaker training environment.

    Returns:
        True if running in SageMaker, False otherwise
    """
    return os.environ.get('SM_MODEL_DIR') is not None


def get_sagemaker_paths() -> Dict[str, Any]:
    """
    Get SageMaker standard paths and environment metadata.

    Returns:
        Dictionary with SageMaker paths and metadata, or empty dict if not in SageMaker
    """
    if not is_sagemaker_environment():
        return {}

    return {
        'model_dir': os.environ.get('SM_MODEL_DIR', '/opt/ml/model'),
        'training_dir': os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'),
        'validation_dir': os.environ.get('SM_CHANNEL_VALIDATION',
                                        os.environ.get('SM_CHANNEL_TRAINING')),
        'output_dir': os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'),
        'checkpoint_dir': '/opt/ml/checkpoints',
        'num_gpus': int(os.environ.get('SM_NUM_GPUS', '1')),
        'hosts': os.environ.get('SM_HOSTS', '').split(',') if os.environ.get('SM_HOSTS') else [],
        'current_host': os.environ.get('SM_CURRENT_HOST', 'algo-1'),
    }


def resolve_path_for_environment(user_path: str, path_type: str = 'dataset') -> str:
    """
    Resolve user-provided path to appropriate environment location.

    In local environments, returns the user path unchanged.
    In SageMaker, returns standard SageMaker paths based on path_type.

    Args:
        user_path: Path provided by user (may be placeholder in SageMaker)
        path_type: Type of path - 'dataset', 'output', or 'checkpoint'

    Returns:
        Resolved path appropriate for current environment
    """
    if not is_sagemaker_environment():
        return user_path

    sm_paths = get_sagemaker_paths()

    if path_type == 'dataset':
        return sm_paths['training_dir']
    elif path_type == 'output':
        return sm_paths['model_dir']
    elif path_type == 'checkpoint':
        return sm_paths['checkpoint_dir']

    return user_path


# ═══════════════════════════════════════════════════════════════════════════════
# ESM3 MODEL WRAPPER FOR HUGGINGFACE TRAINER COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

class ESM3ForMaskedLM(nn.Module):
    """
    Wrapper around ESM3 model to make it compatible with HuggingFace Trainer.
    
    ESM3's forward() doesn't accept 'labels' argument, but Trainer expects it.
    This wrapper handles the labels and computes the MLM loss.
    """
    
    def __init__(self, esm3_model):
        super().__init__()
        self.esm3 = esm3_model
        
        # Create a proper config object compatible with PEFT
        class ESM3Config:
            """Config object for ESM3 model compatible with PEFT"""
            def __init__(self):
                self.vocab_size = 64
                self.hidden_size = 1536
                self.num_hidden_layers = 48
                self.num_attention_heads = 24
                self.intermediate_size = 8192
                self.tie_word_embeddings = False
                self.model_type = "esm3"
                
            def get(self, key, default=None):
                """Dict-like get method for PEFT compatibility"""
                return getattr(self, key, default)
            
            def __getitem__(self, key):
                """Dict-like indexing for PEFT compatibility"""
                return getattr(self, key)
            
            def to_dict(self):
                """Convert config to dictionary for W&B and other integrations"""
                return {
                    'vocab_size': self.vocab_size,
                    'hidden_size': self.hidden_size,
                    'num_hidden_layers': self.num_hidden_layers,
                    'num_attention_heads': self.num_attention_heads,
                    'intermediate_size': self.intermediate_size,
                    'tie_word_embeddings': self.tie_word_embeddings,
                    'model_type': self.model_type,
                }
        
        self.config = ESM3Config()
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        """
        Forward pass compatible with HuggingFace Trainer.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels for MLM (-100 for non-masked positions)
            **kwargs: Other arguments (ignored)
        
        Returns:
            Dictionary with 'loss' and 'logits'
        """
        # Prepare inputs for ESM3
        # ESM3 expects sequence_tokens as input
        from esm.sdk.api import ESMProteinTensor
        
        # Convert input_ids to ESM3 format
        batch_size, seq_len = input_ids.shape
        
        # ESM3 forward expects ESMProteinTensor with sequence field
        # For simplicity, we'll call the model directly with sequence tokens
        outputs = self.esm3(
            sequence_tokens=input_ids,
        )
        
        # Get sequence logits from outputs
        logits = outputs.sequence_logits
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Flatten for loss computation
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
        
        # Return a proper MaskedLMOutput which is both dict-like and indexable (Trainer expects outputs[0] for loss)
        return MaskedLMOutput(loss=loss, logits=logits)
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing if supported"""
        if hasattr(self.esm3, 'gradient_checkpointing_enable'):
            self.esm3.gradient_checkpointing_enable()
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped ESM3 model"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.esm3, name)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA FILTERING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def filter_dataset_by_mode(dataset: Any, mode: str) -> Any:
    """
    Filter dataset based on the fine-tuning mode.

    Args:
        dataset: HuggingFace dataset with 'permutation_key' field
        mode: One of ['mlm', 'tra', 'trb', 'full_tra', 'full_trb', 'tra_trb_pairing',
                      'tcr_mhc', 'peptide_mhc', 'specificity']
              - tcr_mhc: At least one TCR chain AND at least one MHC, excluding examples with peptide
              - peptide_mhc: Peptide AND at least one MHC, excluding examples with TCR chains

    Returns:
        Filtered dataset
    """
    print(f"Filtering dataset for mode: {mode}")
    original_size = len(dataset)

    if mode == "mlm":
        # Use all data
        filtered = dataset

    elif mode in ["tra", "trb"]:
        # Only sequences with the specific chain type
        filtered = dataset.filter(lambda x: x["permutation_key"] == mode)

    elif mode == "full_tra":
        # TRA sequences with full-length sequence available (NEW)
        filtered = dataset.filter(lambda x: (
            x["permutation_key"] == "tra" and
            x.get("tra_full", "") != ""
        ))

    elif mode == "full_trb":
        # TRB sequences with full-length sequence available (NEW)
        filtered = dataset.filter(lambda x: (
            x["permutation_key"] == "trb" and
            x.get("trb_full", "") != ""
        ))

    elif mode == "tra_trb_pairing":
        # Examples containing both TRA and TRB
        filtered = dataset.filter(lambda x: "tra" in x["permutation_key"] and "trb" in x["permutation_key"])

    elif mode == "tcr_mhc":
        # At least one TCR chain (tra/trb) AND at least one MHC (mhc_one/mhc_two), NO peptide
        filtered = dataset.filter(lambda x: (
            ("tra" in x["permutation_key"] or "trb" in x["permutation_key"]) and
            ("mhc_one" in x["permutation_key"] or "mhc_two" in x["permutation_key"]) and
            "peptide" not in x["permutation_key"]
        ))

    elif mode == "peptide_mhc":
        # Peptide AND at least one MHC, NO TCR chains
        filtered = dataset.filter(lambda x: (
            "peptide" in x["permutation_key"] and
            ("mhc_one" in x["permutation_key"] or "mhc_two" in x["permutation_key"]) and
            "tra" not in x["permutation_key"] and
            "trb" not in x["permutation_key"]
        ))

    elif mode == "specificity":
        # At least one TCR chain, peptide, and at least one MHC
        filtered = dataset.filter(lambda x: (
            ("tra" in x["permutation_key"] or "trb" in x["permutation_key"]) and
            "peptide" in x["permutation_key"] and
            ("mhc_one" in x["permutation_key"] or "mhc_two" in x["permutation_key"])
        ))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    filtered_size = len(filtered)
    print(f"Filtered: {original_size:,} -> {filtered_size:,} examples ({filtered_size/original_size*100:.1f}% retained)")

    return filtered


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MASKING STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════════

class TaskSpecificMaskingCollator:
    """
    Custom data collator that applies task-specific masking strategies.
    """
    
    def __init__(
        self,
        tokenizer: Any,
        mode: str = "mlm",
        mlm_probability: float = 0.15,
        cdr3_mask_length: int = 5,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            mode: Masking strategy mode (includes new 'full_tra', 'full_trb')
            mlm_probability: Probability for random MLM masking
            cdr3_mask_length: Number of amino acids to mask in CDR3 region
        """
        self.tokenizer = tokenizer
        self.mode = mode.lower()
        self.mlm_probability = mlm_probability
        self.cdr3_mask_length = cdr3_mask_length

        # Token IDs
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.cls_token_id = tokenizer.cls_token_id if hasattr(tokenizer, 'cls_token_id') else None
        self.sep_token_id = tokenizer.sep_token_id if hasattr(tokenizer, 'sep_token_id') else None

        # Cache vocabulary size (Phase 5 optimization)
        self.vocab_size = len(tokenizer)

        # Pre-compute special token IDs for fast O(1) lookup (Phase 1 optimization)
        self.special_token_ids = self._build_special_token_set()

        # Pre-compute separator token IDs for fast lookup (Phase 2 optimization)
        self.separator_token_ids = self._build_separator_token_set()

        # Initialize CDR identifier for full_tra/full_trb modes (NEW)
        if mode in ['full_tra', 'full_trb'] and CDR_IDENTIFIER_AVAILABLE:
            self.cdr_identifier = CDRRegionIdentifier()
            print(f"✓ Initialized CDR identifier for mode: {mode}")
        else:
            self.cdr_identifier = None
        
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Apply masking and create batch."""
        
        # Extract input_ids
        input_ids_list = [f["input_ids"] for f in features]
        permutation_keys = [f.get("permutation_key", "") for f in features]
        
        # Apply mode-specific masking
        if self.mode == "mlm":
            masked_inputs, labels = self._mask_mlm(input_ids_list)
        elif self.mode in ["tra", "trb"]:
            masked_inputs, labels = self._mask_cdr3_middle(input_ids_list, permutation_keys)
        elif self.mode == "full_tra":  # NEW
            masked_inputs, labels = self._mask_cdr_regions_tra(input_ids_list, features)
        elif self.mode == "full_trb":  # NEW
            masked_inputs, labels = self._mask_cdr_regions_trb(input_ids_list, features)
        elif self.mode == "tra_trb_pairing":
            masked_inputs, labels = self._mask_first_chain(input_ids_list)
        elif self.mode == "tcr_mhc":
            masked_inputs, labels = self._mask_first_molecules_tcr_mhc(input_ids_list, permutation_keys)
        elif self.mode == "peptide_mhc":
            masked_inputs, labels = self._mask_first_molecules_peptide_mhc(input_ids_list, permutation_keys)
        elif self.mode == "specificity":
            masked_inputs, labels = self._mask_first_molecule(input_ids_list)
        else:
            raise ValueError(f"Unknown masking mode: {self.mode}")
        
        # Convert to tensors and pad
        batch = self._create_batch(masked_inputs, labels)
        return batch

    def _build_special_token_set(self) -> set:
        """
        Build set of special token IDs for O(1) lookup.
        Phase 1 optimization: Pre-compute instead of decode() in hot loop.
        """
        special_tokens = set()

        # Add known special tokens from tokenizer attributes
        if self.mask_token_id is not None:
            special_tokens.add(self.mask_token_id)
        if self.pad_token_id is not None:
            special_tokens.add(self.pad_token_id)
        if self.cls_token_id is not None:
            special_tokens.add(self.cls_token_id)
        if self.sep_token_id is not None:
            special_tokens.add(self.sep_token_id)

        # Add other common special tokens by encoding
        for token_str in ['[CLS]', '[SEP]', '[PAD]', '[MASK]', '[UNK]',
                          '<s>', '</s>', '<pad>', '<unk>', '<mask>',
                          '<cls>', '<sep>', '<eos>']:
            try:
                token_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
                if len(token_ids) == 1:
                    special_tokens.add(token_ids[0])
            except:
                pass

        return special_tokens

    def _build_separator_token_set(self) -> set:
        """
        Build set of separator token IDs for fast lookup.
        Phase 2 optimization: Pre-compute instead of decode() in loops.
        """
        separator_tokens = set()

        # Common separator tokens used in this codebase
        for token_str in ['[SEP]', '[ETRA]', '[ETRB]', '[EPEP]', '[EMHO]', '[EMHT]']:
            try:
                token_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
                if len(token_ids) == 1:
                    separator_tokens.add(token_ids[0])
            except:
                pass

        return separator_tokens

    def _is_special_token(self, token_id: int) -> bool:
        """
        Check if token is a special token using pre-computed set.
        Phase 1 optimization: O(1) set lookup instead of decode() calls.
        """
        return token_id in self.special_token_ids
    
    def _find_sequence_boundaries(self, input_ids: List[int]) -> Tuple[int, int]:
        """Find start and end indices of actual sequence (excluding special tokens)."""
        seq_start = 0
        seq_end = len(input_ids)
        
        # Find start (skip special tokens at beginning)
        for i, token_id in enumerate(input_ids):
            if not self._is_special_token(token_id):
                seq_start = i
                break
        
        # Find end (before padding/special tokens at end)
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] != self.pad_token_id and not self._is_special_token(input_ids[i]):
                seq_end = i + 1
                break
        
        return seq_start, seq_end
    
    def _find_separators(self, input_ids: List[int]) -> List[int]:
        """
        Find positions of separator tokens using pre-computed token IDs.
        Phase 2 optimization: Token ID comparison instead of decode() calls.
        """
        return [i for i, token_id in enumerate(input_ids)
                if token_id in self.separator_token_ids]

    def _apply_vectorized_masking(
        self,
        input_ids: List[int],
        label_ids: List[int],
        start_idx: int,
        end_idx: int,
        mask_probability: float = None
    ) -> None:
        """
        Apply MLM masking to a range of tokens using vectorized RNG.
        Phase 4 optimization: Generate all random numbers at once for speed.

        Args:
            input_ids: Token IDs to modify in-place
            label_ids: Label IDs to modify in-place
            start_idx: Start of masking range (inclusive)
            end_idx: End of masking range (exclusive)
            mask_probability: Probability of masking each token (default: self.mlm_probability)
        """
        if mask_probability is None:
            mask_probability = self.mlm_probability

        # Collect maskable token indices in the range
        maskable_indices = [i for i in range(start_idx, end_idx)
                          if i < len(input_ids) and not self._is_special_token(input_ids[i])]

        if not maskable_indices:
            return

        # Generate all random numbers at once (FAST!)
        n_maskable = len(maskable_indices)
        mask_probs = np.random.random(n_maskable)
        action_probs = np.random.random(n_maskable)
        random_tokens = np.random.randint(0, self.vocab_size, n_maskable)

        # Apply masking with vectorized probabilities
        for idx, i in enumerate(maskable_indices):
            if mask_probs[idx] < mask_probability:
                label_ids[i] = input_ids[i]

                # 80% mask, 10% random, 10% keep
                if action_probs[idx] < 0.8:
                    input_ids[i] = self.mask_token_id
                elif action_probs[idx] < 0.9:
                    input_ids[i] = int(random_tokens[idx])
                # else: keep original (10% of masked tokens)

    # ─────────────────────────────────────────────────────────────────────────────
    # MLM Masking (15% random)
    # ─────────────────────────────────────────────────────────────────────────────
    
    def _mask_mlm(self, input_ids_list: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Standard MLM masking: 15% of tokens randomly.
        Phase 4 optimization: Uses vectorized RNG helper for faster masking.
        """
        masked_inputs = []
        labels = []

        for input_ids in input_ids_list:
            input_ids = input_ids.copy() if isinstance(input_ids, list) else input_ids.tolist()
            label_ids = [-100] * len(input_ids)

            # Apply vectorized masking to entire sequence
            self._apply_vectorized_masking(input_ids, label_ids, 0, len(input_ids))

            masked_inputs.append(input_ids)
            labels.append(label_ids)

        return masked_inputs, labels
    
    # ─────────────────────────────────────────────────────────────────────────────
    # TRA/TRB CDR3 Masking (middle 5 amino acids)
    # ─────────────────────────────────────────────────────────────────────────────
    
    def _mask_cdr3_middle(
        self,
        input_ids_list: List[List[int]],
        permutation_keys: List[str]
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Mask a percentage (mlm_probability) of the middle portion of CDR3 region.
        Instead of masking all tokens in the middle region, we mask mlm_probability% of them
        randomly (similar to standard MLM but restricted to the middle region).
        Phase 4 optimization: Uses vectorized RNG helper for faster masking.
        """
        masked_inputs = []
        labels = []

        for input_ids, pkey in zip(input_ids_list, permutation_keys):
            input_ids = input_ids.copy() if isinstance(input_ids, list) else input_ids.tolist()
            label_ids = [-100] * len(input_ids)

            seq_start, seq_end = self._find_sequence_boundaries(input_ids)
            seq_length = seq_end - seq_start

            if seq_length > self.cdr3_mask_length:
                # Calculate middle region
                middle_start = seq_start + (seq_length - self.cdr3_mask_length) // 2
                middle_end = middle_start + self.cdr3_mask_length

                # Apply vectorized masking to middle region
                self._apply_vectorized_masking(input_ids, label_ids, middle_start, middle_end)

            masked_inputs.append(input_ids)
            labels.append(label_ids)

        return masked_inputs, labels
    
    # ─────────────────────────────────────────────────────────────────────────────
    # TRA-TRB Pairing: Mask entire first chain
    # ────────────────────────────────────────────────��────────────────────────────
    
    def _mask_first_chain(self, input_ids_list: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Mask mlm_probability% of first chain (before first separator) with 80/10/10 strategy.
        Phase 4 optimization: Uses vectorized RNG helper for faster masking.
        """
        masked_inputs = []
        labels = []

        for input_ids in input_ids_list:
            input_ids = input_ids.copy() if isinstance(input_ids, list) else input_ids.tolist()
            label_ids = [-100] * len(input_ids)

            separators = self._find_separators(input_ids)
            seq_start, seq_end = self._find_sequence_boundaries(input_ids)

            if separators:
                # Mask from sequence start to first separator
                mask_end = separators[0]
            else:
                # No separator found, mask first half
                mask_end = seq_start + (seq_end - seq_start) // 2

            # Apply vectorized masking to first chain
            self._apply_vectorized_masking(input_ids, label_ids, seq_start, mask_end)
            
            masked_inputs.append(input_ids)
            labels.append(label_ids)
        
        return masked_inputs, labels
    
    # ─────────────────────────────────────────────────────────────────────────────
    # TCR-MHC: Mask first molecule (or both TCR chains if first two)
    # ─────────────────────────────────────────────────────────────────────────────
    
    def _mask_first_molecules_tcr_mhc(
        self,
        input_ids_list: List[List[int]],
        permutation_keys: List[str]
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Mask first molecule, or first two if they're both TCR chains (tra/trb)
        or both MHC chains (mhc_one/mhc_two).
        Phase 4 optimization: Uses vectorized RNG helper for faster masking.
        """
        masked_inputs = []
        labels = []

        for input_ids, pkey in zip(input_ids_list, permutation_keys):
            input_ids = input_ids.copy() if isinstance(input_ids, list) else input_ids.tolist()
            label_ids = [-100] * len(input_ids)

            separators = self._find_separators(input_ids)
            seq_start, _ = self._find_sequence_boundaries(input_ids)

            # Determine masking strategy based on permutation key
            molecules = pkey.lower().split('_')

            if len(separators) >= 1:
                # Check if first two molecules are both TCR or both MHC
                mask_two = False
                if len(molecules) >= 2:
                    if (molecules[0] in ['tra', 'trb'] and molecules[1] in ['tra', 'trb']):
                        mask_two = True
                    elif (molecules[0] in ['mhc_one', 'mhc_two'] and molecules[1] in ['mhc_one', 'mhc_two']):
                        mask_two = True

                if mask_two and len(separators) >= 2:
                    # Mask first two molecules
                    mask_end = separators[1]
                else:
                    # Mask only first molecule
                    mask_end = separators[0]

                # Apply vectorized masking to the region
                self._apply_vectorized_masking(input_ids, label_ids, seq_start, mask_end)

            masked_inputs.append(input_ids)
            labels.append(label_ids)

        return masked_inputs, labels
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Peptide-MHC: Mask first molecule (or both MHC if first two)
    # ─────────────────────────────────────────────────────────────────────────────
    
    def _mask_first_molecules_peptide_mhc(
        self,
        input_ids_list: List[List[int]],
        permutation_keys: List[str]
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Mask first molecule, or both MHC chains if first two are MHC.
        Phase 4 optimization: Uses vectorized RNG helper for faster masking.
        """
        masked_inputs = []
        labels = []

        for input_ids, pkey in zip(input_ids_list, permutation_keys):
            input_ids = input_ids.copy() if isinstance(input_ids, list) else input_ids.tolist()
            label_ids = [-100] * len(input_ids)

            separators = self._find_separators(input_ids)
            seq_start, _ = self._find_sequence_boundaries(input_ids)

            molecules = pkey.lower().split('_')

            if len(separators) >= 1:
                # Check if first two are both MHC
                mask_two = False
                if len(molecules) >= 2:
                    if (molecules[0] in ['mhc_one', 'mhc_two'] and
                        molecules[1] in ['mhc_one', 'mhc_two']):
                        mask_two = True

                if mask_two and len(separators) >= 2:
                    mask_end = separators[1]
                else:
                    mask_end = separators[0]

                # Apply vectorized masking to the region
                self._apply_vectorized_masking(input_ids, label_ids, seq_start, mask_end)

            masked_inputs.append(input_ids)
            labels.append(label_ids)

        return masked_inputs, labels
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Specificity: Mask first molecule
    # ─────────────────────────────────────────────────────────────────────────────
    
    def _mask_first_molecule(self, input_ids_list: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Mask only the first molecule in the sequence.
        Phase 4 optimization: Uses vectorized RNG helper for faster masking.
        """
        masked_inputs = []
        labels = []

        for input_ids in input_ids_list:
            input_ids = input_ids.copy() if isinstance(input_ids, list) else input_ids.tolist()
            label_ids = [-100] * len(input_ids)

            separators = self._find_separators(input_ids)
            seq_start, _ = self._find_sequence_boundaries(input_ids)

            if separators:
                mask_end = separators[0]
            else:
                # Fallback: mask first third
                seq_end = len(input_ids)
                mask_end = seq_start + (seq_end - seq_start) // 3

            # Apply vectorized masking to first molecule
            self._apply_vectorized_masking(input_ids, label_ids, seq_start, mask_end)

            masked_inputs.append(input_ids)
            labels.append(label_ids)

        return masked_inputs, labels

    # ─────────────────────────────────────────────────────────────────────────────
    # Full CDR Masking (NEW - full_tra, full_trb modes)
    # ─────────────────────────────────────────────────────────────────────────────

    def _mask_cdr_regions_tra(
        self,
        input_ids_list: List[List[int]],
        features: List[Dict[str, Any]]
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Mask CDR1, CDR2, and CDR3 regions of TRA full-length sequences.

        Strategy:
        1. First check for pre-calculated CDR positions (tra_cdr1_pos, tra_cdr2_pos, tra_cdr3_pos)
        2. If not available, calculate on-the-fly using CDRRegionIdentifier
        3. Map AA positions to token positions
        4. Mask mlm_probability% of tokens within CDR regions
        5. Use 80/10/10 strategy (80% [MASK], 10% random, 10% keep)
        6. If CDR regions can't be identified, skip masking (all labels = -100)
        """
        masked_inputs = []
        labels = []

        for input_ids, feature in zip(input_ids_list, features):
            input_ids = input_ids.copy() if isinstance(input_ids, list) else input_ids.tolist()
            label_ids = [-100] * len(input_ids)

            # Get required fields
            tra_full = feature.get('tra_full', '')

            # Skip if missing full sequence
            if not tra_full:
                masked_inputs.append(input_ids)
                labels.append(label_ids)
                continue

            # PRIORITY 1: Use pre-calculated CDR positions (from tokenization)
            cdr_regions = {}
            if feature.get('tra_cdr1_pos') is not None:
                cdr_regions['cdr1'] = feature['tra_cdr1_pos']
            if feature.get('tra_cdr2_pos') is not None:
                cdr_regions['cdr2'] = feature['tra_cdr2_pos']
            if feature.get('tra_cdr3_pos') is not None:
                cdr_regions['cdr3'] = feature['tra_cdr3_pos']

            # PRIORITY 2: Calculate on-the-fly if pre-calculated not available
            if not cdr_regions and self.cdr_identifier:
                tra_cdr3 = feature.get('tra', '')
                trav_gene = feature.get('trav_gene', '')

                if tra_cdr3 and trav_gene:
                    cdr_regions = self.cdr_identifier.get_cdr_regions(
                        full_sequence=tra_full,
                        cdr3_sequence=tra_cdr3,
                        v_gene=trav_gene,
                        chain='TRA'
                    )

            # Skip if CDR regions not identified
            if not cdr_regions:
                masked_inputs.append(input_ids)
                labels.append(label_ids)
                continue

            # Map AA positions to token positions
            aa_to_token = self._map_aa_to_tokens(tra_full, input_ids)

            # Mask tokens within CDR regions (Phase 4 optimization: vectorized)
            for region_name, (aa_start, aa_end) in cdr_regions.items():
                # Convert AA positions to token positions
                token_start = aa_to_token.get(aa_start)
                token_end = aa_to_token.get(aa_end - 1)  # End is exclusive

                if token_start is None or token_end is None:
                    continue

                # Apply vectorized masking to this CDR region
                self._apply_vectorized_masking(input_ids, label_ids, token_start, token_end + 1)

            masked_inputs.append(input_ids)
            labels.append(label_ids)

        return masked_inputs, labels

    def _mask_cdr_regions_trb(
        self,
        input_ids_list: List[List[int]],
        features: List[Dict[str, Any]]
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Mask CDR1, CDR2, and CDR3 regions of TRB full-length sequences.
        Same logic as _mask_cdr_regions_tra but for TRB chain.

        Strategy:
        1. First check for pre-calculated CDR positions (trb_cdr1_pos, trb_cdr2_pos, trb_cdr3_pos)
        2. If not available, calculate on-the-fly using CDRRegionIdentifier
        3. Map AA positions to token positions and mask
        """
        masked_inputs = []
        labels = []

        for input_ids, feature in zip(input_ids_list, features):
            input_ids = input_ids.copy() if isinstance(input_ids, list) else input_ids.tolist()
            label_ids = [-100] * len(input_ids)

            # Get required fields
            trb_full = feature.get('trb_full', '')

            # Skip if missing full sequence
            if not trb_full:
                masked_inputs.append(input_ids)
                labels.append(label_ids)
                continue

            # PRIORITY 1: Use pre-calculated CDR positions (from tokenization)
            cdr_regions = {}
            if feature.get('trb_cdr1_pos') is not None:
                cdr_regions['cdr1'] = feature['trb_cdr1_pos']
            if feature.get('trb_cdr2_pos') is not None:
                cdr_regions['cdr2'] = feature['trb_cdr2_pos']
            if feature.get('trb_cdr3_pos') is not None:
                cdr_regions['cdr3'] = feature['trb_cdr3_pos']

            # PRIORITY 2: Calculate on-the-fly if pre-calculated not available
            if not cdr_regions and self.cdr_identifier:
                trb_cdr3 = feature.get('trb', '')
                trbv_gene = feature.get('trbv_gene', '')

                if trb_cdr3 and trbv_gene:
                    cdr_regions = self.cdr_identifier.get_cdr_regions(
                        full_sequence=trb_full,
                        cdr3_sequence=trb_cdr3,
                        v_gene=trbv_gene,
                        chain='TRB'
                    )

            # Skip if CDR regions not identified
            if not cdr_regions:
                masked_inputs.append(input_ids)
                labels.append(label_ids)
                continue

            # Map and mask (Phase 4 optimization: vectorized)
            aa_to_token = self._map_aa_to_tokens(trb_full, input_ids)

            for region_name, (aa_start, aa_end) in cdr_regions.items():
                token_start = aa_to_token.get(aa_start)
                token_end = aa_to_token.get(aa_end - 1)

                if token_start is None or token_end is None:
                    continue

                # Apply vectorized masking to this CDR region
                self._apply_vectorized_masking(input_ids, label_ids, token_start, token_end + 1)

            masked_inputs.append(input_ids)
            labels.append(label_ids)

        return masked_inputs, labels

    def _map_aa_to_tokens(
        self,
        aa_sequence: str,
        token_ids: List[int]
    ) -> Dict[int, int]:
        """
        Map amino acid positions to token positions.

        For character-level tokenizers (ESM2, ProtBERT):
        - 1 AA = 1 token (plus offset for special tokens)

        Args:
            aa_sequence: Full amino acid sequence
            token_ids: Tokenized sequence

        Returns:
            Dict mapping AA position -> token position
        """
        # Find sequence boundaries (skip CLS/special tokens)
        seq_start, seq_end = self._find_sequence_boundaries(token_ids)

        # For character-level tokenization
        # AA position i maps to token position (seq_start + i)
        aa_to_token = {}
        for aa_pos in range(len(aa_sequence)):
            token_pos = seq_start + aa_pos
            if token_pos < seq_end and token_pos < len(token_ids):
                aa_to_token[aa_pos] = token_pos

        return aa_to_token

    # ─────────────────────────────────────────────────────────────────────────────
    # Batch creation with padding
    # ─────────────────────────────────────────────────────────────────────────────

    def _create_batch(
        self,
        masked_inputs: List[List[int]],
        labels: List[List[int]]
    ) -> Dict[str, torch.Tensor]:
        """
        Convert lists to padded tensors.
        Phase 6 optimization: Use numpy pre-allocated arrays for faster padding.
        """
        # Get dimensions
        batch_size = len(masked_inputs)
        max_len = max(len(seq) for seq in masked_inputs)

        # Pre-allocate numpy arrays (faster than list concatenation)
        padded_inputs = np.full((batch_size, max_len), self.pad_token_id, dtype=np.int64)
        padded_labels = np.full((batch_size, max_len), -100, dtype=np.int64)
        attention_masks = np.zeros((batch_size, max_len), dtype=np.int64)

        # Fill arrays (vectorized assignment)
        for i, (inp, lab) in enumerate(zip(masked_inputs, labels)):
            seq_len = len(inp)
            padded_inputs[i, :seq_len] = inp
            padded_labels[i, :seq_len] = lab
            attention_masks[i, :seq_len] = 1

        # Convert to tensors (numpy to torch is very fast)
        return {
            "input_ids": torch.from_numpy(padded_inputs).long(),
            "attention_mask": torch.from_numpy(attention_masks).long(),
            "labels": torch.from_numpy(padded_labels).long(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. METRICS AND LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

class StreamingMetricsTracker:
    """
    Tracks accuracy and perplexity incrementally during evaluation.
    Prevents OOM by processing each batch separately instead of storing all logits.

    This enables evaluating on full validation sets with large models by computing
    metrics batch-by-batch instead of loading all predictions into memory at once.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all counters for a new evaluation run."""
        self.total_correct = 0
        self.total_tokens = 0
        self.total_loss = 0.0
        self.num_batches = 0

    def update(self, logits: np.ndarray, labels: np.ndarray):
        """
        Update metrics with a single batch.

        Args:
            logits: Shape (batch_size, seq_len, vocab_size)
            labels: Shape (batch_size, seq_len)
        """
        # Mask out -100 labels (padding/non-masked tokens)
        mask = labels != -100

        # Calculate accuracy
        preds = np.argmax(logits, axis=-1)
        correct = (preds[mask] == labels[mask]).sum()
        self.total_correct += correct
        self.total_tokens += mask.sum()

        # Calculate loss for perplexity (use float64 for numerical stability)
        logits_masked = logits[mask].astype(np.float64)
        labels_masked = labels[mask].astype(np.int64)

        # Compute cross-entropy loss in chunks to avoid memory spike
        chunk_size = 10000  # Process 10k tokens at a time
        for i in range(0, len(labels_masked), chunk_size):
            chunk_logits = logits_masked[i:i+chunk_size]
            chunk_labels = labels_masked[i:i+chunk_size]

            # Log-softmax
            logits_max = np.max(chunk_logits, axis=-1, keepdims=True)
            logits_shifted = chunk_logits - logits_max
            log_sum_exp = np.log(np.sum(np.exp(logits_shifted), axis=-1, keepdims=True))
            log_probs = logits_shifted - log_sum_exp

            # Negative log likelihood
            nll = -log_probs[np.arange(len(chunk_labels)), chunk_labels]
            self.total_loss += nll.sum()

        self.num_batches += 1

    def compute(self) -> Dict[str, float]:
        """Compute final metrics from accumulated statistics."""
        if self.total_tokens == 0:
            return {"accuracy": 0.0, "perplexity": float('inf')}

        accuracy = self.total_correct / self.total_tokens
        avg_loss = self.total_loss / self.total_tokens
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')

        return {
            "accuracy": float(accuracy),
            "perplexity": float(perplexity),
        }


# Global metrics tracker (initialized before training)
metrics_tracker = StreamingMetricsTracker()


def preprocess_logits_for_metrics(logits, labels):
    """
    Called by Trainer for EACH BATCH before concatenation.
    This is where we do incremental processing to avoid OOM.

    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size)
        labels: Tensor of shape (batch_size, seq_len)

    Returns:
        Dummy tensor (we've already processed the data)
    """
    # Move to CPU and convert to numpy immediately
    logits_np = logits.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    # Update metrics tracker
    metrics_tracker.update(logits_np, labels_np)

    # Return dummy predictions (just argmax) to save memory
    # Trainer still needs something, but we've already computed what we need
    preds = logits.argmax(dim=-1)
    return preds


def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    """
    Memory-efficient metrics computation using streaming approach.

    This function receives full arrays from Trainer, but we've already
    processed them batch-by-batch using preprocess_logits_for_metrics.
    We just return the accumulated results.
    """
    # Return accumulated metrics from streaming tracker
    return metrics_tracker.compute()


def log_prediction_examples(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    num_examples: int = 50,
    device: str = "cuda"
) -> None:
    """
    Log prediction examples to W&B as a table.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        dataset: Dataset to sample from
        num_examples: Number of examples to log
        device: Device to run inference on
    """
    if wandb.run is None:
        return
    
    model.eval()
    model.to(device)
    
    # Sample random examples
    indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
    
    examples_data = []
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="Generating prediction examples"):
            example = dataset[idx]
            
            # Get input_ids and labels
            input_ids = torch.tensor([example["input_ids"]], dtype=torch.long).to(device)
            labels = torch.tensor([example["labels"]], dtype=torch.long).to(device)
            attention_mask = torch.tensor([example["attention_mask"]], dtype=torch.long).to(device)
            
            # Get predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            # Decode masked input (showing [MASK] tokens)
            masked_input = tokenizer.decode(input_ids[0], skip_special_tokens=False)
            
            # Create predicted output by replacing masked positions
            output_ids = input_ids[0].clone()
            mask_positions = labels[0] != -100
            output_ids[mask_positions] = predictions[0][mask_positions]
            predicted_output = tokenizer.decode(output_ids, skip_special_tokens=False)
            
            # Get ground truth
            ground_truth_ids = input_ids[0].clone()
            ground_truth_ids[mask_positions] = labels[0][mask_positions]
            ground_truth = tokenizer.decode(ground_truth_ids, skip_special_tokens=False)
            
            # Calculate accuracy for this example
            correct = (predictions[0][mask_positions] == labels[0][mask_positions]).sum().item()
            total = mask_positions.sum().item()
            accuracy = correct / total if total > 0 else 0.0
            
            examples_data.append([
                masked_input,
                predicted_output,
                ground_truth,
                f"{accuracy:.2%}"
            ])
    
    # Create W&B table
    table = wandb.Table(
        data=examples_data,
        columns=["Masked Input", "Predicted Output", "Ground Truth", "Accuracy"]
    )
    
    wandb.log({"prediction_examples": table})
    print(f"✅ Logged {len(examples_data)} prediction examples to W&B")
    
    model.train()


def log_dataset_statistics(dataset: Any, split_name: str, log_to_wandb: bool = False) -> Dict[str, Any]:
    """Log dataset statistics to W&B and console."""
    num_examples = len(dataset)
    
    # Count permutation keys
    if "permutation_key" in dataset.column_names:
        pkey_counts = Counter(dataset["permutation_key"])
        pkey_counts_str = {k: v for k, v in pkey_counts.items()}
    else:
        pkey_counts_str = {}
    
    stats = {
        f"{split_name}_num_examples": num_examples,
    }
    
    print(f"\n{split_name.upper()} Dataset Statistics:")
    print(f"  Total examples: {num_examples:,}")
    if pkey_counts_str:
        print(f"  Permutation key distribution:")
        for key, count in sorted(pkey_counts_str.items(), key=lambda x: x[1], reverse=True):
            print(f"    {key}: {count:,} ({count/num_examples*100:.1f}%)")
        
        # Create bar chart for W&B
        if log_to_wandb and wandb.run is not None:
            # Sort by count for better visualization
            sorted_keys = sorted(pkey_counts_str.items(), key=lambda x: x[1], reverse=True)
            
            # Create bar chart data
            data = [[key, count] for key, count in sorted_keys]
            table = wandb.Table(data=data, columns=["Permutation Key", "Count"])
            
            stats[f"{split_name}_permutation_distribution"] = wandb.plot.bar(
                table, 
                "Permutation Key", 
                "Count",
                title=f"{split_name.title()} Dataset: Examples per Permutation Key"
            )
    
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ON-THE-FLY TOKENIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def create_tokenizer_for_raw_data(
    tokenizer_type: str,
    model_path: str,
    max_length: int = 512
):
    """
    Create tokenizer for raw data processing.

    Args:
        tokenizer_type: One of 'protbert', 'bert', 'esm2', 'esm3'
        model_path: Model path to load matching tokenizer
        max_length: Maximum sequence length

    Returns:
        Tokenizer instance
    """
    if tokenizer_type == "protbert":
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(
            "Rostlab/prot_bert",
            do_lower_case=False,
            model_max_length=max_length
        )
    elif tokenizer_type == "bert":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=max_length
        )
    elif tokenizer_type in ["esm2", "esm3"]:
        from transformers import EsmTokenizer
        tokenizer = EsmTokenizer.from_pretrained(
            model_path,
            model_max_length=max_length
        )
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    print(f"✓ Created {tokenizer_type} tokenizer (vocab: {len(tokenizer)})")
    return tokenizer


def concatenate_sequences_for_tokenization(
    row: Dict,
    tokenizer_type: str
) -> str:
    """
    Concatenate molecule sequences for tokenization.

    Handles both data formats:
    - New: 'sequence' column (space-separated molecules)
    - Old: Individual columns (tra, trb, peptide, mhc_one, mhc_two)

    Format by tokenizer type:
    - ProtBERT/BERT: Spaces between amino acids, [SEP] between molecules
    - ESM2/ESM3: Contiguous amino acids, dash (-) between molecules

    Args:
        row: Dictionary with sequence data
        tokenizer_type: 'protbert', 'bert', 'esm2', or 'esm3'

    Returns:
        Concatenated sequence string
    """
    # New format: pre-concatenated 'sequence' column
    if 'sequence' in row and row.get('sequence'):
        sequence_str = str(row['sequence'])
        if sequence_str and sequence_str != 'nan':
            molecules = [mol for mol in sequence_str.split()
                        if mol.upper() != 'NA']

            if tokenizer_type in ['protbert', 'bert']:
                spaced = [" ".join(list(mol)) for mol in molecules]
                return " [SEP] ".join(spaced)
            else:  # esm2, esm3
                return "-".join(molecules)

    # Old format: individual columns (tra, trb, peptide, mhc_one, mhc_two)
    sequences = []
    for field in ['tra', 'trb', 'peptide', 'mhc_one', 'mhc_two']:
        val = row.get(field, '')
        if val and str(val) not in ['nan', '', 'NA', 'na']:
            sequences.append(str(val))

    if not sequences:
        return ""

    if tokenizer_type in ['protbert', 'bert']:
        spaced = [" ".join(list(seq)) for seq in sequences]
        return " [SEP] ".join(spaced)
    else:
        return "-".join(sequences)


def identify_cdr_positions_basic(
    row: Dict,
    cdr_identifier
) -> Dict:
    """
    Identify CDR positions for TRA/TRB if full sequences exist.

    NO TCR STITCHING - only processes if tra_full/trb_full already present.
    Uses CDRRegionIdentifier to find CDR1/2/3 positions.

    Args:
        row: Dictionary with sequence data
        cdr_identifier: Optional CDRRegionIdentifier instance

    Returns:
        Dictionary with added CDR position fields
    """
    row_copy = row.copy()

    # Initialize CDR position fields
    for chain in ['tra', 'trb']:
        for region in ['cdr1', 'cdr2', 'cdr3']:
            row_copy[f'{chain}_{region}_pos'] = None

    if cdr_identifier is None:
        return row_copy

    # Process TRA if full sequence exists
    if row.get('tra_full'):
        tra_cdr3 = row.get('tra', '')
        trav_gene = row.get('trav_gene_std', '') or row.get('trav_gene', '')

        if tra_cdr3 and trav_gene:
            try:
                cdr_regions = cdr_identifier.get_cdr_regions(
                    full_sequence=row['tra_full'],
                    cdr3_sequence=tra_cdr3,
                    v_gene=trav_gene,
                    chain='TRA'
                )
                for region, pos in cdr_regions.items():
                    row_copy[f'tra_{region}_pos'] = pos
            except:
                pass

    # Process TRB if full sequence exists
    if row.get('trb_full'):
        trb_cdr3 = row.get('trb', '')
        trbv_gene = row.get('trbv_gene_std', '') or row.get('trbv_gene', '')

        if trb_cdr3 and trbv_gene:
            try:
                cdr_regions = cdr_identifier.get_cdr_regions(
                    full_sequence=row['trb_full'],
                    cdr3_sequence=trb_cdr3,
                    v_gene=trbv_gene,
                    chain='TRB'
                )
                for region, pos in cdr_regions.items():
                    row_copy[f'trb_{region}_pos'] = pos
            except:
                pass

    return row_copy


def create_tokenization_function(
    tokenizer,
    tokenizer_type: str,
    cdr_identifier,
    max_length: int = 512
):
    """
    Create tokenization function for HuggingFace dataset.map().

    Returns a function that processes batches of raw examples.
    OPTIMIZED for maximum throughput with vectorized operations.

    Args:
        tokenizer: Tokenizer instance
        tokenizer_type: Type of tokenizer being used
        cdr_identifier: Optional CDRRegionIdentifier instance
        max_length: Maximum sequence length

    Returns:
        Function that tokenizes a batch of examples
    """
    # Pre-compute separator strings (avoid repeated string operations)
    if tokenizer_type in ['protbert', 'bert']:
        sep_token = " [SEP] "
        def format_sequence(mol: str) -> str:
            # Use string multiplication pattern for faster spacing
            return " ".join(mol) if mol else ""
    else:  # esm2, esm3
        sep_token = "-"
        def format_sequence(mol: str) -> str:
            return mol if mol else ""

    # Pre-define column keys to check (avoid repeated list creation)
    sequence_columns = ['tra', 'trb', 'peptide', 'mhc_one', 'mhc_two']
    invalid_values = {'nan', '', 'NA', 'na', 'None', None}

    def tokenize_batch(examples):
        """Tokenize a batch of raw parquet rows - OPTIMIZED."""
        # Get batch size from first column
        first_key = next(iter(examples.keys()))
        batch_size = len(examples[first_key])

        # Step 1: Concatenate sequences - VECTORIZED
        sequences = []

        # Check if we have the 'sequence' column (new format - faster path)
        if 'sequence' in examples:
            seq_column = examples['sequence']
            for i in range(batch_size):
                seq_val = seq_column[i]
                if seq_val and str(seq_val) not in invalid_values:
                    molecules = [m for m in str(seq_val).split() if m.upper() != 'NA']
                    if molecules:
                        formatted = [format_sequence(m) for m in molecules]
                        sequences.append(sep_token.join(formatted))
                    else:
                        sequences.append("")
                else:
                    sequences.append("")
        else:
            # Old format: individual columns
            # Pre-fetch all columns once (avoid repeated dict lookups)
            col_data = {col: examples.get(col, [None] * batch_size) for col in sequence_columns}

            for i in range(batch_size):
                parts = []
                for col in sequence_columns:
                    val = col_data[col][i] if i < len(col_data[col]) else None
                    if val and str(val) not in invalid_values:
                        parts.append(format_sequence(str(val)))
                sequences.append(sep_token.join(parts) if parts else "")

        # Step 2: Tokenize (HF batching) - Already optimized by HF
        encoded = tokenizer(
            sequences,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors=None
        )

        # Step 3: Add CDR positions if needed (only for full_tra/full_trb modes)
        if cdr_identifier is not None:
            cdr_positions = {
                'tra_cdr1_pos': [],
                'tra_cdr2_pos': [],
                'tra_cdr3_pos': [],
                'trb_cdr1_pos': [],
                'trb_cdr2_pos': [],
                'trb_cdr3_pos': [],
            }

            # Pre-fetch CDR-related columns
            cdr_cols = {
                'tra_full': examples.get('tra_full', [None] * batch_size),
                'trb_full': examples.get('trb_full', [None] * batch_size),
                'tra': examples.get('tra', [None] * batch_size),
                'trb': examples.get('trb', [None] * batch_size),
                'trav_gene_std': examples.get('trav_gene_std', [None] * batch_size),
                'trav_gene': examples.get('trav_gene', [None] * batch_size),
                'trbv_gene_std': examples.get('trbv_gene_std', [None] * batch_size),
                'trbv_gene': examples.get('trbv_gene', [None] * batch_size),
            }

            for i in range(batch_size):
                row = {k: v[i] if i < len(v) else None for k, v in cdr_cols.items()}
                row_with_cdr = identify_cdr_positions_basic(row, cdr_identifier)

                for key in cdr_positions.keys():
                    cdr_positions[key].append(row_with_cdr.get(key))

            encoded.update(cdr_positions)

        return encoded

    return tokenize_batch


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(
    model_path: str,
    use_lora: bool = False,
    lora_config: Optional[Dict[str, Any]] = None,
    enable_gradient_checkpointing: bool = False,
) -> Tuple[Any, Any]:
    """
    Load model and tokenizer, optionally applying LoRA.
    
    Supports both HuggingFace models and ESM3 models via the esm package.
    
    Args:
        model_path: Path to model (local or HuggingFace Hub) or ESM3 model name (e.g., 'esm3-small', 'esm3-medium', 'esm3-large')
        use_lora: Whether to apply LoRA
        lora_config: LoRA configuration dict
        enable_gradient_checkpointing: Whether to enable gradient checkpointing
    
    Returns:
        (model, tokenizer)
    """
    print(f"Loading model and tokenizer from: {model_path}")
    
    # Check if this is an ESM3 model
    is_esm3 = False
    if ESM3_AVAILABLE and (model_path.startswith('esm3') or 'esm3' in model_path.lower()):
        is_esm3 = True
        print(f"Detected ESM3 model: {model_path}")
    
    if is_esm3:
        # Load ESM3 model using the esm package
        print("Loading ESM3 model using esm package...")
        
        # Map common ESM3 model names to loading functions
        esm3_model_loaders = {
            'esm3-small': ESM3_sm_open_v0,
            'esm3-sm': ESM3_sm_open_v0,
            'esm3_sm_open_v0': ESM3_sm_open_v0,
        }
        
        model_loader = esm3_model_loaders.get(model_path.lower())
        if model_loader is None:
            # Try to find a matching loader
            for key, loader in esm3_model_loaders.items():
                if key in model_path.lower():
                    model_loader = loader
                    break
        
        if model_loader is None:
            raise ValueError(
                f"Unknown ESM3 model: {model_path}. "
                f"Available models: {list(esm3_model_loaders.keys())}"
            )
        
        # Load ESM3 model with device placement
        print(f"Loading ESM3 model with {model_loader.__name__}...")
        print("⚠️  Loading large model - this may take a moment and use significant memory...")
        
        # Clear CUDA cache before loading
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🔧 Cleared CUDA cache")
        
        # Load model directly to GPU
        esm3_model = model_loader()
        
        print(f"✅ Model loaded successfully")
        
        # Clear cache again after loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Wrap ESM3 model for HuggingFace Trainer compatibility
        model = ESM3ForMaskedLM(esm3_model)
        print("✅ Wrapped ESM3 model for HuggingFace Trainer compatibility")
        
        # ESM3 uses its own tokenization - we need to wrap it for HuggingFace Trainer compatibility
        # For now, we'll use a simple wrapper that delegates to the model's internal tokenizer
        class ESM3TokenizerWrapper:
            """Wrapper to make ESM3 tokenization compatible with HuggingFace Trainer"""
            def __init__(self, esm3_model):
                self.model = esm3_model
                # Set required attributes for Trainer compatibility
                self.pad_token = "<pad>"
                self.pad_token_id = 0  # ESM3 uses 0 for padding
                self.mask_token = "<mask>"
                self.mask_token_id = 32  # ESM3 mask token
                self.vocab_size = 64  # ESM3 vocabulary size
                
            def __call__(self, text, **kwargs):
                # This won't be used since we're using pre-tokenized data
                raise NotImplementedError("ESM3 tokenization should be done during dataset preparation")
            
            def decode(self, token_ids, **kwargs):
                # Placeholder for decode - needed for logging
                if isinstance(token_ids, list):
                    return " ".join([str(t) for t in token_ids])
                else:
                    return str(token_ids)
        
        tokenizer = ESM3TokenizerWrapper(model)
        
        print(f"✅ Loaded ESM3 model: {model_loader.__name__}")
        
    else:
        # Check if this is a PEFT adapter model
        is_peft_model = False
        if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "adapter_config.json")):
            print(f"Detected existing PEFT adapter at {model_path}")
            is_peft_model = True
            
            # Load base model name from adapter config
            with open(os.path.join(model_path, "adapter_config.json"), 'r') as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path", "Rostlab/prot_bert")
            
            print(f"Loading base model: {base_model_name}")
            base_model = AutoModelForMaskedLM.from_pretrained(base_model_name)
            
            # Load PEFT model and merge weights
            peft_model = PeftModel.from_pretrained(base_model, model_path)
            print("Merging PEFT weights into base model...")
            model = peft_model.merge_and_unload()
            
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        else:
            # Load regular HuggingFace model
            model = AutoModelForMaskedLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # For ESM3 or other models without eos_token
            print("⚠️  No pad token found, using mask token as pad token")
            tokenizer.pad_token = tokenizer.mask_token if hasattr(tokenizer, 'mask_token') else None
    
    # Apply LoRA if requested
    if use_lora:
        print("Applying LoRA configuration...")
        
        # Determine target modules based on model type
        target_modules = lora_config.get("target_modules", ["query", "value"])
        
        # For ESM3, use different target modules
        if is_esm3:
            print("Detected ESM3 model - using ESM3-specific LoRA target modules")
            # ESM3 uses MultiHeadAttention with layernorm_qkv containing the QKV projection
            # and ffn layers. We target the Linear layers inside these modules.
            # Note: ESM3 is wrapped, so we need to prefix with 'esm3.'
            target_modules = lora_config.get("target_modules", [
                "esm3.transformer.blocks.0.attn.layernorm_qkv.1",  # Example path to Linear layer
                "esm3.transformer.blocks.0.attn.out_proj",
                "esm3.transformer.blocks.0.ffn.1",
                "esm3.transformer.blocks.0.ffn.3"
            ])
            print(f"  Target modules: {target_modules}")
        
        lora_cfg = LoraConfig(
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("alpha", 32),
            target_modules=target_modules,
            lora_dropout=lora_config.get("dropout", 0.05),
            bias="none",
        )
        
        try:
            model = get_peft_model(model, lora_cfg)
            
            # Ensure MLM head is trainable
            for name, param in model.named_parameters():
                if "cls" in name or "lm_head" in name or "output" in name.lower():
                    param.requires_grad = True
            
            model.print_trainable_parameters()
        except ValueError as e:
            error_msg = str(e)
            print(f"\n⚠️  LoRA application failed: {error_msg}")
            
            # If the error is about unsupported module types, try to find Linear layers
            if "not supported" in error_msg or "not found" in error_msg:
                print("Attempting to find correct target modules by scanning Linear layers...")
                
                # Try to auto-detect target modules
                import torch
                linear_layers = []
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        linear_layers.append(name)
                
                print(f"Found {len(linear_layers)} Linear layers in model")
                
                # For ESM3, look for specific patterns in the layer names
                # Priority: attention layers, then FFN layers
                attn_layers = [name for name in linear_layers if 'attn' in name.lower()]
                ffn_layers = [name for name in linear_layers if 'ffn' in name.lower()]
                
                # Select a subset of layers to target (from first transformer block)
                found_modules = []
                if attn_layers:
                    # Take first 2-3 attention-related layers
                    found_modules.extend(attn_layers[:3])
                    print(f"  Selected attention layers: {attn_layers[:3]}")
                
                if ffn_layers and len(found_modules) < 4:
                    # Add some FFN layers
                    found_modules.extend(ffn_layers[:2])
                    print(f"  Selected FFN layers: {ffn_layers[:2]}")
                
                if not found_modules:
                    # Fall back to first few linear layers
                    found_modules = [name for name in linear_layers[:4]]
                    print(f"  Using first linear layers: {found_modules}")
                
                if found_modules:
                    print(f"\nRetrying LoRA with target modules: {found_modules}")
                    lora_cfg.target_modules = found_modules
                    model = get_peft_model(model, lora_cfg)
                    model.print_trainable_parameters()
                else:
                    print(f"\n❌ Available linear layers (first 20):")
                    for i, layer in enumerate(linear_layers[:20]):
                        print(f"  {i+1}. {layer}")
                    raise ValueError(
                        f"Could not automatically detect LoRA target modules. "
                        f"Please specify --lora_target_modules manually from the list above."
                    )
            else:
                raise
    
    # Enable gradient checkpointing for memory savings
    # Note: Can slow down training but essential for large models on limited GPU memory
    if enable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        print("✅ Enabling gradient checkpointing for memory efficiency...")
        model.gradient_checkpointing_enable()
    
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CUSTOM STREAMING EVALUATION TRAINER
# ═══════════════════════════════════════════════════════════════════════════════

class StreamingEvalTrainer(Trainer):
    """
    Custom Trainer that supports streaming evaluation for large validation sets.

    Uses DataLoader to process validation data in chunks, accumulating only
    metrics (not predictions) to minimize GPU memory usage.
    """

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Streaming evaluation that processes full dataset without OOM.

        Unlike standard Trainer.evaluate(), this method:
        - Processes data in batches without accumulating all predictions
        - Only stores running metrics (loss, accuracy, perplexity)
        - Handles arbitrarily large validation sets
        """
        # Setup
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        model = self._wrap_model(self.model, training=False)
        model.eval()

        # Initialize metrics
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        num_batches = 0

        # Streaming evaluation loop
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = self._prepare_inputs(batch)

                # Forward pass
                with self.compute_loss_context_manager():
                    outputs = model(**batch)

                loss = outputs.loss
                logits = outputs.logits
                labels = batch.get("labels", batch.get("input_ids"))

                # Accumulate loss
                total_loss += loss.item()
                num_batches += 1

                # Compute accuracy (only for non-padding tokens)
                if labels is not None:
                    # Get predictions
                    predictions = torch.argmax(logits, dim=-1)

                    # Mask for valid tokens (not padding, not special tokens)
                    mask = labels != -100

                    # Count correct predictions
                    correct = (predictions[mask] == labels[mask]).sum().item()
                    total_correct += correct
                    total_tokens += mask.sum().item()

                # Free memory after each batch
                del outputs, logits, predictions, mask
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Compute final metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')

        metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_accuracy": accuracy,
            f"{metric_key_prefix}_perplexity": perplexity,
            f"{metric_key_prefix}_samples": len(eval_dataset),
        }

        # Log callback
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )

        return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune protein language models with task-specific masking"
    )
    
    # Data arguments
    # Data source options (mutually exclusive)
    data_source_group = parser.add_mutually_exclusive_group(required=True)
    data_source_group.add_argument(
        "--dataset_path",
        type=str,
        help="Path to pre-tokenized HuggingFace dataset (existing behavior)"
    )
    data_source_group.add_argument(
        "--raw_data_dir",
        type=str,
        help="Path to directory with raw parquet files (on-the-fly tokenization)"
    )

    # Tokenization options (required when using --raw_data_dir)
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        choices=['protbert', 'bert', 'esm2', 'esm3'],
        help="Tokenizer type for on-the-fly tokenization (required with --raw_data_dir)"
    )
    parser.add_argument(
        "--tokenization_max_length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization (default: 512)"
    )
    parser.add_argument(
        "--tokenization_num_workers",
        type=int,
        default=None,  # Auto-detect based on CPU count
        help="Parallel workers for tokenization (default: auto-detect, typically CPU_COUNT - 4 for safety)"
    )
    parser.add_argument(
        "--tokenization_batch_size",
        type=int,
        default=5000,  # Increased from 1000 for better throughput
        help="Batch size for tokenization (default: 5000, higher = faster but more RAM)"
    )
    parser.add_argument(
        "--use_streaming",
        action="store_true",
        help="Use streaming dataset for truly lazy evaluation. Best for 100M+ examples. "
             "Tokenizes data on-the-fly during training (no pre-processing wait). "
             "Trade-off: Cannot shuffle across full dataset, only within buffer."
    )

    parser.add_argument("--mode", type=str, required=True,
                        choices=["mlm", "tra", "trb", "full_tra", "full_trb", "tra_trb_pairing", "tcr_mhc", "peptide_mhc", "specificity"],
                        help="Fine-tuning mode (determines data filtering and masking strategy). "
                             "If dataset was filtered during tokenization, use --skip-mode-filter")
    parser.add_argument("--skip-mode-filter", action="store_true", default=True,
                        help="Skip mode-based filtering (use if dataset was already filtered during tokenization)")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to pre-trained model. Supports: "
                             "(1) HuggingFace model names (e.g., 'facebook/esm2_t12_35M_UR50D'), "
                             "(2) Local model directories, "
                             "(3) ESM3 model names (e.g., 'esm3-small', 'esm3-medium', 'esm3-large')")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save checkpoints and final model")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size per device")
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Evaluation batch size per device (default: batch_size)")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps (effective batch = batch_size * gradient_accumulation_steps)")
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true",
                        help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=["query", "value"],
                        help="Target modules for LoRA. Default ['query', 'value'] for HuggingFace models. "
                             "For ESM3, auto-detects attention/FFN Linear layers or use specific paths like "
                             "'attn.layernorm_qkv.1', 'attn.out_proj', 'ffn.1', 'ffn.3'")
    
    # Masking arguments
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Probability for MLM masking")
    parser.add_argument("--cdr3_mask_length", type=int, default=5,
                        help="Number of amino acids to mask in CDR3 region (for tra/trb modes)")
    
    # Logging and evaluation
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name for logging")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every N steps")
    parser.add_argument("--eval_steps", type=int, default=None,
                        help="Evaluate every N steps (default: once per epoch)")
    parser.add_argument("--save_steps", type=int, default=None,
                        help="Save checkpoint every N steps (default: once per epoch)")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Maximum number of checkpoints to keep (deletes older ones to save space)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume from checkpoint. Use 'auto' to automatically detect latest checkpoint, or provide path to specific checkpoint")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Maximum number of evaluation samples to use. "
                             "Leave unset to evaluate on full validation set with streaming evaluation.")
    parser.add_argument("--eval_accumulation_steps", type=int, default=10,
                        help="Number of batches to accumulate before moving predictions to CPU. "
                             "Prevents OOM during evaluation with large vocab models (e.g., BERT). "
                             "Lower = less memory but slower. Default: 10")
    parser.add_argument("--log_prediction_examples", action="store_true",
                        help="Log prediction examples to W&B (can be slow for large datasets)")
    
    # Other arguments
    parser.add_argument("--test", action="store_true",
                        help="Run in test mode with smaller dataset")
    parser.add_argument("--use_pre_masked", action="store_true",
                        help="Use pre-masked dataset (skip on-the-fly masking)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 mixed precision")
    parser.add_argument("--bf16", action="store_true",
                        help="Use BF16 mixed precision")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory (slower but reduces memory)")
    parser.add_argument("--optim", type=str, default="adamw_torch",
                        help="Optimizer to use. Use 'adamw_8bit' for 8-bit Adam (requires bitsandbytes) to save memory")
    parser.add_argument("--max_seq_length", type=int, default=None,
                        help="Maximum sequence length. Sequences longer than this will be truncated to save memory")
    
    args = parser.parse_args()

    # ═══════════════════════════════════════════════════════════════════════════
    # Auto-detect optimal tokenization workers if not specified
    # ═══════════════════════════════════════════════════════════════════════════
    if args.tokenization_num_workers is None and args.raw_data_dir:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()

        # Use most CPUs but leave some for system (typically CPU_COUNT - 4)
        # On p4d.24xlarge with 96 vCPUs, this gives 92 workers (massive parallelism!)
        optimal_workers = max(1, cpu_count - 4)
        args.tokenization_num_workers = optimal_workers

        print(f"\n🔧 Auto-detected {cpu_count} CPUs")
        print(f"   Setting tokenization_num_workers={optimal_workers} for maximum throughput")
        print(f"   This should give ~{optimal_workers/8:.1f}x faster tokenization than default (8 workers)")

    # ═══════════════════════════════════════════════════════════════════════════
    # Cache configuration for LOCAL runs (SageMaker handled at module level)
    # ═══════════════════════════════════════════════════════════════════════════
    # For SageMaker, cache was already configured at module level (lines 105-125)
    # For local runs, set cache to raw_data_dir to avoid filling up home directory

    if args.raw_data_dir and not is_sagemaker_environment():
        # Local: Use raw data directory for cache
        hf_cache_path = os.path.join(args.raw_data_dir, ".huggingface_cache")
        os.makedirs(hf_cache_path, exist_ok=True)

        # Set ALL HuggingFace cache environment variables
        os.environ['HF_HOME'] = hf_cache_path
        os.environ['HF_DATASETS_CACHE'] = os.path.join(hf_cache_path, 'datasets')
        os.environ['TRANSFORMERS_CACHE'] = os.path.join(hf_cache_path, 'transformers')
        os.environ['HF_HUB_CACHE'] = os.path.join(hf_cache_path, 'hub')
        os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(hf_cache_path, 'hub')
        os.environ['TORCH_HOME'] = os.path.join(hf_cache_path, 'torch')

        print(f"\n{'='*80}")
        print("🔧 LOCAL CACHE CONFIGURATION")
        print(f"{'='*80}")
        print(f"   HF_HOME:             {os.environ['HF_HOME']}")
        print(f"   HF_DATASETS_CACHE:   {os.environ['HF_DATASETS_CACHE']}")
        print(f"   TRANSFORMERS_CACHE:  {os.environ['TRANSFORMERS_CACHE']}")
        print(f"{'='*80}\n")

    # Set eval batch size if not provided
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Memory optimization warnings and recommendations
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n🔍 GPU Memory: {gpu_memory:.1f} GB")
        
        # Calculate estimated memory usage for BERT
        estimated_model_memory = 0.5  # BERT base ~0.5GB
        estimated_batch_memory = args.batch_size * 0.5  # ~0.5GB per batch for 512 seq len
        estimated_total = estimated_model_memory + estimated_batch_memory
        
        print(f"   Estimated memory usage: ~{estimated_total:.1f} GB")
        print(f"   Model: ~{estimated_model_memory:.1f} GB, Batch: ~{estimated_batch_memory:.1f} GB")
        
        # Auto-apply optimizations if memory is tight
        memory_is_tight = estimated_total > gpu_memory * 0.7
        
        if memory_is_tight or gpu_memory < 25:
            print("\n⚠️  MEMORY OPTIMIZATION RECOMMENDATIONS:")
            print("   Your GPU has limited memory. Consider these options:")
            
            # Auto-reduce batch size if critically low on memory
            if estimated_total > gpu_memory * 0.9 and args.batch_size > 1:
                suggested_batch_size = max(1, args.batch_size // 2)
                print(f"\n   🚨 CRITICAL: Likely to OOM with current settings!")
                print(f"   1. Current batch size: {args.batch_size}")
                print(f"      → STRONGLY RECOMMEND reducing to {suggested_batch_size}")
                if args.gradient_accumulation_steps < 4:
                    suggested_grad_accum = args.gradient_accumulation_steps * (args.batch_size // suggested_batch_size)
                    print(f"      → AND increase gradient_accumulation_steps to {suggested_grad_accum}")
                    print(f"      → This maintains effective batch size of {suggested_batch_size * suggested_grad_accum}")
            else:
                print(f"   1. Current batch size: {args.batch_size}")
                if args.batch_size > 4:
                    print(f"      → Try reducing to 2-4")
                elif args.batch_size > 2:
                    print(f"      → Try reducing to 1-2")
            
            print(f"   2. Gradient accumulation: {args.gradient_accumulation_steps}")
            if args.gradient_accumulation_steps < 4:
                print(f"      → Increase to 4-8 to maintain effective batch size")
            
            if not args.gradient_checkpointing:
                print(f"   3. Gradient checkpointing: OFF")
                print(f"      → Add --gradient_checkpointing (saves ~30-40% memory)")
            
            if args.optim == "adamw_torch":
                print(f"   4. Optimizer: {args.optim}")
                print(f"      → Try --optim adamw_8bit (saves ~50% optimizer memory)")
                print(f"      → Install: pip install bitsandbytes")
            
            if not args.use_lora:
                print(f"   5. LoRA: OFF")
                print(f"      → Add --use_lora --lora_r 8 (reduces trainable params by 99%)")
            
            if args.max_seq_length is None or args.max_seq_length > 512:
                print(f"   6. Max sequence length: {args.max_seq_length if args.max_seq_length else 'unlimited'}")
                print(f"      → Try --max_seq_length 384 or 256 (saves significant memory)")
            
            if not args.fp16 and not args.bf16:
                print(f"   7. Mixed precision: OFF")
                print(f"      → Add --fp16 (saves ~50% memory, faster training)")

            print(f"   8. Eval accumulation steps: {args.eval_accumulation_steps}")
            if args.eval_accumulation_steps > 10:
                print(f"      → Try --eval_accumulation_steps 5 (helps with BERT/large vocab models)")
            else:
                print(f"      → Current setting is optimal for large vocab models")

            print(f"\n   💡 RECOMMENDED COMMAND:")
            print(f"   python scripts/training/fine_tune.py \\")
            print(f"       --batch_size {max(1, args.batch_size // 2)} \\")
            print(f"       --gradient_accumulation_steps {args.gradient_accumulation_steps * 2} \\")
            print(f"       --gradient_checkpointing \\")
            print(f"       --fp16 \\")
            print(f"       --max_seq_length 384 \\")
            print(f"       --use_lora --lora_r 8 \\")
            print(f"       ... (other args)")
            print()
    
    # Set PYTORCH_CUDA_ALLOC_CONF for better memory management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("✅ Enabled expandable_segments for better CUDA memory management")

    # Configure NCCL for multi-GPU training (prevents timeout errors)
    if torch.cuda.device_count() > 1:
        # Increase timeout for large models (default is 10 min, increase to 30 min)
        os.environ["NCCL_TIMEOUT"] = "1800"  # 30 minutes in seconds
        # Enable better error reporting
        os.environ["NCCL_DEBUG"] = "WARN"  # Set to INFO for more verbose debugging
        # Optimize for single-node multi-GPU (p4d.24xlarge)
        os.environ["NCCL_IB_DISABLE"] = "0"  # Enable InfiniBand if available
        os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"  # Skip virtual interfaces
        print(f"✅ Configured NCCL for {torch.cuda.device_count()} GPUs (timeout: 30min)")

    # ─────────────────────────────────────────────────────────────────────────────
    # Initialize Accelerator (for distributed training awareness)
    # ─────────────────────────────────────────────────────────────────────────────
    
    accelerator = Accelerator()
    is_main_process = accelerator.is_main_process
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Initialize W&B (only on main process to avoid duplicate logging)
    # ─────────────────────────────────────────────────────────────────────────────

    if args.wandb_project and is_main_process:
        # Generate run name and config
        run_name = args.wandb_run_name
        config = vars(args).copy()

        if is_sagemaker_environment():
            sm_paths = get_sagemaker_paths()

            # Auto-generate run name with job context
            if run_name is None:
                job_name = os.environ.get('TRAINING_JOB_NAME', 'sagemaker-job')
                run_name = f"{job_name}-{sm_paths['current_host']}"

            # Add SageMaker metadata to config
            config.update({
                'sagemaker_training': True,
                'sagemaker_job_name': os.environ.get('TRAINING_JOB_NAME', 'unknown'),
                'sagemaker_host': sm_paths['current_host'],
                'sagemaker_num_gpus': sm_paths['num_gpus'],
                'sagemaker_hosts': len(sm_paths['hosts']),
            })

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=config,
            settings=wandb.Settings(
                # Disable system metrics to reduce clutter
                _disable_stats=True,
                _disable_meta=True,
            )
        )

        if is_sagemaker_environment():
            print(f"📊 W&B tracking enabled for SageMaker job")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Load and filter dataset
    # ─────────────────────────────────────────────────────────────────────────────

    # Log environment information
    if is_sagemaker_environment():
        sm_paths = get_sagemaker_paths()
        print(f"\n🚀 Running in SageMaker Training Environment")
        print(f"   Training Job: {os.environ.get('TRAINING_JOB_NAME', 'unknown')}")
        print(f"   Current Host: {sm_paths['current_host']}")
        print(f"   Total Hosts: {len(sm_paths['hosts'])}")
        print(f"   GPUs Available: {sm_paths['num_gpus']}")
        print(f"   Checkpoint Dir: {sm_paths['checkpoint_dir']}")
        print(f"   Model Output: {sm_paths['model_dir']}")
    else:
        print(f"\n💻 Running in Local Environment")

    # ═══════════════════════════════════════════════════════════════════════
    # DATA LOADING - Two Paths: Pre-tokenized OR On-the-fly
    # ═══════════════════════════════════════════════════════════════════════

    # Initialize variables that may be set by streaming path
    is_streaming_dataset = False
    total_steps = -1  # -1 means use num_epochs instead
    estimated_train_examples = 0

    if args.dataset_path:
        # ────────────────────────────────────────────────────────────────
        # EXISTING PATH: Load pre-tokenized dataset
        # ────────────────────────────────────────────────────────────────
        dataset_path = resolve_path_for_environment(args.dataset_path, 'dataset')
        print(f"\n📦 Loading PRE-TOKENIZED dataset from: {dataset_path}")
        dataset = load_from_disk(dataset_path)

    elif args.raw_data_dir:
        # ────────────────────────────────────────────────────────────────
        # NEW PATH: On-the-fly tokenization from raw parquet
        # ────────────────────────────────────────────────────────────────
        raw_data_path_str = resolve_path_for_environment(args.raw_data_dir, 'dataset')

        # DDP-AWARE TOKENIZATION: Only main process tokenizes, others wait
        # This prevents 8x redundant work when running with 8 GPUs
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        is_distributed = world_size > 1

        if is_distributed:
            print(f"\n🔄 DDP Mode Detected: Rank {local_rank}/{world_size}")
            if not is_main_process:
                print(f"   [Rank {local_rank}] Waiting for main process to complete tokenization...")

        print(f"\n📦 Loading RAW PARQUET data from: {raw_data_path_str}")
        print(f"   Tokenization: ON-THE-FLY (type: {args.tokenizer_type})")

        # Validate arguments
        if not args.tokenizer_type:
            raise ValueError(
                "--tokenizer_type required when using --raw_data_dir\n"
                "Choose from: protbert, bert, esm2, esm3"
            )

        # Find parquet files
        from pathlib import Path
        from datasets import load_dataset
        import shutil

        raw_data_path = Path(raw_data_path_str)
        parquet_files = list(raw_data_path.glob("*.parquet"))

        # ═══════════════════════════════════════════════════════════════════
        # VERIFY CACHE CONFIGURATION
        # ═══════════════════════════════════════════════════════════════════
        print(f"\n{'='*80}")
        print("🔍 CACHE VERIFICATION")
        print(f"{'='*80}")
        print(f"   HF_HOME:               {os.environ.get('HF_HOME', '❌ NOT SET!')}")
        print(f"   HF_DATASETS_CACHE:     {os.environ.get('HF_DATASETS_CACHE', '❌ NOT SET!')}")
        print(f"   TRANSFORMERS_CACHE:    {os.environ.get('TRANSFORMERS_CACHE', '❌ NOT SET!')}")
        print(f"   HF_HUB_CACHE:          {os.environ.get('HF_HUB_CACHE', '❌ NOT SET!')}")
        print(f"   TORCH_HOME:            {os.environ.get('TORCH_HOME', '❌ NOT SET!')}")
        print(f"\n🚨 TEMP DIRECTORIES (Critical for Arrow/Parquet):")
        print(f"   TMPDIR:                {os.environ.get('TMPDIR', '❌ NOT SET!')}")
        print(f"   ARROW_TMPDIR:          {os.environ.get('ARROW_TMPDIR', '❌ NOT SET!')}")

        # Verify cache is NOT on root filesystem
        cache_home = os.environ.get('HF_HOME', '')
        tmpdir = os.environ.get('TMPDIR', '/tmp')

        if cache_home.startswith('/root/') or cache_home.startswith('/home/'):
            print(f"\n⚠️  WARNING: Cache is on HOME directory: {cache_home}")
            print(f"   This might fill up the root filesystem!")
        elif cache_home.startswith('/opt/ml/'):
            print(f"\n✅ Cache is on EBS volume (not root filesystem)")
        else:
            print(f"\n❓ Cache location unknown - check if it's on large volume")

        # Verify TMPDIR is NOT /tmp
        if tmpdir == '/tmp' or tmpdir.startswith('/tmp/'):
            print(f"⚠️  WARNING: TMPDIR is on root filesystem: {tmpdir}")
            print(f"   PyArrow will create temp files here during parquet processing!")
            print(f"   This is likely causing the disk space error!")
        elif tmpdir.startswith('/opt/ml/'):
            print(f"✅ TMPDIR is on EBS volume: {tmpdir}")
        else:
            print(f"❓ TMPDIR location: {tmpdir}")

        print(f"{'='*80}\n")

        # Print disk usage for debugging
        def print_disk_usage(path="/"):
            """Print disk usage for debugging space issues."""
            try:
                total, used, free = shutil.disk_usage(path)
                print(f"\n💿 Disk Usage ({path}):")
                print(f"   Total: {total / (1024**3):.1f} GB")
                print(f"   Used:  {used / (1024**3):.1f} GB ({used/total*100:.1f}%)")
                print(f"   Free:  {free / (1024**3):.1f} GB ({free/total*100:.1f}%)")
            except Exception as e:
                print(f"   ❌ Could not get disk usage: {e}")

        # Check both root and EBS volume
        print("\n📊 DISK SPACE CHECK:")
        print_disk_usage("/")
        if is_sagemaker_environment():
            print_disk_usage("/opt/ml/input/data/training")

        if not parquet_files:
            raise ValueError(f"No parquet files found in {raw_data_path}")

        print(f"   Found {len(parquet_files)} parquet files")

        # Split files by train/validation based on filename
        train_files = [str(f) for f in parquet_files if 'train' in f.name.lower()]
        val_files = [str(f) for f in parquet_files
                    if 'val' in f.name.lower() or 'validation' in f.name.lower()]

        # If no train/val split in filenames, use all files for both
        # (HuggingFace will handle the actual split via dataset.train_test_split())
        if not train_files and not val_files:
            print("⚠️  No train/val/test split detected in filenames")
            print("   Using all parquet files - will split 80/10/10 during loading")
            all_files = [str(f) for f in parquet_files]

            # Use all files, HuggingFace dataset.load will handle them
            # We'll split after loading
            train_files = all_files
            val_files = None  # Signal to split later
        elif not train_files:
            raise ValueError(
                "No training parquet files found.\n"
                "Expected: Files with 'train' in name (e.g., train_data.parquet)\n"
                "OR: Generic parquet files (will auto-split 80/10/10)"
            )
        elif not val_files:
            print("⚠️  No validation files found")
            print("   Will create 10% validation / 10% test splits from training data")
            val_files = None  # Signal to split later

        # ═══════════════════════════════════════════════════════════════════
        # ESTIMATE DATASET SIZE AND AUTO-ENABLE STREAMING FOR LARGE DATASETS
        # ═══════════════════════════════════════════════════════════════════
        total_file_size = sum(Path(f).stat().st_size for f in parquet_files)
        # Rough estimate: ~200 bytes per example for protein sequences
        estimated_examples = total_file_size // 200

        # Auto-enable streaming for large datasets (>50M examples)
        if estimated_examples > 50_000_000 and not args.use_streaming:
            print(f"\n⚠️  LARGE DATASET DETECTED: ~{estimated_examples:,} examples")
            print(f"   Auto-enabling streaming mode to avoid OOM")
            args.use_streaming = True

        # ═══════════════════════════════════════════════════════════════════
        # STREAMING MODE: HuggingFace Trainer-compatible streaming
        # ═══════════════════════════════════════════════════════════════════
        # HuggingFace Trainer natively supports IterableDataset!
        # - No need to load full dataset into memory
        # - Tokenization happens on-the-fly
        # - Use max_steps instead of num_train_epochs
        if args.use_streaming:
            print(f"\n🌊 STREAMING MODE: HuggingFace Trainer-compatible streaming!")
            print(f"   ✓ Never loads full dataset into memory")
            print(f"   ✓ Tokenizes on-the-fly during training")
            print(f"   ✓ Training starts immediately")
            print(f"   Estimated examples: ~{estimated_examples:,}")

            hf_cache_dir = os.environ.get('HF_DATASETS_CACHE',
                                         '/opt/ml/input/data/training/.huggingface_cache/datasets')

            # Load as streaming dataset
            if val_files is None:
                # All files - split by taking portions
                stream_dataset = load_dataset(
                    'parquet',
                    data_files=train_files,
                    split='train',
                    streaming=True,
                    cache_dir=hf_cache_dir
                )

                # 80/10/10 split
                train_take = int(estimated_examples * 0.8)
                val_take = int(estimated_examples * 0.1)

                shuffled = stream_dataset.shuffle(seed=args.seed, buffer_size=10000)
                raw_streaming_train = shuffled.take(train_take)
                raw_streaming_val = shuffled.skip(train_take).take(val_take)
            else:
                raw_streaming_train = load_dataset(
                    'parquet', data_files=train_files, split='train',
                    streaming=True, cache_dir=hf_cache_dir
                ).shuffle(seed=args.seed, buffer_size=10000)

                raw_streaming_val = load_dataset(
                    'parquet', data_files=val_files, split='train',
                    streaming=True, cache_dir=hf_cache_dir
                )

            is_streaming_dataset = True
            print(f"✅ Streaming datasets ready for Trainer!")

            # Create tokenizer now (needed for streaming tokenization)
            tokenizer = create_tokenizer_for_raw_data(
                args.tokenizer_type,
                args.model_path,
                args.tokenization_max_length
            )

            # Create streaming tokenization function
            # Use max_seq_length if specified, otherwise use tokenization_max_length
            streaming_max_len = args.max_seq_length if args.max_seq_length else args.tokenization_max_length

            def streaming_tokenize(example):
                seq = concatenate_sequences_for_tokenization(example, args.tokenizer_type)
                encoded = tokenizer(
                    seq,
                    padding='max_length',
                    truncation=True,
                    max_length=streaming_max_len,
                    return_tensors=None
                )
                return encoded

            # Apply tokenization (lazy - happens during training)
            train_dataset = raw_streaming_train.map(streaming_tokenize)
            val_dataset = raw_streaming_val.map(streaming_tokenize)

            # Set max_steps based on estimated examples
            estimated_train_examples = int(estimated_examples * 0.8)
            steps_per_epoch = estimated_train_examples // args.batch_size
            total_steps = steps_per_epoch * args.num_epochs
            print(f"   Estimated steps per epoch: {steps_per_epoch:,}")
            print(f"   Total training steps: {total_steps:,}")

            # Store for later use
            dataset = {'train': train_dataset, 'validation': val_dataset}
            dataset_max_len = streaming_max_len  # Use the actual max length for streaming

            # Skip the non-streaming data loading
            # Jump to model loading section below

        # ═══════════════════════════════════════════════════════════════════
        # NON-STREAMING PATH: Load full dataset (small datasets only)
        # ═══════════════════════════════════════════════════════════════════
        else:
            is_streaming_dataset = False
            tokenizer = None  # Will be created later

        # Load as HuggingFace dataset (non-streaming)
        if not args.use_streaming and val_files is None:
            # No validation split - load all as train and split 80/10/10
            print("   Loading parquet files and creating 80/10/10 train/val/test split...")

            # Check disk space before loading
            print_disk_usage("/")
            print_disk_usage(str(raw_data_path))

            # CRITICAL: Explicitly set cache_dir to EBS volume
            # Environment variables alone are not enough - must pass cache_dir directly!
            hf_cache_dir = os.environ.get('HF_DATASETS_CACHE',
                                         '/opt/ml/input/data/training/.huggingface_cache/datasets')
            print(f"\n💾 FORCING cache_dir to: {hf_cache_dir}")

            dataset = load_dataset(
                'parquet',
                data_files=train_files,
                split='train',
                cache_dir=hf_cache_dir  # ← CRITICAL: Force cache to EBS volume
            )

            # Check disk space after loading
            print("\n   After loading parquet:")
            print_disk_usage("/")
            print_disk_usage(str(raw_data_path))

            # Split into train (80%) and temp (20%)
            print("   Creating 80/20 split...")
            print_disk_usage("/")
            split1 = dataset.train_test_split(test_size=0.2, seed=args.seed)

            print("   Creating 10/10 split from remaining 20%...")
            print_disk_usage("/")
            # Split temp (20%) into validation (10%) and test (10%)
            split2 = split1['test'].train_test_split(test_size=0.5, seed=args.seed)

            print("   Splits created")
            print_disk_usage("/")

            # Create final dataset dict
            from datasets import DatasetDict
            dataset = DatasetDict({
                'train': split1['train'],
                'validation': split2['train'],
                'test': split2['test']
            })

            print(f"   Created splits: train={len(dataset['train'])}, "
                  f"validation={len(dataset['validation'])}, test={len(dataset['test'])}")
            print(f"   Test split will be exported AFTER training completes")
        elif not args.use_streaming:
            # Explicit train/val files (non-streaming)
            # CRITICAL: Explicitly set cache_dir to EBS volume
            hf_cache_dir = os.environ.get('HF_DATASETS_CACHE',
                                         '/opt/ml/input/data/training/.huggingface_cache/datasets')
            print(f"\n💾 FORCING cache_dir to: {hf_cache_dir}")

            dataset = load_dataset(
                'parquet',
                data_files={'train': train_files, 'validation': val_files},
                cache_dir=hf_cache_dir  # ← CRITICAL: Force cache to EBS volume
            )

            # Note: test split not available with explicit files unless 'test' files exist
            if 'test' not in dataset:
                print("   Note: No test split available (only train/validation files provided)")

        # ═══════════════════════════════════════════════════════════════════
        # TOKENIZATION: Only needed for non-streaming path
        # ═══════════════════════════════════════════════════════════════════
        # For streaming: tokenization already configured above (tokenizer created there)
        # For non-streaming: create tokenizer and do parallel tokenization here

        if not is_streaming_dataset:
            # Create tokenizer (only for non-streaming - streaming creates it earlier)
            tokenizer = create_tokenizer_for_raw_data(
                args.tokenizer_type,
                args.model_path,
                args.tokenization_max_length
            )
            # Initialize CDR identifier if needed for full_tra/full_trb modes
            cdr_identifier = None
            if args.mode in ['full_tra', 'full_trb']:
                if CDR_IDENTIFIER_AVAILABLE:
                    from parsers.cdr_region_identifier import CDRRegionIdentifier
                    cdr_identifier = CDRRegionIdentifier()
                    print("✓ Initialized CDRRegionIdentifier for CDR masking")
                else:
                    print("⚠️  CDRRegionIdentifier not available")
                    print("   full_tra/full_trb modes may not work correctly")

            # Create tokenization function
            tokenize_fn = create_tokenization_function(
                tokenizer,
                args.tokenizer_type,
                cdr_identifier,
                args.tokenization_max_length
            )
            # ─────────────────────────────────────────────────────────────────
            # PARALLEL TOKENIZATION: Utilize all CPUs for maximum throughput
            # ─────────────────────────────────────────────────────────────────
            # For large datasets (938M+ examples), parallel tokenization is critical.
            # With 96 CPUs on p4d.24xlarge, we can achieve ~100k examples/s
            # instead of ~1k examples/s with single-threaded processing.
            #
            # DDP-AWARE: In distributed training, only rank 0 does tokenization
            # to prevent 8x redundant work. Other ranks wait via file-based sync
            # (NCCL barrier would timeout for long tokenization jobs).

            # Calculate estimated time
            total_examples = sum(len(dataset[split]) for split in dataset.keys())

            # File-based synchronization for DDP (avoids NCCL timeout)
            import time
            sync_file = os.path.join(
                os.environ.get('HF_DATASETS_CACHE', '/tmp'),
                '.tokenization_complete'
            )

            # Determine if we should run tokenization (only main process in DDP)
            should_tokenize = is_main_process or not is_distributed

            if is_distributed:
                if is_main_process:
                    # Remove sync file if it exists from previous run
                    if os.path.exists(sync_file):
                        os.remove(sync_file)
                        print(f"   Removed old sync file: {sync_file}")
                else:
                    # Wait briefly to ensure main process removes old sync file
                    time.sleep(5)

            if should_tokenize:
                print(f"\n⚡ Setting up PARALLEL tokenization...")
                print(f"   Workers: {args.tokenization_num_workers}")
                print(f"   Batch size: {args.tokenization_batch_size}")
                print(f"   Expected throughput: ~{args.tokenization_num_workers * 1000} examples/s")

                estimated_throughput = args.tokenization_num_workers * 1000  # ~1000 ex/s per worker
                estimated_hours = total_examples / estimated_throughput / 3600
                print(f"   Dataset size: {total_examples:,} examples")
                print(f"   Estimated time: {estimated_hours:.1f} hours")

                dataset = dataset.map(
                    tokenize_fn,
                    batched=True,
                    batch_size=args.tokenization_batch_size,
                    num_proc=args.tokenization_num_workers,  # USE ALL CPUS!
                    desc="Tokenizing",
                    load_from_cache_file=True,  # Use automatic caching
                    writer_batch_size=args.tokenization_batch_size * 10,  # Faster writes
                )

                print(f"✓ Parallel tokenization complete!")

                # Signal completion via file (for DDP sync)
                if is_distributed:
                    with open(sync_file, 'w') as f:
                        f.write('done')
                    print(f"   Signaled completion to other ranks")

            else:
                # ───────────────────────────────────────────────────────────
                # FILE-BASED SYNC: Wait for main process to finish tokenization
                # ───────────────────────────────────────────────────────────
                # Using file-based sync instead of NCCL barrier to avoid timeout
                # (tokenization can take hours, NCCL times out after 10 min)
                print(f"\n⏳ [Rank {local_rank}] Waiting for main process to tokenize...")
                print(f"   Checking for sync file: {sync_file}")

                wait_start = time.time()
                check_interval = 30  # Check every 30 seconds

                while not os.path.exists(sync_file):
                    elapsed = time.time() - wait_start
                    print(f"   [Rank {local_rank}] Still waiting... ({elapsed/60:.1f} min elapsed)")
                    time.sleep(check_interval)

                elapsed = time.time() - wait_start
                print(f"   [Rank {local_rank}] Main process finished after {elapsed/60:.1f} min")

                # Load from cache
                print(f"   [Rank {local_rank}] Loading tokenized dataset from cache...")
                dataset = dataset.map(
                    tokenize_fn,
                    batched=True,
                    batch_size=args.tokenization_batch_size,
                    num_proc=1,  # Single-threaded for cache load (fast)
                    desc=f"Loading [Rank {local_rank}]",
                    load_from_cache_file=True,
                )
                print(f"   [Rank {local_rank}] Cache loaded!")

            # Final DDP sync (short barrier - should be fast now)
            if is_distributed:
                print(f"   [Rank {local_rank}] Final sync...")
                accelerator.wait_for_everyone()
                print(f"   [Rank {local_rank}] All ranks ready!")

        # Remove raw columns to save memory (only for non-streaming datasets)
        if not is_streaming_dataset:
            columns_to_remove = [
                col for col in dataset['train'].column_names
                if col not in ['input_ids', 'attention_mask', 'permutation_key',
                              'tra_cdr1_pos', 'tra_cdr2_pos', 'tra_cdr3_pos',
                              'trb_cdr1_pos', 'trb_cdr2_pos', 'trb_cdr3_pos',
                              'tra_full', 'trb_full', 'tra', 'trb',
                              'trav_gene', 'traj_gene', 'trbv_gene', 'trbj_gene']
            ]

            if columns_to_remove:
                print(f"   Removing {len(columns_to_remove)} raw columns to save memory")
                dataset = dataset.remove_columns(columns_to_remove)

    else:
        raise ValueError(
            "Must specify either:\n"
            "  --dataset_path (for pre-tokenized data)\n"
            "  --raw_data_dir (for on-the-fly tokenization)"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Continue with existing training logic
    # ═══════════════════════════════════════════════════════════════════════

    # Handle streaming vs non-streaming datasets differently
    if is_streaming_dataset:
        # Streaming: train_dataset and val_dataset already set above
        # dataset_max_len already set from tokenization_max_length
        print(f"\n🌊 STREAMING DATASET:")
        print(f"   Type: IterableDataset (no len() available)")
        print(f"   Max sequence length: {dataset_max_len}")
        print(f"   Estimated training examples: ~{estimated_train_examples:,}")

        # For test mode with streaming, limit steps instead of examples
        if args.test:
            print("\n⚠️  Running in TEST mode - limiting to 100 steps")
            total_steps = 100  # Override total_steps for test mode

        # For streaming, truncation is applied lazily in the tokenization function
        if args.max_seq_length is not None:
            print(f"\n✂️  Truncation to {args.max_seq_length} applied during streaming tokenization")
            dataset_max_len = args.max_seq_length

        # Limit validation set for streaming using take()
        if args.max_eval_samples is not None:
            print(f"\n⚠️  Limiting validation set to {args.max_eval_samples:,} examples")
            val_dataset = val_dataset.take(args.max_eval_samples)

        print(f"\n📊 Dataset info:")
        print(f"   Training: ~{estimated_train_examples:,} examples (streaming)")
        print(f"   Validation: streaming (size determined at runtime)")
        print(f"   Total training steps: {total_steps:,}")
        print(f"   Eval batch size: {args.eval_batch_size}")
    else:
        # Non-streaming: standard dataset handling
        print(f"\nDataset splits: {list(dataset.keys())}")
        if hasattr(dataset['train'], 'column_names'):
            print(f"Dataset columns: {dataset['train'].column_names}")

        # Filter datasets by mode (skip if dataset was pre-filtered)
        if args.skip_mode_filter:
            print(f"⏭️  Skipping mode filtering (dataset already filtered during tokenization)")
            train_dataset = dataset["train"]
            val_dataset = dataset["validation"]
        else:
            train_dataset = filter_dataset_by_mode(dataset["train"], args.mode)
            val_dataset = filter_dataset_by_mode(dataset["validation"], args.mode)

        # Use smaller subset for testing
        if args.test:
            print("\n⚠️  Running in TEST mode with reduced dataset size")
            train_dataset = train_dataset.shuffle(seed=args.seed).select(range(min(1000, len(train_dataset))))
            val_dataset = val_dataset.shuffle(seed=args.seed).select(range(min(200, len(val_dataset))))

        # Check maximum sequence length in dataset
        max_train_len = max(len(x['input_ids']) for x in train_dataset.select(range(min(1000, len(train_dataset)))))
        max_val_len = max(len(x['input_ids']) for x in val_dataset.select(range(min(1000, len(val_dataset)))))
        dataset_max_len = max(max_train_len, max_val_len)

        # Truncate sequences if max_seq_length is specified (saves memory)
        if args.max_seq_length is not None:
            print(f"\n✂️  Truncating sequences to max length: {args.max_seq_length}")

            def truncate_sequence(example):
                if len(example['input_ids']) > args.max_seq_length:
                    example['input_ids'] = example['input_ids'][:args.max_seq_length]
                    if 'attention_mask' in example:
                        example['attention_mask'] = example['attention_mask'][:args.max_seq_length]
                    if 'labels' in example:
                        example['labels'] = example['labels'][:args.max_seq_length]
                return example

            train_dataset = train_dataset.map(truncate_sequence, desc="Truncating training sequences")
            val_dataset = val_dataset.map(truncate_sequence, desc="Truncating validation sequences")
            dataset_max_len = args.max_seq_length

        # Limit validation set size if specified
        if args.max_eval_samples is not None and len(val_dataset) > args.max_eval_samples:
            print(f"\n⚠️  Limiting validation set from {len(val_dataset):,} to {args.max_eval_samples:,} examples")
            val_dataset = val_dataset.shuffle(seed=args.seed).select(range(args.max_eval_samples))

        print(f"\n📊 Dataset sizes:")
        print(f"   Training: {len(train_dataset):,} examples")
        print(f"   Validation: {len(val_dataset):,} examples")
        print(f"   Eval batch size: {args.eval_batch_size}")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Load model and tokenizer
    # ─────────────────────────────────────────────────────────────────────────────
    
    lora_config = {
        "r": args.lora_r,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
        "target_modules": args.lora_target_modules,
    }
    
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        use_lora=args.use_lora,
        lora_config=lora_config,
        enable_gradient_checkpointing=args.gradient_checkpointing,
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # Model-size-aware memory optimizations
    # ─────────────────────────────────────────────────────────────────────────────

    # Count model parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_params_m = num_params / 1e6
    print(f"\n📊 Model size: {num_params_m:.0f}M parameters")

    # Auto-scale eval_accumulation_steps based on model size
    # Prevents OOM during evaluation by moving predictions to CPU more frequently
    if args.eval_accumulation_steps == 10:  # Using default value
        if num_params_m < 50:  # Small models (8M, 35M)
            default_eval_accum = 10
        elif num_params_m < 200:  # Medium models (150M)
            default_eval_accum = 20
        elif num_params_m < 1000:  # Large models (650M)
            default_eval_accum = 50
        else:  # Very large (3B+)
            default_eval_accum = 100

        if args.eval_accumulation_steps != default_eval_accum:
            args.eval_accumulation_steps = default_eval_accum
            print(f"🔧 Auto-adjusted eval_accumulation_steps to {default_eval_accum} based on model size")

    # Auto-apply memory optimizations for large models (650M+)
    if num_params_m > 500:
        print(f"\n🔧 AUTO-OPTIMIZATION: Large model detected ({num_params_m:.0f}M params)")

        if not args.gradient_checkpointing:
            args.gradient_checkpointing = True
            print("   → Enabling gradient checkpointing (saves ~30-40% memory)")

        if args.batch_size > 4:
            print(f"   ⚠️  WARNING: Batch size {args.batch_size} may be too large for this model!")
            print(f"   → Recommend: --batch_size 4 or lower")
            print(f"   → Current effective batch: {args.batch_size * args.gradient_accumulation_steps}")

        if args.eval_batch_size > 4:
            old_eval_batch = args.eval_batch_size
            args.eval_batch_size = 4
            print(f"   → Reducing eval batch size from {old_eval_batch} to 4")

        print()

    # ─────────────────────────────────────────────────────────────────────────────
    # Check model's max position embeddings vs dataset max length
    # ─────────────────────────────────────────────────────────────────────────────
    
    model_max_length = None
    if hasattr(model.config, 'max_position_embeddings'):
        model_max_length = model.config.max_position_embeddings
        print(f"\n📐 Model max position embeddings: {model_max_length}")
    elif hasattr(model, 'esm3'):  # ESM3 model
        model_max_length = 1024  # ESM3 supports up to 1024
        print(f"\n📐 ESM3 model max length: {model_max_length}")
    
    if model_max_length is not None and dataset_max_len > model_max_length:
        error_msg = (
            f"\n{'='*80}\n"
            f"❌ FATAL ERROR: Dataset sequences exceed model's maximum length!\n"
            f"{'='*80}\n"
            f"   Dataset max length: {dataset_max_len} tokens\n"
            f"   Model max length:   {model_max_length} tokens\n"
            f"   Difference:         {dataset_max_len - model_max_length} tokens over limit\n\n"
            f"This will cause a RuntimeError during training/evaluation.\n\n"
            f"💡 SOLUTIONS (choose one):\n"
            f"   1. Add --max_seq_length {model_max_length} to truncate sequences\n"
            f"   2. Use a model with longer context:\n"
            f"      - ESM2 models support up to 1024 tokens\n"
            f"      - ESM3 models support up to 1024 tokens\n"
            f"      Example: --model_path facebook/esm2_t12_35M_UR50D\n\n"
            f"{'='*80}\n"
        )
        print(error_msg)
        raise RuntimeError(f"Dataset sequences ({dataset_max_len}) exceed model max length ({model_max_length})")
    elif model_max_length is not None and dataset_max_len > model_max_length * 0.9:
        print(f"\n⚠️  WARNING: Dataset sequences are close to model's maximum!")
        print(f"   Dataset max: {dataset_max_len} / Model max: {model_max_length}")
        print(f"   Recommend using --max_seq_length {int(model_max_length * 0.95)} for safety")
    else:
        print(f"\n✅ Dataset sequence lengths are compatible with model")
        print(f"   Dataset max: {dataset_max_len} / Model max: {model_max_length}")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Create data collator with task-specific masking
    # ─────────────────────────────────────────────────────────────────────────────
    
    if args.use_pre_masked:
        # Dataset already has 'labels' column - just use simple padding
        print(f"\n✅ Using PRE-MASKED dataset (much faster!)")
        print(f"   Masking was done offline with mode: {args.mode}")

        # Check if labels column exists (only for non-streaming datasets)
        if not is_streaming_dataset and hasattr(train_dataset, 'column_names'):
            if "labels" not in train_dataset.column_names:
                raise ValueError(
                    "Dataset does not have 'labels' column. "
                    "Please pre-mask the dataset using scripts/data_processing/pre_mask_dataset.py"
                )

        from transformers import default_data_collator
        # Use the simplest collator - just pads to max length in batch
        data_collator = default_data_collator
    else:
        # On-the-fly masking (slower but more flexible)
        print(f"\n⚠️  Using ON-THE-FLY masking (slower)")
        print(f"   Masking strategy: {args.mode}")
        print(f"   Consider pre-masking the dataset for faster training!")
        
        data_collator = TaskSpecificMaskingCollator(
            tokenizer=tokenizer,
            mode=args.mode,
            mlm_probability=args.mlm_probability,
            cdr3_mask_length=args.cdr3_mask_length,
        )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Setup training arguments
    # ─────────────────────────────────────────────────────────────────────────────

    # Resolve checkpoint directory for environment (local vs SageMaker)
    checkpoint_dir = resolve_path_for_environment(args.output_dir, 'checkpoint')
    os.makedirs(checkpoint_dir, exist_ok=True)

    if is_sagemaker_environment():
        print(f"\n💾 Checkpoint Configuration:")
        print(f"   Directory: {checkpoint_dir}")
        print(f"   (Auto-synced to S3 for Spot Training)")

    # Optimize checkpoint frequency for SageMaker Spot Training
    save_strategy = "epoch" if args.save_steps is None else "steps"
    save_steps = args.save_steps
    save_total_limit = args.save_total_limit

    # Eval strategy must match save strategy when load_best_model_at_end=True
    eval_strategy = "epoch" if args.eval_steps is None else "steps"
    eval_steps = args.eval_steps

    if is_sagemaker_environment():
        # Save more frequently for Spot resilience
        if save_steps is None or save_steps > 500:
            save_steps = 500
            save_strategy = 'steps'
            # Match eval strategy to save strategy
            if eval_steps is None:
                eval_steps = save_steps  # Evaluate at same frequency as saving
                eval_strategy = 'steps'
            print(f"\n💾 Checkpointing every {save_steps} steps (Spot resilience)")

        # Keep more checkpoints (S3 storage is cheap)
        if save_total_limit < 5:
            save_total_limit = 5
            print(f"   Keeping last {save_total_limit} checkpoints")

    # Disable compute_metrics for BERT (but not ProtBERT) to save memory during evaluation
    # (BERT's 29k vocab creates huge prediction tensors)
    model_path_lower = args.model_path.lower()
    is_bert = "bert" in model_path_lower
    is_protbert = "prot_bert" in model_path_lower or "protbert" in model_path_lower
    use_metrics = not is_bert or is_protbert  # Disable for BERT, but keep for ProtBERT

    if not use_metrics:
        print("\n⚠️  Note: Using custom streaming evaluation for BERT")
        print("   Metrics (accuracy/perplexity) will be computed efficiently in streaming mode")
        print("   (BERT's large vocabulary is handled without storing all predictions)\n")

    # Use loss for BERT (no metrics), accuracy for others
    best_metric = "loss" if not use_metrics else "accuracy"
    metric_greater_is_better = False if not use_metrics else True

    # For streaming datasets, use max_steps instead of epochs
    # (epochs don't make sense for IterableDataset with unknown length)
    training_epochs = None if is_streaming_dataset else args.num_epochs
    training_max_steps = total_steps if is_streaming_dataset else -1

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,  # Use checkpoint_dir instead of args.output_dir
        num_train_epochs=training_epochs if training_epochs else 1,  # Trainer requires at least 1
        max_steps=training_max_steps,  # For streaming: use max_steps; for non-streaming: -1 (disabled)
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,

        # Evaluation and saving (strategies must match when load_best_model_at_end=True)
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,  # Increased for SageMaker Spot training
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        greater_is_better=metric_greater_is_better,
        
        # Logging
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        report_to="wandb" if args.wandb_project else "none",
        
        # Optimization
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        lr_scheduler_type="cosine",
        optim=args.optim,  # Can be adamw_8bit for memory savings
        
        # Memory optimization & Data Loading
        # Use 8+ workers for multi-GPU to prevent data loading bottlenecks
        dataloader_num_workers=8,  # At least 1 per GPU for 8-GPU instances like p4d.24xlarge
        dataloader_pin_memory=True,  # Enable for faster CPU->GPU transfer (disable if OOM)
        dataloader_drop_last=True,  # Drop incomplete batches to avoid DDP hangs
        auto_find_batch_size=False,  # Don't auto-adjust, use user's settings
        eval_accumulation_steps=args.eval_accumulation_steps,  # CRITICAL: Prevents OOM with large vocab models like BERT

        # Other
        seed=args.seed,
        dataloader_prefetch_factor=2,  # Prefetch 2 batches per worker for better throughput
        remove_unused_columns=True,  # Remove extra columns like permutation_key
        label_names=["labels"],
        include_num_input_tokens_seen=False,
        
        # Disable unnecessary logging
        log_level="warning",
        disable_tqdm=False,
        skip_memory_metrics=True,
    )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Create Trainer
    # ─────────────────────────────────────────────────────────────────────────────

    trainer = StreamingEvalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics if use_metrics else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if use_metrics else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Detect checkpoint for resumption (Local + SageMaker Spot Training)
    # ─────────────────────────────────────────────────────────────────────────────

    # SageMaker Spot Training: Auto-enable resume if checkpoints exist
    if is_sagemaker_environment() and args.resume_from_checkpoint is None:
        if os.path.isdir(checkpoint_dir):
            existing_checkpoints = [
                d for d in os.listdir(checkpoint_dir)
                if d.startswith("checkpoint-") and
                   os.path.isdir(os.path.join(checkpoint_dir, d))
            ]
            if existing_checkpoints:
                print(f"\n🔄 SPOT RECOVERY: Found {len(existing_checkpoints)} checkpoint(s)")
                print(f"   Automatically enabling resumption")
                args.resume_from_checkpoint = "auto"

    resume_checkpoint = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint.lower() == "auto":
            # Auto-detect latest checkpoint
            checkpoints = []
            if os.path.isdir(checkpoint_dir):
                for dirname in os.listdir(checkpoint_dir):
                    if dirname.startswith("checkpoint-"):
                        checkpoint_path = os.path.join(checkpoint_dir, dirname)
                        if os.path.isdir(checkpoint_path):
                            checkpoints.append(checkpoint_path)

            if checkpoints:
                # Sort by checkpoint number to get the latest
                checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
                resume_checkpoint = checkpoints[-1]
                print(f"\n🔄 AUTO-RESUME: Found {len(checkpoints)} checkpoint(s)")
                print(f"   Resuming from latest: {resume_checkpoint}")
            else:
                print(f"\n⚠️  AUTO-RESUME: No checkpoints found in {checkpoint_dir}")
                print(f"   Starting training from scratch")
        else:
            # Use specified checkpoint
            resume_checkpoint = args.resume_from_checkpoint
            if os.path.isdir(resume_checkpoint):
                print(f"\n🔄 RESUME: Using checkpoint: {resume_checkpoint}")
            else:
                print(f"\n⚠️  WARNING: Checkpoint not found: {resume_checkpoint}")
                print(f"   Starting training from scratch")
                resume_checkpoint = None

    # ─────────────────────────────────────────────────────────────────────────────
    # Train
    # ─────────────────────────────────────────────────────────────────────────────

    print("\n" + "="*80)
    print("Starting training...")
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
    print("="*80 + "\n")

    # Clear CUDA cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🔧 Cleared CUDA cache before training")

    try:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    except torch.cuda.OutOfMemoryError as e:
        print("\n" + "="*80)
        print("❌ CUDA OUT OF MEMORY ERROR")
        print("="*80)
        print(f"\nError: {e}\n")
        print("💡 IMMEDIATE SOLUTIONS:")
        print("   1. Reduce batch size further (current: {})".format(args.batch_size))
        print("   2. Add --gradient_checkpointing if not already enabled")
        print("   3. Add --fp16 for half precision training")
        print("   4. Reduce --max_seq_length to 256 or 128")
        print("   5. Use LoRA: --use_lora --lora_r 8")
        print("\n   Try restarting with smaller batch size:")
        print(f"   --batch_size 1 --gradient_accumulation_steps 16 --gradient_checkpointing --fp16")
        print("="*80 + "\n")
        raise
    
    # Clear cache after training before evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\n🔧 Cleared CUDA cache after training")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Save best model
    # ─────────────────────────────────────────────────────────────────────────────

    print("\n" + "="*80)
    print("Training complete! Saving best model...")
    print("="*80 + "\n")

    # Determine final model directory based on environment
    if is_sagemaker_environment():
        sm_paths = get_sagemaker_paths()
        best_model_dir = sm_paths['model_dir']  # /opt/ml/model

        print(f"📦 SageMaker Model Packaging:")
        print(f"   Model saved to: {best_model_dir}")
        print(f"   SageMaker will automatically tar and upload to S3")

        # Also save backup copy in checkpoints
        checkpoint_copy = os.path.join(checkpoint_dir, "best_model")
    else:
        best_model_dir = os.path.join(args.output_dir, "best_model")
        checkpoint_copy = None

    # Save model
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    print(f"✅ Best model saved to: {best_model_dir}")

    # Save backup copy in checkpoints if SageMaker
    if checkpoint_copy:
        print(f"   Backup copy: {checkpoint_copy}")
        trainer.save_model(checkpoint_copy)
        tokenizer.save_pretrained(checkpoint_copy)

    # ─────────────────────────────────────────────────────────────────────────────
    # Export test split (after training, before evaluation)
    # ─────────────────────────────────────────────────────────────────────────────

    if 'test' in dataset and is_main_process:
        print(f"\n{'='*80}")
        print("💾 EXPORTING TEST SPLIT (Post-Training)")
        print(f"{'='*80}")

        # Determine save location
        if is_sagemaker_environment():
            sm_paths = get_sagemaker_paths()
            test_split_dir = os.path.join(sm_paths['output_dir'], 'test_split')
        else:
            test_split_dir = os.path.join(args.output_dir, 'test_split')

        os.makedirs(test_split_dir, exist_ok=True)

        # Export raw test split (parquet)
        test_split_raw_path = os.path.join(test_split_dir, 'test_raw.parquet')
        try:
            dataset['test'].to_parquet(test_split_raw_path)
            print(f"✓ Saved raw test split: {test_split_raw_path}")
            print(f"  Rows: {len(dataset['test'])}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save raw test split: {e}")

        # Export tokenized test split (HuggingFace dataset format)
        test_tokenized_path = os.path.join(test_split_dir, 'test_tokenized')
        try:
            dataset['test'].save_to_disk(test_tokenized_path)
            print(f"✓ Saved tokenized test split: {test_tokenized_path}")
            print(f"  Columns: {dataset['test'].column_names}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save tokenized test split: {e}")

        if is_sagemaker_environment():
            print(f"\n📤 Test splits will be uploaded to S3:")
            print(f"   Raw parquet:  s3://<output-bucket>/.../test_split/test_raw.parquet")
            print(f"   Tokenized:    s3://<output-bucket>/.../test_split/test_tokenized/")

        print(f"{'='*80}\n")

    # ─────────────────────────────────────────────────────────────────────────────
    # Final evaluation and logging
    # ─────────────────────────────────────────────────────────────────────────────
    
    # Clear cache before evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\n🔧 Cleared CUDA cache before final evaluation")
    
    print("\nRunning final evaluation...")
    print(f"  Evaluating on {len(val_dataset):,} validation examples")

    try:
        # Reset metrics tracker before evaluation
        if use_metrics:
            metrics_tracker.reset()

        eval_results = trainer.evaluate()
        
        print("\nFinal Evaluation Results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value:.4f}")
        
        if args.wandb_project and is_main_process:
            wandb.log({"final_eval": eval_results})
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n⚠️  Warning: Final evaluation failed with OOM error")
        print(f"   This is unexpected with streaming evaluation - check eval batch size.")
        print(f"   Training has completed successfully - the best model has been saved.")
        print(f"\n   To evaluate separately, use:")
        print(f"   python scripts/inference/evaluate_mlm.py \\")
        print(f"       --model_path {best_model_dir} \\")
        print(f"       --dataset_path {args.dataset_path} \\")
        print(f"       --batch_size 4")
    except Exception as e:
        print(f"\n⚠️  Warning: Final evaluation failed with error: {e}")
        print("This is likely due to memory constraints with large validation sets.")
        print("Training has completed successfully - the best model has been saved.")
        import traceback
        traceback.print_exc()
    
    if args.wandb_project and is_main_process:
        # Log prediction examples only if requested
        if args.log_prediction_examples:
            print("\nGenerating prediction examples for W&B...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                log_prediction_examples(
                    model=trainer.model,
                    tokenizer=tokenizer,
                    dataset=val_dataset,
                    num_examples=50,
                    device=device
                )
            except Exception as e:
                print(f"⚠️  Warning: Failed to log prediction examples: {e}")
        
        wandb.finish()
    
    print("\n✅ Fine-tuning complete!")


if __name__ == "__main__":
    main()
