#!/usr/bin/env python3
"""
fine_tune.py
────────────────────────────────────────────────────────
Unified fine-tuning script with task-specific data filtering and masking strategies.
Uses HuggingFace Accelerate for distributed training and PEFT for LoRA.

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
from datasets import load_from_disk
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

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
        mode: One of ['mlm', 'tra', 'trb', 'tra_trb_pairing', 'tcr_mhc', 'peptide_mhc', 'specificity']
    
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
        
    elif mode == "tra_trb_pairing":
        # Examples containing both TRA and TRB
        filtered = dataset.filter(lambda x: "tra" in x["permutation_key"] and "trb" in x["permutation_key"])
        
    elif mode == "tcr_mhc":
        # At least one TCR chain (tra/trb) AND at least one MHC (mhc_one/mhc_two)
        filtered = dataset.filter(lambda x: (
            ("tra" in x["permutation_key"] or "trb" in x["permutation_key"]) and
            ("mhc_one" in x["permutation_key"] or "mhc_two" in x["permutation_key"])
        ))
        
    elif mode == "peptide_mhc":
        # Peptide AND at least one MHC
        filtered = dataset.filter(lambda x: (
            "peptide" in x["permutation_key"] and
            ("mhc_one" in x["permutation_key"] or "mhc_two" in x["permutation_key"])
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
            mode: Masking strategy mode
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
    
    def _is_special_token(self, token_id: int) -> bool:
        """Check if token is a special token."""
        if token_id == self.pad_token_id:
            return True
        if self.cls_token_id is not None and token_id == self.cls_token_id:
            return True
        if self.sep_token_id is not None and token_id == self.sep_token_id:
            return True
        
        # Check for special tokens by decoding
        token = self.tokenizer.decode([token_id])
        if token.startswith('[') and token.endswith(']'):
            return True
        if token in ['<s>', '</s>', '<pad>', '<unk>']:
            return True
        
        return False
    
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
        """Find positions of separator tokens ([SEP], [ETRA], [ETRB], etc.)."""
        separators = []
        for i, token_id in enumerate(input_ids):
            token = self.tokenizer.decode([token_id])
            if '[SEP]' in token or token in ['[ETRA]', '[ETRB]', '[EPEP]', '[EMHO]', '[EMHT]']:
                separators.append(i)
        return separators
    
    # ─────────────────────────────────────────────────────────────────────────────
    # MLM Masking (15% random)
    # ─────────────────────────────────────────────────────────────────────────────
    
    def _mask_mlm(self, input_ids_list: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
        """Standard MLM masking: 15% of tokens randomly."""
        masked_inputs = []
        labels = []
        
        for input_ids in input_ids_list:
            input_ids = input_ids.copy() if isinstance(input_ids, list) else input_ids.tolist()
            label_ids = [-100] * len(input_ids)
            
            for i, token_id in enumerate(input_ids):
                if self._is_special_token(token_id):
                    continue
                
                if random.random() < self.mlm_probability:
                    label_ids[i] = input_ids[i]
                    
                    # 80% mask, 10% random, 10% keep
                    prob = random.random()
                    if prob < 0.8:
                        input_ids[i] = self.mask_token_id
                    elif prob < 0.9:
                        input_ids[i] = random.randint(0, len(self.tokenizer) - 1)
                    # else: keep original
            
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
        Mask the middle portion of CDR3 region.
        For simplicity, we mask the middle 5 amino acids of the sequence.
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
                
                # Mask the middle region
                for i in range(middle_start, middle_end):
                    if i < seq_end and not self._is_special_token(input_ids[i]):
                        label_ids[i] = input_ids[i]
                        input_ids[i] = self.mask_token_id
            
            masked_inputs.append(input_ids)
            labels.append(label_ids)
        
        return masked_inputs, labels
    
    # ─────────────────────────────────────────────────────────────────────────────
    # TRA-TRB Pairing: Mask entire first chain
    # ────────────────────────────────────────────────��────────────────────────────
    
    def _mask_first_chain(self, input_ids_list: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
        """Mask the entire first chain (before first separator)."""
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
            
            for i in range(seq_start, mask_end):
                if not self._is_special_token(input_ids[i]):
                    label_ids[i] = input_ids[i]
                    input_ids[i] = self.mask_token_id
            
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
                
                for i in range(seq_start, mask_end):
                    if not self._is_special_token(input_ids[i]):
                        label_ids[i] = input_ids[i]
                        input_ids[i] = self.mask_token_id
            
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
        """Mask first molecule, or both MHC chains if first two are MHC."""
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
                
                for i in range(seq_start, mask_end):
                    if not self._is_special_token(input_ids[i]):
                        label_ids[i] = input_ids[i]
                        input_ids[i] = self.mask_token_id
            
            masked_inputs.append(input_ids)
            labels.append(label_ids)
        
        return masked_inputs, labels
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Specificity: Mask first molecule
    # ─────────────────────────────────────────────────────────────────────────────
    
    def _mask_first_molecule(self, input_ids_list: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
        """Mask only the first molecule in the sequence."""
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
            
            for i in range(seq_start, mask_end):
                if not self._is_special_token(input_ids[i]):
                    label_ids[i] = input_ids[i]
                    input_ids[i] = self.mask_token_id
            
            masked_inputs.append(input_ids)
            labels.append(label_ids)
        
        return masked_inputs, labels
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Batch creation with padding
    # ─────────────────────────────────────────────────────────────────────────────
    
    def _create_batch(
        self,
        masked_inputs: List[List[int]],
        labels: List[List[int]]
    ) -> Dict[str, torch.Tensor]:
        """Convert lists to padded tensors."""
        
        # Pad sequences
        max_len = max(len(seq) for seq in masked_inputs)
        
        padded_inputs = []
        padded_labels = []
        attention_masks = []
        
        for inp, lab in zip(masked_inputs, labels):
            pad_len = max_len - len(inp)
            
            padded_inputs.append(inp + [self.pad_token_id] * pad_len)
            padded_labels.append(lab + [-100] * pad_len)
            attention_masks.append([1] * len(inp) + [0] * pad_len)
        
        return {
            "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. METRICS AND LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    """Compute accuracy and perplexity for evaluation."""
    accuracy_metric = evaluate.load("accuracy")
    
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    
    # Mask out -100 labels
    mask = labels != -100
    
    # Calculate accuracy on masked positions
    preds = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(
        predictions=preds[mask],
        references=labels[mask]
    )["accuracy"]
    
    # Calculate loss and perplexity using numpy to avoid OOM issues
    # Convert to float64 for numerical stability
    logits_masked = logits[mask].astype(np.float64)
    labels_masked = labels[mask].astype(np.int64)
    
    # Compute cross-entropy loss manually using numpy (more memory efficient)
    # Apply log-softmax
    logits_max = np.max(logits_masked, axis=-1, keepdims=True)
    logits_shifted = logits_masked - logits_max
    log_sum_exp = np.log(np.sum(np.exp(logits_shifted), axis=-1, keepdims=True))
    log_probs = logits_shifted - log_sum_exp
    
    # Get log probability of correct class
    nll = -log_probs[np.arange(len(labels_masked)), labels_masked]
    loss = np.mean(nll)
    
    perplexity = math.exp(loss) if loss < 100 else float('inf')
    
    return {
        "accuracy": accuracy,
        "perplexity": perplexity,
    }


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
# 4. MODEL LOADING
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
# 5. MAIN TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune protein language models with task-specific masking"
    )
    
    # Data arguments
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to HuggingFace dataset directory")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["mlm", "tra", "trb", "tra_trb_pairing", "tcr_mhc", "peptide_mhc", "specificity"],
                        help="Fine-tuning mode (determines data filtering and masking strategy)")
    
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
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Maximum number of evaluation samples to use (useful for large validation sets)")
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
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Initialize W&B
    # ─────────────────────────────────────────────────────────────────────────────
    
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            settings=wandb.Settings(
                # Disable system metrics to reduce clutter
                _disable_stats=True,
                _disable_meta=True,
            )
        )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Load and filter dataset
    # ─────────────────────────────────────────────────────────────────────────────
    
    print(f"\nLoading dataset from: {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    
    print(f"Dataset splits: {list(dataset.keys())}")
    print(f"Dataset columns: {dataset['train'].column_names}")
    
    # Filter datasets by mode
    train_dataset = filter_dataset_by_mode(dataset["train"], args.mode)
    val_dataset = filter_dataset_by_mode(dataset["validation"], args.mode)
    
    # Use smaller subset for testing
    if args.test:
        print("\n⚠️  Running in TEST mode with reduced dataset size")
        train_dataset = train_dataset.shuffle(seed=args.seed).select(range(min(1000, len(train_dataset))))
        val_dataset = val_dataset.shuffle(seed=args.seed).select(range(min(200, len(val_dataset))))
    
    # Check maximum sequence length in dataset
    print("\n📏 Checking dataset sequence lengths...")
    max_train_len = max(len(x['input_ids']) for x in train_dataset.select(range(min(1000, len(train_dataset)))))
    max_val_len = max(len(x['input_ids']) for x in val_dataset.select(range(min(1000, len(val_dataset)))))
    dataset_max_len = max(max_train_len, max_val_len)
    print(f"   Max length in sample (train): {max_train_len}")
    print(f"   Max length in sample (val): {max_val_len}")
    print(f"   Dataset max length: {dataset_max_len}")
    
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
    
    # Auto-adjust eval batch size AND limit eval set for very large validation sets to prevent OOM
    if len(val_dataset) > 50000:
        if args.max_eval_samples is None:
            # Automatically limit to 50k examples for very large validation sets
            old_val_size = len(val_dataset)
            args.max_eval_samples = 50000
            val_dataset = val_dataset.shuffle(seed=args.seed).select(range(args.max_eval_samples))
            print(f"\n⚠️  Very large validation set detected ({old_val_size:,} examples)")
            print(f"   Automatically limiting to {args.max_eval_samples:,} examples to prevent OOM during evaluation")
        
        if args.eval_batch_size > 4:
            old_eval_batch = args.eval_batch_size
            args.eval_batch_size = 4
            print(f"   Also reducing eval batch size from {old_eval_batch} to {args.eval_batch_size}")
    
    # Log dataset statistics
    train_stats = log_dataset_statistics(train_dataset, "train", log_to_wandb=args.wandb_project is not None)
    val_stats = log_dataset_statistics(val_dataset, "validation", log_to_wandb=args.wandb_project is not None)
    
    if args.wandb_project:
        wandb.log(train_stats)
        wandb.log(val_stats)
    
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
        
        # Check if labels column exists
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

    # Disable compute_metrics for BERT to save memory during evaluation
    # (BERT's 29k vocab creates huge prediction tensors)
    use_metrics = "bert" not in args.model_path.lower()

    if not use_metrics:
        print("\n⚠️  Disabling accuracy/perplexity metrics for BERT to save memory")
        print("   Will use validation loss as best model metric instead")
        print("   (BERT's 29k vocabulary creates huge prediction tensors)\n")

    # Use loss for BERT (no metrics), accuracy for others
    best_metric = "loss" if not use_metrics else "accuracy"
    metric_greater_is_better = False if not use_metrics else True

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,

        # Evaluation and saving
        eval_strategy="epoch" if args.eval_steps is None else "steps",
        eval_steps=args.eval_steps,
        save_strategy="epoch" if args.save_steps is None else "steps",
        save_steps=args.save_steps,
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
        
        # Memory optimization
        dataloader_pin_memory=True,  # Enable for faster CPU->GPU transfer (disable if OOM)
        auto_find_batch_size=False,  # Don't auto-adjust, use user's settings
        eval_accumulation_steps=args.eval_accumulation_steps,  # CRITICAL: Prevents OOM with large vocab models like BERT

        # Other
        seed=args.seed,
        dataloader_num_workers=4,  # Increased for faster data loading with pre-masked data
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics if use_metrics else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Train
    # ─────────────────────────────────────────────────────────────────────────────
    
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    # Clear CUDA cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🔧 Cleared CUDA cache before training")
    
    try:
        trainer.train()
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
    
    best_model_dir = os.path.join(args.output_dir, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    
    print(f"✅ Best model saved to: {best_model_dir}")
    
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
        eval_results = trainer.evaluate()
        
        print("\nFinal Evaluation Results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value:.4f}")
        
        if args.wandb_project:
            wandb.log({"final_eval": eval_results})
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n⚠️  Warning: Final evaluation failed with OOM error")
        print(f"   This is common with large validation sets.")
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
    
    if args.wandb_project:
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
