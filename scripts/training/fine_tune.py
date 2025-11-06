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
from peft import get_peft_model, LoraConfig, PeftModel
from accelerate import Accelerator
from typing import Any, Dict, List, Optional, Tuple
from tqdm.auto import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
    
    # Calculate loss and perplexity
    loss_fct = torch.nn.CrossEntropyLoss()
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    loss = loss_fct(
        logits_tensor.view(-1, logits_tensor.size(-1)),
        labels_tensor.view(-1)
    ).item()
    
    perplexity = math.exp(loss) if loss < 100 else float('inf')
    
    return {
        "accuracy": accuracy,
        "perplexity": perplexity,
    }


def log_dataset_statistics(dataset: Any, split_name: str) -> Dict[str, Any]:
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
        f"{split_name}_permutation_keys": pkey_counts_str,
    }
    
    print(f"\n{split_name.upper()} Dataset Statistics:")
    print(f"  Total examples: {num_examples:,}")
    if pkey_counts_str:
        print(f"  Permutation key distribution:")
        for key, count in sorted(pkey_counts_str.items(), key=lambda x: x[1], reverse=True):
            print(f"    {key}: {count:,} ({count/num_examples*100:.1f}%)")
    
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(
    model_path: str,
    use_lora: bool = False,
    lora_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any]:
    """
    Load model and tokenizer, optionally applying LoRA.
    
    Args:
        model_path: Path to model (local or HuggingFace Hub)
        use_lora: Whether to apply LoRA
        lora_config: LoRA configuration dict
    
    Returns:
        (model, tokenizer)
    """
    print(f"Loading model and tokenizer from: {model_path}")
    
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
        # Load regular model
        model = AutoModelForMaskedLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Apply LoRA if requested
    if use_lora:
        print("Applying LoRA configuration...")
        
        lora_cfg = LoraConfig(
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("alpha", 32),
            target_modules=lora_config.get("target_modules", ["query", "value"]),
            lora_dropout=lora_config.get("dropout", 0.05),
            bias="none",
        )
        
        model = get_peft_model(model, lora_cfg)
        
        # Ensure MLM head is trainable
        for name, param in model.named_parameters():
            if "cls" in name or "lm_head" in name:
                param.requires_grad = True
        
        model.print_trainable_parameters()
    
    # Enable gradient checkpointing for memory savings
    # Note: Can slow down training but essential for large models on limited GPU memory
    if hasattr(model, "gradient_checkpointing_enable"):
        print("Enabling gradient checkpointing for memory efficiency...")
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
                        help="Path to pre-trained model (local or HuggingFace Hub)")
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
                        help="Target modules for LoRA")
    
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
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Initialize W&B
    # ─────────────────────────────────────────────────────────────────────────────
    
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
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
    
    # Log dataset statistics
    train_stats = log_dataset_statistics(train_dataset, "train")
    val_stats = log_dataset_statistics(val_dataset, "validation")
    
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
        lora_config=lora_config
    )
    
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
        metric_for_best_model="accuracy",
        greater_is_better=True,
        
        # Logging
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        report_to="wandb" if args.wandb_project else "none",
        
        # Optimization
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=False,  # Disabled for LoRA compatibility
        lr_scheduler_type="cosine",
        
        # Other
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=True,  # Remove extra columns like permutation_key
        label_names=["labels"],
        include_num_input_tokens_seen=False,
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
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Train
    # ─────────────────────────────────────────────────────────────────────────────
    
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    trainer.train()
    
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
    
    print("\nRunning final evaluation...")
    eval_results = trainer.evaluate()
    
    print("\nFinal Evaluation Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    if args.wandb_project:
        wandb.log({"final_eval": eval_results})
        wandb.finish()
    
    print("\n✅ Fine-tuning complete!")


if __name__ == "__main__":
    main()
