#!/usr/bin/env python3
"""
pre_mask_dataset.py
────────────────────────────────────────────────────────
Pre-mask datasets for fine-tuning using multiprocessing.
This is MUCH faster than on-the-fly masking during training.

Usage:
    python scripts/data_processing/pre_mask_dataset.py \
        --input_dataset /path/to/input/dataset \
        --output_dataset /path/to/output/dataset \
        --mode mlm \
        --num_proc 32
"""

import argparse
import random
import os
from typing import Dict, List, Any
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer
from tqdm.auto import tqdm

# Try to import ESM3
try:
    from esm.tokenization import get_esm3_model_tokenizers
    HAS_ESM3 = True
except ImportError:
    HAS_ESM3 = False


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-mask dataset for fine-tuning")
    parser.add_argument("--input_dataset", type=str, required=True,
                        help="Path to input HuggingFace dataset")
    parser.add_argument("--output_dataset", type=str, required=True,
                        help="Path to save pre-masked dataset")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["mlm", "tra", "trb", "tra_trb_pairing", "tcr_mhc", "peptide_mhc", "specificity"],
                        help="Masking mode")
    parser.add_argument("--tokenizer_path", type=str, default="Rostlab/prot_bert",
                        help="Tokenizer to use")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="MLM masking probability")
    parser.add_argument("--cdr3_mask_length", type=int, default=5,
                        help="CDR3 masking length for tra/trb modes")
    parser.add_argument("--num_proc", type=int, default=None,
                        help="Number of processes (default: all CPUs)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


class MaskingFunction:
    """Callable class for masking that can be pickled for multiprocessing."""
    
    def __init__(self, tokenizer, mode: str, mlm_probability: float, cdr3_mask_length: int, seed: int):
        self.tokenizer = tokenizer
        self.mode = mode.lower()
        self.mlm_probability = mlm_probability
        self.cdr3_mask_length = cdr3_mask_length
        self.seed = seed
        
        # Token IDs - handle both HuggingFace and ESM3 tokenizers
        self.mask_token_id = getattr(tokenizer, 'mask_token_id', None)
        self.pad_token_id = getattr(tokenizer, 'pad_token_id', None)
        if self.pad_token_id is None:
            self.pad_token_id = 0
        
        self.cls_token_id = getattr(tokenizer, 'cls_token_id', None)
        self.sep_token_id = getattr(tokenizer, 'sep_token_id', None)
        self.unk_token_id = getattr(tokenizer, 'unk_token_id', None)
        
        # Dash token ID (30 in ESM) - used as molecule separator
        self.dash_token_id = tokenizer.convert_tokens_to_ids('-') if hasattr(tokenizer, 'convert_tokens_to_ids') else None
        
        # Vocab size - handle both dict-like and len() tokenizers
        if hasattr(tokenizer, '__len__'):
            self.vocab_size = len(tokenizer)
        elif hasattr(tokenizer, 'vocab_size'):
            self.vocab_size = tokenizer.vocab_size
        else:
            self.vocab_size = 1000  # Fallback
    
    def __call__(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Apply masking to a batch of examples."""
        # Set seed for reproducibility (using example index if available)
        random.seed(self.seed)
        
        input_ids_list = examples["input_ids"]
        permutation_keys = examples.get("permutation_key", [""] * len(input_ids_list))
        
        masked_inputs = []
        labels = []
        
        for input_ids, pkey in zip(input_ids_list, permutation_keys):
            # Check if this example matches the mode's filtering criteria
            if not self._matches_mode(pkey):
                # Skip masking - keep original input with all -100 labels (no loss)
                input_ids_copy = input_ids.copy() if isinstance(input_ids, list) else input_ids.tolist()
                masked_inputs.append(input_ids_copy)
                labels.append([-100] * len(input_ids_copy))
                continue
            
            if self.mode == "mlm":
                masked, lab = self._mask_mlm(input_ids)
            elif self.mode in ["tra", "trb"]:
                masked, lab = self._mask_cdr3_middle(input_ids, pkey)
            elif self.mode == "tra_trb_pairing":
                masked, lab = self._mask_first_chain(input_ids)
            elif self.mode == "tcr_mhc":
                masked, lab = self._mask_first_molecules_tcr_mhc(input_ids, pkey)
            elif self.mode == "peptide_mhc":
                masked, lab = self._mask_first_molecules_peptide_mhc(input_ids, pkey)
            elif self.mode == "specificity":
                masked, lab = self._mask_first_molecule(input_ids)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
            
            masked_inputs.append(masked)
            labels.append(lab)
        
        return {
            "input_ids": masked_inputs,
            "attention_mask": examples["attention_mask"],
            "labels": labels,
            "permutation_key": permutation_keys,
        }
    
    def _matches_mode(self, pkey: str) -> bool:
        """Check if permutation key matches the current masking mode.
        
        Filtering logic:
        - mlm: keep all keys
        - tra: keep only keys that are exactly "tra"
        - trb: keep only keys that are exactly "trb"
        - tra_trb_pairing: keep only keys "tra_trb" or "trb_tra"
        - tcr_mhc: keys with at least one TCR chain AND at least one MHC chain, but NO peptide
        - peptide_mhc: keys with peptide AND at least one MHC chain, but NO TCR chains
        - specificity: keys with at least one TCR chain, peptide, AND at least one MHC chain
        
        Args:
            pkey: Permutation key string (e.g., 'tra', 'trb_peptide_mhc_one')
            
        Returns:
            True if the key matches the mode, False otherwise
        """
        if not pkey:
            return False
        
        pkey_lower = pkey.lower()
        
        # Check for presence of each molecule type using substring matching
        has_tra = 'tra' in pkey_lower.replace('_', ' ').split()
        has_trb = 'trb' in pkey_lower.replace('_', ' ').split()
        has_tcr = has_tra or has_trb
        has_peptide = 'peptide' in pkey_lower
        has_mhc = 'mhc_one' in pkey_lower or 'mhc_two' in pkey_lower or 'mhcone' in pkey_lower or 'mhctwo' in pkey_lower
        
        if self.mode == "mlm":
            # Keep all keys
            return True
        
        elif self.mode == "tra":
            # Keep only keys that are exactly "tra"
            return pkey_lower == "tra"
        
        elif self.mode == "trb":
            # Keep only keys that are exactly "trb"
            return pkey_lower == "trb"
        
        elif self.mode == "tra_trb_pairing":
            # Keep only keys "tra_trb" or "trb_tra"
            return pkey_lower in ["tra_trb", "trb_tra"]
        
        elif self.mode == "tcr_mhc":
            # Keys with at least one TCR chain AND at least one MHC chain, but NO peptide
            return has_tcr and has_mhc and not has_peptide
        
        elif self.mode == "peptide_mhc":
            # Keys with peptide AND at least one MHC chain, but NO TCR chains
            return has_peptide and has_mhc and not has_tcr
        
        elif self.mode == "specificity":
            # Keys with at least one TCR chain, peptide, AND at least one MHC chain
            return has_tcr and has_peptide and has_mhc
        
        else:
            # Unknown mode - keep all
            return True
    
    def _is_special_token(self, token_id: int) -> bool:
        """Check if token is a special token (should not be masked).
        
        For ESM models, dash (-) tokens are used as molecule separators
        and should not be masked. Dash is token ID 30 in ESM vocabulary.
        """
        if token_id == self.pad_token_id:
            return True
        if self.cls_token_id is not None and token_id == self.cls_token_id:
            return True
        if self.sep_token_id is not None and token_id == self.sep_token_id:
            return True
        if self.unk_token_id is not None and token_id == self.unk_token_id:
            return True
        if self.dash_token_id is not None and token_id == self.dash_token_id:
            # Dash (-) is used as molecule separator in ESM models
            return True
        
        # Try to decode token (handle both HuggingFace and ESM3 tokenizers)
        try:
            if hasattr(self.tokenizer, 'decode'):
                token = self.tokenizer.decode([token_id])
            else:
                # Fallback for tokenizers without decode
                return False
        except:
            return False
        
        if token.startswith('[') and token.endswith(']'):
            return True
        if token in ['<s>', '</s>', '<pad>', '<unk>', '<mask>', '<cls>', '<sep>']:
            return True
        
        return False
    
    def _find_sequence_boundaries(self, input_ids: List[int]) -> tuple:
        """Find start and end indices of actual sequence."""
        seq_start = 0
        seq_end = len(input_ids)
        
        for i, token_id in enumerate(input_ids):
            if not self._is_special_token(token_id):
                seq_start = i
                break
        
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] != self.pad_token_id and not self._is_special_token(input_ids[i]):
                seq_end = i + 1
                break
        
        return seq_start, seq_end
    
    def _find_separators(self, input_ids: List[int]) -> List[int]:
        """Find positions of separator tokens."""
        separators = []
        for i, token_id in enumerate(input_ids):
            token = self.tokenizer.decode([token_id])
            if '[SEP]' in token or token in ['[ETRA]', '[ETRB]', '[EPEP]', '[EMHO]', '[EMHT]']:
                separators.append(i)
        return separators
    
    def _mask_mlm(self, input_ids: List[int]) -> tuple:
        """Standard MLM masking: 15% random."""
        input_ids = input_ids.copy() if isinstance(input_ids, list) else input_ids.tolist()
        label_ids = [-100] * len(input_ids)
        
        for i, token_id in enumerate(input_ids):
            if self._is_special_token(token_id):
                continue
            
            if random.random() < self.mlm_probability:
                label_ids[i] = input_ids[i]
                
                prob = random.random()
                if prob < 0.8:
                    input_ids[i] = self.mask_token_id
                elif prob < 0.9:
                    input_ids[i] = random.randint(0, self.vocab_size - 1)
        
        return input_ids, label_ids
    
    def _mask_cdr3_middle(self, input_ids: List[int], pkey: str) -> tuple:
        """Mask 15% of middle portion of CDR3 region.
        
        Instead of masking all tokens in the middle region, we mask 15% of them
        randomly (similar to standard MLM but restricted to the middle region).
        """
        input_ids = input_ids.copy() if isinstance(input_ids, list) else input_ids.tolist()
        label_ids = [-100] * len(input_ids)
        
        seq_start, seq_end = self._find_sequence_boundaries(input_ids)
        seq_length = seq_end - seq_start
        
        if seq_length > self.cdr3_mask_length:
            middle_start = seq_start + (seq_length - self.cdr3_mask_length) // 2
            middle_end = middle_start + self.cdr3_mask_length
            
            # Mask 15% of tokens in the middle region (like MLM but restricted to middle)
            for i in range(middle_start, middle_end):
                if i < seq_end and not self._is_special_token(input_ids[i]):
                    if random.random() < self.mlm_probability:  # 15% probability
                        label_ids[i] = input_ids[i]
                        
                        # Standard MLM: 80% mask, 10% random, 10% keep
                        prob = random.random()
                        if prob < 0.8:
                            input_ids[i] = self.mask_token_id
                        elif prob < 0.9:
                            input_ids[i] = random.randint(0, self.vocab_size - 1)
                        # else: keep original (10%)
        
        return input_ids, label_ids
    
    def _mask_first_chain(self, input_ids: List[int]) -> tuple:
        """Mask mlm_probability% of first chain with 80/10/10 strategy."""
        input_ids = input_ids.copy() if isinstance(input_ids, list) else input_ids.tolist()
        label_ids = [-100] * len(input_ids)
        
        separators = self._find_separators(input_ids)
        seq_start, seq_end = self._find_sequence_boundaries(input_ids)
        
        if separators:
            mask_end = separators[0]
        else:
            mask_end = seq_start + (seq_end - seq_start) // 2
        
        for i in range(seq_start, mask_end):
            if not self._is_special_token(input_ids[i]):
                if random.random() < self.mlm_probability:
                    label_ids[i] = input_ids[i]
                    
                    # Standard MLM: 80% mask, 10% random, 10% keep
                    prob = random.random()
                    if prob < 0.8:
                        input_ids[i] = self.mask_token_id
                    elif prob < 0.9:
                        input_ids[i] = random.randint(0, self.vocab_size - 1)
                    # else: keep original (10%)
        
        return input_ids, label_ids
    
    def _mask_first_molecules_tcr_mhc(self, input_ids: List[int], pkey: str) -> tuple:
        """Mask first molecule(s) for TCR-MHC."""
        input_ids = input_ids.copy() if isinstance(input_ids, list) else input_ids.tolist()
        label_ids = [-100] * len(input_ids)
        
        separators = self._find_separators(input_ids)
        seq_start, _ = self._find_sequence_boundaries(input_ids)
        molecules = pkey.lower().split('_')
        
        if len(separators) >= 1:
            mask_two = False
            if len(molecules) >= 2:
                if (molecules[0] in ['tra', 'trb'] and molecules[1] in ['tra', 'trb']):
                    mask_two = True
                elif (molecules[0] in ['mhc_one', 'mhc_two'] and molecules[1] in ['mhc_one', 'mhc_two']):
                    mask_two = True
            
            if mask_two and len(separators) >= 2:
                mask_end = separators[1]
            else:
                mask_end = separators[0]
            
            for i in range(seq_start, mask_end):
                if not self._is_special_token(input_ids[i]):
                    if random.random() < self.mlm_probability:
                        label_ids[i] = input_ids[i]
                        
                        # Standard MLM: 80% mask, 10% random, 10% keep
                        prob = random.random()
                        if prob < 0.8:
                            input_ids[i] = self.mask_token_id
                        elif prob < 0.9:
                            input_ids[i] = random.randint(0, self.vocab_size - 1)
                        # else: keep original (10%)
        
        return input_ids, label_ids
    
    def _mask_first_molecules_peptide_mhc(self, input_ids: List[int], pkey: str) -> tuple:
        """Mask first molecule(s) for peptide-MHC."""
        input_ids = input_ids.copy() if isinstance(input_ids, list) else input_ids.tolist()
        label_ids = [-100] * len(input_ids)
        
        separators = self._find_separators(input_ids)
        seq_start, _ = self._find_sequence_boundaries(input_ids)
        molecules = pkey.lower().split('_')
        
        if len(separators) >= 1:
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
                    if random.random() < self.mlm_probability:
                        label_ids[i] = input_ids[i]
                        
                        # Standard MLM: 80% mask, 10% random, 10% keep
                        prob = random.random()
                        if prob < 0.8:
                            input_ids[i] = self.mask_token_id
                        elif prob < 0.9:
                            input_ids[i] = random.randint(0, self.vocab_size - 1)
                        # else: keep original (10%)
        
        return input_ids, label_ids
    
    def _mask_first_molecule(self, input_ids: List[int]) -> tuple:
        """Mask first molecule for specificity."""
        input_ids = input_ids.copy() if isinstance(input_ids, list) else input_ids.tolist()
        label_ids = [-100] * len(input_ids)
        
        separators = self._find_separators(input_ids)
        seq_start, _ = self._find_sequence_boundaries(input_ids)
        
        if separators:
            mask_end = separators[0]
        else:
            seq_end = len(input_ids)
            mask_end = seq_start + (seq_end - seq_start) // 3
        
        for i in range(seq_start, mask_end):
            if not self._is_special_token(input_ids[i]):
                if random.random() < self.mlm_probability:
                    label_ids[i] = input_ids[i]
                    
                    # Standard MLM: 80% mask, 10% random, 10% keep
                    prob = random.random()
                    if prob < 0.8:
                        input_ids[i] = self.mask_token_id
                    elif prob < 0.9:
                        input_ids[i] = random.randint(0, self.vocab_size - 1)
                    # else: keep original (10%)
        
        return input_ids, label_ids


def load_tokenizer(tokenizer_path: str):
    """
    Load tokenizer with special handling for ESM3.
    
    Args:
        tokenizer_path: Path or name of tokenizer. Special values:
            - "esm3" or "esm3_sm_open_v1": Load ESM3 tokenizer
            - Otherwise: Load via AutoTokenizer
    
    Returns:
        tokenizer object
    """
    # Check if ESM3 is requested
    if tokenizer_path.lower() in ["esm3", "esm3_sm_open_v1"]:
        if not HAS_ESM3:
            print("⚠️  ESM-3 package not available. Install with: pip install esm")
            print("   Falling back to ESM-2 tokenizer")
            tokenizer_path = "facebook/esm2_t6_8M_UR50D"
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            print(f"✓ Loaded ESM-2 tokenizer as fallback")
        else:
            try:
                print(f"Loading ESM-3 tokenizer...")
                tokenizers = get_esm3_model_tokenizers("esm3_sm_open_v1")
                tokenizer = tokenizers.sequence
                print(f"✓ Loaded ESM-3 sequence tokenizer")
            except Exception as e:
                print(f"⚠️  Failed to load ESM-3 tokenizer: {e}")
                print("   Falling back to ESM-2 tokenizer")
                tokenizer_path = "facebook/esm2_t6_8M_UR50D"
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                print(f"✓ Loaded ESM-2 tokenizer as fallback")
    else:
        # Standard HuggingFace tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"✓ Loaded tokenizer: {tokenizer_path}")
    
    return tokenizer


def main():
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Determine number of processes
    if args.num_proc is None:
        args.num_proc = os.cpu_count()
    
    print(f"Pre-masking dataset with {args.num_proc} processes")
    print(f"Input: {args.input_dataset}")
    print(f"Output: {args.output_dataset}")
    print(f"Mode: {args.mode}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_from_disk(args.input_dataset)
    print(f"Loaded splits: {list(dataset.keys())}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {args.tokenizer_path}")
    tokenizer = load_tokenizer(args.tokenizer_path)
    
    # Create masking function
    mask_fn = MaskingFunction(
        tokenizer=tokenizer,
        mode=args.mode,
        mlm_probability=args.mlm_probability,
        cdr3_mask_length=args.cdr3_mask_length,
        seed=args.seed
    )
    
    # Apply masking to all splits
    print(f"\nApplying {args.mode} masking with {args.num_proc} processes...")
    masked_dataset = DatasetDict()
    
    # Create a filter function based on mode
    def filter_by_mode(example):
        """Filter examples that don't match the masking mode."""
        return mask_fn._matches_mode(example.get("permutation_key", ""))
    
    for split_name in dataset.keys():
        original_count = len(dataset[split_name])
        print(f"\nProcessing {split_name} split ({original_count:,} examples)...")
        
        # Filter to only keep examples that match the mode
        if args.mode != "mlm":
            print(f"   Filtering for {args.mode} mode...")
            filtered_split = dataset[split_name].filter(
                filter_by_mode,
                num_proc=args.num_proc,
                desc=f"Filtering {split_name}"
            )
            filtered_count = len(filtered_split)
            print(f"   Kept {filtered_count:,} / {original_count:,} examples ({100*filtered_count/original_count:.1f}%)")
        else:
            filtered_split = dataset[split_name]
        
        # Apply masking
        masked_split = filtered_split.map(
            mask_fn,
            batched=True,
            batch_size=1000,
            num_proc=args.num_proc,
            desc=f"Masking {split_name}",
            remove_columns=["sequence"] if "sequence" in filtered_split.column_names else []
        )
        
        masked_dataset[split_name] = masked_split
        print(f"✅ {split_name}: {len(masked_split):,} examples masked")
    
    # Save masked dataset
    print(f"\nSaving masked dataset to: {args.output_dataset}")
    masked_dataset.save_to_disk(args.output_dataset)
    
    print("\n✅ Pre-masking complete!")
    print(f"\nDataset saved to: {args.output_dataset}")
    print("\nYou can now use this pre-masked dataset for fast training:")
    print(f"  python scripts/training/fine_tune.py \\")
    print(f"    --dataset_path {args.output_dataset} \\")
    print(f"    --model_path Rostlab/prot_bert \\")
    print(f"    --mode {args.mode} \\")
    print(f"    --output_dir ./checkpoints \\")
    print(f"    --use_pre_masked  # Add this flag!")


if __name__ == "__main__":
    main()
