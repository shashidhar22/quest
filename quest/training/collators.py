"""
quest.training.collators - Data collators for training.

Consolidates collator classes from:
  - scripts/training/esm_native_trainer.py
    (DataCollatorForMLMWithPacking, DataCollatorForMLMDynamic, DataCollatorForMLMWithVarlen)
  - scripts/training/tcr_robust_contrastive_trainer.py
    (ContrastivePairCollator)
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Try to import CDR region identifier (needed by TaskSpecificMaskingCollator)
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from quest.parsers.cdr_region_identifier import CDRRegionIdentifier
    CDR_IDENTIFIER_AVAILABLE = True
except ImportError:
    CDR_IDENTIFIER_AVAILABLE = False


# =============================================================================
# MLM Collators (from esm_native_trainer.py)
# =============================================================================


class DataCollatorForMLMWithPacking:
    """
    Optimized data collator using pure tensor operations.
    Packs multiple sequences into fixed-length samples to eliminate padding.
    """

    def __init__(
        self,
        tokenizer,
        max_seq_length: int = 1024,
        mlm_probability: float = 0.15,
        separator_token_id: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = len(tokenizer)

        # Cache special token IDs as tensor for vectorized isin() operation
        special_ids = [
            tokenizer.pad_token_id,
            tokenizer.cls_token_id,
            tokenizer.eos_token_id,
            tokenizer.sep_token_id,
            tokenizer.unk_token_id,
        ]
        if separator_token_id is not None:
            special_ids.append(separator_token_id)
        special_ids = [x for x in special_ids if x is not None]
        self.special_token_ids = torch.tensor(special_ids, dtype=torch.long)

    def __call__(self, examples):
        """Pack sequences and apply MLM masking using pure tensor operations."""
        sequences = []
        for ex in examples:
            ids = ex.get("input_ids", ex) if isinstance(ex, dict) else ex
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
            elif ids.dtype != torch.long:
                ids = ids.long()
            sequences.append(ids)

        packed_ids, packed_mask = self._pack_sequences(sequences)
        input_ids, labels = self._apply_mlm_masking(packed_ids, packed_mask)

        return {
            "input_ids": input_ids,
            "attention_mask": packed_mask,
            "labels": labels,
        }

    def _pack_sequences(self, sequences):
        """Pack sequences into fixed-length samples using pre-allocated tensors."""
        lengths = [len(s) for s in sequences]
        total_tokens = sum(lengths)
        num_packs = max(1, (total_tokens + self.max_seq_length - 1) // self.max_seq_length)

        packed_ids = torch.full((num_packs, self.max_seq_length), self.pad_token_id, dtype=torch.long)
        packed_mask = torch.zeros(num_packs, self.max_seq_length, dtype=torch.long)

        pack_idx = 0
        pos = 0

        for seq in sequences:
            seq_len = len(seq)
            if pos + seq_len > self.max_seq_length:
                pack_idx += 1
                pos = 0
                if pack_idx >= num_packs:
                    packed_ids = torch.cat([packed_ids, torch.full((1, self.max_seq_length), self.pad_token_id, dtype=torch.long)], dim=0)
                    packed_mask = torch.cat([packed_mask, torch.zeros(1, self.max_seq_length, dtype=torch.long)], dim=0)
                    num_packs += 1

            packed_ids[pack_idx, pos:pos + seq_len] = seq
            packed_mask[pack_idx, pos:pos + seq_len] = 1
            pos += seq_len

        if pack_idx + 1 < num_packs:
            packed_ids = packed_ids[:pack_idx + 1]
            packed_mask = packed_mask[:pack_idx + 1]

        return packed_ids, packed_mask

    def _apply_mlm_masking(self, input_ids, attention_mask):
        """Apply MLM masking using single-pass vectorized operations."""
        labels = input_ids.clone()
        rand_mask = torch.rand(input_ids.shape)
        special_mask = torch.isin(input_ids, self.special_token_ids)
        valid_mask = (attention_mask == 1) & ~special_mask
        masked_indices = valid_mask & (rand_mask < self.mlm_probability)
        labels[~masked_indices] = -100

        mask_type = torch.rand(input_ids.shape)
        input_ids = input_ids.clone()
        input_ids[masked_indices & (mask_type < 0.8)] = self.mask_token_id

        random_token_indices = masked_indices & (mask_type >= 0.8) & (mask_type < 0.9)
        if random_token_indices.any():
            input_ids[random_token_indices] = torch.randint(self.vocab_size, (random_token_indices.sum(),), dtype=torch.long)

        return input_ids, labels


class DataCollatorForMLMDynamic:
    """
    Simple MLM collator with dynamic padding to batch max length.
    More efficient than packing when sequences have similar lengths.
    """

    def __init__(
        self,
        tokenizer,
        mlm_probability: float = 0.15,
        pad_to_multiple_of: int = 8,
        separator_token_id: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = len(tokenizer)
        self.pad_to_multiple_of = pad_to_multiple_of

        special_ids = [
            tokenizer.pad_token_id,
            tokenizer.cls_token_id,
            tokenizer.eos_token_id,
            tokenizer.sep_token_id,
            tokenizer.unk_token_id,
        ]
        if separator_token_id is not None:
            special_ids.append(separator_token_id)
        self.special_token_ids = torch.tensor([x for x in special_ids if x is not None], dtype=torch.long)

    def __call__(self, examples):
        """Dynamically pad to batch max length using vectorized operations."""
        # Convert all sequences to tensors
        sequences = []
        for ex in examples:
            ids = ex.get("input_ids", ex) if isinstance(ex, dict) else ex
            if isinstance(ids, list):
                sequences.append(torch.tensor(ids, dtype=torch.long))
            else:
                sequences.append(ids.long() if ids.dtype != torch.long else ids)

        # Use pad_sequence for efficient padding (C++ implementation)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=self.pad_token_id
        )

        # Create attention mask using broadcasting (no Python loop)
        lengths = torch.tensor([len(s) for s in sequences])
        max_len = input_ids.size(1)
        attention_mask = (torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)).long()

        # Pad to multiple of 8 for tensor core efficiency
        if self.pad_to_multiple_of and max_len % self.pad_to_multiple_of != 0:
            pad_len = self.pad_to_multiple_of - (max_len % self.pad_to_multiple_of)
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=self.pad_token_id)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=0)

        # Apply MLM masking
        input_ids, labels = self._apply_mlm_masking(input_ids, attention_mask)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _apply_mlm_masking(self, input_ids, attention_mask):
        """Apply MLM masking using vectorized operations."""
        labels = input_ids.clone()

        # Single random tensor for efficiency
        rand_vals = torch.rand(input_ids.shape[0], input_ids.shape[1], 2)
        rand_mask = rand_vals[..., 0]
        mask_type = rand_vals[..., 1]

        special_mask = torch.isin(input_ids, self.special_token_ids)
        valid_mask = (attention_mask == 1) & ~special_mask
        masked_indices = valid_mask & (rand_mask < self.mlm_probability)
        labels[~masked_indices] = -100

        input_ids = input_ids.clone()
        input_ids[masked_indices & (mask_type < 0.8)] = self.mask_token_id

        random_token_indices = masked_indices & (mask_type >= 0.8) & (mask_type < 0.9)
        if random_token_indices.any():
            input_ids[random_token_indices] = torch.randint(
                self.vocab_size, (random_token_indices.sum(),), dtype=torch.long
            )

        return input_ids, labels


class DataCollatorForMLMWithVarlen:
    """
    Packs sequences with proper Block Diagonal Masking.
    1. Resets Position IDs (Fixes RoPE embeddings)
    2. Creates 4D Attention Mask (Fixes Cross-Protein Contamination)

    Each sequence in a pack only attends to itself via explicit 4D mask.
    """

    def __init__(
        self,
        tokenizer,
        max_seq_length: int = 1024,
        mlm_probability: float = 0.15,
        separator_token_id: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = len(tokenizer)

        # Cache special token IDs for MLM masking
        special_ids = [
            tokenizer.pad_token_id,
            tokenizer.cls_token_id,
            tokenizer.eos_token_id,
            tokenizer.sep_token_id,
            tokenizer.unk_token_id,
        ]
        if separator_token_id is not None:
            special_ids.append(separator_token_id)
        self.special_token_ids = torch.tensor(
            [x for x in special_ids if x is not None], dtype=torch.long
        )

    def __call__(self, examples):
        """Pack sequences and generate 4D block diagonal attention mask."""
        sequences = []
        for ex in examples:
            ids = ex.get("input_ids", ex) if isinstance(ex, dict) else ex
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
            elif ids.dtype != torch.long:
                ids = ids.long()
            sequences.append(ids)

        # Pack and generate 4D mask
        packed_batch = self._pack_with_4d_mask(sequences)

        # Apply MLM masking
        input_ids, labels = self._apply_mlm_masking(
            packed_batch["input_ids"], packed_batch["valid_mask"]
        )

        return {
            "input_ids": input_ids,
            "attention_mask": packed_batch["attention_mask"],  # 4D Block Diagonal
            "position_ids": packed_batch["position_ids"],      # Resets per protein
            "labels": labels,
        }

    def _pack_with_4d_mask(self, sequences):
        """Pack sequences with 4D block diagonal attention mask."""
        input_ids_list = []
        position_ids_list = []
        attention_masks_list = []

        current_tokens = []
        current_len = 0

        # Truncate sequences that exceed max_seq_length and sort by length
        truncated_seqs = []
        for seq in sequences:
            if len(seq) > self.max_seq_length:
                seq = seq[:self.max_seq_length]
            truncated_seqs.append(seq)

        # Sort sequences by length (descending) for tighter packing
        sequences = sorted(truncated_seqs, key=len, reverse=True)

        for seq in sequences:
            if current_len + len(seq) > self.max_seq_length:
                # Flush current pack if not empty
                if current_tokens:
                    self._flush_pack(
                        current_tokens, input_ids_list, position_ids_list, attention_masks_list
                    )
                current_tokens = []
                current_len = 0

            current_tokens.append(seq)
            current_len += len(seq)

        # Flush final pack
        if current_tokens:
            self._flush_pack(
                current_tokens, input_ids_list, position_ids_list, attention_masks_list
            )

        # Stack into batch
        batch_input_ids = torch.stack(input_ids_list)
        batch_position_ids = torch.stack(position_ids_list)
        batch_attention_mask = torch.stack(attention_masks_list)

        # Valid mask for MLM (1D per sample)
        valid_mask = (batch_input_ids != self.pad_token_id).long()

        # Reshape for Transformers: (Batch, 1, Seq, Seq)
        batch_attention_mask = batch_attention_mask.unsqueeze(1)

        return {
            "input_ids": batch_input_ids,
            "position_ids": batch_position_ids,
            "attention_mask": batch_attention_mask,
            "valid_mask": valid_mask,
        }

    def _flush_pack(self, tokens, input_ids_list, position_ids_list, attention_masks_list):
        """Flush a pack of sequences into the batch lists."""
        full_ids = torch.full((self.max_seq_length,), self.pad_token_id, dtype=torch.long)
        full_pos = torch.zeros((self.max_seq_length,), dtype=torch.long)

        # 2D Block Diagonal Mask - ADDITIVE format for HuggingFace
        # 0 = can attend, -inf = cannot attend (masked out after softmax)
        mask_2d = torch.full(
            (self.max_seq_length, self.max_seq_length),
            float("-inf"),
            dtype=torch.float32
        )

        start_idx = 0
        for seq in tokens:
            end_idx = start_idx + len(seq)

            # Fill IDs
            full_ids[start_idx:end_idx] = seq

            # Fill Positions (0, 1, 2... for each sequence)
            full_pos[start_idx:end_idx] = torch.arange(len(seq), dtype=torch.long)

            # Block Diagonal: this segment attends only to itself
            # Set to 0 (can attend) instead of -inf (masked)
            mask_2d[start_idx:end_idx, start_idx:end_idx] = 0.0

            start_idx = end_idx

        input_ids_list.append(full_ids)
        position_ids_list.append(full_pos)
        attention_masks_list.append(mask_2d)

    def _apply_mlm_masking(self, input_ids, valid_mask):
        """Apply MLM masking using vectorized operations."""
        labels = input_ids.clone()
        rand_mask = torch.rand(input_ids.shape)

        special_mask = torch.isin(input_ids, self.special_token_ids)
        is_token = (valid_mask == 1) & ~special_mask

        masked_indices = is_token & (rand_mask < self.mlm_probability)
        labels[~masked_indices] = -100

        mask_type = torch.rand(input_ids.shape)
        input_ids = input_ids.clone()
        input_ids[masked_indices & (mask_type < 0.8)] = self.mask_token_id

        random_idx = masked_indices & (mask_type >= 0.8) & (mask_type < 0.9)
        if random_idx.any():
            input_ids[random_idx] = torch.randint(
                self.vocab_size, (random_idx.sum(),), dtype=torch.long
            )

        return input_ids, labels


# =============================================================================
# Task-Specific Masking Collator
# =============================================================================


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
            print(f"Initialized CDR identifier for mode: {mode}")
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

    # -------------------------------------------------------------------------
    # MLM Masking (15% random)
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # TRA/TRB CDR3 Masking (middle 5 amino acids)
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # TRA-TRB Pairing: Mask entire first chain
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # TCR-MHC: Mask first molecule (or both TCR chains if first two)
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Peptide-MHC: Mask first molecule (or both MHC if first two)
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Specificity: Mask first molecule
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Full CDR Masking (NEW - full_tra, full_trb modes)
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Batch creation with padding
    # -------------------------------------------------------------------------

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


# =============================================================================
# Contrastive Pair Collator (from tcr_robust_contrastive_trainer.py)
# =============================================================================


class ContrastivePairCollator:
    """
    Collator that handles both tokenization AND similarity mask computation.

    By computing similarity masks here (in DataLoader workers), we:
    1. Offload O(N^2) computation to CPU background processes
    2. Prevent GPU stalls waiting for mask computation
    3. Enable parallel mask computation across batches

    Note: Requires ``SequenceSimilarityCalculator`` from the contrastive
    training module.  Import is deferred to avoid hard dependency.
    """

    def __init__(
        self,
        tokenizer,
        similarity_threshold: float = 0.9,
        max_length: int = 320,
        pad_to_multiple_of: int = 8,
        var_region_len: int = 150,
        similarity_calculator=None,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer instance.
            similarity_threshold: Threshold for the similarity mask.
            max_length: Maximum token length for truncation.
            pad_to_multiple_of: Pad token count to a multiple of this value.
            var_region_len: Truncate sequences to this many amino acids
                before tokenization (0 = no truncation).
            similarity_calculator: Pre-constructed ``SequenceSimilarityCalculator``.
                If ``None``, one will be created (requires the contrastive
                training module to be importable).
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.var_region_len = var_region_len

        if similarity_calculator is not None:
            self.sim_calculator = similarity_calculator
        else:
            # Deferred import to avoid hard dependency
            from scripts.training.tcr_robust_contrastive_trainer import (
                SequenceSimilarityCalculator,
            )
            self.sim_calculator = SequenceSimilarityCalculator(
                threshold=similarity_threshold
            )

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        alpha_seqs = [ex["alpha_seq"] for ex in examples]
        beta_seqs = [ex["beta_seq"] for ex in examples]

        # CRITICAL FIX: Truncate to variable region BEFORE tokenization
        # ESM2's self-attention contextualizes all positions, so masking after encoding
        # is ineffective. The constant region info "bleeds" into variable region embeddings.
        # By truncating here, we prevent the encoder from seeing constant regions entirely.
        if self.var_region_len > 0:
            alpha_seqs = [seq[:self.var_region_len] for seq in alpha_seqs]
            beta_seqs = [seq[:self.var_region_len] for seq in beta_seqs]

        # 1. Compute similarity masks (CPU-intensive, runs in workers)
        alpha_sim_mask = self.sim_calculator.compute_batch_similarity_mask(alpha_seqs)
        beta_sim_mask = self.sim_calculator.compute_batch_similarity_mask(beta_seqs)

        # 2. Tokenize
        alpha_encoded = self.tokenizer(
            alpha_seqs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        beta_encoded = self.tokenizer(
            beta_seqs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        return {
            "alpha_input_ids": alpha_encoded["input_ids"],
            "alpha_attention_mask": alpha_encoded["attention_mask"],
            "beta_input_ids": beta_encoded["input_ids"],
            "beta_attention_mask": beta_encoded["attention_mask"],
            "alpha_sim_mask": alpha_sim_mask,
            "beta_sim_mask": beta_sim_mask,
        }
