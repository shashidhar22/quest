#!/usr/bin/python3
# -*- coding: utf-8 -*-
# type: ignore
"""
datawriter.py  *map-only version*
──────────────────────────────────────────────
• Reads Parquet shards (local or s3://)
• Splits 80/10/10 → train / val / test
• Two branches controlled by --model-name

  1.  bert | protbert | esm | llama
        explode → HF tokenizer → DatasetDict

  2.  custom-rnn | custom-transformer
        ▸ train a BPE tokenizer on **train** split
        ▸ tag sequences with custom boundary tokens
        ▸ encode via AminoAcidDataset **inside datasets.map**
        ▸ returns DatasetDict with {input_ids,target_ids}

No manual pyarrow writing — all handled by hf dataset `map`.
"""

import os
import gc
import argparse
from itertools import combinations, permutations
from pathlib import Path
from typing import List, Dict, Any
import numpy
import s3fs
from datasets import (load_dataset, Dataset, DatasetDict)
from datasets.data_files import DataFilesList
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch
from torch.utils.data import Dataset as TorchDataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from datasets import load_from_disk, concatenate_datasets  # type: ignore
import pandas as pd  # type: ignore
import random



# -- model cards ------------------------------------------------
MODEL_CARDS = {
    "bert": "google-bert/bert-base-uncased",
    "protbert": "Rostlab/prot_bert",
    "esm": "EvolutionaryScale/esm3-sm-open-v1",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "lstm": "rnn",
    "transformer": "tranformer",
}

# ── universal constants ────────────────────────────────────────────────
FIELDS = ["tra","trb","peptide","mhc_one","mhc_two"]
START_STYLE = {m: (lambda s: s) for m in ("bert","protbert")}|{"esm":lambda s:s,"llama":lambda s:s}  # type: ignore
JOIN_STYLE  = {m: (lambda t: " [SEP] ".join(t)) for m in ("bert","protbert")}|{"esm":lambda t:" ".join(t),"llama":lambda t:" ".join(t)}  # type: ignore
END_STYLE   = {m: (lambda s: s) for m in ("bert","protbert")}|{"esm":lambda s:s,"llama":lambda s:s}  # type: ignore

# BPE boundary tokens ----------------------------------------------------
BPE_TOKENS = {
    "pad_token": ["[PAD]"], "unk_token": ["[UNK]"], "end_token": ["[END]"],
    "tra_tokens": ["[TRA]", "[ETRA]"], "trb_tokens": ["[TRB]", "[ETRB]"],
    "pep_tokens": ["[PEP]", "[EPEP]"], "mho_tokens": ["[MHO]", "[EMHO]"],
    "mht_tokens": ["[MHT]", "[EMHT]"],
}
PAD_TOKEN_STR = "[PAD]"; PAD_TOKEN_ID = 0

# MODE dictionary: defines the combo fields that need to be present for each mode
MODE = {"tra": set(["tra"]),
        "trb": set(["trb"]),
        "tcr_pairing": set(["tra", "trb"]),
        "mhc_binding": set(["peptide", "mhc_one", "mhc_two"]),
        "specificity": set(["tra", "trb", "peptide", "mhc_one", "mhc_two"]),
        "default": set(["tra", "trb", "peptide", "mhc_one", "mhc_two"])}

# ── helper: ProtBERT‑style explode ─────────────────────────────────────-
        
def filter_combo_feats_by_mode(combo_feats: List[tuple], mode: str) -> List[tuple]:
    """
    Filter combo_feats based on the analysis mode.

    Args:
        combo_feats: List of feature combination tuples
        mode: 'tra', 'trb', 'tcr_pairing', 'mhc_binding', 'specificity', or 'default'

    Returns:
        Filtered list of combo_feats tuples
    """
    if mode == "default":
        return combo_feats

    filtered = []
    for combo in combo_feats:
        combo_set = set(combo)

        if mode == "tra":
            if "tra" in combo_set:
                filtered.append(combo)

        elif mode == "trb":
            if "trb" in combo_set:
                filtered.append(combo)

        elif mode == "tcr_pairing":
            if combo_set == {"tra", "trb"}:
                filtered.append(combo)

        elif mode == "mhc_binding":
            has_mhc = "mhc_one" in combo_set or "mhc_two" in combo_set
            has_peptide = "peptide" in combo_set
            has_tcr = "tra" in combo_set or "trb" in combo_set
            if has_mhc and has_peptide and not has_tcr:
                filtered.append(combo)

        elif mode == "specificity":
            # Complete TCR complexes: (tra+trb+peptide+mhc_one) or (trb+peptide+mhc_one)
            has_tra = "tra" in combo_set
            has_trb = "trb" in combo_set
            has_peptide = "peptide" in combo_set
            has_mhc_one = "mhc_one" in combo_set

            full_complex = has_tra and has_trb and has_peptide and has_mhc_one
            minimal_complex = has_trb and has_peptide and has_mhc_one and not has_tra
            if full_complex or minimal_complex:
                filtered.append(combo)

    return filtered

def explode_example(ex: Dict[str, Any], mode: str = "specificity") -> Dict[str, List]:
    """
    Generate all permutations of non-null columns for a row.

    Args:
        ex: Row with columns tra, trb, peptide, mhc_one, mhc_two
        mode: Analysis mode determining which fields to use

    Returns:
        Dictionary with 'sequences' (list of tuples) and 'feat_names' (list of tuples)
        where each tuple contains (column_name, value) pairs
    """
    # Step 1: Gather all valid, non-empty features from the input row
    present_features = []
    fields = MODE[mode]
    for f in fields:
        val = ex.get(f)
        if isinstance(val, str) and val and val != "NA":
            # Clean the value
            cleaned_val = val.split('+')[0].strip() if '+' in val else val
            present_features.append((f, cleaned_val))

    # If there are no valid features in this row, exit early
    if not present_features:
        return {"sequences": [], "feat_names": []}

    # Step 2: Generate ALL possible permutations
    all_combos = []
    all_feat_names = []

    if mode in ("tra", "trb"):
        # For single-column modes, just return that column
        if present_features:
            all_combos.append(tuple(present_features))
            all_feat_names.append(tuple(f for f, v in present_features))
    else:
        # For multi-column modes, generate all permutations
        for r in range(1, len(present_features) + 1):
            for combo in permutations(present_features, r):
                all_combos.append(combo)
                all_feat_names.append(tuple(f for f, v in combo))

    # Step 3: Filter out combinations we don't want (always exclude peptide-only)
    filtered_combos = []
    filtered_feat_names = []
    for i, feat_tuple in enumerate(all_feat_names):
        # Always skip peptide-only combinations
        if feat_tuple == ('peptide',):
            continue
        filtered_combos.append(all_combos[i])
        filtered_feat_names.append(feat_tuple)

    # Step 4: Apply mode-specific filtering
    mode_filtered_feat_names = filter_combo_feats_by_mode(filtered_feat_names, mode)

    # Keep only the combos that match filtered features
    final_combos = []
    final_feat_names = []
    for i, feat_tuple in enumerate(filtered_feat_names):
        if feat_tuple in mode_filtered_feat_names:
            final_combos.append(filtered_combos[i])
            final_feat_names.append(feat_tuple)

    return {"sequences": final_combos, "feat_names": final_feat_names}
# ── helpers for custom BPE path ─────────────────────────────────────────

_valid_aa = set("ACDEFGHIKLMNPQRSTVWY")

def format_sequence_for_model(sequence_tuple: tuple, model_name: str) -> str:
    """
    Format a sequence tuple for a specific model type.

    Args:
        sequence_tuple: Tuple of (column_name, value) pairs, e.g., (('tra', 'AAA'), ('trb', 'BBB'))
        model_name: Model type (bert, protbert, esm, llama, lstm, transformer)

    Returns:
        Formatted sequence string
    """
    # Extract just the values from (name, value) pairs
    values = [val for name, val in sequence_tuple]

    if model_name == "protbert":
        # ProtBERT: space-separate amino acids, then join with [SEP]
        spaced_values = [' '.join(list(v)) for v in values]
        return " [SEP] ".join(spaced_values)
    elif model_name in ["esm", "llama"]:
        # ESM/Llama: just space separate the sequences
        return " ".join(values)
    else:  # bert or others
        # BERT: join with [SEP]
        return " [SEP] ".join(values)

# ── masking helpers ─────────────────────────────────────────────────────────

def apply_targeted_masking(combo_id: str, combo_feats: tuple, mode: str, tokenizer, mlm_prob: float = 0.15) -> list:
    """
    Apply targeted masking based on mode.

    Args:
        combo_id: The sequence string
        combo_feats: Tuple of feature names in this combination
        mode: The analysis mode
        tokenizer: Tokenizer to use for encoding/decoding
        mlm_prob: Masking probability

    Returns:
        List of tuples (input_ids, labels) with appropriate masking.
        For multi-molecule examples, returns multiple versions with each molecule masked.
    """
    if mode == "default":
        # Use standard MLM masking - return None to indicate standard processing
        return None

    # Tokenize the full sequence
    tokens = tokenizer.encode(combo_id)
    input_ids = tokens.ids if hasattr(tokens, 'ids') else tokens
    original_input_ids = input_ids.copy()

    # Get mask token ID
    mask_token_id = tokenizer.token_to_id("[MASK]") if hasattr(tokenizer, 'token_to_id') else getattr(tokenizer, 'mask_token_id', 103)

    if mode in ["tra", "trb"] :
        # For TRA/TRB modes: mask everything except first 4 and last 4 amino acids
        labels = [-100] * len(input_ids)

        # Find sequence boundaries (skip special tokens)
        seq_start = 0
        seq_end = len(input_ids)

        # Skip special tokens at the beginning
        for i, token_id in enumerate(input_ids):
            token = tokenizer.decode([token_id]) if hasattr(tokenizer, 'decode') else str(token_id)
            if not (token.startswith('[') or token in ['<s>', '</s>', '<pad>']):
                seq_start = i
                break

        # Find end of sequence (before padding/special tokens)
        for i in range(len(input_ids) - 1, -1, -1):
            if hasattr(tokenizer, 'pad_token_id') and input_ids[i] == tokenizer.pad_token_id:
                continue
            token = tokenizer.decode([input_ids[i]]) if hasattr(tokenizer, 'decode') else str(input_ids[i])
            if not (token.startswith('[') or token in ['<s>', '</s>', '<pad>']):
                seq_end = i + 1
                break

        # Mask middle section, keeping first 4 and last 4 amino acids
        preserve_start = min(seq_start + 4, seq_end)
        preserve_end = max(seq_end - 4, preserve_start)

        if preserve_end > preserve_start:
            # Mask the middle section
            for i in range(preserve_start, preserve_end):
                if hasattr(tokenizer, 'pad_token_id') and input_ids[i] != tokenizer.pad_token_id:
                    labels[i] = input_ids[i]  # Store original for loss calculation
                    input_ids[i] = mask_token_id
                elif not hasattr(tokenizer, 'pad_token_id') or input_ids[i] != 0:  # Assume 0 is pad
                    labels[i] = input_ids[i]
                    input_ids[i] = mask_token_id

        return [(input_ids, labels)]

    else:
        # For all other modes: generate multiple examples, one with each molecule masked
        # TODO: When dealing with single molecule in multiple molecule masking modes, implement a span masking approach
        if len(combo_feats) <= 1:
            # Single feature: mask the entire sequence (excluding special tokens)
            labels = [-100] * len(input_ids)
            seq_start = 0
            seq_end = len(input_ids)

            # Skip special tokens at start
            for i, token_id in enumerate(input_ids):
                token = tokenizer.decode([token_id]) if hasattr(tokenizer, 'decode') else str(token_id)
                if not (token.startswith('[') or token in ['<s>', '</s>', '<pad>']):
                    seq_start = i
                    break

            # Find end before padding
            for i in range(len(input_ids) - 1, -1, -1):
                if hasattr(tokenizer, 'pad_token_id') and input_ids[i] == tokenizer.pad_token_id:
                    continue
                token = tokenizer.decode([input_ids[i]]) if hasattr(tokenizer, 'decode') else str(input_ids[i])
                if not (token.startswith('[') or token in ['<s>', '</s>', '<pad>']):
                    seq_end = i + 1
                    break

            # Mask the entire sequence 
            # TODO: in a single molecule scenario can we do some sort of span masking
            for i in range(seq_start, seq_end):
                if hasattr(tokenizer, 'pad_token_id') and input_ids[i] != tokenizer.pad_token_id:
                    labels[i] = input_ids[i]
                    input_ids[i] = mask_token_id
                elif not hasattr(tokenizer, 'pad_token_id') or input_ids[i] != 0:
                    labels[i] = input_ids[i]
                    input_ids[i] = mask_token_id

            return [(input_ids, labels)]

        else:
            # Multiple features: create one masked example for each molecule
            # Find separators to identify sequence boundaries
            separators = []
            separator_tokens = ['[SEP]', '[ETRA]', '[ETRB]', '[EPEP]', '[EMHO]', '[EMHT]']

            for i, token_id in enumerate(input_ids):
                token = tokenizer.decode([token_id]) if hasattr(tokenizer, 'decode') else str(token_id)
                if any(sep in token for sep in separator_tokens):
                    separators.append(i)

            if len(separators) >= len(combo_feats) - 1:
                # Dataset already contain all permutations, just mask the first 
                # Create a fresh copy for this masked version
                labels = [-100] * len(input_ids)

                # First molecule: from start to first separator
                mask_start = 0
                mask_end = separators[0] if separators else len(input_ids)

                # Skip special tokens at start
                for i, token_id in enumerate(input_ids):
                    token = tokenizer.decode([token_id]) if hasattr(tokenizer, 'decode') else str(token_id)
                    if not (token.startswith('[') or token in ['<s>', '</s>', '<pad>']):
                        mask_start = i
                        break

            
                # Apply masking to this molecule's sequence
                for i in range(mask_start, mask_end):
                    if hasattr(tokenizer, 'pad_token_id') and input_ids[i] != tokenizer.pad_token_id:
                        labels[i] = input_ids[i]
                        input_ids[i] = mask_token_id
                    elif not hasattr(tokenizer, 'pad_token_id') or input_ids[i] != 0:
                        labels[i] = input_ids[i]
                        input_ids[i] = mask_token_id
                return [(input_ids, labels)]

            else:
                # Fallback: apply random 15% masking if we can't identify structure
                labels = [-100] * len(input_ids)
                import random
                for i, token_id in enumerate(input_ids):
                    if hasattr(tokenizer, 'pad_token_id') and token_id == tokenizer.pad_token_id:
                        continue

                    # Skip special tokens
                    token = tokenizer.decode([token_id]) if hasattr(tokenizer, 'decode') else str(token_id)
                    if token.startswith('[') or token in ['<s>', '</s>', '<pad>']:
                        continue

                    if random.random() < mlm_prob:
                        labels[i] = input_ids[i]
                        input_ids[i] = mask_token_id

                return [(input_ids, labels)]

def is_valid_sequence(seq:str):
    return isinstance(seq,str) and seq and all(c in _valid_aa for c in seq)

def validate_and_clean_sequence(seq:str) -> str:
    """
    Validate amino acid sequence and return 'NA' if invalid characters found.
    Optimized version for better performance.

    Args:
        seq: Input sequence (any type)

    Returns:
        str: Original sequence if valid, 'NA' if invalid or empty
    """
    # Fast type and empty checks
    if not seq or not isinstance(seq, str) or seq in ("", "NA", "None"):
        return "NA"

    # Fast character validation using set intersection
    # This is more efficient than all() for longer sequences
    if _valid_aa.issuperset(seq):
        return seq
    else:
        return "NA"

def safe_get(row: Dict[str, Any], key: str) -> str:
    v=row.get(key,"")
    return "" if v in (None,"NA") or (isinstance(v,float) and pd.isna(v)) else v

def deduplicate_by_mode(ds: Dataset, mode: str, num_proc: int = 10, batch_size: int = 10000) -> Dataset:
    """
    Apply mode-level deduplication after file concatenation.

    Args:
        ds: Concatenated dataset with all files merged
        mode: Analysis mode determining deduplication strategy

    Returns:
        Deduplicated dataset based on mode
    """
    print(f"🔄 Applying mode-level deduplication for mode: {mode}")
    original_size = len(ds)

    if mode == "tra":
        # 1. Pull tra column, remove null/empty/invalid, deduplicate
        # Use batched processing for speed
        def process_tra_batch(batch):
            # Optimized vectorized validation - avoid isinstance checks in loop
            tras = batch['tra']
            valid_mask = [
                bool(tra) and tra != 'NA' and _valid_aa.issuperset(tra)
                for tra in tras
            ]
            dedup_keys = [tra if valid else '' for tra, valid in zip(tras, valid_mask)]

            batch['_valid'] = valid_mask
            batch['_dedup_key'] = dedup_keys
            return batch

        ds = ds.map(process_tra_batch, batched=True, batch_size=batch_size, num_proc=num_proc, desc="Validating TRA")
        ds = ds.filter(lambda ex: ex['_valid'], num_proc=num_proc, desc="Filtering valid TRA")
        ds = ds.remove_columns(['_valid'])

        # Sort-based deduplication (optimized for large datasets)
        print(f"   ⏳ Consolidating chunks before sorting...")
        ds = ds.flatten_indices()  # Consolidate chunks to avoid Arrow's 16M chunk limit

        print(f"   ⏳ Sorting {len(ds):,} TRA sequences by dedup key...")
        ds = ds.sort('_dedup_key')

        print(f"   ⏳ Removing duplicates...")
        prev_key = [None]
        def keep_first(ex):
            if ex['_dedup_key'] != prev_key[0]:
                prev_key[0] = ex['_dedup_key']
                return True
            return False
        ds = ds.filter(keep_first, desc="Deduplicating TRA")
        ds = ds.remove_columns(['_dedup_key'])

    elif mode == "trb":
        # 2. Pull trb column, remove null/empty/invalid, deduplicate
        def process_trb_batch(batch):
            # Optimized vectorized validation
            trbs = batch['trb']
            valid_mask = [
                bool(trb) and trb != 'NA' and _valid_aa.issuperset(trb)
                for trb in trbs
            ]
            dedup_keys = [trb if valid else '' for trb, valid in zip(trbs, valid_mask)]

            batch['_valid'] = valid_mask
            batch['_dedup_key'] = dedup_keys
            return batch

        ds = ds.map(process_trb_batch, batched=True, batch_size=batch_size, num_proc=num_proc, desc="Validating TRB")
        ds = ds.filter(lambda ex: ex['_valid'], num_proc=num_proc, desc="Filtering valid TRB")
        ds = ds.remove_columns(['_valid'])

        print(f"   ⏳ Consolidating chunks before sorting...")
        ds = ds.flatten_indices()

        print(f"   ⏳ Sorting {len(ds):,} TRB sequences by dedup key...")
        ds = ds.sort('_dedup_key')

        print(f"   ⏳ Removing duplicates...")
        prev_key = [None]
        def keep_first(ex):
            if ex['_dedup_key'] != prev_key[0]:
                prev_key[0] = ex['_dedup_key']
                return True
            return False
        ds = ds.filter(keep_first, desc="Deduplicating TRB")
        ds = ds.remove_columns(['_dedup_key'])

    elif mode == "tcr_pairing":
        # 3. Pull tra+trb, remove null/empty/invalid, deduplicate on combination
        def process_pairing_batch(batch):
            # Optimized vectorized validation
            tras = batch['tra']
            trbs = batch['trb']

            valid_mask = [
                bool(tra) and tra != 'NA' and _valid_aa.issuperset(tra) and
                bool(trb) and trb != 'NA' and _valid_aa.issuperset(trb)
                for tra, trb in zip(tras, trbs)
            ]
            dedup_keys = [f"{tra}|{trb}" if valid else '' for tra, trb, valid in zip(tras, trbs, valid_mask)]

            batch['_valid'] = valid_mask
            batch['_dedup_key'] = dedup_keys
            return batch

        ds = ds.map(process_pairing_batch, batched=True, batch_size=batch_size, num_proc=num_proc, desc="Validating pairing")
        ds = ds.filter(lambda ex: ex['_valid'], num_proc=num_proc, desc="Filtering valid pairs")
        ds = ds.remove_columns(['_valid'])

        print(f"   ⏳ Consolidating chunks before sorting...")
        ds = ds.flatten_indices()

        print(f"   ⏳ Sorting {len(ds):,} pairs by dedup key...")
        ds = ds.sort('_dedup_key')

        print(f"   ⏳ Removing duplicates...")
        prev_key = [None]
        def keep_first(ex):
            if ex['_dedup_key'] != prev_key[0]:
                prev_key[0] = ex['_dedup_key']
                return True
            return False
        ds = ds.filter(keep_first, desc="Deduplicating pairs")
        ds = ds.remove_columns(['_dedup_key'])

    elif mode == "mhc_binding":
        # 4. Pull peptide+mhc_one+mhc_two, require peptide+mhc_one, deduplicate
        def process_mhc_batch(batch):
            # Optimized vectorized validation
            peptides = batch['peptide']
            mhc_ones = batch['mhc_one']
            mhc_twos = batch['mhc_two']

            # Pre-validate each column
            pep_valid = [bool(p) and p != 'NA' and _valid_aa.issuperset(p) for p in peptides]
            mho_valid = [bool(m) and m != 'NA' and _valid_aa.issuperset(m) for m in mhc_ones]
            mht_valid = [bool(m) and m != 'NA' and _valid_aa.issuperset(m) for m in mhc_twos]

            # Must have peptide AND mhc_one (mhc_two is optional)
            valid_mask = [pv and mv for pv, mv in zip(pep_valid, mho_valid)]
            dedup_keys = [
                f"{p}|{m1}|{m2 if mtv else ''}" if valid else ''
                for p, m1, m2, mtv, valid in zip(peptides, mhc_ones, mhc_twos, mht_valid, valid_mask)
            ]

            batch['_valid'] = valid_mask
            batch['_dedup_key'] = dedup_keys
            return batch

        ds = ds.map(process_mhc_batch, batched=True, batch_size=batch_size, num_proc=num_proc, desc="Validating MHC")
        ds = ds.filter(lambda ex: ex['_valid'], num_proc=num_proc, desc="Filtering valid MHC")
        ds = ds.remove_columns(['_valid'])

        print(f"   ⏳ Consolidating chunks before sorting...")
        ds = ds.flatten_indices()

        print(f"   ⏳ Sorting {len(ds):,} MHC sequences by dedup key...")
        ds = ds.sort('_dedup_key')

        print(f"   ⏳ Removing duplicates...")
        prev_key = [None]
        def keep_first(ex):
            if ex['_dedup_key'] != prev_key[0]:
                prev_key[0] = ex['_dedup_key']
                return True
            return False
        ds = ds.filter(keep_first, desc="Deduplicating MHC")
        ds = ds.remove_columns(['_dedup_key'])

    elif mode == "specificity":
        # 5. All 5 columns, require peptide+mhc_one+(tra OR trb), permutations, deduplicate
        def process_specificity_batch(batch):
            # Optimized vectorized validation
            tras = batch['tra']
            trbs = batch['trb']
            peptides = batch['peptide']
            mhc_ones = batch['mhc_one']
            mhc_twos = batch['mhc_two']

            # Pre-validate each column
            tra_valid = [bool(t) and t != 'NA' and _valid_aa.issuperset(t) for t in tras]
            trb_valid = [bool(t) and t != 'NA' and _valid_aa.issuperset(t) for t in trbs]
            pep_valid = [bool(p) and p != 'NA' and _valid_aa.issuperset(p) for p in peptides]
            mho_valid = [bool(m) and m != 'NA' and _valid_aa.issuperset(m) for m in mhc_ones]
            mht_valid = [bool(m) and m != 'NA' and _valid_aa.issuperset(m) for m in mhc_twos]

            # Must have: peptide + mhc_one + (tra OR trb)
            valid_mask = [pv and mv and (tv or tbv) for pv, mv, tv, tbv in zip(pep_valid, mho_valid, tra_valid, trb_valid)]

            # Build dedup keys
            dedup_keys = []
            for i, valid in enumerate(valid_mask):
                if valid:
                    parts = []
                    if mho_valid[i]: parts.append(f"mhc_one:{mhc_ones[i]}")
                    if mht_valid[i]: parts.append(f"mhc_two:{mhc_twos[i]}")
                    if pep_valid[i]: parts.append(f"peptide:{peptides[i]}")
                    if tra_valid[i]: parts.append(f"tra:{tras[i]}")
                    if trb_valid[i]: parts.append(f"trb:{trbs[i]}")
                    parts.sort()
                    dedup_keys.append('|'.join(parts))
                else:
                    dedup_keys.append('')

            batch['_valid'] = valid_mask
            batch['_dedup_key'] = dedup_keys
            return batch

        ds = ds.map(process_specificity_batch, batched=True, batch_size=batch_size, num_proc=num_proc, desc="Validating specificity")
        ds = ds.filter(lambda ex: ex['_valid'], num_proc=num_proc, desc="Filtering valid specificity")
        ds = ds.remove_columns(['_valid'])

        print(f"   ⏳ Consolidating chunks before sorting...")
        ds = ds.flatten_indices()

        print(f"   ⏳ Sorting {len(ds):,} specificity sequences by dedup key...")
        ds = ds.sort('_dedup_key')

        print(f"   ⏳ Removing duplicates...")
        prev_key = [None]
        def keep_first(ex):
            if ex['_dedup_key'] != prev_key[0]:
                prev_key[0] = ex['_dedup_key']
                return True
            return False
        ds = ds.filter(keep_first, desc="Deduplicating specificity")
        ds = ds.remove_columns(['_dedup_key'])

    elif mode == "default":
        # 6. All 5 columns, validate all, find all permutations of non-null, deduplicate
        def process_default_batch(batch):
            # Optimized vectorized validation
            tras = batch['tra']
            trbs = batch['trb']
            peptides = batch['peptide']
            mhc_ones = batch['mhc_one']
            mhc_twos = batch['mhc_two']

            # Pre-validate each column
            tra_valid = [bool(t) and t != 'NA' and _valid_aa.issuperset(t) for t in tras]
            trb_valid = [bool(t) and t != 'NA' and _valid_aa.issuperset(t) for t in trbs]
            pep_valid = [bool(p) and p != 'NA' and _valid_aa.issuperset(p) for p in peptides]
            mho_valid = [bool(m) and m != 'NA' and _valid_aa.issuperset(m) for m in mhc_ones]
            mht_valid = [bool(m) and m != 'NA' and _valid_aa.issuperset(m) for m in mhc_twos]

            # Build dedup keys - at least one valid column required
            valid_mask = []
            dedup_keys = []
            for i in range(len(tras)):
                parts = []
                if mho_valid[i]: parts.append(f"mhc_one:{mhc_ones[i]}")
                if mht_valid[i]: parts.append(f"mhc_two:{mhc_twos[i]}")
                if pep_valid[i]: parts.append(f"peptide:{peptides[i]}")
                if tra_valid[i]: parts.append(f"tra:{tras[i]}")
                if trb_valid[i]: parts.append(f"trb:{trbs[i]}")

                if parts:
                    valid_mask.append(True)
                    parts.sort()
                    dedup_keys.append('|'.join(parts))
                else:
                    valid_mask.append(False)
                    dedup_keys.append('')

            batch['_valid'] = valid_mask
            batch['_dedup_key'] = dedup_keys
            return batch

        ds = ds.map(process_default_batch, batched=True, batch_size=batch_size, num_proc=num_proc, desc="Validating default")
        ds = ds.filter(lambda ex: ex['_valid'], num_proc=num_proc, desc="Filtering valid default")
        ds = ds.remove_columns(['_valid'])

        print(f"   ⏳ Consolidating chunks before sorting...")
        ds = ds.flatten_indices()

        print(f"   ⏳ Sorting {len(ds):,} default sequences by dedup key...")
        ds = ds.sort('_dedup_key')

        print(f"   ⏳ Removing duplicates...")
        prev_key = [None]
        def keep_first(ex):
            if ex['_dedup_key'] != prev_key[0]:
                prev_key[0] = ex['_dedup_key']
                return True
            return False
        ds = ds.filter(keep_first, desc="Deduplicating default")
        ds = ds.remove_columns(['_dedup_key'])

    final_size = len(ds)
    removed = original_size - final_size
    retention_rate = (final_size / original_size * 100) if original_size > 0 else 0

    print(f"   Original: {original_size:,} rows")
    print(f"   After deduplication: {final_size:,} rows")
    print(f"   Removed: {removed:,} duplicates ({100-retention_rate:.1f}%)")

    return ds

def tag_bpe(row: Dict[str, Any]) -> List[str]:
    mapping=[("tra","tra_tokens"),("trb","trb_tokens"),("peptide","pep_tokens"),
             ("mhc_one","mho_tokens"),("mhc_two","mht_tokens")]
    tagged=[]
    for col,tkkey in mapping:
        val=safe_get(row,col)
        if val and _valid_aa.issuperset(val):
            start_tk,end_tk=BPE_TOKENS[tkkey]
            tagged.append(f"{start_tk} {' '.join(val)} {end_tk}")
    return tagged

def train_bpe_tokenizer(seqs: List[str], vocab_size: int) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Split(pattern=r" ", behavior="removed")
    special = [t for pair in BPE_TOKENS.values() for t in pair if t]
    trainer = trainers.BpeTrainer(special_tokens=special, vocab_size=vocab_size)
    tok.train_from_iterator(seqs, trainer)
    return tok

class AminoAcidDataset(TorchDataset):
    def __init__(self, sequences: List[str], tokenizer: Tokenizer, seq_len: int = 128, model: str = "rnn", step: int = 1, tr_long: bool = True):
        self.tok, self.seq_len=tokenizer,seq_len; self.model=model; self.step=step; self.tr_long=tr_long
        self.pad_id=tokenizer.token_to_id(PAD_TOKEN_STR) or PAD_TOKEN_ID; self.samples=[]; self._build(sequences)
    def _build(self, seqs: List[str]) -> None:
        return self._build_rnn(seqs) if self.model=="rnn" else self._build_tx(seqs)
    def _build_rnn(self, seqs: List[str]) -> None:
        for s in seqs:
            ids=self.tok.encode(s).ids
            if len(ids)>=self.seq_len+1:
                n=(len(ids)-(self.seq_len+1))//self.step+1
                for i in range(n):
                    chunk=ids[i*self.step:i*self.step+self.seq_len+1]
                    self.samples.append((chunk[:-1],chunk[1:]))
            else:
                pad=ids+[self.pad_id]*((self.seq_len+1)-len(ids))
                self.samples.append((pad[:-1],pad[1:]))
    def _build_tx(self, seqs: List[str]) -> None:
        for s in seqs:
            ids=self.tok.encode(s).ids
            if len(ids)>self.seq_len and self.tr_long:
                ids=ids[:self.seq_len]
            if len(ids)<self.seq_len:
                ids+=[self.pad_id]*(self.seq_len-len(ids))
            if len(ids)>=2:
                self.samples.append((ids[:-1],ids[1:]))
    def __len__(self) -> int: return len(self.samples)
    def __getitem__(self, i: int):
        x,y=self.samples[i]; return torch.tensor(x),torch.tensor(y)
        
def load_single_dataset(data_file: Any, storage_options: Dict[str, Any], cache_dir: str = None, keep_in_memory: bool = False):
    """
    Worker function to load a single dataset file. This will be executed
    in parallel by the ThreadPoolExecutor.

    NOTE: File-level deduplication removed for speed - we do global deduplication later anyway.
          Loading 10 files with 50M rows each was taking 2 hours per file with dedup.
          Now takes seconds per file.
    """
    # Simple fast load - no per-file deduplication
    ds = load_dataset(
        "parquet",
        data_files=str(data_file),
        split="train",
        storage_options=storage_options,
        cache_dir=cache_dir,
        keep_in_memory=keep_in_memory
    )
    return ds

    
def encode_bpe_batch(batch: Dict[str, Any], *, tokenizer: Tokenizer, seq_len: int, model_type: str, trunc_long: bool) -> Dict[str, List[List[int]]]:
    out_in,out_tgt=[],[]
    
    # Process each sequence in the batch
    for tagged_seq in batch["tagged"]:
        ds=AminoAcidDataset(tagged_seq,tokenizer,seq_len,model_type,tr_long=trunc_long)
        for x,y in ds:
            # Ensure all sequences have the same length
            x_list = x.tolist()
            y_list = y.tolist()
            
            # Pad or truncate to seq_len
            if len(x_list) < seq_len:
                x_list.extend([tokenizer.token_to_id(PAD_TOKEN_STR) or PAD_TOKEN_ID] * (seq_len - len(x_list)))
                y_list.extend([tokenizer.token_to_id(PAD_TOKEN_STR) or PAD_TOKEN_ID] * (seq_len - len(y_list)))
            elif len(x_list) > seq_len:
                x_list = x_list[:seq_len]
                y_list = y_list[:seq_len]
            
            out_in.append(x_list)
            out_tgt.append(y_list)
    
    # Ensure consistent batch size by padding with dummy examples if needed
    # This is a workaround for the ArrowInvalid error
    if len(out_in) == 0:
        # If no examples were generated, create dummy examples
        pad_id = tokenizer.token_to_id(PAD_TOKEN_STR) or PAD_TOKEN_ID
        dummy_input = [pad_id] * seq_len
        dummy_target = [pad_id] * seq_len
        out_in = [dummy_input]
        out_tgt = [dummy_target]
    
    return {"input_ids":out_in,"target_ids":out_tgt}

# ── Dynamic Resource Detection -----------------------------------------

# Simple resource configuration
def get_num_workers() -> int:
    """Get number of workers for parallel processing."""
    return min(os.cpu_count() or 1, 32)  # Max 32 workers

def get_batch_size() -> int:
    """Get batch size for processing."""
    return 1000

# ── CLI -----------------------------------------------------------------

def cli():
    p=argparse.ArgumentParser()
    p.add_argument("--path",nargs="+", help="Path to original Parquet files (for a full run).")
    p.add_argument("--model-name",required=True,
                   choices=["bert","protbert","esm","llama","lstm","transformer"])
    p.add_argument("--mode", required=False, default="specificity",
                   choices=["tra", "trb", "tcr_pairing", "mhc_binding", "specificity", "default"],
                   help="Analysis mode: tra (TRA only), trb (TRB only), tcr_pairing (TRA+TRB), mhc_binding (MHC+peptide), specificity (complete TCR complexes), default (all combinations except peptide-only)")
    p.add_argument("--input_raw_dir", default=None, help="Optional: Path to an existing 'raw' dataset to start masking from.")
    p.add_argument("--output-raw", required=True, help="Path to save the tokenized dataset with text columns.")
    p.add_argument("--max-len",type=int,default=1024)
    p.add_argument("--bpe-vocab",type=int,default=200)
    p.add_argument("--truncate-long",action="store_true")
    p.add_argument("--mlm-prob",type=float,default=0.15,
                   help="Probability of masking tokens in MLM. Default is 0.15.")
    p.add_argument("--s3-key",default=os.getenv("AWS_ACCESS_KEY_ID"))
    p.add_argument("--s3-secret",default=os.getenv("AWS_SECRET_ACCESS_KEY"))
    p.add_argument("--s3-token",default=os.getenv("AWS_SESSION_TOKEN"))
    p.add_argument("--num-proc", type=int, default=None, help="Number of processes for data processing (default: auto-detect)")
    p.add_argument("--batch-size", type=int, default=10000, help="Batch size for dataset processing operations (default: 10000, recommended: 50000 for x2gd.16xlarge)")
    p.add_argument("--tmp-dir", type=str, default="tmp_processed", help="Directory for temporary/intermediate files during processing.")
    p.add_argument("--sample", type=int, default=None, help="Sample N files for testing (e.g., 10 for first 10 files). Applied during file selection.")
    p.add_argument("--in-memory", action="store_true", help="Keep datasets in memory to avoid disk caching (use for small datasets only)")
    return p.parse_args()

# Multi-node task distribution helper
def main():
    """Simplified main function without Ray."""
    args = cli()

    # Setup cache directories
    cache_dir = os.path.join(args.tmp_dir, "hf_cache")
    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Set all HuggingFace cache environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
    os.environ["HF_HUB_CACHE"] = os.path.join(cache_dir, "hub")
    os.environ["TMPDIR"] = args.tmp_dir
    os.environ["TEMP"] = args.tmp_dir
    os.environ["TMP"] = args.tmp_dir

    # Force Arrow to use tmpdir for memory-mapped files
    os.environ["ARROW_DEFAULT_MEMORY_POOL"] = "system"

    # Disable progress bars and verbose logging for cleaner output (set BEFORE importing datasets)
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_DATASETS_VERBOSITY"] = "error"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    # Set datasets library to use specified cache directory
    import datasets
    datasets.config.HF_DATASETS_CACHE = os.path.join(cache_dir, "datasets")
    datasets.config.DOWNLOADED_DATASETS_PATH = os.path.join(cache_dir, "downloads")
    datasets.config.EXTRACTED_DATASETS_PATH = os.path.join(cache_dir, "extracted")

    # Disable datasets library logging
    import logging
    logging.getLogger("datasets").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # Disable tqdm output for load_dataset
    datasets.logging.set_verbosity_error()

    # Completely disable internal datasets progress bars
    datasets.utils.logging.disable_progress_bar()
    datasets.disable_progress_bar()

    # Disable dataset caching if in-memory mode
    if args.in_memory:
        print("⚠️  IN-MEMORY MODE: Disabling dataset caching to save disk space")
        os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = "0"  # Force in-memory
        from datasets import disable_caching
        disable_caching()

    num_proc = args.num_proc or get_num_workers()
    
    print("="*80)
    print("🧬 T-CELL RECEPTOR ANALYSIS MODE")
    print("="*80)
    print(f"📊 Mode: {args.mode}")
    print(f"💾 Cores: {num_proc}")
    print(f"📦 Batch Size: {args.batch_size:,}")
    print("="*80)
    
    # S3 options
    s3_options = {
        "key": args.s3_key, 
        "secret": args.s3_secret, 
        "token": args.s3_token
    }
    
    # Load data files
    print("Resolving data files...")
    resolved_data_files = DataFilesList.from_patterns(args.path)
    print(f"Found {len(resolved_data_files)} files")

    # Apply sampling if requested
    if args.sample is not None:
        if args.sample <= 0:
            raise ValueError(f"Sample must be positive, got {args.sample}")
        if args.sample < len(resolved_data_files):
            print(f"🎲 Sampling {args.sample} files for testing...")
            resolved_data_files = resolved_data_files[:args.sample]
            print(f"Using {len(resolved_data_files)} files")
        else:
            print(f"⚠️  Sample size ({args.sample}) >= total files ({len(resolved_data_files)}), using all files")

    # Load datasets in parallel
    print("Loading datasets...")
    with ThreadPoolExecutor(max_workers=min(num_proc, len(resolved_data_files))) as executor:
        datasets = list(tqdm(
            executor.map(lambda f: load_single_dataset(f, s3_options, cache_dir=cache_dir, keep_in_memory=args.in_memory), resolved_data_files),
            total=len(resolved_data_files),
            desc="Loading files",
            unit="file"
        ))
    
    # Filter out empty datasets
    datasets = [ds for ds in datasets if len(ds) > 0]
    print(f"Loaded {len(datasets)} non-empty datasets")

    # Concatenate
    print("Concatenating datasets...")
    base = concatenate_datasets(datasets)
    print(f"Total examples before mode deduplication: {len(base):,}")

    # Multi-phase progressive deduplication
    print("\n" + "="*80)
    print("🔄 MULTI-PHASE PROGRESSIVE DEDUPLICATION")
    print("="*80)

    # Phase 1: Initial deduplication on full dataset
    print("\n📍 PHASE 1: Initial mode-level deduplication")
    base = deduplicate_by_mode(base, args.mode, num_proc=num_proc, batch_size=args.batch_size)
    print(f"   After Phase 1: {len(base):,} rows")

    # Phase 2: Split into 20 chunks, deduplicate each, concatenate
    print("\n📍 PHASE 2: Split into 20 chunks, deduplicate & merge")
    total_rows = len(base)
    if total_rows > 1000:  # Only do chunked dedup if we have enough data
        chunk_size = max(1, total_rows // 20)
        chunks = []

        # Calculate chunk ranges
        chunk_ranges = []
        for i in range(0, total_rows, chunk_size):
            end_idx = min(i + chunk_size, total_rows)
            chunk_ranges.append((i, end_idx))

        for i, end_idx in tqdm(chunk_ranges, desc="Phase 2 chunks", unit="chunk"):
            chunk = base.select(range(i, end_idx))
            chunk_deduped = deduplicate_by_mode(chunk, args.mode, num_proc=num_proc, batch_size=args.batch_size)
            chunks.append(chunk_deduped)

        base = concatenate_datasets(chunks)
        print(f"   After Phase 2: {len(base):,} rows (concatenated)")
    else:
        print(f"   Skipping Phase 2 (dataset too small: {total_rows} rows)")

    # Phase 3: Split into 5 chunks, deduplicate each, concatenate
    print("\n📍 PHASE 3: Split into 5 chunks, deduplicate & merge")
    total_rows = len(base)
    if total_rows > 500:  # Only do chunked dedup if we have enough data
        chunk_size = max(1, total_rows // 5)
        chunks = []

        # Calculate chunk ranges
        chunk_ranges = []
        for i in range(0, total_rows, chunk_size):
            end_idx = min(i + chunk_size, total_rows)
            chunk_ranges.append((i, end_idx))

        for i, end_idx in tqdm(chunk_ranges, desc="Phase 3 chunks", unit="chunk"):
            chunk = base.select(range(i, end_idx))
            chunk_deduped = deduplicate_by_mode(chunk, args.mode, num_proc=num_proc, batch_size=args.batch_size)
            chunks.append(chunk_deduped)

        base = concatenate_datasets(chunks)
        print(f"   After Phase 3: {len(base):,} rows (concatenated)")
    else:
        print(f"   Skipping Phase 3 (dataset too small: {total_rows} rows)")

    # Phase 4: Final global deduplication
    print("\n📍 PHASE 4: Final global deduplication")
    base = deduplicate_by_mode(base, args.mode, num_proc=num_proc, batch_size=args.batch_size)
    print(f"   After Phase 4: {len(base):,} rows")

    print("\n" + "="*80)
    print(f"✅ DEDUPLICATION COMPLETE: {len(base):,} unique rows")
    print("="*80 + "\n")

    # Step 2/3: Explode sequences for modes that need it
    if args.mode in ["mhc_binding", "specificity", "default", "tcr_pairing"]:
        print("💥 Exploding sequences...")

        def explode_row(ex):
            return explode_example(ex, args.mode)

        base = base.map(explode_row, batched=False, num_proc=num_proc, desc=None)

        # Unnest
        all_seqs = []
        all_feats = []
        for row in base:
            all_seqs.extend(row['sequences'])
            all_feats.extend(row['feat_names'])

        base = Dataset.from_dict({
            'sequence_tuple': all_seqs,
            'feat_names': all_feats
        })
        print(f"After explosion: {len(base):,} sequences")

        # Deduplicate exact sequences for mhc_binding, specificity, default
        if args.mode in ["mhc_binding", "specificity", "default"]:
            print("🔄 Deduplicating exact sequences...")
            seen = set()
            def dedup(ex):
                key = tuple(sorted(ex['sequence_tuple']))
                if key not in seen:
                    seen.add(key)
                    return True
                return False
            base = base.filter(dedup)
            print(f"After dedup: {len(base):,} unique sequences\n")

    # Split 80/10/10
    print("Splitting dataset...")
    sp1 = base.train_test_split(test_size=0.2, seed=42)
    sp2 = sp1["test"].train_test_split(test_size=0.5, seed=42)
    splits = DatasetDict({
        "train": sp1["train"],
        "validation": sp2["train"],
        "test": sp2["test"]
    })

    # Step 4: Format sequences for model
    model_type = args.model_name
    print(f"\n🎯 Formatting for {model_type}...")

    if model_type in ["bert", "protbert", "esm", "llama"]:
        # Format sequences based on model
        for split_name in ["train", "validation", "test"]:
            print(f"📝 Formatting {split_name} split for {model_type}...")
            ds = splits[split_name]

            if 'sequence_tuple' in ds.column_names:
                # Already exploded, just format
                def format_seq(ex):
                    return {'combo_id': format_sequence_for_model(ex['sequence_tuple'], model_type),
                            'combo_feats': ex['feat_names']}
                ds = ds.map(format_seq, batched=False, num_proc=num_proc, desc=f"Formatting {split_name}")
                ds = ds.remove_columns(['sequence_tuple', 'feat_names'])
            else:
                # Need to explode for tra/trb modes
                def explode_and_format(ex):
                    result = explode_example(ex, args.mode)
                    if result['sequences']:
                        return {'combo_id': [format_sequence_for_model(seq, model_type) for seq in result['sequences']],
                                'combo_feats': result['feat_names']}
                    return {'combo_id': [], 'combo_feats': []}

                ds = ds.map(explode_and_format, batched=False, num_proc=num_proc, desc=f"Exploding & formatting {split_name}")
                # Unnest
                print(f"   Unnesting {split_name} sequences...")
                ds = Dataset.from_dict({
                    'combo_id': sum(ds['combo_id'], []),
                    'combo_feats': sum(ds['combo_feats'], [])
                })

            splits[split_name] = ds
            print(f"   ✓ {split_name}: {len(ds):,} sequences")

        splits.save_to_disk(args.output_raw)
        print(f"✅ Saved to {args.output_raw}")
        
    elif model_type in ["lstm", "transformer"]:
        # Custom BPE path
        print(f"Processing with custom BPE for {model_type}...")

        # Train tokenizer on train split
        def get_sequences(ex):
            return {"tagged": tag_bpe(ex)}

        print("📝 Tagging train sequences for BPE training...")
        train_ds = splits["train"].map(get_sequences, batched=False, num_proc=num_proc, desc="Tagging sequences")
        all_seqs = sum(train_ds["tagged"], [])
        print(f"Training BPE tokenizer on {len(all_seqs):,} sequences...")

        tokenizer = train_bpe_tokenizer(all_seqs, args.bpe_vocab)
        tok_path = f"{args.output_raw}_tokenizer.json"
        tokenizer.save(tok_path)
        print(f"✓ Tokenizer saved to {tok_path}")

        # Encode all splits
        for split_name in ["train", "validation", "test"]:
            print(f"📝 Encoding {split_name} split...")
            ds = splits[split_name].map(get_sequences, num_proc=num_proc, batched=False, desc=f"Tagging {split_name}")
            ds = ds.map(
                lambda batch: encode_bpe_batch(
                    batch,
                    tokenizer=tokenizer,
                    seq_len=args.max_len,
                    model_type="rnn" if model_type == "lstm" else "transformer",
                    trunc_long=args.truncate_long
                ),
                batched=True,
                batch_size=get_batch_size(),
                num_proc=num_proc,
                desc=f"Encoding {split_name}"
            )
            splits[split_name] = ds
            print(f"   ✓ {split_name}: {len(ds):,} encoded sequences")
        
        # Save
        splits.save_to_disk(args.output_raw)
        print(f"✅ Saved to {args.output_raw}")
    
    print("✨ Complete!")


if __name__ == "__main__":
    main()
