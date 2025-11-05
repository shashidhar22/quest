#!/usr/bin/python3
# -*- coding: utf-8 -*-
# type: ignore
"""
datawriter.py  *Ray Data version*
──────────────────────────────────────────────
• Reads Parquet shards from local filesystem
• Handles datasets larger than RAM via disk spilling
• Splits 80/10/10 → train / val / test
• Two branches controlled by --model-name

  1.  bert | protbert | esm | llama
        explode → HF tokenizer → DatasetDict

  2.  custom-rnn | custom-transformer
        ▸ train a BPE tokenizer on **train** split
        ▸ tag sequences with custom boundary tokens
        ▸ encode via AminoAcidDataset **inside datasets.map**
        ▸ returns DatasetDict with {input_ids,target_ids}

Usage:
    python scripts/ray_datawriter.py \
        --path "./data/**/*.parquet" \
        --model-name bert \
        --output-raw ./output
"""

import os
import gc
import argparse
from itertools import combinations, permutations
from pathlib import Path
from typing import List, Dict, Any
import numpy
from datasets import Dataset, DatasetDict  # Only keep what's needed for final output
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset as TorchDataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import pandas as pd  # type: ignore
import random
import psutil
import time
import sys
import json
import ray
import ray.data



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
        "default": set(["tra", "trb", "peptide", "mhc_one", "mhc_two"]),
        "balanced": set(["tra", "trb", "peptide", "mhc_one", "mhc_two"])}

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

def print_progress(stage: str, message: str, indent: int = 1):
    """Helper function for consistent progress reporting."""
    prefix = "   " * indent
    print(f"{prefix}{stage} {message}", flush=True)

def balanced_sample_by_molecules(ds: ray.data.Dataset, samples_per_combo: int = None) -> ray.data.Dataset:
    """
    Apply balanced sampling across all molecule permutations.
    
    For the 5 molecules (tra, trb, peptide, mhc_one, mhc_two), there are:
    - Single molecules: 5 combinations (tra, trb, peptide, mhc_one, mhc_two)
    - Pairs: C(5,2) = 10 combinations
    - Triplets: C(5,3) = 10 combinations  
    - Quadruplets: C(5,4) = 5 combinations
    - All five: 1 combination
    Total: 31 unique combinations (excluding empty set)
    
    But we actually care about PERMUTATIONS (order matters), so:
    - For k molecules chosen from 5: P(5,k) = 5!/(5-k)!
    - Total permutations: sum(P(5,k) for k=1..5) = 325 permutations
    
    This function ensures each permutation gets equal representation in the dataset.
    
    Args:
        ds: Ray Dataset with 'combo_feats' column (tuple of molecule names)
        samples_per_combo: Number of samples per combination. If None, uses minimum count.
    
    Returns:
        Balanced dataset with equal samples per molecule combination
    """
    print("\n🎯 BALANCED SAMPLING BY MOLECULE PERMUTATIONS")
    print("="*80)
    
    import time
    start_time = time.time()
    
    # Step 1: Add permutation key to each row
    def add_perm_key(row):
        """Add permutation key based on combo_feats ordering."""
        combo = row.get('combo_feats', ())
        if isinstance(combo, tuple):
            # Use tuple as-is (order matters for permutations)
            perm_key = '|'.join(sorted(combo))  # Sort for grouping, not for identity
            return {**row, '_perm_key': perm_key}
        return {**row, '_perm_key': ''}
    
    print_progress("⏳", "Step 1/4: Tagging rows with permutation keys...")
    ds_with_keys = ds.map(add_perm_key)
    
    # Step 2: Count samples per permutation
    print_progress("⏳", "Step 2/4: Counting samples per permutation...")
    
    # Use Ray Data's groupby to count
    grouped = ds_with_keys.groupby('_perm_key').count()
    count_results = grouped.materialize()
    
    # Build dictionary of counts
    perm_counts = {}
    for row in count_results.iter_rows():
        key = row['_perm_key']
        count = row['count()']
        if key and key != '':
            perm_counts[key] = count
    
    if not perm_counts:
        print("   ⚠️  No permutations found!")
        return ds
    
    # Print statistics
    total_perms = len(perm_counts)
    total_samples = sum(perm_counts.values())
    min_count = min(perm_counts.values())
    max_count = max(perm_counts.values())
    avg_count = total_samples / total_perms
    
    print(f"\n   📊 Permutation Statistics:")
    print(f"      Total unique permutations: {total_perms}")
    print(f"      Total samples: {total_samples:,}")
    print(f"      Min samples per permutation: {min_count:,}")
    print(f"      Max samples per permutation: {max_count:,}")
    print(f"      Avg samples per permutation: {avg_count:,.1f}")
    print(f"      Imbalance ratio: {max_count/min_count:.1f}x")
    
    # Determine target samples per permutation
    if samples_per_combo is None:
        target_samples = min_count
        print(f"\n   🎯 Using minimum count as target: {target_samples:,} samples per permutation")
    else:
        target_samples = samples_per_combo
        print(f"\n   🎯 Using specified target: {target_samples:,} samples per permutation")
    
    # Show top 10 most and least represented permutations
    sorted_perms = sorted(perm_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n   📈 Top 10 most represented permutations:")
    for i, (perm, count) in enumerate(sorted_perms[:10], 1):
        print(f"      {i:2d}. {perm:40s} : {count:>8,} samples")
    
    print(f"\n   📉 Top 10 least represented permutations:")
    for i, (perm, count) in enumerate(sorted_perms[-10:][::-1], 1):
        print(f"      {i:2d}. {perm:40s} : {count:>8,} samples")
    
    # Step 3: Sample from each permutation
    print_progress("⏳", "Step 3/4: Sampling from each permutation...")
    
    def sample_within_group(batch):
        """Sample target_samples from each permutation group."""
        import pandas as pd
        import random
        
        df = pd.DataFrame(batch)
        if len(df) == 0:
            return df
        
        # Group by permutation key
        sampled_dfs = []
        for perm_key, group in df.groupby('_perm_key'):
            if perm_key and perm_key != '':
                # Sample with replacement if needed, without replacement otherwise
                n_samples = min(target_samples, len(group))
                if n_samples == len(group):
                    # Take all samples
                    sampled = group
                elif n_samples > len(group):
                    # Sample with replacement
                    sampled = group.sample(n=target_samples, replace=True, random_state=42)
                else:
                    # Sample without replacement
                    sampled = group.sample(n=n_samples, random_state=42)
                
                sampled_dfs.append(sampled)
        
        if sampled_dfs:
            result = pd.concat(sampled_dfs, ignore_index=True)
            # Drop temp column
            result = result.drop(columns=['_perm_key'])
            return result
        return pd.DataFrame()
    
    # Sort by permutation key first to group consecutive rows
    print_progress("   ", "→ Sorting by permutation key...", indent=2)
    ds_sorted = ds_with_keys.sort(key='_perm_key')
    
    print_progress("   ", "→ Sampling within each group...", indent=2)
    ds_sampled = ds_sorted.map_batches(sample_within_group, batch_format="pandas")
    
    # Step 4: Materialize and shuffle
    print_progress("⏳", "Step 4/4: Materializing and shuffling...")
    ds_sampled = ds_sampled.materialize()
    
    # Shuffle to mix permutations
    print_progress("   ", "→ Shuffling to mix permutations...", indent=2)
    ds_sampled = ds_sampled.random_shuffle(seed=42)
    
    # Get final count
    final_count = ds_sampled.count()
    expected_count = total_perms * target_samples
    
    elapsed = time.time() - start_time
    
    print(f"\n   ✓ Balanced sampling complete!")
    print(f"      Original samples: {total_samples:,}")
    print(f"      Target samples: {expected_count:,} ({total_perms} perms × {target_samples:,})")
    print(f"      Final samples: {final_count:,}")
    print(f"      Time: {elapsed:.1f}s")
    print(f"      Retention rate: {final_count/total_samples*100:.1f}%")
    print("="*80 + "\n")
    
    return ds_sampled

def deduplicate_ray_data_fast(ds: ray.data.Dataset, mode: str) -> tuple:
    """
    Fast Ray Data deduplication - skips expensive duplicate analysis.

    This is 10-100x faster than the detailed version because it:
    - Skips the initial count (saves one full dataset scan)
    - Skips detailed duplicate statistics (saves expensive groupby + materialize)
    - Only performs the minimal operations needed for deduplication

    Args:
        ds: Ray Dataset (streaming)
        mode: Analysis mode (tra, trb, tcr_pairing, mhc_binding, specificity, default, balanced)

    Returns:
        tuple: (deduplicated_dataset, stats_dict)
    """
    print(f"\n🚀 FAST DEDUPLICATION (Mode: {mode})")
    print("="*80)
    print("⚡ Skipping detailed duplicate analysis for maximum speed")
    stats = {}
    import time

    # Define validation and dedup key function based on mode
    def add_dedup_key(row):
        """Add validation and dedup key to each row. Returns NEW dict (thread-safe)."""
        if mode == "tra":
            tra = row.get('tra', '')
            valid = bool(tra) and tra != 'NA' and _valid_aa.issuperset(tra)
            return {**row, '_dedup_key': tra if valid else '', '_valid': valid}

        elif mode == "trb":
            trb = row.get('trb', '')
            valid = bool(trb) and trb != 'NA' and _valid_aa.issuperset(trb)
            return {**row, '_dedup_key': trb if valid else '', '_valid': valid}

        elif mode == "tcr_pairing":
            tra = row.get('tra', '')
            trb = row.get('trb', '')
            tra_valid = bool(tra) and tra != 'NA' and _valid_aa.issuperset(tra)
            trb_valid = bool(trb) and trb != 'NA' and _valid_aa.issuperset(trb)
            valid = tra_valid and trb_valid
            return {**row, '_dedup_key': f"{tra}|{trb}" if valid else '', '_valid': valid}

        elif mode == "mhc_binding":
            pep = row.get('peptide', '')
            mho = row.get('mhc_one', '')
            mht = row.get('mhc_two', '')
            pep_valid = bool(pep) and pep != 'NA' and _valid_aa.issuperset(pep)
            mho_valid = bool(mho) and mho != 'NA' and _valid_aa.issuperset(mho)
            mht_valid = bool(mht) and mht != 'NA' and _valid_aa.issuperset(mht)
            valid = pep_valid and mho_valid
            return {**row, '_dedup_key': f"{pep}|{mho}|{mht if mht_valid else ''}" if valid else '', '_valid': valid}

        elif mode == "specificity":
            tra = row.get('tra', '')
            trb = row.get('trb', '')
            pep = row.get('peptide', '')
            mho = row.get('mhc_one', '')
            mht = row.get('mhc_two', '')

            tra_valid = bool(tra) and tra != 'NA' and _valid_aa.issuperset(tra)
            trb_valid = bool(trb) and trb != 'NA' and _valid_aa.issuperset(trb)
            pep_valid = bool(pep) and pep != 'NA' and _valid_aa.issuperset(pep)
            mho_valid = bool(mho) and mho != 'NA' and _valid_aa.issuperset(mho)
            mht_valid = bool(mht) and mht != 'NA' and _valid_aa.issuperset(mht)

            valid = pep_valid and mho_valid and (tra_valid or trb_valid)

            if valid:
                parts = []
                if mho_valid: parts.append(f"mhc_one:{mho}")
                if mht_valid: parts.append(f"mhc_two:{mht}")
                if pep_valid: parts.append(f"peptide:{pep}")
                if tra_valid: parts.append(f"tra:{tra}")
                if trb_valid: parts.append(f"trb:{trb}")
                parts.sort()
                dedup_key = '|'.join(parts)
            else:
                dedup_key = ''

            return {**row, '_dedup_key': dedup_key, '_valid': valid}

        else:  # default or balanced mode (both treat data the same way during deduplication)
            tra = row.get('tra', '')
            trb = row.get('trb', '')
            pep = row.get('peptide', '')
            mho = row.get('mhc_one', '')
            mht = row.get('mhc_two', '')

            tra_valid = bool(tra) and tra != 'NA' and _valid_aa.issuperset(tra)
            trb_valid = bool(trb) and trb != 'NA' and _valid_aa.issuperset(trb)
            pep_valid = bool(pep) and pep != 'NA' and _valid_aa.issuperset(pep)
            mho_valid = bool(mho) and mho != 'NA' and _valid_aa.issuperset(mho)
            mht_valid = bool(mht) and mht != 'NA' and _valid_aa.issuperset(mht)

            parts = []
            if mho_valid: parts.append(f"mhc_one:{mho}")
            if mht_valid: parts.append(f"mhc_two:{mht}")
            if pep_valid: parts.append(f"peptide:{pep}")
            if tra_valid: parts.append(f"tra:{tra}")
            if trb_valid: parts.append(f"trb:{trb}")

            valid = len(parts) > 0
            if valid:
                parts.sort()
                dedup_key = '|'.join(parts)
            else:
                dedup_key = ''

            return {**row, '_dedup_key': dedup_key, '_valid': valid}

    # Pipeline: Add keys, filter, sort, deduplicate
    start_time = time.time()

    print_progress("⏳", "Step 1/3: Adding dedup keys and filtering invalid sequences...")
    ds_with_keys = ds.map(add_dedup_key)
    ds_filtered = ds_with_keys.filter(lambda row: row['_valid'])

    print_progress("⏳", "Step 2/3: Sorting by dedup key (external merge sort)...")
    sort_start = time.time()
    ds_sorted = ds_filtered.sort(key='_dedup_key')
    sort_time = time.time() - sort_start
    print_progress("✓", f"Sorting complete (took {sort_time:.1f}s)")

    print_progress("⏳", "Step 3/3: Removing consecutive duplicates...")
    dedup_start = time.time()

    def drop_consecutive_duplicates(batch):
        """Keep only first occurrence of consecutive duplicate keys."""
        import pandas as pd
        df = pd.DataFrame(batch)
        if len(df) == 0:
            return df

        # Mark first occurrence of each unique _dedup_key
        df['_keep'] = df['_dedup_key'].ne(df['_dedup_key'].shift())

        # Filter and drop temp columns
        df_filtered = df[df['_keep']].drop(columns=['_dedup_key', '_valid', '_keep'])

        return df_filtered

    ds_deduped = ds_sorted.map_batches(drop_consecutive_duplicates, batch_format="pandas")

    # Only materialize and count at the end
    print_progress("   ", "→ Materializing deduplicated dataset...", indent=2)
    ds_deduped = ds_deduped.materialize()

    # Get final count (fast after materialization)
    final_count = ds_deduped.count()
    total_time = time.time() - start_time

    print_progress("✓", f"Fast deduplication complete: {final_count:,} unique rows")
    print(f"   ⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"   ⚡ Speed: {final_count/total_time:,.0f} rows/sec")
    print("="*80 + "\n")

    # Minimal stats (don't compute expensive metrics)
    stats['rows_after_deduplication'] = final_count
    stats['rows_before_deduplication'] = 'not_computed_in_fast_mode'
    stats['duplicate_sequences'] = 'not_computed_in_fast_mode'

    return ds_deduped, stats

def deduplicate_ray_data(ds: ray.data.Dataset, mode: str) -> tuple:
    """
    Ray Data deduplication - simple and memory-efficient.

    Args:
        ds: Ray Dataset (streaming)
        mode: Analysis mode (tra, trb, tcr_pairing, mhc_binding, specificity, default, balanced)

    Returns:
        tuple: (deduplicated_dataset, stats_dict)
    """
    print(f"\n🔄 DEDUPLICATION PIPELINE (Mode: {mode})")
    print("=" * 80)
    stats = {}
    import time

    # Define validation and dedup key function based on mode
    def add_dedup_key(row):
        """Add validation and dedup key to each row. Returns NEW dict (thread-safe)."""
        if mode == "tra":
            tra = row.get('tra', '')
            valid = bool(tra) and tra != 'NA' and _valid_aa.issuperset(tra)
            return {**row, '_dedup_key': tra if valid else '', '_valid': valid}

        elif mode == "trb":
            trb = row.get('trb', '')
            valid = bool(trb) and trb != 'NA' and _valid_aa.issuperset(trb)
            return {**row, '_dedup_key': trb if valid else '', '_valid': valid}

        elif mode == "tcr_pairing":
            tra = row.get('tra', '')
            trb = row.get('trb', '')
            tra_valid = bool(tra) and tra != 'NA' and _valid_aa.issuperset(tra)
            trb_valid = bool(trb) and trb != 'NA' and _valid_aa.issuperset(trb)
            valid = tra_valid and trb_valid
            return {**row, '_dedup_key': f"{tra}|{trb}" if valid else '', '_valid': valid}

        elif mode == "mhc_binding":
            pep = row.get('peptide', '')
            mho = row.get('mhc_one', '')
            mht = row.get('mhc_two', '')
            pep_valid = bool(pep) and pep != 'NA' and _valid_aa.issuperset(pep)
            mho_valid = bool(mho) and mho != 'NA' and _valid_aa.issuperset(mho)
            mht_valid = bool(mht) and mht != 'NA' and _valid_aa.issuperset(mht)
            valid = pep_valid and mho_valid
            return {**row, '_dedup_key': f"{pep}|{mho}|{mht if mht_valid else ''}" if valid else '', '_valid': valid}

        elif mode == "specificity":
            tra = row.get('tra', '')
            trb = row.get('trb', '')
            pep = row.get('peptide', '')
            mho = row.get('mhc_one', '')
            mht = row.get('mhc_two', '')

            tra_valid = bool(tra) and tra != 'NA' and _valid_aa.issuperset(tra)
            trb_valid = bool(trb) and trb != 'NA' and _valid_aa.issuperset(trb)
            pep_valid = bool(pep) and pep != 'NA' and _valid_aa.issuperset(pep)
            mho_valid = bool(mho) and mho != 'NA' and _valid_aa.issuperset(mho)
            mht_valid = bool(mht) and mht != 'NA' and _valid_aa.issuperset(mht)

            valid = pep_valid and mho_valid and (tra_valid or trb_valid)

            if valid:
                parts = []
                if mho_valid: parts.append(f"mhc_one:{mho}")
                if mht_valid: parts.append(f"mhc_two:{mht}")
                if pep_valid: parts.append(f"peptide:{pep}")
                if tra_valid: parts.append(f"tra:{tra}")
                if trb_valid: parts.append(f"trb:{trb}")
                parts.sort()
                dedup_key = '|'.join(parts)
            else:
                dedup_key = ''

            return {**row, '_dedup_key': dedup_key, '_valid': valid}

        else:  # default or balanced mode (both treat data the same way during deduplication)
            tra = row.get('tra', '')
            trb = row.get('trb', '')
            pep = row.get('peptide', '')
            mho = row.get('mhc_one', '')
            mht = row.get('mhc_two', '')

            tra_valid = bool(tra) and tra != 'NA' and _valid_aa.issuperset(tra)
            trb_valid = bool(trb) and trb != 'NA' and _valid_aa.issuperset(trb)
            pep_valid = bool(pep) and pep != 'NA' and _valid_aa.issuperset(pep)
            mho_valid = bool(mho) and mho != 'NA' and _valid_aa.issuperset(mho)
            mht_valid = bool(mht) and mht != 'NA' and _valid_aa.issuperset(mht)

            parts = []
            if mho_valid: parts.append(f"mhc_one:{mho}")
            if mht_valid: parts.append(f"mhc_two:{mht}")
            if pep_valid: parts.append(f"peptide:{pep}")
            if tra_valid: parts.append(f"tra:{tra}")
            if trb_valid: parts.append(f"trb:{trb}")

            valid = len(parts) > 0
            if valid:
                parts.sort()
                dedup_key = '|'.join(parts)
            else:
                dedup_key = ''

            return {**row, '_dedup_key': dedup_key, '_valid': valid}

    # Add dedup keys, filter, and collect ALL sequences for duplicate analysis
    step_start = time.time()
    print_progress("⏳", "Step 1/5: Computing deduplication keys and validating sequences...")
    ds_with_keys = ds.map(add_dedup_key)
    ds_filtered = ds_with_keys.filter(lambda row: row['_valid'])

    # Count total input rows BEFORE deduplication (force execution)
    print_progress("⏳", "Step 2/5: Counting total valid rows (this triggers data loading)...")
    count_start = time.time()
    total_before_dedup = ds_filtered.count()
    count_time = time.time() - count_start
    print_progress("✓", f"Found {total_before_dedup:,} valid rows (took {count_time:.1f}s)")

    # Collect duplicate statistics by counting occurrences of each key
    print_progress("⏳", "Step 3/5: Analyzing duplicate patterns...")
    dup_analysis_start = time.time()

    # Group by dedup_key and count occurrences - need to use Ray Data aggregation
    # We'll collect the counts from each batch and merge them
    def count_keys_in_batch(batch):
        """Count occurrences of each dedup key in this batch. Returns augmented batch with counts."""
        import pandas as pd
        df = pd.DataFrame(batch)
        if len(df) == 0:
            return df

        # Count each key in this batch and store in a new column
        key_counts = df['_dedup_key'].value_counts().to_dict()

        # Add count column to each row
        df['_key_count'] = df['_dedup_key'].map(key_counts)

        return df

    # Process all batches to count duplicates within each batch
    print_progress("   ", "→ Grouping sequences by deduplication key...", indent=2)
    ds_with_counts = ds_filtered.map_batches(count_keys_in_batch, batch_format="pandas")

    # Now aggregate counts across ALL batches using groupby
    print_progress("   ", "→ Computing global duplicate counts (this may take several minutes)...", indent=2)

    # Use Ray Data's groupby to get global counts
    # This is the proper way to collect statistics in distributed Ray Data
    grouped = ds_with_counts.groupby('_dedup_key').count()

    # Materialize to get the aggregated counts
    count_results = grouped.materialize()

    # Convert to dictionary for reporting
    from collections import Counter
    duplicate_counter = Counter()

    for row in count_results.iter_rows():
        key = row['_dedup_key']
        count = row['count()']  # Ray Data's count aggregation column name
        if key and key != '':
            duplicate_counter[key] = count

    dup_analysis_time = time.time() - dup_analysis_start
    print_progress("✓", f"Analyzed {len(duplicate_counter):,} unique sequences (took {dup_analysis_time:.1f}s)")

    # Now do the actual deduplication with sorting
    print_progress("⏳", "Step 4/5: Sorting dataset by deduplication key (external merge sort)...")
    sort_start = time.time()
    ds_sorted = ds_filtered.sort(key='_dedup_key')
    sort_time = time.time() - sort_start
    print_progress("✓", f"Sorting complete (took {sort_time:.1f}s)")

    # Final dedup to remove consecutive duplicates
    print_progress("⏳", "Step 5/5: Removing duplicate rows...")
    dedup_start = time.time()

    def drop_consecutive_duplicates(batch):
        """Keep only first occurrence of consecutive duplicate keys."""
        import pandas as pd
        df = pd.DataFrame(batch)
        if len(df) == 0:
            return df

        # Mark first occurrence of each unique _dedup_key
        df['_keep'] = df['_dedup_key'].ne(df['_dedup_key'].shift())

        # Filter and drop temp columns
        df_filtered = df[df['_keep']].drop(columns=['_dedup_key', '_valid', '_keep'])

        return df_filtered

    ds_deduped = ds_sorted.map_batches(drop_consecutive_duplicates, batch_format="pandas")

    # Materialize to get final count (single execution)
    print_progress("   ", "→ Materializing deduplicated dataset to disk...", indent=2)
    ds_deduped = ds_deduped.materialize()

    # Get final count (fast after materialization)
    final_count = ds_deduped.count()
    dedup_time = time.time() - dedup_start
    print_progress("✓", f"Removed duplicates (took {dedup_time:.1f}s)")

    # Analyze and report duplicate statistics
    print("\n" + "="*80)
    print("📊 DUPLICATE ANALYSIS")
    print("="*80)

    if duplicate_counter:
        # Sort by duplicate count (descending)
        sorted_duplicates = duplicate_counter.most_common()

        # Find sequences with actual duplicates (count > 1)
        duplicated_sequences = [(k, v) for k, v in sorted_duplicates if v > 1]

        if duplicated_sequences:
            print(f"Total unique sequences: {len(duplicate_counter):,}")
            print(f"Sequences with duplicates: {len(duplicated_sequences):,}")
            print(f"Total duplicate instances removed: {sum(v - 1 for k, v in duplicated_sequences):,}")
            print(f"\nTop 20 most duplicated sequences:")
            print("-" * 80)

            for i, (key, count) in enumerate(duplicated_sequences[:20], 1):
                # Truncate sequence if too long for display
                display_key = key if len(key) <= 50 else f"{key[:47]}..."
                print(f"{i:2d}. Count: {count:>10,} | Sequence: {display_key}")

            # Distribution analysis
            dup_counts = [v for k, v in duplicated_sequences]
            print(f"\nDuplicate count distribution:")
            print(f"  Min occurrences: {min(dup_counts):,}")
            print(f"  Max occurrences: {max(dup_counts):,}")
            print(f"  Avg occurrences: {sum(dup_counts)/len(dup_counts):.1f}")

            # Save detailed report to file
            import json
            report_path = "/tmp/duplicate_report.json"
            report_data = {
                'total_unique_sequences': len(duplicate_counter),
                'sequences_with_duplicates': len(duplicated_sequences),
                'total_rows_before_dedup': total_before_dedup,
                'total_rows_after_dedup': final_count,
                'total_duplicate_instances_removed': sum(v - 1 for k, v in duplicated_sequences),
                'top_100_duplicates': [
                    {
                        'sequence': k,
                        'count': v,
                        'duplicates_removed': v - 1
                    }
                    for k, v in duplicated_sequences[:100]
                ]
            }
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            print(f"\n💾 Detailed duplicate report saved to: {report_path}")
        else:
            print("No duplicates found!")
    else:
        print("No sequences to analyze!")

    print("="*80 + "\n")

    # Correct stats
    stats['rows_after_deduplication'] = final_count
    stats['rows_before_deduplication'] = total_before_dedup
    stats['duplicate_sequences'] = len([k for k, v in duplicate_counter.items() if v > 1])

    print(f"   ✓ Deduplication complete: {total_before_dedup:,} → {final_count:,} rows")
    return ds_deduped, stats

# Old HF Datasets dedup function - keep for reference but not used
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

        ds = ds.map(process_tra_batch, batched=True, batch_size=batch_size, desc="Validating TRA")
        ds = ds.filter(lambda ex: ex['_valid'], num_proc=num_proc, desc="Filtering valid TRA")
        ds = ds.remove_columns(['_valid'])

        # Sort-based deduplication (optimized for large datasets)
        # Use disk-based operations to avoid OOM
        print(f"   ⏳ Consolidating chunks before sorting...")
        # flatten_indices with keep_in_memory=False forces disk-based consolidation
        ds = ds.flatten_indices(keep_in_memory=False, cache_file_name=None, num_proc=num_proc)

        print(f"   ⏳ Sorting {len(ds):,} TRA sequences by dedup key...")
        # Arrow's sort is already disk-based (external merge sort)
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
        ds = ds.flatten_indices(keep_in_memory=False, cache_file_name=None, num_proc=num_proc)

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
        ds = ds.flatten_indices(keep_in_memory=False, cache_file_name=None, num_proc=num_proc)

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
        ds = ds.flatten_indices(keep_in_memory=False, cache_file_name=None, num_proc=num_proc)

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
        ds = ds.flatten_indices(keep_in_memory=False, cache_file_name=None, num_proc=num_proc)

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
        ds = ds.flatten_indices(keep_in_memory=False, cache_file_name=None, num_proc=num_proc)

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
                   choices=["tra", "trb", "tcr_pairing", "mhc_binding", "specificity", "default", "balanced"],
                   help="Analysis mode: tra (TRA only), trb (TRB only), tcr_pairing (TRA+TRB), mhc_binding (MHC+peptide), specificity (complete TCR complexes), default (all combinations except peptide-only), balanced (all combinations with equal sampling per permutation)")
    p.add_argument("--input_raw_dir", default=None, help="Optional: Path to an existing 'raw' dataset to start masking from.")
    p.add_argument("--output-raw", required=True, help="Path to save the tokenized dataset with text columns.")
    p.add_argument("--max-len",type=int,default=1024)
    p.add_argument("--bpe-vocab",type=int,default=200)
    p.add_argument("--truncate-long",action="store_true")
    p.add_argument("--mlm-prob",type=float,default=0.15,
                   help="Probability of masking tokens in MLM. Default is 0.15.")
    p.add_argument("--num-proc", type=int, default=None, help="Number of processes for data processing (default: auto-detect)")
    p.add_argument("--batch-size", type=int, default=100000, help="Batch size for dataset processing operations (default: 100000, recommended: 100000-200000 for x2gd.16xlarge with 1TB RAM)")
    p.add_argument("--tmp-dir", type=str, default="tmp_processed", help="Directory for temporary/intermediate files during processing.")
    p.add_argument("--sample", type=int, default=None, help="Sample N files for testing (e.g., 10 for first 10 files). Applied during file selection.")
    p.add_argument("--in-memory", action="store_true", help="Keep datasets in memory to avoid disk caching (use for small datasets only)")
    p.add_argument("--fast-mode", action="store_true", help="Enable fast processing mode: skip detailed duplicate analysis and token statistics for 10-100x speedup")
    p.add_argument("--samples-per-combo", type=int, default=None, help="Number of samples per molecule combination (only used with --mode balanced). If None, uses min count across all combinations.")
    return p.parse_args()

# Multi-node task distribution helper
def main():
    """
    Ray Data version for large-scale processing.

    Multi-node best practices implemented:
    1. Thread-safe: All map functions return NEW dicts (no in-place mutations)
    2. Fault tolerance: Materialize after major operations to break lineage
    3. Memory efficient: Disk spilling enabled, streaming operations
    4. Scalable: Works on single node or distributed Ray cluster
    """
    args = cli()

    # Initialize Ray with disk spilling for datasets larger than RAM
    spill_dir = os.path.join(args.tmp_dir, "ray_spill")
    os.makedirs(spill_dir, exist_ok=True)

    # Configure Ray logging to reduce noise - MUST be set before ray.init()
    os.environ["RAY_DEDUP_LOGS"] = "1"  # Enable log deduplication
    os.environ["RAY_object_spilling_config"] = json.dumps({
        "type": "filesystem",
        "params": {"directory_path": spill_dir}
    })

    # Disable Ray Data progress reporting (the verbose logs you're seeing)
    os.environ["RAY_DATA_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["RAY_DATA_TRACE_SCHEDULING"] = "0"  # Disable scheduling traces
    os.environ["RAY_LOG_TO_STDERR"] = "0"  # Disable stderr logging

    # Suppress Ray's verbose logging completely
    import logging
    import warnings

    # Suppress all warnings
    warnings.filterwarnings("ignore")

    # Set Ray logging to CRITICAL (only fatal errors)
    logging.getLogger("ray").setLevel(logging.CRITICAL)
    logging.getLogger("ray.data").setLevel(logging.CRITICAL)
    logging.getLogger("ray.data._internal").setLevel(logging.CRITICAL)
    logging.getLogger("ray.data._internal.execution").setLevel(logging.CRITICAL)
    logging.getLogger("ray.tune").setLevel(logging.CRITICAL)
    logging.getLogger("ray.rllib").setLevel(logging.CRITICAL)
    logging.getLogger("ray._private").setLevel(logging.CRITICAL)

    ray.init(
        _temp_dir=args.tmp_dir,
        object_store_memory=int(psutil.virtual_memory().available * 0.7),  # Use 70% of RAM for object store
        logging_level=logging.CRITICAL,  # Only show critical errors
        log_to_driver=False,  # Don't log worker outputs to driver
        configure_logging=True,
        include_dashboard=False  # Disable dashboard to reduce overhead
    )

    # Setup cache directories
    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(os.path.join(args.tmp_dir, "ray_spill"), exist_ok=True)

    # Disable Ray Data progress bars (programmatic way)
    # Note: ray is already imported at the top of the file
    ray.data.DataContext.get_current().execution_options.verbose_progress = False

    num_proc = args.num_proc or get_num_workers()
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)

    # Batch size for Ray Data operations
    if args.batch_size == 100000:
        if available_gb < 32:
            batch_size = 10000
        elif available_gb < 64:
            batch_size = 50000
        elif available_gb < 128:
            batch_size = 100000
        else:
            batch_size = 200000
    else:
        batch_size = args.batch_size

    print("="*80)
    print("🧬 T-CELL RECEPTOR ANALYSIS MODE (Ray Data)")
    print("="*80)
    print(f"📊 Mode: {args.mode}")
    print(f"💾 Cores: {num_proc}")
    print(f"💾 Available RAM: {available_gb:.1f} GB")
    print(f"📦 Batch Size: {batch_size:,}")
    print("="*80)

    # Prepare file paths and resolve globs
    file_paths = args.path if isinstance(args.path, list) else [args.path]

    # Local filesystem glob
    import glob
    print(f"🔍 Resolving local paths (including partitioned parquet directories)...")
    all_files = []
    for pattern in file_paths:
        matched = glob.glob(pattern, recursive=True)
        for match in matched:
            if os.path.isdir(match):
                # It's a partitioned parquet directory
                part_files = glob.glob(os.path.join(match, "*.parquet"))
                if part_files:
                    all_files.extend(part_files)
                else:
                    # Add directory itself (Ray Data can handle it)
                    all_files.append(match)
            else:
                # It's a single file
                all_files.append(match)

    print(f"Found {len(all_files)} parquet files/partitions")

    # Apply sampling if requested
    if args.sample is not None and args.sample < len(all_files):
        print(f"🎲 Sampling {args.sample} files...")
        all_files = all_files[:args.sample]
        print(f"Using {len(all_files)} files")

    if len(all_files) == 0:
        raise ValueError(f"No files found matching pattern: {file_paths}")

    # Load with Ray Data - streaming, never loads full dataset into memory
    print("📂 Loading parquet files with Ray Data (streaming)...")
    ds = ray.data.read_parquet(all_files)

    print(f"✓ Dataset loaded in streaming mode")

    # Deduplication with Ray Data (streaming, handles > RAM datasets)
    print("\n" + "="*80)
    print("🔄 GLOBAL DEDUPLICATION (Ray Data)")
    print("="*80)
    import time
    start_time = time.time()

    # Use fast mode if requested
    if args.fast_mode:
        ds, dedup_stats = deduplicate_ray_data_fast(ds, args.mode)
    else:
        ds, dedup_stats = deduplicate_ray_data(ds, args.mode)

    # Note: Dataset is already materialized by deduplication functions

    dedup_time = time.time() - start_time

    # Detailed deduplication summary
    print("\n" + "="*80)
    print("📊 DEDUPLICATION SUMMARY")
    print("="*80)
    print(f"⏱️  Time: {dedup_time/60:.1f} minutes ({dedup_time:.0f} seconds)")
    print(f"📤 Final unique rows: {dedup_stats['rows_after_deduplication']:,}")
    print(f"⚡ Processing speed: {dedup_stats['rows_after_deduplication']/dedup_time:,.0f} rows/sec")
    print("="*80 + "\n")

    # Explode sequences for modes that need it
    if args.mode in ["mhc_binding", "specificity", "default", "tcr_pairing"]:
        print("💥 Exploding and unnesting sequences...")

        def explode_and_unnest(row):
            """Explode and flatten in one step for Ray Data."""
            result = explode_example(row, args.mode)
            # Return list of rows (Ray Data will flatten automatically with flat_map)
            if result['sequences']:
                return [
                    {'sequence_tuple': seq, 'feat_names': feat}
                    for seq, feat in zip(result['sequences'], result['feat_names'])
                ]
            return []

        ds = ds.flat_map(explode_and_unnest)
        print("   ✓ Explosion complete")

        # Materialize after explosion to break lineage
        print("   ⏳ Materializing exploded dataset...")
        ds = ds.materialize()

        # Deduplicate exact sequences
        if args.mode in ["mhc_binding", "specificity", "default"]:
            print("🔄 Deduplicating exact sequences...")

            def add_seq_dedup_key(row):
                """Add sequence dedup key. Returns NEW dict (thread-safe)."""
                return {**row, '_seq_key': '|'.join(str(x) for x in sorted(row['sequence_tuple']))}

            ds = ds.map(add_seq_dedup_key)

            # Sort and remove consecutive duplicates
            print("   ⏳ Sorting by sequence key...")
            ds = ds.sort(key='_seq_key')

            print("   ⏳ Removing duplicate sequences...")
            def drop_consecutive_seq_duplicates(batch):
                """Keep only first occurrence of consecutive duplicate sequences."""
                import pandas as pd
                df = pd.DataFrame(batch)

                if len(df) == 0:
                    return df

                # Mark first occurrence
                df['_keep'] = df['_seq_key'].ne(df['_seq_key'].shift())

                # Filter and drop temp column
                df_filtered = df[df['_keep']].drop(columns=['_seq_key', '_keep'])

                return df_filtered

            ds = ds.map_batches(drop_consecutive_seq_duplicates, batch_format="pandas")

            print("   ✓ Sequence dedup complete\n")

    # Split 80/10/10 with Ray Data
    print("Splitting dataset (80/10/10)...")
    train_ds, test_val_ds = ds.train_test_split(test_size=0.2, seed=42)
    val_ds, test_ds = test_val_ds.train_test_split(test_size=0.5, seed=42)
    print("   ✓ Split complete")

    # Format sequences for model
    model_type = args.model_name
    print(f"\n🎯 Formatting for {model_type}...")

    if model_type in ["bert", "protbert", "esm", "llama"]:
        def format_row(row):
            """Format a single row for transformer models. Returns NEW dict (thread-safe)."""
            if 'sequence_tuple' in row:
                # Already exploded - return new dict without temp columns
                return {
                    **{k: v for k, v in row.items() if k not in ('sequence_tuple', 'feat_names')},
                    'combo_id': format_sequence_for_model(row['sequence_tuple'], model_type),
                    'combo_feats': row['feat_names']
                }
            else:
                # Need to explode for tra/trb modes
                result = explode_example(row, args.mode)
                if result['sequences']:
                    # Just take first sequence for tra/trb (they're single-molecule)
                    return {
                        **row,
                        'combo_id': format_sequence_for_model(result['sequences'][0], model_type),
                        'combo_feats': result['feat_names'][0]
                    }
                return row

        print("📝 Formatting train split...")
        train_ds = train_ds.map(format_row)
        train_count = train_ds.count()

        print("📝 Formatting validation split...")
        val_ds = val_ds.map(format_row)
        val_count = val_ds.count()

        print("📝 Formatting test split...")
        test_ds = test_ds.map(format_row)
        test_count = test_ds.count()

        # Apply balanced sampling if mode is "balanced"
        if args.mode == "balanced":
            print("\n🎯 APPLYING BALANCED SAMPLING (Mode: balanced)")
            print("="*80)
            print("This mode ensures equal representation across all molecule permutations.")
            print("Balancing across molecule permutations...")
            
            # Apply to each split
            print("\n📝 Balancing train split...")
            train_ds = balanced_sample_by_molecules(train_ds, args.samples_per_combo)
            train_count = train_ds.count()
            
            print("\n📝 Balancing validation split...")
            val_ds = balanced_sample_by_molecules(val_ds, args.samples_per_combo)
            val_count = val_ds.count()
            
            print("\n📝 Balancing test split...")
            test_ds = balanced_sample_by_molecules(test_ds, args.samples_per_combo)
            test_count = test_ds.count()
            
            print("="*80 + "\n")

        # Calculate token statistics (sample first 1000 rows)
        if args.fast_mode:
            print("\n⚡ Skipping token statistics (fast mode)")
            avg_tokens = min_tokens = max_tokens = total_tokens_est = 0
        else:
            print("\n📊 Computing token statistics...")
            def count_tokens_in_row(row):
                """Count tokens in combo_id (space-separated for ProtBERT)."""
                seq = row.get('combo_id', '')
                if model_type == "protbert":
                    # ProtBERT uses space-separated amino acids
                    tokens = seq.split()
                else:
                    # For other models, rough estimate
                    tokens = seq.split()
                return {'token_count': len(tokens), **row}

            # Sample and compute token stats
            train_sample = train_ds.limit(1000).map(count_tokens_in_row)
            token_counts = [row['token_count'] for row in train_sample.take_all()]

            if token_counts:
                avg_tokens = sum(token_counts) / len(token_counts)
                min_tokens = min(token_counts)
                max_tokens = max(token_counts)
                total_tokens_est = int(avg_tokens * train_count)
            else:
                avg_tokens = min_tokens = max_tokens = total_tokens_est = 0

        # Save as HF Datasets format (materialize Ray Data to disk)
        print("\n💾 Saving to disk...")
        os.makedirs(args.output_raw, exist_ok=True)
        train_ds.write_parquet(os.path.join(args.output_raw, "train"))
        val_ds.write_parquet(os.path.join(args.output_raw, "validation"))
        test_ds.write_parquet(os.path.join(args.output_raw, "test"))

        # Final summary
        print("\n" + "="*80)
        print(f"📊 FINAL DATASET SUMMARY ({model_type.upper()})")
        print("="*80)
        print(f"📁 Output directory: {args.output_raw}")
        print(f"🧬 Analysis mode: {args.mode}")
        print(f"\n📦 Split Statistics:")
        print(f"   Train:      {train_count:>12,} examples ({train_count/(train_count+val_count+test_count)*100:.1f}%)")
        print(f"   Validation: {val_count:>12,} examples ({val_count/(train_count+val_count+test_count)*100:.1f}%)")
        print(f"   Test:       {test_count:>12,} examples ({test_count/(train_count+val_count+test_count)*100:.1f}%)")
        print(f"   Total:      {train_count+val_count+test_count:>12,} examples")
        if not args.fast_mode:
            print(f"\n🔤 Token Statistics (train split):")
            print(f"   Average tokens/sequence: {avg_tokens:.1f}")
            print(f"   Min tokens:              {min_tokens:,}")
            print(f"   Max tokens:              {max_tokens:,}")
            print(f"   Estimated total tokens:  {total_tokens_est:,}")
        else:
            print(f"\n⚡ Fast mode: Token statistics skipped for speed")
        print("="*80)
        print(f"✅ Dataset saved to {args.output_raw}")
        print("="*80 + "\n")
        
    elif model_type in ["lstm", "transformer"]:
        # Custom BPE path - need to materialize train split for tokenizer training
        print(f"Processing with custom BPE for {model_type}...")

        def tag_row(row):
            row['tagged'] = tag_bpe(row)
            return row

        print("📝 Collecting train sequences for tokenizer training...")
        # Sample train data for tokenizer (to avoid materializing full dataset)
        train_sample = train_ds.limit(100000)  # Use first 100k for tokenizer training
        all_seqs = []
        for row in train_sample.iter_rows():
            all_seqs.extend(tag_bpe(row))

        print(f"Training BPE tokenizer on {len(all_seqs):,} sequences...")
        tokenizer = train_bpe_tokenizer(all_seqs, args.bpe_vocab)
        tok_path = f"{args.output_raw}_tokenizer.json"
        tokenizer.save(tok_path)
        print(f"✓ Tokenizer saved to {tok_path}")

        # Encode all splits (streaming)
        print("📝 Encoding splits with BPE...")
        # Note: This is simplified - full BPE encoding may need batch processing
        # For now, save tagged sequences and let downstream training handle encoding
        train_ds = train_ds.map(tag_row)
        val_ds = val_ds.map(tag_row)
        test_ds = test_ds.map(tag_row)

        print("💾 Saving to disk...")
        os.makedirs(args.output_raw, exist_ok=True)
        train_ds.write_parquet(os.path.join(args.output_raw, "train"))
        val_ds.write_parquet(os.path.join(args.output_raw, "validation"))
        test_ds.write_parquet(os.path.join(args.output_raw, "test"))
        print(f"✅ Saved to {args.output_raw}")

    # Shutdown Ray
    ray.shutdown()
    print("✨ Complete!")


if __name__ == "__main__":
    main()
