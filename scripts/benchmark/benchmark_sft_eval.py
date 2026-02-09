#!/usr/bin/env python
"""
SFT Benchmark Evaluation Script

Evaluates pre-trained SFT models (ESM2 + TransformerDecoder) on TCR generation tasks.
Computes generation metrics: sequence identity, BLOSUM similarity, perplexity.

This script is separate from tcr_seq2seq_trainer.py to maintain clean separation
between training/validation and benchmark evaluation.

Usage:
    # Evaluate a trained model on all test splits
    python scripts/benchmark/benchmark_sft_eval.py \
        --model_path output/sft_training/tra_trb_peptide_mhc_one/best_model.pt \
        --task tra_trb_peptide_mhc_one \
        --data_dir data/icml/tasks/tcr_specificity/tcr90pep80 \
        --output_dir output/sft_benchmark

    # Evaluate with sampling (for diversity metrics)
    python scripts/benchmark/benchmark_sft_eval.py \
        --model_path output/sft_training/tra_trb_peptide_mhc_one/best_model.pt \
        --task tra_trb_peptide_mhc_one \
        --data_dir data/icml/tasks/tcr_specificity/tcr90pep80 \
        --output_dir output/sft_benchmark \
        --do_sample \
        --temperature 0.8 \
        --num_samples 5

    # Run all tasks
    python scripts/benchmark/benchmark_sft_eval.py \
        --model_path output/sft_training \
        --task all \
        --data_dir data/icml/tasks/tcr_specificity/tcr90pep80 \
        --output_dir output/sft_benchmark
"""

import argparse
import json
import math
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# TCR stitching for full-length sequences
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from quest.parsers.tcr_stitcher import add_full_tcr_sequences
    STITCHER_AVAILABLE = True
except ImportError:
    STITCHER_AVAILABLE = False
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import from trainer module
from scripts.training.tcr_seq2seq_trainer import (
    GenerationTask,
    PositionalEncoding,
    TCRSeq2SeqModel,
    Seq2SeqCollator,
    greedy_decode,
    SelfAttnDropoutDecoderLayer,
    SelfAttnDropoutDecoder,
    apply_lora_to_encoder,
)

from scripts.benchmark.benchmark_metrics import compute_all_unified_metrics

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Task Configurations
# =============================================================================

SFT_TASK_CONFIGS = {
    # =========================================================================
    # PEPTIDE PREDICTION TASKS (predict peptide from TCR + MHC context)
    # =========================================================================
    "tra_peptide_mhc_one": {
        "mhc_class": "class_one",
        "sft_task": GenerationTask.PEPTIDE,
        "target_col": "peptide",
        "context_cols": ["tra", "mhc_one"],
        "description": "Generate peptide from TRA + MHC-I",
    },
    "trb_peptide_mhc_one": {
        "mhc_class": "class_one",
        "sft_task": GenerationTask.PEPTIDE,
        "target_col": "peptide",
        "context_cols": ["trb", "mhc_one"],
        "description": "Generate peptide from TRB + MHC-I",
    },
    "tra_trb_peptide_mhc_one": {
        "mhc_class": "class_one",
        "sft_task": GenerationTask.PEPTIDE,
        "target_col": "peptide",
        "context_cols": ["tra", "trb", "mhc_one"],
        "description": "Generate peptide from TRA + TRB + MHC-I",
    },
    "tra_peptide_mhc_two": {
        "mhc_class": "class_two",
        "sft_task": GenerationTask.PEPTIDE,
        "target_col": "peptide",
        "context_cols": ["tra", "mhc_one", "mhc_two"],
        "description": "Generate peptide from TRA + MHC-II",
    },
    "trb_peptide_mhc_two": {
        "mhc_class": "class_two",
        "sft_task": GenerationTask.PEPTIDE,
        "target_col": "peptide",
        "context_cols": ["trb", "mhc_one", "mhc_two"],
        "description": "Generate peptide from TRB + MHC-II",
    },
    "tra_trb_peptide_mhc_two": {
        "mhc_class": "class_two",
        "sft_task": GenerationTask.PEPTIDE,
        "target_col": "peptide",
        "context_cols": ["tra", "trb", "mhc_one", "mhc_two"],
        "description": "Generate peptide from TRA + TRB + MHC-II",
    },
}

TEST_SPLITS = [
    "test_seen_epitope",
    "test_unseen_tcr_seen_epitope",
    "test_unseen_epitope",
    "test_unseen_allele",
]

# BLOSUM62 scoring matrix for amino acid similarity
BLOSUM62 = {
    'A': {'A': 4, 'R': -1, 'N': -2, 'D': -2, 'C': 0, 'Q': -1, 'E': -1, 'G': 0, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 0, 'W': -3, 'Y': -2, 'V': 0},
    'R': {'A': -1, 'R': 5, 'N': 0, 'D': -2, 'C': -3, 'Q': 1, 'E': 0, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 2, 'M': -1, 'F': -3, 'P': -2, 'S': -1, 'T': -1, 'W': -3, 'Y': -2, 'V': -3},
    'N': {'A': -2, 'R': 0, 'N': 6, 'D': 1, 'C': -3, 'Q': 0, 'E': 0, 'G': 0, 'H': 1, 'I': -3, 'L': -3, 'K': 0, 'M': -2, 'F': -3, 'P': -2, 'S': 1, 'T': 0, 'W': -4, 'Y': -2, 'V': -3},
    'D': {'A': -2, 'R': -2, 'N': 1, 'D': 6, 'C': -3, 'Q': 0, 'E': 2, 'G': -1, 'H': -1, 'I': -3, 'L': -4, 'K': -1, 'M': -3, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -4, 'Y': -3, 'V': -3},
    'C': {'A': 0, 'R': -3, 'N': -3, 'D': -3, 'C': 9, 'Q': -3, 'E': -4, 'G': -3, 'H': -3, 'I': -1, 'L': -1, 'K': -3, 'M': -1, 'F': -2, 'P': -3, 'S': -1, 'T': -1, 'W': -2, 'Y': -2, 'V': -1},
    'Q': {'A': -1, 'R': 1, 'N': 0, 'D': 0, 'C': -3, 'Q': 5, 'E': 2, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 1, 'M': 0, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -2, 'Y': -1, 'V': -2},
    'E': {'A': -1, 'R': 0, 'N': 0, 'D': 2, 'C': -4, 'Q': 2, 'E': 5, 'G': -2, 'H': 0, 'I': -3, 'L': -3, 'K': 1, 'M': -2, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
    'G': {'A': 0, 'R': -2, 'N': 0, 'D': -1, 'C': -3, 'Q': -2, 'E': -2, 'G': 6, 'H': -2, 'I': -4, 'L': -4, 'K': -2, 'M': -3, 'F': -3, 'P': -2, 'S': 0, 'T': -2, 'W': -2, 'Y': -3, 'V': -3},
    'H': {'A': -2, 'R': 0, 'N': 1, 'D': -1, 'C': -3, 'Q': 0, 'E': 0, 'G': -2, 'H': 8, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -1, 'P': -2, 'S': -1, 'T': -2, 'W': -2, 'Y': 2, 'V': -3},
    'I': {'A': -1, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -3, 'E': -3, 'G': -4, 'H': -3, 'I': 4, 'L': 2, 'K': -3, 'M': 1, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -3, 'Y': -1, 'V': 3},
    'L': {'A': -1, 'R': -2, 'N': -3, 'D': -4, 'C': -1, 'Q': -2, 'E': -3, 'G': -4, 'H': -3, 'I': 2, 'L': 4, 'K': -2, 'M': 2, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -2, 'Y': -1, 'V': 1},
    'K': {'A': -1, 'R': 2, 'N': 0, 'D': -1, 'C': -3, 'Q': 1, 'E': 1, 'G': -2, 'H': -1, 'I': -3, 'L': -2, 'K': 5, 'M': -1, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
    'M': {'A': -1, 'R': -1, 'N': -2, 'D': -3, 'C': -1, 'Q': 0, 'E': -2, 'G': -3, 'H': -2, 'I': 1, 'L': 2, 'K': -1, 'M': 5, 'F': 0, 'P': -2, 'S': -1, 'T': -1, 'W': -1, 'Y': -1, 'V': 1},
    'F': {'A': -2, 'R': -3, 'N': -3, 'D': -3, 'C': -2, 'Q': -3, 'E': -3, 'G': -3, 'H': -1, 'I': 0, 'L': 0, 'K': -3, 'M': 0, 'F': 6, 'P': -4, 'S': -2, 'T': -2, 'W': 1, 'Y': 3, 'V': -1},
    'P': {'A': -1, 'R': -2, 'N': -2, 'D': -1, 'C': -3, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -4, 'P': 7, 'S': -1, 'T': -1, 'W': -4, 'Y': -3, 'V': -2},
    'S': {'A': 1, 'R': -1, 'N': 1, 'D': 0, 'C': -1, 'Q': 0, 'E': 0, 'G': 0, 'H': -1, 'I': -2, 'L': -2, 'K': 0, 'M': -1, 'F': -2, 'P': -1, 'S': 4, 'T': 1, 'W': -3, 'Y': -2, 'V': -2},
    'T': {'A': 0, 'R': -1, 'N': 0, 'D': -1, 'C': -1, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 5, 'W': -2, 'Y': -2, 'V': 0},
    'W': {'A': -3, 'R': -3, 'N': -4, 'D': -4, 'C': -2, 'Q': -2, 'E': -3, 'G': -2, 'H': -2, 'I': -3, 'L': -2, 'K': -3, 'M': -1, 'F': 1, 'P': -4, 'S': -3, 'T': -2, 'W': 11, 'Y': 2, 'V': -3},
    'Y': {'A': -2, 'R': -2, 'N': -2, 'D': -3, 'C': -2, 'Q': -1, 'E': -2, 'G': -3, 'H': 2, 'I': -1, 'L': -1, 'K': -2, 'M': -1, 'F': 3, 'P': -3, 'S': -2, 'T': -2, 'W': 2, 'Y': 7, 'V': -1},
    'V': {'A': 0, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -2, 'E': -2, 'G': -3, 'H': -3, 'I': 3, 'L': 1, 'K': -2, 'M': 1, 'F': -1, 'P': -2, 'S': -2, 'T': 0, 'W': -3, 'Y': -1, 'V': 4},
}


# =============================================================================
# Model Loading
# =============================================================================


def load_sft_model(
    checkpoint_path: str,
    device: str = "cuda",
    use_flash_attention: bool = True,
) -> Tuple[TCRSeq2SeqModel, AutoTokenizer, Dict]:
    """
    Load a trained SFT model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file (.pt) or directory containing best_model.pt
        device: Device to load model on
        use_flash_attention: Whether to use flash attention (if available)

    Returns:
        Tuple of (model, tokenizer, config)
    """
    # Handle directory vs file path
    checkpoint_file = Path(checkpoint_path)
    if checkpoint_file.is_dir():
        checkpoint_file = checkpoint_file / "best_model.pt"

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    print(f"Loading checkpoint from: {checkpoint_file}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    print(f"  Checkpoint from epoch {checkpoint.get('epoch', 'N/A')}, "
          f"step {checkpoint.get('global_step', 'N/A')}")
    if "metrics" in checkpoint:
        metrics = checkpoint["metrics"]
        eval_loss = metrics.get('eval_loss', None)
        if eval_loss is not None:
            print(f"  Metrics: loss={eval_loss:.4f}")

    # Determine attention implementation
    attn_impl = "auto"
    if not use_flash_attention:
        attn_impl = "eager"

    # Recreate model with same architecture
    model = TCRSeq2SeqModel(
        encoder_model_name=config.get("model_name", "facebook/esm2_t33_650M_UR50D"),
        decoder_layers=config.get("decoder_layers", 6),
        decoder_heads=config.get("decoder_heads", 20),
        decoder_dim=config.get("decoder_dim", 1280),
        decoder_ffn_dim=config.get("decoder_ffn_dim", 5120),
        dropout=config.get("dropout", 0.1),
        attn_implementation=attn_impl,
        torch_dtype=torch.float32,  # Use float32 for inference stability
        decoder_warm_start=False,
        self_attn_drop_initial=0.0,  # No dropout during inference
        self_attn_drop_final=0.0,
    )

    # Apply LoRA if the model was trained with it
    if config.get("use_lora", False):
        print(f"  Applying LoRA (r={config.get('lora_r', 16)}, alpha={config.get('lora_alpha', 32)})")
        model = apply_lora_to_encoder(model, config)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.get("model_name", "facebook/esm2_t33_650M_UR50D")
    )

    print(f"  Model loaded successfully on {device}")
    print(f"  Encoder: {config.get('model_name', 'facebook/esm2_t33_650M_UR50D')}")
    print(f"  Decoder: {config.get('decoder_layers', 6)} layers, "
          f"{config.get('decoder_heads', 20)} heads")
    if config.get("use_lora", False):
        print(f"  LoRA: r={config.get('lora_r', 16)}, alpha={config.get('lora_alpha', 32)}")

    return model, tokenizer, config


# =============================================================================
# Data Preparation
# =============================================================================


class SFTEvalDataset(Dataset):
    """
    Dataset for SFT model evaluation.

    Loads test data from parquet files and prepares encoder context / decoder target pairs.
    """

    def __init__(
        self,
        parquet_path: str,
        task_config: Dict,
    ):
        """
        Args:
            parquet_path: Path to parquet file
            task_config: Task configuration dict with context_cols, target_col, etc.
        """
        self.task_config = task_config
        self.df = self._load_and_prepare_data(parquet_path)

    def _load_and_prepare_data(self, parquet_path: str) -> pd.DataFrame:
        """Load parquet and prepare columns."""
        df = pd.read_parquet(parquet_path)

        # Rename columns if needed (mhc_one_id -> mhc_one)
        rename_map = {
            "mhc_one_id": "mhc_one",
            "mhc_two_id": "mhc_two",
        }
        for old_col, new_col in rename_map.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})

        # Stitch full-length TCR sequences from CDR3 + gene annotations
        if STITCHER_AVAILABLE:
            has_gene_cols = any(col in df.columns for col in ["trav_gene", "trbv_gene"])
            if has_gene_cols:
                df = add_full_tcr_sequences(df, species="HUMAN")
                tra_stitched = (df["tra_full"].notna() & (df["tra_full"] != "")).sum() if "tra_full" in df.columns else 0
                trb_stitched = (df["trb_full"].notna() & (df["trb_full"] != "")).sum() if "trb_full" in df.columns else 0
                print(f"  TCR stitching: {tra_stitched}/{len(df)} TRA, {trb_stitched}/{len(df)} TRB full-length")

        # Filter rows with required columns
        required_cols = self.task_config["context_cols"] + [self.task_config["target_col"]]
        for col in required_cols:
            if col in df.columns:
                df = df[df[col].notna() & (df[col] != "")]

        return df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get encoder context and decoder target.

        Returns:
            dict with:
                - encoder_context: List[str] of context sequences
                - decoder_target: str target sequence
                - mhc_class: 'I' or 'II'
        """
        row = self.df.iloc[idx]

        # Build encoder context in order, preferring full-length TCR sequences
        encoder_context = []
        for col in self.task_config["context_cols"]:
            if col in self.df.columns and pd.notna(row.get(col)):
                # Prefer full-length stitched TCR over CDR3
                full_col = f"{col}_full"
                if col in ("tra", "trb") and full_col in self.df.columns:
                    full_val = row.get(full_col)
                    if pd.notna(full_val) and full_val:
                        encoder_context.append(str(full_val))
                        continue
                encoder_context.append(str(row[col]))

        # Get target
        decoder_target = str(row[self.task_config["target_col"]])

        # Determine MHC class
        mhc_class = "II" if "mhc_two" in self.task_config["context_cols"] else "I"

        return {
            "encoder_context": encoder_context,
            "decoder_target": decoder_target,
            "mhc_class": mhc_class,
        }


def prepare_eval_dataloader(
    parquet_path: str,
    task_config: Dict,
    tokenizer,
    batch_size: int = 16,
    max_encoder_length: int = 1024,
    max_decoder_length: int = 350,
) -> Tuple[DataLoader, pd.DataFrame]:
    """
    Prepare DataLoader for evaluation.

    Args:
        parquet_path: Path to parquet file
        task_config: Task configuration
        tokenizer: ESM2 tokenizer
        batch_size: Batch size
        max_encoder_length: Max encoder sequence length
        max_decoder_length: Max decoder sequence length

    Returns:
        Tuple of (DataLoader, DataFrame with original data)
    """
    dataset = SFTEvalDataset(parquet_path, task_config)
    collator = Seq2SeqCollator(
        tokenizer=tokenizer,
        max_encoder_length=max_encoder_length,
        max_decoder_length=max_decoder_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,  # Avoid multiprocessing issues
    )

    return dataloader, dataset.df


# =============================================================================
# Sequence Generation
# =============================================================================


@torch.no_grad()
def generate_sequences(
    model: TCRSeq2SeqModel,
    encoder_input_ids: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    tokenizer,
    max_length: int = 350,
    temperature: float = 1.0,
    do_sample: bool = False,
) -> List[str]:
    """
    Generate target sequences given encoded context.

    Args:
        model: TCRSeq2SeqModel
        encoder_input_ids: (batch, enc_len) encoder input
        encoder_attention_mask: (batch, enc_len) encoder mask
        tokenizer: ESM2 tokenizer
        max_length: Maximum generation length
        temperature: Sampling temperature (only used if do_sample=True)
        do_sample: Whether to use sampling (True) or greedy decoding (False)

    Returns:
        List of generated sequences (decoded strings)
    """
    model.eval()
    device = encoder_input_ids.device
    batch_size = encoder_input_ids.size(0)

    bos_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    if do_sample:
        # Sampling-based generation
        generated = _sample_decode(
            model, encoder_input_ids, encoder_attention_mask,
            bos_id, eos_id, pad_id, max_length, temperature
        )
    else:
        # Greedy decoding using the imported function
        generated = greedy_decode(
            model, encoder_input_ids, encoder_attention_mask,
            bos_id, eos_id, max_length, pad_id
        )

    # Decode to strings
    sequences = []
    for i in range(batch_size):
        tokens = generated[i].tolist()
        # Remove BOS and EOS tokens, stop at first EOS
        if bos_id in tokens:
            tokens = tokens[tokens.index(bos_id) + 1:]
        if eos_id in tokens:
            tokens = tokens[:tokens.index(eos_id)]
        # Remove padding
        tokens = [t for t in tokens if t != pad_id]

        seq = tokenizer.decode(tokens, skip_special_tokens=True)
        # Clean up sequence (remove spaces between amino acids)
        seq = seq.replace(" ", "")
        sequences.append(seq)

    return sequences


@torch.no_grad()
def _sample_decode(
    model: TCRSeq2SeqModel,
    encoder_input_ids: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_length: int,
    temperature: float,
) -> torch.Tensor:
    """
    Sampling-based decoding for sequence generation.

    Args:
        model: TCRSeq2SeqModel
        encoder_input_ids: (batch, enc_len) encoder input
        encoder_attention_mask: (batch, enc_len) encoder mask
        bos_id: Beginning of sequence token ID
        eos_id: End of sequence token ID
        pad_id: Padding token ID
        max_length: Maximum generation length
        temperature: Sampling temperature

    Returns:
        generated: (batch, gen_len) generated token IDs
    """
    device = encoder_input_ids.device
    batch_size = encoder_input_ids.size(0)

    # Encode context
    encoder_hidden = model.encode(encoder_input_ids, encoder_attention_mask)

    # Initialize decoder input with BOS
    decoder_input = torch.full(
        (batch_size, 1), bos_id, dtype=torch.long, device=device
    )

    # Track which sequences are finished
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_length - 1):
        # Create attention mask for current decoder input
        decoder_mask = torch.ones_like(decoder_input)

        # Get logits for next token
        logits = model.decode(
            decoder_input,
            decoder_mask,
            encoder_hidden,
            encoder_attention_mask,
        )

        # Apply temperature and sample
        next_token_logits = logits[:, -1, :] / temperature
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Replace finished sequences' next tokens with pad
        next_token = torch.where(
            finished.unsqueeze(-1),
            torch.full_like(next_token, pad_id),
            next_token,
        )

        # Append to decoder input
        decoder_input = torch.cat([decoder_input, next_token], dim=1)

        # Update finished status
        finished = finished | (next_token.squeeze(-1) == eos_id)

        if finished.all():
            break

    return decoder_input


# =============================================================================
# Evaluation Metrics
# =============================================================================


def compute_sequence_identity(seq1: str, seq2: str) -> float:
    """
    Compute sequence identity between two sequences.

    Uses global alignment approach - matching positions / max(len(seq1), len(seq2))
    """
    if not seq1 or not seq2:
        return 0.0

    # Use minimum length for comparison
    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))

    if max_len == 0:
        return 0.0

    # Count matching positions
    matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])

    return matches / max_len


def compute_blosum_similarity(seq1: str, seq2: str) -> float:
    """
    Compute normalized BLOSUM62 similarity between two sequences.

    Returns value between 0 and 1 (normalized by max possible score).
    """
    if not seq1 or not seq2:
        return 0.0

    min_len = min(len(seq1), len(seq2))

    if min_len == 0:
        return 0.0

    score = 0
    max_score = 0

    for i in range(min_len):
        aa1, aa2 = seq1[i].upper(), seq2[i].upper()

        # Get BLOSUM score (default to -4 for unknown amino acids)
        if aa1 in BLOSUM62 and aa2 in BLOSUM62.get(aa1, {}):
            score += BLOSUM62[aa1][aa2]
        else:
            score += -4  # Penalty for unknown

        # Max score is diagonal (self-match)
        if aa1 in BLOSUM62 and aa1 in BLOSUM62.get(aa1, {}):
            max_score += BLOSUM62[aa1][aa1]
        else:
            max_score += 4  # Default max

    if max_score == 0:
        return 0.0

    # Normalize to 0-1 range
    # BLOSUM scores can be negative, so we shift and scale
    normalized = (score + min_len * 4) / (max_score + min_len * 4)
    return max(0.0, min(1.0, normalized))


def compute_generation_metrics(
    generated_seqs: List[str],
    reference_seqs: List[str],
) -> Dict[str, float]:
    """
    Compute metrics comparing generated vs reference sequences.

    Args:
        generated_seqs: List of generated sequences
        reference_seqs: List of reference (ground truth) sequences

    Returns:
        Dictionary of metrics
    """
    n = len(generated_seqs)
    if n == 0:
        return {}

    # Exact match
    exact_matches = sum(1 for g, r in zip(generated_seqs, reference_seqs) if g == r)
    exact_match_rate = exact_matches / n

    # Sequence identity
    identities = [
        compute_sequence_identity(g, r)
        for g, r in zip(generated_seqs, reference_seqs)
    ]
    mean_identity = np.mean(identities)
    std_identity = np.std(identities)

    # BLOSUM similarity
    blosum_scores = [
        compute_blosum_similarity(g, r)
        for g, r in zip(generated_seqs, reference_seqs)
    ]
    mean_blosum = np.mean(blosum_scores)
    std_blosum = np.std(blosum_scores)

    # Length accuracy
    length_matches = sum(1 for g, r in zip(generated_seqs, reference_seqs) if len(g) == len(r))
    length_accuracy = length_matches / n

    # Length statistics
    gen_lengths = [len(g) for g in generated_seqs]
    ref_lengths = [len(r) for r in reference_seqs]
    mean_length_diff = np.mean([abs(len(g) - len(r)) for g, r in zip(generated_seqs, reference_seqs)])

    # Per-position accuracy (for same-length pairs)
    position_accuracies = []
    for g, r in zip(generated_seqs, reference_seqs):
        if len(g) == len(r) and len(g) > 0:
            matches = sum(1 for a, b in zip(g, r) if a == b)
            position_accuracies.append(matches / len(g))

    mean_position_accuracy = np.mean(position_accuracies) if position_accuracies else 0.0

    # Valid amino acid check
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    valid_seqs = sum(1 for g in generated_seqs if all(aa.upper() in valid_aas for aa in g))
    valid_aa_rate = valid_seqs / n

    return {
        "exact_match_rate": float(exact_match_rate),
        "mean_sequence_identity": float(mean_identity),
        "std_sequence_identity": float(std_identity),
        "mean_blosum_similarity": float(mean_blosum),
        "std_blosum_similarity": float(std_blosum),
        "length_accuracy": float(length_accuracy),
        "mean_length_diff": float(mean_length_diff),
        "mean_position_accuracy": float(mean_position_accuracy),
        "valid_aa_rate": float(valid_aa_rate),
        "mean_generated_length": float(np.mean(gen_lengths)),
        "mean_reference_length": float(np.mean(ref_lengths)),
        "num_samples": n,
    }


@torch.no_grad()
def compute_perplexity(
    model: TCRSeq2SeqModel,
    dataloader: DataLoader,
    device: str = "cuda",
) -> float:
    """
    Compute perplexity on a dataset.

    Args:
        model: TCRSeq2SeqModel
        dataloader: DataLoader with evaluation data
        device: Device

    Returns:
        Perplexity value
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        encoder_ids = batch["encoder_input_ids"].to(device)
        encoder_mask = batch["encoder_attention_mask"].to(device)
        decoder_ids = batch["decoder_input_ids"].to(device)
        decoder_mask = batch["decoder_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            encoder_input_ids=encoder_ids,
            encoder_attention_mask=encoder_mask,
            decoder_input_ids=decoder_ids,
            decoder_attention_mask=decoder_mask,
            labels=labels,
        )

        # Count non-padding tokens
        num_tokens = (labels != -100).sum().item()
        total_loss += outputs["loss"].item() * num_tokens
        total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(min(avg_loss, 100))  # Cap to avoid overflow

    return perplexity


@torch.no_grad()
def compute_per_sample_log_likelihood(
    model: TCRSeq2SeqModel,
    dataloader: DataLoader,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Compute per-sample log-likelihood via teacher-forced forward pass.

    For each sample, feeds the true target to the decoder and computes:
      - nll_per_token: mean NLL per token (lower = better)
      - log_prob_total: -sum(NLL) = total log P(target|context)
      - perplexity: exp(nll_per_token)

    Args:
        model: TCRSeq2SeqModel
        dataloader: DataLoader with evaluation data
        device: Device

    Returns:
        Dictionary with:
          - per_sample: list of dicts with nll_per_token, log_prob_total, perplexity
          - aggregate: dict with mean/std of each metric and corpus perplexity
    """
    model.eval()
    per_sample_results = []
    total_nll = 0.0
    total_tokens = 0

    for batch in dataloader:
        encoder_ids = batch["encoder_input_ids"].to(device)
        encoder_mask = batch["encoder_attention_mask"].to(device)
        decoder_ids = batch["decoder_input_ids"].to(device)
        decoder_mask = batch["decoder_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass to get logits
        encoder_hidden = model.encode(encoder_ids, encoder_mask)
        logits = model.decode(decoder_ids, decoder_mask, encoder_hidden, encoder_mask)

        # logits: (batch, dec_len, vocab_size), labels: (batch, dec_len)
        batch_size, dec_len, vocab_size = logits.shape

        # Compute per-token cross entropy (no reduction)
        # Reshape for F.cross_entropy: (batch*dec_len, vocab) vs (batch*dec_len,)
        loss_per_token = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
            reduction='none',
        )  # (batch * dec_len,)
        loss_per_token = loss_per_token.reshape(batch_size, dec_len)

        # Mask: valid tokens where labels != -100
        valid_mask = (labels != -100).float()  # (batch, dec_len)
        num_valid_tokens = valid_mask.sum(dim=1)  # (batch,)

        # Per-sample NLL sum and mean
        nll_sum = (loss_per_token * valid_mask).sum(dim=1)  # (batch,)
        nll_mean = nll_sum / num_valid_tokens.clamp(min=1)  # (batch,)

        for i in range(batch_size):
            ntok = int(num_valid_tokens[i].item())
            if ntok == 0:
                continue
            nll_tok = float(nll_mean[i].item())
            nll_total = float(nll_sum[i].item())
            ppl = math.exp(min(nll_tok, 100))
            per_sample_results.append({
                "nll_per_token": nll_tok,
                "log_prob_total": -nll_total,
                "perplexity": ppl,
                "num_tokens": ntok,
            })
            total_nll += nll_total
            total_tokens += ntok

    # Aggregate metrics
    nll_per_token_vals = [r["nll_per_token"] for r in per_sample_results]
    log_prob_vals = [r["log_prob_total"] for r in per_sample_results]
    ppl_vals = [r["perplexity"] for r in per_sample_results]

    corpus_ppl = math.exp(min(total_nll / total_tokens, 100)) if total_tokens > 0 else float('inf')

    aggregate = {
        "perplexity": corpus_ppl,
        "mean_nll_per_token": float(np.mean(nll_per_token_vals)),
        "std_nll_per_token": float(np.std(nll_per_token_vals)),
        "mean_log_prob_total": float(np.mean(log_prob_vals)),
        "std_log_prob_total": float(np.std(log_prob_vals)),
        "mean_perplexity_per_sample": float(np.mean(ppl_vals)),
        "std_perplexity_per_sample": float(np.std(ppl_vals)),
    }

    return {"per_sample": per_sample_results, "aggregate": aggregate}


@torch.no_grad()
def score_candidate_peptides(
    model: TCRSeq2SeqModel,
    encoder_hidden: torch.Tensor,
    encoder_mask: torch.Tensor,
    candidate_decoder_ids: torch.Tensor,
    candidate_decoder_masks: torch.Tensor,
    candidate_labels: torch.Tensor,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Score a set of candidate peptides against a single encoded context.

    Args:
        model: TCRSeq2SeqModel (eval mode)
        encoder_hidden: (1, enc_len, dim) encoder hidden states for one sample
        encoder_mask: (1, enc_len) encoder attention mask for one sample
        candidate_decoder_ids: (num_candidates, max_dec_len) decoder input ids for all candidates
        candidate_decoder_masks: (num_candidates, max_dec_len) decoder attention masks
        candidate_labels: (num_candidates, max_dec_len) labels for all candidates
        batch_size: Number of candidates to score in each forward pass

    Returns:
        log_probs: (num_candidates,) log P(candidate | context) for each candidate
    """
    num_candidates = candidate_decoder_ids.size(0)
    device = encoder_hidden.device
    log_probs = np.zeros(num_candidates, dtype=np.float64)

    for start_idx in range(0, num_candidates, batch_size):
        end_idx = min(start_idx + batch_size, num_candidates)
        bs = end_idx - start_idx

        # Expand encoder hidden to batch of candidates
        enc_hidden_batch = encoder_hidden.expand(bs, -1, -1)  # (bs, enc_len, dim)
        enc_mask_batch = encoder_mask.expand(bs, -1)  # (bs, enc_len)

        dec_ids_batch = candidate_decoder_ids[start_idx:end_idx].to(device)
        dec_mask_batch = candidate_decoder_masks[start_idx:end_idx].to(device)
        labels_batch = candidate_labels[start_idx:end_idx].to(device)

        # Forward pass through decoder
        logits = model.decode(dec_ids_batch, dec_mask_batch, enc_hidden_batch, enc_mask_batch)

        # Compute per-token log probabilities
        _, dec_len, vocab_size = logits.shape
        loss_per_token = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels_batch.reshape(-1),
            ignore_index=-100,
            reduction='none',
        ).reshape(bs, dec_len)

        valid_mask = (labels_batch != -100).float()
        # log P = -sum(NLL over valid tokens)
        nll_sum = (loss_per_token * valid_mask).sum(dim=1)  # (bs,)
        log_probs[start_idx:end_idx] = -nll_sum.cpu().numpy()

    return log_probs


@torch.no_grad()
def compute_retrieval_metrics(
    model: TCRSeq2SeqModel,
    dataset: SFTEvalDataset,
    tokenizer,
    device: str = "cuda",
    batch_size: int = 16,
    max_encoder_length: int = 1024,
    max_decoder_length: int = 350,
) -> Dict[str, Any]:
    """
    Compute retrieval metrics by ranking all unique peptides per query.

    For each test sample (TCR + MHC context), ranks all unique peptides in the
    split by generation likelihood and measures where the true peptide falls.

    Args:
        model: TCRSeq2SeqModel
        dataset: SFTEvalDataset instance
        tokenizer: ESM2 tokenizer
        device: Device
        batch_size: Batch size for scoring candidates
        max_encoder_length: Max encoder sequence length
        max_decoder_length: Max decoder sequence length

    Returns:
        Dictionary with:
          - per_sample: list of dicts with true_peptide_rank, true_peptide_log_prob
          - aggregate: dict with retrieval metrics (hit@1, hit@5, recall@10, mrr, etc.)
    """
    model.eval()
    target_col = dataset.task_config["target_col"]

    # Collect all unique peptides in the split
    all_targets = dataset.df[target_col].tolist()
    unique_peptides = sorted(set(all_targets))
    num_candidates = len(unique_peptides)
    print(f"    Retrieval: {len(all_targets)} samples, {num_candidates} unique candidates")

    # Pre-tokenize all candidate peptides
    bos_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    # For each candidate: decoder_input_ids = [BOS] + tokens, labels = tokens + [EOS]
    candidate_token_lists = []
    for pep in unique_peptides:
        # Tokenize the peptide (space-separated amino acids for ESM tokenizer)
        spaced_pep = " ".join(list(pep))
        tokens = tokenizer.encode(spaced_pep, add_special_tokens=False)
        candidate_token_lists.append(tokens)

    # Determine max decoder length for candidates
    max_cand_len = min(max(len(t) for t in candidate_token_lists) + 1, max_decoder_length)  # +1 for BOS/EOS

    # Build padded tensors for decoder inputs and labels
    candidate_decoder_ids = torch.full((num_candidates, max_cand_len), pad_id, dtype=torch.long)
    candidate_decoder_masks = torch.zeros((num_candidates, max_cand_len), dtype=torch.long)
    candidate_labels = torch.full((num_candidates, max_cand_len), -100, dtype=torch.long)

    for i, tokens in enumerate(candidate_token_lists):
        # decoder_input_ids: [BOS, tok1, tok2, ..., tokN]
        dec_ids = [bos_id] + tokens
        dec_len = min(len(dec_ids), max_cand_len)
        candidate_decoder_ids[i, :dec_len] = torch.tensor(dec_ids[:dec_len], dtype=torch.long)
        candidate_decoder_masks[i, :dec_len] = 1

        # labels: [tok1, tok2, ..., tokN, EOS]
        lab = tokens + [eos_id]
        lab_len = min(len(lab), max_cand_len)
        candidate_labels[i, :lab_len] = torch.tensor(lab[:lab_len], dtype=torch.long)

    # Build a mapping from peptide string to index in unique_peptides
    peptide_to_idx = {pep: idx for idx, pep in enumerate(unique_peptides)}

    # Collator for encoding single samples
    collator = Seq2SeqCollator(
        tokenizer=tokenizer,
        max_encoder_length=max_encoder_length,
        max_decoder_length=max_decoder_length,
    )

    per_sample_results = []
    score_matrix_rows = []
    true_indices_list = []
    sample_peptide_list = []

    for sample_idx in tqdm(range(len(dataset)), desc="    Retrieval scoring"):
        sample = dataset[sample_idx]
        true_peptide = sample["decoder_target"]

        if true_peptide not in peptide_to_idx:
            continue

        true_idx = peptide_to_idx[true_peptide]

        # Encode the context for this sample
        batch = collator([sample])
        encoder_ids = batch["encoder_input_ids"].to(device)
        encoder_mask = batch["encoder_attention_mask"].to(device)

        encoder_hidden = model.encode(encoder_ids, encoder_mask)  # (1, enc_len, dim)

        # Score all candidates against this context
        log_probs = score_candidate_peptides(
            model=model,
            encoder_hidden=encoder_hidden,
            encoder_mask=encoder_mask,
            candidate_decoder_ids=candidate_decoder_ids,
            candidate_decoder_masks=candidate_decoder_masks,
            candidate_labels=candidate_labels,
            batch_size=batch_size,
        )

        # Rank candidates by log-likelihood (descending = highest first)
        sorted_indices = np.argsort(-log_probs)
        rank = int(np.where(sorted_indices == true_idx)[0][0]) + 1  # 1-indexed

        per_sample_results.append({
            "true_peptide_rank": rank,
            "true_peptide_log_prob": float(log_probs[true_idx]),
        })
        score_matrix_rows.append(log_probs)
        true_indices_list.append(true_idx)
        sample_peptide_list.append(true_peptide)

    # Build score matrix for unified metrics
    if len(score_matrix_rows) > 0:
        score_matrix = np.array(score_matrix_rows)
        true_indices_arr = np.array(true_indices_list)
        sample_peptides_arr = np.array(sample_peptide_list)

        # Compute all unified metrics using shared module
        unified = compute_all_unified_metrics(
            scores=score_matrix,
            true_indices=true_indices_arr,
            sample_peptides=sample_peptides_arr,
            candidate_peptides=unique_peptides,
            n_bootstrap=1000,
            min_samples_per_peptide=5,
            seed=42,
        )
    else:
        unified = {
            "retrieval_hit_at_1": 0.0,
            "retrieval_hit_at_5": 0.0,
            "retrieval_recall_at_10": 0.0,
            "retrieval_mrr": 0.0,
            "retrieval_mean_rank": float('inf'),
            "retrieval_median_rank": float('inf'),
            "retrieval_num_candidates": num_candidates,
        }

    return {"per_sample": per_sample_results, "unified_metrics": unified}


# =============================================================================
# Main Evaluation Loop
# =============================================================================


def evaluate_on_split(
    model: TCRSeq2SeqModel,
    tokenizer,
    parquet_path: str,
    task_config: Dict,
    output_dir: Path,
    split_name: str,
    device: str = "cuda",
    batch_size: int = 16,
    temperature: float = 1.0,
    do_sample: bool = False,
    max_length: int = 350,
) -> Dict:
    """
    Evaluate model on a single test split.

    Args:
        model: Loaded SFT model
        tokenizer: ESM2 tokenizer
        parquet_path: Path to test split parquet file
        task_config: Task configuration
        output_dir: Output directory for this split
        split_name: Name of the split
        device: Device
        batch_size: Batch size
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        max_length: Maximum generation length

    Returns:
        Dictionary with metrics
    """
    print(f"  Evaluating on {split_name}...")

    # Prepare data
    dataloader, df = prepare_eval_dataloader(
        parquet_path, task_config, tokenizer, batch_size
    )

    print(f"    Samples: {len(df)}")

    # Generate sequences
    all_generated = []
    all_references = []

    target_col = task_config["target_col"]

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"    Generating")):
        encoder_ids = batch["encoder_input_ids"].to(device)
        encoder_mask = batch["encoder_attention_mask"].to(device)

        # Generate
        generated = generate_sequences(
            model, encoder_ids, encoder_mask, tokenizer,
            max_length=max_length,
            temperature=temperature,
            do_sample=do_sample,
        )
        all_generated.extend(generated)

    # Get reference sequences from dataframe
    all_references = df[target_col].tolist()

    # Compute generation metrics
    metrics = compute_generation_metrics(all_generated, all_references)

    # Compute per-sample log-likelihood (replaces old compute_perplexity)
    ll_results = compute_per_sample_log_likelihood(model, dataloader, device)
    metrics.update(ll_results["aggregate"])

    print(f"    Exact match: {metrics['exact_match_rate']:.4f}")
    print(f"    Sequence identity: {metrics['mean_sequence_identity']:.4f}")
    print(f"    BLOSUM similarity: {metrics['mean_blosum_similarity']:.4f}")
    print(f"    Perplexity: {metrics['perplexity']:.2f}")
    print(f"    Mean NLL/token: {metrics['mean_nll_per_token']:.4f}")

    # Compute retrieval metrics
    dataset = SFTEvalDataset(parquet_path, task_config)
    retrieval_results = compute_retrieval_metrics(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )
    unified = retrieval_results["unified_metrics"]
    per_epitope_all = unified.pop("per_epitope_all", [])
    metrics.update(unified)

    print(f"    Retrieval Hit@1: {metrics['retrieval_hit_at_1']:.4f}")
    print(f"    Retrieval MRR: {metrics['retrieval_mrr']:.4f}")
    print(f"    Retrieval Mean Rank: {metrics['retrieval_mean_rank']:.2f}")
    if metrics.get("per_peptide_auc_mean") is not None:
        print(f"    Per-peptide AUC: {metrics['per_peptide_auc_mean']:.4f}")

    # Save predictions
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame({
        "generated": all_generated,
        "reference": all_references,
        "exact_match": [g == r for g, r in zip(all_generated, all_references)],
        "sequence_identity": [
            compute_sequence_identity(g, r)
            for g, r in zip(all_generated, all_references)
        ],
        "blosum_similarity": [
            compute_blosum_similarity(g, r)
            for g, r in zip(all_generated, all_references)
        ],
    })

    # Add per-sample likelihood columns
    ll_per_sample = ll_results["per_sample"]
    if len(ll_per_sample) == len(pred_df):
        pred_df["nll_per_token"] = [r["nll_per_token"] for r in ll_per_sample]
        pred_df["log_prob_total"] = [r["log_prob_total"] for r in ll_per_sample]
        pred_df["perplexity"] = [r["perplexity"] for r in ll_per_sample]

    # Add retrieval columns
    retrieval_per_sample = retrieval_results["per_sample"]
    if len(retrieval_per_sample) == len(pred_df):
        pred_df["true_peptide_rank"] = [r["true_peptide_rank"] for r in retrieval_per_sample]
        pred_df["true_peptide_log_prob"] = [r["true_peptide_log_prob"] for r in retrieval_per_sample]

    pred_df.to_csv(output_dir / "predictions.csv", index=False)

    # Save per-epitope breakdown CSV
    if per_epitope_all:
        epitope_df = pd.DataFrame(per_epitope_all)
        epitope_df.to_csv(output_dir / "per_epitope_breakdown.csv", index=False)

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def evaluate_on_all_splits(
    model_path: str,
    data_dir: str,
    task_name: str,
    output_dir: str,
    device: str = "cuda",
    batch_size: int = 16,
    temperature: float = 1.0,
    do_sample: bool = False,
    max_length: int = 350,
    use_flash_attention: bool = True,
) -> Dict:
    """
    Evaluate model on all test splits for a given task.

    Args:
        model_path: Path to model checkpoint
        data_dir: Base data directory
        task_name: Name of the task
        output_dir: Output directory for results
        device: Device
        batch_size: Batch size
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        max_length: Maximum generation length
        use_flash_attention: Whether to use flash attention

    Returns:
        Results dictionary
    """
    task_config = SFT_TASK_CONFIGS[task_name]
    mhc_class = task_config["mhc_class"]

    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"Description: {task_config['description']}")
    print(f"{'='*60}")

    # Load model
    model, tokenizer, config = load_sft_model(model_path, device, use_flash_attention)

    # Create output directory
    task_output_dir = Path(output_dir) / task_name
    task_output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "task": task_name,
        "model_path": str(model_path),
        "config": config,
        "temperature": temperature,
        "do_sample": do_sample,
        "splits": {},
    }

    # Evaluate on each split
    for split_name in TEST_SPLITS:
        split_path = Path(data_dir) / mhc_class / task_name / f"{split_name}.parquet"

        if not split_path.exists():
            print(f"\n  Skipping {split_name} - file not found: {split_path}")
            continue

        split_output_dir = task_output_dir / split_name
        metrics = evaluate_on_split(
            model=model,
            tokenizer=tokenizer,
            parquet_path=str(split_path),
            task_config=task_config,
            output_dir=split_output_dir,
            split_name=split_name,
            device=device,
            batch_size=batch_size,
            temperature=temperature,
            do_sample=do_sample,
            max_length=max_length,
        )

        results["splits"][split_name] = metrics

    # Save task results
    with open(task_output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {task_output_dir}")

    return results


def run_all_tasks(
    model_dir: str,
    data_dir: str,
    output_dir: str,
    tasks: Optional[List[str]] = None,
    **kwargs,
) -> Dict:
    """
    Run evaluation for all tasks (or specified tasks).

    Args:
        model_dir: Directory containing task subdirectories with checkpoints
        data_dir: Base data directory
        output_dir: Output directory
        tasks: List of tasks to evaluate (None = all)
        **kwargs: Additional arguments passed to evaluate_on_all_splits

    Returns:
        Summary dictionary with all results
    """
    if tasks is None:
        tasks = list(SFT_TASK_CONFIGS.keys())

    summary = {
        "timestamp": datetime.now().isoformat(),
        "model_dir": model_dir,
        "data_dir": data_dir,
        "tasks": {},
    }

    for task_name in tasks:
        # Look for model checkpoint
        task_model_path = Path(model_dir) / task_name
        if not task_model_path.exists():
            # Try using model_dir directly as checkpoint path
            task_model_path = Path(model_dir)

        if not task_model_path.exists():
            print(f"\nSkipping {task_name} - model not found at {task_model_path}")
            continue

        try:
            results = evaluate_on_all_splits(
                model_path=str(task_model_path),
                data_dir=data_dir,
                task_name=task_name,
                output_dir=output_dir,
                **kwargs,
            )
            summary["tasks"][task_name] = results
        except Exception as e:
            print(f"\nError evaluating {task_name}: {e}")
            import traceback
            traceback.print_exc()
            summary["tasks"][task_name] = {"error": str(e)}

    # Save summary
    summary_path = Path(output_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Benchmark complete!")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}")

    return summary


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SFT models on TCR generation tasks"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint or directory containing task checkpoints",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=list(SFT_TASK_CONFIGS.keys()) + ["all"],
        default="all",
        help="Task to evaluate (default: all)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/sravisha/projects/quest/data/icml/tasks/tcr_specificity/tcr90pep80",
        help="Base data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/sravisha/projects/quest/results/sft_benchmark",
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (only used with --do_sample)",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling instead of greedy decoding",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples per input (only used with --do_sample)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=350,
        help="Maximum generation length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--no_flash_attention",
        action="store_true",
        help="Disable flash attention",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"SFT Benchmark Evaluation")
    print(f"========================")
    print(f"Model path: {args.model_path}")
    print(f"Task: {args.task}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sampling: {args.do_sample} (temp={args.temperature})")

    if args.task == "all":
        # Run all tasks
        run_all_tasks(
            model_dir=args.model_path,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
            temperature=args.temperature,
            do_sample=args.do_sample,
            max_length=args.max_length,
            use_flash_attention=not args.no_flash_attention,
        )
    else:
        # Run single task
        evaluate_on_all_splits(
            model_path=args.model_path,
            data_dir=args.data_dir,
            task_name=args.task,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
            temperature=args.temperature,
            do_sample=args.do_sample,
            max_length=args.max_length,
            use_flash_attention=not args.no_flash_attention,
        )


if __name__ == "__main__":
    main()
