#!/usr/bin/env python3
"""
Peptide-to-MHC Conditional Sequence Generation Trainer

Encoder-decoder architecture for predicting MHC sequences from peptide input.
Uses ESM2 as encoder with a TransformerDecoder for autoregressive generation.

Architecture:
    [Encoder: ESM2]           [Decoder: torch.nn.TransformerDecoder]
    [Peptide sequence]  -->  Cross-Attention  -->  MHC sequence(s)
         (8-25 AA)            + Causal LM         (~365-520 AA)

Task Types:
- MHC_CLASS_I: Peptide -> single MHC chain (~365 AA)
- MHC_CLASS_II: Peptide -> two MHC chains concatenated with <eos> separator (~520 AA)

Features:
- ESM2 encoder with flash attention 2, frozen or with LoRA
- TransformerDecoder with cross-attention to encoder outputs
- Weight tying: decoder embeddings shared with ESM2 encoder embeddings
- Autoregressive next-token prediction with cross-entropy loss
- DDP for multi-GPU training
- Mixed precision (bfloat16)

Usage:
    # Class I (single chain)
    torchrun --nproc_per_node=8 scripts/training/peptide_mhc_seq2seq_trainer.py \
        --data_path data/icml/tasks/peptide_mhc3/class_one/peptide_mhc_one/ \
        --output_dir ./output/peptide_mhc_class_i \
        --task MHC_CLASS_I \
        --model_name facebook/esm2_t33_650M_UR50D \
        --decoder_dim 1280 --decoder_heads 20 \
        --batch_size 8 --use_lora

    # Class II (two chains)
    torchrun --nproc_per_node=8 scripts/training/peptide_mhc_seq2seq_trainer.py \
        --data_path data/icml/tasks/peptide_mhc3/class_two/peptide_mhc_one_mhc_two/ \
        --output_dir ./output/peptide_mhc_class_ii \
        --task MHC_CLASS_II \
        --max_decoder_length 550 \
        --batch_size 4
"""

import argparse
import copy
import glob
import math
import os
import sys
from datetime import datetime, timedelta

# Add project root to path for direct script execution
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Backend abstraction for hardware-agnostic training
try:
    from .backends import AcceleratorBackend, get_backend
    from .base_trainer import BaseTCRTrainer
except ImportError:
    from scripts.training.backends import AcceleratorBackend, get_backend
    from scripts.training.base_trainer import BaseTCRTrainer

# Optional: LoRA support
try:
    from peft import LoraConfig, TaskType, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# Optional: wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============================================================================
# Enums and Constants
# =============================================================================


class MHCGenerationTask(Enum):
    """Task types for MHC sequence generation from peptide."""
    MHC_CLASS_I = "mhc_class_i"    # Single chain output (~365 AA)
    MHC_CLASS_II = "mhc_class_ii"  # Two chains: alpha + beta (~520 AA total)


# BLOSUM62 substitution matrix for similarity metrics
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

# Standard amino acids
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


# =============================================================================
# Positional Encoding
# =============================================================================


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:seq_len].unsqueeze(0).to(x.dtype)
        return self.dropout(x)


# =============================================================================
# Model: PeptideMHCSeq2SeqModel (reuses TCRSeq2SeqModel architecture)
# =============================================================================


class PeptideMHCSeq2SeqModel(nn.Module):
    """
    Encoder-decoder model for peptide-to-MHC sequence generation.

    Uses ESM2 as encoder and torch.nn.TransformerDecoder for generation.
    Decoder embeddings are tied to ESM2 encoder embeddings for better convergence.

    Args:
        encoder_model_name: Pretrained ESM2 model name
        decoder_layers: Number of decoder transformer layers
        decoder_heads: Number of attention heads in decoder
        decoder_dim: Hidden dimension (must match ESM2)
        decoder_ffn_dim: FFN intermediate dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        encoder_model_name: str = "facebook/esm2_t33_650M_UR50D",
        decoder_layers: int = 6,
        decoder_heads: int = 20,
        decoder_dim: int = 1280,
        decoder_ffn_dim: int = 5120,
        dropout: float = 0.1,
        attn_implementation: Optional[str] = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        decoder_warm_start: bool = False,
    ):
        super().__init__()

        # Load ESM2 encoder
        encoder_kwargs = {"dtype": torch_dtype}

        if attn_implementation == "auto":
            try:
                import flash_attn
                encoder_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                encoder_kwargs["attn_implementation"] = "eager"
        elif attn_implementation:
            encoder_kwargs["attn_implementation"] = attn_implementation

        import logging as _logging
        _hf_logger = _logging.getLogger("transformers.modeling_utils")
        _prev_level = _hf_logger.level
        _hf_logger.setLevel(_logging.ERROR)
        self.encoder = AutoModel.from_pretrained(
            encoder_model_name,
            **encoder_kwargs,
        )
        _hf_logger.setLevel(_prev_level)

        # Get encoder config
        self.encoder_dim = self.encoder.config.hidden_size
        self.vocab_size = self.encoder.config.vocab_size

        # Verify dimensions match
        assert decoder_dim == self.encoder_dim, (
            f"decoder_dim ({decoder_dim}) must match encoder hidden size ({self.encoder_dim})"
        )

        # Share encoder embeddings with decoder
        self.decoder_embed = self.encoder.embeddings.word_embeddings

        # Positional encoding for decoder
        self.decoder_pos_encoding = PositionalEncoding(
            d_model=decoder_dim,
            max_len=700,  # Max MHC sequence length (Class II)
            dropout=dropout
        )

        # Standard PyTorch decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers,
        )

        # Output projection tied to decoder embeddings (weight tying)
        self.output_proj = nn.Linear(decoder_dim, self.vocab_size, bias=False)
        self.output_proj.weight = self.decoder_embed.weight

        # Layer norm before output projection
        self.output_norm = nn.LayerNorm(decoder_dim)

        # Convert decoder components to same dtype as encoder
        self.decoder_pos_encoding = self.decoder_pos_encoding.to(torch_dtype)
        self.decoder = self.decoder.to(torch_dtype)
        self.output_norm = self.output_norm.to(torch_dtype)

        # Initialize decoder from encoder weights if requested
        if decoder_warm_start:
            self._initialize_decoder_from_encoder()

    def _initialize_decoder_from_encoder(self) -> None:
        """Initialize decoder transformer layers from pre-trained ESM2 encoder."""
        num_encoder_layers = len(self.encoder.encoder.layer)
        num_decoder_layers = len(self.decoder.layers)
        num_layers_to_copy = min(num_encoder_layers, num_decoder_layers)

        print(f"[Warm-Start] Initializing {num_layers_to_copy} decoder layers from ESM2 encoder...")

        for i in range(num_layers_to_copy):
            enc_layer = self.encoder.encoder.layer[i]
            dec_layer = self.decoder.layers[i]

            # Self-Attention Q, K, V -> in_proj_weight
            q_weight = enc_layer.attention.self.query.weight.data
            k_weight = enc_layer.attention.self.key.weight.data
            v_weight = enc_layer.attention.self.value.weight.data
            in_proj_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            dec_layer.self_attn.in_proj_weight.data.copy_(in_proj_weight)

            if enc_layer.attention.self.query.bias is not None:
                q_bias = enc_layer.attention.self.query.bias.data
                k_bias = enc_layer.attention.self.key.bias.data
                v_bias = enc_layer.attention.self.value.bias.data
                in_proj_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                dec_layer.self_attn.in_proj_bias.data.copy_(in_proj_bias)

            # Self-Attention Output Projection
            dec_layer.self_attn.out_proj.weight.data.copy_(
                enc_layer.attention.output.dense.weight.data
            )
            if enc_layer.attention.output.dense.bias is not None:
                dec_layer.self_attn.out_proj.bias.data.copy_(
                    enc_layer.attention.output.dense.bias.data
                )

            # FFN Layers
            dec_layer.linear1.weight.data.copy_(enc_layer.intermediate.dense.weight.data)
            dec_layer.linear2.weight.data.copy_(enc_layer.output.dense.weight.data)

            if enc_layer.intermediate.dense.bias is not None:
                dec_layer.linear1.bias.data.copy_(enc_layer.intermediate.dense.bias.data)
            if enc_layer.output.dense.bias is not None:
                dec_layer.linear2.bias.data.copy_(enc_layer.output.dense.bias.data)

            # Layer Norms
            dec_layer.norm1.weight.data.copy_(enc_layer.attention.LayerNorm.weight.data)
            dec_layer.norm1.bias.data.copy_(enc_layer.attention.LayerNorm.bias.data)
            dec_layer.norm3.weight.data.copy_(enc_layer.LayerNorm.weight.data)
            dec_layer.norm3.bias.data.copy_(enc_layer.LayerNorm.bias.data)

        print(f"[Warm-Start] Complete. Cross-attention layers remain randomly initialized.")

    def _generate_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Generate causal attention mask for decoder."""
        mask = torch.triu(
            torch.full((seq_len, seq_len), -10000.0, device=device, dtype=dtype),
            diagonal=1
        )
        return mask

    def encode(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode peptide sequences with ESM2."""
        outputs = self.encoder(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
        )
        return outputs.last_hidden_state

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        encoder_hidden: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode MHC sequence with cross-attention to encoder."""
        batch_size, dec_len = decoder_input_ids.shape
        device = decoder_input_ids.device

        # Embed decoder inputs (shared with encoder)
        decoder_emb = self.decoder_embed(decoder_input_ids)
        decoder_emb = self.decoder_pos_encoding(decoder_emb)

        dtype = decoder_emb.dtype

        # Causal mask for autoregressive decoding
        causal_mask = self._generate_causal_mask(dec_len, device, dtype=dtype)

        # Convert padding masks to additive float masks
        tgt_key_padding_mask = (1.0 - decoder_attention_mask.to(dtype)) * -10000.0
        memory_key_padding_mask = (1.0 - encoder_attention_mask.to(dtype)) * -10000.0

        # Run decoder
        decoder_output = self.decoder(
            tgt=decoder_emb,
            memory=encoder_hidden,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        # Project to vocabulary
        decoder_output = self.output_norm(decoder_output)
        logits = self.output_proj(decoder_output)

        return logits

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass for training."""
        # Encode context
        encoder_hidden = self.encode(encoder_input_ids, encoder_attention_mask)

        # Decode target
        logits = self.decode(
            decoder_input_ids,
            decoder_attention_mask,
            encoder_hidden,
            encoder_attention_mask,
        )

        output = {"logits": logits}

        # Compute loss if labels provided
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss

        return output


def apply_lora_to_encoder(model: PeptideMHCSeq2SeqModel, config: dict) -> PeptideMHCSeq2SeqModel:
    """Apply LoRA to the ESM2 encoder."""
    if not PEFT_AVAILABLE:
        raise ImportError("peft is required for LoRA. Install with: pip install peft")

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=["query", "key", "value", "dense"],
        bias="none",
    )

    model.encoder = get_peft_model(model.encoder, lora_config)
    return model


def load_encoder_checkpoint(
    model: PeptideMHCSeq2SeqModel,
    checkpoint_path: str,
    strict: bool = False,
    verbose: bool = True,
) -> PeptideMHCSeq2SeqModel:
    """Load pre-trained encoder weights from a checkpoint file."""
    if verbose:
        print(f"Loading encoder checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")

    encoder_state_dict = model.encoder.state_dict()
    encoder_keys = set(encoder_state_dict.keys())

    key_mapping = {}
    matched_keys = []

    for ckpt_key in state_dict.keys():
        if ckpt_key in encoder_keys:
            key_mapping[ckpt_key] = ckpt_key
            matched_keys.append(ckpt_key)
            continue

        modified_key = ckpt_key
        if modified_key.startswith("encoder."):
            modified_key = modified_key[len("encoder."):]
            if modified_key in encoder_keys:
                key_mapping[ckpt_key] = modified_key
                matched_keys.append(ckpt_key)
                continue

        if modified_key in encoder_keys:
            key_mapping[ckpt_key] = modified_key
            matched_keys.append(ckpt_key)

    new_state_dict = {}
    for ckpt_key, encoder_key in key_mapping.items():
        new_state_dict[encoder_key] = state_dict[ckpt_key]

    if verbose:
        print(f"  Matched {len(matched_keys)} / {len(state_dict)} checkpoint keys")
        print(f"  Loading {len(new_state_dict)} weights into encoder")

    model.encoder.load_state_dict(new_state_dict, strict=strict)

    if verbose:
        print("  Encoder checkpoint loaded successfully!")

    return model


# =============================================================================
# Dataset: PeptideMHCDataset
# =============================================================================


class PeptideMHCDataset(Dataset):
    """
    Dataset for peptide-to-MHC sequence generation.

    Loads pre-split parquet files with peptide and MHC columns.
    Encoder input is the peptide sequence.
    Decoder target is the MHC sequence (or concatenated chains for Class II).

    Args:
        data_path: Path to directory containing train.parquet, val.parquet, test.parquet
        task: MHC generation task (MHC_CLASS_I or MHC_CLASS_II)
        split: One of 'train', 'val', or 'test'
        local_rank: Local rank for DDP logging
    """

    _cache: Dict[str, List[Dict[str, str]]] = {}

    def __init__(
        self,
        data_path: str,
        task: MHCGenerationTask,
        split: str = "train",
        local_rank: int = 0,
    ):
        import pyarrow.parquet as pq

        self.task = task
        self.split = split
        is_main = local_rank == 0

        # Create cache key
        cache_key = f"{data_path}:{split}"

        if cache_key in PeptideMHCDataset._cache:
            if is_main:
                print(f"Using cached data for {split} split...")
            self.data = PeptideMHCDataset._cache[cache_key]
        else:
            # Load the appropriate split file
            split_file = os.path.join(data_path, f"{split}.parquet")

            if not os.path.exists(split_file):
                # Fallback: load all parquet files and split manually
                if is_main:
                    print(f"Split file {split_file} not found, loading all parquet files...")
                parquet_files = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
                if not parquet_files:
                    raise ValueError(f"No parquet files found in {data_path}")

                all_data = []
                for pf in tqdm(parquet_files, desc="Loading", disable=not is_main):
                    try:
                        table = pq.read_table(pf)
                        df = table.to_pandas()

                        for _, row in df.iterrows():
                            record = self._extract_record(row, task)
                            if record is not None:
                                all_data.append(record)
                    except Exception as e:
                        if is_main:
                            print(f"Warning: Error reading {pf}: {e}")
                        continue

                # Create deterministic train/val/test split
                n_total = len(all_data)
                np.random.seed(42)
                indices = np.random.permutation(n_total)

                train_end = int(n_total * 0.8)
                val_end = int(n_total * 0.9)

                if split == "train":
                    selected_indices = indices[:train_end]
                elif split == "val":
                    selected_indices = indices[train_end:val_end]
                else:  # test
                    selected_indices = indices[val_end:]

                self.data = [all_data[i] for i in selected_indices]
            else:
                # Load from pre-split file
                if is_main:
                    print(f"Loading {split} split from {split_file}...")

                table = pq.read_table(split_file)
                df = table.to_pandas()

                self.data = []
                for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing", disable=not is_main):
                    record = self._extract_record(row, task)
                    if record is not None:
                        self.data.append(record)

            if not self.data:
                raise ValueError(f"No valid records found for {task.value} task in {data_path}")

            PeptideMHCDataset._cache[cache_key] = self.data

            if is_main:
                print(f"Loaded {len(self.data):,} {split} records for {task.value}")

    def _extract_record(self, row, task: MHCGenerationTask) -> Optional[Dict[str, str]]:
        """Extract peptide and MHC from a row based on task."""
        peptide = row.get("peptide", "")
        mhc_one = row.get("mhc_one", "")
        mhc_two = row.get("mhc_two", "")

        # Validate peptide
        if not peptide or len(peptide) < 5:
            return None

        # Validate MHC based on task
        if task == MHCGenerationTask.MHC_CLASS_I:
            if not mhc_one or len(mhc_one) < 50:
                return None
            return {
                "peptide": peptide,
                "mhc_target": mhc_one,
                "mhc_one_id": row.get("mhc_one_id", ""),
            }
        else:  # MHC_CLASS_II
            if not mhc_one or not mhc_two or len(mhc_one) < 50 or len(mhc_two) < 50:
                return None
            return {
                "peptide": peptide,
                "mhc_one": mhc_one,
                "mhc_two": mhc_two,
                "mhc_one_id": row.get("mhc_one_id", ""),
                "mhc_two_id": row.get("mhc_two_id", ""),
            }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get encoder input (peptide) and decoder target (MHC)."""
        record = self.data[idx]

        # Encoder context is just the peptide
        encoder_context = [record["peptide"]]

        # Decoder target depends on task
        if self.task == MHCGenerationTask.MHC_CLASS_I:
            decoder_target = record["mhc_target"]
        else:  # MHC_CLASS_II
            # Concatenate two chains with separator (will be handled by collator)
            decoder_target = (record["mhc_one"], record["mhc_two"])

        return {
            "encoder_context": encoder_context,
            "decoder_target": decoder_target,
            "task": self.task.value,
        }


# =============================================================================
# Data Collator: PeptideMHCCollator
# =============================================================================


class PeptideMHCCollator:
    """
    Collates peptide-MHC sequences for seq2seq training.

    Handles:
    - Short encoder inputs (peptides, max ~64 tokens)
    - Long decoder outputs (MHC, max ~400-550 tokens)
    - Class II: concatenates two MHC chains with <eos> separator

    Args:
        tokenizer: ESM2 tokenizer
        max_encoder_length: Maximum encoder sequence length (64 for peptides)
        max_decoder_length: Maximum decoder sequence length (400 for Class I, 550 for Class II)
    """

    def __init__(
        self,
        tokenizer,
        max_encoder_length: int = 64,
        max_decoder_length: int = 400,
    ):
        self.tokenizer = tokenizer
        self.max_encoder_length = max_encoder_length
        self.max_decoder_length = max_decoder_length

        # Get special token IDs
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.cls_token_id  # ESM2 uses <cls> as BOS
        self.eos_id = tokenizer.eos_token_id

        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id
        if self.bos_id is None:
            self.bos_id = tokenizer.bos_token_id or 0

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of examples."""
        batch_size = len(examples)

        # Process encoder inputs (peptides)
        encoder_seqs = []
        for ex in examples:
            # encoder_context is a list with just the peptide
            peptide = ex["encoder_context"][0] if ex["encoder_context"] else ""
            encoder_seqs.append(peptide)

        # Tokenize encoder inputs
        encoder_encoded = self.tokenizer(
            encoder_seqs,
            padding="max_length",
            truncation=True,
            max_length=self.max_encoder_length,
            return_tensors="pt",
        )

        # Process decoder inputs and labels
        decoder_input_ids_list = []
        decoder_attention_mask_list = []
        labels_list = []

        for ex in examples:
            target = ex["decoder_target"]

            # Handle Class II: concatenate chains with EOS separator
            if isinstance(target, tuple):
                # Class II: (mhc_one, mhc_two) -> "mhc_one<eos>mhc_two"
                mhc_one, mhc_two = target
                target_str = f"{mhc_one}{self.tokenizer.eos_token}{mhc_two}"
            else:
                # Class I: single MHC chain
                target_str = target

            # Tokenize target (without special tokens)
            target_encoded = self.tokenizer(
                target_str,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_decoder_length - 2,  # Reserve space for BOS/EOS
            )
            target_ids = target_encoded["input_ids"]

            # Decoder input: BOS + target (for teacher forcing)
            decoder_input = [self.bos_id] + target_ids

            # Labels: target + EOS
            labels = target_ids + [self.eos_id]

            # Pad to max length
            dec_len = len(decoder_input)
            pad_len = self.max_decoder_length - dec_len

            if pad_len > 0:
                decoder_input = decoder_input + [self.pad_id] * pad_len
                labels = labels + [-100] * pad_len
                attention_mask = [1] * dec_len + [0] * pad_len
            else:
                decoder_input = decoder_input[:self.max_decoder_length]
                labels = labels[:self.max_decoder_length]
                attention_mask = [1] * self.max_decoder_length

            decoder_input_ids_list.append(decoder_input)
            decoder_attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        return {
            "encoder_input_ids": encoder_encoded["input_ids"],
            "encoder_attention_mask": encoder_encoded["attention_mask"],
            "decoder_input_ids": torch.tensor(decoder_input_ids_list, dtype=torch.long),
            "decoder_attention_mask": torch.tensor(decoder_attention_mask_list, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }


# =============================================================================
# Generation Utilities
# =============================================================================


@torch.no_grad()
def greedy_decode(
    model: PeptideMHCSeq2SeqModel,
    encoder_input_ids: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_length: int = 400,
    pad_id: Optional[int] = None,
) -> torch.Tensor:
    """Greedy decoding for MHC sequence generation."""
    model.eval()
    device = encoder_input_ids.device
    batch_size = encoder_input_ids.size(0)
    pad_id = pad_id if pad_id is not None else eos_id

    # Encode peptide context
    encoder_hidden = model.encode(encoder_input_ids, encoder_attention_mask)

    # Initialize decoder input with BOS
    decoder_input = torch.full(
        (batch_size, 1), bos_id, dtype=torch.long, device=device
    )

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_length - 1):
        decoder_mask = torch.ones_like(decoder_input)

        logits = model.decode(
            decoder_input,
            decoder_mask,
            encoder_hidden,
            encoder_attention_mask,
        )

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        next_token = torch.where(
            finished.unsqueeze(-1),
            torch.full_like(next_token, pad_id),
            next_token,
        )

        decoder_input = torch.cat([decoder_input, next_token], dim=1)
        finished = finished | (next_token.squeeze(-1) == eos_id)

        if finished.all():
            break

    return decoder_input


# =============================================================================
# Sequence Similarity Metrics
# =============================================================================


def levenshtein_distance(seq1: str, seq2: str) -> int:
    """Compute Levenshtein (edit) distance between two sequences."""
    if len(seq1) < len(seq2):
        return levenshtein_distance(seq2, seq1)

    if len(seq2) == 0:
        return len(seq1)

    previous_row = range(len(seq2) + 1)
    for i, c1 in enumerate(seq1):
        current_row = [i + 1]
        for j, c2 in enumerate(seq2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def sequence_identity(seq1: str, seq2: str) -> float:
    """Compute sequence identity (fraction of matching positions)."""
    if not seq1 or not seq2:
        return 0.0

    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))

    matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
    return matches / max_len


def blosum62_similarity(seq1: str, seq2: str) -> float:
    """Compute BLOSUM62 similarity score between two sequences."""
    if not seq1 or not seq2:
        return 0.0

    min_len = min(len(seq1), len(seq2))
    gap_penalty = -4

    score = 0.0
    for i in range(min_len):
        aa1 = seq1[i].upper()
        aa2 = seq2[i].upper()
        if aa1 in BLOSUM62 and aa2 in BLOSUM62[aa1]:
            score += BLOSUM62[aa1][aa2]
        else:
            score += gap_penalty

    len_diff = abs(len(seq1) - len(seq2))
    score += len_diff * gap_penalty

    return score


def blosum62_normalized(seq1: str, seq2: str) -> float:
    """Compute length-normalized BLOSUM62 similarity."""
    if not seq1 or not seq2:
        return 0.0

    score = blosum62_similarity(seq1, seq2)
    max_len = max(len(seq1), len(seq2))
    return score / max_len


# =============================================================================
# Evaluator: MHCGenerationEvaluator
# =============================================================================


class MHCGenerationEvaluator:
    """
    Evaluator for MHC sequence generation.

    Metrics:
    - Cross-entropy loss (teacher-forced)
    - Perplexity
    - Exact match rate
    - Sequence identity / BLOSUM62 similarity
    - Per-chain metrics for Class II
    - Length statistics
    """

    def __init__(self, tokenizer, task: MHCGenerationTask):
        self.tokenizer = tokenizer
        self.task = task

    def _compute_sequence_metrics(
        self,
        generated_seqs: List[str],
        reference_seqs: List[str],
    ) -> Dict[str, Any]:
        """Compute comprehensive sequence similarity metrics."""
        metrics = {}

        # Clean sequences
        gen_clean = [s.replace(" ", "") for s in generated_seqs]
        ref_clean = [s.replace(" ", "") for s in reference_seqs]

        # Sequence similarity metrics
        lev_distances = []
        identities = []
        blosum_scores = []
        blosum_norm_scores = []

        for gen, ref in zip(gen_clean, ref_clean):
            if gen and ref:
                lev_distances.append(levenshtein_distance(gen, ref))
                identities.append(sequence_identity(gen, ref))
                blosum_scores.append(blosum62_similarity(gen, ref))
                blosum_norm_scores.append(blosum62_normalized(gen, ref))

        if lev_distances:
            metrics["levenshtein_mean"] = float(np.mean(lev_distances))
            metrics["levenshtein_std"] = float(np.std(lev_distances))
            metrics["identity_mean"] = float(np.mean(identities))
            metrics["identity_std"] = float(np.std(identities))
            metrics["blosum62_mean"] = float(np.mean(blosum_scores))
            metrics["blosum62_normalized"] = float(np.mean(blosum_norm_scores))

        # Per-chain metrics for Class II
        if self.task == MHCGenerationTask.MHC_CLASS_II:
            chain1_identities = []
            chain2_identities = []

            eos_token = self.tokenizer.eos_token
            for gen, ref in zip(gen_clean, ref_clean):
                if eos_token in gen and eos_token in ref:
                    gen_parts = gen.split(eos_token)
                    ref_parts = ref.split(eos_token)
                    if len(gen_parts) >= 2 and len(ref_parts) >= 2:
                        chain1_identities.append(sequence_identity(gen_parts[0], ref_parts[0]))
                        chain2_identities.append(sequence_identity(gen_parts[1], ref_parts[1]))

            if chain1_identities:
                metrics["chain1_identity_mean"] = float(np.mean(chain1_identities))
                metrics["chain2_identity_mean"] = float(np.mean(chain2_identities))

        return metrics

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        generate_samples: bool = False,
        num_generate: int = 100,
    ) -> Dict[str, Any]:
        """Evaluate model on generation task."""
        model.eval()

        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        generated_seqs = []
        reference_seqs = []

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            encoder_ids = batch["encoder_input_ids"].to(device)
            encoder_mask = batch["encoder_attention_mask"].to(device)
            decoder_ids = batch["decoder_input_ids"].to(device)
            decoder_mask = batch["decoder_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    encoder_input_ids=encoder_ids,
                    encoder_attention_mask=encoder_mask,
                    decoder_input_ids=decoder_ids,
                    decoder_attention_mask=decoder_mask,
                    labels=labels,
                )

            loss = outputs["loss"]
            valid_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
            num_batches += 1

            # Generate samples
            if generate_samples and len(generated_seqs) < num_generate:
                batch_size = encoder_ids.size(0)
                samples_needed = min(batch_size, num_generate - len(generated_seqs))

                gen_output = greedy_decode(
                    model.module if hasattr(model, "module") else model,
                    encoder_ids[:samples_needed],
                    encoder_mask[:samples_needed],
                    bos_id=self.tokenizer.cls_token_id,
                    eos_id=self.tokenizer.eos_token_id,
                    max_length=decoder_ids.size(1),
                )

                for i in range(samples_needed):
                    gen_seq = self.tokenizer.decode(
                        gen_output[i].tolist(),
                        skip_special_tokens=True,
                    )
                    ref_seq = self.tokenizer.decode(
                        labels[i][labels[i] != -100].tolist(),
                        skip_special_tokens=True,
                    )

                    generated_seqs.append(gen_seq)
                    reference_seqs.append(ref_seq)

        # Calculate metrics
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 100))

        metrics = {
            "eval_loss": avg_loss,
            "perplexity": perplexity,
            "num_batches": num_batches,
            "total_tokens": total_tokens,
        }

        if generated_seqs:
            gen_clean = [s.replace(" ", "") for s in generated_seqs]
            ref_clean = [s.replace(" ", "") for s in reference_seqs]

            exact_matches = sum(1 for g, r in zip(gen_clean, ref_clean) if g == r)
            metrics["exact_match_rate"] = exact_matches / len(generated_seqs)
            metrics["num_generated"] = len(generated_seqs)

            # Length statistics
            gen_lengths = [len(s) for s in gen_clean]
            ref_lengths = [len(s) for s in ref_clean]

            metrics["avg_gen_length"] = float(np.mean(gen_lengths))
            metrics["avg_ref_length"] = float(np.mean(ref_lengths))
            metrics["length_ratio"] = float(np.mean(gen_lengths)) / max(float(np.mean(ref_lengths)), 1)

            # Sequence metrics
            detailed_metrics = self._compute_sequence_metrics(generated_seqs, reference_seqs)
            metrics.update(detailed_metrics)

            # Sample outputs
            metrics["sample_generated"] = generated_seqs[:3]
            metrics["sample_reference"] = reference_seqs[:3]

        return metrics


# =============================================================================
# Trainer: PeptideMHCSeq2SeqTrainer
# =============================================================================


class PeptideMHCSeq2SeqTrainer(BaseTCRTrainer):
    """
    Trainer for peptide-to-MHC sequence generation.

    Extends BaseTCRTrainer with MHC-specific functionality:
    - No CDR weighting (MHC has no CDR regions)
    - Support for Class I (single chain) and Class II (two chains)
    - MHC-specific evaluation metrics
    """

    def __init__(self, config: Dict[str, Any], backend: Optional[AcceleratorBackend] = None):
        super().__init__(config, backend)

        # Parse task
        task_str = self.config.get("task", "MHC_CLASS_I")
        self.task = MHCGenerationTask[task_str.upper()]

        # Evaluator will be set up during data setup
        self.evaluator = None

    def _create_model(self) -> nn.Module:
        """Create PeptideMHCSeq2SeqModel with backend-appropriate settings."""
        model_name = self.config.get("model_name", "facebook/esm2_t33_650M_UR50D")

        self._log(f"Loading tokenizer and model: {model_name}")

        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get backend-specific model loading kwargs
        load_kwargs = self._get_model_load_kwargs()

        # Create model
        model = PeptideMHCSeq2SeqModel(
            encoder_model_name=model_name,
            decoder_layers=self.config.get("decoder_layers", 6),
            decoder_heads=self.config.get("decoder_heads", 20),
            decoder_dim=self.config.get("decoder_dim", 1280),
            decoder_ffn_dim=self.config.get("decoder_ffn_dim", 5120),
            dropout=self.config.get("dropout", 0.1),
            decoder_warm_start=self.config.get("decoder_warm_start", False),
            **load_kwargs,
        )

        # Apply LoRA if requested
        if self.config.get("use_lora", False):
            if not PEFT_AVAILABLE:
                self._log("Warning: peft not available, skipping LoRA")
            else:
                model = apply_lora_to_encoder(model, self.config)
                self._log("Applied LoRA to encoder")
                if self._is_main_process():
                    model.encoder.print_trainable_parameters()

        # Load pre-trained encoder checkpoint if provided
        encoder_checkpoint = self.config.get("encoder_checkpoint")
        if encoder_checkpoint:
            self._log(f"Loading encoder checkpoint: {encoder_checkpoint}")
            model = load_encoder_checkpoint(
                model,
                encoder_checkpoint,
                strict=False,
                verbose=self._is_main_process(),
            )

        # Freeze encoder if not using LoRA
        if not self.config.get("use_lora", False) and self.config.get("freeze_encoder", True):
            for param in model.encoder.parameters():
                param.requires_grad = False
            self._log("Encoder frozen (no LoRA)")

        # Enable gradient checkpointing
        if self.config.get("gradient_checkpointing", True):
            if self.backend.name == "xla":
                self._log("Gradient checkpointing disabled (not supported on XLA/Trainium)")
            else:
                model.encoder.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                self._log("Gradient checkpointing enabled")

        return model

    def _create_datasets(self) -> Tuple[Dataset, Dataset]:
        """Create train and validation datasets."""
        train_dataset = PeptideMHCDataset(
            data_path=self.config["data_path"],
            task=self.task,
            split="train",
            local_rank=self.local_rank,
        )

        val_dataset = PeptideMHCDataset(
            data_path=self.config["data_path"],
            task=self.task,
            split="val",
            local_rank=self.local_rank,
        )

        return train_dataset, val_dataset

    def _create_collator(self) -> PeptideMHCCollator:
        """Create MHC data collator."""
        if self.tokenizer is None:
            model_name = self.config.get("model_name", "facebook/esm2_t33_650M_UR50D")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        return PeptideMHCCollator(
            tokenizer=self.tokenizer,
            max_encoder_length=self.config.get("max_encoder_length", 64),
            max_decoder_length=self.config.get("max_decoder_length", 400),
        )

    def _compute_loss(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute standard cross-entropy loss (no CDR weighting for MHC)."""
        outputs = model(
            encoder_input_ids=batch["encoder_input_ids"],
            encoder_attention_mask=batch["encoder_attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=batch["labels"],
        )

        return {
            "loss": outputs["loss"],
        }

    def setup_data(self, include_val: bool = True) -> None:
        """Setup data with MHC-specific evaluator."""
        super().setup_data(include_val)

        if include_val:
            self.evaluator = MHCGenerationEvaluator(
                self.tokenizer,
                task=self.task,
            )

    def overfit_single_batch(self):
        """Overfit check: Train on a single batch to verify the pipeline works."""
        if self.model is None:
            self.setup_model()
        if self.train_loader is None:
            self.setup_data(include_val=False)

        self._log("\n" + "=" * 60)
        self._log("SINGLE BATCH OVERFIT CHECK")
        self._log("=" * 60)
        self._log("Expected: Loss should decrease significantly")
        self._log("=" * 60 + "\n")

        num_steps = self.config.get("overfit_steps", 500)
        lr = self.config.get("overfit_lr") or self.config.get("learning_rate") or 1e-4

        self.setup_optimizer(lr=lr)
        self.model.train()

        # Get a single batch
        batch = next(iter(self.train_loader))
        encoder_ids = batch["encoder_input_ids"].to(self.device)
        encoder_mask = batch["encoder_attention_mask"].to(self.device)
        decoder_ids = batch["decoder_input_ids"].to(self.device)
        decoder_mask = batch["decoder_attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        self._log(f"[Batch Info]")
        self._log(f"  Encoder shape: {encoder_ids.shape}")
        self._log(f"  Decoder shape: {decoder_ids.shape}")
        self._log(f"  Learning rate: {lr}")
        self._log("")

        initial_loss = None

        for step in range(num_steps):
            self.optimizer.zero_grad()

            with self.backend.autocast_context(self.backend.get_model_dtype()):
                outputs = self.model(
                    encoder_input_ids=encoder_ids,
                    encoder_attention_mask=encoder_mask,
                    decoder_input_ids=decoder_ids,
                    decoder_attention_mask=decoder_mask,
                    labels=labels,
                )

            loss = outputs["loss"]
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.get("max_grad_norm", 1.0)
            )

            self.backend.optimizer_step(self.optimizer, self.model)

            if step == 0:
                initial_loss = loss.item()

            if step % 10 == 0 or step == num_steps - 1:
                self._log(
                    f"Step {step:4d}: loss={loss.item():.4f}, "
                    f"ppl={math.exp(min(loss.item(), 100)):.2f}, "
                    f"grad_norm={grad_norm:.4f}"
                )

        final_loss = loss.item()

        self._log("\n" + "=" * 60)
        self._log("OVERFIT CHECK RESULTS")
        self._log("=" * 60)
        self._log(f"Initial loss: {initial_loss:.4f}")
        self._log(f"Final loss:   {final_loss:.4f}")
        self._log(f"Loss reduction: {100 * (initial_loss - final_loss) / initial_loss:.1f}%")
        self._log(f"Initial perplexity: {math.exp(min(initial_loss, 100)):.2f}")
        self._log(f"Final perplexity:   {math.exp(min(final_loss, 100)):.2f}")

        if final_loss < initial_loss * 0.5:
            self._log("\nSUCCESS: Model can overfit a single batch!")
        else:
            self._log("\nWARNING: Loss reduction is limited.")

        self._log("=" * 60 + "\n")

    def _save_checkpoint(self, epoch: int, step: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        if not self._is_main_process():
            return

        model_to_save = self.model.module if self.is_distributed else self.model

        checkpoint = {
            "epoch": epoch,
            "global_step": step,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if hasattr(self, "scheduler") else None,
            "metrics": metrics,
            "config": self.config,
        }

        checkpoint_path = self.output_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        self._log(f"Saved checkpoint: {checkpoint_path}")

        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self._log(f"New best model saved! Loss: {metrics.get('eval_loss', 'N/A'):.4f}")

        # Cleanup old checkpoints
        checkpoints = sorted(self.output_dir.glob("checkpoint_step_*.pt"))
        for old_ckpt in checkpoints[:-3]:
            old_ckpt.unlink()

    def train(self):
        """Full training loop with validation, checkpointing, and early stopping."""
        if self.model is None:
            self.setup_model()
        if self.train_loader is None:
            self.setup_data(include_val=True)

        self._log("\n" + "=" * 60)
        self._log("STARTING PEPTIDE-TO-MHC TRAINING")
        self._log(f"Task: {self.task.value}")
        self._log("=" * 60 + "\n")

        num_epochs = self.config.get("num_epochs", 10)
        grad_accum_steps = self.config.get("gradient_accumulation_steps", 1)
        eval_steps = self.config.get("eval_steps", 1000)
        save_steps = self.config.get("save_steps", 5000)
        log_steps = self.config.get("log_steps", 100)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        patience = self.config.get("patience", 5)

        num_training_steps = len(self.train_loader) * num_epochs // grad_accum_steps
        self.setup_optimizer()
        self.setup_scheduler(num_training_steps)

        self._log(f"Training configuration:")
        self._log(f"  Epochs: {num_epochs}")
        self._log(f"  Batch size: {self.config.get('batch_size', 16)}")
        self._log(f"  Gradient accumulation: {grad_accum_steps}")
        self._log(f"  Effective batch size: {self.config.get('batch_size', 16) * grad_accum_steps * self.world_size}")
        self._log(f"  Total training steps: {num_training_steps}")
        self._log(f"  Early stopping patience: {patience}")
        self._log("")

        patience_counter = 0
        running_loss = 0.0
        running_batches = 0

        # Setup wandb
        use_wandb = self.config.get("report_to") == "wandb" and self._is_main_process() and WANDB_AVAILABLE
        if use_wandb:
            wandb.init(
                project=self.config.get("wandb_project", "peptide-mhc-seq2seq"),
                name=self.config.get("wandb_run_name"),
                config=self.config,
                settings=wandb.Settings(init_timeout=300),
            )

        self.model.train()

        for epoch in range(num_epochs):
            if self.is_distributed:
                self.train_loader.sampler.set_epoch(epoch)

            self._log(f"\n{'='*60}")
            self._log(f"Epoch {epoch + 1}/{num_epochs}")
            self._log(f"{'='*60}")

            epoch_loss = 0.0
            epoch_steps = 0

            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}",
                disable=not self._is_main_process(),
            )

            self.optimizer.zero_grad()

            for batch_idx, batch in enumerate(progress_bar):
                encoder_ids = batch["encoder_input_ids"].to(self.device)
                encoder_mask = batch["encoder_attention_mask"].to(self.device)
                decoder_ids = batch["decoder_input_ids"].to(self.device)
                decoder_mask = batch["decoder_attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                with self.backend.autocast_context(self.backend.get_model_dtype()):
                    outputs = self.model(
                        encoder_input_ids=encoder_ids,
                        encoder_attention_mask=encoder_mask,
                        decoder_input_ids=decoder_ids,
                        decoder_attention_mask=decoder_mask,
                        labels=labels,
                    )
                    loss = outputs["loss"] / grad_accum_steps

                loss.backward()

                epoch_loss += outputs["loss"].item()
                epoch_steps += 1

                if (batch_idx + 1) % grad_accum_steps == 0:
                    running_loss += outputs["loss"].item()
                    running_batches += 1

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=max_grad_norm
                    )

                    self.backend.optimizer_step(self.optimizer, self.model)
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Logging
                    if self.global_step % log_steps == 0 and self._is_main_process():
                        avg_loss = running_loss / running_batches
                        lr = self.scheduler.get_last_lr()[0]

                        self._log(
                            f"Step {self.global_step}: loss={avg_loss:.4f}, "
                            f"ppl={math.exp(min(avg_loss, 100)):.2f}, "
                            f"lr={lr:.2e}, grad_norm={grad_norm:.4f}"
                        )

                        if use_wandb:
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/perplexity": math.exp(min(avg_loss, 100)),
                                "train/learning_rate": lr,
                                "train/grad_norm": grad_norm,
                                "train/global_step": self.global_step,
                            })

                        running_loss = 0.0
                        running_batches = 0

                    # Evaluation
                    if self.global_step % eval_steps == 0:
                        self._log(f"\n--- Validation at step {self.global_step} ---")
                        val_metrics = self.evaluator.evaluate(
                            self.model,
                            self.val_loader,
                            self.device,
                            generate_samples=True,
                            num_generate=self.config.get("num_generate", 100),
                        )

                        self._log(
                            f"Val: loss={val_metrics['eval_loss']:.4f}, "
                            f"ppl={val_metrics['perplexity']:.2f}"
                        )

                        if "exact_match_rate" in val_metrics:
                            self._log(f"     exact_match={val_metrics['exact_match_rate']:.4f}")

                        if "identity_mean" in val_metrics:
                            self._log(
                                f"     identity={val_metrics['identity_mean']:.4f} (+/-{val_metrics.get('identity_std', 0):.3f})"
                            )

                        # Per-chain metrics for Class II
                        if "chain1_identity_mean" in val_metrics:
                            self._log(
                                f"     chain1_id={val_metrics['chain1_identity_mean']:.4f}, "
                                f"chain2_id={val_metrics['chain2_identity_mean']:.4f}"
                            )

                        if "sample_generated" in val_metrics:
                            self._log("\n[Sample outputs]")
                            for i, (gen, ref) in enumerate(zip(
                                val_metrics["sample_generated"][:2],
                                val_metrics["sample_reference"][:2]
                            )):
                                self._log(f"  [{i}] Gen: {gen[:80]}...")
                                self._log(f"      Ref: {ref[:80]}...")

                        if use_wandb:
                            wandb.log({
                                f"eval/{k}": v
                                for k, v in val_metrics.items()
                                if isinstance(v, (int, float))
                            } | {"eval/global_step": self.global_step})

                        # Check for improvement
                        is_best = val_metrics["eval_loss"] < self.best_val_loss
                        if is_best:
                            self.best_val_loss = val_metrics["eval_loss"]
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            self._log(f"No improvement. Patience: {patience_counter}/{patience}")

                        self._save_checkpoint(epoch, self.global_step, val_metrics, is_best=is_best)

                        if patience_counter >= patience:
                            self._log(f"\nEarly stopping triggered at step {self.global_step}")
                            break

                        self.model.train()

                    # Periodic checkpoint
                    if self.global_step % save_steps == 0:
                        self._save_checkpoint(
                            epoch, self.global_step,
                            {"train_loss": epoch_loss / epoch_steps}
                        )

                progress_bar.set_postfix({
                    "loss": f"{epoch_loss/max(epoch_steps, 1):.4f}",
                    "step": f"{self.global_step}",
                })

            if patience_counter >= patience:
                break

            self._log(f"\n--- End of Epoch {epoch + 1} ---")
            self._log(f"Train: loss={epoch_loss/epoch_steps:.4f}, ppl={math.exp(min(epoch_loss/epoch_steps, 100)):.2f}")

        # Final summary
        self._log("\n" + "=" * 60)
        self._log("TRAINING COMPLETE")
        self._log("=" * 60)
        self._log(f"Best validation loss: {self.best_val_loss:.4f}")
        self._log(f"Best validation perplexity: {math.exp(min(self.best_val_loss, 100)):.2f}")
        self._log(f"Total steps: {self.global_step}")
        self._log(f"Best model saved to: {self.output_dir / 'best_model.pt'}")

        if use_wandb:
            wandb.finish()

        self._save_checkpoint(
            num_epochs - 1, self.global_step,
            {"final": True, "best_val_loss": self.best_val_loss}
        )


# =============================================================================
# CLI Arguments
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Peptide-to-MHC Conditional Sequence Generation Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Backend
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "cuda", "xla", "neuron", "trainium"],
                        help="Training backend")

    # Data
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to directory containing parquet files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for logs and checkpoints")
    parser.add_argument("--task", type=str, default="MHC_CLASS_I",
                        choices=["MHC_CLASS_I", "MHC_CLASS_II"],
                        help="MHC generation task")
    parser.add_argument("--max_encoder_length", type=int, default=64,
                        help="Maximum encoder sequence length (peptides are short)")
    parser.add_argument("--max_decoder_length", type=int, default=400,
                        help="Maximum decoder sequence length (400 for Class I, 550 for Class II)")

    # Model
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D",
                        help="Pretrained ESM2 model name")
    parser.add_argument("--decoder_layers", type=int, default=6,
                        help="Number of decoder transformer layers")
    parser.add_argument("--decoder_heads", type=int, default=20,
                        help="Number of decoder attention heads")
    parser.add_argument("--decoder_dim", type=int, default=1280,
                        help="Decoder hidden dimension (must match ESM2)")
    parser.add_argument("--decoder_ffn_dim", type=int, default=5120,
                        help="Decoder FFN intermediate dimension")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--decoder_warm_start", action="store_true",
                        help="Initialize decoder from encoder weights")

    # LoRA
    parser.add_argument("--use_lora", action="store_true",
                        help="Apply LoRA to encoder")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--freeze_encoder", action="store_true", default=True,
                        help="Freeze encoder if not using LoRA")

    # Custom encoder checkpoint
    parser.add_argument("--encoder_checkpoint", type=str, default=None,
                        help="Path to pre-trained encoder checkpoint for SFT")

    # Training
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of data loader workers")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing")

    # Logging and checkpointing
    parser.add_argument("--log_steps", type=int, default=100,
                        help="Log every N steps")
    parser.add_argument("--eval_steps", type=int, default=1000,
                        help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=5000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")

    # Wandb
    parser.add_argument("--report_to", type=str, default="none",
                        choices=["wandb", "none"],
                        help="Reporting destination")
    parser.add_argument("--wandb_project", type=str, default="peptide-mhc-seq2seq",
                        help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Wandb run name")

    # Overfit check
    parser.add_argument("--overfit_check", action="store_true",
                        help="Run single-batch overfit check")
    parser.add_argument("--overfit_steps", type=int, default=500,
                        help="Number of steps for overfit check")
    parser.add_argument("--overfit_lr", type=float, default=None,
                        help="Learning rate for overfit check")

    # Evaluation
    parser.add_argument("--num_generate", type=int, default=100,
                        help="Number of samples to generate for evaluation")

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()
    config = vars(args)

    # Get backend
    backend = get_backend(config.get("backend", "auto"))

    # Create trainer
    trainer = PeptideMHCSeq2SeqTrainer(config, backend=backend)

    if args.overfit_check:
        trainer.overfit_single_batch()
    else:
        trainer.train()

    # Cleanup distributed
    if trainer.is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
