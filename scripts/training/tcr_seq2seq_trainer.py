#!/usr/bin/env python3
"""
TCR-Peptide-MHC Conditional Sequence Generation Trainer

Encoder-decoder architecture for conditional TCR sequence generation.
Uses ESM2 as encoder with a TransformerDecoder for autoregressive generation.

Architecture:
    [Encoder: ESM2]                              [Decoder: torch.nn.TransformerDecoder]
    [Beta <eos> Peptide <eos> MHC <eos>]  -->  Cross-Attention  -->  Alpha chain
             (context)                           + Causal LM            (target)

Features:
- ESM2 encoder with flash attention 2, frozen or with LoRA
- TransformerDecoder with cross-attention to encoder outputs
- Weight tying: decoder embeddings shared with ESM2 encoder embeddings
- Autoregressive next-token prediction with cross-entropy loss
- Support for ALPHA, BETA, and PEPTIDE generation tasks
- DDP for multi-GPU training
- Mixed precision (bfloat16)

Usage:
    python scripts/training/tcr_seq2seq_trainer.py \
        --data_path data/deduplicated/full/foundation_permutations/ \
        --output_dir ./output/tcr_seq2seq \
        --task ALPHA \
        --permutation_keys tra_trb_peptide_mhc_one tra_trb_peptide_mhc_one_mhc_two \
        --batch_size 16 \
        --gradient_accumulation_steps 8 \
        --num_epochs 10 \
        --use_lora \
        --overfit_check
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


class GenerationTask(Enum):
    """Task types for conditional sequence generation."""
    ALPHA = "alpha"    # Generate alpha chain from beta + peptide + MHC
    BETA = "beta"      # Generate beta chain from alpha + peptide + MHC
    PEPTIDE = "peptide"  # Generate peptide from alpha + beta + MHC


# Fields present in parquet files
FIELDS = ["tra", "trb", "peptide", "mhc_one", "mhc_two"]


# =============================================================================
# Permutation Key Utilities
# =============================================================================


def permutation_key_has_fields(pkey: str, required_fields: List[str], any_of: List[str] = None) -> bool:
    """
    Check if a permutation key contains required fields.

    Args:
        pkey: Permutation key string (e.g., 'tra_trb_peptide_mhc_one')
        required_fields: List of field names that must ALL be present
        any_of: List of field names where at least ONE must be present (optional)

    Returns:
        True if all required fields are present (and at least one of any_of if specified)
    """
    pkey_lower = pkey.lower()
    words = pkey_lower.replace('_', ' ').split()

    # Check all required fields are present
    for field in required_fields:
        if field in ('tra', 'trb'):
            # Exact word match to avoid 'tra' matching 'trav'
            if field not in words:
                return False
        else:
            # Substring match for peptide, mhc_one, mhc_two
            if field not in pkey_lower:
                return False

    # Check at least one of any_of is present
    if any_of:
        found_any = False
        for field in any_of:
            if field in ('tra', 'trb'):
                if field in words:
                    found_any = True
                    break
            elif field in pkey_lower:
                found_any = True
                break
        if not found_any:
            return False

    return True


def get_required_fields_for_task(task: "GenerationTask") -> Tuple[List[str], List[str]]:
    """
    Get required fields for a generation task.

    Args:
        task: GenerationTask enum value

    Returns:
        Tuple of (required_fields, any_of_fields)
    """
    if task == GenerationTask.ALPHA:
        # ALPHA: need tra (target) + peptide + mhc_one; trb optional
        return ['tra', 'peptide', 'mhc_one'], []
    elif task == GenerationTask.BETA:
        # BETA: need trb (target) + peptide + mhc_one; tra optional
        return ['trb', 'peptide', 'mhc_one'], []
    elif task == GenerationTask.PEPTIDE:
        # PEPTIDE: need peptide (target) + mhc_one + at least one TCR
        return ['peptide', 'mhc_one'], ['tra', 'trb']
    else:
        return ['tra', 'trb', 'peptide', 'mhc_one'], []


def discover_permutation_keys(data_path: str, task: "GenerationTask", is_main: bool = True) -> List[str]:
    """
    Discover all permutation keys from parquet files and filter by task requirements.

    Args:
        data_path: Path to directory containing parquet files
        task: Generation task (determines required fields)
        is_main: Whether this is the main process (for logging)

    Returns:
        List of matching permutation keys
    """
    import pyarrow.parquet as pq

    # Get required fields for this task
    required_fields, any_of_fields = get_required_fields_for_task(task)

    # Scan parquet files to find unique permutation keys
    parquet_files = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_path}")

    all_keys = set()
    for pf in parquet_files[:10]:  # Sample first 10 files for speed
        try:
            table = pq.read_table(pf, columns=['permutation_key'])
            if 'permutation_key' in table.column_names:
                keys = table['permutation_key'].to_pylist()
                all_keys.update(k for k in keys if k)
        except Exception:
            continue

    # Filter keys by required fields
    matching_keys = [
        key for key in all_keys
        if permutation_key_has_fields(key, required_fields, any_of_fields)
    ]

    if is_main:
        print(f"Discovered {len(all_keys)} unique permutation keys")
        print(f"Selected {len(matching_keys)} keys for {task.value} task")
        if matching_keys:
            print(f"  Examples: {matching_keys[:5]}")

    if not matching_keys:
        raise ValueError(
            f"No permutation keys found matching {task.value} requirements. "
            f"Need keys containing: {required_fields}" +
            (f" and at least one of: {any_of_fields}" if any_of_fields else "")
        )

    return sorted(matching_keys)


# =============================================================================
# Positional Encoding
# =============================================================================


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding as in the original Transformer paper.
    """
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
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model) with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len].unsqueeze(0).to(x.dtype)
        return self.dropout(x)


# =============================================================================
# Custom Decoder Layer with Self-Attention Dropout (Phase 8: Cross-Attention Forcing)
# =============================================================================


class SelfAttnDropoutDecoderLayer(nn.Module):
    """
    Custom TransformerDecoderLayer with optional self-attention dropout.

    During training, randomly skips self-attention to force the model to rely
    on cross-attention for information from the encoder. This addresses mode
    collapse where the decoder ignores encoder context.

    Uses Pre-LN (norm_first=True) architecture for stability.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        batch_first: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )

        # Cross-attention
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer norms (Pre-LN style)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Activation
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        self_attn_drop_prob: float = 0.0,
    ) -> torch.Tensor:
        """
        Forward pass with optional self-attention dropout.

        Args:
            tgt: Target sequence (batch, seq_len, d_model)
            memory: Encoder output (batch, enc_len, d_model)
            tgt_mask: Causal mask for self-attention
            memory_mask: Mask for cross-attention (usually None)
            tgt_key_padding_mask: Padding mask for target
            memory_key_padding_mask: Padding mask for encoder output
            self_attn_drop_prob: Probability of skipping self-attention (0-1)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Pre-LN Self-Attention (with optional dropout)
        if self.training and self_attn_drop_prob > 0 and torch.rand(1).item() < self_attn_drop_prob:
            # Skip self-attention entirely - use residual only
            # This forces the model to rely on cross-attention for context
            x = tgt
        else:
            # Normal self-attention path
            x2 = self.norm1(tgt)
            x2, _ = self.self_attn(
                x2, x2, x2,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                need_weights=False,
            )
            x = tgt + self.dropout1(x2)

        # Pre-LN Cross-Attention (always active - this is what we want to force)
        x2 = self.norm2(x)
        x2, _ = self.multihead_attn(
            x2, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout2(x2)

        # Pre-LN FFN
        x2 = self.norm3(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x2))))
        x = x + self.dropout3(x2)

        return x


class SelfAttnDropoutDecoder(nn.Module):
    """
    Transformer decoder with self-attention dropout support.

    Wraps multiple SelfAttnDropoutDecoderLayer modules.
    """

    def __init__(self, decoder_layer: SelfAttnDropoutDecoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            copy.deepcopy(decoder_layer) for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        self_attn_drop_prob: float = 0.0,
    ) -> torch.Tensor:
        """Forward through all decoder layers."""
        output = tgt
        for layer in self.layers:
            output = layer(
                output, memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                self_attn_drop_prob=self_attn_drop_prob,
            )
        return output


# =============================================================================
# Model: TCRSeq2SeqModel
# =============================================================================


class TCRSeq2SeqModel(nn.Module):
    """
    Encoder-decoder model for conditional TCR sequence generation.

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
        # Self-attention dropout schedule for cross-attention forcing (Phase 8)
        self_attn_drop_initial: float = 0.0,
        self_attn_drop_final: float = 0.0,
        self_attn_drop_anneal_epochs: int = 3,
    ):
        super().__init__()

        # Store self-attention dropout schedule parameters
        self.self_attn_drop_initial = self_attn_drop_initial
        self.self_attn_drop_final = self_attn_drop_final
        self.self_attn_drop_anneal_epochs = self_attn_drop_anneal_epochs
        self.use_custom_decoder = self_attn_drop_initial > 0 or self_attn_drop_final > 0
        self.current_epoch = 0  # Updated by trainer

        # Load ESM2 encoder with appropriate attention implementation
        # attn_implementation: "flash_attention_2" for CUDA, "eager" or None for XLA/Trainium
        encoder_kwargs = {"torch_dtype": torch_dtype}

        # Try flash_attention_2 first, fall back to eager if not available
        if attn_implementation == "auto":
            try:
                import flash_attn
                encoder_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                encoder_kwargs["attn_implementation"] = "eager"
        elif attn_implementation:
            encoder_kwargs["attn_implementation"] = attn_implementation

        self.encoder = AutoModel.from_pretrained(
            encoder_model_name,
            **encoder_kwargs,
        )

        # Get encoder config
        self.encoder_dim = self.encoder.config.hidden_size
        self.vocab_size = self.encoder.config.vocab_size

        # Verify dimensions match
        assert decoder_dim == self.encoder_dim, (
            f"decoder_dim ({decoder_dim}) must match encoder hidden size ({self.encoder_dim})"
        )

        # Share encoder embeddings with decoder (critical for convergence!)
        # This ties the decoder input space to ESM2's learned token representations
        self.decoder_embed = self.encoder.embeddings.word_embeddings

        # Positional encoding for decoder
        self.decoder_pos_encoding = PositionalEncoding(
            d_model=decoder_dim,
            max_len=512,  # Max decoder sequence length
            dropout=dropout
        )

        # Transformer decoder layers
        # Use custom decoder with self-attention dropout if enabled (Phase 8)
        if self.use_custom_decoder:
            print(f"[Cross-Attn Forcing] Using custom decoder with self-attention dropout")
            print(f"  Initial dropout: {self_attn_drop_initial}, Final: {self_attn_drop_final}")
            print(f"  Anneal epochs: {self_attn_drop_anneal_epochs}")
            decoder_layer = SelfAttnDropoutDecoderLayer(
                d_model=decoder_dim,
                nhead=decoder_heads,
                dim_feedforward=decoder_ffn_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.decoder = SelfAttnDropoutDecoder(
                decoder_layer,
                num_layers=decoder_layers,
            )
        else:
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
        # This anchors output predictions in ESM2's learned token space
        self.output_proj = nn.Linear(decoder_dim, self.vocab_size, bias=False)
        self.output_proj.weight = self.decoder_embed.weight  # Weight tying

        # Layer norm before output projection
        self.output_norm = nn.LayerNorm(decoder_dim)

        # Convert decoder components to same dtype as encoder for consistency
        # (encoder is loaded with torch_dtype, decoder is created in float32 by default)
        self.decoder_pos_encoding = self.decoder_pos_encoding.to(torch_dtype)
        self.decoder = self.decoder.to(torch_dtype)
        self.output_norm = self.output_norm.to(torch_dtype)

        # Initialize decoder from encoder weights if requested (warm-start)
        if decoder_warm_start:
            self._initialize_decoder_from_encoder()

    def _initialize_decoder_from_encoder(self) -> None:
        """
        Initialize decoder transformer layers from pre-trained ESM2 encoder.

        This addresses mode collapse by giving the decoder a head start with
        weights that already understand the encoder's representation space.

        Copies:
        - Self-attention Q, K, V weights (concatenated for PyTorch format)
        - Self-attention output projection
        - FFN layers (linear1, linear2)
        - Layer norms (attention and FFN)

        Keeps random:
        - Cross-attention weights (multihead_attn) - no encoder equivalent
        - Cross-attention layer norm (norm2)
        """
        # Get number of layers to copy (min of encoder and decoder layers)
        num_encoder_layers = len(self.encoder.encoder.layer)
        num_decoder_layers = len(self.decoder.layers)
        num_layers_to_copy = min(num_encoder_layers, num_decoder_layers)

        print(f"[Warm-Start] Initializing {num_layers_to_copy} decoder layers from ESM2 encoder...")

        for i in range(num_layers_to_copy):
            enc_layer = self.encoder.encoder.layer[i]
            dec_layer = self.decoder.layers[i]

            # ================================================
            # 1. Self-Attention Q, K, V -> in_proj_weight
            # ================================================
            # ESM2 has separate Q, K, V weights
            # PyTorch decoder expects concatenated [Q; K; V]
            q_weight = enc_layer.attention.self.query.weight.data  # (1280, 1280)
            k_weight = enc_layer.attention.self.key.weight.data
            v_weight = enc_layer.attention.self.value.weight.data

            # Concatenate: in_proj_weight is (3*embed_dim, embed_dim)
            in_proj_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            dec_layer.self_attn.in_proj_weight.data.copy_(in_proj_weight)

            # Biases (if present)
            if enc_layer.attention.self.query.bias is not None:
                q_bias = enc_layer.attention.self.query.bias.data
                k_bias = enc_layer.attention.self.key.bias.data
                v_bias = enc_layer.attention.self.value.bias.data
                in_proj_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                dec_layer.self_attn.in_proj_bias.data.copy_(in_proj_bias)

            # ================================================
            # 2. Self-Attention Output Projection
            # ================================================
            dec_layer.self_attn.out_proj.weight.data.copy_(
                enc_layer.attention.output.dense.weight.data
            )
            if enc_layer.attention.output.dense.bias is not None:
                dec_layer.self_attn.out_proj.bias.data.copy_(
                    enc_layer.attention.output.dense.bias.data
                )

            # ================================================
            # 3. FFN Layers
            # ================================================
            # ESM2: intermediate.dense -> linear1, output.dense -> linear2
            dec_layer.linear1.weight.data.copy_(enc_layer.intermediate.dense.weight.data)
            dec_layer.linear2.weight.data.copy_(enc_layer.output.dense.weight.data)

            if enc_layer.intermediate.dense.bias is not None:
                dec_layer.linear1.bias.data.copy_(enc_layer.intermediate.dense.bias.data)
            if enc_layer.output.dense.bias is not None:
                dec_layer.linear2.bias.data.copy_(enc_layer.output.dense.bias.data)

            # ================================================
            # 4. Layer Norms
            # ================================================
            # norm1 = self-attention LN (maps to ESM2 attention.LayerNorm)
            # norm2 = cross-attention LN (keep random - no encoder equivalent)
            # norm3 = FFN LN (maps to ESM2 LayerNorm)

            dec_layer.norm1.weight.data.copy_(enc_layer.attention.LayerNorm.weight.data)
            dec_layer.norm1.bias.data.copy_(enc_layer.attention.LayerNorm.bias.data)

            dec_layer.norm3.weight.data.copy_(enc_layer.LayerNorm.weight.data)
            dec_layer.norm3.bias.data.copy_(enc_layer.LayerNorm.bias.data)

            # NOTE: norm2 (cross-attention) and multihead_attn stay randomly initialized
            # This is intentional - cross-attention has no encoder equivalent

        print(f"[Warm-Start] Complete. Copied self-attention, FFN, and layer norms.")
        print(f"[Warm-Start] Cross-attention layers remain randomly initialized.")

    def get_self_attn_drop_prob(self) -> float:
        """
        Get self-attention dropout probability based on current epoch (annealed schedule).

        Schedule:
        - Epochs 0 to (anneal_epochs - 1): Use initial_prob (high dropout)
        - Epochs >= anneal_epochs: Use final_prob (low dropout)

        This forces the model to rely on cross-attention early in training,
        then allows self-attention to refine syntax later.
        """
        if not self.use_custom_decoder:
            return 0.0

        if self.current_epoch < self.self_attn_drop_anneal_epochs:
            return self.self_attn_drop_initial
        else:
            return self.self_attn_drop_final

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for dropout schedule."""
        self.current_epoch = epoch
        if self.use_custom_decoder:
            prob = self.get_self_attn_drop_prob()
            print(f"[Cross-Attn Forcing] Epoch {epoch}: self-attn dropout = {prob:.2f}")

    def _generate_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Generate causal attention mask for decoder.

        XLA/Trainium Fix: Use additive float mask with -1e4 instead of -inf or bool.
        Using -inf causes NaN gradients on Trainium due to bfloat16 handling.

        Returns:
            Float mask of shape (seq_len, seq_len) where 0.0 = attend, -1e4 = block
        """
        # Create upper triangle with large negative number (block attention)
        # Use -1e4 because it zeros out softmax in bfloat16 but avoids NaN gradients
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
        """
        Encode context sequences with ESM2.

        Args:
            encoder_input_ids: (batch, enc_len) tokenized context
            encoder_attention_mask: (batch, enc_len) attention mask

        Returns:
            encoder_hidden: (batch, enc_len, hidden_dim) encoder outputs
        """
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
        """
        Decode target sequence with cross-attention to encoder.

        Args:
            decoder_input_ids: (batch, dec_len) tokenized target input
            decoder_attention_mask: (batch, dec_len) decoder attention mask
            encoder_hidden: (batch, enc_len, hidden_dim) encoder outputs
            encoder_attention_mask: (batch, enc_len) encoder attention mask

        Returns:
            logits: (batch, dec_len, vocab_size) output logits
        """
        batch_size, dec_len = decoder_input_ids.shape
        device = decoder_input_ids.device

        # Embed decoder inputs (shared with encoder)
        decoder_emb = self.decoder_embed(decoder_input_ids)  # (batch, dec_len, dim)
        decoder_emb = self.decoder_pos_encoding(decoder_emb)

        # Get dtype for masks (XLA/Trainium compatibility)
        dtype = decoder_emb.dtype

        # Causal mask for autoregressive decoding (additive float mask)
        causal_mask = self._generate_causal_mask(dec_len, device, dtype=dtype)

        # XLA/Trainium Fix: Convert boolean padding masks to additive float masks
        # Using -1e4 instead of -inf to avoid NaN gradients in bfloat16
        # HF attention_mask: 1 = attend, 0 = ignore
        # Additive mask: 0.0 = attend, -1e4 = ignore
        tgt_key_padding_mask = (1.0 - decoder_attention_mask.to(dtype)) * -10000.0
        memory_key_padding_mask = (1.0 - encoder_attention_mask.to(dtype)) * -10000.0

        # Run decoder with cross-attention
        if self.use_custom_decoder:
            # Custom decoder with self-attention dropout
            decoder_output = self.decoder(
                tgt=decoder_emb,
                memory=encoder_hidden,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                self_attn_drop_prob=self.get_self_attn_drop_prob(),
            )
        else:
            # Standard PyTorch decoder
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
        """
        Full forward pass for training.

        Args:
            encoder_input_ids: (batch, enc_len) context tokens
            encoder_attention_mask: (batch, enc_len) context attention mask
            decoder_input_ids: (batch, dec_len) target input tokens (teacher forcing)
            decoder_attention_mask: (batch, dec_len) target attention mask
            labels: (batch, dec_len) target labels for loss computation

        Returns:
            dict with logits, and optionally loss if labels provided
        """
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
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss

        return output


def apply_lora_to_encoder(model: TCRSeq2SeqModel, config: dict) -> TCRSeq2SeqModel:
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


# =============================================================================
# Dataset: TCRSeq2SeqDataset
# =============================================================================


class TCRSeq2SeqDataset(Dataset):
    """
    Dataset for TCR conditional sequence generation.

    Loads parquet files with individual columns (tra, trb, peptide, mhc_one, mhc_two)
    and prepares encoder context / decoder target pairs based on the task.

    Args:
        data_path: Path to directory containing parquet files
        permutation_keys: List of permutation keys to filter sequences.
            If None, auto-discovers keys matching task requirements.
        task: Generation task (ALPHA, BETA, or PEPTIDE)
        split: One of 'train', 'val', or 'test'
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        seed: Random seed for reproducible splits
        local_rank: Local rank for DDP
    """

    # Class-level cache for loaded data
    _cache: Dict[str, List[Dict[str, str]]] = {}

    def __init__(
        self,
        data_path: str,
        permutation_keys: Optional[List[str]],
        task: GenerationTask,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        local_rank: int = 0,
    ):
        self.task = task
        self.split = split
        is_main = local_rank == 0

        # Auto-discover permutation keys if not provided
        if permutation_keys is None or permutation_keys == ["auto"]:
            permutation_keys = discover_permutation_keys(data_path, task, is_main)

        # Create cache key
        cache_key = f"{data_path}:{','.join(sorted(permutation_keys))}"

        if cache_key in TCRSeq2SeqDataset._cache:
            if is_main:
                print(f"Using cached data for {split} split...")
            all_data = TCRSeq2SeqDataset._cache[cache_key]
        else:
            import pyarrow.parquet as pq

            parquet_files = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
            if not parquet_files:
                raise ValueError(f"No parquet files found in {data_path}")

            if is_main:
                print(f"Loading {len(parquet_files)} parquet files...")

            all_data = []

            # Determine required columns based on permutation keys
            required_cols = ["permutation_key", "sequence"]
            # Also try to load individual columns if available
            optional_cols = ["tra", "trb", "peptide", "mhc_one", "mhc_two"]

            iterator = tqdm(parquet_files, desc="Loading", disable=not is_main)
            for pf in iterator:
                try:
                    # Try to read with individual columns first
                    table = pq.read_table(pf)
                    df = table.to_pandas()

                    # Filter by permutation keys
                    if "permutation_key" in df.columns:
                        df = df[df["permutation_key"].isin(permutation_keys)]

                    # Check if we have individual columns or need to parse sequence
                    has_individual_cols = all(col in df.columns for col in ["tra", "trb", "peptide", "mhc_one"])

                    for _, row in df.iterrows():
                        if has_individual_cols:
                            # Use individual columns directly
                            record = {
                                "tra": row.get("tra", ""),
                                "trb": row.get("trb", ""),
                                "peptide": row.get("peptide", ""),
                                "mhc_one": row.get("mhc_one", ""),
                                "mhc_two": row.get("mhc_two", ""),
                            }
                        else:
                            # Parse from concatenated sequence column
                            # Format: "TRA TRB PEPTIDE MHC_ONE MHC_TWO" (space-separated)
                            seq = row.get("sequence", "")
                            perm_key = row.get("permutation_key", "")
                            record = self._parse_sequence(seq, perm_key)

                        # Only keep records with required fields for this task
                        if self._has_required_fields(record):
                            all_data.append(record)

                except Exception as e:
                    if is_main:
                        print(f"Warning: Error reading {pf}: {e}")
                    continue

            if not all_data:
                raise ValueError(f"No valid sequences found with permutation_keys: {permutation_keys}")

            if is_main:
                print(f"Loaded {len(all_data):,} records")

            TCRSeq2SeqDataset._cache[cache_key] = all_data

        # Create deterministic train/val/test split
        n_total = len(all_data)
        np.random.seed(seed)
        indices = np.random.permutation(n_total)

        train_end = int(n_total * train_ratio)
        val_end = int(n_total * (train_ratio + val_ratio))

        if split == "train":
            selected_indices = indices[:train_end]
        elif split == "val":
            selected_indices = indices[train_end:val_end]
        elif split == "test":
            selected_indices = indices[val_end:]
        else:
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split}")

        self.data = [all_data[i] for i in selected_indices]

        if is_main:
            print(f"{split.capitalize()} set: {len(self.data):,} examples")

    def _parse_sequence(self, sequence: str, perm_key: str) -> Dict[str, str]:
        """
        Parse concatenated sequence based on permutation key.

        Args:
            sequence: Space-separated concatenated sequence
            perm_key: Permutation key indicating which fields are present

        Returns:
            Dict mapping field names to sequences
        """
        parts = sequence.split(" ")
        record = {"tra": "", "trb": "", "peptide": "", "mhc_one": "", "mhc_two": ""}

        # Find position of each field in the permutation key to determine order
        field_positions = []
        for field in FIELDS:
            pos = perm_key.find(field)
            if pos != -1:
                field_positions.append((pos, field))

        # Sort by position in key string to get actual order
        field_positions.sort()
        fields_in_order = [field for _, field in field_positions]

        # Assign parts to fields in correct order
        for i, field in enumerate(fields_in_order):
            if i < len(parts):
                record[field] = parts[i]

        return record

    def _has_required_fields(self, record: Dict[str, str]) -> bool:
        """
        Check if record has required fields for the generation task.

        Relaxed validation:
        - ALPHA: tra (target) + peptide + mhc_one required; trb optional
        - BETA: trb (target) + peptide + mhc_one required; tra optional
        - PEPTIDE: peptide (target) + mhc_one + at least one TCR required

        Args:
            record: Dict with tra, trb, peptide, mhc_one, mhc_two

        Returns:
            True if record has required fields for the task
        """
        # Helper to check if a field is non-empty and valid
        def is_valid(val: str) -> bool:
            return bool(val) and val not in ("", "NA", "None")

        # Common requirement: peptide and mhc_one
        if not is_valid(record.get("peptide", "")) or not is_valid(record.get("mhc_one", "")):
            return False

        if self.task == GenerationTask.ALPHA:
            # Need tra (target); trb optional for context
            return is_valid(record.get("tra", ""))

        elif self.task == GenerationTask.BETA:
            # Need trb (target); tra optional for context
            return is_valid(record.get("trb", ""))

        elif self.task == GenerationTask.PEPTIDE:
            # Need at least one TCR chain for context
            return is_valid(record.get("tra", "")) or is_valid(record.get("trb", ""))

        return False

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get encoder context sequences and decoder target.

        Handles optional TCR chains - only includes present chains in encoder context.

        Returns:
            dict with:
                - encoder_context: List[str] of context sequences
                - decoder_target: str target sequence
                - mhc_class: 'I' or 'II' based on mhc_two presence
        """
        record = self.data[idx]

        # Determine MHC class
        mhc_class = "II" if record.get("mhc_two", "") else "I"

        if self.task == GenerationTask.ALPHA:
            # Encoder context: [trb if present] + peptide + mhc
            encoder_context = []
            if record.get("trb"):
                encoder_context.append(record["trb"])
            encoder_context.extend([record["peptide"], record["mhc_one"]])
            if mhc_class == "II" and record.get("mhc_two"):
                encoder_context.append(record["mhc_two"])
            decoder_target = record["tra"]

        elif self.task == GenerationTask.BETA:
            # Encoder context: [tra if present] + peptide + mhc
            encoder_context = []
            if record.get("tra"):
                encoder_context.append(record["tra"])
            encoder_context.extend([record["peptide"], record["mhc_one"]])
            if mhc_class == "II" and record.get("mhc_two"):
                encoder_context.append(record["mhc_two"])
            decoder_target = record["trb"]

        elif self.task == GenerationTask.PEPTIDE:
            # Encoder context: available TCRs + mhc
            encoder_context = []
            if record.get("tra"):
                encoder_context.append(record["tra"])
            if record.get("trb"):
                encoder_context.append(record["trb"])
            encoder_context.append(record["mhc_one"])
            if mhc_class == "II" and record.get("mhc_two"):
                encoder_context.append(record["mhc_two"])
            decoder_target = record["peptide"]

        return {
            "encoder_context": encoder_context,
            "decoder_target": decoder_target,
            "mhc_class": mhc_class,
        }


# =============================================================================
# Data Collator: Seq2SeqCollator
# =============================================================================


class Seq2SeqCollator:
    """
    Collates TCR sequences for seq2seq training.

    Handles:
    - Joining encoder context sequences with <eos> token between entities
    - Creating decoder inputs (BOS + target) for teacher forcing
    - Creating labels (target + EOS) with padding set to -100

    Args:
        tokenizer: ESM2 tokenizer
        max_encoder_length: Maximum encoder sequence length
        max_decoder_length: Maximum decoder sequence length
    """

    def __init__(
        self,
        tokenizer,
        max_encoder_length: int = 1024,
        max_decoder_length: int = 350,
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
        """
        Collate batch of examples.

        Args:
            examples: List of dicts with encoder_context and decoder_target

        Returns:
            dict with:
                - encoder_input_ids: (batch, enc_len)
                - encoder_attention_mask: (batch, enc_len)
                - decoder_input_ids: (batch, dec_len)
                - decoder_attention_mask: (batch, dec_len)
                - labels: (batch, dec_len)
        """
        batch_size = len(examples)

        # Process encoder inputs: join context sequences with EOS separator
        encoder_seqs = []
        for ex in examples:
            # Join with EOS token between sequences
            joined = f"{self.tokenizer.eos_token}".join(ex["encoder_context"])
            encoder_seqs.append(joined)

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

            # Tokenize target (without special tokens, we add them manually)
            target_encoded = self.tokenizer(
                target,
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
                labels = labels + [-100] * pad_len  # -100 for ignored positions
                attention_mask = [1] * dec_len + [0] * pad_len
            else:
                # Truncate if needed
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
    model: TCRSeq2SeqModel,
    encoder_input_ids: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_length: int = 350,
    pad_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Greedy decoding for sequence generation.

    Args:
        model: TCRSeq2SeqModel
        encoder_input_ids: (batch, enc_len) encoder input
        encoder_attention_mask: (batch, enc_len) encoder mask
        bos_id: Beginning of sequence token ID
        eos_id: End of sequence token ID
        max_length: Maximum generation length
        pad_id: Padding token ID (defaults to eos_id)

    Returns:
        generated: (batch, gen_len) generated token IDs
    """
    model.eval()
    device = encoder_input_ids.device
    batch_size = encoder_input_ids.size(0)
    pad_id = pad_id if pad_id is not None else eos_id

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

        # Get next token (greedy: argmax)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

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


@torch.no_grad()
def beam_search(
    model: TCRSeq2SeqModel,
    encoder_input_ids: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    beam_width: int = 5,
    max_length: int = 350,
    length_penalty: float = 1.0,
    pad_id: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Beam search decoding for sequence generation.

    Args:
        model: TCRSeq2SeqModel
        encoder_input_ids: (1, enc_len) single encoder input
        encoder_attention_mask: (1, enc_len) encoder mask
        bos_id: Beginning of sequence token ID
        eos_id: End of sequence token ID
        beam_width: Number of beams
        max_length: Maximum generation length
        length_penalty: Length penalty for beam scoring
        pad_id: Padding token ID

    Returns:
        best_sequence: (1, gen_len) best generated sequence
        best_score: (1,) score of best sequence
    """
    model.eval()
    device = encoder_input_ids.device
    pad_id = pad_id if pad_id is not None else eos_id

    # Only support batch_size=1 for beam search
    assert encoder_input_ids.size(0) == 1, "Beam search only supports batch_size=1"

    # Encode context
    encoder_hidden = model.encode(encoder_input_ids, encoder_attention_mask)

    # Expand encoder outputs for beam search
    encoder_hidden = encoder_hidden.expand(beam_width, -1, -1)
    encoder_attention_mask = encoder_attention_mask.expand(beam_width, -1)

    # Initialize beams: (beam_width, 1)
    beams = torch.full((beam_width, 1), bos_id, dtype=torch.long, device=device)
    beam_scores = torch.zeros(beam_width, device=device)
    beam_scores[1:] = -float("inf")  # Only first beam is active initially

    finished_beams = []
    finished_scores = []

    for step in range(max_length - 1):
        # Create attention mask
        decoder_mask = torch.ones_like(beams)

        # Get logits
        logits = model.decode(
            beams,
            decoder_mask,
            encoder_hidden,
            encoder_attention_mask,
        )

        # Get log probabilities for next token
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)  # (beam_width, vocab)

        # Calculate scores for all possible next tokens
        vocab_size = log_probs.size(-1)
        next_scores = beam_scores.unsqueeze(-1) + log_probs  # (beam_width, vocab)

        # Flatten and get top-k
        next_scores = next_scores.view(-1)  # (beam_width * vocab)
        top_scores, top_indices = next_scores.topk(beam_width * 2, dim=0)

        # Convert flat indices to beam and token indices
        beam_indices = top_indices // vocab_size
        token_indices = top_indices % vocab_size

        # Build new beams
        new_beams = []
        new_scores = []

        for score, beam_idx, token_idx in zip(top_scores, beam_indices, token_indices):
            if len(new_beams) >= beam_width:
                break

            beam_idx = beam_idx.item()
            token_idx = token_idx.item()

            # Create new beam
            new_beam = torch.cat([
                beams[beam_idx],
                torch.tensor([token_idx], device=device)
            ])

            if token_idx == eos_id:
                # Finished beam - apply length penalty
                length = new_beam.size(0)
                final_score = score / (length ** length_penalty)
                finished_beams.append(new_beam)
                finished_scores.append(final_score.item())
            else:
                new_beams.append(new_beam)
                new_scores.append(score)

        if not new_beams:
            break

        # Pad beams to same length
        max_len = max(b.size(0) for b in new_beams)
        padded_beams = []
        for b in new_beams:
            if b.size(0) < max_len:
                padding = torch.full(
                    (max_len - b.size(0),), pad_id, dtype=torch.long, device=device
                )
                b = torch.cat([b, padding])
            padded_beams.append(b)

        beams = torch.stack(padded_beams[:beam_width])
        beam_scores = torch.tensor(new_scores[:beam_width], device=device)

    # Add remaining beams to finished
    for i, (beam, score) in enumerate(zip(beams, beam_scores)):
        length = beam.size(0)
        final_score = score / (length ** length_penalty)
        finished_beams.append(beam)
        finished_scores.append(final_score.item())

    # Return best beam
    if finished_beams:
        best_idx = np.argmax(finished_scores)
        return finished_beams[best_idx].unsqueeze(0), torch.tensor([finished_scores[best_idx]])
    else:
        return beams[0].unsqueeze(0), beam_scores[0].unsqueeze(0)


@torch.no_grad()
def sample_with_temperature(
    model: TCRSeq2SeqModel,
    encoder_input_ids: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    max_length: int = 350,
    pad_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Temperature sampling for sequence generation.

    Args:
        model: TCRSeq2SeqModel
        encoder_input_ids: (batch, enc_len) encoder input
        encoder_attention_mask: (batch, enc_len) encoder mask
        bos_id: Beginning of sequence token ID
        eos_id: End of sequence token ID
        temperature: Sampling temperature (higher = more random)
        top_k: If > 0, only sample from top-k tokens
        top_p: If < 1.0, use nucleus sampling
        max_length: Maximum generation length
        pad_id: Padding token ID

    Returns:
        generated: (batch, gen_len) generated token IDs
    """
    model.eval()
    device = encoder_input_ids.device
    batch_size = encoder_input_ids.size(0)
    pad_id = pad_id if pad_id is not None else eos_id

    # Encode context
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

        # Apply temperature
        logits = logits[:, -1, :] / temperature

        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        # Sample next token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Replace finished sequences' tokens with pad
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
# Biological Constants for Sequence Metrics
# =============================================================================

# BLOSUM62 substitution matrix (symmetric, only store upper triangle + diagonal)
# Source: Henikoff & Henikoff (1992)
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

# Kyte-Doolittle hydrophobicity scale
# Positive = hydrophobic, Negative = hydrophilic
KYTE_DOOLITTLE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}

# Amino acid charge at physiological pH (~7.4)
AA_CHARGE = {
    'D': -1.0, 'E': -1.0,  # Acidic (negative)
    'K': 1.0, 'R': 1.0, 'H': 0.1,  # Basic (positive, H partial)
    'A': 0.0, 'N': 0.0, 'C': 0.0, 'Q': 0.0, 'G': 0.0,
    'I': 0.0, 'L': 0.0, 'M': 0.0, 'F': 0.0, 'P': 0.0,
    'S': 0.0, 'T': 0.0, 'W': 0.0, 'Y': 0.0, 'V': 0.0,
}

# Standard amino acids
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


# =============================================================================
# Sequence Similarity Metrics
# =============================================================================


def levenshtein_distance(seq1: str, seq2: str) -> int:
    """
    Compute Levenshtein (edit) distance between two sequences.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Edit distance (insertions, deletions, substitutions)
    """
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
    """
    Compute sequence identity (fraction of matching positions).

    Uses global alignment where shorter sequence is compared position-by-position.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Identity fraction (0.0 to 1.0)
    """
    if not seq1 or not seq2:
        return 0.0

    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))

    matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])

    # Normalize by max length to penalize length differences
    return matches / max_len


def blosum62_similarity(seq1: str, seq2: str) -> float:
    """
    Compute BLOSUM62 similarity score between two sequences.

    Compares aligned positions using BLOSUM62 substitution scores.
    Unaligned positions (length difference) are penalized with gap penalty.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Total BLOSUM62 score (can be negative)
    """
    if not seq1 or not seq2:
        return 0.0

    min_len = min(len(seq1), len(seq2))
    gap_penalty = -4  # Standard BLOSUM62 gap penalty

    score = 0.0
    for i in range(min_len):
        aa1 = seq1[i].upper()
        aa2 = seq2[i].upper()
        if aa1 in BLOSUM62 and aa2 in BLOSUM62[aa1]:
            score += BLOSUM62[aa1][aa2]
        else:
            # Unknown amino acid, use gap penalty
            score += gap_penalty

    # Penalize length differences
    len_diff = abs(len(seq1) - len(seq2))
    score += len_diff * gap_penalty

    return score


def blosum62_normalized(seq1: str, seq2: str) -> float:
    """
    Compute length-normalized BLOSUM62 similarity.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Normalized score (score per position)
    """
    if not seq1 or not seq2:
        return 0.0

    score = blosum62_similarity(seq1, seq2)
    max_len = max(len(seq1), len(seq2))
    return score / max_len


# =============================================================================
# Amino Acid Composition Metrics
# =============================================================================


def compute_hydrophobicity_profile(seq: str) -> np.ndarray:
    """
    Compute per-position hydrophobicity using Kyte-Doolittle scale.

    Args:
        seq: Amino acid sequence

    Returns:
        Array of hydrophobicity values per position
    """
    profile = []
    for aa in seq.upper():
        profile.append(KYTE_DOOLITTLE.get(aa, 0.0))
    return np.array(profile)


def hydrophobicity_correlation(seq1: str, seq2: str) -> float:
    """
    Compute Pearson correlation between hydrophobicity profiles.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Correlation coefficient (-1 to 1), or 0 if cannot compute
    """
    if not seq1 or not seq2:
        return 0.0

    # Truncate to same length for comparison
    min_len = min(len(seq1), len(seq2))
    if min_len < 3:
        return 0.0

    profile1 = compute_hydrophobicity_profile(seq1[:min_len])
    profile2 = compute_hydrophobicity_profile(seq2[:min_len])

    # Compute Pearson correlation
    if np.std(profile1) < 1e-8 or np.std(profile2) < 1e-8:
        return 0.0

    corr = np.corrcoef(profile1, profile2)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def compute_net_charge(seq: str) -> float:
    """
    Compute net charge of a sequence at physiological pH.

    Args:
        seq: Amino acid sequence

    Returns:
        Net charge
    """
    return sum(AA_CHARGE.get(aa.upper(), 0.0) for aa in seq)


def aa_frequency_distribution(seq: str) -> Dict[str, float]:
    """
    Compute amino acid frequency distribution.

    Args:
        seq: Amino acid sequence

    Returns:
        Dict mapping amino acid to frequency (0-1)
    """
    if not seq:
        return {aa: 0.0 for aa in AMINO_ACIDS}

    counts = {aa: 0 for aa in AMINO_ACIDS}
    total = 0
    for aa in seq.upper():
        if aa in counts:
            counts[aa] += 1
            total += 1

    if total == 0:
        return {aa: 0.0 for aa in AMINO_ACIDS}

    return {aa: count / total for aa, count in counts.items()}


def jensen_shannon_divergence(dist1: Dict[str, float], dist2: Dict[str, float]) -> float:
    """
    Compute Jensen-Shannon divergence between two distributions.

    Args:
        dist1: First distribution (dict of probabilities)
        dist2: Second distribution (dict of probabilities)

    Returns:
        JS divergence (0 = identical, 1 = maximally different)
    """
    # Get all keys
    keys = set(dist1.keys()) | set(dist2.keys())

    p = np.array([dist1.get(k, 0.0) for k in keys])
    q = np.array([dist2.get(k, 0.0) for k in keys])

    # Normalize
    p = p / (p.sum() + 1e-10)
    q = q / (q.sum() + 1e-10)

    # Average distribution
    m = 0.5 * (p + q)

    # KL divergences
    def kl_div(a, b):
        mask = (a > 0) & (b > 0)
        return np.sum(a[mask] * np.log(a[mask] / b[mask]))

    js = 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)
    return float(js)


# =============================================================================
# Position-Wise Analysis
# =============================================================================


class PositionWiseAnalyzer:
    """
    Analyze per-position accuracy and confusion patterns.

    Tracks:
    - Accuracy at each sequence position
    - CDR3 conserved position accuracy (first 3 and last 2 positions)
    - Most common substitution errors
    """

    def __init__(self):
        self.position_correct: Dict[int, int] = {}
        self.position_total: Dict[int, int] = {}
        self.confusion: Dict[Tuple[str, str], int] = {}  # (ref, gen) -> count
        self.total_samples = 0

    def update(self, generated: str, reference: str) -> None:
        """
        Update statistics with a new generated/reference pair.

        Args:
            generated: Generated sequence
            reference: Reference sequence
        """
        self.total_samples += 1
        min_len = min(len(generated), len(reference))

        for i in range(min_len):
            ref_aa = reference[i].upper()
            gen_aa = generated[i].upper()

            # Update position accuracy
            if i not in self.position_total:
                self.position_total[i] = 0
                self.position_correct[i] = 0

            self.position_total[i] += 1
            if ref_aa == gen_aa:
                self.position_correct[i] += 1
            else:
                # Track confusion
                key = (ref_aa, gen_aa)
                self.confusion[key] = self.confusion.get(key, 0) + 1

    def get_position_accuracy(self) -> Dict[int, float]:
        """
        Get accuracy at each position.

        Returns:
            Dict mapping position to accuracy (0-1)
        """
        return {
            pos: self.position_correct[pos] / self.position_total[pos]
            for pos in sorted(self.position_total.keys())
            if self.position_total[pos] > 0
        }

    def get_mean_position_accuracy(self) -> float:
        """Get mean accuracy across all positions."""
        accuracies = self.get_position_accuracy()
        if not accuracies:
            return 0.0
        return float(np.mean(list(accuracies.values())))

    def get_cdr3_conserved_accuracy(self) -> Dict[str, float]:
        """
        Get accuracy for CDR3 conserved positions.

        CDR3 typically has conserved residues at:
        - First 3 positions (often Cys at position 0)
        - Last 2 positions (often Phe/Trp at -2, Gly at -1)

        Returns:
            Dict with 'first3' and 'last2' accuracy
        """
        pos_acc = self.get_position_accuracy()

        # First 3 positions
        first3_acc = []
        for i in range(3):
            if i in pos_acc:
                first3_acc.append(pos_acc[i])

        # Last 2 positions (need to find max position)
        if pos_acc:
            max_pos = max(pos_acc.keys())
            last2_acc = []
            for i in range(max(0, max_pos - 1), max_pos + 1):
                if i in pos_acc:
                    last2_acc.append(pos_acc[i])
        else:
            last2_acc = []

        return {
            'first3': float(np.mean(first3_acc)) if first3_acc else 0.0,
            'last2': float(np.mean(last2_acc)) if last2_acc else 0.0,
        }

    def get_top_substitution_errors(self, top_k: int = 10) -> List[Tuple[str, str, int]]:
        """
        Get most common substitution errors.

        Args:
            top_k: Number of top errors to return

        Returns:
            List of (reference_aa, generated_aa, count) tuples
        """
        sorted_errors = sorted(
            self.confusion.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [(ref, gen, count) for (ref, gen), count in sorted_errors[:top_k]]

    def reset(self) -> None:
        """Reset all accumulators."""
        self.position_correct.clear()
        self.position_total.clear()
        self.confusion.clear()
        self.total_samples = 0


# =============================================================================
# CDR Region Weighting (ANARCI-based)
# =============================================================================


# Preset configurations for CDR region weighting
CDR_WEIGHT_PRESETS = {
    "uniform": {"cdr1": 1.0, "cdr2": 1.0, "cdr3": 1.0, "framework": 1.0},
    "cdr3_focused": {"cdr1": 2.0, "cdr2": 2.0, "cdr3": 4.0, "framework": 1.0},
    "all_cdr_equal": {"cdr1": 3.0, "cdr2": 3.0, "cdr3": 3.0, "framework": 1.0},
    "extreme_cdr3": {"cdr1": 1.5, "cdr2": 1.5, "cdr3": 8.0, "framework": 0.5},
}


# Optional: ANARCI for CDR annotation
try:
    from anarci import anarci as run_anarci
    ANARCI_AVAILABLE = True
except ImportError:
    ANARCI_AVAILABLE = False


def get_cdr_positions_anarci(sequence: str, chain_type: str = "B") -> Optional[Dict[str, Tuple[int, int]]]:
    """
    Get CDR region positions using ANARCI IMGT numbering.

    Args:
        sequence: Full TCR sequence
        chain_type: 'A' for alpha, 'B' for beta

    Returns:
        Dict mapping region names to (start, end) positions, or None if annotation fails
    """
    if not ANARCI_AVAILABLE:
        return None

    try:
        # Run ANARCI with IMGT scheme
        results = run_anarci([("seq", sequence)], scheme="imgt", allowed_species=["human"])

        if results[0][0] is None:
            return None

        # Extract numbering
        numbering = results[0][0][0][0]  # First hit, first domain

        # IMGT CDR definitions for TCRs:
        # CDR1: positions 27-38 (IMGT)
        # CDR2: positions 56-65 (IMGT)
        # CDR3: positions 105-117 (IMGT) + insertions

        cdr_positions = {
            'cdr1': None,
            'cdr2': None,
            'cdr3': None,
        }

        # Map IMGT positions to sequence positions
        seq_pos = 0
        cdr1_start, cdr1_end = None, None
        cdr2_start, cdr2_end = None, None
        cdr3_start, cdr3_end = None, None

        for (imgt_pos, insertion), aa in numbering:
            if aa == '-':
                continue

            # Track CDR1 (IMGT 27-38)
            if 27 <= imgt_pos <= 38:
                if cdr1_start is None:
                    cdr1_start = seq_pos
                cdr1_end = seq_pos + 1

            # Track CDR2 (IMGT 56-65)
            elif 56 <= imgt_pos <= 65:
                if cdr2_start is None:
                    cdr2_start = seq_pos
                cdr2_end = seq_pos + 1

            # Track CDR3 (IMGT 105-117 + insertions)
            elif imgt_pos >= 105 and imgt_pos <= 117:
                if cdr3_start is None:
                    cdr3_start = seq_pos
                cdr3_end = seq_pos + 1
            elif insertion and cdr3_start is not None:
                # CDR3 insertions
                cdr3_end = seq_pos + 1

            seq_pos += 1

        if cdr1_start is not None:
            cdr_positions['cdr1'] = (cdr1_start, cdr1_end)
        if cdr2_start is not None:
            cdr_positions['cdr2'] = (cdr2_start, cdr2_end)
        if cdr3_start is not None:
            cdr_positions['cdr3'] = (cdr3_start, cdr3_end)

        return cdr_positions

    except Exception:
        return None


class CDRAnnotationCache:
    """
    Thread-safe cache for CDR annotations to avoid re-running ANARCI.
    """

    def __init__(self, max_size: int = 10000):
        self._cache: Dict[str, Optional[Dict[str, Tuple[int, int]]]] = {}
        self._max_size = max_size
        import threading
        self._lock = threading.Lock()

    def get_or_compute(self, sequence: str, chain_type: str = "B") -> Optional[Dict[str, Tuple[int, int]]]:
        """Get CDR positions from cache or compute if not cached."""
        cache_key = f"{chain_type}:{sequence[:50]}"  # Use prefix for key

        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Compute outside lock
        positions = get_cdr_positions_anarci(sequence, chain_type)

        with self._lock:
            if len(self._cache) < self._max_size:
                self._cache[cache_key] = positions

        return positions


# Global CDR annotation cache
_cdr_cache = CDRAnnotationCache()


def build_cdr_position_weights(
    seq_len: int,
    cdr_annotations: Optional[Dict[str, Tuple[int, int]]],
    cdr1_weight: float = 2.0,
    cdr2_weight: float = 2.0,
    cdr3_weight: float = 4.0,
    framework_weight: float = 1.0,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Build per-position weights based on CDR annotations.

    Args:
        seq_len: Sequence length
        cdr_annotations: Dict with 'cdr1', 'cdr2', 'cdr3' -> (start, end) tuples
        cdr1_weight: Weight for CDR1 region
        cdr2_weight: Weight for CDR2 region
        cdr3_weight: Weight for CDR3 region (most important!)
        framework_weight: Weight for framework regions
        device: Target device

    Returns:
        Tensor of shape (seq_len,) with per-position weights
    """
    weights = torch.full((seq_len,), framework_weight, device=device)

    if cdr_annotations is None:
        return weights

    # Apply CDR-specific weights
    for region, weight in [('cdr1', cdr1_weight), ('cdr2', cdr2_weight), ('cdr3', cdr3_weight)]:
        if region in cdr_annotations and cdr_annotations[region] is not None:
            start, end = cdr_annotations[region]
            if start is not None and end is not None and start < seq_len and end <= seq_len:
                weights[start:end] = weight

    return weights


# =============================================================================
# BLOSUM62 Soft Loss
# =============================================================================


class BLOSUM62SoftLoss(nn.Module):
    """
    Soft cross-entropy loss weighted by BLOSUM62 biological similarity.

    Instead of treating all errors equally, penalize biologically dissimilar
    substitutions more:
    - Predicting L when target is I (similar hydrophobic) -> small penalty
    - Predicting K when target is D (opposite charge) -> large penalty

    Args:
        tokenizer: ESM2 tokenizer
        temperature: Temperature for softmax normalization of BLOSUM scores
        alpha: Mixing weight - (1-alpha)*CE + alpha*BLOSUM_KL
    """

    def __init__(self, tokenizer, temperature: float = 1.0, alpha: float = 0.1):
        super().__init__()
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.alpha = alpha

        # Build BLOSUM62 soft target matrix
        self.register_buffer("blosum_targets", self._build_blosum_soft_targets())

    def _build_blosum_soft_targets(self) -> torch.Tensor:
        """Convert BLOSUM62 to soft probability targets."""
        vocab_size = self.tokenizer.vocab_size

        # Initialize with uniform (for non-AA tokens)
        soft_targets = torch.zeros(vocab_size, vocab_size)

        # Map amino acids to token IDs
        aa_to_id = {}
        for aa in AMINO_ACIDS:
            tokens = self.tokenizer.encode(aa, add_special_tokens=False)
            if tokens:
                aa_to_id[aa] = tokens[0]

        # Fill in BLOSUM62 scores for AA pairs
        for aa1, aa1_id in aa_to_id.items():
            for aa2, aa2_id in aa_to_id.items():
                score = BLOSUM62.get(aa1, {}).get(aa2, -4)
                soft_targets[aa1_id, aa2_id] = score

        # Normalize rows to sum to 1 (softmax over BLOSUM scores)
        # Only for rows corresponding to amino acids
        for aa, aa_id in aa_to_id.items():
            row = soft_targets[aa_id]
            # Apply softmax only to AA positions
            aa_indices = list(aa_to_id.values())
            aa_scores = row[aa_indices]
            aa_probs = F.softmax(aa_scores / self.temperature, dim=0)
            for i, idx in enumerate(aa_indices):
                soft_targets[aa_id, idx] = aa_probs[i]

        return soft_targets

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute BLOSUM-weighted soft loss.

        Args:
            logits: (batch, seq_len, vocab_size)
            labels: (batch, seq_len) hard labels

        Returns:
            Combined loss scalar
        """
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device

        # Standard cross-entropy (hard targets)
        ce_loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100,
            reduction='none'
        )

        # BLOSUM soft targets
        valid_mask = (labels != -100).float()
        valid_labels = labels.clone()
        valid_labels[~(labels != -100)] = 0

        # Look up soft targets for each label
        soft_targets = self.blosum_targets.to(device)[valid_labels]  # (batch, seq_len, vocab_size)

        # KL divergence from predicted distribution to BLOSUM soft targets
        log_probs = F.log_softmax(logits, dim=-1)
        kl_loss = F.kl_div(
            log_probs.view(-1, vocab_size),
            soft_targets.view(-1, vocab_size),
            reduction='none'
        ).sum(dim=-1)

        # Mask invalid positions
        flat_mask = valid_mask.view(-1)
        kl_loss = kl_loss * flat_mask
        ce_loss = ce_loss * flat_mask

        # Combine losses
        n_valid = flat_mask.sum().clamp(min=1)
        total_loss = (1 - self.alpha) * (ce_loss.sum() / n_valid) + self.alpha * (kl_loss.sum() / n_valid)

        return total_loss


# =============================================================================
# Sequence Alignment Utilities
# =============================================================================


# Optional: Biopython for alignment
try:
    from Bio import pairwise2
    from Bio.Align import substitution_matrices
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False


def aligned_sequence_metrics(generated: str, reference: str) -> Dict[str, float]:
    """
    Compute metrics on properly aligned sequences using Biopython.

    Uses Needleman-Wunsch global alignment with BLOSUM62 scoring.

    Args:
        generated: Generated sequence
        reference: Reference sequence

    Returns:
        Dict with alignment-based metrics
    """
    if not BIOPYTHON_AVAILABLE:
        # Fallback to simple position-wise comparison
        return {
            "identity_aligned": sequence_identity(generated, reference),
            "blosum_score": blosum62_similarity(generated, reference),
            "alignment_available": False,
        }

    if not generated or not reference:
        return {
            "identity_aligned": 0.0,
            "blosum_score": 0.0,
            "alignment_available": True,
        }

    try:
        # Load BLOSUM62 for alignment scoring
        blosum62_matrix = substitution_matrices.load("BLOSUM62")

        # Global alignment (Needleman-Wunsch style)
        alignments = pairwise2.align.globalds(
            generated, reference,
            blosum62_matrix,
            -10,   # Gap open penalty
            -0.5,  # Gap extend penalty
        )

        if not alignments:
            return {
                "identity_aligned": 0.0,
                "blosum_score": 0.0,
                "alignment_available": True,
            }

        best_alignment = alignments[0]
        aligned_gen, aligned_ref, score, begin, end = best_alignment

        # Compute identity on aligned sequences
        matches = sum(1 for a, b in zip(aligned_gen, aligned_ref)
                      if a == b and a != '-')
        non_gap_positions = sum(1 for a, b in zip(aligned_gen, aligned_ref)
                                 if a != '-' or b != '-')
        identity = matches / max(non_gap_positions, 1)

        # Gap statistics
        gen_gaps = aligned_gen.count('-')
        ref_gaps = aligned_ref.count('-')

        return {
            "identity_aligned": identity,
            "blosum_score": score,
            "blosum_per_position": score / max(len(aligned_gen), 1),
            "alignment_length": len(aligned_gen),
            "gen_gaps": gen_gaps,
            "ref_gaps": ref_gaps,
            "alignment_available": True,
        }

    except Exception:
        return {
            "identity_aligned": sequence_identity(generated, reference),
            "blosum_score": blosum62_similarity(generated, reference),
            "alignment_available": False,
        }


# =============================================================================
# Evaluator: GenerationEvaluator
# =============================================================================


class GenerationEvaluator:
    """
    Evaluator for sequence generation with comprehensive biological metrics.

    Metrics:
    - Cross-entropy loss (teacher-forced)
    - Perplexity
    - Exact match rate
    - Length statistics
    - Sequence similarity (Levenshtein, identity, BLOSUM62)
    - Position-specific accuracy (CDR3 conserved positions)
    - Amino acid composition (hydrophobicity, charge, distribution)
    """

    def __init__(self, tokenizer, compute_detailed_metrics: bool = True):
        """
        Initialize the evaluator.

        Args:
            tokenizer: ESM2 tokenizer
            compute_detailed_metrics: Whether to compute advanced biological metrics
        """
        self.tokenizer = tokenizer
        self.compute_detailed_metrics = compute_detailed_metrics
        self.position_analyzer = PositionWiseAnalyzer() if compute_detailed_metrics else None

    def _compute_sequence_metrics(
        self,
        generated_seqs: List[str],
        reference_seqs: List[str],
    ) -> Dict[str, Any]:
        """
        Compute comprehensive sequence similarity and composition metrics.

        Args:
            generated_seqs: List of generated sequences
            reference_seqs: List of reference sequences

        Returns:
            Dict with sequence similarity, position, and composition metrics
        """
        metrics = {}

        # Clean sequences (remove spaces from tokenizer output)
        gen_clean = [s.replace(" ", "") for s in generated_seqs]
        ref_clean = [s.replace(" ", "") for s in reference_seqs]

        # ===== Sequence Similarity Metrics =====
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
            metrics["blosum62_std"] = float(np.std(blosum_scores))
            metrics["blosum62_normalized"] = float(np.mean(blosum_norm_scores))

        # ===== Position-Specific Accuracy =====
        if self.position_analyzer is not None:
            self.position_analyzer.reset()
            for gen, ref in zip(gen_clean, ref_clean):
                if gen and ref:
                    self.position_analyzer.update(gen, ref)

            metrics["mean_position_accuracy"] = self.position_analyzer.get_mean_position_accuracy()

            cdr3_acc = self.position_analyzer.get_cdr3_conserved_accuracy()
            metrics["cdr3_first3_accuracy"] = cdr3_acc["first3"]
            metrics["cdr3_last2_accuracy"] = cdr3_acc["last2"]

            # Top substitution errors
            top_errors = self.position_analyzer.get_top_substitution_errors(top_k=5)
            metrics["top_substitution_errors"] = top_errors

        # ===== Amino Acid Composition Metrics =====
        hydro_corrs = []
        charge_errors = []
        js_divergences = []

        for gen, ref in zip(gen_clean, ref_clean):
            if gen and ref and len(gen) >= 3 and len(ref) >= 3:
                # Hydrophobicity correlation
                hydro_corrs.append(hydrophobicity_correlation(gen, ref))

                # Charge error
                gen_charge = compute_net_charge(gen)
                ref_charge = compute_net_charge(ref)
                charge_errors.append(abs(gen_charge - ref_charge))

                # AA distribution divergence
                gen_dist = aa_frequency_distribution(gen)
                ref_dist = aa_frequency_distribution(ref)
                js_divergences.append(jensen_shannon_divergence(gen_dist, ref_dist))

        if hydro_corrs:
            metrics["hydrophobicity_corr_mean"] = float(np.mean(hydro_corrs))
            metrics["hydrophobicity_corr_std"] = float(np.std(hydro_corrs))
            metrics["charge_mae"] = float(np.mean(charge_errors))
            metrics["aa_distribution_js"] = float(np.mean(js_divergences))

        return metrics

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        generate_samples: bool = False,
        num_generate: int = 100,
        compute_detailed_metrics: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate model on generation task with comprehensive metrics.

        Args:
            model: TCRSeq2SeqModel
            dataloader: DataLoader for evaluation
            device: Device to run on
            generate_samples: Whether to generate samples for quality metrics
            num_generate: Number of samples to generate (default 100 for good statistics)
            compute_detailed_metrics: Whether to compute advanced biological metrics

        Returns:
            dict with all metrics
        """
        model.eval()

        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        # For generation metrics
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

            # Count non-padding tokens
            valid_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
            num_batches += 1

            # Generate samples for first num_generate batches
            if generate_samples and len(generated_seqs) < num_generate:
                # Generate for entire batch to be more efficient
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

                # Decode sequences
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

        # Calculate base metrics
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 100))  # Cap to avoid overflow

        metrics = {
            "eval_loss": avg_loss,
            "perplexity": perplexity,
            "num_batches": num_batches,
            "total_tokens": total_tokens,
        }

        # Calculate generation metrics if samples were generated
        if generated_seqs:
            # Clean sequences for comparison
            gen_clean = [s.replace(" ", "") for s in generated_seqs]
            ref_clean = [s.replace(" ", "") for s in reference_seqs]

            exact_matches = sum(
                1 for g, r in zip(gen_clean, ref_clean)
                if g == r
            )
            metrics["exact_match_rate"] = exact_matches / len(generated_seqs)
            metrics["num_generated"] = len(generated_seqs)

            # Length statistics
            gen_lengths = [len(s) for s in gen_clean]
            ref_lengths = [len(s) for s in ref_clean]

            metrics["avg_gen_length"] = float(np.mean(gen_lengths))
            metrics["avg_ref_length"] = float(np.mean(ref_lengths))
            metrics["length_ratio"] = float(np.mean(gen_lengths)) / max(float(np.mean(ref_lengths)), 1)

            # Compute detailed biological metrics
            if compute_detailed_metrics and self.compute_detailed_metrics:
                detailed_metrics = self._compute_sequence_metrics(
                    generated_seqs, reference_seqs
                )
                metrics.update(detailed_metrics)

            # Sample outputs for logging (first 3)
            metrics["sample_generated"] = generated_seqs[:3]
            metrics["sample_reference"] = reference_seqs[:3]

        return metrics


# =============================================================================
# Trainer: TCRSeq2SeqTrainer
# =============================================================================


class TCRSeq2SeqTrainer(BaseTCRTrainer):
    """
    Trainer for TCR conditional sequence generation.

    Extends BaseTCRTrainer with seq2seq-specific functionality:
    - Encoder-decoder architecture with ESM2 encoder
    - Cross-attention based generation
    - Comprehensive biological metrics for evaluation
    - Support for CUDA and Trainium backends

    Features:
    - DDP/XLA support for multi-device training
    - Mixed precision (bfloat16)
    - LoRA for encoder (optional)
    - Gradient checkpointing
    - Warmup + cosine LR schedule
    - Early stopping
    - Checkpointing with best model tracking
    - Overfit check mode
    """

    def __init__(self, config: Dict[str, Any], backend: Optional[AcceleratorBackend] = None):
        """
        Initialize seq2seq trainer.

        Args:
            config: Training configuration
            backend: Accelerator backend (auto-detected if None)
        """
        # Initialize base trainer (handles device setup, logging, etc.)
        super().__init__(config, backend)

        # Seq2seq specific: parse task
        task_str = self.config.get("task", "ALPHA")
        self.task = GenerationTask[task_str.upper()]

        # Evaluator will be set up during data setup
        self.evaluator = None

        # BLOSUM loss function (initialized after tokenizer is set up)
        self.blosum_loss_fn = None

        # CDR annotation cache
        self.cdr_cache = CDRAnnotationCache()

    def _create_model(self) -> nn.Module:
        """Create TCRSeq2SeqModel with backend-appropriate settings."""
        model_name = self.config.get("model_name", "facebook/esm2_t33_650M_UR50D")

        self._log(f"Loading tokenizer and model: {model_name}")

        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get backend-specific model loading kwargs
        load_kwargs = self._get_model_load_kwargs()

        # Create model with backend-appropriate attention implementation
        model = TCRSeq2SeqModel(
            encoder_model_name=model_name,
            decoder_layers=self.config.get("decoder_layers", 6),
            decoder_heads=self.config.get("decoder_heads", 20),
            decoder_dim=self.config.get("decoder_dim", 1280),
            decoder_ffn_dim=self.config.get("decoder_ffn_dim", 5120),
            dropout=self.config.get("dropout", 0.1),
            decoder_warm_start=self.config.get("decoder_warm_start", False),
            self_attn_drop_initial=self.config.get("self_attn_drop_initial", 0.0),
            self_attn_drop_final=self.config.get("self_attn_drop_final", 0.0),
            self_attn_drop_anneal_epochs=self.config.get("self_attn_drop_anneal_epochs", 3),
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

        # Freeze encoder if not using LoRA
        if not self.config.get("use_lora", False) and self.config.get("freeze_encoder", True):
            for param in model.encoder.parameters():
                param.requires_grad = False
            self._log("Encoder frozen (no LoRA)")

        # Enable gradient checkpointing for encoder (not supported on XLA/Trainium)
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
        # Get permutation keys from config (can be None for auto-discovery)
        permutation_keys = self.config.get("permutation_keys")
        if isinstance(permutation_keys, str):
            permutation_keys = [permutation_keys]
        # Handle empty list from nargs="*" as None for auto-discovery
        if permutation_keys is not None and len(permutation_keys) == 0:
            permutation_keys = None

        train_dataset = TCRSeq2SeqDataset(
            data_path=self.config["data_path"],
            permutation_keys=permutation_keys,
            task=self.task,
            split="train",
            local_rank=self.local_rank,
        )

        val_dataset = TCRSeq2SeqDataset(
            data_path=self.config["data_path"],
            permutation_keys=permutation_keys,
            task=self.task,
            split="val",
            local_rank=self.local_rank,
        )

        return train_dataset, val_dataset

    def _create_collator(self) -> Seq2SeqCollator:
        """Create seq2seq data collator."""
        # Ensure tokenizer is set up
        if self.tokenizer is None:
            model_name = self.config.get("model_name", "facebook/esm2_t33_650M_UR50D")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        return Seq2SeqCollator(
            tokenizer=self.tokenizer,
            max_encoder_length=self.config.get("max_encoder_length", 1024),
            max_decoder_length=self.config.get("max_decoder_length", 350),
        )

    def _compute_loss(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute biologically-informed loss for seq2seq training.

        Loss components (all configurable):
        1. CDR-weighted CE: Weight CDR3 > CDR1/2 > Framework via ANARCI positions
        2. BLOSUM62 soft loss: Penalize biologically dissimilar substitutions more
        3. Auxiliary losses: Identity, length consistency

        For PEPTIDE task, CDR weighting is disabled (peptides don't have CDR regions).
        """
        outputs = model(
            encoder_input_ids=batch["encoder_input_ids"],
            encoder_attention_mask=batch["encoder_attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=batch["labels"],
        )

        logits = outputs["logits"]  # (batch, seq_len, vocab_size)
        labels = batch["labels"]     # (batch, seq_len)
        batch_size, seq_len, vocab_size = logits.shape

        # =====================================================
        # COMPONENT 1: CDR-Weighted Cross-Entropy Loss
        # Only for TCR generation (ALPHA/BETA), not PEPTIDE
        # =====================================================
        use_cdr_weighting = (
            self.config.get("use_cdr_weighting", False) and
            self.task != GenerationTask.PEPTIDE  # Peptides don't have CDR regions
        )

        if use_cdr_weighting:
            # Get CDR annotations from batch (if available from collator)
            # If not in batch, compute on-the-fly from labels using ANARCI (cached)
            cdr_annotations = batch.get("cdr_annotations")
            if cdr_annotations is None and ANARCI_AVAILABLE:
                cdr_annotations = self._get_cdr_annotations_from_labels(labels)

            # Ensure cdr_annotations is a list (default to all None if not available)
            if cdr_annotations is None:
                cdr_annotations = [None] * batch_size

            # Build per-position weights for each sequence in batch
            position_weights = torch.ones(batch_size, seq_len, device=self.device)

            for b in range(batch_size):
                annot = cdr_annotations[b] if b < len(cdr_annotations) else None
                if annot is not None:
                    weights_b = build_cdr_position_weights(
                        seq_len=seq_len,
                        cdr_annotations=annot,
                        cdr1_weight=self.config.get("cdr1_weight", 2.0),
                        cdr2_weight=self.config.get("cdr2_weight", 2.0),
                        cdr3_weight=self.config.get("cdr3_weight", 4.0),
                        framework_weight=self.config.get("framework_weight", 1.0),
                        device=self.device,
                    )
                    position_weights[b] = weights_b

            # Weighted cross-entropy per position
            ce_per_position = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=-100,
                reduction='none'
            ).view(batch_size, seq_len)

            valid_mask = (labels != -100).float()
            weighted_sum = (ce_per_position * position_weights * valid_mask).sum()
            weight_sum = (position_weights * valid_mask).sum().clamp(min=1)
            weighted_ce = weighted_sum / weight_sum
        else:
            # Standard CE loss (used for PEPTIDE task or when CDR weighting disabled)
            weighted_ce = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        # =====================================================
        # COMPONENT 2: BLOSUM62 Soft Loss
        # Penalize biologically dissimilar substitutions more
        # =====================================================
        blosum_loss = torch.tensor(0.0, device=self.device)
        blosum_alpha = self.config.get("blosum_alpha", 0.0)

        if self.config.get("use_blosum_loss", False) and blosum_alpha > 0:
            if hasattr(self, 'blosum_loss_fn') and self.blosum_loss_fn is not None:
                blosum_loss = self.blosum_loss_fn(logits, labels)

        # =====================================================
        # COMPONENT 3: Auxiliary Losses (identity, length)
        # CRITICAL: Must be differentiable - no argmax!
        # =====================================================
        aux_loss = torch.tensor(0.0, device=self.device)

        if self.config.get("use_auxiliary_losses", False):
            # Differentiable soft identity loss using softmax probabilities
            # (argmax breaks gradients - we use probability of correct label instead)
            probs = F.softmax(logits, dim=-1)  # (batch, seq_len, vocab_size)

            # Handle padding: replace -100 with 0 for gather, then mask later
            valid_mask = (labels != -100).float()
            n_valid = valid_mask.sum().clamp(min=1)
            labels_clamped = labels.clamp(min=0)  # Replace -100 with 0 for indexing

            # Get probability assigned to correct label at each position
            target_probs = probs.gather(2, labels_clamped.unsqueeze(2)).squeeze(2)  # (batch, seq_len)

            # Soft identity: mean probability of correct label (masked)
            soft_identity = (target_probs * valid_mask).sum() / n_valid
            identity_loss = 1.0 - soft_identity  # Minimize this = maximize target prob

            # Differentiable length consistency loss using expected EOS position
            # Use softmax to compute expected EOS position (soft argmax)
            eos_id = self.tokenizer.eos_token_id if self.tokenizer else 2
            eos_probs = probs[:, :, eos_id]  # (batch, seq_len) - prob of EOS at each position

            # Position indices
            positions = torch.arange(seq_len, device=self.device, dtype=torch.float).unsqueeze(0)  # (1, seq_len)

            # Expected EOS position = sum(position * EOS_prob) / sum(EOS_prob)
            # Add small epsilon to avoid division by zero
            eos_prob_sum = eos_probs.sum(dim=1, keepdim=True).clamp(min=1e-6)
            pred_expected_eos = (positions * eos_probs).sum(dim=1) / eos_prob_sum.squeeze(1)

            # Label EOS position (non-differentiable, used as target)
            label_eos_pos = torch.where(
                labels == eos_id,
                torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1),
                torch.full((batch_size, seq_len), seq_len, device=self.device)
            ).min(dim=1).values.float()

            length_loss = F.l1_loss(pred_expected_eos, label_eos_pos)

            aux_loss = (
                self.config.get("aux_identity_weight", 0.05) * identity_loss +
                self.config.get("aux_length_weight", 0.02) * length_loss
            )

        # =====================================================
        # COMBINE ALL LOSSES
        # =====================================================
        # Main loss: (1 - blosum_alpha) * CDR_weighted_CE + blosum_alpha * BLOSUM_loss
        if blosum_alpha > 0:
            main_loss = (1.0 - blosum_alpha) * weighted_ce + blosum_alpha * blosum_loss
        else:
            main_loss = weighted_ce

        # Add auxiliary losses
        total_loss = main_loss + aux_loss

        return {
            "loss": total_loss,
            "ce_loss": weighted_ce.detach(),
            "blosum_loss": blosum_loss.detach() if isinstance(blosum_loss, torch.Tensor) else torch.tensor(0.0),
            "aux_loss": aux_loss.detach() if isinstance(aux_loss, torch.Tensor) else torch.tensor(0.0),
        }

    def _compute_stopping_metric(self, val_metrics: Dict[str, float]) -> float:
        """
        Compute composite early stopping metric from biological metrics.

        Lower is better. Combines:
        - Loss (lower is better)
        - Identity (higher is better -> invert)
        - Position accuracy (higher is better -> invert)
        - Hydrophobicity correlation (higher is better -> invert)

        Returns:
            Composite metric (lower is better)
        """
        # Get weights from config
        loss_weight = self.config.get("stopping_loss_weight", 0.4)
        identity_weight = self.config.get("stopping_identity_weight", 0.3)
        position_weight = self.config.get("stopping_position_weight", 0.2)
        hydro_weight = self.config.get("stopping_hydro_weight", 0.1)

        # Normalize weights
        total_weight = loss_weight + identity_weight + position_weight + hydro_weight
        loss_weight /= total_weight
        identity_weight /= total_weight
        position_weight /= total_weight
        hydro_weight /= total_weight

        # Get metrics (with defaults)
        loss = val_metrics.get("eval_loss", 1.0)
        identity = val_metrics.get("identity_mean", 0.0)
        position_acc = val_metrics.get("mean_position_accuracy", 0.0)
        hydro_corr = val_metrics.get("hydrophobicity_corr_mean", 0.0)

        # Compute composite (lower is better)
        # For metrics where higher is better, use (1 - metric)
        composite = (
            loss_weight * loss +
            identity_weight * (1.0 - identity) +
            position_weight * (1.0 - position_acc) +
            hydro_weight * (1.0 - (hydro_corr + 1.0) / 2.0)  # Normalize hydro_corr from [-1,1] to [0,1]
        )

        return composite

    def _get_cdr_annotations_from_labels(
        self,
        labels: torch.Tensor
    ) -> List[Optional[Dict[str, Tuple[int, int]]]]:
        """
        Compute CDR annotations from label tokens using ANARCI.

        Decodes labels to sequences and uses cached ANARCI annotation.

        Args:
            labels: (batch, seq_len) label tensor

        Returns:
            List of CDR annotation dicts, one per batch item
        """
        if not ANARCI_AVAILABLE or self.tokenizer is None:
            return [None] * labels.shape[0]

        annotations = []
        for b in range(labels.shape[0]):
            # Get valid tokens (not -100 padding)
            valid_tokens = labels[b][labels[b] != -100].tolist()
            if len(valid_tokens) < 10:  # Too short for meaningful annotation
                annotations.append(None)
                continue

            # Decode to sequence
            try:
                sequence = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)
                sequence = sequence.replace(" ", "")  # Remove spaces

                if len(sequence) < 50:  # Too short for full TCR
                    annotations.append(None)
                    continue

                # Determine chain type from task
                chain_type = "A" if self.task == GenerationTask.ALPHA else "B"

                # Get CDR positions (cached)
                cdr_pos = self.cdr_cache.get_or_compute(sequence, chain_type)
                annotations.append(cdr_pos)

            except Exception:
                annotations.append(None)

        return annotations

    def setup_data(self, include_val: bool = True) -> None:
        """
        Setup data with seq2seq-specific evaluator and biological loss functions.

        Overrides base class to add:
        - GenerationEvaluator for comprehensive metrics
        - BLOSUM62SoftLoss for biological similarity-aware training
        """
        # Call base class setup
        super().setup_data(include_val)

        # Setup seq2seq-specific evaluator
        if include_val:
            self.evaluator = GenerationEvaluator(
                self.tokenizer,
                compute_detailed_metrics=self.config.get("compute_detailed_metrics", True),
            )

        # Setup BLOSUM loss function if enabled
        if self.config.get("use_blosum_loss", False) and self.tokenizer is not None:
            self.blosum_loss_fn = BLOSUM62SoftLoss(
                tokenizer=self.tokenizer,
                temperature=self.config.get("blosum_temperature", 1.0),
                alpha=self.config.get("blosum_alpha", 0.1),
            )
            # Move to device
            if hasattr(self, 'device'):
                self.blosum_loss_fn = self.blosum_loss_fn.to(self.device)
            self._log(f"Initialized BLOSUM62SoftLoss (alpha={self.config.get('blosum_alpha', 0.1)})")

        # Log CDR weighting configuration
        if self.config.get("use_cdr_weighting", False):
            if self.task == GenerationTask.PEPTIDE:
                self._log("Note: CDR weighting disabled for PEPTIDE task (peptides have no CDR regions)")
            else:
                self._log(f"CDR weighting enabled: CDR3={self.config.get('cdr3_weight', 4.0)}, "
                         f"CDR1/2={self.config.get('cdr1_weight', 2.0)}/{self.config.get('cdr2_weight', 2.0)}, "
                         f"Framework={self.config.get('framework_weight', 1.0)}")
                if not ANARCI_AVAILABLE:
                    self._log("Warning: ANARCI not available - CDR positions will use fallback (uniform weights)")

    # Keep legacy method name for backward compatibility
    def _setup_data(self, include_val: bool = True):
        """Legacy method name - calls setup_data."""
        self.setup_data(include_val)

    def _setup_optimizer(self, lr: Optional[float] = None):
        """Legacy method - calls setup_optimizer."""
        self.setup_optimizer(lr)

    def _setup_scheduler(self, num_training_steps: int):
        """Legacy method - calls setup_scheduler."""
        self.setup_scheduler(num_training_steps)

    def overfit_single_batch(self):
        """
        Overfit check: Train on a single batch to verify the pipeline works.

        This should drive loss to near zero and demonstrate the model can learn.
        """
        # Setup model and data if not already done
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

        self._setup_optimizer(lr=lr)
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
            self._log("The encoder-decoder architecture is working correctly.")
        else:
            self._log("\nWARNING: Loss reduction is limited.")
            self._log("Consider checking: gradients, learning rate, architecture")

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

        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        self._log(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self._log(f"New best model saved! Loss: {metrics.get('eval_loss', 'N/A'):.4f}")

        # Cleanup old checkpoints (keep 3)
        checkpoints = sorted(self.output_dir.glob("checkpoint_step_*.pt"))
        for old_ckpt in checkpoints[:-3]:
            old_ckpt.unlink()

    def train(self):
        """Full training loop with validation, checkpointing, and early stopping."""
        # Setup model and data if not already done
        if self.model is None:
            self.setup_model()
        if self.train_loader is None:
            self.setup_data(include_val=True)

        self._log("\n" + "=" * 60)
        self._log("STARTING FULL TRAINING")
        self._log("=" * 60 + "\n")

        num_epochs = self.config.get("num_epochs", 10)
        grad_accum_steps = self.config.get("gradient_accumulation_steps", 1)
        eval_steps = self.config.get("eval_steps", 1000)
        save_steps = self.config.get("save_steps", 5000)
        log_steps = self.config.get("log_steps", 100)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        patience = self.config.get("patience", 5)

        num_training_steps = len(self.train_loader) * num_epochs // grad_accum_steps
        self._setup_optimizer()
        self._setup_scheduler(num_training_steps)

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
                project=self.config.get("wandb_project", "tcr-seq2seq"),
                name=self.config.get("wandb_run_name"),
                config=self.config,
                settings=wandb.Settings(init_timeout=300),  # 5 min timeout for slow connections
            )

        self.model.train()

        for epoch in range(num_epochs):
            if self.is_distributed:
                self.train_loader.sampler.set_epoch(epoch)

            self._log(f"\n{'='*60}")
            self._log(f"Epoch {epoch + 1}/{num_epochs}")
            self._log(f"{'='*60}")

            # Update self-attention dropout schedule for cross-attention forcing
            self.model.set_epoch(epoch)

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
                            compute_detailed_metrics=self.config.get("compute_detailed_metrics", True),
                        )

                        # Log base metrics
                        self._log(
                            f"Val: loss={val_metrics['eval_loss']:.4f}, "
                            f"ppl={val_metrics['perplexity']:.2f}"
                        )

                        if "exact_match_rate" in val_metrics:
                            self._log(f"     exact_match={val_metrics['exact_match_rate']:.4f}")

                        # Log sequence similarity metrics
                        if "identity_mean" in val_metrics:
                            self._log(
                                f"     identity={val_metrics['identity_mean']:.4f} (+/-{val_metrics.get('identity_std', 0):.3f}), "
                                f"levenshtein={val_metrics.get('levenshtein_mean', 0):.2f}"
                            )

                        # Log position-specific metrics
                        if "mean_position_accuracy" in val_metrics:
                            self._log(
                                f"     pos_acc={val_metrics['mean_position_accuracy']:.4f}, "
                                f"cdr3_first3={val_metrics.get('cdr3_first3_accuracy', 0):.4f}, "
                                f"cdr3_last2={val_metrics.get('cdr3_last2_accuracy', 0):.4f}"
                            )

                        # Log composition metrics
                        if "hydrophobicity_corr_mean" in val_metrics:
                            self._log(
                                f"     hydro_corr={val_metrics['hydrophobicity_corr_mean']:.4f}, "
                                f"charge_mae={val_metrics.get('charge_mae', 0):.3f}, "
                                f"aa_js_div={val_metrics.get('aa_distribution_js', 0):.4f}"
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

                        # Check for improvement using biological metrics if enabled
                        if self.config.get("use_biological_stopping", False):
                            # Compute composite stopping metric (lower is better)
                            stopping_metric = self._compute_stopping_metric(val_metrics)
                            is_best = stopping_metric < getattr(self, 'best_stopping_metric', float('inf'))
                            if is_best:
                                self.best_stopping_metric = stopping_metric
                                self.best_val_loss = val_metrics["eval_loss"]
                                patience_counter = 0
                                self._log(f"New best composite metric: {stopping_metric:.4f}")
                            else:
                                patience_counter += 1
                                self._log(f"No improvement (composite={stopping_metric:.4f}). Patience: {patience_counter}/{patience}")
                        else:
                            # Standard: only check loss
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

            # End of epoch logging
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

        # Save final model
        self._save_checkpoint(
            num_epochs - 1, self.global_step,
            {"final": True, "best_val_loss": self.best_val_loss}
        )


# =============================================================================
# CLI Arguments
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="TCR Conditional Sequence Generation Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Backend
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "cuda", "xla", "neuron", "trainium"],
                        help="Training backend: auto (detect), cuda (NVIDIA GPU), xla/neuron/trainium (AWS Trainium)")

    # Data
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to directory containing parquet files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for logs and checkpoints")
    parser.add_argument("--task", type=str, default="ALPHA",
                        choices=["ALPHA", "BETA", "PEPTIDE"],
                        help="Generation task")
    parser.add_argument("--permutation_keys", type=str, nargs="*",
                        default=None,
                        help="Permutation keys to filter sequences. "
                             "If not specified, auto-discovers keys matching task requirements.")
    parser.add_argument("--max_encoder_length", type=int, default=512,
                        help="Maximum encoder sequence length (512 for trn1.2xlarge, 1024 for larger instances)")
    parser.add_argument("--max_decoder_length", type=int, default=256,
                        help="Maximum decoder sequence length (256 for trn1.2xlarge, 350 for larger instances)")

    # Model (default to smallest ESM2 for trn1.2xlarge memory constraints)
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t6_8M_UR50D",
                        help="Pretrained ESM2 model name (esm2_t6_8M for trn1.2xlarge, esm2_t33_650M for larger instances)")
    parser.add_argument("--decoder_layers", type=int, default=4,
                        help="Number of decoder transformer layers")
    parser.add_argument("--decoder_heads", type=int, default=4,
                        help="Number of decoder attention heads (4 for ESM2-8M, 20 for ESM2-650M)")
    parser.add_argument("--decoder_dim", type=int, default=320,
                        help="Decoder hidden dimension (320 for ESM2-8M, 1280 for ESM2-650M)")
    parser.add_argument("--decoder_ffn_dim", type=int, default=1280,
                        help="Decoder FFN intermediate dimension")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--decoder_warm_start", action="store_true",
                        help="Initialize decoder self-attention and FFN from pre-trained encoder weights (fixes mode collapse)")

    # Cross-attention forcing (Phase 8 - fixes mode collapse when decoder ignores encoder)
    parser.add_argument("--self_attn_drop_initial", type=float, default=0.0,
                        help="Initial self-attention dropout probability (e.g., 0.5 to force cross-attention usage)")
    parser.add_argument("--self_attn_drop_final", type=float, default=0.0,
                        help="Final self-attention dropout probability after annealing (e.g., 0.1)")
    parser.add_argument("--self_attn_drop_anneal_epochs", type=int, default=3,
                        help="Number of epochs before annealing from initial to final dropout")

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

    # Training (defaults optimized for trn1.2xlarge with ESM2-8M)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size per device (8 for trn1.2xlarge with ESM2-8M)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of data loader workers (2 for trn1.2xlarge to leave CPU for XLA compilation)")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps (4 with batch_size=8 for effective batch=32)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
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
    parser.add_argument("--wandb_project", type=str, default="tcr-seq2seq",
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

    # Enhanced evaluation
    parser.add_argument("--num_generate", type=int, default=100,
                        help="Number of samples to generate for evaluation metrics")
    parser.add_argument("--compute_detailed_metrics", action="store_true", default=True,
                        help="Compute advanced sequence metrics (BLOSUM62, position accuracy, etc.)")

    # ==========================================================================
    # Biological Loss Configuration
    # ==========================================================================

    # CDR Region Weighting (for ALPHA/BETA tasks only, not PEPTIDE)
    parser.add_argument("--use_cdr_weighting", action="store_true",
                        help="Enable CDR region-weighted loss (requires ANARCI). "
                             "Weights CDR3 > CDR1/2 > Framework regions.")
    parser.add_argument("--cdr_weight_preset", type=str, default=None,
                        choices=["uniform", "cdr3_focused", "all_cdr_equal", "extreme_cdr3"],
                        help="Preset CDR weight configuration. Overrides individual weights if set.")
    parser.add_argument("--cdr1_weight", type=float, default=2.0,
                        help="Loss weight for CDR1 region")
    parser.add_argument("--cdr2_weight", type=float, default=2.0,
                        help="Loss weight for CDR2 region")
    parser.add_argument("--cdr3_weight", type=float, default=4.0,
                        help="Loss weight for CDR3 region (most important for specificity)")
    parser.add_argument("--framework_weight", type=float, default=1.0,
                        help="Loss weight for framework regions (conserved)")

    # BLOSUM62 Soft Loss
    parser.add_argument("--use_blosum_loss", action="store_true",
                        help="Enable BLOSUM62 soft loss (penalize biologically dissimilar substitutions more)")
    parser.add_argument("--blosum_alpha", type=float, default=0.1,
                        help="BLOSUM loss mixing weight: (1-alpha)*CE + alpha*BLOSUM_KL")
    parser.add_argument("--blosum_temperature", type=float, default=1.0,
                        help="Temperature for BLOSUM score softmax normalization")

    # Auxiliary Losses
    parser.add_argument("--use_auxiliary_losses", action="store_true",
                        help="Enable auxiliary losses (identity, length consistency)")
    parser.add_argument("--aux_identity_weight", type=float, default=0.05,
                        help="Weight for soft sequence identity auxiliary loss")
    parser.add_argument("--aux_length_weight", type=float, default=0.02,
                        help="Weight for length consistency auxiliary loss")

    # Multi-objective Early Stopping
    parser.add_argument("--use_biological_stopping", action="store_true",
                        help="Use biological metrics for early stopping (not just loss)")
    parser.add_argument("--stopping_loss_weight", type=float, default=0.4,
                        help="Weight for loss in early stopping composite metric")
    parser.add_argument("--stopping_identity_weight", type=float, default=0.3,
                        help="Weight for identity in early stopping composite metric")
    parser.add_argument("--stopping_position_weight", type=float, default=0.2,
                        help="Weight for position accuracy in early stopping composite metric")
    parser.add_argument("--stopping_hydro_weight", type=float, default=0.1,
                        help="Weight for hydrophobicity correlation in early stopping")

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()
    config = vars(args)

    # Apply CDR weight preset if specified
    if config.get("cdr_weight_preset"):
        preset_name = config["cdr_weight_preset"]
        if preset_name in CDR_WEIGHT_PRESETS:
            preset = CDR_WEIGHT_PRESETS[preset_name]
            config["cdr1_weight"] = preset["cdr1"]
            config["cdr2_weight"] = preset["cdr2"]
            config["cdr3_weight"] = preset["cdr3"]
            config["framework_weight"] = preset["framework"]
            print(f"Applied CDR weight preset '{preset_name}': {preset}")

    # Get backend from config (auto-detected if not specified)
    backend = get_backend(config.get("backend", "auto"))

    # Create trainer with backend
    trainer = TCRSeq2SeqTrainer(config, backend=backend)

    if args.overfit_check:
        # Use overfit check from base trainer
        trainer.overfit_single_batch()
    else:
        # Use full training loop
        trainer.train()

    # Cleanup distributed
    if trainer.is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
