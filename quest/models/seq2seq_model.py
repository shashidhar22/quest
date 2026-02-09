"""
TCR Seq2Seq Model for conditional TCR sequence generation.

Encoder-decoder architecture using ESM2 as encoder and a Transformer decoder
for autoregressive generation. Supports self-attention dropout for cross-attention
forcing, decoder warm-start from encoder weights, and BLOSUM62-weighted soft loss.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .positional_encoding import PositionalEncoding
from .seq2seq_decoder import SelfAttnDropoutDecoder, SelfAttnDropoutDecoderLayer


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
        encoder_kwargs = {"dtype": torch_dtype}

        # Try flash_attention_2 first, fall back to eager if not available
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
