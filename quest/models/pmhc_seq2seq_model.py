"""
Peptide-MHC Seq2Seq Model for peptide-to-MHC sequence generation.

Encoder-decoder architecture using ESM2 as encoder and a standard Transformer
decoder for autoregressive MHC sequence generation from peptide input.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .positional_encoding import PositionalEncoding


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
