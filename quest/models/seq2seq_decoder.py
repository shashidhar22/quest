"""
Custom Transformer decoder with self-attention dropout support.

Provides SelfAttnDropoutDecoderLayer and SelfAttnDropoutDecoder for
cross-attention forcing during training. By randomly skipping
self-attention, the decoder is forced to rely on cross-attention
for information from the encoder, addressing mode collapse.
"""

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
