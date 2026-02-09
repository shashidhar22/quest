"""
Positional Encoding module for Transformer-based models.

Standard sinusoidal positional encoding as described in "Attention Is All You Need"
(Vaswani et al., 2017). Used by seq2seq models for decoder position information.
"""

import math

import torch
import torch.nn as nn


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
