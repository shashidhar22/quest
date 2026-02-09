"""
Attention pooling and projection head modules for contrastive learning.

Provides reusable pooling and projection components used in dual encoder
and contrastive learning architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Learned attention-weighted pooling.

    Better for ESM2 than CLS token because ESM2 uses MLM objective,
    not next-sentence prediction. The CLS token embedding may not
    contain optimal sequence-level information.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len)
        Returns:
            pooled: (batch, hidden_dim)
        """
        attn_scores = self.attention(hidden_states).squeeze(-1)  # (batch, seq_len)
        attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
        return (hidden_states * attn_weights).sum(dim=1)


class DeepProjectionHead(nn.Module):
    """
    Deep projection head following SimCLR recommendations.

    SimCLR found that deeper projection heads (2-3 hidden layers)
    with the same hidden dimension perform better than shallow ones.

    Args:
        use_batchnorm: If True, use BatchNorm1d instead of LayerNorm.
            BatchNorm normalizes across the batch dimension, which can help
            break embedding collapse ("North Cone" problem) by forcing the
            model to use inter-sample variance.
    """

    def __init__(
        self,
        hidden_dim: int,
        projection_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_batchnorm: bool = False,
    ):
        super().__init__()

        layers = []
        for _ in range(num_layers - 1):
            norm_layer = nn.BatchNorm1d(hidden_dim) if use_batchnorm else nn.LayerNorm(hidden_dim)
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                norm_layer,
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(hidden_dim, projection_dim))

        self.projection = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)
