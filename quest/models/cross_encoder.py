"""
TCR Cross-Encoder for alpha-beta pairing classification.

Cross-encoder approach that concatenates alpha and beta sequences with a
separator and uses ESM2's self-attention to learn interaction patterns,
rather than encoding chains separately.

Architecture:
    [CLS] Alpha - Beta [EOS] -> ESM2 -> CLS embedding -> Classifier -> Binary score
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class AttentionPooling(nn.Module):
    """Learned attention pooling for sequence representations."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # hidden_states: (batch, seq_len, hidden_dim)
        # attention_mask: (batch, seq_len)
        attn_scores = self.attention(hidden_states).squeeze(-1)  # (batch, seq_len)
        attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
        return (hidden_states * attn_weights).sum(dim=1)


class TCRCrossEncoder(nn.Module):
    """
    Cross-encoder for TCR alpha-beta pairing classification.

    Concatenates alpha and beta sequences with a separator and classifies
    whether they are a true paired match using ESM2's self-attention.
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        hidden_dim: int = 256,
        dropout: float = 0.1,
        pooling: str = "cls",
        attn_implementation: Optional[str] = "flash_attention_2",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        # Load encoder with backend-appropriate attention implementation
        encoder_kwargs = {"torch_dtype": torch_dtype}
        if attn_implementation:
            encoder_kwargs["attn_implementation"] = attn_implementation

        self.encoder = AutoModel.from_pretrained(
            model_name,
            **encoder_kwargs,
        )

        esm_hidden = self.encoder.config.hidden_size
        self.pooling_type = pooling

        if pooling == "attention":
            self.pooler = AttentionPooling(esm_hidden)
        else:
            self.pooler = None

        # 2-layer classification head
        self.classifier = nn.Sequential(
            nn.Linear(esm_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for cross-encoder.

        Args:
            input_ids: (batch, seq_len) tokenized concatenated sequences
            attention_mask: (batch, seq_len) attention mask

        Returns:
            logits: (batch,) binary classification logits
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if self.pooling_type == "cls":
            # Use CLS token (index 0)
            pooled = outputs.last_hidden_state[:, 0, :].float()
        elif self.pooling_type == "mean":
            # Mean pooling over non-padding tokens
            hidden = outputs.last_hidden_state.float()
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            # Attention pooling
            pooled = self.pooler(
                outputs.last_hidden_state.float(),
                attention_mask,
            )

        logits = self.classifier(pooled).squeeze(-1)  # (batch,)
        return logits


class CrossEncoderBCELoss(nn.Module):
    """Binary cross-entropy loss for cross-encoder pair classification."""

    def __init__(self, pos_weight: Optional[float] = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute BCE loss with optional positive class weighting.

        Args:
            logits: (batch,) model outputs
            labels: (batch,) binary labels (1=matched, 0=mismatched)

        Returns:
            dict with loss and accuracy
        """
        if self.pos_weight is not None:
            pw = torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype)
        else:
            pw = None

        loss = F.binary_cross_entropy_with_logits(
            logits, labels.float(), pos_weight=pw
        )

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            accuracy = (preds == labels).float().mean()

            # Compute per-class metrics
            pos_mask = labels == 1
            neg_mask = labels == 0

            if pos_mask.sum() > 0:
                pos_acc = (preds[pos_mask] == labels[pos_mask]).float().mean()
            else:
                pos_acc = torch.tensor(0.0)

            if neg_mask.sum() > 0:
                neg_acc = (preds[neg_mask] == labels[neg_mask]).float().mean()
            else:
                neg_acc = torch.tensor(0.0)

        return {
            "loss": loss,
            "accuracy": accuracy,
            "pos_accuracy": pos_acc,
            "neg_accuracy": neg_acc,
        }
