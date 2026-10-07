"""ESM SFT scorer: ESM2/ESM-C encoder + attention pooling + scalar head.

The encoder is frozen for `base` variants. For `cpt` variants the LoRA
adapter weights are trainable (PEFT marks them so by default), but the
base encoder stays frozen. The pooling + head are always trainable.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from track_a.esm_sft.encoders import LoadedEncoder, encoder_forward


class _AttentionPool(nn.Module):
    """Single-head attention pool over token positions (mask-aware)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, L, H), mask: (B, L) bool — True for valid tokens
        logits = self.score(x).squeeze(-1)  # (B, L)
        logits = logits.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(logits, dim=-1)  # (B, L)
        return (weights.unsqueeze(-1) * x).sum(dim=1)  # (B, H)


class ESMScorer(nn.Module):
    """f([cls] TRB ... PEP ... MHC) → scalar binding logit.

    The encoder ingests a single concatenated sequence (chains separated
    by special tokens). The attention pool produces a fixed vector and a
    2-layer MLP returns the binary logit.

    `freeze_base`: if True, freeze the underlying encoder parameters
    (LoRA params remain trainable on cpt variants).
    """

    def __init__(self, loaded: LoadedEncoder, head_dim: int = 256, dropout: float = 0.1, freeze_base: bool = True):
        super().__init__()
        # Register encoder as a child module so .to(device) moves its weights.
        self.encoder = loaded.encoder
        self._is_esmc = loaded.is_esmc
        self._tokenizer = loaded.tokenizer
        self._hidden_dim = loaded.hidden_dim
        self.pool = _AttentionPool(loaded.hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(loaded.hidden_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )
        if freeze_base:
            for p in self.encoder.parameters():
                p.requires_grad = False
            # PEFT adapters set requires_grad=True on LoRA params at attach time;
            # iterate again and re-enable any parameter whose name contains "lora"
            for n, p in self.encoder.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True

    @property
    def loaded(self) -> LoadedEncoder:
        # Reconstruct the dataclass at access time so encoder_forward keeps working.
        return LoadedEncoder(
            encoder=self.encoder, tokenizer=self._tokenizer,
            hidden_dim=self._hidden_dim, is_esmc=self._is_esmc,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # The ENCODER runs under bf16 autocast on CUDA (encoder_forward does that itself). The
        # pool and head run in fp32, OUTSIDE autocast. Running them in bf16 (the previous
        # "entire forward under autocast" fix for a dtype crash) quantised the final logit to
        # ~3 significant digits: 100 retrieval candidates that differ only in a 9-residue
        # peptide produced logits within one bf16 ulp of each other, every candidate tied, and
        # ties count against the true peptide — Recall@10 = 0.0, MRR = 1/100 and AUC = 0.5
        # exactly, on 15,000 queries (2026-10-07, wave-1 cell 1, 18 GPU-hours). A ranking head
        # needs fp32 resolution; the fp32 cast of (B, L, H) costs nothing that matters.
        h = encoder_forward(self.loaded, input_ids, attention_mask).float()  # (B, L, H) fp32
        with torch.autocast(device_type=h.device.type, enabled=False):
            pooled = self.pool(h, attention_mask.bool())  # (B, H)
            return self.head(pooled).squeeze(-1)  # (B,) fp32
