"""LSTM and MLP binary scorers for AS retrieval baselines.

Both scorers take (TCR, peptide, MHC) and return a binding logit. At eval
time we score a TCR against every peptide in the candidate pool to rank
them; Recall@K / MRR / Hit@1 are computed from those ranks.

This is intentionally simple — these are the floor baselines, not SOTA.
Architecture mirrors the encoder choices in
scripts/benchmark/benchmark_baseline.py but uses a binary head (so the
model generalizes to peptides held out from training).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

# Amino-acid vocab: 0 = PAD, 1..20 = AAs, 21 = UNK
AA_TO_IDX = {
    aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")
}
AA_TO_IDX["X"] = 21  # any non-standard
PAD_IDX = 0
UNK_IDX = 21
VOCAB_SIZE = 22


def encode_seq(seq: Optional[str], max_len: int) -> tuple[torch.Tensor, int]:
    """Tokenize an AA sequence, return (ids, length). Pads with PAD_IDX."""
    if seq is None or not isinstance(seq, str) or len(seq) == 0:
        return torch.zeros(max_len, dtype=torch.long), 0
    ids = [AA_TO_IDX.get(c.upper(), UNK_IDX) for c in seq[:max_len]]
    length = len(ids)
    if length < max_len:
        ids += [PAD_IDX] * (max_len - length)
    return torch.tensor(ids, dtype=torch.long), length


class _LSTMEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = hidden_dim * 2

    def forward(self, ids: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        emb = self.embed(ids)  # (B, L, E)
        # pack/pad would be more efficient, but for our sizes simple last-step works.
        # We use the last non-pad position of forward + first position of backward.
        # Simpler: take mean over non-pad positions.
        mask = (ids != PAD_IDX).unsqueeze(-1).float()  # (B, L, 1)
        out, _ = self.lstm(emb)  # (B, L, 2H)
        masked = out * mask
        # Mean-pool over non-pad
        denom = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        return masked.sum(dim=1) / denom  # (B, 2H)


class _MLPEncoder(nn.Module):
    """Mean-pool of AA embeddings (no recurrence) → linear projection."""

    def __init__(self, vocab_size: int, embed_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out_dim = out_dim

    def forward(self, ids: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        emb = self.embed(ids)  # (B, L, E)
        mask = (ids != PAD_IDX).unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (emb * mask).sum(dim=1) / denom  # (B, E)
        return self.proj(pooled)


class TCRPeptideMHCScorer(nn.Module):
    """Binary scorer: f(TCR, peptide, MHC) -> logit.

    `mhc_dim` is the size of the one-hot MHC vector (allele vocabulary size).
    `chains` is "trb" for TRB-only or "paired" for TRA+TRB.
    """

    def __init__(
        self,
        encoder_type: str,  # "lstm" | "mlp"
        chains: str,        # "trb" | "paired"
        mhc_dim: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        head_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        if encoder_type not in {"lstm", "mlp"}:
            raise ValueError(f"encoder_type must be 'lstm' or 'mlp', got {encoder_type}")
        if chains not in {"trb", "paired"}:
            raise ValueError(f"chains must be 'trb' or 'paired', got {chains}")
        self.encoder_type = encoder_type
        self.chains = chains

        def _make_encoder():
            if encoder_type == "lstm":
                return _LSTMEncoder(VOCAB_SIZE, embed_dim, hidden_dim, num_layers, dropout)
            return _MLPEncoder(VOCAB_SIZE, embed_dim, hidden_dim * 2, dropout)

        self.trb_enc = _make_encoder()
        self.pep_enc = _make_encoder()
        if chains == "paired":
            self.tra_enc = _make_encoder()
        else:
            self.tra_enc = None
        enc_dim = self.trb_enc.out_dim
        concat_dim = enc_dim + enc_dim + mhc_dim + (enc_dim if chains == "paired" else 0)

        self.head = nn.Sequential(
            nn.Linear(concat_dim, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )

    def forward(
        self,
        trb_ids: torch.Tensor, trb_len: torch.Tensor,
        pep_ids: torch.Tensor, pep_len: torch.Tensor,
        mhc_oh: torch.Tensor,
        tra_ids: Optional[torch.Tensor] = None, tra_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        trb = self.trb_enc(trb_ids, trb_len)
        pep = self.pep_enc(pep_ids, pep_len)
        feats = [trb, pep, mhc_oh]
        if self.chains == "paired":
            assert tra_ids is not None and tra_len is not None
            tra = self.tra_enc(tra_ids, tra_len)
            feats.append(tra)
        cat = torch.cat(feats, dim=-1)
        return self.head(cat).squeeze(-1)
