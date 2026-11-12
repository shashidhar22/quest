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


def build_head(in_dim: int, head_dim: int = 256, dropout: float = 0.1) -> nn.Module:
    """Per-feature normalization followed by the binding MLP.

    On the DG1 cached peptide-feature fixture, normalization lets this head beat
    the class prior within two epochs; the unnormalized control does not.
    This establishes a conditioning problem in the head, not receptor dependence
    of a retrained judge (ACS findings 9.14). Eval uses fixed running statistics,
    so scores do not depend on the other candidates in a batch.
    """
    return nn.Sequential(
        nn.BatchNorm1d(in_dim),
        nn.Linear(in_dim, head_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(head_dim, 1),
    )


class ESMScorer(nn.Module):
    """Encoder and scalar binding head.

    Default: the original Track A single-pool architecture and checkpoint format.
    With peptide_pool=True (the DG1 judge), concatenate whole-sequence and peptide
    attention pools, then normalize features before the MLP. The separate peptide
    pool keeps its representation available without dilution by the longer MHC.

    freeze_base freezes encoder weights except LoRA; pools and head stay trainable.
    """

    def __init__(self, loaded: LoadedEncoder, head_dim: int = 256, dropout: float = 0.1, freeze_base: bool = True,
                 peptide_pool: bool = False):
        super().__init__()
        # Register encoder as a child module so .to(device) moves its weights.
        self.encoder = loaded.encoder
        self._is_esmc = loaded.is_esmc
        self._tokenizer = loaded.tokenizer
        self._hidden_dim = loaded.hidden_dim
        self.pool = _AttentionPool(loaded.hidden_dim)
        # The original Track A callers and checkpoints retain their architecture.
        # The DG1 judge opts into the repaired architecture explicitly.
        self.peptide_pool = peptide_pool
        if peptide_pool:
            self.pool_peptide = _AttentionPool(loaded.hidden_dim)
            self.head = build_head(2 * loaded.hidden_dim, head_dim, dropout)
        else:
            self.head = nn.Sequential(
                nn.Linear(loaded.hidden_dim, head_dim), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(head_dim, 1),
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

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                peptide_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # The ENCODER runs under bf16 autocast on CUDA (encoder_forward does that itself). The
        # pool and head run in fp32, OUTSIDE autocast. Running them in bf16 (the previous
        # "entire forward under autocast" fix for a dtype crash) quantised the final logit to
        # ~3 significant digits: 100 retrieval candidates that differ only in a 9-residue
        # peptide produced logits within one bf16 ulp of each other, every candidate tied, and
        # ties count against the true peptide — Recall@10 = 0.0, MRR = 1/100 and AUC = 0.5
        # exactly, on 15,000 queries (2026-10-07, wave-1 cell 1, 18 GPU-hours). A ranking head
        # needs fp32 resolution; the fp32 cast of (B, L, H) costs nothing that matters.
        # `peptide_mask` is REQUIRED, and is not quietly defaulted to the full mask: a scorer
        # that cannot locate the peptide is the instrument finding 9.14 describes, and a silent
        # fallback would reproduce it with no sign in the output. A row whose peptide survived
        # no token (an empty or fully truncated peptide slot) is a serialization fault, so it is
        # refused here rather than pooled over the whole molecule twice.
        if not self.peptide_pool:
            h = encoder_forward(self.loaded, input_ids, attention_mask).float()
            with torch.autocast(device_type=h.device.type, enabled=False):
                return self.head(self.pool(h, attention_mask.bool())).squeeze(-1)
        if peptide_mask is None:
            raise ValueError(
                "ESMScorer.forward needs peptide_mask: the peptide is ~2-4 % of a serialized "
                "molecule's tokens. The repaired judge pools these separately "
                "(quest-cpt/findings.md 9.14). Build the mask with "
                "cpt_training_lib.annotate_token_regions, as track_a/esm_sft/judge.py does.")
        valid = attention_mask.bool()
        pep = peptide_mask.bool() & valid
        empty = ~pep.any(dim=-1)
        if bool(empty.any()):
            raise ValueError(
                f"{int(empty.sum())} of {pep.shape[0]} rows carry no peptide token, so their "
                f"peptide pool is undefined. A softmax over an all -inf row is NaN, and a NaN "
                f"logit is a technical fault that must not be scored as a rank.")
        h = encoder_forward(self.loaded, input_ids, attention_mask).float()  # (B, L, H) fp32
        with torch.autocast(device_type=h.device.type, enabled=False):
            pooled = torch.cat([self.pool(h, valid), self.pool_peptide(h, pep)], dim=-1)
            return self.head(pooled).squeeze(-1)  # (B,) fp32
