"""
TCR Dual Encoder for alpha-beta pairing with contrastive learning.

Provides the TCRDualEncoder (shared ESM2 backbone with projection heads),
MomentumEncoder (MoCo-style EMA for stable distractor embeddings), and
DistractorManager (manages distractor embedding banks).
"""

import copy
import random
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm
from transformers import AutoModel

from .attention_pooling import AttentionPooling, DeepProjectionHead


class TCRDualEncoder(nn.Module):
    """
    Dual encoder for TCR alpha-beta pairing.

    Features:
    - Shared ESM2 backbone with LoRA
    - Attention-weighted pooling
    - Deep projection head
    - Learnable temperature with sigmoid bounds (always has gradients)
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        projection_dim: int = 256,
        pooling: str = "attention",
        initial_temperature: float = 0.07,
        temp_min: float = 0.01,
        temp_max: float = 0.5,
        projection_layers: int = 3,
        projection_dropout: float = 0.1,
        use_batchnorm: bool = False,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )

        hidden_dim = self.encoder.config.hidden_size
        self.pooling_type = pooling

        # Pooling (keep in float32 for gradient precision)
        if pooling == "attention":
            self.pooler = AttentionPooling(hidden_dim)  # float32
        else:
            self.pooler = None

        # Projection (keep in float32 for gradient precision)
        self.projection = DeepProjectionHead(
            hidden_dim, projection_dim, projection_layers, projection_dropout,
            use_batchnorm=use_batchnorm,
        )  # float32

        # Temperature with sigmoid bounds (Fix #8)
        self.temp_min = temp_min
        self.temp_max = temp_max
        # Initialize logit to achieve initial_temperature
        init_normalized = (initial_temperature - temp_min) / (temp_max - temp_min)
        init_normalized = np.clip(init_normalized, 1e-6, 1 - 1e-6)
        init_logit = np.log(init_normalized / (1 - init_normalized))
        self.temperature_logit = nn.Parameter(
            torch.tensor(init_logit, dtype=torch.float32)
        )

    @property
    def temperature(self) -> torch.Tensor:
        """Temperature with smooth sigmoid bounds - always has gradients."""
        return self.temp_min + (self.temp_max - self.temp_min) * torch.sigmoid(
            self.temperature_logit
        )

    def _pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # NOTE: Sequences are now truncated to variable region in the collator
        # BEFORE encoding. This is more effective than post-encoding masking because
        # ESM2's self-attention contextualizes all positions during the forward pass.

        # Convert to float32 for pooling (better gradient precision)
        hidden_float = hidden_states.float()
        if self.pooling_type == "attention":
            return self.pooler(hidden_float, attention_mask)
        elif self.pooling_type == "cls":
            return hidden_float[:, 0, :]
        else:  # mean pooling
            mask = attention_mask.unsqueeze(-1).expand(hidden_float.size()).float()
            return (hidden_float * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        recenter: bool = True,
    ) -> torch.Tensor:
        """Encode sequences to L2-normalized embeddings.

        Args:
            recenter: If True, subtract batch mean before projection.
                      This removes ESM2's strong bias toward a common direction,
                      which causes all embeddings to have high cosine similarity.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(outputs.last_hidden_state, attention_mask)  # Returns float32

        # Re-center embeddings to remove ESM2's mean direction bias
        # ESM2 maps all sequences to a narrow cone; subtracting mean spreads them out
        if recenter and pooled.size(0) > 1:
            pooled = pooled - pooled.mean(dim=0, keepdim=True)

        projected = self.projection(pooled)  # float32 in, float32 out
        # Normalize in float32 for numerical stability
        return F.normalize(projected, p=2, dim=-1)

    def forward(
        self,
        alpha_input_ids: torch.Tensor,
        alpha_attention_mask: torch.Tensor,
        beta_input_ids: torch.Tensor,
        beta_attention_mask: torch.Tensor,
        recenter: bool = True,
    ) -> Dict[str, torch.Tensor]:
        alpha_emb = self.encode(alpha_input_ids, alpha_attention_mask, recenter=recenter)
        beta_emb = self.encode(beta_input_ids, beta_attention_mask, recenter=recenter)

        return {
            "alpha_embeddings": alpha_emb,
            "beta_embeddings": beta_emb,
            "temperature": self.temperature,
        }


class MomentumEncoder:
    """
    Momentum-updated encoder for stable distractor embeddings.

    Similar to MoCo: maintains an exponentially moving average of model weights.
    This prevents distractor embeddings from becoming stale during training.
    """

    def __init__(
        self,
        model: nn.Module,
        momentum: float = 0.999,
        device: torch.device = None,
    ):
        self.momentum = momentum
        self.device = device

        # Create copy (no gradients)
        base_model = model.module if hasattr(model, 'module') else model
        self.encoder = copy.deepcopy(base_model).to(device)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update with exponential moving average."""
        base_model = model.module if hasattr(model, 'module') else model

        for param_q, param_k in zip(base_model.parameters(), self.encoder.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    @torch.no_grad()
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.encoder.eval()
        return self.encoder.encode(input_ids, attention_mask)


class DistractorManager:
    """
    Manages distractor embeddings with:
    - Momentum encoder for stability
    - Random sampling from full pool (not truncation)
    - Warnings for edge cases
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        unpaired_alphas: List[str],
        unpaired_betas: List[str],
        device: torch.device,
        initial_pool_size: int = 50_000,
        batch_size: int = 128,
        is_main: bool = True,
        momentum: float = 0.999,
    ):
        self.device = device
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.is_main = is_main

        self.unpaired_alphas = unpaired_alphas
        self.unpaired_betas = unpaired_betas

        # Momentum encoder
        self.momentum_encoder = MomentumEncoder(model, momentum, device)

        # Initial embedding computation (random sample, not first N)
        if is_main:
            print("Computing initial distractor embeddings...")

        alpha_sample = random.sample(
            unpaired_alphas, min(initial_pool_size, len(unpaired_alphas))
        ) if unpaired_alphas else []
        beta_sample = random.sample(
            unpaired_betas, min(initial_pool_size, len(unpaired_betas))
        ) if unpaired_betas else []

        self.alpha_embeddings = self._compute_embeddings(alpha_sample, "alpha")
        self.beta_embeddings = self._compute_embeddings(beta_sample, "beta")

        if is_main:
            print(f"Distractor bank: {len(self.alpha_embeddings):,} alphas, "
                  f"{len(self.beta_embeddings):,} betas")

    @torch.no_grad()
    def _compute_embeddings(
        self,
        sequences: List[str],
        name: str,
    ) -> torch.Tensor:
        if not sequences:
            if self.is_main:
                warnings.warn(f"No {name} sequences for distractors!")
            return torch.empty(0, 256)

        embeddings = []

        iterator = range(0, len(sequences), self.batch_size)
        if self.is_main:
            iterator = tqdm(iterator, desc=f"Encoding {name}", leave=False)

        for i in iterator:
            batch_seqs = sequences[i:i + self.batch_size]

            encoded = self.tokenizer(
                batch_seqs,
                padding=True,
                truncation=True,
                max_length=320,
                return_tensors="pt",
            )

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                emb = self.momentum_encoder.encode(
                    encoded["input_ids"].to(self.device),
                    encoded["attention_mask"].to(self.device),
                )

            embeddings.append(emb.cpu())

        return torch.cat(embeddings, dim=0)

    def sample(
        self,
        num_alpha: int,
        num_beta: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample distractor embeddings with warnings for edge cases."""

        # Fix #13: Warn if not enough available
        if num_alpha > len(self.alpha_embeddings) and self.is_main:
            warnings.warn(
                f"Requested {num_alpha} alpha distractors but only "
                f"{len(self.alpha_embeddings)} available"
            )
        if num_beta > len(self.beta_embeddings) and self.is_main:
            warnings.warn(
                f"Requested {num_beta} beta distractors but only "
                f"{len(self.beta_embeddings)} available"
            )

        # Sample
        if len(self.alpha_embeddings) > 0:
            alpha_idx = np.random.choice(
                len(self.alpha_embeddings),
                min(num_alpha, len(self.alpha_embeddings)),
                replace=False,
            )
            alpha_sample = self.alpha_embeddings[alpha_idx].to(self.device)
        else:
            alpha_sample = torch.empty(0, 256, device=self.device)

        if len(self.beta_embeddings) > 0:
            beta_idx = np.random.choice(
                len(self.beta_embeddings),
                min(num_beta, len(self.beta_embeddings)),
                replace=False,
            )
            beta_sample = self.beta_embeddings[beta_idx].to(self.device)
        else:
            beta_sample = torch.empty(0, 256, device=self.device)

        return alpha_sample, beta_sample

    def update_momentum(self, model: nn.Module):
        """Update momentum encoder."""
        self.momentum_encoder.update(model)

    def refresh(self, pool_size: int = 50_000):
        """Refresh embeddings with updated momentum encoder."""
        if self.is_main:
            print("Refreshing distractor embeddings...")

        alpha_sample = random.sample(
            self.unpaired_alphas, min(pool_size, len(self.unpaired_alphas))
        ) if self.unpaired_alphas else []
        beta_sample = random.sample(
            self.unpaired_betas, min(pool_size, len(self.unpaired_betas))
        ) if self.unpaired_betas else []

        self.alpha_embeddings = self._compute_embeddings(alpha_sample, "alpha")
        self.beta_embeddings = self._compute_embeddings(beta_sample, "beta")
