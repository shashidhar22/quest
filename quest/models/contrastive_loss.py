"""
Contrastive loss functions for TCR alpha-beta pairing.

Provides DistributedRobustInfoNCE loss with cross-GPU gathering support,
hidden positive masking, and external distractor integration.
"""

from typing import Dict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def gather_embeddings(embeddings: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    Gather embeddings from all GPUs.

    Important: Preserves gradients for the local batch by replacing
    the gathered tensor at the current rank's position.
    """
    if world_size == 1:
        return embeddings

    gathered = [torch.zeros_like(embeddings) for _ in range(world_size)]
    dist.all_gather(gathered, embeddings)

    # Replace with original to preserve gradients
    gathered[dist.get_rank()] = embeddings

    return torch.cat(gathered, dim=0)


class DistributedRobustInfoNCE(nn.Module):
    """
    Symmetric InfoNCE loss with:
    - Correct cross-GPU indexing (rank offset for labels)
    - Sequence-based hidden positive masking
    - External distractors

    Key insight: When gathering embeddings across GPUs, each GPU's local
    batch occupies a specific slice of the gathered tensor. Labels must
    account for this offset.
    """

    def __init__(self, rank: int, world_size: int, symmetric: bool = True):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.symmetric = symmetric

    def forward(
        self,
        alpha_local: torch.Tensor,      # (N, D) - local batch
        beta_local: torch.Tensor,       # (N, D) - local batch
        alpha_sim_mask: torch.Tensor,   # (N, N) - local similarity mask
        beta_sim_mask: torch.Tensor,    # (N, N) - local similarity mask
        alpha_all: torch.Tensor,        # (W*N, D) - gathered across GPUs
        beta_all: torch.Tensor,         # (W*N, D) - gathered across GPUs
        ext_alpha: torch.Tensor,        # (M, D) - distractors
        ext_beta: torch.Tensor,         # (M, D) - distractors
        temperature: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        local_bs = alpha_local.size(0)
        device = alpha_local.device

        # Labels: positive for alpha_local[i] is beta_all[rank * local_bs + i]
        rank_offset = self.rank * local_bs
        labels = torch.arange(local_bs, device=device) + rank_offset

        # Prepare masks (exclude diagonal - true positives)
        alpha_hidden_mask = alpha_sim_mask.to(device).clone()
        alpha_hidden_mask.fill_diagonal_(False)

        beta_hidden_mask = beta_sim_mask.to(device).clone()
        beta_hidden_mask.fill_diagonal_(False)

        total_masked = 0

        # === Direction 1: Alpha -> Beta ===
        if ext_beta is not None and ext_beta.numel() > 0:
            targets_b = torch.cat([beta_all, ext_beta], dim=0)
        else:
            targets_b = beta_all

        logits_a2b = (alpha_local @ targets_b.T) / temperature

        # Mask hidden positives in LOCAL block only
        # Other GPUs' betas and distractors are treated as true negatives
        logits_a2b[:, rank_offset:rank_offset + local_bs].masked_fill_(
            beta_hidden_mask, float("-inf")
        )
        total_masked += beta_hidden_mask.sum().item()

        loss_a2b = F.cross_entropy(logits_a2b, labels)

        with torch.no_grad():
            acc_a2b = (logits_a2b.argmax(dim=1) == labels).float().mean()

        if self.symmetric:
            # === Direction 2: Beta -> Alpha ===
            if ext_alpha is not None and ext_alpha.numel() > 0:
                targets_a = torch.cat([alpha_all, ext_alpha], dim=0)
            else:
                targets_a = alpha_all

            logits_b2a = (beta_local @ targets_a.T) / temperature

            logits_b2a[:, rank_offset:rank_offset + local_bs].masked_fill_(
                alpha_hidden_mask, float("-inf")
            )
            total_masked += alpha_hidden_mask.sum().item()

            loss_b2a = F.cross_entropy(logits_b2a, labels)

            with torch.no_grad():
                acc_b2a = (logits_b2a.argmax(dim=1) == labels).float().mean()

            loss = (loss_a2b + loss_b2a) / 2
            accuracy = (acc_a2b + acc_b2a) / 2
        else:
            loss = loss_a2b
            accuracy = acc_a2b
            loss_b2a = torch.tensor(0.0, device=device)

        return {
            "loss": loss,
            "loss_a2b": loss_a2b,
            "loss_b2a": loss_b2a if self.symmetric else torch.tensor(0.0),
            "accuracy": accuracy,
            "masked_count": total_masked,
        }
