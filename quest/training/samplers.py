"""
quest.training.samplers - Custom data samplers for training.

Consolidates sampler implementations from:
  - scripts/training/esm_native_trainer.py
    (LengthBucketSampler, DistributedLengthBucketSampler)
  - scripts/training/tcr_robust_contrastive_trainer.py
    (CurriculumSampler, DistributedCurriculumSampler)
"""

from typing import List, Optional

import numpy as np
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


# =============================================================================
# Length Bucket Samplers (from esm_native_trainer.py)
# =============================================================================


class LengthBucketSampler(Sampler):
    """
    Groups sequences by length buckets to minimize padding waste.

    Instead of packing (which causes cross-attention contamination),
    this sampler groups similar-length sequences so that when padded
    to batch max, the padding overhead is minimal.

    For a distribution where 92% of data is 256-384 tokens:
    - Sequences in same bucket get batched together
    - Each batch pads to its own max (not global max)
    - No 4D attention masks needed, Flash Attention 2 works perfectly
    """

    def __init__(
        self,
        lengths: np.ndarray,
        batch_size: int,
        bucket_boundaries: List[int] = [128, 256, 384, 512, 768],
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
    ):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.bucket_boundaries = sorted(bucket_boundaries)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # Assign each sequence to a bucket (0 to num_buckets)
        # np.digitize returns bucket index for each length
        self.bucket_ids = np.digitize(self.lengths, self.bucket_boundaries)
        self.num_buckets = len(self.bucket_boundaries) + 1

        # Pre-compute indices per bucket for efficiency
        self.bucket_indices = [
            np.where(self.bucket_ids == b)[0] for b in range(self.num_buckets)
        ]

    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling."""
        self.epoch = epoch

    def __len__(self):
        if self.drop_last:
            return (len(self.lengths) // self.batch_size) * self.batch_size
        return len(self.lengths)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)

        # Collect all batch chunks
        all_batches = []

        for bucket_idx in range(self.num_buckets):
            indices = self.bucket_indices[bucket_idx].copy()
            if len(indices) == 0:
                continue

            if self.shuffle:
                rng.shuffle(indices)

            # Split into batch-sized chunks
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:
                    all_batches.append(batch)
                elif not self.drop_last:
                    all_batches.append(batch)

        # Shuffle batch order (keeps sequences within batch from same bucket)
        if self.shuffle:
            rng.shuffle(all_batches)

        # Flatten and yield
        if all_batches:
            all_indices = np.concatenate(all_batches)
            return iter(all_indices.tolist())
        return iter([])


class DistributedLengthBucketSampler(LengthBucketSampler):
    """
    Distributed version of LengthBucketSampler for multi-GPU training.

    Each rank gets a subset of the data while maintaining:
    - Length-based bucketing within each rank
    - Deterministic shuffling across epochs
    - Proper data partitioning (no overlap between ranks)
    """

    def __init__(
        self,
        lengths: np.ndarray,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        bucket_boundaries: List[int] = [128, 256, 384, 512, 768],
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
    ):
        # Get distributed info
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0

        self.num_replicas = num_replicas
        self.rank = rank

        # Initialize parent (don't drop_last yet, we handle it at distributed level)
        super().__init__(
            lengths=lengths,
            batch_size=batch_size,
            bucket_boundaries=bucket_boundaries,
            shuffle=shuffle,
            drop_last=False,  # Handle at distributed level
            seed=seed,
        )

        self.drop_last_distributed = drop_last

        # Calculate samples per replica
        total_size = len(self.lengths)
        if self.drop_last_distributed:
            # Make divisible by (batch_size * num_replicas)
            self.total_size = (total_size // (batch_size * num_replicas)) * (batch_size * num_replicas)
        else:
            # Pad to make divisible
            self.total_size = ((total_size + num_replicas - 1) // num_replicas) * num_replicas

        self.num_samples = self.total_size // self.num_replicas

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)

        # Build global ordering with bucket grouping
        all_batches = []

        for bucket_idx in range(self.num_buckets):
            indices = self.bucket_indices[bucket_idx].copy()
            if len(indices) == 0:
                continue

            if self.shuffle:
                rng.shuffle(indices)

            # Create batches
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:
                    all_batches.append(batch)

        # Shuffle batch order globally
        if self.shuffle:
            rng.shuffle(all_batches)

        # Flatten to global indices
        if all_batches:
            all_indices = np.concatenate(all_batches).tolist()
        else:
            all_indices = list(range(len(self.lengths)))

        # Pad if necessary
        if len(all_indices) < self.total_size:
            padding = all_indices[:self.total_size - len(all_indices)]
            all_indices.extend(padding)

        # Truncate if necessary
        all_indices = all_indices[:self.total_size]

        # Subsample for this rank
        indices = all_indices[self.rank:self.total_size:self.num_replicas]

        return iter(indices)


# =============================================================================
# Curriculum Learning Samplers (from tcr_robust_contrastive_trainer.py)
# =============================================================================


class CurriculumSampler(Sampler):
    """
    Curriculum learning: start with short sequences, gradually introduce longer ones.

    Phases (by epoch progress):
    - 0-25%:  Only sequences <= 25th percentile length
    - 25-50%: Only sequences <= 50th percentile length
    - 50-75%: Only sequences <= 75th percentile length
    - 75-100%: All sequences

    Args:
        dataset: A dataset with ``alpha_seqs`` and ``beta_seqs`` attributes
            (e.g., ``TCRPairingDataset``).  Combined sequence length is used
            for curriculum ordering.
        num_epochs: Total number of training epochs.
        current_epoch: Starting epoch (default 0).
        shuffle: Whether to shuffle within each curriculum phase.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_epochs: int,
        current_epoch: int = 0,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.current_epoch = current_epoch
        self.shuffle = shuffle
        self.seed = seed

        # Compute combined sequence lengths
        self.lengths = np.array([
            len(a) + len(b)
            for a, b in zip(dataset.alpha_seqs, dataset.beta_seqs)
        ])

        self.p25 = np.percentile(self.lengths, 25)
        self.p50 = np.percentile(self.lengths, 50)
        self.p75 = np.percentile(self.lengths, 75)

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def _get_curriculum_mask(self) -> np.ndarray:
        progress = self.current_epoch / max(1, self.num_epochs - 1)

        if progress < 0.25:
            return self.lengths <= self.p25
        elif progress < 0.5:
            return self.lengths <= self.p50
        elif progress < 0.75:
            return self.lengths <= self.p75
        else:
            return np.ones(len(self.lengths), dtype=bool)

    def __iter__(self):
        mask = self._get_curriculum_mask()
        indices = np.where(mask)[0]

        if self.shuffle:
            rng = np.random.RandomState(self.seed + self.current_epoch)
            rng.shuffle(indices)

        return iter(indices.tolist())

    def __len__(self):
        return int(self._get_curriculum_mask().sum())


class DistributedCurriculumSampler(CurriculumSampler):
    """Distributed version of curriculum sampler."""

    def __init__(
        self,
        dataset: Dataset,
        num_epochs: int,
        num_replicas: int,
        rank: int,
        current_epoch: int = 0,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = True,
    ):
        super().__init__(dataset, num_epochs, current_epoch, shuffle, seed)
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last

    def __iter__(self):
        mask = self._get_curriculum_mask()
        indices = np.where(mask)[0]

        if self.shuffle:
            rng = np.random.RandomState(self.seed + self.current_epoch)
            rng.shuffle(indices)

        indices = indices.tolist()

        # Ensure divisibility
        total_size = len(indices)
        if self.drop_last:
            total_size = (total_size // self.num_replicas) * self.num_replicas
        else:
            padding = self.num_replicas - (total_size % self.num_replicas)
            if padding != self.num_replicas:
                indices += indices[:padding]
                total_size = len(indices)

        indices = indices[:total_size]

        # Subsample for this rank
        indices = indices[self.rank:total_size:self.num_replicas]

        return iter(indices)

    def __len__(self):
        mask = self._get_curriculum_mask()
        total = int(mask.sum())
        if self.drop_last:
            return total // self.num_replicas
        return (total + self.num_replicas - 1) // self.num_replicas
