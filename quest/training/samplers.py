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
# Token-budget, step-synchronised batch sampler (Stage 4.3, 2026-10-05)
# =============================================================================


class TokenBudgetStepSampler(Sampler):
    """Batch sampler whose optimizer steps are always ``rows_per_step`` rows and whose
    micro-batches fill a token budget — identically on every rank.

    Why it exists: ``DistributedLengthBucketSampler`` uses one fixed ``batch_size`` for every
    length bucket, so batch 16 (sized for the rare 600-token rows) starves the GPU on the 99.5 %
    of rows that are ~17 tokens (Stage 4.1 arms 2–3: 23–26 % GPU). Letting the micro-batch size
    vary by length fixes that, but only if every rank agrees on how many micro-batches make a
    step — otherwise DDP all-reduces mismatched counts and hangs or mis-averages. The old
    sampler stride-shards one flat index list, so it cannot give that guarantee.

    Design (deterministic in ``seed + epoch``, so every rank computes the same global plan):
      1. Shuffle rows within each length bucket, then concatenate buckets shortest-first.
      2. Chunk that sequence into global step-batches of exactly ``rows_per_step`` rows. Only the
         final partial chunk is dropped (< ``rows_per_step`` rows per epoch in total) — never a
         whole sparse bucket of long rows, which would silently discard the full-MHC rows.
      3. Each step-batch's micro-batch size comes from its GLOBAL max length: the largest divisor
         ``d`` of ``rows_per_step // world_size`` with ``d * max_len <= token_budget``. All ranks
         see the same step-batch, so all derive the same ``d`` and the same accumulation count.
      4. Shuffle rows within the step-batch (balances long rows across ranks), give rank ``r``
         its contiguous slice, split that slice into micro-batches of ``d`` rows.
      5. Shuffle the order of step-batches.

    Use as ``DataLoader(ds, batch_sampler=sampler, ...)``. The trainer must step the optimizer
    by ``step_plan()`` (micro-batches per optimizer step, in yield order), not by a fixed
    ``gradient_accumulation_steps``.

    Loss weighting note: the trainer keeps the existing mean-of-micro-batch-means, so changing
    the micro-batch partition reweights rows slightly versus Stage 4.1. The Stage 4.3
    equivalence run (within 2 % at step 2000) is the check that this is immaterial.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        rows_per_step: int = 1024,
        token_budget: int = 16384,
        max_micro_rows: int = 256,
        bucket_boundaries: List[int] = [64, 128, 256, 512, 1024],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        if rows_per_step % num_replicas:
            raise ValueError(f"rows_per_step={rows_per_step} not divisible by world size {num_replicas}")
        self.lengths = np.asarray(lengths)
        self.rows_per_step = rows_per_step
        self.rows_per_rank = rows_per_step // num_replicas
        self.token_budget = token_budget
        self.max_micro_rows = max_micro_rows
        self.bucket_boundaries = sorted(bucket_boundaries)
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        bucket_ids = np.digitize(self.lengths, self.bucket_boundaries)
        self.bucket_indices = [np.where(bucket_ids == b)[0] for b in range(len(self.bucket_boundaries) + 1)]
        self._divisors = [d for d in range(1, self.rows_per_rank + 1) if self.rows_per_rank % d == 0]
        self._cache_key = None
        self._cache = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def micro_rows_for(self, max_len: int) -> int:
        """Largest divisor of rows_per_rank that fits the token budget (and the row cap)."""
        fits = [d for d in self._divisors
                if d <= self.max_micro_rows and d * max(int(max_len), 1) <= self.token_budget]
        return fits[-1] if fits else 1

    def _plan(self):
        key = (self.epoch, self.rank)
        if self._cache_key == key:
            return self._cache
        rng = np.random.default_rng(self.seed + self.epoch)
        ordered = []
        for idx in self.bucket_indices:
            idx = idx.copy()
            if self.shuffle:
                rng.shuffle(idx)
            ordered.append(idx)
        flat = np.concatenate(ordered) if ordered else np.array([], dtype=np.int64)
        # Drop the epoch remainder from the SHORT end. The sequence is shortest-first, so cutting
        # the tail would drop the longest (full-MHC) rows every epoch — caught by a test.
        flat = flat[len(flat) % self.rows_per_step:]
        n_steps = len(flat) // self.rows_per_step
        steps = [flat[i * self.rows_per_step:(i + 1) * self.rows_per_step].copy() for i in range(n_steps)]
        for s in steps:
            if self.shuffle:
                rng.shuffle(s)
        if self.shuffle:
            order = rng.permutation(n_steps)
            steps = [steps[i] for i in order]
        micro_batches, plan = [], []
        lo, hi = self.rank * self.rows_per_rank, (self.rank + 1) * self.rows_per_rank
        for s in steps:
            d = self.micro_rows_for(int(self.lengths[s].max()))
            mine = s[lo:hi]
            micro_batches.extend(mine[j:j + d].tolist() for j in range(0, self.rows_per_rank, d))
            plan.append(self.rows_per_rank // d)
        self._cache_key, self._cache = key, (micro_batches, plan)
        return self._cache

    def step_plan(self) -> List[int]:
        """Micro-batches per optimizer step for the current epoch, in yield order."""
        return list(self._plan()[1])

    def __iter__(self):
        return iter(self._plan()[0])

    def __len__(self) -> int:
        return len(self._plan()[0])


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
