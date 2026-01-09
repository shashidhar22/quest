#!/usr/bin/env python3
"""
TCR Alpha-Beta Pairing with Robust Contrastive Learning (V3 - Production Ready).

A complete implementation combining all fixes and best practices:

Key Features:
- Momentum encoder for stable distractor embeddings (MoCo-style)
- Sequence-based hidden positive masking (CPU-offloaded to DataLoader workers)
- Full-corpus retrieval evaluation (realistic metrics)
- Cross-GPU negative gathering with correct rank-offset indexing
- Proper temperature parameterization (sigmoid bounds, always has gradients)
- Deep projection heads (3-layer, following SimCLR)
- Attention-weighted pooling (better for ESM2 than CLS)
- Curriculum learning (short→long sequences)
- GradScaler, cosine LR scheduler with warmup, early stopping
- Checkpoint management with best model tracking
- Thread-safe caching with size limits
- Comprehensive logging via wandb

Usage:
    torchrun --nproc_per_node=4 tcr_contrastive_v3.py \
        --data_path data/deduplicated/full/foundation_permutations/ \
        --output_dir ./output/tcr_v3 \
        --paired_permutation_keys tra_trb \
        --alpha_distractor_keys tra \
        --beta_distractor_keys trb

Author: Refactored with fixes from V1, V2, V2.1
"""

import argparse
import copy
import glob
import os
import random
import threading
import warnings
from collections import OrderedDict
from datetime import timedelta
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# =============================================================================
# Hardware Optimizations
# =============================================================================

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# =============================================================================
# Thread-Safe Caching (Fix #11, #12)
# =============================================================================

class ThreadSafeCache:
    """
    Thread-safe LRU cache with size limits.
    Prevents memory leaks in multi-worker DataLoader scenarios.
    """
    
    def __init__(self, max_size: int = 3):
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.RLock()
        self._max_size = max_size
    
    def get(self, key: str) -> Optional[any]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None
    
    def set(self, key: str, value: any):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
                self._cache[key] = value
    
    def clear(self):
        with self._lock:
            self._cache.clear()


# Global caches
PAIRED_DATA_CACHE = ThreadSafeCache(max_size=3)
UNPAIRED_DATA_CACHE = ThreadSafeCache(max_size=3)

# Global debug logger
DEBUG_LOG_FILE = None


def debug_log(msg: str, also_print: bool = True):
    """Write debug message to log file and optionally to stdout."""
    if also_print:
        print(msg)
    if DEBUG_LOG_FILE is not None:
        try:
            with open(DEBUG_LOG_FILE, "a") as f:
                f.write(msg + "\n")
        except Exception as e:
            print(f"Warning: Could not write to debug log: {e}")


# =============================================================================
# Sequence Similarity Calculator (Fix #2)
# =============================================================================

class SequenceSimilarityCalculator:
    """
    Computes sequence-level similarity for hidden positive detection.
    
    Key insight: We use SEQUENCE similarity (Levenshtein), not EMBEDDING similarity.
    Embedding similarity is unreliable early in training, but sequence similarity
    is deterministic and biologically meaningful.
    
    This runs in CPU DataLoader workers, not on GPU.
    """
    
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold
    
    @staticmethod
    @lru_cache(maxsize=50000)
    def levenshtein_ratio(s1: str, s2: str) -> float:
        """
        Compute normalized Levenshtein similarity (0-1).
        Cached to avoid recomputation for repeated pairs.
        """
        if s1 == s2:
            return 1.0
        
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # Standard DP for edit distance
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # deletion
                    dp[i][j-1] + 1,      # insertion
                    dp[i-1][j-1] + cost  # substitution
                )
        
        return 1.0 - dp[len1][len2] / max(len1, len2)
    
    def compute_batch_similarity_mask(self, sequences: List[str]) -> torch.Tensor:
        """
        Compute boolean mask indicating which sequence pairs are similar.
        
        Returns:
            mask: (batch, batch) tensor where mask[i,j] = True if sequences[i] 
                  and sequences[j] have similarity >= threshold.
                  Diagonal is always False (self-similarity doesn't count).
        """
        batch_size = len(sequences)
        mask = torch.zeros(batch_size, batch_size, dtype=torch.bool)
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                sim = self.levenshtein_ratio(sequences[i], sequences[j])
                if sim >= self.threshold:
                    mask[i, j] = True
                    mask[j, i] = True
        
        # Diagonal is False (handled by loop starting at j = i + 1)
        return mask


# =============================================================================
# Dataset (Fix #11, #12, #14)
# =============================================================================

class TCRPairingDataset(Dataset):
    """
    Dataset for TCR alpha-beta pairs with:
    - Thread-safe caching
    - Deduplication
    - Proper split handling
    - Unpaired sequence loading for distractors
    """
    
    def __init__(
        self,
        data_path: str,
        paired_permutation_keys: List[str] = ["tra_trb"],
        alpha_distractor_keys: List[str] = ["tra"],
        beta_distractor_keys: List[str] = ["trb"],
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        local_rank: int = 0,
        distractor_sample_size: int = 500_000,
        deduplicate: bool = True,
        skip_distractors: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.split = split
        self.is_main = local_rank == 0
        self.deduplicate = deduplicate
        self.distractor_sample_size = distractor_sample_size
        self.max_samples = max_samples

        # Load paired data
        self._load_paired_data(
            data_path, paired_permutation_keys, split,
            train_ratio, val_ratio, seed
        )

        # Load unpaired distractors (training only, unless skipped)
        if split == "train" and not skip_distractors:
            self._load_unpaired_data(data_path, alpha_distractor_keys, "alpha")
            self._load_unpaired_data(data_path, beta_distractor_keys, "beta")
        else:
            self.unpaired_alphas = []
            self.unpaired_betas = []
            if skip_distractors and self.is_main:
                print("Skipping distractor loading (fast mode)")
    
    def _load_paired_data(
        self,
        data_path: str,
        permutation_keys: List[str],
        split: str,
        train_ratio: float,
        val_ratio: float,
        seed: int,
    ):
        """Load paired alpha-beta sequences."""
        cache_key = f"{data_path}:paired:{','.join(sorted(permutation_keys))}"
        cached = PAIRED_DATA_CACHE.get(cache_key)
        
        if cached is not None:
            if self.is_main:
                print(f"Using cached paired data for {split}...")
            alpha_seqs, beta_seqs = cached
        else:
            import pyarrow.parquet as pq
            
            parquet_files = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
            if not parquet_files:
                raise ValueError(f"No parquet files found in {data_path}")
            
            if self.is_main:
                print(f"Loading paired data from {len(parquet_files)} files...")
            
            alpha_seqs = []
            beta_seqs = []
            seen_pairs: Set[Tuple[str, str]] = set()
            
            # For fast mode, calculate how many we need before splitting
            # We need enough samples that after split we still have max_samples
            early_stop_count = None
            if self.max_samples is not None:
                # Load extra to account for split ratio (e.g., 80% train)
                early_stop_count = int(self.max_samples / 0.8 * 1.5)  # 1.5x buffer
                if self.is_main:
                    print(f"Early stop after {early_stop_count:,} samples (fast mode)")

            iterator = tqdm(parquet_files, desc="Loading paired", disable=not self.is_main)
            for pf in iterator:
                table = pq.read_table(pf, columns=["permutation_key", "sequence"])
                df = table.to_pandas()
                df = df[df["permutation_key"].isin(permutation_keys)]

                for seq in df["sequence"].values:
                    parts = seq.split(" ")
                    if len(parts) >= 2:
                        alpha, beta = parts[0], parts[1]

                        if self.deduplicate:
                            pair_key = (alpha, beta)
                            if pair_key in seen_pairs:
                                continue
                            seen_pairs.add(pair_key)

                        alpha_seqs.append(alpha)
                        beta_seqs.append(beta)

                        # Early stop for fast mode
                        if early_stop_count and len(alpha_seqs) >= early_stop_count:
                            if self.is_main:
                                print(f"Early stopping data load at {len(alpha_seqs):,} samples")
                            break

                # Check if we should stop loading more files
                if early_stop_count and len(alpha_seqs) >= early_stop_count:
                    break
            
            if not alpha_seqs:
                raise ValueError(f"No sequences found with keys: {permutation_keys}")
            
            if self.is_main:
                print(f"Loaded {len(alpha_seqs):,} unique paired sequences")
            
            PAIRED_DATA_CACHE.set(cache_key, (alpha_seqs, beta_seqs))
        
        # Deterministic split
        n_total = len(alpha_seqs)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_total)
        
        train_end = int(n_total * train_ratio)
        val_end = int(n_total * (train_ratio + val_ratio))
        
        if split == "train":
            selected = indices[:train_end]
        elif split == "val":
            selected = indices[train_end:val_end]
        else:  # test
            selected = indices[val_end:]
        
        self.alpha_seqs = [alpha_seqs[i] for i in selected]
        self.beta_seqs = [beta_seqs[i] for i in selected]

        # Apply max_samples limit if specified
        if self.max_samples is not None and len(self.alpha_seqs) > self.max_samples:
            self.alpha_seqs = self.alpha_seqs[:self.max_samples]
            self.beta_seqs = self.beta_seqs[:self.max_samples]
            if self.is_main:
                print(f"Limited to {self.max_samples:,} samples (fast mode)")

        if self.is_main:
            print(f"{split.capitalize()} set: {len(self.alpha_seqs):,} pairs")
    
    def _load_unpaired_data(
        self,
        data_path: str,
        permutation_keys: List[str],
        chain_type: str,
    ):
        """
        Load unpaired sequences for distractors using streaming reservoir sampling.

        This avoids loading all 50-60M sequences into memory by sampling as we read.
        Uses Algorithm R (reservoir sampling) for uniform random sampling.
        """
        cache_key = f"{data_path}:unpaired:{','.join(sorted(permutation_keys))}:{self.distractor_sample_size}"
        cached = UNPAIRED_DATA_CACHE.get(cache_key)

        if cached is not None:
            if self.is_main:
                print(f"Using cached unpaired {chain_type} sequences ({len(cached):,})...")
            sequences = cached
        else:
            import pyarrow.parquet as pq

            parquet_files = sorted(glob.glob(os.path.join(data_path, "*.parquet")))

            if self.is_main:
                print(f"Loading unpaired {chain_type} sequences with streaming sampling...")
                print(f"  Target sample size: {self.distractor_sample_size:,}")

            # Reservoir sampling: maintain a fixed-size sample as we stream through data
            reservoir: List[str] = []
            seen_sequences: Set[str] = set()  # For deduplication within sample
            total_seen = 0

            iterator = tqdm(
                parquet_files, desc=f"Sampling {chain_type}", disable=not self.is_main
            )
            for pf in iterator:
                table = pq.read_table(pf, columns=["permutation_key", "sequence"])
                df = table.to_pandas()
                df = df[df["permutation_key"].isin(permutation_keys)]

                for seq in df["sequence"].values:
                    # Single chain sequences (no space separator)
                    if " " not in seq and len(seq) > 50:
                        # Skip duplicates
                        if seq in seen_sequences:
                            continue

                        total_seen += 1

                        # Reservoir sampling (Algorithm R)
                        if len(reservoir) < self.distractor_sample_size:
                            reservoir.append(seq)
                            seen_sequences.add(seq)
                        else:
                            # Replace with decreasing probability
                            j = random.randint(0, total_seen - 1)
                            if j < self.distractor_sample_size:
                                # Remove old sequence from dedup set
                                old_seq = reservoir[j]
                                seen_sequences.discard(old_seq)
                                # Add new sequence
                                reservoir[j] = seq
                                seen_sequences.add(seq)

                # Update progress bar with current stats
                iterator.set_postfix({
                    "sampled": len(reservoir),
                    "seen": f"{total_seen:,}"
                })

            sequences = reservoir

            if self.is_main:
                print(f"Sampled {len(sequences):,} from {total_seen:,} total {chain_type} sequences")

            UNPAIRED_DATA_CACHE.set(cache_key, sequences)

        if chain_type == "alpha":
            self.unpaired_alphas = sequences
        else:
            self.unpaired_betas = sequences
    
    def __len__(self) -> int:
        return len(self.alpha_seqs)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        return {
            "alpha_seq": self.alpha_seqs[idx],
            "beta_seq": self.beta_seqs[idx],
        }


# =============================================================================
# Collator with CPU-Offloaded Similarity Computation
# =============================================================================

class ContrastivePairCollator:
    """
    Collator that handles both tokenization AND similarity mask computation.
    
    By computing similarity masks here (in DataLoader workers), we:
    1. Offload O(N²) computation to CPU background processes
    2. Prevent GPU stalls waiting for mask computation
    3. Enable parallel mask computation across batches
    """
    
    def __init__(
        self,
        tokenizer,
        similarity_threshold: float = 0.9,
        max_length: int = 320,
        pad_to_multiple_of: int = 8,
        var_region_len: int = 150,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.var_region_len = var_region_len
        # Fix #3: Pass threshold to calculator
        self.sim_calculator = SequenceSimilarityCalculator(threshold=similarity_threshold)
    
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        alpha_seqs = [ex["alpha_seq"] for ex in examples]
        beta_seqs = [ex["beta_seq"] for ex in examples]

        # CRITICAL FIX: Truncate to variable region BEFORE tokenization
        # ESM2's self-attention contextualizes all positions, so masking after encoding
        # is ineffective. The constant region info "bleeds" into variable region embeddings.
        # By truncating here, we prevent the encoder from seeing constant regions entirely.
        if self.var_region_len > 0:
            alpha_seqs = [seq[:self.var_region_len] for seq in alpha_seqs]
            beta_seqs = [seq[:self.var_region_len] for seq in beta_seqs]

        # DEBUG: Print input sequences info
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 1

        if self._debug_count <= 3:  # Only print first 3 batches to avoid spam
            debug_log(f"\n[DEBUG Collator] Batch {self._debug_count}:")
            debug_log(f"  Number of examples: {len(examples)}")
            debug_log(f"  Alpha sequences: {len(alpha_seqs)}, Beta sequences: {len(beta_seqs)}")
            if alpha_seqs:
                debug_log(f"  [First 3 Alpha sequences]")
                for i, seq in enumerate(alpha_seqs[:3]):
                    debug_log(f"    [{i}] (len={len(seq)}): {seq}")
                debug_log(f"  [First 3 Beta sequences]")
                for i, seq in enumerate(beta_seqs[:3]):
                    debug_log(f"    [{i}] (len={len(seq)}): {seq}")

        # 1. Compute similarity masks (CPU-intensive, runs in workers)
        alpha_sim_mask = self.sim_calculator.compute_batch_similarity_mask(alpha_seqs)
        beta_sim_mask = self.sim_calculator.compute_batch_similarity_mask(beta_seqs)

        # 2. Tokenize
        alpha_encoded = self.tokenizer(
            alpha_seqs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        beta_encoded = self.tokenizer(
            beta_seqs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # DEBUG: Print tokenization results
        if self._debug_count <= 3:
            debug_log(f"  [Tokenizer Output]")
            debug_log(f"    Alpha input_ids shape: {alpha_encoded['input_ids'].shape}")
            debug_log(f"    Alpha attention_mask shape: {alpha_encoded['attention_mask'].shape}")
            debug_log(f"    Beta input_ids shape: {beta_encoded['input_ids'].shape}")
            debug_log(f"    Beta attention_mask shape: {beta_encoded['attention_mask'].shape}")
            debug_log(f"    Alpha input_ids (first seq, first 10 tokens): {alpha_encoded['input_ids'][0, :10].tolist()}")
            debug_log(f"    Alpha attention_mask (first seq, first 10): {alpha_encoded['attention_mask'][0, :10].tolist()}")
            debug_log(f"    Non-padding tokens in first alpha: {alpha_encoded['attention_mask'][0].sum().item()}")
            debug_log(f"    Non-padding tokens in first beta: {beta_encoded['attention_mask'][0].sum().item()}")
            # Check for all-padding sequences (potential issue)
            alpha_token_counts = alpha_encoded['attention_mask'].sum(dim=1)
            beta_token_counts = beta_encoded['attention_mask'].sum(dim=1)
            debug_log(f"    Alpha token count range: [{alpha_token_counts.min().item()}, {alpha_token_counts.max().item()}]")
            debug_log(f"    Beta token count range: [{beta_token_counts.min().item()}, {beta_token_counts.max().item()}]")
            if (alpha_token_counts == 0).any() or (beta_token_counts == 0).any():
                debug_log(f"    WARNING: Found sequences with 0 tokens!")
            debug_log(f"    Vocab size: {self.tokenizer.vocab_size}, Pad token id: {self.tokenizer.pad_token_id}")

        return {
            "alpha_input_ids": alpha_encoded["input_ids"],
            "alpha_attention_mask": alpha_encoded["attention_mask"],
            "beta_input_ids": beta_encoded["input_ids"],
            "beta_attention_mask": beta_encoded["attention_mask"],
            "alpha_sim_mask": alpha_sim_mask,
            "beta_sim_mask": beta_sim_mask,
        }


# =============================================================================
# Curriculum Learning Sampler (Fix #15)
# =============================================================================

class CurriculumSampler(Sampler):
    """
    Curriculum learning: start with short sequences, gradually introduce longer ones.
    
    Phases (by epoch progress):
    - 0-25%:  Only sequences <= 25th percentile length
    - 25-50%: Only sequences <= 50th percentile length
    - 50-75%: Only sequences <= 75th percentile length
    - 75-100%: All sequences
    """
    
    def __init__(
        self,
        dataset: TCRPairingDataset,
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
        dataset: TCRPairingDataset,
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


# =============================================================================
# Model Components (Fix #5, #6, #7, #8)
# =============================================================================

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


def apply_lora(model: TCRDualEncoder, config: dict) -> TCRDualEncoder:
    """Apply LoRA to ESM2 encoder."""
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=["query", "key", "value", "dense"],
        bias="none",
    )
    model.encoder = get_peft_model(model.encoder, lora_config)
    return model


# =============================================================================
# Momentum Encoder (Fix #1)
# =============================================================================

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


# =============================================================================
# Distractor Manager (Fix #1, #4, #13)
# =============================================================================

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


# =============================================================================
# Cross-GPU Gathering (Fix #16)
# =============================================================================

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


# =============================================================================
# Loss Function (Fix #1, #2, #16)
# =============================================================================

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


# =============================================================================
# Full-Corpus Evaluator (Fix #3, #9)
# =============================================================================

class FullCorpusEvaluator:
    """
    Evaluate with full-corpus retrieval.
    
    Unlike in-batch evaluation (which inflates metrics), this computes
    Recall@K against the ENTIRE validation set, giving realistic metrics.
    
    Expected metric ranges:
    - In-batch Recall@10 with batch_size=64: ~95%
    - Full-corpus Recall@10 with 10K candidates: ~15-40%
    """
    
    def __init__(self, k_values: List[int] = [1, 5, 10, 20, 100]):
        self.k_values = k_values
    
    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataset: TCRPairingDataset,
        tokenizer,
        device: torch.device,
        batch_size: int = 128,
        max_length: int = 320,
    ) -> Dict[str, float]:
        model.eval()
        
        base_model = model.module if hasattr(model, 'module') else model
        
        # Encode all sequences
        def encode_all(sequences: List[str]) -> torch.Tensor:
            embeddings = []
            
            for i in range(0, len(sequences), batch_size):
                batch_seqs = sequences[i:i + batch_size]
                
                encoded = tokenizer(
                    batch_seqs,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    emb = base_model.encode(
                        encoded["input_ids"].to(device),
                        encoded["attention_mask"].to(device),
                    )
                
                embeddings.append(emb.cpu())
            
            return torch.cat(embeddings, dim=0)
        
        # Encode all alphas and betas
        alpha_emb = encode_all(dataset.alpha_seqs).to(device)
        beta_emb = encode_all(dataset.beta_seqs).to(device)
        
        # Compute ranks in chunks (avoid OOM)
        num_samples = len(dataset)
        all_ranks = []
        chunk_size = 1000
        
        for start in range(0, num_samples, chunk_size):
            end = min(start + chunk_size, num_samples)
            
            # Similarity: (chunk, all_betas)
            sim = alpha_emb[start:end] @ beta_emb.T
            
            # Sort and find rank of true positive
            sorted_idx = sim.argsort(dim=1, descending=True)
            
            for i, global_idx in enumerate(range(start, end)):
                rank = (sorted_idx[i] == global_idx).nonzero(as_tuple=True)[0]
                if len(rank) > 0:
                    all_ranks.append(rank[0].item() + 1)  # 1-indexed
        
        if not all_ranks:
            return {}
        
        ranks = np.array(all_ranks)
        
        metrics = {
            "mrr": float(np.mean(1.0 / ranks)),
            "mean_rank": float(np.mean(ranks)),
            "median_rank": float(np.median(ranks)),
        }
        
        for k in self.k_values:
            metrics[f"recall@{k}"] = float(np.mean(ranks <= k))
        
        return metrics


# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.mode == "max":
            improved = score - self.best_score > self.min_delta
        else:
            improved = self.best_score - score > self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


# =============================================================================
# Trainer
# =============================================================================

class TCRContrastiveTrainer:
    """
    Complete trainer with all fixes applied.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.is_distributed = "LOCAL_RANK" in os.environ

        # Setup debug logging
        global DEBUG_LOG_FILE
        if config.get("debug_log_file"):
            DEBUG_LOG_FILE = config["debug_log_file"]
        else:
            # Default to output_dir/debug.log
            DEBUG_LOG_FILE = os.path.join(config["output_dir"], "debug.log")

        # Create output dir and initialize log file
        os.makedirs(config["output_dir"], exist_ok=True)
        if not self.is_distributed or int(os.environ.get("RANK", 0)) == 0:
            with open(DEBUG_LOG_FILE, "w") as f:
                f.write(f"Debug log started at {os.popen('date').read().strip()}\n")
                f.write(f"Config: {config}\n")
                f.write("=" * 60 + "\n\n")
            print(f"Debug log: {DEBUG_LOG_FILE}")

        # Setup distributed
        if self.is_distributed:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.global_rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=timedelta(minutes=30),
            )
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.local_rank = 0
            self.global_rank = 0
            self.world_size = 1
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._setup_model()
        self._setup_data()
        self._setup_training()
        
        # Wandb
        self.use_wandb = (
            config.get("report_to") == "wandb" 
            and WANDB_AVAILABLE 
            and self._is_main()
        )
        if self.use_wandb:
            wandb.init(
                project=config.get("wandb_project", "tcr-pairing-v3"),
                name=config.get("wandb_run_name"),
                config=config,
            )
    
    def _is_main(self) -> bool:
        return self.global_rank == 0
    
    def _setup_model(self):
        if self._is_main():
            print(f"Loading model: {self.config['model_name']}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Temperature: use user-provided value, or 0.5 for overfit check, or 0.07 default
        user_temp = self.config.get("temperature", None)
        if user_temp is not None and user_temp != 0.07:  # User explicitly set temperature
            initial_temp = user_temp
        elif self.config.get("overfit_check", False):
            initial_temp = 0.5  # Softer distribution for learning
        else:
            initial_temp = 0.07

        self.model = TCRDualEncoder(
            model_name=self.config["model_name"],
            projection_dim=self.config.get("projection_dim", 256),
            pooling=self.config.get("pooling", "attention"),
            initial_temperature=initial_temp,
            projection_layers=self.config.get("projection_layers", 3),
            use_batchnorm=self.config.get("use_batchnorm", False),
        )

        if self.config.get("use_lora", True):
            self.model = apply_lora(self.model, self.config)
            if self._is_main():
                self.model.encoder.print_trainable_parameters()
        else:
            # Freeze encoder when LoRA is disabled
            for param in self.model.encoder.parameters():
                param.requires_grad = False

            # Optionally unfreeze the last N transformer layers
            unfreeze_layers = self.config.get("unfreeze_layers", 0)
            if unfreeze_layers > 0:
                # ESM2 structure: model.encoder.layer[i] (from HuggingFace transformers)
                encoder = self.model.encoder
                num_layers = len(encoder.encoder.layer)

                if unfreeze_layers > num_layers:
                    if self._is_main():
                        print(f"Warning: unfreeze_layers={unfreeze_layers} > num_layers={num_layers}, unfreezing all")
                    unfreeze_layers = num_layers

                # Unfreeze last N layers
                for i in range(num_layers - unfreeze_layers, num_layers):
                    for param in encoder.encoder.layer[i].parameters():
                        param.requires_grad = True

                # Also unfreeze layer norm after transformer blocks
                if hasattr(encoder.encoder, 'emb_layer_norm_after'):
                    for param in encoder.encoder.emb_layer_norm_after.parameters():
                        param.requires_grad = True

                if self._is_main():
                    unfrozen = sum(p.numel() for p in self.model.encoder.parameters() if p.requires_grad)
                    total = sum(p.numel() for p in self.model.encoder.parameters())
                    print(f"LoRA disabled - unfroze last {unfreeze_layers} encoder layers: {unfrozen:,}/{total:,} params")
            else:
                if self._is_main():
                    print("LoRA disabled - encoder frozen, training projection/pooler only")

        # Load pretrained PEFT checkpoint if provided
        pretrained_path = self.config.get("pretrained_checkpoint")
        if pretrained_path:
            if self._is_main():
                print(f"Loading pretrained checkpoint: {pretrained_path}")

            ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)

            # The checkpoint has keys like: base_model.model.esm.encoder...
            # Our model.encoder has keys like: base_model.model.esm.encoder...
            # We need to load into model.encoder

            # Filter to only encoder keys and adjust prefix
            encoder_state = {}
            for k, v in state_dict.items():
                # Keep keys that are part of the ESM encoder
                if k.startswith("base_model.model.esm"):
                    encoder_state[k] = v

            if encoder_state:
                # Load with strict=False to allow missing projection head keys
                missing, unexpected = self.model.encoder.load_state_dict(
                    encoder_state, strict=False
                )
                if self._is_main():
                    print(f"  Loaded {len(encoder_state)} encoder weights")
                    if missing:
                        # Filter out expected missing keys (projection head, etc.)
                        real_missing = [m for m in missing if "lora" in m.lower()]
                        if real_missing:
                            print(f"  Missing LoRA keys: {len(real_missing)}")
                    if unexpected:
                        print(f"  Unexpected keys: {len(unexpected)}")
            else:
                if self._is_main():
                    print("  Warning: No encoder keys found in checkpoint!")

        # Enable gradient checkpointing (saves memory, required for 650M model)
        if self.config.get("gradient_checkpointing", True):
            self.model.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        self.model = self.model.to(self.device)

        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=True,
            )
    
    def _setup_data(self):
        config = self.config

        # Fast mode for overfit check: skip distractors and limit samples
        is_overfit_check = config.get("overfit_check", False)
        skip_distractors = is_overfit_check

        # Use smaller batch size for overfit check (faster iterations)
        if is_overfit_check:
            config["batch_size"] = min(config.get("batch_size", 64), 32)  # Max 32 for overfit check

        # Need enough samples for at least 1 batch after train/val split (80% train)
        max_samples = config.get("batch_size", 64) * 4 if is_overfit_check else None

        if is_overfit_check and self._is_main():
            print("\n[Fast Mode] Overfit check enabled:")
            print(f"  - Skipping distractor loading")
            print(f"  - Using batch_size={config['batch_size']} (reduced for memory)")
            print(f"  - Limiting to {max_samples} samples\n")

        # Datasets
        self.train_dataset = TCRPairingDataset(
            data_path=config["data_path"],
            paired_permutation_keys=config.get("paired_permutation_keys", ["tra_trb"]),
            alpha_distractor_keys=config.get("alpha_distractor_keys", ["tra"]),
            beta_distractor_keys=config.get("beta_distractor_keys", ["trb"]),
            split="train",
            local_rank=self.local_rank,
            distractor_sample_size=config.get("distractor_sample_size", 500_000),
            skip_distractors=skip_distractors,
            max_samples=max_samples,
        )

        self.val_dataset = TCRPairingDataset(
            data_path=config["data_path"],
            paired_permutation_keys=config.get("paired_permutation_keys", ["tra_trb"]),
            split="val",
            local_rank=self.local_rank,
            max_samples=max_samples if is_overfit_check else None,
        )
        
        # Collator
        self.collator = ContrastivePairCollator(
            tokenizer=self.tokenizer,
            similarity_threshold=config.get("similarity_threshold", 0.9),
            max_length=config.get("max_length", 320),
            var_region_len=config.get("var_region_len", 150),
        )
        
        # Sampler - disable curriculum for overfit check (use all data)
        num_epochs = config.get("num_epochs", 10)
        use_curriculum = config.get("use_curriculum", True) and not is_overfit_check
        if use_curriculum:
            if self.is_distributed:
                self.train_sampler = DistributedCurriculumSampler(
                    self.train_dataset,
                    num_epochs=num_epochs,
                    num_replicas=self.world_size,
                    rank=self.global_rank,
                )
            else:
                self.train_sampler = CurriculumSampler(
                    self.train_dataset,
                    num_epochs=num_epochs,
                )
        else:
            if self.is_distributed:
                # For overfit check with small dataset, don't drop_last
                self.train_sampler = DistributedSampler(
                    self.train_dataset, shuffle=True,
                    drop_last=not is_overfit_check
                )
            else:
                self.train_sampler = None
        
        # DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.get("batch_size", 64),
            sampler=self.train_sampler,
            shuffle=(self.train_sampler is None),
            num_workers=config.get("num_workers", 4) if not is_overfit_check else 0,
            prefetch_factor=2 if not is_overfit_check else None,
            pin_memory=True,
            persistent_workers=False,  # Fix #12
            drop_last=not is_overfit_check,  # Don't drop for overfit check
            collate_fn=self.collator,
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.get("batch_size", 64),
            num_workers=2,
            pin_memory=True,
            collate_fn=self.collator,
        )
        
        # Distractors
        if self.train_dataset.unpaired_alphas or self.train_dataset.unpaired_betas:
            self.distractor_manager = DistractorManager(
                model=self.model,
                tokenizer=self.tokenizer,
                unpaired_alphas=self.train_dataset.unpaired_alphas,
                unpaired_betas=self.train_dataset.unpaired_betas,
                device=self.device,
                initial_pool_size=config.get("initial_distractor_pool", 50_000),
                is_main=self._is_main(),
            )
        else:
            self.distractor_manager = None
            if self._is_main():
                warnings.warn("No distractors available!")
        
        # Evaluator
        self.evaluator = FullCorpusEvaluator()
    
    def _setup_training(self):
        config = self.config

        # Learning rate
        base_lr = config.get("learning_rate", 2e-4)
        if config.get("overfit_check", False):
            # Moderate LR for overfit check - training LoRA + projection/pooler
            # LoRA needs lower LR than randomly initialized projection head
            lr = config.get("overfit_lr") or 1e-3
            if self._is_main():
                print(f"Using overfit check LR: {lr} (LoRA + projection/pooler)")
        else:
            lr = base_lr

        # Optimizer with separate param groups for different learning rates
        base_model = self.model.module if hasattr(self.model, 'module') else self.model

        # Group parameters: projection/pooler get higher LR than LoRA
        projection_params = []
        pooler_params = []
        lora_params = []
        other_params = []

        for name, param in base_model.named_parameters():
            if not param.requires_grad:
                continue
            if 'projection' in name:
                projection_params.append(param)
            elif 'pooler' in name and 'encoder' not in name:  # Custom pooler, not ESM pooler
                pooler_params.append(param)
            elif 'lora' in name.lower():
                lora_params.append(param)
            else:
                other_params.append(param)

        # Projection/pooler get 10x LR (randomly initialized, need faster learning)
        param_groups = [
            {'params': projection_params, 'lr': lr * 10, 'name': 'projection'},
            {'params': pooler_params, 'lr': lr * 10, 'name': 'pooler'},
            {'params': lora_params, 'lr': lr, 'name': 'lora'},
            {'params': other_params, 'lr': lr, 'name': 'other'},
        ]

        # Filter out empty groups
        param_groups = [g for g in param_groups if len(g['params']) > 0]

        if self._is_main():
            print(f"[Optimizer] Parameter groups:")
            for g in param_groups:
                total_params = sum(p.numel() for p in g['params'])
                print(f"  {g['name']}: {len(g['params'])} tensors, {total_params:,} params, lr={g['lr']:.6f}")

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,  # Default LR (overridden by param groups)
            weight_decay=config.get("weight_decay", 0.01),
        )
        
        # Scheduler (cosine with warmup)
        grad_accum = config.get("gradient_accumulation_steps", 1)
        num_epochs = config.get("num_epochs", 10)
        steps_per_epoch = len(self.train_loader) // grad_accum
        total_steps = steps_per_epoch * num_epochs
        warmup_steps = int(total_steps * config.get("warmup_ratio", 0.1))
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # GradScaler - NOT needed for bfloat16 (only for float16)
        # bfloat16 has larger dynamic range and doesn't need loss scaling
        self.scaler = None  # Disabled for bfloat16
        
        # Loss
        self.criterion = DistributedRobustInfoNCE(
            rank=self.global_rank,
            world_size=self.world_size,
            symmetric=config.get("symmetric_loss", True),
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get("early_stopping_patience", 5),
            mode="max",
        ) if config.get("early_stopping", True) else None
        
        # State
        self.global_step = 0
        self.best_metric = float("-inf")
    
    def overfit_single_batch(self, num_steps: int = 100):
        """
        Debug utility: Try to overfit on a single batch.

        If the model is working correctly, loss should decrease toward 0.
        If loss doesn't decrease, there's a fundamental issue with:
        - Model architecture
        - Loss function
        - Optimizer setup
        - Gradient flow
        """
        if self._is_main():
            debug_log(f"\n{'='*60}")
            debug_log("SINGLE BATCH OVERFIT CHECK")
            debug_log(f"{'='*60}")
            debug_log(f"Running {num_steps} iterations on a single batch...")
            debug_log("Expected: Loss should decrease significantly (ideally toward 0)")
            debug_log(f"{'='*60}\n")

        self.model.train()

        # Train full model (LoRA + projection/pooler)
        # The encoder needs to learn which positions matter for TCR pairing.
        # With frozen encoder, embeddings are dominated by conserved framework regions.
        base_model = self.model.module if hasattr(self.model, 'module') else self.model

        # Freeze temperature during overfit check to reduce instability
        if hasattr(base_model, 'temperature_logit'):
            base_model.temperature_logit.requires_grad = False

        if self._is_main():
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            lora_params = sum(p.numel() for n, p in base_model.named_parameters()
                             if p.requires_grad and 'lora' in n.lower())
            debug_log(f"[Trainable] {trainable:,} params (LoRA: {lora_params:,}, projection/pooler: {trainable - lora_params:,})")

        # Check dataset size
        if self._is_main():
            debug_log(f"[Dataset Info]")
            debug_log(f"  Train dataset size: {len(self.train_dataset)}")
            debug_log(f"  Train loader batches: {len(self.train_loader)}")
            if len(self.train_dataset) == 0:
                debug_log("  ERROR: Train dataset is empty!")
                debug_log("  Check your data_path and paired_permutation_keys")
                return [], []
            if len(self.train_loader) == 0:
                debug_log("  ERROR: Train loader is empty!")
                debug_log(f"  Dataset has {len(self.train_dataset)} samples but batch_size={self.config.get('batch_size', 64)}")
                debug_log("  This can happen if dataset size < batch_size with drop_last=True")
                return [], []

        # Get a single batch
        batch_iter = iter(self.train_loader)
        try:
            batch = next(batch_iter)
        except StopIteration:
            if self._is_main():
                debug_log("ERROR: Could not get a batch from DataLoader!")
                debug_log(f"  Dataset size: {len(self.train_dataset)}")
                debug_log(f"  Batch size: {self.config.get('batch_size', 64)}")
            return [], []

        # Unpack batch
        alpha_ids = batch["alpha_input_ids"].to(self.device)
        alpha_mask = batch["alpha_attention_mask"].to(self.device)
        beta_ids = batch["beta_input_ids"].to(self.device)
        beta_mask = batch["beta_attention_mask"].to(self.device)
        alpha_sim_mask = batch["alpha_sim_mask"]
        beta_sim_mask = batch["beta_sim_mask"]

        if self._is_main():
            debug_log(f"[Batch Info]")
            debug_log(f"  Alpha IDs shape: {alpha_ids.shape}")
            debug_log(f"  Beta IDs shape: {beta_ids.shape}")
            debug_log(f"  Alpha attention sum: {alpha_mask.sum().item()}")
            debug_log(f"  Beta attention sum: {beta_mask.sum().item()}")
            debug_log("")

        initial_loss = None
        losses = []
        accuracies = []

        for step in range(num_steps):
            self.optimizer.zero_grad()

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                # Forward - disable re-centering for overfit test to simplify optimization
                outputs = self.model(alpha_ids, alpha_mask, beta_ids, beta_mask, recenter=False)

                alpha_local = outputs["alpha_embeddings"]
                beta_local = outputs["beta_embeddings"]
                temperature = outputs["temperature"]

                # Gather across GPUs (no-op if single GPU)
                alpha_all = gather_embeddings(alpha_local, self.world_size)
                beta_all = gather_embeddings(beta_local, self.world_size)

                # No distractors for overfit test (simpler)
                ext_alpha = torch.empty(0, alpha_local.shape[-1], device=self.device)
                ext_beta = torch.empty(0, beta_local.shape[-1], device=self.device)

                # Loss
                loss_dict = self.criterion(
                    alpha_local, beta_local,
                    alpha_sim_mask, beta_sim_mask,
                    alpha_all, beta_all,
                    ext_alpha, ext_beta,
                    temperature,
                )
                loss = loss_dict["loss"]

            # Backward - use gradient clipping to stabilize training
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            loss_val = loss_dict["loss"].item()
            acc_val = loss_dict["accuracy"].item()
            losses.append(loss_val)
            accuracies.append(acc_val)

            if initial_loss is None:
                initial_loss = loss_val

            if self._is_main() and (step % 10 == 0 or step == num_steps - 1):
                # Compute similarity stats to detect embedding collapse
                with torch.no_grad():
                    sim_matrix = torch.mm(alpha_local.float(), beta_local.float().t())
                    diag_sim = sim_matrix.diag().mean().item()  # matched pairs
                    off_diag_mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
                    off_diag_sim = sim_matrix[off_diag_mask].mean().item()  # non-matched pairs

                debug_log(f"Step {step:3d}: loss={loss_val:.4f}, acc={acc_val:.4f}, "
                          f"temp={temperature.item():.4f}, grad_norm={grad_norm:.4f}")
                debug_log(f"         sim(matched)={diag_sim:.4f}, sim(non-matched)={off_diag_sim:.4f}")

        # Summary
        if self._is_main():
            final_loss = losses[-1]
            final_acc = accuracies[-1]
            loss_reduction = (initial_loss - final_loss) / initial_loss * 100 if initial_loss > 0 else 0

            debug_log(f"\n{'='*60}")
            debug_log("OVERFIT CHECK RESULTS")
            debug_log(f"{'='*60}")
            debug_log(f"Initial loss: {initial_loss:.4f}")
            debug_log(f"Final loss:   {final_loss:.4f}")
            debug_log(f"Loss reduction: {loss_reduction:.1f}%")
            debug_log(f"Initial accuracy: {accuracies[0]:.4f}")
            debug_log(f"Final accuracy:   {final_acc:.4f}")
            debug_log("")

            if final_loss < 0.1 and final_acc > 0.95:
                debug_log("SUCCESS: Model can overfit a single batch!")
                debug_log("  The training loop and model architecture are working.")
            elif loss_reduction > 50:
                debug_log("PARTIAL: Loss decreased but didn't fully converge.")
                debug_log("  Training is working but may need more steps or tuning.")
            elif loss_reduction > 10:
                debug_log("SLOW: Loss is decreasing slowly.")
                debug_log("  Check learning rate, optimizer, or gradient flow.")
            else:
                debug_log("FAILURE: Model cannot overfit a single batch!")
                debug_log("  Possible issues:")
                debug_log("  - Gradients not flowing (check requires_grad)")
                debug_log("  - Learning rate too low")
                debug_log("  - Loss function issue")
                debug_log("  - Model architecture problem")

            # Always show diagnostics
            debug_log("\n[Additional Diagnostics]")
            base_model = self.model.module if hasattr(self.model, 'module') else self.model

            # Check trainable parameters by component
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            debug_log(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

            # Break down by component
            debug_log("\n  [Trainable params by component]")
            for name, param in base_model.named_parameters():
                if param.requires_grad:
                    debug_log(f"    {name}: {param.numel():,} (grad_fn={param.grad is not None})")

            # Check gradient magnitudes
            debug_log("\n  [Gradient magnitudes]")
            has_grad = False
            grad_stats = []
            for name, param in base_model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm = param.grad.abs().mean().item()
                    if grad_norm > 0:
                        has_grad = True
                        grad_stats.append((name, grad_norm))

            # Show top 5 largest gradients
            grad_stats.sort(key=lambda x: x[1], reverse=True)
            for name, grad_norm in grad_stats[:5]:
                debug_log(f"    {name}: {grad_norm:.6f}")
            if not grad_stats:
                debug_log("    No gradients found!")

            debug_log(f"\n  Gradients present: {has_grad}")

            # Check embedding similarity
            with torch.no_grad():
                outputs = self.model(alpha_ids, alpha_mask, beta_ids, beta_mask)
                alpha_emb = outputs["alpha_embeddings"]
                beta_emb = outputs["beta_embeddings"]

                # Cosine similarity of matched pairs
                cos_sim = (alpha_emb * beta_emb).sum(dim=1).mean().item()
                debug_log(f"  Avg cosine sim (matched pairs): {cos_sim:.4f}")

                # Check if embeddings are collapsed (all same)
                alpha_std = alpha_emb.std().item()
                beta_std = beta_emb.std().item()
                debug_log(f"  Alpha embedding std: {alpha_std:.6f}")
                debug_log(f"  Beta embedding std: {beta_std:.6f}")
                if alpha_std < 0.01 or beta_std < 0.01:
                    debug_log("  WARNING: Embeddings may be collapsed!")

            debug_log(f"{'='*60}\n")

        return losses, accuracies

    def train(self):
        config = self.config

        # Run overfit check if requested
        if config.get("overfit_check", False):
            self.overfit_single_batch(num_steps=config.get("overfit_steps", 100))
            return

        num_epochs = config.get("num_epochs", 10)
        grad_accum = config.get("gradient_accumulation_steps", 1)
        logging_steps = config.get("logging_steps", 50)
        eval_steps = config.get("eval_steps", 500)
        save_steps = config.get("save_steps", 500)
        refresh_steps = config.get("refresh_distractors_steps", 2000)
        num_distractors = config.get("num_distractors", 1000)
        
        if self._is_main():
            print(f"\n{'='*60}")
            print(f"Starting training for {num_epochs} epochs")
            print(f"  Batch size: {config.get('batch_size', 64)}")
            print(f"  Gradient accumulation: {grad_accum}")
            print(f"  World size: {self.world_size}")
            print(f"  Effective batch size: {config.get('batch_size', 64) * self.world_size * grad_accum}")
            print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            if hasattr(self.train_sampler, 'set_epoch'):
                self.train_sampler.set_epoch(epoch)
            
            self._train_epoch(
                epoch, grad_accum, logging_steps, eval_steps, 
                save_steps, refresh_steps, num_distractors
            )
            
            # Epoch-end evaluation
            metrics = self._validate()
            if self._is_main():
                print(f"\nEpoch {epoch+1} - Recall@10: {metrics.get('recall@10', 0):.4f}, "
                      f"MRR: {metrics.get('mrr', 0):.4f}")
            
            # Early stopping
            if self.early_stopping:
                if self.early_stopping(metrics.get("recall@10", 0)):
                    if self._is_main():
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        
        self._save_final()
        
        if self.use_wandb:
            wandb.finish()
    
    def _train_epoch(
        self,
        epoch: int,
        grad_accum: int,
        logging_steps: int,
        eval_steps: int,
        save_steps: int,
        refresh_steps: int,
        num_distractors: int,
    ):
        self.model.train()
        
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch+1}",
            disable=not self._is_main(),
        )
        
        self.optimizer.zero_grad()
        
        for step, batch in pbar:
            # Unpack batch
            alpha_ids = batch["alpha_input_ids"].to(self.device)
            alpha_mask = batch["alpha_attention_mask"].to(self.device)
            beta_ids = batch["beta_input_ids"].to(self.device)
            beta_mask = batch["beta_attention_mask"].to(self.device)
            alpha_sim_mask = batch["alpha_sim_mask"]
            beta_sim_mask = batch["beta_sim_mask"]
            
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                # Forward
                outputs = self.model(alpha_ids, alpha_mask, beta_ids, beta_mask)
                
                alpha_local = outputs["alpha_embeddings"]
                beta_local = outputs["beta_embeddings"]
                temperature = outputs["temperature"]
                
                # Gather across GPUs
                alpha_all = gather_embeddings(alpha_local, self.world_size)
                beta_all = gather_embeddings(beta_local, self.world_size)
                
                # Sample distractors
                if self.distractor_manager:
                    ext_alpha, ext_beta = self.distractor_manager.sample(
                        num_distractors, num_distractors
                    )
                else:
                    ext_alpha = torch.empty(0, 256, device=self.device)
                    ext_beta = torch.empty(0, 256, device=self.device)
                
                # Loss
                loss_dict = self.criterion(
                    alpha_local, beta_local,
                    alpha_sim_mask, beta_sim_mask,
                    alpha_all, beta_all,
                    ext_alpha, ext_beta,
                    temperature,
                )
                loss = loss_dict["loss"] / grad_accum
            
            # Backward
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss_dict["loss"].item()
            total_acc += loss_dict["accuracy"].item()
            num_batches += 1
            
            # Optimizer step
            if (step + 1) % grad_accum == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Update momentum encoder
                if self.distractor_manager:
                    self.distractor_manager.update_momentum(self.model)
                
                pbar.set_postfix({
                    "loss": f"{loss_dict['loss'].item():.4f}",
                    "acc": f"{loss_dict['accuracy'].item():.4f}",
                    "temp": f"{temperature.item():.4f}",
                })
                
                # Logging
                if self.global_step % logging_steps == 0 and self._is_main():
                    if self.use_wandb:
                        wandb.log({
                            "train/loss": total_loss / num_batches,
                            "train/accuracy": total_acc / num_batches,
                            "train/temperature": temperature.item(),
                            "train/lr": self.scheduler.get_last_lr()[0],
                            "train/step": self.global_step,
                        })
                
                # Refresh distractors
                if self.global_step % refresh_steps == 0 and self.distractor_manager:
                    self.distractor_manager.refresh()
                
                # Evaluation
                if self.global_step % eval_steps == 0:
                    metrics = self._validate()
                    if self._is_main():
                        print(f"\nStep {self.global_step} - Recall@10: {metrics.get('recall@10', 0):.4f}")
                        
                        if self.use_wandb:
                            wandb.log({f"eval/{k}": v for k, v in metrics.items()})
                        
                        if metrics.get("recall@10", 0) > self.best_metric:
                            self.best_metric = metrics["recall@10"]
                            self._save_checkpoint(is_best=True)
                    
                    self.model.train()
                
                # Checkpoint
                if self.global_step % save_steps == 0:
                    self._save_checkpoint()
        
        pbar.close()
    
    def _validate(self) -> Dict[str, float]:
        return self.evaluator.evaluate(
            self.model,
            self.val_dataset,
            self.tokenizer,
            self.device,
        )
    
    def _save_checkpoint(self, is_best: bool = False):
        if not self._is_main():
            return
        
        output_dir = self.config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        model_to_save = self.model.module if self.is_distributed else self.model
        
        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config,
        }
        
        path = os.path.join(output_dir, f"checkpoint-{self.global_step}.pt")
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
        
        if is_best:
            best_path = os.path.join(output_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"New best model (recall@10={self.best_metric:.4f})")
        
        # Cleanup old checkpoints (keep 3)
        checkpoints = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*.pt")))
        for old in checkpoints[:-3]:
            os.remove(old)
    
    def _save_final(self):
        if not self._is_main():
            return
        
        final_dir = os.path.join(self.config["output_dir"], "final")
        os.makedirs(final_dir, exist_ok=True)
        
        model_to_save = self.model.module if self.is_distributed else self.model
        
        torch.save(
            {"model_state_dict": model_to_save.state_dict(), "config": self.config},
            os.path.join(final_dir, "model.pt"),
        )
        self.tokenizer.save_pretrained(final_dir)
        
        print(f"\nFinal model saved to: {final_dir}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="TCR Alpha-Beta Pairing with Robust Contrastive Learning (V3)"
    )
    
    # Required
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    
    # Data
    parser.add_argument("--paired_permutation_keys", type=str, nargs="+", default=["tra_trb"])
    parser.add_argument("--alpha_distractor_keys", type=str, nargs="+", default=["tra"])
    parser.add_argument("--beta_distractor_keys", type=str, nargs="+", default=["trb"])
    parser.add_argument("--max_length", type=int, default=320)
    parser.add_argument("--distractor_sample_size", type=int, default=500_000)
    parser.add_argument("--similarity_threshold", type=float, default=0.9)
    parser.add_argument("--var_region_len", type=int, default=150,
                        help="Truncate sequences to this length before encoding (variable region only). "
                             "Set to 0 to disable truncation.")

    # Model
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--pretrained_checkpoint", type=str, default=None,
                        help="Path to pretrained PEFT checkpoint (.pt file with model_state_dict)")
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--pooling", type=str, default="attention", choices=["attention", "cls", "mean"])
    parser.add_argument("--projection_layers", type=int, default=3)
    parser.add_argument("--use_batchnorm", action="store_true", default=False,
                        help="Use BatchNorm1d instead of LayerNorm in projection head. "
                             "BatchNorm can help break embedding collapse by normalizing across batch dimension.")
    parser.add_argument("--temperature", type=float, default=0.07)

    # LoRA
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", action="store_false", dest="use_lora",
                        help="Disable LoRA (use frozen encoder for debugging)")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--unfreeze_layers", type=int, default=0,
                        help="Number of last encoder layers to unfreeze when LoRA is disabled. "
                             "ESM2-650M has 33 layers. Set to 0 to freeze all (projection only).")
    
    # Training
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_distractors", type=int, default=1000)
    parser.add_argument("--initial_distractor_pool", type=int, default=50_000)
    
    # Curriculum & Features
    parser.add_argument("--use_curriculum", action="store_true", default=True)
    parser.add_argument("--no_curriculum", action="store_false", dest="use_curriculum")
    parser.add_argument("--symmetric_loss", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--use_amp", action="store_true", default=True)
    
    # Evaluation & Logging
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--refresh_distractors_steps", type=int, default=2000)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Early Stopping
    parser.add_argument("--early_stopping", action="store_true", default=True)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    
    # Wandb
    parser.add_argument("--report_to", type=str, default="wandb", choices=["wandb", "none"])
    parser.add_argument("--wandb_project", type=str, default="tcr-pairing-v3")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    # Debug / Overfit check
    parser.add_argument("--overfit_check", action="store_true", default=False,
                        help="Run single batch overfit check instead of full training")
    parser.add_argument("--overfit_steps", type=int, default=100,
                        help="Number of steps for overfit check")
    parser.add_argument("--overfit_lr", type=float, default=None,
                        help="Learning rate for overfit check (default: 10x normal lr)")
    parser.add_argument("--debug_log_file", type=str, default=None,
                        help="Path to write debug logs (default: output_dir/debug.log)")

    return parser.parse_args()


def main():
    args = parse_args()
    config = vars(args)

    trainer = TCRContrastiveTrainer(config)
    trainer.train()
    
    if trainer.is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()