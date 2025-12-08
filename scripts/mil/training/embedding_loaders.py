#!/usr/bin/env python3
"""
Embedding loaders for different formats (HDF5 and Parquet).

Parquet is actually more efficient for:
- Sequential reads
- Compression
- Columnar storage

HDF5 is better for:
- Random access patterns
- Very frequent small reads
"""

import os
import numpy as np
import pickle
import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm


class ParquetEmbeddingLoader:
    """Load embeddings from Parquet file (recommended)."""

    def __init__(self, parquet_path: str, max_cache_size: int = 500000):
        self.parquet_path = parquet_path
        self.seq_to_idx = {}
        self.embedding_dim = None
        self._parquet_file = None
        self._cache = {}  # Cache for frequently accessed embeddings
        self._max_cache_size = max_cache_size  # Increased default from 10k to 500k
        self._load_index()

    def _load_index(self):
        """Load or build sequence index."""
        index_path = self.parquet_path + '.idx.pkl'

        if os.path.exists(index_path):
            with open(index_path, 'rb') as f:
                self.seq_to_idx = pickle.load(f)

            # Get embedding dim from first row
            pf = pq.ParquetFile(self.parquet_path)
            first_batch = pf.read_row_group(0, columns=['embedding']).to_pandas()
            self.embedding_dim = len(first_batch['embedding'].iloc[0])

            print(f"Loaded index: {len(self.seq_to_idx):,} sequences, dim={self.embedding_dim}")
        else:
            # Build index
            print("Building index from parquet...")
            pf = pq.ParquetFile(self.parquet_path)

            idx = 0
            for i in tqdm(range(pf.num_row_groups), desc="Indexing"):
                batch = pf.read_row_group(i).to_pandas()

                if self.embedding_dim is None:
                    self.embedding_dim = len(batch['embedding'].iloc[0])

                for seq in batch['sequence_id']:
                    self.seq_to_idx[seq] = idx
                    idx += 1

            # Save index
            with open(index_path, 'wb') as f:
                pickle.dump(self.seq_to_idx, f)

            print(f"Built and saved index: {len(self.seq_to_idx):,} sequences")

    def get_batch(self, sequences: list) -> np.ndarray:
        """
        Get embeddings for a batch of sequences.

        Uses caching for frequently accessed sequences.
        """
        embeddings = []
        missing_seqs = []
        missing_indices = []

        # Check cache first
        for i, seq in enumerate(sequences):
            if seq in self._cache:
                embeddings.append((i, self._cache[seq]))
            elif seq in self.seq_to_idx:
                missing_seqs.append(seq)
                missing_indices.append((i, self.seq_to_idx[seq]))

        # Load missing embeddings from parquet
        if missing_seqs:
            if self._parquet_file is None:
                self._parquet_file = pq.ParquetFile(self.parquet_path)

            # Group by row group for efficient reading
            rowgroup_indices = {}
            for seq_idx, global_idx in missing_indices:
                # Determine which row group this index belongs to
                rows_per_group = self._parquet_file.metadata.row_group(0).num_rows
                rowgroup = global_idx // rows_per_group
                local_idx = global_idx % rows_per_group

                if rowgroup not in rowgroup_indices:
                    rowgroup_indices[rowgroup] = []
                rowgroup_indices[rowgroup].append((seq_idx, local_idx))

            # Read row groups
            for rowgroup, indices in rowgroup_indices.items():
                batch_df = self._parquet_file.read_row_group(rowgroup).to_pandas()

                for seq_idx, local_idx in indices:
                    emb = np.array(batch_df['embedding'].iloc[local_idx], dtype=np.float32)
                    embeddings.append((seq_idx, emb))

                    # Add to cache if not full
                    if len(self._cache) < self._max_cache_size:
                        self._cache[sequences[seq_idx]] = emb

        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        result = np.array([emb for _, emb in embeddings], dtype=np.float32)

        if len(result) == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        return result

    def prefetch_sequences(self, sequences: set):
        """
        Prefetch embeddings for a set of sequences into cache.

        This is much faster than loading on-demand during training.
        Call this once before training with all sequences in your dataset.
        """
        print(f"Prefetching {len(sequences):,} sequences into cache...")

        # Get indices for all sequences
        seq_list = []
        indices = []
        for seq in sequences:
            if seq in self.seq_to_idx and seq not in self._cache:
                seq_list.append(seq)
                indices.append(self.seq_to_idx[seq])

        if not indices:
            print("All sequences already in cache!")
            return

        # Estimate memory usage
        mem_gb = len(indices) * self.embedding_dim * 4 / (1024**3)
        print(f"Loading {len(indices):,} new sequences (estimated memory: {mem_gb:.2f} GB)...")

        if self._parquet_file is None:
            self._parquet_file = pq.ParquetFile(self.parquet_path)

        # Group by row group
        rows_per_group = self._parquet_file.metadata.row_group(0).num_rows
        rowgroup_map = {}

        for seq, idx in zip(seq_list, indices):
            rowgroup = idx // rows_per_group
            local_idx = idx % rows_per_group

            if rowgroup not in rowgroup_map:
                rowgroup_map[rowgroup] = []
            rowgroup_map[rowgroup].append((seq, local_idx))

        # Read row groups in order
        for rowgroup in tqdm(sorted(rowgroup_map.keys()), desc="Loading row groups"):
            batch_df = self._parquet_file.read_row_group(rowgroup).to_pandas()

            for seq, local_idx in rowgroup_map[rowgroup]:
                emb = np.array(batch_df['embedding'].iloc[local_idx], dtype=np.float32)
                self._cache[seq] = emb

        print(f"✓ Prefetched {len(indices):,} sequences. Cache now contains {len(self._cache):,} sequences.")

    def close(self):
        """Close parquet file if open."""
        if self._parquet_file is not None:
            # Parquet files don't need explicit closing
            self._parquet_file = None


class HDF5EmbeddingLoader:
    """Load embeddings from HDF5 file (legacy support)."""

    def __init__(self, h5_path: str):
        import h5py

        self.h5_path = h5_path
        self.h5file = None
        self.seq_to_idx = {}
        self.embedding_dim = None
        self._load_index()

    def _load_index(self):
        """Load sequence index."""
        import h5py

        index_path = self.h5_path + '.idx.pkl'

        if os.path.exists(index_path):
            with open(index_path, 'rb') as f:
                self.seq_to_idx = pickle.load(f)

            with h5py.File(self.h5_path, 'r') as f:
                self.embedding_dim = f['embeddings'].shape[1]

            print(f"Loaded index: {len(self.seq_to_idx):,} sequences, dim={self.embedding_dim}")
        else:
            # Build index
            print("Building index...")
            with h5py.File(self.h5_path, 'r') as f:
                sequences = f['sequences'][:]
                for idx, seq in enumerate(tqdm(sequences)):
                    if isinstance(seq, bytes):
                        seq = seq.decode('utf-8')
                    self.seq_to_idx[seq] = idx
                self.embedding_dim = f['embeddings'].shape[1]

            # Save index
            with open(index_path, 'wb') as f:
                pickle.dump(self.seq_to_idx, f)
            print(f"Built and saved index: {len(self.seq_to_idx):,} sequences")

    def get_batch(self, sequences: list) -> np.ndarray:
        """Get embeddings for sequences."""
        import h5py

        if self.h5file is None:
            self.h5file = h5py.File(self.h5_path, 'r')

        embeddings = []
        for seq in sequences:
            if seq in self.seq_to_idx:
                idx = self.seq_to_idx[seq]
                embeddings.append(self.h5file['embeddings'][idx])

        if not embeddings:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        return np.array(embeddings, dtype=np.float32)

    def prefetch_sequences(self, sequences: set):
        """
        Prefetch embeddings for a set of sequences.

        Note: HDF5 doesn't have a cache, so this is a no-op.
        Included for API compatibility with ParquetEmbeddingLoader.
        """
        print(f"Prefetch requested for {len(sequences):,} sequences, but HDF5 loader doesn't cache.")
        print("Consider using ParquetEmbeddingLoader for better performance.")

    def close(self):
        if self.h5file is not None:
            self.h5file.close()


def get_embedding_loader(path: str):
    """
    Factory function to get appropriate loader based on file extension.

    Args:
        path: Path to embedding file (.parquet or .h5)

    Returns:
        EmbeddingLoader instance
    """
    if path.endswith('.parquet'):
        return ParquetEmbeddingLoader(path)
    elif path.endswith('.h5') or path.endswith('.hdf5'):
        return HDF5EmbeddingLoader(path)
    else:
        raise ValueError(f"Unsupported embedding format: {path}")
