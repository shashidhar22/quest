#!/usr/bin/env python
"""
Baseline Benchmark Script for TCR-Peptide Specificity Prediction

This script implements simple neural network baselines (LSTM and MLP) for
predicting peptide specificity given TCR sequences and MHC information.
These serve as comparison baselines against more sophisticated models.
"""

import argparse
import json
import os
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# Task configurations for all 6 tasks
TASK_CONFIGS = {
    "tra_peptide_mhc_one": {
        "mhc_class": "class_one",
        "tcr_cols": ["tra"],
        "mhc_cols": ["mhc_one_id"],
        "description": "TCR alpha only + MHC-I -> peptide",
    },
    "trb_peptide_mhc_one": {
        "mhc_class": "class_one",
        "tcr_cols": ["trb"],
        "mhc_cols": ["mhc_one_id"],
        "description": "TCR beta only + MHC-I -> peptide",
    },
    "tra_trb_peptide_mhc_one": {
        "mhc_class": "class_one",
        "tcr_cols": ["tra", "trb"],
        "mhc_cols": ["mhc_one_id"],
        "description": "TCR alpha+beta paired + MHC-I -> peptide",
    },
    "tra_peptide_mhc_two": {
        "mhc_class": "class_two",
        "tcr_cols": ["tra"],
        "mhc_cols": ["mhc_one_id", "mhc_two_id"],
        "description": "TCR alpha only + MHC-II -> peptide",
    },
    "trb_peptide_mhc_two": {
        "mhc_class": "class_two",
        "tcr_cols": ["trb"],
        "mhc_cols": ["mhc_one_id", "mhc_two_id"],
        "description": "TCR beta only + MHC-II -> peptide",
    },
    "tra_trb_peptide_mhc_two": {
        "mhc_class": "class_two",
        "tcr_cols": ["tra", "trb"],
        "mhc_cols": ["mhc_one_id", "mhc_two_id"],
        "description": "TCR alpha+beta paired + MHC-II -> peptide",
    },
}

TEST_SPLITS = [
    "test_seen_epitope",
    "test_unseen_tcr_seen_epitope",
    "test_unseen_epitope",
    "test_unseen_allele",
]

# Amino acid vocabulary
AA_VOCAB = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
AA_VOCAB["<PAD>"] = 0
AA_VOCAB["<UNK>"] = 21
VOCAB_SIZE = 22  # 20 amino acids + padding + unknown

# Extended vocabulary for Seq2Seq models
SEQ2SEQ_SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<BOS>": 22,  # Beginning of sequence
    "<EOS>": 23,  # End of sequence
    "<SEP>": 24,  # Separator between TCR chains
}
SEQ2SEQ_VOCAB_SIZE = 25  # 20 AA + UNK + PAD + BOS + EOS + SEP

# Reverse vocab for decoding
ID_TO_AA = {v: k for k, v in AA_VOCAB.items()}
ID_TO_AA[SEQ2SEQ_SPECIAL_TOKENS["<BOS>"]] = "<BOS>"
ID_TO_AA[SEQ2SEQ_SPECIAL_TOKENS["<EOS>"]] = "<EOS>"
ID_TO_AA[SEQ2SEQ_SPECIAL_TOKENS["<SEP>"]] = "<SEP>"

# =============================================================================
# Biology Constants for Sequence Metrics
# =============================================================================

# BLOSUM62 substitution matrix (symmetric)
# Source: Henikoff & Henikoff (1992)
BLOSUM62 = {
    'A': {'A': 4, 'R': -1, 'N': -2, 'D': -2, 'C': 0, 'Q': -1, 'E': -1, 'G': 0, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 0, 'W': -3, 'Y': -2, 'V': 0},
    'R': {'A': -1, 'R': 5, 'N': 0, 'D': -2, 'C': -3, 'Q': 1, 'E': 0, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 2, 'M': -1, 'F': -3, 'P': -2, 'S': -1, 'T': -1, 'W': -3, 'Y': -2, 'V': -3},
    'N': {'A': -2, 'R': 0, 'N': 6, 'D': 1, 'C': -3, 'Q': 0, 'E': 0, 'G': 0, 'H': 1, 'I': -3, 'L': -3, 'K': 0, 'M': -2, 'F': -3, 'P': -2, 'S': 1, 'T': 0, 'W': -4, 'Y': -2, 'V': -3},
    'D': {'A': -2, 'R': -2, 'N': 1, 'D': 6, 'C': -3, 'Q': 0, 'E': 2, 'G': -1, 'H': -1, 'I': -3, 'L': -4, 'K': -1, 'M': -3, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -4, 'Y': -3, 'V': -3},
    'C': {'A': 0, 'R': -3, 'N': -3, 'D': -3, 'C': 9, 'Q': -3, 'E': -4, 'G': -3, 'H': -3, 'I': -1, 'L': -1, 'K': -3, 'M': -1, 'F': -2, 'P': -3, 'S': -1, 'T': -1, 'W': -2, 'Y': -2, 'V': -1},
    'Q': {'A': -1, 'R': 1, 'N': 0, 'D': 0, 'C': -3, 'Q': 5, 'E': 2, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 1, 'M': 0, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -2, 'Y': -1, 'V': -2},
    'E': {'A': -1, 'R': 0, 'N': 0, 'D': 2, 'C': -4, 'Q': 2, 'E': 5, 'G': -2, 'H': 0, 'I': -3, 'L': -3, 'K': 1, 'M': -2, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
    'G': {'A': 0, 'R': -2, 'N': 0, 'D': -1, 'C': -3, 'Q': -2, 'E': -2, 'G': 6, 'H': -2, 'I': -4, 'L': -4, 'K': -2, 'M': -3, 'F': -3, 'P': -2, 'S': 0, 'T': -2, 'W': -2, 'Y': -3, 'V': -3},
    'H': {'A': -2, 'R': 0, 'N': 1, 'D': -1, 'C': -3, 'Q': 0, 'E': 0, 'G': -2, 'H': 8, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -1, 'P': -2, 'S': -1, 'T': -2, 'W': -2, 'Y': 2, 'V': -3},
    'I': {'A': -1, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -3, 'E': -3, 'G': -4, 'H': -3, 'I': 4, 'L': 2, 'K': -3, 'M': 1, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -3, 'Y': -1, 'V': 3},
    'L': {'A': -1, 'R': -2, 'N': -3, 'D': -4, 'C': -1, 'Q': -2, 'E': -3, 'G': -4, 'H': -3, 'I': 2, 'L': 4, 'K': -2, 'M': 2, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -2, 'Y': -1, 'V': 1},
    'K': {'A': -1, 'R': 2, 'N': 0, 'D': -1, 'C': -3, 'Q': 1, 'E': 1, 'G': -2, 'H': -1, 'I': -3, 'L': -2, 'K': 5, 'M': -1, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
    'M': {'A': -1, 'R': -1, 'N': -2, 'D': -3, 'C': -1, 'Q': 0, 'E': -2, 'G': -3, 'H': -2, 'I': 1, 'L': 2, 'K': -1, 'M': 5, 'F': 0, 'P': -2, 'S': -1, 'T': -1, 'W': -1, 'Y': -1, 'V': 1},
    'F': {'A': -2, 'R': -3, 'N': -3, 'D': -3, 'C': -2, 'Q': -3, 'E': -3, 'G': -3, 'H': -1, 'I': 0, 'L': 0, 'K': -3, 'M': 0, 'F': 6, 'P': -4, 'S': -2, 'T': -2, 'W': 1, 'Y': 3, 'V': -1},
    'P': {'A': -1, 'R': -2, 'N': -2, 'D': -1, 'C': -3, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -4, 'P': 7, 'S': -1, 'T': -1, 'W': -4, 'Y': -3, 'V': -2},
    'S': {'A': 1, 'R': -1, 'N': 1, 'D': 0, 'C': -1, 'Q': 0, 'E': 0, 'G': 0, 'H': -1, 'I': -2, 'L': -2, 'K': 0, 'M': -1, 'F': -2, 'P': -1, 'S': 4, 'T': 1, 'W': -3, 'Y': -2, 'V': -2},
    'T': {'A': 0, 'R': -1, 'N': 0, 'D': -1, 'C': -1, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 5, 'W': -2, 'Y': -2, 'V': 0},
    'W': {'A': -3, 'R': -3, 'N': -4, 'D': -4, 'C': -2, 'Q': -2, 'E': -3, 'G': -2, 'H': -2, 'I': -3, 'L': -2, 'K': -3, 'M': -1, 'F': 1, 'P': -4, 'S': -3, 'T': -2, 'W': 11, 'Y': 2, 'V': -3},
    'Y': {'A': -2, 'R': -2, 'N': -2, 'D': -3, 'C': -2, 'Q': -1, 'E': -2, 'G': -3, 'H': 2, 'I': -1, 'L': -1, 'K': -2, 'M': -1, 'F': 3, 'P': -3, 'S': -2, 'T': -2, 'W': 2, 'Y': 7, 'V': -1},
    'V': {'A': 0, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -2, 'E': -2, 'G': -3, 'H': -3, 'I': 3, 'L': 1, 'K': -2, 'M': 1, 'F': -1, 'P': -2, 'S': -2, 'T': 0, 'W': -3, 'Y': -1, 'V': 4},
}

# Kyte-Doolittle hydrophobicity scale
# Positive = hydrophobic, Negative = hydrophilic
KYTE_DOOLITTLE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}

# Amino acid charge at physiological pH (~7.4)
AA_CHARGE = {
    'D': -1.0, 'E': -1.0,  # Acidic (negative)
    'K': 1.0, 'R': 1.0, 'H': 0.1,  # Basic (positive, H partial)
    'A': 0.0, 'N': 0.0, 'C': 0.0, 'Q': 0.0, 'G': 0.0,
    'I': 0.0, 'L': 0.0, 'M': 0.0, 'F': 0.0, 'P': 0.0,
    'S': 0.0, 'T': 0.0, 'W': 0.0, 'Y': 0.0, 'V': 0.0,
}


# =============================================================================
# Sequence Encoding Functions
# =============================================================================


def encode_sequence(seq: str, max_len: int = 512) -> Tuple[List[int], int]:
    """
    Convert amino acid sequence to token IDs.

    Args:
        seq: Amino acid sequence string
        max_len: Maximum sequence length

    Returns:
        Tuple of (token IDs list, actual length)
    """
    tokens = [AA_VOCAB.get(aa.upper(), AA_VOCAB["<UNK>"]) for aa in seq]
    length = len(tokens)

    # Pad or truncate
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
        length = max_len
    else:
        tokens = tokens + [0] * (max_len - len(tokens))

    return tokens, length


def encode_aa_frequencies(seq: str) -> np.ndarray:
    """
    Compute normalized amino acid frequencies (bag-of-AAs).

    Args:
        seq: Amino acid sequence string

    Returns:
        Normalized frequency vector of shape (21,)
    """
    counts = np.zeros(21)  # 20 AAs + unknown
    for aa in seq.upper():
        idx = AA_VOCAB.get(aa, AA_VOCAB["<UNK>"]) - 1  # Shift by 1 (skip padding)
        if 0 <= idx < 21:
            counts[idx] += 1

    # Normalize
    total = counts.sum()
    if total > 0:
        counts /= total

    return counts


# =============================================================================
# Dataset Class
# =============================================================================


class TCRPeptideDataset(Dataset):
    """Dataset for TCR-peptide classification."""

    def __init__(
        self,
        df: pd.DataFrame,
        task_config: dict,
        label_encoder: LabelEncoder,
        max_seq_len: int = 512,
        encoding: str = "tokens",  # "tokens" for LSTM, "frequencies" for MLP
    ):
        """
        Initialize dataset.

        Args:
            df: DataFrame with TCR and peptide data
            task_config: Task configuration dictionary
            label_encoder: Fitted LabelEncoder for peptide labels
            max_seq_len: Maximum sequence length for tokenization
            encoding: "tokens" for LSTM or "frequencies" for MLP
        """
        self.df = df.reset_index(drop=True)
        self.task_config = task_config
        self.label_encoder = label_encoder
        self.max_seq_len = max_seq_len
        self.encoding = encoding

        # Encode labels - handle unseen labels
        self.labels = []
        self.valid_indices = []
        for idx, peptide in enumerate(df["peptide"].values):
            if peptide in label_encoder.classes_:
                self.labels.append(label_encoder.transform([peptide])[0])
                self.valid_indices.append(idx)

        self.labels = np.array(self.labels)
        self.valid_mask = np.isin(np.arange(len(df)), self.valid_indices)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.df.iloc[actual_idx]

        # Concatenate TCR sequences with separator
        sequences = []
        for col in self.task_config["tcr_cols"]:
            if col in self.df.columns and pd.notna(row[col]):
                sequences.append(str(row[col]).upper())

        # Add MHC allele as string (simplified encoding)
        for col in self.task_config["mhc_cols"]:
            if col in self.df.columns and pd.notna(row[col]):
                # Remove special characters from MHC for tokenization
                mhc_str = str(row[col]).replace("*", "").replace(":", "")
                sequences.append(mhc_str)

        combined_seq = "".join(sequences)

        if self.encoding == "tokens":
            tokens, length = encode_sequence(combined_seq, self.max_seq_len)
            return {
                "input_ids": torch.tensor(tokens, dtype=torch.long),
                "length": torch.tensor(length, dtype=torch.long),
                "label": torch.tensor(self.labels[idx], dtype=torch.long),
            }
        else:
            # Frequency encoding for MLP
            freq = encode_aa_frequencies(combined_seq)
            return {
                "features": torch.tensor(freq, dtype=torch.float32),
                "label": torch.tensor(self.labels[idx], dtype=torch.long),
            }


class TCRPeptideSeq2SeqDataset(Dataset):
    """Dataset for generative peptide prediction (Seq2Seq)."""

    def __init__(
        self,
        df: pd.DataFrame,
        task_config: dict,
        max_src_len: int = 512,
        max_tgt_len: int = 30,
    ):
        """
        Initialize Seq2Seq dataset.

        Args:
            df: DataFrame with TCR and peptide data
            task_config: Task configuration dictionary
            max_src_len: Maximum source (TCR+MHC) sequence length
            max_tgt_len: Maximum target (peptide) sequence length
        """
        self.df = df.reset_index(drop=True)
        self.task_config = task_config
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        # Filter out rows with missing peptides
        self.valid_indices = [
            i for i, row in df.iterrows() if pd.notna(row.get("peptide"))
        ]

    def __len__(self):
        return len(self.valid_indices)

    def _encode_sequence_seq2seq(self, seq: str, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode sequence for Seq2Seq with padding mask."""
        tokens = [AA_VOCAB.get(aa.upper(), AA_VOCAB["<UNK>"]) for aa in seq]
        length = len(tokens)

        # Truncate or pad
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
            length = max_len
        else:
            tokens = tokens + [0] * (max_len - len(tokens))

        # Padding mask: True where padded
        padding_mask = torch.tensor([i >= length for i in range(max_len)], dtype=torch.bool)
        return torch.tensor(tokens, dtype=torch.long), padding_mask

    def _encode_peptide_with_special_tokens(self, peptide: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode peptide with BOS/EOS tokens for teacher forcing.

        Returns:
            decoder_input: BOS + peptide tokens (for decoder input)
            labels: peptide tokens + EOS (for loss computation)
            padding_mask: Mask for decoder input
        """
        peptide_tokens = [AA_VOCAB.get(aa.upper(), AA_VOCAB["<UNK>"]) for aa in peptide]

        # Decoder input: BOS + peptide (teacher forcing)
        decoder_input = [SEQ2SEQ_SPECIAL_TOKENS["<BOS>"]] + peptide_tokens
        # Labels: peptide + EOS (shifted by 1)
        labels = peptide_tokens + [SEQ2SEQ_SPECIAL_TOKENS["<EOS>"]]

        # Truncate if needed
        if len(decoder_input) > self.max_tgt_len:
            decoder_input = decoder_input[: self.max_tgt_len]
            labels = labels[: self.max_tgt_len]

        length = len(decoder_input)

        # Pad
        pad_len = self.max_tgt_len - length
        decoder_input = decoder_input + [0] * pad_len
        labels = labels + [-100] * pad_len  # -100 is ignored by CrossEntropyLoss

        # Padding mask
        padding_mask = torch.tensor([i >= length for i in range(self.max_tgt_len)], dtype=torch.bool)

        return (
            torch.tensor(decoder_input, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            padding_mask,
        )

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.df.iloc[actual_idx]

        # Build source sequence: TCR + MHC
        sequences = []
        for col in self.task_config["tcr_cols"]:
            if col in self.df.columns and pd.notna(row[col]):
                sequences.append(str(row[col]).upper())

        for col in self.task_config["mhc_cols"]:
            if col in self.df.columns and pd.notna(row[col]):
                mhc_str = str(row[col]).replace("*", "").replace(":", "")
                sequences.append(mhc_str)

        combined_src = "".join(sequences)

        # Encode source
        src_tokens, src_padding_mask = self._encode_sequence_seq2seq(combined_src, self.max_src_len)

        # Encode target peptide
        peptide = str(row["peptide"]).upper()
        decoder_input, labels, tgt_padding_mask = self._encode_peptide_with_special_tokens(peptide)

        return {
            "src": src_tokens,
            "src_padding_mask": src_padding_mask,
            "decoder_input": decoder_input,
            "labels": labels,
            "tgt_padding_mask": tgt_padding_mask,
            "reference_peptide": peptide,
        }


# =============================================================================
# Model Architectures
# =============================================================================


class LSTMClassifier(nn.Module):
    """Bi-directional LSTM for sequence classification."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 100,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        self.bidirectional = bidirectional
        hidden_factor = 2 if bidirectional else 1
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * hidden_factor, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, lengths=None):
        """
        Forward pass.

        Args:
            x: Input token IDs of shape (batch, seq_len)
            lengths: Optional tensor of actual sequence lengths

        Returns:
            Logits of shape (batch, num_classes)
        """
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        if lengths is not None:
            # Pack padded sequence for efficient LSTM processing
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        _, (hidden, _) = self.lstm(embedded)  # hidden: (num_layers*dirs, batch, hidden)

        # Concatenate last forward and backward hidden states
        if self.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            hidden = hidden[-1]

        return self.fc(hidden)  # (batch, num_classes)


class MLPClassifier(nn.Module):
    """Simple MLP using bag-of-amino-acids representation."""

    def __init__(
        self,
        input_dim: int = 21,  # AA frequencies
        hidden_dims: List[int] = None,
        num_classes: int = 100,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input features of shape (batch, input_dim)

        Returns:
            Logits of shape (batch, num_classes)
        """
        return self.network(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer models."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    """Encoder-decoder transformer for peptide generation."""

    def __init__(
        self,
        vocab_size: int = SEQ2SEQ_VOCAB_SIZE,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Shared embedding for encoder and decoder
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout)

        # Standard PyTorch Transformer
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection (weight-tied with embedding)
        self.output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
        self.output_proj.weight = self.embedding.weight  # Weight tying

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for decoder self-attention."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training with teacher forcing.

        Args:
            src: Source (encoder) input of shape (batch, src_len)
            tgt: Target (decoder) input of shape (batch, tgt_len)
            src_key_padding_mask: Padding mask for source (True = padded)
            tgt_key_padding_mask: Padding mask for target (True = padded)

        Returns:
            Logits of shape (batch, tgt_len, vocab_size)
        """
        # Embed and add positional encoding
        src_emb = self.pos_encoding(self.embedding(src))
        tgt_emb = self.pos_encoding(self.embedding(tgt))

        # Generate causal mask for decoder
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), tgt.device)

        # Transformer forward pass
        output = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        # Project to vocabulary
        logits = self.output_proj(output)
        return logits

    def encode(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode source sequence."""
        src_emb = self.pos_encoding(self.embedding(src))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        return memory

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode target sequence given encoder memory."""
        tgt_emb = self.pos_encoding(self.embedding(tgt))
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), tgt.device)
        output = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_proj(output)


# =============================================================================
# Seq2Seq Inference Functions
# =============================================================================


def greedy_decode(
    model: Seq2SeqTransformer,
    src: torch.Tensor,
    src_padding_mask: Optional[torch.Tensor] = None,
    max_len: int = 30,
    device: str = "cuda",
) -> List[str]:
    """
    Autoregressive greedy decoding for Seq2Seq generation.

    Args:
        model: Trained Seq2SeqTransformer model
        src: Source tokens of shape (batch, src_len)
        src_padding_mask: Padding mask for source (True = padded)
        max_len: Maximum generation length
        device: Device to use

    Returns:
        List of generated peptide sequences
    """
    model.eval()
    batch_size = src.size(0)
    bos_token = SEQ2SEQ_SPECIAL_TOKENS["<BOS>"]
    eos_token = SEQ2SEQ_SPECIAL_TOKENS["<EOS>"]

    with torch.no_grad():
        # Encode source
        memory = model.encode(src, src_key_padding_mask=src_padding_mask)

        # Initialize decoder input with BOS
        decoder_input = torch.full((batch_size, 1), bos_token, dtype=torch.long, device=device)

        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        generated_tokens = [[] for _ in range(batch_size)]

        for _ in range(max_len):
            # Decode one step
            logits = model.decode(
                decoder_input,
                memory,
                memory_key_padding_mask=src_padding_mask,
            )

            # Get next token (greedy)
            next_token = logits[:, -1, :].argmax(dim=-1)  # (batch,)

            # Update generated sequences
            for i in range(batch_size):
                if not finished[i]:
                    token_id = next_token[i].item()
                    if token_id == eos_token:
                        finished[i] = True
                    elif token_id != 0:  # Not padding
                        generated_tokens[i].append(token_id)

            # Check if all sequences are finished
            if finished.all():
                break

            # Append next token to decoder input
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=1)

    # Convert token IDs to sequences
    generated_sequences = []
    for tokens in generated_tokens:
        seq = ""
        for token_id in tokens:
            if token_id in ID_TO_AA:
                aa = ID_TO_AA[token_id]
                if aa not in ["<PAD>", "<BOS>", "<EOS>", "<SEP>", "<UNK>"]:
                    seq += aa
        generated_sequences.append(seq)

    return generated_sequences


# =============================================================================
# Biology-Informed Sequence Metrics
# =============================================================================


def levenshtein_distance(seq1: str, seq2: str) -> int:
    """
    Compute Levenshtein (edit) distance between two sequences.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Edit distance (insertions, deletions, substitutions)
    """
    if len(seq1) < len(seq2):
        return levenshtein_distance(seq2, seq1)

    if len(seq2) == 0:
        return len(seq1)

    previous_row = range(len(seq2) + 1)
    for i, c1 in enumerate(seq1):
        current_row = [i + 1]
        for j, c2 in enumerate(seq2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def sequence_identity(seq1: str, seq2: str) -> float:
    """
    Compute sequence identity (fraction of matching positions).

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Identity fraction (0.0 to 1.0)
    """
    if not seq1 or not seq2:
        return 0.0

    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))

    matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])

    # Normalize by max length to penalize length differences
    return matches / max_len


def blosum62_similarity(seq1: str, seq2: str) -> float:
    """
    Compute BLOSUM62 similarity score between two sequences.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Total BLOSUM62 score (can be negative)
    """
    if not seq1 or not seq2:
        return 0.0

    min_len = min(len(seq1), len(seq2))
    gap_penalty = -4  # Standard BLOSUM62 gap penalty

    score = 0.0
    for i in range(min_len):
        aa1 = seq1[i].upper()
        aa2 = seq2[i].upper()
        if aa1 in BLOSUM62 and aa2 in BLOSUM62[aa1]:
            score += BLOSUM62[aa1][aa2]
        else:
            score += gap_penalty

    # Penalize length differences
    len_diff = abs(len(seq1) - len(seq2))
    score += len_diff * gap_penalty

    return score


def blosum62_normalized(seq1: str, seq2: str) -> float:
    """
    Compute length-normalized BLOSUM62 similarity.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Normalized score (score per position)
    """
    if not seq1 or not seq2:
        return 0.0

    score = blosum62_similarity(seq1, seq2)
    max_len = max(len(seq1), len(seq2))
    return score / max_len


def compute_hydrophobicity_profile(seq: str) -> np.ndarray:
    """
    Compute per-position hydrophobicity using Kyte-Doolittle scale.

    Args:
        seq: Amino acid sequence

    Returns:
        Array of hydrophobicity values per position
    """
    profile = []
    for aa in seq.upper():
        profile.append(KYTE_DOOLITTLE.get(aa, 0.0))
    return np.array(profile)


def hydrophobicity_correlation(seq1: str, seq2: str) -> float:
    """
    Compute Pearson correlation between hydrophobicity profiles.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Correlation coefficient (-1 to 1), or 0 if cannot compute
    """
    if not seq1 or not seq2:
        return 0.0

    # Truncate to same length for comparison
    min_len = min(len(seq1), len(seq2))
    if min_len < 3:
        return 0.0

    profile1 = compute_hydrophobicity_profile(seq1[:min_len])
    profile2 = compute_hydrophobicity_profile(seq2[:min_len])

    # Compute Pearson correlation
    if np.std(profile1) < 1e-8 or np.std(profile2) < 1e-8:
        return 0.0

    corr = np.corrcoef(profile1, profile2)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def compute_net_charge(seq: str) -> float:
    """
    Compute net charge of a sequence at physiological pH.

    Args:
        seq: Amino acid sequence

    Returns:
        Net charge
    """
    return sum(AA_CHARGE.get(aa.upper(), 0.0) for aa in seq)


def compute_generation_metrics(generated: List[str], references: List[str]) -> Dict:
    """
    Compute metrics for generative models.

    Args:
        generated: List of generated peptide sequences
        references: List of reference peptide sequences

    Returns:
        Dictionary of metrics
    """
    assert len(generated) == len(references), "Length mismatch"

    n = len(generated)
    if n == 0:
        return {}

    # Initialize accumulators
    exact_matches = 0
    identities = []
    levenshtein_distances = []
    blosum_scores = []
    hydro_corrs = []
    charge_diffs = []
    length_diffs = []

    for gen, ref in zip(generated, references):
        # Exact match
        if gen == ref:
            exact_matches += 1

        # Sequence similarity
        identities.append(sequence_identity(gen, ref))
        levenshtein_distances.append(levenshtein_distance(gen, ref))
        blosum_scores.append(blosum62_normalized(gen, ref))

        # Biochemical properties
        if len(gen) >= 3 and len(ref) >= 3:
            hydro_corrs.append(hydrophobicity_correlation(gen, ref))

        gen_charge = compute_net_charge(gen)
        ref_charge = compute_net_charge(ref)
        charge_diffs.append(abs(gen_charge - ref_charge))

        # Length
        length_diffs.append(abs(len(gen) - len(ref)))

    metrics = {
        "exact_match": exact_matches / n,
        "mean_sequence_identity": float(np.mean(identities)),
        "mean_levenshtein": float(np.mean(levenshtein_distances)),
        "mean_blosum62_normalized": float(np.mean(blosum_scores)),
        "mean_hydrophobicity_corr": float(np.mean(hydro_corrs)) if hydro_corrs else 0.0,
        "mean_charge_diff": float(np.mean(charge_diffs)),
        "mean_length_diff": float(np.mean(length_diffs)),
        "num_samples": n,
    }

    return metrics


# =============================================================================
# Training Functions
# =============================================================================


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    device: str = "cuda",
    encoding: str = "tokens",
) -> Tuple[nn.Module, Dict]:
    """
    Train model with early stopping.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Maximum number of epochs
        lr: Learning rate
        patience: Early stopping patience
        device: Device to train on
        encoding: "tokens" or "frequencies"

    Returns:
        Tuple of (trained model, training history dict)
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "best_epoch": 0,
    }

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            if encoding == "tokens":
                logits = model(
                    batch["input_ids"].to(device), batch["length"].to(device)
                )
            else:
                logits = model(batch["features"].to(device))

            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                if encoding == "tokens":
                    logits = model(
                        batch["input_ids"].to(device), batch["length"].to(device)
                    )
                else:
                    logits = model(batch["features"].to(device))

                loss = criterion(logits, batch["label"].to(device))
                val_loss += loss.item()
                val_batches += 1

                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == batch["label"].to(device)).sum().item()
                val_total += len(batch["label"])

        avg_val_loss = val_loss / val_batches
        val_accuracy = val_correct / val_total

        scheduler.step(avg_val_loss)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(val_accuracy)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
            history["best_epoch"] = epoch
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Val Acc: {val_accuracy:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def train_seq2seq_model(
    model: Seq2SeqTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-4,
    patience: int = 10,
    device: str = "cuda",
) -> Tuple[nn.Module, Dict]:
    """
    Train Seq2Seq model with early stopping.

    Args:
        model: Seq2SeqTransformer model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Maximum number of epochs
        lr: Learning rate
        patience: Early stopping patience
        device: Device to train on

    Returns:
        Tuple of (trained model, training history dict)
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padding in labels

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_exact_match": [],
        "best_epoch": 0,
    }

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            src = batch["src"].to(device)
            src_padding_mask = batch["src_padding_mask"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            labels = batch["labels"].to(device)
            tgt_padding_mask = batch["tgt_padding_mask"].to(device)

            # Forward pass
            logits = model(
                src,
                decoder_input,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask,
            )

            # Compute loss (reshape for CrossEntropyLoss)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        all_generated = []
        all_references = []

        with torch.no_grad():
            for batch in val_loader:
                src = batch["src"].to(device)
                src_padding_mask = batch["src_padding_mask"].to(device)
                decoder_input = batch["decoder_input"].to(device)
                labels = batch["labels"].to(device)
                tgt_padding_mask = batch["tgt_padding_mask"].to(device)

                # Compute loss
                logits = model(
                    src,
                    decoder_input,
                    src_key_padding_mask=src_padding_mask,
                    tgt_key_padding_mask=tgt_padding_mask,
                )
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )
                val_loss += loss.item()
                val_batches += 1

                # Generate sequences for metrics
                generated = greedy_decode(model, src, src_padding_mask, device=device)
                all_generated.extend(generated)
                all_references.extend(batch["reference_peptide"])

        avg_val_loss = val_loss / val_batches

        # Compute exact match rate
        exact_matches = sum(1 for g, r in zip(all_generated, all_references) if g == r)
        val_exact_match = exact_matches / len(all_references) if all_references else 0

        scheduler.step(avg_val_loss)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_exact_match"].append(val_exact_match)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
            history["best_epoch"] = epoch
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Val Exact Match: {val_exact_match:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# =============================================================================
# Evaluation Functions
# =============================================================================


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, num_classes: int
) -> Dict:
    """
    Compute classification metrics.

    Args:
        y_true: True class indices
        y_pred: Predicted class indices
        y_prob: Prediction probabilities (N x num_classes)
        num_classes: Number of classes

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Top-1 accuracy
    metrics["top1_accuracy"] = float(accuracy_score(y_true, y_pred))

    # Top-5 accuracy (if we have enough classes)
    k = min(5, num_classes)
    if k > 1:
        metrics["top5_accuracy"] = float(
            top_k_accuracy_score(y_true, y_prob, k=k, labels=range(num_classes))
        )
    else:
        metrics["top5_accuracy"] = metrics["top1_accuracy"]

    # Macro F1
    metrics["macro_f1"] = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )

    # Weighted F1
    metrics["weighted_f1"] = float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )

    # Macro precision/recall
    metrics["macro_precision"] = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    metrics["macro_recall"] = float(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )

    # AUC-ROC (one-vs-rest) - only if we have at least 2 classes represented
    try:
        unique_true = np.unique(y_true)
        if len(unique_true) >= 2:
            # Only compute AUC for classes that appear in y_true
            # Filter to classes that have at least one sample
            y_true_onehot = np.zeros((len(y_true), num_classes))
            y_true_onehot[np.arange(len(y_true)), y_true] = 1

            # Mask to valid classes (those with at least one positive sample)
            valid_classes = y_true_onehot.sum(axis=0) > 0
            if valid_classes.sum() >= 2:
                auc = roc_auc_score(
                    y_true_onehot[:, valid_classes],
                    y_prob[:, valid_classes],
                    average="macro",
                    multi_class="ovr"
                )
                # Check for nan
                if np.isnan(auc):
                    metrics["auc_roc_ovr"] = None
                else:
                    metrics["auc_roc_ovr"] = float(auc)
            else:
                metrics["auc_roc_ovr"] = None
        else:
            metrics["auc_roc_ovr"] = None
    except (ValueError, Exception):
        metrics["auc_roc_ovr"] = None

    metrics["num_samples"] = int(len(y_true))
    metrics["num_classes"] = int(num_classes)
    metrics["num_unique_true_classes"] = int(len(np.unique(y_true)))

    return metrics


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    encoding: str,
    num_classes: int,
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataset.

    Args:
        model: Trained model
        data_loader: Data loader
        device: Device
        encoding: "tokens" or "frequencies"
        num_classes: Number of classes

    Returns:
        Tuple of (metrics dict, y_true, y_pred, y_prob)
    """
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            if encoding == "tokens":
                logits = model(
                    batch["input_ids"].to(device), batch["length"].to(device)
                )
            else:
                logits = model(batch["features"].to(device))

            probs = torch.softmax(logits, dim=1)
            all_labels.append(batch["label"].numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    y_pred = np.argmax(y_prob, axis=1)

    metrics = compute_metrics(y_true, y_pred, y_prob, num_classes)

    return metrics, y_true, y_pred, y_prob


def evaluate_seq2seq_model(
    model: Seq2SeqTransformer,
    data_loader: DataLoader,
    device: str,
) -> Tuple[Dict, List[str], List[str]]:
    """
    Evaluate Seq2Seq model on a dataset.

    Args:
        model: Trained Seq2SeqTransformer model
        data_loader: Data loader
        device: Device

    Returns:
        Tuple of (metrics dict, generated sequences, reference sequences)
    """
    model.eval()
    all_generated = []
    all_references = []

    with torch.no_grad():
        for batch in data_loader:
            src = batch["src"].to(device)
            src_padding_mask = batch["src_padding_mask"].to(device)

            # Generate sequences
            generated = greedy_decode(model, src, src_padding_mask, device=device)
            all_generated.extend(generated)
            all_references.extend(batch["reference_peptide"])

    # Compute generation metrics
    metrics = compute_generation_metrics(all_generated, all_references)

    return metrics, all_generated, all_references


# =============================================================================
# Data Loading
# =============================================================================


def load_data(parquet_path: str, task_config: dict) -> pd.DataFrame:
    """
    Load parquet file and filter out rows with missing required columns.

    Args:
        parquet_path: Path to parquet file
        task_config: Task configuration dictionary

    Returns:
        Filtered DataFrame
    """
    df = pd.read_parquet(parquet_path)

    # Filter out rows with NaN in required columns
    required_cols = task_config["tcr_cols"] + ["peptide"]
    for col in required_cols:
        if col in df.columns:
            df = df[df[col].notna()]

    return df


# =============================================================================
# Main Training and Evaluation Pipeline
# =============================================================================


def train_and_evaluate(
    model_type: str,
    task_name: str,
    data_dir: str,
    output_dir: str,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    batch_size: int = 64,
    num_epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    max_seq_len: int = 512,
    device: str = None,
    # Seq2Seq-specific parameters
    embed_dim: int = 128,
    num_heads: int = 4,
    num_encoder_layers: int = 3,
    num_decoder_layers: int = 3,
    ffn_dim: int = 512,
    max_tgt_len: int = 30,
    seq2seq_epochs: int = 500,
    early_stop: bool = True,
) -> Dict:
    """
    Train model and evaluate on all test splits.

    Args:
        model_type: "lstm", "mlp", or "seq2seq"
        task_name: Name of the task
        data_dir: Base data directory
        output_dir: Output directory for results
        hidden_dim: Hidden dimension for LSTM models
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        batch_size: Batch size
        num_epochs: Number of training epochs for LSTM/MLP
        lr: Learning rate
        patience: Early stopping patience
        max_seq_len: Maximum sequence length
        device: Device to use (auto-detect if None)
        embed_dim: Embedding dimension for seq2seq
        num_heads: Number of attention heads for seq2seq
        num_encoder_layers: Number of encoder layers for seq2seq
        num_decoder_layers: Number of decoder layers for seq2seq
        ffn_dim: Feed-forward dimension for seq2seq
        max_tgt_len: Maximum target length for seq2seq
        seq2seq_epochs: Number of training epochs for seq2seq (default: 500)
        early_stop: Whether to use early stopping (default: True)

    Returns:
        Results dictionary
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    task_config = TASK_CONFIGS[task_name]
    mhc_class = task_config["mhc_class"]
    task_data_dir = os.path.join(data_dir, mhc_class, task_name)

    # Create output directory
    task_output_dir = os.path.join(output_dir, model_type, task_name)
    os.makedirs(task_output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Model: {model_type.upper()}")
    print(f"Task: {task_name}")
    print(f"Description: {task_config['description']}")
    print(f"Device: {device}")
    print(f"{'=' * 60}")

    # Determine encoding type and if this is a generative model
    is_generative = model_type == "seq2seq"
    encoding = "tokens" if model_type == "lstm" else "frequencies"

    # Load training data
    print("\nLoading training data...")
    train_path = os.path.join(task_data_dir, "train.parquet")
    train_df = load_data(train_path, task_config)

    # Load validation data
    print("Loading validation data...")
    val_path = os.path.join(task_data_dir, "val.parquet")
    val_df = load_data(val_path, task_config)

    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    # Fit label encoder on training data (for classification models)
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["peptide"].values)
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes (unique peptides): {num_classes}")

    # Create datasets
    if is_generative:
        train_dataset = TCRPeptideSeq2SeqDataset(
            train_df, task_config, max_src_len=max_seq_len, max_tgt_len=max_tgt_len
        )
        val_dataset = TCRPeptideSeq2SeqDataset(
            val_df, task_config, max_src_len=max_seq_len, max_tgt_len=max_tgt_len
        )
    else:
        train_dataset = TCRPeptideDataset(
            train_df, task_config, label_encoder, max_seq_len, encoding
        )
        val_dataset = TCRPeptideDataset(
            val_df, task_config, label_encoder, max_seq_len, encoding
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Initialize model
    print(f"\nInitializing {model_type.upper()} model...")
    if model_type == "lstm":
        model = LSTMClassifier(
            vocab_size=VOCAB_SIZE,
            embed_dim=64,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=True,
        )
    elif model_type == "mlp":
        model = MLPClassifier(
            input_dim=21,  # AA frequencies
            hidden_dims=[256, 128, 64],
            num_classes=num_classes,
            dropout=dropout,
        )
    elif model_type == "seq2seq":
        model = Seq2SeqTransformer(
            vocab_size=SEQ2SEQ_VOCAB_SIZE,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Determine epochs and patience based on model type and early_stop setting
    actual_epochs = seq2seq_epochs if is_generative else num_epochs
    actual_patience = patience if early_stop else actual_epochs  # No early stop = patience >= epochs

    # Train model
    print("\nTraining model...")
    print(f"  Epochs: {actual_epochs}, Early stopping: {early_stop} (patience: {actual_patience})")
    if is_generative:
        model, history = train_seq2seq_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=actual_epochs,
            lr=lr,
            patience=actual_patience,
            device=device,
        )
    else:
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=actual_epochs,
            lr=lr,
            patience=actual_patience,
            device=device,
            encoding=encoding,
        )

    # Save model and config
    model_path = os.path.join(task_output_dir, "model.pt")
    torch.save(model.state_dict(), model_path)

    config = {
        "model_type": model_type,
        "task": task_name,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "batch_size": batch_size,
        "num_epochs": actual_epochs,
        "lr": lr,
        "patience": actual_patience,
        "early_stop": early_stop,
        "max_seq_len": max_seq_len,
        "num_classes": num_classes,
        "num_params": num_params,
        "encoding": encoding if not is_generative else "seq2seq",
    }
    if is_generative:
        config.update({
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_encoder_layers": num_encoder_layers,
            "num_decoder_layers": num_decoder_layers,
            "ffn_dim": ffn_dim,
            "max_tgt_len": max_tgt_len,
        })
    config_path = os.path.join(task_output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Save training history
    history_path = os.path.join(task_output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Store results
    results = {
        "model_type": model_type,
        "task": task_name,
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "n_classes": num_classes,
        "best_epoch": history["best_epoch"],
        "test_results": {},
    }

    # Evaluate on each test split
    print("\nEvaluating on test splits...")
    for split_name in TEST_SPLITS:
        test_path = os.path.join(task_data_dir, f"{split_name}.parquet")
        if not os.path.exists(test_path):
            print(f"  Skipping {split_name} - file not found")
            continue

        print(f"\n  Evaluating on {split_name}...")

        # Load test data
        test_df = load_data(test_path, task_config)

        if is_generative:
            # Seq2Seq: Can evaluate on all splits including unseen epitopes
            test_dataset = TCRPeptideSeq2SeqDataset(
                test_df, task_config, max_src_len=max_seq_len, max_tgt_len=max_tgt_len
            )

            if len(test_dataset) == 0:
                print(f"    No valid samples - skipping")
                results["test_results"][split_name] = {
                    "error": "No valid samples",
                    "n_total": len(test_df),
                }
                continue

            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            # Evaluate with generation metrics
            metrics, generated, references = evaluate_seq2seq_model(
                model, test_loader, device
            )
            metrics["n_total"] = int(len(test_df))
            metrics["n_evaluated"] = int(len(test_dataset))

            results["test_results"][split_name] = metrics

            print(f"    Exact Match: {metrics['exact_match']:.4f}")
            print(f"    Mean Sequence Identity: {metrics['mean_sequence_identity']:.4f}")
            print(f"    Mean BLOSUM62 Normalized: {metrics['mean_blosum62_normalized']:.4f}")

            # Save predictions
            split_output_dir = os.path.join(task_output_dir, split_name)
            os.makedirs(split_output_dir, exist_ok=True)

            # Save metrics
            metrics_path = os.path.join(split_output_dir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            # Save predictions (generated vs reference)
            pred_df = pd.DataFrame({
                "generated_peptide": generated,
                "reference_peptide": references,
                "exact_match": [g == r for g, r in zip(generated, references)],
            })
            pred_path = os.path.join(split_output_dir, "predictions.csv")
            pred_df.to_csv(pred_path, index=False)

        else:
            # Classification models: Can only evaluate on splits with known labels
            test_dataset = TCRPeptideDataset(
                test_df, task_config, label_encoder, max_seq_len, encoding
            )

            if len(test_dataset) == 0:
                print(f"    No samples with known labels - skipping")
                results["test_results"][split_name] = {
                    "error": "No samples with known labels",
                    "n_total": len(test_df),
                    "n_known": 0,
                }
                continue

            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            # Evaluate
            metrics, y_true, y_pred, y_prob = evaluate_model(
                model, test_loader, device, encoding, num_classes
            )
            metrics["n_total"] = int(len(test_df))
            metrics["n_known_labels"] = int(len(test_dataset))
            metrics["n_unknown_labels"] = int(len(test_df) - len(test_dataset))

            results["test_results"][split_name] = metrics

            print(f"    Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")
            print(f"    Macro F1: {metrics['macro_f1']:.4f}")
            if metrics.get("auc_roc_ovr"):
                print(f"    AUC-ROC: {metrics['auc_roc_ovr']:.4f}")

            # Save predictions
            split_output_dir = os.path.join(task_output_dir, split_name)
            os.makedirs(split_output_dir, exist_ok=True)

            # Save metrics
            metrics_path = os.path.join(split_output_dir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            # Save predictions
            pred_df = pd.DataFrame(
                {
                    "true_label_idx": y_true,
                    "true_label": label_encoder.inverse_transform(y_true),
                    "predicted_label_idx": y_pred,
                    "predicted_label": label_encoder.inverse_transform(y_pred),
                    "predicted_prob": y_prob.max(axis=1),
                }
            )
            pred_path = os.path.join(split_output_dir, "predictions.csv")
            pred_df.to_csv(pred_path, index=False)

    # Save aggregated results
    results_path = os.path.join(task_output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {task_output_dir}")

    return results


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Baseline Neural Network Benchmark for TCR-peptide specificity prediction"
    )
    parser.add_argument(
        "--model",
        choices=["lstm", "mlp", "seq2seq", "all"],
        default="lstm",
        help="Model type to run (default: lstm)",
    )
    parser.add_argument(
        "--task",
        choices=list(TASK_CONFIGS.keys()) + ["all"],
        default="trb_peptide_mhc_one",
        help="Task to run (default: trb_peptide_mhc_one)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/sravisha/projects/quest/data/icml/tasks/tcr_specificity/tcr90pep80",
        help="Base data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/sravisha/projects/quest/results/baseline_benchmark",
        help="Output directory for results",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension for LSTM (default: 128)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of LSTM layers (default: 2)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate (default: 0.3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of epochs (default: 50)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect)",
    )
    # Seq2Seq-specific arguments
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=128,
        help="Embedding dimension for seq2seq (default: 128)",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="Number of attention heads for seq2seq (default: 4)",
    )
    parser.add_argument(
        "--num_encoder_layers",
        type=int,
        default=3,
        help="Number of encoder layers for seq2seq (default: 3)",
    )
    parser.add_argument(
        "--num_decoder_layers",
        type=int,
        default=3,
        help="Number of decoder layers for seq2seq (default: 3)",
    )
    parser.add_argument(
        "--ffn_dim",
        type=int,
        default=512,
        help="Feed-forward dimension for seq2seq (default: 512)",
    )
    parser.add_argument(
        "--max_tgt_len",
        type=int,
        default=30,
        help="Maximum target length for seq2seq (default: 30)",
    )
    parser.add_argument(
        "--seq2seq_epochs",
        type=int,
        default=500,
        help="Number of epochs for seq2seq model (default: 500)",
    )
    parser.add_argument(
        "--no_early_stop",
        action="store_true",
        help="Disable early stopping",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine models and tasks to run
    models = ["lstm", "mlp", "seq2seq"] if args.model == "all" else [args.model]
    tasks = list(TASK_CONFIGS.keys()) if args.task == "all" else [args.task]

    # Track all results
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "results": {},
    }

    # Run benchmarks
    for model_type in models:
        all_results["results"][model_type] = {}
        for task in tasks:
            try:
                results = train_and_evaluate(
                    model_type=model_type,
                    task_name=task,
                    data_dir=args.data_dir,
                    output_dir=args.output_dir,
                    hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    batch_size=args.batch_size,
                    num_epochs=args.num_epochs,
                    lr=args.lr,
                    patience=args.patience,
                    max_seq_len=args.max_seq_len,
                    device=args.device,
                    # Seq2Seq parameters
                    embed_dim=args.embed_dim,
                    num_heads=args.num_heads,
                    num_encoder_layers=args.num_encoder_layers,
                    num_decoder_layers=args.num_decoder_layers,
                    ffn_dim=args.ffn_dim,
                    max_tgt_len=args.max_tgt_len,
                    seq2seq_epochs=args.seq2seq_epochs,
                    early_stop=not args.no_early_stop,
                )
                all_results["results"][model_type][task] = results
            except Exception as e:
                print(f"\nError running {model_type}/{task}: {e}")
                import traceback

                traceback.print_exc()
                all_results["results"][model_type][task] = {"error": str(e)}

    # Save summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Benchmark complete!")
    print(f"Summary saved to: {summary_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
