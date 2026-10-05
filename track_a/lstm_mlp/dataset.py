"""Triplet dataset for AS binary scoring + retrieval candidate dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from track_a.lstm_mlp.model import encode_seq

ALLELE_UNK = "__UNK__"
ALLELE_PAD_IDX = 0
DEFAULT_MAX_TRB = 32
DEFAULT_MAX_TRA = 32
DEFAULT_MAX_PEP = 25


def _is_paired_task(task: str) -> bool:
    return task.startswith("as_paired")


def _is_class_one(task: str) -> bool:
    return task.endswith("_i")


def _mhc_col(task: str) -> str:
    return "mhc_one_allele" if _is_class_one(task) else "mhc_two_allele"


def build_allele_vocab(
    train_df: pd.DataFrame,
    mhc_col: str,
    cache_path: Optional[Path] = None,
) -> dict[str, int]:
    """Return {allele: idx} (1-indexed; 0 reserved for OOV) and persist if cache_path."""
    alleles = sorted(set(a for a in train_df[mhc_col].dropna().unique() if isinstance(a, str)))
    vocab = {a: i + 1 for i, a in enumerate(alleles)}
    vocab[ALLELE_UNK] = 0
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(vocab))
    return vocab


def encode_allele(allele: Optional[str], vocab: dict[str, int]) -> int:
    if allele is None or not isinstance(allele, str):
        return ALLELE_PAD_IDX
    return vocab.get(allele, ALLELE_PAD_IDX)


def one_hot_mhc(allele_idx: int, vocab_size: int) -> torch.Tensor:
    oh = torch.zeros(vocab_size, dtype=torch.float32)
    oh[allele_idx] = 1.0
    return oh


class TripletDataset(Dataset):
    """A DataFrame of (TCR, peptide, MHC, label=0/1) → torch tensors.

    `df` must have columns: trb_cdr3 (always), peptide, label,
    and either mhc_one_allele or mhc_two_allele depending on task,
    and tra_cdr3 if `chains=="paired"`.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        task: str,
        allele_vocab: dict[str, int],
        max_trb: int = DEFAULT_MAX_TRB,
        max_tra: int = DEFAULT_MAX_TRA,
        max_pep: int = DEFAULT_MAX_PEP,
    ):
        self.df = df.reset_index(drop=True)
        self.task = task
        self.allele_vocab = allele_vocab
        self.allele_vocab_size = len(allele_vocab)
        self.max_trb = max_trb
        self.max_tra = max_tra
        self.max_pep = max_pep
        self.chains = "paired" if _is_paired_task(task) else "trb"
        self.mhc_col = _mhc_col(task)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        trb_ids, trb_len = encode_seq(row.get("trb_cdr3"), self.max_trb)
        pep_ids, pep_len = encode_seq(row.get("peptide"), self.max_pep)
        allele_idx = encode_allele(row.get(self.mhc_col), self.allele_vocab)
        mhc_oh = one_hot_mhc(allele_idx, self.allele_vocab_size)
        item = {
            "trb_ids": trb_ids, "trb_len": torch.tensor(trb_len),
            "pep_ids": pep_ids, "pep_len": torch.tensor(pep_len),
            "mhc_oh": mhc_oh,
            "label": torch.tensor(float(row.get("label", 1.0)), dtype=torch.float32),
        }
        if self.chains == "paired":
            tra_ids, tra_len = encode_seq(row.get("tra_cdr3"), self.max_tra)
            item["tra_ids"] = tra_ids
            item["tra_len"] = torch.tensor(tra_len)
        return item


def collate(batch: list[dict]) -> dict:
    """Stack a list of dicts into batched tensors."""
    out = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out
