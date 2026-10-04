"""Dataset for ESM SFT: concatenate (TRA?, TRB, peptide, MHC pseudo-seq)
into a single tokenized sequence with chain-delimiter strings.

For ESM2 (HF AutoTokenizer): we use raw string sequences and let HF
tokenize as a single "protein" with chain markers as repeated single-char
spacers (`X`) since ESM2's vocab is amino-acid only.

For ESM-C (EsmSequenceTokenizer): same approach.

Both encoders accept any AA-string input; we don't have access to the
QUEST custom delimiter tokens (`[TRA]`, `[TRB]`, etc.) in stock ESM
vocabularies. Instead we use a simple separator pattern:

    "{TRA}XX{TRB}XX{PEPTIDE}XX{MHC_PSEUDOSEQ}"

Two `X` tokens (mapped to the unknown-AA) act as a soft chain boundary.
This is a deliberate simplification consistent with how ESM models are
typically used in TCR papers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CHAIN_SEP = "XX"  # two unknown-AA tokens as a soft chain boundary


def _safe(seq: object) -> str:
    return str(seq).upper() if isinstance(seq, str) and seq else ""


def assemble_sequence(row: pd.Series, chains: str, mhc_col: str, pseudoseq_col: Optional[str] = None) -> str:
    """Build a single AA string from a benchmark_v2 row.

    chains: "trb" or "paired"
    mhc_col: 'mhc_one_allele' (we try mhc_one_pocket_contact for sequence form first)

    The MHC pseudo-sequence is preferred over the raw allele name because
    ESM vocab can't tokenize HLA name strings. If the row only has the
    allele-name form, we fall back to a fixed-length 0-string of the same
    soft length so the model gets *something* per-row.
    """
    parts = []
    if chains == "paired":
        parts.append(_safe(row.get("tra_cdr3")))
        parts.append(CHAIN_SEP)
    parts.append(_safe(row.get("trb_cdr3")))
    parts.append(CHAIN_SEP)
    parts.append(_safe(row.get("peptide")))
    parts.append(CHAIN_SEP)
    if pseudoseq_col is not None:
        parts.append(_safe(row.get(pseudoseq_col)))
    else:
        parts.append(_safe(row.get(mhc_col)))
    return "".join(parts)


@dataclass
class TokenizedAS(Dataset):
    df: pd.DataFrame
    tokenizer: object        # HF AutoTokenizer or EsmSequenceTokenizer
    is_esmc: bool
    chains: str              # "trb" | "paired"
    mhc_col: str             # "mhc_one_allele" | "mhc_two_allele"
    pseudoseq_col: Optional[str] = None
    max_len: int = 256

    def __post_init__(self):
        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = assemble_sequence(row, chains=self.chains, mhc_col=self.mhc_col, pseudoseq_col=self.pseudoseq_col)
        if self.is_esmc:
            # EsmSequenceTokenizer: encode returns a tensor of token ids
            ids = self.tokenizer.encode(seq)
            ids = torch.as_tensor(ids, dtype=torch.long)
            ids = ids[: self.max_len]
            attn = torch.ones_like(ids)
        else:
            # HF AutoTokenizer
            enc = self.tokenizer(
                seq, max_length=self.max_len, truncation=True,
                padding=False, return_tensors="pt",
            )
            ids = enc["input_ids"][0]
            attn = enc["attention_mask"][0]
        return {
            "input_ids": ids,
            "attention_mask": attn,
            "label": torch.tensor(float(row.get("label", 1.0)), dtype=torch.float32),
        }


def collate_pad(batch: list[dict], pad_id: int = 1) -> dict:
    """Pad batch to max length, return tensors + attention mask."""
    max_l = max(b["input_ids"].size(0) for b in batch)
    n = len(batch)
    ids = torch.full((n, max_l), pad_id, dtype=torch.long)
    attn = torch.zeros((n, max_l), dtype=torch.long)
    labels = torch.zeros(n, dtype=torch.float32)
    for i, b in enumerate(batch):
        l = b["input_ids"].size(0)
        ids[i, :l] = b["input_ids"]
        attn[i, :l] = b["attention_mask"]
        labels[i] = b["label"]
    return {"input_ids": ids, "attention_mask": attn, "label": labels}
