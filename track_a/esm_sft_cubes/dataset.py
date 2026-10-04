"""Per-checkpoint sequence assembly + Dataset for the 8-cell cubes.

Parameterizes the (tcr_variant, mhc_variant, chains) per the 5 Stage-4
CPT checkpoints. Uses the literal `<eos>` separator (matches CPT pretraining
input distribution exactly — ESM-C's `EsmSequenceTokenizer` emits one EOS id
per `<eos>` string).

Assembled sequence (peptide-in-middle, matches CPT training):

    <TCR pieces> <eos> <peptide_candidate> <eos> <MHC piece>

Examples
--------
Checkpoint A (cdr3, pocket), chains=trb:
    "CASSEKL<eos>SIINFEKL<eos>DRLTAVQ..."   (37-char pocket)
Checkpoint B (cdr123, pocket_contact), chains=paired:
    "tra_cdr1<eos>tra_cdr2<eos>tra_cdr3<eos>trb_cdr1<eos>trb_cdr2<eos>trb_cdr3<eos>SIINFEKL<eos>...47-char..."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

EOS = "<eos>"  # ESM-C EsmSequenceTokenizer tokenizes this literal as eos_token_id (=2)


# Column maps per checkpoint config
TCR_COLS = {
    "cdr3":   {"trb":    ["trb_cdr3"],
               "paired": ["tra_cdr3", "trb_cdr3"]},
    "cdr123": {"trb":    ["trb_cdr1", "trb_cdr2", "trb_cdr3"],
               "paired": ["tra_cdr1", "tra_cdr2", "tra_cdr3",
                          "trb_cdr1", "trb_cdr2", "trb_cdr3"]},
    "full":   {"trb":    ["trb_full"],
               "paired": ["tra_full", "trb_full"]},
}
MHC_COL = {
    "pocket":         "mhc_one_pocket",
    "pocket_contact": "mhc_one_pocket_contact",
    "full":           "mhc_one",
}


def _safe(s: object) -> str:
    """Return the upper-cased AA string, or '' for null/empty."""
    if isinstance(s, str) and s:
        return s.upper()
    return ""


def required_columns(tcr_variant: str, mhc_variant: str, chains: str) -> list[str]:
    """Columns that must be non-null/non-empty for a row to be usable."""
    tcr = TCR_COLS[tcr_variant][chains]
    return list(tcr) + [MHC_COL[mhc_variant], "peptide"]


def assemble_cpt_sequence(
    row: pd.Series,
    tcr_variant: str,        # 'cdr3' | 'cdr123' | 'full'
    mhc_variant: str,        # 'pocket' | 'pocket_contact' | 'full'
    chains: str,             # 'trb' | 'paired'
) -> str:
    """Build the encoder input string for a single row.

    Returns '' if any required column is empty (caller should skip the row).
    """
    tcr_cols = TCR_COLS[tcr_variant][chains]
    mhc_col = MHC_COL[mhc_variant]

    tcr_parts = [_safe(row.get(c)) for c in tcr_cols]
    if any(p == "" for p in tcr_parts):
        return ""
    mhc_str = _safe(row.get(mhc_col))
    if not mhc_str:
        return ""
    pep_str = _safe(row.get("peptide"))
    if not pep_str:
        return ""

    tcr_segment = EOS.join(tcr_parts)
    return f"{tcr_segment}{EOS}{pep_str}{EOS}{mhc_str}"


@dataclass
class TokenizedCube(Dataset):
    """Wrap a DataFrame of (cube-row + label) into a tokenized dataset.

    Rows whose `assemble_cpt_sequence(...)` returns '' are silently dropped
    at construction time.
    """
    df: pd.DataFrame
    tokenizer: object          # EsmSequenceTokenizer
    tcr_variant: str
    mhc_variant: str
    chains: str
    max_len: int = 320

    def __post_init__(self):
        # Pre-assemble strings so we can drop bad rows once
        df = self.df.reset_index(drop=True)
        seqs = [assemble_cpt_sequence(df.iloc[i], self.tcr_variant, self.mhc_variant, self.chains)
                for i in range(len(df))]
        keep = [i for i, s in enumerate(seqs) if s]
        self.df = df.iloc[keep].reset_index(drop=True)
        self._seqs = [seqs[i] for i in keep]

    def __len__(self) -> int:
        return len(self._seqs)

    def __getitem__(self, idx: int) -> dict:
        ids = self.tokenizer.encode(self._seqs[idx])
        ids = torch.as_tensor(ids, dtype=torch.long)[: self.max_len]
        attn = torch.ones_like(ids)
        label = float(self.df.iloc[idx].get("label", 1.0))
        return {
            "input_ids": ids,
            "attention_mask": attn,
            "label": torch.tensor(label, dtype=torch.float32),
        }


def collate_pad_esmc(batch: list[dict], pad_id: int = 1) -> dict:
    """Pad batch to max length in batch."""
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
