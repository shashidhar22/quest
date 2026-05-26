"""Shared CPT (continued pre-training) MLM library for TCRBench v3.

This module is the foundation reused across all 8 CPT training stages. It is
deliberately a SLIM wrapper that delegates to existing, battle-tested QUEST
infrastructure where possible. Specifically it reuses:

- ``quest.training.collators.DataCollatorForMLMDynamic`` (token-id MLM collator
  that already excludes ``<eos>`` from masking via its special_token_ids tensor).
- ``quest.training.samplers.DistributedLengthBucketSampler`` for length-bucketed
  batching (cuts padding waste on highly variable-length CPT data).
- ``quest.training.callbacks.EarlyStopping`` for stop-on-plateau.
- The DDP / AMP / wandb / checkpointing pattern from
  ``scripts/training/esm_native_trainer.py`` (mirrored, not subclassed, because
  the existing trainer is a single 966-LOC class with no clean extension hook).

What is genuinely new here:
- ``preprocess_for_tokenizer`` — Fix B (whitespace around ``<eos>``) for
  cross-tokenizer robustness.
- ``setup_lora_model`` — auto-detects LoRA target modules based on the
  underlying model class (ESM-2 vs ESM-C).
- ``bucketed_eval`` — runs deterministic MLM eval on the canonical eval set
  and reports per-``stratum_key`` perplexity.
- ``CPTTrainer`` — a CPT-specialised training loop that hooks bucketed_eval
  every ``eval_every_n_steps`` against the canonical eval set and selects the
  best checkpoint by ``canonical/overall/loss``.
"""

from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

import wandb

from quest.training.callbacks import EarlyStopping
from quest.training.collators import DataCollatorForMLMDynamic
from quest.training.samplers import DistributedLengthBucketSampler, LengthBucketSampler

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# Tokenizer preprocessing (Fix B)
# ---------------------------------------------------------------------------

EOS_LITERAL = "<eos>"


def preprocess_for_tokenizer(text: str) -> str:
    """Pad ``<eos>`` with whitespace.

    The ESM-2 fast tokenizer already pre-splits on ``<eos>`` because it is
    registered as a special token, but applying this transformation makes the
    code tokenizer-agnostic — the same dataset can train ESM-2, ESM-C, or any
    future protein PLM without re-checking each tokenizer's quirks.
    """
    return text.replace(EOS_LITERAL, " " + EOS_LITERAL + " ")


def get_tokenizer(model_name: str):
    """Load tokenizer and verify ``<eos>`` round-trips correctly.

    For ESM-C, returns the ``EsmSequenceTokenizer`` directly — Fix B
    verification for that tokenizer happens in Stage 2 ``_verify_esmc.py``.
    """
    if _is_esmc_model_name(model_name):
        from esm.tokenization import EsmSequenceTokenizer  # lazy import
        tok = EsmSequenceTokenizer()
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token or tok.unk_token
        return tok

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    # Sanity: a literal mid-string '<eos>' tokenizes to one id (after Fix B).
    sample = preprocess_for_tokenizer("CASS<eos>NEKL")
    ids = tok(sample, add_special_tokens=True)["input_ids"]
    n_eos = sum(1 for tid in ids if tid == tok.eos_token_id)
    if n_eos != 2:
        raise RuntimeError(
            f"Tokenizer {model_name} did not pre-split <eos>. Got {n_eos} eos ids in "
            f"{ids}. Run verify_tokenizer.py to diagnose."
        )
    return tok


def _is_esmc_model_name(model_name: str) -> bool:
    """Detect ESM-C base-model names (esmc_300m / esmc_600m / esmc-300m...)."""
    n = model_name.lower()
    return n.startswith("esmc_") or n.startswith("esmc-") or "/esmc" in n


# ---------------------------------------------------------------------------
# Stage 2: TCR representation projection + per-token region annotation
# ---------------------------------------------------------------------------

# Faithful Python mirror of ``scripts/data_processing/cpt_dataset_builder.py``
# (git commit 0610b61): given a canonical_eval_v2 row carrying full molecular
# columns, produce the same ``input_text`` a run-dataset row would have, plus
# a parallel ``segment_kinds`` list naming each `<eos>`-separated piece. The
# region annotator then uses ``segment_kinds`` to label each token by
# biological region (cdr3, cdr12, pocket, peptide, framework).


_VALID_TCR_VARIANTS = ("cdr3", "cdr123", "full")
_VALID_MHC_VARIANTS = ("pocket", "pocket_contact", "full")


def _chain_segments(row: dict, chain: str, variant: str) -> list[tuple[str, str]]:
    """Return ``[(seg_text, seg_kind), ...]`` for one chain (chain ∈ {'tra','trb'})."""
    if variant == "cdr3":
        v = row.get(f"{chain}_cdr3")
        return [(v, f"{chain}_cdr3")] if v else []
    if variant == "cdr123":
        out: list[tuple[str, str]] = []
        for i in (1, 2, 3):
            v = row.get(f"{chain}_cdr{i}")
            if v:
                out.append((v, f"{chain}_cdr{i}"))
        return out
    if variant == "full":
        v = row.get(f"{chain}_full")
        return [(v, f"{chain}_full")] if v else []
    raise ValueError(f"unknown tcr_variant: {variant}")


def _mhc_segment(row: dict, mhc: str, variant: str) -> tuple[str, str] | None:
    """Return ``(seg_text, seg_kind)`` for mhc_one / mhc_two, or None if empty."""
    if variant == "pocket":
        v = row.get(f"{mhc}_pocket")
        return (v, f"{mhc}_pocket") if v else None
    if variant == "pocket_contact":
        v = row.get(f"{mhc}_pocket_contact")
        return (v, f"{mhc}_pocket_contact") if v else None
    if variant == "full":
        v = row.get(mhc)
        return (v, mhc) if v else None
    raise ValueError(f"unknown mhc_variant: {variant}")


def _parse_order_key(order_key: str) -> list[str]:
    """Parse an order_key like 'tra_trb_peptide_mhc_one_mhc_two' into ordered tokens.

    Mirrors the builder's logic: temporarily substitute multi-token names so
    splitting on '_' is unambiguous.
    """
    if not order_key:
        return []
    s = order_key.replace("mhc_one", "M1").replace("mhc_two", "M2")
    out = []
    for tok in s.split("_"):
        if tok == "M1":
            out.append("mhc_one")
        elif tok == "M2":
            out.append("mhc_two")
        elif tok in ("tra", "trb", "peptide"):
            out.append(tok)
    return out


def project_canonical_row_to_run_input(
    row: dict, tcr_variant: str, mhc_variant: str
) -> tuple[str, list[str]]:
    """Project a canonical_eval_v2 row to its ``input_text`` for a given (TCR, MHC)
    representation, mirroring the run-dataset builder.

    Returns ``(input_text, segment_kinds)`` where ``segment_kinds`` is a list
    of strings, one per `<eos>`-separated piece, e.g.
    ``['tra_cdr3', 'trb_cdr3', 'peptide', 'mhc_one_pocket']``.

    Used by region_restricted_eval to align tokens with biological regions.
    Empty result (``("", [])``) means the row contributes no usable input.
    """
    if tcr_variant not in _VALID_TCR_VARIANTS:
        raise ValueError(f"tcr_variant must be in {_VALID_TCR_VARIANTS}")
    if mhc_variant not in _VALID_MHC_VARIANTS:
        raise ValueError(f"mhc_variant must be in {_VALID_MHC_VARIANTS}")

    order_key = row.get("order_key") or ""
    tokens = _parse_order_key(order_key)

    pieces: list[tuple[str, str]] = []
    for tok in tokens:
        if tok in ("tra", "trb"):
            pieces.extend(_chain_segments(row, tok, tcr_variant))
        elif tok == "peptide":
            v = row.get("peptide")
            if v:
                pieces.append((v, "peptide"))
        elif tok in ("mhc_one", "mhc_two"):
            seg = _mhc_segment(row, tok, mhc_variant)
            if seg is not None:
                pieces.append(seg)

    if not pieces:
        return ("", [])
    input_text = EOS_LITERAL.join(p[0] for p in pieces)
    segment_kinds = [p[1] for p in pieces]
    return input_text, segment_kinds


def identify_segment_regions(seg_text: str, seg_kind: str, row: dict) -> list[str]:
    """Per-AA region labels for one segment.

    Label set: 'cdr3', 'cdr12', 'pocket', 'pocket_contact_only', 'peptide',
    'framework'. ``'special'`` is not used here; that's reserved for
    <cls>/<eos>/<sep> tokens added later by the annotator.
    """
    n = len(seg_text)
    if n == 0:
        return []

    # Bare-CDR / bare-pocket / bare-peptide segments: uniform label.
    if seg_kind in ("tra_cdr3", "trb_cdr3"):
        return ["cdr3"] * n
    if seg_kind in ("tra_cdr1", "tra_cdr2", "trb_cdr1", "trb_cdr2"):
        return ["cdr12"] * n
    if seg_kind == "peptide":
        return ["peptide"] * n
    if seg_kind in ("mhc_one_pocket", "mhc_two_pocket"):
        return ["pocket"] * n

    labels = ["framework"] * n

    # Full chain TCR: overlay cdr3 then cdr1/cdr2 (don't overwrite cdr3).
    if seg_kind in ("tra_full", "trb_full"):
        chain = seg_kind.split("_")[0]
        cdr3 = row.get(f"{chain}_cdr3")
        if cdr3 and cdr3 in seg_text:
            start = seg_text.find(cdr3)
            for i in range(start, start + len(cdr3)):
                if i < n:
                    labels[i] = "cdr3"
        for cdr_num in (1, 2):
            cdr = row.get(f"{chain}_cdr{cdr_num}")
            if cdr and cdr in seg_text:
                start = seg_text.find(cdr)
                for i in range(start, start + len(cdr)):
                    if i < n and labels[i] == "framework":
                        labels[i] = "cdr12"
        return labels

    # pocket_contact substring: default to 'pocket_contact_only', overlay 'pocket'.
    # Pocket residues are structural positions in the binding groove; in the
    # pocket_contact string they are interleaved with contact residues by
    # chain position (not a contiguous substring). Use greedy subsequence
    # match in chain-position order — pocket positions are a sorted subset
    # of pocket_contact positions, so left-to-right greedy is correct
    # whenever residues are non-degenerate (and a tolerable label-swap
    # otherwise; loss is still attributed to a residue at the adjacent
    # chain position).
    if seg_kind in ("mhc_one_pocket_contact", "mhc_two_pocket_contact"):
        labels = ["pocket_contact_only"] * n
        mhc = seg_kind.replace("_pocket_contact", "")
        pocket = row.get(f"{mhc}_pocket")
        if pocket:
            for i in _subsequence_positions(seg_text, pocket):
                if i < n:
                    labels[i] = "pocket"
        return labels

    # Full MHC chain: default 'framework', overlay pocket_contact_only then pocket.
    # Same subsequence rationale as above; for the full chain, pocket_contact
    # is itself a subsequence (not a substring) of the full sequence.
    if seg_kind in ("mhc_one", "mhc_two"):
        mhc = seg_kind
        contact = row.get(f"{mhc}_pocket_contact")
        if contact:
            for i in _subsequence_positions(seg_text, contact):
                if i < n and labels[i] == "framework":
                    labels[i] = "pocket_contact_only"
        pocket = row.get(f"{mhc}_pocket")
        if pocket:
            for i in _subsequence_positions(seg_text, pocket):
                if i < n:
                    labels[i] = "pocket"
        return labels

    return labels  # unknown kind — keep as framework


def _subsequence_positions(haystack: str, needle: str) -> list[int]:
    """Greedy left-to-right subsequence match: indices in ``haystack`` where
    each ``needle`` character is consumed in order. Returns ``[]`` if
    ``needle`` is not a subsequence of ``haystack``.

    Used by ``identify_segment_regions`` to locate MHC pocket residues
    inside the pocket_contact or full chain — those residues are at
    fixed chain positions but the canonical eval strings interleave
    pocket and contact residues by position, so a contiguous-substring
    search fails (see Stage 3 preflight).
    """
    if not needle:
        return []
    out: list[int] = []
    j = 0
    nlen = len(needle)
    for i, c in enumerate(haystack):
        if c == needle[j]:
            out.append(i)
            j += 1
            if j == nlen:
                return out
    return [] if j < nlen else out


def annotate_token_regions(
    input_text: str,
    row: dict,
    segment_kinds: list[str],
    input_ids: list[int],
    *,
    separator_token_id: int,
    special_token_ids: set[int],
) -> list[str]:
    """Per-token region labels (length == len(input_ids)).

    Algorithm: any input_id in ``special_token_ids`` (cls/eos/sep/pad/unk) or
    equal to ``separator_token_id`` is labeled ``'special'``. Remaining
    positions are AA tokens; they are paired in order with the AAs of each
    segment, and each AA is labeled per ``identify_segment_regions``.

    Truncation-safe: if input_ids were truncated, only the surviving AA
    positions are labeled (the rest of the segment is dropped silently).
    """
    n_tokens = len(input_ids)
    labels = ["framework"] * n_tokens
    special_set = set(special_token_ids) | {separator_token_id}
    for i, tid in enumerate(input_ids):
        if tid in special_set:
            labels[i] = "special"

    segments = input_text.split(EOS_LITERAL) if input_text else []
    # Defensive: if segment_kinds count mismatches, leave AA tokens at default.
    if len(segments) != len(segment_kinds):
        return labels

    aa_positions = [i for i in range(n_tokens) if labels[i] != "special"]
    cursor = 0
    for seg_text, seg_kind in zip(segments, segment_kinds):
        if cursor >= len(aa_positions):
            break
        avail = len(aa_positions) - cursor
        take = min(len(seg_text), avail)
        if take <= 0:
            continue
        seg_regions = identify_segment_regions(seg_text[:take], seg_kind, row)
        for j in range(take):
            labels[aa_positions[cursor + j]] = seg_regions[j]
        cursor += take

    return labels


# ---------------------------------------------------------------------------
# Dataset loading + tokenization
# ---------------------------------------------------------------------------


def load_cpt_dataset(
    path: str,
    tokenizer,
    *,
    split: str | None = None,
    max_length: int = 128,
    num_proc: int = 8,
    add_length_col: bool = True,
    eos_strategy: str = "fixb",
) -> Dataset:
    """Load an HF dataset (or a specific split subdir), tokenize.

    ``eos_strategy`` selects how literal ``<eos>`` in ``input_text`` is
    preprocessed before tokenization (see ``_preprocess_for_eos_strategy``).
    Stage 0/1 default ``"fixb"`` preserves backward compatibility for ESM-2.
    Stage 2 uses ``"raw"`` for ESM-C.

    Returns a Dataset with at minimum ``input_ids`` and ``attention_mask``
    columns. Stratification metadata (``mhc_class``, ``n_segments``, etc.) is
    preserved when present so bucketed_eval can group on it.
    """
    target = os.path.join(path, split) if split else path
    ds = load_from_disk(target)

    if "input_text" not in ds.column_names:
        raise ValueError(
            f"Dataset at {target} has no 'input_text' column; got {ds.column_names}"
        )

    def _prep_and_tok(batch):
        preped = [_preprocess_for_eos_strategy(t, eos_strategy) for t in batch["input_text"]]
        enc = tokenizer(
            preped,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        out = {
            "input_ids": enc["input_ids"],
        }
        am = enc.get("attention_mask")
        if am is None:
            am = [[1] * len(x) for x in enc["input_ids"]]
        out["attention_mask"] = am
        if add_length_col:
            out["length"] = [len(x) for x in enc["input_ids"]]
        return out

    keep_cols = [
        c
        for c in (
            "mhc_class",
            "n_segments",
            "subset_key",
            "stratum_key",
            "source_row_hash",
        )
        if c in ds.column_names
    ]
    drop_cols = [c for c in ds.column_names if c not in keep_cols]

    ds = ds.map(
        _prep_and_tok,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=drop_cols,
        desc=f"tokenize ({os.path.basename(target)})",
    )
    return ds


# ---------------------------------------------------------------------------
# LoRA setup
# ---------------------------------------------------------------------------


ESM2_LORA_TARGETS = ["query", "key", "value", "dense", "intermediate.dense", "output.dense"]
# ESM-C uses a different attention/FFN module naming convention. These names are
# the conventional set for facebookresearch/esmc; if the underlying ESM-C
# implementation drifts, this list is the place to update. Full validation of
# ESM-C LoRA happens in Stage 2.
ESMC_LORA_TARGETS = ["layernorm_qkv.1", "out_proj", "ffn.1", "ffn.3"]


def _autodetect_lora_targets(model) -> list[str]:
    cls_name = type(model).__name__
    if "Esm" in cls_name and "ESMC" not in cls_name.upper():
        return ESM2_LORA_TARGETS
    if "ESMC" in cls_name.upper():
        return ESMC_LORA_TARGETS
    raise ValueError(
        f"Could not autodetect LoRA targets for model class {cls_name}. "
        f"Pass target_modules explicitly."
    )


class _ESMCHFConfigShim(dict):
    """Minimal HF-style ``.config`` shim for ESM-C so PEFT's get_peft_model works.

    PEFT expects ``model.config.use_return_dict`` and ``model.config.get(...)``.
    The shim provides both via dict + attribute access. Mirrors the pattern in
    scripts/training/esmc_native_trainer.py:250-263.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def _load_esmc(model_name: str, *, dtype: torch.dtype = torch.bfloat16):
    """Load an ESM-C model via the ``esm`` package and attach the PEFT-shim
    config so ``get_peft_model`` can wrap it."""
    from esm.models.esmc import ESMC  # lazy import (heavy)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESMC.from_pretrained(model_name, device=device)
    if dtype != torch.bfloat16:
        # ESMC.from_pretrained already lands in bf16; cast only if caller asked
        # for something else.
        model = model.to(dtype)
    if not hasattr(model, "config"):
        model.config = _ESMCHFConfigShim(
            use_return_dict=True,
            tie_word_embeddings=False,
        )
    return model


def load_base_model(
    model_name: str,
    *,
    attn_implementation: str = "flash_attention_2",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load a base PLM with the preferred attention impl.

    Routes to ``esm.models.esmc.ESMC.from_pretrained`` if ``model_name`` looks
    like an ESM-C id (e.g. ``esmc_300m``). Otherwise loads via
    ``AutoModelForMaskedLM`` with Flash Attention 2, falling back to SDPA if
    FA2 isn't available.
    """
    if _is_esmc_model_name(model_name):
        return _load_esmc(model_name, dtype=dtype)
    try:
        return AutoModelForMaskedLM.from_pretrained(
            model_name, attn_implementation=attn_implementation, dtype=dtype
        )
    except Exception as e:
        if attn_implementation == "flash_attention_2":
            print(f"[warn] Flash Attention 2 unavailable ({e}); falling back to SDPA")
            return AutoModelForMaskedLM.from_pretrained(
                model_name, attn_implementation="sdpa", dtype=dtype
            )
        raise


def setup_lora_model(
    model_name: str,
    *,
    rank: int = 32,
    alpha: int = 64,
    dropout: float = 0.05,
    bias: str = "none",
    target_modules: list[str] | str = "auto",
    attn_implementation: str = "flash_attention_2",
):
    """Load a base MLM model and wrap it with PEFT LoRA.

    Returns ``(model, info_dict)`` where info_dict contains the auto-detected
    target modules, trainable param count, and total param count.
    """
    base = load_base_model(model_name, attn_implementation=attn_implementation)
    if target_modules == "auto":
        target_modules = _autodetect_lora_targets(base)
    lora = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias=bias,
    )
    model = get_peft_model(base, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    info = {
        "target_modules": target_modules,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100.0 * trainable / total,
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
    }
    return model, info


# ---------------------------------------------------------------------------
# Bucketed evaluation
# ---------------------------------------------------------------------------


def _apply_deterministic_mlm_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    special_token_ids: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    mlm_probability: float,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic 80/10/10 MLM masking driven by a torch.Generator.

    Same recipe as DataCollatorForMLMDynamic but with a passable RNG so
    repeated calls with the same seed give the same mask pattern. This makes
    bucketed eval reproducible.
    """
    labels = input_ids.clone()
    rand_vals = torch.rand(*input_ids.shape, 2, generator=rng)
    rand_mask = rand_vals[..., 0]
    mask_type = rand_vals[..., 1]

    special_mask = torch.isin(input_ids, special_token_ids)
    valid = (attention_mask == 1) & ~special_mask
    masked = valid & (rand_mask < mlm_probability)
    labels[~masked] = -100

    input_ids = input_ids.clone()
    input_ids[masked & (mask_type < 0.8)] = mask_token_id
    rnd_idx = masked & (mask_type >= 0.8) & (mask_type < 0.9)
    if rnd_idx.any():
        input_ids[rnd_idx] = torch.randint(
            vocab_size, (int(rnd_idx.sum()),), dtype=torch.long, generator=rng
        )
    return input_ids, labels


def _ppl(loss: float) -> float:
    return math.exp(loss) if loss < 100 else float("inf")


@torch.no_grad()
def bucketed_eval(
    model,
    eval_dataset: Dataset,
    tokenizer,
    *,
    device: torch.device,
    batch_size: int = 64,
    mlm_probability: float = 0.15,
    max_length: int = 128,
    mask_seed: int = 12345,
    bucket_cols: tuple[str, ...] = ("stratum_key", "mhc_class", "n_segments", "subset_key"),
    distributed: bool = False,
) -> dict[str, dict[str, dict[str, float]]]:
    """Run MLM eval on the canonical eval set, grouped by each bucket column.

    Returns ``{bucket_dim: {bucket_value: {loss, perplexity, n_rows, n_masked_tokens}}}``
    plus the special key ``{"overall": {"_": {...}}}``.

    Eval is single-process. In distributed setups, run on rank 0 only and
    broadcast results — callers control this via the ``distributed`` arg.
    """
    available_cols = set(eval_dataset.column_names)
    bucket_cols = tuple(c for c in bucket_cols if c in available_cols)

    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id
    cls_id = tokenizer.cls_token_id
    unk_id = tokenizer.unk_token_id
    sep_id = tokenizer.sep_token_id
    special_ids_list = [x for x in (pad_id, cls_id, eos_id, sep_id, unk_id) if x is not None]
    special_token_ids = torch.tensor(special_ids_list, dtype=torch.long, device=device)

    model.eval()
    rng = torch.Generator(device="cpu").manual_seed(mask_seed)

    # Bucket accumulators: per bucket dim, per bucket value, {loss_sum, n_tokens, n_rows}
    acc: dict[str, dict[Any, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"loss_sum": 0.0, "n_tokens": 0, "n_rows": 0})
    )

    n_examples = len(eval_dataset)

    # We need raw per-row metadata to know which bucket each row in a batch
    # belongs to. Pull all bucket columns up-front (cheap; ~20K rows).
    meta = {c: eval_dataset[c] for c in bucket_cols}

    for batch_start in tqdm(
        range(0, n_examples, batch_size),
        desc="bucketed eval",
        disable=False,
    ):
        batch_end = min(batch_start + batch_size, n_examples)
        rows = list(range(batch_start, batch_end))
        sub = eval_dataset.select(rows)

        # Pad to batch max length.
        seqs = [torch.tensor(x, dtype=torch.long) for x in sub["input_ids"]]
        if not seqs:
            continue
        input_ids = torch.nn.utils.rnn.pad_sequence(
            seqs, batch_first=True, padding_value=pad_id
        )
        lengths = torch.tensor([len(s) for s in seqs])
        max_len = int(input_ids.size(1))
        attention_mask = (
            torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
        ).long()

        masked_ids, labels = _apply_deterministic_mlm_mask(
            input_ids,
            attention_mask,
            special_token_ids=special_token_ids.cpu(),
            mask_token_id=mask_id,
            vocab_size=len(tokenizer),
            mlm_probability=mlm_probability,
            rng=rng,
        )
        masked_ids = masked_ids.to(device)
        attention_mask_d = attention_mask.to(device)
        labels_d = labels.to(device)

        with autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16):
            outputs = model(
                input_ids=masked_ids,
                attention_mask=attention_mask_d,
                labels=labels_d,
            )

        # Per-row loss & masked-token counts.
        logits = outputs.logits  # (B, L, V)
        # Reduction='none' cross-entropy per token, mask via labels != -100.
        flat_logits = logits.view(-1, logits.size(-1)).float()
        flat_labels = labels_d.view(-1)
        token_loss = torch.nn.functional.cross_entropy(
            flat_logits, flat_labels, reduction="none", ignore_index=-100
        )
        token_loss = token_loss.view(labels_d.shape)
        per_row_mask = (labels_d != -100)
        per_row_n_tokens = per_row_mask.sum(dim=1).cpu().tolist()
        per_row_loss_sum = (token_loss * per_row_mask.float()).sum(dim=1).cpu().tolist()

        for i, row_idx in enumerate(rows):
            n_tok = int(per_row_n_tokens[i])
            if n_tok == 0:
                continue  # nothing masked in this row (rare on short seqs)
            l_sum = float(per_row_loss_sum[i])
            # Overall
            acc["overall"]["_"]["loss_sum"] += l_sum
            acc["overall"]["_"]["n_tokens"] += n_tok
            acc["overall"]["_"]["n_rows"] += 1
            for c in bucket_cols:
                val = meta[c][row_idx]
                acc[c][val]["loss_sum"] += l_sum
                acc[c][val]["n_tokens"] += n_tok
                acc[c][val]["n_rows"] += 1

    # Reduce to {bucket_dim: {bucket_value: {loss, perplexity, n_rows, n_masked_tokens}}}
    out: dict[str, dict[str, dict[str, float]]] = {}
    for dim, vals in acc.items():
        out[dim] = {}
        for val, stats in vals.items():
            if stats["n_tokens"] == 0:
                continue
            loss = stats["loss_sum"] / stats["n_tokens"]
            out[dim][str(val)] = {
                "loss": loss,
                "perplexity": _ppl(loss),
                "n_rows": int(stats["n_rows"]),
                "n_masked_tokens": int(stats["n_tokens"]),
            }
    return out


def flatten_bucketed_for_wandb(buckets: dict, prefix: str = "canonical") -> dict[str, float]:
    """Flatten the nested bucketed_eval output into wandb-friendly scalars."""
    flat: dict[str, float] = {}
    for dim, vals in buckets.items():
        for val, m in vals.items():
            key = f"{prefix}/{dim}/{val}/perplexity" if dim != "overall" else f"{prefix}/overall/perplexity"
            flat[key] = m["perplexity"]
            lkey = f"{prefix}/{dim}/{val}/loss" if dim != "overall" else f"{prefix}/overall/loss"
            flat[lkey] = m["loss"]
    return flat


# ---------------------------------------------------------------------------
# Region-restricted evaluation on canonical_eval_v2 (Stage 2+)
# ---------------------------------------------------------------------------


def _model_forward_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Return per-token logits for ESM-2 (HF MLM) OR ESM-C base models, accounting
    for PEFT and DDP wrapping.

    For PEFT-wrapped ESM-C we bypass the PEFT forward (which remaps kwargs to
    HF-style names that ESMC doesn't accept) and call the underlying ESMC
    directly. LoRA adapters are still applied because they're injected into the
    base Linear modules.
    """
    # Unwrap DDP
    m = model.module if hasattr(model, "module") else model
    # Unwrap torch.compile
    m = getattr(m, "_orig_mod", m)
    # PEFT-wrapped ESM-C
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        inner = m.base_model.model
        if type(inner).__name__.upper() == "ESMC":
            seq_id = attention_mask.bool()
            out = inner(sequence_tokens=input_ids, sequence_id=seq_id)
            return out.sequence_logits
    # Raw ESM-C
    if type(m).__name__.upper() == "ESMC":
        seq_id = attention_mask.bool()
        out = m(sequence_tokens=input_ids, sequence_id=seq_id)
        return out.sequence_logits
    # Default: HF MaskedLM (ESM-2 etc.)
    out = m(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits


def _collect_special_token_ids(tokenizer) -> set[int]:
    out: set[int] = set()
    for attr in (
        "pad_token_id",
        "cls_token_id",
        "eos_token_id",
        "sep_token_id",
        "unk_token_id",
        "bos_token_id",
        "mask_token_id",
    ):
        sid = getattr(tokenizer, attr, None)
        if sid is not None:
            out.add(int(sid))
    return out


def _preprocess_for_eos_strategy(text: str, eos_strategy: str) -> str:
    """Apply the chosen preprocessing for ``<eos>`` separators.

    Strategies (chosen per-tokenizer via Stage 2 verification):
      - ``"raw"``: no preprocessing. Both ESM-2 and ESM-C tokenizers natively
        emit a single ``<eos>`` id for the literal ``<eos>`` string.
      - ``"fixb"``: pad ``<eos>`` with whitespace. Works for ESM-2 (kept for
        Stage 0/1 backward compatibility). BREAKS ESM-C because the whitespace
        is not in its vocabulary.
      - ``"pipe_sub"``: substitute ``<eos>`` with pipe ``|``. ESM-C-only
        fallback; pipe is ESM-C separator token id 31.
    """
    if eos_strategy == "raw":
        return text
    if eos_strategy == "fixb":
        return preprocess_for_tokenizer(text)
    if eos_strategy == "pipe_sub":
        return text.replace(EOS_LITERAL, "|")
    raise ValueError(f"unknown eos_strategy: {eos_strategy}")


@torch.no_grad()
def region_restricted_eval(
    model,
    canonical_eval_v2: Dataset,
    tcr_variant: str,
    mhc_variant: str,
    tokenizer,
    *,
    device: torch.device,
    eos_strategy: str = "fixb",
    separator_token_id: int | None = None,
    regions: tuple[str, ...] = (
        "overall",
        "cdr3",
        "cdr12",
        "pocket",
        "pocket_contact_only",
        "peptide",
        "framework",
    ),
    bucket_cols: tuple[str, ...] = (
        "stratum_key",
        "mhc_class",
        "n_segments",
        "subset_key",
    ),
    mlm_probability: float = 0.15,
    mask_seed: int = 12345,
    max_length: int = 1024,
    batch_size: int = 16,
    verbose: bool = True,
) -> dict:
    """Region-restricted MLM eval on canonical_eval_v2.

    Pipeline per row:
      1. Project to ``(tcr_variant, mhc_variant)`` input_text via
         ``project_canonical_row_to_run_input``.
      2. Apply EOS strategy (Fix B whitespace pad, or substitute ``<eos>`` →
         ``|`` for ESM-C tokenizer without recognized ``<eos>``).
      3. Tokenize. Get input_ids + attention_mask.
      4. Annotate per-token region via ``annotate_token_regions``.
      5. Apply 15% MLM masking to all valid (non-special) tokens
         (deterministic via ``mask_seed``).
      6. Forward pass; per-token cross-entropy loss.
      7. Each masked token's loss is attributed to its region, and accumulated
         per ``(region, bucket_col, bucket_value)``.

    Returns:
      {region: {bucket_col: {bucket_value: {loss, perplexity, n_rows,
       n_masked_tokens}}}} plus {region: {"overall": {"_": {...}}}}
    """
    if eos_strategy in ("fixb", "raw"):
        sep_id = int(tokenizer.eos_token_id)
    elif eos_strategy == "pipe_sub":
        if separator_token_id is None:
            raise ValueError("pipe_sub mode requires separator_token_id")
        sep_id = int(separator_token_id)
    else:
        raise ValueError(f"unknown eos_strategy: {eos_strategy}")

    pad_id = int(tokenizer.pad_token_id)
    mask_id = int(tokenizer.mask_token_id)
    vocab_size = len(tokenizer)
    special_ids = _collect_special_token_ids(tokenizer)
    special_ids.add(sep_id)

    available_buckets = tuple(c for c in bucket_cols if c in canonical_eval_v2.column_names)

    model.eval()
    rng = torch.Generator(device="cpu").manual_seed(int(mask_seed))

    # acc[region][dim][val] = {loss_sum, n_tokens, n_rows_set}
    acc: dict[str, dict[str, dict[Any, dict[str, Any]]]] = {
        r: defaultdict(lambda: defaultdict(lambda: {"loss_sum": 0.0, "n_tokens": 0, "rows": set()}))
        for r in regions
    }

    # Pre-pull metadata columns to avoid per-row dataset access overhead.
    meta_cols: dict[str, list] = {c: canonical_eval_v2[c] for c in available_buckets}
    n_examples = len(canonical_eval_v2)

    # Tokenize + annotate per row (Python; eval set is ~20k rows).
    # Then group rows into batches and run forward passes.
    cached: list[dict] = []
    annot_fail_rows: list[int] = []
    empty_rows: list[int] = []
    for i in range(n_examples):
        row = canonical_eval_v2[i]
        text, kinds = project_canonical_row_to_run_input(row, tcr_variant, mhc_variant)
        if not text:
            empty_rows.append(i)
            continue
        prepped = _preprocess_for_eos_strategy(text, eos_strategy)
        enc = tokenizer(
            prepped,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        ids = list(enc["input_ids"])
        # Defensive: some ESM-C tokenizer paths may return tensors.
        if hasattr(ids[0], "item"):
            ids = [int(x) for x in ids]
        # attention_mask may not be returned by every tokenizer; default to all 1.
        am = enc.get("attention_mask")
        if am is None:
            am = [1] * len(ids)
        token_regions = annotate_token_regions(
            text,
            row,
            kinds,
            ids,
            separator_token_id=sep_id,
            special_token_ids=special_ids,
        )
        # Verify segment-count parity once per row; record annotation failures.
        n_segments_in_text = text.count(EOS_LITERAL) + 1
        if n_segments_in_text != len(kinds):
            annot_fail_rows.append(i)
        cached.append({
            "row_idx": i,
            "input_ids": ids,
            "attention_mask": am,
            "token_regions": token_regions,
            "length": len(ids),
        })

    if verbose:
        print(
            f"  region_restricted_eval: tokenized {len(cached):,}/{n_examples:,} rows "
            f"(empty={len(empty_rows)}, annot_fail={len(annot_fail_rows)})",
            flush=True,
        )

    # Sort by length for tighter batches (length bucketing reduces padding waste).
    cached.sort(key=lambda x: x["length"])

    for batch_start in tqdm(
        range(0, len(cached), batch_size),
        desc="region_restricted_eval",
        disable=not verbose,
    ):
        batch = cached[batch_start: batch_start + batch_size]
        if not batch:
            continue
        seqs = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
        ams = [torch.tensor(b["attention_mask"], dtype=torch.long) for b in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(ams, batch_first=True, padding_value=0)
        max_len = int(input_ids.size(1))

        # Build a parallel per-token region tensor for attribution.
        region_table: list[list[str]] = []
        for b in batch:
            tr = list(b["token_regions"])
            if len(tr) < max_len:
                tr = tr + ["special"] * (max_len - len(tr))
            elif len(tr) > max_len:
                tr = tr[:max_len]
            region_table.append(tr)

        # Deterministic mask: 80/10/10 like training, only on valid tokens
        # (attention_mask == 1 AND not in special_token_ids).
        special_ids_t = torch.tensor(sorted(special_ids), dtype=torch.long)
        rand_vals = torch.rand(*input_ids.shape, 2, generator=rng)
        rand_mask = rand_vals[..., 0]
        mask_type = rand_vals[..., 1]
        special_mask = torch.isin(input_ids, special_ids_t)
        valid = (attention_mask == 1) & ~special_mask
        masked = valid & (rand_mask < mlm_probability)
        labels = input_ids.clone()
        labels[~masked] = -100
        masked_ids = input_ids.clone()
        masked_ids[masked & (mask_type < 0.8)] = mask_id
        rnd_idx = masked & (mask_type >= 0.8) & (mask_type < 0.9)
        if rnd_idx.any():
            masked_ids[rnd_idx] = torch.randint(
                vocab_size, (int(rnd_idx.sum()),), dtype=torch.long, generator=rng
            )

        masked_ids = masked_ids.to(device)
        attention_mask_d = attention_mask.to(device)
        labels_d = labels.to(device)

        with autocast(
            device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16
        ):
            logits = _model_forward_logits(model, masked_ids, attention_mask_d)

        flat_logits = logits.view(-1, logits.size(-1)).float()
        flat_labels = labels_d.view(-1)
        token_loss = torch.nn.functional.cross_entropy(
            flat_logits, flat_labels, reduction="none", ignore_index=-100
        )
        token_loss = token_loss.view(labels_d.shape).cpu()
        per_mask = (labels_d != -100).cpu()

        for bi, b in enumerate(batch):
            row_idx = b["row_idx"]
            tr = region_table[bi]
            for pos in range(max_len):
                if not bool(per_mask[bi, pos]):
                    continue
                tl = float(token_loss[bi, pos])
                # Attribute to 'overall' always
                acc["overall"]["overall"]["_"]["loss_sum"] += tl
                acc["overall"]["overall"]["_"]["n_tokens"] += 1
                acc["overall"]["overall"]["_"]["rows"].add(row_idx)
                for dim in available_buckets:
                    val = meta_cols[dim][row_idx]
                    acc["overall"][dim][val]["loss_sum"] += tl
                    acc["overall"][dim][val]["n_tokens"] += 1
                    acc["overall"][dim][val]["rows"].add(row_idx)
                # Attribute to the token's region
                region = tr[pos]
                if region == "special":
                    continue
                if region in acc:
                    acc[region]["overall"]["_"]["loss_sum"] += tl
                    acc[region]["overall"]["_"]["n_tokens"] += 1
                    acc[region]["overall"]["_"]["rows"].add(row_idx)
                    for dim in available_buckets:
                        val = meta_cols[dim][row_idx]
                        acc[region][dim][val]["loss_sum"] += tl
                        acc[region][dim][val]["n_tokens"] += 1
                        acc[region][dim][val]["rows"].add(row_idx)

    # Reduce.
    out: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for region, dim_map in acc.items():
        out[region] = {}
        for dim, val_map in dim_map.items():
            out[region][dim] = {}
            for val, stats in val_map.items():
                n_tok = stats["n_tokens"]
                if n_tok == 0:
                    continue
                loss = stats["loss_sum"] / n_tok
                out[region][dim][str(val)] = {
                    "loss": loss,
                    "perplexity": _ppl(loss),
                    "n_rows": len(stats["rows"]),
                    "n_masked_tokens": int(n_tok),
                }
    out["_diagnostics"] = {  # type: ignore[assignment]
        "n_rows_total": n_examples,
        "n_rows_tokenized": len(cached),
        "n_rows_empty": len(empty_rows),
        "n_rows_annot_fail": len(annot_fail_rows),
        "tcr_variant": tcr_variant,
        "mhc_variant": mhc_variant,
        "eos_strategy": eos_strategy,
        "mlm_probability": mlm_probability,
        "mask_seed": mask_seed,
    }
    return out


def flatten_region_eval_for_wandb(buckets: dict, prefix: str = "canonical_v2") -> dict[str, float]:
    """Flatten region_restricted_eval output into wandb-friendly scalars.

    Emits keys like ``canonical_v2/cdr3/overall/perplexity``,
    ``canonical_v2/cdr3/stratum_key/none_nseg2/perplexity``, etc.
    """
    flat: dict[str, float] = {}
    for region, dim_map in buckets.items():
        if region.startswith("_"):
            continue  # skip _diagnostics
        for dim, val_map in dim_map.items():
            for val, m in val_map.items():
                if dim == "overall":
                    pkey = f"{prefix}/{region}/overall/perplexity"
                    lkey = f"{prefix}/{region}/overall/loss"
                else:
                    pkey = f"{prefix}/{region}/{dim}/{val}/perplexity"
                    lkey = f"{prefix}/{region}/{dim}/{val}/loss"
                flat[pkey] = m["perplexity"]
                flat[lkey] = m["loss"]
    return flat


# ---------------------------------------------------------------------------
# Early-stopping callback driven by region_restricted_eval's cdr3_ppl
# ---------------------------------------------------------------------------


class CdrPplPlateauCallback:
    """Stop training when canonical ``cdr3_ppl`` plateaus.

    Walks ``metric_path`` through the nested region_restricted_eval result to
    extract the scalar metric, then compares to running best. If the last
    ``n_patience`` evaluations all show relative improvement less than
    ``min_rel_improvement`` (vs running best), returns True (stop).

    Default ``metric_path`` reads
    ``buckets['cdr3']['overall']['_']['perplexity']``.
    """

    def __init__(
        self,
        n_patience: int = 3,
        min_rel_improvement: float = 0.01,
        metric_path: tuple[str, ...] = ("cdr3", "overall", "_", "perplexity"),
    ):
        self.n_patience = int(n_patience)
        self.min_rel_improvement = float(min_rel_improvement)
        self.metric_path = tuple(metric_path)
        self.best: float = float("inf")
        self.history: list[dict] = []

    def _get_metric(self, buckets: dict) -> float | None:
        cur = buckets
        try:
            for k in self.metric_path:
                cur = cur[k]
        except (KeyError, TypeError):
            return None
        return float(cur)

    def __call__(self, buckets: dict) -> bool:
        m = self._get_metric(buckets)
        if m is None or not math.isfinite(m):
            self.history.append({"metric": None, "best": self.best, "improved": False})
            return False
        improved = m < self.best * (1.0 - self.min_rel_improvement)
        if m < self.best:
            self.best = m
        self.history.append({"metric": m, "best": self.best, "improved": improved})
        if len(self.history) < self.n_patience:
            return False
        recent = self.history[-self.n_patience:]
        # Stop only if NONE of the last `n_patience` evals improved meaningfully.
        return not any(h["improved"] for h in recent)

    def state_dict(self) -> dict:
        return {
            "n_patience": self.n_patience,
            "min_rel_improvement": self.min_rel_improvement,
            "metric_path": list(self.metric_path),
            "best": self.best,
            "history": self.history,
        }


# ---------------------------------------------------------------------------
# CPT Trainer
# ---------------------------------------------------------------------------


@dataclass
class CPTConfig:
    base_model: str = "facebook/esm2_t30_150M_UR50D"
    dataset_path: str = ""
    canonical_eval_path: str = ""
    output_dir: str = ""

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | str = "auto"

    # Optim
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    warmup_ratio: float = 0.05
    gradient_clip: float = 1.0

    # Schedule
    num_epochs: int = 1
    batch_size: int = 256
    gradient_accumulation_steps: int = 2
    eval_every_n_steps: int = 500
    save_every_n_steps: int = 2000
    logging_steps: int = 50

    # Data
    max_length: int = 128
    mlm_probability: float = 0.15
    eos_strategy: str = "fixb"  # "raw" | "fixb" | "pipe_sub"

    # Runtime
    attn_implementation: str = "flash_attention_2"
    use_compile: bool = True
    use_length_bucketing: bool = True
    bucket_boundaries: list[int] = field(default_factory=lambda: [32, 48, 64, 96, 128])
    dataloader_num_workers: int = 4
    eval_batch_size: int = 64
    gradient_checkpointing: bool = False

    # Canonical eval mode
    # If region_eval is True, the canonical eval callback runs
    # region_restricted_eval on canonical_eval_path (which must be a v2-format
    # dataset with full molecular columns), instead of bucketed_eval.
    region_eval: bool = False
    tcr_variant: str = "cdr3"      # for region_restricted_eval projection
    mhc_variant: str = "pocket"
    region_eval_batch_size: int = 16

    # Early stop & best
    early_stopping_patience: int = 8
    # When region_eval is True, the early-stop callback is CdrPplPlateauCallback
    # monitoring the metric at ``early_stop_metric_path`` (default cdr3 ppl).
    early_stop_metric_path: tuple = ("cdr3", "overall", "_", "perplexity")
    early_stop_n_patience: int = 3
    early_stop_min_rel: float = 0.01

    # Wandb
    wandb_project: str = "tcrbench-v3-cpt"
    wandb_run_name: str | None = None
    report_to: str = "wandb"

    # Reproducibility
    seed: int = 42


class CPTTrainer:
    """DDP MLM trainer specialised for CPT.

    Lifecycle mirrors ``scripts/training/esm_native_trainer.py`` (DDP, bf16
    autocast, optional torch.compile, cosine LR with warmup) but evaluates
    against the canonical eval set every ``eval_every_n_steps`` and selects
    ``best_model`` based on ``canonical/overall/loss`` rather than the run's
    native val split.
    """

    def __init__(self, cfg: CPTConfig):
        self.cfg = cfg
        self.is_distributed = "LOCAL_RANK" in os.environ
        if self.is_distributed:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.global_rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(
                backend="nccl", init_method="env://", timeout=timedelta(minutes=30)
            )
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.local_rank = 0
            self.global_rank = 0
            self.world_size = 1
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.set_float32_matmul_precision("high")
        torch.manual_seed(cfg.seed + self.global_rank)

        self._setup_tokenizer_and_model()
        self._setup_data()
        self._setup_optimizer()

        self.early_stopping = EarlyStopping(
            patience=cfg.early_stopping_patience, min_delta=1e-4, mode="min"
        )
        self.global_step = 0
        self.best_eval_loss = float("inf")
        self.should_stop = False
        self.last_eval_result: dict | None = None

        # WandB (rank 0 only).
        self.use_wandb = cfg.report_to == "wandb" and self.global_rank == 0
        if self.use_wandb:
            wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                config={k: v for k, v in cfg.__dict__.items() if not callable(v)},
            )

    # -- setup helpers -------------------------------------------------------

    def _is_main(self) -> bool:
        return self.global_rank == 0

    def _log(self, msg: str) -> None:
        if self._is_main():
            print(msg, flush=True)

    def _setup_tokenizer_and_model(self) -> None:
        self.tokenizer = get_tokenizer(self.cfg.base_model)
        self.is_esmc = _is_esmc_model_name(self.cfg.base_model)
        model, info = setup_lora_model(
            self.cfg.base_model,
            rank=self.cfg.lora_r,
            alpha=self.cfg.lora_alpha,
            dropout=self.cfg.lora_dropout,
            target_modules=self.cfg.lora_target_modules,
            attn_implementation=self.cfg.attn_implementation,
        )
        self.lora_info = info
        self._log(
            f"LoRA targets={info['target_modules']}  "
            f"trainable={info['trainable_params']:,}/{info['total_params']:,} "
            f"({info['trainable_pct']:.2f}%)  ESM-C={self.is_esmc}"
        )
        model = model.to(self.device)
        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], find_unused_parameters=False)
        if self.cfg.use_compile:
            self._log("Compiling model with torch.compile (mode='default') ...")
            model = torch.compile(model, mode="default")
        self.model = model

        # Initialize the region-eval early-stop callback if requested.
        if self.cfg.region_eval:
            self.cdr3_plateau = CdrPplPlateauCallback(
                n_patience=self.cfg.early_stop_n_patience,
                min_rel_improvement=self.cfg.early_stop_min_rel,
                metric_path=tuple(self.cfg.early_stop_metric_path),
            )
        else:
            self.cdr3_plateau = None

    def _setup_data(self) -> None:
        train_path = os.path.join(self.cfg.dataset_path, "train")
        val_path = os.path.join(self.cfg.dataset_path, "validation")
        if not os.path.exists(val_path):
            val_path = os.path.join(self.cfg.dataset_path, "val")

        # When region_eval is True, the canonical eval dataset is v2 (has
        # molecular columns; region_restricted_eval handles its own
        # tokenization/projection per row). For legacy bucketed_eval mode the
        # canonical eval is pre-tokenized like train/val.
        def _load_train_val(num_proc):
            ds = load_cpt_dataset(
                train_path,
                self.tokenizer,
                max_length=self.cfg.max_length,
                num_proc=num_proc,
                eos_strategy=self.cfg.eos_strategy,
            )
            vds = load_cpt_dataset(
                val_path,
                self.tokenizer,
                max_length=self.cfg.max_length,
                num_proc=num_proc,
                eos_strategy=self.cfg.eos_strategy,
            )
            return ds, vds

        # Tokenize on rank 0 only; barriers ensure other ranks read the cache.
        if self._is_main():
            self.train_ds, self.val_ds = _load_train_val(num_proc=8)
            if self.cfg.region_eval:
                self.canonical_ds = load_from_disk(self.cfg.canonical_eval_path)
            else:
                self.canonical_ds = load_cpt_dataset(
                    self.cfg.canonical_eval_path,
                    self.tokenizer,
                    max_length=self.cfg.max_length,
                    add_length_col=False,
                    eos_strategy=self.cfg.eos_strategy,
                )
        if self.is_distributed:
            dist.barrier()
        if not self._is_main():
            self.train_ds, self.val_ds = _load_train_val(num_proc=1)
            if self.cfg.region_eval:
                self.canonical_ds = load_from_disk(self.cfg.canonical_eval_path)
            else:
                self.canonical_ds = load_cpt_dataset(
                    self.cfg.canonical_eval_path,
                    self.tokenizer,
                    max_length=self.cfg.max_length,
                    add_length_col=False,
                    num_proc=1,
                    eos_strategy=self.cfg.eos_strategy,
                )

        self._log(f"Train rows: {len(self.train_ds):,}")
        self._log(f"Val rows:   {len(self.val_ds):,}")
        self._log(f"Canonical eval rows: {len(self.canonical_ds):,}")

        # Collator: standard dynamic MLM, eos already excluded as special token.
        self.collator = DataCollatorForMLMDynamic(
            tokenizer=self.tokenizer,
            mlm_probability=self.cfg.mlm_probability,
            pad_to_multiple_of=8,
            separator_token_id=self.tokenizer.eos_token_id,
        )

        # Length-bucketed sampler if 'length' column is present.
        if self.cfg.use_length_bucketing and "length" in self.train_ds.column_names:
            lengths = np.array(self.train_ds["length"])
            if self.is_distributed:
                self.train_sampler = DistributedLengthBucketSampler(
                    lengths=lengths,
                    batch_size=self.cfg.batch_size,
                    bucket_boundaries=self.cfg.bucket_boundaries,
                    shuffle=True,
                    drop_last=True,
                    num_replicas=self.world_size,
                    rank=self.global_rank,
                )
            else:
                self.train_sampler = LengthBucketSampler(
                    lengths=lengths,
                    batch_size=self.cfg.batch_size,
                    bucket_boundaries=self.cfg.bucket_boundaries,
                    shuffle=True,
                    drop_last=True,
                )
            self.train_loader = DataLoader(
                self.train_ds,
                batch_size=self.cfg.batch_size,
                sampler=self.train_sampler,
                num_workers=self.cfg.dataloader_num_workers,
                pin_memory=True,
                collate_fn=self.collator,
                persistent_workers=self.cfg.dataloader_num_workers > 0,
            )
        else:
            if self.is_distributed:
                self.train_sampler = DistributedSampler(
                    self.train_ds, shuffle=True, drop_last=True
                )
            else:
                self.train_sampler = None
            self.train_loader = DataLoader(
                self.train_ds,
                batch_size=self.cfg.batch_size,
                sampler=self.train_sampler,
                shuffle=(self.train_sampler is None),
                num_workers=self.cfg.dataloader_num_workers,
                pin_memory=True,
                collate_fn=self.collator,
                persistent_workers=self.cfg.dataloader_num_workers > 0,
                drop_last=True,
            )

    def _setup_optimizer(self) -> None:
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
            fused=True,
        )
        steps_per_epoch = max(1, len(self.train_loader) // self.cfg.gradient_accumulation_steps)
        total_steps = steps_per_epoch * self.cfg.num_epochs
        warmup_steps = int(total_steps * self.cfg.warmup_ratio)
        self.total_steps = total_steps

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self._log(
            f"Total steps: {total_steps}  warmup: {warmup_steps}  "
            f"steps/epoch: {steps_per_epoch}"
        )

    # -- training -----------------------------------------------------------

    def train(self) -> None:
        cfg = self.cfg
        self._log(f"\nStarting training: {cfg.num_epochs} epoch(s), bs={cfg.batch_size}*"
                  f"ga{cfg.gradient_accumulation_steps}*ws{self.world_size}="
                  f"{cfg.batch_size*cfg.gradient_accumulation_steps*self.world_size} effective\n")
        for epoch in range(cfg.num_epochs):
            if self.is_distributed and hasattr(self.train_sampler, "set_epoch"):
                self.train_sampler.set_epoch(epoch)
            self._train_one_epoch(epoch)
            if self.should_stop:
                self._log(f"Stopping after epoch {epoch+1} due to early-stop trigger")
                break
        # Final eval + save.
        self._run_canonical_eval(final=True)
        self._save_final()
        if self.use_wandb:
            wandb.finish()
        if self.is_distributed:
            dist.barrier()
            dist.destroy_process_group()

    def _train_one_epoch(self, epoch: int) -> None:
        cfg = self.cfg
        self.model.train()
        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"epoch {epoch+1}",
            disable=not self._is_main(),
        )
        self.optimizer.zero_grad()
        accum_loss = 0.0

        for step, batch in pbar:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                if self.is_esmc:
                    logits = _model_forward_logits(self.model, input_ids, attention_mask)
                    raw_loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100,
                    )
                else:
                    out = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    raw_loss = out.loss
                loss = raw_loss / cfg.gradient_accumulation_steps
            loss.backward()
            accum_loss += loss.item()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.gradient_clip
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                pbar.set_postfix(
                    {"loss": f"{accum_loss:.4f}",
                     "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"}
                )
                if self._is_main() and self.global_step % cfg.logging_steps == 0 and self.use_wandb:
                    wandb.log(
                        {
                            "train/loss": accum_loss,
                            "train/perplexity": _ppl(accum_loss),
                            "train/grad_norm": float(grad_norm) if torch.is_tensor(grad_norm) else grad_norm,
                            "train/learning_rate": self.scheduler.get_last_lr()[0],
                            "train/global_step": self.global_step,
                            "train/epoch_frac": epoch + (step + 1) / len(self.train_loader),
                        }
                    )
                accum_loss = 0.0

                if self.global_step % cfg.eval_every_n_steps == 0:
                    self._run_canonical_eval()
                    self.model.train()
                    if self.should_stop:
                        self._log(
                            f"  Early stop triggered at step {self.global_step} "
                            f"(cdr3 plateau)"
                        )
                        pbar.close()
                        return

                if self.global_step % cfg.save_every_n_steps == 0:
                    self._save_checkpoint(epoch, tag=f"step_{self.global_step}")

        pbar.close()

    def _run_canonical_eval(self, final: bool = False) -> None:
        """Run canonical eval on rank 0; broadcast scalar metric to all ranks
        for early-stopping rank-consistency.

        Mode-aware:
          - cfg.region_eval=False: classical ``bucketed_eval`` (Stage 0/1
            behavior). Best checkpoint by ``overall_loss``.
          - cfg.region_eval=True: region_restricted_eval against canonical
            eval v2. Best checkpoint by ``cdr3_ppl``; early stop via
            ``CdrPplPlateauCallback``. Also persists per-region history to
            ``{output_dir}/early_stop_history.json``.
        """
        cfg = self.cfg
        # Default for non-main ranks; will be broadcast over by main.
        eval_metric = float("inf")
        stop_now = False

        if self._is_main():
            unwrapped = self.model.module if isinstance(self.model, DDP) else self.model
            if cfg.region_eval:
                sep_token_id = (
                    None
                    if cfg.eos_strategy != "pipe_sub"
                    else 31  # ESM-C pipe id
                )
                regions_eval = region_restricted_eval(
                    unwrapped,
                    self.canonical_ds,
                    cfg.tcr_variant,
                    cfg.mhc_variant,
                    self.tokenizer,
                    device=self.device,
                    eos_strategy=cfg.eos_strategy,
                    separator_token_id=sep_token_id,
                    mlm_probability=cfg.mlm_probability,
                    max_length=cfg.max_length,
                    batch_size=cfg.region_eval_batch_size,
                )
                self.last_eval_result = regions_eval
                overall = regions_eval["overall"]["overall"]["_"]
                cdr3 = regions_eval.get("cdr3", {}).get("overall", {}).get("_")
                cdr3_ppl = cdr3["perplexity"] if cdr3 else float("inf")
                fw = regions_eval.get("framework", {}).get("overall", {}).get("_")
                fw_ppl = fw["perplexity"] if fw else float("nan")
                self._log(
                    f"\n[canonical_v2 eval @ step {self.global_step}] "
                    f"overall_ppl={overall['perplexity']:.3f}  "
                    f"cdr3_ppl={cdr3_ppl:.3f}  framework_ppl={fw_ppl:.3f}"
                )
                for region in ("cdr3", "cdr12", "pocket", "pocket_contact_only", "peptide", "framework"):
                    rb = regions_eval.get(region, {}).get("stratum_key", {})
                    if not rb:
                        continue
                    self._log(f"  [{region}]")
                    for val, m in sorted(rb.items()):
                        self._log(
                            f"    {val:>22}  ppl={m['perplexity']:>6.3f}  "
                            f"n_rows={m['n_rows']:>5,}  n_masked={m['n_masked_tokens']:>7,}"
                        )
                if self.use_wandb:
                    flat = flatten_region_eval_for_wandb(regions_eval)
                    flat["canonical_v2/global_step"] = self.global_step
                    wandb.log(flat)

                # Save best by cdr3_ppl.
                is_best = cdr3_ppl < self.best_eval_loss
                if is_best:
                    self.best_eval_loss = cdr3_ppl
                    self._save_checkpoint(epoch=-1, tag="best", buckets=regions_eval)

                # Early stopping decision.
                if self.cdr3_plateau is not None:
                    stop_now = bool(self.cdr3_plateau(regions_eval))
                    history_path = os.path.join(cfg.output_dir, "early_stop_history.json")
                    with open(history_path, "w") as f:
                        json.dump(self.cdr3_plateau.state_dict(), f, indent=2, default=str)

                eval_metric = float(cdr3_ppl)
            else:
                buckets = bucketed_eval(
                    unwrapped,
                    self.canonical_ds,
                    self.tokenizer,
                    device=self.device,
                    batch_size=cfg.eval_batch_size,
                    mlm_probability=cfg.mlm_probability,
                    max_length=cfg.max_length,
                    distributed=False,
                )
                overall = buckets["overall"]["_"]
                self._log(
                    f"\n[canonical eval @ step {self.global_step}] "
                    f"overall loss={overall['loss']:.4f} ppl={overall['perplexity']:.3f} "
                    f"n_rows={overall['n_rows']:,} n_masked={overall['n_masked_tokens']:,}"
                )
                for dim in ("stratum_key",):
                    if dim in buckets:
                        for val, m in sorted(buckets[dim].items()):
                            self._log(
                                f"    [{dim}={val:>22}] ppl={m['perplexity']:>6.3f}  "
                                f"n_rows={m['n_rows']:>6,}  n_masked={m['n_masked_tokens']:>7,}"
                            )
                if self.use_wandb:
                    flat = flatten_bucketed_for_wandb(buckets)
                    flat["canonical/global_step"] = self.global_step
                    wandb.log(flat)

                eval_loss = overall["loss"]
                is_best = eval_loss < self.best_eval_loss
                if is_best:
                    self.best_eval_loss = eval_loss
                    self._save_checkpoint(epoch=-1, tag="best", buckets=buckets)
                eval_metric = float(eval_loss)

        # Broadcast scalar eval_metric and stop signal to all ranks.
        if self.is_distributed:
            t = torch.tensor([eval_metric, 1.0 if stop_now else 0.0], device=self.device)
            dist.broadcast(t, src=0)
            eval_metric = float(t[0].item())
            stop_now = bool(t[1].item() >= 0.5)

        # Early-stopping decision (rank-consistent).
        if not final:
            if cfg.region_eval:
                # CdrPplPlateauCallback already evaluated on rank 0.
                # If it fired, we just set a flag — let train loop break.
                self.should_stop = self.should_stop or stop_now
            else:
                self.early_stopping(eval_metric)

    def _save_checkpoint(self, epoch: int, tag: str, buckets: dict | None = None) -> None:
        if not self._is_main():
            return
        out_dir = os.path.join(self.cfg.output_dir, f"checkpoint_{tag}")
        os.makedirs(out_dir, exist_ok=True)
        model = self.model.module if isinstance(self.model, DDP) else self.model
        unwrapped = getattr(model, "_orig_mod", model)  # undo torch.compile wrapper
        unwrapped.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        meta = {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
            "lora_info": self.lora_info,
            "config": {k: v for k, v in self.cfg.__dict__.items() if not callable(v)},
        }
        if buckets is not None:
            meta["canonical_buckets"] = buckets
        with open(os.path.join(out_dir, "trainer_state.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        self._log(f"  saved checkpoint: {out_dir}")

        # Cleanup old step checkpoints (keep 3).
        step_ckpts = sorted(glob.glob(os.path.join(self.cfg.output_dir, "checkpoint_step_*")))
        for old in step_ckpts[:-3]:
            import shutil
            shutil.rmtree(old, ignore_errors=True)

    def _save_final(self) -> None:
        if not self._is_main():
            return
        out_dir = os.path.join(self.cfg.output_dir, "final")
        os.makedirs(out_dir, exist_ok=True)
        model = self.model.module if isinstance(self.model, DDP) else self.model
        unwrapped = getattr(model, "_orig_mod", model)
        try:
            unwrapped.save_pretrained(out_dir)
        except Exception as e:
            self._log(f"  WARN: save_pretrained failed ({type(e).__name__}: {e})")
        try:
            self.tokenizer.save_pretrained(out_dir)
        except Exception as e:
            self._log(f"  WARN: tokenizer save_pretrained failed ({type(e).__name__}: {e})")
        with open(os.path.join(out_dir, "trainer_state.json"), "w") as f:
            json.dump(
                {
                    "global_step": self.global_step,
                    "best_eval_loss": self.best_eval_loss,
                    "lora_info": self.lora_info,
                    "config": {k: v for k, v in self.cfg.__dict__.items() if not callable(v)},
                },
                f,
                indent=2,
                default=str,
            )
        # If region-restricted final eval ran, persist it next to the model
        # for downstream verifier / summary scripts to read.
        if self.cfg.region_eval and self.last_eval_result is not None:
            cv2_path = os.path.join(out_dir, "canonical_v2_results.json")
            import datetime
            payload = {
                "config": {
                    "base_model": self.cfg.base_model,
                    "tcr_variant": self.cfg.tcr_variant,
                    "mhc_variant": self.cfg.mhc_variant,
                    "eos_strategy": self.cfg.eos_strategy,
                },
                "global_step": self.global_step,
                "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                "result": self.last_eval_result,
            }
            with open(cv2_path, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            self._log(f"  saved canonical_v2_results.json: {cv2_path}")
        self._log(f"  saved final model: {out_dir}")
