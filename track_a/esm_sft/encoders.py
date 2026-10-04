"""ESM2 and ESM-C encoder loaders for Track A.

Implements the verified loading patterns from `docs/CPT_HF_CHECKPOINTS.md`.

Variants:
    esm2_8m  (base | cpt)
    esm2_35m (base | cpt)
    esmc_300m (base | cpt)

For ESM2 (base): HuggingFace `AutoModel.from_pretrained(facebook/esm2_t6_8M_UR50D)` etc.
For ESM2 (cpt):  base + PEFT adapter from HF (`sravisha/esm2-{8m,35m}-cpt-fnd100m`)
For ESM-C: load via EvolutionaryScale `esm` package, then attach PEFT adapter for cpt variant.

The base encoder is wrapped in a small adapter `EsmEncoderAdapter` that exposes a
unified `forward(input_ids, attention_mask) -> (B, L, hidden)` interface across
ESM2 and ESM-C.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

# HF repo + adapter ids
ESM2_8M_BASE = "facebook/esm2_t6_8M_UR50D"
ESM2_35M_BASE = "facebook/esm2_t12_35M_UR50D"
ESM2_8M_CPT_ADAPTER = "sravisha/esm2-8m-cpt-fnd100m"
ESM2_35M_CPT_ADAPTER = "sravisha/esm2-35m-cpt-fnd100m"

ESMC_BASE_ID = "esmc_300m"  # passed to ESMC.from_pretrained
ESMC_CPT_ADAPTER = "sravisha/esmc300m-foundation100m-c3-lora-r32"


@dataclass
class LoadedEncoder:
    encoder: nn.Module          # the underlying model (with PEFT adapter if cpt)
    tokenizer: object           # HF AutoTokenizer for esm2; EsmSequenceTokenizer for esmc
    hidden_dim: int             # last-hidden-state dimension
    is_esmc: bool               # True for esm-c, False for esm2


def load_esm2(variant: str, base_id: str, cpt_adapter_id: Optional[str]) -> LoadedEncoder:
    """Load an ESM2 encoder. variant is 'base' or 'cpt'."""
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    base = AutoModel.from_pretrained(base_id)
    if variant == "cpt":
        from peft import PeftModel
        if cpt_adapter_id is None:
            raise ValueError("cpt variant requires cpt_adapter_id")
        # PEFT expects a base AutoModelForMaskedLM normally; AutoModel works for the encoder forward
        # path because the adapter targets attention/MLP layers. Verified per CPT_HF_CHECKPOINTS.md.
        # However, the adapter was trained against AutoModelForMaskedLM — the LoRA target modules
        # exist in both. To be safe, load the MaskedLM head and access .esm for the encoder.
        from transformers import AutoModelForMaskedLM
        base_mlm = AutoModelForMaskedLM.from_pretrained(base_id)
        base_mlm = PeftModel.from_pretrained(base_mlm, cpt_adapter_id)
        # The encoder is base_mlm.base_model.model.esm (PEFT-wrapped) for ESM2.
        # We wrap and forward through it, returning last_hidden_state.
        encoder = base_mlm
    else:
        encoder = base
    hidden_dim = encoder.config.hidden_size if hasattr(encoder, "config") else base.config.hidden_size
    return LoadedEncoder(encoder=encoder, tokenizer=tokenizer, hidden_dim=hidden_dim, is_esmc=False)


def load_esmc(variant: str) -> LoadedEncoder:
    """Load ESM-C 300M. variant is 'base' or 'cpt'.

    Mirrors the canonical pattern in `scripts/training/esmc_native_trainer.py`:
    pass `device=torch.device("cuda")` to ESMC.from_pretrained so the model
    loads in its native bf16 on CUDA. The forward path then uses autocast
    bf16 in `encoder_forward`.
    """
    from esm.models.esmc import ESMC
    from esm.tokenization import EsmSequenceTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = ESMC.from_pretrained(ESMC_BASE_ID, device=device)

    # ESM-C has no .config; PEFT requires one. Apply the documented shim
    # (matches `_HFConfigShim` in scripts/training/esmc_native_trainer.py).
    class _ConfigShim(dict):
        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)

    if not hasattr(base, "config"):
        base.config = _ConfigShim(use_return_dict=True, tie_word_embeddings=False)

    if variant == "cpt":
        from peft import PeftModel
        encoder = PeftModel.from_pretrained(base, ESMC_CPT_ADAPTER)
    else:
        encoder = base

    tokenizer = EsmSequenceTokenizer()
    hidden_dim = 960  # ESM-C 300M
    return LoadedEncoder(encoder=encoder, tokenizer=tokenizer, hidden_dim=hidden_dim, is_esmc=True)


def load_encoder(name: str) -> LoadedEncoder:
    """Dispatch on a flat name like 'esm2_8m-base', 'esmc_300m-cpt'."""
    parts = name.split("-")
    if len(parts) != 2:
        raise ValueError(f"name must be '<encoder>-<variant>', got {name}")
    enc, var = parts
    if var not in {"base", "cpt"}:
        raise ValueError(f"variant must be base|cpt, got {var}")

    if enc == "esm2_8m":
        return load_esm2(var, ESM2_8M_BASE, ESM2_8M_CPT_ADAPTER if var == "cpt" else None)
    if enc == "esm2_35m":
        return load_esm2(var, ESM2_35M_BASE, ESM2_35M_CPT_ADAPTER if var == "cpt" else None)
    if enc == "esmc_300m":
        return load_esmc(var)
    raise ValueError(f"unknown encoder: {enc}")


def _is_peft_wrapped(model: nn.Module) -> bool:
    """Detect a PEFT-wrapped model (PeftModel) vs a plain HF / EvolutionaryScale model."""
    return hasattr(model, "peft_config")


def encoder_forward(loaded: LoadedEncoder, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Run the encoder forward and return the per-token hidden states (B, L, H)."""
    if loaded.is_esmc:
        # Canonical ESM-C call pattern (matches scripts/training/esmc_native_trainer.py):
        # forward under autocast bf16, with sequence_tokens + sequence_id as bool mask.
        if _is_peft_wrapped(loaded.encoder):
            inner = loaded.encoder.base_model.model
        else:
            inner = loaded.encoder
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=input_ids.is_cuda):
            out = inner(
                sequence_tokens=input_ids, sequence_id=attention_mask.bool()
            )
        if hasattr(out, "embeddings"):
            return out.embeddings
        return out.hidden_states
    # ESM2 path
    if _is_peft_wrapped(loaded.encoder):
        # PEFT-wrapped AutoModelForMaskedLM. Underlying model has an .esm encoder.
        inner = loaded.encoder.base_model.model  # AutoModelForMaskedLM
        if hasattr(inner, "esm"):
            out = inner.esm(input_ids=input_ids, attention_mask=attention_mask)
            return out.last_hidden_state
        out = inner(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return out.hidden_states[-1]
    # Plain AutoModel (EsmModel) → BaseModelOutput
    out = loaded.encoder(input_ids=input_ids, attention_mask=attention_mask)
    return out.last_hidden_state
