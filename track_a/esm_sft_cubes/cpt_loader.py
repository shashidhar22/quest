"""Load an ESM-C 300M encoder with a *local* CPT LoRA adapter.

Mirrors `track_a/esm_sft/encoders.py:load_esmc` exactly, except the LoRA
adapter is loaded from a local filesystem path (the 5 Stage-4 checkpoints
under /home/ubuntu/quest/checkpoints/cpt_stage4_*/final/) instead of the
hardcoded HF repo id.

Returns the same `LoadedEncoder` dataclass that `track_a/esm_sft/model.ESMScorer`
already consumes, so the downstream training code (model + forward + scorer)
needs no modification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

from track_a.esm_sft.encoders import LoadedEncoder

ESMC_BASE_ID = "esmc_300m"


def load_esmc_from_local_lora(lora_path) -> LoadedEncoder:
    """Load ESM-C 300M + apply a CPT LoRA adapter from a local directory.

    Args:
        lora_path: directory containing `adapter_config.json` +
                   `adapter_model.safetensors` (the `final/` subdir of a
                   Stage-4 checkpoint). If None/'baseline'/'null', applies
                   a fresh randomly-initialized LoRA r=32 (same shape as CPT)
                   for the no-CPT baseline comparison.

    Returns:
        LoadedEncoder with `is_esmc=True`, `hidden_dim=960`.
    """
    from esm.models.esmc import ESMC
    from esm.tokenization import EsmSequenceTokenizer
    from peft import PeftModel, LoraConfig, get_peft_model, TaskType

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = ESMC.from_pretrained(ESMC_BASE_ID, device=device)

    class _ConfigShim(dict):
        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)

    if not hasattr(base, "config"):
        base.config = _ConfigShim(use_return_dict=True, tie_word_embeddings=False)

    if lora_path in (None, "None", "baseline", "null", ""):
        # Baseline mode: fresh randomly-init LoRA with same shape as CPT
        lora_cfg = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
            target_modules=["layernorm_qkv.1", "out_proj", "ffn.1", "ffn.3"],
        )
        encoder = get_peft_model(base, lora_cfg)
    else:
        lora_path = Path(lora_path)
        if not (lora_path / "adapter_config.json").exists():
            raise FileNotFoundError(f"adapter_config.json not found in {lora_path}")
        encoder = PeftModel.from_pretrained(base, str(lora_path))

    tokenizer = EsmSequenceTokenizer()
    return LoadedEncoder(encoder=encoder, tokenizer=tokenizer, hidden_dim=960, is_esmc=True)
