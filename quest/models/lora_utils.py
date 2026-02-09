"""
LoRA utilities for encoder models.

Functions for applying LoRA (Low-Rank Adaptation) to ESM2 encoders
and loading pre-trained encoder checkpoints. Used across multiple
trainer files (seq2seq, peptide-MHC, contrastive, etc.).
"""

import re

import torch
import torch.nn as nn

# Optional: LoRA support
try:
    from peft import LoraConfig, TaskType, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def apply_lora_to_encoder(model: nn.Module, config: dict) -> nn.Module:
    """
    Apply LoRA to the ESM2 encoder.

    Args:
        model: A model with an `encoder` attribute (e.g., TCRSeq2SeqModel,
               PeptideMHCSeq2SeqModel).
        config: Dict with optional keys: lora_r, lora_alpha, lora_dropout.

    Returns:
        The model with LoRA applied to its encoder.
    """
    if not PEFT_AVAILABLE:
        raise ImportError("peft is required for LoRA. Install with: pip install peft")

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


def load_encoder_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
    verbose: bool = True,
) -> nn.Module:
    """
    Load pre-trained encoder weights from a checkpoint file.

    Supports checkpoints from:
    - MLM pre-training (ESM2 + LoRA fine-tuned)
    - Previous seq2seq training runs

    The function handles different checkpoint formats:
    - Full checkpoint with 'model_state_dict' key
    - Direct state dict

    Args:
        model: A model with an `encoder` attribute (e.g., TCRSeq2SeqModel,
               PeptideMHCSeq2SeqModel).
        checkpoint_path: Path to checkpoint .pt file
        strict: If True, raise error on missing/unexpected keys
        verbose: If True, print loading details

    Returns:
        Model with loaded encoder weights
    """
    if verbose:
        print(f"Loading encoder checkpoint from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract state dict
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            if verbose and "config" in checkpoint:
                ckpt_config = checkpoint["config"]
                print(f"  Checkpoint config: model={ckpt_config.get('model_name', 'unknown')}, "
                      f"lora_r={ckpt_config.get('lora_r', 'N/A')}")
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            # Assume it's a direct state dict
            state_dict = checkpoint
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")

    # Get current encoder state dict for reference
    encoder_state_dict = model.encoder.state_dict()
    encoder_keys = set(encoder_state_dict.keys())

    # Try to match checkpoint keys to encoder keys
    # The checkpoint might have different prefixes depending on how it was saved
    matched_keys = []
    unmatched_ckpt_keys = []

    # Build a mapping from checkpoint keys to encoder keys
    key_mapping = {}

    for ckpt_key in state_dict.keys():
        # Direct match
        if ckpt_key in encoder_keys:
            key_mapping[ckpt_key] = ckpt_key
            matched_keys.append(ckpt_key)
            continue

        # Try removing common prefixes
        # e.g., "base_model.model.encoder..." -> "base_model.model..."
        # or "encoder.base_model.model..." -> "base_model.model..."
        modified_key = ckpt_key

        # Remove leading "encoder." if present (from seq2seq checkpoint)
        if modified_key.startswith("encoder."):
            modified_key = modified_key[len("encoder."):]
            if modified_key in encoder_keys:
                key_mapping[ckpt_key] = modified_key
                matched_keys.append(ckpt_key)
                continue

        # Handle EsmForMaskedLM vs EsmModel prefix difference:
        # Foundation model (EsmForMaskedLM): base_model.model.esm.encoder.layer...
        # SFT encoder (EsmModel):            base_model.model.encoder.layer...
        if "base_model.model.esm." in modified_key:
            modified_key = modified_key.replace("base_model.model.esm.", "base_model.model.")
            if modified_key in encoder_keys:
                key_mapping[ckpt_key] = modified_key
                matched_keys.append(ckpt_key)
                continue

        # Also handle without base_model prefix:
        # Foundation: esm.encoder.layer... -> encoder.layer...
        if modified_key.startswith("esm."):
            modified_key = modified_key[len("esm."):]
            if modified_key in encoder_keys:
                key_mapping[ckpt_key] = modified_key
                matched_keys.append(ckpt_key)
                continue

        # The checkpoint key is already in the right format
        # (standalone PEFT model saves as base_model.model.*)
        if modified_key in encoder_keys:
            key_mapping[ckpt_key] = modified_key
            matched_keys.append(ckpt_key)
            continue

        unmatched_ckpt_keys.append(ckpt_key)

    # Load matched weights
    new_state_dict = {}
    for ckpt_key, encoder_key in key_mapping.items():
        new_state_dict[encoder_key] = state_dict[ckpt_key]

    # Check for missing encoder keys
    loaded_keys = set(new_state_dict.keys())
    missing_keys = encoder_keys - loaded_keys

    if verbose:
        print(f"  Matched {len(matched_keys)} / {len(state_dict)} checkpoint keys")
        print(f"  Loading {len(new_state_dict)} weights into encoder")
        if missing_keys:
            # Filter out expected missing keys (e.g., new LoRA adapters)
            important_missing = [k for k in missing_keys if "lora" not in k.lower()]
            if important_missing:
                print(f"  Missing (non-LoRA) keys: {len(important_missing)}")
                if len(important_missing) <= 5:
                    for k in important_missing:
                        print(f"    - {k}")
        if unmatched_ckpt_keys:
            print(f"  Unmatched checkpoint keys: {len(unmatched_ckpt_keys)}")

    # Load the state dict
    load_result = model.encoder.load_state_dict(new_state_dict, strict=strict)

    if verbose:
        if load_result.missing_keys:
            lora_missing = [k for k in load_result.missing_keys if "lora" in k.lower()]
            other_missing = [k for k in load_result.missing_keys if "lora" not in k.lower()]
            if lora_missing:
                print(f"  LoRA layers initialized fresh: {len(lora_missing)}")
            if other_missing:
                print(f"  Other missing keys: {len(other_missing)}")
        if load_result.unexpected_keys:
            print(f"  Unexpected keys (ignored): {len(load_result.unexpected_keys)}")
        print("  Encoder checkpoint loaded successfully!")

    return model
