"""
Tensor parallelism utilities for ESM2 models on NeuronX Distributed.

This module provides functions to replace standard PyTorch layers in ESM2
with NeuronX Distributed tensor-parallel equivalents (ColumnParallelLinear,
RowParallelLinear, ParallelEmbedding) for training across multiple Trainium
chips.

Usage:
    from quest.models.tensor_parallel import apply_tensor_parallelism, get_tp_loss_fn

    model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")
    apply_tensor_parallelism(model, tp_degree=2)
    loss_fn = get_tp_loss_fn(tp_degree=2)
"""

import logging
import math
from typing import Callable, Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


def pad_model(model: nn.Module, tp_degree: int) -> int:
    """
    Pad ESM2 attention heads so num_heads is divisible by tp_degree.

    ESM2 t33_650M_UR50D has 20 attention heads. For TP degrees that don't
    divide 20 evenly (e.g., TP=8), we pad to the next multiple (24 heads).

    This pads the QKV projection weight/bias in-place by adding zero-initialized
    rows for the extra heads, and adjusts the attention output projection
    to accept the padded hidden dimension.

    Args:
        model: ESM2 model (modifies in-place)
        tp_degree: Tensor parallelism degree

    Returns:
        New number of attention heads after padding
    """
    config = model.config
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads

    if num_heads % tp_degree == 0:
        return num_heads

    new_num_heads = math.ceil(num_heads / tp_degree) * tp_degree
    new_attn_dim = new_num_heads * head_dim
    pad_size = new_attn_dim - (num_heads * head_dim)

    logger.warning(
        f"Padding attention heads from {num_heads} to {new_num_heads} "
        f"for TP degree {tp_degree} (adding {new_num_heads - num_heads} heads)"
    )

    for layer in model.esm.encoder.layer:
        attn = layer.attention.self

        # Pad Q, K, V projections: add zero rows for extra heads
        for proj_name in ("query", "key", "value"):
            proj = getattr(attn, proj_name)
            old_weight = proj.weight.data  # [out_features, in_features]
            new_weight = torch.zeros(
                old_weight.shape[0] + pad_size,
                old_weight.shape[1],
                dtype=old_weight.dtype,
                device=old_weight.device,
            )
            new_weight[:old_weight.shape[0]] = old_weight

            new_proj = nn.Linear(
                old_weight.shape[1],
                old_weight.shape[0] + pad_size,
                bias=proj.bias is not None,
            )
            new_proj.weight = nn.Parameter(new_weight)
            if proj.bias is not None:
                new_bias = torch.zeros(
                    old_weight.shape[0] + pad_size,
                    dtype=proj.bias.data.dtype,
                    device=proj.bias.data.device,
                )
                new_bias[:proj.bias.data.shape[0]] = proj.bias.data
                new_proj.bias = nn.Parameter(new_bias)

            setattr(attn, proj_name, new_proj)

        # Update num_attention_heads attribute on the attention module
        attn.num_attention_heads = new_num_heads
        attn.all_head_size = new_attn_dim

        # Pad attention output dense: add zero columns for extra input dims
        out_dense = layer.attention.output.dense
        old_weight = out_dense.weight.data  # [hidden_size, attn_dim]
        new_weight = torch.zeros(
            old_weight.shape[0],
            old_weight.shape[1] + pad_size,
            dtype=old_weight.dtype,
            device=old_weight.device,
        )
        new_weight[:, :old_weight.shape[1]] = old_weight

        new_out = nn.Linear(
            old_weight.shape[1] + pad_size,
            old_weight.shape[0],
            bias=out_dense.bias is not None,
        )
        new_out.weight = nn.Parameter(new_weight)
        if out_dense.bias is not None:
            new_out.bias = nn.Parameter(out_dense.bias.data.clone())
        layer.attention.output.dense = new_out

    config.num_attention_heads = new_num_heads
    return new_num_heads


def apply_tensor_parallelism(model: nn.Module, tp_degree: int) -> None:
    """
    Replace ESM2 linear layers with NeuronX Distributed tensor-parallel layers.

    This walks the ESM2 module tree and replaces:
    - Word embeddings -> ParallelEmbedding
    - QKV projections -> ColumnParallelLinear(gather_output=False)
    - Attention output -> RowParallelLinear(input_is_parallel=True)
    - FFN intermediate -> ColumnParallelLinear(gather_output=False)
    - FFN output -> RowParallelLinear(input_is_parallel=True)
    - LM head decoder -> ColumnParallelLinear (sharded vocab output)

    After replacement, num_attention_heads per layer is divided by tp_degree.

    Args:
        model: ESM2 model (modified in-place)
        tp_degree: Tensor parallelism degree (must be >= 1)

    Raises:
        ImportError: If neuronx_distributed is not available
        ValueError: If tp_degree < 1
    """
    if tp_degree <= 1:
        return

    try:
        from neuronx_distributed.parallel_layers import (
            ColumnParallelLinear,
            RowParallelLinear,
            ParallelEmbedding,
        )
    except ImportError:
        raise ImportError(
            "neuronx_distributed is required for tensor parallelism. "
            "Install with: pip install neuronx_distributed"
        )

    # Pad heads if needed
    num_heads = pad_model(model, tp_degree)
    config = model.config
    head_dim = config.hidden_size // num_heads
    heads_per_partition = num_heads // tp_degree

    logger.info(
        f"Applying tensor parallelism (TP={tp_degree}): "
        f"{num_heads} heads -> {heads_per_partition} heads/partition"
    )

    # Replace word embeddings
    old_emb = model.esm.embeddings.word_embeddings
    model.esm.embeddings.word_embeddings = ParallelEmbedding(
        old_emb.num_embeddings,
        old_emb.embedding_dim,
        padding_idx=old_emb.padding_idx,
    )
    logger.info("Replaced word_embeddings -> ParallelEmbedding")

    # Replace layers in each transformer block
    for i, layer in enumerate(model.esm.encoder.layer):
        attn = layer.attention.self
        attn_out = layer.attention.output

        # QKV: ColumnParallelLinear (shard output dim across TP ranks)
        for proj_name in ("query", "key", "value"):
            old_proj = getattr(attn, proj_name)
            new_proj = ColumnParallelLinear(
                old_proj.in_features,
                old_proj.out_features,
                bias=old_proj.bias is not None,
                gather_output=False,
            )
            setattr(attn, proj_name, new_proj)

        # Attention output: RowParallelLinear (input is already partitioned)
        old_out = attn_out.dense
        attn_out.dense = RowParallelLinear(
            old_out.in_features,
            old_out.out_features,
            bias=old_out.bias is not None,
            input_is_parallel=True,
        )

        # Update per-layer head count
        attn.num_attention_heads = heads_per_partition
        attn.all_head_size = heads_per_partition * head_dim

        # FFN intermediate: ColumnParallelLinear
        old_intermediate = layer.intermediate.dense
        layer.intermediate.dense = ColumnParallelLinear(
            old_intermediate.in_features,
            old_intermediate.out_features,
            bias=old_intermediate.bias is not None,
            gather_output=False,
        )

        # FFN output: RowParallelLinear
        old_output = layer.output.dense
        layer.output.dense = RowParallelLinear(
            old_output.in_features,
            old_output.out_features,
            bias=old_output.bias is not None,
            input_is_parallel=True,
        )

    # Replace LM head decoder
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "decoder"):
        old_decoder = model.lm_head.decoder
        model.lm_head.decoder = ColumnParallelLinear(
            old_decoder.in_features,
            old_decoder.out_features,
            bias=old_decoder.bias is not None,
            gather_output=False,
        )
        logger.info("Replaced lm_head.decoder -> ColumnParallelLinear")

    logger.info(f"Tensor parallelism applied to {len(model.esm.encoder.layer)} layers")


def get_tp_loss_fn(tp_degree: int) -> Optional[Callable]:
    """
    Return a loss function appropriate for the tensor parallelism degree.

    When TP > 1, returns NeuronX Distributed's parallel_cross_entropy which
    operates on sharded logits without gathering across TP ranks (saves memory
    and communication).

    When TP == 1, returns None (use standard CrossEntropyLoss).

    Args:
        tp_degree: Tensor parallelism degree

    Returns:
        parallel_cross_entropy callable if TP > 1, else None
    """
    if tp_degree <= 1:
        return None

    try:
        from neuronx_distributed.parallel_layers.loss_functions import (
            parallel_cross_entropy,
        )
        return parallel_cross_entropy
    except ImportError:
        raise ImportError(
            "neuronx_distributed is required for parallel cross entropy. "
            "Install with: pip install neuronx_distributed"
        )
