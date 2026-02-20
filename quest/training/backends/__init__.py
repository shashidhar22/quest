"""
Accelerator backends for hardware-agnostic training.

This package provides a unified interface for training on different
hardware backends (NVIDIA GPUs and AWS Trainium).

Usage:
    from backends import get_backend, AcceleratorBackend

    # Auto-detect backend based on available hardware
    backend = get_backend("auto")

    # Or explicitly specify
    backend = get_backend("cuda")  # NVIDIA GPUs
    backend = get_backend("xla")   # AWS Trainium

Example:
    backend = get_backend("auto")
    device = backend.get_device(local_rank=0)
    model = model.to(device)

    if is_distributed:
        backend.init_distributed(local_rank, world_size)
        model = backend.wrap_model_distributed(model, local_rank)

    for batch in backend.wrap_dataloader(dataloader, device):
        with backend.autocast_context():
            outputs = model(batch)
            loss = compute_loss(outputs)

        loss.backward()
        backend.optimizer_step(optimizer)
"""
from typing import Optional

from .base import AcceleratorBackend


def get_backend(
    backend_type: str = "auto",
    tp_degree: int = 1,
    pp_degree: int = 1,
) -> AcceleratorBackend:
    """
    Factory function to get the appropriate accelerator backend.

    Args:
        backend_type: One of:
            - 'auto': Auto-detect based on available hardware (XLA first, then CUDA)
            - 'cuda': NVIDIA GPU backend with NCCL
            - 'xla', 'neuron', 'trainium': AWS Trainium backend with XLA
        tp_degree: Tensor parallelism degree (only used for Neuron backend).
            1 = no TP (default). Values > 1 require neuronx_distributed.
        pp_degree: Pipeline parallelism degree (only used for Neuron backend).
            1 = no PP (default).

    Returns:
        AcceleratorBackend instance for the specified/detected hardware

    Raises:
        ValueError: If backend_type is not recognized
        ImportError: If required dependencies for the backend are not available

    Example:
        >>> backend = get_backend("auto")
        >>> print(backend.name)
        'cuda'  # or 'xla' if on Trainium
    """
    if backend_type == "auto":
        # Try XLA first (Trainium), then CUDA
        try:
            from .neuron_backend import NeuronBackend
            return NeuronBackend(tp_degree=tp_degree, pp_degree=pp_degree)
        except ImportError:
            pass

        from .cuda_backend import CUDABackend
        cuda_backend = CUDABackend()
        if cuda_backend.is_available():
            return cuda_backend

        # Fallback: return CUDA backend even if no GPU (will use CPU)
        return cuda_backend

    elif backend_type == "cuda":
        from .cuda_backend import CUDABackend
        return CUDABackend()

    elif backend_type in ("xla", "neuron", "trainium"):
        from .neuron_backend import NeuronBackend
        return NeuronBackend(tp_degree=tp_degree, pp_degree=pp_degree)

    else:
        raise ValueError(
            f"Unknown backend type: '{backend_type}'. "
            f"Valid options: 'auto', 'cuda', 'xla', 'neuron', 'trainium'"
        )


def is_xla_available() -> bool:
    """
    Check if XLA (Trainium) backend is available.

    Returns:
        True if torch_xla is importable, False otherwise
    """
    try:
        import torch_xla
        return True
    except ImportError:
        return False


def is_cuda_available() -> bool:
    """
    Check if CUDA backend is available.

    Returns:
        True if CUDA is available, False otherwise
    """
    import torch
    return torch.cuda.is_available()


def detect_backend() -> str:
    """
    Detect the best available backend.

    Returns:
        'xla' if Trainium/XLA is available, otherwise 'cuda'
    """
    if is_xla_available():
        return "xla"
    return "cuda"


# Public API
__all__ = [
    "AcceleratorBackend",
    "get_backend",
    "is_xla_available",
    "is_cuda_available",
    "detect_backend",
]
