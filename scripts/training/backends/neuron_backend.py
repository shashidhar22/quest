"""
Neuron backend for AWS Trainium/Inferentia training.

This backend provides training on AWS Trainium chips using:
- XLA for graph compilation and execution
- torch_xla for PyTorch integration
- Neuron SDK for hardware-specific optimizations

Note: This backend requires torch-xla-neuronx to be installed,
which is only available on Trainium/Inferentia instances.
"""
import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from .base import AcceleratorBackend


# Lazy imports for XLA - not available on non-Trainium instances
_XLA_AVAILABLE = False
_xm = None
_pl = None

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_backend  # Registers 'xla' backend
    _XLA_AVAILABLE = True
    _xm = xm
    _pl = pl
except ImportError:
    pass


class NeuronBackend(AcceleratorBackend):
    """
    Backend for AWS Trainium training with XLA.

    Features:
    - XLA graph compilation for optimized execution
    - Eager attention (Flash Attention not supported)
    - BF16 via environment variable (not torch.amp)
    - MpDeviceLoader for efficient data loading
    - xm.optimizer_step() for proper gradient synchronization

    Note: Requires torch-xla-neuronx package, which is pre-installed
    in the AWS Neuron PyTorch container images.
    """

    def __init__(self):
        """Initialize Neuron backend, verifying XLA availability."""
        if not _XLA_AVAILABLE:
            raise ImportError(
                "torch_xla is required for Neuron backend. "
                "This package is only available on AWS Trainium/Inferentia instances. "
                "Use the Neuron PyTorch container or install with: pip install torch-xla-neuronx"
            )

    @property
    def name(self) -> str:
        return "xla"

    @property
    def distributed_backend(self) -> str:
        return "xla"

    def is_available(self) -> bool:
        """Check if XLA/Neuron is available."""
        return _XLA_AVAILABLE

    def init_distributed(
        self,
        local_rank: int,
        world_size: int,
        timeout_minutes: int = 30
    ) -> None:
        """
        Initialize XLA distributed training.

        Args:
            local_rank: Local process rank (used differently in XLA)
            world_size: Total number of processes
            timeout_minutes: Timeout (unused in XLA)
        """
        # Set Neuron-specific environment variables
        os.environ.setdefault("XLA_USE_BF16", "1")

        # Enhanced compiler flags for transformer training
        os.environ.setdefault(
            "NEURON_CC_FLAGS",
            "--model-type transformer --distribution-strategy llm-training"
        )

        # XLA Compilation Cache (critical for fast restarts)
        os.environ.setdefault("NEURON_COMPILE_CACHE_URL", "/var/tmp/neuron-compile-cache")

        # Performance optimizations
        os.environ.setdefault("NEURON_FUSE_SOFTMAX", "1")  # Fuse softmax into attention
        os.environ.setdefault("NEURON_RT_STOCHASTIC_ROUNDING_EN", "1")  # Better BF16 training

        # Memory management for trn1.2xlarge (32GB HBM)
        os.environ.setdefault("NEURON_NUM_RECENT_MODELS_TO_KEEP", "2")

        # XLA distributed initialization
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(backend="xla")

    def get_device(self, local_rank: int = 0) -> torch.device:
        """
        Get XLA device.

        Args:
            local_rank: Unused - XLA handles device assignment

        Returns:
            XLA device
        """
        return _xm.xla_device()

    def get_attention_implementation(self) -> Optional[str]:
        """
        Return eager attention for XLA.

        Flash Attention is not supported on Trainium, so we use
        the standard eager attention implementation.
        """
        return "eager"

    def get_model_dtype(self) -> torch.dtype:
        """
        Return float32 for Trainium model loading.

        Note: Trainium handles dtype conversion via XLA_USE_BF16 environment
        variable. Loading models in bfloat16 explicitly can cause NaN issues
        when combined with XLA's automatic casting.
        """
        return torch.float32

    def wrap_model_distributed(
        self,
        model: nn.Module,
        local_rank: int,
        find_unused_parameters: bool = True
    ) -> nn.Module:
        """
        No explicit wrapping needed for XLA.

        XLA handles data parallelism through its execution model.
        When using xmp.spawn or torchrun, the model is automatically
        replicated across devices.

        Args:
            model: Model (returned unchanged)
            local_rank: Unused
            find_unused_parameters: Unused

        Returns:
            Original model unchanged
        """
        return model

    def wrap_dataloader(
        self,
        dataloader: DataLoader,
        device: torch.device
    ) -> DataLoader:
        """
        Wrap DataLoader with MpDeviceLoader for XLA.

        MpDeviceLoader handles efficient data transfer to XLA devices
        and prefetching for better performance.

        Args:
            dataloader: PyTorch DataLoader
            device: XLA device

        Returns:
            MpDeviceLoader wrapping the original dataloader
        """
        return _pl.MpDeviceLoader(dataloader, device)

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        model: Optional[nn.Module] = None
    ) -> None:
        """
        XLA optimizer step with mark_step.

        Uses xm.optimizer_step() which:
        1. Calls optimizer.step()
        2. Calls xm.mark_step() to compile and execute the graph
        3. Properly handles gradient synchronization across devices

        Args:
            optimizer: Optimizer to step
            model: Unused
        """
        _xm.optimizer_step(optimizer)

    @contextmanager
    def autocast_context(
        self,
        dtype: torch.dtype = torch.bfloat16
    ) -> Generator[None, None, None]:
        """
        XLA autocast context (effectively a no-op).

        XLA handles mixed precision through the XLA_USE_BF16 environment
        variable rather than torch.amp.autocast. Using autocast with XLA
        can cause graph compilation issues.

        Args:
            dtype: Unused

        Yields:
            Null context (no autocast)
        """
        # XLA handles dtype via XLA_USE_BF16 env var
        # Using autocast with XLA can cause issues
        yield

    def synchronize(self) -> None:
        """
        XLA synchronization via mark_step.

        mark_step() triggers graph compilation and execution,
        synchronizing all pending operations.
        """
        _xm.mark_step()

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: str = "sum"
    ) -> torch.Tensor:
        """
        XLA all-reduce operation.

        Args:
            tensor: Tensor to reduce
            op: Reduction operation ('sum', 'avg', 'max', 'min')

        Returns:
            Reduced tensor
        """
        # XLA uses string operation names
        xla_ops = {
            "sum": "sum",
            "avg": "mean",  # XLA uses 'mean' not 'avg'
            "max": "max",
            "min": "min",
        }
        xla_op = xla_ops.get(op, "sum")
        return _xm.all_reduce(xla_op, tensor)

    def all_gather(
        self,
        tensor: torch.Tensor,
        world_size: int
    ) -> torch.Tensor:
        """
        XLA all-gather operation.

        Args:
            tensor: Tensor to gather
            world_size: Number of processes

        Returns:
            Concatenated tensor from all processes
        """
        if world_size == 1:
            return tensor
        return _xm.all_gather(tensor, dim=0)

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        path: str,
        is_main_process: bool
    ) -> None:
        """
        Save checkpoint using xm.save.

        xm.save() handles XLA tensor serialization properly and
        ensures only the master process saves when master_only=True.

        Args:
            state_dict: State dictionary to save
            path: Save path
            is_main_process: Unused - xm.save handles this
        """
        _xm.save(state_dict, path, master_only=True)

    def load_checkpoint(
        self,
        path: str,
        map_location: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint for XLA.

        Args:
            path: Checkpoint path
            map_location: Unused - tensors moved to XLA device after loading

        Returns:
            Loaded state dictionary
        """
        # Load to CPU first, then move to XLA device
        return torch.load(path, map_location="cpu", weights_only=False)

    def set_seed(self, seed: int) -> None:
        """
        Set random seed including XLA-specific seeds.

        Args:
            seed: Random seed
        """
        super().set_seed(seed)
        # XLA handles seeding through torch.manual_seed
        # which propagates to XLA devices

    def rendezvous(self, tag: str) -> None:
        """
        XLA rendezvous point for synchronization.

        Creates a synchronization barrier with a named tag,
        useful for coordinating operations across processes.

        Args:
            tag: Rendezvous tag name
        """
        _xm.rendezvous(tag)

    def mark_step(self) -> None:
        """
        Explicit mark_step for graph execution.

        Should be called periodically during training to trigger
        graph compilation and execution. Usually called automatically
        by optimizer_step(), but can be called manually if needed.
        """
        _xm.mark_step()

    def get_ordinal(self) -> int:
        """
        Get XLA device ordinal (equivalent to local rank).

        Returns:
            Device ordinal for this process
        """
        return _xm.get_ordinal()

    def master_print(self, message: str) -> None:
        """
        Print only on the master process.

        Args:
            message: Message to print
        """
        _xm.master_print(message)
