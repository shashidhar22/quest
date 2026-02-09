"""
Accelerator backend abstraction for hardware-agnostic training.

This module provides an abstract base class for hardware backends,
enabling training scripts to run on both NVIDIA GPUs (CUDA) and
AWS Trainium (XLA) without code changes.
"""
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader


class AcceleratorBackend(ABC):
    """
    Abstract base class for hardware accelerator backends.

    Implementations:
    - CUDABackend: NVIDIA GPUs with NCCL distributed backend
    - NeuronBackend: AWS Trainium/Inferentia with XLA

    All hardware-specific operations are abstracted through this interface,
    allowing training code to be written once and run on multiple backends.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name (e.g., 'cuda', 'xla')."""
        pass

    @property
    @abstractmethod
    def distributed_backend(self) -> str:
        """Distributed training backend name (e.g., 'nccl', 'xla')."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the current system."""
        pass

    @abstractmethod
    def init_distributed(
        self,
        local_rank: int,
        world_size: int,
        timeout_minutes: int = 30
    ) -> None:
        """
        Initialize distributed training environment.

        Args:
            local_rank: Local process rank on this node
            world_size: Total number of processes
            timeout_minutes: Timeout for initialization
        """
        pass

    @abstractmethod
    def get_device(self, local_rank: int = 0) -> torch.device:
        """
        Get the device for this accelerator.

        Args:
            local_rank: Local rank for multi-device setups

        Returns:
            torch.device for this backend
        """
        pass

    @abstractmethod
    def get_attention_implementation(self) -> Optional[str]:
        """
        Get attention implementation for transformers models.

        Returns:
            'flash_attention_2' for CUDA, None/'eager' for XLA
        """
        pass

    @abstractmethod
    def get_model_dtype(self) -> torch.dtype:
        """
        Get the preferred dtype for model weights.

        Returns:
            torch.dtype (typically torch.bfloat16)
        """
        pass

    @abstractmethod
    def wrap_model_distributed(
        self,
        model: nn.Module,
        local_rank: int,
        find_unused_parameters: bool = True
    ) -> nn.Module:
        """
        Wrap model for distributed training.

        Args:
            model: Model to wrap
            local_rank: Local process rank
            find_unused_parameters: Whether to detect unused parameters (DDP)

        Returns:
            Wrapped model (DDP for CUDA, unchanged for XLA)
        """
        pass

    @abstractmethod
    def wrap_dataloader(
        self,
        dataloader: DataLoader,
        device: torch.device
    ) -> DataLoader:
        """
        Wrap dataloader if needed for this backend.

        Args:
            dataloader: PyTorch DataLoader
            device: Target device

        Returns:
            Wrapped dataloader (MpDeviceLoader for XLA, unchanged for CUDA)
        """
        pass

    @abstractmethod
    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        model: Optional[nn.Module] = None
    ) -> None:
        """
        Perform optimizer step with backend-specific handling.

        Args:
            optimizer: PyTorch optimizer
            model: Model (needed for some backends)
        """
        pass

    @abstractmethod
    @contextmanager
    def autocast_context(
        self,
        dtype: torch.dtype = torch.bfloat16
    ) -> Generator[None, None, None]:
        """
        Get autocast context manager for mixed precision.

        Args:
            dtype: Data type for autocast

        Yields:
            Context manager for mixed precision training
        """
        pass

    @abstractmethod
    def synchronize(self) -> None:
        """
        Synchronize across devices.

        CUDA: torch.cuda.synchronize()
        XLA: xm.mark_step()
        """
        pass

    @abstractmethod
    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: str = "sum"
    ) -> torch.Tensor:
        """
        All-reduce operation across distributed workers.

        Args:
            tensor: Tensor to reduce
            op: Reduction operation ('sum', 'avg', 'max', 'min')

        Returns:
            Reduced tensor
        """
        pass

    @abstractmethod
    def all_gather(
        self,
        tensor: torch.Tensor,
        world_size: int
    ) -> torch.Tensor:
        """
        All-gather operation across distributed workers.

        Args:
            tensor: Tensor to gather
            world_size: Number of processes

        Returns:
            Gathered tensor from all processes
        """
        pass

    @abstractmethod
    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        path: str,
        is_main_process: bool
    ) -> None:
        """
        Save checkpoint with backend-specific serialization.

        Args:
            state_dict: Checkpoint state dictionary
            path: Save path
            is_main_process: Whether this is the main process
        """
        pass

    @abstractmethod
    def load_checkpoint(
        self,
        path: str,
        map_location: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint with backend-specific deserialization.

        Args:
            path: Checkpoint path
            map_location: Device to map tensors to

        Returns:
            Loaded state dictionary
        """
        pass

    def set_seed(self, seed: int) -> None:
        """
        Set random seed for reproducibility.

        Args:
            seed: Random seed
        """
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def get_world_size(self) -> int:
        """Get the number of distributed processes."""
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_world_size()
        return 1

    def get_rank(self) -> int:
        """Get the global rank of this process."""
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank()
        return 0

    def barrier(self) -> None:
        """Synchronization barrier across all processes."""
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()
