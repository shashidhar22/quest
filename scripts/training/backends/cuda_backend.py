"""
CUDA backend for NVIDIA GPU training.

This backend provides optimized training on NVIDIA GPUs using:
- NCCL for distributed communication
- Flash Attention 2 for efficient attention computation
- torch.amp for mixed precision training
- DistributedDataParallel for multi-GPU training
"""
import os
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, Dict, Generator, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .base import AcceleratorBackend


class CUDABackend(AcceleratorBackend):
    """
    Backend for NVIDIA GPU training with NCCL distributed communication.

    Features:
    - Flash Attention 2 support for efficient attention
    - NCCL backend for fast GPU-to-GPU communication
    - Mixed precision training with torch.amp
    - DistributedDataParallel wrapper for multi-GPU
    """

    @property
    def name(self) -> str:
        return "cuda"

    @property
    def distributed_backend(self) -> str:
        return "nccl"

    def is_available(self) -> bool:
        """Check if CUDA is available."""
        return torch.cuda.is_available()

    def init_distributed(
        self,
        local_rank: int,
        world_size: int,
        timeout_minutes: int = 30
    ) -> None:
        """
        Initialize NCCL distributed training.

        Args:
            local_rank: Local GPU rank
            world_size: Total number of GPUs
            timeout_minutes: Timeout for NCCL initialization
        """
        torch.cuda.set_device(local_rank)

        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=timedelta(minutes=timeout_minutes),
            )

        # Performance optimizations for CUDA
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def get_device(self, local_rank: int = 0) -> torch.device:
        """Get CUDA device for the given local rank."""
        if torch.cuda.is_available():
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")

    def get_attention_implementation(self) -> Optional[str]:
        """Return Flash Attention 2 for CUDA if available, else eager."""
        try:
            import flash_attn
            return "flash_attention_2"
        except ImportError:
            return "eager"

    def get_model_dtype(self) -> torch.dtype:
        """Return bfloat16 for modern GPUs."""
        return torch.bfloat16

    def wrap_model_distributed(
        self,
        model: nn.Module,
        local_rank: int,
        find_unused_parameters: bool = True
    ) -> nn.Module:
        """
        Wrap model with DistributedDataParallel.

        Args:
            model: Model to wrap
            local_rank: Local GPU rank
            find_unused_parameters: Whether to find unused parameters

        Returns:
            DDP-wrapped model
        """
        return DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=find_unused_parameters,
        )

    def wrap_dataloader(
        self,
        dataloader: DataLoader,
        device: torch.device
    ) -> DataLoader:
        """
        No wrapping needed for CUDA DataLoader.

        Args:
            dataloader: PyTorch DataLoader
            device: Target device (unused)

        Returns:
            Original dataloader unchanged
        """
        return dataloader

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        model: Optional[nn.Module] = None
    ) -> None:
        """
        Standard PyTorch optimizer step.

        Args:
            optimizer: Optimizer to step
            model: Model (unused for CUDA)
        """
        optimizer.step()

    @contextmanager
    def autocast_context(
        self,
        dtype: torch.dtype = torch.bfloat16
    ) -> Generator[None, None, None]:
        """
        CUDA autocast context manager for mixed precision.

        Args:
            dtype: Data type for autocast (default: bfloat16)

        Yields:
            Autocast context
        """
        with autocast(device_type="cuda", dtype=dtype):
            yield

    def synchronize(self) -> None:
        """Synchronize CUDA streams."""
        torch.cuda.synchronize()

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: str = "sum"
    ) -> torch.Tensor:
        """
        NCCL all-reduce operation.

        Args:
            tensor: Tensor to reduce
            op: Reduction operation ('sum', 'avg', 'max', 'min')

        Returns:
            Reduced tensor
        """
        if not dist.is_initialized():
            return tensor

        ops_map = {
            "sum": dist.ReduceOp.SUM,
            "avg": dist.ReduceOp.AVG,
            "max": dist.ReduceOp.MAX,
            "min": dist.ReduceOp.MIN,
        }
        reduce_op = ops_map.get(op, dist.ReduceOp.SUM)
        dist.all_reduce(tensor, op=reduce_op)
        return tensor

    def all_gather(
        self,
        tensor: torch.Tensor,
        world_size: int
    ) -> torch.Tensor:
        """
        NCCL all-gather operation.

        Args:
            tensor: Tensor to gather (must be same size on all ranks)
            world_size: Number of processes

        Returns:
            Concatenated tensor from all processes
        """
        if not dist.is_initialized() or world_size == 1:
            return tensor

        # Create output tensor list
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        path: str,
        is_main_process: bool
    ) -> None:
        """
        Save checkpoint using torch.save.

        Args:
            state_dict: State dictionary to save
            path: Save path
            is_main_process: Only save on main process
        """
        if is_main_process:
            torch.save(state_dict, path)

    def load_checkpoint(
        self,
        path: str,
        map_location: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint using torch.load.

        Args:
            path: Checkpoint path
            map_location: Device to map tensors to

        Returns:
            Loaded state dictionary
        """
        return torch.load(path, map_location=map_location, weights_only=False)

    def set_seed(self, seed: int) -> None:
        """
        Set random seed including CUDA-specific seeds.

        Args:
            seed: Random seed
        """
        super().set_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def empty_cache(self) -> None:
        """Clear CUDA memory cache."""
        torch.cuda.empty_cache()

    def memory_stats(self) -> Dict[str, float]:
        """
        Get CUDA memory statistics.

        Returns:
            Dict with allocated, reserved, and max memory in GB
        """
        if not torch.cuda.is_available():
            return {}

        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
