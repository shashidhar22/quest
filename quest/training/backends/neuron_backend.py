"""
Neuron backend for AWS Trainium/Inferentia training.

This backend provides training on AWS Trainium chips using:
- XLA for graph compilation and execution
- torch_xla for PyTorch integration
- Neuron SDK for hardware-specific optimizations
- NeuronX Distributed for tensor parallelism (optional)

Note: This backend requires torch-xla-neuronx to be installed,
which is only available on Trainium/Inferentia instances.
"""
import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from .base import AcceleratorBackend

logger = logging.getLogger(__name__)

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
    - Optional tensor parallelism via NeuronX Distributed

    Note: Requires torch-xla-neuronx package, which is pre-installed
    in the AWS Neuron PyTorch container images.
    """

    def __init__(self, tp_degree: int = 1, pp_degree: int = 1):
        """
        Initialize Neuron backend, verifying XLA availability.

        Args:
            tp_degree: Tensor parallelism degree. 1 = no TP (default).
                Compatible with ESM2 head counts: 1, 2, 4, 5, 10, 20
                (TP=8 auto-pads heads from 20 to 24).
            pp_degree: Pipeline parallelism degree. 1 = no PP (default).
        """
        if not _XLA_AVAILABLE:
            raise ImportError(
                "torch_xla is required for Neuron backend. "
                "This package is only available on AWS Trainium/Inferentia instances. "
                "Use the Neuron PyTorch container or install with: pip install torch-xla-neuronx"
            )

        self.tp_degree = tp_degree
        self.pp_degree = pp_degree
        self._nxd_initialized = False

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

        When tp_degree > 1, also initializes NeuronX Distributed model
        parallel groups for tensor parallelism.

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

        # Initialize NeuronX Distributed parallel state for TP
        if self.tp_degree > 1:
            import neuronx_distributed
            from neuronx_distributed.parallel_layers import parallel_state

            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=self.tp_degree,
                pipeline_model_parallel_size=self.pp_degree,
            )
            self._nxd_initialized = True

            dp_size = self.get_data_parallel_world_size()
            logger.info(
                f"NeuronX Distributed initialized: "
                f"TP={self.tp_degree}, PP={self.pp_degree}, DP={dp_size}, "
                f"total ranks={world_size}"
            )

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
        Wrap model for distributed training on XLA.

        When TP > 1, uses NeuronX Distributed's move_model_to_device()
        to properly place tensor-parallel layers on the XLA device.
        Otherwise, returns the model unchanged (XLA handles data
        parallelism through its execution model).

        Args:
            model: Model to wrap
            local_rank: Unused for XLA
            find_unused_parameters: Unused for XLA

        Returns:
            Model (moved to device if TP > 1, unchanged otherwise)
        """
        if self.tp_degree > 1:
            from neuronx_distributed.parallel_layers import move_model_to_device
            move_model_to_device(model, _xm.xla_device())
            logger.info("Model moved to XLA device via NeuronX Distributed move_model_to_device()")
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

        When TP > 1, passes the data parallel group to ensure gradients
        are only synchronized across data-parallel ranks (not TP ranks).

        Args:
            optimizer: Optimizer to step
            model: Unused
        """
        if self.tp_degree > 1:
            from neuronx_distributed.parallel_layers import parallel_state
            _xm.optimizer_step(
                optimizer,
                groups=parallel_state.get_data_parallel_group(as_list=True),
            )
        else:
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
        Save checkpoint using xm.save or NeuronX Distributed sharded save.

        When TP > 1, uses neuronx_distributed sharded checkpointing so each
        TP rank saves its own shard. Otherwise, uses standard xm.save().

        Args:
            state_dict: State dictionary to save
            path: Save path
            is_main_process: Unused - xm.save handles this
        """
        if self.tp_degree > 1:
            from neuronx_distributed.parallel_layers import save
            save(state_dict, path)
            logger.info(f"Saved TP-sharded checkpoint to {path}")
        else:
            _xm.save(state_dict, path, master_only=True)

    def load_checkpoint(
        self,
        path: str,
        map_location: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint for XLA.

        When TP > 1, uses neuronx_distributed sharded loading so each
        TP rank loads its own shard. Otherwise, loads to CPU first.

        Args:
            path: Checkpoint path
            map_location: Unused - tensors moved to XLA device after loading

        Returns:
            Loaded state dictionary
        """
        if self.tp_degree > 1:
            from neuronx_distributed.parallel_layers import load
            return load(path)
        # Load to CPU first, then move to XLA device
        return torch.load(path, map_location="cpu", weights_only=False)

    def clip_grad_norm(
        self,
        model: nn.Module,
        max_norm: float
    ) -> torch.Tensor:
        """
        TP-aware gradient clipping.

        When TP > 1, uses NeuronX Distributed's clip_grad_norm which
        correctly computes the global gradient norm across TP ranks.
        Otherwise, falls back to standard PyTorch clip_grad_norm_.

        Args:
            model: Model whose gradients to clip
            max_norm: Maximum gradient norm

        Returns:
            Total gradient norm before clipping
        """
        if self.tp_degree > 1:
            from neuronx_distributed.parallel_layers import clip_grad_norm
            return clip_grad_norm(model.parameters(), max_norm)
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

    def get_data_parallel_world_size(self) -> int:
        """
        Get the data-parallel world size (excluding TP/PP ranks).

        For a 16-chip trn1.32xlarge with TP=2:
            total_ranks=16, TP=2, PP=1 -> DP=8

        Returns:
            Number of data-parallel replicas
        """
        if self.tp_degree > 1:
            from neuronx_distributed.parallel_layers import parallel_state
            return parallel_state.get_data_parallel_size()
        return self.get_world_size()

    def get_data_parallel_rank(self) -> int:
        """
        Get the data-parallel rank of this process (excluding TP/PP).

        Returns:
            Data-parallel rank (0-indexed)
        """
        if self.tp_degree > 1:
            from neuronx_distributed.parallel_layers import parallel_state
            return parallel_state.get_data_parallel_rank()
        return self.get_rank()

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
