"""
quest.training - Training utilities for the QUEST framework.

Consolidates training components (callbacks, collators, samplers) that were
previously duplicated across multiple trainer scripts.

Will be populated with BaseTCRTrainer and backend re-exports in Phase 3.
"""

from quest.training.callbacks import EarlyStopping, StreamingMetricsTracker
from quest.training.collators import (
    DataCollatorForMLMDynamic,
    DataCollatorForMLMWithPacking,
    DataCollatorForMLMWithVarlen,
    TaskSpecificMaskingCollator,
    ContrastivePairCollator,
)
from quest.training.samplers import (
    LengthBucketSampler,
    DistributedLengthBucketSampler,
    CurriculumSampler,
    DistributedCurriculumSampler,
)
from quest.training.base_trainer import BaseTCRTrainer
from quest.training.backends import AcceleratorBackend, get_backend

__all__ = [
    # Callbacks
    "EarlyStopping",
    "StreamingMetricsTracker",
    # Collators
    "DataCollatorForMLMDynamic",
    "DataCollatorForMLMWithPacking",
    "DataCollatorForMLMWithVarlen",
    "TaskSpecificMaskingCollator",
    "ContrastivePairCollator",
    # Samplers
    "LengthBucketSampler",
    "DistributedLengthBucketSampler",
    "CurriculumSampler",
    "DistributedCurriculumSampler",
    # Base trainer
    "BaseTCRTrainer",
    # Backends
    "AcceleratorBackend",
    "get_backend",
]
