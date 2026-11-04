"""
quest.training - Training utilities for the QUEST framework.

Consolidates training components (callbacks, collators, samplers) that were
previously duplicated across multiple trainer scripts.

Re-exports are LAZY (PEP 562). They used to be eager, which made this package
unimportable wherever torch is absent -- and that reached further than the heavy
modules themselves: `import quest.training.resume_compat`, a stdlib-only module,
executed this file first and so pulled in collators and torch. Extracting a rule
into a light module achieves nothing if its package drags the ML stack in behind
it, so submodule imports must stay cheap.

`from quest.training import EarlyStopping` still works and still raises if the
dependency is genuinely missing; it just defers the cost to first use.
"""
import importlib
from typing import TYPE_CHECKING

# name -> submodule that defines it
_EXPORTS = {
    # Callbacks
    "EarlyStopping": "quest.training.callbacks",
    "StreamingMetricsTracker": "quest.training.callbacks",
    # Collators
    "DataCollatorForMLMDynamic": "quest.training.collators",
    "DataCollatorForMLMWithPacking": "quest.training.collators",
    "DataCollatorForMLMWithVarlen": "quest.training.collators",
    "TaskSpecificMaskingCollator": "quest.training.collators",
    "ContrastivePairCollator": "quest.training.collators",
    # Samplers
    "LengthBucketSampler": "quest.training.samplers",
    "DistributedLengthBucketSampler": "quest.training.samplers",
    "CurriculumSampler": "quest.training.samplers",
    "DistributedCurriculumSampler": "quest.training.samplers",
    # Base trainer
    "BaseTCRTrainer": "quest.training.base_trainer",
    # Backends
    "AcceleratorBackend": "quest.training.backends",
    "get_backend": "quest.training.backends",
}

__all__ = [  # literal, so static checkers can read it; kept in step with _EXPORTS by a test
    "EarlyStopping",
    "StreamingMetricsTracker",
    "DataCollatorForMLMDynamic",
    "DataCollatorForMLMWithPacking",
    "DataCollatorForMLMWithVarlen",
    "TaskSpecificMaskingCollator",
    "ContrastivePairCollator",
    "LengthBucketSampler",
    "DistributedLengthBucketSampler",
    "CurriculumSampler",
    "DistributedCurriculumSampler",
    "BaseTCRTrainer",
    "AcceleratorBackend",
    "get_backend",
]


def __getattr__(name):
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value          # resolve once
    return value


def __dir__():
    return sorted(set(__all__) | set(globals()))


if TYPE_CHECKING:  # so type checkers and IDEs still see the real names
    from quest.training.backends import AcceleratorBackend, get_backend
    from quest.training.base_trainer import BaseTCRTrainer
    from quest.training.callbacks import EarlyStopping, StreamingMetricsTracker
    from quest.training.collators import (
        ContrastivePairCollator,
        DataCollatorForMLMDynamic,
        DataCollatorForMLMWithPacking,
        DataCollatorForMLMWithVarlen,
        TaskSpecificMaskingCollator,
    )
    from quest.training.samplers import (
        CurriculumSampler,
        DistributedCurriculumSampler,
        DistributedLengthBucketSampler,
        LengthBucketSampler,
    )
