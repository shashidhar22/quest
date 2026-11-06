"""What a score vector may become: the rank rules, and what a checkpoint must contain.

Lifted out of `judge.py` on 2026-11-06 (audit finding 2). Two defects lived in one line each,
and neither could be tested where they were, because `judge.py` imports the CPT trainer and
therefore peft and torch. Nothing here needs more than NumPy.

**A non-finite score used to become a perfect rank.** The rank was

    1 + (sc[1:] >= sc[0]).sum()

and every comparison against NaN is False, so an all-NaN query ranked 1. Rank 1 is a legal rank,
so no downstream validation -- not the gate's rank check, not the comparator's -- could tell it
from a genuine best-possible retrieval. A numerical failure would have been read as the strongest
result in the one number the visit exists to produce. `rank_of` refuses instead, and the caller
turns that into a technical failure with no evaluation artifact.

**An incomplete checkpoint used to evaluate as if trained.** The evaluator merged the saved
trainable tensors into a fresh `model.state_dict()`, so a key absent from the file silently kept
its random initialisation while `load_state_dict` still reported success. Extra keys were already
caught (strict loading rejects them); missing ones were not. `missing_trainable_keys` names the
full expected set from the live model, so the check cannot drift from the model's own structure.

A rank alone also cannot distinguish a dead scorer from an unlucky one -- an all-tie query and a
genuinely last-placed one both score rank N -- so `score_diagnostics` records the spread and the
tie count beside it. That is the failure wave 1 spent 18 GPU-hours discovering.
"""
import numpy as np


class ScoreError(Exception):
    """A score vector that no rank may be computed from. Never downgraded to a caveat."""


def rank_of(scores) -> int:
    """The true peptide's rank among its candidates. Index 0 is the true peptide.

    Ties count AGAINST the true peptide (``>=``), the convention every rank in this programme
    was produced under. Refuses a non-finite score and a candidate set with nothing to rank
    against; both are technical faults, and a fault must not be representable as a result.
    """
    sc = np.asarray(scores, dtype=np.float64).ravel()
    if sc.size < 2:
        raise ScoreError(
            f"a rank needs at least two candidates (the true peptide and one distractor), "
            f"got {sc.size}")
    n_bad = int((~np.isfinite(sc)).sum())
    if n_bad:
        raise ScoreError(
            f"{n_bad} of {sc.size} candidate scores are non-finite, so this query has no rank. "
            f"Ranking anyway would have produced {1 + int((sc[1:] >= sc[0]).sum())}, because "
            f"every comparison against NaN is False -- a numerical failure reported as a "
            f"retrieval result. Fix the scorer or the checkpoint; do not rank around it.")
    return 1 + int((sc[1:] >= sc[0]).sum())


def score_diagnostics(scores) -> dict:
    """Evidence about a score vector, computed even when it is unusable.

    Survives non-finite input on purpose: it exists to describe a failure, so raising on one
    would remove exactly the evidence a reader needs.
    """
    sc = np.asarray(scores, dtype=np.float64).ravel()
    finite = sc[np.isfinite(sc)]
    n_distinct = int(np.unique(np.round(finite, 6)).size) if finite.size else 0
    return {
        "spread": float(finite.max() - finite.min()) if finite.size else 0.0,
        "n_distinct": n_distinct,
        "n_non_finite": int(sc.size - finite.size),
        # a dead scorer: every candidate identical. Only meaningful when scores are usable.
        "all_tied": bool(finite.size == sc.size and sc.size > 1 and n_distinct < 2),
    }


def missing_trainable_keys(saved_keys, model_keys, patterns):
    """(missing, unexpected) for a checkpoint's trainable state dict.

    The expected set is derived from the LIVE model's keys filtered by ``patterns``, so it
    cannot drift from the model's structure the way a hardcoded list would. The frozen base is
    deliberately absent from a checkpoint and is never demanded.
    """
    expected = {k for k in model_keys if any(p in k for p in patterns)}
    saved = set(saved_keys)
    return sorted(expected - saved), sorted(saved - expected)


# The settings a checkpoint must agree with the evaluation on. A checkpoint trained on a
# different task, or with a different serialization, scores a different quantity; the audit asked
# for "compatible checkpoint metadata" alongside the trainable-key check. `adapter` is here
# because the SFT head was trained on top of one specific encoder init: evaluating a base-arm
# checkpoint through the arm-D adapter mixes a trained head with an encoder it never saw.
CKPT_COMPAT_KEYS = ("task", "tcr_variant", "mhc_variant", "adapter", "max_len")


def incompatible_checkpoint_args(saved_args, current_args, keys=CKPT_COMPAT_KEYS):
    """Settings on which a checkpoint and the evaluation reading it disagree.

    A key the checkpoint never recorded is reported as unrecorded rather than treated as a
    match, for the same reason absence is not agreement in a provenance manifest.
    """
    out = []
    saved = saved_args if isinstance(saved_args, dict) else {}
    for k in keys:
        if k not in saved:
            out.append(f"{k}: the checkpoint records nothing; the evaluation wants "
                       f"{current_args.get(k)!r}")
        elif saved[k] != current_args.get(k):
            out.append(f"{k}: trained with {saved[k]!r}, evaluated as {current_args.get(k)!r}")
    return out


class DeviceError(Exception):
    """The evaluation would run somewhere nobody chose. Refused rather than reported."""


def select_device(cuda_available: bool, allow_cpu: bool = False) -> str:
    """Which device an evaluation may use, and a refusal when the answer is "not the one asked for".

    Audit finding 5, 2026-11-06: every entry point read

        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    so a box whose driver had not come up, or whose CUDA build was wrong, evaluated silently on
    the CPU. On a GPU instance that is a result nobody chose to buy, produced at roughly a
    hundredth of the throughput, and the output file says only `cpu` in a field no one reads.
    Opting into CPU explicitly stays available, because the small off-box checks are useful.
    """
    if cuda_available:
        return "cuda"
    if allow_cpu:
        return "cpu"
    raise DeviceError(
        "CUDA is not available, so this would run on the CPU. On a GPU instance that is a "
        "silent fallback to a result nobody chose to pay for, at a fraction of the throughput. "
        "Check the driver (`nvidia-smi`) and the torch build, or pass --allow_cpu if a CPU run "
        "is genuinely what is wanted.")


# What the JUDGE optimises. A constant, not a CLI choice: it is a property of the code, and a
# field the launcher declares is a field the launcher can get wrong.
SFT_OBJECTIVE = "bce-pair-logits/v1"
# The training settings two arms of a paired comparison must share. Only the CPT init and the
# checkpoint trained on top of it may differ (round two of the 2026-11-06 review, finding 4).
SFT_RECIPE_KEYS = ("lr", "batch_size", "epochs", "patience", "max_train_pos", "max_val_pos",
                   "num_workers")
UNRECORDED = "unrecorded"


def sft_recipe(saved_args) -> str:
    """A deterministic string naming the SFT recipe a checkpoint was trained under.

    Taken from the checkpoint's OWN recorded args, so it describes the training that happened
    rather than the defaults in force when it is read. Any missing key makes the whole recipe
    `unrecorded`, because a partially known recipe cannot be shown equal to another: provenance
    refuses that sentinel rather than letting two unrecorded recipes match each other.
    """
    a = saved_args if isinstance(saved_args, dict) else {}
    missing = [k for k in SFT_RECIPE_KEYS if a.get(k) is None]
    if missing:
        return UNRECORDED
    return ";".join(f"{k}={a[k]}" for k in SFT_RECIPE_KEYS)


def cotenants() -> str:
    """Which GPUs this process could see, read from the environment rather than declared.

    Not a claim about what else is running -- nothing visible from inside a process can be --
    but a record of the visibility the run had, which is what differs between a cell that ran
    alone and one that shared a box.
    """
    import os
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    return f"cuda_visible_devices={vis if vis is not None else 'unset'}"
