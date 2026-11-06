"""The exact global masked-target mean, as three small pieces the training loop composes.

Decision 1 of 2026-11-04, approved by Yuexin. The objective for future CPT baselines is

    L = (sum of the loss over every supervised target in the optimizer step)
        / (count of those targets, summed over all ranks)

and the point of it is that every supervised token carries the same weight however the step's
rows were partitioned and however they were spread across ranks.

WHY THE SHIPPED `token` WEIGHTING IS NOT THAT. `TokenBudgetStepSampler.step_token_plan()` sums
INPUT lengths, while the trainer's `raw_loss` is a mean over the REALIZED masked targets. Scaling
a masked-target mean by an input-length share makes per-target weight uniform only where mask
count tracks length. Two 100-token micro-batches with 10 and 20 supervised targets each get
weight 0.5, so their targets get 0.05 and 0.025. The existing test could not see this: it allows
15 % relative error and builds mask counts as round(0.15 x length), which is proportional to
length by construction.

WHY SUMMED AND NOT MEAN. Three reasons, all forced:
  - a mean over zero elements is NaN, and NaN times a zero weight is still NaN, so a micro-batch
    that masking happened to leave with no supervised target would poison the step;
  - DDP requires every rank to run the same number of backward passes, so skipping such a
    micro-batch is not available;
  - the denominator belongs to the STEP, not the micro-batch, so dividing inside the micro-batch
    and multiplying back is a detour that loses precision for nothing.

WHY world_size APPEARS. DDP all-reduces gradients as a MEAN over ranks. The objective is a sum
over the step's targets wherever they live, so each rank multiplies by the world size to undo
that averaging. With equal target counts per rank this is the familiar no-op; with unequal counts
it is the whole correction.

CLIPPING ORDER is unchanged and already correct in the loop: accumulate every micro-batch of the
step, then clip the resulting gradient once, then step the optimizer. Clipping per micro-batch
would make the result depend on the partition again, which is the property this objective exists
to remove.

WHAT THE LOOP STILL HAS TO DO, and why this module cannot do it alone: the scale needs the step's
GLOBAL target count before the step's FIRST backward. The current loop pulls one micro-batch at a
time from the DataLoader, so it cannot know that count yet. Landing this objective therefore means
buffering the step's micro-batches, counting their targets, one all-reduce of the count, and then
forward/backward over the buffer. Until that lands, `micro_batch_loss_scale` refuses the name, so
no run can be started under an objective the loop does not implement. See
`plans/v04-audit-repair.md` step 5.

Imports torch and nothing else from this repo, so the contract is testable on a CPU with no peft
or esm present (`tests/test_loss_normalization.py`).
"""
import torch

IGNORE_INDEX = -100


def masked_target_count(labels) -> int:
    """How many positions are supervised, i.e. not the ignore index."""
    return int((labels != IGNORE_INDEX).sum())


def summed_masked_loss(logits, labels, ignore_index: int = IGNORE_INDEX):
    """Cross entropy SUMMED over the supervised targets, 0.0 when there are none.

    Summed rather than averaged so that an empty micro-batch is finite: see the module docstring.
    """
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="sum",
    )


def sum_from_mean(mean_loss, n_targets_microbatch: int):
    """Recover the summed target loss from a mean over the same targets.

    The HuggingFace path reports ``out.loss``, a mean over the micro-batch's supervised targets,
    so the trainer multiplies it back by that count rather than recomputing from logits.

    ZERO TARGETS. ``out.loss`` is a mean over no elements, i.e. NaN, and ``NaN * 0`` is NaN, so
    the naive multiply let one unmasked micro-batch poison a whole step's accumulated gradient
    (audit finding 4, 2026-11-06). Such a micro-batch contributes nothing, so the honest return
    is a finite zero. It is DETACHED from the graph, which is the one thing a caller must know:
    DDP with ``find_unused_parameters=False`` wants every rank's backward to touch the same
    parameters, and a detached zero touches none. A loop that must keep that invariant should
    take the summed path instead -- ``summed_masked_loss`` returns a graph-connected 0.0 for an
    all-ignore batch, because cross entropy with ``reduction="sum"`` sums zero terms rather than
    dividing by zero. That is the main reason the summed path is the one this objective specifies.
    """
    if n_targets_microbatch < 0:
        raise ValueError(
            f"a micro-batch cannot have a negative supervised-target count, got "
            f"{n_targets_microbatch}")
    if n_targets_microbatch == 0:
        return torch.zeros_like(mean_loss) if torch.is_tensor(mean_loss) else 0.0
    return mean_loss * n_targets_microbatch


def exact_microbatch_scale(n_targets_step_global: int, world_size: int = 1) -> float:
    """The factor a micro-batch's SUMMED loss is multiplied by before backward.

    It is ``world_size / N_global`` and deliberately does NOT depend on the micro-batch. That is
    the whole content of the objective: the denominator belongs to the optimizer step, so

        L = sum_over_microbatches(summed loss) / N_global

    and every micro-batch carries the same 1/N_global whatever it happens to contain. A scale of
    ``n_i / N_global`` would be right for a micro-batch MEAN, and using it on a summed loss
    weights a micro-batch by its target count twice over.

    ``world_size`` undoes DDP's averaging across ranks; pass 1 on a single process.
    """
    if n_targets_step_global <= 0:
        raise ValueError(
            "this optimizer step has no supervised targets anywhere, so its loss cannot be "
            "normalized. A step of entirely unmasked rows is a data or collator fault, not a "
            "number to divide by zero."
        )
    if world_size < 1:
        raise ValueError(f"world_size must be at least 1, got {world_size}")
    return world_size / float(n_targets_step_global)


def global_target_count(local_counts, group=None) -> int:
    """Sum the step's local target counts across ranks, with one all-reduce.

    Returns the local sum unchanged when distributed training is not initialised, so the
    single-process path needs no special case at the call site.
    """
    total = int(sum(local_counts))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        t = torch.tensor([total], dtype=torch.long)
        if torch.cuda.is_available():
            t = t.cuda()
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM, group=group)
        total = int(t.item())
    return total


def micro_batch_loss_scale(n_rows: int, rows_per_rank: int, n_acc: int, token_mode: bool,
                           weighting: str = "row", mb_tokens: int = 0,
                           step_tokens: int = 0) -> float:
    """Weight of one micro-batch's mean loss within the optimizer step.

    ``raw_loss`` is already a mean over the micro-batch's MASKED TOKENS, so this scale decides how
    much each token ends up contributing to the step's gradient.

    Fixed-batch path (Stage 4.1): ``1 / n_acc``, the mean of ``n_acc`` equal micro-batch means.
    Its micro-batches hold 16 randomly chosen rows, so each one's token mix is close to the global
    mix and per-token weight comes out near uniform.

    Token-budget paths, selected by ``weighting``:

    ``"row"`` — ``n_rows / rows_per_rank``, the micro-batch's ROW share of the rank's step. Weights
      every row alike. **This is not equivalent to the fixed-batch path**, because the v2 cuts are
      length-homogeneous: full-MHC rows land in their own small, token-rich micro-batch and their
      tokens are divided down. Measured against the real sampler at the Stage 4.1 length mix
      (99.5 % of rows ~17 tokens), MHC-bearing tokens receive **0.43x** the per-token gradient
      weight they had under the fixed-batch path, while short rows keep theirs. That is the
      mechanism behind Stage 4.3's better-overall-worse-pocket split (quest-cpt/findings.md 9.5).
      Kept as the default only so existing runs stay reproducible.

    ``"token"`` — ``mb_tokens / step_tokens`` from ``TokenBudgetStepSampler.step_token_plan()``,
      the micro-batch's TOKEN share of the rank's step. Per-token weight is then uniform across
      the step however the rows were partitioned, which is what the fixed-batch path approximated.
      Use this to keep the token-budget speedup without moving the objective.

    Both token-budget paths coincide with ``1 / n_acc`` when micro-batches are equal in rows and
    tokens (``n_rows * n_acc == rows_per_rank``).
    """
    # Validation comes FIRST, before any return. Until 2026-11-06 the fixed-batch branch
    # returned 1/n_acc without ever looking at `weighting`, so `token_exact` with fixed batches
    # silently produced a number instead of refusing (audit finding 4). A guard that only covers
    # the branch the CLI happens to take is not a guard.
    if weighting == "token_exact":
        # The objective approved 2026-11-04: summed target loss over the step's GLOBAL target
        # count. It cannot be expressed as a per-micro-batch factor, because the denominator is
        # not known until the step's micro-batches have been counted and all-reduced. Kept inline
        # and self-contained so this function refuses correctly even when extracted on its own.
        raise ValueError(
            "loss_weighting 'token_exact' (global-masked-target-mean/v2) is not implemented in "
            "the training loop yet. Its contract and gradient tests live in "
            "quest/training/loss_normalization.py; the loop must buffer each optimizer step and "
            "all-reduce its supervised-target count first. See plans/v04-audit-repair.md step 5. "
            "Use 'token' only to reproduce the historical input-length-share objective."
        )
    if weighting not in ("row", "token"):
        raise ValueError(
            f"unknown weighting {weighting!r}; expected 'row', 'token' or 'token_exact'")
    if not token_mode:
        return 1.0 / n_acc
    if weighting == "token":
        if step_tokens <= 0:
            raise ValueError("token-share weighting needs step_tokens > 0 "
                             "(from TokenBudgetStepSampler.step_token_plan())")
        return mb_tokens / float(step_tokens)
    return n_rows / float(rows_per_rank)
