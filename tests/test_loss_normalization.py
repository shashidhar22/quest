"""The exact global masked-target mean: one target, one weight, however rows are split.

Decision 1 of 2026-11-04, approved by Yuexin. The objective for future CPT baselines is

    L = (sum of the loss over every supervised target in the optimizer step)
        / (count of those targets, summed over all ranks)

The shipped `token` weighting does not compute that. `TokenBudgetStepSampler.step_token_plan()`
sums INPUT lengths while `raw_loss` is a mean over the REALIZED masked targets, so per-target
weight is uniform only where mask count happens to track length. The audit's algebra: two
100-token micro-batches with 10 and 20 supervised targets each receive weight 0.5, so their
targets receive 0.05 and 0.025.

These are real gradient tests on CPU torch, not arithmetic on the scale factor alone. The property
that matters is that the ACCUMULATED gradient is invariant to how the step's rows are partitioned
and to how they are spread across ranks, because that is what "the objective does not depend on
batching" means. The existing test for the old weighting allows 15 % relative error and builds
mask counts as round(0.15 * length) at lengths 17 and 600, so mask count is proportional to length
by construction and it cannot detect the gap it was meant to cover.
"""
import pytest

torch = pytest.importorskip("torch", reason="gradient equivalence needs CPU torch")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quest.training.loss_normalization import (  # noqa: E402
    exact_microbatch_scale,
    masked_target_count,
    sum_from_mean,
    summed_masked_loss,
)

VOCAB, DIM = 11, 6
IGNORE = -100


def model(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Linear(DIM, VOCAB)


def batch(n_rows, seq, n_targets_per_row, seed=0):
    """Fixed inputs and labels: `n_targets_per_row` supervised positions per row, rest ignored."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_rows, seq, DIM, generator=g)
    labels = torch.full((n_rows, seq), IGNORE, dtype=torch.long)
    for r in range(n_rows):
        k = n_targets_per_row if isinstance(n_targets_per_row, int) else n_targets_per_row[r]
        labels[r, :k] = torch.randint(0, VOCAB, (k,), generator=g)
    return x, labels


def grad_of(m):
    return torch.cat([p.grad.reshape(-1) for p in m.parameters()])


def reference_gradient(m, x, labels):
    """One batch, loss = summed target loss / target count. The objective, by definition."""
    m.zero_grad()
    logits = m(x)
    total = summed_masked_loss(logits, labels)
    n = masked_target_count(labels)
    (total / n).backward()
    return grad_of(m).clone()


def accumulated_gradient(m, pieces, n_global, world_size=1):
    """Backward each micro-batch with the exact scale, as the trainer would."""
    m.zero_grad()
    scale = exact_microbatch_scale(n_global, world_size=world_size)
    for x, labels in pieces:
        logits = m(x)
        total = summed_masked_loss(logits, labels)
        (total * scale).backward()
    return grad_of(m).clone()


def split(x, labels, sizes):
    out, i = [], 0
    for s in sizes:
        out.append((x[i:i + s], labels[i:i + s]))
        i += s
    assert i == x.size(0), "the partition must cover the batch"
    return out


# --------------------------------------------------------- the invariance that defines it
def test_the_gradient_does_not_depend_on_the_micro_batch_partition():
    """Unequal targets per row is the case the old weighting gets wrong."""
    x, labels = batch(8, 12, [1, 2, 3, 4, 5, 6, 7, 8])
    n_global = masked_target_count(labels)

    m = model()
    ref = reference_gradient(m, x, labels)
    for sizes in ([8], [4, 4], [1, 7], [2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1], [5, 3]):
        got = accumulated_gradient(m, split(x, labels, sizes), n_global)
        assert torch.allclose(got, ref, atol=1e-6), (
            f"partition {sizes} moved the gradient: max diff {(got - ref).abs().max():.3e}"
        )


def test_the_gradient_does_not_depend_on_the_world_size():
    """DDP averages across ranks, so each rank scales by world_size to undo that."""
    x, labels = batch(8, 12, [1, 2, 3, 4, 5, 6, 7, 8])
    n_global = masked_target_count(labels)
    m = model()
    ref = reference_gradient(m, x, labels)

    for shards in ([[0, 1, 2, 3], [4, 5, 6, 7]],            # 2 ranks, equal rows
                   [[0, 1, 2, 3, 4, 5], [6, 7]],            # 2 ranks, unequal rows AND targets
                   [[0, 1], [2, 3], [4, 5], [6, 7]]):       # 4 ranks
        world = len(shards)
        per_rank = []
        for rows in shards:
            idx = torch.tensor(rows)
            pieces = [(x[idx[i:i + 1]], labels[idx[i:i + 1]]) for i in range(len(rows))]
            per_rank.append(accumulated_gradient(m, pieces, n_global, world_size=world))
        averaged = torch.stack(per_rank).mean(0)      # exactly what DDP does
        assert torch.allclose(averaged, ref, atol=1e-6), (
            f"world_size {world} moved the gradient: "
            f"max diff {(averaged - ref).abs().max():.3e}"
        )


def test_every_target_carries_the_same_weight():
    """The contract in its own terms, read off the gradient of a one-target-per-row batch."""
    x, labels = batch(4, 6, 1)
    n_global = masked_target_count(labels)
    assert n_global == 4
    m = model()
    # Each single-row micro-batch must contribute exactly a quarter of the gradient.
    per_row = []
    for i in range(4):
        m.zero_grad()
        logits = m(x[i:i + 1])
        total = summed_masked_loss(logits, labels[i:i + 1])
        (total * exact_microbatch_scale(n_global)).backward()
        per_row.append(grad_of(m).clone())
    ref = reference_gradient(m, x, labels)
    assert torch.allclose(torch.stack(per_row).sum(0), ref, atol=1e-6)


# --------------------------------------------------------- the control: the old scale fails it
def test_the_input_length_share_scale_is_not_partition_invariant():
    """A verification that cannot fail proves nothing, so the old contract is measured here.

    Rows of equal length carrying different numbers of supervised targets get equal weight
    under an input-length share, so their targets do not.
    """
    seq = 12
    x, labels = batch(4, seq, [1, 1, 9, 9])          # equal length, very unequal target counts
    n_global = masked_target_count(labels)
    m = model()
    ref = reference_gradient(m, x, labels)

    m.zero_grad()
    pieces = split(x, labels, [2, 2])                 # [1,1] targets vs [9,9] targets
    step_tokens = sum(p[0].size(0) * seq for p in pieces)
    for xi, li in pieces:
        logits = m(xi)
        n_i = masked_target_count(li)
        raw_mean = summed_masked_loss(logits, li) / n_i        # what the trainer's raw_loss is
        mb_tokens = xi.size(0) * seq
        (raw_mean * (mb_tokens / step_tokens)).backward()      # the shipped `token` scale
        del logits
    old = grad_of(m).clone()
    rel = (old - ref).abs().max() / ref.abs().max()
    assert rel > 0.1, (
        f"the old scale matched the exact objective to {rel:.3%}; this control is vacuous and "
        "the fixture needs mask counts that do not track length"
    )


# --------------------------------------------------------- edges the reviewer named
def test_a_micro_batch_with_no_targets_contributes_nothing_and_no_nan():
    """Masking can leave a micro-batch with zero supervised targets.

    A mean over zero elements is NaN, and NaN times zero is still NaN, so the summed form is
    not a stylistic choice: it is what makes this case finite. DDP also requires every rank to
    run the same number of backward passes, so skipping is not an option.
    """
    x, labels = batch(4, 6, [2, 2, 0, 0])
    n_global = masked_target_count(labels)
    assert n_global == 4
    m = model()
    ref = reference_gradient(m, x, labels)

    got = accumulated_gradient(m, split(x, labels, [2, 2]), n_global)
    assert torch.isfinite(got).all(), "an empty micro-batch produced a non-finite gradient"
    assert torch.allclose(got, ref, atol=1e-6)


def test_an_entirely_empty_micro_batch_is_finite_on_its_own():
    x, labels = batch(2, 6, 0)
    m = model()
    m.zero_grad()
    logits = m(x)
    total = summed_masked_loss(logits, labels)
    assert float(total) == 0.0
    assert masked_target_count(labels) == 0
    (total * exact_microbatch_scale(4)).backward()
    assert torch.isfinite(grad_of(m)).all()


def test_a_zero_global_target_count_is_refused():
    """A whole step with no supervised target cannot be normalized; it must not divide by zero."""
    with pytest.raises(ValueError, match="no supervised targets"):
        exact_microbatch_scale(0)


def test_the_scale_belongs_to_the_step_not_the_micro_batch():
    """It is world_size / N_global and does not depend on what the micro-batch holds.

    Scaling a SUMMED loss by n_i / N_global instead would weight the micro-batch by its target
    count twice, which is the bug these tests caught while this module was being written.
    """
    assert exact_microbatch_scale(12) == pytest.approx(1 / 12)
    assert exact_microbatch_scale(12, world_size=4) == pytest.approx(4 / 12)


def test_sum_from_mean_matches_the_summed_reduction():
    """The HF path reports a mean, so the trainer multiplies back by the count."""
    x, labels = batch(3, 8, [2, 3, 4])
    m = model()
    logits = m(x)
    n = masked_target_count(labels)
    mean = torch.nn.functional.cross_entropy(
        logits.reshape(-1, VOCAB), labels.reshape(-1), ignore_index=IGNORE)
    assert float(sum_from_mean(mean, n)) == pytest.approx(
        float(summed_masked_loss(logits, labels)), rel=1e-6)


# --------------------------------------------------------- the name must match the behaviour
def test_the_semantics_table_names_all_three_objectives():
    from quest.training.resume_compat import LOSS_WEIGHTING_SEMANTICS as S

    assert S["row"] == "row-share/v1"
    assert S["token"] == "input-length-share/v1", (
        "the shipped `token` must keep its v1 label while the trainer still computes v1"
    )
    assert S["token_exact"] == "global-masked-target-mean/v2"


def test_the_trainer_refuses_the_new_objective_until_its_loop_lands():
    """Fail closed: a name the training loop does not implement must not start a run.

    The exact mean needs the step's global target count before its first backward, which the
    current loop cannot supply. Until that lands, asking for it has to stop the run rather than
    silently train the old objective under the new name.
    """
    from quest.training.loss_normalization import micro_batch_loss_scale

    with pytest.raises(ValueError, match="not implemented in the training loop"):
        micro_batch_loss_scale(4, 16, 4, token_mode=True, weighting="token_exact",
                               mb_tokens=10, step_tokens=40)


def test_the_historical_weightings_still_compute_what_they_did():
    """Moving the helper must not move a number: Stage 4.1 and 4.3 runs are reproducible."""
    from quest.training.loss_normalization import micro_batch_loss_scale as f

    assert f(16, 1024, 64, token_mode=False) == pytest.approx(1 / 64)
    assert f(4, 16, 4, token_mode=True, weighting="row") == pytest.approx(0.25)
    assert f(4, 16, 4, token_mode=True, weighting="token",
             mb_tokens=10, step_tokens=40) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="step_tokens"):
        f(4, 16, 4, token_mode=True, weighting="token", mb_tokens=10, step_tokens=0)
    with pytest.raises(ValueError, match="unknown weighting"):
        f(4, 16, 4, token_mode=True, weighting="nonsense")
