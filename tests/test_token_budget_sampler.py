"""Falsifiable tests for TokenBudgetStepSampler (Stage 4.3). Laptop-only: ranks are simulated
by constructing one sampler per rank with explicit rank/num_replicas; no torch.distributed init.

Each test states what a failure would look like. The length profile mimics run05 (99.5 % short
rows, a thin tail of long full-MHC rows) plus a deliberately sparse long bucket, because silently
dropping those rows is the failure that matters most for this programme.
"""
import numpy as np
import pytest

from quest.training.samplers import TokenBudgetStepSampler

WORLD, STEP, BUDGET = 4, 1024, 16384


def _lengths(n_short=40_000, seed=0):
    rng = np.random.default_rng(seed)
    short = rng.integers(10, 30, size=n_short)          # ~17-token rows (bucket <64)
    mid = rng.integers(70, 250, size=300)               # buckets <128, <256
    long_ = rng.integers(520, 610, size=7)              # SPARSE: 7 rows in [512,1024)
    edge = np.array([1024, 1024])                       # exactly 1024 -> np.digitize overflow bucket
    return np.concatenate([short, mid, long_, edge])


def _ranks(lengths, **kw):
    return [TokenBudgetStepSampler(lengths, rows_per_step=STEP, token_budget=BUDGET,
                                   num_replicas=WORLD, rank=r, **kw) for r in range(WORLD)]


def test_all_ranks_agree_on_accumulation_plan():
    """Failure: any two ranks with different micro-batch counts for the same step (DDP desync)."""
    plans = [s.step_plan() for s in _ranks(_lengths())]
    assert all(p == plans[0] for p in plans), "ranks disagree on micro-batches per step"
    assert len(plans[0]) > 0


def test_every_step_is_exactly_rows_per_step():
    """Failure: an optimizer step whose rows, summed over ranks, differ from 1024."""
    samplers = _ranks(_lengths())
    per_rank = [list(iter(s)) for s in samplers]
    plan = samplers[0].step_plan()
    cursor = 0
    for n_micro in plan:
        total = sum(len(mb) for r in range(WORLD) for mb in per_rank[r][cursor:cursor + n_micro])
        assert total == STEP, f"step has {total} rows"
        cursor += n_micro
    assert cursor == len(per_rank[0])


def test_micro_batches_respect_token_budget():
    """Failure: rows x longest row in any micro-batch exceeds the budget (OOM risk)."""
    lengths = _lengths()
    for s in _ranks(lengths):
        for mb in s:
            assert len(mb) * lengths[mb].max() <= BUDGET, (len(mb), lengths[mb].max())


def test_no_duplicates_and_sparse_long_rows_are_kept():
    """Failure: a row yielded twice, or the 9 long rows (7 sparse + 2 at exactly 1024) dropped.
    Per-bucket chunking with per-bucket remainder dropping would lose all 9."""
    lengths = _lengths()
    seen = [i for s in _ranks(lengths) for mb in s for i in mb]
    assert len(seen) == len(set(seen)), "row yielded twice in one epoch"
    dropped = set(range(len(lengths))) - set(seen)
    assert len(dropped) < STEP, f"{len(dropped)} rows dropped (expected < one step)"
    long_rows = set(np.where(lengths >= 70)[0])
    assert not (long_rows & dropped), f"{len(long_rows & dropped)} non-short rows dropped"
    assert all(lengths[i] < 64 for i in dropped), "the remainder must come from the short end"


def test_short_rows_get_large_micro_batches():
    """Failure: short-row steps still at 16 rows (the Stage 4.1 starvation this fixes)."""
    s = TokenBudgetStepSampler(np.full(8192, 20), rows_per_step=STEP, token_budget=BUDGET,
                               num_replicas=WORLD, rank=0)
    assert set(len(mb) for mb in s) == {256}, "short rows should fill 256-row micro-batches"
    assert set(s.step_plan()) == {1}, "one micro-batch per rank per step for short rows"


def test_long_rows_stay_within_the_memory_budget():
    """Failure: a 600-token micro-batch exceeding the token budget that Stage 4.1 proved fits in
    memory (16 rows x 1024 = 16,384 tokens), or ranks disagreeing on the cut plan."""
    samplers = [TokenBudgetStepSampler(np.full(4096, 600), rows_per_step=STEP, token_budget=BUDGET,
                                       num_replicas=WORLD, rank=r) for r in range(WORLD)]
    for s in samplers:
        for mb in s:
            assert len(mb) * 600 <= BUDGET, len(mb)
        assert all(n >= 1 for n in s.step_plan())
    assert all(s.step_plan() == samplers[0].step_plan() for s in samplers)
    assert set(len(mb) for mb in samplers[0]) == {27, 256 - 9 * 27}, "budget // 600 = 27 rows, then the tail"


def test_deterministic_per_epoch_and_reshuffled_across_epochs():
    """Failure: same seed+epoch gives a different order, or epoch 1 repeats epoch 0."""
    lengths = _lengths()
    a, b = _ranks(lengths)[0], _ranks(lengths)[0]
    assert list(iter(a)) == list(iter(b))
    e0 = list(iter(a))
    a.set_epoch(1)
    assert list(iter(a)) != e0


def test_world_size_invariant_step_size():
    """Failure: effective rows per optimizer step depends on GPU count (1 vs 4)."""
    lengths = _lengths()
    one = TokenBudgetStepSampler(lengths, rows_per_step=STEP, token_budget=BUDGET, num_replicas=1, rank=0)
    rows, cur = list(iter(one)), 0
    for n in one.step_plan():
        assert sum(len(mb) for mb in rows[cur:cur + n]) == STEP
        cur += n


def test_naive_per_rank_sizing_would_desync():
    """Falsifies the design choice: if each rank sized micro-batches from ITS OWN slice's max
    length (the obvious implementation), ranks would disagree on some step. If this test ever
    fails, the global-max rule is no longer necessary — and the sampler's docstring is wrong."""
    # One step-batch: 1023 short rows and a single 600-token row. Only one rank's slice holds it.
    lengths = np.full(STEP, 20)
    lengths[0] = 600
    s = TokenBudgetStepSampler(lengths, rows_per_step=STEP, token_budget=BUDGET,
                               num_replicas=WORLD, rank=0, shuffle=False)
    per_rank = STEP // WORLD
    naive = {per_rank // s.micro_rows_for(lengths[r * per_rank:(r + 1) * per_rank].max())
             for r in range(WORLD)}
    assert len(naive) > 1, "naive per-rank sizing agreed — the global-max rule would be unneeded"
    plans = [TokenBudgetStepSampler(lengths, rows_per_step=STEP, token_budget=BUDGET,
                                    num_replicas=WORLD, rank=r, shuffle=False).step_plan()
             for r in range(WORLD)]
    assert all(p == plans[0] for p in plans), plans
    # The one 600-token row bounds the first cut (16384 // 600 = 27 rows) on EVERY rank, even the
    # three ranks that do not hold it; the remaining 229 short rows form one 256-capped micro-batch.
    assert plans[0] == [2], plans[0]


def test_rejects_indivisible_world_size():
    with pytest.raises(ValueError):
        TokenBudgetStepSampler(np.full(100, 20), rows_per_step=1024, num_replicas=3, rank=0)


def test_steps_are_content_mixed_not_length_homogeneous():
    """The v2 fix. Failure: the sparse long rows are concentrated into one or two steps (v1 put
    all of them into a single length-homogeneous step; measured 2026-10-06, that concentration
    cost 17-31 % on framework/pocket perplexity at matched steps). With a global permutation the
    9 long rows should land in ~9 distinct steps out of 39."""
    lengths = _lengths()
    s = TokenBudgetStepSampler(lengths, rows_per_step=STEP, token_budget=BUDGET, num_replicas=1, rank=0)
    rows, cur, steps_with_long = list(iter(s)), 0, 0
    for n in s.step_plan():
        step_rows = [i for mb in rows[cur:cur + n] for i in mb]
        steps_with_long += any(lengths[i] >= 512 for i in step_rows)
        cur += n
    assert steps_with_long >= 7, f"long rows concentrated into {steps_with_long} step(s)"


def test_within_a_step_ranks_are_dealt_round_robin_longest_first():
    """Failure: the round-robin deal is wrong, so some rank's row at position k is longer than the
    global bound used for the cut at k — the token budget would then be violated on that rank.
    Checks the invariant directly: for every step and position, max over ranks == the bound."""
    lengths = _lengths()
    samplers = _ranks(lengths)
    per_rank = [list(iter(s)) for s in samplers]
    plan = samplers[0].step_plan()
    cur = 0
    for n in plan:
        shares = [[i for mb in per_rank[r][cur:cur + n] for i in mb] for r in range(WORLD)]
        assert len(set(map(len, shares))) == 1
        step_rows = sorted((i for sh in shares for i in sh), key=lambda i: -lengths[i])
        bound = [lengths[step_rows[k * WORLD]] for k in range(len(shares[0]))]
        for k in range(len(shares[0])):
            assert max(lengths[sh[k]] for sh in shares) == bound[k], (k, bound[k])
        cur += n


def test_micro_batch_loss_scale_matches_fixed_batch_only_on_equal_micro_batches():
    """Renamed 2026-11-04. Every assertion below is sound, but the old name --
    ``..._reproduces_fixed_batch_weighting`` -- states the claim findings.md 9.5 REFUTED, and a
    reader grepping for whether row-share equals the fixed-batch path would find a passing test
    with that name and conclude it does. It does not: row-share coincides with 1/n_acc ONLY when
    micro-batches are equal in rows and tokens, which is exactly what the token-budget sampler's
    length-homogeneous cuts make them NOT. At the real length mix it gives MHC-bearing tokens
    0.43x the fixed-batch per-token weight. The unequal case lives in
    tests/test_micro_batch_loss_weighting.py, which pins all four weightings.

    Failure: row-weighting disagrees with 1/n_acc on EQUAL micro-batches, or a micro-batch's
    weight does not scale with its rows under the token-budget path."""
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "training"))
    from quest.training.loss_normalization import micro_batch_loss_scale as w
    assert w(16, 256, 16, token_mode=True) == pytest.approx(w(16, 256, 16, token_mode=False)) == pytest.approx(1 / 16)
    assert w(256, 256, 1, True) == pytest.approx(1.0)
    assert w(27, 256, 2, True) + w(229, 256, 2, True) == pytest.approx(1.0)   # a step's weights sum to 1
    assert w(27, 256, 2, True) / w(229, 256, 2, True) == pytest.approx(27 / 229)
