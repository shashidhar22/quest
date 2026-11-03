"""Stage 4.3: what the token-budget sampler does to per-region gradient weight.

`raw_loss` is a mean over a micro-batch's masked tokens, so the micro-batch's weight inside its
step decides how much each token contributes. The v2 cuts are length-homogeneous and in this data
length IS content (a row is long because it carries the full MHC chain), so the weighting rule and
the partition interact. Three distinct objectives result, and none of them is the other:

    token-uniform   every masked token weighted alike (a plain full-batch MLM loss)
    fixed-16        Stage 4.1: equal weight per 16-random-row micro-batch
    v2 row-share    the shipped v2 default
    v2 token-share  --loss_weighting token

These tests pin the relationships down so that "v2 only changes speed" cannot be believed again,
and so that adopting token-share is recognised as a recipe change rather than an equivalence fix.
Measured 2026-10-15; see quest-cpt/findings.md 9.5.
"""
import sys
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "training"))
from quest.training.samplers import TokenBudgetStepSampler  # noqa: E402
from cpt_training_lib import micro_batch_loss_scale as scale  # noqa: E402

ROWS_PER_STEP = 1024
SHORT, LONG = 17, 600          # the Stage 4.1 mix: 99.5 % of rows ~17 tokens, the rest full-MHC
MASK_RATE = 0.15


def _masked(length):
    return max(1, int(round(MASK_RATE * length)))


def weights(frac_long=0.005, n_rows=1024 * 20, seed=0):
    """Total per-token gradient weight landing on long vs short rows, under each scheme."""
    rng = np.random.default_rng(seed)
    lengths = np.full(n_rows, SHORT)
    lengths[rng.choice(n_rows, int(n_rows * frac_long), replace=False)] = LONG
    tok = np.array([_masked(L) for L in lengths])
    is_long = lengths == LONG

    s = TokenBudgetStepSampler(lengths, rows_per_step=ROWS_PER_STEP,
                               num_replicas=1, rank=0, seed=42)
    plan, tplan, mbs = s.step_plan(), s.step_token_plan(), list(iter(s))
    acc = {k: np.zeros(2) for k in ("fixed16", "v2_row", "v2_token")}
    i = 0
    for si, n_acc in enumerate(plan):
        step, i = mbs[i:i + n_acc], i + n_acc
        step_tokens = sum(tplan[si])
        for mi, mb in enumerate(step):
            idx = np.asarray(mb)
            m = tok[idx].sum()
            for key, w in (
                ("v2_row", scale(len(idx), s.rows_per_rank, n_acc, True, "row")),
                ("v2_token", scale(len(idx), s.rows_per_rank, n_acc, True, "token",
                                   tplan[si][mi], step_tokens)),
            ):
                acc[key] += (w / m) * np.array([tok[idx][is_long[idx]].sum(),
                                                tok[idx][~is_long[idx]].sum()])
        # Stage 4.1: the same rows in random 16-row micro-batches, each weighted 1/n_acc
        rows = np.concatenate([np.asarray(m) for m in step])
        rng.shuffle(rows)
        groups = [rows[j:j + 16] for j in range(0, len(rows), 16)]
        for mb in groups:
            acc["fixed16"] += (scale(16, s.rows_per_rank, len(groups), False) / tok[mb].sum()) \
                * np.array([tok[mb][is_long[mb]].sum(), tok[mb][~is_long[mb]].sum()])
    uniform = tok[is_long].sum() / tok[~is_long].sum()
    return acc, uniform


@pytest.fixture(scope="module")
def w():
    return weights()


def test_row_share_starves_mhc_tokens_relative_to_stage41(w):
    """The Stage 4.3 regression: v2's default gives MHC tokens ~0.43x Stage 4.1's weight."""
    acc, _ = w
    ratio = acc["v2_row"][0] / acc["fixed16"][0]
    assert ratio < 0.6, f"expected MHC tokens starved under row share, got {ratio:.3f}"
    assert 0.35 < ratio < 0.55, f"measured 0.43 on 2026-10-15, got {ratio:.3f}"


def test_row_share_leaves_short_rows_alone(w):
    """Only the long rows move; that asymmetry is what split overall from pocket."""
    acc, _ = w
    assert 0.95 < acc["v2_row"][1] / acc["fixed16"][1] < 1.15


def test_token_share_is_token_uniform(w):
    """--loss_weighting token makes per-token weight uniform, which is the MLM objective."""
    acc, uniform = w
    got = acc["v2_token"][0] / acc["v2_token"][1]
    assert got == pytest.approx(uniform, rel=0.15), \
        f"token share should match the token-count ratio {uniform:.4f}, got {got:.4f}"


def test_token_share_is_not_equivalent_to_stage41_either(w):
    """The load-bearing correction: token share UP-weights MHC tokens ~2.8x vs Stage 4.1.

    So it cannot be adopted as an 'equivalence fix' — every Stage 4.1 control stops being valid
    for runs trained under it.
    """
    acc, _ = w
    ratio = acc["v2_token"][0] / acc["fixed16"][0]
    assert ratio > 2.0, f"expected token share to raise MHC weight well above 4.1, got {ratio:.3f}"


def test_stage41_was_never_token_uniform(w):
    """Stage 4.1's own weighting is an artifact of batch size 16, not a principled choice."""
    acc, uniform = w
    assert acc["fixed16"][0] / acc["fixed16"][1] < 0.5 * uniform


@pytest.mark.parametrize("frac_long", [0.005, 0.01, 0.02])
def test_the_starvation_worsens_as_mhc_rows_get_denser(frac_long):
    """Balanced datasets (Stage 4.4) carry far more full-MHC rows, so this matters more there."""
    acc, _ = weights(frac_long=frac_long, n_rows=1024 * 10)
    assert acc["v2_row"][0] / acc["fixed16"][0] < 0.6


def test_token_share_requires_the_step_total():
    """A caller that forgets step_token_plan() must fail loudly, not silently mis-weight."""
    with pytest.raises(ValueError, match="step_tokens"):
        scale(256, 1024, 5, True, "token", 4352, 0)


# --------------------------------------------------------------------------- early stopping off
# Stage 4.3's V2_FINAL compared an early-stopped run (step 4000 of 7800) with one that ran its
# full budget. plan.md already forbade early stopping in single-configuration runs; these pin the
# switch that enforces it.
from cpt_training_lib import CdrPplPlateauCallback  # noqa: E402


def _feed(cb, seq):
    return [cb({"m": v}) for v in seq]


RISE_THEN_STALL = [3.0, 2.0, 1.9, 1.85, 1.86, 1.87, 1.88, 1.89]


def test_patience_zero_never_stops():
    cb = CdrPplPlateauCallback(n_patience=0, metric_path=("m",))
    assert not any(_feed(cb, RISE_THEN_STALL))
    assert all(h.get("disabled") for h in cb.history)


def test_negative_patience_also_means_off():
    cb = CdrPplPlateauCallback(n_patience=-1, metric_path=("m",))
    assert not any(_feed(cb, RISE_THEN_STALL))


def test_patience_three_still_stops_on_a_plateau():
    """The guard must not disable early stopping for runs that do want it."""
    cb = CdrPplPlateauCallback(n_patience=3, min_rel_improvement=0.01, metric_path=("m",))
    out = _feed(cb, RISE_THEN_STALL)
    assert any(out), "a genuine plateau must still stop"


def test_disabled_callback_still_records_the_curve():
    """Turning the stop off must not cost the eval history that the verdict is read from."""
    cb = CdrPplPlateauCallback(n_patience=0, metric_path=("m",))
    _feed(cb, [3.0, 2.0, 1.9])
    assert [h["metric"] for h in cb.history] == [3.0, 2.0, 1.9]
