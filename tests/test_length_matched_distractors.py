"""The retrieval eval's distractors must match the true peptide's length.

The AS class-I candidate pool is 76 % 9-mers (2,718 of 3,563). Drawing 99 distractors
uniformly from it therefore leaks the true peptide's length into the ranking: inside
as_paired_i test_level4, 9-mers scored 0.553 and 14-mers 0.013, an order of magnitude
within one condition, and test_iid's 15-mers scored 0.875 against its 9-mers' 0.050.
A scorer with nothing but a length prior reproduces that pattern.

These tests pin the length-matched draw added 2026-11-03, and keep `uniform` available
so the 2026-10-08 wave-1 cell stays reproducible. See quest-cpt/findings.md 9.6.
"""
import csv
import pathlib
import sys
from collections import Counter

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from track_a.esm_sft.judge import (  # noqa: E402
    POOL_SIZE,
    _draw_distractors,
    _pool_by_length,
    _qseed,
)

N = POOL_SIZE - 1                     # 99 distractors per query
FIXTURE = ROOT / "tests" / "fixtures" / "s42_as_paired_i_true_peptides.csv"


def _aa(rng, n):
    return "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=n))


def _pool(rng, spec):
    """spec = {length: count}; distinct peptides, like the real pool."""
    out = []
    for length, count in spec.items():
        while sum(1 for q in out if len(q) == length) < count:
            q = _aa(rng, length)
            if q not in out:
                out.append(q)
    return out


def _realistic_pool(rng):
    """The measured shape: 76 % 9-mers, the rest spread over 8-16."""
    spec = {9: 760}
    spec.update({L: 30 for L in (8, 10, 11, 12, 13, 14, 15, 16)})
    return _pool(rng, spec)


# ------------------------------------------------------------------ the draw itself

def test_pool_by_length_partitions_every_peptide():
    pool = _realistic_pool(np.random.default_rng(0))
    by_len = _pool_by_length(pool)
    assert sum(len(v) for v in by_len.values()) == len(pool)
    assert all(all(len(q) == L for q in v) for L, v in by_len.items())


def test_uniform_mode_reproduces_the_shipped_rng_call():
    """--distractor_draw uniform must reproduce 2026-10-08 bit for bit."""
    pool = _realistic_pool(np.random.default_rng(1))
    by_len = _pool_by_length(pool)
    true = next(q for q in pool if len(q) == 9)

    shipped = [q for q in np.random.default_rng(7).choice(
        pool, size=POOL_SIZE + 8, replace=False) if q != true][: POOL_SIZE - 1]
    got, _ = _draw_distractors(pool, by_len, true, np.random.default_rng(7), N, "uniform")

    assert list(map(str, got)) == list(map(str, shipped))


def test_length_matched_draw_is_all_same_length_when_the_pool_allows():
    pool = _realistic_pool(np.random.default_rng(2))
    by_len = _pool_by_length(pool)
    true = next(q for q in pool if len(q) == 9)          # 760 available, plenty
    got, n_same = _draw_distractors(
        pool, by_len, true, np.random.default_rng(3), N, "length_matched")
    assert len(got) == N
    assert n_same == N
    assert {len(q) for q in got} == {9}


def test_the_true_peptide_is_never_its_own_distractor():
    pool = _realistic_pool(np.random.default_rng(4))
    by_len = _pool_by_length(pool)
    for mode in ("length_matched", "uniform"):
        for length in (9, 14):
            true = next(q for q in pool if len(q) == length)
            got, _ = _draw_distractors(
                pool, by_len, true, np.random.default_rng(5), N, mode)
            assert true not in list(map(str, got)), mode


def test_a_rare_length_fills_from_nearest_first_and_reports_the_shortfall():
    """Only 12 fourteen-mers exist, so 99 cannot be matched. Say so, don't hide it.

    This is the NORMAL case for a long true peptide, not an edge case: the real pool's
    845 non-9-mers are spread over many lengths, so no long peptide has 99 siblings.
    Matching is therefore partial exactly where the confound bit hardest, which is why
    frac_distractors_same_length is reported per condition.
    """
    rng = np.random.default_rng(6)
    pool = _pool(rng, {9: 400, 13: 40, 14: 12, 15: 40})
    by_len = _pool_by_length(pool)
    true = next(q for q in pool if len(q) == 14)

    got, n_same = _draw_distractors(
        pool, by_len, true, np.random.default_rng(8), N, "length_matched")

    assert len(got) == N
    assert n_same == 11                                   # the 12 fourteen-mers minus the true one
    lens = Counter(len(q) for q in got)
    assert lens[14] == 11
    assert lens[13] + lens[15] == 80                      # distance 1 exhausted next
    assert lens[9] == N - 11 - 80                         # only then the far 9-mers
    assert sorted(lens) == [9, 13, 14, 15]                # nothing from lengths the pool lacks


def test_the_draw_is_deterministic_per_row_id():
    pool = _realistic_pool(np.random.default_rng(9))
    by_len = _pool_by_length(pool)
    true = next(q for q in pool if len(q) == 12)
    draw = lambda rid: list(map(str, _draw_distractors(
        pool, by_len, true, np.random.default_rng(_qseed(rid)), N, "length_matched")[0]))
    assert draw("row-1") == draw("row-1")
    assert draw("row-1") != draw("row-2")


# ------------------------------------------------- what the draw does to the metric

def _rank(true, others, favoured=9, eps=0.1):
    """A scorer that is almost nothing but a length prior toward `favoured`.

    It gets a length bonus of 1.0 and a genuine preference for the true peptide of only
    eps, so the length term dominates. Ties count against the true peptide, exactly as
    cmd_eval scores them.
    """
    score = lambda q: 1.0 if len(q) == favoured else 0.0
    s_true = score(true) + eps
    return 1 + sum(1 for q in others if score(q) >= s_true)


def test_a_length_prior_alone_reproduces_the_observed_long_peptide_collapse():
    """Uniform draw: a 14-mer true peptide loses to 9-mer distractors on length alone.

    This is the mechanism behind 14-mers scoring 0.013 while 9-mers scored 0.553 in the
    same condition, with the same model and the same weak real signal.
    """
    pool = _realistic_pool(np.random.default_rng(10))
    by_len = _pool_by_length(pool)
    long_true = next(q for q in pool if len(q) == 14)
    short_true = next(q for q in pool if len(q) == 9)

    long_u, _ = _draw_distractors(
        pool, by_len, long_true, np.random.default_rng(11), N, "uniform")
    short_u, _ = _draw_distractors(
        pool, by_len, short_true, np.random.default_rng(11), N, "uniform")

    assert _rank(long_true, long_u) > 10        # missed, on length and nothing else
    assert _rank(short_true, short_u) == 1      # hit, for the same reason


def test_length_matching_recovers_the_genuine_signal_the_uniform_draw_buried():
    """Same scorer, same weak real signal: rank goes from deep to 1 once length is matched."""
    pool = _realistic_pool(np.random.default_rng(12))
    by_len = _pool_by_length(pool)
    long_true = next(q for q in pool if len(q) == 14)

    uni, _ = _draw_distractors(
        pool, by_len, long_true, np.random.default_rng(13), N, "uniform")
    matched, n_same = _draw_distractors(
        pool, by_len, long_true, np.random.default_rng(13), N, "length_matched")

    # Only 30 fourteen-mers exist here, so the match is partial and honest about it.
    assert n_same == 29
    assert 9 not in {len(q) for q in matched}   # the fill never reaches the 9-mer mass
    assert _rank(long_true, uni) > 10           # buried by 9-mer distractors
    assert _rank(long_true, matched) == 1       # recovered: no distractor wins on length


def test_length_matching_does_not_flatter_a_scorer_with_no_signal():
    """A scorer that cannot tell the true peptide apart must still fail. eps = 0."""
    pool = _realistic_pool(np.random.default_rng(14))
    by_len = _pool_by_length(pool)
    true = next(q for q in pool if len(q) == 14)
    matched, _ = _draw_distractors(
        pool, by_len, true, np.random.default_rng(15), N, "length_matched")
    assert _rank(true, matched, eps=0.0) == POOL_SIZE     # all tied, ties count against


# ------------------------------------------------------- tied to the real eval's data

def test_the_real_cell_spans_the_lengths_that_make_this_matter():
    """The confound is not hypothetical: the scored true peptides are multi-length."""
    assert FIXTURE.exists(), FIXTURE
    with FIXTURE.open() as fh:
        peps = [r["peptide"] for r in csv.DictReader(fh)]
    lens = Counter(len(p) for p in peps)
    assert len(peps) > 1_500
    assert len(lens) >= 4                                  # several lengths in one eval
    assert max(lens.values()) / len(peps) < 0.95           # not one length dominating
