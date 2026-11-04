"""Pins the Stage 4.2 readability predicate against the REAL V03 base cell.

Fixtures are the shipped per-query ranks and the row-to-peptide/cluster map pulled from the
benchmark splits, so these tests fail if the estimator drifts rather than if a toy example drifts.
Every number here was measured on 2026-11-04 under the UNIFORM distractor draw, which is what the
V03 ranks were scored against; the length-matched draw changes the pool and these need recomputing
when a cell runs under it.

Pre-registered effect and power convention: quest-cpt/box-jobs/prereg-4.2-effect.md, committed
before any of these numbers existed.
"""
import csv
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts" / "training" / "stage42_box"))
import readability as R  # noqa: E402

PREREG_DELTA = 0.05
PUBLISHED_CLUSTER_CI = {            # findings.md 9.6, computed independently there
    "test_level4": (0.172, 0.633),
    "test_iid": (0.027, 0.353),
    "test_novel_pep": (0.043, 0.147),
}
COUNTS = {                           # molecules, peptides, pep_clusters
    "test_level4": (202, 108, 18),
    "test_iid": (201, 53, 39),
    "test_novel_pep": (1589, 463, 214),
}


@pytest.fixture(scope="module")
def cell():
    ev = json.loads((HERE / "fixtures" / "s42_eval_base_paired_ranks_20261008.json").read_text())
    units = {}
    with open(HERE / "fixtures" / "s42_as_paired_i_units.csv") as f:
        for row in csv.DictReader(f):
            units[(row["condition"], row["row_id"])] = (row["peptide"], row["pep_cluster"])
    out = {}
    for cond, m in ev["conditions"].items():
        ranks, peps, clus = [], [], []
        for rid, rank in m["ranks"].items():
            ranks.append(rank)
            p, k = units[(cond, rid)]
            peps.append(p)
            clus.append(k)
        out[cond] = (ranks, peps, clus, m)
    return ev["pool_size"], out


def test_counts_are_what_the_splits_say(cell):
    _, conds = cell
    for name, (nm, npep, nclus) in COUNTS.items():
        ranks, peps, clus, _ = conds[name]
        assert (len(ranks), len(set(peps)), len(set(clus))) == (nm, npep, nclus)


def test_cluster_bootstrap_reproduces_the_published_interval(cell):
    """Independent reimplementation of findings 9.6's corrected CI. Tolerance 0.02 absolute."""
    pool, conds = cell
    for name, (lo_pub, hi_pub) in PUBLISHED_CLUSTER_CI.items():
        ranks, peps, _, _ = conds[name]
        _, _, lo, hi, _, _ = R.cluster_bootstrap(ranks, peps, pool, "recall_at_10")
        assert abs(lo - lo_pub) < 0.02 and abs(hi - hi_pub) < 0.02, (name, lo, hi)


def test_per_query_bootstrap_is_three_to_seven_times_too_narrow(cell):
    """The error 9.6 retired. A ratio near 1.0 would mean clustering never mattered."""
    pool, conds = cell
    for name in COUNTS:
        ranks, peps, _, _ = conds[name]
        _, se_pep, _, _, _, _ = R.cluster_bootstrap(ranks, peps, pool, "recall_at_10")
        se_naive = R.naive_bootstrap_se(ranks, pool, "recall_at_10")
        assert 2.5 < se_pep / se_naive < 8.0, (name, se_pep / se_naive)


def test_peptide_is_not_the_last_rung_and_it_bites_only_on_level4(cell):
    """108 peptides in test_level4 collapse into 18 clusters, so the unit question is still open."""
    pool, conds = cell
    ranks, peps, clus, _ = conds["test_level4"]
    _, se_pep, _, _, _, _ = R.cluster_bootstrap(ranks, peps, pool, "recall_at_10")
    _, se_clu, _, _, _, _ = R.cluster_bootstrap(ranks, clus, pool, "recall_at_10")
    assert se_clu / se_pep > 1.35, se_clu / se_pep
    for name in ("test_iid", "test_novel_pep"):
        ranks, peps, clus, _ = conds[name]
        _, sp, _, _, _, _ = R.cluster_bootstrap(ranks, peps, pool, "recall_at_10")
        _, sc, _, _, _, _ = R.cluster_bootstrap(ranks, clus, pool, "recall_at_10")
        assert sc / sp < 1.3, (name, sc / sp)


def test_paired_mde_at_rho_zero_is_the_unpaired_case(cell):
    pool, conds = cell
    ranks, peps, _, _ = conds["test_level4"]
    _, se, _, _, _, _ = R.cluster_bootstrap(ranks, peps, pool, "recall_at_10")
    assert R.paired_mde(se, 0.0) == pytest.approx(R.POWER_Z * (2 ** 0.5) * se)


def test_paired_mde_falls_as_rho_rises():
    se = 0.1
    vals = [R.paired_mde(se, r) for r in (0.0, 0.5, 0.8, 0.9, 0.95)]
    assert vals == sorted(vals, reverse=True)
    with pytest.raises(ValueError):
        R.paired_mde(se, 1.0)


def test_lambda_hits_the_pre_registered_recall_delta(cell):
    pool, conds = cell
    ranks, peps, _, _ = conds["test_level4"]
    lam = R.lambda_for_recall_delta(ranks, pool, PREREG_DELTA)
    got = R.effect_under_lambda(ranks, peps, pool, lam, "recall_at_10")
    assert got >= PREREG_DELTA and got < PREREG_DELTA + 0.02, (lam, got)


def test_effect_is_monotone_in_lambda(cell):
    pool, conds = cell
    ranks, peps, _, _ = conds["test_novel_pep"]
    effs = [R.effect_under_lambda(ranks, peps, pool, l, "norm_rank") for l in (1.0, 1.2, 1.5, 2.0)]
    assert effs == sorted(effs)


def test_level4_with_recall_at_10_cannot_see_the_pre_registered_effect(cell):
    """The headline. The condition the gate currently keys on, with the metric it reports,
    cannot resolve +0.05 even at an implausibly generous rho of 0.95."""
    pool, conds = cell
    ranks, peps, _, _ = conds["test_level4"]
    _, se, _, _, _, _ = R.cluster_bootstrap(ranks, peps, pool, "recall_at_10")
    lam = R.lambda_for_recall_delta(ranks, pool, PREREG_DELTA)
    eff = R.effect_under_lambda(ranks, peps, pool, lam, "recall_at_10")
    assert eff < R.paired_mde(se, 0.95), (eff, R.paired_mde(se, 0.95))


def test_novel_pep_with_norm_rank_can(cell):
    """Same eval pass, no extra GPU: the readable condition is already in hand."""
    pool, conds = cell
    ranks, peps, _, _ = conds["test_novel_pep"]
    lvl_ranks, _, _, _ = conds["test_level4"]
    lam = R.lambda_for_recall_delta(lvl_ranks, pool, PREREG_DELTA)   # ONE common lambda
    _, se, _, _, _, _ = R.cluster_bootstrap(ranks, peps, pool, "norm_rank")
    eff = R.effect_under_lambda(ranks, peps, pool, lam, "norm_rank")
    assert eff >= R.paired_mde(se, 0.8), (eff, R.paired_mde(se, 0.8))


def test_unreachable_delta_is_reported_not_crashed(cell):
    pool, conds = cell
    ranks, peps, _, _ = conds["test_level4"]
    out = R.readability(ranks, peps, pool, 0.90, n_boot=200)
    assert out["verdict"] == "UNREACHABLE" and out["lambda"] is None
