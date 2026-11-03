"""The judge's recall CI must resample peptides, not queries.

Queries sharing a true peptide are not independent draws for a peptide-retrieval metric. On the
real V03 base cell the per-query bootstrap was 3-4x too narrow, which manufactured a
"significantly below random" reading of test_novel_pep (0.081, per-query CI [0.068, 0.094],
peptide-clustered CI [0.043, 0.147] — straddling random 0.10). See quest-cpt/findings.md 9.6.
"""
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from track_a.esm_sft.judge import cluster_bootstrap_ci  # noqa: E402


def test_all_singletons_matches_a_plain_bootstrap():
    """With one query per peptide the clustered CI is the per-query CI, so nothing is lost."""
    rng = np.random.default_rng(0)
    hits = (rng.random(400) < 0.3).astype(float)
    peps = np.arange(400).astype(str)             # every peptide distinct
    lo, hi, k = cluster_bootstrap_ci(hits, peps, n_boot=3000, seed=1)
    assert k == 400
    se = np.sqrt(hits.mean() * (1 - hits.mean()) / len(hits))
    assert (hi - lo) == pytest.approx(2 * 1.96 * se, rel=0.25)


def test_heavy_clustering_widens_the_interval():
    """The V03 test_iid shape: most queries on a handful of peptides."""
    hits = np.concatenate([np.zeros(90), np.ones(10), np.zeros(101)])
    heavy = np.array(["A"] * 90 + ["B"] * 10 + [f"p{i}" for i in range(101)])
    singles = np.array([f"q{i}" for i in range(201)])
    w_cluster = np.subtract(*cluster_bootstrap_ci(hits, heavy, n_boot=3000, seed=1)[1::-1])
    w_single = np.subtract(*cluster_bootstrap_ci(hits, singles, n_boot=3000, seed=1)[1::-1])
    assert w_cluster > 2 * w_single, (w_cluster, w_single)


def test_cluster_count_is_reported():
    """n_peptides is the number the power floor is judged on."""
    peps = np.array(["A", "A", "A", "B", "C", "C"])
    _, _, k = cluster_bootstrap_ci(np.ones(6), peps, n_boot=200)
    assert k == 3


def test_a_single_peptide_cannot_resolve_anything():
    """One peptide means one independent observation, so the CI must span the whole outcome."""
    hits = np.array([1.0, 1.0, 0.0, 1.0])
    lo, hi, k = cluster_bootstrap_ci(hits, np.array(["A"] * 4), n_boot=500)
    assert k == 1 and lo == hi == pytest.approx(hits.mean())


def test_empty_input_is_safe():
    assert cluster_bootstrap_ci([], []) == (0.0, 0.0, 0)


def test_deterministic_for_a_seed():
    rng = np.random.default_rng(3)
    hits = (rng.random(120) < 0.4).astype(float)
    peps = np.repeat(np.arange(30), 4).astype(str)
    assert cluster_bootstrap_ci(hits, peps, seed=7) == cluster_bootstrap_ci(hits, peps, seed=7)


REAL_EVAL = ROOT / "docs" / "cpt_stage4.2" / "wave1_v03" / "base_paired" / "eval.json"
TRUE_PEPS = pathlib.Path(__file__).parent / "fixtures" / "s42_as_paired_i_true_peptides.csv"

# The real V03 base cell, recomputed 2026-10-15. Per-query CIs are what shipped; the clustered
# ones are correct. Outcomes are strongly correlated within a peptide (a peptide tends to be
# easy or hard for every TCR paired with it), which is what inflates the interval.
EXPECTED = {
    #                 recall  per-query CI        clustered CI (what this code must produce)
    "test_level4":    (0.337, (0.267, 0.406), (0.17, 0.64)),
    "test_iid":       (0.144, (0.095, 0.194), (0.02, 0.36)),
    "test_novel_pep": (0.081, (0.068, 0.094), (0.04, 0.15)),
}


def _real(cond):
    import json
    import pandas as pd
    ranks = json.loads(REAL_EVAL.read_text())["conditions"][cond]["ranks"]
    tp = pd.read_csv(TRUE_PEPS, dtype=str)
    tp = tp[tp.condition == cond]
    tp["rank"] = tp.row_id.map(ranks)
    tp = tp[tp["rank"].notna()]
    return (tp["rank"].astype(int).to_numpy() <= 10).astype(float), tp.peptide.to_numpy()


@pytest.mark.parametrize("cond", list(EXPECTED))
def test_clustered_ci_on_the_real_v03_cell(cond):
    recall, per_query, clustered = EXPECTED[cond]
    hits, peps = _real(cond)
    assert hits.mean() == pytest.approx(recall, abs=0.002), "fixture no longer matches eval.json"
    lo, hi, k = cluster_bootstrap_ci(hits, peps, n_boot=4000, seed=42)
    assert (lo, hi) == pytest.approx(clustered, abs=0.03), f"{cond}: got [{lo:.3f}, {hi:.3f}]"
    assert (hi - lo) > 2.5 * (per_query[1] - per_query[0]), \
        f"{cond}: clustering must widen the interval substantially"


def test_novel_pep_is_not_below_random_once_clustered():
    """The retraction in findings 9.6. Per-query CI said [0.068, 0.094], excluding random 0.10."""
    hits, peps = _real("test_novel_pep")
    lo, hi, _ = cluster_bootstrap_ci(hits, peps, n_boot=4000, seed=42)
    assert hi > 0.10, f"clustered CI [{lo:.3f}, {hi:.3f}] must reach random, not sit below it"


def test_level4_still_clears_random_once_clustered():
    """The gate fix depends on this: test_level4 must survive the correct interval."""
    hits, peps = _real("test_level4")
    lo, hi, _ = cluster_bootstrap_ci(hits, peps, n_boot=4000, seed=42)
    assert lo > 0.10, f"clustered CI [{lo:.3f}, {hi:.3f}] must stay clear of random"


def test_outcomes_really_are_clustered_within_a_peptide():
    """Why the inflation happens: recall is near-binary per peptide, not independent per query."""
    import pandas as pd
    hits, peps = _real("test_level4")
    g = pd.DataFrame({"h": hits, "p": peps}).groupby("p").h.mean()
    extreme = ((g == 0.0) | (g == 1.0)).mean()
    assert extreme > 0.8, f"only {extreme:.0%} of peptides are all-hit or all-miss"
