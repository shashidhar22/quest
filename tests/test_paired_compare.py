"""The paired comparator must measure the LOCKED endpoint, on the right unit.

What `judge.py compare` did instead (audit finding 4, 2026-11-04): it intersected row
IDs silently, took differences of Recall@10 indicators, and bootstrapped over ROWS. No
peptide clustering, no normalized rank, no rho, no pool or draw identity check, no equal
query sets. The gate locks `test_novel_pep / peptide / norm_rank`; the comparator computed
a different quantity on the wrong resampling unit, so its interval was too narrow by
construction. The launcher never called it at all.

The auditor's demonstration is the first test below: 100 molecules in two peptides, every
molecule of one peptide improving and none of the other. The row bootstrap calls that a
resolved +0.50; resampling peptides has to admit the draw where only the flat peptide is
drawn, so the interval must reach zero.

Stdlib only, like readability.py, so the whole statistical contract is testable with no
ML stack present.
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BOX = ROOT / "scripts" / "training" / "stage42_box"
sys.path.insert(0, str(BOX))

import paired_compare as PC  # noqa: E402

POOL = 100


def prov(ckpt="cccccccccccccccc", adapter="none", pool_digest="aaaaaaaaaaaaaaaa", **over):
    """The block a real evaluation records about what it read (audit finding 1, 2026-11-06)."""
    lineage = ({"cpt_adapter": "none", "cpt_adapter_digest": "none", "cpt_objective": "none",
                "cpt_run_id": "none", "cpt_steps": None, "cpt_seed": None,
                "cpt_token_budget": None} if adapter == "none" else
               {"cpt_adapter": adapter, "cpt_adapter_digest": "3afeaf40bc860fef",
                "cpt_objective": "fixed-batch-mean/v0", "cpt_run_id": "cpt_stage41_pointD_run05",
                "cpt_steps": 7800, "cpt_seed": 42, "cpt_token_budget": 0})
    d = {"pool_digest": pool_digest, "pool_path": "/x/pool.tsv",
         "split_digests": {"test_novel_pep": "bbbbbbbbbbbbbbbb"},
         "ckpt_digest": ckpt, "ckpt_path": f"/x/{ckpt}.pt", "device": "cuda",
         "code_commit": "abcdef123456", "pool_size": POOL,
         "sft_objective": "bce-pair-logits/v1", "cotenants": "cuda_visible_devices=0",
         "sft_seed": 42, "sft_max_len": 1024,
         "sft_recipe": "lr=0.0001;bs=16;epochs=2;train_cap=50000;val_cap=5000", **lineage}
    d.update(over)
    return d


def ev(conditions, pool=POOL, draw="length_matched", task="as_paired_i",
       tcr_variant="cdr3", mhc_variant="full", provenance=None, omit_provenance=False,
       max_len=1024):
    """An eval document in the judge's shape."""
    d = {
        "pool_size": pool,
        "args": {"distractor_draw": draw, "task": task, "max_len": max_len,
                 "tcr_variant": tcr_variant, "mhc_variant": mhc_variant},
        "conditions": {c: {"ranks": r} for c, r in conditions.items()},
    }
    if not omit_provenance:
        d["provenance"] = provenance if provenance is not None else prov()
    return d


def units_for(cond, mapping):
    """{(condition, row_id): (peptide, pep_cluster)} from {row_id: (peptide, cluster)}."""
    return {(cond, rid): v for rid, v in mapping.items()}


# ----------------------------------------------------------------- the auditor's case
def two_peptides_one_improving():
    """50 molecules of P1 improve by 40 ranks; 50 of P2 do not move."""
    a, b, u = {}, {}, {}
    for i in range(50):
        rid = f"p1_{i}"
        a[rid], b[rid] = 60, 5          # crosses into the top ten
        u[rid] = ("P1", "C1")
    for i in range(50):
        rid = f"p2_{i}"
        a[rid] = b[rid] = 60            # unchanged
        u[rid] = ("P2", "C2")
    return a, b, u


def test_one_improving_peptide_cannot_resolve_an_effect():
    """The defect the audit demonstrated: two peptides is two observations, not a hundred."""
    a, b, u = two_peptides_one_improving()
    r = PC.compare_condition("test_novel_pep", a, b, units_for("test_novel_pep", u),
                             POOL, metric="recall_at_10", unit="peptide")
    assert r["n_molecules"] == 100
    assert r["n_units"] == 2
    assert r["point"] == pytest.approx(0.50, abs=1e-9)
    assert r["ci_lo"] <= 0.0 <= r["ci_hi"], (
        f"resampling 2 peptides gave CI [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}], which excludes "
        "zero although only one peptide moved"
    )
    assert r["verdict"] == "no difference resolved"


def test_the_row_bootstrap_is_the_narrower_wrong_answer():
    """Kept measurable rather than asserted, as readability.py does for the marginal CI."""
    a, b, u = two_peptides_one_improving()
    clustered = PC.compare_condition("c", a, b, units_for("c", u), POOL,
                                     metric="recall_at_10", unit="peptide")
    naive = PC.naive_row_se(a, b, POOL, metric="recall_at_10")
    assert naive < clustered["se"] / 3, (
        f"row SE {naive:.4f} should be far below the peptide SE {clustered['se']:.4f}"
    )


# ----------------------------------------------------------------- it still detects real effects
def many_peptides_all_improving(n_pep=80, per_pep=3, gain=30):
    a, b, u = {}, {}, {}
    for p in range(n_pep):
        for m in range(per_pep):
            rid = f"pep{p}_{m}"
            a[rid] = 50 + (p % 7)
            b[rid] = max(1, a[rid] - gain)
            u[rid] = (f"P{p}", f"C{p // 4}")
    return a, b, u


def test_a_consistent_effect_across_many_peptides_is_resolved():
    a, b, u = many_peptides_all_improving()
    r = PC.compare_condition("test_novel_pep", a, b, units_for("test_novel_pep", u),
                             POOL, metric="norm_rank", unit="peptide")
    assert r["n_units"] == 80
    assert r["point"] > 0
    assert r["ci_lo"] > 0, f"a uniform improvement over 80 peptides was not resolved: {r}"
    assert r["verdict"] == "B > A"


def test_no_effect_is_not_resolved():
    a, b, u = many_peptides_all_improving(gain=0)
    r = PC.compare_condition("c", a, b, units_for("c", u), POOL,
                             metric="norm_rank", unit="peptide")
    assert r["point"] == pytest.approx(0.0, abs=1e-9)
    assert r["ci_lo"] <= 0.0 <= r["ci_hi"]
    assert r["verdict"] == "no difference resolved"


def test_a_regression_is_reported_as_one():
    a, b, u = many_peptides_all_improving(gain=-30)
    r = PC.compare_condition("c", a, b, units_for("c", u), POOL,
                             metric="norm_rank", unit="peptide")
    assert r["ci_hi"] < 0
    assert r["verdict"] == "B < A"


# ----------------------------------------------------------------- cluster sensitivity and rho
def effect_shared_within_clusters(n_cluster=20, pep_per_cluster=4, per_pep=3):
    """Homologous peptides respond alike: the gain is a property of the cluster.

    This is the case the sensitivity analysis exists for. Half the clusters improve by 30
    ranks, half not at all, so a peptide-level read sees 80 units whose effects are not
    independent.
    """
    a, b, u = {}, {}, {}
    for c in range(n_cluster):
        gain = 30 if c % 2 == 0 else 0
        for p in range(pep_per_cluster):
            pid = f"P{c}_{p}"
            for m in range(per_pep):
                rid = f"{pid}_{m}"
                a[rid] = 50 + (p % 3)
                b[rid] = max(1, a[rid] - gain)
                u[rid] = (pid, f"C{c}")
    return a, b, u


def test_cluster_sensitivity_widens_the_interval_when_effects_are_shared():
    """80 peptides in 20 clusters are 20 observations when the cluster drives the effect."""
    a, b, u = effect_shared_within_clusters()
    per_pep = PC.compare_condition("c", a, b, units_for("c", u), POOL,
                                   metric="norm_rank", unit="peptide")
    per_cl = PC.compare_condition("c", a, b, units_for("c", u), POOL,
                                  metric="norm_rank", unit="pep_cluster")
    assert per_pep["n_units"] == 80 and per_cl["n_units"] == 20
    assert per_cl["n_molecules"] == per_pep["n_molecules"]
    assert per_cl["point"] == pytest.approx(per_pep["point"], abs=1e-12), (
        "the estimand must not change with the resampling unit"
    )
    assert per_cl["se"] > per_pep["se"], (
        f"clustered SE {per_cl['se']:.4f} should exceed the peptide SE {per_pep['se']:.4f} "
        "when peptides within a cluster share their effect"
    )


def test_coarser_clustering_does_not_always_widen_the_interval():
    """Stated honestly: with no within-cluster correlation the two reads agree.

    Clustering is not a penalty applied for its own sake. It matters exactly when the
    coarser unit carries the dependence, which is why both are reported rather than one
    being declared conservative.
    """
    a, b, u = many_peptides_all_improving(gain=0)
    for rid in list(u):
        pep, _ = u[rid]
        u[rid] = (pep, "C_ALL")      # one cluster, no shared structure to find
    per_pep = PC.compare_condition("c", a, b, units_for("c", u), POOL,
                                   metric="norm_rank", unit="peptide")
    per_cl = PC.compare_condition("c", a, b, units_for("c", u), POOL,
                                  metric="norm_rank", unit="pep_cluster")
    assert per_cl["n_units"] == 1
    assert per_cl["point"] == pytest.approx(per_pep["point"], abs=1e-12)


def test_measured_rho_is_reported_and_bounded():
    """The pre-run analysis ASSUMED 0.8; the comparison has to measure it."""
    a, b, u = many_peptides_all_improving()
    r = PC.compare_condition("c", a, b, units_for("c", u), POOL,
                             metric="norm_rank", unit="peptide")
    assert r["measured_rho"] is not None
    assert -1.0 - 1e-9 <= r["measured_rho"] <= 1.0 + 1e-9


def test_rho_is_none_when_it_cannot_be_estimated():
    """One unit gives no cross-arm correlation; reporting 0 or 1 would both be lies."""
    a = {"x_0": 40, "x_1": 41}
    b = {"x_0": 20, "x_1": 21}
    u = units_for("c", {"x_0": ("P", "C"), "x_1": ("P", "C")})
    r = PC.compare_condition("c", a, b, u, POOL, metric="norm_rank", unit="peptide")
    assert r["n_units"] == 1
    assert r["measured_rho"] is None


# ----------------------------------------------------------------- validation: fail, never guess
def test_mismatched_query_sets_are_refused():
    a, b, u = many_peptides_all_improving(n_pep=5)
    del b[next(iter(b))]
    with pytest.raises(PC.ComparisonError, match="pairing"):
        PC.compare_condition("c", a, b, units_for("c", u), POOL,
                             metric="norm_rank", unit="peptide")


def test_mismatched_query_sets_may_be_accounted_for_explicitly():
    a, b, u = many_peptides_all_improving(n_pep=40)
    dropped = next(iter(b))
    del b[dropped]
    r = PC.compare_condition("c", a, b, units_for("c", u), POOL, metric="norm_rank",
                             unit="peptide", allow_partial_pairing=True)
    assert r["n_only_in_a"] == 1 and r["n_only_in_b"] == 0
    assert r["n_molecules"] == len(a) - 1


def test_ranks_outside_the_pool_are_refused():
    a, b, u = many_peptides_all_improving(n_pep=5)
    b[next(iter(b))] = POOL + 7
    with pytest.raises(PC.ComparisonError, match="rank"):
        PC.compare_condition("c", a, b, units_for("c", u), POOL,
                             metric="norm_rank", unit="peptide")


def test_unmapped_rows_are_refused():
    a, b, u = many_peptides_all_improving(n_pep=5)
    u.pop(next(iter(u)))
    with pytest.raises(PC.ComparisonError, match="unit map|coverage"):
        PC.compare_condition("c", a, b, units_for("c", u), POOL,
                             metric="norm_rank", unit="peptide")


def test_an_empty_comparison_is_refused():
    with pytest.raises(PC.ComparisonError):
        PC.compare_condition("c", {}, {}, {}, POOL, metric="norm_rank", unit="peptide")


# ----------------------------------------------------------------- metadata compatibility
def test_identical_metadata_is_compatible():
    assert PC.check_metadata(ev({}), ev({})) == []


@pytest.mark.parametrize("field,value", [
    ("pool", 50),
    ("draw", "uniform"),
    ("task", "as_trb_i"),
    ("tcr_variant", "cdr123"),
    ("mhc_variant", "pocket"),
])
def test_incompatible_metadata_is_named(field, value):
    bad = PC.check_metadata(ev({}), ev({}, **{field: value}))
    assert bad, f"{field} differing was not caught"
    assert any(field.replace("pool", "pool_size") in b for b in bad), bad


def test_metadata_absent_on_both_sides_is_refused():
    """The auditor's second probe line: `check_metadata({pool_size}, {pool_size}) == []`.

    Draw, task and serialization missing from both evals used to read as agreement, so an eval
    that never recorded its draw was comparable with any other. A uniform-draw number and a
    length-matched one are not comparable, and absence does not make them so.
    """
    bare = {"pool_size": POOL, "args": {}, "conditions": {}, "provenance": prov()}
    bad = PC.check_metadata(bare, dict(bare))
    assert bad, "two evals that recorded no metadata were accepted as comparable"
    for field in ("distractor_draw", "task", "tcr_variant", "mhc_variant"):
        assert any(field in b for b in bad), field


def test_an_eval_with_no_provenance_block_is_refused(tmp_path):
    """The comparator must apply the same validation as the launcher, not trust it to have."""
    a, b, u = many_peptides_all_improving(n_pep=12)
    pa = tmp_path / "a.json"
    pb = tmp_path / "b.json"
    pa.write_text(json.dumps(ev({"test_novel_pep": a}, omit_provenance=True)))
    pb.write_text(json.dumps(ev({"test_novel_pep": b}, provenance=prov(ckpt="dddddddddddddddd"))))
    up = write_units(tmp_path, "test_novel_pep", u)
    rc, res = PC.compare(str(pa), str(pb), str(up))
    assert rc == PC.EXIT_ERROR
    assert res is None


def test_a_changed_pool_between_arms_is_refused_by_the_comparator(tmp_path):
    a, b, u = many_peptides_all_improving(n_pep=12)
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(ev({"test_novel_pep": a})))
    pb.write_text(json.dumps(ev({"test_novel_pep": b},
                                provenance=prov(ckpt="dddddddddddddddd",
                                                pool_digest="ffffffffffffffff"))))
    up = write_units(tmp_path, "test_novel_pep", u)
    rc, res = PC.compare(str(pa), str(pb), str(up))
    assert rc == PC.EXIT_ERROR, "two arms scored against different pools were compared"
    assert res is None


def test_an_arm_compared_with_itself_is_refused_by_the_comparator(tmp_path):
    """Identical checkpoint and identical adapter: one model, so any effect is noise. This is
    the shape a reused stale evaluation takes."""
    a, b, u = many_peptides_all_improving(n_pep=12)
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(ev({"test_novel_pep": a})))
    pb.write_text(json.dumps(ev({"test_novel_pep": b})))      # same provenance = same model
    up = write_units(tmp_path, "test_novel_pep", u)
    rc, res = PC.compare(str(pa), str(pb), str(up))
    assert rc == PC.EXIT_ERROR
    assert res is None


def test_a_real_pair_reports_the_treatment_difference(tmp_path, capsys):
    a, b, u = many_peptides_all_improving(n_pep=12)
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(ev({"test_novel_pep": a})))
    pb.write_text(json.dumps(ev(
        {"test_novel_pep": b},
        provenance=prov(ckpt="dddddddddddddddd",
                        adapter="checkpoints/cpt_stage41_pointD_run05/final"))))
    up = write_units(tmp_path, "test_novel_pep", u)
    rc, res = PC.compare(str(pa), str(pb), str(up))
    assert rc == PC.EXIT_OK, capsys.readouterr().out
    out = capsys.readouterr().out
    assert "cpt_objective" in out and "fixed-batch-mean/v0" in out, (
        "the one difference the comparison is about must travel with the result")
    assert res["provenance"]["arm_b"]["cpt_run_id"] == "cpt_stage41_pointD_run05"


# ----------------------------------------------------------------- round two, finding 3
def _pair(tmp_path, a_doc, b_doc, u):
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a_doc))
    pb.write_text(json.dumps(b_doc))
    return pa, pb, write_units(tmp_path, "test_novel_pep", u)


def test_the_comparator_applies_the_same_validator_as_the_launcher(tmp_path):
    """The review's third finding. `check_eval_provenance` required five fields and treated ANY
    difference among the treatment fields -- including a run LABEL -- as evidence of two models,
    so two evals with no checkpoint or adapter hashes and empty split hashes were compared and a
    result was written. The launcher's manifest validator would have refused all three. One rule
    for both entry points, or the weaker one is the real contract.
    """
    a, b, u = many_peptides_all_improving(n_pep=12)
    weak = {"pool_digest": "pool", "split_digests": {}, "device": "cuda",
            "code_commit": "commit", "pool_size": POOL, "cpt_run_id": "one"}
    pa, pb, up = _pair(tmp_path,
                       ev({"test_novel_pep": a}, provenance=dict(weak)),
                       ev({"test_novel_pep": b}, provenance=dict(weak, cpt_run_id="two")), u)
    rc, res = PC.compare(str(pa), str(pb), str(up), n_boot=20)
    assert rc == PC.EXIT_ERROR, "the weak provenance path is still reachable"
    assert res is None


def test_the_comparator_requires_a_digest_for_every_condition_it_compares(tmp_path):
    """A split hash that covers a condition nobody compared proves nothing about the one that
    was compared."""
    a, b, u = many_peptides_all_improving(n_pep=12)
    other = {"test_iid": "cccccccccccccccc"}            # not the compared condition
    pa, pb, up = _pair(tmp_path,
                       ev({"test_novel_pep": a}, provenance=prov(split_digests=other)),
                       ev({"test_novel_pep": b},
                          provenance=prov(ckpt="dddddddddddddddd", split_digests=other)), u)
    rc, res = PC.compare(str(pa), str(pb), str(up), n_boot=20)
    assert rc == PC.EXIT_ERROR
    assert res is None


def test_a_different_eval_time_max_len_is_refused(tmp_path):
    """Finding 4's probe: max_len 1024 against 128 passed both checks. Two arms serialized to
    different lengths are not scoring the same inputs."""
    a, b, u = many_peptides_all_improving(n_pep=12)
    pa, pb, up = _pair(tmp_path,
                       ev({"test_novel_pep": a}, max_len=1024),
                       ev({"test_novel_pep": b}, max_len=128,
                          provenance=prov(ckpt="dddddddddddddddd")), u)
    rc, res = PC.compare(str(pa), str(pb), str(up), n_boot=20)
    assert rc == PC.EXIT_ERROR
    assert res is None
    assert "max_len" in PC.check_metadata(ev({}, max_len=1024), ev({}, max_len=128))[0]


def test_a_different_sft_recipe_is_refused(tmp_path):
    """The SFT recipe is fixed across arms by construction: only the CPT init differs."""
    a, b, u = many_peptides_all_improving(n_pep=12)
    pa, pb, up = _pair(tmp_path,
                       ev({"test_novel_pep": a}),
                       ev({"test_novel_pep": b},
                          provenance=prov(ckpt="dddddddddddddddd",
                                          sft_recipe="lr=0.001;bs=4;epochs=9")), u)
    rc, res = PC.compare(str(pa), str(pb), str(up), n_boot=20)
    assert rc == PC.EXIT_ERROR
    assert res is None


def test_an_unrecorded_sft_recipe_is_refused_rather_than_assumed(tmp_path):
    a, b, u = many_peptides_all_improving(n_pep=12)
    pa, pb, up = _pair(tmp_path,
                       ev({"test_novel_pep": a}, provenance=prov(sft_recipe=None)),
                       ev({"test_novel_pep": b},
                          provenance=prov(ckpt="dddddddddddddddd", sft_recipe=None)), u)
    rc, res = PC.compare(str(pa), str(pb), str(up), n_boot=20)
    assert rc == PC.EXIT_ERROR


# ----------------------------------------------------------------- the CLI contract
def write_pair(tmp_path, a, b, **kw):
    """Two arms, and TWO MODELS: arm B carries the arm-D adapter and its own checkpoint.

    Giving both sides the same provenance would now (rightly) be refused as one model compared
    with itself, so a CLI fixture has to describe a real pair.
    """
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(ev({"test_novel_pep": a}, **kw)))
    kw_b = dict(kw)
    kw_b.setdefault("provenance", prov(
        ckpt="dddddddddddddddd", adapter="checkpoints/cpt_stage41_pointD_run05/final"))
    pb.write_text(json.dumps(ev({"test_novel_pep": b}, **kw_b)))
    return pa, pb


def write_units(tmp_path, cond, mapping):
    p = tmp_path / "units.csv"
    lines = ["condition,row_id,peptide,pep_cluster"]
    lines += [f"{cond},{rid},{pep},{cl}" for rid, (pep, cl) in mapping.items()]
    p.write_text("\n".join(lines) + "\n")
    return p


def cli(*args):
    return subprocess.run([sys.executable, str(BOX / "paired_compare.py"), *map(str, args)],
                          capture_output=True, text=True, timeout=300)


def test_cli_produces_the_locked_read_and_exits_zero(tmp_path):
    a, b, u = many_peptides_all_improving()
    pa, pb = write_pair(tmp_path, a, b)
    pu = write_units(tmp_path, "test_novel_pep", u)
    r = cli("--a", pa, "--b", pb, "--units", pu, "--out", tmp_path / "cmp.json")
    assert r.returncode == PC.EXIT_OK, r.stdout + r.stderr
    out = r.stdout
    assert "PRIMARY" in out and "test_novel_pep" in out and "norm_rank" in out
    assert "peptide" in out
    res = json.loads((tmp_path / "cmp.json").read_text())
    assert res["primary"]["metric"] == "norm_rank"
    assert res["primary"]["unit"] == "peptide"
    assert res["primary"]["condition"] == "test_novel_pep"
    assert res["primary"]["ci_lo"] > 0


def test_cli_reports_secondary_metrics_as_secondary(tmp_path):
    """Decision 3: one primary inferential endpoint, the rest named secondary."""
    a, b, u = many_peptides_all_improving()
    pa, pb = write_pair(tmp_path, a, b)
    pu = write_units(tmp_path, "test_novel_pep", u)
    r = cli("--a", pa, "--b", pb, "--units", pu, "--out", tmp_path / "cmp.json")
    out = r.stdout
    assert "SECONDARY" in out
    assert "recall_at_10" in out and "mrr" in out
    res = json.loads((tmp_path / "cmp.json").read_text())
    assert "secondary" in res and res["secondary"], "secondary results were not recorded"


def test_cli_states_that_the_primary_does_not_establish_the_utility_target(tmp_path):
    """Decision 3: a normalized-rank effect is not the +0.05 Recall@10 claim."""
    a, b, u = many_peptides_all_improving()
    pa, pb = write_pair(tmp_path, a, b)
    pu = write_units(tmp_path, "test_novel_pep", u)
    out = cli("--a", pa, "--b", pb, "--units", pu, "--out", tmp_path / "c.json").stdout
    assert "does not establish" in out.lower()
    assert "0.05" in out or "+0.050" in out


def test_cli_reports_cluster_sensitivity(tmp_path):
    a, b, u = many_peptides_all_improving()
    pa, pb = write_pair(tmp_path, a, b)
    pu = write_units(tmp_path, "test_novel_pep", u)
    r = cli("--a", pa, "--b", pb, "--units", pu, "--out", tmp_path / "c.json")
    assert "pep_cluster" in r.stdout
    res = json.loads((tmp_path / "c.json").read_text())
    assert res["cluster_sensitivity"]["unit"] == "pep_cluster"


def test_cli_refuses_incompatible_metadata_with_the_error_code(tmp_path):
    a, b, u = many_peptides_all_improving(n_pep=10)
    pa = tmp_path / "a.json"
    pb = tmp_path / "b.json"
    pa.write_text(json.dumps(ev({"test_novel_pep": a}, draw="uniform")))
    pb.write_text(json.dumps(ev({"test_novel_pep": b}, draw="length_matched")))
    pu = write_units(tmp_path, "test_novel_pep", u)
    r = cli("--a", pa, "--b", pb, "--units", pu, "--out", tmp_path / "c.json")
    assert r.returncode == PC.EXIT_ERROR
    assert "draw" in (r.stdout + r.stderr)
    assert not (tmp_path / "c.json").exists(), "a refused comparison still wrote a result file"


def test_cli_refuses_a_missing_primary_condition(tmp_path):
    a, b, u = many_peptides_all_improving(n_pep=10)
    pa, pb = write_pair(tmp_path, a, b)
    pu = write_units(tmp_path, "test_novel_pep", u)
    r = cli("--a", pa, "--b", pb, "--units", pu, "--out", tmp_path / "c.json",
            "--primary-condition", "test_iid")
    assert r.returncode == PC.EXIT_ERROR
    assert "test_iid" in r.stdout + r.stderr


# ----------------------------------------------------------------- off-box by construction
def test_the_comparator_imports_with_no_ml_stack_present():
    code = textwrap.dedent(
        r"""
        import importlib.abc, sys
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                for bad in ("torch", "peft", "esm", "numpy", "pandas"):
                    if fullname == bad or fullname.startswith(bad + "."):
                        raise ImportError(bad + " is blocked for this test")
                return None
        sys.meta_path.insert(0, Block())
        sys.path.insert(0, "scripts/training/stage42_box")
        import paired_compare
        print("OK")
        """
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, (
        "the comparator pulled in a heavy dependency:\n" + r.stdout + r.stderr
    )


def test_an_unverified_training_data_gap_is_carried_into_the_result(tmp_path, capsys):
    """The legacy base checkpoint records no training-data digests. The comparison proceeds,
    and the gap travels with the result instead of being labelled a treatment difference."""
    a, b, u = many_peptides_all_improving(n_pep=12)
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(ev({"test_novel_pep": a})))            # no sft_train_digests
    pb.write_text(json.dumps(ev(
        {"test_novel_pep": b},
        provenance=prov(ckpt="dddddddddddddddd",
                        adapter="checkpoints/cpt_stage41_pointD_run05/final",
                        sft_train_digests={"train": "1" * 16, "val": "2" * 16}))))
    up = write_units(tmp_path, "test_novel_pep", u)
    rc, res = PC.compare(str(pa), str(pb), str(up), n_boot=20)
    assert rc == PC.EXIT_OK, capsys.readouterr().out
    out = capsys.readouterr().out
    assert "UNVERIFIED sft_train_digests" in out
    assert "TREATMENT DIFFERENCE (expected, and what the result is about): UNVERIFIED" not in out
    assert res["provenance"]["unverified"], "the gap must be in the written result too"
    assert not any("sft_train_digests" in x for x in res["provenance"]["treatment_differences"])
