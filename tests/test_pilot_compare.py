"""The step-12 pilot read-out: validation-only paired contrasts that refuse incomparable cells."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "training" / "pilot12_box"))
import pilot_compare as P  # noqa: E402

ARGS = {"distractor_draw": "length_matched", "task": "as_paired_i", "tcr_variant": "cdr3",
        "mhc_variant": "full", "max_len": 1024}
PROV = {k: f"{k}-value" for k in P.SAME}
ROWS = [("r1", "AAAAAAAAA", "c1"), ("r2", "CCCCCCCCC", "c1"), ("r3", "DDDDDDDDD", "c2"),
        ("r4", "EEEEEEEEE", "c2"), ("r5", "FFFFFFFFF", "c3"), ("r6", "GGGGGGGGG", "c3")]


def units():
    return {(P.CONDITION, r): (p, c) for r, p, c in ROWS}


def cell(ranks, *, prov=None, args=None, conditions=None, pool=100):
    cond = {P.CONDITION: {"n_queries": len(ranks), "n_peptides": 6, "recall_at_10": 0.5, "mrr": 0.3,
                          "n_truncated_seqs": 0, "ranks": dict(ranks),
                          "true_peptides": {r: p for r, p, _c in ROWS}}}
    for extra in conditions or ():
        cond[extra] = {"ranks": {"r1": 1}, "true_peptides": {"r1": "AAAAAAAAA"}}
    return {"args": {**ARGS, **(args or {})}, "pool_size": pool, "conditions": cond,
            "provenance": {**PROV, "cpt_adapter": "a", "cpt_objective": "o", **(prov or {})}}


def three_cells(shift=0):
    base = {r: 50 for r, _p, _c in ROWS}
    better = {r: max(1, 50 - 10 - shift) for r, _p, _c in ROWS}
    mid = {r: 45 for r, _p, _c in ROWS}
    return {"B": cell(base), "P": cell(mid), "E": cell(better)}


def test_a_clean_pilot_compares_every_contrast_metric_and_unit():
    out = P.compare(three_cells(), units(), ["E-B", "P-B", "E-P"], n_boot=200)
    assert set(out["contrasts"]) == {"E-B", "P-B", "E-P"}
    for results in out["contrasts"].values():
        assert set(results) == {f"{m}/{u}" for m in P.METRICS for u in ("peptide", "pep_cluster")}
        for r in results.values():
            assert r["condition"] == "val" and r["n_molecules"] == 6 and r["ci_lo"] <= r["point"] <= r["ci_hi"]
    # E ranks the true peptide higher than B, so norm_rank rises and the interval excludes zero.
    eb = out["contrasts"]["E-B"]["norm_rank/pep_cluster"]
    assert eb["point"] > 0 and eb["ci_lo"] > 0
    assert out["cells"]["E"]["n_queries"] == 6
    assert "in-sample" in out["reading"] and "cannot establish" in out["reading"]


def test_the_contrast_is_b_minus_a():
    out = P.compare(three_cells(), units(), ["E-B", "B-E"], n_boot=200)
    a = out["contrasts"]["E-B"]["norm_rank/peptide"]["point"]
    b = out["contrasts"]["B-E"]["norm_rank/peptide"]["point"]
    assert a == pytest.approx(-b) and a > 0


@pytest.mark.parametrize("name, break_it", [
    ("other candidate draw", lambda c: c["args"].update(distractor_draw="uniform")),
    ("other serialization length", lambda c: c["args"].update(max_len=128)),
    ("draw not recorded", lambda c: c["args"].pop("distractor_draw")),
    ("other code", lambda c: c["provenance"].update(code_commit="deadbeef")),
    ("other recipe", lambda c: c["provenance"].update(sft_recipe="lr=9")),
    ("other training data", lambda c: c["provenance"].update(sft_train_digests="x")),
    ("recipe unrecorded", lambda c: c["provenance"].update(sft_recipe="unrecorded")),
    ("provenance missing", lambda c: c.update(provenance={})),
    ("other queries", lambda c: c["conditions"]["val"]["true_peptides"].update(r1="MMMMMMMMM")),
    ("a test part was scored", lambda c: c["conditions"].update(test_novel_pep={"ranks": {}, "true_peptides": {}})),
    ("no validation condition", lambda c: c["conditions"].pop("val")),
])
def test_each_incomparability_refuses(name, break_it):
    cells = three_cells()
    break_it(cells["E"])
    with pytest.raises(P.PC.ComparisonError):
        P.compare(cells, units(), ["E-B"], n_boot=50)


def test_unpaired_queries_or_a_missing_unit_refuse():
    cells = three_cells()
    del cells["E"]["conditions"]["val"]["ranks"]["r6"]
    del cells["E"]["conditions"]["val"]["true_peptides"]["r6"]
    with pytest.raises(P.PC.ComparisonError, match="true peptides differ"):
        P.compare(cells, units(), ["E-B"], n_boot=50)
    cells = three_cells()
    u = units()
    del u[(P.CONDITION, "r3")]
    with pytest.raises(P.PC.ComparisonError, match="unit map coverage"):
        P.compare(cells, u, ["E-B"], n_boot=50)
    with pytest.raises(P.PC.ComparisonError, match="names a cell"):
        P.compare(three_cells(), units(), ["X-B"], n_boot=50)


def test_a_rank_outside_the_pool_refuses():
    cells = three_cells()
    cells["E"]["conditions"]["val"]["ranks"]["r1"] = 101
    with pytest.raises(P.PC.ComparisonError, match="rank outside"):
        P.compare(cells, units(), ["E-B"], n_boot=50)


def test_the_units_come_from_the_split_and_duplicates_refuse(tmp_path):
    import pandas as pd
    good = tmp_path / "val.parquet"
    pd.DataFrame({"row_id": ["r1", "r1", "r2"], "peptide": ["AA", "AA", "CC"],
                  "pep_cluster": ["c1", "c1", "c2"]}).to_parquet(good)      # repeated rows are fine
    assert P.units_from_split(good) == {("val", "r1"): ("AA", "c1"), ("val", "r2"): ("CC", "c2")}
    bad = tmp_path / "bad.parquet"
    pd.DataFrame({"row_id": ["r1", "r1"], "peptide": ["AA", "ZZ"], "pep_cluster": ["c1", "c1"]}).to_parquet(bad)
    with pytest.raises(P.PC.ComparisonError, match="more than one peptide"):
        P.units_from_split(bad)


def test_the_cli_writes_once_and_refuses_an_incomparable_pair(tmp_path):
    import pandas as pd
    val = tmp_path / "val.parquet"
    pd.DataFrame({"row_id": [r for r, _p, _c in ROWS], "peptide": [p for _r, p, _c in ROWS],
                  "pep_cluster": [c for _r, _p, c in ROWS]}).to_parquet(val)
    cells = three_cells()
    paths = {}
    for name, doc in cells.items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(doc))
    args = ["--cells", *[f"{n}={p}" for n, p in paths.items()], "--val-split", str(val),
            "--contrasts", "E-B", "P-B", "--n-boot", "200"]
    out = tmp_path / "pilot.json"
    assert P.main(args + ["--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert set(report["contrasts"]) == {"E-B", "P-B"} and report["cell_paths"]["E"] == str(paths["E"])
    with pytest.raises(SystemExit):                       # never overwrites
        P.main(args + ["--out", str(out)])
    doc = json.loads(paths["E"].read_text())
    doc["provenance"]["code_commit"] = "other"
    paths["E"].write_text(json.dumps(doc))
    assert P.main(args + ["--out", str(tmp_path / "second.json")]) == P.PC.EXIT_ERROR
    assert not (tmp_path / "second.json").exists()


def test_queries_recorded_on_neither_cell_are_not_agreement():
    cells = three_cells()
    for name in ("E", "B"):
        del cells[name]["conditions"]["val"]["true_peptides"]
    with pytest.raises(P.PC.ComparisonError, match="true peptides"):
        P.compare(cells, units(), ["E-B"], n_boot=50)


def test_a_cell_outside_every_contrast_is_still_held_to_the_closed_test_parts():
    cells = three_cells()
    cells["P"]["conditions"]["test_novel_pep"] = {"ranks": {}, "true_peptides": {}}
    with pytest.raises(P.PC.ComparisonError, match="test parts closed"):
        P.compare(cells, units(), ["E-B"], n_boot=50)


def test_no_positional_verdict_label_is_saved():
    """`compare_condition` says "B > A" by position; for the contrast E-B its "B" is cell E."""
    out = P.compare(three_cells(), units(), ["E-B"], n_boot=50)
    for r in out["contrasts"]["E-B"].values():
        assert "verdict" not in r and "point" in r
