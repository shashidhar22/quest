"""The audit of a built corpus (corpus repair step 11): leakage against the benchmark and coverage,
both from the built rows themselves."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.data_processing.cpt_built_audit import (  # noqa: E402
    audit,
    deletions,
    edit_distance_at_most_one,
    near_heldout,
    normalize,
)

MHC_A = "GSHSMRYF"
KNOWN = {MHC_A: ("I", True)}
classify = lambda seq: KNOWN.get(seq, (None, None))
BENCH = {"train": [("CAVT", "CASST", "TRAINPEPK")],
         "val": [("CAVV", "CASSV", "VALPEPTIDE")],
         "test_novel_pep": [("CAVH", "CASSH", "HELDOUTPEP"), ("CAVH", "CASSH2", "HELDOUTPEP")]}
PARTS = ["val", "test_novel_pep"]
CLEAN = [
    ("train", "CAVR<eos>CASSL<eos>GILGFVFTL<eos>" + MHC_A, "tra_trb_peptide_mhc_one"),
    ("train", "TRAINPEPK<eos>CASST", "peptide_trb"),          # a benchmark TRAIN peptide is allowed
    ("train", "CASSH", "trb"),                                 # a held-out receptor alone is allowed
    ("validation", "NLVPMVATV", "peptide"),
    ("test", "CAVQ", "tra"),
]


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def test_edit_distance_check_agrees_with_levenshtein():
    rng = random.Random(3)
    words = ["".join(rng.choice("ACDE") for _ in range(rng.randint(1, 6))) for _ in range(400)]
    for a in words[:60]:
        for b in words:
            assert edit_distance_at_most_one(a, b) == (levenshtein(a, b) <= 1), (a, b)


def test_the_deletion_index_finds_every_near_peptide_and_nothing_else():
    rng = random.Random(5)
    heldout = {"".join(rng.choice("ACDEFG") for _ in range(rng.randint(4, 7))) for _ in range(40)}
    built = {"".join(rng.choice("ACDEFG") for _ in range(rng.randint(3, 8))) for _ in range(3000)} | set(list(heldout)[:5])
    got = near_heldout(built, heldout)
    brute = {p for p in built if any(levenshtein(p, h) <= 1 for h in heldout)}
    assert set(got) == brute and len(brute) > 20
    assert all(levenshtein(p, h) <= 1 and h in heldout for p, h in got.items())
    assert deletions("ABC") == {"ABC", "BC", "AC", "AB"}


def test_a_clean_corpus_passes_and_is_counted():
    report, lines = audit(CLEAN, BENCH, PARTS, classify)
    assert report["status"] == "PASS"
    assert report["rows_checked"] == 5 and report["peptide_bearing_rows"] == 3
    assert report["minimum_edit_distance_to_a_heldout_peptide"] == "at least 2"
    assert report["built_peptides_also_in_benchmark_train"] == 1
    assert all(v == 0 for part in report["per_heldout_part"].values() for v in part.values())
    by = {(l["split"], l["stratum"]): l for l in lines}
    s1 = by[("train", "s1_paired_tcr_peptide_classI")]
    assert (s1["rows"], s1["distinct_examples"], s1["distinct_classical_class_i_heavy_chains"]) == (1, 1, 1)
    assert by[("train", "background_single_beta_chain")]["distinct_trb_cdr3"] == 1
    assert by[("test", "background_single_alpha_chain")]["rows"] == 1


@pytest.mark.parametrize("row, key, distance", [
    (("train", "HELDOUTPEP", "peptide"), "rows_with_a_heldout_peptide", 0),
    (("train", "heldoutpep<eos>CASSZ", "peptide_trb"), "rows_with_a_heldout_peptide", 0),         # case
    (("validation", "HELDOUTPEP<eos>CASSH", "peptide_trb"), "rows_with_a_heldout_trb_peptide_association", 0),
    (("train", "CAVH<eos>CASSH<eos>HELDOUTPEP", "tra_trb_peptide"), "rows_with_a_heldout_tra_trb_peptide_association", 0),
    (("train", "HELDOUTPEA", "peptide"), None, 1),            # one substitution
    (("train", "HELDOUTPE", "peptide"), None, 1),             # one deletion
    (("test", "HELDOUTPEPK", "peptide"), None, 1),            # one insertion
])
def test_each_kind_of_leak_fails_the_audit(row, key, distance):
    report, _ = audit(CLEAN + [row], BENCH, PARTS, classify)
    assert report["status"] == "FAIL"
    assert report["minimum_edit_distance_to_a_heldout_peptide"] == distance
    if key:
        assert report["per_heldout_part"]["test_novel_pep"][key] == 1
        assert report["per_heldout_part"]["val"][key] == 0
    assert report["rows_within_one_edit_of_a_heldout_peptide"] == 1


def test_two_edits_away_is_not_a_leak():
    report, _ = audit(CLEAN + [("train", "HELDOUTAAP", "peptide")], BENCH, PARTS, classify)
    assert report["status"] == "PASS"


def test_an_audit_that_checked_nothing_does_not_pass():
    assert audit([], BENCH, PARTS, classify)[0]["status"] == "FAIL"
    assert audit([("train", "CASSL", "trb")], BENCH, PARTS, classify)[0]["status"] == "FAIL"    # no peptide rows


def test_normalization_is_letters_only_and_upper_case():
    assert normalize("gil gfv ftl") == "GILGFVFTL" and normalize(None) == ""


def test_an_audit_against_no_heldout_peptide_does_not_pass():
    empty = {**BENCH, "test_novel_pep": [("CAVH", "CASSH", None)]}
    report, _ = audit(CLEAN, empty, PARTS, classify)
    assert report["status"] == "FAIL" and report["heldout_parts_without_a_peptide"] == ["test_novel_pep"]
    assert audit(CLEAN, BENCH, [], classify)[0]["status"] == "FAIL"


def test_receptor_case_does_not_hide_an_association():
    bench = {**BENCH, "test_novel_pep": [("cavh", "cassh", "HELDOUTPEP")]}
    report, _ = audit(CLEAN + [("train", "CAVH<eos>CASSH<eos>HELDOUTPEP", "tra_trb_peptide")], bench, PARTS, classify)
    hit = report["per_heldout_part"]["test_novel_pep"]
    assert hit["rows_with_a_heldout_trb_peptide_association"] == 1
    assert hit["rows_with_a_heldout_tra_trb_peptide_association"] == 1


def test_benchmark_rows_sharing_an_id_all_count(tmp_path):
    import pandas as pd
    from scripts.data_processing.cpt_built_audit import load_benchmark
    pd.DataFrame({"row_id": [1, 1], "tra_cdr3": ["CAVA", "CAVA"], "trb_cdr3": ["CASSA", "CASSA"],
                  "peptide": ["FIRSTPEP", "SECONDPEP"]}).to_parquet(tmp_path / "t_val.parquet")
    molecules, _ = load_benchmark(tmp_path, "t", ["val"])
    assert {p for _a, _b, p in molecules["val"]} == {"FIRSTPEP", "SECONDPEP"}
