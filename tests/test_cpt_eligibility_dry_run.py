"""Step-5 dry run: the old/new comparison must count what it claims, on a corpus small enough to
check by hand."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("duckdb", reason="the dry run compares the SQL filter; duckdb required")
pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

from scripts.data_processing.cpt_eligibility_dry_run import footer_scan, run  # noqa: E402


def chain(length, salt):
    motif = (salt + "GAVLIPFMWSTCYNQDEKRH")[:20]
    return (motif * (length // 20 + 1))[:length]


LONG_A, SHORT_A, DRA, DRB, UNKNOWN = chain(365, "A"), chain(275, "N"), chain(254, "D"), chain(266, "R"), chain(300, "K")

# (subset_key, mhc_one, mhc_one_allele, mhc_two, mhc_two_allele, copies)
ROWS = [
    ("trb_mhc_one", LONG_A, "HLA-A*02:01", None, None, 5),             # legacy fail (F5) -> new pass
    ("trb_mhc_one", SHORT_A, "HLA-A*11:01", None, None, 3),            # pass -> pass
    ("trb_mhc_one", UNKNOWN, "HLA-A*02:01", None, None, 2),            # legacy pass -> new fail
    ("trb_mhc_one", SHORT_A, "HLA-B*07:02", None, None, 4),            # pass -> fail, on the name only
    ("trb_mhc_one_mhc_two", DRA, "HLA-DRA*01:01", DRB, "HLA-DRB1*15:01", 6),   # pass -> pass
    ("trb_mhc_one_mhc_two", SHORT_A, "HLA-A*11:01", DRB, "HLA-DRB1*15:01", 1),  # pass -> fail, pairing
    ("trb", None, None, None, None, 7),                                 # receptor-only, untouched
]


@pytest.fixture
def corpus(tmp_path):
    src = tmp_path / "src"
    by_subset = {}
    for subset, m1, a1, m2, a2, n in ROWS:
        by_subset.setdefault(subset, []).extend([(m1, a1, m2, a2)] * n)
    for subset, rows in by_subset.items():
        d = src / f"subset_key={subset}" / f"order_key={subset}"
        d.mkdir(parents=True)
        cols = list(zip(*rows))
        pq.write_table(pa.table({k: pa.array(v, type=pa.string()) for k, v in zip(
            ("mhc_one", "mhc_one_allele", "mhc_two", "mhc_two_allele"), cols)}), d / "data_0.parquet")
    fasta = tmp_path / "mini_prot.fasta"
    fasta.write_text("".join(f">{h}\n{s}\n" for h, s in [
        ("HLA:1 A*02:01:01:01 365 bp", LONG_A), ("HLA:2 A*11:01:01:01 275 bp", SHORT_A),
        ("HLA:3 DRA*01:01:01:01 254 bp", DRA), ("HLA:4 DRB1*15:01:01:01 266 bp", DRB)]))
    return src, fasta, tmp_path / "out"


def test_dry_run_counts_every_transition(corpus):
    src, fasta, out = corpus
    s = run(src, fasta, out)
    assert s["rows_scanned"] == 28 and s["rows_with_any_mhc_chain"] == 21
    t = {k: v["rows"] for k, v in s["transition_legacy_to_new"].items()}
    assert t == {"False | True": 5, "True | True": 3 + 6 + 7, "True | False": 2 + 4 + 1}
    assert s["legacy_length_only_rejections"]["rows"] == 5
    assert s["gained_legacy_fail_new_pass"]["rows"] == 5
    assert {k: v["rows"] for k, v in s["lost_legacy_pass_new_fail"].items()} == {
        "mhc_one:reference_absent": 2, "mhc_one:name_locus_conflict": 4, "pair_mismatch": 1}
    assert {k: v["rows"] for k, v in s["rejected_on_allele_name_only"].items()} == {
        "mhc_one:name_locus_conflict": 4}
    assert s["new_accepted_mhc_one_over_320"] == {"I | A": {"rows": 5, "combinations": 1}}
    assert s["by_subset_key"]["trb"] == {"rows": 7, "legacy_pass_rows": 7, "new_pass_rows": 7}
    assert s["sql_python_disagreements"] == 0
    assert s["acceptance"]["sql_and_python_agree_on_every_combination"] is True
    assert (out / "combinations.csv").exists() and (out / "summary.json").exists()
    assert LONG_A not in (out / "combinations.csv").read_text()   # hashes, never sequences


def test_dry_run_refuses_to_overwrite(corpus):
    src, fasta, out = corpus
    run(src, fasta, out)
    with pytest.raises(FileExistsError):
        run(src, fasta, out)


def test_footer_scan_reports_rows_and_non_null_chains(corpus):
    src, _fasta, _out = corpus
    got = footer_scan(str(src))
    assert got["trb"]["rows"] == 7 and got["trb"].get("mhc_one_non_null", 0) == 0
    assert got["trb_mhc_one"]["rows"] == 14 and got["trb_mhc_one"]["mhc_one_non_null"] == 14
    assert got["trb_mhc_one_mhc_two"]["mhc_two_non_null"] == 7
