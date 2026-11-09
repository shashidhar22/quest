"""Benchmark isolation of the CPT corpus (corpus repair step 6).

Why this file exists. The builder never consulted the benchmark, and run05's train split held the
exact receptor-peptide association for 103 of 1,589 locked-test molecules. Policy
`heldout-peptides-v2` drops every row carrying a held-out peptide or, at min_edit_distance=2, a
one-edit neighbour of one; these tests pin what that means, including what it must NOT remove.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from quest.data.benchmark_exclusion import (  # noqa: E402
    HELDOUT_PEPTIDES,
    build_exclusion,
    exclusion_list_info,
    load_excluded_peptides,
    one_edit_neighbours,
    peptide_of_output_row,
    write_exclusion,
)

TASK = "as_paired_i"


def write_splits(tmp_path, parts):
    """parts: {part: [(row_id, peptide), ...]}; each molecule is written twice, as order variants."""
    for part, rows in parts.items():
        frame = pd.DataFrame([{"row_id": r, "peptide": p, "order_key": o}
                              for r, p in rows for o in ("a", "b")])
        frame.to_parquet(tmp_path / f"{TASK}_{part}.parquet")
    return tmp_path


@pytest.fixture
def splits(tmp_path):
    return write_splits(tmp_path, {
        "train": [("t1", "GILGFVFTL"), ("t2", "NLVPMVATV")],
        "val": [("v1", "ELAGIGILTV"), ("v2", "ELAGIGILTV")],
        "test_novel_pep": [("n1", "KLGGALQAK"), ("n2", " elagigiltv ")],
        # one row_id carrying two peptides: both must be listed, never just the first
        "test_level4": [("d1", "AAAWYYY"), ("d1", "CCCWYYY"), ("d2", None), ("d3", "  ")],
        "test_novel_tcr": [("x1", "GILGFVFTL")],          # its peptide is a TRAIN peptide by design
    })


def test_list_counts_molecules_not_order_variants(splits):
    rows, meta = build_exclusion(splits, TASK, ["val", "test_novel_pep"], min_edit_distance=1)
    assert rows == [
        {"peptide": "ELAGIGILTV", "distance": 0, "heldout_peptide": "ELAGIGILTV",
         "parts": "test_novel_pep,val", "molecules": 3},
        {"peptide": "KLGGALQAK", "distance": 0, "heldout_peptide": "KLGGALQAK",
         "parts": "test_novel_pep", "molecules": 1},
    ]
    assert meta["policy"] == HELDOUT_PEPTIDES and meta["n_peptides"] == 2
    assert meta["min_edit_distance"] == 1 and meta["n_heldout_peptides"] == 2
    assert meta["heldout_parts"] == ["val", "test_novel_pep"]
    assert set(meta["input_sha256"]) == {f"{TASK}_{p}.parquet" for p in ("train", "val", "test_novel_pep")}


def test_refuses_a_part_whose_peptides_are_training_peptides(splits):
    """Excluding them would hide from CPT what fine-tuning is allowed to see. Such parts need
    pair-level exclusion; widening this one silently would be the wrong repair."""
    with pytest.raises(ValueError, match="also occur in as_paired_i train"):
        build_exclusion(splits, TASK, ["val", "test_novel_tcr"], min_edit_distance=1)


def test_train_is_never_a_heldout_part(splits):
    with pytest.raises(ValueError):
        build_exclusion(splits, TASK, ["train"], min_edit_distance=1)
    with pytest.raises(ValueError):
        build_exclusion(splits, TASK, [], min_edit_distance=1)
    with pytest.raises(ValueError):
        build_exclusion(splits, TASK, ["val"], min_edit_distance=3)   # cannot be enumerated


def test_list_round_trips_and_is_hashed(splits, tmp_path):
    rows, meta = build_exclusion(splits, TASK, ["val", "test_novel_pep"], min_edit_distance=1)
    out = tmp_path / "heldout.tsv"
    digest = write_exclusion(rows, meta, out)
    peptides, loaded_digest = load_excluded_peptides(out)
    assert peptides == {"ELAGIGILTV", "KLGGALQAK"} and loaded_digest == digest
    assert exclusion_list_info(out) == {"policy": HELDOUT_PEPTIDES, "min_edit_distance": 1}
    assert f'"list_sha256": "{digest}"' in (tmp_path / "heldout.tsv.meta.json").read_text()
    with pytest.raises(FileExistsError):
        write_exclusion(rows, meta, out)


HEAD = "#policy=heldout-peptides-v2;min_edit_distance=2\npeptide\tdistance\theldout_peptide\tparts\tmolecules\n"


@pytest.mark.parametrize("content", [
    "", HEAD,                                                   # empty, or no peptides
    "peptide\tdistance\theldout_peptide\tparts\tmolecules\nKLM\t0\tKLM\tval\t1\n",   # no policy line
    HEAD.replace("=2", "=5") + "KLM\t0\tKLM\tval\t1\n",        # a distance that was never built
    HEAD + "KLMVAL1\n",                                         # tabs lost
    HEAD + "KL M\t0\tKLM\tval\t1\n",                            # edited peptide
    HEAD + "klm\t0\tKLM\tval\t1\n",                             # not as written
    HEAD + "KLM\t7\tKLM\tval\t1\n",                             # impossible distance
    HEAD + "KLM\t0\tKLM\tval\t1\n\n",                           # truncated or padded
    "\ufeff" + HEAD + "KLM\t0\tKLM\tval\t1\n",                  # byte-order mark
])
def test_a_file_that_is_not_an_exclusion_list_is_refused(tmp_path, content):
    """An empty or mis-headed file must not load as 'exclude nothing'."""
    bad = tmp_path / "bad.tsv"
    bad.write_text(content)
    with pytest.raises(ValueError):
        load_excluded_peptides(bad)


@pytest.mark.parametrize("text,order,expected", [
    ("CASSF<eos>GILGFVFTL", "trb_peptide", "GILGFVFTL"),
    ("GILGFVFTL<eos>CAVR<eos>CASSF<eos>MHCSEQ", "peptide_tra_trb_mhc_one", "GILGFVFTL"),
    ("MHCA<eos>MHCB<eos>kLgGaLqAk", "mhc_one_mhc_two_peptide", "KLGGALQAK"),
    ("GILGFVFTL", "peptide", "GILGFVFTL"),
    ("CAVR<eos>CASSF", "tra_trb", None),
])
def test_peptide_is_read_from_a_serialized_row_by_its_order_key(text, order, expected):
    assert peptide_of_output_row(text, order) == expected


def test_component_count_mismatch_raises_rather_than_guessing():
    with pytest.raises(ValueError):
        peptide_of_output_row("CASSF<eos>GILGFVFTL<eos>EXTRA", "trb_peptide")


@pytest.mark.parametrize("order", ["trb_peptides", "trb_peptide_x", "peptide-trb", ""])
def test_unknown_order_key_raises_rather_than_being_read_loosely(order):
    with pytest.raises(ValueError):
        peptide_of_output_row("CASSF<eos>GILGFVFTL", order)


def test_one_row_id_with_two_peptides_lists_both(splits):
    rows, _ = build_exclusion(splits, TASK, ["test_level4"], min_edit_distance=1)
    assert [(r["peptide"], r["molecules"]) for r in rows] == [("AAAWYYY", 1), ("CCCWYYY", 1)]


# ---------- one-edit neighbours (min_edit_distance=2) ----------

def test_one_edit_neighbours_are_exactly_the_strings_at_distance_one():
    n = one_edit_neighbours("ACD")
    assert {"AC", "AD", "CD"} <= n                       # deletions
    assert {"ACE", "WCD", "AWD"} <= n                    # substitutions
    assert {"WACD", "ACDW", "ACWD"} <= n                 # insertions
    assert "ACD" not in n and "A" not in n and "ACDWW" not in n and "WWD" not in n
    assert len(n) == 3 + 3 * 25 + (4 * 26 - 3)           # dels + subs + distinct insertions


def test_neighbours_are_listed_with_the_heldout_peptide_they_are_near(splits):
    rows, meta = build_exclusion(splits, TASK, ["val", "test_novel_pep"], min_edit_distance=2)
    by = {r["peptide"]: r for r in rows}
    assert meta["n_heldout_peptides"] == 2 and meta["n_peptides"] == len(rows) > 2
    assert by["ELAGIGILTV"]["distance"] == 0
    for near in ("ELAGIGILTA", "ELAGIGILT", "ELAGIGILTVV", "WELAGIGILTV"):   # sub, del, two insertions
        assert by[near]["distance"] == 1 and by[near]["heldout_peptide"] == "ELAGIGILTV"
        assert by[near]["molecules"] == 0
    assert "ELAGIGILAA" not in by                        # two substitutions away: allowed to stay
    assert "GILGFVFTL" not in by                         # an unrelated training peptide


def test_refuses_when_a_training_peptide_is_one_edit_from_a_heldout_one(tmp_path):
    """The same rule as for exact matches: never hide from CPT what fine-tuning may see."""
    d = write_splits(tmp_path, {"train": [("t1", "GILGFVFTL")], "val": [("v1", "GILGFVFTV")]})
    build_exclusion(d, TASK, ["val"], min_edit_distance=1)            # exact-only is fine
    with pytest.raises(ValueError, match="one edit from a held-out peptide"):
        build_exclusion(d, TASK, ["val"], min_edit_distance=2)


def test_a_min_distance_two_list_removes_neighbours_through_the_sql_filter(splits, tmp_path):
    """End to end: build the list, load it, filter. A neighbour goes, a two-edit peptide stays."""
    duckdb_ = pytest.importorskip("duckdb")
    from scripts.data_processing.cpt_dataset_builder import LEGACY_POLICY as LEG, filter_where_sql as fws
    rows, meta = build_exclusion(splits, TASK, ["val", "test_novel_pep"], min_edit_distance=2)
    out = tmp_path / "list.tsv"
    write_exclusion(rows, meta, out)
    excluded, _ = load_excluded_peptides(out)
    con = duckdb_.connect()
    try:
        con.execute("CREATE TABLE t (tag VARCHAR, peptide VARCHAR, mhc_one VARCHAR, mhc_two VARCHAR, "
                    "mhc_one_allele VARCHAR, mhc_two_allele VARCHAR)")
        con.executemany("INSERT INTO t VALUES (?, ?, NULL, NULL, NULL, NULL)", [
            ("exact", "ELAGIGILTV"), ("substitution", "ELAGIGILTA"), ("deletion", "LAGIGILTV"),
            ("insertion", "ELAGIGGILTV"), ("two_edits", "ELAGIGILAA"), ("train", "GILGFVFTL"),
            ("no_peptide", None)])
        con.execute("CREATE TABLE x (excluded_peptide VARCHAR)")
        con.executemany("INSERT INTO x VALUES (?)", [(p,) for p in excluded])
        kept = {r[0] for r in con.execute(
            f"SELECT tag FROM t {fws(LEG, excluded_peptides_table='x')}").fetchall()}
    finally:
        con.close()
    assert kept == {"two_edits", "train", "no_peptide"}


@pytest.mark.parametrize("raw", ["ELAGIGILTV", " elagigiltv\t", "ELAG\u00a0IGILTV", "ELAGIGILTV\u200b",
                                 "ELAG\x0bIGILTV", "ELAG-IGILTV", "", "123"])
def test_sql_and_python_peptide_normalization_agree(raw):
    """Wider than chain normalization: any non-letter is removed, so a held-out peptide written
    with a non-breaking or zero-width space is still excluded. Pinned in both languages."""
    duckdb_ = pytest.importorskip("duckdb")
    from quest.data.benchmark_exclusion import normalize_peptide, normalize_peptide_sql
    con = duckdb_.connect()
    try:
        got = con.execute(f"SELECT {normalize_peptide_sql('?')}", [raw]).fetchone()[0]
    finally:
        con.close()
    assert got == normalize_peptide(raw)


# ---------- the SQL filter ----------

duckdb = pytest.importorskip("duckdb", reason="the isolation filter is SQL; duckdb required")

from scripts.data_processing.cpt_dataset_builder import (  # noqa: E402
    LEGACY_POLICY,
    filter_where_sql,
)

ROWS = [  # (tag, peptide, trb_cdr3)
    ("heldout_pair", "ELAGIGILTV", "CASSF"),
    ("heldout_peptide_other_receptor", "ELAGIGILTV", "CASSQ"),
    ("heldout_peptide_only", "ELAGIGILTV", None),
    ("heldout_spelled_differently", " elagigiltv\t", None),
    ("heldout_with_invisible_characters", "ELAG\u00a0IGIL\u200bTV", "CASSF"),
    ("train_peptide", "GILGFVFTL", "CASSF"),
    ("heldout_receptor_alone", None, "CASSF"),
    ("superstring_is_a_different_peptide", "ELAGIGILTVX", None),
]


def survivors(excluded, **kw):
    con = duckdb.connect()
    try:
        con.execute("CREATE TABLE t (tag VARCHAR, peptide VARCHAR, trb_cdr3 VARCHAR, mhc_one VARCHAR, "
                    "mhc_two VARCHAR, mhc_one_allele VARCHAR, mhc_two_allele VARCHAR)")
        con.executemany("INSERT INTO t VALUES (?, ?, ?, NULL, NULL, NULL, NULL)", ROWS)
        if excluded is not None:
            con.execute("CREATE TABLE x (excluded_peptide VARCHAR)")
            if excluded:
                con.executemany("INSERT INTO x VALUES (?)", [(p,) for p in excluded])
            kw["excluded_peptides_table"] = "x"
        where = filter_where_sql(LEGACY_POLICY, **kw)
        return {r[0] for r in con.execute(f"SELECT tag FROM t {where}").fetchall()}
    finally:
        con.close()


def test_every_row_carrying_a_heldout_peptide_is_dropped_in_every_projection():
    assert survivors(["ELAGIGILTV"]) == {
        "train_peptide", "heldout_receptor_alone", "superstring_is_a_different_peptide"}


def test_a_receptor_may_recur_alone_and_other_peptides_are_untouched():
    """Also the regression for a real defect: with the list column named `peptide`, the
    correlated subquery compared the list with itself and dropped EVERY peptide-bearing row."""
    kept = survivors(["ELAGIGILTV"])
    assert "heldout_receptor_alone" in kept and "train_peptide" in kept


def test_without_a_list_nothing_is_dropped_and_legacy_sql_is_unchanged():
    assert survivors(None) == {r[0] for r in ROWS}
    assert "excluded" not in filter_where_sql(LEGACY_POLICY)
    assert filter_where_sql(LEGACY_POLICY, excluded_peptides_table="x").startswith(
        filter_where_sql(LEGACY_POLICY))


def test_an_empty_exclusion_table_drops_nothing():
    assert survivors([]) == {r[0] for r in ROWS}
