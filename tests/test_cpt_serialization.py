"""Serialization of a CPT row (corpus repair step 7).

Two things are pinned here. The builder writes `input_text` in SQL and the trainer re-derives it
in Python (`project_canonical_row_to_run_input`); they must give the same bytes. And the SQL must
run on the installed DuckDB at all: it used `list_filter(..., x -> ...)`, which DuckDB 2.0 rejects,
and nothing had executed it off the box.
"""
import ast
import itertools
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

duckdb = pytest.importorskip("duckdb", reason="the builder's projection is SQL; duckdb required")

from scripts.data_processing.cpt_dataset_builder import (  # noqa: E402
    TOKEN_LIMIT,
    build_input_text_sql,
    token_count_sql,
)

TCR, MHC = ("cdr3", "cdr123", "full"), ("pocket", "pocket_contact", "full")
COLUMNS = ("tra_full", "trb_full", "peptide", "mhc_one", "mhc_two", "tra_cdr1", "tra_cdr2", "tra_cdr3",
           "trb_cdr1", "trb_cdr2", "trb_cdr3", "mhc_one_pocket", "mhc_one_pocket_contact",
           "mhc_two_pocket", "mhc_two_pocket_contact", "order_key")


def trainer_projection():
    """The trainer's functions, lifted by name: importing the module needs the training stack."""
    lib = ROOT / "scripts/training/cpt_training_lib.py"
    want = {"_chain_segments", "_mhc_segment", "_parse_order_key", "project_canonical_row_to_run_input"}
    const = {"EOS_LITERAL", "_VALID_TCR_VARIANTS", "_VALID_MHC_VARIANTS"}
    keep = [n for n in ast.parse(lib.read_text()).body
            if (isinstance(n, ast.FunctionDef) and n.name in want)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in const for t in n.targets))]
    assert {n.name for n in keep if isinstance(n, ast.FunctionDef)} == want
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(lib), "exec"), ns)
    return ns["project_canonical_row_to_run_input"]


def rows():
    full = {"tra_full": "AAAATRAFULL", "trb_full": "BBBBTRBFULL", "peptide": "GILGFVFTL",
            "mhc_one": "M" * 365, "mhc_two": "N" * 266, "tra_cdr1": "TA", "tra_cdr2": "TB", "tra_cdr3": "CAVR",
            "trb_cdr1": "SA", "trb_cdr2": "SB", "trb_cdr3": "CASSF", "mhc_one_pocket": "YFAM",
            "mhc_one_pocket_contact": "YFAMYQ", "mhc_two_pocket": "HDLE", "mhc_two_pocket_contact": "HDLEWK"}
    out = []
    tokens = ["tra", "trb", "peptide", "mhc_one", "mhc_two"]
    for n in range(1, 6):
        for perm in itertools.permutations(tokens, n):
            out.append({**full, "order_key": "_".join(perm)})
    # Missing and empty components: both languages must drop them the same way.
    for blank in ({"tra_cdr1": None}, {"trb_cdr2": ""}, {"peptide": ""}, {"mhc_two": None, "mhc_two_pocket": None},
                  {"tra_cdr1": None, "tra_cdr2": None, "tra_cdr3": None, "tra_full": None}):
        out.append({**full, **blank, "order_key": "tra_trb_peptide_mhc_one_mhc_two"})
    return out


@pytest.mark.parametrize("tcr", TCR)
@pytest.mark.parametrize("mhc", MHC)
def test_builder_sql_and_trainer_python_serialize_identically(tcr, mhc):
    project = trainer_projection()
    data = rows()
    con = duckdb.connect()
    try:
        con.execute("CREATE TABLE t (i INTEGER, " + ", ".join(f"{c} VARCHAR" for c in COLUMNS) + ")")
        con.executemany(f"INSERT INTO t VALUES ({', '.join('?' * (len(COLUMNS) + 1))})",
                        [(i, *(r.get(c) for c in COLUMNS)) for i, r in enumerate(data)])
        got = con.execute(f"SELECT {build_input_text_sql(tcr, mhc)} FROM t ORDER BY i").fetchall()
    finally:
        con.close()
    assert len(got) == len(data) == 325 + 5
    for row, (sql_text,) in zip(data, got):
        assert (sql_text or "") == project(row, tcr, mhc)[0], row["order_key"]


@pytest.mark.parametrize("text,tokens", [
    ("GILGFVFTL", 9 + 2),
    ("CASSF<eos>GILGFVFTL", 5 + 9 + 1 + 2),
    ("CAVR<eos>CASSF<eos>GILGFVFTL<eos>" + "M" * 365, 4 + 5 + 9 + 365 + 3 + 2),
])
def test_token_count_is_residues_plus_separators_plus_two(text, tokens):
    """The identity the real ESM-C tokenizer satisfied on every one of 64,440 real rows."""
    con = duckdb.connect()
    try:
        assert con.execute(f"SELECT {token_count_sql('?')}".replace("?", "$1"), [text]).fetchone()[0] == tokens
    finally:
        con.close()
    assert TOKEN_LIMIT == 1024
