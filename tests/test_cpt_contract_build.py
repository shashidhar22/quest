"""The builder's contract mode on a small source (corpus repair step 10).

`fill_staging_from_contract` is SQL. Here the expected selection is derived separately, in
Python, from `quest.data.contract_sampling`'s and `split_routing`'s Python twins, and the rows the
SQL inserts must be exactly those. The refusals are tested the same way: a contract sized on a
different supply, or asking for more than the source holds, must raise, never insert fewer.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
duckdb = pytest.importorskip("duckdb", reason="the builder's contract mode is SQL; duckdb required")

from quest.data.contract_sampling import (  # noqa: E402
    background_key,
    choose_order_variant,
    example_hash,
    select_examples,
    stratum_of,
)
from quest.data.split_routing import combine, peptide_split, receptor_split  # noqa: E402
from scripts.data_processing.cpt_dataset_builder import (  # noqa: E402
    REFERENCE_AWARE_POLICY,
    ContractNotMet,
    fill_staging_from_contract,
    filter_where_sql,
)

SEED, ROUTING = 42, "receptor-hash-v2"
SRC_COLS = ["tra_full", "trb_full", "tra_cdr1", "tra_cdr2", "tra_cdr3", "trb_cdr1", "trb_cdr2", "trb_cdr3",
            "peptide", "mhc_one", "mhc_two", "mhc_one_pocket", "mhc_one_pocket_contact", "mhc_two_pocket",
            "mhc_two_pocket_contact", "mhc_one_allele", "mhc_two_allele", "sequence"]
ALL_COLS = SRC_COLS + ["subset_key", "order_key"]
MHC_I, MHC_E = "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRF", "GSHSLKYFHTSVSRPGRGEPRFISVGYVDDTQFVRF"
PEPTIDES = ["GILGFVFTL", "NLVPMVATV", "KLGGALQAK", "ELAGIGILTV", "LLWNGPMAV", "YVLDHLIVV", "GLCTLVAML"]


def _row(subset_key, order_key, **kw):
    r = dict.fromkeys(ALL_COLS)
    r.update(kw, subset_key=subset_key, order_key=order_key)
    r["sequence"] = "|".join(str(r[c]) for c in ("tra_cdr3", "trb_cdr3", "peptide", "mhc_one") if r[c])
    return r


def source_rows():
    rows = []
    for i in range(60):                    # paired TCR + peptide + classical class I, two order variants
        kw = dict(tra_cdr3=f"CAV{'AGQ'[i % 3]}{'DNSTY'[i % 5]}{chr(65 + i % 20)}KLIF".replace("B", "S").replace("J", "L").replace("O", "P"),
                  trb_cdr3=f"CASS{'LQRPG'[i % 5]}{'ETNYD'[i % 4]}{chr(65 + (i * 7) % 20)}QYF".replace("B", "S").replace("J", "L").replace("O", "P"),
                  peptide=PEPTIDES[0] if i % 2 else PEPTIDES[1 + i % 3], mhc_one=MHC_I, mhc_one_allele="HLA-A*02:01")
        for ok in ("tra_trb_peptide_mhc_one", "mhc_one_peptide_trb_tra"):
            rows.append(_row("tra_trb_peptide_mhc_one", ok, **kw))
    for i in range(30):                    # one-chain TCR + peptide, no MHC
        rows.append(_row("trb_peptide", "trb_peptide" if i % 2 else "peptide_trb",
                         trb_cdr3=f"CASR{'GLT'[i % 3]}{'ADNQ'[i % 4]}{chr(65 + (i * 3) % 20)}EAFF".replace("B", "S").replace("J", "L").replace("O", "P"),
                         peptide=PEPTIDES[i % 5]))
    for i in range(6):                     # non-classical class I, no peptide
        rows.append(_row("trb_mhc_one", "trb_mhc_one", trb_cdr3=f"CASSE{'ADGNQT'[i]}YEQYF", mhc_one=MHC_E,
                         mhc_one_allele="HLA-E*01:01"))
    for p in PEPTIDES:                     # peptide only
        rows.append(_row("peptide", "peptide", peptide=p))
    for i in range(400):                   # single beta chains
        rows.append(_row("trb", "trb", trb_cdr3=f"CASS{'ADEFGHIKLMNPQRSTVWY'[i % 19]}{'ADEFGHIKLM'[(i // 19) % 10]}{'NPQRSTVWY'[(i // 190) % 9]}TQYF"))
    for i in range(150):                   # single alpha chains
        rows.append(_row("tra", "tra", tra_cdr3=f"CAV{'ADEFGHIKLMNPQRSTVWY'[i % 19]}{'ADEFGHIKLM'[(i // 19) % 10]}GKLIF"))
    rows.append(_row("tra_trb", "tra_trb", tra_full="MAAA", trb_full="MBBB"))   # no CDR3: no text under cdr3
    return rows


def connect(rows):
    con = duckdb.connect()
    ddl = ", ".join(f"{c} VARCHAR" for c in ALL_COLS)
    con.execute(f"CREATE TABLE source ({ddl})")
    con.executemany(f"INSERT INTO source VALUES ({', '.join('?' * len(ALL_COLS))})",
                    [[r[c] for c in ALL_COLS] for r in rows])
    con.execute(f"CREATE TABLE staging_raw ({ddl})")
    con.execute("CREATE TABLE elig (seq_norm VARCHAR, allele_norm VARCHAR, chain_role VARCHAR, "
                "mhc_class VARCHAR, classical BOOLEAN, pair_group VARCHAR)")
    con.execute("INSERT INTO elig VALUES (?, 'HLA-A*02:01', 'heavy', 'I', true, 'I'), "
                "(?, 'HLA-E*01:01', 'heavy', 'I', false, 'I')", [MHC_I, MHC_E])
    con.execute("CREATE TABLE pep_lkp (mol VARCHAR, cluster_id BIGINT, split VARCHAR)")
    con.execute("INSERT INTO pep_lkp VALUES (?, 7, 'train'), (?, 7, 'train')", PEPTIDES[:2])   # one shared cluster
    return con


CLASS = {MHC_I: ("I", True), MHC_E: ("I", False), None: (None, None)}
CLUSTER = {PEPTIDES[0]: 7, PEPTIDES[1]: 7}
RULES = {"s1_paired_tcr_peptide_classI": ({"cap_per_peptide": 3}, 0.5),
         "s7_tcr_peptide_no_mhc": ({"cap_per_peptide": 2, "examples": 4}, None),
         "s5_nonclassical_classI": ("all", None),
         "s8_peptide_only": ({"examples": 2}, None)}
BACKGROUND = {"background_single_beta_chain": ("trb", "trb_cdr3", 0.4),
              "background_single_alpha_chain": ("tra", "tra_cdr3", 0.3)}


def expected(rows):
    """The selection, derived in Python: (contract version, {identity of each expected row})."""
    examples = {}
    for r in rows:
        if r["subset_key"] in ("tra", "trb") or not (r["tra_cdr3"] or r["trb_cdr3"] or r["peptide"]):
            continue
        tup = tuple(r[c] for c in ("tra_cdr3", "trb_cdr3", "peptide", "mhc_one", "mhc_two"))
        split = combine([receptor_split(r["trb_cdr3"], "trb_cdr3", SEED), receptor_split(r["tra_cdr3"], "tra_cdr3", SEED),
                         peptide_split(CLUSTER.get(r["peptide"]), r["peptide"], SEED)])
        stratum = stratum_of(*CLASS[r["mhc_one"]], r["peptide"], r["tra_cdr3"], r["trb_cdr3"], r["mhc_two"])
        examples.setdefault((stratum, split), {}).setdefault(tup, []).append(r["order_key"])
    version, want = {"splits": {}}, set()
    for split in ("train", "val", "test"):
        strata = {}
        for stratum, (rule, fraction) in RULES.items():
            pool = [(example_hash(*tup), tup[2], tup, oks) for tup, oks in examples.get((stratum, split), {}).items()]
            cap = None if rule == "all" else rule.get("cap_per_peptide")
            under_cap = select_examples(pool, cap, None)
            n = len(under_cap) if rule == "all" or "examples" not in rule else min(rule["examples"], len(under_cap))
            if fraction is not None:
                n = int(len(under_cap) * fraction)
            chosen = select_examples(pool, cap, n)
            strata[stratum] = {"rule": rule, "target_examples": n, "target_rows": n,
                               "examples_available": len(pool), "examples_available_under_cap": len(under_cap)}
            want |= {(tup, choose_order_variant(SEED, h, oks)) for h, _p, tup, oks in chosen}
        for name, (sk, axis, fraction) in BACKGROUND.items():
            pool = sorted((background_key(SEED, axis, r[axis]), r[axis]) for r in rows
                          if r["subset_key"] == sk and receptor_split(r[axis], axis, SEED) == split)
            n = int(len(pool) * fraction)
            strata[name] = {"target_rows": n, "rule": {"fill": "test"}}
            want |= {((cdr3 if sk == "tra" else None, cdr3 if sk == "trb" else None, None, None, None), sk)
                     for _k, cdr3 in pool[:n]}
        version["splits"][split] = {"by_stratum": strata,
                                    "target_rows_total": sum(v["target_rows"] for v in strata.values())}
    return version, want


def run(con, version):
    return fill_staging_from_contract(
        con, src="source", where=filter_where_sql(REFERENCE_AWARE_POLICY, eligibility_table="elig"),
        eligibility_table="elig", peptide_lookup="pep_lkp", version=version, routing=ROUTING, seed=SEED,
        tcr_variant="cdr3", mhc_variant="full", src_cols=SRC_COLS)


def staged(con):
    return {((a, b, p, m1, m2), ok) for a, b, p, m1, m2, ok in con.execute(
        "SELECT tra_cdr3, trb_cdr3, peptide, mhc_one, mhc_two, order_key FROM staging_raw").fetchall()}


def test_the_fixture_exercises_every_branch():
    version, want = expected(source_rows())
    by = {}
    for split, v in version["splits"].items():
        for stratum, s in v["by_stratum"].items():
            by[stratum] = by.get(stratum, 0) + s.get("target_examples", s["target_rows"])
    assert all(n > 0 for n in by.values()), by                       # every stratum selects something
    s1 = version["splits"]["train"]["by_stratum"]["s1_paired_tcr_peptide_classI"]
    assert s1["examples_available_under_cap"] < s1["examples_available"]   # the cap binds
    assert s1["target_examples"] < s1["examples_available_under_cap"]      # and the count binds after it
    assert len(want) == sum(v["target_rows_total"] for v in version["splits"].values())


def test_sql_selects_exactly_the_rows_python_expects():
    rows = source_rows()
    version, want = expected(rows)
    con = connect(rows)
    stats = run(con, version)
    assert staged(con) == want
    assert con.execute("SELECT COUNT(*) FROM staging_raw").fetchone()[0] == len(want)   # no row twice
    for split, v in version["splits"].items():
        for stratum, s in v["by_stratum"].items():
            assert stats["achieved"][f"{split}/{stratum}"] == s.get("target_examples", s["target_rows"])


def test_one_row_per_example_and_the_cap_holds_per_peptide():
    rows = source_rows()
    version, _ = expected(rows)
    con = connect(rows)
    run(con, version)
    n, distinct = con.execute("""SELECT COUNT(*), COUNT(DISTINCT (tra_cdr3, trb_cdr3, peptide, mhc_one, mhc_two))
        FROM staging_raw WHERE subset_key NOT IN ('tra', 'trb')""").fetchone()
    assert n == distinct > 0
    worst = con.execute("""SELECT MAX(n) FROM (SELECT COUNT(*) n FROM staging_raw
        WHERE subset_key = 'tra_trb_peptide_mhc_one' GROUP BY peptide)""").fetchone()[0]
    assert worst <= 3 * 3                       # cap 3 per peptide, in each of at most three splits
    assert con.execute("SELECT COUNT(*) FROM staging_raw WHERE subset_key = 'tra_trb'").fetchone()[0] == 0


def test_the_selection_does_not_depend_on_source_row_order():
    rows = source_rows()
    version, want = expected(rows)
    con = connect(list(reversed(rows)))
    run(con, version)
    assert staged(con) == want


def test_a_duplicated_source_row_still_yields_one_row():
    rows = source_rows()
    version, want = expected(rows)
    con = connect(rows + [dict(r, tra_full="MEXTRA") for r in rows[:40]])   # same example and order, other content
    run(con, version)
    assert staged(con) == want
    assert con.execute("SELECT COUNT(*) FROM staging_raw").fetchone()[0] == len(want)


def test_background_takes_each_receptor_once():
    rows = source_rows()
    version, want = expected(rows)
    again = [dict(r, trb_full="MEXTRA" + r["trb_cdr3"]) for r in rows if r["subset_key"] == "trb"]
    con = connect(rows + again + again[:50])            # every beta receptor two or three times
    stats = run(con, version)
    assert staged(con) == want
    n, d = con.execute("SELECT COUNT(*), COUNT(DISTINCT trb_cdr3) FROM staging_raw WHERE subset_key = 'trb'").fetchone()
    assert n == d > 0 and stats["background_rows_sharing_a_receptor"] == {"tra": 0, "trb": 0}


@pytest.mark.parametrize("field", ["examples_available", "examples_available_under_cap"])
def test_refuses_a_contract_sized_on_a_different_supply(field):
    rows = source_rows()
    version, _ = expected(rows)
    version = copy.deepcopy(version)
    version["splits"]["train"]["by_stratum"]["s7_tcr_peptide_no_mhc"][field] += 1
    with pytest.raises(ContractNotMet, match="sized on"):
        run(connect(rows), version)


def test_refuses_a_target_the_stratum_cannot_supply():
    rows = source_rows()
    version, _ = expected(rows)
    version = copy.deepcopy(version)
    s = version["splits"]["train"]["by_stratum"]["s5_nonclassical_classI"]
    s["target_examples"] = s["examples_available"] + 1
    with pytest.raises(ContractNotMet, match="selected"):
        run(connect(rows), version)


def test_refuses_background_the_source_cannot_supply():
    rows = source_rows()
    version, _ = expected(rows)
    version = copy.deepcopy(version)
    version["splits"]["val"]["by_stratum"]["background_single_alpha_chain"]["target_rows"] = 10_000
    with pytest.raises(ContractNotMet, match="single-receptor"):
        run(connect(rows), version)


def test_refuses_an_example_that_lands_in_two_strata():
    rows = source_rows()
    version, _ = expected(rows)
    con = connect(rows)
    # The same chain under a second allele name that decides differently: one example, two strata.
    con.execute("INSERT INTO elig VALUES (?, 'HLA-E*01:03', 'heavy', 'I', true, 'I')", [MHC_E])
    con.execute("INSERT INTO source SELECT * REPLACE ('HLA-E*01:03' AS mhc_one_allele) FROM source "
                "WHERE mhc_one_allele = 'HLA-E*01:01'")
    with pytest.raises(ContractNotMet, match="not unique"):
        run(con, version)


@pytest.mark.parametrize("column", ["tra_cdr3", "mhc_two"])
def test_refuses_an_empty_string_where_a_component_is_absent(column):
    """'' is present to the stratum rule and the hash, absent from the text: never selected silently."""
    rows = source_rows()
    version, _ = expected(rows)
    rows = rows + [_row("trb_peptide", "trb_peptide", trb_cdr3="CASSWWWEQYF", peptide=PEPTIDES[3], **{column: ""})]
    with pytest.raises(ContractNotMet, match="empty-string"):
        run(connect(rows), version)

