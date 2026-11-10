"""Example-level sampling for the CPT sampling contract (corpus repair, steps 9-10).

The contract (`cpt-sampling-contract-v1`) names targets per stratum and split, in biological
examples. This module is the single definition of the four things that turn rows into those
examples, so the sizing scripts that produced the contract and the builder that has to meet it
cannot drift:

  * the stratum of a row, from its components and the reference-derived class of its MHC chain;
  * the example hash, md5 of the five components joined by '|' with a missing component written
    as the empty string. `concat_ws` on the bare columns skips NULLs, which made an alpha-only
    and a beta-only example with the same CDR3 collide; every component is therefore coalesced;
  * the selection within one (stratum, split): the per-peptide cap first (lowest hashes within a
    peptide), then the N lowest hashes of what the cap leaves. No replacement;
  * the one order variant kept per selected example: the order_key with the lowest seeded md5.

Each SQL builder has a Python twin and a test pins their agreement. Selection is by md5, not
DuckDB's hash(), whose values are not promised stable across versions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

STRATA = (
    "s1_paired_tcr_peptide_classI",
    "s2_one_chain_tcr_peptide_classI",
    "s3_peptide_classI_no_tcr",
    "s4_classI_no_peptide",
    "s5_nonclassical_classI",
    "s6_classII",
    "s7_tcr_peptide_no_mhc",
    "s8_peptide_only",
    "s9_paired_receptors_only",
)
# Strata whose examples carry a receptor and a peptide; the contract caps these per peptide.
TCR_PEPTIDE_STRATA = (STRATA[0], STRATA[1], STRATA[6])
BACKGROUND_STRATA = {"background_single_beta_chain": "trb", "background_single_alpha_chain": "tra"}
NO_PEPTIDE = "<none>"
COMPONENTS = ("tra_cdr3", "trb_cdr3", "peptide", "mhc_one", "mhc_two")


def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def stratum_sql(*, mhc_class: str, classical: str, peptide: str, tra: str, trb: str, mhc_two: str) -> str:
    """CASE expression for a row's stratum. Arguments are SQL expressions.

    `mhc_class` and `classical` describe the row's `mhc_one` chain and come from the eligibility
    decision, never from an allele name. `peptide` must already be NULL when the row has none
    (NULLIF(peptide, '')). The first matching branch wins, so the order is part of the definition.
    """
    c1 = f"{mhc_class} = 'I' AND {classical}"
    any_tcr = f"({tra} IS NOT NULL OR {trb} IS NOT NULL)"
    return f"""CASE
        WHEN {c1} AND {peptide} IS NOT NULL AND {tra} IS NOT NULL AND {trb} IS NOT NULL THEN '{STRATA[0]}'
        WHEN {c1} AND {peptide} IS NOT NULL AND {any_tcr} THEN '{STRATA[1]}'
        WHEN {c1} AND {peptide} IS NOT NULL THEN '{STRATA[2]}'
        WHEN {c1} THEN '{STRATA[3]}'
        WHEN {mhc_class} = 'I' THEN '{STRATA[4]}'
        WHEN {mhc_class} = 'II' OR {mhc_two} IS NOT NULL THEN '{STRATA[5]}'
        WHEN {peptide} IS NOT NULL AND {any_tcr} THEN '{STRATA[6]}'
        WHEN {peptide} IS NOT NULL THEN '{STRATA[7]}'
        ELSE '{STRATA[8]}' END"""


def stratum_of(mhc_class, classical, peptide, tra, trb, mhc_two) -> str:
    """Python twin of `stratum_sql`. None is absent; `classical` counts only when it is True."""
    c1 = mhc_class == "I" and classical is True
    has_pep, any_tcr = peptide is not None, tra is not None or trb is not None
    if c1 and has_pep and tra is not None and trb is not None:
        return STRATA[0]
    if c1 and has_pep and any_tcr:
        return STRATA[1]
    if c1 and has_pep:
        return STRATA[2]
    if c1:
        return STRATA[3]
    if mhc_class == "I":
        return STRATA[4]
    if mhc_class == "II" or mhc_two is not None:
        return STRATA[5]
    if has_pep and any_tcr:
        return STRATA[6]
    if has_pep:
        return STRATA[7]
    return STRATA[8]


def example_hash_sql(tra: str = "tra_cdr3", trb: str = "trb_cdr3", peptide: str = "peptide",
                     mhc_one: str = "mhc_one", mhc_two: str = "mhc_two") -> str:
    """md5 of the example tuple; every component coalesced so that NULLs keep their position."""
    parts = ", ".join(f"coalesce({c}, '')" for c in (tra, trb, peptide, mhc_one, mhc_two))
    return f"md5(concat_ws('|', {parts}))"


def example_hash(tra, trb, peptide, mhc_one, mhc_two) -> str:
    """Python twin of `example_hash_sql`."""
    return _md5("|".join("" if c is None else c for c in (tra, trb, peptide, mhc_one, mhc_two)))


def peptide_rank_sql(stratum: str = "stratum", split: str = "split", peptide: str = "peptide",
                     ex_hash: str = "ex_hash") -> str:
    """Rank of an example within its peptide, by hash: the column a per-peptide cap filters on."""
    return (f"row_number() OVER (PARTITION BY {stratum}, {split}, coalesce({peptide}, '{NO_PEPTIDE}') "
            f"ORDER BY {ex_hash})")


def selection_sql(table: str, cap, n, rank_col: str = "rn_pep", ex_hash: str = "ex_hash") -> str:
    """SELECT of one (stratum, split)'s selection from `table`; two `?` parameters, in that order.

    `table` holds one line per example with `stratum`, `split`, `ex_hash` and `rank_col`
    (`peptide_rank_sql`). `cap` and `n` are None when the rule has no cap or no count.
    ORDER BY ... LIMIT, not QUALIFY row_number() <= n: DuckDB 1.5 rewrites the latter to a top-N
    aggregate that refuses n >= 1,000,000.
    """
    cap_sql = "TRUE" if cap is None else f"{rank_col} <= {int(cap)}"
    limit_sql = "" if n is None else f" ORDER BY {ex_hash} LIMIT {int(n)}"
    return f"SELECT * FROM {table} WHERE stratum = ? AND split = ? AND {cap_sql}{limit_sql}"


def select_examples(examples, cap, n) -> list:
    """Python twin of the selection. `examples` is an iterable of (ex_hash, peptide, ...) tuples
    of ONE (stratum, split); returns the chosen tuples in hash order."""
    pool = sorted(examples, key=lambda e: e[0])
    if cap is not None:
        seen: dict = {}
        kept = []
        for e in pool:
            key = NO_PEPTIDE if e[1] is None else e[1]
            if seen.get(key, 0) < cap:
                seen[key] = seen.get(key, 0) + 1
                kept.append(e)
        pool = kept
    return pool if n is None else pool[:n]


def rule_cap_and_count(rule) -> tuple:
    """(cap, n) of a contract rule: 'all', {'cap_per_peptide': K}, {'examples': N}, or both."""
    if rule == "all":
        return None, None
    unknown = set(rule) - {"cap_per_peptide", "examples"}
    if not isinstance(rule, dict) or unknown or not rule:
        raise ValueError(f"not a sampling rule: {rule!r}")
    return rule.get("cap_per_peptide"), rule.get("examples")


def order_variant_key_sql(seed: int, ex_hash: str = "ex_hash", order_key: str = "order_key") -> str:
    """Sort key for choosing an example's single order variant: the lowest key is kept."""
    return f"md5(concat('{int(seed)}|', {ex_hash}, '|', {order_key}))"


def choose_order_variant(seed: int, ex_hash: str, order_keys) -> str:
    """Python twin: the order_key with the lowest seeded md5."""
    order_keys = list(order_keys)
    if not order_keys:
        raise ValueError("an example has no order variant")
    return min(order_keys, key=lambda ok: _md5(f"{int(seed)}|{ex_hash}|{ok}"))


# ---------- the contract file ----------

CONTRACT_NAME = "cpt-sampling-contract-v1"
SPLITS = ("train", "val", "test")
ROW_SAMPLED, EXAMPLE_SAMPLED = "rows per partition", "examples per stratum"


def load_contract_version(path, version: str) -> tuple[dict, dict, str]:
    """Read a sampling contract. Returns (contract, the named version, sha256 of the file)."""
    raw = Path(path).read_bytes()
    contract = json.loads(raw)
    if contract.get("contract") != CONTRACT_NAME:
        raise ValueError(f"{path} is not a {CONTRACT_NAME} file")
    versions = contract.get("versions", {})
    if version not in versions:
        raise ValueError(f"{path} has no version {version!r}; it has {sorted(versions)}")
    return contract, versions[version], hashlib.sha256(raw).hexdigest()


def version_kind(version: dict) -> str:
    """Whether a contract version fixes rows per partition (run05's sampling) or examples per stratum."""
    if "target_rows_by_partition" in version:
        return ROW_SAMPLED
    if "splits" in version:
        return EXAMPLE_SAMPLED
    raise ValueError("contract version has neither partition targets nor per-split strata")


def contract_mismatches(contract: dict, **build) -> list[str]:
    """Every way the build's own settings differ from what the contract was sized under.

    A contract's numbers hold only for the universe it was counted on, so the builder refuses
    on any entry here instead of sampling a different universe to the same targets. `build`
    must name every key below; a setting the caller forgot is a mismatch, not a pass.
    """
    p = contract["policies"]
    want = {
        "eligibility_policy": p["eligibility"]["policy"],
        "reference_sha256": p["eligibility"]["hla_reference_sha256"],
        "partial_chain_fraction": p["eligibility"]["partial_chain_fraction"],
        "exclusion_policy": p["benchmark_exclusion"]["policy"],
        "exclusion_list_sha256": p["benchmark_exclusion"]["list_sha256"],
        "min_edit_distance": p["benchmark_exclusion"]["min_edit_distance"],
        "routing_policy": p["split_routing"]["policy"],
        "seed": p["split_routing"]["seed"],
        "serialization": p["serialization"],
    }
    extra = set(build) - set(want)
    if extra:
        raise TypeError(f"unknown build settings: {sorted(extra)}")
    return [f"{k}: contract {v!r}, build {build.get(k)!r}" for k, v in want.items() if build.get(k) != v]


def serialization_name(tcr_variant: str, mhc_variant: str) -> str:
    return f"{tcr_variant} x {mhc_variant}"


def partition_targets(version: dict) -> dict:
    """{(subset_key, order_key): rows} of a row-sampled version."""
    return {tuple(k.split("|")): int(n) for k, n in version["target_rows_by_partition"].items()}


def stratum_targets(version: dict) -> list[dict]:
    """One entry per (split, stratum) of an example-sampled version, in a fixed order.

    Non-background entries carry the rule's cap, the exact example target and the supply the
    contract was sized on; background entries carry the row target and their subset_key.
    """
    out = []
    for split in SPLITS:
        strata = version["splits"][split]["by_stratum"]
        unknown = set(strata) - set(STRATA) - set(BACKGROUND_STRATA)
        if unknown:
            raise ValueError(f"contract names unknown strata in {split}: {sorted(unknown)}")
        for name in sorted(strata):
            v = strata[name]
            if name in BACKGROUND_STRATA:
                out.append({"split": split, "stratum": name, "background": BACKGROUND_STRATA[name],
                            "target_rows": int(v["target_rows"])})
                continue
            cap, n = rule_cap_and_count(v["rule"])
            # The rule and the numbers beside it must say the same thing, or the contract
            # contradicts itself and whichever one the builder follows is a guess.
            implied = n if n is not None else (v["examples_available"] if cap is None
                                               else v["examples_available_under_cap"])
            if implied != v["target_examples"] or v["target_examples"] > v["examples_available_under_cap"]:
                raise ValueError(f"{split}/{name}: rule {v['rule']!r} implies {implied:,} examples, the target "
                                 f"is {v['target_examples']:,} of {v['examples_available_under_cap']:,} available")
            out.append({"split": split, "stratum": name, "background": None, "cap": cap,
                        "target_examples": int(v["target_examples"]),
                        "examples_available": int(v["examples_available"]),
                        "examples_available_under_cap": int(v["examples_available_under_cap"])})
    return out


def background_key_sql(seed: int, axis: str, cdr3: str) -> str:
    """Sort key for background rows: the N lowest per split are kept. Same hash family as the
    receptor's routing vote, with its own prefix so that the two are independent."""
    return f"md5(concat('{int(seed)}|background|{axis}|', {cdr3}))"


def background_key(seed: int, axis: str, cdr3: str) -> str:
    """Python twin of `background_key_sql`."""
    return _md5(f"{int(seed)}|background|{axis}|{cdr3}")


# ---------- checking a built dataset against the contract ----------

EOS = "<eos>"
_ORDER_TOKENS = ("mhc_one", "mhc_two", "tra", "trb", "peptide")
CHECKABLE_SERIALIZATION = "cdr3 x full"      # one text segment per component, so a row can be parsed back
_CENSUS_STRATUM = {s[1]: s for s in STRATA}  # the census names its strata '1 paired TCR + ...'


def components_of_output_row(input_text: str, order_key: str) -> dict:
    """{component: text} of a serialized row under cdr3 x full. Raises when the order_key does
    not name exactly the segments present, because every count built on the parse would be wrong."""
    order, rest = [], order_key
    while rest:
        token = next((t for t in _ORDER_TOKENS if rest == t or rest.startswith(t + "_")), None)
        if token is None:
            raise ValueError(f"order_key {order_key!r} is not a '_'-joined list of known components")
        order.append(token)
        rest = rest[len(token) + 1:]
    parts = input_text.split(EOS)
    if len(parts) != len(order) or len(set(order)) != len(order) or not all(parts):
        raise ValueError(f"order_key {order_key!r} names {len(order)} components, input_text has {len(parts)}")
    return dict(zip(order, parts))


def count_built_split(rows, class_of_chain) -> dict:
    """Recount one built split from its own rows: `rows` yields (input_text, order_key).

    `class_of_chain(sequence)` returns (mhc_class, classical) of an `mhc_one` chain from the
    reference, so the strata are re-derived from the built text and not from any label the
    builder wrote. Returns per-stratum rows and residues, per-peptide maxima, and the number of
    examples (and, for background rows, receptors) that occur more than once.
    """
    rows_by, residues_by, per_peptide, seen, repeated = {}, {}, {}, set(), 0
    receptors = {v: set() for v in BACKGROUND_STRATA.values()}
    for text, order_key in rows:
        c = components_of_output_row(text, order_key)
        residues = len(text) - len(EOS) * (len(c) - 1)
        if order_key in BACKGROUND_STRATA.values():
            name = next(k for k, v in BACKGROUND_STRATA.items() if v == order_key)
            if text in receptors[order_key]:
                repeated += 1
            receptors[order_key].add(text)
        else:
            tup = (c.get("tra"), c.get("trb"), c.get("peptide"), c.get("mhc_one"), c.get("mhc_two"))
            mhc_class, classical = class_of_chain(tup[3]) if tup[3] is not None else (None, None)
            name = stratum_of(mhc_class, classical, tup[2], tup[0], tup[1], tup[4])
            if tup in seen:
                repeated += 1
            seen.add(tup)
            key = (name, NO_PEPTIDE if tup[2] is None else tup[2])
            per_peptide[key] = per_peptide.get(key, 0) + 1
        rows_by[name] = rows_by.get(name, 0) + 1
        residues_by[name] = residues_by.get(name, 0) + residues
    largest = {}
    for (name, _pep), n in per_peptide.items():
        largest[name] = max(largest.get(name, 0), n)
    return {"rows": rows_by, "residues": residues_by, "largest_peptide": largest, "repeated_examples": repeated,
            # Kept for the cross-split check; not for serializing.
            "examples": seen, "background_receptors": receptors}


def contract_violations(version: dict, built: dict) -> list[str]:
    """Every way a built dataset departs from a contract version. `built` maps the contract's
    split names to `count_built_split` results. An empty list is the only pass."""
    out = []
    if version_kind(version) == EXAMPLE_SAMPLED:
        for split in SPLITS:
            want, got = version["splits"][split], built[split]
            for name, v in sorted(want["by_stratum"].items()):
                n = got["rows"].get(name, 0)
                if n != v["target_rows"]:
                    out.append(f"{split}/{name}: {n:,} rows, target {v['target_rows']:,}")
                if name in BACKGROUND_STRATA:
                    est, res = v["residue_tokens_estimate"], got["residues"].get(name, 0)
                    if abs(res - est) > 0.01 * est:
                        out.append(f"{split}/{name}: {res:,} residue tokens, estimate {est:,} (tolerance 1%)")
                    continue
                if got["residues"].get(name, 0) != v["residue_tokens"]:
                    out.append(f"{split}/{name}: {got['residues'].get(name, 0):,} residue tokens, "
                               f"target {v['residue_tokens']:,}")
                cap, _n = rule_cap_and_count(v["rule"])
                if cap is not None and got["largest_peptide"].get(name, 0) > cap:
                    out.append(f"{split}/{name}: a peptide has {got['largest_peptide'][name]:,} examples, cap {cap:,}")
            for name in sorted(set(got["rows"]) - set(want["by_stratum"])):
                out.append(f"{split}/{name}: {got['rows'][name]:,} rows in a stratum the contract does not name")
            if got["repeated_examples"]:
                out.append(f"{split}: {got['repeated_examples']:,} examples occur more than once")
            if sum(got["rows"].values()) != want["target_rows_total"]:
                out.append(f"{split}: {sum(got['rows'].values()):,} rows, target {want['target_rows_total']:,}")
        # An example, or a background receptor, routes to one split; in two it is a leak.
        for i, a in enumerate(SPLITS):
            for b in SPLITS[i + 1:]:
                shared = len(built[a].get("examples", set()) & built[b].get("examples", set()))
                shared += sum(len(built[a].get("background_receptors", {}).get(k, set())
                                  & built[b].get("background_receptors", {}).get(k, set()))
                              for k in BACKGROUND_STRATA.values())
                if shared:
                    out.append(f"{a} and {b} share {shared:,} examples or background receptors")
        return out
    # Row-sampled: split and stratum counts are outcomes of routing, held to the stated tolerance.
    for census_name, by_split in sorted(version["split_and_stratum_counts"]["expected"].items()):
        if census_name.startswith("background single "):
            name = next(k for k, v in BACKGROUND_STRATA.items() if v == census_name.split()[2])
        else:
            name = _CENSUS_STRATUM[census_name[0]]
        for split, v in by_split.items():
            e, n = v["expected_rows"], built[split]["rows"].get(name, 0)
            if e >= 100 and abs(n - e) > 4 * e ** 0.5:
                out.append(f"{split}/{name}: {n:,} rows, expected {e:,} +/- {4 * e ** 0.5:,.0f}")
    return out

