#!/usr/bin/env python3
"""Audit a BUILT CPT corpus from its own files (corpus repair step 11).

Two reports, both computed from the saved dataset and nothing the builder wrote about it:

  leakage-audit.json     Overlap with the benchmark. For every held-out part: built rows carrying
                         one of its peptides, one of its (TRB, peptide) associations, one of its
                         (TRA, TRB, peptide) associations; and the smallest edit distance from any
                         built peptide to any held-out peptide. Under heldout-peptides-v2 the
                         counts must be zero and the distance at least 2.
  coverage-by-split.csv  Per split and stratum: rows, distinct examples, peptides, receptors and
                         classical class-I heavy chains, with MHC class re-decided against the
                         reference.

Independence. Rows are parsed with `contract_sampling.components_of_output_row`; nothing else
is shared with the builder. In particular the near-peptide search does not use the exclusion
list or `benchmark_exclusion.one_edit_neighbours`: two strings are within one edit exactly when
they are equal or share a string obtained by deleting at most one letter from each, so candidates
are found through a deletion index and then confirmed with a plain edit-distance routine.

Only cdr3 x full can be audited: one text segment per component.

Usage:
    python3 cpt_built_audit.py --dataset <dir> --splits-dir <benchmark splits> --task as_paired_i \
        --heldout-parts val test_iid test_novel_pep test_level4 --hla-reference hla_prot.fasta \
        --out-dir <dir>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quest.data.contract_sampling import (  # noqa: E402
    BACKGROUND_STRATA,
    components_of_output_row,
    stratum_of,
)

DATASET_SPLITS = ("train", "validation", "test")


def normalize(peptide) -> str:
    """Letters only, upper case; a missing value is ''."""
    return re.sub(r"[^A-Za-z]", "", peptide).upper() if isinstance(peptide, str) else ""


def edit_distance_at_most_one(a: str, b: str) -> bool:
    """Plain check, no enumeration: equal, one substitution, or one insertion/deletion."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long_ = (a, b) if la < lb else (b, a)
    i = 0
    while i < len(short) and short[i] == long_[i]:
        i += 1
    return short[i:] == long_[i + 1:]


def deletions(s: str) -> set:
    """`s` and every string obtained by deleting one letter from it."""
    return {s} | {s[:i] + s[i + 1:] for i in range(len(s))}


def near_heldout(built_peptides, heldout) -> dict:
    """{built peptide: a held-out peptide within one edit}, for every built peptide that has one."""
    index: dict = {}
    for h in heldout:
        for d in deletions(h):
            index.setdefault(d, set()).add(h)
    hits = {}
    for p in built_peptides:
        candidates = set().union(*(index.get(d, ()) for d in deletions(p)))
        for h in sorted(candidates):
            if edit_distance_at_most_one(p, h):
                hits[p] = h
                break
    return hits


def built_rows(dataset_dir):
    """Yield (split, input_text, order_key) for every row of a saved DatasetDict."""
    from datasets import load_from_disk
    ds = load_from_disk(str(dataset_dir))
    for split in DATASET_SPLITS:
        table = ds[split].data.table.select(["input_text", "order_key"])
        for batch in table.to_batches(max_chunksize=200_000):
            for text, order in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
                yield split, text, str(order)


def audit(rows, benchmark: dict, heldout_parts, class_of_chain) -> tuple[dict, list]:
    """Run both audits over `rows` ((split, text, order_key)).

    `benchmark` maps part name to a list of (tra_cdr3, trb_cdr3, peptide) molecules, `train`
    included. Returns (leakage report, coverage lines).
    """
    heldout = {part: {normalize(p) for _a, _b, p in benchmark[part]} - {""} for part in heldout_parts}
    pairs = {part: {(normalize(b), normalize(p)) for _a, b, p in benchmark[part] if normalize(b) and normalize(p)}
             for part in heldout_parts}
    triples = {part: {(normalize(a), normalize(b), normalize(p)) for a, b, p in benchmark[part]
                      if normalize(a) and normalize(b) and normalize(p)} for part in heldout_parts}
    empty_parts = [part for part in heldout_parts if not heldout[part]]
    all_heldout = set().union(*heldout.values())
    hits = {part: {"rows_with_a_heldout_peptide": 0, "rows_with_a_heldout_trb_peptide_association": 0,
                   "rows_with_a_heldout_tra_trb_peptide_association": 0} for part in heldout_parts}
    built_peptides: dict = {}
    cover: dict = {}
    n_rows = 0
    for split, text, order_key in rows:
        n_rows += 1
        c = components_of_output_row(text, order_key)
        tra, trb, pep, m1, m2 = (c.get(k) for k in ("tra", "trb", "peptide", "mhc_one", "mhc_two"))
        if order_key in BACKGROUND_STRATA.values():
            stratum = next(k for k, v in BACKGROUND_STRATA.items() if v == order_key)
            classical = False
        else:
            mhc_class, classical = class_of_chain(m1) if m1 is not None else (None, None)
            stratum = stratum_of(mhc_class, classical, pep, tra, trb, m2)
        s = cover.setdefault((split, stratum), {"rows": 0, "examples": set(), "peptides": set(),
                                                "trb": set(), "tra": set(), "class_i_chains": set()})
        s["rows"] += 1
        s["examples"].add((tra, trb, pep, m1, m2))
        if trb is not None:
            s["trb"].add(trb)
        if tra is not None:
            s["tra"].add(tra)
        if m1 is not None and classical is True:
            s["class_i_chains"].add(m1)
        if pep is None:
            continue
        p = normalize(pep)
        s["peptides"].add(p)
        built_peptides[p] = built_peptides.get(p, 0) + 1
        if p in all_heldout:
            for part in heldout_parts:
                if p in heldout[part]:
                    hits[part]["rows_with_a_heldout_peptide"] += 1
                if trb is not None and (normalize(trb), p) in pairs[part]:
                    hits[part]["rows_with_a_heldout_trb_peptide_association"] += 1
                if tra is not None and trb is not None and (normalize(tra), normalize(trb), p) in triples[part]:
                    hits[part]["rows_with_a_heldout_tra_trb_peptide_association"] += 1
    near = near_heldout(built_peptides, all_heldout)
    exact = sorted(p for p in near if p in all_heldout)
    one_edit = sorted(p for p in near if p not in all_heldout)
    train_peptides = {normalize(p) for _a, _b, p in benchmark.get("train", [])} - {""}
    report = {
        "rows_checked": n_rows, "peptide_bearing_rows": sum(built_peptides.values()),
        "distinct_built_peptides": len(built_peptides), "heldout_parts": list(heldout_parts),
        "distinct_heldout_peptides": len(all_heldout),
        "heldout_parts_without_a_peptide": empty_parts,
        "per_heldout_part": hits,
        "built_peptides_equal_to_a_heldout_one": len(exact),
        "built_peptides_one_edit_from_a_heldout_one": len(one_edit),
        "rows_within_one_edit_of_a_heldout_peptide": sum(built_peptides[p] for p in near),
        "examples_within_one_edit": [{"built": p, "heldout": near[p], "rows": built_peptides[p]}
                                     for p in (exact + one_edit)[:20]],
        "minimum_edit_distance_to_a_heldout_peptide": 0 if exact else (1 if one_edit else "at least 2"),
        "built_peptides_also_in_benchmark_train": len(set(built_peptides) & train_peptides),
        "not_covered": ("receptor-only recurrence, which is permitted; benchmark parts not named above; "
                        "peptide clusters (no crosswalk)"),
    }
    # Nothing to compare against is not a pass: every named part must contribute a peptide.
    report["status"] = "PASS" if (n_rows > 0 and report["peptide_bearing_rows"] > 0 and heldout_parts
                                  and not empty_parts and not near and all(
        v == 0 for part in hits.values() for v in part.values())) else "FAIL"
    lines = [{"split": split, "stratum": stratum, "rows": s["rows"], "distinct_examples": len(s["examples"]),
              "distinct_peptides": len(s["peptides"]), "distinct_trb_cdr3": len(s["trb"]),
              "distinct_tra_cdr3": len(s["tra"]),
              "distinct_classical_class_i_heavy_chains": len(s["class_i_chains"])}
             for (split, stratum), s in sorted(cover.items())]
    return report, lines


def load_benchmark(splits_dir, task: str, parts) -> tuple[dict, dict]:
    import pandas as pd
    out, shas = {}, {}
    for part in parts:
        path = Path(splits_dir) / f"{task}_{part}.parquet"
        # Every row counts: molecules are compared as sets, so repeated rows cost nothing and
        # dropping by row_id could discard a peptide if two rows ever shared an id.
        frame = pd.read_parquet(path, columns=["tra_cdr3", "trb_cdr3", "peptide"])
        text = lambda v: v if isinstance(v, str) else None
        out[part] = sorted({(text(a), text(b), text(p))
                            for a, b, p in zip(frame.tra_cdr3, frame.trb_cdr3, frame.peptide)},
                           key=lambda m: tuple(x or "" for x in m))
        shas[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out, shas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--heldout-parts", nargs="+", required=True)
    ap.add_argument("--hla-reference", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    for name in ("leakage-audit.json", "coverage-by-split.csv"):
        if (a.out_dir / name).exists():
            ap.error(f"{a.out_dir / name} exists; an audit never overwrites one")
    manifest = json.loads((a.dataset / "_build_artifacts" / "build_manifest.json").read_text())
    serialization = f"{manifest['config']['tcr_variant']} x {manifest['config']['mhc_variant']}"
    if serialization != "cdr3 x full":
        ap.error(f"only cdr3 x full can be parsed back into components, not {serialization}")

    from quest.data.mhc_eligibility import build_index_from_fasta
    ref_sha = hashlib.sha256(a.hla_reference.read_bytes()).hexdigest()
    index = build_index_from_fasta(a.hla_reference, reference_version=f"{a.hla_reference.name}:{ref_sha[:12]}")
    cache: dict = {}

    def class_of_chain(seq):
        if seq not in cache:
            d = index.decide(seq)
            cache[seq] = (d.mhc_class, d.classical) if d.decision == "accept" else (None, None)
        return cache[seq]

    benchmark, split_shas = load_benchmark(a.splits_dir, a.task, ["train", *a.heldout_parts])
    report, lines = audit(built_rows(a.dataset), benchmark, a.heldout_parts, class_of_chain)
    if not manifest.get("run_id"):
        ap.error("the build manifest has no run_id: the audit could not be tied to its build")
    report = {"dataset": str(a.dataset), "run_id": manifest["run_id"], "task": a.task,
              "benchmark_split_sha256": split_shas, "hla_reference_sha256": ref_sha, **report}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with (a.out_dir / "coverage-by-split.csv").open("x", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lines[0]))
        w.writeheader()
        w.writerows(lines)
    with (a.out_dir / "leakage-audit.json").open("x") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "examples_within_one_edit"}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
