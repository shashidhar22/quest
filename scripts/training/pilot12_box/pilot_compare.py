#!/usr/bin/env python3
"""Read out the step-12 pilot: paired comparisons of SFT cells on the VALIDATION condition only.

The locked comparator (`stage42_box/paired_compare.py`) refuses without `test_novel_pep`, by
design. The pilot keeps the test parts closed (new-cpt-plan gate 7), so this wraps that module's
own `compare_condition` on `val`, with the unit map built from the validation split itself. Every
contrast is paired on identical queries; it refuses when two cells are not comparable. One SFT
seed, and the selection set is the validation split: this is triage, never a result, so it prints
no verdict of benefit and no utility read.

    python pilot_compare.py --cells B=<dir>/eval.json P=... E=... \
        --val-split <splits>/as_paired_i_val.parquet --contrasts E-B P-B E-P --out pilot.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "stage42_box"))
import paired_compare as PC  # noqa: E402

CONDITION = "val"
METRICS = ("norm_rank", "recall_at_10", "mrr")
# Provenance each cell must record, and on which the cells must agree: same code, same inputs,
# same recipe. Absence never counts as agreement (the rule of check_metadata).
SAME = ("pool_digest", "split_digests", "code_commit", "sft_objective", "sft_seed", "sft_max_len",
        "sft_recipe", "sft_train_digests")


def units_from_split(path) -> dict:
    """{(condition, row_id): (peptide, pep_cluster)} for the validation split."""
    import pandas as pd
    frame = pd.read_parquet(path, columns=["row_id", "peptide", "pep_cluster"]).drop_duplicates()
    if frame.row_id.duplicated().any():
        raise PC.ComparisonError("a validation row_id carries more than one peptide or cluster")
    return {(CONDITION, str(r)): (p, str(c))
            for r, p, c in zip(frame.row_id, frame.peptide, frame.pep_cluster)}


def problems_between(name_a, a, name_b, b) -> list:
    """Every reason these two cells cannot be compared."""
    bad = [f"{name_a} vs {name_b}: {m}" for m in PC.check_metadata(a, b)]
    pa, pb = a.get("provenance") or {}, b.get("provenance") or {}
    for k in SAME:
        if pa.get(k) in (None, "", "unrecorded") or pb.get(k) in (None, "", "unrecorded"):
            bad.append(f"{name_a} vs {name_b}: {k} is not recorded on both")
        elif pa[k] != pb[k]:
            bad.append(f"{name_a} vs {name_b}: {k} differs")
    ca, cb = a["conditions"].get(CONDITION), b["conditions"].get(CONDITION)
    if not ca or not cb:
        bad.append(f"{name_a} vs {name_b}: both must have scored {CONDITION!r}")
    elif not ca.get("true_peptides") or not cb.get("true_peptides"):
        bad.append(f"{name_a} vs {name_b}: the true peptides are not recorded on both")
    elif ca["true_peptides"] != cb["true_peptides"]:
        bad.append(f"{name_a} vs {name_b}: the queries or their true peptides differ")
    return bad


def compare(cells: dict, units: dict, contrasts, n_boot=PC.N_BOOT, seed=PC.SEED) -> dict:
    out = {"condition": CONDITION, "n_boot": n_boot, "seed": seed, "cells": {}, "contrasts": {},
           "reading": ("Validation triage on one SFT seed. The validation split selected each "
                       "cell's checkpoint, so these numbers are in-sample and cannot establish "
                       "benefit; the locked primary read stays on the closed test parts.")}
    for name, doc in cells.items():
        # Every cell given is reported, so every one is held to this, in a contrast or not.
        extra = sorted(set(doc["conditions"]) - {CONDITION})
        if extra:
            raise PC.ComparisonError(f"{name}: also scored {extra}; the pilot keeps the test parts closed")
        c = doc["conditions"].get(CONDITION) or {}
        out["cells"][name] = {k: c.get(k) for k in
                              ("n_queries", "n_peptides", "recall_at_10", "mrr", "n_truncated_seqs")}
        out["cells"][name]["cpt_adapter"] = (doc.get("provenance") or {}).get("cpt_adapter")
        out["cells"][name]["cpt_objective"] = (doc.get("provenance") or {}).get("cpt_objective")
    for contrast in contrasts:
        b_name, _, a_name = contrast.partition("-")     # "E-B" is E minus B
        if a_name not in cells or b_name not in cells:
            raise PC.ComparisonError(f"contrast {contrast!r} names a cell that was not given")
        a, b = cells[a_name], cells[b_name]
        bad = problems_between(a_name, a, b_name, b)
        if bad:
            raise PC.ComparisonError("; ".join(bad))
        ranks_a = a["conditions"][CONDITION]["ranks"]
        ranks_b = b["conditions"][CONDITION]["ranks"]
        out["contrasts"][contrast] = {
            f"{metric}/{unit}": PC.compare_condition(
                CONDITION, ranks_a, ranks_b, units, a["pool_size"], metric=metric, unit=unit,
                n_boot=n_boot, seed=seed)
            for metric in METRICS for unit in ("peptide", "pep_cluster")}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", nargs="+", required=True, metavar="NAME=eval.json")
    ap.add_argument("--val-split", required=True)
    ap.add_argument("--contrasts", nargs="+", required=True, metavar="B-A", help="'E-B' is E minus B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=PC.N_BOOT)
    ap.add_argument("--seed", type=int, default=PC.SEED)
    a = ap.parse_args(argv)
    if pathlib.Path(a.out).exists():
        ap.error(f"{a.out} exists; a comparison never overwrites one")
    cells = {}
    for item in a.cells:
        name, sep, path = item.partition("=")
        if not sep or not name:
            ap.error(f"--cells takes NAME=eval.json, not {item!r}")
        if name in cells:
            ap.error(f"--cells names {name!r} twice")
        cells[name] = PC.load_eval(path)
    try:
        report = compare(cells, units_from_split(a.val_split), a.contrasts, a.n_boot, a.seed)
    except PC.ComparisonError as e:
        print(f"REFUSED: {e}")
        return PC.EXIT_ERROR
    report["cell_paths"] = {n: p for n, _s, p in (i.partition("=") for i in a.cells)}
    with open(a.out, "x") as fh:
        json.dump(report, fh, indent=1)
        fh.write("\n")
    for contrast, results in report["contrasts"].items():
        for key, r in results.items():
            # rho is None when one arm's per-unit means have no variance, which is information
            # about the pairing, not a number to format.
            rho = "n/a" if r["measured_rho"] is None else f"{r['measured_rho']:.3f}"
            print(f"{contrast:>5} {key:<24} {r['point']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] "
                  f"n={r['n_molecules']} units={r['n_units']} rho={rho}")
    print(report["reading"])
    return PC.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
