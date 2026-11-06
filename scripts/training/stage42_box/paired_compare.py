"""The paired base-versus-CPT comparison, on the locked endpoint and the declared unit.

This replaces what `judge.py compare` computed. That function (audit finding 4, 2026-11-04)
intersected row IDs silently, took differences of Recall@10 indicators, and bootstrapped over
ROWS. So it measured a different quantity from the one the gate licenses, on a resampling unit
that treats molecules of one peptide as independent observations, which makes its interval too
narrow by construction. The launcher never invoked it at all, so a completed visit produced
evaluation files and no comparison.

What this module computes, and why each choice is forced:

1. **The unit is the peptide.** Molecules nest within peptides and move together, so resampling
   molecules manufactures precision. The auditor's case makes it concrete: 100 molecules in two
   peptides, every molecule of one improving, gives a row bootstrap a resolved +0.50; resampling
   peptides must admit the draw in which only the flat peptide appears, so the interval reaches
   zero. Two peptides is two observations.
2. **The estimator stays molecule-weighted.** The eval reports a molecule-weighted mean, and
   changing the estimand while fixing the variance would be a different quantity again. So
   peptides are resampled, and within a draw every molecule of a drawn peptide contributes.
3. **The comparison is paired per molecule.** Both arms score the same rows against the same
   pool, so peptide difficulty cancels and the variance that matters is that of the difference.
4. **rho is MEASURED, not assumed.** The pre-run power analysis assumed 0.8 against a measured
   breakeven of 0.765. That assumption was a planning parameter; it is not evidence about a
   result, so the cross-arm peptide-level correlation is reported from the data.
5. **One primary endpoint, the rest secondary** (decision 3, 2026-11-04). The primary is
   `test_novel_pep / peptide / norm_rank`, locked before any CPT cell existed. Recall@10, MRR and
   the other conditions are reported and labelled secondary. A normalized-rank effect does NOT
   establish the separately declared +0.05 Recall@10 utility target, and this module says so in
   its own output so a reader cannot quietly conflate them.
6. **Validate, never guess.** Unequal query sets, ranks outside the pool, rows missing from the
   unit map, and incompatible pool/draw/task/serialization metadata all refuse. An exclusion may
   be tolerated only when the caller asks for it, and is then counted in the output.

Stdlib only, deliberately, like `readability.py`: the judge pulls in the trainer and therefore
peft, so anything living there cannot be exercised away from the box. Every verdict that cost
this programme a visit was produced by code that could not be run on a laptop.
"""
import argparse
import csv
import json
import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "v04_box"))

import provenance as PROV  # noqa: E402  (stdlib only; the same rule the launcher applies)
import readability as R  # noqa: E402  (metrics, by_peptide, weighted_mean; stdlib only)

EXIT_OK = 0        # a valid comparison was produced; its content may be inconclusive
EXIT_ERROR = 2     # no valid comparison could be produced. Same codes as s42_check.

N_BOOT = 4000
SEED = 42
# The locked primary read. Changing it means editing box-jobs/README.md first.
PRIMARY = ("test_novel_pep", "peptide", "norm_rank")
SECONDARY_METRICS = ("recall_at_10", "mrr")
# The practical-utility target, declared separately from the inferential endpoint.
UTILITY_DELTA = 0.05
UTILITY_METRIC = "recall_at_10"
UTILITY_CONDITION = "test_level4"
# Metadata that must agree for two evals to be comparable at all.
META_KEYS = ("pool_size", "distractor_draw", "task", "tcr_variant", "mhc_variant")


class ComparisonError(Exception):
    """Raised when no valid comparison exists. Never downgraded to a printed caveat."""


def _val(metric, rank, pool, k=10):
    f = R.METRICS[metric]
    return f(rank, pool, k) if metric == "recall_at_10" else f(rank, pool)


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None          # one arm is constant across units; rho is undefined, not 0 or 1
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def load_eval(path):
    d = json.loads(pathlib.Path(path).read_text())
    if "conditions" not in d:
        raise ComparisonError(f"{path} has no 'conditions' block")
    return d


def meta_of(doc):
    """The comparison-relevant metadata, from wherever the evaluator recorded it."""
    args = doc.get("args") or {}
    out = {"pool_size": doc.get("pool_size")}
    for k in ("distractor_draw", "task", "tcr_variant", "mhc_variant"):
        out[k] = args.get(k)
    return out


def check_metadata(a, b):
    """Offending keys, as printable strings. Empty means comparable.

    REQUIRED, not merely equal (audit finding 1, 2026-11-06). Until the repair this compared
    `ma.get(k) != mb.get(k)`, so a field missing from both evals read as agreement: an eval that
    never recorded its distractor draw was comparable with any other, although a uniform-draw
    number and a length-matched one are not comparable at all. Absence is the one thing that
    must never pass a compatibility check, because it is indistinguishable from a mismatch.
    """
    ma, mb = meta_of(a), meta_of(b)
    bad = []
    for k in META_KEYS:
        for lbl, m in (("arm A", ma), ("arm B", mb)):
            if m.get(k) in (None, ""):
                bad.append(f"{k}: {lbl} did not record it, so agreement cannot be established")
        if ma.get(k) != mb.get(k):
            bad.append(f"{k}: {ma.get(k)!r} vs {mb.get(k)!r}")
    if ma.get("pool_size") in (None, 0):
        bad.append("pool_size: missing on arm A; every metric divides by it")
    return bad


def load_units(path):
    """{(condition, row_id): (peptide, pep_cluster)}."""
    units = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            units[(row["condition"], row["row_id"])] = (row["peptide"], row["pep_cluster"])
    return units


def naive_row_se(ranks_a, ranks_b, pool, metric="recall_at_10", n_boot=N_BOOT, seed=SEED):
    """The WRONG resampling unit, kept so the inflation is measurable rather than asserted."""
    ids = sorted(set(ranks_a) & set(ranks_b))
    d = [_val(metric, ranks_b[i], pool) - _val(metric, ranks_a[i], pool) for i in ids]
    if not d:
        return float("nan")
    rng = random.Random(seed)
    draws = [R.weighted_mean([d[rng.randrange(len(d))] for _ in d]) for _ in range(n_boot)]
    mean = sum(draws) / len(draws)
    return math.sqrt(sum((x - mean) ** 2 for x in draws) / (len(draws) - 1))


def compare_condition(condition, ranks_a, ranks_b, units, pool, metric="norm_rank",
                      unit="peptide", n_boot=N_BOOT, seed=SEED, allow_partial_pairing=False):
    """The paired estimate for one (condition, metric, unit).

    Returns a dict with point, se, ci_lo, ci_hi, verdict, n_molecules, n_units, measured_rho and
    the pairing accounting. Raises ComparisonError rather than returning a degraded answer.
    """
    ids_a, ids_b = set(ranks_a), set(ranks_b)
    only_a, only_b = sorted(ids_a - ids_b), sorted(ids_b - ids_a)
    if (only_a or only_b) and not allow_partial_pairing:
        raise ComparisonError(
            f"{condition}: incomplete pairing — {len(only_a)} row(s) only in A, "
            f"{len(only_b)} only in B, e.g. {(only_a or only_b)[:5]}. A paired comparison needs "
            f"the same queries in both arms; pass allow_partial_pairing to exclude them on "
            f"purpose and have the exclusions counted."
        )
    ids = sorted(ids_a & ids_b)
    if not ids:
        raise ComparisonError(f"{condition}: no rows are present in both arms")

    bad_rank = [(i, ranks_a[i], ranks_b[i]) for i in ids
                if not all(isinstance(r, (int, float)) and math.isfinite(r) and 1 <= r <= pool
                           for r in (ranks_a[i], ranks_b[i]))]
    if bad_rank:
        raise ComparisonError(
            f"{condition}: {len(bad_rank)} row(s) carry a rank outside [1, {pool}] or "
            f"non-finite, e.g. {bad_rank[:3]}"
        )
    missing = [i for i in ids if (condition, i) not in units]
    if missing:
        raise ComparisonError(
            f"{condition}: unit map coverage is incomplete — {len(missing)} of {len(ids)} paired "
            f"rows have no peptide/cluster, e.g. {missing[:5]}. Dropping them would shrink the "
            f"comparison's support silently."
        )

    idx = 0 if unit == "peptide" else 1
    key = {i: units[(condition, i)][idx] for i in ids}
    # Paired per-molecule differences, grouped by the resampling unit.
    diffs = {}
    per_unit_a, per_unit_b = {}, {}
    for i in ids:
        u = key[i]
        va = _val(metric, ranks_a[i], pool)
        vb = _val(metric, ranks_b[i], pool)
        diffs.setdefault(u, []).append(vb - va)
        per_unit_a.setdefault(u, []).append(va)
        per_unit_b.setdefault(u, []).append(vb)

    keys = list(diffs)
    point = R.weighted_mean([v for u in keys for v in diffs[u]])
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        pooled = []
        for _ in range(len(keys)):
            pooled.extend(diffs[keys[rng.randrange(len(keys))]])
        draws.append(R.weighted_mean(pooled))
    draws.sort()
    mean = sum(draws) / len(draws)
    se = math.sqrt(sum((x - mean) ** 2 for x in draws) / (len(draws) - 1))
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    if not all(math.isfinite(x) for x in (point, se, lo, hi)):
        raise ComparisonError(f"{condition}: the estimate is non-finite (point={point}, se={se})")

    rho = _pearson([R.weighted_mean(per_unit_a[u]) for u in keys],
                   [R.weighted_mean(per_unit_b[u]) for u in keys])
    verdict = "B > A" if lo > 0 else ("B < A" if hi < 0 else "no difference resolved")
    return {
        "condition": condition, "metric": metric, "unit": unit,
        "point": point, "se": se, "ci_lo": lo, "ci_hi": hi, "verdict": verdict,
        "n_molecules": len(ids), "n_units": len(keys),
        "measured_rho": rho,
        "n_only_in_a": len(only_a), "n_only_in_b": len(only_b),
        "n_boot": n_boot, "seed": seed,
    }


def _fmt(r):
    rho = "unmeasurable" if r["measured_rho"] is None else f"{r['measured_rho']:+.3f}"
    return (f"{r['condition']:15s} {r['unit']:11s} {r['metric']:12s} "
            f"n={r['n_molecules']:5d} mol {r['n_units']:4d} {r['unit']} "
            f"delta={r['point']:+.4f} SE={r['se']:.4f} "
            f"95% CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] rho={rho}  -> {r['verdict']}")


def compare(a_path, b_path, units_path, primary=PRIMARY, n_boot=N_BOOT, seed=SEED,
            allow_partial_pairing=False, out=None):
    """The whole report. Returns (exit_code, result dict)."""
    a, b = load_eval(a_path), load_eval(b_path)
    # The comparator applies the provenance rule ITSELF rather than trusting a launcher to have
    # done it: `judge compare` can be run by hand on any two files, and the 2026-11-06 audit
    # found exactly that gap. Same function the launcher calls, so there is one rule.
    prov_bad, prov_tolerated = PROV.check_eval_provenance(a, b)
    bad = check_metadata(a, b) + prov_bad
    if bad:
        print("[cmp] REFUSED: these evaluations are not comparable")
        for line in bad:
            print(f"[cmp]   {line}")
        print("[cmp] No comparison was produced. Fix the mismatch; do not reinterpret it.")
        return EXIT_ERROR, None
    pool = int(meta_of(a)["pool_size"])
    units = load_units(units_path)
    pc, pu, pm = primary

    ca, cb = a["conditions"], b["conditions"]
    if pc not in ca or pc not in cb:
        print(f"[cmp] REFUSED: the locked primary condition {pc} is absent from "
              f"{'arm A' if pc not in ca else 'arm B'}. Not substituting another condition — "
              f"that is the selection surface the lock exists to close.")
        return EXIT_ERROR, None

    def ranks(doc_conds, cond):
        return doc_conds.get(cond, {}).get("ranks") or {}

    try:
        prim = compare_condition(pc, ranks(ca, pc), ranks(cb, pc), units, pool,
                                 metric=pm, unit=pu, n_boot=n_boot, seed=seed,
                                 allow_partial_pairing=allow_partial_pairing)
        cluster = compare_condition(pc, ranks(ca, pc), ranks(cb, pc), units, pool,
                                    metric=pm, unit="pep_cluster", n_boot=n_boot, seed=seed,
                                    allow_partial_pairing=allow_partial_pairing)
        secondary = []
        for cond in sorted(set(ca) & set(cb)):
            if not (ranks(ca, cond) and ranks(cb, cond)):
                continue
            for metric in SECONDARY_METRICS + ((pm,) if cond != pc else ()):
                secondary.append(
                    compare_condition(cond, ranks(ca, cond), ranks(cb, cond), units, pool,
                                      metric=metric, unit=pu, n_boot=n_boot, seed=seed,
                                      allow_partial_pairing=allow_partial_pairing))
    except ComparisonError as e:
        print(f"[cmp] REFUSED: {e}")
        return EXIT_ERROR, None

    print(f"[cmp] pool={pool} draw={meta_of(a)['distractor_draw']} task={meta_of(a)['task']} "
          f"serialization={meta_of(a)['tcr_variant']}x{meta_of(a)['mhc_variant']}")
    for line in prov_tolerated:
        print(f"[cmp]   TREATMENT DIFFERENCE (expected, and what the result is about): {line}")
    print(f"[cmp] PRIMARY (locked before any CPT cell existed): {_fmt(prim)}")
    if prim["n_only_in_a"] or prim["n_only_in_b"]:
        print(f"[cmp]   pairing: {prim['n_only_in_a']} excluded from A, "
              f"{prim['n_only_in_b']} from B, by explicit request")
    print(f"[cmp]   cluster sensitivity: {_fmt(cluster)}")
    if cluster["verdict"] != prim["verdict"]:
        print("[cmp]   NOTE: the cluster read does not agree with the peptide read. The peptide "
              "unit is not established as independent (findings 9.4, 9.6), so this disagreement "
              "is part of the result, not a footnote.")
    rho_txt = ("unmeasurable" if prim["measured_rho"] is None
               else f"{prim['measured_rho']:+.3f}")
    print(f"[cmp]   measured cross-arm rho = {rho_txt}; the pre-run analysis ASSUMED 0.80 against "
          f"a breakeven of 0.765. That assumption was a planning parameter and is not evidence "
          f"about this result.")
    print("[cmp] SECONDARY (reported, not the inferential endpoint):")
    for r in secondary:
        print(f"[cmp]   {_fmt(r)}")
    print(f"[cmp] The primary endpoint is a normalized-rank effect. It does not establish the "
          f"separately declared practical-utility target of {UTILITY_DELTA:+.3f} "
          f"{UTILITY_METRIC} on {UTILITY_CONDITION}: a mean-rank improvement can occur with no "
          f"top-ten crossings, and the common-lambda mapping used for pre-run power is a modelling "
          f"assumption, not an observed equivalence. Report both; conflate neither.")

    result = {"pool_size": pool, "metadata": meta_of(a), "primary": prim,
              "cluster_sensitivity": cluster, "secondary": secondary,
              "utility_target": {"delta": UTILITY_DELTA, "metric": UTILITY_METRIC,
                                 "condition": UTILITY_CONDITION, "established": False},
              # What produced each side, carried INTO the result: a comparison read later
              # without its provenance is the failure family this programme started with.
              "provenance": {"arm_a": a.get("provenance"), "arm_b": b.get("provenance"),
                             "treatment_differences": prov_tolerated}}
    if out:
        pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(out).write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"[cmp] wrote {out}")
    return EXIT_OK, result


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--a", required=True, help="arm A eval.json (the base)")
    ap.add_argument("--b", required=True, help="arm B eval.json (the CPT arm)")
    ap.add_argument("--units", required=True,
                    help="csv: condition,row_id,peptide,pep_cluster")
    ap.add_argument("--out", help="write the result as json")
    ap.add_argument("--primary-condition", dest="pc", default=PRIMARY[0])
    ap.add_argument("--primary-unit", dest="pu", default=PRIMARY[1],
                    choices=["peptide", "pep_cluster"])
    ap.add_argument("--primary-metric", dest="pm", default=PRIMARY[2],
                    choices=sorted(R.METRICS))
    ap.add_argument("--n-boot", dest="n_boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--allow-partial-pairing", action="store_true",
                    help="exclude rows missing from one arm ON PURPOSE; the exclusions are "
                         "counted in the output. Without this an unequal query set refuses.")
    a = ap.parse_args()
    try:
        rc, _ = compare(a.a, a.b, a.units, primary=(a.pc, a.pu, a.pm), n_boot=a.n_boot,
                        seed=a.seed, allow_partial_pairing=a.allow_partial_pairing, out=a.out)
    except ComparisonError as e:
        print(f"[cmp] REFUSED: {e}")
        return EXIT_ERROR
    return rc


if __name__ == "__main__":
    sys.exit(main())
