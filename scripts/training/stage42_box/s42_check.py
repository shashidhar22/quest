"""Stage 4.2 wave-1 gate and report.

  gate --eval <base cell's eval.json> [--condition test_level4]
      The harness check from plans/stage4-2.md, on the condition that HAS signal. Three outcomes,
      and only BROKEN is worth stopping the box for:

        PASS          CI lower bound clears random  -> the harness, head and pool all work
        UNDERPOWERED  CI straddles random           -> the data cannot resolve it; report, continue
        BROKEN        CI entirely at/below random,
                      or Recall@10 = 0, or MRR ~ 1/pool (every candidate tied)

      Gated on `test_level4` since 2026-10-15. It used to gate on `test_iid`, whose 201 molecules
      after de-duplication cannot separate 0.144 from 0.100: on 2026-10-08 that reported "the
      harness is broken" about a working harness (test_level4 0.337, CI [0.267, 0.406], val AUC
      0.702) and skipped wave-1 cells 2-4 for nothing. See quest-cpt/findings.md 9.6.

      BROKEN is a statement about the harness only on the harness condition. `test_novel_pep` sits
      significantly below random by construction -- its distractors come from a training-dominated
      pool while its true peptide is unseen -- so gating there would always read BROKEN.

      This condition tests the harness against CHANCE. It does NOT license reading a
      base-versus-CPT comparison: test_level4 clears chance decisively and its interval is still
      0.46 wide. That second question is the readability predicate below.

  gate ... --units <row->peptide/cluster csv> [--prereg-delta 0.05]
      The SECOND condition: can any (condition, metric) pairing in this eval resolve the
      pre-registered effect? Paired, clustered over peptides, reported against the unmeasurable
      cross-arm correlation rho. Returns nonzero when nothing is readable at rho <= 0.9, which is
      the only honest reason to refuse the remaining cells. See readability.py and
      quest-cpt/box-jobs/prereg-4.2-effect.md.

  report --dir <wave-1 out dir>
      One table of every cell that finished: molecules, PEPTIDES, Recall@10 with CI, MRR, and the
      underpowered flag. The power floor is peptides, never queries -- `--min_queries` was removed
      from the judge on 2026-11-03 and the old "n < 500 queries" floor here went with it, because
      molecules nest within peptides (findings.md 9.6).
"""
import argparse
import json
import pathlib
import sys

CELLS = [("base_paired", "base", "as_paired_i"), ("armD_paired", "arm D", "as_paired_i"),
         ("base_trb", "base", "as_trb_i"), ("armD_trb", "arm D", "as_trb_i")]
CONDS = ["test_level4", "test_novel_pep", "test_iid"]


def _conds(path):
    d = json.loads(pathlib.Path(path).read_text())
    return d.get("conditions", d)


def _pool_size(path, default):
    """POOL_SIZE the eval actually used; the tie signature is MRR == 1/pool."""
    d = json.loads(pathlib.Path(path).read_text())
    return d.get("pool_size") or default


def gate(args):
    cond = args.condition
    try:
        cs = _conds(args.eval)
        m = cs[cond]
    except Exception as e:
        print(f"[gate] ERROR reading {cond} from {args.eval}: {e!r}")
        return 1
    rnd = m.get("random_recall_at_10") or 0.10
    r, n, mrr = m.get("recall_at_10"), m.get("n_queries"), m.get("mrr")
    npep = m.get("n_peptides")       # the real unit; absent on evals from before 2026-10-15
    ci = m.get("recall_at_10_ci95") or [None, None]
    lo, hi = ci[0], ci[1]
    if r is None or lo is None or hi is None:
        print(f"[gate] ERROR: no recall/CI on {cond} (n={n})")
        return 1

    # Every candidate tied is the 2026-10-07 bf16 failure: rank is always the pool size, so
    # Recall@10 collapses to 0 and MRR to exactly 1/pool. Catch it by name, not by the CI.
    pool = _pool_size(args.eval, args.pool_size)
    tied = r == 0.0 or (mrr is not None and abs(mrr - 1.0 / pool) < 1e-6)
    if tied:
        verdict, rc = "BROKEN", 1
        why = f"Recall@10={r:.4f}, MRR={mrr:.4f} ~ 1/{pool} — every candidate scored alike"
    elif lo > rnd:
        verdict, rc = "PASS", 0
        why = "CI lower bound clears random"
    elif hi <= rnd:
        verdict, rc = "BROKEN", 1
        why = "CI lies entirely at or below random"
    else:
        verdict, rc = "UNDERPOWERED", 0
        why = (f"CI straddles random; n={n} cannot resolve {r:.4f} from {rnd:.4f}. "
               "Reported, not failed — this is the data, not a harness fault")

    unit = f"n={n} molecules" + (f", {npep} peptides" if npep else ", peptides NOT REPORTED")
    print(f"[gate] {verdict}: harness vs chance — base {cond} {unit} recall@10={r:.4f} "
          f"CI=[{lo:.4f}, {hi:.4f}] vs random {rnd:.4f} ({why})")
    if not npep:
        print("[gate]   note: this eval predates n_peptides; its CI unit cannot be confirmed here.")
    for other in (c for c in CONDS if c != cond and c in cs):
        o = cs[other]
        oci = o.get("recall_at_10_ci95") or [None, None]
        if o.get("recall_at_10") is not None and oci[0] is not None:
            print(f"[gate]   context {other}: n={o.get('n_queries')} "
                  f"recall@10={o['recall_at_10']:.4f} CI=[{oci[0]:.4f}, {oci[1]:.4f}]")
    if rc:
        print(f"[gate] {cond} is the harness condition — stopping before the remaining cells.")
        return rc
    if getattr(args, "units", None):
        rc = max(rc, _readability_gate(args, cs))
    else:
        print("[gate] READABILITY not evaluated: pass --units to check whether any condition "
              "can resolve the pre-registered effect. Passing the harness check does not "
              "license reading a base-versus-CPT comparison.")
    return rc


def _readability_gate(args, cs):
    """Second condition: is any (condition, metric) pairing able to see the pre-registered effect?"""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import csv as _csv

    import readability as R
    units = {}
    with open(args.units) as f:
        for row in _csv.DictReader(f):
            units[(row["condition"], row["row_id"])] = (row["peptide"], row["pep_cluster"])
    pool = args.pool_size
    # ONE lambda, fixed on the harness condition, so metrics are judged against one effect.
    anchor = cs.get(args.condition, {}).get("ranks") or {}
    if not anchor:
        print("[gate] READABILITY SKIPPED: no per-query ranks in this eval.")
        return 0
    lam = R.lambda_for_recall_delta(list(anchor.values()), pool, args.prereg_delta)
    if lam is None:
        print(f"[gate] READABILITY UNREACHABLE: {args.prereg_delta:+.3f} Recall@10 is not "
              f"attainable on {args.condition} by any rank improvement.")
        return 1
    print(f"[gate] READABILITY: one lambda={lam:.3f} (= {100*(lam-1):.0f}% rank gain) fixed on "
          f"{args.condition} at the pre-registered {args.prereg_delta:+.3f} Recall@10")
    best = None
    for cond, m in cs.items():
        ranks_d = m.get("ranks")
        if not ranks_d:
            continue
        rk, pp, cl = [], [], []
        for rid, rank in ranks_d.items():
            if (cond, rid) not in units:
                continue
            rk.append(rank)
            p, k = units[(cond, rid)]
            pp.append(p)
            cl.append(k)
        if not rk:
            continue
        for uname, uu in (("peptide", pp), ("pep_cluster", cl)):
            for metric in R.METRICS:
                _, se, _, _, _, _ = R.cluster_bootstrap(rk, uu, pool, metric)
                eff = R.effect_under_lambda(rk, uu, pool, lam, metric)
                rho = next((x for x in (0.0, 0.5, 0.8, 0.9, 0.95)
                            if eff >= R.paired_mde(se, x)), None)
                print(f"[gate]   {cond:15s} {uname:11s} n={len(set(uu)):4d} {metric:12s} "
                      f"effect={eff:+.4f} MDE(rho=.8)={R.paired_mde(se, 0.8):.4f} "
                      f"readable at rho>={rho}")
                if rho is not None and (best is None or rho < best[0]):
                    best = (rho, cond, uname, metric)
    if best is None:
        print("[gate] READABILITY UNREADABLE: no condition, unit and metric resolves the "
              "pre-registered effect even at rho=0.95. A comparison must not be read, and the "
              "remaining cells cannot answer the question as specified.")
        return 1
    rho, cond, uname, metric = best
    verdict = "READABLE" if rho <= 0.9 else "MARGINAL"
    print(f"[gate] READABILITY {verdict}: best pairing is {cond} / {uname} / {metric}, "
          f"readable at rho>={rho}. Read the comparison THERE, not on the harness condition.")
    return 0 if rho <= 0.9 else 1


def report(args):
    root = pathlib.Path(args.dir)
    rows = []
    for d, arm, task in CELLS:
        p = root / d / "eval.json"
        if not p.exists():
            rows.append((arm, task, "-", "not run", "", "", ""))
            continue
        try:
            cs = _conds(p)
        except Exception as e:
            rows.append((arm, task, "-", f"unreadable: {e!r}", "", "", ""))
            continue
        for cond in CONDS:
            m = cs.get(cond)
            if not m:
                continue
            r, ci = m.get("recall_at_10"), m.get("recall_at_10_ci95") or [None, None]
            rows.append((arm, task, cond, f"{m.get('n_queries')}/{m.get('n_peptides') or '?'}",
                         "n/a" if r is None else f"{r:.4f}",
                         "" if ci[0] is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]",
                         "UNDERPOWERED" if m.get("underpowered") else ""))
    w = [max(len(str(r[i])) for r in rows + [("arm", "task", "condition", "mol/pep", "recall@10", "CI95", "flag")])
         for i in range(7)]
    head = ("arm", "task", "condition", "mol/pep", "recall@10", "CI95", "flag")
    print("  ".join(h.ljust(w[i]) for i, h in enumerate(head)))
    print("  ".join("-" * w[i] for i in range(7)))
    for r in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
    for name in ("paired", "trb"):
        f = root / f"compare_{name}.txt"
        if f.exists():
            print(f"\n[compare {name}] (B = arm D, A = base; positive delta favours CPT)")
            print("\n".join("  " + l for l in f.read_text().strip().splitlines()))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gate"); g.add_argument("--eval", required=True)
    g.add_argument("--condition", default="test_level4", choices=CONDS)
    g.add_argument("--pool-size", dest="pool_size", type=int, default=100)
    g.add_argument("--units", help="csv: condition,row_id,peptide,pep_cluster — enables the "
                                   "readability condition")
    g.add_argument("--prereg-delta", dest="prereg_delta", type=float, default=0.05,
                   help="pre-registered Recall@10 effect; see box-jobs/prereg-4.2-effect.md")
    r = sp.add_parser("report"); r.add_argument("--dir", required=True)
    a = ap.parse_args()
    sys.exit({"gate": gate, "report": report}[a.cmd](a))
