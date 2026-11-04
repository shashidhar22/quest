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
import math
import pathlib
import sys

# THREE outcomes, because two conflated a spending decision with a software fault. The audit of
# 2026-11-04 found the consequence: with `--units` passed but no ranks in the eval, the readability
# gate printed SKIPPED and returned 0, so a visit proceeded on absent evidence; and a technical
# failure was indistinguishable from the pre-agreed refusal, so the launcher treated both as "the
# science says stop". A caller must be able to tell them apart, and run_v04.sh now does.
EXIT_OK = 0        # the locked read is available and affordable: proceed
EXIT_REFUSED = 1   # SCIENTIFIC: the data cannot resolve the pre-registered effect, or retrieval
                   # does not clear chance. A spending decision, not a bug.
EXIT_ERROR = 2     # TECHNICAL: missing or malformed evidence, incompatible metadata, software
                   # fault. Must stop the visit; must never read as a refusal.

CELLS = [("base_paired", "base", "as_paired_i"), ("armD_paired", "arm D", "as_paired_i"),
         ("base_trb", "base", "as_trb_i"), ("armD_trb", "arm D", "as_trb_i")]
CONDS = ["test_level4", "test_novel_pep", "test_iid"]


def _load(path):
    return json.loads(pathlib.Path(path).read_text())


def _conds(path):
    d = _load(path)
    return d.get("conditions", d)


def _pool_size(path, cli_default):
    """The pool the eval ACTUALLY used.

    Previously this fell back to the CLI value, so the gate could score against a pool the eval
    never used: random Recall@10 is k/pool and the tie signature is MRR == 1/pool, so both the
    verdict and the fault check depend on it. It is now required, and a CLI value that disagrees
    is a refusal to guess.
    """
    pool = _load(path).get("pool_size")
    if not pool:
        raise ValueError(
            "this eval records no pool_size, so random Recall@10 and the tie signature cannot "
            "be computed; re-score with an evaluator that records it"
        )
    if cli_default is not None and int(cli_default) != int(pool):
        raise ValueError(
            f"--pool-size {cli_default} disagrees with the pool the eval used ({pool}); "
            "the eval's own value is authoritative, so fix the caller"
        )
    return int(pool)


def _finite(**named):
    """Names of the given statistics that are present but not finite."""
    return [k for k, v in named.items()
            if v is not None and isinstance(v, float) and not math.isfinite(v)]


def gate(args):
    cond = args.condition
    try:
        cs = _conds(args.eval)
        m = cs[cond]
    except Exception as e:
        print(f"[gate] ERROR reading {cond} from {args.eval}: {e!r}")
        return EXIT_ERROR
    rnd = m.get("random_recall_at_10") or 0.10
    r, n, mrr = m.get("recall_at_10"), m.get("n_queries"), m.get("mrr")
    npep = m.get("n_peptides")       # the real unit; absent on evals from before 2026-10-15
    ci = m.get("recall_at_10_ci95") or [None, None]
    lo, hi = ci[0], ci[1]
    if r is None or lo is None or hi is None:
        print(f"[gate] ERROR: no recall/CI on {cond} (n={n})")
        return EXIT_ERROR
    bad = _finite(recall_at_10=r, ci_lo=lo, ci_hi=hi, mrr=mrr, random_recall_at_10=rnd)
    if bad:
        print(f"[gate] ERROR: non-finite statistic on {cond}: {', '.join(bad)}")
        return EXIT_ERROR
    try:
        pool = _pool_size(args.eval, getattr(args, "pool_size", None))
    except ValueError as e:
        print(f"[gate] ERROR: {e}")
        return EXIT_ERROR

    # SOFTWARE check, kept narrow and direct. Every candidate tied is the 2026-10-07 bf16 failure:
    # every rank equals the pool size, so MRR is EXACTLY 1/pool. That signature is a fault.
    # Recall@10 == 0 is NOT: a working scorer can vary across candidates and still never place the
    # true peptide in the top ten, so reading zero recall as "every score tied" licensed a false
    # conclusion about the software (audit finding 7).
    if mrr is not None and abs(mrr - 1.0 / pool) < 1e-6:
        print(f"[gate] TIED: harness fault — base {cond} MRR={mrr:.4f} is exactly 1/{pool}, so "
              f"every candidate scored alike and rank carries no information (Recall@10={r:.4f})")
        return EXIT_ERROR

    if lo > rnd:
        verdict, rc = "PASS", EXIT_OK
        why = "CI lower bound clears random"
    elif hi <= rnd:
        # A working model can sit at or below chance, especially under a changed distractor draw.
        # So this stops the chain as a SCIENTIFIC refusal, not as evidence of broken software.
        verdict, rc = "BELOW-CHANCE", EXIT_REFUSED
        why = ("CI lies entirely at or below random — a result, not a fault; no base-versus-CPT "
               "read on this condition would be interpretable")
    else:
        verdict, rc = "UNDERPOWERED", EXIT_OK
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
        rc = max(rc, _readability_gate(args, cs, pool))
    else:
        print("[gate] READABILITY not evaluated: pass --units to check whether any condition "
              "can resolve the pre-registered effect. Passing the harness check does not "
              "license reading a base-versus-CPT comparison.")
    return rc


def _readability_gate(args, cs, pool):
    """Second condition: is any (condition, metric) pairing able to see the pre-registered effect?

    `pool` is the eval's own recorded pool, passed in rather than read off the CLI: every metric
    here divides by it.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import csv as _csv

    import readability as R
    units = {}
    with open(args.units) as f:
        for row in _csv.DictReader(f):
            units[(row["condition"], row["row_id"])] = (row["peptide"], row["pep_cluster"])
    # ONE lambda, fixed on the harness condition, so metrics are judged against one effect.
    anchor = cs.get(args.condition, {}).get("ranks") or {}
    if not anchor:
        # THE 2026-11-04 fail-open. This printed SKIPPED and returned 0, so a visit that asked for
        # the locked read proceeded without it. A requested read that cannot be performed is
        # missing evidence, which is a technical error, not an affordability refusal.
        print(f"[gate] ERROR: --units was given, so the locked read was requested, but "
              f"{args.condition} carries no per-query ranks in {args.eval}. Re-score with an "
              f"evaluator that emits ranks; this is missing evidence, not a refusal.")
        return EXIT_ERROR
    bad_ranks = [rid for rid, rk in anchor.items()
                 if not isinstance(rk, (int, float)) or not math.isfinite(rk)
                 or rk < 1 or rk > pool]
    if bad_ranks:
        print(f"[gate] ERROR: {len(bad_ranks)} rank(s) on {args.condition} are not in [1, {pool}] "
              f"or are non-finite, e.g. {bad_ranks[:5]}")
        return EXIT_ERROR
    # Ranks absent from the unit map used to be dropped silently, shrinking the support of the read
    # without saying so. Coverage is now required for every condition that carries ranks.
    for cond, m in cs.items():
        rd = m.get("ranks") or {}
        missing = [rid for rid in rd if (cond, rid) not in units]
        if rd and missing:
            print(f"[gate] ERROR: unit map coverage is incomplete for {cond}: "
                  f"{len(missing)} of {len(rd)} ranked rows have no peptide/cluster, "
                  f"e.g. {missing[:5]}. Dropping them would shrink the read's support silently.")
            return EXIT_ERROR
    lam = R.lambda_for_recall_delta(list(anchor.values()), pool, args.prereg_delta)
    if lam is None:
        print(f"[gate] READABILITY UNREACHABLE: {args.prereg_delta:+.3f} Recall@10 is not "
              f"attainable on {args.condition} by any rank improvement.")
        return EXIT_REFUSED
    print(f"[gate] READABILITY: one lambda={lam:.3f} (= {100*(lam-1):.0f}% rank gain) fixed on "
          f"{args.condition} at the pre-registered {args.prereg_delta:+.3f} Recall@10")
    best = None
    per_pair = {}
    for cond, m in cs.items():
        ranks_d = m.get("ranks")
        if not ranks_d:
            continue
        rk, pp, cl = [], [], []
        for rid, rank in ranks_d.items():
            if (cond, rid) not in units:
                continue       # unreachable: coverage was validated above, kept as a guard
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
                be = R.breakeven_rho(se, eff)
                per_pair[(cond, uname, metric)] = (eff, se, be)
                print(f"[gate]   {cond:15s} {uname:11s} n={len(set(uu)):4d} {metric:12s} "
                      f"effect={eff:+.4f} MDE(rho=.8)={R.paired_mde(se, 0.8):.4f} "
                      f"breakeven rho={'n/a' if be is None else format(be, '.3f')}")
                if rho is not None and (best is None or rho < best[0]):
                    best = (rho, cond, uname, metric)
    # The LOCKED primary read decides the gate. "Something is readable" across twelve pairings is
    # a post-hoc selection surface; the pairing was locked on 2026-11-04 on pre-cell power, with no
    # CPT cell in existence. box-jobs/README.md 3.4.
    pc, pu, pm = args.primary_condition, args.primary_unit, args.primary_metric
    prim = per_pair.get((pc, pu, pm))
    if prim is None:
        print(f"[gate] ERROR: the locked primary read {pc}/{pu}/{pm} is not present "
              f"in this eval. Not substituting another pairing — that is the selection surface "
              f"the lock exists to close.")
        return EXIT_ERROR
    eff, se, be = prim
    nf = _finite(effect=eff, se=se, breakeven_rho=be)
    if nf:
        print(f"[gate] ERROR: the locked read produced non-finite {', '.join(nf)}; "
              f"the estimate cannot be compared with a floor")
        return EXIT_ERROR
    floor = args.rho_floor
    ok = be is not None and be <= floor
    print(f"[gate] READABILITY {'PASS' if ok else 'FAIL'} on the LOCKED primary read "
          f"{pc} / {pu} / {pm}: effect={eff:+.4f} SE={se:.4f} breakeven rho={be:.3f} "
          f"vs floor {floor:.2f}")
    if not ok:
        print(f"[gate]   the comparison is NOT readable unless cross-arm correlation exceeds "
              f"{be:.3f}. Nothing may be read off it, and the comparison cells cannot answer "
              f"the question. Refusing them is the correct outcome, not a failure of the run.")
    # Length stratification, in the SAME pass. Recall tracks peptide length against a pool that
    # is 76 % 9-mers, so the confound has to be read off the condition that carries the decision,
    # not checked later. 463 peptides is the first n in this programme that can be stratified --
    # and at the peptide unit every non-9 stratum is thin, which is itself the finding.
    rk, pp = [], []
    for rid, rank in (cs.get(pc, {}).get("ranks") or {}).items():
        if (pc, rid) in units:
            rk.append(rank)
            pp.append(units[(pc, rid)][0 if pu == "peptide" else 1])
    if rk and pu == "peptide":
        st = R.stratify(rk, pp, pool, pm)
        print(f"[gate]   length strata of {pc} on {pm} (confound check, same pass):")
        for length, (nmol, npep, val) in st.items():
            thin = "  <-- too thin to compare" if npep < 30 else ""
            print(f"[gate]     len {length:>2}: {nmol:5d} mol  {npep:4d} pep  {pm}={val:.4f}{thin}")

    if best is not None and best[1:] != (pc, pu, pm):
        print(f"[gate]   note: {best[1]}/{best[2]}/{best[3]} clears at a lower rho ({best[0]}). "
              f"Secondary only. Changing the primary read requires writing it into "
              f"box-jobs/README.md first.")
    # Affordability only. A pass licenses spending the comparison cells; it says nothing about the
    # outcome, and the +0.05 Recall@10 utility target remains a separate claim on its own condition
    # (decision 3, 2026-11-04). Reaching this line at all means the evidence was present and valid.
    return EXIT_OK if ok else EXIT_REFUSED


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
    g.add_argument("--pool-size", dest="pool_size", type=int, default=None,
                   help="Optional cross-check only. The eval's own recorded pool_size is "
                        "authoritative and required; passing a value that disagrees with it is a "
                        "technical error rather than an override, because every metric here "
                        "divides by the pool the eval actually used.")
    g.add_argument("--units", help="csv: condition,row_id,peptide,pep_cluster — enables the "
                                   "readability condition")
    g.add_argument("--prereg-delta", dest="prereg_delta", type=float, default=0.05,
                   help="pre-registered Recall@10 effect; see box-jobs/prereg-4.2-effect.md")
    g.add_argument("--primary-condition", dest="primary_condition", default="test_novel_pep")
    g.add_argument("--primary-unit", dest="primary_unit", default="peptide",
                   choices=("peptide", "pep_cluster"))
    g.add_argument("--primary-metric", dest="primary_metric", default="norm_rank",
                   choices=("recall_at_10", "mrr", "norm_rank"))
    g.add_argument("--rho-floor", dest="rho_floor", type=float, default=0.8,
                   help="the cross-arm correlation we are willing to assume; the locked read "
                        "must have a breakeven at or below this")
    r = sp.add_parser("report"); r.add_argument("--dir", required=True)
    a = ap.parse_args()
    sys.exit({"gate": gate, "report": report}[a.cmd](a))
