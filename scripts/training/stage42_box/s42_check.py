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

  report --dir <wave-1 out dir>
      One table of every cell that finished: n, Recall@10 with CI, MRR, and whether the cell is
      underpowered (n < 500 queries, where a bootstrap CI cannot separate anything).
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

    print(f"[gate] {verdict}: base {cond} n={n} recall@10={r:.4f} "
          f"CI=[{lo:.4f}, {hi:.4f}] vs random {rnd:.4f} ({why})")
    for other in (c for c in CONDS if c != cond and c in cs):
        o = cs[other]
        oci = o.get("recall_at_10_ci95") or [None, None]
        if o.get("recall_at_10") is not None and oci[0] is not None:
            print(f"[gate]   context {other}: n={o.get('n_queries')} "
                  f"recall@10={o['recall_at_10']:.4f} CI=[{oci[0]:.4f}, {oci[1]:.4f}]")
    if rc:
        print(f"[gate] {cond} is the harness condition — stopping before the remaining cells.")
    return rc


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
            rows.append((arm, task, cond, str(m.get("n_queries")),
                         "n/a" if r is None else f"{r:.4f}",
                         "" if ci[0] is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]",
                         "UNDERPOWERED" if m.get("underpowered") else ""))
    w = [max(len(str(r[i])) for r in rows + [("arm", "task", "condition", "n", "recall@10", "CI95", "flag")])
         for i in range(7)]
    head = ("arm", "task", "condition", "n", "recall@10", "CI95", "flag")
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
    r = sp.add_parser("report"); r.add_argument("--dir", required=True)
    a = ap.parse_args()
    sys.exit({"gate": gate, "report": report}[a.cmd](a))
