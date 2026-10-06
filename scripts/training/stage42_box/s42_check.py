"""Stage 4.2 wave-1 gate and report.

  gate --eval <base cell's eval.json>
      Exit 0 only if the base arm beats the analytic random Recall@10 on test_iid with its 95 %
      CI clear of it, and the condition is not underpowered. This is the harness check from
      plans/stage4-2.md: an in-distribution split the model has every chance on. Failing it means
      the judge is broken, so no later number is worth the GPU time.

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


def gate(args):
    try:
        m = _conds(args.eval)["test_iid"]
    except Exception as e:
        print(f"[gate] ERROR reading test_iid from {args.eval}: {e!r}")
        return 1
    rnd = m.get("random_recall_at_10") or 0.10
    r, n = m.get("recall_at_10"), m.get("n_queries")
    ci = m.get("recall_at_10_ci95") or [None, None]
    lo, hi = ci[0], ci[1]
    if r is None or lo is None or hi is None:
        print(f"[gate] ERROR: no recall/CI on test_iid (n={n})")
        return 1
    if m.get("underpowered"):
        print(f"[gate] FAIL: test_iid underpowered, n={n} (<500): a CI here separates nothing")
        return 1
    ok = lo > rnd
    print(f"[gate] {'PASS' if ok else 'FAIL'}: base test_iid n={n} recall@10={r:.4f} "
          f"CI=[{lo:.4f}, {hi:.4f}] vs random {rnd:.4f} "
          f"({'CI lower bound clears random' if ok else 'CI touches or falls below random'})")
    return 0 if ok else 1


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
    r = sp.add_parser("report"); r.add_argument("--dir", required=True)
    a = ap.parse_args()
    sys.exit({"gate": gate, "report": report}[a.cmd](a))
