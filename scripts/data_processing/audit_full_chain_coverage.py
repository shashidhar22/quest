"""Audit TCR full-chain coverage across benchmark_v2 splits + foundation files.

Reports per-file counts of rows where CDR is populated but the corresponding
full chain (`tra_full` / `trb_full`) is missing — the upstream gap inherited
from the enrichment stage.

Outputs:
- data/benchmark_v2/audit/full_chain_coverage.json
- data/benchmark_v2/audit/full_chain_coverage_summary.md
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import duckdb

OUT = Path("/home/ubuntu/quest/data/benchmark_v2/audit")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=64")

    files = sorted(glob.glob("/home/ubuntu/quest/data/benchmark_v2/splits/*.parquet"))
    # Skip the lookup subdir and pair_negatives (which is just (tra,trb,label),
    # no full-chain columns).
    files = [f for f in files if "lookup" not in f and "pair_negatives" not in f]
    files += [
        "/home/ubuntu/quest/data/benchmark_v2/foundation/foundation_val.parquet",
        "/home/ubuntu/quest/data/benchmark_v2/foundation/foundation_test.parquet",
    ]

    per_file = []
    for f in files:
        rec = con.execute(f"""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE tra_cdr3 IS NOT NULL AND tra_cdr3 <> '') AS tra_cdr3_pop,
                COUNT(*) FILTER (WHERE tra_full IS NOT NULL AND tra_full <> '') AS tra_full_pop,
                COUNT(*) FILTER (WHERE tra_cdr3 IS NOT NULL AND tra_cdr3 <> '' AND tra_full IS NOT NULL AND tra_full <> '') AS tra_cdr3_with_full,
                COUNT(*) FILTER (WHERE tra_cdr3 IS NOT NULL AND tra_cdr3 <> '' AND (tra_full IS NULL OR tra_full = '')) AS tra_cdr3_no_full,
                COUNT(*) FILTER (WHERE tra_cdr1 IS NOT NULL AND tra_cdr1 <> '' AND (tra_full IS NULL OR tra_full = '')) AS tra_cdr1_no_full,
                COUNT(*) FILTER (WHERE trb_cdr3 IS NOT NULL AND trb_cdr3 <> '') AS trb_cdr3_pop,
                COUNT(*) FILTER (WHERE trb_full IS NOT NULL AND trb_full <> '') AS trb_full_pop,
                COUNT(*) FILTER (WHERE trb_cdr3 IS NOT NULL AND trb_cdr3 <> '' AND trb_full IS NOT NULL AND trb_full <> '') AS trb_cdr3_with_full,
                COUNT(*) FILTER (WHERE trb_cdr3 IS NOT NULL AND trb_cdr3 <> '' AND (trb_full IS NULL OR trb_full = '')) AS trb_cdr3_no_full,
                COUNT(*) FILTER (WHERE trb_cdr1 IS NOT NULL AND trb_cdr1 <> '' AND (trb_full IS NULL OR trb_full = '')) AS trb_cdr1_no_full
            FROM read_parquet('{f}')
        """).fetchone()
        cols = ["total", "tra_cdr3_pop", "tra_full_pop", "tra_cdr3_with_full",
                "tra_cdr3_no_full", "tra_cdr1_no_full", "trb_cdr3_pop", "trb_full_pop",
                "trb_cdr3_with_full", "trb_cdr3_no_full", "trb_cdr1_no_full"]
        d = dict(zip(cols, rec))
        d = {k: int(v) for k, v in d.items()}
        d["file"] = f.split("/")[-1]
        d["dir"] = "splits" if "/splits/" in f else "foundation"
        per_file.append(d)
        print(f"  {d['file']:<60s} total={d['total']:>12,d} tra_gap={d['tra_cdr3_no_full']:>10,d} trb_gap={d['trb_cdr3_no_full']:>10,d}", flush=True)

    def task_of(name: str) -> str:
        if name.startswith("as_"):
            return "as"
        if name.startswith("pm_"):
            return "pm"
        if name.startswith("pair_"):
            return "pair"
        if name.startswith("mr_"):
            return "mr"
        if name.startswith("foundation_"):
            return "foundation"
        return "other"

    tasks: dict = {}
    for r in per_file:
        t = task_of(r["file"])
        if t not in tasks:
            tasks[t] = {k: 0 for k in r if k not in ("file", "dir")}
        for k in tasks[t]:
            tasks[t][k] += r[k]

    overall = {k: sum(t[k] for t in tasks.values()) for k in next(iter(tasks.values()))}

    result = {"per_file": per_file, "per_task": tasks, "overall": overall}
    (OUT / "full_chain_coverage.json").write_text(json.dumps(result, indent=2))

    pct_tra = 100 * overall["tra_cdr3_with_full"] / max(1, overall["tra_cdr3_pop"])
    pct_trb = 100 * overall["trb_cdr3_with_full"] / max(1, overall["trb_cdr3_pop"])

    lines = [
        "# Full-Chain Coverage Audit",
        "",
        "Counts of rows where `tra_cdr3` / `trb_cdr3` is populated but the",
        "corresponding `tra_full` / `trb_full` is null. The gap is inherited",
        "from the upstream enrichment stage (likely missing V/J gene calls",
        "during stitching) and is documented in the wiki.",
        "",
        "## Overall",
        "",
        f"- Total rows scanned: **{overall['total']:,}**",
        f"- TRA CDR3 populated: {overall['tra_cdr3_pop']:,}",
        f"  - With tra_full: {overall['tra_cdr3_with_full']:,} ({pct_tra:.1f}%)",
        f"  - **Without tra_full: {overall['tra_cdr3_no_full']:,} ({100 - pct_tra:.1f}%)**",
        f"- TRB CDR3 populated: {overall['trb_cdr3_pop']:,}",
        f"  - With trb_full: {overall['trb_cdr3_with_full']:,} ({pct_trb:.1f}%)",
        f"  - **Without trb_full: {overall['trb_cdr3_no_full']:,} ({100 - pct_trb:.1f}%)**",
        "",
        "## Per task",
        "",
        "| Task | total | tra_cdr3_pop | tra_with_full | tra_no_full | tra_no_full_pct | trb_cdr3_pop | trb_with_full | trb_no_full | trb_no_full_pct |",
        "|------|------:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|",
    ]
    for t, st in sorted(tasks.items()):
        tra_pct = 100 * st["tra_cdr3_no_full"] / max(1, st["tra_cdr3_pop"])
        trb_pct = 100 * st["trb_cdr3_no_full"] / max(1, st["trb_cdr3_pop"])
        lines.append(
            f"| {t} | {st['total']:,} | {st['tra_cdr3_pop']:,} | {st['tra_cdr3_with_full']:,} | "
            f"{st['tra_cdr3_no_full']:,} | {tra_pct:.1f}% | "
            f"{st['trb_cdr3_pop']:,} | {st['trb_cdr3_with_full']:,} | "
            f"{st['trb_cdr3_no_full']:,} | {trb_pct:.1f}% |"
        )

    lines += ["", "## Per file (sorted by total gap size)", ""]
    lines.append("| file | total | tra_no_full | trb_no_full |")
    lines.append("|------|------:|-----:|-----:|")
    for r in sorted(per_file, key=lambda x: -(x["tra_cdr3_no_full"] + x["trb_cdr3_no_full"])):
        lines.append(
            f"| {r['file']} | {r['total']:,} | {r['tra_cdr3_no_full']:,} | {r['trb_cdr3_no_full']:,} |"
        )

    (OUT / "full_chain_coverage_summary.md").write_text("\n".join(lines) + "\n")

    print()
    print(f"OVERALL: TRA full present where cdr3 present: {pct_tra:.1f}%")
    print(f"OVERALL: TRB full present where cdr3 present: {pct_trb:.1f}%")
    print(f"Wrote: {OUT / 'full_chain_coverage.json'}")
    print(f"Wrote: {OUT / 'full_chain_coverage_summary.md'}")


if __name__ == "__main__":
    main()
