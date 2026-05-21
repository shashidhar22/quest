"""Stage 4: row-fraction validation on the full filtered universe.

Stream /home/ubuntu/quest/data/deduplicated_again/exploded_deduped_enriched/
through DuckDB, apply F1-F6 cleanup filters, join each of the 5 axis
columns to master_cluster_assignments.parquet, apply priority routing
(test > val > train), and tally the combined row fractions.

Writes _build_artifacts/row_routing_stats.json with per-axis (informational
already from Stage 3) plus the combined target metric.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

ROOT = Path("/home/ubuntu/quest/data/molecule_clusters")
ARTIFACT_DIR = ROOT / "_build_artifacts"
MASTER_PATH = ROOT / "master_cluster_assignments.parquet"
INPUT_GLOB = (
    "/home/ubuntu/quest/data/deduplicated_again/exploded_deduped_enriched/"
    "**/*.parquet"
)
ROUTING_PATH = ARTIFACT_DIR / "row_routing_stats.json"

# Target row-fractions
TARGET = {"train": 0.80, "val": 0.10, "test": 0.10}
TOL = 0.02  # ±2%


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{now_iso()}] {msg}", flush=True)


HLA_CLASS_MACRO = r"""
CREATE OR REPLACE MACRO hla_class(allele) AS (
  CASE
    WHEN allele IS NULL OR TRIM(CAST(allele AS VARCHAR)) = '' THEN NULL
    ELSE
      CASE
        WHEN UPPER(regexp_replace(regexp_replace(TRIM(CAST(allele AS VARCHAR)),
              '^HLA[-_*]?', ''), '^[-_*]+', '')) = ''
          THEN 'unknown'
        WHEN regexp_matches(
              UPPER(regexp_replace(regexp_replace(TRIM(CAST(allele AS VARCHAR)),
                '^HLA[-_*]?', ''), '^[-_*]+', '')),
              '^[ABCEFG]($|[*:0-9])'
        ) THEN 'I'
        WHEN regexp_matches(
              UPPER(regexp_replace(regexp_replace(TRIM(CAST(allele AS VARCHAR)),
                '^HLA[-_*]?', ''), '^[-_*]+', '')),
              '^D[RPQMO]A'
        ) THEN 'II_alpha'
        WHEN regexp_matches(
              UPPER(regexp_replace(regexp_replace(TRIM(CAST(allele AS VARCHAR)),
                '^HLA[-_*]?', ''), '^[-_*]+', '')),
              '^D[RPQMO]B'
        ) THEN 'II_beta'
        WHEN regexp_matches(
              UPPER(regexp_replace(regexp_replace(TRIM(CAST(allele AS VARCHAR)),
                '^HLA[-_*]?', ''), '^[-_*]+', '')),
              '^D[RPQMO]'
        ) THEN 'II_unknown_chain'
        ELSE 'unknown'
      END
  END
);
"""

# F1-F6 cleanup filter predicate (rows to KEEP).
# Each conjunct is COALESCEd to FALSE so a NULL value (very common for
# mhc_one/mhc_two on TCR-only rows) doesn't propagate NULL through the WHERE
# clause and drop the row. Stage 1 reports 1,449,240,249 rows kept out of
# 1,461,360,146 total -> F1-F6 should drop only ~12M rows.
F1_F6_KEEP = """
NOT (
  COALESCE(hla_class(mhc_one_allele) = 'II_beta', FALSE)                              -- F1
  OR COALESCE(hla_class(mhc_two_allele) = 'I', FALSE)                                 -- F2
  OR COALESCE(hla_class(mhc_two_allele) = 'II_alpha', FALSE)                          -- F3
  OR COALESCE(mhc_one_allele IS NOT NULL AND hla_class(mhc_one_allele) = 'unknown', FALSE) -- F4
  OR COALESCE(LENGTH(mhc_one) > 320, FALSE)                                           -- F5
  OR COALESCE(LENGTH(mhc_two) > 320, FALSE)                                           -- F6
)
"""


def main():
    overall_t0 = time.time()
    log("=== Stage 4: row-fraction validation on full filtered universe ===")

    con = duckdb.connect(":memory:")
    con.execute("PRAGMA threads = 64;")
    con.execute("PRAGMA memory_limit = '900GB';")
    con.execute("PRAGMA temp_directory = '/home/ubuntu/quest/data/molecule_clusters/_build_artifacts/duckdb_tmp';")
    Path("/home/ubuntu/quest/data/molecule_clusters/_build_artifacts/duckdb_tmp").mkdir(exist_ok=True)
    con.execute(HLA_CLASS_MACRO)

    # First: just count filtered universe size (sanity vs Stage 1's 1,449,240,249)
    log("counting filtered universe size (sanity vs Stage 1)...")
    t_count = time.time()
    n_filt = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{INPUT_GLOB}', hive_partitioning=true)
        WHERE {F1_F6_KEEP}
    """).fetchone()[0]
    log(f"  filtered universe: {n_filt:,} rows ({time.time()-t_count:.1f}s)")

    log("computing combined row fractions via priority routing test>val>train...")
    t_route = time.time()

    # Build per-axis lookup CTEs, then 5 LEFT JOINs, then priority routing.
    sql = f"""
    WITH filtered AS (
      SELECT trb_cdr3, tra_cdr3, peptide, mhc_one_pocket, mhc_two_pocket
      FROM read_parquet('{INPUT_GLOB}', hive_partitioning=true)
      WHERE {F1_F6_KEEP}
    ),
    trb_lkp AS (
      SELECT molecule_string AS k, split FROM read_parquet('{MASTER_PATH}') WHERE axis='trb_cdr3'
    ),
    tra_lkp AS (
      SELECT molecule_string AS k, split FROM read_parquet('{MASTER_PATH}') WHERE axis='tra_cdr3'
    ),
    pep_lkp AS (
      SELECT molecule_string AS k, split FROM read_parquet('{MASTER_PATH}') WHERE axis='peptide'
    ),
    mho_lkp AS (
      SELECT molecule_string AS k, split FROM read_parquet('{MASTER_PATH}') WHERE axis='mhc_one_pocket'
    ),
    mht_lkp AS (
      SELECT molecule_string AS k, split FROM read_parquet('{MASTER_PATH}') WHERE axis='mhc_two_pocket'
    ),
    joined AS (
      SELECT
        t.split AS trb_split,
        a.split AS tra_split,
        p.split AS pep_split,
        mo.split AS mho_split,
        mt.split AS mht_split
      FROM filtered f
      LEFT JOIN trb_lkp t ON f.trb_cdr3 = t.k
      LEFT JOIN tra_lkp a ON f.tra_cdr3 = a.k
      LEFT JOIN pep_lkp p ON f.peptide = p.k
      LEFT JOIN mho_lkp mo ON f.mhc_one_pocket = mo.k
      LEFT JOIN mht_lkp mt ON f.mhc_two_pocket = mt.k
    ),
    routed AS (
      SELECT CASE
        WHEN 'test' IN (trb_split, tra_split, pep_split, mho_split, mht_split) THEN 'test'
        WHEN 'val'  IN (trb_split, tra_split, pep_split, mho_split, mht_split) THEN 'val'
        WHEN 'train' IN (trb_split, tra_split, pep_split, mho_split, mht_split) THEN 'train'
        ELSE 'none'
      END AS final_split,
      -- Also count: how many axes had any cluster (signal of coverage)
      ( (trb_split IS NOT NULL)::INT
      + (tra_split IS NOT NULL)::INT
      + (pep_split IS NOT NULL)::INT
      + (mho_split IS NOT NULL)::INT
      + (mht_split IS NOT NULL)::INT ) AS n_axes_present
      FROM joined
    )
    SELECT final_split,
           SUM(1) AS n_rows,
           AVG(n_axes_present) AS avg_axes_present
    FROM routed
    GROUP BY final_split
    ORDER BY final_split;
    """
    rows = con.execute(sql).fetchall()
    log(f"  routing completed in {time.time()-t_route:.1f}s")
    log(f"  result: {rows}")

    total = sum(r[1] for r in rows)
    fractions = {r[0]: int(r[1]) for r in rows}
    avg_axes = {r[0]: float(r[2]) for r in rows}
    fraction_pct = {k: v / total for k, v in fractions.items()}

    log(f"  combined row counts: {fractions}")
    log(f"  combined row fractions: " +
        ", ".join(f"{k}={v:.4f}" for k, v in fraction_pct.items()))

    # Compare to target (within ±2%)
    deviations = {}
    pass_gate = True
    for split in ("train", "val", "test"):
        actual = fraction_pct.get(split, 0.0)
        target = TARGET[split]
        dev = actual - target
        deviations[split] = dev
        if abs(dev) > TOL:
            pass_gate = False
    log(f"  deviations from target {TARGET}: " +
        ", ".join(f"{k}={v:+.4f}" for k, v in deviations.items()) +
        f"; tolerance ±{TOL} -> {'PASS' if pass_gate else 'FAIL'}")

    payload = {
        "stage": "stage4_row_routing",
        "filtered_universe_rows": int(n_filt),
        "combined_row_counts": fractions,
        "combined_row_fractions": fraction_pct,
        "target_row_fractions": TARGET,
        "deviations_from_target": deviations,
        "tolerance_pct": TOL,
        "within_tolerance": pass_gate,
        "avg_axes_present_per_split": avg_axes,
        "iteration": 1,
        "max_iterations": 5,
        "elapsed_s": float(time.time() - overall_t0),
        "timestamp_utc": now_iso(),
        "note": (
            "Stage 4 single-iteration row-routing validation. If deviation "
            "exceeds tolerance, the spec calls for adjusting cluster fractions "
            "and re-running Stage 3 for affected axes (up to 5 iters)."
        ),
    }
    ROUTING_PATH.write_text(json.dumps(payload, indent=2))
    log(f"wrote {ROUTING_PATH}")
    log(f"DONE in {time.time()-overall_t0:.1f}s")


if __name__ == "__main__":
    main()
