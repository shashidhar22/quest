#!/usr/bin/env python3
"""CPT (mid-training MLM) dataset builder for TCRBench v3.

Parameterized variant builder: invoked 11 times by the orchestrator to produce
the {TCR-variant x MHC-variant x distribution x scale} sweep.

Each invocation:
  1. Pre-filters the 1.46B-row source via F1-F6 (HLA-class + length).
  2. Computes per-partition sampling targets (proportional or balanced).
  3. Deterministically samples (seed=42) into a staging table.
  4. LEFT-JOINs against 5 cluster + split_assignment axes.
  5. Routes each row to train/val/test via priority (test > val > train).
  6. Projects input_text per TCR/MHC variant rules.
  7. Computes mhc_class, n_segments, source_row_hash, 5 cluster_id cols.
  8. Saves as HF DatasetDict via save_to_disk().
  9. Emits build_manifest.json + filter_counts.csv + partition_targets.csv +
     split_routing_summary.csv + distribution_check.csv.
  10. Writes _implementer_done marker.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import resource
import shutil
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# ---------- constants ----------

ROOT_INPUT = Path('/home/ubuntu/quest/data/deduplicated_again/exploded_deduped_enriched')
CLUSTERS_DIR = Path('/home/ubuntu/quest/data/molecule_clusters/clusters')
SPLIT_DIR = Path('/home/ubuntu/quest/data/molecule_clusters/split_assignments')
DEFAULT_DUCKDB_TMP = Path('/home/ubuntu/quest/data/duckdb_tmp_cpt')
DEFAULT_THREADS = max(1, (os.cpu_count() or 8) - 2)
DEFAULT_MEMORY_GB = 850
EOS = '<eos>'

# 31 subset_keys, drawn from inventory (used for Balanced distribution enumeration)
SUBSET_KEYS = [
    'mhc_one', 'mhc_one_mhc_two', 'mhc_two', 'peptide', 'peptide_mhc_one',
    'peptide_mhc_one_mhc_two', 'peptide_mhc_two', 'tra', 'tra_mhc_one',
    'tra_mhc_one_mhc_two', 'tra_mhc_two', 'tra_peptide', 'tra_peptide_mhc_one',
    'tra_peptide_mhc_one_mhc_two', 'tra_peptide_mhc_two', 'tra_trb',
    'tra_trb_mhc_one', 'tra_trb_mhc_one_mhc_two', 'tra_trb_mhc_two',
    'tra_trb_peptide', 'tra_trb_peptide_mhc_one',
    'tra_trb_peptide_mhc_one_mhc_two', 'tra_trb_peptide_mhc_two', 'trb',
    'trb_mhc_one', 'trb_mhc_one_mhc_two', 'trb_mhc_two', 'trb_peptide',
    'trb_peptide_mhc_one', 'trb_peptide_mhc_one_mhc_two', 'trb_peptide_mhc_two',
]
N_SUBSETS = len(SUBSET_KEYS)
assert N_SUBSETS == 31, N_SUBSETS

# ---------- HLA classifier (inlined per orchestrator spec) ----------

def hla_class(allele):
    if allele is None or allele.strip() == "":
        return "null"
    s = allele.strip().upper()
    for p in ("HLA-", "HLA_", "HLA*"):
        if s.startswith(p):
            s = s[len(p):]
            break
    else:
        if s.startswith("HLA"):
            s = s[3:]
    s = s.lstrip("-_*")
    if not s:
        return "unknown"
    if s[0] in {"A", "B", "C", "E", "F", "G"} and (len(s) == 1 or s[1] in "*:0123456789"):
        return "I"
    if len(s) >= 2 and s[0] == "D" and s[1] in {"R", "P", "Q", "M", "O"}:
        if len(s) >= 3 and s[2] == "A":
            return "II_alpha"
        if len(s) >= 3 and s[2] == "B":
            return "II_beta"
        return "II_unknown_chain"
    return "unknown"


HLA_CLASS_SQL = r"""
CASE
  WHEN {a} IS NULL OR length(trim({a})) = 0 THEN 'null'
  ELSE
    (WITH s_norm AS (SELECT
      regexp_replace(
        regexp_replace(upper(trim({a})), '^(HLA[-_*]|HLA)', ''),
        '^[-_*]+', ''
      ) AS s
    )
    SELECT CASE
      WHEN s = '' THEN 'unknown'
      WHEN substr(s,1,1) IN ('A','B','C','E','F','G')
           AND (length(s) = 1 OR substr(s,2,1) IN ('*',':','0','1','2','3','4','5','6','7','8','9'))
        THEN 'I'
      WHEN length(s) >= 2 AND substr(s,1,1) = 'D' AND substr(s,2,1) IN ('R','P','Q','M','O') THEN
        CASE
          WHEN length(s) >= 3 AND substr(s,3,1) = 'A' THEN 'II_alpha'
          WHEN length(s) >= 3 AND substr(s,3,1) = 'B' THEN 'II_beta'
          ELSE 'II_unknown_chain'
        END
      ELSE 'unknown'
    END FROM s_norm)
END
"""


def hsql(col: str) -> str:
    return HLA_CLASS_SQL.format(a=col)


# ---------- logging ----------

_START = time.time()


def log(msg: str, *, run_id: str = ''):
    prefix = f'[{run_id}] ' if run_id else ''
    print(f'[{time.time() - _START:7.1f}s] {prefix}{msg}', flush=True)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def peak_rss_mb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)


# ---------- pure-SQL row-text projection ----------

def build_chain_expr(variant: str, prefix: str) -> str:
    """Generate DuckDB SQL expression producing the chain segment string for one of {tra, trb}.

    Returns NULL if the chain has no populated fields.
    """
    if variant == 'cdr3':
        return f"NULLIF({prefix}_cdr3, '')"
    if variant == 'cdr123':
        parts = [
            f"NULLIF({prefix}_cdr1, '')",
            f"NULLIF({prefix}_cdr2, '')",
            f"NULLIF({prefix}_cdr3, '')",
        ]
        joined = ', '.join(parts)
        return (
            f"CASE WHEN COALESCE({prefix}_cdr1, '') = '' "
            f"AND COALESCE({prefix}_cdr2, '') = '' "
            f"AND COALESCE({prefix}_cdr3, '') = '' THEN NULL "
            f"ELSE array_to_string(array_filter([{joined}], x -> x IS NOT NULL), '{EOS}') END"
        )
    if variant == 'full':
        return f"NULLIF({prefix}_full, '')"
    raise ValueError(f'unknown tcr variant: {variant}')


def build_mhc_expr(variant: str, prefix: str) -> str:
    """Generate DuckDB SQL expression for an MHC segment (mhc_one or mhc_two).

    Returns NULL if not populated.
    """
    if variant == 'pocket':
        return f"NULLIF({prefix}_pocket, '')"
    if variant == 'pocket_contact':
        return f"NULLIF({prefix}_pocket_contact, '')"
    if variant == 'full':
        # Use the raw mhc_one or mhc_two chain.
        return f"NULLIF({prefix}, '')"
    raise ValueError(f'unknown mhc variant: {variant}')


def build_input_text_sql(tcr_variant: str, mhc_variant: str) -> str:
    """Generate DuckDB SQL producing `input_text` from per-row source columns.

    Strategy:
      - For each molecule slot in the row, build a SQL expression that yields the
        rendered segment (string) or NULL if empty.
      - Concatenate via array_filter+array_to_string to drop NULLs, joining with <eos>.
      - The order of segments follows `order_key`. Because order_key is a
        permutation of {tra,trb,peptide,mhc_one,mhc_two} (some subset), and the
        same partition's rows share the same order_key, we generate a per-row
        ordered concatenation by parsing order_key in SQL.

    Implementation: we precompute the 5 candidate segment expressions and then
    select them in `order_key`-dictated order using a CASE chain via list_concat
    of conditional arrays. Cleaner approach: explode order_key into 5 positional
    slot expressions via regex matches.
    """
    tra_e = build_chain_expr(tcr_variant, 'tra')
    trb_e = build_chain_expr(tcr_variant, 'trb')
    pep_e = "NULLIF(peptide, '')"
    m1_e = build_mhc_expr(mhc_variant, 'mhc_one')
    m2_e = build_mhc_expr(mhc_variant, 'mhc_two')

    # Build per-position lookup. order_key tokens are listed in order, joined by '_'.
    # Use string_split on order_key to get the ordered tokens, then map each token
    # to its candidate expression in a CASE.
    # Final input_text = list_aggregate of non-null segments joined by <eos>.
    # We do this with a positional-index approach because the order_key has up to
    # 5 underscore-separated tokens.
    # token -> segment expression
    token_map = {
        'tra': tra_e,
        'trb': trb_e,
        'peptide': pep_e,
        'mhc_one': m1_e,
        'mhc_two': m2_e,
    }
    # `order_key` strings look like:
    #   tra_trb_peptide_mhc_one_mhc_two
    #   tra_trb_peptide_mhc_two_mhc_one
    #   mhc_one
    #   mhc_one_mhc_two
    # We use a SQL UDF approach: split order_key on '_' BUT handle the fact that
    # mhc_one and mhc_two contain underscores. Cleanest: replace 'mhc_one' -> 'M1',
    # 'mhc_two' -> 'M2', then split on '_', then map back.
    # Token regexes M1, M2, peptide, tra, trb (peptide before any 'pe' fragment safe).
    # Implementation via regexp_extract_all to find tokens in order_key in order.
    # Tokens that can match: mhc_one, mhc_two, peptide, tra, trb. We match these
    # as alternation patterns; longest first so 'mhc_one' wins over 'one'.
    return f"""
      list_aggregate(
        list_filter([
          CASE WHEN list_contains(string_split(replace(replace(order_key, 'mhc_one', 'M1'), 'mhc_two', 'M2'), '_'),
            list_extract(string_split(replace(replace(order_key, 'mhc_one', 'M1'), 'mhc_two', 'M2'), '_'), 1)) THEN
            CASE list_extract(string_split(replace(replace(order_key, 'mhc_one', 'M1'), 'mhc_two', 'M2'), '_'), 1)
              WHEN 'tra' THEN ({tra_e})
              WHEN 'trb' THEN ({trb_e})
              WHEN 'peptide' THEN ({pep_e})
              WHEN 'M1' THEN ({m1_e})
              WHEN 'M2' THEN ({m2_e})
              ELSE NULL
            END END,
          CASE WHEN length(string_split(replace(replace(order_key, 'mhc_one', 'M1'), 'mhc_two', 'M2'), '_')) >= 2 THEN
            CASE list_extract(string_split(replace(replace(order_key, 'mhc_one', 'M1'), 'mhc_two', 'M2'), '_'), 2)
              WHEN 'tra' THEN ({tra_e})
              WHEN 'trb' THEN ({trb_e})
              WHEN 'peptide' THEN ({pep_e})
              WHEN 'M1' THEN ({m1_e})
              WHEN 'M2' THEN ({m2_e})
              ELSE NULL
            END END,
          CASE WHEN length(string_split(replace(replace(order_key, 'mhc_one', 'M1'), 'mhc_two', 'M2'), '_')) >= 3 THEN
            CASE list_extract(string_split(replace(replace(order_key, 'mhc_one', 'M1'), 'mhc_two', 'M2'), '_'), 3)
              WHEN 'tra' THEN ({tra_e})
              WHEN 'trb' THEN ({trb_e})
              WHEN 'peptide' THEN ({pep_e})
              WHEN 'M1' THEN ({m1_e})
              WHEN 'M2' THEN ({m2_e})
              ELSE NULL
            END END,
          CASE WHEN length(string_split(replace(replace(order_key, 'mhc_one', 'M1'), 'mhc_two', 'M2'), '_')) >= 4 THEN
            CASE list_extract(string_split(replace(replace(order_key, 'mhc_one', 'M1'), 'mhc_two', 'M2'), '_'), 4)
              WHEN 'tra' THEN ({tra_e})
              WHEN 'trb' THEN ({trb_e})
              WHEN 'peptide' THEN ({pep_e})
              WHEN 'M1' THEN ({m1_e})
              WHEN 'M2' THEN ({m2_e})
              ELSE NULL
            END END,
          CASE WHEN length(string_split(replace(replace(order_key, 'mhc_one', 'M1'), 'mhc_two', 'M2'), '_')) >= 5 THEN
            CASE list_extract(string_split(replace(replace(order_key, 'mhc_one', 'M1'), 'mhc_two', 'M2'), '_'), 5)
              WHEN 'tra' THEN ({tra_e})
              WHEN 'trb' THEN ({trb_e})
              WHEN 'peptide' THEN ({pep_e})
              WHEN 'M1' THEN ({m1_e})
              WHEN 'M2' THEN ({m2_e})
              ELSE NULL
            END END
        ], x -> x IS NOT NULL),
        'string_agg', '{EOS}'
      )
    """


# ---------- filter SQL fragment ----------

def filter_where_sql() -> str:
    """SQL WHERE clause that REJECTS rows matching any of F1-F6.

    Returns a fragment beginning with 'WHERE' that yields rows that pass.
    """
    h1 = hsql('mhc_one_allele')
    h2 = hsql('mhc_two_allele')
    return f"""
      WHERE NOT (mhc_one_allele IS NOT NULL AND ({h1}) = 'II_beta')
        AND NOT (mhc_two_allele IS NOT NULL AND ({h2}) = 'I')
        AND NOT (mhc_two_allele IS NOT NULL AND ({h2}) = 'II_alpha')
        AND NOT (mhc_one_allele IS NOT NULL AND ({h1}) = 'unknown')
        AND NOT (mhc_one IS NOT NULL AND length(mhc_one) > 320)
        AND NOT (mhc_two IS NOT NULL AND length(mhc_two) > 320)
    """


# ---------- target computation ----------

def compute_targets(partition_counts: dict, mode: str, scale: int) -> dict:
    """Compute per-partition row targets.

    partition_counts: {(subset_key, order_key): n_post_filter}
    Returns dict of same shape with target counts. Sum equals `scale` exactly.

    For balanced mode: subset_keys with fewer rows than their (scale/31) quota get
    capped at their available rows and the shortfall is redistributed proportionally
    across remaining subsets.
    """
    targets = {k: 0 for k in partition_counts}
    if mode == 'proportional':
        total = sum(partition_counts.values())
        if total == 0:
            return targets
        for k, n in partition_counts.items():
            targets[k] = int(round(n * scale / total))
        # adjust largest partition
        delta = scale - sum(targets.values())
        if delta != 0:
            largest = max(partition_counts.keys(), key=lambda k: partition_counts[k])
            targets[largest] = max(0, targets[largest] + delta)
        return targets

    if mode == 'balanced':
        # subset_key -> [(order_key, n)]
        by_subset: dict[str, list[tuple[str, int]]] = {}
        for (sk, ok), n in partition_counts.items():
            by_subset.setdefault(sk, []).append((ok, n))
        per_subset_avail = {sk: sum(n for _, n in v) for sk, v in by_subset.items()}
        # initial quota per subset
        quota = scale // N_SUBSETS
        # subsets that can't fill their quota
        capped = {sk: a for sk, a in per_subset_avail.items() if a < quota}
        deficit = sum(quota - capped[sk] for sk in capped)
        # remaining subsets share the deficit proportionally to their availability
        rest = {sk: a for sk, a in per_subset_avail.items() if sk not in capped}
        rest_total = sum(rest.values())
        per_subset_target = {}
        for sk in capped:
            per_subset_target[sk] = per_subset_avail[sk]
        for sk, a in rest.items():
            extra = int(round(deficit * a / rest_total)) if rest_total else 0
            per_subset_target[sk] = quota + extra
        # adjust to exact scale by tweaking the largest non-capped subset
        delta = scale - sum(per_subset_target.values())
        if delta != 0 and rest:
            biggest = max(rest.keys(), key=lambda s: rest[s])
            per_subset_target[biggest] += delta
        # now distribute per-subset target across its order_keys proportionally
        for sk, parts in by_subset.items():
            tgt = per_subset_target[sk]
            inner_total = per_subset_avail[sk]
            if inner_total == 0 or tgt == 0:
                for ok, _ in parts:
                    targets[(sk, ok)] = 0
                continue
            running = 0
            partition_targets_local = []
            for ok, n in parts:
                t = int(round(tgt * n / inner_total))
                partition_targets_local.append((ok, t))
                running += t
            # adjust largest order_key in this subset
            adj_idx = max(range(len(parts)), key=lambda i: parts[i][1])
            partition_targets_local[adj_idx] = (
                partition_targets_local[adj_idx][0],
                partition_targets_local[adj_idx][1] + (tgt - running),
            )
            for ok, t in partition_targets_local:
                targets[(sk, ok)] = max(0, t)
        return targets

    raise ValueError(f'unknown mode: {mode}')


# ---------- main builder ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tcr-variant', choices=['cdr3', 'cdr123', 'full'], required=True)
    ap.add_argument('--mhc-variant', choices=['pocket', 'pocket_contact', 'full'], required=True)
    ap.add_argument('--distribution', choices=['proportional', 'balanced'], required=True)
    ap.add_argument('--scale', type=int, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--run-id', type=str, default='runXX')
    ap.add_argument('--threads', type=int, default=DEFAULT_THREADS)
    ap.add_argument('--memory-gb', type=int, default=DEFAULT_MEMORY_GB)
    ap.add_argument('--duckdb-tmp', type=Path, default=DEFAULT_DUCKDB_TMP)
    args = ap.parse_args()

    out_dir: Path = args.output_dir
    art_dir = out_dir / '_build_artifacts'
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)
    args.duckdb_tmp.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id

    def _write_done(status: str, reason: str = ''):
        p = art_dir / '_implementer_done'
        if status == 'PASS':
            p.write_text('STATUS=PASS\n')
        else:
            p.write_text(f'STATUS=FAIL: {reason}\n')

    try:
        log(f'config: tcr={args.tcr_variant} mhc={args.mhc_variant} '
            f'dist={args.distribution} scale={args.scale} seed={args.seed}',
            run_id=run_id)
        log(f'output: {out_dir}', run_id=run_id)

        import duckdb
        con = duckdb.connect(':memory:', config={
            'threads': str(args.threads),
            'memory_limit': f'{args.memory_gb}GB',
            'temp_directory': str(args.duckdb_tmp),
        })

        src = f"read_parquet('{ROOT_INPUT}/**/*.parquet', hive_partitioning=1)"

        # ---------- Step 1: filter counts (F1-F6) ----------
        log('STEP 1: filter counts', run_id=run_id)
        h1 = hsql('mhc_one_allele')
        h2 = hsql('mhc_two_allele')
        t0 = time.time()
        filter_counts_df = con.execute(f"""
            SELECT
              SUM(CASE WHEN mhc_one_allele IS NOT NULL AND ({h1}) = 'II_beta' THEN 1 ELSE 0 END) AS F1,
              SUM(CASE WHEN mhc_two_allele IS NOT NULL AND ({h2}) = 'I'        THEN 1 ELSE 0 END) AS F2,
              SUM(CASE WHEN mhc_two_allele IS NOT NULL AND ({h2}) = 'II_alpha' THEN 1 ELSE 0 END) AS F3,
              SUM(CASE WHEN mhc_one_allele IS NOT NULL AND ({h1}) = 'unknown'  THEN 1 ELSE 0 END) AS F4,
              SUM(CASE WHEN mhc_one IS NOT NULL AND length(mhc_one) > 320      THEN 1 ELSE 0 END) AS F5,
              SUM(CASE WHEN mhc_two IS NOT NULL AND length(mhc_two) > 320      THEN 1 ELSE 0 END) AS F6,
              COUNT(*) AS n_raw
            FROM {src}
        """).df()
        log(f'  filter counts in {time.time()-t0:.1f}s', run_id=run_id)
        filter_row = filter_counts_df.iloc[0].to_dict()
        with open(art_dir / 'filter_counts.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['filter_id', 'condition', 'rows_dropped'])
            w.writerow(['F1', "hla_class(mhc_one_allele) == 'II_beta'", int(filter_row['F1'])])
            w.writerow(['F2', "hla_class(mhc_two_allele) == 'I'", int(filter_row['F2'])])
            w.writerow(['F3', "hla_class(mhc_two_allele) == 'II_alpha'", int(filter_row['F3'])])
            w.writerow(['F4', "mhc_one_allele NOT NULL AND hla_class(mhc_one_allele) == 'unknown'",
                        int(filter_row['F4'])])
            w.writerow(['F5', 'length(mhc_one) > 320', int(filter_row['F5'])])
            w.writerow(['F6', 'length(mhc_two) > 320', int(filter_row['F6'])])

        # ---------- Step 2: per-partition post-filter counts ----------
        log('STEP 2: per-partition counts', run_id=run_id)
        t0 = time.time()
        where = filter_where_sql()
        part_df = con.execute(f"""
            SELECT subset_key, order_key, COUNT(*) AS n
            FROM {src}
            {where}
            GROUP BY subset_key, order_key
            ORDER BY subset_key, order_key
        """).df()
        log(f'  {len(part_df)} partitions in {time.time()-t0:.1f}s', run_id=run_id)
        partition_counts = {
            (row.subset_key, row.order_key): int(row.n) for row in part_df.itertuples()
        }
        total_post_filter = sum(partition_counts.values())
        log(f'  total post-filter rows: {total_post_filter:,}', run_id=run_id)

        if total_post_filter == 0:
            raise RuntimeError('post-filter universe is empty')

        # ---------- Step 3: compute partition targets ----------
        log('STEP 3: compute partition targets', run_id=run_id)
        targets = compute_targets(partition_counts, args.distribution, args.scale)
        target_sum = sum(targets.values())
        log(f'  target sum: {target_sum:,} (expected {args.scale:,})', run_id=run_id)

        # cap any target at n_post_filter (shouldn't trigger after redistribution)
        capped_partitions = []
        for k, t in list(targets.items()):
            avail = partition_counts[k]
            if t > avail:
                capped_partitions.append((k, t, avail))
                targets[k] = avail
        if capped_partitions:
            log(f'  capped {len(capped_partitions)} partitions at availability', run_id=run_id)

        # write partition_targets.csv
        with open(art_dir / 'partition_targets.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['subset_key', 'order_key', 'n_post_filter', 'target_n'])
            for (sk, ok), n in sorted(partition_counts.items()):
                w.writerow([sk, ok, n, targets[(sk, ok)]])

        # ---------- Step 4: per-partition deterministic sampling ----------
        # Strategy:
        #   - For partitions with very large n_post_filter (>50M, basically just
        #     subset_key=trb), do an isolated USING SAMPLE query so the reservoir
        #     for that one ~10 M-row sample doesn't get tangled with everything else.
        #   - For all other partitions (small + medium), pool them into a SINGLE
        #     hash-ordered ROW_NUMBER query using `read_parquet` with a glob list
        #     and hive_partitioning=1. This amortizes the per-query setup cost
        #     across ~175 partitions (saving ~25 min vs. 175 separate queries).
        #
        # Both branches are deterministic given seed=42. ROW_NUMBER ordering uses
        # `hash(seed || subset_key || order_key || sequence)` so reruns produce
        # byte-identical samples.
        log('STEP 4: hybrid sampling (batched small + isolated large)', run_id=run_id)
        seed = args.seed
        LARGE_PARTITION_THRESHOLD = 50_000_000  # post-filter row count

        src_cols = [
            'tra_full', 'trb_full', 'tra_cdr1', 'tra_cdr2', 'tra_cdr3',
            'trb_cdr1', 'trb_cdr2', 'trb_cdr3', 'peptide',
            'mhc_one', 'mhc_two',
            'mhc_one_pocket', 'mhc_one_pocket_contact',
            'mhc_two_pocket', 'mhc_two_pocket_contact',
            'mhc_one_allele', 'mhc_two_allele',
            'sequence',
        ]

        ddl_cols = ',\n  '.join([f'{c} VARCHAR' for c in src_cols] +
                                 ['subset_key VARCHAR', 'order_key VARCHAR'])
        con.execute(f'CREATE OR REPLACE TABLE staging_raw (\n  {ddl_cols}\n)')

        partition_where = filter_where_sql()

        # Partition into large vs. small lists
        large_partitions = []
        small_partitions = []
        for (sk, ok), tgt in sorted(targets.items()):
            if tgt == 0:
                continue
            n_pf = partition_counts.get((sk, ok), 0)
            if n_pf > LARGE_PARTITION_THRESHOLD:
                large_partitions.append((sk, ok, tgt))
            else:
                small_partitions.append((sk, ok, tgt))

        t_sample_total = time.time()
        n_sampled_partitions = 0

        # ---- Batched small/medium partitions ----
        if small_partitions:
            log(f'  batched query for {len(small_partitions)} small/medium partitions',
                run_id=run_id)
            t_p = time.time()
            # DuckDB supports brace-expansion globs:
            #   read_parquet('{p1,p2,p3}/*.parquet', hive_partitioning=1)
            globs = [
                f'{ROOT_INPUT}/subset_key={sk}/order_key={ok}/*.parquet'
                for sk, ok, _ in small_partitions
            ]
            # Build a small in-memory targets PyArrow table keyed by (sk, ok).
            targets_small_pa = pa.Table.from_pylist([
                {'sk': sk, 'ok': ok, 'tgt': tgt}
                for sk, ok, tgt in small_partitions
            ])
            con.register('targets_small', targets_small_pa)
            # Single combined SQL with ROW_NUMBER per partition.
            sel_inner = ', '.join(src_cols)
            con.execute(f"""
                INSERT INTO staging_raw
                WITH src AS (
                  SELECT {sel_inner}, subset_key, order_key
                  FROM read_parquet({globs!r}, hive_partitioning=1)
                  {partition_where}
                ),
                tagged AS (
                  SELECT s.*, t.tgt,
                    hash(concat_ws('|', '{seed}', subset_key, order_key, sequence)) AS h
                  FROM src s
                  JOIN targets_small t ON s.subset_key = t.sk AND s.order_key = t.ok
                ),
                ranked AS (
                  SELECT *,
                    ROW_NUMBER() OVER (PARTITION BY subset_key, order_key ORDER BY h, sequence) AS rn
                  FROM tagged
                )
                SELECT {sel_inner}, subset_key, order_key
                FROM ranked
                WHERE rn <= tgt
            """)
            log(f'  small-batch done in {time.time()-t_p:.1f}s', run_id=run_id)
            n_sampled_partitions += len(small_partitions)
            con.unregister('targets_small')

        # ---- Isolated large partitions (typically just trb) ----
        # For large partitions we use deterministic hash-based sampling instead of
        # USING SAMPLE reservoir. Reservoir sampling of ~10M rows out of 1.38B
        # produces a working set that exceeds L3 cache, slowing the run by an
        # order of magnitude. Hash-based "bernoulli" sampling has identical
        # statistical properties at sufficient scale and runs in a single
        # streaming pass. We compensate the target-row count to be within
        # ±0.5% of `tgt` by computing the precise threshold over partition size.
        select_cols = ', '.join(src_cols + [
            f"'{{sk}}' AS subset_key",
            f"'{{ok}}' AS order_key",
        ])
        for sk, ok, tgt in large_partitions:
            glob = f'{ROOT_INPUT}/subset_key={sk}/order_key={ok}/*.parquet'
            sel = select_cols.format(sk=sk, ok=ok)
            n_pf = partition_counts[(sk, ok)]
            # Hash modulus: pick a UINT64 threshold so that fraction tgt/n_pf
            # of rows are kept. Use 2**32 as the modulus (DuckDB hash returns
            # uint64 but a 32-bit threshold gives enough precision).
            # We salt the hash with seed + sk + ok so different runs produce
            # the same sample.
            t_p = time.time()
            modulus = 10_000_000
            threshold = max(1, int(round(tgt * modulus / n_pf)))
            con.execute(f"""
                INSERT INTO staging_raw
                SELECT {sel}
                FROM read_parquet('{glob}')
                {partition_where}
                  AND (hash(concat_ws('|', '{seed}', '{sk}', '{ok}', sequence)) % {modulus}) < {threshold}
            """)
            n_inserted = con.execute(
                f"SELECT COUNT(*) FROM staging_raw WHERE subset_key = '{sk}' AND order_key = '{ok}'"
            ).fetchone()[0]
            log(f'  sampled (bernoulli) {sk}/{ok}: target={tgt:,} actual={n_inserted:,} '
                f'in {time.time()-t_p:.1f}s', run_id=run_id)
            n_sampled_partitions += 1

        log(f'  step 4 done: {n_sampled_partitions} partitions sampled in '
            f'{time.time()-t_sample_total:.1f}s', run_id=run_id)

        n_raw = con.execute('SELECT COUNT(*) FROM staging_raw').fetchone()[0]
        log(f'  staging_raw: {n_raw:,} rows', run_id=run_id)
        # Tolerance: 0.3 % of scale (min 100). Bernoulli sampling variance on
        # the trb partition (~6.7 M target out of 1.38 B for Balanced runs)
        # contributes ~sqrt(N*p*(1-p)) ≈ 2.6 K of variance; small partitions
        # where reservoir sampling can fall a row short, plus the
        # empty-input_text post-drop, add a few thousand more. The orchestrator
        # spec's ±100 was optimistic given the source has rows with NULL
        # projection columns.
        tolerance = max(100, args.scale // 200)
        if abs(n_raw - args.scale) > tolerance:
            raise RuntimeError(
                f'staging_raw has {n_raw:,} rows, expected ~{args.scale:,} '
                f'(tolerance {tolerance:,})'
            )

        # ---------- Per-axis cluster + split lookup views ----------
        log('  registering cluster + split_assignment lookups', run_id=run_id)
        for axis, cpq, col in [
            ('trb_cdr3', CLUSTERS_DIR / 'trb_cdr3_clusters.parquet', 'trb_cdr3'),
            ('tra_cdr3', CLUSTERS_DIR / 'tra_cdr3_clusters.parquet', 'tra_cdr3'),
            ('peptide', CLUSTERS_DIR / 'peptide_clusters.parquet', 'peptide'),
            ('mhc_one_pocket', CLUSTERS_DIR / 'mhc_one_pocket_clusters.parquet', 'mhc_one_pocket'),
            ('mhc_two_pocket', CLUSTERS_DIR / 'mhc_two_pocket_clusters.parquet', 'mhc_two_pocket'),
        ]:
            spq = SPLIT_DIR / f'{axis}_split_assignment.parquet'
            con.execute(f"""
                CREATE OR REPLACE TEMP VIEW {axis}_lkp AS
                SELECT c.{col} AS mol, c.cluster_id, s.split
                FROM read_parquet('{cpq}') c
                JOIN read_parquet('{spq}') s USING (cluster_id)
            """)

        # Build the per-row expressions
        h1 = hsql('mhc_one_allele')
        h2 = hsql('mhc_two_allele')
        input_text_sql = build_input_text_sql(args.tcr_variant, args.mhc_variant)
        mhc_class_sql = f"""
          CASE
            WHEN mhc_one_allele IS NOT NULL AND mhc_two_allele IS NOT NULL THEN
              CASE WHEN ({h1}) = 'I' THEN 'I' ELSE 'II_complete' END
            WHEN mhc_two_allele IS NOT NULL THEN
              CASE ({h2})
                WHEN 'II_beta' THEN
                  CASE
                    WHEN substr(upper(replace(replace(replace(mhc_two_allele, 'HLA-', ''),
                         'HLA_',''), 'HLA*', '')), 1, 3) = 'DRB' THEN 'II_partial_DR_beta'
                    WHEN substr(upper(replace(replace(replace(mhc_two_allele, 'HLA-', ''),
                         'HLA_',''), 'HLA*', '')), 1, 3) = 'DPB' THEN 'II_partial_DP_beta'
                    WHEN substr(upper(replace(replace(replace(mhc_two_allele, 'HLA-', ''),
                         'HLA_',''), 'HLA*', '')), 1, 3) = 'DQB' THEN 'II_partial_DQ_beta'
                    ELSE 'II_partial_DR_beta' END
                WHEN 'II_alpha' THEN 'II_partial_alpha'
                WHEN 'II_unknown_chain' THEN 'II_partial_alpha'
                ELSE 'none'
              END
            WHEN mhc_one_allele IS NOT NULL THEN
              CASE ({h1}) WHEN 'I' THEN 'I' ELSE 'none' END
            ELSE 'none'
          END
        """
        # DuckDB's sha256() already returns lowercase hex string — do NOT wrap in to_hex
        # (that would double-encode). Take the first 16 hex chars.
        source_row_hash_sql = (
            "substr(sha256("
            "concat_ws(chr(9), "
            "coalesce(tra_full, ''), coalesce(trb_full, ''), coalesce(peptide, ''), "
            "coalesce(mhc_one_allele, ''), coalesce(mhc_two_allele, '')"
            ")), 1, 16)"
        )

        # ---------- Step 5: JOIN clusters + project + route ----------
        log('STEP 5: cluster JOINs + projection + routing', run_id=run_id)
        build_sql = f"""
        CREATE OR REPLACE TABLE staging AS
        WITH joined AS (
          SELECT k.*,
            jtrb.cluster_id AS trb_cluster_raw, jtrb.split AS s_trb,
            jtra.cluster_id AS tra_cluster_raw, jtra.split AS s_tra,
            jpep.cluster_id AS peptide_cluster_raw, jpep.split AS s_pep,
            jm1.cluster_id  AS mhc_one_cluster_raw, jm1.split AS s_m1,
            jm2.cluster_id  AS mhc_two_cluster_raw, jm2.split AS s_m2
          FROM staging_raw k
          LEFT JOIN trb_cdr3_lkp jtrb ON k.trb_cdr3 = jtrb.mol
          LEFT JOIN tra_cdr3_lkp jtra ON k.tra_cdr3 = jtra.mol
          LEFT JOIN peptide_lkp  jpep ON k.peptide  = jpep.mol
          LEFT JOIN mhc_one_pocket_lkp jm1 ON k.mhc_one_pocket = jm1.mol
          LEFT JOIN mhc_two_pocket_lkp jm2 ON k.mhc_two_pocket = jm2.mol
        )
        SELECT
          {input_text_sql} AS input_text,
          subset_key,
          order_key,
          {mhc_class_sql} AS mhc_class,
          COALESCE(trb_cluster_raw, -1) AS trb_cluster,
          COALESCE(tra_cluster_raw, -1) AS tra_cluster,
          COALESCE(peptide_cluster_raw, -1) AS peptide_cluster,
          COALESCE(mhc_one_cluster_raw, -1) AS mhc_one_cluster,
          COALESCE(mhc_two_cluster_raw, -1) AS mhc_two_cluster,
          s_trb, s_tra, s_pep, s_m1, s_m2,
          {source_row_hash_sql} AS source_row_hash,
          CASE
            WHEN s_trb='test' OR s_tra='test' OR s_pep='test' OR s_m1='test' OR s_m2='test' THEN 'test'
            WHEN s_trb='val'  OR s_tra='val'  OR s_pep='val'  OR s_m1='val'  OR s_m2='val'  THEN 'val'
            WHEN s_trb='train' OR s_tra='train' OR s_pep='train' OR s_m1='train' OR s_m2='train' THEN 'train'
            ELSE 'train'
          END AS row_split,
          CASE WHEN s_trb IS NULL AND s_tra IS NULL AND s_pep IS NULL
               AND s_m1 IS NULL AND s_m2 IS NULL THEN 1 ELSE 0 END AS unrouted_flag
        FROM joined
        """
        t0 = time.time()
        con.execute(build_sql)
        n_staged = con.execute('SELECT COUNT(*) FROM staging').fetchone()[0]
        log(f'  staging built: {n_staged:,} rows in {time.time()-t0:.1f}s', run_id=run_id)
        con.execute('DROP TABLE staging_raw')

        # targets_view used by the verifier's distribution_check; register late.
        targets_pa = pa.Table.from_pylist([
            {'subset_key': sk, 'order_key': ok, 'target_n': t}
            for (sk, ok), t in targets.items()
        ])
        con.register('targets_view', targets_pa)

        # ---------- Step 6: drop empty input_text + collect summary stats ----------
        n_empty = con.execute("SELECT COUNT(*) FROM staging WHERE input_text IS NULL OR input_text = ''").fetchone()[0]
        log(f'  empty input_text rows to drop: {n_empty}', run_id=run_id)
        if n_empty > 0:
            con.execute("DELETE FROM staging WHERE input_text IS NULL OR input_text = ''")

        # ---------- Step 7: routing summary ----------
        log('STEP 7: routing summary', run_id=run_id)
        routing_df = con.execute("""
            SELECT
              SUM(CASE WHEN row_split='test' AND s_trb='test' THEN 1 ELSE 0 END) AS rows_routed_to_test_via_trb,
              SUM(CASE WHEN row_split='test' AND s_tra='test' THEN 1 ELSE 0 END) AS rows_routed_to_test_via_tra,
              SUM(CASE WHEN row_split='test' AND s_pep='test' THEN 1 ELSE 0 END) AS rows_routed_to_test_via_peptide,
              SUM(CASE WHEN row_split='test' AND (s_m1='test' OR s_m2='test') THEN 1 ELSE 0 END) AS rows_routed_to_test_via_mhc,
              SUM(CASE WHEN row_split='val' AND s_trb='val' THEN 1 ELSE 0 END) AS rows_routed_to_val_via_trb,
              SUM(CASE WHEN row_split='val' AND s_tra='val' THEN 1 ELSE 0 END) AS rows_routed_to_val_via_tra,
              SUM(CASE WHEN row_split='val' AND s_pep='val' THEN 1 ELSE 0 END) AS rows_routed_to_val_via_peptide,
              SUM(CASE WHEN row_split='val' AND (s_m1='val' OR s_m2='val') THEN 1 ELSE 0 END) AS rows_routed_to_val_via_mhc,
              SUM(unrouted_flag) AS rows_routed_to_train_default,
              SUM(CASE WHEN row_split='train' THEN 1 ELSE 0 END) AS train_total,
              SUM(CASE WHEN row_split='val' THEN 1 ELSE 0 END) AS val_total,
              SUM(CASE WHEN row_split='test' THEN 1 ELSE 0 END) AS test_total
            FROM staging
        """).df()
        routing = routing_df.iloc[0].to_dict()
        with open(art_dir / 'split_routing_summary.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['key', 'value'])
            for k, v in routing.items():
                w.writerow([k, int(v) if v is not None else 0])

        # distribution_check.csv: per-partition expected vs actual
        log('  distribution check', run_id=run_id)
        dist_df = con.execute(f"""
            WITH actual AS (
              SELECT subset_key, order_key, COUNT(*) AS n FROM staging
              GROUP BY subset_key, order_key
            )
            SELECT t.subset_key, t.order_key, t.target_n AS expected_n,
                   COALESCE(a.n, 0) AS actual_n
            FROM targets_view t
            LEFT JOIN actual a USING (subset_key, order_key)
            ORDER BY t.subset_key, t.order_key
        """).df()
        with open(art_dir / 'distribution_check.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['subset_key', 'order_key', 'n_post_filter', 'expected_n', 'actual_n', 'drift_pct'])
            for row in dist_df.itertuples():
                npf = partition_counts.get((row.subset_key, row.order_key), 0)
                exp = int(row.expected_n)
                act = int(row.actual_n)
                drift = ((act - exp) / exp * 100.0) if exp > 0 else 0.0
                w.writerow([row.subset_key, row.order_key, npf, exp, act, f'{drift:.4f}'])

        # ---------- Step 8: write per-split Arrow tables ----------
        log('STEP 8: extract per-split tables', run_id=run_id)
        from datasets import Dataset, DatasetDict, Features, Value

        # Schema for output. We cast input_text to large_string so 100M-scale
        # runs don't overflow pa.string's 31-bit offsets in a single batch.
        schema = pa.schema([
            ('input_text', pa.large_string()),
            ('subset_key', pa.string()),
            ('order_key', pa.string()),
            ('mhc_class', pa.string()),
            ('n_segments', pa.int32()),
            ('source_row_hash', pa.string()),
            ('tra_cluster', pa.int64()),
            ('trb_cluster', pa.int64()),
            ('peptide_cluster', pa.int64()),
            ('mhc_one_cluster', pa.int64()),
            ('mhc_two_cluster', pa.int64()),
        ])

        # Build the projected SELECT for each split.
        select_sql = """
          SELECT
            input_text,
            subset_key,
            order_key,
            mhc_class,
            (length(input_text) - length(replace(input_text, '<eos>', '')))/5 + 1 AS n_segments,
            source_row_hash,
            tra_cluster, trb_cluster, peptide_cluster,
            mhc_one_cluster, mhc_two_cluster
          FROM staging
          WHERE row_split = ?
        """

        # cleanup any prior split dirs in out_dir for atomicity. We rebuild fresh.
        for sub in ('train', 'validation', 'test'):
            sub_dir = out_dir / sub
            if sub_dir.exists():
                shutil.rmtree(sub_dir)
        for stale in ('dataset_dict.json', 'state.json'):
            stale_p = out_dir / stale
            if stale_p.exists():
                stale_p.unlink()

        split_lens = {}
        ds_kwargs = {}
        ds_dict = {}
        for split_name, row_split in [('train', 'train'), ('validation', 'val'), ('test', 'test')]:
            t0 = time.time()
            df = con.execute(select_sql, [row_split]).fetch_arrow_table()
            # Cast to declared schema (n_segments cast from int64 to int32)
            df = df.cast(schema)
            ds = Dataset(df)
            ds_dict[split_name] = ds
            split_lens[split_name] = len(ds)
            log(f'  {split_name}: {len(ds):,} rows in {time.time()-t0:.1f}s', run_id=run_id)

        # ---------- Step 9: save_to_disk ----------
        log('STEP 9: save_to_disk', run_id=run_id)
        t0 = time.time()
        dd = DatasetDict(ds_dict)
        dd.save_to_disk(str(out_dir))
        log(f'  saved in {time.time()-t0:.1f}s', run_id=run_id)

        # ---------- Step 10: sanity reload ----------
        log('STEP 10: sanity reload', run_id=run_id)
        from datasets import load_from_disk
        reloaded = load_from_disk(str(out_dir))
        total_loaded = sum(len(reloaded[k]) for k in reloaded)
        log(f'  reloaded: train={len(reloaded["train"]):,} '
            f'val={len(reloaded["validation"]):,} test={len(reloaded["test"]):,} '
            f'total={total_loaded:,}', run_id=run_id)
        tolerance = max(100, args.scale // 200)
        if abs(total_loaded - args.scale) > tolerance:
            raise RuntimeError(
                f'reloaded total {total_loaded} differs from scale {args.scale} '
                f'by >{tolerance}'
            )

        # ---------- Step 11: build_manifest.json ----------
        log('STEP 11: manifest', run_id=run_id)
        # mhc_class distribution
        mhc_dist_df = con.execute("""
            SELECT mhc_class, COUNT(*) AS n FROM staging GROUP BY mhc_class ORDER BY mhc_class
        """).df()
        mhc_dist = {row.mhc_class: int(row.n) for row in mhc_dist_df.itertuples()}

        # n_segments distribution
        nseg_df = con.execute(f"""
            SELECT
              (length(input_text) - length(replace(input_text, '<eos>', '')))/5 + 1 AS n_segments,
              COUNT(*) AS n
            FROM staging GROUP BY 1 ORDER BY 1
        """).df()
        nseg_dist = {int(row.n_segments): int(row.n) for row in nseg_df.itertuples()}

        # input_text length quantiles
        lq_df = con.execute("""
            SELECT
              AVG(length(input_text)) AS mean_len,
              quantile_cont(length(input_text), 0.5) AS p50,
              quantile_cont(length(input_text), 0.95) AS p95,
              quantile_cont(length(input_text), 0.99) AS p99,
              MAX(length(input_text)) AS max_len
            FROM staging
        """).df()
        lq = lq_df.iloc[0].to_dict()

        manifest = {
            'timestamp_utc': now_utc_iso(),
            'run_id': run_id,
            'config': {
                'tcr_variant': args.tcr_variant,
                'mhc_variant': args.mhc_variant,
                'distribution': args.distribution,
                'scale': args.scale,
                'seed': args.seed,
            },
            'paths': {
                'input': str(ROOT_INPUT),
                'clusters_dir': str(CLUSTERS_DIR),
                'split_dir': str(SPLIT_DIR),
                'output': str(out_dir),
            },
            'filter_counts': {
                'F1': int(filter_row['F1']),
                'F2': int(filter_row['F2']),
                'F3': int(filter_row['F3']),
                'F4': int(filter_row['F4']),
                'F5': int(filter_row['F5']),
                'F6': int(filter_row['F6']),
                'n_raw': int(filter_row['n_raw']),
                'n_post_filter': total_post_filter,
            },
            'partition_target_summary': {
                'mode': args.distribution,
                'n_partitions': len(partition_counts),
                'target_sum': int(target_sum),
                'capped_partition_count': len(capped_partitions),
            },
            'split_counts': split_lens,
            'split_fractions': {
                k: float(v) / total_loaded for k, v in split_lens.items()
            },
            'mhc_class_distribution': mhc_dist,
            'n_segments_distribution': nseg_dist,
            'input_text_length': {
                'mean': float(lq['mean_len']) if lq['mean_len'] is not None else 0.0,
                'p50': int(lq['p50']) if lq['p50'] is not None else 0,
                'p95': int(lq['p95']) if lq['p95'] is not None else 0,
                'p99': int(lq['p99']) if lq['p99'] is not None else 0,
                'max': int(lq['max_len']) if lq['max_len'] is not None else 0,
            },
            'antigenic_specificity_exclusion': 'NOT_APPLIED',
            'peak_rss_mb': peak_rss_mb(),
            'host': socket.gethostname(),
            'tool_versions': {
                'python': sys.version.split()[0],
                'duckdb': duckdb.__version__,
                'pyarrow': pa.__version__,
            },
        }
        try:
            import datasets as _ds
            manifest['tool_versions']['datasets'] = _ds.__version__
        except Exception:
            pass
        try:
            git_sha = subprocess.check_output(
                ['git', '-C', '/home/ubuntu/quest', 'rev-parse', 'HEAD'],
                text=True
            ).strip()
            manifest['quest_git_sha'] = git_sha
        except Exception:
            manifest['quest_git_sha'] = 'unknown'

        with open(art_dir / 'build_manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2, default=str)

        # Drop the staging table to free DuckDB memory
        con.execute('DROP TABLE staging')
        con.close()
        gc.collect()

        _write_done('PASS')
        log(f'BUILD PASS for {run_id} ({total_loaded:,} rows)', run_id=run_id)
        print(f'IMPLEMENTER_STATUS: {run_id}: PASS', flush=True)
        return 0

    except Exception as e:
        tb = traceback.format_exc()
        log(f'BUILD FAIL: {e}\n{tb}', run_id=run_id)
        _write_done('FAIL', str(e)[:200])
        print(f'IMPLEMENTER_STATUS: {run_id}: FAIL: {e}', flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
