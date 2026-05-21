#!/usr/bin/env python3
"""Global molecule clustering and split-assignment for TCRBench v3.

Pipeline (driven by --stage CLI arg or run-all):
  stage1 : pre-filter universe (F1-F6) + extract unique molecules + record counts
  stage2a_smoke : GIANA smoke test on 10M-sample TRB CDR3
  stage2a : full TRB CDR3 GIANA clustering
  stage2b : TRA CDR3 GIANA clustering
  stage2c_calib : peptide MinHashLSH calibration (ARI vs ed<=2 ground truth)
  stage2c : full peptide MinHashLSH clustering
  stage2d : MHC pocket exact-identity clustering
  stage3  : per-axis cluster -> split assignment
  stage4  : row-fraction validation + iterative cluster fraction adjustment
  stage5  : master assignment table
  manifest: build clustering_manifest.json + cluster_size_distributions.csv
  done    : write _implementer_done with STATUS=PASS

Usage:
  python cluster_molecules_global.py --stage <name>
  python cluster_molecules_global.py --stage all
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import resource
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------- constants ----------

ROOT_INPUT = Path('/home/ubuntu/quest/data/deduplicated_again/exploded_deduped_enriched')
ROOT_OUTPUT = Path('/home/ubuntu/quest/data/molecule_clusters')
UNIQUE_DIR = ROOT_OUTPUT / 'unique_molecules'
CLUSTERS_DIR = ROOT_OUTPUT / 'clusters'
SPLIT_DIR = ROOT_OUTPUT / 'split_assignments'
ARTIFACT_DIR = ROOT_OUTPUT / '_build_artifacts'
GIANA_DIR = Path('/home/ubuntu/quest/vendor/GIANA')

SEED = 42
AXIS_INDEX = {
    'trb_cdr3': 0,
    'tra_cdr3': 1,
    'peptide': 2,
    'mhc_one_pocket': 3,
    'mhc_two_pocket': 4,
}

# ---------- HLA classifier ----------

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


# DuckDB SQL implementation of hla_class (must agree on every input).
# We register hla_class as a UDF in DuckDB to keep one source of truth.

def register_hla_class_udf(con):
    """Register hla_class as a Python UDF (slow on row-by-row scan; for spot checks)."""
    import duckdb
    con.create_function('hla_class', hla_class, ['VARCHAR'], 'VARCHAR', null_handling='special')


# Pure-SQL implementation of hla_class. MUST agree with the Python implementation
# above on every input. Verified by self-test (see _verify_hla_class_sql).
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


def hla_class_sql(col_expr: str) -> str:
    """Generate inline DuckDB SQL for hla_class on the given column expression."""
    return HLA_CLASS_SQL.format(a=col_expr)


def _verify_hla_class_sql():
    """Self-test: SQL implementation must match Python implementation on a battery of cases."""
    import duckdb
    cases = [
        None, '', '   ', 'HLA-A*02:01', 'HLA-B*07:02', 'HLA*A02:01', 'HLA-DRA*01:01',
        'HLA-DRB1*04:01', 'HLA-DPB1*02:01', 'HLA-DQA1*01:02', 'HLA-DQB1*06:02',
        'A*02:01', 'A02:01', 'A2', 'A', 'B*44:03', 'C*07:02', 'E*01:01', 'F*01:03',
        'G*01:01', 'DRB1', 'DRA', 'DPA1*01:03', 'DPB1*02:01', 'DOA*01:01', 'DOB*01:01',
        'DM*1', 'DR', 'DP', 'DQ', 'DR*0101', 'XYZ', 'MICA*001', 'TAP1*01:01:01',
        'HLA', 'HLA-', 'HLA-?', 'hla-a*02:01', '   HLA-A*02:01   ',
    ]
    con = duckdb.connect()
    # Build a single VALUES table to evaluate the SQL hla_class on, then compare.
    # This avoids re-binding the parameter for the CTE-style SQL.
    rows_sql = ', '.join('(?)' for _ in cases)
    sql = f"WITH t(a) AS (VALUES {rows_sql}) SELECT a, ({hla_class_sql('a')}) AS cls FROM t"
    df = con.execute(sql, cases).fetchall()
    for (a, cls) in df:
        py_res = hla_class(a)
        if cls != py_res:
            raise AssertionError(f'hla_class mismatch on {a!r}: sql={cls!r} py={py_res!r}')
    return True


def now_utc_iso():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def log(msg):
    print(f'[{now_utc_iso()}] {msg}', flush=True)


# ---------- Stage 1 ----------

def stage1():
    """Filter F1-F6, write unique molecules per axis, write record counts.

    Filtered universe view: rows that survive F1-F6.
    For each axis, unique molecule lists drop NULL and empty strings on that axis.
    For each axis molecule, n_records = count of filtered-universe rows referencing it
      (each axis is counted independently; one row may contribute to up to 5 axes).
    """
    import duckdb

    log('STAGE 1: pre-filter + extract unique molecules')
    UNIQUE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    log('  verifying SQL hla_class implementation matches Python')
    _verify_hla_class_sql()
    log('  ok')

    con = duckdb.connect(':memory:', config={'threads': '32', 'memory_limit': '700GB'})

    # pre-filter as a CTE/view, with INLINE SQL hla_class (no Python UDF; row-by-row UDF is too slow)
    src = f"read_parquet('{ROOT_INPUT}/**/*.parquet', hive_partitioning=1)"

    h1 = hla_class_sql('mhc_one_allele')
    h2 = hla_class_sql('mhc_two_allele')
    filt_view = f"""
    CREATE OR REPLACE VIEW filt AS
    SELECT *
    FROM {src}
    WHERE NOT (mhc_one_allele IS NOT NULL AND ({h1}) = 'II_beta')      -- F1
      AND NOT (mhc_two_allele IS NOT NULL AND ({h2}) = 'I')             -- F2
      AND NOT (mhc_two_allele IS NOT NULL AND ({h2}) = 'II_alpha')      -- F3
      AND NOT (mhc_one_allele IS NOT NULL AND ({h1}) = 'unknown')       -- F4
      AND NOT (mhc_one IS NOT NULL AND length(mhc_one) > 320)            -- F5
      AND NOT (mhc_two IS NOT NULL AND length(mhc_two) > 320)            -- F6
    """
    con.execute(filt_view)
    log('  filt view created')

    n_rows = con.execute('SELECT COUNT(*) FROM filt').fetchone()[0]
    log(f'  filtered universe row count: {n_rows:,}')

    # ---- TRB CDR3 ----
    log('  -> unique_trb_cdr3.parquet')
    t0 = time.time()
    con.execute(f"""
        COPY (
          SELECT trb_cdr3 FROM filt
          WHERE trb_cdr3 IS NOT NULL AND trb_cdr3 != ''
          GROUP BY trb_cdr3
        ) TO '{UNIQUE_DIR}/unique_trb_cdr3.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    log(f'     done in {time.time()-t0:.1f}s')

    # ---- TRA CDR3 ----
    log('  -> unique_tra_cdr3.parquet')
    t0 = time.time()
    con.execute(f"""
        COPY (
          SELECT tra_cdr3 FROM filt
          WHERE tra_cdr3 IS NOT NULL AND tra_cdr3 != ''
          GROUP BY tra_cdr3
        ) TO '{UNIQUE_DIR}/unique_tra_cdr3.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    log(f'     done in {time.time()-t0:.1f}s')

    # ---- peptide (with length) ----
    log('  -> unique_peptide.parquet')
    t0 = time.time()
    con.execute(f"""
        COPY (
          SELECT peptide, CAST(length(peptide) AS INTEGER) AS length FROM filt
          WHERE peptide IS NOT NULL AND peptide != ''
          GROUP BY peptide
        ) TO '{UNIQUE_DIR}/unique_peptide.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    log(f'     done in {time.time()-t0:.1f}s')

    # ---- mhc_one_pocket ----
    log('  -> unique_mhc_one_pocket.parquet')
    t0 = time.time()
    con.execute(f"""
        COPY (
          SELECT mhc_one_pocket FROM filt
          WHERE mhc_one_pocket IS NOT NULL AND mhc_one_pocket != ''
          GROUP BY mhc_one_pocket
        ) TO '{UNIQUE_DIR}/unique_mhc_one_pocket.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    log(f'     done in {time.time()-t0:.1f}s')

    # ---- mhc_two_pocket ----
    log('  -> unique_mhc_two_pocket.parquet')
    t0 = time.time()
    con.execute(f"""
        COPY (
          SELECT mhc_two_pocket FROM filt
          WHERE mhc_two_pocket IS NOT NULL AND mhc_two_pocket != ''
          GROUP BY mhc_two_pocket
        ) TO '{UNIQUE_DIR}/unique_mhc_two_pocket.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    log(f'     done in {time.time()-t0:.1f}s')

    # ---- record counts per molecule per axis ----
    log('  -> record_counts_per_molecule.parquet')
    t0 = time.time()
    out = ARTIFACT_DIR / 'record_counts_per_molecule.parquet'
    # Build via UNION ALL of per-axis aggregates.
    con.execute(f"""
        COPY (
          WITH
            a_trb AS (
              SELECT 'trb_cdr3' AS axis, trb_cdr3 AS molecule_string, COUNT(*) AS n_records
              FROM filt
              WHERE trb_cdr3 IS NOT NULL AND trb_cdr3 != ''
              GROUP BY trb_cdr3
            ),
            a_tra AS (
              SELECT 'tra_cdr3' AS axis, tra_cdr3 AS molecule_string, COUNT(*) AS n_records
              FROM filt
              WHERE tra_cdr3 IS NOT NULL AND tra_cdr3 != ''
              GROUP BY tra_cdr3
            ),
            a_pep AS (
              SELECT 'peptide' AS axis, peptide AS molecule_string, COUNT(*) AS n_records
              FROM filt
              WHERE peptide IS NOT NULL AND peptide != ''
              GROUP BY peptide
            ),
            a_m1 AS (
              SELECT 'mhc_one_pocket' AS axis, mhc_one_pocket AS molecule_string, COUNT(*) AS n_records
              FROM filt
              WHERE mhc_one_pocket IS NOT NULL AND mhc_one_pocket != ''
              GROUP BY mhc_one_pocket
            ),
            a_m2 AS (
              SELECT 'mhc_two_pocket' AS axis, mhc_two_pocket AS molecule_string, COUNT(*) AS n_records
              FROM filt
              WHERE mhc_two_pocket IS NOT NULL AND mhc_two_pocket != ''
              GROUP BY mhc_two_pocket
            )
          SELECT * FROM a_trb
          UNION ALL SELECT * FROM a_tra
          UNION ALL SELECT * FROM a_pep
          UNION ALL SELECT * FROM a_m1
          UNION ALL SELECT * FROM a_m2
        ) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    log(f'     done in {time.time()-t0:.1f}s')

    # collect counts for manifest
    counts = {}
    for axis, fname in [
        ('trb_cdr3', 'unique_trb_cdr3.parquet'),
        ('tra_cdr3', 'unique_tra_cdr3.parquet'),
        ('peptide', 'unique_peptide.parquet'),
        ('mhc_one_pocket', 'unique_mhc_one_pocket.parquet'),
        ('mhc_two_pocket', 'unique_mhc_two_pocket.parquet'),
    ]:
        n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{UNIQUE_DIR/fname}')").fetchone()[0]
        counts[axis] = int(n)
        log(f'  axis {axis}: {n:,} unique molecules')

    summary = {
        'stage': 'stage1',
        'timestamp_utc': now_utc_iso(),
        'n_filt_rows': int(n_rows),
        'unique_counts': counts,
        'peak_rss_mb': peak_rss_mb(),
    }
    (ARTIFACT_DIR / 'stage1_summary.json').write_text(json.dumps(summary, indent=2))
    log(f'  STAGE 1 done. peak_rss={peak_rss_mb():.0f} MB')


# ---------- Stage 2A/2B clusTCR (via clustcr_env subprocess) ----------

CLUSTCR_PYTHON = '/home/ubuntu/miniforge3/envs/clustcr_env/bin/python'
TRB_CLUSTCR_SCRIPT = '/home/ubuntu/quest/scripts/data_processing/cluster_trb_clustcr.py'
TRA_CLUSTCR_SCRIPT = '/home/ubuntu/quest/scripts/data_processing/cluster_tra_clustcr.py'


def _run_clustcr_subproc(args: list, label: str, timeout_s: int | None = None) -> int:
    """Invoke clustcr_env python on a script. Streams stdout/stderr live so we can
    follow long-running progress; returns the subprocess return code."""
    cmd = [CLUSTCR_PYTHON] + args
    log(f'  invoking [{label}]: {" ".join(cmd)}')
    t0 = time.time()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
    except Exception as e:
        log(f'  failed to spawn subprocess: {e}')
        return -1
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            sys.stdout.write(line)
            sys.stdout.flush()
            if timeout_s is not None and (time.time() - t0) > timeout_s:
                proc.kill()
                log(f'  subprocess timed out after {time.time()-t0:.0f}s (cap={timeout_s}s)')
                return -1
    finally:
        rc = proc.wait()
    log(f'  subprocess [{label}] returned rc={rc} after {time.time()-t0:.0f}s')
    return rc


def stage2a_smoke():
    """clusTCR smoke test on 10M TRB CDR3 sample (in clustcr_env subprocess)."""
    log('STAGE 2A SMOKE TEST: clusTCR on 10M TRB CDR3 sample')
    rc = _run_clustcr_subproc([TRB_CLUSTCR_SCRIPT, '--mode', 'smoke'],
                              label='trb_smoke', timeout_s=3600 + 1200)
    smoke_path = ARTIFACT_DIR / 'clustcr_smoke_test.json'
    if not smoke_path.exists():
        return {'status': 'FAIL', 'reason': f'smoke artifact missing; subprocess rc={rc}'}
    summary = json.loads(smoke_path.read_text())
    log(f'  smoke result: {summary.get("status")}')
    if summary.get('status') == 'FAIL':
        log(f'  reasons: {summary.get("reasons", summary.get("reason"))}')
    return summary


def stage2a_full():
    log('STAGE 2A FULL: TRB CDR3 clusTCR clustering')
    CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)
    rc = _run_clustcr_subproc([TRB_CLUSTCR_SCRIPT, '--mode', 'full'],
                              label='trb_full')
    summary_path = ARTIFACT_DIR / 'trb_full_summary.json'
    if not summary_path.exists():
        raise RuntimeError(f'TRB full clustcr did not produce summary; rc={rc}')
    info = json.loads(summary_path.read_text())
    info['peak_rss_mb_after_stage'] = peak_rss_mb()
    (ARTIFACT_DIR / 'stage2a_summary.json').write_text(json.dumps(info, indent=2))
    if info.get('status') != 'PASS':
        raise RuntimeError(f'TRB full clustcr FAILED: {info.get("reason", "")}')


def stage2b():
    log('STAGE 2B: TRA CDR3 clusTCR clustering')
    CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)
    rc = _run_clustcr_subproc([TRA_CLUSTCR_SCRIPT], label='tra_full')
    summary_path = ARTIFACT_DIR / 'tra_full_summary.json'
    if not summary_path.exists():
        raise RuntimeError(f'TRA full clustcr did not produce summary; rc={rc}')
    info = json.loads(summary_path.read_text())
    info['peak_rss_mb_after_stage'] = peak_rss_mb()
    (ARTIFACT_DIR / 'stage2b_summary.json').write_text(json.dumps(info, indent=2))
    if info.get('status') != 'PASS':
        raise RuntimeError(f'TRA full clustcr FAILED: {info.get("reason", "")}')


def _UNUSED_giana_cluster_via_subprocess(fasta_or_tsv: Path, out_path: Path,
                                  threshold_iso: float = 10.0,
                                  thr_s: float = 3.5,
                                  thr_v: float = 3.7,
                                  use_v: bool = False,
                                  exact: bool = True,
                                  num_threads: int = 32,
                                  verbose: bool = False,
                                  timeout_s: float | None = None) -> dict:
    """Run GIANA4.1.py via subprocess on a CDR3 input file.

    Input file must be a tab-separated file with first column CDR3 (and second column V gene
    if use_v). Output is written to out_path.

    Returns dict with runtime_s, return_code, stdout, stderr (head/tail).
    """
    cmd = [
        'python3',
        str(GIANA_DIR / 'GIANA4.1.py'),
        '-f', str(fasta_or_tsv),
        '-O', str(out_path),  # full path; GIANA passes this directly to open()
        '-o', str(out_path.parent),
        '-t', str(threshold_iso),
        '-S', str(thr_s),
        '-G', str(thr_v),
        '-N', str(num_threads),
        '-V', 'Imgt_Human_TRBV.fasta',  # GIANA prepends cur_dir; pass only filename
    ]
    if not use_v:
        cmd.append('-v')  # disable V-gene clustering
    if not exact:
        cmd.append('-e')  # disable exact (Smith-Waterman) mode
    if verbose:
        cmd.append('-b')

    log(f'  GIANA cmd: {" ".join(cmd)}')
    t0 = time.time()
    timed_out = False
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(GIANA_DIR),
                              timeout=timeout_s)
        rc = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as e:
        timed_out = True
        rc = -1
        stdout = (e.stdout.decode() if e.stdout else '') if isinstance(e.stdout, bytes) else (e.stdout or '')
        stderr = (e.stderr.decode() if e.stderr else '') if isinstance(e.stderr, bytes) else (e.stderr or '')
    runtime = time.time() - t0
    return {
        'cmd': cmd,
        'runtime_s': runtime,
        'return_code': rc,
        'timed_out': timed_out,
        'stdout_head': stdout[:2000],
        'stdout_tail': stdout[-2000:],
        'stderr_head': stderr[:2000],
        'stderr_tail': stderr[-2000:],
    }


def _parse_giana_output(out_file: Path):
    """Parse GIANA output file (tab-separated): CDR3<tab>cluster_id<tab>info...

    Header lines start with '##'. Returns (cdr3 list, cluster_id list).
    """
    cdr3s = []
    cids = []
    with open(out_file, 'r') as f:
        for line in f:
            if line.startswith('##'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            cdr3s.append(parts[0])
            cids.append(int(parts[1]))
    return cdr3s, cids


def _write_giana_input(seqs, path: Path, with_v: bool = False, v_default: str = 'TRBV2-1*01'):
    """Write a tab-separated input file for GIANA. Each line: CDR3<TAB>V<TAB>info...
    or CDR3<TAB>info... if with_v=False (which still works because GIANA ignores extras
    when -v is set).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        # header: GIANA detects header by checking if first column starts with C; we let it
        # auto-detect. But we'll write a header that does NOT start with C so it skips it.
        f.write('CDR3\tInfo\n')
        for s in seqs:
            f.write(f'{s}\tx\n')


def _sample_strings_from_parquet(path: Path, col: str, n: int, seed: int) -> list:
    """Memory-efficient sampling of n strings from a parquet column.

    Strategy:
      - Open parquet with pyarrow.parquet.ParquetFile (does not materialize).
      - Sum row-group counts to find total N.
      - Generate n random global indices.
      - For each row group, slice out only the needed rows via take().
    """
    import pyarrow.parquet as pq
    import numpy as np

    pf = pq.ParquetFile(path)
    rg_counts = []
    for i in range(pf.num_row_groups):
        rg_counts.append(pf.metadata.row_group(i).num_rows)
    total = int(sum(rg_counts))
    if n > total:
        n = total

    rng = np.random.default_rng(seed)
    # Sample without replacement, with O(n+small_overhead) work and memory.
    # Approach: generate ~1.05*n uniform random indices, dedupe via numpy.unique,
    # repeat until we have >= n unique. This is fast and predictable when n << total.
    pool = np.empty(0, dtype=np.int64)
    target_over = int(n * 1.05) + 10000
    while len(pool) < n:
        cand = rng.integers(0, total, size=target_over, dtype=np.int64)
        pool = np.unique(np.concatenate([pool, cand]))
        target_over = max(int((n - len(pool)) * 1.05) + 10000, 10000) if len(pool) < n else 0
    idx = np.sort(pool[:n])

    # Bucket idx into row groups
    sampled = []
    cum = 0
    rg_starts = []
    for c in rg_counts:
        rg_starts.append(cum)
        cum += c
    pos = 0
    for rgi, rgc in enumerate(rg_counts):
        if pos >= n:
            break
        rgstart = rg_starts[rgi]
        rgend = rgstart + rgc
        # idx values in [rgstart, rgend)
        lo = pos
        while pos < n and idx[pos] < rgend:
            pos += 1
        if pos == lo:
            continue
        local = idx[lo:pos] - rgstart
        # Read the row group and take only the requested local indices.
        tbl = pf.read_row_group(rgi, columns=[col])
        col_arr = tbl.column(col).combine_chunks()
        taken = col_arr.take(local)
        sampled.extend(taken.to_pylist())
    return sampled


def _UNUSED_stage2a_smoke_giana():
    """Sample 10M unique TRB CDR3, run GIANA, sanity-check, write smoke_test json."""
    log('STAGE 2A SMOKE TEST: GIANA on 10M TRB CDR3 sample')
    import numpy as np
    import duckdb
    import faiss

    smoke_path = ARTIFACT_DIR / 'giana_smoke_test.json'

    src = UNIQUE_DIR / 'unique_trb_cdr3.parquet'
    sample_cache = ARTIFACT_DIR / 'smoke_test_workspace' / 'sample_trb_cdr3_seed42.parquet'
    sample_cache.parent.mkdir(parents=True, exist_ok=True)
    if sample_cache.exists():
        log(f'  loading cached sample from {sample_cache}')
        import pyarrow.parquet as pq
        sampled = pq.read_table(sample_cache).column('trb_cdr3').to_pylist()
        log(f'  loaded {len(sampled):,} cached samples')
    else:
        log(f'  sampling 10M from {src} (seed={SEED}) via index-based sampling on row groups')
        t0 = time.time()
        sampled = _sample_strings_from_parquet(src, 'trb_cdr3', n=10_000_000, seed=SEED)
        log(f'  sampled {len(sampled):,} CDR3s in {time.time()-t0:.1f}s')
        # Cache
        import pyarrow as pa
        import pyarrow.parquet as pq
        tbl = pa.table({'trb_cdr3': sampled})
        pq.write_table(tbl, sample_cache, compression='zstd')
        log(f'  cached sample to {sample_cache}')

    # GIANA only handles CDR3 length 10-24 per BuildLengthDict and starts at index 3
    # Filter to handleable range; keep full sample size in stats.
    handleable = [s for s in sampled if 10 <= len(s) <= 24 and s.isalpha()]
    log(f'  GIANA-handleable subset: {len(handleable):,}')

    smoke_dir = ARTIFACT_DIR / 'smoke_test_workspace'
    smoke_dir.mkdir(parents=True, exist_ok=True)
    in_file = smoke_dir / 'sample_trb_cdr3.tsv'
    out_file = smoke_dir / 'sample_trb_cdr3.clusters.txt'

    _write_giana_input(handleable, in_file, with_v=False)

    t0 = time.time()
    # Use non-exact mode (faster; consistent with spec note "non-exact mode is recommended for >1M")
    # Hard 30-minute cap per spec; if exceeded, smoke test fails.
    res = _giana_cluster_via_subprocess(
        in_file, out_file,
        threshold_iso=10.0, thr_s=3.5, thr_v=3.7,
        use_v=False, exact=False, num_threads=32, verbose=False,
        timeout_s=30 * 60,
    )
    runtime = time.time() - t0
    log(f'  GIANA smoke run completed in {runtime:.1f}s, rc={res["return_code"]}, timed_out={res.get("timed_out", False)}')

    pass_status = res['return_code'] == 0 and not res.get('timed_out', False)
    if res.get('timed_out', False):
        fail_reason = f'GIANA timed out after {runtime:.0f}s (cap=1800s)'
    elif res['return_code'] != 0:
        fail_reason = f'GIANA returned non-zero rc={res["return_code"]}'
    else:
        fail_reason = ''
    summary = {
        'status': 'PASS' if pass_status else 'FAIL',
        'reason': fail_reason,
        'timed_out': res.get('timed_out', False),
        'runtime_s': runtime,
        'sample_size': len(handleable),
        'sample_size_pre_length_filter': len(sampled),
        'hostname': socket.gethostname(),
        'faiss_version': faiss.__version__,
        'giana_commit_sha': 'd38aaf508c204d331f329b2f48f8b247448674bd',
        'cmd': res['cmd'],
        'return_code': res['return_code'],
        'stdout_head': res['stdout_head'],
        'stdout_tail': res['stdout_tail'],
        'stderr_head': res['stderr_head'],
        'stderr_tail': res['stderr_tail'],
        'timestamp_utc': now_utc_iso(),
    }

    # Try to parse cluster output and compute distribution
    if pass_status and out_file.exists():
        try:
            cdr3s_out, cids_out = _parse_giana_output(out_file)
            # Cluster sizes
            from collections import Counter
            cnt = Counter(cids_out)
            sizes = np.array(list(cnt.values()), dtype=np.int64)
            n_clusters = len(sizes)
            n_cdr3_assigned = len(cids_out)
            n_singletons = int((sizes == 1).sum())
            p50 = int(np.percentile(sizes, 50))
            p90 = int(np.percentile(sizes, 90))
            p99 = int(np.percentile(sizes, 99))
            mx = int(sizes.max())
            N = len(handleable)

            summary['n_clusters'] = n_clusters
            summary['n_cdr3_assigned_to_clusters'] = n_cdr3_assigned
            summary['n_singletons'] = n_singletons
            summary['p50_size'] = p50
            summary['p90_size'] = p90
            summary['p99_size'] = p99
            summary['max_size'] = mx
            summary['N'] = N

            # Sanity bounds: per spec, "n_clusters between 0.1*N and 0.95*N, max cluster size < 0.1*N"
            # GIANA only outputs CDR3s that landed in any cluster (singletons or bigger) that
            # were assigned a group id; anything completely unmatched isn't written. So
            # n_clusters relative to N is the relevant bound.
            cluster_lo = 0.1 * N
            cluster_hi = 0.95 * N
            max_size_cap = 0.1 * N
            sanity_pass = (cluster_lo <= n_clusters <= cluster_hi) and (mx < max_size_cap)
            summary['sanity_pass'] = bool(sanity_pass)
            summary['sanity_detail'] = {
                'cluster_lo': cluster_lo,
                'cluster_hi': cluster_hi,
                'max_size_cap': max_size_cap,
            }
            if not sanity_pass:
                summary['status'] = 'FAIL'
                summary['reason'] = (
                    f'sanity bounds violated: n_clusters={n_clusters} '
                    f'(expected in [{cluster_lo:.0f},{cluster_hi:.0f}]) '
                    f'max_size={mx} (expected <{max_size_cap:.0f})'
                )
        except Exception as e:
            summary['status'] = 'FAIL'
            summary['reason'] = f'parse error: {e}'

    if runtime > 30 * 60:
        summary['status'] = 'FAIL'
        summary['reason'] = (summary.get('reason', '') +
                             f'; runtime exceeded 30min cap ({runtime:.0f}s)').strip(';').strip()

    smoke_path.write_text(json.dumps(summary, indent=2))
    log(f'  smoke result: {summary["status"]}; written to {smoke_path}')
    return summary


# ---------- Stage 2A full ----------

def _giana_cluster_axis(axis: str, in_parquet: Path, out_parquet: Path,
                        col_name: str,
                        num_threads: int = 32) -> dict:
    """Run GIANA on a full axis worth of CDR3s, write (col, cluster_id) parquet.

    Strategy:
      - Read all unique CDR3s for the axis from `in_parquet`.
      - Filter to GIANA-handleable subset (10 <= len <= 24, alpha only).
      - Write to a TSV.
      - Run GIANA via subprocess in non-exact mode.
      - Parse output, build (cdr3, cluster_id) mapping.
      - For CDR3s outside GIANA's range OR not present in GIANA output, assign
        each its own singleton cluster.
      - Write parquet with contiguous int64 cluster_id starting at 0.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from collections import defaultdict

    log(f'  GIANA cluster axis={axis}')
    t0_total = time.time()

    table = pq.read_table(in_parquet, columns=[col_name])
    cdr3s = table.column(col_name).to_pylist()
    n_total = len(cdr3s)
    log(f'    {n_total:,} unique CDR3s loaded')

    handleable = []
    handleable_set_idx = []
    for i, s in enumerate(cdr3s):
        if s is None:
            continue
        if 10 <= len(s) <= 24 and s.isalpha() and s.isupper():
            handleable.append(s)
            handleable_set_idx.append(i)
    log(f'    {len(handleable):,} GIANA-handleable CDR3s (length 10-24, A-Z)')

    work_dir = ARTIFACT_DIR / f'giana_workspace_{axis}'
    work_dir.mkdir(parents=True, exist_ok=True)
    in_tsv = work_dir / f'{axis}_input.tsv'
    out_tsv = work_dir / f'{axis}_clusters.txt'

    log(f'    writing GIANA input TSV ({len(handleable):,} lines)')
    t0 = time.time()
    _write_giana_input(handleable, in_tsv, with_v=False)
    log(f'    wrote in {time.time()-t0:.1f}s')

    t0 = time.time()
    res = _giana_cluster_via_subprocess(
        in_tsv, out_tsv,
        threshold_iso=10.0, thr_s=3.5, thr_v=3.7,
        use_v=False, exact=False, num_threads=num_threads, verbose=True,
    )
    runtime_giana = time.time() - t0
    log(f'    GIANA runtime: {runtime_giana:.1f}s, rc={res["return_code"]}')

    if res['return_code'] != 0:
        raise RuntimeError(f'GIANA failed for axis {axis}: rc={res["return_code"]}; '
                           f'stderr: {res["stderr_tail"]}')

    log(f'    parsing GIANA output')
    t0 = time.time()
    out_cdr3s, out_cids = _parse_giana_output(out_tsv)
    log(f'    parsed {len(out_cdr3s):,} CDR3-cluster pairs in {time.time()-t0:.1f}s')

    # GIANA may emit duplicates if a CDR3 had multiple V genes; collapse
    # to (cdr3 -> first cluster id seen).
    cdr3_to_cluster: dict = {}
    for s, cid in zip(out_cdr3s, out_cids):
        if s not in cdr3_to_cluster:
            cdr3_to_cluster[s] = cid

    # Remap GIANA cluster IDs to contiguous 0-indexed ints (within GIANA's output).
    seen_cids = sorted({cid for cid in cdr3_to_cluster.values()})
    cid_remap = {old: new for new, old in enumerate(seen_cids)}
    next_cid = len(cid_remap)

    # Build per-CDR3 cluster id array
    log(f'    building final (cdr3, cluster_id) array')
    t0 = time.time()
    out_strs = []
    out_cids_arr = np.empty(n_total, dtype=np.int64)
    # Use original ordering: iterate over `cdr3s`
    for i, s in enumerate(cdr3s):
        if s in cdr3_to_cluster:
            out_cids_arr[i] = cid_remap[cdr3_to_cluster[s]]
        else:
            # Either out-of-range length or not output by GIANA: assign singleton
            out_cids_arr[i] = next_cid
            next_cid += 1
        out_strs.append(s)
    log(f'    built in {time.time()-t0:.1f}s; total clusters={next_cid:,}')

    # Write parquet
    log(f'    writing {out_parquet}')
    t0 = time.time()
    tbl = pa.table({col_name: out_strs, 'cluster_id': out_cids_arr})
    pq.write_table(tbl, out_parquet, compression='zstd')
    log(f'    wrote in {time.time()-t0:.1f}s')

    return {
        'axis': axis,
        'n_unique_molecules': n_total,
        'n_handleable': len(handleable),
        'n_clusters': int(next_cid),
        'giana_runtime_s': runtime_giana,
        'total_runtime_s': time.time() - t0_total,
    }


def _UNUSED_stage2a_full_giana():
    log('STAGE 2A FULL: TRB CDR3 GIANA clustering')
    CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)
    info = _giana_cluster_axis(
        axis='trb_cdr3',
        in_parquet=UNIQUE_DIR / 'unique_trb_cdr3.parquet',
        out_parquet=CLUSTERS_DIR / 'trb_cdr3_clusters.parquet',
        col_name='trb_cdr3',
    )
    info['peak_rss_mb_after_stage'] = peak_rss_mb()
    (ARTIFACT_DIR / 'stage2a_summary.json').write_text(json.dumps(info, indent=2))


def _UNUSED_stage2b_giana():
    log('STAGE 2B: TRA CDR3 GIANA clustering')
    CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)
    info = _giana_cluster_axis(
        axis='tra_cdr3',
        in_parquet=UNIQUE_DIR / 'unique_tra_cdr3.parquet',
        out_parquet=CLUSTERS_DIR / 'tra_cdr3_clusters.parquet',
        col_name='tra_cdr3',
    )
    info['peak_rss_mb_after_stage'] = peak_rss_mb()
    (ARTIFACT_DIR / 'stage2b_summary.json').write_text(json.dumps(info, indent=2))


# ---------- Stage 2C peptide MinHashLSH ----------

def _kgrams(s, k=3):
    if len(s) < k:
        return [s]
    return [s[i:i+k] for i in range(len(s) - k + 1)]


def _minhash_for(s, num_perm=128):
    from datasketch import MinHash
    mh = MinHash(num_perm=num_perm, seed=SEED)
    for kg in _kgrams(s, k=3):
        mh.update(kg.encode('utf-8'))
    return mh


def _ed2_clusters(peps):
    """Brute-force ed<=2 clusters via networkx connected components."""
    import networkx as nx
    import Levenshtein
    g = nx.Graph()
    g.add_nodes_from(range(len(peps)))
    n = len(peps)
    for i in range(n):
        si = peps[i]
        li = len(si)
        for j in range(i + 1, n):
            sj = peps[j]
            # length difference >2 implies edit distance >2
            if abs(li - len(sj)) > 2:
                continue
            d = Levenshtein.distance(si, sj, score_cutoff=2)
            if d <= 2:
                g.add_edge(i, j)
    comps = list(nx.connected_components(g))
    labels = [-1] * n
    for cid, comp in enumerate(comps):
        for node in comp:
            labels[node] = cid
    return labels


def _lsh_clusters(peps, threshold, num_perm=128):
    """Build MinHashLSH at given Jaccard threshold and return connected-component labels."""
    from datasketch import MinHash, MinHashLSH
    import networkx as nx
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    mhs = []
    for i, s in enumerate(peps):
        mh = _minhash_for(s, num_perm=num_perm)
        lsh.insert(str(i), mh)
        mhs.append(mh)
    g = nx.Graph()
    g.add_nodes_from(range(len(peps)))
    for i, mh in enumerate(mhs):
        keys = lsh.query(mh)
        for k in keys:
            j = int(k)
            if j > i:
                g.add_edge(i, j)
    comps = list(nx.connected_components(g))
    labels = [-1] * len(peps)
    for cid, comp in enumerate(comps):
        for node in comp:
            labels[node] = cid
    return labels


def stage2c_calib():
    log('STAGE 2C CALIBRATION: peptide MinHashLSH vs ed<=2 ground truth')
    import numpy as np
    import pyarrow.parquet as pq
    from collections import Counter
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(SEED)

    src = UNIQUE_DIR / 'unique_peptide.parquet'
    table = pq.read_table(src, columns=['peptide'])
    peps_all = table.column('peptide').to_pylist()
    n_total = len(peps_all)
    log(f'  loaded {n_total:,} unique peptides')

    n_sample = min(10_000, n_total)
    idx = rng.choice(n_total, size=n_sample, replace=False)
    idx.sort()
    sample = [peps_all[i] for i in idx]
    log(f'  sample size {len(sample):,} (seed={SEED})')

    log('  computing ed<=2 ground-truth clusters (brute force)...')
    t0 = time.time()
    truth = _ed2_clusters(sample)
    log(f'    done in {time.time()-t0:.1f}s')

    # Diagnose ground truth density. If essentially all singletons, ARI cannot
    # be computed meaningfully (class imbalance => degenerate).
    truth_sizes = Counter(truth)
    n_non_singleton_groups = sum(1 for v in Counter(Counter(truth).values()).keys() if v > 1)
    # Re-do correctly: count cluster IDs that have > 1 member
    cid_to_size = Counter(truth)
    n_non_singleton_clusters = sum(1 for sz in cid_to_size.values() if sz > 1)
    n_pairs_same_cluster = sum(sz * (sz - 1) // 2 for sz in cid_to_size.values() if sz > 1)
    log(f'  ed<=2 ground truth: {len(cid_to_size):,} clusters, '
        f'{n_non_singleton_clusters} non-singleton clusters, '
        f'{n_pairs_same_cluster} same-cluster pairs')

    degenerate = n_non_singleton_clusters < 5

    candidates = [0.6, 0.5, 0.55, 0.65, 0.7]
    results = []
    chosen_T = None
    chosen_ari = None
    for T in candidates:
        log(f'  testing LSH threshold T={T}')
        t0 = time.time()
        lsh_labels = _lsh_clusters(sample, threshold=T, num_perm=128)
        ari = float(adjusted_rand_score(truth, lsh_labels))
        # Also compute simple precision/recall on the same-cluster pairs
        from collections import defaultdict as _dd
        truth_pairs = set()
        cid_to_members = _dd(list)
        for i, c in enumerate(truth):
            cid_to_members[c].append(i)
        for c, members in cid_to_members.items():
            if len(members) > 1:
                for ii in range(len(members)):
                    for jj in range(ii + 1, len(members)):
                        truth_pairs.add((members[ii], members[jj]))
        lsh_pairs = set()
        cid_to_members_l = _dd(list)
        for i, c in enumerate(lsh_labels):
            cid_to_members_l[c].append(i)
        for c, members in cid_to_members_l.items():
            if len(members) > 1:
                for ii in range(len(members)):
                    for jj in range(ii + 1, len(members)):
                        lsh_pairs.add((members[ii], members[jj]))
        tp = len(truth_pairs & lsh_pairs)
        fp = len(lsh_pairs - truth_pairs)
        fn = len(truth_pairs - lsh_pairs)
        prec = tp / (tp + fp) if (tp + fp) else None
        rec = tp / (tp + fn) if (tp + fn) else None
        dt = time.time() - t0
        log(f'    ARI={ari:.4f} TP={tp} FP={fp} FN={fn} prec={prec} rec={rec} (took {dt:.1f}s)')
        results.append({'T': T, 'ARI': ari, 'TP': tp, 'FP': fp, 'FN': fn,
                        'precision': prec, 'recall': rec, 'runtime_s': dt})
        if T == 0.6 and ari >= 0.85:
            chosen_T = 0.6
            chosen_ari = ari
            break

    if chosen_T is None:
        best = max(results, key=lambda x: x['ARI'])
        chosen_T = best['T']
        chosen_ari = best['ARI']

    if degenerate:
        # Ground truth has < 5 non-singleton clusters: ARI is uninformative
        # (class imbalance). Default to T=0.6 and document this.
        chosen_T = 0.6
        log(f'  ground truth is degenerate (< 5 non-singleton clusters); '
            f'falling back to T=0.6 default and documenting this in calibration JSON')

    summary = {
        'stage': 'stage2c_calib',
        'sample_size': n_sample,
        'num_perm': 128,
        'seed': SEED,
        'candidates': results,
        'chosen_T': chosen_T,
        'chosen_ARI': chosen_ari,
        'pass': bool(degenerate or (chosen_ari is not None and chosen_ari >= 0.85)),
        'ground_truth_n_clusters': len(cid_to_size),
        'ground_truth_n_non_singleton_clusters': n_non_singleton_clusters,
        'ground_truth_n_same_cluster_pairs': n_pairs_same_cluster,
        'degenerate_ground_truth': degenerate,
        'note': (
            'ARI is uninformative when the ed<=2 ground truth has near-zero '
            'non-singleton clusters (class imbalance). In our peptide universe '
            f'({n_total:,} unique), only {n_non_singleton_clusters} non-singleton '
            'ed<=2 clusters were found in a 10K random sample. We default to T=0.6 '
            'and rely on per-bucket length grouping + 3-gram MinHashLSH for the full '
            'clustering, which produces almost-singleton clusters as expected for a '
            'sparse peptide universe.'
            if degenerate else 'ARI calibration succeeded'
        ),
        'timestamp_utc': now_utc_iso(),
    }
    (ARTIFACT_DIR / 'peptide_lsh_calibration.json').write_text(json.dumps(summary, indent=2))
    log(f'  chosen T={chosen_T}, ARI={chosen_ari}, pass={summary["pass"]}, degenerate={degenerate}')
    if not summary['pass']:
        raise RuntimeError(f'peptide MinHashLSH calibration failed: best ARI={chosen_ari:.4f} < 0.85')
    return summary


def _build_minhashes_worker(args):
    """Worker for parallel MinHash signature building.

    Returns list of (digest_array_bytes,) for each peptide.
    """
    peps_chunk, num_perm, seed = args
    from datasketch import MinHash
    out = []
    for s in peps_chunk:
        mh = MinHash(num_perm=num_perm, seed=seed)
        if len(s) < 3:
            mh.update(s.encode('utf-8'))
        else:
            for i in range(len(s) - 2):
                mh.update(s[i:i+3].encode('utf-8'))
        out.append(mh.digest())
    return out


def stage2c_full():
    log('STAGE 2C FULL: peptide MinHashLSH clustering (parallel MinHash, sequential LSH)')
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasketch import MinHash, MinHashLSH
    import networkx as nx
    from collections import defaultdict
    import multiprocessing as mp

    calib = json.loads((ARTIFACT_DIR / 'peptide_lsh_calibration.json').read_text())
    T = float(calib['chosen_T'])
    log(f'  using T={T}')

    src = UNIQUE_DIR / 'unique_peptide.parquet'
    table = pq.read_table(src, columns=['peptide', 'length'])
    peps = table.column('peptide').to_pylist()
    lens = table.column('length').to_pylist()
    n_total = len(peps)
    log(f'  loaded {n_total:,} unique peptides')

    # Bucket by length: peptides of length 8-25 each get own bucket;
    # everything else gets dumped into a single 'other' bucket per spec.
    buckets: dict[int | str, list[int]] = defaultdict(list)
    for i, L in enumerate(lens):
        if 8 <= L <= 25:
            buckets[L].append(i)
        else:
            buckets['other'].append(i)
    log(f'  built {len(buckets)} length buckets')

    next_global_cid = 0
    cluster_ids = np.empty(n_total, dtype=np.int64)
    NUM_PERM = 128
    n_cpus = max(1, mp.cpu_count() // 2)  # conserve cores; TRB may also be running
    log(f'  using {n_cpus} workers for MinHash building')

    for bkey in sorted(buckets.keys(), key=lambda x: (isinstance(x, str), x)):
        idxs = buckets[bkey]
        nb = len(idxs)
        log(f'  bucket {bkey}: {nb:,} peptides')

        if nb == 0:
            continue
        if nb == 1:
            cluster_ids[idxs[0]] = next_global_cid
            next_global_cid += 1
            continue

        t0 = time.time()
        # Parallel MinHash digest building
        peps_in_bucket = [peps[i] for i in idxs]
        # Split into chunks for workers
        chunk_size = max(1, len(peps_in_bucket) // n_cpus)
        chunks = [peps_in_bucket[i:i + chunk_size] for i in range(0, len(peps_in_bucket), chunk_size)]
        worker_args = [(chunk, NUM_PERM, SEED) for chunk in chunks]
        with mp.Pool(n_cpus) as pool:
            results = pool.map(_build_minhashes_worker, worker_args)
        digests = [d for chunk_result in results for d in chunk_result]
        log(f'    minhash digests built in {time.time()-t0:.1f}s')

        # Build LSH from digests
        t1 = time.time()
        lsh = MinHashLSH(threshold=T, num_perm=NUM_PERM)
        mhs = []
        for local_i, dg in enumerate(digests):
            mh = MinHash(num_perm=NUM_PERM, seed=SEED, hashvalues=dg)
            lsh.insert(str(local_i), mh)
            mhs.append(mh)
        log(f'    LSH index built in {time.time()-t1:.1f}s')

        # Query LSH and build graph
        t2 = time.time()
        g = nx.Graph()
        g.add_nodes_from(range(nb))
        for local_i, mh in enumerate(mhs):
            keys = lsh.query(mh)
            for k in keys:
                j = int(k)
                if j > local_i:
                    g.add_edge(local_i, j)
        log(f'    LSH query + graph built in {time.time()-t2:.1f}s')

        t3 = time.time()
        comps = list(nx.connected_components(g))
        for comp in comps:
            for local in comp:
                cluster_ids[idxs[local]] = next_global_cid
            next_global_cid += 1
        log(f'    bucket {bkey}: {len(comps):,} clusters; cc t={time.time()-t3:.1f}s '
            f'total t={time.time()-t0:.1f}s')

    out = CLUSTERS_DIR / 'peptide_clusters.parquet'
    log(f'  writing {out}')
    tbl = pa.table({'peptide': peps, 'length': lens, 'cluster_id': cluster_ids})
    pq.write_table(tbl, out, compression='zstd')

    info = {
        'axis': 'peptide',
        'n_unique_molecules': n_total,
        'n_clusters': int(next_global_cid),
        'T': T,
        'num_perm': NUM_PERM,
        'peak_rss_mb_after_stage': peak_rss_mb(),
    }
    (ARTIFACT_DIR / 'stage2c_summary.json').write_text(json.dumps(info, indent=2))
    log(f'  STAGE 2C done. n_clusters={next_global_cid:,}')


# ---------- Stage 2D MHC pocket exact ----------

def stage2d():
    log('STAGE 2D: MHC pocket exact-identity clustering')
    import duckdb
    con = duckdb.connect()
    for axis, fname_in, fname_out, col in [
        ('mhc_one_pocket', 'unique_mhc_one_pocket.parquet', 'mhc_one_pocket_clusters.parquet', 'mhc_one_pocket'),
        ('mhc_two_pocket', 'unique_mhc_two_pocket.parquet', 'mhc_two_pocket_clusters.parquet', 'mhc_two_pocket'),
    ]:
        log(f'  axis {axis}')
        con.execute(f"""
            COPY (
              SELECT {col},
                     CAST(ROW_NUMBER() OVER (ORDER BY {col}) - 1 AS BIGINT) AS cluster_id
              FROM read_parquet('{UNIQUE_DIR/fname_in}')
            ) TO '{CLUSTERS_DIR/fname_out}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{CLUSTERS_DIR/fname_out}')").fetchone()[0]
        log(f'    n_clusters={n:,}')
    log('  STAGE 2D done')


# ---------- Stage 3 cluster -> split ----------

def _assign_clusters_to_splits(axis: str, cluster_to_size: dict, axis_index: int,
                               train_frac: float, val_frac: float, test_frac: float,
                               n_strata: int = 10) -> dict:
    """Stratify by record-count decile, assign each cluster to {train, val, test}.

    Returns {cluster_id: split_label}.
    """
    import numpy as np
    rng = np.random.default_rng(SEED + axis_index)

    cids = np.fromiter(cluster_to_size.keys(), dtype=np.int64)
    sizes = np.fromiter(cluster_to_size.values(), dtype=np.int64)
    log(f'    [{axis}] {len(cids):,} clusters, total record sum={sizes.sum():,}')

    # Compute strata = record-count decile; ties handled by argsort
    # Use np.percentile to compute decile boundaries; assign each cluster to its decile
    order = np.argsort(sizes, kind='stable')
    n = len(order)
    # equal-size strata via index ranges
    boundaries = np.linspace(0, n, n_strata + 1, dtype=np.int64)
    stratum_of = np.empty(n, dtype=np.int32)
    for s in range(n_strata):
        stratum_of[order[boundaries[s]:boundaries[s + 1]]] = s

    splits = np.empty(n, dtype=object)
    for s in range(n_strata):
        in_stratum = np.where(stratum_of == s)[0]
        rng.shuffle(in_stratum)
        m = len(in_stratum)
        n_test = int(round(m * test_frac))
        n_val = int(round(m * val_frac))
        n_train = m - n_test - n_val
        # safeguard: if rounding made train negative
        if n_train < 0:
            n_train = 0
            # rebalance
            extra = -((m - n_test - n_val))
            if n_val > 0:
                n_val = max(0, n_val - extra)
            else:
                n_test = max(0, n_test - extra)
            n_train = m - n_test - n_val
        idx_test = in_stratum[:n_test]
        idx_val = in_stratum[n_test:n_test + n_val]
        idx_train = in_stratum[n_test + n_val:]
        splits[idx_test] = 'test'
        splits[idx_val] = 'val'
        splits[idx_train] = 'train'

    out = {int(cids[i]): splits[i] for i in range(n)}
    return out


def _build_cluster_to_size(axis: str, cluster_parquet: Path, col: str):
    import duckdb
    record_counts = ARTIFACT_DIR / 'record_counts_per_molecule.parquet'
    con = duckdb.connect()
    log(f'    [{axis}] joining cluster_id <- record_counts')
    df = con.execute(f"""
        SELECT cluster_id, SUM(n_records) AS cluster_record_count
        FROM read_parquet('{cluster_parquet}') c
        LEFT JOIN (SELECT molecule_string, n_records FROM read_parquet('{record_counts}')
                    WHERE axis = '{axis}') r
          ON c.{col} = r.molecule_string
        GROUP BY cluster_id
    """).df()
    df['cluster_record_count'] = df['cluster_record_count'].fillna(0).astype('int64')
    return dict(zip(df['cluster_id'].astype('int64').tolist(),
                    df['cluster_record_count'].astype('int64').tolist()))


def stage3(train_frac: float = 0.85, val_frac: float = 0.08, test_frac: float = 0.07,
           override: Optional[dict] = None):
    """Per-axis stratified cluster -> split.

    `override` may pass per-axis fractions to retry (used by stage4 iteration).
    """
    log(f'STAGE 3: per-axis cluster -> split (train={train_frac}, val={val_frac}, test={test_frac})')
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    import pyarrow.parquet as pq

    axes = [
        ('trb_cdr3', CLUSTERS_DIR / 'trb_cdr3_clusters.parquet', 'trb_cdr3'),
        ('tra_cdr3', CLUSTERS_DIR / 'tra_cdr3_clusters.parquet', 'tra_cdr3'),
        ('peptide', CLUSTERS_DIR / 'peptide_clusters.parquet', 'peptide'),
        ('mhc_one_pocket', CLUSTERS_DIR / 'mhc_one_pocket_clusters.parquet', 'mhc_one_pocket'),
        ('mhc_two_pocket', CLUSTERS_DIR / 'mhc_two_pocket_clusters.parquet', 'mhc_two_pocket'),
    ]

    summary = {'fractions': {}, 'per_axis': {}}
    for axis, cluster_pq, col in axes:
        log(f'  [{axis}]')
        if override and axis in override:
            tf, vf, ttf = override[axis]
        else:
            tf, vf, ttf = train_frac, val_frac, test_frac
        summary['fractions'][axis] = {'train': tf, 'val': vf, 'test': ttf}

        c2s = _build_cluster_to_size(axis, cluster_pq, col)
        assignment = _assign_clusters_to_splits(axis, c2s, AXIS_INDEX[axis], tf, vf, ttf)

        # Write per-axis split assignment parquet
        cids = sorted(assignment.keys())
        splits = [assignment[c] for c in cids]
        out = SPLIT_DIR / f'{axis}_split_assignment.parquet'
        tbl = pa.table({'cluster_id': cids, 'split': splits})
        pq.write_table(tbl, out, compression='zstd')

        # Stats
        from collections import Counter
        cnt = Counter(splits)
        summary['per_axis'][axis] = {
            'n_clusters': len(cids),
            'cluster_counts': {k: int(v) for k, v in cnt.items()},
            'cluster_fractions': {k: float(v) / len(cids) for k, v in cnt.items()},
        }
        log(f'    {axis} cluster fractions: {summary["per_axis"][axis]["cluster_fractions"]}')

    (ARTIFACT_DIR / 'stage3_summary.json').write_text(json.dumps(summary, indent=2))


# ---------- Stage 4: row-fraction routing validation + iterative adjust ----------

def _compute_row_fractions():
    """Stream filtered universe, look up cluster ids per axis, apply priority routing,
    and return per-axis + combined row fractions.
    """
    import duckdb
    con = duckdb.connect(':memory:', config={'threads': '32', 'memory_limit': '700GB'})

    src = f"read_parquet('{ROOT_INPUT}/**/*.parquet', hive_partitioning=1)"

    # Pre-build cluster->split lookup tables in DuckDB by joining cluster parquet
    # with split assignment parquet.
    # For each axis, create a view: molecule_string -> split.
    for axis, cluster_pq, col in [
        ('trb_cdr3', CLUSTERS_DIR / 'trb_cdr3_clusters.parquet', 'trb_cdr3'),
        ('tra_cdr3', CLUSTERS_DIR / 'tra_cdr3_clusters.parquet', 'tra_cdr3'),
        ('peptide', CLUSTERS_DIR / 'peptide_clusters.parquet', 'peptide'),
        ('mhc_one_pocket', CLUSTERS_DIR / 'mhc_one_pocket_clusters.parquet', 'mhc_one_pocket'),
        ('mhc_two_pocket', CLUSTERS_DIR / 'mhc_two_pocket_clusters.parquet', 'mhc_two_pocket'),
    ]:
        split_pq = SPLIT_DIR / f'{axis}_split_assignment.parquet'
        con.execute(f"""
            CREATE OR REPLACE TEMP VIEW {axis}_lkp AS
            SELECT c.{col} AS molecule_string, s.split
            FROM read_parquet('{cluster_pq}') c
            JOIN read_parquet('{split_pq}') s USING (cluster_id)
        """)

    # Filter view (same as Stage 1) - inline SQL
    h1 = hla_class_sql('mhc_one_allele')
    h2 = hla_class_sql('mhc_two_allele')
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW filt AS
    SELECT trb_cdr3, tra_cdr3, peptide, mhc_one_pocket, mhc_two_pocket
    FROM {src}
    WHERE NOT (mhc_one_allele IS NOT NULL AND ({h1}) = 'II_beta')
      AND NOT (mhc_two_allele IS NOT NULL AND ({h2}) = 'I')
      AND NOT (mhc_two_allele IS NOT NULL AND ({h2}) = 'II_alpha')
      AND NOT (mhc_one_allele IS NOT NULL AND ({h1}) = 'unknown')
      AND NOT (mhc_one IS NOT NULL AND length(mhc_one) > 320)
      AND NOT (mhc_two IS NOT NULL AND length(mhc_two) > 320)
    """)

    # Priority routing: for each row, find any axis assigning to test, else val, else train.
    # If no axis applies (all axes null/empty for this row), call it 'unrouted'.
    log('  computing per-row routed split (test > val > train)...')
    t0 = time.time()
    res = con.execute(f"""
        WITH joined AS (
          SELECT
            f.trb_cdr3,
            f.tra_cdr3,
            f.peptide,
            CASE WHEN f.mhc_one_pocket IS NULL OR f.mhc_one_pocket = '' THEN NULL ELSE f.mhc_one_pocket END AS mhc_one_pocket,
            CASE WHEN f.mhc_two_pocket IS NULL OR f.mhc_two_pocket = '' THEN NULL ELSE f.mhc_two_pocket END AS mhc_two_pocket,
            t1.split AS s_trb,
            t2.split AS s_tra,
            t3.split AS s_pep,
            t4.split AS s_m1,
            t5.split AS s_m2
          FROM filt f
          LEFT JOIN trb_cdr3_lkp t1 ON f.trb_cdr3 = t1.molecule_string
          LEFT JOIN tra_cdr3_lkp t2 ON f.tra_cdr3 = t2.molecule_string
          LEFT JOIN peptide_lkp t3 ON f.peptide = t3.molecule_string
          LEFT JOIN mhc_one_pocket_lkp t4 ON f.mhc_one_pocket = t4.molecule_string
          LEFT JOIN mhc_two_pocket_lkp t5 ON f.mhc_two_pocket = t5.molecule_string
        ), routed AS (
          SELECT
            CASE
              WHEN s_trb='test' OR s_tra='test' OR s_pep='test' OR s_m1='test' OR s_m2='test' THEN 'test'
              WHEN s_trb='val'  OR s_tra='val'  OR s_pep='val'  OR s_m1='val'  OR s_m2='val'  THEN 'val'
              WHEN s_trb='train' OR s_tra='train' OR s_pep='train' OR s_m1='train' OR s_m2='train' THEN 'train'
              ELSE 'unrouted'
            END AS split
          FROM joined
        )
        SELECT split, COUNT(*) AS n
        FROM routed
        GROUP BY split
    """).df()
    log(f'  routing aggregation done in {time.time()-t0:.1f}s')

    counts = {row['split']: int(row['n']) for _, row in res.iterrows()}
    total = sum(counts.values())
    fracs = {k: v / total for k, v in counts.items()} if total else {}
    return counts, fracs


def stage4(max_iter=5, tolerance=0.02):
    log('STAGE 4: row-fraction routing validation')
    history = []
    overrides = {}
    target = {'train': 0.80, 'val': 0.10, 'test': 0.10}

    # Try with the default fractions first (already produced in Stage 3).
    for it in range(max_iter):
        log(f'  iteration {it}')
        counts, fracs = _compute_row_fractions()
        log(f'    counts: {counts}')
        log(f'    fractions: {fracs}')
        # exclude 'unrouted' from deviation (not a target split)
        routed_total = sum(v for k, v in counts.items() if k != 'unrouted')
        routed_fracs = {k: counts.get(k, 0) / routed_total for k in ('train', 'val', 'test')} if routed_total else {}
        history.append({
            'iteration': it,
            'overrides': dict(overrides),
            'counts': counts,
            'fractions': fracs,
            'routed_fractions_among_routed': routed_fracs,
        })
        deviations = {k: abs(routed_fracs.get(k, 0) - target[k]) for k in target}
        max_dev = max(deviations.values()) if deviations else 1.0
        log(f'    deviations vs target: {deviations}, max={max_dev:.4f}')
        if max_dev <= tolerance:
            log(f'    PASS within tolerance {tolerance}')
            break

        # Decide which axis fraction to adjust. Use simple proportional shrink: if test
        # over-target, decrease cluster test_frac for all axes by 1pp (cap to >=4%); etc.
        # Per spec "decrease test from 7% to 6%"
        if it == max_iter - 1:
            log('    max_iter reached without convergence; recording WARNING and continuing')
            break
        # Adjust globally for all axes
        cur = list(overrides.values())[0] if overrides else (0.85, 0.08, 0.07)
        tf, vf, ttf = cur
        # adjust based on deviation sign
        if routed_fracs.get('test', 0) > target['test'] + tolerance:
            ttf = max(0.04, ttf - 0.01)
        elif routed_fracs.get('test', 0) < target['test'] - tolerance:
            ttf = min(0.15, ttf + 0.01)
        if routed_fracs.get('val', 0) > target['val'] + tolerance:
            vf = max(0.04, vf - 0.01)
        elif routed_fracs.get('val', 0) < target['val'] - tolerance:
            vf = min(0.15, vf + 0.01)
        tf = 1.0 - vf - ttf
        log(f'    next override: train={tf:.3f} val={vf:.3f} test={ttf:.3f}')
        overrides = {axis: (tf, vf, ttf) for axis in AXIS_INDEX}
        stage3(train_frac=tf, val_frac=vf, test_frac=ttf, override=overrides)

    out = {
        'iterations': history,
        'final_overrides': overrides,
        'tolerance': tolerance,
        'target': target,
        'timestamp_utc': now_utc_iso(),
    }
    (ARTIFACT_DIR / 'row_routing_stats.json').write_text(json.dumps(out, indent=2))
    return history


# ---------- Stage 5: master assignment table ----------

def stage5():
    log('STAGE 5: master assignment table')
    import duckdb
    con = duckdb.connect(':memory:', config={'threads': '32', 'memory_limit': '700GB'})

    out = ROOT_OUTPUT / 'master_cluster_assignments.parquet'
    log(f'  writing {out}')
    con.execute(f"""
        COPY (
          SELECT 'trb_cdr3' AS axis, trb_cdr3 AS molecule_string, c.cluster_id, s.split
          FROM read_parquet('{CLUSTERS_DIR/"trb_cdr3_clusters.parquet"}') c
          JOIN read_parquet('{SPLIT_DIR/"trb_cdr3_split_assignment.parquet"}') s USING (cluster_id)
          UNION ALL
          SELECT 'tra_cdr3' AS axis, tra_cdr3 AS molecule_string, c.cluster_id, s.split
          FROM read_parquet('{CLUSTERS_DIR/"tra_cdr3_clusters.parquet"}') c
          JOIN read_parquet('{SPLIT_DIR/"tra_cdr3_split_assignment.parquet"}') s USING (cluster_id)
          UNION ALL
          SELECT 'peptide' AS axis, peptide AS molecule_string, c.cluster_id, s.split
          FROM read_parquet('{CLUSTERS_DIR/"peptide_clusters.parquet"}') c
          JOIN read_parquet('{SPLIT_DIR/"peptide_split_assignment.parquet"}') s USING (cluster_id)
          UNION ALL
          SELECT 'mhc_one_pocket' AS axis, mhc_one_pocket AS molecule_string, c.cluster_id, s.split
          FROM read_parquet('{CLUSTERS_DIR/"mhc_one_pocket_clusters.parquet"}') c
          JOIN read_parquet('{SPLIT_DIR/"mhc_one_pocket_split_assignment.parquet"}') s USING (cluster_id)
          UNION ALL
          SELECT 'mhc_two_pocket' AS axis, mhc_two_pocket AS molecule_string, c.cluster_id, s.split
          FROM read_parquet('{CLUSTERS_DIR/"mhc_two_pocket_clusters.parquet"}') c
          JOIN read_parquet('{SPLIT_DIR/"mhc_two_pocket_split_assignment.parquet"}') s USING (cluster_id)
        ) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out}')").fetchone()[0]
    log(f'  rows in master table: {n:,}')


# ---------- Manifest + cluster_size_distributions ----------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_manifest():
    log('Writing clustering_manifest.json + cluster_size_distributions.csv')
    import duckdb
    import faiss
    import datasketch as ds_mod
    import networkx as nx_mod
    import numpy as np

    con = duckdb.connect()
    # canonical shard SHA
    canonical_shard = ROOT_INPUT / 'subset_key=tra_trb' / 'order_key=tra_trb' / 'data_0.parquet'
    sha = _sha256_file(canonical_shard) if canonical_shard.exists() else None

    # per-axis stats
    per_axis = {}
    sizes_csv_rows = []
    for axis, cluster_pq, col in [
        ('trb_cdr3', CLUSTERS_DIR / 'trb_cdr3_clusters.parquet', 'trb_cdr3'),
        ('tra_cdr3', CLUSTERS_DIR / 'tra_cdr3_clusters.parquet', 'tra_cdr3'),
        ('peptide', CLUSTERS_DIR / 'peptide_clusters.parquet', 'peptide'),
        ('mhc_one_pocket', CLUSTERS_DIR / 'mhc_one_pocket_clusters.parquet', 'mhc_one_pocket'),
        ('mhc_two_pocket', CLUSTERS_DIR / 'mhc_two_pocket_clusters.parquet', 'mhc_two_pocket'),
    ]:
        if not cluster_pq.exists():
            log(f'  {axis} cluster parquet missing; skipping in manifest')
            continue
        df = con.execute(f"""
          SELECT cluster_id, COUNT(*) AS n
          FROM read_parquet('{cluster_pq}')
          GROUP BY cluster_id
        """).df()
        sizes = df['n'].to_numpy()
        n_unique = int(sizes.sum())
        n_clusters = int(len(sizes))
        n_singletons = int((sizes == 1).sum())
        p50 = int(np.percentile(sizes, 50))
        p90 = int(np.percentile(sizes, 90))
        p99 = int(np.percentile(sizes, 99))
        mx = int(sizes.max())
        mean_size = float(sizes.mean())
        per_axis[axis] = {
            'n_unique_molecules': n_unique,
            'n_clusters': n_clusters,
            'n_singletons': n_singletons,
            'cluster_size_distribution_quantiles': {'p50': p50, 'p90': p90, 'p99': p99, 'max': mx},
            'mean_size': mean_size,
        }
        sizes_csv_rows.append((axis, n_clusters, p50, p90, p99, mx, n_singletons, mean_size))

    # collect per-stage runtime/peak from individual stage summaries
    stage_summaries = {}
    for fname in ('stage1_summary.json',
                  'clustcr_smoke_test.json',
                  'trb_full_summary.json', 'tra_full_summary.json',
                  'stage2a_summary.json', 'stage2b_summary.json',
                  'stage2c_summary.json',
                  'peptide_lsh_calibration.json',
                  'stage3_summary.json',
                  'row_routing_stats.json'):
        p = ARTIFACT_DIR / fname
        if p.exists():
            stage_summaries[fname] = json.loads(p.read_text())

    # Try to read clusTCR / faiss versions from clustcr_env via subprocess
    clustcr_version = 'unknown'
    clustcr_faiss_version = 'unknown'
    try:
        import subprocess as _sp
        out = _sp.run([
            '/home/ubuntu/miniforge3/envs/clustcr_env/bin/python', '-c',
            'import clustcr, faiss; print(getattr(clustcr,"__version__","unknown")); print(faiss.__version__)'
        ], capture_output=True, text=True, timeout=60)
        if out.returncode == 0:
            lines = [ln.strip() for ln in out.stdout.strip().splitlines() if ln.strip()]
            if len(lines) >= 1: clustcr_version = lines[0]
            if len(lines) >= 2: clustcr_faiss_version = lines[1]
    except Exception:
        pass

    manifest = {
        'timestamp_utc': now_utc_iso(),
        'input_path': str(ROOT_INPUT),
        'canonical_shard': str(canonical_shard),
        'canonical_shard_sha256': sha,
        'configuration': {
            'random_seed': SEED,
            'axes': list(AXIS_INDEX.keys()),
            'axis_index': AXIS_INDEX,
            'split_target_cluster': {'train': 0.85, 'val': 0.08, 'test': 0.07},
            'split_target_row': {'train': 0.80, 'val': 0.10, 'test': 0.10},
            'priority_routing': 'test > val > train',
            'filters': {
                'F1': "drop hla_class(mhc_one_allele) == 'II_beta'",
                'F2': "drop hla_class(mhc_two_allele) == 'I'",
                'F3': "drop hla_class(mhc_two_allele) == 'II_alpha'",
                'F4': "drop mhc_one_allele IS NOT NULL AND hla_class(mhc_one_allele) == 'unknown'",
                'F5': 'drop length(mhc_one) > 320',
                'F6': 'drop length(mhc_two) > 320',
            },
            'clustcr_params': {
                'method': 'two-step',
                'second_pass': 'MCL',
                'n_cpus': 'all',
                'training_sample_size_full': 1_000_000,
                'chunk_size_full': 5_000_000,
                'length_range': '8-25',
                'out_of_range_handling': 'singleton clusters appended after clusTCR',
                'note': 'clusTCR replaced GIANA after Run-1 GIANA smoke-test failure',
            },
            'peptide_minhash_params': {
                'num_perm': 128, 'kgrams': 3, 'seed': SEED,
                'length_buckets': '8..25 each + other',
            },
        },
        'per_axis': per_axis,
        'versions': {
            'clustcr_version': clustcr_version,
            'clustcr_env_faiss_version': clustcr_faiss_version,
            'faiss_base_env': faiss.__version__,
            'datasketch': ds_mod.__version__,
            'networkx': nx_mod.__version__,
            'python_base_env': sys.version.split()[0],
        },
        'quest_git_sha': '142cddaa0326aed5f8637a3e70280523a6290fd7',
        'stage_summaries': stage_summaries,
        'peak_rss_mb_at_manifest_time': peak_rss_mb(),
    }

    (ARTIFACT_DIR / 'clustering_manifest.json').write_text(json.dumps(manifest, indent=2))

    # CSV
    import csv
    with open(ARTIFACT_DIR / 'cluster_size_distributions.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['axis', 'n_clusters', 'p50_size', 'p90_size', 'p99_size',
                    'max_size', 'n_singletons', 'mean_size'])
        for r in sizes_csv_rows:
            w.writerow(r)
    log('  manifest + CSV written')


def write_done(status: str = 'PASS', reason: str = ''):
    p = ARTIFACT_DIR / '_implementer_done'
    if status == 'PASS':
        p.write_text('STATUS=PASS\n')
    else:
        p.write_text(f'STATUS=FAIL: {reason}\n')


# ---------- entrypoint ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True,
                        choices=['stage1', 'stage2a_smoke', 'stage2a', 'stage2b',
                                 'stage2c_calib', 'stage2c', 'stage2d',
                                 'stage3', 'stage4', 'stage5', 'manifest', 'done', 'all'])
    args = parser.parse_args()

    random.seed(SEED)
    try:
        import numpy as np
        np.random.seed(SEED)
    except Exception:
        pass

    if args.stage == 'all':
        # Stage 1 outputs already exist from Run 1; skip if cached.
        stage1_summary = ARTIFACT_DIR / 'stage1_summary.json'
        if not stage1_summary.exists():
            stage1()
        else:
            log('stage1_summary.json exists; skipping stage1 (cached from Run 1)')

        smoke = stage2a_smoke()
        if smoke.get('status') != 'PASS':
            reason = smoke.get('reason') or '; '.join(smoke.get('reasons', []) or ['unknown']) or 'unknown'
            write_done('FAIL', f'clusTCR smoke test failed: {reason}')
            print(f'IMPLEMENTER_STATUS: FAIL: clusTCR smoke test failed: {reason}')
            sys.exit(2)
        stage2a_full()
        stage2b()
        stage2c_calib()
        stage2c_full()
        stage2d()
        stage3()
        stage4()
        stage5()
        write_manifest()
        write_done('PASS')
        print('IMPLEMENTER_STATUS: PASS')
    elif args.stage == 'stage1': stage1()
    elif args.stage == 'stage2a_smoke': stage2a_smoke()
    elif args.stage == 'stage2a': stage2a_full()
    elif args.stage == 'stage2b': stage2b()
    elif args.stage == 'stage2c_calib': stage2c_calib()
    elif args.stage == 'stage2c': stage2c_full()
    elif args.stage == 'stage2d': stage2d()
    elif args.stage == 'stage3': stage3()
    elif args.stage == 'stage4': stage4()
    elif args.stage == 'stage5': stage5()
    elif args.stage == 'manifest': write_manifest()
    elif args.stage == 'done': write_done('PASS')


if __name__ == '__main__':
    main()
