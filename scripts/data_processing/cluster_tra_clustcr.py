"""TRA CDR3 clustering with clusTCR (single-call fit on 29M sequences).

Runs inside clustcr_env (Python 3.10).

Strategy:
- Use clusTCR.fit() with method='two-step' (no batch API).
- Force faiss_cluster_size=200 to keep ncentroids ~ 145 (training=29M).
  Hmm wait, with single-call fit, training_data == fitting_data == 29M.
  Default faiss_cluster_size=5000 → ncentroids = 5800.
  Avg precluster = 5000 (by design).
  K-means search: 29M × 5800 × 225 = 38T → ~20 min on Graviton with single-thread.
  K-means TRAINING: 29M × 5800 × 225 × 25 = 945T → 8.7 hours. TOO SLOW.

- Solution: use the BATCH API with training=200K, override=10 (similar to TRB)
  but smaller ncentroids and chunk size since data is 30x smaller.
  training=200K, override=10 → ncentroids=20K. Avg precluster = 29M/20K = 1450.
  Training: 200K × 20K × 225 × 25 = 22.5T → 750s = 13 min.
  Search: 29M × 20K × 225 / IVF speedup = 130T / 100GFLOPS = 1300s = 22 min (with IVF).
  MCL: 20K iters / 64 = 313 batches × ~2s/MCL on 1450 seqs = 626s = 10 min.

- Total: ~45 min.

Usage:
  python cluster_tra_clustcr.py
"""
from __future__ import annotations

import json
import platform
import resource
import shutil
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

ROOT_OUTPUT = Path('/home/ubuntu/quest/data/molecule_clusters')
UNIQUE_DIR = ROOT_OUTPUT / 'unique_molecules'
CLUSTERS_DIR = ROOT_OUTPUT / 'clusters'
ARTIFACT_DIR = ROOT_OUTPUT / '_build_artifacts'
TRA_SHARDS_DIR = ARTIFACT_DIR / 'tra_cdr3_cluster_shards'

SEED = 42
LEN_MIN = 8
LEN_MAX = 25


def now_utc_iso():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def log(msg):
    print(f'[{now_utc_iso()}] [tra_clustcr] {msg}', flush=True)


def main():
    log('=== TRA CDR3 clusTCR FULL (using batch API for scaling) ===')
    log(f'hostname={socket.gethostname()}, platform={platform.platform()}')

    # Import the TRB helper module (reuse cluster_via_batch + reassign_and_fill_missing)
    sys.path.insert(0, '/home/ubuntu/quest/scripts/data_processing')
    import cluster_trb_clustcr as trb

    import clustcr
    try:
        import faiss
        faiss_ver = faiss.__version__
    except Exception:
        faiss_ver = 'unknown'

    CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)

    src = UNIQUE_DIR / 'unique_tra_cdr3.parquet'
    log(f'reading {src}')
    t = pq.read_table(src, columns=['tra_cdr3'])
    arr = t.column('tra_cdr3')
    n_total = arr.length()
    log(f'  total rows: {n_total:,}')

    lens = pc.utf8_length(arr)
    in_mask = pc.and_(pc.greater_equal(lens, LEN_MIN), pc.less_equal(lens, LEN_MAX))
    out_mask = pc.invert(in_mask)
    in_arr = arr.filter(in_mask).combine_chunks()
    out_arr = arr.filter(out_mask).combine_chunks()
    n_in = len(in_arr)
    n_out = len(out_arr)
    log(f'  in-range [{LEN_MIN},{LEN_MAX}]: {n_in:,}')
    log(f'  out-of-range: {n_out:,}')

    # Run batch clustering via the TRB helper (column-name awareness)
    in_chunked = pa.chunked_array([in_arr])

    t_overall = time.time()
    try:
        # TRA strategy: training=200K, faiss_cluster_size_override=10 (ncentroids=20K),
        # chunk_size=5M (only ~6 chunks for 29M), IVF for fast search.
        # Override col_name to 'tra_cdr3' for the shard files; use trb pipeline routines.
        # We need to monkey-patch the col_name into helpers since they're hard-coded to 'trb_cdr3'.
        # Simplest: write 'tra_cdr3' shards directly into TRA_SHARDS_DIR by reusing trb's
        # cluster_via_batch but it writes 'trb_cdr3' as the column name. We then rename.

        # Reuse trb.cluster_via_batch (it stores under trb_cdr3 column) — we'll rename later
        # by reading shards and renaming column.

        stats = trb.cluster_via_batch(
            seqs_arr=in_chunked,
            training_sample_size=200_000,
            chunk_size=5_000_000,
            shards_dir=TRA_SHARDS_DIR,
            max_seq_size=LEN_MAX,
            stage_label='tra_full',
            use_ivf=True,
            ivf_nlist=512,
            ivf_nprobe=16,
            faiss_cluster_size_override=10,
        )
    except Exception as e:
        tb = traceback.format_exc()
        log(f'EXCEPTION: {e}\n{tb}')
        (ARTIFACT_DIR / 'tra_full_summary.json').write_text(json.dumps({
            'status': 'FAIL',
            'reason': f'clustcr exception: {type(e).__name__}: {e}',
            'traceback': tb,
            'elapsed_s': time.time() - t_overall,
            'timestamp_utc': now_utc_iso(),
        }, indent=2))
        sys.exit(2)

    log(f'TRA clustering done in {time.time()-t_overall:.1f}s')

    # Concatenate + fill in missing (clusTCR-dropped) seqs as singletons.
    # Helper writes shards with column 'trb_cdr3' (hard-coded in cluster_via_batch).
    # Pass col_name='trb_cdr3' to the helper, then rename column on final concat.
    concat_path, n_cids_in = trb.reassign_and_fill_missing(
        TRA_SHARDS_DIR, in_arr, col_name='trb_cdr3')
    log(f'concat parquet at {concat_path}, n_cids_after_fill={n_cids_in:,}')

    # Append out-of-range singletons
    log(f'appending {n_out:,} out-of-range singletons')
    out_seqs = out_arr.to_pylist()
    out_cids = np.arange(n_cids_in, n_cids_in + n_out, dtype=np.int64)
    n_total_clusters = n_cids_in + n_out

    final_path = CLUSTERS_DIR / 'tra_cdr3_clusters.parquet'
    schema = pa.schema([pa.field('tra_cdr3', pa.large_string()),
                        pa.field('cluster_id', pa.int64())])
    writer = pq.ParquetWriter(final_path, schema=schema, compression='zstd')
    # Read concat (column name is 'trb_cdr3' from helper) and rename.
    concat_tbl = pq.read_table(concat_path)
    concat_renamed = concat_tbl.rename_columns(['tra_cdr3', 'cluster_id'])
    writer.write_table(concat_renamed.cast(schema))
    if n_out > 0:
        sing_tbl = pa.table({
            'tra_cdr3': pa.array(out_seqs, type=pa.large_string()),
            'cluster_id': pa.array(out_cids, type=pa.int64()),
        })
        writer.write_table(sing_tbl)
    writer.close()
    final_n = pq.read_metadata(final_path).num_rows
    expected = n_in + n_out
    log(f'final parquet rows={final_n:,}, expected={expected:,}, n_clusters={n_total_clusters:,}')

    summary = {
        'status': 'PASS' if final_n == expected else 'FAIL',
        'n_unique_input': int(expected),
        'n_in_range': int(n_in),
        'n_out_of_range': int(n_out),
        'n_clusters_clustcr': int(n_cids_in),
        'n_clusters_singletons': int(n_out),
        'n_clusters_total': int(n_total_clusters),
        'final_parquet_rows': int(final_n),
        'elapsed_s': float(time.time() - t_overall),
        'clustcr_version': getattr(clustcr, '__version__', 'unknown'),
        'faiss_version': faiss_ver,
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'timestamp_utc': now_utc_iso(),
        'peak_rss_mb': peak_rss_mb(),
        'len_min': LEN_MIN,
        'len_max': LEN_MAX,
    }
    (ARTIFACT_DIR / 'tra_full_summary.json').write_text(json.dumps(summary, indent=2))
    log(f'wrote tra_full_summary.json status={summary["status"]}')
    sys.exit(0 if summary['status'] == 'PASS' else 2)


if __name__ == '__main__':
    main()
