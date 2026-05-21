"""TRB CDR3 clustering with clusTCR batch API.

Runs inside clustcr_env (Python 3.10).

Usage:
  python cluster_trb_clustcr.py --mode smoke
  python cluster_trb_clustcr.py --mode full

Inputs:
  /home/ubuntu/quest/data/molecule_clusters/unique_molecules/unique_trb_cdr3.parquet

Outputs:
  --mode smoke:
    /home/ubuntu/quest/data/molecule_clusters/_build_artifacts/clustcr_smoke_test.json
  --mode full:
    /home/ubuntu/quest/data/molecule_clusters/clusters/trb_cdr3_clusters.parquet
    /home/ubuntu/quest/data/molecule_clusters/_build_artifacts/trb_full_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
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

# clusTCR imports deferred to inside try blocks so we can capture import-time tracebacks.

ROOT_OUTPUT = Path('/home/ubuntu/quest/data/molecule_clusters')
UNIQUE_DIR = ROOT_OUTPUT / 'unique_molecules'
CLUSTERS_DIR = ROOT_OUTPUT / 'clusters'
ARTIFACT_DIR = ROOT_OUTPUT / '_build_artifacts'
SHARDS_DIR = ARTIFACT_DIR / 'trb_cdr3_cluster_shards'
SMOKE_SHARDS_DIR = ARTIFACT_DIR / 'trb_cdr3_smoke_shards'

SEED = 42
LEN_MIN = 8
LEN_MAX = 25

CHUNK_SIZE_FULL = 1_000_000  # rows per batch_precluster call (was 5M; reduced for finer progress + better thread utilization)
CHUNK_SIZE_SMOKE = 1_000_000
TRAINING_SAMPLE_SIZE = 1_000_000  # for FAISS training (full)
SMOKE_TRAIN_SAMPLE = 100_000

PROGRESS_PATH = ARTIFACT_DIR / 'clustcr_progress.json'


def now_utc_iso():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def log(msg):
    print(f'[{now_utc_iso()}] [trb_clustcr] {msg}', flush=True)


def write_progress(stage: str, n_chunks_processed: int, n_seqs_processed: int,
                   elapsed_s: float, extra: dict | None = None):
    payload = {
        'stage': stage,
        'n_chunks_processed': n_chunks_processed,
        'n_seqs_processed': n_seqs_processed,
        'elapsed_s': elapsed_s,
        'timestamp_utc': now_utc_iso(),
    }
    if extra:
        payload.update(extra)
    PROGRESS_PATH.write_text(json.dumps(payload, indent=2))


def load_trb_in_range(unique_path: Path):
    """Read unique TRB CDR3 column; split into in-range (8..25) and out-of-range arrays."""
    log(f'reading {unique_path}')
    t = pq.read_table(unique_path, columns=['trb_cdr3'])
    # Use large_string to avoid 32-bit offset overflow on full-table take()
    t = t.cast(pa.schema([pa.field('trb_cdr3', pa.large_string())]))
    arr = t.column('trb_cdr3')
    # arr is a ChunkedArray of LargeStringArray
    log(f'  total rows: {arr.length():,}')
    # Compute lengths via pyarrow.compute
    import pyarrow.compute as pc
    lens = pc.utf8_length(arr)
    in_mask = pc.and_(pc.greater_equal(lens, LEN_MIN), pc.less_equal(lens, LEN_MAX))
    out_mask = pc.invert(in_mask)
    in_arr = arr.filter(in_mask)
    out_arr = arr.filter(out_mask)
    log(f'  in-range [{LEN_MIN},{LEN_MAX}]: {in_arr.length():,}')
    log(f'  out-of-range: {out_arr.length():,}')
    return in_arr, out_arr


def _swap_kmeans_to_ivf(clustering, nlist: int = None, nprobe: int = 16):
    """Replace clustering.faiss_clustering.kmeans.index (IndexFlatL2) with an
    IVF index built on the trained centroids. Speeds up subsequent
    .index.search() by ~nlist/nprobe x at very small accuracy cost.

    Must be called AFTER faiss training; before any batch_precluster.
    """
    import faiss
    fc = clustering.faiss_clustering
    assert fc.kmeans is not None, '_swap_kmeans_to_ivf requires trained Kmeans'
    centroids = fc.kmeans.centroids  # ndarray (k, d)
    k, d = centroids.shape
    if nlist is None:
        nlist = max(64, int(np.sqrt(k)))
    nlist = min(nlist, k)  # nlist must be <= k
    nprobe = min(nprobe, nlist)
    log(f'  _swap_kmeans_to_ivf: ncentroids={k}, dim={d}, nlist={nlist}, nprobe={nprobe}')

    quantizer = faiss.IndexFlatL2(d)
    ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    # Train IVF on the centroids themselves.
    if not ivf.is_trained:
        ivf.train(np.ascontiguousarray(centroids, dtype='float32'))
    ivf.add(np.ascontiguousarray(centroids, dtype='float32'))
    ivf.nprobe = nprobe
    fc.kmeans.index = ivf
    log(f'  IVF index swapped in (nlist={nlist}, nprobe={nprobe})')


def _fast_batch_precluster(clustering, cdr3_series: pd.Series, name: str = ''):
    """Vectorized replacement for Clustering.batch_precluster that avoids
    per-row open/append. Maintains the same on-disk format expected by
    Clustering._batch_process_preclusters: file '<cluster_id>' in
    BATCH_TMP_DIRECTORY with lines '<seq>,<name>\\n'.

    This is byte-identical to upstream output but ~50x faster on Graviton.
    """
    from os.path import join as _join
    assert clustering.faiss_clustering is not None, 'fast_batch_precluster needs trained faiss'
    # Run the FAISS K-means assignment (CPU multi-process via parmap inside clusTCR)
    clustered = clustering._faiss(cdr3_series)
    df = clustered.clusters_df  # has 'junction_aa' and 'cluster' columns
    # Group by cluster id, write each group's sequences to a single file in append mode.
    tmp_dir = type(clustering).BATCH_TMP_DIRECTORY
    suffix = f',{name}\n'
    for cid, grp in df.groupby('cluster', sort=False):
        path = _join(tmp_dir, str(int(cid)))
        # Build the bytes once and write in one syscall.
        # Each row: '<seq>,<name>\n'
        seqs_arr = grp['junction_aa'].to_numpy()
        # Use list comprehension (faster than join via str.add); single open/write/close.
        body = ''.join(s + suffix for s in seqs_arr)
        with open(path, 'a') as f:
            f.write(body)


def _fast_batch_cluster(clustering):
    """Faster replacement for Clustering.batch_cluster that uses a persistent
    multiprocessing Pool to amortize 64-worker startup cost across all iterations.

    Yields ClusteringResult objects, identical to clustering.batch_cluster().
    """
    import multiprocessing as _mp
    from clustcr.clustering.methods import MCL_multi
    from clustcr.clustering.tools import create_edgelist
    from clustcr.clustering.multirepertoire_cluster_matrix import MultiRepertoireClusterMatrix
    from clustcr.clustering.clustering import ClusteringResult
    import parmap as _parmap
    import pandas as _pd

    clustering.cluster_matrix = MultiRepertoireClusterMatrix()
    clusters_per_batch = max(1, min(clustering.n_cpus,
                                    50000 // clustering.faiss_cluster_size))
    npreclusters = clustering.faiss_clustering.ncentroids()
    log(f'  _fast_batch_cluster: ncentroids={npreclusters}, '
        f'clusters_per_batch={clusters_per_batch}, n_cpus={clustering.n_cpus}')

    pool = _mp.Pool(clustering.n_cpus)
    try:
        max_cluster_id = 0
        for i in range(0, npreclusters, clusters_per_batch):
            cluster_ids = range(i, min(i + clusters_per_batch, npreclusters))
            preclusters = clustering._batch_process_preclusters(cluster_ids)

            # Build edges via list comprehension; parallel-MCL only on those with edges
            cluster_contents = preclusters.cluster_contents()
            edges = {idx: create_edgelist(cluster) for idx, cluster in enumerate(cluster_contents)}
            clean = {idx: e for idx, e in edges.items() if len(e) > 0}
            remaining_edges = list(clean.values())

            if remaining_edges:
                cdr3 = None
                nodelist = _parmap.map(MCL_multi, remaining_edges, cdr3,
                                       mcl_hyper=clustering.mcl_params,
                                       pm_parallel=True, pm_pool=pool)
                # Re-id clusters globally unique
                for c in range(len(nodelist)):
                    if c != 0:
                        prev_max = nodelist[c - 1]['cluster'].max()
                        if _pd.notna(prev_max):
                            nodelist[c]['cluster'] += int(prev_max) + 1
                if nodelist:
                    mcl_result = _pd.concat(nodelist, ignore_index=True)
                else:
                    mcl_result = _pd.DataFrame({'junction_aa': [], 'cluster': []})
            else:
                mcl_result = _pd.DataFrame({'junction_aa': [], 'cluster': []})

            if len(mcl_result) > 0:
                mcl_result['cluster'] += max_cluster_id + 1
                cmax = mcl_result['cluster'].max()
                if _pd.notna(cmax):
                    max_cluster_id = int(cmax)
            yield ClusteringResult(mcl_result, chain=clustering.chain)
    finally:
        pool.close()
        pool.join()


def cluster_via_batch(seqs_arr, training_sample_size: int, chunk_size: int,
                      shards_dir: Path, max_seq_size: int = LEN_MAX,
                      stage_label: str = 'full', use_ivf: bool = False,
                      ivf_nlist: int | None = None, ivf_nprobe: int = 16,
                      faiss_cluster_size_override: int | None = None):
    """Run the clusTCR batch pipeline on seqs_arr (pyarrow LargeChunkedArray of strings).

    Uses a fast monkey-patched batch_precluster (vectorized write) for performance.
    Writes per-iteration parquet shards to shards_dir.

    If faiss_cluster_size_override is given, we lie about fitting_data_size to bypass
    clusTCR's auto-adjust, and force the resulting faiss_cluster_size to that value.
    This lets us pick a target ncentroids = training_sample_size / override.

    Returns dict with stats.
    """
    from clustcr import Clustering
    import faiss
    # Force FAISS to use all CPU cores for both K-means training (which does
    # internal index.search per iteration) and post-training search.
    faiss.omp_set_num_threads(64)

    rng = np.random.default_rng(SEED)
    n_total = seqs_arr.length()
    log(f'cluster_via_batch: n_total={n_total:,}, training_sample={training_sample_size:,}, '
        f'chunk_size={chunk_size:,}, shards_dir={shards_dir}, '
        f'faiss_cluster_size_override={faiss_cluster_size_override}, use_ivf={use_ivf}; '
        f'faiss.omp_get_max_threads()={faiss.omp_get_max_threads()}')

    if shards_dir.exists():
        shutil.rmtree(shards_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)

    # ---- Sample training data ----
    sample_idx = rng.choice(n_total, size=min(training_sample_size, n_total), replace=False)
    sample_idx.sort()
    log(f'  taking training sample of {len(sample_idx):,} (sorted indices)')
    t0 = time.time()
    # Convert ChunkedArray to a single combined LargeStringArray for take
    combined = seqs_arr.combine_chunks() if hasattr(seqs_arr, 'combine_chunks') else seqs_arr
    # combined is a single Array
    train_arr = combined.take(pa.array(sample_idx, type=pa.int64()))
    train_list = train_arr.to_pylist()
    train_series = pd.Series(train_list, dtype=object, name='junction_aa')
    log(f'  training sample materialized in {time.time()-t0:.1f}s')

    # ---- Build Clustering with FAISS training ----
    # If override is set, pass fitting_data_size=training_size so the auto-adjust is a no-op,
    # then set faiss_cluster_size=override and re-train.
    t_train = time.time()
    if faiss_cluster_size_override is not None:
        clustering = Clustering(
            chain='B',
            method='two-step',
            n_cpus='all',
            second_pass='MCL',
            faiss_cluster_size=faiss_cluster_size_override,
            faiss_training_data=train_series,
            fitting_data_size=len(train_series),  # lie: ratio=1, no auto-adjust
            max_sequence_size=max_seq_size,
        )
    else:
        clustering = Clustering(
            chain='B',
            method='two-step',
            n_cpus='all',
            second_pass='MCL',
            faiss_training_data=train_series,
            fitting_data_size=n_total,
            max_sequence_size=max_seq_size,
        )
    log(f'  FAISS training completed in {time.time()-t_train:.1f}s; '
        f'ncentroids={clustering.faiss_clustering.ncentroids()}, '
        f'faiss_cluster_size={clustering.faiss_cluster_size}')

    if use_ivf:
        _swap_kmeans_to_ivf(clustering, nlist=ivf_nlist, nprobe=ivf_nprobe)

    # Force FAISS to use all 64 threads for index search (default may be 1).
    import faiss
    faiss.omp_set_num_threads(64)
    log(f'  faiss.omp_set_num_threads(64); current={faiss.omp_get_max_threads()}')

    # ---- Stream chunks through fast_batch_precluster ----
    n_done = 0
    chunk_id = 0
    t_pc = time.time()
    write_progress(f'{stage_label}_precluster', 0, 0, 0.0,
                   extra={'n_total': n_total})
    for start in range(0, n_total, chunk_size):
        stop = min(start + chunk_size, n_total)
        idx_arr = pa.array(np.arange(start, stop, dtype=np.int64), type=pa.int64())
        chunk_arr = combined.take(idx_arr)
        chunk_list = chunk_arr.to_pylist()
        chunk_series = pd.Series(chunk_list, dtype=object, name='junction_aa')
        t_chunk = time.time()
        _fast_batch_precluster(clustering, chunk_series, name=f'b{chunk_id}')
        n_done += len(chunk_list)
        chunk_id += 1
        elapsed = time.time() - t_pc
        log(f'  precluster chunk {chunk_id}: {start:,}..{stop:,} '
            f'({n_done:,}/{n_total:,}) chunk_t={time.time()-t_chunk:.1f}s '
            f'cum_t={elapsed:.1f}s')
        write_progress(f'{stage_label}_precluster', chunk_id, n_done, elapsed,
                       extra={'n_total': n_total})
    log(f'  precluster done. n_chunks={chunk_id}, total elapsed={time.time()-t_pc:.1f}s')

    # ---- Stream batch_cluster -> parquet shards (using fast persistent-pool variant) ----
    t_bc = time.time()
    n_clusters_total = 0
    shard_idx = 0
    n_seqs_in_shards = 0
    write_progress(f'{stage_label}_cluster', 0, 0, 0.0)
    for result in _fast_batch_cluster(clustering):
        df = result.clusters_df
        # df has 'junction_aa', 'cluster' columns
        if df is None or len(df) == 0:
            continue
        # Rename 'cluster' -> 'cluster_id' and convert to int64
        out_df = pd.DataFrame({
            'trb_cdr3': df['junction_aa'].astype(str).values,
            'cluster_id': df['cluster'].astype(np.int64).values,
        })
        shard_path = shards_dir / f'shard_{shard_idx:05d}.parquet'
        pq.write_table(pa.Table.from_pandas(out_df, preserve_index=False),
                       shard_path, compression='zstd')
        n_clusters_local = int(out_df['cluster_id'].nunique())
        n_seqs_in_shards += len(out_df)
        n_clusters_total += n_clusters_local
        log(f'  shard {shard_idx}: {len(out_df):,} seqs, '
            f'{n_clusters_local:,} clusters; max_cid_so_far~={int(out_df["cluster_id"].max())}')
        shard_idx += 1
        write_progress(f'{stage_label}_cluster', shard_idx, n_seqs_in_shards,
                       time.time() - t_bc)
    log(f'  batch_cluster done. n_shards={shard_idx}, n_seqs_in_shards={n_seqs_in_shards:,}, '
        f'elapsed={time.time()-t_bc:.1f}s')

    # ---- Cleanup tmp files ----
    try:
        clustering.batch_cleanup()
    except Exception as e:
        log(f'  batch_cleanup raised {type(e).__name__}: {e}')

    return {
        'n_input': int(n_total),
        'n_seqs_in_shards': int(n_seqs_in_shards),
        'n_shards': int(shard_idx),
        'precluster_elapsed_s': float(time.time() - t_pc - (time.time() - t_bc)),
        # The above is rough; reuse t_pc/t_bc explicitly:
    }


def reassign_and_fill_missing(shards_dir: Path, all_input_seqs: pa.Array,
                              col_name: str = 'trb_cdr3') -> tuple[Path, int]:
    """Concatenate shards, repack cluster_ids contiguous, and append singleton
    clusters for any input sequence not present in the shard output.

    clusTCR's MCL_multiprocessing_from_preclusters silently drops preclusters
    that have no Hamming-distance-1 edges (clean_edgelist); those sequences are
    missing from the output. We add them back as one cluster_id per missing seq.

    Returns (concat_path, n_clusters_total).
    """
    log('reassign_and_fill_missing: scanning shards')
    shard_paths = sorted(shards_dir.glob('shard_*.parquet'))
    log(f'  found {len(shard_paths)} shards')
    # Pass 1: collect unique cluster_ids and track which input seqs appeared.
    all_cids = set()
    n_rows_total = 0
    seen_seqs: set = set()
    for sp in shard_paths:
        t = pq.read_table(sp)
        seqs = t.column(col_name).to_pylist()
        cids = t.column('cluster_id').to_numpy(zero_copy_only=False)
        seen_seqs.update(seqs)
        all_cids.update(np.unique(cids).tolist())
        n_rows_total += t.num_rows
    log(f'  shards: n_rows={n_rows_total:,}, n_unique_cids={len(all_cids):,}, '
        f'unique_seqs_seen={len(seen_seqs):,}')

    # Determine missing seqs (input - seen)
    n_input = len(all_input_seqs)
    log(f'  identifying missing input seqs (input={n_input:,})')
    input_seqs_pylist = all_input_seqs.to_pylist()
    # Use set for fast lookup
    missing_seqs = []
    for s in input_seqs_pylist:
        if s not in seen_seqs:
            missing_seqs.append(s)
    log(f'  missing seqs (will become singletons): {len(missing_seqs):,}')

    # Build mapping for clusTCR output cluster ids -> contiguous 0..K-1
    sorted_cids = sorted(all_cids)
    remap = {old: new for new, old in enumerate(sorted_cids)}
    n_clustcr_clusters = len(remap)
    next_cid = n_clustcr_clusters
    log(f'  clustcr clusters (after remap): 0..{n_clustcr_clusters-1}')

    # Pass 2: rewrite as single concatenated parquet
    concat_path = shards_dir / '_concat.parquet'
    if concat_path.exists():
        concat_path.unlink()
    schema = pa.schema([pa.field(col_name, pa.large_string()),
                        pa.field('cluster_id', pa.int64())])
    writer = pq.ParquetWriter(concat_path, schema=schema, compression='zstd')
    for sp in shard_paths:
        t = pq.read_table(sp)
        seqs = t.column(col_name).to_pylist()
        cids = t.column('cluster_id').to_numpy(zero_copy_only=False)
        new_cids = np.fromiter((remap[int(c)] for c in cids), dtype=np.int64, count=len(cids))
        out_tbl = pa.table({
            col_name: pa.array(seqs, type=pa.large_string()),
            'cluster_id': pa.array(new_cids, type=pa.int64()),
        })
        writer.write_table(out_tbl)

    # Append singletons for missing seqs
    if missing_seqs:
        sing_cids = np.arange(next_cid, next_cid + len(missing_seqs), dtype=np.int64)
        next_cid += len(missing_seqs)
        sing_tbl = pa.table({
            col_name: pa.array(missing_seqs, type=pa.large_string()),
            'cluster_id': pa.array(sing_cids, type=pa.int64()),
        })
        writer.write_table(sing_tbl)
    writer.close()
    log(f'  total clusters after singletons: {next_cid:,}')
    return concat_path, next_cid


# Back-compat alias
reassign_globally_unique = reassign_and_fill_missing


def smoke_main():
    log('=== TRB CDR3 clusTCR SMOKE TEST ===')
    log(f'hostname={socket.gethostname()}, platform={platform.platform()}')
    import clustcr
    log(f'clustcr file={clustcr.__file__}')
    try:
        import faiss
        faiss_ver = faiss.__version__
    except Exception:
        faiss_ver = 'unknown'

    SMOKE_SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    in_arr, _out_arr = load_trb_in_range(UNIQUE_DIR / 'unique_trb_cdr3.parquet')
    n_in = in_arr.length()
    rng = np.random.default_rng(SEED)
    n_smoke = min(10_000_000, n_in)
    idx = rng.choice(n_in, size=n_smoke, replace=False)
    idx.sort()
    log(f'smoke sample: {n_smoke:,} sequences (seed={SEED})')

    # Materialize the smoke sample as a separate Array to pass through cluster_via_batch
    combined = in_arr.combine_chunks()
    smoke_arr_concrete = combined.take(pa.array(idx, type=pa.int64()))
    smoke_chunked = pa.chunked_array([smoke_arr_concrete])
    log(f'smoke sample materialized as ChunkedArray length={smoke_chunked.length():,}')

    t_overall = time.time()
    try:
        stats = cluster_via_batch(
            seqs_arr=smoke_chunked,
            training_sample_size=SMOKE_TRAIN_SAMPLE,
            chunk_size=CHUNK_SIZE_SMOKE,
            shards_dir=SMOKE_SHARDS_DIR,
            max_seq_size=LEN_MAX,
            stage_label='smoke',
        )
    except Exception as e:
        tb = traceback.format_exc()
        log(f'EXCEPTION: {e}\n{tb}')
        out = {
            'status': 'FAIL',
            'reason': f'clustcr exception: {type(e).__name__}: {e}',
            'traceback': tb,
            'clustcr_version': getattr(clustcr, '__version__', 'unknown'),
            'faiss_version': faiss_ver,
            'hostname': socket.gethostname(),
            'platform': platform.platform(),
            'elapsed_s': time.time() - t_overall,
            'timestamp_utc': now_utc_iso(),
        }
        (ARTIFACT_DIR / 'clustcr_smoke_test.json').write_text(json.dumps(out, indent=2))
        return False

    elapsed = time.time() - t_overall
    log(f'smoke clustering done in {elapsed:.1f}s')

    # Concatenate + repack + fill in missing-from-clusTCR-output as singletons
    concat_path, n_cids = reassign_and_fill_missing(
        SMOKE_SHARDS_DIR, smoke_arr_concrete, col_name='trb_cdr3')
    tbl = pq.read_table(concat_path)
    seqs_out = tbl.column('trb_cdr3').to_pylist()
    cids_out = tbl.column('cluster_id').to_numpy(zero_copy_only=False)
    n_out = len(seqs_out)
    log(f'concat: n_rows={n_out:,}, n_clusters={n_cids:,}')

    sizes = np.bincount(cids_out)
    max_size = int(sizes.max())
    p50 = int(np.percentile(sizes, 50))
    p90 = int(np.percentile(sizes, 90))
    p99 = int(np.percentile(sizes, 99))
    n_singletons = int((sizes == 1).sum())

    # Sanity gates per spec:
    #  - n_clusters in (0.1*N, 0.95*N)
    #  - max cluster < 0.1*N
    #  - all input sequences accounted for in output
    pass_clusters = (0.1 * n_smoke) < n_cids < (0.95 * n_smoke)
    pass_max = max_size < 0.1 * n_smoke
    pass_account = n_out == n_smoke
    pass_time = elapsed < 3600.0

    overall_pass = pass_clusters and pass_max and pass_account and pass_time

    out = {
        'status': 'PASS' if overall_pass else 'FAIL',
        'reasons': [],
        'n_input': int(n_smoke),
        'n_output_rows': int(n_out),
        'n_clusters': int(n_cids),
        'max_cluster_size': max_size,
        'p50_size': p50,
        'p90_size': p90,
        'p99_size': p99,
        'n_singletons': n_singletons,
        'sanity_pass_clusters': bool(pass_clusters),
        'sanity_pass_max': bool(pass_max),
        'sanity_pass_account': bool(pass_account),
        'sanity_pass_time_under_3600s': bool(pass_time),
        'elapsed_s': elapsed,
        'clustcr_version': getattr(clustcr, '__version__', 'unknown'),
        'faiss_version': faiss_ver,
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'timestamp_utc': now_utc_iso(),
        'peak_rss_mb': peak_rss_mb(),
        'training_sample_size': SMOKE_TRAIN_SAMPLE,
        'chunk_size': CHUNK_SIZE_SMOKE,
    }
    if not pass_clusters: out['reasons'].append(f'n_clusters={n_cids} not in (0.1*N, 0.95*N)=({0.1*n_smoke}, {0.95*n_smoke})')
    if not pass_max: out['reasons'].append(f'max_cluster_size={max_size} >= 0.1*N={0.1*n_smoke}')
    if not pass_account: out['reasons'].append(f'n_output={n_out} != n_input={n_smoke}')
    if not pass_time: out['reasons'].append(f'elapsed={elapsed} >= 3600s')

    (ARTIFACT_DIR / 'clustcr_smoke_test.json').write_text(json.dumps(out, indent=2))
    log(f'wrote clustcr_smoke_test.json status={out["status"]}')
    return overall_pass


def full_main():
    log('=== TRB CDR3 clusTCR FULL ===')
    log(f'hostname={socket.gethostname()}, platform={platform.platform()}')
    import clustcr
    try:
        import faiss
        faiss_ver = faiss.__version__
    except Exception:
        faiss_ver = 'unknown'

    SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)

    in_arr, out_arr = load_trb_in_range(UNIQUE_DIR / 'unique_trb_cdr3.parquet')
    n_in = in_arr.length()
    n_out = out_arr.length()

    t_overall = time.time()
    try:
        # FULL TRB strategy:
        #   training=200K, faiss_cluster_size=10 (override -> ncentroids=20K).
        #   With 864M total: avg precluster size = 864M / 20K = 43K seqs (tractable for MCL).
        #   IVF index swap (nlist=512, nprobe=16) accelerates K-means SEARCH.
        # Empirical: chunk takes ~6 min in this config => ~17 hours full + cluster phase.
        # If cluster phase scales similarly to smoke (12s/iter × 313 iters parallel),
        # cluster phase is ~63 min.
        # Total: ~18 hours. Still meets spec (no hard time bound on full TRB).
        stats = cluster_via_batch(
            seqs_arr=in_arr,
            training_sample_size=200_000,
            chunk_size=CHUNK_SIZE_FULL,    # 5M
            shards_dir=SHARDS_DIR,
            max_seq_size=LEN_MAX,
            stage_label='full',
            use_ivf=True,
            ivf_nlist=512,
            ivf_nprobe=16,
            faiss_cluster_size_override=10,
        )
    except Exception as e:
        tb = traceback.format_exc()
        log(f'EXCEPTION: {e}\n{tb}')
        (ARTIFACT_DIR / 'trb_full_summary.json').write_text(json.dumps({
            'status': 'FAIL',
            'reason': f'clustcr exception: {type(e).__name__}: {e}',
            'traceback': tb,
            'elapsed_s': time.time() - t_overall,
            'timestamp_utc': now_utc_iso(),
        }, indent=2))
        return False

    log(f'full clustering done in {time.time()-t_overall:.1f}s')
    # Concatenate shards + fill in clusTCR-dropped seqs as singletons
    in_arr_combined = in_arr.combine_chunks() if hasattr(in_arr, 'combine_chunks') else in_arr
    concat_path, n_cids_in = reassign_and_fill_missing(
        SHARDS_DIR, in_arr_combined, col_name='trb_cdr3')
    log(f'concat parquet at {concat_path}, n_cids_after_fill={n_cids_in:,}')

    # Append out-of-range singletons starting at id n_cids_in
    log(f'appending {n_out:,} out-of-range singletons')
    out_seqs = out_arr.to_pylist()
    out_cids = np.arange(n_cids_in, n_cids_in + n_out, dtype=np.int64)
    n_total_clusters = n_cids_in + n_out

    # Final write: concat + singletons -> clusters/trb_cdr3_clusters.parquet
    final_path = CLUSTERS_DIR / 'trb_cdr3_clusters.parquet'
    schema = pa.schema([pa.field('trb_cdr3', pa.large_string()),
                        pa.field('cluster_id', pa.int64())])
    writer = pq.ParquetWriter(final_path, schema=schema, compression='zstd')
    # Stream the concat shard
    concat_tbl = pq.read_table(concat_path)
    writer.write_table(concat_tbl.cast(schema))
    # Write singletons
    if n_out > 0:
        sing_tbl = pa.table({
            'trb_cdr3': pa.array(out_seqs, type=pa.large_string()),
            'cluster_id': pa.array(out_cids, type=pa.int64()),
        })
        writer.write_table(sing_tbl)
    writer.close()

    # Verify final row count matches
    final_n = pq.read_metadata(final_path).num_rows
    expected_n = n_in + n_out
    log(f'final parquet: {final_n:,} rows (expected {expected_n:,}), '
        f'n_clusters={n_total_clusters:,}')

    summary = {
        'status': 'PASS' if final_n == expected_n else 'FAIL',
        'n_unique_input': int(expected_n),
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
    (ARTIFACT_DIR / 'trb_full_summary.json').write_text(json.dumps(summary, indent=2))
    log(f'wrote trb_full_summary.json: {summary["status"]}')
    return final_n == expected_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['smoke', 'full'])
    args = parser.parse_args()
    if args.mode == 'smoke':
        ok = smoke_main()
    else:
        ok = full_main()
    sys.exit(0 if ok else 2)


if __name__ == '__main__':
    main()
