"""HD<=1 wildcard-hash union-find clustering for TRB CDR3.

Strategy
--------
For each length bucket L in [LEN_MIN, LEN_MAX]:
  For each position pos in [0, L):
    Drop column pos from the (N, L) byte array -> (N, L-1)
    View each row as a fixed-width byte string (S(L-1))
    Sort indices by that masked string
    Adjacent indices with equal masked strings are an HD<=1 pair (differ
      only at position pos, or are identical) -- record as edges.
After collecting all edges for a length bucket, compute connected
components via scipy.sparse.csgraph.connected_components. Each component
is a cluster.

Sequences outside [LEN_MIN, LEN_MAX] are emitted as singletons (one
cluster_id per sequence).

Output: /home/ubuntu/quest/data/molecule_clusters/clusters/trb_cdr3_clusters.parquet
        cols: (trb_cdr3 large_string, cluster_id int64), contiguous 0..K-1
"""
from __future__ import annotations

import argparse
import json
import resource
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

ROOT = Path("/home/ubuntu/quest/data/molecule_clusters")
UNIQUE_DIR = ROOT / "unique_molecules"
CLUSTERS_DIR = ROOT / "clusters"
ARTIFACT_DIR = ROOT / "_build_artifacts"
# Note: PROGRESS_PATH and SUMMARY_PATH are axis-dependent and set in main()
# based on --axis, so multiple invocations don't clobber each other.

LEN_MIN = 8
LEN_MAX = 25


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def log(msg: str) -> None:
    print(f"[{now_iso()}] [trb_hd1] {msg}", flush=True)


def write_progress(progress_path: Path, stage: str, length: int | None,
                   frac_done: float, extra: dict | None = None) -> None:
    payload = {
        "stage": stage,
        "current_length_bucket": length,
        "frac_done": frac_done,
        "timestamp_utc": now_iso(),
        "peak_rss_mb": peak_rss_mb(),
    }
    if extra:
        payload.update(extra)
    progress_path.write_text(json.dumps(payload, indent=2))


def cluster_one_length_bucket(seqs_arr: pa.Array, length: int,
                              cluster_offset: int) -> tuple[np.ndarray, int]:
    """Cluster a same-length pa.Array (large_string) by HD<=1.

    Returns (cluster_ids, next_cluster_offset). cluster_ids has length
    seqs_arr.length() and values in [cluster_offset, next_cluster_offset).
    """
    n = len(seqs_arr)
    if n == 0:
        return np.zeros(0, dtype=np.int64), cluster_offset
    if n == 1:
        return np.array([cluster_offset], dtype=np.int64), cluster_offset + 1

    t_start = time.time()

    # Materialize sequences as a contiguous (n, length) byte array.
    # large_string -> Python str -> ASCII bytes
    seqs_list = seqs_arr.to_pylist()
    # Concatenate to a single bytes object then view as (n, length)
    flat = bytes("".join(seqs_list), "ascii")
    if len(flat) != n * length:
        # Defensive: some sequence wasn't the expected length
        raise ValueError(
            f"length-{length} bucket has total bytes {len(flat)} != {n*length}"
        )
    byte_arr = np.frombuffer(flat, dtype=np.uint8).reshape(n, length)
    log(f"  L={length}: materialized ({n:,}, {length}) byte array "
        f"in {time.time()-t_start:.1f}s, peak_rss_mb={peak_rss_mb():.0f}")

    # Collect edges across all positions.
    edge_a_list: list[np.ndarray] = []
    edge_b_list: list[np.ndarray] = []

    masked_dtype = np.dtype(f"S{length - 1}")

    for pos in range(length):
        t_pos = time.time()
        # Build (n, L-1) by dropping column pos.
        if pos == 0:
            masked = np.ascontiguousarray(byte_arr[:, 1:])
        elif pos == length - 1:
            masked = np.ascontiguousarray(byte_arr[:, :-1])
        else:
            masked = np.empty((n, length - 1), dtype=np.uint8)
            masked[:, :pos] = byte_arr[:, :pos]
            masked[:, pos:] = byte_arr[:, pos + 1:]
        # View each row as a single fixed-width byte string.
        masked_view = masked.view(masked_dtype).ravel()
        # Sort indices by masked string.
        order = np.argsort(masked_view, kind="stable")
        sorted_masked = masked_view[order]
        # Find consecutive equal pairs (HD<=1 neighbors at position pos).
        same = sorted_masked[1:] == sorted_masked[:-1]
        if same.any():
            same_idx = np.where(same)[0]
            edge_a_list.append(order[same_idx])
            edge_b_list.append(order[same_idx + 1])
        # Free intermediates eagerly
        del masked, masked_view, order, sorted_masked
        log(f"    pos {pos}: {time.time()-t_pos:.1f}s")

    # Build sparse adjacency and compute connected components.
    if edge_a_list:
        edge_a = np.concatenate(edge_a_list).astype(np.int64)
        edge_b = np.concatenate(edge_b_list).astype(np.int64)
        n_edges = len(edge_a)
        log(f"  L={length}: {n_edges:,} edges; computing CCs")
        data = np.ones(n_edges, dtype=np.int8)
        adj = sp.coo_matrix((data, (edge_a, edge_b)), shape=(n, n))
        # connected_components ignores direction when directed=False
        n_comp, labels = connected_components(adj, directed=False)
    else:
        labels = np.arange(n, dtype=np.int64)
        n_comp = n
        log(f"  L={length}: 0 edges (all singletons)")

    cluster_ids = cluster_offset + labels.astype(np.int64)
    next_offset = cluster_offset + int(n_comp)
    log(f"  L={length}: {n_comp:,} clusters; total {time.time()-t_start:.1f}s")
    return cluster_ids, next_offset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=UNIQUE_DIR / "unique_trb_cdr3.parquet")
    parser.add_argument("--output", type=Path,
                        default=CLUSTERS_DIR / "trb_cdr3_clusters.parquet")
    parser.add_argument("--axis", default="trb_cdr3",
                        help="Column name in the input parquet")
    args = parser.parse_args()

    overall_t0 = time.time()
    log(f"=== HD<=1 wildcard-hash UF clustering for {args.axis} ===")
    log(f"host={socket.gethostname()} platform={sys.platform}")
    log(f"input={args.input}")
    log(f"output={args.output}")

    CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    progress_path = ARTIFACT_DIR / f"hd1_{args.axis}_progress.json"
    summary_path = ARTIFACT_DIR / f"{args.axis}_hd1_summary.json"

    write_progress(progress_path, "loading", None, 0.0)

    t_load = time.time()
    table = pq.read_table(args.input, columns=[args.axis])
    # Cast to large_string to allow .take() across the full universe.
    table = table.cast(pa.schema([pa.field(args.axis, pa.large_string())]))
    arr = table.column(args.axis).combine_chunks()
    n_total = len(arr)
    log(f"loaded {n_total:,} rows in {time.time()-t_load:.1f}s "
        f"(peak_rss_mb={peak_rss_mb():.0f})")

    # Compute lengths once.
    t_len = time.time()
    lens = pc.utf8_length(arr).to_numpy(zero_copy_only=False)
    log(f"lengths computed in {time.time()-t_len:.1f}s")

    # Histogram
    hist = {}
    for L in range(LEN_MIN, LEN_MAX + 1):
        c = int((lens == L).sum())
        if c:
            hist[L] = c
    n_in_range = sum(hist.values())
    n_out_range = int(n_total - n_in_range)
    log(f"length histogram (in-range): {hist}")
    log(f"in-range total: {n_in_range:,}  out-of-range: {n_out_range:,}")

    output_seqs: list[pa.Array] = []
    output_cids: list[pa.Array] = []
    cluster_offset = 0

    for L in range(LEN_MIN, LEN_MAX + 1):
        n_L = hist.get(L, 0)
        if n_L == 0:
            continue
        write_progress(progress_path, "clustering", L,
                       cluster_offset / max(n_total, 1),
                       extra={"n_L": n_L, "cluster_offset": cluster_offset})
        log(f"--- length {L}: {n_L:,} seqs (cluster_offset={cluster_offset}) ---")
        mask = pa.array(lens == L)
        seqs_L = arr.filter(mask)
        cids, cluster_offset = cluster_one_length_bucket(seqs_L, L, cluster_offset)
        output_seqs.append(seqs_L)
        output_cids.append(pa.array(cids, type=pa.int64()))

    # Out-of-range singletons.
    if n_out_range > 0:
        log(f"--- out-of-range singletons: {n_out_range:,} ---")
        out_mask = pa.array((lens < LEN_MIN) | (lens > LEN_MAX))
        seqs_out = arr.filter(out_mask)
        cids_out = np.arange(cluster_offset, cluster_offset + n_out_range,
                             dtype=np.int64)
        output_seqs.append(seqs_out)
        output_cids.append(pa.array(cids_out, type=pa.int64()))
        cluster_offset += n_out_range

    # Concatenate and write.
    log(f"concatenating output ({cluster_offset:,} total clusters)")
    write_progress(progress_path, "writing", None, 1.0,
                   extra={"n_clusters_total": cluster_offset})
    # Use chunked array for memory friendliness.
    all_seqs = pa.chunked_array([s for s in output_seqs])
    all_cids = pa.chunked_array([c for c in output_cids])
    out_table = pa.table({args.axis: all_seqs, "cluster_id": all_cids})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, args.output, compression="zstd")
    log(f"wrote {args.output} ({len(out_table):,} rows)")

    elapsed = time.time() - overall_t0
    summary = {
        "status": "PASS",
        "method": "HD<=1 wildcard-hash union-find (scipy connected_components)",
        "axis": args.axis,
        "input_path": str(args.input),
        "output_path": str(args.output),
        "n_unique_input": int(n_total),
        "n_in_range": int(n_in_range),
        "n_out_of_range": int(n_out_range),
        "n_clusters_total": int(cluster_offset),
        "length_histogram": hist,
        "elapsed_s": float(elapsed),
        "peak_rss_mb": peak_rss_mb(),
        "hostname": socket.gethostname(),
        "timestamp_utc": now_iso(),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    log(f"wrote {summary_path}")
    log(f"DONE in {elapsed:.1f}s "
        f"(peak_rss_mb={peak_rss_mb():.0f}, n_clusters={cluster_offset:,})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
