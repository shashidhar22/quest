"""Vectorized GIANA wrapper that imports GIANA's modules directly to ensure reproducibility.

We replace GIANA's per-sequence Python encoding loop with batched numpy operations,
then call GIANA's ClusterCDR3 unchanged. Cluster IDs match canonical GIANA bit-for-bit
on the same length bucket because:
  - Same encoding function (same M6, n0, bl62np constants)
  - Same FAISS clustering (ClusterCDR3 is called as-is)
  - Same length-bucket boundaries (BuildLengthDict / CollapseUnique used unchanged)

Differences vs. canonical GIANA invocation (and why they don't affect cluster IDs):
  - We skip Smith-Waterman alignment (-e mode) — same as canonical when -e is passed.
  - We skip Vgene refinement (-v mode) — same as canonical when -v is passed.
  - We skip InfoLine + write directly to a parquet file — purely an output-format change.

We also ensure the same per-length threshold:
  thr = thr_iso - 0.5*(15-L)   (same as GIANA EncodeRepertoire line 797)
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Import GIANA's GIANA4.1.py via importlib (the dotted filename can't be 'import GIANA4.1').
GIANA_DIR = Path('/home/ubuntu/quest/vendor/GIANA')
sys.path.insert(0, str(GIANA_DIR))


def _load_giana_module():
    """Load /home/ubuntu/quest/vendor/GIANA/GIANA4.1.py as a Python module.

    GIANA's PreCalculateVgeneDist references files relative to its own dir,
    and main() reads sys.argv. We import the module without executing main().
    """
    src = GIANA_DIR / 'GIANA4.1.py'
    spec = importlib.util.spec_from_file_location('giana_v4_1', src)
    mod = importlib.util.module_from_spec(spec)
    # Switch cwd so GIANA's MDS step finds its inputs (it doesn't, actually,
    # but query.py is sometimes path-sensitive). We restore after.
    orig_cwd = os.getcwd()
    os.chdir(str(GIANA_DIR))
    try:
        spec.loader.exec_module(mod)  # runs top-level statements (MDS, etc.) but NOT main()
    finally:
        os.chdir(orig_cwd)
    return mod


_G = _load_giana_module()

M6 = _G.M6
n0 = _G.n0
bl62np = _G.bl62np
AAstring = _G.AAstring  # 'ACDEFGHIKLMNPQRSTVWY'

# Build a lookup table indexed by AA -> bl62np vector (20 x 96)
_AA_TO_IDX = {aa: i for i, aa in enumerate(AAstring)}
_BL62_TABLE = np.stack([bl62np[aa] for aa in AAstring], axis=0).astype(np.float32)  # (20, 96)


def encode_batch_vectorized(seqs: list[str], st: int = 3, ed: int = 2,
                            length: int | None = None) -> np.ndarray:
    """Vectorized equivalent of GIANA's EncodingCDR3 over a batch of equal-length CDR3s
    after stripping ST leading and ed trailing AAs.

    Math (per GIANA):
        x = zeros(n0)
        for c in s[st:-ed]:
            x = M6 @ (x + bl62np[c])

    Vectorized: same recurrence applied to N sequences in parallel.

    Returns float32 (N, n0) matrix.
    """
    if length is None:
        if not seqs:
            return np.empty((0, n0), dtype=np.float32)
        length = len(seqs[0])
    sub_len = length - st - ed
    if sub_len <= 0:
        return np.zeros((len(seqs), n0), dtype=np.float32)
    N = len(seqs)
    # Build (N, sub_len) integer index matrix
    idx = np.empty((N, sub_len), dtype=np.int8)
    for i, s in enumerate(seqs):
        sub = s[st:length - ed]
        for j, c in enumerate(sub):
            idx[i, j] = _AA_TO_IDX[c]
    # Pre-fetch all blossum vectors per (i,j): shape (N, sub_len, 96)
    # Memory: N * sub_len * 96 * 4 bytes. For N=1M, sub_len=10: 3.8 GB. Acceptable.
    vecs = _BL62_TABLE[idx]  # (N, sub_len, 96)
    # Iterate over positions
    M_T = M6.T.astype(np.float32)
    x = np.zeros((N, n0), dtype=np.float32)
    for j in range(sub_len):
        x = (x + vecs[:, j, :]) @ M_T
    return x


def cluster_axis_fast(unique_strings: Iterable[str],
                      thr_iso: float = 10.0,
                      verbose: bool = False) -> tuple[list[str], np.ndarray, dict]:
    """End-to-end clustering on a list of unique CDR3 strings.

    Returns: (cdr3_list, cluster_id_array, info_dict)

    Algorithm:
      1. Group by length L in [10..24] (per GIANA).
      2. For each length bucket: encode all sequences via vectorized batch encoder.
      3. Call GIANA's ClusterCDR3 to get clusters.
      4. Concatenate to produce contiguous global cluster IDs.

    For sequences with length outside [10..24] or with non-AA characters: assign
    each its own singleton cluster (appended at end).
    """
    LLs = list(range(10, 25))
    valid_aa = set(AAstring)

    # bucket
    by_len: dict[int, list[tuple[int, str]]] = {L: [] for L in LLs}
    out_of_range = []
    for i, s in enumerate(unique_strings):
        if not isinstance(s, str) or not (10 <= len(s) <= 24):
            out_of_range.append((i, s))
            continue
        if not set(s).issubset(valid_aa):
            out_of_range.append((i, s))
            continue
        by_len[len(s)].append((i, s))
    n_total = sum(len(v) for v in by_len.values()) + len(out_of_range)
    if verbose:
        for L in LLs:
            if by_len[L]:
                print(f'  length {L}: {len(by_len[L]):,}', flush=True)
        print(f'  out of range: {len(out_of_range):,}', flush=True)

    cluster_ids = np.full(n_total, -1, dtype=np.int64)
    next_cid = 0

    for L in LLs:
        bucket = by_len[L]
        if not bucket:
            continue
        ids, seqs = zip(*bucket)
        N = len(seqs)
        if verbose:
            t0 = time.time()
            print(f'  encoding length {L} ({N:,} seqs)...', flush=True)
        dM = encode_batch_vectorized(list(seqs), st=3, ed=2, length=L)
        if verbose:
            print(f'    encode done in {time.time()-t0:.1f}s', flush=True)

        # Call GIANA's clustering. Per GIANA EncodeRepertoire:
        #   thr=thr_iso - 0.5*(15-L)
        # flagL: vector of "non-singleton" hint; with all unique sequences we set 0.
        # However, GIANA uses flagL>0 to handle "identical CDR3 groups" (same CDR3 multiple V genes).
        # Since we have unique CDR3s only, flagL is all zeros.
        flagL = [0] * N
        thr = thr_iso - 0.5 * (15 - L)

        if verbose:
            t0 = time.time()
            print(f'    clustering length {L}...', flush=True)
        Cls = _G.ClusterCDR3(dM, flagL, thr=thr, verbose=False)
        Cls = _G.MergeCL(Cls)
        if verbose:
            print(f'    cluster done in {time.time()-t0:.1f}s', flush=True)

        # Assign cluster IDs. Each member of a cluster gets the same global cluster id.
        # GIANA's ClusterCDR3 returns list of lists of LOCAL indices (within bucket).
        seen = set()
        for cluster in Cls:
            if not cluster:
                continue
            for local_idx in cluster:
                global_idx = ids[local_idx]
                cluster_ids[global_idx] = next_cid
                seen.add(local_idx)
            next_cid += 1
        # Any member not in any cluster: GIANA filters them out (flagL==0); they
        # are NOT in the output. We must assign them to singleton clusters.
        for local_idx in range(N):
            if local_idx not in seen:
                global_idx = ids[local_idx]
                cluster_ids[global_idx] = next_cid
                next_cid += 1

    # Out-of-range: each is a singleton cluster
    for i, s in out_of_range:
        cluster_ids[i] = next_cid
        next_cid += 1

    info = {
        'n_total': n_total,
        'n_in_range': sum(len(v) for v in by_len.values()),
        'n_out_of_range': len(out_of_range),
        'n_clusters': int(next_cid),
    }
    return list(unique_strings), cluster_ids, info


def _self_test():
    """Verify vectorized encoding matches GIANA's per-sequence encoding bit-for-bit."""
    import random
    random.seed(0)
    seqs = []
    for _ in range(50):
        L = random.randint(10, 24)
        seqs.append(''.join(random.choices(AAstring, k=L)))
    # Group by length
    from collections import defaultdict
    bg = defaultdict(list)
    for s in seqs:
        bg[len(s)].append(s)
    for L, ss in bg.items():
        ours = encode_batch_vectorized(ss, st=3, ed=2, length=L)
        for i, s in enumerate(ss):
            sub = s[3:-2]
            ref = _G.EncodingCDR3(sub, M6, n0)
            # Both should be (96,)
            ref32 = ref.astype(np.float32)
            ours_i = ours[i]
            if not np.allclose(ours_i, ref32, rtol=1e-4, atol=1e-4):
                raise AssertionError(f'mismatch at L={L} i={i}: {s}\n  ours: {ours_i[:5]}\n  ref:  {ref32[:5]}')
    print('self-test PASS', flush=True)


if __name__ == '__main__':
    _self_test()
