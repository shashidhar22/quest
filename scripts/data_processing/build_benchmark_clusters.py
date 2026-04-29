"""Build clustering artifacts for benchmark_v2 split generation.

Scans the Hive-partitioned enriched corpus at
``data/deduplicated_again/exploded_deduped_enriched/``, extracts unique TCR
CDR3s / peptides / MHC protein sequences at several scopes (interaction /
restriction / all / pm), then clusters them with MMseqs2 (CDR3s, peptides) and
rapidfuzz Levenshtein + single-linkage (peptides).

Outputs land in ``data/benchmark_v2/clusters/``. MMseqs2 scratch goes on
``/scratch`` (ephemeral NVMe).

Usage:
    python build_benchmark_clusters.py --task {collect,tcr_cluster,peptide_cluster,allele_inventory,summary,all}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SOURCE_ROOT = Path("/home/ubuntu/quest/data/deduplicated_again/exploded_deduped_enriched")
OUT_ROOT = Path("/home/ubuntu/quest/data/benchmark_v2/clusters")
SCRATCH = Path("/scratch")
SCRATCH_BIG = Path("/scratch2")
LOG_DIR = OUT_ROOT / "logs"
FASTA_DIR = SCRATCH / "fasta"
MMSEQS_TMP = SCRATCH / "mmseqs_tmp"
MMSEQS_TMP_BIG = SCRATCH_BIG / "mmseqs_tmp_trb_all"
TIMINGS_PATH = OUT_ROOT / "timings.json"
COUNTS_PATH = OUT_ROOT / "sequence_counts.json"
SUMMARY_PATH = OUT_ROOT / "clustering_summary.json"
TRB_ALL_PID_FILE = LOG_DIR / "trb_all.pid"
TRB_ALL_LOG = LOG_DIR / "trb_all.log"

THREADS = 64


# ---------------------------------------------------------------------------
# Subset-key classification
# ---------------------------------------------------------------------------

def classify_subset_key(key: str) -> Dict[str, bool]:
    """Return membership flags for the given subset_key string."""
    has_tra = "tra" in key
    has_trb = "trb" in key
    has_peptide = "peptide" in key
    has_mhc_one = "mhc_one" in key
    has_mhc_two = "mhc_two" in key
    has_tcr = has_tra or has_trb
    has_mhc = has_mhc_one or has_mhc_two
    return {
        "interaction": has_tcr and has_peptide,
        "restriction": has_tcr and has_mhc,
        "pm": has_peptide and has_mhc,
        "has_tra": has_tra,
        "has_trb": has_trb,
        "has_peptide": has_peptide,
        "has_mhc_one": has_mhc_one,
        "has_mhc_two": has_mhc_two,
    }


# ---------------------------------------------------------------------------
# Task 1: collect unique sequences
# ---------------------------------------------------------------------------

def _scan_file_arrow(file_path: Path, columns: List[str]) -> Dict[str, pa.Array]:
    """Read one parquet file and return per-column non-null, unique Arrow string array.

    All dedup happens in C++ (pyarrow.compute), avoiding expensive Python iteration.
    """
    table = pq.read_table(str(file_path), columns=columns, use_threads=False)
    result: Dict[str, pa.Array] = {}
    for col in columns:
        a = table.column(col).combine_chunks()
        a = pc.unique(pc.drop_null(a))
        # Drop empty strings as well
        if len(a) > 0:
            mask = pc.greater(pc.utf8_length(a), 0)
            a = a.filter(mask)
        result[col] = a
    return result


SCOPE_NAMES: List[str] = [
    "trb_cdr3_interaction",
    "trb_cdr3_restriction",
    "trb_cdr3_all",
    "tra_cdr3_interaction",
    "tra_cdr3_restriction",
    "tra_cdr3_all",
    "peptides_interaction",
    "peptides_pm",
    "mhc_one_all",
    "mhc_two_all",
]

SCOPE_TO_OUTFILE: Dict[str, str] = {
    "trb_cdr3_interaction": "unique_trb_cdr3_interaction.txt",
    "trb_cdr3_restriction": "unique_trb_cdr3_restriction.txt",
    "trb_cdr3_all": "unique_trb_cdr3_all.txt",
    "tra_cdr3_interaction": "unique_tra_cdr3_interaction.txt",
    "tra_cdr3_restriction": "unique_tra_cdr3_restriction.txt",
    "tra_cdr3_all": "unique_tra_cdr3_all.txt",
    "peptides_interaction": "unique_peptides_interaction.txt",
    "peptides_pm": "unique_peptides_pm.txt",
    "mhc_one_all": "unique_mhc_one_alleles.txt",
    "mhc_two_all": "unique_mhc_two_alleles.txt",
}


def _scope_predicate(scope: str, cls: Dict[str, bool]) -> bool:
    """Return True if a partition with the given classification belongs to this scope."""
    if scope == "trb_cdr3_all":
        return cls["has_trb"]
    if scope == "trb_cdr3_interaction":
        return cls["has_trb"] and cls["interaction"]
    if scope == "trb_cdr3_restriction":
        return cls["has_trb"] and cls["restriction"]
    if scope == "tra_cdr3_all":
        return cls["has_tra"]
    if scope == "tra_cdr3_interaction":
        return cls["has_tra"] and cls["interaction"]
    if scope == "tra_cdr3_restriction":
        return cls["has_tra"] and cls["restriction"]
    if scope == "peptides_interaction":
        return cls["has_peptide"] and cls["interaction"]
    if scope == "peptides_pm":
        return cls["has_peptide"] and cls["pm"]
    if scope == "mhc_one_all":
        return cls["has_mhc_one"]
    if scope == "mhc_two_all":
        return cls["has_mhc_two"]
    return False


SCOPE_TO_COLUMN: Dict[str, str] = {
    "trb_cdr3_all": "trb_cdr3",
    "trb_cdr3_interaction": "trb_cdr3",
    "trb_cdr3_restriction": "trb_cdr3",
    "tra_cdr3_all": "tra_cdr3",
    "tra_cdr3_interaction": "tra_cdr3",
    "tra_cdr3_restriction": "tra_cdr3",
    "peptides_interaction": "peptide",
    "peptides_pm": "peptide",
    "mhc_one_all": "mhc_one",
    "mhc_two_all": "mhc_two",
}


def collect_unique_sequences() -> Dict[str, int]:
    """Per scope, collect distinct non-null non-empty values from the relevant
    partitions and write them as newline-separated text files.

    Uses DuckDB because it (a) scans parquet in parallel in C++, (b) computes
    DISTINCT efficiently using hash tables with spill-to-disk, and (c) emits
    text output without Python-side list materialization. Previous attempts
    (pyarrow pc.unique, Python set streams, GNU sort -u) all stalled on the
    645K TRB-all CDR3 scope at this data size.
    """
    import duckdb

    t0 = time.time()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    partition_dirs = sorted(
        p for p in SOURCE_ROOT.iterdir() if p.is_dir() and p.name.startswith("subset_key=")
    )
    print(f"[collect] {len(partition_dirs)} subset_key partitions", flush=True)

    con = duckdb.connect()
    con.execute("SET threads=64")
    con.execute("SET memory_limit='400GB'")
    con.execute(f"SET temp_directory='{SCRATCH}'")

    counts: Dict[str, int] = {}
    for scope in SCOPE_NAMES:
        col = SCOPE_TO_COLUMN[scope]
        files: List[str] = []
        for pdir in partition_dirs:
            key = pdir.name.split("=", 1)[1]
            cls = classify_subset_key(key)
            if not _scope_predicate(scope, cls):
                continue
            files.extend(str(f) for f in pdir.rglob("*.parquet"))
        out_path = OUT_ROOT / SCOPE_TO_OUTFILE[scope]
        if not files:
            out_path.write_text("")
            counts[SCOPE_TO_OUTFILE[scope].replace("unique_", "").replace(".txt", "")] = 0
            print(f"[collect]   {scope}: no files, wrote empty", flush=True)
            continue

        t_scope = time.time()
        con.execute(
            f"""
            COPY (
                SELECT DISTINCT {col} AS v
                FROM read_parquet(?)
                WHERE {col} IS NOT NULL AND {col} != ''
                ORDER BY v
            ) TO '{out_path}' (HEADER false, FORMAT CSV, QUOTE '', DELIMITER ',');
            """,
            [files],
        )
        # Count lines
        with open(out_path, "rb") as f:
            n = sum(1 for _ in f)
        counts[SCOPE_TO_OUTFILE[scope].replace("unique_", "").replace(".txt", "")] = n
        size_gb = out_path.stat().st_size / 1e9
        print(
            f"[collect]   {scope}: {len(files)} files -> {n:,} unique ({size_gb:.2f}GB) in {time.time()-t_scope:.1f}s",
            flush=True,
        )

    con.close()

    with open(COUNTS_PATH, "w") as f:
        json.dump(counts, f, indent=2, sort_keys=True)
    _record_timing("collect", time.time() - t0)
    print(f"[collect] done in {time.time()-t0:.1f}s", flush=True)
    return counts


# ---------------------------------------------------------------------------
# Task 2: MMseqs2 easy-cluster
# ---------------------------------------------------------------------------

def _seqs_to_fasta(sequences: List[str], fasta_path: Path) -> None:
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fasta_path, "w") as f:
        for i, seq in enumerate(sequences):
            # Use index as header; MMseqs truncates at whitespace
            f.write(f">{i}\n{seq}\n")


def _txt_to_fasta_awk(txt_path: Path, fasta_path: Path) -> None:
    """Convert a newline-delimited text file to FASTA using awk.

    Header = sequence itself (lets us skip the seqidx lookup). Much faster than
    the Python equivalent for 100M+ sequences.
    """
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fasta_path, "w") as out:
        subprocess.run(
            ["awk", '{print ">"$0"\\n"$0}', str(txt_path)],
            stdout=out,
            check=True,
        )


def _read_seqs(path: Path) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _run_easy_cluster(
    fasta_path: Path,
    out_prefix: Path,
    tmp_dir: Path,
    min_seq_id: float = 0.9,
    coverage: float = 0.8,
    threads: int = THREADS,
    capture_log: Optional[Path] = None,
) -> None:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mmseqs", "easy-cluster",
        str(fasta_path), str(out_prefix), str(tmp_dir),
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage),
        "--cluster-mode", "1",
        "--threads", str(threads),
    ]
    print(f"[mmseqs] {' '.join(cmd)}", flush=True)
    if capture_log:
        with open(capture_log, "w") as lf:
            subprocess.run(cmd, check=True, stdout=lf, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, check=True)


def _cluster_one(
    name: str,
    source_txt: Path,
    out_tsv: Path,
    min_seq_id: float = 0.9,
    coverage: float = 0.8,
    tmp_dir: Optional[Path] = None,
) -> int:
    """Cluster sequences from a newline-text file via mmseqs easy-cluster.

    Uses awk to convert txt -> FASTA (header == sequence), so the resulting
    {prefix}_cluster.tsv already has ``rep_seq\\tmember_seq`` rows and no
    index-to-sequence lookup is needed.
    """
    tmp_dir = tmp_dir or (MMSEQS_TMP / name)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    fasta = FASTA_DIR / f"{name}.fasta"
    prefix = SCRATCH / "mmseqs_out" / name
    prefix.parent.mkdir(parents=True, exist_ok=True)
    src_size = source_txt.stat().st_size
    if src_size == 0:
        out_tsv.write_text("sequence\tcluster_id\n")
        return 0
    print(f"[cluster] {name}: source {src_size/1e9:.3f}GB -> {out_tsv.name}", flush=True)
    t0 = time.time()
    _txt_to_fasta_awk(source_txt, fasta)
    print(f"[cluster]   awk fasta in {time.time()-t0:.1f}s", flush=True)
    _run_easy_cluster(fasta, prefix, tmp_dir, min_seq_id=min_seq_id, coverage=coverage)
    raw = Path(str(prefix) + "_cluster.tsv")
    # Copy raw tsv with a header renamed to our convention
    n_clusters = _copy_fasta_header_tsv(raw, out_tsv)
    print(f"[cluster]   {name}: {n_clusters:,} clusters", flush=True)
    return n_clusters


def _copy_fasta_header_tsv(raw_tsv: Path, out_tsv: Path) -> int:
    """Copy mmseqs ``{prefix}_cluster.tsv`` (rep_seq \\t member_seq) to our
    ``sequence \\t cluster_id`` format. Counts unique reps while copying.

    Streaming I/O; safe for 1B-row files.
    """
    n_clusters = 0
    seen: Set[str] = set()
    with open(raw_tsv, "rb") as rf, open(out_tsv, "wb") as wf:
        wf.write(b"sequence\tcluster_id\n")
        for line in rf:
            parts = line.rstrip(b"\n").split(b"\t")
            if len(parts) < 2:
                continue
            rep, member = parts[0], parts[1]
            wf.write(member + b"\t" + rep + b"\n")
            if rep not in seen:
                seen.add(rep)
                n_clusters += 1
    return n_clusters


def run_tcr_clustering() -> Dict[str, int]:
    t0 = time.time()
    cluster_counts: Dict[str, int] = {}

    # 1. Launch TRB foundation clustering in background (nohup subprocess running this same script)
    trb_all_src = OUT_ROOT / "unique_trb_cdr3_all.txt"
    if trb_all_src.exists() and trb_all_src.stat().st_size > 0:
        if TRB_ALL_PID_FILE.exists():
            print(f"[cluster] TRB foundation PID file exists at {TRB_ALL_PID_FILE}; skipping re-launch", flush=True)
        else:
            pid = _launch_trb_foundation_background()
            print(f"[cluster] launched TRB foundation clustering in background PID={pid}", flush=True)
    else:
        print("[cluster] skip TRB foundation launch — source txt empty/missing", flush=True)

    # 2. Foreground jobs 1-4 (interaction/restriction)
    jobs = [
        ("trb_cdr3_interaction", "unique_trb_cdr3_interaction.txt", "trb_cdr3_interaction_clusters.tsv"),
        ("tra_cdr3_interaction", "unique_tra_cdr3_interaction.txt", "tra_cdr3_interaction_clusters.tsv"),
        ("trb_cdr3_restriction", "unique_trb_cdr3_restriction.txt", "trb_cdr3_restriction_clusters.tsv"),
        ("tra_cdr3_restriction", "unique_tra_cdr3_restriction.txt", "tra_cdr3_restriction_clusters.tsv"),
    ]
    for name, src, out in jobs:
        src_path = OUT_ROOT / src
        out_path = OUT_ROOT / out
        cluster_counts[name] = _cluster_one(name, src_path, out_path, min_seq_id=0.9, coverage=0.8)

    # 3. TRA foundation (smaller than TRB, do in foreground)
    cluster_counts["tra_cdr3_all"] = _cluster_one(
        "tra_cdr3_all",
        OUT_ROOT / "unique_tra_cdr3_all.txt",
        OUT_ROOT / "tra_cdr3_all_clusters.tsv",
        min_seq_id=0.9,
        coverage=0.8,
    )

    _record_timing("tcr_cluster", time.time() - t0)
    with open(OUT_ROOT / "_cluster_counts_partial.json", "w") as f:
        json.dump(cluster_counts, f, indent=2, sort_keys=True)
    return cluster_counts


def _launch_trb_foundation_background() -> int:
    """Fork-and-detach a mmseqs easy-cluster run for trb_cdr3_all.

    Writes the FASTA via awk (fast), then nohups mmseqs easy-cluster, then
    post-processes the result TSV into our final output format — all wrapped
    in a single shell script so the whole pipeline runs detached.
    """
    name = "trb_cdr3_all"
    src_txt = OUT_ROOT / "unique_trb_cdr3_all.txt"
    fasta = SCRATCH_BIG / "fasta" / f"{name}.fasta"
    fasta.parent.mkdir(parents=True, exist_ok=True)
    prefix = SCRATCH_BIG / "mmseqs_out" / name
    prefix.parent.mkdir(parents=True, exist_ok=True)
    MMSEQS_TMP_BIG.mkdir(parents=True, exist_ok=True)

    runner_sh = LOG_DIR / "trb_all_runner.sh"
    final_tsv = OUT_ROOT / "trb_cdr3_all_clusters.tsv"
    raw_tsv = Path(str(prefix) + "_cluster.tsv")
    done_flag = prefix.with_suffix(".done")
    runner_sh.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        f"echo '[trb_all] writing fasta...' >&2\n"
        f"awk '{{print \">\"$0\"\\n\"$0}}' {src_txt} > {fasta}\n"
        f"echo '[trb_all] running mmseqs...' >&2\n"
        f"mmseqs easy-cluster {fasta} {prefix} {MMSEQS_TMP_BIG} "
        f"--min-seq-id 0.9 -c 0.8 --cluster-mode 1 --threads {THREADS}\n"
        f"echo '[trb_all] post-processing tsv...' >&2\n"
        f"(printf 'sequence\\tcluster_id\\n'; awk -v OFS='\\t' '{{print $2,$1}}' {raw_tsv}) > {final_tsv}\n"
        f"touch {done_flag}\n"
        f"echo '[trb_all] done' >&2\n"
    )
    runner_sh.chmod(0o755)

    log = TRB_ALL_LOG
    with open(log, "wb") as lf:
        proc = subprocess.Popen(
            ["nohup", "bash", str(runner_sh)],
            stdout=lf,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    TRB_ALL_PID_FILE.write_text(f"{proc.pid}\n{int(time.time())}\n{prefix}\n")
    return proc.pid


def finalize_trb_foundation_if_done() -> Tuple[str, Optional[int]]:
    """Report whether the background TRB foundation clustering is done.

    The runner script writes the final TSV itself, so ``summary`` does not
    need to convert here — only check the done-flag and count clusters.
    """
    if not TRB_ALL_PID_FILE.exists():
        return ("not_launched", None)
    lines = TRB_ALL_PID_FILE.read_text().strip().splitlines()
    if len(lines) < 3:
        return ("bad_pid_file", None)
    pid = int(lines[0])
    prefix = Path(lines[2])
    done_flag = prefix.with_suffix(".done")
    final_tsv = OUT_ROOT / "trb_cdr3_all_clusters.tsv"
    alive = Path(f"/proc/{pid}").exists()
    if not done_flag.exists() and alive:
        return ("running", None)
    if not done_flag.exists():
        return ("failed_no_output", None)
    # Count clusters by streaming TSV and counting distinct cluster_id (column 2)
    seen: Set[str] = set()
    with open(final_tsv, "rb") as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip(b"\n").split(b"\t")
            if len(parts) < 2:
                continue
            seen.add(parts[1])
    return ("complete", len(seen))


# ---------------------------------------------------------------------------
# Task 3: peptide edit-distance clustering + MMseqs2 appendix
# ---------------------------------------------------------------------------

_WORKER_PEPTIDES: List[str] = []
_WORKER_LENS: List[int] = []
_WORKER_FRAC = 0.2


def _worker_init(peptides: List[str], frac: float) -> None:
    global _WORKER_PEPTIDES, _WORKER_LENS, _WORKER_FRAC
    _WORKER_PEPTIDES = peptides
    _WORKER_LENS = [len(p) for p in peptides]
    _WORKER_FRAC = frac


def _worker_edges(i_start_end: Tuple[int, int]) -> List[Tuple[int, int]]:
    from rapidfuzz.distance import Levenshtein
    i_start, i_end = i_start_end
    peps = _WORKER_PEPTIDES
    lens = _WORKER_LENS
    frac = _WORKER_FRAC
    n = len(peps)
    edges: List[Tuple[int, int]] = []
    for i in range(i_start, i_end):
        li = lens[i]
        pi = peps[i]
        # Max edit we might allow with any j: ceil(frac * max_len_ever)
        for j in range(i + 1, n):
            lj = lens[j]
            min_len = li if li < lj else lj
            thr = math.ceil(frac * min_len)
            if abs(li - lj) > thr:
                continue
            d = Levenshtein.distance(pi, peps[j], score_cutoff=thr)
            if d <= thr:
                edges.append((i, j))
    return edges


def _dsu_cluster(n: int, edges: List[Tuple[int, int]]) -> List[int]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return [find(i) for i in range(n)]


def _cluster_peptides_edit_distance(
    peptides: List[str], frac: float = 0.2, threads: int = THREADS, chunk: int = 512
) -> Dict[str, str]:
    """Return dict peptide -> cluster_representative (shortest peptide in component)."""
    # Sort by length (stable) so short peps tend to be representatives
    idx_order = sorted(range(len(peptides)), key=lambda i: (len(peptides[i]), peptides[i]))
    peps_sorted = [peptides[i] for i in idx_order]
    n = len(peps_sorted)
    if n <= 1:
        return {p: p for p in peptides}

    chunks: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        chunks.append((i, min(i + chunk, n)))
        i += chunk

    print(f"[peptide-ed] {n:,} peptides, {len(chunks)} outer-chunks of {chunk}, {threads} workers", flush=True)
    all_edges: List[Tuple[int, int]] = []
    t0 = time.time()
    with Pool(processes=threads, initializer=_worker_init, initargs=(peps_sorted, frac)) as pool:
        for i_chunk, edges in enumerate(pool.imap_unordered(_worker_edges, chunks, chunksize=1)):
            all_edges.extend(edges)
            if (i_chunk + 1) % max(1, len(chunks) // 20) == 0:
                pct = 100 * (i_chunk + 1) / len(chunks)
                print(
                    f"[peptide-ed]   {pct:.0f}% ({i_chunk+1}/{len(chunks)} chunks, {len(all_edges):,} edges so far, {time.time()-t0:.0f}s)",
                    flush=True,
                )
    print(f"[peptide-ed] {len(all_edges):,} edges; running DSU", flush=True)

    roots = _dsu_cluster(n, all_edges)
    # Pick a canonical representative per root: shortest peptide, then lexicographic
    comp_to_rep: Dict[int, str] = {}
    for i, r in enumerate(roots):
        cur = comp_to_rep.get(r)
        if cur is None or (len(peps_sorted[i]), peps_sorted[i]) < (len(cur), cur):
            comp_to_rep[r] = peps_sorted[i]

    return {peps_sorted[i]: comp_to_rep[roots[i]] for i in range(n)}


def _write_peptide_cluster_tsv(mapping: Dict[str, str], out_tsv: Path) -> int:
    with open(out_tsv, "w") as f:
        f.write("peptide\tcluster_id\n")
        for pep, cid in mapping.items():
            f.write(f"{pep}\t{cid}\n")
    return len(set(mapping.values()))


def run_peptide_clustering() -> Dict[str, int]:
    """Peptide clustering.

    - Interaction peptides (~5K): full all-pairs rapidfuzz Levenshtein +
      single-linkage DSU. Tractable and honors the original spec.
    - PM peptides (~4.7M): full all-pairs ED is infeasible (≈11 trillion
      pairs even at rapidfuzz C-extension speeds). Fall back to MMseqs2
      easy-cluster at 80% identity for PM. This deviation is documented in
      ``docs/BENCHMARK_CLUSTERING.md``.
    - Appendix: MMseqs2 on interaction peptides + agreement stats vs the ED
      method, so the methods comparison the user asked for is preserved at
      interaction scale.
    """
    t0 = time.time()
    counts: Dict[str, int] = {}

    # Interaction: edit distance (feasible)
    src = OUT_ROOT / "unique_peptides_interaction.txt"
    peptides = _read_seqs(src)
    print(f"[peptide-ed] interaction: {len(peptides):,} unique peptides", flush=True)
    mapping = _cluster_peptides_edit_distance(peptides)
    n_int = _write_peptide_cluster_tsv(mapping, OUT_ROOT / "peptide_clusters_interaction.tsv")
    counts["peptide_clusters_interaction"] = n_int
    print(f"[peptide-ed]   -> {n_int:,} clusters", flush=True)

    # PM: MMseqs2 fallback (full ED on ~5M peptides is infeasible — see docstring)
    pm_count = _cluster_one(
        "peptides_pm_mmseqs",
        OUT_ROOT / "unique_peptides_pm.txt",
        OUT_ROOT / "peptide_clusters_pm.tsv",
        min_seq_id=0.8,
        coverage=0.8,
    )
    counts["peptide_clusters_pm"] = pm_count

    # Appendix: MMseqs2 on interaction peptides for methods comparison
    mm_count = _cluster_one(
        "peptides_mmseqs2",
        OUT_ROOT / "unique_peptides_interaction.txt",
        OUT_ROOT / "peptide_clusters_mmseqs2.tsv",
        min_seq_id=0.8,
        coverage=0.8,
    )
    counts["peptide_clusters_mmseqs2"] = mm_count

    # Agreement stats (interaction scope)
    ed_map = _read_cluster_tsv(OUT_ROOT / "peptide_clusters_interaction.tsv", seq_col="peptide")
    mm_map = _read_cluster_tsv(OUT_ROOT / "peptide_clusters_mmseqs2.tsv", seq_col="sequence")
    stats = _compute_agreement(ed_map, mm_map, max_pairs=5_000_000)
    with open(OUT_ROOT / "peptide_clustering_agreement.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[peptide-ed] agreement vs MMseqs2: {json.dumps(stats, indent=2)}", flush=True)

    _record_timing("peptide_cluster", time.time() - t0)
    return counts


def _read_cluster_tsv(path: Path, seq_col: str = "sequence") -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            si = header.index(seq_col)
        except ValueError:
            si = 0
        try:
            ci = header.index("cluster_id")
        except ValueError:
            ci = 1
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            mapping[parts[si]] = parts[ci]
    return mapping


def _compute_agreement(
    map_a: Dict[str, str], map_b: Dict[str, str], max_pairs: int = 5_000_000
) -> Dict[str, object]:
    import random

    common = sorted(set(map_a.keys()) & set(map_b.keys()))
    n = len(common)
    total_pairs = n * (n - 1) // 2
    if total_pairs == 0:
        return {"n_common": n, "total_pairs": 0}
    # Sample pairs
    if total_pairs <= max_pairs:
        sampled = total_pairs
        iter_pairs = ((i, j) for i in range(n) for j in range(i + 1, n))
    else:
        sampled = max_pairs
        rng = random.Random(1234)
        seen = set()
        def _gen() -> object:
            while len(seen) < max_pairs:
                i = rng.randrange(n)
                j = rng.randrange(n)
                if i == j:
                    continue
                a, b = (i, j) if i < j else (j, i)
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                yield a, b
        iter_pairs = _gen()

    both = only_a = only_b = neither = 0
    for i, j in iter_pairs:
        a_same = map_a[common[i]] == map_a[common[j]]
        b_same = map_b[common[i]] == map_b[common[j]]
        if a_same and b_same:
            both += 1
        elif a_same:
            only_a += 1
        elif b_same:
            only_b += 1
        else:
            neither += 1
    agree = both + neither
    return {
        "n_common": n,
        "total_pairs": total_pairs,
        "sampled_pairs": sampled,
        "ed_only_pairs": only_a,
        "mmseqs_only_pairs": only_b,
        "both_agree_same": both,
        "both_agree_diff": neither,
        "pct_agreement": 100.0 * agree / max(1, sampled),
    }


# ---------------------------------------------------------------------------
# Task 4: allele inventory
# ---------------------------------------------------------------------------

def run_allele_inventory() -> None:
    t0 = time.time()
    for src_name, out_name in [
        ("unique_mhc_one_alleles.txt", "mhc_one_alleles.json"),
        ("unique_mhc_two_alleles.txt", "mhc_two_alleles.json"),
    ]:
        src = OUT_ROOT / src_name
        seqs = sorted(_read_seqs(src))
        with open(OUT_ROOT / out_name, "w") as f:
            json.dump({"count": len(seqs), "alleles": seqs}, f, indent=2)
        print(f"[allele] {out_name}: {len(seqs):,} entries", flush=True)
    _record_timing("allele_inventory", time.time() - t0)


# ---------------------------------------------------------------------------
# Task 5: summary
# ---------------------------------------------------------------------------

def _cluster_size_distribution(tsv: Path, seq_col: str = "sequence") -> Dict[str, object]:
    if not tsv.exists():
        return {"status": "missing"}
    sizes: Dict[str, int] = {}
    with open(tsv) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            sizes[parts[1]] = sizes.get(parts[1], 0) + 1
    if not sizes:
        return {"n_clusters": 0}
    vals = sorted(sizes.values())
    n = len(vals)
    median = vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) // 2
    top10 = sorted(sizes.items(), key=lambda x: -x[1])[:10]
    return {
        "n_clusters": n,
        "min_size": vals[0],
        "median_size": median,
        "max_size": vals[-1],
        "top10": [{"cluster_id": c, "size": s} for c, s in top10],
    }


def generate_summary() -> None:
    t0 = time.time()
    summary: Dict[str, object] = {}

    if COUNTS_PATH.exists():
        summary["unique_sequence_counts"] = json.loads(COUNTS_PATH.read_text())

    cluster_jobs = [
        ("trb_cdr3_interaction_clusters.tsv", "sequence"),
        ("tra_cdr3_interaction_clusters.tsv", "sequence"),
        ("trb_cdr3_restriction_clusters.tsv", "sequence"),
        ("tra_cdr3_restriction_clusters.tsv", "sequence"),
        ("trb_cdr3_all_clusters.tsv", "sequence"),
        ("tra_cdr3_all_clusters.tsv", "sequence"),
        ("peptide_clusters_interaction.tsv", "peptide"),
        ("peptide_clusters_pm.tsv", "peptide"),
        ("peptide_clusters_mmseqs2.tsv", "sequence"),
    ]
    summary["cluster_size_distributions"] = {}
    for fname, seq_col in cluster_jobs:
        summary["cluster_size_distributions"][fname] = _cluster_size_distribution(OUT_ROOT / fname, seq_col)

    # TRB foundation status
    status, n_clusters = finalize_trb_foundation_if_done()
    summary["trb_foundation"] = {
        "status": status,
        "n_clusters": n_clusters,
        "pid_file": str(TRB_ALL_PID_FILE),
        "log": str(TRB_ALL_LOG),
    }

    # Peptide agreement
    agree_path = OUT_ROOT / "peptide_clustering_agreement.json"
    if agree_path.exists():
        summary["peptide_method_agreement"] = json.loads(agree_path.read_text())

    # Timings
    if TIMINGS_PATH.exists():
        summary["timings_seconds"] = json.loads(TIMINGS_PATH.read_text())

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] wrote {SUMMARY_PATH}", flush=True)
    print(json.dumps(summary, indent=2))
    _record_timing("summary", time.time() - t0)


# ---------------------------------------------------------------------------
# Shared timing helper
# ---------------------------------------------------------------------------

def _record_timing(task: str, seconds: float) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    current: Dict[str, float] = {}
    if TIMINGS_PATH.exists():
        try:
            current = json.loads(TIMINGS_PATH.read_text())
        except json.JSONDecodeError:
            current = {}
    current[task] = round(seconds, 2)
    with open(TIMINGS_PATH, "w") as f:
        json.dump(current, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        required=True,
        choices=["collect", "tcr_cluster", "peptide_cluster", "allele_inventory", "summary", "all"],
    )
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FASTA_DIR.mkdir(parents=True, exist_ok=True)
    MMSEQS_TMP.mkdir(parents=True, exist_ok=True)

    if args.task == "collect":
        collect_unique_sequences()
    elif args.task == "tcr_cluster":
        run_tcr_clustering()
    elif args.task == "peptide_cluster":
        run_peptide_clustering()
    elif args.task == "allele_inventory":
        run_allele_inventory()
    elif args.task == "summary":
        generate_summary()
    elif args.task == "all":
        collect_unique_sequences()
        run_tcr_clustering()
        run_peptide_clustering()
        run_allele_inventory()
        generate_summary()


if __name__ == "__main__":
    sys.exit(main())
