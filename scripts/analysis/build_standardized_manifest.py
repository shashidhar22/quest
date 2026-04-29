"""Build a machine-readable inventory of the standardized output dir.

Walks `data/standardized_again/{bucket}/` (18 buckets) and computes per-bucket:
  - manifest summary (row_count, dropped_count, elapsed_seconds, source_checksum_count)
  - parquet stats (file_count, total bytes, row_count_from_parquet, shard breakdown,
    row_count_matches_manifest flag)
  - column coverage (populated count + pct + unique cardinality for ID-like cols)
    across all 25 TARGET_COLUMNS using streaming iter_batches()
  - drop-reason aggregation parsed streamingly from dropped.tsv (only the `reason`
    column is needed; we don't load the full file)
  - drop_rate_pct = dropped / (row_count + dropped) * 100
  - yield rate vs raw_data inventory.json (when available)

Outputs:
  - docs/wiki/standardized_data/inventory.json (full machine-readable)
  - docs/wiki/standardized_data/inventory.csv  (flattened per-bucket summary)

Run from quest repo root:
    python scripts/analysis/build_standardized_manifest.py --embed
    python scripts/analysis/build_standardized_manifest.py --bucket vdjdb \
        --out /tmp/check.json --csv /tmp/check.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("std_manifest")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "data" / "standardized_again"
DEFAULT_OUT_JSON = REPO_ROOT / "docs" / "wiki" / "standardized_data" / "inventory.json"
DEFAULT_OUT_CSV = REPO_ROOT / "docs" / "wiki" / "standardized_data" / "inventory.csv"
DEFAULT_WIKI_ROOT = REPO_ROOT / "docs" / "wiki" / "standardized_data"
RAW_INVENTORY_JSON = REPO_ROOT / "docs" / "wiki" / "raw_data" / "inventory.json"

# The 25-column unified TARGET_COLUMNS schema. Mirrors quest/data/standardization.py:18-44.
TARGET_COLUMNS: list[str] = [
    "tra",
    "trav_gene",
    "trad_gene",
    "traj_gene",
    "tra_cdr1",
    "tra_cdr2",
    "tra_cdr3",
    "tra_full",
    "trb",
    "trbv_gene",
    "trbd_gene",
    "trbj_gene",
    "trb_cdr1",
    "trb_cdr2",
    "trb_cdr3",
    "trb_full",
    "peptide",
    "mhc_one",
    "mhc_two",
    "mhc_one_allele",
    "mhc_two_allele",
    "binding",
    "score",
    "source",
    "study_id",
]

# Track unique cardinality (via Python set) for these ID-like columns.
ID_LIKE_COLUMNS = {"tra", "trb", "peptide", "study_id", "mhc_one_allele"}

# Cap unique-tracking sets at this size; once exceeded, switch to a sentinel
# meaning "more than UNIQUE_CAP distinct values".
UNIQUE_CAP = 1_000_000

# Map standardized bucket name -> raw_data inventory source name.
# cedar_pmhc and iedb_pmhc share input data with cedar/iedb on the TCR side;
# imgthla, netmhcpan need normalisation to upper-case for the raw side.
# studies is a roll-up of all studies/ entries — handled specially in yield calc.
STD_TO_RAW: dict[str, str] = {
    "adc": "adc",
    "batman": "BATMAN",
    "cedar": "CEDAR",
    "cedar_pmhc": "CEDAR",
    "iedb": "IEDB",
    "iedb_pmhc": "IEDB",
    "imgthla": "IMGTHLA",
    "immuneaccess": "immuneACCESS",
    "immunecode": "immuneCODE",
    "mcpas": "McPAS-TCR",
    "netmhcpan": "NetMHCPan",
    "ots": "OTS",
    "rcc_atlas": "RCC_ATLAS",
    "tadb": "TaDB",
    "tcrdb": "tcrdb",
    "trait": "trait",
    "vdjdb": "vdjdb",
    # "studies" handled specially: sums all kind=study entries from raw inventory.
}

# Buckets whose raw counterpart was a sampled estimate in the raw_data inventory.
# Used to flag yield numbers as approximate.
SAMPLED_RAW_SOURCES = {"adc", "immuneaccess", "immunecode", "ots", "tcrdb"}

# Buckets that share input data with another bucket (note this in the yield block).
SHARED_INPUT_BUCKETS = {"cedar_pmhc": "cedar", "iedb_pmhc": "iedb"}


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclass
class ColumnCoverage:
    populated: int = 0
    pct: float = 0.0
    unique: int | str = 0  # int, or str sentinel like ">1000000" or "n/a"


@dataclass
class ShardEntry:
    file: str
    size_bytes: int
    rows: int


@dataclass
class DropReason:
    reason: str
    count: int
    pct: float


@dataclass
class BucketEntry:
    name: str
    path: str
    manifest: dict[str, Any] = field(default_factory=dict)
    parquet: dict[str, Any] = field(default_factory=dict)
    column_coverage: dict[str, dict[str, Any]] = field(default_factory=dict)
    drop_reasons: list[dict[str, Any]] = field(default_factory=list)
    drop_rate_pct: float | None = None
    yield_info: dict[str, Any] | None = None
    notes: str = ""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def human_size(n: int) -> str:
    """Match build_raw_data_manifest.py human_size formatting."""
    if n < 0:
        return f"{n}B"
    val: float = float(n)
    for unit in ("B", "K", "M", "G", "T", "P"):
        if val < 1024 or unit == "P":
            return f"{int(val)}{unit}" if unit == "B" else f"{val:.1f}{unit}"
        val /= 1024
    return f"{val}"


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Parquet metadata + column coverage
# -----------------------------------------------------------------------------


def parquet_stats(parquet_paths: list[Path]) -> tuple[int, list[ShardEntry]]:
    """Return (total_rows_from_metadata, list_of_shard_entries)."""
    import pyarrow.parquet as pq

    total = 0
    shards: list[ShardEntry] = []
    for p in parquet_paths:
        try:
            num_rows = pq.ParquetFile(p).metadata.num_rows
        except Exception as e:
            logger.warning("Failed to read parquet metadata for %s: %s", p, e)
            num_rows = -1
        size = p.stat().st_size
        shards.append(ShardEntry(file=p.name, size_bytes=size, rows=num_rows))
        if num_rows > 0:
            total += num_rows
    return total, shards


def compute_column_coverage(
    parquet_paths: list[Path],
    total_rows: int,
    *,
    max_shards: int = 30,
    bucket_name: str = "",
) -> tuple[dict[str, ColumnCoverage], dict[str, Any]]:
    """Stream parquet shards once, computing populated counts for every
    TARGET_COLUMN and unique cardinality for ID_LIKE_COLUMNS in a single pass.

    Strategy: for each shard, iter_batches() loading all present TARGET_COLUMNS
    at once. For each batch, run pyarrow.compute kernels per column (cheap;
    columnar in C). Only ID_LIKE columns trigger Python-side `to_pylist()` for
    set accumulation, and those bail out once the set exceeds UNIQUE_CAP.

    Single-pass design avoids paying parquet column-decoding overhead 25× per
    file, which matters for buckets like adc (2188 part files).

    When `len(parquet_paths) > max_shards`, sample evenly across the shards and
    extrapolate `populated` by `total_rows / sampled_rows`. Coverage `pct` is
    computed against the *sampled rows* directly so percentages remain accurate
    (sampling preserves proportions), while populated absolute counts are
    extrapolated. Unique counts on ID-like columns are reported as observed in
    the sample with a sentinel suffix indicating the sample size.

    Returns: (coverage_dict, sampling_info_dict). sampling_info contains:
       - sampled (bool)
       - shards_scanned (int) / shards_total (int)
       - rows_scanned (int)
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    out: dict[str, ColumnCoverage] = {col: ColumnCoverage() for col in TARGET_COLUMNS}
    sampling_info: dict[str, Any] = {
        "sampled": False,
        "shards_scanned": len(parquet_paths),
        "shards_total": len(parquet_paths),
        "rows_scanned": 0,
    }

    if total_rows <= 0 or not parquet_paths:
        for col in TARGET_COLUMNS:
            out[col].unique = "n/a" if col not in ID_LIKE_COLUMNS else 0
        return out, sampling_info

    # If too many shards, evenly sample down to max_shards. Always include
    # first + last for visibility.
    scan_paths = parquet_paths
    if len(parquet_paths) > max_shards:
        n = len(parquet_paths)
        # Evenly-spaced indices across [0, n-1]
        step = (n - 1) / (max_shards - 1) if max_shards > 1 else n
        idxs = sorted({int(round(i * step)) for i in range(max_shards)})
        idxs = [min(i, n - 1) for i in idxs]
        scan_paths = [parquet_paths[i] for i in idxs]
        sampling_info["sampled"] = True
        sampling_info["shards_scanned"] = len(scan_paths)
        logger.info(
            "[%s] sampling %d of %d shards for column coverage",
            bucket_name, len(scan_paths), len(parquet_paths),
        )

    # Probe the first shard's schema to determine which TARGET_COLUMNS exist.
    present_cols: set[str] = set()
    try:
        schema = pq.ParquetFile(parquet_paths[0]).schema_arrow
        present_cols = set(schema.names) & set(TARGET_COLUMNS)
    except Exception as e:
        logger.warning("Failed to read schema from %s: %s", parquet_paths[0], e)
        present_cols = set(TARGET_COLUMNS)

    # Initialize accumulators only for present columns.
    populated_counts: dict[str, int] = {col: 0 for col in present_cols}
    unique_sets: dict[str, set[str]] = {
        col: set() for col in (ID_LIKE_COLUMNS & present_cols)
    }
    unique_overflowed: dict[str, bool] = {
        col: False for col in (ID_LIKE_COLUMNS & present_cols)
    }

    cols_to_load = sorted(present_cols)
    if not cols_to_load:
        # Nothing to scan
        for col in TARGET_COLUMNS:
            out[col].populated = 0
            out[col].pct = 0.0
            out[col].unique = "n/a" if col not in ID_LIKE_COLUMNS else 0
        return out, sampling_info

    rows_scanned = 0
    for ppath in scan_paths:
        try:
            pfile = pq.ParquetFile(ppath)
        except Exception as e:
            logger.warning("Failed to open %s: %s", ppath, e)
            continue
        try:
            for batch in pfile.iter_batches(columns=cols_to_load, batch_size=65536):
                rows_scanned += batch.num_rows
                # batch.schema.names is the order of cols_to_load
                for ci, col in enumerate(batch.schema.names):
                    arr = batch.column(ci)
                    nn_mask = pc.is_valid(arr)
                    if pa.types.is_string(arr.type) or pa.types.is_large_string(
                        arr.type
                    ):
                        empty_mask = pc.equal(arr, pa.scalar("", type=arr.type))
                        pop_mask = pc.and_(nn_mask, pc.invert(empty_mask))
                    else:
                        pop_mask = nn_mask
                    populated_counts[col] += int(pc.sum(pop_mask).as_py() or 0)

                    if col in unique_sets and not unique_overflowed[col]:
                        filtered = arr.filter(pop_mask)
                        uset = unique_sets[col]
                        for v in filtered.to_pylist():
                            if v is None:
                                continue
                            uset.add(v if isinstance(v, str) else str(v))
                            if len(uset) > UNIQUE_CAP:
                                unique_overflowed[col] = True
                                # Free memory
                                unique_sets[col] = set()
                                break
        except Exception as e:
            logger.warning("Failed iter_batches on %s: %s", ppath, e)
            continue

    sampling_info["rows_scanned"] = rows_scanned
    is_sampled = sampling_info["sampled"]
    extrap_factor = (
        (total_rows / rows_scanned) if (is_sampled and rows_scanned > 0) else 1.0
    )

    for col in TARGET_COLUMNS:
        if col not in present_cols:
            out[col].populated = 0
            out[col].pct = 0.0
            out[col].unique = "n/a" if col not in ID_LIKE_COLUMNS else 0
            continue
        pop = populated_counts.get(col, 0)
        # pct is computed from the sampled population (preserves proportion).
        denom = rows_scanned if is_sampled and rows_scanned > 0 else total_rows
        out[col].pct = (pop / denom * 100.0) if denom > 0 else 0.0
        # Extrapolate populated count to whole-dataset estimate when sampling.
        out[col].populated = (
            int(round(pop * extrap_factor)) if is_sampled else pop
        )
        if col in ID_LIKE_COLUMNS:
            if unique_overflowed.get(col):
                out[col].unique = f">{UNIQUE_CAP}"
            else:
                u = len(unique_sets.get(col, set()))
                # Mark unique counts from sampled buckets so consumers know
                # they're a lower bound, not a population estimate.
                out[col].unique = (
                    f"{u} (sampled from {sampling_info['shards_scanned']}"
                    f"/{sampling_info['shards_total']} shards)"
                    if is_sampled
                    else u
                )
        else:
            out[col].unique = "n/a"

    return out, sampling_info


# -----------------------------------------------------------------------------
# Drop-reason aggregation (streaming)
# -----------------------------------------------------------------------------


def aggregate_drop_reasons(dropped_tsv: Path) -> tuple[list[DropReason], int]:
    """Stream dropped.tsv line-by-line, count by `reason` column.

    The file format is: reason\tsource_file\trow_index\tfield\traw_value
    Header is line 0. Returns (sorted_drop_reasons, total_data_rows). The total
    is included as a sanity-check against manifest.dropped_count.
    """
    counter: Counter[str] = Counter()
    total = 0
    if not dropped_tsv.exists():
        return [], 0
    try:
        with dropped_tsv.open("r", encoding="utf-8", errors="replace") as f:
            header = f.readline()  # discard header
            if not header:
                return [], 0
            # Parse just the first tab-delimited field per line (the reason).
            for line in f:
                # Fast path: find the first tab; if none, treat the whole line as
                # the reason (defensive, shouldn't happen on well-formed files).
                tab = line.find("\t")
                if tab < 0:
                    reason = line.rstrip("\n")
                else:
                    reason = line[:tab]
                if not reason:
                    continue
                counter[reason] += 1
                total += 1
    except Exception as e:
        logger.warning("Failed to parse %s: %s", dropped_tsv, e)
        return [], 0

    if total == 0:
        return [], 0
    out: list[DropReason] = []
    for reason, count in counter.most_common():
        out.append(DropReason(reason=reason, count=count, pct=count / total * 100.0))
    return out, total


# -----------------------------------------------------------------------------
# Yield computation (vs raw_data inventory)
# -----------------------------------------------------------------------------


def load_raw_inventory_index() -> dict[str, dict[str, Any]] | None:
    """Load raw_data/inventory.json and index by source name. Also include a
    synthetic 'studies' aggregate (sum of total_records across kind=study).
    Returns None if the raw inventory is missing.
    """
    if not RAW_INVENTORY_JSON.exists():
        return None
    try:
        payload = json.loads(RAW_INVENTORY_JSON.read_text())
    except Exception as e:
        logger.warning("Failed to read raw inventory: %s", e)
        return None
    idx: dict[str, dict[str, Any]] = {}
    studies_total = 0
    studies_n = 0
    studies_any_sampled = False
    for s in payload.get("sources", []):
        idx[s["name"]] = s
        if s.get("kind") == "study" and s.get("total_records") is not None:
            studies_total += int(s["total_records"])
            studies_n += 1
            if not s.get("exact_record_count", True):
                studies_any_sampled = True
    if studies_n > 0:
        idx["__studies_aggregate__"] = {
            "name": "__studies_aggregate__",
            "kind": "study",
            "total_records": studies_total,
            "exact_record_count": not studies_any_sampled,
            "study_count": studies_n,
        }
    return idx


def compute_yield(
    bucket_name: str,
    row_count: int,
    raw_index: dict[str, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if raw_index is None:
        return None
    raw_key: str | None
    if bucket_name == "studies":
        raw_key = "__studies_aggregate__"
    else:
        raw_key = STD_TO_RAW.get(bucket_name)
    if raw_key is None or raw_key not in raw_index:
        return None
    raw_entry = raw_index[raw_key]
    raw_records = raw_entry.get("total_records")
    if raw_records is None or raw_records == 0:
        return None
    sampled = bucket_name in SAMPLED_RAW_SOURCES or not raw_entry.get(
        "exact_record_count", True
    )
    yield_pct = row_count / raw_records * 100.0
    info: dict[str, Any] = {
        "raw_records": int(raw_records),
        "raw_source": raw_key,
        "yield_pct": round(yield_pct, 2),
        "sampled_raw": sampled,
    }
    if bucket_name == "studies":
        info["study_count"] = raw_entry.get("study_count")
        info["note"] = "aggregate of all kind=study sources in raw_data inventory"
    if bucket_name in SHARED_INPUT_BUCKETS:
        info["shared_input_with"] = SHARED_INPUT_BUCKETS[bucket_name]
        info["note"] = (
            f"shares raw input with `{SHARED_INPUT_BUCKETS[bucket_name]}` bucket "
            f"(both standardize from {raw_key})"
        )
    if sampled:
        info["yield_pct_str"] = f"~{yield_pct:.2f}%"
        existing_note = info.get("note", "")
        sample_note = "raw record total is a sampled estimate; yield is approximate"
        info["note"] = f"{existing_note}; {sample_note}".strip("; ")
    # Yield > 100% means: (a) the standardizer emits >1 row per raw record by
    # design (vdjdb's score-0 split, NetMHCpan ID-expansion), (b) the raw count
    # is undercounted, or (c) the standardizer ran on a different snapshot
    # than the inventory observed. Surface this loudly — readers should not
    # treat a >100% yield as a measurement.
    if yield_pct > 100.0:
        existing_note = info.get("note", "")
        warn = (
            f"yield exceeds 100% ({yield_pct:.1f}%) — likely row-multiplication "
            f"in standardizer (e.g. vdjdb score-0 split, peptide list explosion) "
            f"or raw-count undercount; do not treat as a literal proportion"
        )
        info["yield_exceeds_100"] = True
        info["note"] = f"{existing_note}; {warn}".strip("; ")
    return info


# -----------------------------------------------------------------------------
# Per-bucket inventory
# -----------------------------------------------------------------------------


def inventory_bucket(
    name: str,
    path: Path,
    *,
    raw_index: dict[str, dict[str, Any]] | None = None,
    max_coverage_shards: int = 30,
) -> BucketEntry:
    rel_path = str(path.relative_to(REPO_ROOT))
    entry = BucketEntry(name=name, path=rel_path)

    manifest_path = path / "manifest.json"
    dropped_path = path / "dropped.tsv"
    parquet_paths = sorted(path.glob("part_*.parquet"))

    # ---- Manifest summary
    if manifest_path.exists():
        try:
            mf = load_manifest(manifest_path)
            entry.manifest = {
                "timestamp": mf.get("timestamp"),
                "row_count": mf.get("row_count"),
                "dropped_count": mf.get("dropped_count"),
                "elapsed_seconds": mf.get("elapsed_seconds"),
                "source_checksum_count": len(mf.get("source_checksums") or {}),
            }
        except Exception as e:
            entry.notes = f"manifest read error: {type(e).__name__}: {e}"
            entry.manifest = {}
    else:
        entry.notes = "manifest.json missing"

    manifest_row_count: int = int(entry.manifest.get("row_count") or 0)
    manifest_dropped: int = int(entry.manifest.get("dropped_count") or 0)

    # ---- Parquet stats
    total_rows_pq, shards = parquet_stats(parquet_paths)
    total_size = sum(s.size_bytes for s in shards)
    matches = (manifest_row_count == total_rows_pq) if entry.manifest else None
    entry.parquet = {
        "file_count": len(parquet_paths),
        "size_bytes": total_size,
        "size_human": human_size(total_size),
        "row_count_from_parquet": total_rows_pq,
        "row_count_matches_manifest": matches,
        "shard_breakdown": [asdict(s) for s in shards],
    }
    if matches is False:
        logger.warning(
            "[%s] parquet row count %d != manifest row_count %d",
            name, total_rows_pq, manifest_row_count,
        )

    # ---- Column coverage
    # Prefer the parquet-derived total for the denominator (it's the source of
    # truth for what's actually in the files); fall back to manifest if parquet
    # was unreadable.
    coverage_total = total_rows_pq if total_rows_pq > 0 else manifest_row_count
    if parquet_paths and coverage_total > 0:
        cov, samp_info = compute_column_coverage(
            parquet_paths,
            coverage_total,
            max_shards=max_coverage_shards,
            bucket_name=name,
        )
        entry.column_coverage = {
            col: {
                "populated": c.populated,
                "pct": round(c.pct, 2),
                "unique": c.unique,
            }
            for col, c in cov.items()
        }
        entry.parquet["coverage_sampling"] = samp_info
    else:
        entry.column_coverage = {}

    # ---- Drop reasons (streaming)
    drop_reasons, drop_total = aggregate_drop_reasons(dropped_path)
    entry.drop_reasons = [asdict(d) for d in drop_reasons]
    if drop_total and manifest_dropped and drop_total != manifest_dropped:
        logger.warning(
            "[%s] dropped.tsv data lines (%d) != manifest.dropped_count (%d)",
            name, drop_total, manifest_dropped,
        )

    # ---- Drop rate
    denom = manifest_row_count + manifest_dropped
    if denom > 0:
        entry.drop_rate_pct = round(manifest_dropped / denom * 100.0, 4)

    # ---- Yield rate
    entry.yield_info = compute_yield(name, manifest_row_count, raw_index)

    return entry


# -----------------------------------------------------------------------------
# Embedding (markdown)
# -----------------------------------------------------------------------------


PLACEHOLDER_RE = re.compile(
    r"<!--\s*TODO:\s*filled by Quantifier.*?-->",
    re.IGNORECASE | re.DOTALL,
)
AUTO_BLOCK_RE = re.compile(
    r"<!--\s*BEGIN:\s*AUTO-INVENTORY.*?<!--\s*END:\s*AUTO-INVENTORY\s*-->",
    re.DOTALL | re.IGNORECASE,
)


def render_markdown(entry: BucketEntry) -> str:
    """Render an inventory block for one BucketEntry to embed under the
    Quantifier-placeholder in the corresponding sources/{bucket}.md page."""
    lines: list[str] = []
    lines.append("<!-- BEGIN: AUTO-INVENTORY (build_standardized_manifest.py) -->")
    lines.append("")

    # Manifest summary
    mf = entry.manifest or {}
    rc = mf.get("row_count")
    dc = mf.get("dropped_count")
    el = mf.get("elapsed_seconds")
    sc = mf.get("source_checksum_count")
    ts = mf.get("timestamp")
    lines.append(f"- **Path**: `{entry.path}`")
    if ts:
        lines.append(f"- **Standardized at**: {ts}")
    if rc is not None:
        rc_str = f"{rc:,}"
        lines.append(f"- **Rows (manifest)**: {rc_str}")
    if dc is not None:
        dc_str = f"{dc:,}"
        rate = (
            f" ({entry.drop_rate_pct:.2f}% drop rate)"
            if entry.drop_rate_pct is not None
            else ""
        )
        lines.append(f"- **Dropped**: {dc_str}{rate}")
    if el is not None:
        lines.append(f"- **Standardization elapsed**: {el:,.1f}s")
    if sc is not None:
        lines.append(f"- **Source files (checksummed)**: {sc}")

    # Parquet stats
    pq = entry.parquet or {}
    if pq:
        match_str = ""
        if pq.get("row_count_matches_manifest") is True:
            match_str = " (matches manifest)"
        elif pq.get("row_count_matches_manifest") is False:
            match_str = (
                f" (MISMATCH vs manifest row_count="
                f"{mf.get('row_count'):,})"
                if mf.get("row_count") is not None
                else " (MISMATCH vs manifest)"
            )
        lines.append(
            f"- **Parquet**: {pq.get('file_count', 0)} files, "
            f"{pq.get('size_human', '?')} ({pq.get('size_bytes', 0):,} bytes), "
            f"{int(pq.get('row_count_from_parquet') or 0):,} rows"
            f"{match_str}"
        )

    # Yield
    if entry.yield_info:
        yi = entry.yield_info
        ypct = yi.get("yield_pct_str") or f"{yi['yield_pct']:.2f}%"
        line = (
            f"- **Yield vs raw**: {ypct} "
            f"({rc:,} of {yi['raw_records']:,} raw records from `{yi['raw_source']}`)"
        )
        if yi.get("note"):
            line += f" — _{yi['note']}_"
        lines.append(line)

    # Top 8 most-populated columns
    if entry.column_coverage:
        ranked = sorted(
            entry.column_coverage.items(),
            key=lambda kv: -kv[1]["populated"],
        )
        non_zero = [(k, v) for k, v in ranked if v["populated"] > 0]
        zero = [(k, v) for k, v in ranked if v["populated"] == 0]

        lines.append("")
        lines.append("**Top 8 most-populated columns:**")
        lines.append("")
        lines.append("| Column | Populated | Coverage | Unique |")
        lines.append("|--------|-----------|----------|--------|")
        for col, c in non_zero[:8]:
            uniq = c["unique"]
            uniq_str = f"{uniq:,}" if isinstance(uniq, int) else str(uniq)
            lines.append(
                f"| `{col}` | {c['populated']:,} | {c['pct']:.2f}% | {uniq_str} |"
            )

        # Bottom 5 least-populated (across all 25 columns; choose 5 lowest non-empty
        # if any, else show 5 columns at 0% as a sparseness signal).
        lines.append("")
        lines.append("**Bottom 5 least-populated columns:**")
        lines.append("")
        lines.append("| Column | Populated | Coverage |")
        lines.append("|--------|-----------|----------|")
        # Bottom 5 = the 5 with the lowest pct (zeros first); we want the
        # *least populated* including zeros, so just pick from ranked tail.
        bottom = ranked[-5:]
        for col, c in bottom:
            lines.append(f"| `{col}` | {c['populated']:,} | {c['pct']:.2f}% |")

    # Drop reasons (top 10)
    if entry.drop_reasons:
        lines.append("")
        lines.append("**Top 10 drop reasons:**")
        lines.append("")
        lines.append("| Reason | Count | Share |")
        lines.append("|--------|-------|-------|")
        for d in entry.drop_reasons[:10]:
            lines.append(
                f"| `{d['reason']}` | {d['count']:,} | {d['pct']:.2f}% |"
            )

    if entry.notes:
        lines.append("")
        lines.append(f"_Notes_: {entry.notes}")

    lines.append("")
    lines.append("<!-- END: AUTO-INVENTORY -->")
    return "\n".join(lines)


def embed_inventory_into_wiki(
    inventory_json: Path, wiki_root: Path
) -> dict[str, list[str]]:
    """Substitute the placeholder (or replace existing AUTO-INVENTORY block) in
    each docs/wiki/standardized_data/sources/{bucket}.md with rendered inventory.
    Idempotent: re-running on already-embedded files re-renders the block.

    Returns: {"updated": [paths], "missing_md": [paths], "no_placeholder_in_md": [paths]}
    """
    payload = json.loads(inventory_json.read_text())
    sources_dir = wiki_root / "sources"
    updated: list[str] = []
    missing_md: list[str] = []
    no_placeholder: list[str] = []

    for entry_dict in payload.get("buckets", []):
        bucket_name = entry_dict["name"]
        md_path = sources_dir / f"{bucket_name}.md"
        if not md_path.exists():
            missing_md.append(str(md_path))
            continue

        # Re-hydrate just enough to render. We render from the dict directly so
        # there's no risk of dataclass schema drift.
        entry = BucketEntry(
            name=entry_dict["name"],
            path=entry_dict.get("path", ""),
            manifest=entry_dict.get("manifest", {}),
            parquet=entry_dict.get("parquet", {}),
            column_coverage=entry_dict.get("column_coverage", {}),
            drop_reasons=entry_dict.get("drop_reasons", []),
            drop_rate_pct=entry_dict.get("drop_rate_pct"),
            yield_info=entry_dict.get("yield_info"),
            notes=entry_dict.get("notes", ""),
        )
        block = render_markdown(entry)

        text = md_path.read_text()
        if AUTO_BLOCK_RE.search(text):
            # Use a lambda returning the literal block to avoid re.sub
            # interpreting backreferences in the block.
            new_text = AUTO_BLOCK_RE.sub(lambda _m: block, text)
        elif PLACEHOLDER_RE.search(text):
            new_text = PLACEHOLDER_RE.sub(lambda _m: block, text)
        else:
            no_placeholder.append(str(md_path))
            continue
        if new_text != text:
            md_path.write_text(new_text)
            updated.append(str(md_path))
    return {
        "updated": updated,
        "missing_md": missing_md,
        "no_placeholder_in_md": no_placeholder,
    }


MASTER_TABLE_RE = re.compile(
    r"<!--\s*BEGIN:\s*AUTO-MASTER-TABLE.*?<!--\s*END:\s*AUTO-MASTER-TABLE\s*-->",
    re.DOTALL | re.IGNORECASE,
)


def render_master_table(inventory_json: Path) -> str:
    """Render a master summary table from inventory.json. Returns the block
    delimited by AUTO-MASTER-TABLE markers."""
    payload = json.loads(inventory_json.read_text())
    rows: list[str] = []

    for b in payload.get("buckets", []):
        name = b["name"]
        manifest = b.get("manifest", {})
        parquet = b.get("parquet", {})
        cov = b.get("column_coverage", {})
        yield_info = b.get("yield") or b.get("yield_info") or {}

        rcount = manifest.get("row_count", 0)
        dcount = manifest.get("dropped_count", 0)
        drate = b.get("drop_rate_pct")
        size_b = parquet.get("size_bytes", 0)
        size_h = parquet.get("size_human") or human_size(size_b)
        files = parquet.get("file_count", 0)

        # Headline coverage: % of rows with TRA-CDR3 or TRB-CDR3 populated, plus peptide
        def _pct(col: str) -> str:
            c = cov.get(col, {})
            p = c.get("pct")
            return f"{p:.0f}%" if isinstance(p, (int, float)) else "—"

        cov_brief = (
            f"trb={_pct('trb')} "
            f"tra={_pct('tra')} "
            f"pep={_pct('peptide')} "
            f"mhc1={_pct('mhc_one')}"
        )

        if isinstance(drate, (int, float)):
            drate_str = f"{drate:.2f}%"
        else:
            drate_str = "—"
        yp = yield_info.get("yield_pct")
        yield_str = f"{yp:.1f}%" if isinstance(yp, (int, float)) else "—"

        rows.append(
            f"| [`{name}`](sources/{name}.md) | "
            f"{rcount:,} | {dcount:,} | {drate_str} | "
            f"{size_h} | {files} | {cov_brief} | {yield_str} |"
        )

    rows.sort()
    totals = payload.get("totals", {})
    total_rows = totals.get("total_row_count", sum(
        b.get("manifest", {}).get("row_count", 0) for b in payload.get("buckets", [])
    ))
    total_dropped = totals.get("total_dropped_count", sum(
        b.get("manifest", {}).get("dropped_count", 0) for b in payload.get("buckets", [])
    ))

    out: list[str] = []
    out.append("<!-- BEGIN: AUTO-MASTER-TABLE (build_standardized_manifest.py) -->")
    out.append("")
    out.append(
        f"**Last regenerated**: {payload.get('generated_at', '?')} · "
        f"**Buckets**: {len(payload.get('buckets', []))} · "
        f"**Total standardized rows**: {total_rows:,} · "
        f"**Total dropped (during validation)**: {total_dropped:,}"
    )
    out.append("")
    out.append("| Bucket | Rows | Dropped | Drop rate | Parquet size | Files | Coverage (trb/tra/pep/mhc1) | Yield vs raw |")
    out.append("|--------|-----:|--------:|----------:|-------------:|------:|-----------------------------|-------------:|")
    out.extend(rows)
    out.append("")
    out.append(
        "*Rows column*: count from `manifest.json:row_count` (verified against parquet metadata). "
        "*Yield* = standardized rows / raw input rows (where the raw counterpart is unambiguous; some "
        "standardizers — e.g. vdjdb — emit >1 row per input record by design, so >100% is expected). "
        "*Coverage* shows the percent of rows where the column is non-null AND non-empty."
    )
    out.append("")
    out.append("<!-- END: AUTO-MASTER-TABLE -->")
    return "\n".join(out)


def embed_master_table(inventory_json: Path, wiki_root: Path) -> bool:
    readme = wiki_root / "README.md"
    if not readme.exists():
        return False
    text = readme.read_text()
    if not MASTER_TABLE_RE.search(text):
        return False
    block = render_master_table(inventory_json)
    new_text = MASTER_TABLE_RE.sub(lambda _m: block, text)
    if new_text != text:
        readme.write_text(new_text)
        return True
    return False


# -----------------------------------------------------------------------------
# Output writers
# -----------------------------------------------------------------------------


def discover_buckets(root: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            out.append((d.name, d))
    return out


def write_outputs(
    buckets: list[BucketEntry],
    out_json: Path,
    out_csv: Path,
    *,
    started_at: float,
    args: argparse.Namespace,
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.time() - started_at, 1),
        "args": {
            "root": str(args.root),
            "bucket_filter": args.bucket,
            "max_coverage_shards": args.max_coverage_shards,
        },
        "buckets": [_bucket_to_dict(b) for b in buckets],
        "totals": {
            "bucket_count": len(buckets),
            "total_parquet_bytes": sum(
                int(b.parquet.get("size_bytes") or 0) for b in buckets
            ),
            "total_rows": sum(
                int(b.manifest.get("row_count") or 0) for b in buckets
            ),
            "total_dropped": sum(
                int(b.manifest.get("dropped_count") or 0) for b in buckets
            ),
            "buckets_with_row_count_mismatch": [
                b.name for b in buckets
                if b.parquet.get("row_count_matches_manifest") is False
            ],
        },
    }
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("Wrote %s (%d buckets)", out_json, len(buckets))

    # Flat CSV
    fields = [
        "name",
        "path",
        "row_count",
        "dropped_count",
        "drop_rate_pct",
        "elapsed_seconds",
        "source_checksum_count",
        "parquet_files",
        "parquet_size_bytes",
        "parquet_size_human",
        "row_count_from_parquet",
        "row_count_matches_manifest",
        "top_drop_reason",
        "top_drop_reason_count",
        "yield_pct",
        "raw_records",
        "raw_source",
        "notes",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for b in buckets:
            mf = b.manifest or {}
            pq = b.parquet or {}
            top_dr = b.drop_reasons[0] if b.drop_reasons else {}
            yi = b.yield_info or {}
            w.writerow(
                {
                    "name": b.name,
                    "path": b.path,
                    "row_count": mf.get("row_count", ""),
                    "dropped_count": mf.get("dropped_count", ""),
                    "drop_rate_pct": (
                        f"{b.drop_rate_pct:.4f}"
                        if b.drop_rate_pct is not None
                        else ""
                    ),
                    "elapsed_seconds": mf.get("elapsed_seconds", ""),
                    "source_checksum_count": mf.get("source_checksum_count", ""),
                    "parquet_files": pq.get("file_count", ""),
                    "parquet_size_bytes": pq.get("size_bytes", ""),
                    "parquet_size_human": pq.get("size_human", ""),
                    "row_count_from_parquet": pq.get("row_count_from_parquet", ""),
                    "row_count_matches_manifest": pq.get(
                        "row_count_matches_manifest", ""
                    ),
                    "top_drop_reason": top_dr.get("reason", ""),
                    "top_drop_reason_count": top_dr.get("count", ""),
                    "yield_pct": yi.get("yield_pct", ""),
                    "raw_records": yi.get("raw_records", ""),
                    "raw_source": yi.get("raw_source", ""),
                    "notes": b.notes,
                }
            )
    logger.info("Wrote %s", out_csv)


def _bucket_to_dict(b: BucketEntry) -> dict[str, Any]:
    """Custom serialization: rename `yield_info` -> `yield` in JSON for
    readability (Python attribute can't be `yield`).
    """
    d = asdict(b)
    yi = d.pop("yield_info", None)
    d["yield"] = yi
    return d


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                   help="Root of standardized data (default: data/standardized_again/)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_JSON,
                   help="Output JSON path")
    p.add_argument("--csv", type=Path, default=DEFAULT_OUT_CSV,
                   help="Output CSV path")
    p.add_argument("--bucket", type=str, default=None,
                   help="Limit to one bucket name (e.g. vdjdb). Requires explicit "
                        "--out and --csv to avoid clobbering the full inventory.")
    p.add_argument("--max-coverage-shards", type=int, default=30,
                   help="When a bucket has more than this many parquet shards, "
                        "sample evenly across them for column-coverage calc and "
                        "extrapolate. Set very high to disable sampling. "
                        "Default 30 keeps full inventory under ~3 min.")
    p.add_argument("--embed", action="store_true",
                   help="After inventorying, embed results into per-source markdown "
                        "files at docs/wiki/standardized_data/sources/{bucket}.md.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    # Safety: --bucket NAME with default --out OR --csv would overwrite the
    # full inventory with a 1-bucket file. Refuse, mirroring raw-data script.
    # Both --out AND --csv must be redirected — code review C1 caught that the
    # csv side was previously unguarded.
    if args.bucket and (args.out == DEFAULT_OUT_JSON or args.csv == DEFAULT_OUT_CSV):
        p.error(
            "--bucket NAME without explicit --out AND --csv would overwrite the "
            "full inventory. Pass both, e.g. "
            "--out /tmp/check.json --csv /tmp/check.csv."
        )

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    started = time.time()
    raw_index = load_raw_inventory_index()
    if raw_index is None:
        logger.warning(
            "Raw-data inventory not found at %s; yield rates will be unavailable",
            RAW_INVENTORY_JSON,
        )

    discovered = discover_buckets(args.root)
    logger.info("Discovered %d buckets under %s", len(discovered), args.root)

    pending = [
        (n, pp) for (n, pp) in discovered
        if (args.bucket is None or n == args.bucket)
    ]

    try:
        from tqdm import tqdm
        bar = tqdm(pending, unit="bucket")
    except ImportError:
        bar = pending

    buckets: list[BucketEntry] = []
    for name, path_ in bar:
        if hasattr(bar, "set_description"):
            bar.set_description(name)
        else:
            logger.info("Inventorying %s", name)
        t0 = time.time()
        try:
            entry = inventory_bucket(
                name, path_,
                raw_index=raw_index,
                max_coverage_shards=args.max_coverage_shards,
            )
        except Exception as e:
            logger.exception("Failed to inventory %s: %s", name, e)
            entry = BucketEntry(
                name=name,
                path=str(path_.relative_to(REPO_ROOT)),
                notes=f"INVENTORY ERROR: {type(e).__name__}: {e}",
            )
        dt = time.time() - t0
        rc = entry.manifest.get("row_count") if entry.manifest else None
        rc_str = f"{rc:,}" if isinstance(rc, int) else "—"
        logger.info(
            "  %s done in %.1fs (rows=%s, parts=%d, drop_reasons=%d)",
            name, dt, rc_str,
            int(entry.parquet.get("file_count") or 0),
            len(entry.drop_reasons),
        )
        buckets.append(entry)

    write_outputs(buckets, args.out, args.csv, started_at=started, args=args)

    if args.embed:
        try:
            res = embed_inventory_into_wiki(args.out, DEFAULT_WIKI_ROOT)
            logger.info(
                "Embedder: updated=%d, missing_md=%d, no_placeholder=%d",
                len(res["updated"]),
                len(res["missing_md"]),
                len(res["no_placeholder_in_md"]),
            )
            if res["missing_md"]:
                logger.info("  missing_md: %s", res["missing_md"])
            if res["no_placeholder_in_md"]:
                logger.info("  no_placeholder: %s", res["no_placeholder_in_md"])
        except Exception as e:
            logger.exception("Embedder failed: %s", e)
        try:
            updated = embed_master_table(args.out, DEFAULT_WIKI_ROOT)
            logger.info("Master table: %s", "updated" if updated else "no change")
        except Exception as e:
            logger.exception("Master table embedder failed: %s", e)

    logger.info("Total wall time: %.1fs", time.time() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
