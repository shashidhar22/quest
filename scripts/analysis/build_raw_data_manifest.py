"""Build a machine-readable inventory of raw_data/.

Walks `data/raw_data/databases/` and `data/raw_data/studies/`, computing per-source:
  - byte size, file count, format breakdown
  - per-file record counts (with method + provenance)
  - aggregate record total per source
  - cross-reference to per-source standardizer (when present)

Outputs:
  - docs/wiki/raw_data/inventory.json (full machine-readable)
  - docs/wiki/raw_data/inventory.csv  (flattened per-source summary)

Counting rules (per-format):
  - csv/tsv:      `wc -l` minus 1 (header). Special-case OTS = wc -l minus 2 (json metadata + header).
  - parquet:      pyarrow.parquet.ParquetFile(path).metadata.num_rows
  - fasta:        count of `^>` lines (entries)
  - jsonl:        `wc -l`
  - json:         len(json.load(...)) if list else 1 (size-bounded; otherwise null)
  - xlsx:         pandas.read_excel per sheet; sum across sheets
  - zip/tar.gz:   if archive < 1 GB, list members and try to count CSV/TSV/FASTA inside
                  by streaming; otherwise report null with notes
  - sql/sql.gz:   record_count=null, notes='sql dump, not counted'
  - h5/bw/...:    record_count=null, notes='binary; not row-shaped'

Sampling: when a primary-data file glob has >100 files of identical format and the
source is in the "huge" set, count a stratified sample of N (default 30) files and
extrapolate; otherwise count exactly. Use --exact to disable sampling everywhere.

Run from quest repo root:
    python scripts/analysis/build_raw_data_manifest.py
    python scripts/analysis/build_raw_data_manifest.py --source vdjdb --exact
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import logging
import math
import os
import random
import re
import statistics
import subprocess
import sys
import tarfile
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("manifest")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = REPO_ROOT / "data" / "raw_data"
DEFAULT_OUT_JSON = REPO_ROOT / "docs" / "wiki" / "raw_data" / "inventory.json"
DEFAULT_OUT_CSV = REPO_ROOT / "docs" / "wiki" / "raw_data" / "inventory.csv"
STD_DIR = REPO_ROOT / "scripts" / "data_processing" / "standardize"

# Files / paths to skip wholesale (not record-counted, not listed as data files).
EXCLUDE_NAME_PREFIXES = (
    "download_",
    ".",  # hidden
)
EXCLUDE_NAME_SUFFIXES = (
    ".log",
    ".sh",
    ".py",  # download scripts only end up here; per-source we whitelist data .py if any
    ".jar",
    ".pper",
    ".pack",
    ".idx",
    ".gitattributes",
    ".properties",
)
EXCLUDE_BASENAMES = {
    "checkpoint.json",
    "README.md",
    "README",
    "LICENSE.txt",
    "LICENSE",
    "LICENCE.md",
    "LICENCE",
    "Manual.md",
    "manifest.tsv",  # in studies/, this is curator metadata
    "download_summary.txt",
    "filelist.txt",  # GEO study filelists
    ".DS_Store",
}
EXCLUDE_DIR_NAMES = {
    ".git",
    ".github",
    ".claude",
    "__MACOSX",
    "__pycache__",
}

# Documentation-style files: surfaced under non_data_artifacts but not record-counted.
DOC_NAME_REGEX = re.compile(
    r"^(README|readme|LICEN[SC]E|licen[sc]e|Manual|MANUAL|NOMENCLATURE|.*\.md)$"
)

# Plain-text files in IMGTHLA / NetMHCPan that are non-tabular metadata.
NON_TABULAR_TXT_BASENAMES = {
    "Allele_status.txt",
    "Allelelist_history.txt",
    "Deleted_alleles.txt",
    "Nomenclature_2009.txt",
    "change_log.txt",
    "release_version.txt",
    "sversion_history.txt",
    "version_report.txt",
    "md5checksum.txt",
    "latest-version.txt",
    "motif_pwms.txt",
    "cluster_members.txt",
    "MHC_pseudo.dat",
    "pseudosequence.2016.all.X.dat",
}

# Sources we treat as "huge" for sampling decisions.
HUGE_SOURCES = {"adc", "immuneACCESS", "immuneCODE", "OTS", "tcrdb"}

# When sampling huge sources, skip files larger than this for the per-file count
# (we still include their size in size_bytes via gather_files). Anything larger
# would dominate `wc -l` time without improving mean estimate quality much.
SAMPLE_SKIP_FILE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB

ARCHIVE_INSPECT_LIMIT_BYTES = 1 * 1024 * 1024 * 1024  # 1 GiB
JSON_INSPECT_LIMIT_BYTES = 50 * 1024 * 1024  # 50 MiB

# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclass
class FileEntry:
    path: str  # path relative to source root
    format: str
    size_bytes: int
    record_count: int | None = None
    record_definition: str = ""
    counting_method: str = ""
    notes: str = ""


@dataclass
class SourceEntry:
    name: str
    kind: str  # "database" | "study"
    path: str  # relative to repo root
    size_bytes: int = 0
    size_human: str = ""
    file_count: int = 0
    format_breakdown: dict[str, int] = field(default_factory=dict)
    primary_data_files: list[FileEntry] = field(default_factory=list)
    total_records: int | None = None
    total_records_method: str = ""
    total_records_ci: list[float] | None = None
    exact_record_count: bool = True
    record_definition: str = ""
    notes: str = ""
    non_data_artifacts: list[str] = field(default_factory=list)
    standardizer: dict[str, Any] | None = None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def human_size(n: int) -> str:
    for unit in ("B", "K", "M", "G", "T", "P"):
        if n < 1024 or unit == "P":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}{unit}"
        n /= 1024
    return f"{n}"


def detect_format(path: Path) -> str:
    s = path.name.lower()
    # double extensions first
    if s.endswith(".tar.gz"):
        return "tar.gz"
    if s.endswith(".sql.gz"):
        return "sql.gz"
    if s.endswith(".fasta.gz") or s.endswith(".fa.gz"):
        return "fasta.gz"
    if s.endswith(".csv.gz"):
        return "csv.gz"
    if s.endswith(".tsv.gz"):
        return "tsv.gz"
    if s.endswith(".jsonl"):
        return "jsonl"
    suffix = path.suffix.lower().lstrip(".")
    aliases = {
        "fa": "fasta",
        "fas": "fasta",
        "fna": "fasta",
        "tgz": "tar.gz",
        "tar": "tar",
        "h5": "h5",
        "bw": "bw",
        "bedgraph": "bedgraph",
        "xls": "xlsx",
        "yml": "yaml",
        "yaml": "yaml",
    }
    return aliases.get(suffix, suffix or "noext")


def is_excluded(path: Path) -> bool:
    name = path.name
    if name in EXCLUDE_BASENAMES:
        return True
    if any(name.startswith(p) for p in EXCLUDE_NAME_PREFIXES):
        return True
    if any(name.endswith(s) for s in EXCLUDE_NAME_SUFFIXES):
        # special-case: keep .py only if it is data (none in our raw_data set)
        return True
    return False


def is_doc_artifact(path: Path) -> bool:
    return bool(DOC_NAME_REGEX.match(path.name))


def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        # Skip excluded directories anywhere on the path
        rel = p.relative_to(root)
        if any(part in EXCLUDE_DIR_NAMES for part in rel.parts):
            continue
        if not p.is_file():
            continue
        yield p


def wc_l(path: Path) -> int:
    """Run `wc -l` on a file and return the line count.

    Note: `wc -l` counts newline characters. If the last line lacks a trailing
    newline, the final record is missed; we detect and add 1 in that case. For
    speed we tail the last KiB and check if the final byte is `\\n`.
    """
    out = subprocess.check_output(["wc", "-l", str(path)], text=True)
    n = int(out.split()[0])
    # Adjust for missing trailing newline
    try:
        sz = path.stat().st_size
        if sz == 0:
            return 0
        with path.open("rb") as f:
            f.seek(-1, os.SEEK_END)
            last = f.read(1)
        if last != b"\n":
            n += 1
    except Exception:
        pass
    return n


def count_fasta(path: Path) -> int:
    """Count `^>` lines using grep -c (fast, streaming)."""
    try:
        out = subprocess.check_output(["grep", "-c", "^>", str(path)], text=True)
        return int(out.strip())
    except subprocess.CalledProcessError as e:
        # grep returns 1 with no matches
        if e.returncode == 1:
            return 0
        raise


def count_jsonl(path: Path) -> int:
    return wc_l(path)


def count_json(path: Path) -> tuple[int | None, str]:
    sz = path.stat().st_size
    if sz > JSON_INSPECT_LIMIT_BYTES:
        return None, f"json too large ({human_size(sz)}); not counted"
    try:
        with path.open() as f:
            data = json.load(f)
    except Exception as e:
        return None, f"json parse error: {type(e).__name__}"
    if isinstance(data, list):
        return len(data), "len(json.load) — top-level list"
    if isinstance(data, dict):
        return 1, "json object — counted as 1 record"
    return 1, "json scalar — counted as 1 record"


def count_xlsx(path: Path) -> tuple[int | None, str, list[dict]]:
    """Returns (total, method, per_sheet_breakdown)."""
    try:
        import pandas as pd  # local import keeps startup fast
        # Use openpyxl read-only mode for memory efficiency on large workbooks
        from openpyxl import load_workbook

        wb = load_workbook(filename=str(path), read_only=True, data_only=True)
        per_sheet = []
        total = 0
        for ws in wb.worksheets:
            # ws.max_row counts header + data rows
            mr = ws.max_row or 0
            rows = max(0, mr - 1)
            per_sheet.append({"sheet": ws.title, "rows": rows})
            total += rows
        wb.close()
        return total, "openpyxl read-only max_row minus 1 per sheet", per_sheet
    except Exception as e:
        return None, f"xlsx error: {type(e).__name__}: {e}", []


def count_parquet(path: Path) -> tuple[int | None, str]:
    try:
        import pyarrow.parquet as pq

        meta = pq.ParquetFile(path).metadata
        return meta.num_rows, "parquet metadata num_rows"
    except Exception as e:
        return None, f"parquet error: {type(e).__name__}"


def count_archive_members(path: Path, fmt: str) -> tuple[int | None, str, list[dict]]:
    """List members in zip/tar.gz/tgz archives.

    Returns (record_count, method, member_breakdown). record_count is the count
    of CSV/TSV/FASTA records inside if archive is small enough to introspect;
    otherwise None.
    """
    sz = path.stat().st_size
    if sz > ARCHIVE_INSPECT_LIMIT_BYTES:
        return None, f"archive >{human_size(ARCHIVE_INSPECT_LIMIT_BYTES)}; not extracted", []
    members: list[dict] = []
    record_total = 0
    have_data = False
    method_parts: list[str] = []
    try:
        if fmt == "zip":
            with zipfile.ZipFile(path) as z:
                for info in z.infolist():
                    if info.is_dir():
                        continue
                    members.append({"name": info.filename, "size": info.file_size})
                    inner_fmt = detect_format(Path(info.filename))
                    if inner_fmt in ("csv", "tsv", "txt"):
                        try:
                            with z.open(info) as f:
                                lines = sum(1 for _ in io.TextIOWrapper(f, errors="replace"))
                            if lines > 0:
                                record_total += max(0, lines - 1)
                                have_data = True
                        except Exception:
                            pass
                    elif inner_fmt == "fasta":
                        try:
                            with z.open(info) as f:
                                n = sum(1 for line in io.TextIOWrapper(f, errors="replace") if line.startswith(">"))
                            record_total += n
                            have_data = True
                        except Exception:
                            pass
            method_parts.append("zip member inspection")
        elif fmt in ("tar.gz", "tar"):
            mode = "r:gz" if fmt == "tar.gz" else "r:"
            with tarfile.open(path, mode) as t:
                for info in t:
                    if not info.isfile():
                        continue
                    members.append({"name": info.name, "size": info.size})
                    inner_fmt = detect_format(Path(info.name))
                    if inner_fmt in ("csv", "tsv", "txt"):
                        f = t.extractfile(info)
                        if f is not None:
                            try:
                                lines = sum(1 for _ in io.TextIOWrapper(f, errors="replace"))
                                if lines > 0:
                                    record_total += max(0, lines - 1)
                                    have_data = True
                            except Exception:
                                pass
                    elif inner_fmt == "fasta":
                        f = t.extractfile(info)
                        if f is not None:
                            try:
                                n = sum(1 for line in io.TextIOWrapper(f, errors="replace") if line.startswith(">"))
                                record_total += n
                                have_data = True
                            except Exception:
                                pass
            method_parts.append("tar member inspection")
        else:
            return None, f"unknown archive format {fmt}", []
    except Exception as e:
        return None, f"archive error: {type(e).__name__}", members
    if not have_data:
        return None, "; ".join(method_parts) + "; no tabular members", members
    return record_total, "; ".join(method_parts) + "; sum(lines-1) over csv/tsv/txt + fasta entries", members


def count_records(path: Path, fmt: str, *, source_name: str = "") -> tuple[int | None, str, str, str]:
    """Dispatch counting by format.

    Returns: (record_count, counting_method, record_definition, notes)
    """
    notes = ""
    try:
        if fmt in ("csv", "tsv"):
            n = wc_l(path)
            # OTS files have a JSON metadata first row plus header — record = n - 2
            if source_name == "OTS" and path.name.endswith("_Paired_All.csv"):
                rec = max(0, n - 2)
                return rec, "wc -l minus 2 (json metadata + header)", "paired-chain TCR sequence row", ""
            rec = max(0, n - 1)
            return rec, "wc -l minus 1 (header)", "tab/comma-delimited row excluding header", ""
        if fmt in ("csv.gz", "tsv.gz"):
            return None, "gz tabular not counted (would require streaming decompress)", "", "gz-compressed tabular; not counted"
        if fmt == "txt":
            # NetMHCpan-style headerless training files: c*_ba, c*_el, train_*.txt, test_*.txt
            base = path.name
            if path.parent.name == "NetMHCpan_train" and re.fullmatch(r"c\d{3}_(ba|el)", base):
                n = wc_l(path)
                return n, "wc -l (no header)", "peptide-MHC binding/eluted-ligand row", ""
            if path.parent.name == "NetMHCIIpan_train" and (
                base.startswith("train_") or base.startswith("test_")
            ):
                n = wc_l(path)
                return n, "wc -l (no header)", "peptide-MHC binding/eluted-ligand row", ""
            # IMGTHLA Allelelist.txt etc. — try header detection (check before vdjdb rule)
            if base in NON_TABULAR_TXT_BASENAMES:
                return None, "non-tabular metadata txt", "", "non-tabular plain text"
            # vdjdb_*.txt files in vdjdb/ are TSVs with headers
            if source_name == "vdjdb":
                n = wc_l(path)
                return max(0, n - 1), "wc -l minus 1 (header)", "tab-delimited row excluding header", ""
            # default: treat as line-delimited records (no header assumption)
            n = wc_l(path)
            return n, "wc -l (assumed no header)", "line", "txt with unknown schema; counted as lines"
        if fmt == "noext":
            # Headerless extension-less tabular files in NetMHCpan_train (c000_ba, c001_el, …)
            base = path.name
            if path.parent.name == "NetMHCpan_train" and re.fullmatch(r"c\d{3}_(ba|el)", base):
                n = wc_l(path)
                return n, "wc -l (no header)", "peptide-MHC binding/eluted-ligand row", ""
            return None, "binary/non-tabular; not counted", "", "noext: binary or non-tabular"
        if fmt == "fasta":
            n = count_fasta(path)
            return n, "grep -c '^>'", "FASTA entry (>header)", ""
        if fmt == "fasta.gz":
            return None, "fasta.gz not counted", "", "gz-compressed fasta; not counted"
        if fmt == "jsonl":
            n = count_jsonl(path)
            return n, "wc -l", "JSON line", ""
        if fmt == "json":
            cnt, method = count_json(path)
            return cnt, method, "json record (list element or document)", ""
        if fmt == "xlsx":
            cnt, method, per_sheet = count_xlsx(path)
            sheet_summary = "; ".join(f"{s['sheet']}={s['rows']}" for s in per_sheet)
            return cnt, method, "xlsx data row excluding header", sheet_summary
        if fmt == "parquet":
            cnt, method = count_parquet(path)
            return cnt, method, "parquet row", ""
        if fmt in ("zip", "tar.gz", "tar"):
            cnt, method, _members = count_archive_members(path, fmt)
            return cnt, method, "sum of records in archived csv/tsv/fasta members", ""
        if fmt == "tgz":
            cnt, method, _members = count_archive_members(path, "tar.gz")
            return cnt, method, "sum of records in archived csv/tsv/fasta members", ""
        if fmt in ("sql.gz", "sql"):
            return None, "sql dump not counted", "", "sql dump, not counted"
        if fmt in ("h5", "bw", "bedgraph", "pdf", "xsd", "yaml", "msf", "pir", "sample", "dat", "rdata", "r"):
            return None, "binary/non-tabular; not counted", "", f"{fmt}: binary or non-tabular"
        # Catch-all
        return None, f"unsupported format: {fmt}", "", f"unsupported format: {fmt}"
    except Exception as e:
        return None, f"counting error: {type(e).__name__}", "", str(e)


def t_interval(samples: list[float], confidence: float = 0.95) -> tuple[float, float] | None:
    if len(samples) < 2:
        return None
    mean = statistics.fmean(samples)
    sd = statistics.stdev(samples)
    n = len(samples)
    # 1.96 z-score for ~95% (good enough for n>=30); fall back to z if scipy missing
    z = 1.96 if confidence == 0.95 else 2.576
    margin = z * sd / math.sqrt(n)
    return (mean - margin, mean + margin)


# -----------------------------------------------------------------------------
# Standardizer cross-reference
# -----------------------------------------------------------------------------

# Map of source-name (as found in raw_data/databases/) -> standardizer file basename(s).
# A given source may have multiple standardizer modules (e.g., IEDB has iedb.py and
# iedb_pmhc.py, CEDAR has cedar.py and cedar_pmhc.py).
DB_TO_STANDARDIZERS: dict[str, list[str]] = {
    "vdjdb": ["vdjdb.py"],
    "McPAS-TCR": ["mcpas.py"],
    "IEDB": ["iedb.py", "iedb_pmhc.py"],
    "CEDAR": ["cedar.py", "cedar_pmhc.py"],
    "trait": ["trait.py"],
    "IMGTHLA": ["imgthla.py"],
    "NetMHCPan": ["netmhcpan.py"],
    "OTS": ["ots.py"],
    "adc": ["adc.py"],
    "immuneACCESS": ["immuneaccess.py"],
    "immuneCODE": ["immunecode.py"],
    "tcrdb": ["tcrdb.py"],
    "RCC_ATLAS": ["rcc_atlas.py"],
    "TaDB": ["tadb.py"],
    "BATMAN": ["batman.py"],
}


def parse_standardizer(file_path: Path) -> dict[str, Any] | None:
    if not file_path.exists():
        return None
    text = file_path.read_text(errors="replace")
    cls_match = re.search(r"^class\s+(\w+Standardizer)\(", text, re.MULTILINE)
    cls = cls_match.group(1) if cls_match else None
    name_match = re.search(r'^\s*name\s*=\s*"([^"]+)"', text, re.MULTILINE)
    inputs: list[str] = []
    # Collect explicit file paths under self.source_dir / "..."
    inputs.extend(
        re.findall(r'self\.source_dir\s*/\s*"([^"]+)"', text)
    )
    # Collect glob patterns inside source_dir.glob/rglob calls
    inputs.extend(re.findall(r'\.r?glob\(\s*"([^"]+)"\s*\)', text))
    # Filter unique while preserving order
    seen = set()
    inputs_unique: list[str] = []
    for x in inputs:
        if x not in seen:
            seen.add(x)
            inputs_unique.append(x)
    # Quick filter summary heuristic
    filt = []
    if "score" in text.lower() and ("vdjdb" in text.lower() or "score >=" in text or "score>=" in text):
        m = re.findall(r"score\s*[<>]=?\s*\d+", text)
        if m:
            filt.append("; ".join(set(m)))
    return {
        "path": str(file_path.relative_to(REPO_ROOT)),
        "class": cls,
        "name": name_match.group(1) if name_match else None,
        "expected_input_files": inputs_unique[:30],
        "filters_summary": "; ".join(filt) if filt else "",
    }


def standardizer_for(source_name: str) -> Any:
    files = DB_TO_STANDARDIZERS.get(source_name, [])
    out = []
    for fname in files:
        info = parse_standardizer(STD_DIR / fname)
        if info:
            out.append(info)
    if not out:
        return None
    if len(out) == 1:
        return out[0]
    return out


# -----------------------------------------------------------------------------
# Per-source inventory
# -----------------------------------------------------------------------------


def gather_files(root: Path) -> tuple[list[Path], list[Path], int, int]:
    """Walk `root` once. Return (data_files, non_data_artifacts, total_size_bytes,
    total_disk_files) where total_size_bytes/total_disk_files include EVERY file
    on disk (including those in excluded dirs like .git) so `du -sh` matches."""
    data: list[Path] = []
    non_data: list[Path] = []
    total_size = 0
    total_disk_files = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        total_disk_files += 1
        try:
            total_size += p.stat().st_size
        except OSError:
            pass
        # Determine if this file should be excluded from per-source counting:
        rel = p.relative_to(root)
        if any(part in EXCLUDE_DIR_NAMES for part in rel.parts):
            continue  # not even surfaced
        if is_excluded(p):
            non_data.append(p)
            continue
        if is_doc_artifact(p):
            non_data.append(p)
            continue
        data.append(p)
    return data, non_data, total_size, total_disk_files


def directory_size(root: Path) -> int:
    total = 0
    for p in iter_files(root):
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


def inventory_source(
    name: str,
    path: Path,
    kind: str,
    *,
    sample_n: int,
    exact: bool,
) -> SourceEntry:
    rel_path = str(path.relative_to(REPO_ROOT))
    src = SourceEntry(
        name=name,
        kind=kind,
        path=rel_path,
        size_bytes=0,
        size_human="",
        file_count=0,
        format_breakdown={},
        primary_data_files=[],
        non_data_artifacts=[],
        standardizer=standardizer_for(name) if kind == "database" else None,
    )

    if not path.exists():
        src.notes = "directory does not exist"
        return src

    data_files, non_data_files, total_size, total_disk_files = gather_files(path)
    fmt_counts: dict[str, int] = {}
    src.size_bytes = total_size
    src.size_human = human_size(total_size)
    src.file_count = total_disk_files

    # Format breakdown across ALL files (data + non-data) for visibility
    for p in data_files + non_data_files:
        f = detect_format(p)
        fmt_counts[f] = fmt_counts.get(f, 0) + 1
    src.format_breakdown = dict(sorted(fmt_counts.items(), key=lambda kv: -kv[1]))

    src.non_data_artifacts = sorted(
        str(p.relative_to(path)) for p in non_data_files
    )[:200]  # cap to avoid huge lists

    # Group data files by format for sampling decisions
    by_fmt: dict[str, list[Path]] = {}
    for p in data_files:
        f = detect_format(p)
        by_fmt.setdefault(f, []).append(p)

    primary_entries: list[FileEntry] = []
    total_records: int | None = 0
    sampled_used = False
    sample_method_strs: list[str] = []
    overall_definitions: set[str] = set()

    for fmt, paths in by_fmt.items():
        # Skip non-data formats outright (they shouldn't be in data_files but
        # double-defensive)
        if fmt in ("md", "yaml", "log", "py", "sh"):
            continue

        # Decide sampling: only sample huge sources with >100 files of one fmt.
        do_sample = (
            (not exact)
            and (name in HUGE_SOURCES)
            and (fmt in ("csv", "tsv", "txt"))
            and (len(paths) > 100)
        )

        if do_sample:
            sampled_used = True
            # Random sample (size-blind, seed=42 for reproducibility). Top-K-by-size
            # biases the mean estimate upward when file size correlates with row
            # count, AND it dominates wall time. Files >2GB are excluded from the
            # sample but still counted toward total size; the mean is extrapolated
            # over ALL files in the format group. Notes call this out.
            rng = random.Random(42)
            sampleable = [p for p in paths if p.stat().st_size <= SAMPLE_SKIP_FILE_BYTES]
            n_pick = min(sample_n, len(sampleable))
            sample = rng.sample(sampleable, k=n_pick) if sampleable else []
            counts: list[float] = []
            sample_entries: list[FileEntry] = []
            for p in sample:
                cnt, method, definition, n_notes = count_records(p, fmt, source_name=name)
                if cnt is not None:
                    counts.append(float(cnt))
                    overall_definitions.add(definition)
                size = p.stat().st_size
                sample_entries.append(
                    FileEntry(
                        path=str(p.relative_to(path)),
                        format=fmt,
                        size_bytes=size,
                        record_count=cnt,
                        record_definition=definition,
                        counting_method=method,
                        notes=("sampled" + ("; " + n_notes if n_notes else "")),
                    )
                )
            primary_entries.extend(sample_entries)
            if counts:
                mean = statistics.fmean(counts)
                est = int(round(mean * len(paths)))
                ci = t_interval(counts)
                ci_total = (ci[0] * len(paths), ci[1] * len(paths)) if ci else None
                skipped_huge = len(paths) - len(sampleable)
                method_str = (
                    f"sampled ({len(sample)} of {len(paths)} {fmt} files, "
                    f"mean={mean:.1f} records/file, extrapolated"
                    + (f"; {skipped_huge} files >{human_size(SAMPLE_SKIP_FILE_BYTES)} excluded from sample"
                       if skipped_huge else "")
                    + ")"
                )
                sample_method_strs.append(method_str)
                # Add a synthetic summary entry for the un-sampled remainder
                primary_entries.append(
                    FileEntry(
                        path=f"<{len(paths) - len(sample)} more {fmt} files (not sampled)>",
                        format=fmt,
                        size_bytes=sum(p.stat().st_size for p in paths) - sum(
                            e.size_bytes for e in sample_entries
                        ),
                        record_count=est - sum(int(c) for c in counts),
                        record_definition=next(iter(overall_definitions), ""),
                        counting_method="extrapolated from sample",
                        notes=(
                            f"95% CI for total: "
                            f"[{ci_total[0]:.0f}, {ci_total[1]:.0f}]"
                            if ci_total
                            else "no CI (n<2)"
                        ),
                    )
                )
                if total_records is not None:
                    total_records += est
                if ci_total:
                    src.total_records_ci = list(ci_total)
            else:
                # All samples failed to count — mark None
                total_records = None
        else:
            # Count every file in this format
            for p in paths:
                cnt, method, definition, n_notes = count_records(p, fmt, source_name=name)
                if definition:
                    overall_definitions.add(definition)
                size = p.stat().st_size
                primary_entries.append(
                    FileEntry(
                        path=str(p.relative_to(path)),
                        format=fmt,
                        size_bytes=size,
                        record_count=cnt,
                        record_definition=definition,
                        counting_method=method,
                        notes=n_notes,
                    )
                )
                if cnt is None:
                    # This particular file didn't get a count — total stays additive
                    # over what we DO know.
                    pass
                else:
                    if total_records is not None:
                        total_records += cnt

    # ---- Dedup pass: suppress double-counting from archives & duplicate dumps
    # Four rules, applied in order. Each rule may set entry.record_count = None
    # and append a note. After dedup, total_records is recomputed from the
    # surviving (non-None) per-file record_counts.
    primary_entries = _apply_dedup_rules(name, path, primary_entries)

    # Recompute total_records from deduped entries (only if not sampled).
    if not sampled_used:
        recount: int | None = 0
        for fe in primary_entries:
            if fe.record_count is None:
                continue
            # Skip the synthetic "more files (not sampled)" rows
            if fe.path.startswith("<") and fe.path.endswith(">"):
                continue
            recount += fe.record_count
        total_records = recount

    src.primary_data_files = primary_entries
    src.total_records = total_records
    src.exact_record_count = not sampled_used and (total_records is not None)
    if sampled_used:
        src.total_records_method = "; ".join(sample_method_strs)
    elif total_records is not None:
        src.total_records_method = "exact: sum of per-file record_count (after dedup)"
    else:
        src.total_records_method = "not all files countable"
    src.record_definition = "; ".join(sorted(overall_definitions))[:500]

    return src


# Sources where multiple data files are alternate dumps of the same underlying
# records. Only files in the canonical allowlist contribute to total_records.
CANONICAL_DATA_FILES: dict[str, set[str]] = {
    # vdjdb ships ~9 files that are alternate views of the same data
    # (slim/full × scored/filtered + zip). vdjdb.txt is the superset (all
    # entries, paired and unpaired); vdjdb_full.txt is the paired view.
    # We keep both as canonical so callers can pick based on use case.
    "vdjdb": {"vdjdb.txt", "vdjdb_full.txt"},
}


def _apply_dedup_rules(
    name: str, path: Path, entries: list[FileEntry]
) -> list[FileEntry]:
    """Mutate entries to suppress double-counting from archives and redundant
    dumps. Sets record_count = None on redundant entries and appends a note.
    """
    archive_fmts = {"zip", "tar.gz", "tar", "tgz"}

    # ---- Rule A: archive with extracted sibling directory.
    # Examples: NetMHCpan_train.tar.gz + NetMHCpan_train/,
    #           NetMHCIIpan_train.tar.gz + NetMHCIIpan_train/.
    for e in entries:
        if e.format not in archive_fmts or e.record_count is None:
            continue
        archive_full = path / e.path
        candidate_stems: list[str] = []
        nm = archive_full.name
        for ext in (".tar.gz", ".tar.bz2", ".tgz", ".tar", ".zip"):
            if nm.endswith(ext):
                candidate_stems.append(nm[: -len(ext)])
                break
        if not candidate_stems:
            candidate_stems.append(archive_full.stem)
        for stem in candidate_stems:
            sib = archive_full.parent / stem
            if sib.is_dir() and any(sib.rglob("*")):
                rec = e.record_count
                e.record_count = None
                note = f"archive duplicates extracted sibling dir `{stem}/`; not double-counted (was {rec:,})"
                e.notes = (e.notes + "; " if e.notes else "") + note
                break

    # ---- Rule B: archive whose stem matches a sibling loose data file.
    # Examples: trait `A0201_X_pos.zip` + `A0201_X_pos.txt` (same dir),
    #           CEDAR `tcell_full_v3.zip` + `tcell_full_v3.csv`,
    #           vdjdb `vdjdb-2025-12-29.zip` (bundles all the loose .txt files).
    on_disk_basenames: dict[str, str] = {}
    for e in entries:
        on_disk_basenames[Path(e.path).name] = e.path

    for e in entries:
        if e.format not in archive_fmts or e.record_count is None:
            continue
        archive_full = path / e.path
        nm = archive_full.name
        # strip archive extension
        stem = nm
        for ext in (".tar.gz", ".tar.bz2", ".tgz", ".tar", ".zip"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        # Look for any sibling file (in any subdir of this source) whose
        # basename overlaps with the archive's stem prefix.
        # Common patterns: stem == loose_basename_without_extension, OR
        # stem starts with same token (e.g., "vdjdb-2025-12-29" vs "vdjdb").
        matched_sibling = None
        for base, opath in on_disk_basenames.items():
            if opath == e.path:
                continue
            base_stem = Path(base).stem
            # exact stem match (CEDAR tcell_full_v3.zip vs tcell_full_v3.csv)
            if base_stem == stem:
                matched_sibling = base
                break
            # token-prefix match for versioned bundles (vdjdb-2025-12-29.zip vs vdjdb.txt)
            stem_token = re.split(r"[-_.]", stem, maxsplit=1)[0]
            if stem_token and stem_token == base_stem:
                matched_sibling = base
                break
        if matched_sibling:
            rec = e.record_count
            e.record_count = None
            note = (
                f"archive likely bundles loose sibling files (e.g. `{matched_sibling}`); "
                f"not double-counted (was {rec:,})"
            )
            e.notes = (e.notes + "; " if e.notes else "") + note

    # ---- Rule C: per-source canonical allowlist for redundant alternate dumps.
    canonical = CANONICAL_DATA_FILES.get(name)
    if canonical:
        for e in entries:
            base = Path(e.path).name
            if e.format in archive_fmts:
                continue
            if e.record_count is None:
                continue
            if base not in canonical:
                rec = e.record_count
                e.record_count = None
                note = (
                    f"redundant alternate dump for `{name}`; canonical: {sorted(canonical)} "
                    f"(was {rec:,})"
                )
                e.notes = (e.notes + "; " if e.notes else "") + note

    # ---- Rule D: same-basename + same-size dedup
    # (IMGTHLA hla_nuc.fasta exists at root and under fasta/ — byte-identical.)
    seen_key_to_path: dict[tuple[str, int], str] = {}
    for e in entries:
        if e.record_count is None:
            continue
        if e.format in archive_fmts:
            continue
        base = Path(e.path).name
        key = (base, e.size_bytes)
        if key in seen_key_to_path:
            rec = e.record_count
            e.record_count = None
            note = (
                f"duplicates `{seen_key_to_path[key]}` (same basename+size); "
                f"not double-counted (was {rec:,})"
            )
            e.notes = (e.notes + "; " if e.notes else "") + note
        else:
            seen_key_to_path[key] = e.path

    return entries


# -----------------------------------------------------------------------------
# Top-level orchestration
# -----------------------------------------------------------------------------


def discover_sources(root: Path) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]]]:
    db_root = root / "databases"
    st_root = root / "studies"
    dbs = []
    if db_root.exists():
        for d in sorted(db_root.iterdir()):
            if d.is_dir() and d.name not in EXCLUDE_DIR_NAMES:
                dbs.append((d.name, d))
    studies = []
    if st_root.exists():
        for d in sorted(st_root.iterdir()):
            if d.is_dir() and d.name not in EXCLUDE_DIR_NAMES:
                studies.append((d.name, d))
    return dbs, studies


def write_outputs(
    sources: list[SourceEntry],
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
            "sample": args.sample,
            "exact": bool(args.exact),
            "source_filter": args.source,
        },
        "sources": [asdict(s) for s in sources],
        "totals": {
            "source_count": len(sources),
            "total_bytes": sum(s.size_bytes for s in sources),
            "total_records": sum(
                (s.total_records or 0) for s in sources
            ),
            "exact_sources": sum(1 for s in sources if s.exact_record_count),
            "sampled_sources": sum(1 for s in sources if not s.exact_record_count),
        },
    }
    out_json.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s (%d sources)", out_json, len(sources))

    # Flat CSV: one row per source
    fields = [
        "name",
        "kind",
        "path",
        "size_bytes",
        "size_human",
        "file_count",
        "primary_data_file_count",
        "total_records",
        "exact_record_count",
        "total_records_method",
        "record_definition",
        "standardizer_path",
        "standardizer_class",
        "format_breakdown",
        "notes",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in sources:
            std = s.standardizer
            if isinstance(std, list):
                std_path = "; ".join(x.get("path", "") for x in std)
                std_cls = "; ".join(x.get("class") or "" for x in std)
            elif isinstance(std, dict):
                std_path = std.get("path", "")
                std_cls = std.get("class") or ""
            else:
                std_path = ""
                std_cls = ""
            w.writerow(
                {
                    "name": s.name,
                    "kind": s.kind,
                    "path": s.path,
                    "size_bytes": s.size_bytes,
                    "size_human": s.size_human,
                    "file_count": s.file_count,
                    "primary_data_file_count": len(s.primary_data_files),
                    "total_records": s.total_records if s.total_records is not None else "",
                    "exact_record_count": s.exact_record_count,
                    "total_records_method": s.total_records_method,
                    "record_definition": s.record_definition,
                    "standardizer_path": std_path,
                    "standardizer_class": std_cls,
                    "format_breakdown": json.dumps(s.format_breakdown),
                    "notes": s.notes,
                }
            )
    logger.info("Wrote %s", out_csv)


def render_markdown(s: SourceEntry) -> str:
    """Render a markdown block for one SourceEntry to embed under
    'Raw inventory' in the corresponding wiki page."""
    lines: list[str] = []
    lines.append("<!-- BEGIN: AUTO-INVENTORY (build_raw_data_manifest.py) -->")
    lines.append("")
    lines.append(f"- **Path**: `{s.path}`")
    lines.append(f"- **Size**: {s.size_human} ({s.size_bytes:,} bytes)")
    lines.append(f"- **Files**: {s.file_count}")
    if s.format_breakdown:
        fmt_pairs = ", ".join(f"`.{k}`={v}" for k, v in s.format_breakdown.items())
        lines.append(f"- **Format breakdown**: {fmt_pairs}")
    if s.total_records is not None:
        if s.exact_record_count:
            lines.append(f"- **Records**: {s.total_records:,} (exact)")
        else:
            ci = ""
            if s.total_records_ci:
                lo, hi = s.total_records_ci
                ci = f" (95% CI: {int(lo):,}–{int(hi):,})"
            lines.append(f"- **Records**: ~{s.total_records:,} (sampled){ci}")
        if s.total_records_method:
            lines.append(f"- **Counting method**: {s.total_records_method}")
        if s.record_definition:
            lines.append(f"- **Record definition**: {s.record_definition}")
    if isinstance(s.standardizer, dict):
        std = s.standardizer
        lines.append(
            f"- **Standardizer**: `{std.get('path','?')}` "
            f"(`{std.get('class','?')}`)"
        )
        if std.get("expected_input_files"):
            lines.append(
                "- **Expected input files**: "
                + ", ".join(f"`{x}`" for x in std["expected_input_files"][:8])
            )
    elif isinstance(s.standardizer, list):
        for std in s.standardizer:
            lines.append(
                f"- **Standardizer**: `{std.get('path','?')}` "
                f"(`{std.get('class','?')}`)"
            )
    if s.primary_data_files:
        lines.append("")
        lines.append("**Primary data files** (top 12 by size):")
        lines.append("")
        lines.append("| File | Format | Size | Records | Method |")
        lines.append("|------|--------|------|---------|--------|")
        sorted_files = sorted(
            s.primary_data_files, key=lambda f: -f.size_bytes
        )[:12]
        for fe in sorted_files:
            rc = f"{fe.record_count:,}" if fe.record_count is not None else "—"
            lines.append(
                f"| `{fe.path}` | {fe.format} | {human_size(fe.size_bytes)} "
                f"| {rc} | {fe.counting_method} |"
            )
    if s.non_data_artifacts:
        lines.append("")
        lines.append(
            "**Non-data artifacts**: "
            + ", ".join(f"`{x}`" for x in s.non_data_artifacts[:10])
            + (f" (+{len(s.non_data_artifacts)-10} more)" if len(s.non_data_artifacts) > 10 else "")
        )
    lines.append("")
    lines.append("<!-- END: AUTO-INVENTORY -->")
    return "\n".join(lines)


PLACEHOLDER_RE = re.compile(
    r"<!--\s*TODO:\s*filled by Quantifier agent from inventory\.json\s*-->",
    re.IGNORECASE,
)
AUTO_BLOCK_RE = re.compile(
    r"<!--\s*BEGIN:\s*AUTO-INVENTORY.*?<!--\s*END:\s*AUTO-INVENTORY\s*-->",
    re.DOTALL | re.IGNORECASE,
)


# Map raw_data/databases/ source names to the wiki .md filename stem
# (lit-review agents use lowercased canonical names).
DB_NAME_TO_WIKI_STEM: dict[str, str] = {
    "vdjdb": "vdjdb",
    "McPAS-TCR": "mcpas",
    "IEDB": "iedb",
    "CEDAR": "cedar",
    "trait": "trait",
    "IMGTHLA": "imgthla",
    "NetMHCPan": "netmhcpan",
    "OTS": "ots",
    "adc": "adc",
    "immuneACCESS": "immuneaccess",
    "immuneCODE": "immunecode",
    "tcrdb": "tcrdb",
    "RCC_ATLAS": "rcc_atlas",
    "TaDB": "tadb",
    "BATMAN": "batman",
}


def embed_inventory_into_wiki(inventory_json: Path, wiki_root: Path) -> dict:
    """Substitute the placeholder (or replace existing AUTO-INVENTORY block) in
    each docs/wiki/raw_data/{databases,studies}/{name}.md with rendered
    inventory. Idempotent."""
    payload = json.loads(inventory_json.read_text())
    by_kind_name: dict[tuple[str, str], dict] = {
        (s["kind"], s["name"]): s for s in payload["sources"]
    }
    updated: list[str] = []
    placeholders_left: list[str] = []
    missing_md: list[str] = []
    for (kind, name), s_dict in by_kind_name.items():
        sub = "databases" if kind == "database" else "studies"
        # Resolve wiki filename
        if kind == "database":
            stem = DB_NAME_TO_WIKI_STEM.get(name, name.lower())
        else:
            stem = name
        md_path = wiki_root / sub / f"{stem}.md"
        if not md_path.exists():
            # Try a couple of common alternates
            alts = [wiki_root / sub / f"{name}.md", wiki_root / sub / f"{name.lower()}.md"]
            md_path = next((p for p in alts if p.exists()), md_path)
        if not md_path.exists():
            missing_md.append(str(md_path))
            continue
        text = md_path.read_text()
        # Reconstruct SourceEntry-shaped dataclass for rendering
        s_obj = SourceEntry(**{
            **{k: v for k, v in s_dict.items() if k != "primary_data_files"},
            "primary_data_files": [FileEntry(**f) for f in s_dict.get("primary_data_files", [])],
        })
        block = render_markdown(s_obj)
        if AUTO_BLOCK_RE.search(text):
            new_text = AUTO_BLOCK_RE.sub(lambda _m: block, text)
        elif PLACEHOLDER_RE.search(text):
            new_text = PLACEHOLDER_RE.sub(lambda _m: block, text)
        else:
            placeholders_left.append(str(md_path))
            continue
        if new_text != text:
            md_path.write_text(new_text)
            updated.append(str(md_path))
    return {
        "updated": updated,
        "missing_md": missing_md,
        "no_placeholder_in_md": placeholders_left,
    }


# Pull the leading 1-5 from a "Confidence (1-5)" or "Confidence (1–5)" line.
FIDELITY_RE = re.compile(
    r"\*\*Confidence\s*\(1[‐-―\-]+5\)\*\*[:\s]*\*?\*?\s*(\d)",
    re.IGNORECASE,
)


def _extract_fidelity(md_text: str) -> str:
    m = FIDELITY_RE.search(md_text)
    return m.group(1) if m else "—"


MASTER_TABLE_RE = re.compile(
    r"<!--\s*BEGIN:\s*AUTO-MASTER-TABLE.*?<!--\s*END:\s*AUTO-MASTER-TABLE\s*-->",
    re.DOTALL | re.IGNORECASE,
)


def render_master_table(inventory_json: Path, wiki_root: Path) -> str:
    """Render a master summary table from inventory.json. Pulls fidelity from
    the per-source wiki page (lit-reviewed). Returns the table as a markdown
    block delimited by AUTO-MASTER-TABLE markers."""
    payload = json.loads(inventory_json.read_text())
    db_rows: list[str] = []
    study_rows: list[str] = []

    def _row_for(s: dict) -> tuple[str, str]:
        kind = s["kind"]
        name = s["name"]
        sub = "databases" if kind == "database" else "studies"
        stem = (DB_NAME_TO_WIKI_STEM.get(name, name.lower())
                if kind == "database" else name)
        md_path = wiki_root / sub / f"{stem}.md"
        fidelity = _extract_fidelity(md_path.read_text()) if md_path.exists() else "—"
        size = s.get("size_human") or "—"
        files = s.get("file_count") or 0
        records = s.get("total_records")
        if records is None:
            rec_str = "—"
        else:
            exact = s.get("exact_record_count")
            ci = s.get("total_records_ci")
            if exact:
                rec_str = f"{records:,}"
            elif ci:
                rec_str = f"~{records:,} (CI {int(ci[0]):,}–{int(ci[1]):,})"
            else:
                rec_str = f"~{records:,}"
        std = s.get("standardizer")
        if isinstance(std, list) and std:
            std_link = "; ".join(
                f"[`{Path(x.get('path','')).name}`]({'../' * 2}{x.get('path','')})"
                for x in std if x.get("path")
            )
        elif isinstance(std, dict) and std.get("path"):
            std_link = f"[`{Path(std['path']).name}`]({'../' * 2}{std['path']})"
        else:
            std_link = "—"
        link = f"[`{name}`]({sub}/{stem}.md)"
        return kind, (
            f"| {link} | {kind} | {rec_str} | {size} | {files} | {fidelity} | {std_link} |"
        )

    for s in payload["sources"]:
        kind, row = _row_for(s)
        if kind == "database":
            db_rows.append(row)
        else:
            study_rows.append(row)

    db_rows.sort()
    study_rows.sort()

    totals = payload.get("totals", {})
    total_bytes = totals.get("total_bytes", 0)
    total_records = totals.get("total_records", 0)
    total_sources = totals.get("source_count", 0)

    out: list[str] = []
    out.append("<!-- BEGIN: AUTO-MASTER-TABLE (build_raw_data_manifest.py) -->")
    out.append("")
    out.append(
        f"**Last regenerated**: {payload.get('generated_at', '?')} · "
        f"**Sources**: {total_sources} · "
        f"**Total bytes**: {human_size(total_bytes)} · "
        f"**Total records (sum across sources, deduped)**: {total_records:,}"
    )
    out.append("")
    out.append("### Databases")
    out.append("")
    out.append("| Source | Kind | Records | Size | Files | Fidelity | Standardizer |")
    out.append("|--------|------|---------|------|-------|----------|--------------|")
    out.extend(db_rows)
    out.append("")
    out.append("### Studies")
    out.append("")
    out.append("| Source | Kind | Records | Size | Files | Fidelity | Standardizer |")
    out.append("|--------|------|---------|------|-------|----------|--------------|")
    out.extend(study_rows)
    out.append("")
    out.append(
        "*Records column*: exact counts for fully-counted sources; "
        "`~N (CI lo–hi)` for sampled sources. `—` means the source has no "
        "row-shaped data or counting was not possible. *Fidelity* uses the "
        "[1–5 rubric](FIDELITY_RUBRIC.md). *Standardizer* links to the "
        "Python module that maps this source to the 25-column TARGET_COLUMNS "
        "schema; sources without a standardizer are not (yet) ingested into "
        "training datasets."
    )
    out.append("")
    out.append("<!-- END: AUTO-MASTER-TABLE -->")
    return "\n".join(out)


def embed_master_table(inventory_json: Path, wiki_root: Path) -> bool:
    """Replace the AUTO-MASTER-TABLE block in README.md with the rendered table."""
    readme = wiki_root / "README.md"
    if not readme.exists():
        return False
    text = readme.read_text()
    if not MASTER_TABLE_RE.search(text):
        return False
    block = render_master_table(inventory_json, wiki_root)
    new_text = MASTER_TABLE_RE.sub(lambda _m: block, text)
    if new_text != text:
        readme.write_text(new_text)
        return True
    return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_RAW)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--csv", type=Path, default=DEFAULT_OUT_CSV)
    p.add_argument("--sample", type=int, default=30,
                   help="Files to sample per huge source per format (default 30)")
    p.add_argument("--exact", action="store_true",
                   help="Disable sampling; count every file (slow on adc/immuneACCESS)")
    p.add_argument("--source", type=str, default=None,
                   help="Limit to one source name (database or study)")
    p.add_argument("--embed", action="store_true",
                   help="After inventorying, also embed results into wiki .md files")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    # Safety: --source NAME mode would overwrite the full inventory with a
    # 1-source file when default --out / --csv are used. Refuse, and require
    # the caller to specify alternate output paths explicitly.
    if args.source and args.out == DEFAULT_OUT_JSON:
        p.error(
            "--source NAME without --out would overwrite the full inventory.json. "
            "Pass explicit --out PATH and --csv PATH (e.g. --out /tmp/check.json "
            "--csv /tmp/check.csv) when scoping to a single source."
        )

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    started = time.time()
    dbs, studies = discover_sources(args.root)
    logger.info("Discovered %d databases, %d studies under %s",
                len(dbs), len(studies), args.root)

    pending: list[tuple[str, Path, str]] = []
    for n, p_ in dbs:
        if args.source and n != args.source:
            continue
        pending.append((n, p_, "database"))
    for n, p_ in studies:
        if args.source and n != args.source:
            continue
        pending.append((n, p_, "study"))

    try:
        from tqdm import tqdm
        bar = tqdm(pending, unit="source")
    except ImportError:
        bar = pending

    sources: list[SourceEntry] = []
    for name, path_, kind in bar:
        if hasattr(bar, "set_description"):
            bar.set_description(f"{kind}/{name}")
        else:
            logger.info("Inventorying %s/%s", kind, name)
        t0 = time.time()
        try:
            entry = inventory_source(
                name, path_, kind,
                sample_n=args.sample,
                exact=args.exact,
            )
        except Exception as e:
            logger.exception("Failed to inventory %s/%s: %s", kind, name, e)
            entry = SourceEntry(
                name=name, kind=kind, path=str(path_.relative_to(REPO_ROOT)),
                notes=f"INVENTORY ERROR: {type(e).__name__}: {e}",
            )
        dt = time.time() - t0
        logger.info("  %s/%s done in %.1fs (%d files, %s, %s records)",
                    kind, name, dt,
                    entry.file_count, entry.size_human,
                    f"{entry.total_records:,}" if entry.total_records is not None else "—")
        sources.append(entry)

    write_outputs(sources, args.out, args.csv, started_at=started, args=args)

    if args.embed:
        try:
            res = embed_inventory_into_wiki(args.out, args.out.parent)
            logger.info("Embedder: updated=%d, missing_md=%d, no_placeholder=%d",
                        len(res["updated"]), len(res["missing_md"]), len(res["no_placeholder_in_md"]))
        except Exception as e:
            logger.exception("Embedder failed: %s", e)
        try:
            updated = embed_master_table(args.out, args.out.parent)
            logger.info("Master table: %s", "updated" if updated else "no change")
        except Exception as e:
            logger.exception("Master table embedder failed: %s", e)

    logger.info("Total wall time: %.1fs", time.time() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
