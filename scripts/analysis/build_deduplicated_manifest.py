"""Build a machine-readable inventory of the deduplicated output dir.

Walks `data/deduplicated_again/` and inventories:
  - top-level files: deduped_parquet/ (canonical post-dedup table),
    deduped.db (DuckDB intermediate; size only — not opened),
    mhc_pseudo_lookup.json
  - the 31 subset_keys under exploded_deduped/ and exploded_deduped_enriched/,
    each with N order_keys (= factorial of column count)

For each subset_key, computes:
  - combination_count from combination_counts.json (or null)
  - permutation_count from permutation_counts.json (= rows per order_key
    partition; the JSON name is a misnomer — it's the canonical row count
    for the subset, not the count after explosion)
  - per-order-key parquet stats: file_count, size_bytes, row_count_from_parquet
  - cross-checks:
      row_count_matches_permutation: sum of parquet rows in deduped exploded ==
        permutation_count * factorial(num_cols) (i.e. one rotation per order_key)
      row_count_matches_deduped: enriched row count == deduped row count
        (enrichment never adds/drops rows)

All counts come from `pyarrow.parquet.ParquetFile.metadata.num_rows` — we never
read parquet table data into memory.

Outputs:
  - docs/wiki/deduplicated_data/inventory.json
  - docs/wiki/deduplicated_data/inventory.csv

Run from quest repo root:
    python scripts/analysis/build_deduplicated_manifest.py --embed
    python scripts/analysis/build_deduplicated_manifest.py --subset tra_trb \
        --out /tmp/check.json --csv /tmp/check.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("dedup_manifest")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "data" / "deduplicated_again"
DEFAULT_OUT_JSON = REPO_ROOT / "docs" / "wiki" / "deduplicated_data" / "inventory.json"
DEFAULT_OUT_CSV = REPO_ROOT / "docs" / "wiki" / "deduplicated_data" / "inventory.csv"
DEFAULT_WIKI_ROOT = REPO_ROOT / "docs" / "wiki" / "deduplicated_data"

# Columns added by the enrichment step on top of the deduped exploded schema.
EXPECTED_ENRICHED_ADDED = {
    "subset_key",
    "order_key",
    "mhc_one_pocket",
    "mhc_one_contact",
    "mhc_one_pocket_contact",
    "mhc_two_pocket",
    "mhc_two_contact",
    "mhc_two_pocket_contact",
}


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclass
class PartitionStats:
    order_key_count: int
    file_count: int
    size_bytes: int
    row_count_from_parquet: int
    # True when the cross-check passes; False when it fails; None when expected
    # value is unavailable (e.g. permutation_counts.json missing the key).
    row_count_matches_expected: bool | None
    schema: list[str]
    # Per-order-key row counts so we can sanity-check that explosion preserves
    # row count across rotations (every order_key should have identical rows).
    per_order_key_rows: dict[str, int] = field(default_factory=dict)
    # When per-order-key rows differ from the canonical count, list the offenders.
    order_key_row_anomalies: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SubsetEntry:
    subset_key: str
    populated_columns: list[str]
    combination_count: int | None
    permutation_count: int | None
    order_keys: list[str]
    deduped: dict[str, Any] = field(default_factory=dict)
    enriched: dict[str, Any] = field(default_factory=dict)
    # When deduped rows != enriched rows (an actual bug), populated.
    deduped_vs_enriched_mismatch: bool | None = None
    # The set difference enriched.schema - deduped.schema; should equal
    # EXPECTED_ENRICHED_ADDED. We also record extras/missing in case it doesn't.
    schema_added: list[str] = field(default_factory=list)
    schema_added_unexpected: list[str] = field(default_factory=list)
    schema_added_missing: list[str] = field(default_factory=list)
    notes: str = ""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def human_size(n: int) -> str:
    """Match build_raw_data_manifest.py / build_standardized_manifest.py."""
    if n < 0:
        return f"{n}B"
    val: float = float(n)
    for unit in ("B", "K", "M", "G", "T", "P"):
        if val < 1024 or unit == "P":
            return f"{int(val)}{unit}" if unit == "B" else f"{val:.1f}{unit}"
        val /= 1024
    return f"{val}"


def parse_subset_columns(subset_key: str) -> list[str]:
    """Decompose a subset_key like 'tra_trb_peptide_mhc_one' into ordered cols.

    Treats 'mhc_one' and 'mhc_two' as single columns, not two underscore-
    separated tokens.
    """
    cols: list[str] = []
    parts = subset_key.split("_")
    i = 0
    while i < len(parts):
        if parts[i] == "mhc" and i + 1 < len(parts) and parts[i + 1] in ("one", "two"):
            cols.append(f"mhc_{parts[i + 1]}")
            i += 2
        else:
            cols.append(parts[i])
            i += 1
    return cols


def safe_load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None


# -----------------------------------------------------------------------------
# Top-level inventory
# -----------------------------------------------------------------------------


def inventory_top_level(root: Path) -> dict[str, Any]:
    """Inventory the loose files at the dedup root: deduped_parquet/, deduped.db,
    mhc_pseudo_lookup.json. Walks deduped_parquet/data_*.parquet and sums
    parquet metadata rows."""
    import pyarrow.parquet as pq

    out: dict[str, Any] = {}

    # ---- deduped_parquet/
    dpq = root / "deduped_parquet"
    if dpq.exists():
        files = sorted(dpq.glob("data_*.parquet"))
        size = sum(p.stat().st_size for p in files)
        rows = 0
        schema: list[str] = []
        for i, p in enumerate(files):
            try:
                pf = pq.ParquetFile(p)
                rows += pf.metadata.num_rows
                if i == 0:
                    schema = list(pf.schema_arrow.names)
            except Exception as e:
                logger.warning("deduped_parquet: failed to read %s: %s", p, e)
        out["deduped_parquet"] = {
            "path": str(dpq.relative_to(REPO_ROOT)),
            "file_count": len(files),
            "size_bytes": size,
            "size_human": human_size(size),
            "row_count": rows,
            "schema": schema,
        }
    else:
        out["deduped_parquet"] = None

    # ---- deduped.db (DuckDB intermediate — record size only)
    db = root / "deduped.db"
    if db.exists():
        out["deduped_db"] = {
            "path": str(db.relative_to(REPO_ROOT)),
            "size_bytes": db.stat().st_size,
            "size_human": human_size(db.stat().st_size),
        }
    else:
        out["deduped_db"] = None

    # ---- mhc_pseudo_lookup.json
    mhc = root / "mhc_pseudo_lookup.json"
    if mhc.exists():
        size = mhc.stat().st_size
        try:
            payload = json.loads(mhc.read_text())
        except Exception as e:
            logger.warning("Failed to parse mhc_pseudo_lookup.json: %s", e)
            payload = None
        # Format is `{class_i: {allele: pseudo}, class_ii: {allele: pseudo}}`.
        if isinstance(payload, dict):
            classes = {k: len(v) if isinstance(v, dict) else None for k, v in payload.items()}
            entry_count = sum(v for v in classes.values() if isinstance(v, int))
        else:
            classes = {}
            entry_count = None
        out["mhc_pseudo_lookup"] = {
            "path": str(mhc.relative_to(REPO_ROOT)),
            "size_bytes": size,
            "size_human": human_size(size),
            "entry_count": entry_count,
            "entries_by_class": classes,
        }
    else:
        out["mhc_pseudo_lookup"] = None

    return out


# -----------------------------------------------------------------------------
# Per-subset inventory
# -----------------------------------------------------------------------------


def _stats_for_subset_dir(
    subset_dir: Path,
    *,
    expected_per_order_key: int | None,
    expected_total: int | None,
) -> PartitionStats:
    """Walk subset_key=X/order_key=Y/data_*.parquet under `subset_dir`. Return
    aggregate stats and per-order-key counts for sanity-checking.
    """
    import pyarrow.parquet as pq

    order_dirs = sorted(d for d in subset_dir.iterdir() if d.is_dir() and d.name.startswith("order_key="))
    file_count = 0
    size_bytes = 0
    total_rows = 0
    per_ok: dict[str, int] = {}
    schema: list[str] = []
    schema_set = False
    for od in order_dirs:
        ok = od.name[len("order_key="):]
        rows_here = 0
        for pf_path in sorted(od.glob("data_*.parquet")):
            file_count += 1
            try:
                size_bytes += pf_path.stat().st_size
            except OSError as e:
                logger.warning("stat failed for %s: %s", pf_path, e)
            try:
                pf = pq.ParquetFile(pf_path)
                rows_here += pf.metadata.num_rows
                if not schema_set:
                    schema = list(pf.schema_arrow.names)
                    schema_set = True
            except Exception as e:
                logger.warning("parquet metadata read failed for %s: %s", pf_path, e)
        per_ok[ok] = rows_here
        total_rows += rows_here

    # Detect rotations whose row count differs from the canonical (the median is
    # robust against e.g. one short shard); flag any deviations as anomalies.
    anomalies: list[dict[str, Any]] = []
    if per_ok and expected_per_order_key is not None:
        for ok, rows in per_ok.items():
            if rows != expected_per_order_key:
                anomalies.append(
                    {"order_key": ok, "rows": rows, "expected": expected_per_order_key}
                )
    elif per_ok:
        # No expected value (e.g. permutation_counts missing); compare to mode.
        from collections import Counter

        c = Counter(per_ok.values())
        most_common, _ = c.most_common(1)[0]
        for ok, rows in per_ok.items():
            if rows != most_common:
                anomalies.append(
                    {"order_key": ok, "rows": rows, "expected": most_common}
                )

    matches: bool | None
    if expected_total is None:
        matches = None
    else:
        matches = total_rows == expected_total

    return PartitionStats(
        order_key_count=len(order_dirs),
        file_count=file_count,
        size_bytes=size_bytes,
        row_count_from_parquet=total_rows,
        row_count_matches_expected=matches,
        schema=schema,
        per_order_key_rows=per_ok,
        order_key_row_anomalies=anomalies,
    )


def inventory_subset(
    subset_key: str,
    exploded_root: Path,
    enriched_root: Path,
    combinations: dict[str, int],
    permutations: dict[str, int],
) -> SubsetEntry:
    populated_cols = parse_subset_columns(subset_key)
    n_cols = len(populated_cols)
    fact = math.factorial(n_cols)
    perm_count = permutations.get(subset_key)
    comb_count = combinations.get(subset_key)
    expected_total = perm_count * fact if perm_count is not None else None

    entry = SubsetEntry(
        subset_key=subset_key,
        populated_columns=populated_cols,
        combination_count=comb_count,
        permutation_count=perm_count,
        order_keys=[],
    )

    # Deduped exploded
    sub_dir = exploded_root / f"subset_key={subset_key}"
    if sub_dir.exists():
        ds = _stats_for_subset_dir(
            sub_dir,
            expected_per_order_key=perm_count,
            expected_total=expected_total,
        )
        entry.order_keys = sorted(ds.per_order_key_rows.keys())
        entry.deduped = asdict(ds)
    else:
        entry.notes = (entry.notes + " deduped subset dir missing;").strip()

    # Enriched exploded
    enr_dir = enriched_root / f"subset_key={subset_key}"
    if enr_dir.exists():
        es = _stats_for_subset_dir(
            enr_dir,
            expected_per_order_key=perm_count,
            expected_total=expected_total,
        )
        entry.enriched = asdict(es)
    else:
        entry.notes = (entry.notes + " enriched subset dir missing;").strip()

    # Cross-check deduped vs enriched row counts
    drows = entry.deduped.get("row_count_from_parquet") if entry.deduped else None
    erows = entry.enriched.get("row_count_from_parquet") if entry.enriched else None
    if drows is not None and erows is not None:
        entry.deduped_vs_enriched_mismatch = drows != erows

    # Schema diff
    d_schema = set(entry.deduped.get("schema") or [])
    e_schema = set(entry.enriched.get("schema") or [])
    if d_schema or e_schema:
        added = e_schema - d_schema
        entry.schema_added = sorted(added)
        entry.schema_added_unexpected = sorted(added - EXPECTED_ENRICHED_ADDED)
        entry.schema_added_missing = sorted(EXPECTED_ENRICHED_ADDED - added)

    return entry


# -----------------------------------------------------------------------------
# Markdown rendering / embedding
# -----------------------------------------------------------------------------


PLACEHOLDER_RE = re.compile(
    r"<!--\s*TODO:\s*filled by Quantifier.*?-->",
    re.IGNORECASE | re.DOTALL,
)
AUTO_BLOCK_RE = re.compile(
    r"<!--\s*BEGIN:\s*AUTO-INVENTORY.*?<!--\s*END:\s*AUTO-INVENTORY\s*-->",
    re.DOTALL | re.IGNORECASE,
)
MASTER_TABLE_RE = re.compile(
    r"<!--\s*BEGIN:\s*AUTO-MASTER-TABLE.*?<!--\s*END:\s*AUTO-MASTER-TABLE\s*-->",
    re.DOTALL | re.IGNORECASE,
)


def render_markdown(entry: SubsetEntry) -> str:
    """Render an AUTO-INVENTORY block for a single subset_key."""
    lines: list[str] = []
    lines.append("<!-- BEGIN: AUTO-INVENTORY (build_deduplicated_manifest.py) -->")
    lines.append("")

    cols = ", ".join(f"`{c}`" for c in entry.populated_columns)
    n = len(entry.populated_columns)
    fact = math.factorial(n)
    lines.append(f"- **Populated columns** ({n}): {cols}")

    if entry.combination_count is not None:
        lines.append(f"- **Combination count** (canonical, from `combination_counts.json`): {entry.combination_count:,}")
    else:
        lines.append("- **Combination count**: _not present in `combination_counts.json`_ (combination_counts only tracks 11 of the 31 subsets — those exposing all `populated_columns` in the same form as a unique key)")
    if entry.permutation_count is not None:
        lines.append(
            f"- **Permutation row count** (rows per order_key, from `permutation_counts.json`): "
            f"{entry.permutation_count:,}"
        )
        lines.append(
            f"- **Total exploded rows expected**: {entry.permutation_count:,} × {fact} ({n}! permutations of {n} columns) "
            f"= {entry.permutation_count * fact:,}"
        )
    else:
        lines.append("- **Permutation row count**: _missing_")

    # Order keys
    if entry.order_keys:
        if len(entry.order_keys) <= 8:
            okstr = ", ".join(f"`{ok}`" for ok in entry.order_keys)
        else:
            head = ", ".join(f"`{ok}`" for ok in entry.order_keys[:8])
            okstr = f"{head}, … ({len(entry.order_keys) - 8} more)"
        lines.append(f"- **Order keys** ({len(entry.order_keys)}): {okstr}")

    # Deduped block
    d = entry.deduped or {}
    if d:
        match = d.get("row_count_matches_expected")
        if match is True:
            mstr = " (matches expected)"
        elif match is False:
            mstr = (
                f" (MISMATCH; expected {(entry.permutation_count or 0) * fact:,} "
                f"= {entry.permutation_count or 0:,} × {fact})"
            )
        else:
            mstr = ""
        lines.append("")
        lines.append(
            f"**Deduped exploded** (`exploded_deduped/subset_key={entry.subset_key}/`):"
        )
        lines.append("")
        lines.append(
            f"- {d.get('order_key_count', 0)} order_keys · "
            f"{d.get('file_count', 0)} parquet files · "
            f"{human_size(d.get('size_bytes', 0))} "
            f"({d.get('size_bytes', 0):,} bytes) · "
            f"{int(d.get('row_count_from_parquet') or 0):,} rows{mstr}"
        )
        if d.get("schema"):
            lines.append(f"- Schema: {', '.join(f'`{c}`' for c in d['schema'])}")
        anom = d.get("order_key_row_anomalies") or []
        if anom:
            lines.append(
                f"- ⚠ Per-order-key row anomalies: {len(anom)} order_keys diverge from canonical row count"
            )

    # Enriched block
    e = entry.enriched or {}
    if e:
        match = e.get("row_count_matches_expected")
        if match is True:
            mstr = " (matches expected)"
        elif match is False:
            mstr = " (MISMATCH vs permutation expectation)"
        else:
            mstr = ""
        de_match = entry.deduped_vs_enriched_mismatch
        if de_match is False:
            de_str = " · enriched row count == deduped row count"
        elif de_match is True:
            de_str = " · ⚠ enriched row count != deduped row count"
        else:
            de_str = ""
        lines.append("")
        lines.append(
            f"**Enriched exploded** (`exploded_deduped_enriched/subset_key={entry.subset_key}/`):"
        )
        lines.append("")
        lines.append(
            f"- {e.get('order_key_count', 0)} order_keys · "
            f"{e.get('file_count', 0)} parquet files · "
            f"{human_size(e.get('size_bytes', 0))} "
            f"({e.get('size_bytes', 0):,} bytes) · "
            f"{int(e.get('row_count_from_parquet') or 0):,} rows{mstr}{de_str}"
        )
        if entry.schema_added:
            lines.append(
                "- Schema columns added vs deduped: "
                + ", ".join(f"`{c}`" for c in entry.schema_added)
            )
        if entry.schema_added_unexpected:
            lines.append(
                "- ⚠ Unexpected enriched columns: "
                + ", ".join(f"`{c}`" for c in entry.schema_added_unexpected)
            )
        if entry.schema_added_missing:
            lines.append(
                "- ⚠ Expected enriched columns missing: "
                + ", ".join(f"`{c}`" for c in entry.schema_added_missing)
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
    each docs/wiki/deduplicated_data/subsets/{subset_key}.md with rendered
    inventory. Idempotent.
    """
    payload = json.loads(inventory_json.read_text())
    subsets_dir = wiki_root / "subsets"
    updated: list[str] = []
    missing_md: list[str] = []
    no_placeholder: list[str] = []

    for entry_dict in payload.get("subsets", []):
        subset_key = entry_dict["subset_key"]
        md_path = subsets_dir / f"{subset_key}.md"
        if not md_path.exists():
            missing_md.append(str(md_path))
            continue

        # Re-hydrate from dict (guards against dataclass schema drift).
        entry = SubsetEntry(
            subset_key=entry_dict["subset_key"],
            populated_columns=entry_dict.get("populated_columns", []),
            combination_count=entry_dict.get("combination_count"),
            permutation_count=entry_dict.get("permutation_count"),
            order_keys=entry_dict.get("order_keys", []),
            deduped=entry_dict.get("deduped", {}) or {},
            enriched=entry_dict.get("enriched", {}) or {},
            deduped_vs_enriched_mismatch=entry_dict.get("deduped_vs_enriched_mismatch"),
            schema_added=entry_dict.get("schema_added", []),
            schema_added_unexpected=entry_dict.get("schema_added_unexpected", []),
            schema_added_missing=entry_dict.get("schema_added_missing", []),
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


def render_master_table(inventory_json: Path) -> str:
    """Render the master summary table for the README."""
    payload = json.loads(inventory_json.read_text())
    subsets = payload.get("subsets", [])
    totals = payload.get("totals", {})

    lines: list[str] = []
    lines.append("<!-- BEGIN: AUTO-MASTER-TABLE (build_deduplicated_manifest.py) -->")
    lines.append("")

    # Headline
    tl = payload.get("top_level", {}) or {}
    pq_block = tl.get("deduped_parquet") or {}
    db_block = tl.get("deduped_db") or {}
    mhc_block = tl.get("mhc_pseudo_lookup") or {}
    pipe = payload.get("pipeline_run") or {}

    lines.append(
        f"**Last regenerated**: {payload.get('generated_at', '?')} · "
        f"**Subsets**: {len(subsets)}"
    )
    lines.append("")
    lines.append("**Pipeline run**:")
    lines.append("")
    if pipe:
        lines.append(f"- Run timestamp: `{pipe.get('run_timestamp', '?')}`")
        lines.append(f"- Dedup elapsed: {pipe.get('dedup_elapsed_seconds', 0):,.1f}s · output rows: {int(pipe.get('dedup_rows_out') or 0):,}")
        lines.append(f"- Explosion elapsed: {pipe.get('explosion_elapsed_seconds', 0):,.1f}s · output rows: {int(pipe.get('explosion_rows_out') or 0):,}")
    lines.append("")
    lines.append("**Top-level artifacts**:")
    lines.append("")
    if pq_block:
        lines.append(
            f"- `deduped_parquet/`: {pq_block.get('file_count', 0)} files, "
            f"{pq_block.get('size_human', '?')} ({pq_block.get('size_bytes', 0):,} bytes), "
            f"{int(pq_block.get('row_count') or 0):,} rows"
        )
    if db_block:
        lines.append(
            f"- `deduped.db` (DuckDB intermediate): {db_block.get('size_human', '?')} "
            f"({db_block.get('size_bytes', 0):,} bytes)"
        )
    if mhc_block:
        ec = mhc_block.get("entry_count")
        ec_str = f"{ec:,}" if isinstance(ec, int) else "?"
        lines.append(
            f"- `mhc_pseudo_lookup.json`: {mhc_block.get('size_human', '?')}, "
            f"{ec_str} entries"
        )
    lines.append("")

    # Aggregate totals
    lines.append(
        f"**Totals across {len(subsets)} subsets**: "
        f"deduped exploded rows = {int(totals.get('total_deduped_rows') or 0):,} · "
        f"enriched exploded rows = {int(totals.get('total_enriched_rows') or 0):,} · "
        f"deduped exploded size = {human_size(int(totals.get('total_deduped_bytes') or 0))} · "
        f"enriched exploded size = {human_size(int(totals.get('total_enriched_bytes') or 0))}"
    )
    lines.append("")

    # Master subset table
    lines.append("| Subset | Populated cols | Order keys | Combination | Permutation rows | Total exploded rows | Deduped size | Enriched size |")
    lines.append("|--------|---------------:|-----------:|------------:|-----------------:|--------------------:|-------------:|--------------:|")
    rows: list[str] = []
    for s in subsets:
        sk = s["subset_key"]
        pc = s.get("populated_columns") or []
        oks = s.get("order_keys") or []
        cc = s.get("combination_count")
        pr = s.get("permutation_count")
        d = s.get("deduped") or {}
        e = s.get("enriched") or {}
        cc_str = f"{cc:,}" if cc is not None else "—"
        pr_str = f"{pr:,}" if pr is not None else "—"
        n = len(pc)
        fact = math.factorial(n) if n else 1
        total_rows = pr * fact if pr is not None else None
        tr_str = f"{total_rows:,}" if total_rows is not None else "—"
        d_size = human_size(int(d.get("size_bytes", 0)))
        e_size = human_size(int(e.get("size_bytes", 0)))
        rows.append(
            f"| [`{sk}`](subsets/{sk}.md) | {n} ({', '.join(pc)}) | {len(oks)} | "
            f"{cc_str} | {pr_str} | {tr_str} | {d_size} | {e_size} |"
        )
    rows.sort()
    lines.extend(rows)
    lines.append("")
    lines.append(
        "*Permutation rows* = canonical rows per order_key partition (= rows after dedup for that column subset). "
        "*Total exploded rows* = permutation rows × `n!` rotations. *Combination* counts come from "
        "`combination_counts.json` (only 11 of 31 subsets are tracked there — namely the ones whose populated "
        "columns include all of the chains that anchor a complete training record)."
    )
    lines.append("")

    # Surface bugs in the master block too so they aren't buried
    bugs = totals.get("buggy_subsets", []) or []
    if bugs:
        lines.append("**⚠ Anomalies detected**:")
        lines.append("")
        for b in bugs:
            lines.append(
                f"- `{b['subset_key']}`: {b['issue']}"
            )
        lines.append("")
    else:
        lines.append("_All cross-checks passed: every subset's exploded row count matches `permutation_count × n!`, "
                     "and every subset's enriched row count equals its deduped row count._")
        lines.append("")

    lines.append("<!-- END: AUTO-MASTER-TABLE -->")
    return "\n".join(lines)


def embed_master_table(inventory_json: Path, wiki_root: Path) -> bool:
    """Embed the master-table block into README.md.

    Replacement target priority (idempotent):
      1. existing AUTO-MASTER-TABLE block (re-render in place)
      2. Quantifier TODO placeholder (first-time embed)
    Either branch makes subsequent runs replace the AUTO-MASTER-TABLE block.
    """
    readme = wiki_root / "README.md"
    if not readme.exists():
        return False
    text = readme.read_text()
    block = render_master_table(inventory_json)
    if MASTER_TABLE_RE.search(text):
        new_text = MASTER_TABLE_RE.sub(lambda _m: block, text)
    elif PLACEHOLDER_RE.search(text):
        new_text = PLACEHOLDER_RE.sub(lambda _m: block, text)
    else:
        return False
    if new_text != text:
        readme.write_text(new_text)
        return True
    return False


# -----------------------------------------------------------------------------
# Output writers
# -----------------------------------------------------------------------------


def discover_subset_keys(exploded_root: Path, enriched_root: Path) -> list[str]:
    keys: set[str] = set()
    for r in (exploded_root, enriched_root):
        if not r.exists():
            continue
        for d in r.iterdir():
            if d.is_dir() and d.name.startswith("subset_key="):
                keys.add(d.name[len("subset_key="):])
    return sorted(keys)


def write_outputs(
    top_level: dict[str, Any],
    pipeline_run: dict[str, Any] | None,
    subsets: list[SubsetEntry],
    out_json: Path,
    out_csv: Path,
    *,
    started_at: float,
    args: argparse.Namespace,
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate totals
    total_deduped_rows = sum(int((s.deduped or {}).get("row_count_from_parquet") or 0) for s in subsets)
    total_enriched_rows = sum(int((s.enriched or {}).get("row_count_from_parquet") or 0) for s in subsets)
    total_deduped_bytes = sum(int((s.deduped or {}).get("size_bytes") or 0) for s in subsets)
    total_enriched_bytes = sum(int((s.enriched or {}).get("size_bytes") or 0) for s in subsets)
    total_deduped_files = sum(int((s.deduped or {}).get("file_count") or 0) for s in subsets)
    total_enriched_files = sum(int((s.enriched or {}).get("file_count") or 0) for s in subsets)

    buggy: list[dict[str, str]] = []
    for s in subsets:
        d_match = (s.deduped or {}).get("row_count_matches_expected")
        e_match = (s.enriched or {}).get("row_count_matches_expected")
        if d_match is False:
            buggy.append({
                "subset_key": s.subset_key,
                "issue": (
                    f"deduped exploded rows {int(s.deduped.get('row_count_from_parquet') or 0):,} != "
                    f"expected {(s.permutation_count or 0) * math.factorial(len(s.populated_columns)):,}"
                ),
            })
        if e_match is False:
            buggy.append({
                "subset_key": s.subset_key,
                "issue": (
                    f"enriched exploded rows {int(s.enriched.get('row_count_from_parquet') or 0):,} != "
                    f"expected (permutation_count × n!)"
                ),
            })
        if s.deduped_vs_enriched_mismatch is True:
            buggy.append({
                "subset_key": s.subset_key,
                "issue": (
                    f"enriched row count "
                    f"{int(s.enriched.get('row_count_from_parquet') or 0):,} != "
                    f"deduped row count {int(s.deduped.get('row_count_from_parquet') or 0):,}"
                ),
            })
        if s.schema_added_unexpected:
            buggy.append({
                "subset_key": s.subset_key,
                "issue": f"unexpected enriched cols: {', '.join(s.schema_added_unexpected)}",
            })
        if s.schema_added_missing:
            buggy.append({
                "subset_key": s.subset_key,
                "issue": f"missing expected enriched cols: {', '.join(s.schema_added_missing)}",
            })
        anom_d = (s.deduped or {}).get("order_key_row_anomalies") or []
        anom_e = (s.enriched or {}).get("order_key_row_anomalies") or []
        if anom_d:
            buggy.append({
                "subset_key": s.subset_key,
                "issue": f"deduped per-order-key row anomalies: {len(anom_d)} order_keys",
            })
        if anom_e:
            buggy.append({
                "subset_key": s.subset_key,
                "issue": f"enriched per-order-key row anomalies: {len(anom_e)} order_keys",
            })

    payload: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.time() - started_at, 1),
        "args": {
            "root": str(args.root),
            "subset_filter": args.subset,
        },
        "pipeline_run": pipeline_run,
        "top_level": top_level,
        "subsets": [asdict(s) for s in subsets],
        "totals": {
            "subset_count": len(subsets),
            "total_deduped_rows": total_deduped_rows,
            "total_enriched_rows": total_enriched_rows,
            "total_deduped_bytes": total_deduped_bytes,
            "total_enriched_bytes": total_enriched_bytes,
            "total_deduped_files": total_deduped_files,
            "total_enriched_files": total_enriched_files,
            "buggy_subsets": buggy,
        },
    }
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("Wrote %s (%d subsets)", out_json, len(subsets))

    # ---- CSV
    fields = [
        "subset_key",
        "populated_columns",
        "n_columns",
        "factorial",
        "combination_count",
        "permutation_count",
        "expected_total_exploded",
        "deduped_order_keys",
        "deduped_files",
        "deduped_size_bytes",
        "deduped_size_human",
        "deduped_rows",
        "deduped_rows_match_expected",
        "enriched_order_keys",
        "enriched_files",
        "enriched_size_bytes",
        "enriched_size_human",
        "enriched_rows",
        "enriched_rows_match_expected",
        "deduped_vs_enriched_match",
        "schema_added",
        "schema_added_unexpected",
        "schema_added_missing",
        "deduped_anomaly_order_keys",
        "enriched_anomaly_order_keys",
        "notes",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in subsets:
            n = len(s.populated_columns)
            fact = math.factorial(n) if n else 1
            d = s.deduped or {}
            e = s.enriched or {}
            anom_d = d.get("order_key_row_anomalies") or []
            anom_e = e.get("order_key_row_anomalies") or []
            de_match: str
            if s.deduped_vs_enriched_mismatch is None:
                de_match = ""
            else:
                de_match = "True" if s.deduped_vs_enriched_mismatch is False else "False"
            w.writerow({
                "subset_key": s.subset_key,
                "populated_columns": "|".join(s.populated_columns),
                "n_columns": n,
                "factorial": fact,
                "combination_count": s.combination_count if s.combination_count is not None else "",
                "permutation_count": s.permutation_count if s.permutation_count is not None else "",
                "expected_total_exploded": (
                    s.permutation_count * fact if s.permutation_count is not None else ""
                ),
                "deduped_order_keys": d.get("order_key_count", ""),
                "deduped_files": d.get("file_count", ""),
                "deduped_size_bytes": d.get("size_bytes", ""),
                "deduped_size_human": human_size(int(d.get("size_bytes") or 0)) if d else "",
                "deduped_rows": d.get("row_count_from_parquet", ""),
                "deduped_rows_match_expected": d.get("row_count_matches_expected", ""),
                "enriched_order_keys": e.get("order_key_count", ""),
                "enriched_files": e.get("file_count", ""),
                "enriched_size_bytes": e.get("size_bytes", ""),
                "enriched_size_human": human_size(int(e.get("size_bytes") or 0)) if e else "",
                "enriched_rows": e.get("row_count_from_parquet", ""),
                "enriched_rows_match_expected": e.get("row_count_matches_expected", ""),
                "deduped_vs_enriched_match": de_match,
                "schema_added": "|".join(s.schema_added),
                "schema_added_unexpected": "|".join(s.schema_added_unexpected),
                "schema_added_missing": "|".join(s.schema_added_missing),
                "deduped_anomaly_order_keys": len(anom_d),
                "enriched_anomaly_order_keys": len(anom_e),
                "notes": s.notes,
            })
    logger.info("Wrote %s", out_csv)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                   help="Root of deduplicated data (default: data/deduplicated_again/)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_JSON,
                   help="Output JSON path")
    p.add_argument("--csv", type=Path, default=DEFAULT_OUT_CSV,
                   help="Output CSV path")
    p.add_argument("--subset", type=str, default=None,
                   help="Limit to one subset_key (e.g. tra_trb). Requires explicit "
                        "--out AND --csv to avoid clobbering the full inventory.")
    p.add_argument("--embed", action="store_true",
                   help="After inventorying, embed results into per-subset markdown "
                        "files at docs/wiki/deduplicated_data/subsets/{subset_key}.md "
                        "and the master README.md table.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    # Mirrors C1/C3 footgun guard from the standardized-data script: --subset
    # NAME with default --out OR --csv would silently clobber the whole-file
    # inventory with a 1-subset view. Refuse unless BOTH are redirected.
    if args.subset and (args.out == DEFAULT_OUT_JSON or args.csv == DEFAULT_OUT_CSV):
        p.error(
            "--subset NAME without explicit --out AND --csv would overwrite the "
            "full inventory. Pass both, e.g. "
            "--out /tmp/check.json --csv /tmp/check.csv."
        )

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    started = time.time()
    root: Path = args.root
    if not root.exists():
        logger.error("Root %s does not exist", root)
        return 2

    # Top-level
    logger.info("Inventorying top-level files under %s", root)
    top = inventory_top_level(root)

    # Pipeline run log
    pipeline_log = safe_load_json(root / "pipeline_run_log.json")
    pipeline_run: dict[str, Any] | None = None
    if pipeline_log:
        steps = pipeline_log.get("steps", {}) or {}
        dedup = steps.get("dedup", {}) or {}
        explosion = steps.get("explosion", {}) or {}
        pipeline_run = {
            "run_timestamp": pipeline_log.get("run_timestamp"),
            "dedup_elapsed_seconds": dedup.get("elapsed_seconds"),
            "dedup_rows_out": dedup.get("rows_out"),
            "explosion_elapsed_seconds": explosion.get("elapsed_seconds"),
            "explosion_rows_out": explosion.get("rows_out"),
        }

    # Counts
    combinations = safe_load_json(root / "combination_counts.json") or {}
    permutations = safe_load_json(root / "permutation_counts.json") or {}

    exploded_root = root / "exploded_deduped"
    enriched_root = root / "exploded_deduped_enriched"

    keys = discover_subset_keys(exploded_root, enriched_root)
    if args.subset:
        keys = [k for k in keys if k == args.subset]
    logger.info("Discovered %d subset_keys", len(keys))

    try:
        from tqdm import tqdm
        bar = tqdm(keys, unit="subset")
    except ImportError:
        bar = keys

    subsets: list[SubsetEntry] = []
    for sk in bar:
        if hasattr(bar, "set_description"):
            bar.set_description(sk)
        else:
            logger.info("Inventorying %s", sk)
        t0 = time.time()
        try:
            entry = inventory_subset(
                sk, exploded_root, enriched_root, combinations, permutations
            )
        except Exception as e:
            logger.exception("Failed to inventory %s: %s", sk, e)
            entry = SubsetEntry(
                subset_key=sk,
                populated_columns=parse_subset_columns(sk),
                combination_count=combinations.get(sk),
                permutation_count=permutations.get(sk),
                order_keys=[],
                notes=f"INVENTORY ERROR: {type(e).__name__}: {e}",
            )
        dt = time.time() - t0
        d = entry.deduped or {}
        logger.info(
            "  %s done in %.1fs (order_keys=%d, dedup_rows=%s, dedup_files=%d)",
            sk, dt,
            d.get("order_key_count", 0),
            f"{int(d.get('row_count_from_parquet') or 0):,}",
            d.get("file_count", 0),
        )
        subsets.append(entry)

    write_outputs(top, pipeline_run, subsets, args.out, args.csv, started_at=started, args=args)

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
                logger.info("  missing_md (first 10): %s", res["missing_md"][:10])
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
