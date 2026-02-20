"""Base class for per-database standardizers.

Each database standardizer inherits from BaseStandardizer and implements
get_column_map() and load_and_standardize() to produce parquet files in
the unified 15-column schema.
"""

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from quest.data.standardization import TARGET_COLUMNS

logger = logging.getLogger(__name__)

# Maximum rows per output parquet file (for chunked output)
ROWS_PER_FILE = 1_000_000


class BaseStandardizer(ABC):
    """Abstract base for per-database standardizers."""

    name: str = ""  # e.g., "batman"
    streaming: bool = False  # True for large datasets (>10M rows)

    def __init__(
        self,
        source_dir: str | Path,
        output_dir: str | Path | None = None,
    ):
        self.source_dir = Path(source_dir)
        if output_dir is None:
            output_dir = (
                self.source_dir.parents[1] / "standardized" / self.name
            )
        self.output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abstractmethod
    def get_column_map(self) -> dict:
        """Return {source_col: target_col} mapping."""

    @abstractmethod
    def load_and_standardize(self) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """Yield (standardized_df, dropped_df) tuples.

        standardized_df must have exactly the 15 TARGET_COLUMNS.
        dropped_df has columns: reason, source_file, row_index, field, raw_value.
        Small datasets can yield a single tuple.
        Large datasets should yield chunks of ~1M rows.
        """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, force: bool = False, progress: bool = True) -> dict:
        """Main entry: load, standardize, write parquet.

        Args:
            force: Re-run even if source files haven't changed.
            progress: Show a tqdm progress bar for rows processed.

        Returns a summary dict with row_count, dropped_count, timing.
        """
        if not force and not self.is_stale():
            logger.info("%s: up-to-date (skipped)", self.name)
            return {"status": "skipped", "name": self.name}

        self.output_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        total_rows = 0
        total_dropped = 0
        file_idx = 0

        dropped_path = self.output_dir / "dropped.tsv"
        dropped_header_written = False

        pbar = tqdm(
            desc=self.name,
            unit=" rows",
            unit_scale=True,
            disable=not progress,
            dynamic_ncols=True,
        )

        # Buffer for streaming mode: accumulate chunks and flush at ROWS_PER_FILE
        write_buffer: list[pd.DataFrame] = []
        write_buffer_rows = 0

        for chunk_df, dropped_df in self.load_and_standardize():
            if chunk_df.empty:
                continue

            if dropped_df is not None and not dropped_df.empty:
                total_dropped += len(dropped_df)
                dropped_df.to_csv(
                    dropped_path,
                    sep="\t",
                    index=False,
                    mode="a",
                    header=not dropped_header_written,
                )
                dropped_header_written = True

            # Validate schema
            assert list(chunk_df.columns) == TARGET_COLUMNS, (
                f"Expected {TARGET_COLUMNS}, got {list(chunk_df.columns)}"
            )

            # Write parquet file(s)
            if self.streaming:
                # Buffer chunks and flush when we have enough rows
                write_buffer.append(chunk_df)
                write_buffer_rows += len(chunk_df)

                while write_buffer_rows >= ROWS_PER_FILE:
                    combined = pd.concat(write_buffer, ignore_index=True)
                    to_write = combined.iloc[:ROWS_PER_FILE]
                    remainder = combined.iloc[ROWS_PER_FILE:]

                    out_file = self.output_dir / f"part_{file_idx:04d}.parquet"
                    table = pa.Table.from_pandas(to_write, preserve_index=False)
                    pq.write_table(table, out_file, compression="snappy")
                    file_idx += 1
                    total_rows += len(to_write)
                    pbar.update(len(to_write))

                    write_buffer = [remainder] if len(remainder) > 0 else []
                    write_buffer_rows = len(remainder)
            else:
                out_file = self.output_dir / f"part_{file_idx:04d}.parquet"
                table = pa.Table.from_pandas(chunk_df, preserve_index=False)
                pq.write_table(table, out_file, compression="snappy")
                file_idx += 1
                total_rows += len(chunk_df)
                pbar.update(len(chunk_df))

            pbar.set_postfix(
                dropped=f"{total_dropped:,}", files=file_idx, refresh=False
            )

        # Flush remaining buffered rows for streaming mode
        if self.streaming and write_buffer:
            combined = pd.concat(write_buffer, ignore_index=True)
            out_file = self.output_dir / f"part_{file_idx:04d}.parquet"
            table = pa.Table.from_pandas(combined, preserve_index=False)
            pq.write_table(table, out_file, compression="snappy")
            file_idx += 1
            total_rows += len(combined)
            pbar.update(len(combined))

        pbar.close()

        # Write empty dropped file if none was written
        if not dropped_header_written:
            pd.DataFrame(
                columns=["reason", "source_file", "row_index", "field", "raw_value"]
            ).to_csv(dropped_path, sep="\t", index=False)

        elapsed = time.time() - start
        checksums = self._compute_source_checksums()
        self._write_manifest(total_rows, total_dropped, checksums, elapsed)

        summary = {
            "status": "completed",
            "name": self.name,
            "rows": total_rows,
            "dropped": total_dropped,
            "files": file_idx,
            "elapsed_s": round(elapsed, 1),
        }
        logger.info(
            "%s: %s rows (%s dropped) in %.1fs",
            self.name,
            f"{total_rows:,}",
            f"{total_dropped:,}",
            elapsed,
        )
        return summary

    def is_stale(self) -> bool:
        """Check if source files changed since last run.

        Compares MD5 of source files against stored checksums in manifest.json.
        Returns True if any source file is new/modified/deleted, or manifest missing.
        """
        manifest_path = self.output_dir / "manifest.json"
        if not manifest_path.exists():
            return True
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            stored = manifest.get("source_checksums", {})
            current = self._compute_source_checksums()
            return stored != current
        except (json.JSONDecodeError, KeyError):
            return True

    def verify(self) -> dict:
        """Post-run verification: row counts, null ratios, sample values."""
        parquet_files = sorted(self.output_dir.glob("*.parquet"))
        if not parquet_files:
            return {"error": "no parquet files found"}

        total_rows = 0
        null_counts = {col: 0 for col in TARGET_COLUMNS}
        sample_values = {}

        for pf in parquet_files:
            table = pq.read_table(pf)
            total_rows += table.num_rows
            df = table.to_pandas()
            for col in TARGET_COLUMNS:
                null_counts[col] += (df[col] == "").sum()
                if col not in sample_values and not df[col].empty:
                    non_empty = df.loc[df[col] != "", col]
                    if not non_empty.empty:
                        sample_values[col] = non_empty.iloc[0]

        null_ratios = {
            col: round(null_counts[col] / total_rows, 4) if total_rows > 0 else 0
            for col in TARGET_COLUMNS
        }

        return {
            "name": self.name,
            "total_rows": total_rows,
            "parquet_files": len(parquet_files),
            "null_ratios": null_ratios,
            "sample_values": sample_values,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_source_checksums(self) -> dict:
        """Compute MD5 checksums for all source files."""
        checksums = {}
        if not self.source_dir.exists():
            return checksums
        for f in sorted(self.source_dir.rglob("*")):
            if f.is_file() and not f.name.startswith("."):
                checksums[str(f.relative_to(self.source_dir))] = _md5(f)
        return checksums

    def _write_manifest(
        self,
        row_count: int,
        dropped_count: int,
        checksums: dict,
        elapsed: float,
    ):
        """Write manifest.json with source checksums, row counts, timestamp."""
        import datetime

        manifest = {
            "name": self.name,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "row_count": row_count,
            "dropped_count": dropped_count,
            "elapsed_seconds": round(elapsed, 1),
            "source_checksums": checksums,
        }
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def _write_dropped(self, dropped_df: pd.DataFrame):
        """Write dropped records to TSV."""
        dropped_path = self.output_dir / "dropped.tsv"
        dropped_df.to_csv(dropped_path, sep="\t", index=False)

    def _get_source_files(self, pattern: str = "*") -> list[Path]:
        """Get list of source files matching pattern."""
        return sorted(self.source_dir.glob(pattern))


def _md5(filepath: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
