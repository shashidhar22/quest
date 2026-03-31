"""Base class for per-database standardizers.

Each database standardizer inherits from BaseStandardizer and implements
get_column_map() and load_and_standardize() to produce parquet files in
the unified 25-column schema.
"""

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _process_file_worker(args: tuple) -> dict:
    """Worker function for parallel file processing.

    Runs in a separate process. Takes a tuple of (standardizer_cls, init_kwargs,
    file_path, worker_output_dir, worker_idx) and writes parquet files directly.
    Returns a summary dict with row/dropped counts and output file paths.
    """
    import warnings
    warnings.filterwarnings("ignore")

    cls, init_kwargs, file_path, worker_output_dir, worker_idx = args
    standardizer = cls(**init_kwargs)

    total_rows = 0
    total_dropped = 0
    output_files = []
    dropped_records = []
    file_idx = 0

    try:
        for chunk_df, dropped_df in standardizer.process_file(Path(file_path)):
            if chunk_df.empty:
                continue

            if dropped_df is not None and not dropped_df.empty:
                total_dropped += len(dropped_df)
                dropped_records.append(dropped_df)

            assert list(chunk_df.columns) == TARGET_COLUMNS

            out_file = Path(worker_output_dir) / f"w{worker_idx:04d}_{file_idx:04d}.parquet"
            table = pa.Table.from_pandas(chunk_df, preserve_index=False)
            pq.write_table(table, out_file, compression="snappy")
            output_files.append(str(out_file))
            file_idx += 1
            total_rows += len(chunk_df)
    except Exception as e:
        return {
            "rows": total_rows,
            "dropped": total_dropped,
            "files": output_files,
            "error": str(e),
            "source_file": str(file_path),
        }

    # Write dropped records
    dropped_path = None
    if dropped_records:
        dropped_df_all = pd.concat(dropped_records, ignore_index=True)
        dropped_path = str(Path(worker_output_dir) / f"dropped_w{worker_idx:04d}.tsv")
        dropped_df_all.to_csv(dropped_path, sep="\t", index=False)

    return {
        "rows": total_rows,
        "dropped": total_dropped,
        "files": output_files,
        "dropped_path": dropped_path,
    }


class BaseStandardizer(ABC):
    """Abstract base for per-database standardizers."""

    name: str = ""  # e.g., "batman"
    streaming: bool = False  # True for large datasets (>10M rows)

    def __init__(
        self,
        source_dir: str | Path,
        output_dir: str | Path | None = None,
        stitch: bool = False,
        hla_dir: str = "",
    ):
        self.source_dir = Path(source_dir)
        if output_dir is None:
            output_dir = (
                self.source_dir.parents[1] / "standardized" / self.name
            )
        self.output_dir = Path(output_dir)
        self.stitch = stitch
        self.hla_dir = hla_dir

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abstractmethod
    def get_column_map(self) -> dict:
        """Return {source_col: target_col} mapping."""

    @abstractmethod
    def load_and_standardize(self) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """Yield (standardized_df, dropped_df) tuples.

        standardized_df must have exactly the 25 TARGET_COLUMNS.
        dropped_df has columns: reason, source_file, row_index, field, raw_value.
        Small datasets can yield a single tuple.
        Large datasets should yield chunks of ~1M rows.
        """

    # ------------------------------------------------------------------
    # Parallel processing interface (optional override for large DBs)
    # ------------------------------------------------------------------
    def get_file_list(self) -> list[Path]:
        """Return list of source files to process in parallel.

        Override this in subclasses that support parallel processing.
        Returns empty list if parallel processing is not supported.
        """
        return []

    def process_file(self, file_path: Path) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """Process a single source file, yielding (standardized_df, dropped_df).

        Override this in subclasses that support parallel processing.
        Must be self-contained (no shared mutable state).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support parallel file processing"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def parallel_run(
        self,
        workers: int = 8,
        force: bool = False,
        progress: bool = True,
    ) -> dict:
        """Run standardization with parallel file processing.

        Uses ProcessPoolExecutor to process files in parallel. Each worker
        writes its own parquet files, which are then renamed into the final
        sequential naming scheme.

        Falls back to serial run() if get_file_list() returns empty.
        """
        file_list = self.get_file_list()
        if not file_list:
            return self.run(force=force, progress=progress)

        if not force and not self.is_stale():
            logger.info("%s: up-to-date (skipped)", self.name)
            return {"status": "skipped", "name": self.name}

        self.output_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()

        init_kwargs = {
            "source_dir": str(self.source_dir),
            "output_dir": str(self.output_dir),
            "stitch": self.stitch,
            "hla_dir": self.hla_dir,
        }

        # Build work items
        work_items = [
            (self.__class__, init_kwargs, str(fp), str(self.output_dir), idx)
            for idx, fp in enumerate(file_list)
        ]

        total_rows = 0
        total_dropped = 0
        all_output_files = []
        all_dropped_paths = []
        errors = []

        pbar = tqdm(
            total=len(work_items),
            desc=f"{self.name} (parallel)",
            unit=" files",
            disable=not progress,
            dynamic_ncols=True,
        )

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_process_file_worker, item): idx
                for idx, item in enumerate(work_items)
            }
            for future in as_completed(futures):
                result = future.result()
                total_rows += result.get("rows", 0)
                total_dropped += result.get("dropped", 0)
                all_output_files.extend(result.get("files", []))
                if result.get("dropped_path"):
                    all_dropped_paths.append(result["dropped_path"])
                if "error" in result:
                    errors.append(result)
                pbar.update(1)
                pbar.set_postfix(
                    rows=f"{total_rows:,}",
                    dropped=f"{total_dropped:,}",
                    refresh=False,
                )

        pbar.close()

        # Rename worker output files to sequential part_NNNN.parquet
        all_output_files.sort()
        for new_idx, old_path in enumerate(all_output_files):
            old = Path(old_path)
            new = self.output_dir / f"part_{new_idx:04d}.parquet"
            if old != new:
                old.rename(new)

        # Merge dropped record files
        dropped_path = self.output_dir / "dropped.tsv"
        if all_dropped_paths:
            with open(dropped_path, "w") as out_fh:
                header_written = False
                for dp in sorted(all_dropped_paths):
                    dp_path = Path(dp)
                    with open(dp_path) as in_fh:
                        header = in_fh.readline()
                        if not header_written:
                            out_fh.write(header)
                            header_written = True
                        for line in in_fh:
                            out_fh.write(line)
                    dp_path.unlink()
        else:
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
            "files": len(all_output_files),
            "elapsed_s": round(elapsed, 1),
        }
        if errors:
            summary["errors"] = len(errors)
        logger.info(
            "%s: %s rows (%s dropped) in %.1fs [%d workers]",
            self.name,
            f"{total_rows:,}",
            f"{total_dropped:,}",
            elapsed,
            workers,
        )
        return summary

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

        # Buffer for streaming mode: accumulate Arrow tables and flush at ROWS_PER_FILE
        write_buffer: list[pa.Table] = []
        write_buffer_rows = 0

        # Keep dropped file handle open for duration to avoid repeated open/close
        dropped_fh = open(dropped_path, "w", newline="")
        try:
            for chunk_df, dropped_df in self.load_and_standardize():
                if chunk_df.empty:
                    continue

                if dropped_df is not None and not dropped_df.empty:
                    total_dropped += len(dropped_df)
                    dropped_df.to_csv(
                        dropped_fh,
                        sep="\t",
                        index=False,
                        header=not dropped_header_written,
                    )
                    dropped_fh.flush()
                    dropped_header_written = True

                # Validate schema
                assert list(chunk_df.columns) == TARGET_COLUMNS, (
                    f"Expected {TARGET_COLUMNS}, got {list(chunk_df.columns)}"
                )

                # Enforce string dtype on all non-numeric columns so Arrow
                # schemas stay consistent across heterogeneous source files.
                _str_cols = [c for c in chunk_df.columns if c not in ("binding", "score")]
                for _c in _str_cols:
                    if chunk_df[_c].dtype != object:
                        chunk_df[_c] = chunk_df[_c].astype(str).replace("nan", pd.NA)

                # Write parquet file(s)
                if self.streaming:
                    # Buffer chunks as Arrow tables and flush when enough rows
                    write_buffer.append(
                        pa.Table.from_pandas(chunk_df, preserve_index=False)
                    )
                    write_buffer_rows += len(chunk_df)

                    while write_buffer_rows >= ROWS_PER_FILE:
                        combined = pa.concat_tables(write_buffer, promote_options="default")
                        to_write = combined.slice(0, ROWS_PER_FILE)
                        remainder = combined.slice(ROWS_PER_FILE)

                        out_file = self.output_dir / f"part_{file_idx:04d}.parquet"
                        pq.write_table(to_write, out_file, compression="snappy")
                        file_idx += 1
                        total_rows += to_write.num_rows
                        pbar.update(to_write.num_rows)

                        write_buffer = [remainder] if remainder.num_rows > 0 else []
                        write_buffer_rows = remainder.num_rows
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
                combined = pa.concat_tables(write_buffer, promote_options="default")
                out_file = self.output_dir / f"part_{file_idx:04d}.parquet"
                pq.write_table(combined, out_file, compression="snappy")
                file_idx += 1
                total_rows += combined.num_rows
                pbar.update(combined.num_rows)

            pbar.close()

            # Write empty dropped file if none was written
            if not dropped_header_written:
                pd.DataFrame(
                    columns=["reason", "source_file", "row_index", "field", "raw_value"]
                ).to_csv(dropped_fh, sep="\t", index=False)
        finally:
            dropped_fh.close()

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
        """Post-run verification: row counts, null ratios, sample values, combo stats.

        Uses PyArrow compute to avoid pandas conversion overhead.
        """
        import pyarrow.compute as pc

        parquet_files = sorted(self.output_dir.glob("*.parquet"))
        if not parquet_files:
            return {"error": "no parquet files found"}

        total_rows = 0
        null_counts = {col: 0 for col in TARGET_COLUMNS}
        sample_values = {}
        combo_counts = {
            'tra_only': 0, 'trb_only': 0, 'tra_trb': 0,
            'peptide_only': 0, 'pep_mhcI': 0, 'pep_mhcII': 0,
            'tcr_pep_mhcI': 0, 'tcr_pep_mhcII': 0, 'tcr_peptide': 0,
            'other': 0,
        }
        unique_sets = {
            'tra': set(), 'trb': set(), 'peptide': set(), 'mhc_one': set(),
        }

        empty = pa.scalar("")

        for pf in parquet_files:
            table = pq.read_table(pf)
            n = table.num_rows
            total_rows += n

            # Null counts and sample values (Arrow compute, no pandas)
            for col in TARGET_COLUMNS:
                arr = table.column(col)
                is_empty = pc.equal(arr, empty)
                null_counts[col] += pc.sum(is_empty).as_py()
                if col not in sample_values:
                    not_empty = pc.invert(is_empty)
                    if pc.any(not_empty).as_py():
                        idx = pc.index(not_empty, True).as_py()
                        sample_values[col] = arr[idx].as_py()

            # Molecule combination counting (Arrow compute)
            has_tra = pc.not_equal(table.column('tra'), empty)
            has_trb = pc.not_equal(table.column('trb'), empty)
            has_tcr = pc.or_(has_tra, has_trb)
            has_pep = pc.not_equal(table.column('peptide'), empty)
            has_mhc_one = pc.not_equal(table.column('mhc_one'), empty)
            has_mhc_two = pc.not_equal(table.column('mhc_two'), empty)
            is_mhcI = pc.and_(has_mhc_one, pc.invert(has_mhc_two))
            is_mhcII = pc.and_(has_mhc_one, has_mhc_two)
            has_mhc = pc.or_(is_mhcI, is_mhcII)

            not_trb = pc.invert(has_trb)
            not_tra = pc.invert(has_tra)
            not_pep = pc.invert(has_pep)
            not_tcr = pc.invert(has_tcr)
            not_mhc = pc.invert(has_mhc)

            combo_counts['tra_only'] += pc.sum(pc.and_(has_tra, pc.and_(not_trb, not_pep))).as_py()
            combo_counts['trb_only'] += pc.sum(pc.and_(not_tra, pc.and_(has_trb, not_pep))).as_py()
            combo_counts['tra_trb'] += pc.sum(pc.and_(has_tra, pc.and_(has_trb, not_pep))).as_py()
            combo_counts['peptide_only'] += pc.sum(pc.and_(has_pep, pc.and_(not_mhc, not_tcr))).as_py()
            combo_counts['pep_mhcI'] += pc.sum(pc.and_(has_pep, pc.and_(is_mhcI, not_tcr))).as_py()
            combo_counts['pep_mhcII'] += pc.sum(pc.and_(has_pep, pc.and_(is_mhcII, not_tcr))).as_py()
            combo_counts['tcr_pep_mhcI'] += pc.sum(pc.and_(has_tcr, pc.and_(has_pep, is_mhcI))).as_py()
            combo_counts['tcr_pep_mhcII'] += pc.sum(pc.and_(has_tcr, pc.and_(has_pep, is_mhcII))).as_py()
            combo_counts['tcr_peptide'] += pc.sum(pc.and_(has_tcr, pc.and_(has_pep, not_mhc))).as_py()
            combo_counts['other'] += pc.sum(pc.and_(not_tcr, not_pep)).as_py()

            # Unique values per key field (Arrow unique, no pandas)
            for field in unique_sets:
                arr = table.column(field)
                non_empty = pc.filter(arr, pc.not_equal(arr, empty))
                unique_sets[field].update(pc.unique(non_empty).to_pylist())

        null_ratios = {
            col: round(null_counts[col] / total_rows, 4) if total_rows > 0 else 0
            for col in TARGET_COLUMNS
        }
        unique_counts = {field: len(vals) for field, vals in unique_sets.items()}

        return {
            "name": self.name,
            "total_rows": total_rows,
            "parquet_files": len(parquet_files),
            "null_ratios": null_ratios,
            "sample_values": sample_values,
            "combo_counts": combo_counts,
            "unique_counts": unique_counts,
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
