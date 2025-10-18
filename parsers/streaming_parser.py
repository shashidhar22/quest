"""
Streaming parser for efficiently processing large TCR datasets (2TB+).

Key optimizations:
1. Chunked reading (never load full files into memory)
2. Format detection from header only
3. Streaming write to Parquet partitions
4. Parallel processing across files
5. Memory-efficient standardization
"""

import os
import csv
import yaml
import logging
from pathlib import Path
from typing import Iterator, Dict, List, Tuple, Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

from .utils import standardize_sequence, standardize_mri

logger = logging.getLogger(__name__)


class StreamingParser:
    """
    Memory-efficient parser that processes files in chunks and writes directly to Parquet.
    
    Args:
        format_config: Path to YAML config defining column mappings for each format
        output_dir: Directory to write Parquet files
        chunk_size: Number of rows to process at once (default: 100,000)
        compression: Parquet compression (default: 'snappy')
        test_mode: If True, only process 10% of data
    """
    
    def __init__(
        self,
        format_config: str,
        output_dir: str,
        chunk_size: int = 100_000,
        compression: str = 'snappy',
        test_mode: bool = False
    ):
        self.format_config = format_config
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.compression = compression
        self.test_mode = test_mode
        
        # Create output directories
        self.mri_dir = self.output_dir / "mri"
        self.seq_dir = self.output_dir / "seq"
        self.mri_dir.mkdir(parents=True, exist_ok=True)
        self.seq_dir.mkdir(parents=True, exist_ok=True)
        
        # Load format definitions
        with open(self.format_config, 'r') as f:
            self.format_dict = yaml.safe_load(f)
    
    def detect_format(self, file_path: str) -> Tuple[str, str, str]:
        """
        Detect file format by reading only the header.
        
        Returns:
            (format_type, format_name, delimiter)
        """
        with open(file_path, 'r') as f:
            # Read first few lines to detect delimiter and format
            sample = f.read(20_000)
            
            # Detect delimiter
            if "\t" in sample[:1000]:
                delimiter = "\t"
            elif "," in sample[:1000]:
                delimiter = ","
            else:
                sniffer = csv.Sniffer()
                try:
                    dialect = sniffer.sniff(sample[:5000])
                    delimiter = dialect.delimiter
                except:
                    delimiter = "\t"
            
            # Parse header
            f.seek(0)
            if "MiTCRFullExportV1.1" in sample:
                f.readline()  # skip special first line
            
            header_line = f.readline().strip()
            columns = set([c.strip() for c in header_line.split(delimiter)])
        
        # Match against known formats
        for format_type in ['bulk', 'misc', 'paired']:
            if format_type not in self.format_dict:
                continue
            for format_name, format_columns in self.format_dict[format_type].items():
                if columns == set(format_columns):
                    return format_type, format_name, delimiter
        
        raise ValueError(f"Unknown format for {file_path}. Columns: {columns}")
    
    def parse_file_streaming(
        self,
        file_path: str,
        format_type: str,
        format_name: str,
        delimiter: str
    ) -> Tuple[int, int]:
        """
        Parse a single file in chunks, writing directly to Parquet.
        
        Returns:
            (mri_rows_written, seq_rows_written)
        """
        logger.info(f"Parsing {file_path} as {format_type}/{format_name}")
        
        # Extract metadata from path
        file_path_obj = Path(file_path)
        repertoire_id = file_path_obj.stem
        
        # Determine skiprows for special formats
        skiprows = 1 if "MiTCR" in format_name else 0
        
        # Setup Parquet writers
        mri_output = self.mri_dir / f"{repertoire_id}_mri.parquet"
        seq_output = self.seq_dir / f"{repertoire_id}_seq.parquet"
        
        mri_writer = None
        seq_writer = None
        mri_rows = 0
        seq_rows = 0
        
        try:
            # Process in chunks
            chunk_iter = pd.read_csv(
                file_path,
                sep=delimiter,
                dtype=str,
                na_filter=False,
                chunksize=self.chunk_size,
                skiprows=skiprows
            )
            
            for chunk_idx, chunk in enumerate(chunk_iter):
                # Test mode: only process first chunk (10%)
                if self.test_mode and chunk_idx > 0:
                    break
                
                # Convert chunk based on format
                mri_chunk, seq_chunk = self._convert_chunk(
                    chunk,
                    format_type,
                    format_name,
                    file_path
                )
                
                if mri_chunk.empty:
                    continue
                
                # Write MRI chunk
                mri_table = pa.Table.from_pandas(mri_chunk, preserve_index=False)
                if mri_writer is None:
                    mri_writer = pq.ParquetWriter(
                        mri_output,
                        mri_table.schema,
                        compression=self.compression
                    )
                mri_writer.write_table(mri_table)
                mri_rows += len(mri_chunk)
                
                # Write sequence chunk (deduplicated)
                if not seq_chunk.empty:
                    seq_table = pa.Table.from_pandas(seq_chunk, preserve_index=False)
                    if seq_writer is None:
                        seq_writer = pq.ParquetWriter(
                            seq_output,
                            seq_table.schema,
                            compression=self.compression
                        )
                    seq_writer.write_table(seq_table)
                    seq_rows += len(seq_chunk)
                
                # Log progress every 10 chunks
                if (chunk_idx + 1) % 10 == 0:
                    logger.info(f"  Processed {(chunk_idx + 1) * self.chunk_size:,} rows")
        
        finally:
            if mri_writer:
                mri_writer.close()
            if seq_writer:
                seq_writer.close()
        
        logger.info(f"  ✓ Wrote {mri_rows:,} MRI rows, {seq_rows:,} seq rows")
        return mri_rows, seq_rows
    
    def _convert_chunk(
        self,
        chunk: pd.DataFrame,
        format_type: str,
        format_name: str,
        file_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert a chunk from its native format to standardized schema.
        
        Returns:
            (mri_chunk, sequence_chunk)
        """
        # Get metadata from file path
        file_path_obj = Path(file_path)
        repertoire_id = file_path_obj.stem
        
        # Determine appropriate converter based on format
        if format_type == "bulk":
            return self._convert_bulk_chunk(chunk, format_name, file_path)
        elif format_type == "paired":
            return self._convert_paired_chunk(chunk, format_name, file_path)
        elif format_type == "misc":
            return self._convert_misc_chunk(chunk, format_name, file_path)
        else:
            raise ValueError(f"Unknown format type: {format_type}")
    
    def _convert_bulk_chunk(
        self,
        chunk: pd.DataFrame,
        format_name: str,
        file_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert bulk format chunk to standard schema."""
        
        # Extract path metadata
        path_obj = Path(file_path)
        repertoire_id = path_obj.stem
        parts = path_obj.parts
        
        # Typical structure: .../category/study_id/molecule_type/file_type/file.tsv
        try:
            study_id = parts[-4] if len(parts) >= 4 else "unknown"
            molecule_type = parts[-3] if len(parts) >= 3 else "unknown"
            category = parts[-5] if len(parts) >= 5 else "unknown"
        except:
            study_id = "unknown"
            molecule_type = "unknown"
            category = "unknown"
        
        # Detect chain type from first row
        if chunk.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Column mapping based on format (simplified - extend as needed)
        if format_name == "format_one":
            # Drop unnecessary columns
            chunk = chunk.drop(
                columns=['VJCombo', 'Copy', 'ntCDR3', 'NetInsertionLength'],
                errors='ignore'
            )
            
            # Detect chain from VGene column
            first_vgene = str(chunk.iloc[0].get('VGene', ''))
            if 'TRAV' in first_vgene:
                chunk = chunk.rename(columns={
                    'VGene': 'trav_gene',
                    'JGene': 'traj_gene',
                    'aaCDR3': 'tra'
                })
            elif 'TRBV' in first_vgene:
                chunk = chunk.rename(columns={
                    'VGene': 'trbv_gene',
                    'JGene': 'trbj_gene',
                    'aaCDR3': 'trb'
                })
        
        elif format_name == "format_two":
            # Handle format_two columns
            chunk = chunk.drop(
                columns=[
                    'Read count', 'Percentage', 'CDR3 nucleotide sequence',
                    'CDR3 nucleotide quality', 'Min quality'
                ],
                errors='ignore'
            )
            
            first_v = str(chunk.iloc[0].get('V alleles', ''))
            if 'TRAV' in first_v:
                chunk = chunk.rename(columns={
                    'V alleles': 'trav_gene',
                    'J alleles': 'traj_gene',
                    'D alleles': 'trad_gene',
                    'CDR3 amino acid sequence': 'tra'
                })
            elif 'TRBV' in first_v:
                chunk = chunk.rename(columns={
                    'V alleles': 'trbv_gene',
                    'J alleles': 'trbj_gene',
                    'D alleles': 'trbd_gene',
                    'CDR3 amino acid sequence': 'trb'
                })
        
        # Add metadata
        chunk['repertoire_id'] = repertoire_id
        chunk['study_id'] = study_id
        chunk['category'] = category
        chunk['molecule_type'] = molecule_type
        chunk['host_organism'] = 'human'
        chunk['source'] = 'bulk_survey'
        
        # Build sequence table (deduplicated)
        mri_table = standardize_mri(chunk)
        
        # Sequence table: keep only unique sequences
        seq_cols = ['source', 'tid', 'tra', 'trad_gene', 'traj_gene', 'trav_gene',
                    'trb', 'trbd_gene', 'trbj_gene', 'trbv_gene', 'peptide', 
                    'mhc_one', 'mhc_two']
        seq_table = chunk[['source'] + [c for c in seq_cols[1:] if c in chunk.columns]].copy()
        seq_table['source'] = f"bulk_survey_{format_name}"
        seq_table = standardize_sequence(seq_table).drop_duplicates()
        
        return mri_table, seq_table
    
    def _convert_paired_chunk(
        self,
        chunk: pd.DataFrame,
        format_name: str,
        file_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert single-cell paired format chunk to standard schema."""
        
        # This would need specific logic for contigs, clonotypes, etc.
        # Simplified version:
        path_obj = Path(file_path)
        repertoire_id = path_obj.stem
        
        # Add metadata
        chunk['repertoire_id'] = repertoire_id
        chunk['source'] = 'single_cell'
        chunk['host_organism'] = 'human'
        
        mri_table = standardize_mri(chunk)
        seq_table = standardize_sequence(chunk).drop_duplicates()
        
        return mri_table, seq_table
    
    def _convert_misc_chunk(
        self,
        chunk: pd.DataFrame,
        format_name: str,
        file_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert misc format chunk to standard schema."""
        
        path_obj = Path(file_path)
        repertoire_id = path_obj.stem
        
        # Add metadata
        chunk['repertoire_id'] = repertoire_id
        chunk['source'] = 'misc_format'
        chunk['host_organism'] = 'human'
        
        mri_table = standardize_mri(chunk)
        seq_table = standardize_sequence(chunk).drop_duplicates()
        
        return mri_table, seq_table
    
    def parse_directory(
        self,
        input_dir: str,
        pattern: str = "**/*.tsv",
        max_workers: int = 4
    ) -> Dict[str, int]:
        """
        Parse all files in a directory tree in parallel.
        
        Args:
            input_dir: Root directory to search
            pattern: Glob pattern for files (e.g., '**/*.tsv', '**/*.csv')
            max_workers: Number of parallel workers
        
        Returns:
            Dict with statistics
        """
        input_path = Path(input_dir)
        files = list(input_path.glob(pattern))
        
        logger.info(f"Found {len(files)} files matching {pattern}")
        
        total_mri = 0
        total_seq = 0
        processed = 0
        failed = []
        
        # Process files in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_file = {
                executor.submit(self._process_single_file, str(f)): f
                for f in files
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    mri_rows, seq_rows = future.result()
                    total_mri += mri_rows
                    total_seq += seq_rows
                    processed += 1
                    
                    if processed % 10 == 0:
                        logger.info(f"Progress: {processed}/{len(files)} files")
                
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    failed.append(str(file_path))
        
        logger.info(f"✓ Completed: {processed} files, {total_mri:,} MRI rows, {total_seq:,} seq rows")
        if failed:
            logger.warning(f"✗ Failed: {len(failed)} files")
        
        return {
            'processed': processed,
            'failed': len(failed),
            'total_mri_rows': total_mri,
            'total_seq_rows': total_seq,
            'failed_files': failed
        }
    
    def _process_single_file(self, file_path: str) -> Tuple[int, int]:
        """Process a single file (used by parallel executor)."""
        try:
            format_type, format_name, delimiter = self.detect_format(file_path)
            return self.parse_file_streaming(file_path, format_type, format_name, delimiter)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            raise


def merge_parquet_partitions(
    input_dir: Path,
    output_file: Path,
    compression: str = 'snappy'
) -> int:
    """
    Merge multiple Parquet partition files into a single file.
    Uses PyArrow for memory-efficient streaming merge.
    
    Args:
        input_dir: Directory containing .parquet files
        output_file: Output merged parquet file
        compression: Compression codec
    
    Returns:
        Total number of rows
    """
    parquet_files = list(input_dir.glob("*.parquet"))
    
    if not parquet_files:
        logger.warning(f"No parquet files found in {input_dir}")
        return 0
    
    logger.info(f"Merging {len(parquet_files)} parquet files -> {output_file}")
    
    # Read first file to get schema
    first_table = pq.read_table(parquet_files[0])
    schema = first_table.schema
    
    total_rows = 0
    
    with pq.ParquetWriter(output_file, schema, compression=compression) as writer:
        for pf in parquet_files:
            table = pq.read_table(pf)
            writer.write_table(table)
            total_rows += len(table)
            
            # Progress logging
            if (len(parquet_files)) % 100 == 0:
                logger.info(f"  Merged {total_rows:,} rows...")
    
    logger.info(f"✓ Merged {total_rows:,} total rows")
    return total_rows
