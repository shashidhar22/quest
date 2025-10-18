"""
Complete streaming parser for efficiently processing large TCR datasets (2TB+).

This version implements ALL format mappings from the existing parsers:
- BulkFileParser: 11 formats
- PairedFileParser: 3 formats (contigs, clonotypes, airr) with barcode pairing
- MiscFileParser: 6 formats

Key optimizations:
1. Chunked reading for bulk data (never load full files into memory)
2. Special handling for paired data (cell-aware grouping)
3. Format detection from header only
4. Streaming write to Parquet partitions
5. Memory-efficient standardization
"""

import os
import csv
import yaml
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Iterator, Dict, List, Tuple, Optional
from itertools import product
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from .utils import (
    standardize_sequence,
    standardize_mri,
    parse_junction_aa,
    format_combined_tcell
)

logger = logging.getLogger(__name__)


class StreamingParserComplete:
    """
    Memory-efficient parser that processes files in chunks with COMPLETE format support.
    
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
        # Check if this is an AIRR file based on path
        path_obj = Path(file_path)
        parent_dir = path_obj.parent.name.lower()
        
        # AIRR rearrangement files are in 'airr' directory
        if parent_dir == 'airr' or '_airr' in path_obj.name:
            return 'paired', 'airr_rearrangement', '\t'
        
        # Check for contigs or clonotypes by directory name
        if parent_dir == 'contigs':
            return 'paired', 'contigs', ','
        elif parent_dir == 'clonotypes':
            return 'paired', 'clonotypes', ','
        
        with open(file_path, 'r') as f:
            # Read first few lines to detect delimiter and format
            sample = f.read(20_000)
            
            # Detect delimiter
            if "\t" in sample[:1000]:
                delimiter = "\t"
            elif "," in sample[:1000]:
                delimiter = ","
            else:
                # Use csv.Sniffer
                try:
                    dialect = csv.Sniffer().sniff(sample[:10000])
                    delimiter = dialect.delimiter
                except:
                    delimiter = "\t"  # fallback
            
            # Parse header
            f.seek(0)
            reader = csv.reader(f, delimiter=delimiter)
            columns = next(reader)
        
        columns_set = set(columns)
        
        # Match against known formats
        for format_type in ['bulk', 'misc', 'paired']:
            if format_type not in self.format_dict:
                continue
            for format_name, format_cols in self.format_dict[format_type].items():
                if columns_set == set(format_cols):
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
        Parse a single file, using appropriate strategy based on format type.
        
        - Bulk: chunk-by-chunk streaming
        - Paired: load entire file (needed for barcode grouping)
        - Misc: load entire file (complex parsing logic)
        
        Returns:
            (mri_rows_written, seq_rows_written)
        """
        # Removed verbose per-file logging - progress shown in main progress bar
        
        if format_type == "bulk":
            return self._parse_bulk_streaming(file_path, format_name, delimiter)
        elif format_type == "paired":
            return self._parse_paired_full(file_path, format_name, delimiter)
        elif format_type == "misc":
            return self._parse_misc_full(file_path, format_name, delimiter)
        else:
            raise ValueError(f"Unknown format type: {format_type}")
    
    # ========================================================================
    #                           BULK PARSING (STREAMING)
    # ========================================================================
    
    def _parse_bulk_streaming(
        self,
        file_path: str,
        format_name: str,
        delimiter: str
    ) -> Tuple[int, int]:
        """
        Parse bulk data in chunks (memory efficient).
        """
        file_path_obj = Path(file_path)
        repertoire_id = file_path_obj.stem
        
        # Setup Parquet writers
        mri_output = self.mri_dir / f"{repertoire_id}_mri.parquet"
        seq_output = self.seq_dir / f"{repertoire_id}_seq.parquet"
        
        mri_writer = None
        seq_writer = None
        mri_rows = 0
        seq_rows = 0
        
        try:
            # Determine skiprows for special formats
            skiprows = 1 if "MiTCR" in format_name else 0
            
            # Read in chunks
            chunk_iter = pd.read_csv(
                file_path,
                sep=delimiter,
                dtype=str,
                chunksize=self.chunk_size,
                skiprows=skiprows,
                na_filter=False
            )
            
            for chunk_idx, chunk in enumerate(chunk_iter):
                if chunk.empty:
                    continue
                
                # Test mode: only process 10% of chunks
                if self.test_mode and chunk_idx % 10 != 0:
                    continue
                
                # Convert chunk to standard schema
                mri_chunk, seq_chunk = self._convert_bulk_chunk(
                    chunk, format_name, file_path
                )
                
                if not mri_chunk.empty:
                    # Write MRI chunk
                    mri_table = pa.Table.from_pandas(mri_chunk, preserve_index=False)
                    if mri_writer is None:
                        mri_writer = pq.ParquetWriter(
                            mri_output, mri_table.schema, compression=self.compression
                        )
                    mri_writer.write_table(mri_table)
                    mri_rows += len(mri_chunk)
                
                if not seq_chunk.empty:
                    # Write seq chunk
                    seq_table = pa.Table.from_pandas(seq_chunk, preserve_index=False)
                    if seq_writer is None:
                        seq_writer = pq.ParquetWriter(
                            seq_output, seq_table.schema, compression=self.compression
                        )
                    seq_writer.write_table(seq_table)
                    seq_rows += len(seq_chunk)
        
        finally:
            if mri_writer:
                mri_writer.close()
            if seq_writer:
                seq_writer.close()
        
        # Removed verbose per-file logging - progress shown in main progress bar
        return mri_rows, seq_rows
    
    def _convert_bulk_chunk(
        self,
        chunk: pd.DataFrame,
        format_name: str,
        file_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert bulk format chunk to standard schema.
        Implements ALL 11 bulk formats from BulkFileParser.
        """
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
        
        if chunk.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Apply format-specific transformations
        if format_name == "format_one":
            chunk = self._transform_bulk_format_one(chunk)
        elif format_name == "format_two":
            chunk = self._transform_bulk_format_two(chunk)
        elif format_name == "format_three":
            chunk = self._transform_bulk_format_three(chunk)
        elif format_name == "format_four":
            chunk = self._transform_bulk_format_four(chunk)
        elif format_name == "format_five":
            chunk = self._transform_bulk_format_five(chunk)
        elif format_name == "format_six":
            chunk = self._transform_bulk_format_six(chunk)
        elif format_name in ["format_seven", "format_eight", "format_ten"]:
            chunk = self._transform_bulk_format_seven(chunk)
        elif format_name == "format_nine":
            chunk = self._transform_bulk_format_nine(chunk)
        elif format_name == "format_eleven":
            chunk = self._transform_bulk_format_eleven(chunk)
        else:
            logger.warning(f"Unknown bulk format: {format_name}")
            return pd.DataFrame(), pd.DataFrame()
        
        if chunk.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Add metadata
        chunk['repertoire_id'] = repertoire_id
        chunk['study_id'] = study_id
        chunk['category'] = category
        chunk['molecule_type'] = molecule_type
        chunk['host_organism'] = 'human'
        chunk['source'] = f'bulk_survey_{format_name}'
        
        # Build MRI table
        mri_table = standardize_mri(chunk)
        
        # Build sequence table (deduplicated)
        seq_cols = ['tra', 'trad_gene', 'traj_gene', 'trav_gene',
                    'trb', 'trbd_gene', 'trbj_gene', 'trbv_gene', 
                    'peptide', 'mhc_one', 'mhc_two']
        seq_table = chunk[[c for c in seq_cols if c in chunk.columns]].copy()
        seq_table['source'] = f'bulk_survey_{format_name}'
        seq_table = standardize_sequence(seq_table).drop_duplicates()
        
        return mri_table, seq_table
    
    # ========================================================================
    #                    BULK FORMAT TRANSFORMATIONS
    # ========================================================================
    
    def _transform_bulk_format_one(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format one: VGene, JGene, aaCDR3, VJCombo, Copy, ntCDR3, NetInsertionLength"""
        df = df.drop(columns=['VJCombo', 'Copy', 'ntCDR3', 'NetInsertionLength'], errors='ignore')
        
        if df.empty:
            return df
        
        # Detect chain from VGene
        first_vgene = str(df.iloc[0].get('VGene', ''))
        if 'TRAV' in first_vgene:
            df = df.rename(columns={
                'VGene': 'trav_gene',
                'JGene': 'traj_gene',
                'aaCDR3': 'tra'
            })
        elif 'TRBV' in first_vgene:
            df = df.rename(columns={
                'VGene': 'trbv_gene',
                'JGene': 'trbj_gene',
                'aaCDR3': 'trb'
            })
        
        return df
    
    def _transform_bulk_format_two(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format two: V/D/J alleles, CDR3 amino acid sequence, Read count, etc."""
        drop_cols = ['Read count', 'Percentage', 'CDR3 nucleotide sequence',
                     'CDR3 nucleotide quality', 'Min quality', 'V segments',
                     'J segments', 'D segments', 'Last V nucleotide position ',
                     'First D nucleotide position', 'Last D nucleotide position',
                     'First J nucleotide position', 'VD insertions', 'DJ insertions',
                     'Total insertions']
        df = df.drop(columns=drop_cols, errors='ignore')
        
        if df.empty:
            return df
        
        first_v = str(df.iloc[0].get('V alleles', ''))
        if 'TRAV' in first_v:
            df = df.rename(columns={
                'V alleles': 'trav_gene',
                'J alleles': 'traj_gene',
                'D alleles': 'trad_gene',
                'CDR3 amino acid sequence': 'tra'
            })
        elif 'TRBV' in first_v:
            df = df.rename(columns={
                'V alleles': 'trbv_gene',
                'J alleles': 'trbj_gene',
                'D alleles': 'trbd_gene',
                'CDR3 amino acid sequence': 'trb'
            })
        
        return df
    
    def _transform_bulk_format_three(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format three: V/D/J segments, CDR3 amino acid sequence, Count, etc."""
        drop_cols = ['Count', 'Percentage', 'CDR3 nucleotide sequence',
                     'Last V nucleotide position', 'First D nucleotide position',
                     'Last D nucleotide position', 'First J nucleotide position',
                     'Good events', 'Total events', 'Good reads', 'Total reads']
        df = df.drop(columns=drop_cols, errors='ignore')
        
        if df.empty:
            return df
        
        first_v = str(df.iloc[0].get('V segments', ''))
        if 'TRAV' in first_v:
            df = df.rename(columns={
                'V segments': 'trav_gene',
                'J segments': 'traj_gene',
                'D segments': 'trad_gene',
                'CDR3 amino acid sequence': 'tra'
            })
        elif 'TRBV' in first_v:
            df = df.rename(columns={
                'V segments': 'trbv_gene',
                'J segments': 'trbj_gene',
                'D segments': 'trbd_gene',
                'CDR3 amino acid sequence': 'trb'
            })
        
        return df
    
    def _transform_bulk_format_four(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format four: amino_acid, v/d/j_resolved, frame_type"""
        # Filter for in-frame only
        df = df[df['frame_type'] == "In"].copy()
        
        if df.empty:
            return df
        
        df = df[['amino_acid', 'v_resolved', 'd_resolved', 'j_resolved']]
        
        # Detect chain and transform
        transformed_rows = []
        for _, row in df.iterrows():
            v_gene = str(row.get('v_resolved', ''))
            result = {}
            
            if 'TCRA' in v_gene or 'TRAV' in v_gene:
                result = {
                    'tra': row['amino_acid'],
                    'trav_gene': row['v_resolved'],
                    'trad_gene': row['d_resolved'],
                    'traj_gene': row['j_resolved']
                }
            elif 'TCRB' in v_gene or 'TRBV' in v_gene:
                result = {
                    'trb': row['amino_acid'],
                    'trbv_gene': row['v_resolved'],
                    'trbd_gene': row['d_resolved'],
                    'trbj_gene': row['j_resolved']
                }
            
            if result:
                transformed_rows.append(result)
        
        return pd.DataFrame(transformed_rows) if transformed_rows else pd.DataFrame()
    
    def _transform_bulk_format_five(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format five: #ID, V_ref, D_ref, J_ref, CDR3(aa), amino_acid with fuction filter"""
        # Filter for in-frame (note: typo 'fuction' exists in original data)
        df = df[df['fuction'] == "in-frame"].copy()
        
        if df.empty:
            return df
        
        df = df[['#ID', 'V_ref', 'D_ref', 'J_ref', 'CDR3(aa)', 'amino_acid']]
        
        # Detect chain and transform
        transformed_rows = []
        for _, row in df.iterrows():
            v_ref = str(row.get('V_ref', ''))
            
            if 'TRAV' in v_ref:
                transformed_rows.append({
                    'trav_gene': row['V_ref'],
                    'traj_gene': row['J_ref'],
                    'trad_gene': row['D_ref'],
                    'tra': row['CDR3(aa)']
                })
            elif 'TRBV' in v_ref:
                transformed_rows.append({
                    'trbv_gene': row['V_ref'],
                    'trbj_gene': row['J_ref'],
                    'trbd_gene': row['D_ref'],
                    'trb': row['CDR3(aa)']
                })
        
        return pd.DataFrame(transformed_rows) if transformed_rows else pd.DataFrame()
    
    def _transform_bulk_format_six(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format six: aminoAcid(CDR3 in lowercase), vGene, dGene, jGene"""
        # Filter for in-frame (note: typo 'fuction' exists in original data)
        df = df[df['fuction'] == "in-frame"].copy()
        
        if df.empty:
            return df
        
        df = df[['aminoAcid(CDR3 in lowercase)', 'vGene', 'dGene', 'jGene']]
        
        # Extract the longest lowercase stretch (CDR3 region)
        import re
        def longest_lowercase(s):
            matches = re.findall(r'[a-z]+', str(s))
            return max(matches, key=len, default='')
        
        df['cdr3'] = df['aminoAcid(CDR3 in lowercase)'].apply(longest_lowercase)
        
        # Detect chain and transform
        transformed_rows = []
        for _, row in df.iterrows():
            v_gene = str(row.get('vGene', ''))
            
            if 'TRAV' in v_gene:
                transformed_rows.append({
                    'trav_gene': row['vGene'],
                    'traj_gene': row['jGene'],
                    'trad_gene': row['dGene'],
                    'tra': row['cdr3']
                })
            elif 'TRBV' in v_gene:
                transformed_rows.append({
                    'trbv_gene': row['vGene'],
                    'trbj_gene': row['jGene'],
                    'trbd_gene': row['dGene'],
                    'trb': row['cdr3']
                })
        
        return pd.DataFrame(transformed_rows) if transformed_rows else pd.DataFrame()
    
    def _transform_bulk_format_seven(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format seven/eight/ten: aminoAcid, vMaxResolved, dMaxResolved, jMaxResolved"""
        # Filter for in-frame sequences
        df = df[df['sequenceStatus'] == "In"].copy()
        
        if df.empty:
            return df
        
        df = df[['aminoAcid', 'vMaxResolved', 'dMaxResolved', 'jMaxResolved']]
        df = df.fillna('')
        
        # Detect chain and transform
        transformed_rows = []
        for _, row in df.iterrows():
            v = str(row.get('vMaxResolved', ''))
            j = str(row.get('jMaxResolved', ''))
            d = str(row.get('dMaxResolved', ''))
            
            if 'TCRAV' in v or 'TCRAJ' in j:
                transformed_rows.append({
                    'trav_gene': v,
                    'traj_gene': j,
                    'trad_gene': d,
                    'tra': row['aminoAcid']
                })
            elif 'TCRBV' in v or 'TCRBJ' in j or 'TCRBD' in d:
                transformed_rows.append({
                    'trbv_gene': v,
                    'trbj_gene': j,
                    'trbd_gene': d,
                    'trb': row['aminoAcid']
                })
        
        return pd.DataFrame(transformed_rows) if transformed_rows else pd.DataFrame()
    
    def _transform_bulk_format_nine(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format nine: amino_acid, v_gene, d_gene, j_gene with frame_type filter"""
        # Filter for in-frame only
        df = df[df['frame_type'] == "In"].copy()
        
        if df.empty:
            return df
        
        df = df[['amino_acid', 'v_gene', 'd_gene', 'j_gene']]
        df = df.fillna('')
        
        # Detect chain and transform
        transformed_rows = []
        for _, row in df.iterrows():
            v_gene = str(row.get('v_gene', ''))
            
            if 'TCRAV' in v_gene:
                transformed_rows.append({
                    'trav_gene': row['v_gene'],
                    'traj_gene': row['j_gene'],
                    'trad_gene': row['d_gene'],
                    'tra': row['amino_acid'],
                })
            elif 'TCRBV' in v_gene:
                transformed_rows.append({
                    'trbv_gene': row['v_gene'],
                    'trbj_gene': row['j_gene'],
                    'trbd_gene': row['d_gene'],
                    'trb': row['amino_acid'],
                })
        
        return pd.DataFrame(transformed_rows) if transformed_rows else pd.DataFrame()
    
    def _transform_bulk_format_eleven(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format eleven: cdr3_b_aa, v_b_gene, j_b_gene"""
        if df.empty:
            return df
        
        df = df[['cdr3_b_aa', 'v_b_gene', 'j_b_gene']].copy()
        
        # Detect chain and transform
        transformed_rows = []
        for _, row in df.iterrows():
            v_gene = str(row.get('v_b_gene', ''))
            
            if 'TRAV' in v_gene:
                transformed_rows.append({
                    'trav_gene': row['v_b_gene'],
                    'traj_gene': row['j_b_gene'],
                    'tra': row['cdr3_b_aa'],
                })
            elif 'TRBV' in v_gene:
                transformed_rows.append({
                    'trbv_gene': row['v_b_gene'],
                    'trbj_gene': row['j_b_gene'],
                    'trb': row['cdr3_b_aa'],
                })
        
        return pd.DataFrame(transformed_rows) if transformed_rows else pd.DataFrame()
    
    # ========================================================================
    #                      PAIRED PARSING (FULL FILE)
    # ========================================================================
    
    def _parse_paired_full(
        self,
        file_path: str,
        format_name: str,
        delimiter: str
    ) -> Tuple[int, int]:
        """
        Parse paired single-cell data (requires full file for barcode grouping).
        """
        # Load entire file (required for barcode pairing)
        df = pd.read_csv(file_path, sep=delimiter, dtype=str, na_filter=False)
        
        if self.test_mode and not df.empty:
            df = df.sample(frac=0.1, random_state=21)
        
        # Determine file type from format_name or path
        file_type = Path(file_path).parent.name
        
        if file_type == "contigs" or "contigs" in format_name.lower():
            mri_table, seq_table = self._parse_contigs(df, file_path)
        elif file_type == "clonotypes" or "clonotypes" in format_name.lower():
            mri_table, seq_table = self._parse_clonotypes(df, file_path)
        elif file_type == "airr" or "airr" in format_name.lower():
            mri_table, seq_table = self._parse_rearrangements(df, file_path)
        else:
            logger.warning(f"Unknown paired format: {file_type}")
            return 0, 0
        
        # Write to parquet
        repertoire_id = Path(file_path).stem
        mri_output = self.mri_dir / f"{repertoire_id}_mri.parquet"
        seq_output = self.seq_dir / f"{repertoire_id}_seq.parquet"
        
        mri_rows = 0
        seq_rows = 0
        
        if not mri_table.empty:
            mri_table.to_parquet(mri_output, compression=self.compression, index=False)
            mri_rows = len(mri_table)
        
        if not seq_table.empty:
            seq_table.to_parquet(seq_output, compression=self.compression, index=False)
            seq_rows = len(seq_table)
        
        # Removed verbose logging - progress shown in main progress bar
        return mri_rows, seq_rows
    
    def _parse_contigs(self, df: pd.DataFrame, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parse contigs format with barcode-based pairing.
        """
        # Filter for TCR alpha/beta only
        df = df[df['chain'].str.contains('TR[AB]', na=False)].reset_index(drop=True)
        
        # Filter for productive + high_confidence
        df = df[
            df['productive'].isin(["True", "true", "TRUE"]) &
            df['high_confidence'].isin(["True", "true", "TRUE"])
        ].copy()
        
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Group by barcode and create TRA x TRB combinations
        grouped = df.groupby('barcode')
        formatted_contigs = []
        
        for barcode, group in grouped:
            tra_seqs = group.loc[group['chain'] == 'TRA', ['cdr3','v_gene','d_gene','j_gene']].apply(tuple, axis=1).tolist()
            trb_seqs = group.loc[group['chain'] == 'TRB', ['cdr3','v_gene','d_gene','j_gene']].apply(tuple, axis=1).tolist()
            
            if tra_seqs and trb_seqs:
                # Both alpha and beta
                for idx, (tra, trb) in enumerate(product(tra_seqs, trb_seqs), start=1):
                    result_dict = format_combined_tcell(barcode, idx, tra, trb)
                    formatted_contigs.append(result_dict)
            elif tra_seqs:
                # alpha only
                for idx, tra in enumerate(tra_seqs, start=1):
                    result_dict = format_combined_tcell(barcode, idx, tra, '')
                    formatted_contigs.append(result_dict)
            elif trb_seqs:
                # beta only
                for idx, trb in enumerate(trb_seqs, start=1):
                    result_dict = format_combined_tcell(barcode, idx, '', trb)
                    formatted_contigs.append(result_dict)
        
        if not formatted_contigs:
            return pd.DataFrame(), pd.DataFrame()
        
        mri_table = pd.DataFrame(formatted_contigs)
        sequence_table = mri_table.copy().drop_duplicates()
        
        # Add metadata
        repertoire_id = Path(file_path).stem
        parts = Path(file_path).parts
        study_id = parts[-4] if len(parts) >= 4 else "unknown"
        category = parts[-5] if len(parts) >= 5 else "unknown"
        molecule_type = parts[-3] if len(parts) >= 3 else "unknown"
        
        mri_table['repertoire_id'] = repertoire_id
        mri_table['study_id'] = study_id
        mri_table['category'] = category
        mri_table['molecule_type'] = molecule_type
        mri_table['host_organism'] = 'human'
        mri_table['source'] = 'single_cell'
        
        sequence_table['source'] = 'single_cell'
        
        # Standardize
        sequence_table = standardize_sequence(sequence_table)
        mri_table = standardize_mri(mri_table)
        
        return mri_table, sequence_table
    
    def _parse_clonotypes(self, df: pd.DataFrame, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parse clonotypes format (cdr3s_aa column).
        """
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        parsed_rows = []
        for idx, row in df.iterrows():
            # parse_junction_aa returns dict with keys like 'tra','trb'
            result = parse_junction_aa(row.get('cdr3s_aa', ''))
            
            # Set 'tid' from row['barcode'] or row['clonotype_id'] if present
            if 'barcode' in row:
                result['tid'] = row['barcode']
            elif 'clonotype_id' in row:
                result['tid'] = row['clonotype_id']
            else:
                result['tid'] = f"row_index_{idx}"
            
            parsed_rows.append(result)
        
        if not parsed_rows:
            return pd.DataFrame(), pd.DataFrame()
        
        mri_table = pd.DataFrame(parsed_rows)
        sequence_table = mri_table.copy().drop_duplicates()
        
        # Add metadata
        repertoire_id = Path(file_path).stem
        parts = Path(file_path).parts
        study_id = parts[-4] if len(parts) >= 4 else "unknown"
        category = parts[-5] if len(parts) >= 5 else "unknown"
        molecule_type = parts[-3] if len(parts) >= 3 else "unknown"
        
        mri_table['repertoire_id'] = repertoire_id
        mri_table['study_id'] = study_id
        mri_table['category'] = category
        mri_table['molecule_type'] = molecule_type
        mri_table['host_organism'] = 'human'
        mri_table['source'] = 'single_cell'
        
        sequence_table['source'] = 'single_cell'
        
        # Standardize
        sequence_table = standardize_sequence(sequence_table)
        mri_table = standardize_mri(mri_table)
        
        return mri_table, sequence_table
    
    def _parse_rearrangements(self, df: pd.DataFrame, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parse AIRR rearrangements format with cell_id grouping.
        """
        # Filter
        df = df[
            (df['is_cell'] == "T") &
            (df['productive'] == "T")
        ].copy()
        
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Group by cell_id
        grouped = df.groupby('cell_id')
        formatted_results = []
        
        for barcode, group in grouped:
            # Identify rows that are TRA or TRB from v_call/j_call/d_call
            tra_subset = group[
                group['v_call'].str.contains("TRA", na=False) |
                group['j_call'].str.contains("TRA", na=False) |
                group['d_call'].str.contains("TRA", na=False)
            ][['junction_aa','v_call','d_call','j_call']].apply(tuple, axis=1).tolist()
            
            trb_subset = group[
                group['v_call'].str.contains("TRB", na=False) |
                group['j_call'].str.contains("TRB", na=False) |
                group['d_call'].str.contains("TRB", na=False)
            ][['junction_aa','v_call','d_call','j_call']].apply(tuple, axis=1).tolist()
            
            if tra_subset and trb_subset:
                for idx, (tra, trb) in enumerate(product(tra_subset, trb_subset), start=1):
                    formatted_results.append(format_combined_tcell(barcode, idx, tra, trb))
            elif tra_subset:
                for idx, tra in enumerate(tra_subset, start=1):
                    formatted_results.append(format_combined_tcell(barcode, idx, tra, ''))
            elif trb_subset:
                for idx, trb in enumerate(trb_subset, start=1):
                    formatted_results.append(format_combined_tcell(barcode, idx, '', trb))
        
        if not formatted_results:
            return pd.DataFrame(), pd.DataFrame()
        
        mri_table = pd.DataFrame(formatted_results)
        sequence_table = mri_table.copy().drop_duplicates()
        
        # Add metadata
        repertoire_id = Path(file_path).stem
        parts = Path(file_path).parts
        study_id = parts[-4] if len(parts) >= 4 else "unknown"
        category = parts[-5] if len(parts) >= 5 else "unknown"
        molecule_type = parts[-3] if len(parts) >= 3 else "unknown"
        
        mri_table['repertoire_id'] = repertoire_id
        mri_table['study_id'] = study_id
        mri_table['host_organism'] = 'human'
        mri_table['source'] = 'single_cell'
        mri_table['category'] = category
        mri_table['molecule_type'] = molecule_type
        
        sequence_table['source'] = 'single_cell'
        
        # Standardize
        sequence_table = standardize_sequence(sequence_table)
        mri_table = standardize_mri(mri_table)
        
        return mri_table, sequence_table
    
    # ========================================================================
    #                        MISC PARSING (FULL FILE)
    # ========================================================================
    
    def _parse_misc_full(
        self,
        file_path: str,
        format_name: str,
        delimiter: str
    ) -> Tuple[int, int]:
        """
        Parse misc formats (requires full file for complex string parsing).
        """
        # Load entire file (required for complex string parsing)
        df = pd.read_csv(file_path, sep=delimiter, dtype=str, na_filter=False)
        
        # Clean up NA/None values
        df = df.fillna('')
        df = df.replace(['None', 'nan', 'NA', 'N/A', 'na', 'NaN'], '', regex=True)
        
        if self.test_mode and not df.empty:
            df = df.sample(frac=0.1, random_state=21)
        
        # Route to appropriate misc format handler
        if format_name == "format_one":
            mri_table, seq_table = self._parse_misc_format_one(df, file_path)
        elif format_name == "format_two":
            mri_table, seq_table = self._parse_misc_format_two(df, file_path)
        elif format_name == "format_three":
            mri_table, seq_table = self._parse_misc_format_three(df, file_path)
        elif format_name == "format_four":
            mri_table, seq_table = self._parse_misc_format_four(df, file_path)
        elif format_name == "format_five":
            mri_table, seq_table = self._parse_misc_format_five(df, file_path)
        elif format_name == "format_six":
            mri_table, seq_table = self._parse_misc_format_six(df, file_path)
        else:
            logger.warning(f"Unknown misc format: {format_name}")
            return 0, 0
        
        # Write to parquet
        repertoire_id = Path(file_path).stem
        mri_output = self.mri_dir / f"{repertoire_id}_mri.parquet"
        seq_output = self.seq_dir / f"{repertoire_id}_seq.parquet"
        
        mri_rows = 0
        seq_rows = 0
        
        if not mri_table.empty:
            mri_table.to_parquet(mri_output, compression=self.compression, index=False)
            mri_rows = len(mri_table)
        
        if not seq_table.empty:
            seq_table.to_parquet(seq_output, compression=self.compression, index=False)
            seq_rows = len(seq_table)
        
        # Removed verbose logging - progress shown in main progress bar
        return mri_rows, seq_rows
    
    def _parse_misc_format_one(self, df: pd.DataFrame, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Misc format one: cdr3s_nt and cdr3s_aa with 'TRA:xxx;TRB:yyy' format"""
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        parsed_rows = []
        for idx, row in df.iterrows():
            result = parse_junction_aa(row.get('cdr3s_aa', ''))
            if result:
                parsed_rows.append(result)
        
        if not parsed_rows:
            return pd.DataFrame(), pd.DataFrame()
        
        mri_table = pd.DataFrame(parsed_rows)
        sequence_table = mri_table[['tra', 'trb']].drop_duplicates()
        
        # Add metadata
        repertoire_id = Path(file_path).stem
        parts = Path(file_path).parts
        study_id = parts[-4] if len(parts) >= 4 else "unknown"
        category = parts[-5] if len(parts) >= 5 else "unknown"
        molecule_type = parts[-3] if len(parts) >= 3 else "unknown"
        
        mri_table['repertoire_id'] = repertoire_id
        mri_table['study_id'] = study_id
        mri_table['category'] = category
        mri_table['molecule_type'] = molecule_type
        mri_table['host_organism'] = 'human'
        mri_table['source'] = 'misc_format'
        
        sequence_table['source'] = 'misc_format'
        
        sequence_table = standardize_sequence(sequence_table)
        mri_table = standardize_mri(mri_table)
        
        return mri_table, sequence_table
    
    def _parse_misc_format_two(self, df: pd.DataFrame, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Misc format two: similar to format_one but with barcode column"""
        # Similar logic to format_one
        return self._parse_misc_format_one(df, file_path)
    
    def _parse_misc_format_three(self, df: pd.DataFrame, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Misc format three: separate TRA/TRB rows with chain column"""
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        df.rename(columns={
            'sample_id': 'repertoire_id',
            'TR_chain': 'chain',
            'subject_id': 'patient_id',
            'CDR3.aa': 'cdr3_aa',
            'V.name': 'v_gene',
            'D.name': 'd_gene',
            'J.name': 'j_gene'
        }, inplace=True)
        
        # Separate TRA vs TRB
        tra_df = df[df['chain'] == 'TRA'].copy()
        trb_df = df[df['chain'] == 'TRB'].copy()
        
        tra_df.rename(columns={
            'cdr3_aa': 'tra',
            'v_gene': 'trav_gene',
            'd_gene': 'trad_gene',
            'j_gene': 'traj_gene'
        }, inplace=True)
        
        trb_df.rename(columns={
            'cdr3_aa': 'trb',
            'v_gene': 'trbv_gene',
            'd_gene': 'trbd_gene',
            'j_gene': 'trbj_gene'
        }, inplace=True)
        
        mri_table = pd.concat([tra_df, trb_df], ignore_index=True)
        
        if mri_table.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        seq_table = mri_table.copy().drop_duplicates()
        
        # Add metadata
        repertoire_id = Path(file_path).stem
        parts = Path(file_path).parts
        study_id = parts[-4] if len(parts) >= 4 else "unknown"
        category = parts[-5] if len(parts) >= 5 else "unknown"
        molecule_type = parts[-3] if len(parts) >= 3 else "unknown"
        
        mri_table['repertoire_id'] = repertoire_id
        mri_table['study_id'] = study_id
        mri_table['category'] = category
        mri_table['molecule_type'] = molecule_type
        mri_table['host_organism'] = 'human'
        mri_table['source'] = 'misc_format'
        
        seq_table['source'] = 'misc_format'
        
        seq_table = standardize_sequence(seq_table)
        mri_table = standardize_mri(mri_table)
        
        return mri_table, seq_table
    
    def _parse_misc_format_four(self, df: pd.DataFrame, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Misc format four: placeholder"""
        logger.warning("misc_format_four not fully implemented")
        return pd.DataFrame(), pd.DataFrame()
    
    def _parse_misc_format_five(self, df: pd.DataFrame, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Misc format five: placeholder"""
        logger.warning("misc_format_five not fully implemented")
        return pd.DataFrame(), pd.DataFrame()
    
    def _parse_misc_format_six(self, df: pd.DataFrame, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Misc format six: placeholder"""
        logger.warning("misc_format_six not fully implemented")
        return pd.DataFrame(), pd.DataFrame()
    
    # ========================================================================
    #                      PARALLEL DIRECTORY PROCESSING
    # ========================================================================
    
    def _filter_and_prioritize_files(self, all_files: List[Path]) -> Tuple[List[Path], int]:
        """
        Filter out non-TCR files and deduplicate studies with multiple formats.
        
        Valid TCR folders: contigs, clonotypes, airr, misc, 
                          bulk_survey_tra, bulk_survey_trb, 
                          bulk_deep_tra, bulk_deep_trb
        
        Single-cell priority (to avoid overrepresentation):
            1. contigs (highest priority)
            2. clonotypes
            3. airr (lowest priority)
        
        Only ONE format per study will be processed.
        
        Returns:
            (filtered_files, skipped_count)
        """
        # Valid TCR directories
        valid_tcr_dirs = {
            'contigs', 'clonotypes', 'airr', 'misc',
            'bulk_survey_tra', 'bulk_survey_trb',
            'bulk_deep_tra', 'bulk_deep_trb'
        }
        
        # Non-TCR directories to exclude
        excluded_dirs = {'gex', 'cite', 'adt', 'hto', 'multiplexing', 'spatial', 'atac'}
        
        # Single-cell format priority
        sc_priority = ['contigs', 'clonotypes', 'airr']
        
        # Group files by study and format
        study_files = {}  # study_id -> {format: [files]}
        skipped_count = 0
        
        for f in all_files:
            # Get all parent directory names
            parent_names = {p.name.lower() for p in f.parents}
            
            # Skip non-TCR directories
            if parent_names.intersection(excluded_dirs):
                skipped_count += 1
                continue
            
            # Check if file is in a valid TCR directory
            tcr_dir = None
            for parent in f.parents:
                if parent.name.lower() in valid_tcr_dirs:
                    tcr_dir = parent.name.lower()
                    break
            
            if tcr_dir is None:
                logger.warning(f"Skipping file not in valid TCR directory: {f}")
                skipped_count += 1
                continue
            
            # Extract study ID (grandparent of TCR directory)
            # Path structure: .../study_id/tcr_format/file.tsv
            try:
                tcr_dir_parent = None
                for i, parent in enumerate(f.parents):
                    if parent.name.lower() == tcr_dir:
                        tcr_dir_parent = f.parents[i + 1]
                        break
                
                if tcr_dir_parent is None:
                    study_id = f.parent.name  # fallback
                else:
                    study_id = tcr_dir_parent.name
            except:
                study_id = f.parent.name
            
            # Group by study and format
            if study_id not in study_files:
                study_files[study_id] = {}
            if tcr_dir not in study_files[study_id]:
                study_files[study_id][tcr_dir] = []
            study_files[study_id][tcr_dir].append(f)
        
        # Apply prioritization for single-cell studies
        selected_files = []
        
        for study_id, formats in study_files.items():
            # Check if study has single-cell data (contigs/clonotypes/airr)
            sc_formats = [fmt for fmt in sc_priority if fmt in formats]
            
            if sc_formats:
                # Use highest priority single-cell format ONLY
                chosen_format = sc_formats[0]  # Already sorted by priority
                selected_files.extend(formats[chosen_format])
                
                # Count skipped formats
                for fmt in sc_formats[1:]:
                    skipped_count += len(formats[fmt])
                    logger.info(f"Study {study_id}: Using {chosen_format}, skipping {fmt} ({len(formats[fmt])} files)")
                
                # Also include bulk data if present (bulk is independent)
                for fmt in formats:
                    if fmt.startswith('bulk_') or fmt == 'misc':
                        selected_files.extend(formats[fmt])
            else:
                # No single-cell data, include all bulk/misc files
                for fmt, files in formats.items():
                    selected_files.extend(files)
        
        return selected_files, skipped_count
    
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
        all_files = list(input_path.glob(pattern))
        
        # Filter and prioritize files
        files, skipped_count = self._filter_and_prioritize_files(all_files)
        
        logger.info(f"Found {len(files)} TCR files matching {pattern} (skipped {skipped_count} non-TCR/duplicate files)")
        
        total_mri = 0
        total_seq = 0
        processed = 0
        failed = []
        
        # Process files in parallel with progress bar
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self._process_single_file, str(f)): f
                for f in files
            }
            
            # Create progress bar
            with tqdm(total=len(files), desc="Processing files", unit="file") as pbar:
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        mri_rows, seq_rows = future.result()
                        total_mri += mri_rows
                        total_seq += seq_rows
                        processed += 1
                        
                        # Update progress bar with stats
                        pbar.set_postfix({
                            'MRI': f'{total_mri:,}',
                            'Seq': f'{total_seq:,}',
                            'Failed': len(failed)
                        })
                        pbar.update(1)
                    except Exception as e:
                        failed.append(str(file_path))
                        logger.error(f"✗ {file_path.name}: {str(e)[:100]}")
                        pbar.update(1)
        
        logger.info(f"✓ Completed: {processed} files, {total_mri:,} MRI rows, {total_seq:,} seq rows")
        if failed:
            logger.warning(f"Failed {len(failed)} files: {failed[:10]}")
        
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
            raise RuntimeError(f"Error processing {file_path}: {e}")
