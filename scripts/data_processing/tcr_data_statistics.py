#!/usr/bin/env python3
"""
TCR Data Statistics Tool

Analyzes raw TCR data to produce comprehensive statistics including:
- Total distinct studies and databases
- Sequence counts by type (MHC, peptide, TCR alpha/beta chains)
- Paired vs unpaired sequence breakdown
- Per-category and per-study breakdowns

Usage:
    python tcr_data_statistics.py --path /path/to/raw_data
    python tcr_data_statistics.py --path /path/to/raw_data --output stats.json
"""

import argparse
import gzip
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SequenceStats:
    """Statistics for a single file or aggregated group."""
    total_sequences: int = 0
    tra_sequences: int = 0
    trb_sequences: int = 0
    paired_sequences: int = 0  # Both TRA and TRB present
    unpaired_sequences: int = 0  # Only one chain
    with_peptide: int = 0
    with_mhc: int = 0
    tcr_only: int = 0  # No peptide or MHC

    def __add__(self, other: 'SequenceStats') -> 'SequenceStats':
        return SequenceStats(
            total_sequences=self.total_sequences + other.total_sequences,
            tra_sequences=self.tra_sequences + other.tra_sequences,
            trb_sequences=self.trb_sequences + other.trb_sequences,
            paired_sequences=self.paired_sequences + other.paired_sequences,
            unpaired_sequences=self.unpaired_sequences + other.unpaired_sequences,
            with_peptide=self.with_peptide + other.with_peptide,
            with_mhc=self.with_mhc + other.with_mhc,
            tcr_only=self.tcr_only + other.tcr_only,
        )


@dataclass
class StudyStats:
    """Statistics for a single study."""
    name: str
    category: str
    file_count: int = 0
    stats: SequenceStats = field(default_factory=SequenceStats)
    file_types: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class DatabaseStats:
    """Statistics for a database."""
    name: str
    file_count: int = 0
    stats: SequenceStats = field(default_factory=SequenceStats)
    description: str = ""
    errors: List[str] = field(default_factory=list)


# =============================================================================
# Constants
# =============================================================================

# File extensions to process
VALID_EXTENSIONS = {'.tsv', '.csv', '.txt', '.gz'}

# Directories to skip
SKIP_DIRS = {'.DS_Store', '__pycache__', '.git'}

# Column patterns for chain detection
TRA_V_PATTERNS = ['TRAV', 'TCRAV', 'v_alpha', 'va_gene', 'trav']
TRB_V_PATTERNS = ['TRBV', 'TCRBV', 'v_beta', 'vb_gene', 'trbv']

# Column patterns for paired data detection
PAIRED_INDICATORS = ['barcode', 'cell_id', 'contig_id', 'clonotype_id']

# Column patterns for peptide/MHC
PEPTIDE_COLUMNS = ['peptide', 'epitope', 'antigen', 'antigen.epitope']
MHC_COLUMNS = ['mhc', 'hla', 'mhc_class', 'mhc_a', 'mhc_b', 'mhc.a', 'mhc.b',
               'mhc_one', 'mhc_two', 'mhc_restriction']

# CDR3 column patterns
CDR3_COLUMNS = ['cdr3', 'cdr3_aa', 'cdr3_b_aa', 'cdr3_a_aa', 'junction_aa',
                'aacdr3', 'amino_acid', 'cdr3s_aa', 'tra', 'trb',
                'cdr3.alpha.aa', 'cdr3.beta.aa']


def is_valid_value(val) -> bool:
    """Check if value is valid (not NA, empty, or null)."""
    if pd.isna(val):
        return False
    val_str = str(val).strip().upper()
    return val_str not in ('', 'NA', 'NAN', 'NONE', 'NULL', '0')


# =============================================================================
# File Classification
# =============================================================================

def get_file_type(file_path: Path, header: Optional[List[str]] = None) -> str:
    """
    Classify file type based on path and headers.

    Returns one of:
    - 'bulk_tra': Bulk TCR alpha chain data
    - 'bulk_trb': Bulk TCR beta chain data
    - 'bulk_mixed': Bulk data with both chains in different rows
    - 'paired_contigs': Single-cell contig annotations
    - 'paired_clonotypes': Clonotype data with paired chains
    - 'database': Database file format
    - 'unknown': Unrecognized format
    """
    path_str = str(file_path).lower()

    # Check path hints
    if 'contigs' in path_str or 'contig_annotations' in path_str:
        return 'paired_contigs'
    if 'clonotypes' in path_str:
        return 'paired_clonotypes'

    # Check for bulk data indicators in path
    if 'bulk' in path_str:
        # Check for TRA/TRB directory patterns
        if 'bulk_survey_tra' in path_str or '/tra/' in path_str or '_tra/' in path_str:
            return 'bulk_tra'
        if 'bulk_survey_trb' in path_str or '/trb/' in path_str or '_trb/' in path_str:
            return 'bulk_trb'
        # Generic bulk - will determine chain from V gene values
        return 'bulk_mixed'

    # Check headers if available
    if header:
        header_lower = [h.lower() for h in header]

        # Check for paired indicators
        if any(ind in header_lower for ind in PAIRED_INDICATORS):
            if 'chain' in header_lower or 'locus' in header_lower:
                return 'paired_contigs'
            if 'cdr3s_aa' in header_lower or 'cdr3s_nt' in header_lower:
                return 'paired_clonotypes'

        # Check for chain-specific V gene columns (e.g., TRAV, TRBV as column names)
        has_tra_v = any(any(p.lower() in h for p in TRA_V_PATTERNS) for h in header_lower)
        has_trb_v = any(any(p.lower() in h for p in TRB_V_PATTERNS) for h in header_lower)

        if has_tra_v and has_trb_v:
            return 'paired_clonotypes'  # Both chains in columns
        elif has_tra_v:
            return 'bulk_tra'
        elif has_trb_v:
            return 'bulk_trb'

        # Check for generic V gene column - need to inspect data for chain type
        if 'v_gene' in header_lower or 'vgene' in header_lower or 'v_resolved' in header_lower:
            return 'bulk_mixed'  # Will determine from data

        # Check for amino_acid column (adaptive bulk format)
        if 'amino_acid' in header_lower and ('v_gene' in header_lower or 'v_resolved' in header_lower):
            return 'bulk_mixed'

    return 'unknown'


def read_file_header(file_path: Path, delimiter: str = '\t') -> Optional[List[str]]:
    """Read the header row from a file."""
    try:
        if str(file_path).endswith('.gz'):
            with gzip.open(file_path, 'rt') as f:
                header = f.readline().strip().split(delimiter)
        else:
            with open(file_path, 'r') as f:
                header = f.readline().strip().split(delimiter)

        # Try comma if tab didn't work well
        if len(header) == 1 and ',' in header[0]:
            if str(file_path).endswith('.gz'):
                with gzip.open(file_path, 'rt') as f:
                    header = f.readline().strip().split(',')
            else:
                with open(file_path, 'r') as f:
                    header = f.readline().strip().split(',')

        return header
    except Exception:
        return None


# =============================================================================
# Chain Detection
# =============================================================================

def detect_chain_from_value(v_gene: str) -> Optional[str]:
    """Detect chain type from V gene value."""
    if pd.isna(v_gene) or not v_gene:
        return None
    v_gene_upper = str(v_gene).upper()
    if 'TRAV' in v_gene_upper or 'TCRAV' in v_gene_upper:
        return 'TRA'
    if 'TRBV' in v_gene_upper or 'TCRBV' in v_gene_upper:
        return 'TRB'
    return None


def find_v_gene_column(columns: List[str]) -> Optional[str]:
    """Find the V gene column in a dataframe."""
    columns_lower = {c.lower(): c for c in columns}

    # Check specific patterns first (in order of preference)
    for pattern in ['v_gene', 'vgene', 'v_call', 'v_segment', 'v_resolved', 'v.segm']:
        if pattern in columns_lower:
            return columns_lower[pattern]

    # Check for TRA/TRB specific columns
    for pattern in TRA_V_PATTERNS + TRB_V_PATTERNS:
        for col in columns:
            if pattern.lower() in col.lower():
                return col

    return None


def find_cdr3_columns(columns: List[str]) -> Dict[str, str]:
    """Find CDR3 columns for alpha and beta chains."""
    result = {'tra': None, 'trb': None, 'generic': None}
    columns_lower = {c.lower(): c for c in columns}

    # Look for chain-specific CDR3 columns
    for col in columns:
        col_lower = col.lower()
        if 'cdr3' in col_lower or 'junction' in col_lower:
            # Check for alpha chain patterns
            if ('alpha' in col_lower or '.alpha' in col_lower or
                '_a_' in col_lower or col_lower.endswith('_a') or
                col_lower.endswith('.a.aa') or 'cdr3_a' in col_lower):
                result['tra'] = col
            # Check for beta chain patterns
            elif ('beta' in col_lower or '.beta' in col_lower or
                  '_b_' in col_lower or col_lower.endswith('_b') or
                  col_lower.endswith('.b.aa') or 'cdr3_b' in col_lower):
                result['trb'] = col
            elif result['generic'] is None:
                result['generic'] = col

    # Check for 'tra' and 'trb' columns directly
    if 'tra' in columns_lower:
        result['tra'] = columns_lower['tra']
    if 'trb' in columns_lower:
        result['trb'] = columns_lower['trb']

    # Also check for amino_acid column (common in adaptive bulk data)
    if result['generic'] is None and 'amino_acid' in columns_lower:
        result['generic'] = columns_lower['amino_acid']

    return result


def has_peptide(row: pd.Series, peptide_cols: List[str]) -> bool:
    """Check if row has peptide annotation."""
    for col in peptide_cols:
        if col in row.index:
            if is_valid_value(row[col]):
                return True
    return False


def has_mhc(row: pd.Series, mhc_cols: List[str]) -> bool:
    """Check if row has MHC annotation."""
    for col in mhc_cols:
        if col in row.index:
            if is_valid_value(row[col]):
                return True
    return False


# =============================================================================
# File Parsing
# =============================================================================

def parse_bulk_file(file_path: Path, file_type: str) -> SequenceStats:
    """Parse a bulk TCR file and return statistics."""
    stats = SequenceStats()

    try:
        # Determine delimiter
        delimiter = ',' if file_path.suffix == '.csv' or 'csv' in str(file_path).lower() else '\t'

        # For bulk_tra/bulk_trb files, we know the chain type - use fast counting
        if file_type in ('bulk_tra', 'bulk_trb'):
            # Fast path: just count rows without iterating
            total_rows = 0
            chunks = pd.read_csv(
                file_path,
                sep=delimiter,
                chunksize=100000,
                low_memory=False,
                on_bad_lines='skip',
                usecols=[0]  # Only read first column for counting
            )
            for chunk in chunks:
                total_rows += len(chunk)

            stats.total_sequences = total_rows
            stats.unpaired_sequences = total_rows
            stats.tcr_only = total_rows
            if file_type == 'bulk_tra':
                stats.tra_sequences = total_rows
            else:
                stats.trb_sequences = total_rows
            return stats

        # For bulk_mixed, need to check each row's V gene
        chunks = pd.read_csv(
            file_path,
            sep=delimiter,
            chunksize=100000,
            low_memory=False,
            on_bad_lines='skip'
        )

        for chunk in chunks:
            columns = chunk.columns.tolist()
            v_gene_col = find_v_gene_column(columns)

            # Find peptide/MHC columns
            peptide_cols = [c for c in columns if any(p in c.lower() for p in PEPTIDE_COLUMNS)]
            mhc_cols = [c for c in columns if any(m in c.lower() for m in MHC_COLUMNS)]

            chunk_size = len(chunk)
            stats.total_sequences += chunk_size

            # Vectorized chain detection if v_gene column exists
            if v_gene_col and v_gene_col in chunk.columns:
                v_genes = chunk[v_gene_col].astype(str).str.upper()
                tra_mask = v_genes.str.contains('TRAV|TCRAV', regex=True, na=False)
                trb_mask = v_genes.str.contains('TRBV|TCRBV', regex=True, na=False)
                stats.tra_sequences += tra_mask.sum()
                stats.trb_sequences += trb_mask.sum()

            stats.unpaired_sequences += chunk_size

            # Check annotations (vectorized)
            if peptide_cols:
                has_pep = chunk[peptide_cols].notna().any(axis=1) & (chunk[peptide_cols].astype(str) != '').any(axis=1)
                stats.with_peptide += has_pep.sum()
            else:
                has_pep = pd.Series([False] * chunk_size)

            if mhc_cols:
                has_mhc_ann = chunk[mhc_cols].notna().any(axis=1) & (chunk[mhc_cols].astype(str) != '').any(axis=1)
                stats.with_mhc += has_mhc_ann.sum()
            else:
                has_mhc_ann = pd.Series([False] * chunk_size)

            stats.tcr_only += (~has_pep & ~has_mhc_ann).sum()

    except Exception as e:
        # Return partial stats with error noted
        pass

    return stats


def parse_paired_contigs(file_path: Path) -> SequenceStats:
    """Parse single-cell contig annotations."""
    stats = SequenceStats()

    try:
        delimiter = ',' if '.csv' in str(file_path).lower() else '\t'
        df = pd.read_csv(file_path, sep=delimiter, low_memory=False, on_bad_lines='skip')

        columns = df.columns.tolist()

        # Find barcode and chain columns
        barcode_col = None
        for col in ['barcode', 'cell_id', 'contig_id']:
            if col in [c.lower() for c in columns]:
                barcode_col = [c for c in columns if c.lower() == col][0]
                break

        chain_col = None
        for col in ['chain', 'locus']:
            if col in [c.lower() for c in columns]:
                chain_col = [c for c in columns if c.lower() == col][0]
                break

        if barcode_col is None:
            # Fall back to row-based counting
            stats.total_sequences = len(df)
            stats.unpaired_sequences = len(df)
            return stats

        # Group by barcode/cell
        cell_chains = defaultdict(set)

        for _, row in df.iterrows():
            barcode = row.get(barcode_col, '')
            chain = None

            if chain_col and chain_col in row.index:
                chain_val = str(row[chain_col]).upper()
                if 'TRA' in chain_val or 'ALPHA' in chain_val:
                    chain = 'TRA'
                elif 'TRB' in chain_val or 'BETA' in chain_val:
                    chain = 'TRB'

            if barcode and chain:
                cell_chains[barcode].add(chain)

        # Count based on chains per cell
        for barcode, chains in cell_chains.items():
            stats.total_sequences += 1
            if 'TRA' in chains:
                stats.tra_sequences += 1
            if 'TRB' in chains:
                stats.trb_sequences += 1
            if 'TRA' in chains and 'TRB' in chains:
                stats.paired_sequences += 1
            else:
                stats.unpaired_sequences += 1

        stats.tcr_only = stats.total_sequences  # Contigs typically don't have peptide/MHC

    except Exception:
        pass

    return stats


def parse_paired_clonotypes(file_path: Path) -> SequenceStats:
    """Parse clonotype data with paired chains."""
    stats = SequenceStats()

    try:
        delimiter = ',' if '.csv' in str(file_path).lower() else '\t'
        df = pd.read_csv(file_path, sep=delimiter, low_memory=False, on_bad_lines='skip')

        columns = df.columns.tolist()
        cdr3_cols = find_cdr3_columns(columns)

        # Find peptide/MHC columns
        peptide_cols = [c for c in columns if any(p in c.lower() for p in PEPTIDE_COLUMNS)]
        mhc_cols = [c for c in columns if any(m in c.lower() for m in MHC_COLUMNS)]

        # Check for cdr3s_aa format (e.g., "TRA:CAVS;TRB:CASS")
        cdr3s_col = None
        for col in columns:
            if 'cdr3s' in col.lower():
                cdr3s_col = col
                break

        for _, row in df.iterrows():
            stats.total_sequences += 1

            has_tra = False
            has_trb = False

            if cdr3s_col and cdr3s_col in row.index:
                # Parse combined CDR3 format
                cdr3s = str(row[cdr3s_col])
                if 'TRA' in cdr3s.upper():
                    has_tra = True
                if 'TRB' in cdr3s.upper():
                    has_trb = True
            else:
                # Check separate columns
                if cdr3_cols['tra'] and pd.notna(row.get(cdr3_cols['tra'], None)):
                    has_tra = True
                if cdr3_cols['trb'] and pd.notna(row.get(cdr3_cols['trb'], None)):
                    has_trb = True

            if has_tra:
                stats.tra_sequences += 1
            if has_trb:
                stats.trb_sequences += 1
            if has_tra and has_trb:
                stats.paired_sequences += 1
            else:
                stats.unpaired_sequences += 1

            # Check annotations
            has_pep = has_peptide(row, peptide_cols)
            has_mhc_ann = has_mhc(row, mhc_cols)

            if has_pep:
                stats.with_peptide += 1
            if has_mhc_ann:
                stats.with_mhc += 1
            if not has_pep and not has_mhc_ann:
                stats.tcr_only += 1

    except Exception:
        pass

    return stats


def parse_generic_file(file_path: Path) -> SequenceStats:
    """Parse a generic TCR file."""
    stats = SequenceStats()

    try:
        # Try to detect delimiter and read
        delimiter = ',' if '.csv' in str(file_path).lower() else '\t'

        chunks = pd.read_csv(
            file_path,
            sep=delimiter,
            chunksize=100000,
            low_memory=False,
            on_bad_lines='skip'
        )

        for chunk in chunks:
            columns = chunk.columns.tolist()
            v_gene_col = find_v_gene_column(columns)
            cdr3_cols = find_cdr3_columns(columns)

            peptide_cols = [c for c in columns if any(p in c.lower() for p in PEPTIDE_COLUMNS)]
            mhc_cols = [c for c in columns if any(m in c.lower() for m in MHC_COLUMNS)]

            for _, row in chunk.iterrows():
                stats.total_sequences += 1

                has_tra = False
                has_trb = False

                # Check for chain-specific CDR3
                if cdr3_cols['tra'] and cdr3_cols['tra'] in row.index:
                    if pd.notna(row[cdr3_cols['tra']]) and str(row[cdr3_cols['tra']]).strip():
                        has_tra = True
                if cdr3_cols['trb'] and cdr3_cols['trb'] in row.index:
                    if pd.notna(row[cdr3_cols['trb']]) and str(row[cdr3_cols['trb']]).strip():
                        has_trb = True

                # Fallback to V gene detection
                if not has_tra and not has_trb and v_gene_col:
                    chain = detect_chain_from_value(row.get(v_gene_col, ''))
                    if chain == 'TRA':
                        has_tra = True
                    elif chain == 'TRB':
                        has_trb = True

                if has_tra:
                    stats.tra_sequences += 1
                if has_trb:
                    stats.trb_sequences += 1
                if has_tra and has_trb:
                    stats.paired_sequences += 1
                elif has_tra or has_trb:
                    stats.unpaired_sequences += 1
                else:
                    stats.unpaired_sequences += 1

                # Annotations
                has_pep = has_peptide(row, peptide_cols)
                has_mhc_ann = has_mhc(row, mhc_cols)

                if has_pep:
                    stats.with_peptide += 1
                if has_mhc_ann:
                    stats.with_mhc += 1
                if not has_pep and not has_mhc_ann:
                    stats.tcr_only += 1

    except Exception:
        pass

    return stats


# =============================================================================
# Database Handlers
# =============================================================================

def analyze_vdjdb(db_path: Path) -> DatabaseStats:
    """Analyze VDJdb database."""
    stats = DatabaseStats(name="VDJdb", description="VDJ Database of TCR-epitope pairs")

    # VDJdb has two formats:
    # - vdjdb_full.txt: paired format with cdr3.alpha and cdr3.beta columns
    # - vdjdb.txt: single-chain format with gene column (TRA/TRB) and complex.id for pairing
    # Process vdjdb_full.txt if available (cleaner paired data)

    full_file = db_path / 'vdjdb_full.txt'
    std_file = db_path / 'vdjdb.txt'

    if full_file.exists():
        # Process paired format (vdjdb_full.txt)
        stats.file_count = 1
        try:
            df = pd.read_csv(full_file, sep='\t', low_memory=False, on_bad_lines='skip')

            for _, row in df.iterrows():
                stats.stats.total_sequences += 1

                # Check for alpha and beta chains
                cdr3_alpha = row.get('cdr3.alpha', None)
                cdr3_beta = row.get('cdr3.beta', None)

                has_tra = is_valid_value(cdr3_alpha)
                has_trb = is_valid_value(cdr3_beta)

                if has_tra:
                    stats.stats.tra_sequences += 1
                if has_trb:
                    stats.stats.trb_sequences += 1
                if has_tra and has_trb:
                    stats.stats.paired_sequences += 1
                else:
                    stats.stats.unpaired_sequences += 1

                # Check epitope and MHC
                epitope_val = row.get('antigen.epitope', None)
                mhc_a = row.get('mhc.a', None)
                mhc_b = row.get('mhc.b', None)

                has_pep = is_valid_value(epitope_val)
                has_mhc = is_valid_value(mhc_a) or is_valid_value(mhc_b)

                if has_pep:
                    stats.stats.with_peptide += 1
                if has_mhc:
                    stats.stats.with_mhc += 1
                if not has_pep and not has_mhc:
                    stats.stats.tcr_only += 1

        except Exception as e:
            stats.errors.append(f"vdjdb_full.txt: {str(e)}")

    elif std_file.exists():
        # Process single-chain format (vdjdb.txt)
        stats.file_count = 1
        try:
            df = pd.read_csv(std_file, sep='\t', low_memory=False, on_bad_lines='skip')

            # Group by complex.id for paired detection
            complex_chains = defaultdict(lambda: {'TRA': False, 'TRB': False, 'has_peptide': False, 'has_mhc': False})

            for _, row in df.iterrows():
                complex_id = row.get('complex.id', None)
                gene = str(row.get('gene', '')).upper()

                # Track chain types per complex
                if complex_id and pd.notna(complex_id) and complex_id != 0:
                    if 'TRA' in gene:
                        complex_chains[complex_id]['TRA'] = True
                    elif 'TRB' in gene:
                        complex_chains[complex_id]['TRB'] = True

                    epitope_val = row.get('antigen.epitope', None)
                    if is_valid_value(epitope_val):
                        complex_chains[complex_id]['has_peptide'] = True
                    mhc_a = row.get('mhc.a', None)
                    mhc_b = row.get('mhc.b', None)
                    if is_valid_value(mhc_a) or is_valid_value(mhc_b):
                        complex_chains[complex_id]['has_mhc'] = True
                else:
                    # No complex.id - count as individual sequence
                    stats.stats.total_sequences += 1
                    if 'TRA' in gene:
                        stats.stats.tra_sequences += 1
                    elif 'TRB' in gene:
                        stats.stats.trb_sequences += 1
                    stats.stats.unpaired_sequences += 1

                    epitope_val = row.get('antigen.epitope', None)
                    has_pep = is_valid_value(epitope_val)
                    mhc_a = row.get('mhc.a', None)
                    mhc_b = row.get('mhc.b', None)
                    has_mhc = is_valid_value(mhc_a) or is_valid_value(mhc_b)

                    if has_pep:
                        stats.stats.with_peptide += 1
                    if has_mhc:
                        stats.stats.with_mhc += 1
                    if not has_pep and not has_mhc:
                        stats.stats.tcr_only += 1

            # Count complexes (paired/unpaired)
            for complex_id, data in complex_chains.items():
                stats.stats.total_sequences += 1

                if data['TRA']:
                    stats.stats.tra_sequences += 1
                if data['TRB']:
                    stats.stats.trb_sequences += 1

                if data['TRA'] and data['TRB']:
                    stats.stats.paired_sequences += 1
                else:
                    stats.stats.unpaired_sequences += 1

                if data['has_peptide']:
                    stats.stats.with_peptide += 1
                if data['has_mhc']:
                    stats.stats.with_mhc += 1
                if not data['has_peptide'] and not data['has_mhc']:
                    stats.stats.tcr_only += 1

        except Exception as e:
            stats.errors.append(f"vdjdb.txt: {str(e)}")

    return stats


def analyze_mcpas(db_path: Path) -> DatabaseStats:
    """Analyze McPAS-TCR database."""
    stats = DatabaseStats(name="McPASDB", description="McPAS TCR-peptide-MHC database")

    file_path = db_path / "McPAS-TCR.csv"
    if file_path.exists():
        stats.file_count = 1
        try:
            df = pd.read_csv(file_path, low_memory=False, on_bad_lines='skip')

            for _, row in df.iterrows():
                stats.stats.total_sequences += 1

                # McPAS has CDR3.alpha.aa and CDR3.beta.aa columns (note the .aa suffix)
                tra_val = row.get('CDR3.alpha.aa', None)
                trb_val = row.get('CDR3.beta.aa', None)

                has_tra = pd.notna(tra_val) and str(tra_val).strip() and str(tra_val).upper() != 'NA'
                has_trb = pd.notna(trb_val) and str(trb_val).strip() and str(trb_val).upper() != 'NA'

                if has_tra:
                    stats.stats.tra_sequences += 1
                if has_trb:
                    stats.stats.trb_sequences += 1
                if has_tra and has_trb:
                    stats.stats.paired_sequences += 1
                else:
                    stats.stats.unpaired_sequences += 1

                # McPAS has Epitope.peptide and MHC columns
                epitope_val = row.get('Epitope.peptide', None)
                mhc_val = row.get('MHC', None)

                has_pep = pd.notna(epitope_val) and str(epitope_val).strip() and str(epitope_val).upper() != 'NA'
                has_mhc = pd.notna(mhc_val) and str(mhc_val).strip() and str(mhc_val).upper() != 'NA'

                if has_pep:
                    stats.stats.with_peptide += 1
                if has_mhc:
                    stats.stats.with_mhc += 1
                if not has_pep and not has_mhc:
                    stats.stats.tcr_only += 1

        except Exception as e:
            stats.errors.append(str(e))

    return stats


def analyze_tcrdb(db_path: Path) -> DatabaseStats:
    """Analyze TCRdb database."""
    stats = DatabaseStats(name="TCRdb", description="TCR repertoire database")

    # TCRdb has multiple PRJNA*.tsv files
    tsv_files = list(db_path.glob("*.tsv"))
    stats.file_count = len(tsv_files)

    for file_path in tqdm(tsv_files, desc="  TCRdb files", leave=False):
        try:
            chunks = pd.read_csv(file_path, sep='\t', chunksize=100000,
                               low_memory=False, on_bad_lines='skip')

            for chunk in chunks:
                stats.stats.total_sequences += len(chunk)

                # TCRdb typically has v_gene column
                v_gene_col = find_v_gene_column(chunk.columns.tolist())

                if v_gene_col:
                    for v_gene in chunk[v_gene_col]:
                        chain = detect_chain_from_value(v_gene)
                        if chain == 'TRA':
                            stats.stats.tra_sequences += 1
                        elif chain == 'TRB':
                            stats.stats.trb_sequences += 1

                stats.stats.unpaired_sequences += len(chunk)
                stats.stats.tcr_only += len(chunk)  # TCRdb doesn't have peptide/MHC

        except Exception as e:
            stats.errors.append(f"{file_path.name}: {str(e)}")

    return stats


def analyze_iedb(db_path: Path) -> DatabaseStats:
    """Analyze IEDB database."""
    stats = DatabaseStats(name="IEDB", description="Immune Epitope Database")

    # IEDB has multiple table files
    for pattern in ['*.tsv', '*.csv']:
        for file_path in db_path.glob(pattern):
            stats.file_count += 1
            try:
                delimiter = ',' if '.csv' in str(file_path) else '\t'
                chunks = pd.read_csv(file_path, sep=delimiter, chunksize=100000,
                                   low_memory=False, on_bad_lines='skip')

                for chunk in chunks:
                    columns = [c.lower() for c in chunk.columns]

                    # Check if this is a T-cell receptor file
                    has_tcr_cols = any('cdr3' in c or 'receptor' in c for c in columns)

                    if has_tcr_cols:
                        stats.stats.total_sequences += len(chunk)

                        # IEDB has various column names
                        peptide_cols = [c for c in chunk.columns
                                      if any(p in c.lower() for p in ['epitope', 'peptide', 'antigen'])]
                        mhc_cols = [c for c in chunk.columns
                                  if any(m in c.lower() for m in ['mhc', 'hla', 'restriction'])]

                        for _, row in chunk.iterrows():
                            # Check chain type
                            chain_col = [c for c in chunk.columns if 'chain' in c.lower()]
                            if chain_col:
                                chain_val = str(row.get(chain_col[0], '')).upper()
                                if 'ALPHA' in chain_val or 'TRA' in chain_val:
                                    stats.stats.tra_sequences += 1
                                elif 'BETA' in chain_val or 'TRB' in chain_val:
                                    stats.stats.trb_sequences += 1

                            stats.stats.unpaired_sequences += 1

                            # Annotations
                            if any(pd.notna(row.get(c, None)) for c in peptide_cols):
                                stats.stats.with_peptide += 1
                            if any(pd.notna(row.get(c, None)) for c in mhc_cols):
                                stats.stats.with_mhc += 1
                            if not any(pd.notna(row.get(c, None)) for c in peptide_cols + mhc_cols):
                                stats.stats.tcr_only += 1

            except Exception as e:
                stats.errors.append(f"{file_path.name}: {str(e)}")

    return stats


def analyze_ireceptor(db_path: Path) -> DatabaseStats:
    """Analyze iReceptor database."""
    stats = DatabaseStats(name="iReceptor", description="iReceptor AIRR-compliant data")

    for pattern in ['*.tsv', '*.csv', '*.json']:
        for file_path in db_path.glob(pattern):
            if file_path.suffix == '.json':
                continue  # Skip metadata files

            stats.file_count += 1
            try:
                delimiter = ',' if '.csv' in str(file_path) else '\t'
                file_stats = parse_generic_file(file_path)
                stats.stats = stats.stats + file_stats
            except Exception as e:
                stats.errors.append(f"{file_path.name}: {str(e)}")

    return stats


def analyze_cedar(db_path: Path) -> DatabaseStats:
    """Analyze CEDAR database."""
    stats = DatabaseStats(name="CEDAR", description="CEDAR MHC ligand and T-cell assay database")

    for file_path in db_path.glob("*.tsv"):
        stats.file_count += 1
        try:
            # Check if this is a receptor results file (has TCR data)
            if 'receptor' in file_path.name.lower():
                chunks = pd.read_csv(file_path, sep='\t', chunksize=100000,
                                   low_memory=False, on_bad_lines='skip')

                for chunk in chunks:
                    columns = [c.lower() for c in chunk.columns]

                    # Check for TCR-related columns
                    if any('cdr3' in c or 'tcr' in c for c in columns):
                        stats.stats.total_sequences += len(chunk)
                        stats.stats.unpaired_sequences += len(chunk)

                        # CEDAR has MHC and peptide data
                        peptide_cols = [c for c in chunk.columns
                                      if any(p in c.lower() for p in PEPTIDE_COLUMNS)]
                        mhc_cols = [c for c in chunk.columns
                                  if any(m in c.lower() for m in MHC_COLUMNS)]

                        for _, row in chunk.iterrows():
                            if any(pd.notna(row.get(c, None)) for c in peptide_cols):
                                stats.stats.with_peptide += 1
                            if any(pd.notna(row.get(c, None)) for c in mhc_cols):
                                stats.stats.with_mhc += 1

        except Exception as e:
            stats.errors.append(f"{file_path.name}: {str(e)}")

    return stats


def analyze_warrendb(db_path: Path) -> DatabaseStats:
    """Analyze WarrenDB database."""
    stats = DatabaseStats(name="WarrenDB", description="Viral-specific TCR sequences")

    for file_path in db_path.glob("*.csv"):
        stats.file_count += 1
        try:
            df = pd.read_csv(file_path, low_memory=False, on_bad_lines='skip')
            stats.stats.total_sequences += len(df)

            # Warren DB files are organized by virus, contain TRB sequences
            stats.stats.trb_sequences += len(df)
            stats.stats.unpaired_sequences += len(df)
            stats.stats.tcr_only += len(df)  # Typically no explicit peptide/MHC columns

        except Exception as e:
            stats.errors.append(f"{file_path.name}: {str(e)}")

    return stats


def analyze_gliphdb(db_path: Path) -> DatabaseStats:
    """Analyze GLIPH database."""
    stats = DatabaseStats(name="GLIPHDB", description="GLIPH reference epitope database")

    # Look for TSV files
    for file_path in list(db_path.glob("*.tsv")) + list(db_path.glob("**/*.tsv")):
        stats.file_count += 1
        try:
            df = pd.read_csv(file_path, sep='\t', low_memory=False, on_bad_lines='skip')
            columns = df.columns.tolist()

            # Check if it has TCR data
            if any('cdr3' in c.lower() or 'tcr' in c.lower() for c in columns):
                file_stats = parse_generic_file(file_path)
                stats.stats = stats.stats + file_stats

        except Exception as e:
            stats.errors.append(f"{file_path.name}: {str(e)}")

    return stats


def analyze_imgthla(db_path: Path) -> DatabaseStats:
    """Analyze IMGT/HLA database (HLA sequences only, no TCR)."""
    stats = DatabaseStats(name="IMGTHLA", description="IMGT/HLA allele sequences (HLA only, no TCR)")

    # Count FASTA files for reference
    fasta_files = list(db_path.glob("**/*.fasta")) + list(db_path.glob("**/*.fa"))
    stats.file_count = len(fasta_files)

    # This database doesn't contain TCR sequences, just HLA alleles
    stats.stats.total_sequences = 0

    return stats


def analyze_netmhcpan(db_path: Path) -> DatabaseStats:
    """Analyze netMHCpan database (MHC binding predictions, no TCR)."""
    stats = DatabaseStats(name="netMHCpan", description="MHC binding prediction data (no TCR)")

    # This is prediction tool data, not TCR sequences
    stats.file_count = len(list(db_path.glob("**/*")))
    stats.stats.total_sequences = 0

    return stats


# =============================================================================
# Directory Scanning
# =============================================================================

def scan_study_directory(study_path: Path, category: str) -> StudyStats:
    """Scan a single study directory."""
    study = StudyStats(name=study_path.name, category=category)

    # Find all relevant files
    files_to_process = []
    for ext in VALID_EXTENSIONS:
        files_to_process.extend(study_path.rglob(f"*{ext}"))

    # Filter out non-TCR files (e.g., gene expression data)
    tcr_files = []
    for f in files_to_process:
        path_str = str(f).lower()
        # Skip gene expression files
        if any(skip in path_str for skip in ['gex/', 'expression', 'matrix', 'features', 'barcodes']):
            continue
        # Skip image/document files
        if any(f.suffix.lower() == ext for ext in ['.pdf', '.png', '.jpg', '.html']):
            continue
        tcr_files.append(f)

    study.file_count = len(tcr_files)

    for file_path in tcr_files:
        try:
            header = read_file_header(file_path)
            file_type = get_file_type(file_path, header)

            # Track file types
            study.file_types[file_type] = study.file_types.get(file_type, 0) + 1

            # Parse based on type
            if file_type in ['bulk_tra', 'bulk_trb', 'bulk_mixed']:
                file_stats = parse_bulk_file(file_path, file_type)
            elif file_type == 'paired_contigs':
                file_stats = parse_paired_contigs(file_path)
            elif file_type == 'paired_clonotypes':
                file_stats = parse_paired_clonotypes(file_path)
            else:
                file_stats = parse_generic_file(file_path)

            study.stats = study.stats + file_stats

        except Exception as e:
            study.errors.append(f"{file_path.name}: {str(e)}")

    return study


def scan_category(category_path: Path, category_name: str) -> List[StudyStats]:
    """Scan all studies in a category."""
    studies = []

    # Get subdirectories (each is a study)
    study_dirs = [d for d in category_path.iterdir()
                  if d.is_dir() and d.name not in SKIP_DIRS]

    for study_dir in tqdm(study_dirs, desc=f"  {category_name}", leave=False):
        study = scan_study_directory(study_dir, category_name)
        if study.stats.total_sequences > 0 or study.file_count > 0:
            studies.append(study)

    return studies


def analyze_databases(databases_path: Path) -> List[DatabaseStats]:
    """Analyze all databases."""
    databases = []

    db_handlers = {
        'VdjDB': analyze_vdjdb,
        'McPASDB': analyze_mcpas,
        'TCRdb': analyze_tcrdb,
        'IEDB': analyze_iedb,
        'ireceptor': analyze_ireceptor,
        'CEDAR': analyze_cedar,
        'WarrenDB': analyze_warrendb,
        'GLIPHDB': analyze_gliphdb,
        'IMGTHLA': analyze_imgthla,
        'netMHCpan': analyze_netmhcpan,
    }

    for db_name, handler in tqdm(db_handlers.items(), desc="  Databases"):
        db_path = databases_path / db_name
        if db_path.exists():
            db_stats = handler(db_path)
            databases.append(db_stats)

    return databases


# =============================================================================
# Reporting
# =============================================================================

def format_number(n: int) -> str:
    """Format number with commas."""
    return f"{n:,}"


def format_percentage(part: int, total: int) -> str:
    """Format percentage."""
    if total == 0:
        return "0.0%"
    return f"{part / total * 100:.1f}%"


def print_report(studies: List[StudyStats], databases: List[DatabaseStats]):
    """Print formatted console report."""

    # Aggregate totals
    total_stats = SequenceStats()
    categories = defaultdict(lambda: {'studies': [], 'stats': SequenceStats()})

    for study in studies:
        total_stats = total_stats + study.stats
        categories[study.category]['studies'].append(study)
        categories[study.category]['stats'] = categories[study.category]['stats'] + study.stats

    db_total_stats = SequenceStats()
    for db in databases:
        db_total_stats = db_total_stats + db.stats

    combined_total = total_stats + db_total_stats

    print("\n" + "=" * 78)
    print("TCR DATA STATISTICS REPORT")
    print("=" * 78)

    # Summary
    print("\nSUMMARY")
    print("-" * 40)
    print(f"{'Total Categories:':<30} {len(categories)}")
    print(f"{'Total Studies:':<30} {len(studies)}")
    print(f"{'Total Databases:':<30} {len([d for d in databases if d.stats.total_sequences > 0])}")
    print(f"{'Total TCR Sequences:':<30} {format_number(combined_total.total_sequences)}")
    print(f"{'  From Studies:':<30} {format_number(total_stats.total_sequences)}")
    print(f"{'  From Databases:':<30} {format_number(db_total_stats.total_sequences)}")

    # Sequence Type Breakdown
    print("\nSEQUENCE TYPE BREAKDOWN")
    print("-" * 40)
    print(f"{'TCR Alpha (TRA):':<30} {format_number(combined_total.tra_sequences):>15} ({format_percentage(combined_total.tra_sequences, combined_total.total_sequences)})")
    print(f"{'TCR Beta (TRB):':<30} {format_number(combined_total.trb_sequences):>15} ({format_percentage(combined_total.trb_sequences, combined_total.total_sequences)})")
    print(f"{'Paired (TRA+TRB):':<30} {format_number(combined_total.paired_sequences):>15} ({format_percentage(combined_total.paired_sequences, combined_total.total_sequences)})")
    print(f"{'Unpaired (single chain):':<30} {format_number(combined_total.unpaired_sequences):>15} ({format_percentage(combined_total.unpaired_sequences, combined_total.total_sequences)})")

    # Annotation Coverage
    print("\nANNOTATION COVERAGE")
    print("-" * 40)
    print(f"{'With Peptide:':<30} {format_number(combined_total.with_peptide):>15} ({format_percentage(combined_total.with_peptide, combined_total.total_sequences)})")
    print(f"{'With MHC:':<30} {format_number(combined_total.with_mhc):>15} ({format_percentage(combined_total.with_mhc, combined_total.total_sequences)})")
    print(f"{'TCR-only:':<30} {format_number(combined_total.tcr_only):>15} ({format_percentage(combined_total.tcr_only, combined_total.total_sequences)})")

    # By Category
    print("\nBY CATEGORY")
    print("-" * 40)
    for cat_name, cat_data in sorted(categories.items()):
        cat_stats = cat_data['stats']
        n_studies = len(cat_data['studies'])
        print(f"\n{cat_name}: {n_studies} studies, {format_number(cat_stats.total_sequences)} sequences")
        print(f"  Paired: {format_percentage(cat_stats.paired_sequences, cat_stats.total_sequences)}, "
              f"TRA: {format_percentage(cat_stats.tra_sequences, cat_stats.total_sequences)}, "
              f"TRB: {format_percentage(cat_stats.trb_sequences, cat_stats.total_sequences)}")

        # Top studies by sequence count
        sorted_studies = sorted(cat_data['studies'], key=lambda s: s.stats.total_sequences, reverse=True)
        for study in sorted_studies[:5]:  # Top 5
            if study.stats.total_sequences > 0:
                print(f"    {study.name[:30]:<32} {format_number(study.stats.total_sequences):>12}")

    # Database Breakdown
    print("\n\nDATABASE BREAKDOWN")
    print("-" * 40)
    for db in sorted(databases, key=lambda d: d.stats.total_sequences, reverse=True):
        if db.stats.total_sequences > 0:
            print(f"\n{db.name}: {format_number(db.stats.total_sequences)} sequences")
            print(f"  Paired: {format_percentage(db.stats.paired_sequences, db.stats.total_sequences)}, "
                  f"TRA: {format_percentage(db.stats.tra_sequences, db.stats.total_sequences)}, "
                  f"TRB: {format_percentage(db.stats.trb_sequences, db.stats.total_sequences)}")
            print(f"  With peptide: {format_percentage(db.stats.with_peptide, db.stats.total_sequences)}, "
                  f"With MHC: {format_percentage(db.stats.with_mhc, db.stats.total_sequences)}")
        else:
            print(f"\n{db.name}: {db.description}")

    print("\n" + "=" * 78)
    print("END OF REPORT")
    print("=" * 78)


def save_json_report(studies: List[StudyStats], databases: List[DatabaseStats], output_path: Path):
    """Save detailed JSON report."""

    # Convert to serializable format
    def stats_to_dict(stats: SequenceStats) -> dict:
        return asdict(stats)

    report = {
        'summary': {
            'total_categories': len(set(s.category for s in studies)),
            'total_studies': len(studies),
            'total_databases': len([d for d in databases if d.stats.total_sequences > 0]),
        },
        'studies': [],
        'databases': [],
        'by_category': {}
    }

    # Studies
    for study in studies:
        report['studies'].append({
            'name': study.name,
            'category': study.category,
            'file_count': study.file_count,
            'stats': stats_to_dict(study.stats),
            'file_types': study.file_types,
            'errors': study.errors
        })

    # Databases
    for db in databases:
        report['databases'].append({
            'name': db.name,
            'description': db.description,
            'file_count': db.file_count,
            'stats': stats_to_dict(db.stats),
            'errors': db.errors
        })

    # By category
    categories = defaultdict(lambda: {'studies': [], 'stats': SequenceStats()})
    for study in studies:
        categories[study.category]['studies'].append(study.name)
        categories[study.category]['stats'] = categories[study.category]['stats'] + study.stats

    for cat_name, cat_data in categories.items():
        report['by_category'][cat_name] = {
            'study_count': len(cat_data['studies']),
            'studies': cat_data['studies'],
            'stats': stats_to_dict(cat_data['stats'])
        }

    # Calculate totals
    total_stats = SequenceStats()
    for study in studies:
        total_stats = total_stats + study.stats
    for db in databases:
        total_stats = total_stats + db.stats

    report['totals'] = stats_to_dict(total_stats)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)

    print(f"\nJSON report saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze TCR raw data statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tcr_data_statistics.py --path /path/to/raw_data
    python tcr_data_statistics.py --path /path/to/raw_data --output stats.json
"""
    )

    parser.add_argument("--path", type=Path, required=True,
                       help="Path to raw_data directory")
    parser.add_argument("--output", type=Path, default=None,
                       help="Path for JSON output (default: tcr_statistics.json in current dir)")

    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: Path does not exist: {args.path}")
        return

    output_path = args.output or Path("tcr_statistics.json")

    print(f"Analyzing TCR data in: {args.path}")
    print("=" * 78)

    # Get categories (top-level directories except 'databases')
    all_dirs = [d for d in args.path.iterdir()
                if d.is_dir() and d.name not in SKIP_DIRS]

    category_dirs = [d for d in all_dirs if d.name != 'databases']
    databases_path = args.path / 'databases'

    # Scan studies
    print("\nScanning study categories...")
    all_studies = []
    for cat_dir in tqdm(category_dirs, desc="Categories"):
        studies = scan_category(cat_dir, cat_dir.name)
        all_studies.extend(studies)

    # Analyze databases
    print("\nAnalyzing databases...")
    databases = []
    if databases_path.exists():
        databases = analyze_databases(databases_path)

    # Generate reports
    print_report(all_studies, databases)
    save_json_report(all_studies, databases, output_path)


if __name__ == "__main__":
    main()
