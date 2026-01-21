"""
Output parsers for peptide-MHC prediction tools.

Parses prediction results from:
- netMHCpan 4.2 (MHC Class I)
- netMHCIIpan 4.3 (MHC Class II)
- MHCflurry (MHC Class I)
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd


def parse_netmhcpan_output(
    output_file: Union[str, Path],
    include_raw: bool = False,
) -> pd.DataFrame:
    """
    Parse netMHCpan 4.2 XLS output file.

    netMHCpan outputs tab-separated data with columns:
    - Pos: Position in protein (0 for single peptides)
    - MHC: MHC allele
    - Peptide: Peptide sequence
    - Core: Core binding sequence (9-mer)
    - Of: Offset of core in peptide
    - Gp: Gap position
    - Gl: Gap length
    - Ip: Insertion position
    - Il: Insertion length
    - Icore: Interaction core
    - Identity: Protein identifier
    - Score_EL: EL prediction score
    - %Rank_EL: EL percentile rank
    - Score_BA: BA prediction score (if -BA flag used)
    - %Rank_BA: BA percentile rank (if -BA flag used)
    - Aff(nM): Binding affinity in nM (if -BA flag used)
    - BindLevel: SB (strong), WB (weak), or empty

    Args:
        output_file: Path to netMHCpan XLS output file
        include_raw: Whether to include all raw columns

    Returns:
        DataFrame with parsed predictions
    """
    output_file = Path(output_file)

    if not output_file.exists():
        raise FileNotFoundError(f"Output file not found: {output_file}")

    # Read the file, skipping comment lines
    lines = []
    header_line = None
    data_start = False

    with open(output_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Header line starts after the dashed separator
            if line.startswith('---'):
                data_start = True
                continue

            if data_start:
                if header_line is None:
                    # This is the header row
                    header_line = line
                else:
                    # This is a data row
                    lines.append(line)

    if not lines:
        return pd.DataFrame()

    # Parse header
    header = header_line.split()

    # Parse data rows
    data = []
    for line in lines:
        # Skip separator lines
        if line.startswith('-'):
            continue
        parts = line.split()
        if len(parts) >= len(header):
            data.append(parts[:len(header)])

    if not data:
        return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame(data, columns=header)

    # Standardize column names and convert types
    rename_map = {
        'MHC': 'allele',
        'Peptide': 'peptide',
        'Core': 'core',
        'Score_EL': 'score_el',
        '%Rank_EL': 'rank_el',
        'Score_BA': 'score_ba',
        '%Rank_BA': 'rank_ba',
        'Aff(nM)': 'affinity_nm',
        'BindLevel': 'bind_level',
    }

    df = df.rename(columns=rename_map)

    # Convert numeric columns
    numeric_cols = ['score_el', 'rank_el', 'score_ba', 'rank_ba', 'affinity_nm']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Select output columns
    output_cols = ['peptide', 'allele', 'core', 'score_el', 'rank_el']
    if 'score_ba' in df.columns:
        output_cols.extend(['score_ba', 'rank_ba', 'affinity_nm'])
    if 'bind_level' in df.columns:
        output_cols.append('bind_level')

    if include_raw:
        return df
    else:
        available_cols = [c for c in output_cols if c in df.columns]
        return df[available_cols]


def parse_netmhcpan_stdout(stdout: str) -> pd.DataFrame:
    """
    Parse netMHCpan output from stdout (when not using -xls).

    Args:
        stdout: Raw stdout from netMHCpan

    Returns:
        DataFrame with parsed predictions
    """
    lines = stdout.strip().split('\n')
    data = []
    in_results = False
    header = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect start of results
        if '----' in line:
            in_results = True
            continue

        if in_results:
            if header is None:
                # Parse header
                header = line.split()
                continue

            # Parse data row
            parts = line.split()
            if len(parts) >= 5:  # Minimum expected columns
                data.append(parts)

    if not data or header is None:
        return pd.DataFrame()

    # Truncate rows to header length
    data = [row[:len(header)] for row in data if len(row) >= len(header)]

    df = pd.DataFrame(data, columns=header)

    # Standardize column names
    rename_map = {
        'MHC': 'allele',
        'Peptide': 'peptide',
        'Core': 'core',
        'EL_score': 'score_el',
        '%Rank_EL': 'rank_el',
        'Score_EL': 'score_el',
        'BA_score': 'score_ba',
        '%Rank_BA': 'rank_ba',
        'Aff(nM)': 'affinity_nm',
    }

    df = df.rename(columns=rename_map)

    # Convert numeric columns
    numeric_cols = ['score_el', 'rank_el', 'score_ba', 'rank_ba', 'affinity_nm']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def parse_netmhciipan_output(
    output_file: Union[str, Path],
    include_raw: bool = False,
) -> pd.DataFrame:
    """
    Parse netMHCIIpan 4.3 XLS output file (multi-allele wide format).

    netMHCIIpan -xls outputs a wide format where:
    - Line 1: Allele names (tab-separated, empty cells for shared columns)
    - Line 2: Column headers repeated for each allele
    - Line 3+: Data rows with predictions for all alleles

    Shared columns: Pos, Peptide, ID, Target
    Per-allele columns: Core, Inverted, Score, Rank, Score_BA, nM, Rank_BA

    Args:
        output_file: Path to netMHCIIpan XLS output file
        include_raw: Whether to include all raw columns

    Returns:
        DataFrame with parsed predictions (long format: one row per peptide-allele pair)
    """
    output_file = Path(output_file)

    if not output_file.exists():
        raise FileNotFoundError(f"Output file not found: {output_file}")

    # Read the file
    with open(output_file, 'r') as f:
        lines = [line.rstrip('\n') for line in f if line.strip()]

    if len(lines) < 3:
        return pd.DataFrame()

    # Parse line 1: allele names
    # Format: \t\t\tAllele1\t\t\t\t\t\t\t\tAllele2\t\t\t...
    allele_line = lines[0].split('\t')
    alleles = [a for a in allele_line if a.strip()]

    # Parse line 2: column headers
    header_line = lines[1].split('\t')

    # Count columns per allele block
    # Shared columns: Pos, Peptide, ID, Target (4 columns)
    # Per-allele columns: Core, Inverted, Score, Rank, Score_BA, nM, Rank_BA (7 columns)
    # Plus Ave and NB at the end (2 columns)
    shared_cols = 4
    per_allele_cols = 7

    # Parse data rows
    all_predictions = []

    for line in lines[2:]:
        parts = line.split('\t')
        if len(parts) < shared_cols:
            continue

        # Extract shared values
        pos = parts[0] if len(parts) > 0 else ''
        peptide = parts[1] if len(parts) > 1 else ''
        identity = parts[2] if len(parts) > 2 else ''
        target = parts[3] if len(parts) > 3 else ''

        # Extract per-allele predictions
        for i, allele in enumerate(alleles):
            start_idx = shared_cols + i * per_allele_cols

            if start_idx + per_allele_cols > len(parts):
                continue

            try:
                core = parts[start_idx] if len(parts) > start_idx else ''
                inverted = parts[start_idx + 1] if len(parts) > start_idx + 1 else ''
                score_el = parts[start_idx + 2] if len(parts) > start_idx + 2 else ''
                rank_el = parts[start_idx + 3] if len(parts) > start_idx + 3 else ''
                score_ba = parts[start_idx + 4] if len(parts) > start_idx + 4 else ''
                affinity_nm = parts[start_idx + 5] if len(parts) > start_idx + 5 else ''
                rank_ba = parts[start_idx + 6] if len(parts) > start_idx + 6 else ''

                all_predictions.append({
                    'peptide': peptide,
                    'allele': allele,
                    'core': core,
                    'score_el': float(score_el) if score_el else None,
                    'rank_el': float(rank_el) if rank_el else None,
                    'score_ba': float(score_ba) if score_ba else None,
                    'affinity_nm': float(affinity_nm) if affinity_nm else None,
                    'rank_ba': float(rank_ba) if rank_ba else None,
                })
            except (ValueError, IndexError):
                continue

    if not all_predictions:
        return pd.DataFrame()

    df = pd.DataFrame(all_predictions)

    # Select output columns
    output_cols = ['peptide', 'allele', 'core', 'score_el', 'rank_el']
    if 'score_ba' in df.columns:
        output_cols.extend(['score_ba', 'rank_ba', 'affinity_nm'])

    if include_raw:
        return df
    else:
        available_cols = [c for c in output_cols if c in df.columns]
        return df[available_cols]


def parse_netmhciipan_stdout(stdout: str) -> pd.DataFrame:
    """
    Parse netMHCIIpan output from stdout.

    Args:
        stdout: Raw stdout from netMHCIIpan

    Returns:
        DataFrame with parsed predictions
    """
    lines = stdout.strip().split('\n')
    data = []
    in_results = False
    header = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if '----' in line:
            in_results = True
            continue

        if in_results:
            if header is None:
                header = line.split()
                continue

            parts = line.split()
            if len(parts) >= 5:
                data.append(parts)

    if not data or header is None:
        return pd.DataFrame()

    data = [row[:len(header)] for row in data if len(row) >= len(header)]

    df = pd.DataFrame(data, columns=header)

    rename_map = {
        'MHC': 'allele',
        'Peptide': 'peptide',
        'Core': 'core',
        'Score_EL': 'score_el',
        '%Rank_EL': 'rank_el',
        'Score_BA': 'score_ba',
        '%Rank_BA': 'rank_ba',
        'Affinity(nM)': 'affinity_nm',
    }

    df = df.rename(columns=rename_map)

    numeric_cols = ['score_el', 'rank_el', 'score_ba', 'rank_ba', 'affinity_nm']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def parse_mhcflurry_output(
    output_file: Union[str, Path],
    include_raw: bool = False,
) -> pd.DataFrame:
    """
    Parse MHCflurry CSV output file.

    MHCflurry outputs CSV with columns:
    - allele: MHC allele
    - peptide: Peptide sequence
    - mhcflurry_affinity: Predicted binding affinity (nM)
    - mhcflurry_affinity_percentile: Affinity percentile rank
    - mhcflurry_processing_score: Processing score
    - mhcflurry_presentation_score: Presentation score
    - mhcflurry_presentation_percentile: Presentation percentile rank

    Args:
        output_file: Path to MHCflurry output CSV
        include_raw: Whether to include all raw columns

    Returns:
        DataFrame with parsed predictions
    """
    output_file = Path(output_file)

    if not output_file.exists():
        raise FileNotFoundError(f"Output file not found: {output_file}")

    df = pd.read_csv(output_file)

    # Standardize column names
    rename_map = {
        'mhcflurry_affinity': 'affinity_nm',
        'mhcflurry_affinity_percentile': 'rank_affinity',
        'mhcflurry_processing_score': 'processing_score',
        'mhcflurry_presentation_score': 'presentation_score',
        'mhcflurry_presentation_percentile': 'rank_presentation',
    }

    df = df.rename(columns=rename_map)

    # Convert affinity to EL-like score (higher = better binding)
    # Use inverse log transform similar to netMHC tools
    if 'affinity_nm' in df.columns:
        df['score_affinity'] = 1 - (df['affinity_nm'].apply(
            lambda x: min(x, 50000) / 50000 if pd.notna(x) else 1.0
        ))

    # Select output columns
    output_cols = ['peptide', 'allele', 'affinity_nm', 'rank_affinity',
                   'score_affinity', 'processing_score', 'presentation_score',
                   'rank_presentation']

    if include_raw:
        return df
    else:
        available_cols = [c for c in output_cols if c in df.columns]
        return df[available_cols]


def unify_predictions(
    predictions: Dict[str, pd.DataFrame],
    tool_name: str,
) -> pd.DataFrame:
    """
    Unify predictions from a tool into a standard format.

    Standard output columns:
    - peptide: Peptide sequence
    - allele: MHC allele
    - score: Primary prediction score (higher = better binding)
    - rank: Percentile rank (lower = better binding)
    - affinity_nm: Binding affinity in nM (if available)
    - tool: Tool name

    Args:
        predictions: Dict mapping allele to prediction DataFrame
        tool_name: Name of the prediction tool

    Returns:
        Unified DataFrame with standard columns
    """
    all_preds = []

    for allele, df in predictions.items():
        if len(df) == 0:
            continue

        unified = df.copy()
        unified['tool'] = tool_name

        # Map score column based on tool
        if tool_name == 'netmhcpan':
            if 'score_el' in unified.columns:
                unified['score'] = unified['score_el']
            if 'rank_el' in unified.columns:
                unified['rank'] = unified['rank_el']
        elif tool_name == 'netmhciipan':
            if 'score_el' in unified.columns:
                unified['score'] = unified['score_el']
            if 'rank_el' in unified.columns:
                unified['rank'] = unified['rank_el']
        elif tool_name == 'mhcflurry':
            if 'score_affinity' in unified.columns:
                unified['score'] = unified['score_affinity']
            if 'rank_affinity' in unified.columns:
                unified['rank'] = unified['rank_affinity']

        all_preds.append(unified)

    if not all_preds:
        return pd.DataFrame()

    combined = pd.concat(all_preds, ignore_index=True)

    # Select standard columns
    standard_cols = ['peptide', 'allele', 'score', 'rank', 'affinity_nm', 'tool']
    available_cols = [c for c in standard_cols if c in combined.columns]

    return combined[available_cols]


def merge_predictions_with_labels(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    peptide_col: str = 'peptide',
    allele_col: str = 'allele',
    label_allele_col: str = 'mhc_one_id',
) -> pd.DataFrame:
    """
    Merge predictions with ground truth labels.

    Args:
        predictions: DataFrame with predictions
        labels: DataFrame with ground truth labels
        peptide_col: Peptide column name in predictions
        allele_col: Allele column name in predictions
        label_allele_col: Allele column name in labels

    Returns:
        Merged DataFrame with predictions and labels
    """
    # Standardize labels DataFrame
    labels_std = labels.copy()
    labels_std['_peptide'] = labels_std['peptide']
    labels_std['_allele'] = labels_std[label_allele_col]

    # Add label column (1 for positive pairs in labels)
    labels_std['label'] = 1

    # Create merge keys
    predictions['_peptide'] = predictions[peptide_col]
    predictions['_allele'] = predictions[allele_col]

    # Merge
    merged = predictions.merge(
        labels_std[['_peptide', '_allele', 'label']],
        on=['_peptide', '_allele'],
        how='left',
    )

    # Fill missing labels with 0 (negatives)
    merged['label'] = merged['label'].fillna(0).astype(int)

    # Clean up
    merged = merged.drop(columns=['_peptide', '_allele'])

    return merged
