"""
Input formatters for peptide-MHC prediction tools.

Handles data formatting and allele name conversion for:
- netMHCpan 4.2 (MHC Class I)
- netMHCIIpan 4.3 (MHC Class II)
- MHCflurry (MHC Class I)
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# Allele conversion patterns
# Standard HLA format: HLA-A*02:01
# netMHCpan format: HLA-A*02:01 (same as standard)
# netMHCIIpan format: DRB1_0101, HLA-DRB1*01:01, etc.
# MHCflurry format: HLA-A0201 (no colon, no asterisk)


def convert_allele_netmhcpan(allele: str) -> Optional[str]:
    """
    Convert allele name to netMHCpan format.

    netMHCpan 4.2 accepts HLA alleles in format: HLA-A02:01 (no asterisk)
    Note: The tool internally uses asterisks but input must NOT have them.

    Args:
        allele: Input allele name (e.g., "HLA-A*02:01", "A*02:01", "A0201")

    Returns:
        Formatted allele string for netMHCpan, or None if conversion fails
    """
    if not allele or pd.isna(allele):
        return None

    allele = str(allele).strip().upper()

    # Format: HLA-A*02:01 -> HLA-A02:01 (remove asterisk)
    match = re.match(r'^HLA-([ABC])\*(\d{2}):(\d{2})(:\d{2})?$', allele)
    if match:
        return f"HLA-{match.group(1)}{match.group(2)}:{match.group(3)}"

    # Already in netMHCpan format: HLA-A02:01
    if re.match(r'^HLA-[ABC]\d{2}:\d{2}$', allele):
        return allele

    # Format: A*02:01 -> HLA-A02:01
    match = re.match(r'^([ABC])\*(\d{2}):(\d{2})$', allele)
    if match:
        return f"HLA-{match.group(1)}{match.group(2)}:{match.group(3)}"

    # Format: HLA-A0201 or A0201 -> HLA-A02:01
    match = re.match(r'^(HLA-)?([ABC])(\d{2})(\d{2})$', allele)
    if match:
        gene = match.group(2)
        g1 = match.group(3)
        g2 = match.group(4)
        return f"HLA-{gene}{g1}:{g2}"

    # Format: HLA-A*0201 -> HLA-A02:01
    match = re.match(r'^HLA-([ABC])\*(\d{2})(\d{2})$', allele)
    if match:
        gene = match.group(1)
        g1 = match.group(2)
        g2 = match.group(3)
        return f"HLA-{gene}{g1}:{g2}"

    # Mouse MHC (H-2)
    if allele.startswith('H-2') or allele.startswith('H2'):
        # Return as-is, netMHCpan handles H-2 alleles
        return allele.replace('H2', 'H-2')

    return None


def convert_allele_netmhciipan(allele: str) -> Optional[str]:
    """
    Convert allele name to netMHCIIpan format.

    netMHCIIpan 4.3 accepts alleles in different formats:
    - DRB alleles: DRB1_0101 (underscore separator)
    - DQ/DP alleles: HLA-DQA10101-DQB10501 (paired alpha-beta)

    For HLA-DR: HLA-DRB1*01:01 -> DRB1_0101
    For HLA-DQ: HLA-DQA1*01:01/HLA-DQB1*05:01 -> HLA-DQA10101-DQB10501
    For HLA-DP: HLA-DPA1*01:03/HLA-DPB1*04:01 -> HLA-DPA10103-DPB10401

    For beta-chain-only DQ/DP alleles, default alpha chains are assumed:
    - DQB1*05:01 -> HLA-DQA10101-DQB10501 (assumes DQA1*01:01)
    - DPB1*04:01 -> HLA-DPA10103-DPB10401 (assumes DPA1*01:03)

    Args:
        allele: Input allele name

    Returns:
        Formatted allele string for netMHCIIpan, or None if conversion fails
    """
    if not allele or pd.isna(allele):
        return None

    allele = str(allele).strip().upper()

    # Already in netMHCIIpan format: DRB1_0101
    if re.match(r'^DRB[135]_\d{4}$', allele):
        return allele

    # Format: HLA-DRB1*01:01 -> DRB1_0101
    match = re.match(r'^HLA-(DRB[135])\*(\d{2}):(\d{2})$', allele)
    if match:
        gene = match.group(1)
        g1 = match.group(2)
        g2 = match.group(3)
        return f"{gene}_{g1}{g2}"

    # Format: DRB1*01:01 -> DRB1_0101
    match = re.match(r'^(DRB[135])\*(\d{2}):(\d{2})$', allele)
    if match:
        gene = match.group(1)
        g1 = match.group(2)
        g2 = match.group(3)
        return f"{gene}_{g1}{g2}"

    # Format: HLA-DRB10101 or DRB10101 -> DRB1_0101
    match = re.match(r'^(HLA-)?(DRB[135])(\d{2})(\d{2})$', allele)
    if match:
        gene = match.group(2)
        g1 = match.group(3)
        g2 = match.group(4)
        return f"{gene}_{g1}{g2}"

    # Handle DQ/DP paired alleles
    # Format: HLA-DQA1*01:01-DQB1*05:01 -> HLA-DQA10101-DQB10501
    match = re.match(
        r'^HLA-(D[QP]A1)\*(\d{2}):(\d{2})[-/](HLA-)?(D[QP]B1)\*(\d{2}):(\d{2})$',
        allele
    )
    if match:
        alpha = f"{match.group(1)}{match.group(2)}{match.group(3)}"
        beta = f"{match.group(5)}{match.group(6)}{match.group(7)}"
        return f"HLA-{alpha}-{beta}"

    # Handle DQ beta-chain only alleles with default alpha chain
    # Format: DQB1*05:01 -> HLA-DQA10101-DQB10501 (assumes DQA1*01:01)
    # Format: HLA-DQB1*05:01 -> HLA-DQA10101-DQB10501
    match = re.match(r'^(HLA-)?(DQB1)\*(\d{2}):(\d{2})$', allele)
    if match:
        beta = f"DQB1{match.group(3)}{match.group(4)}"
        # Default alpha chain DQA1*01:01
        return f"HLA-DQA10101-{beta}"

    # Handle DP beta-chain only alleles with default alpha chain
    # Format: DPB1*04:01 -> HLA-DPA10103-DPB10401 (assumes DPA1*01:03)
    # Format: HLA-DPB1*04:01 -> HLA-DPA10103-DPB10401
    match = re.match(r'^(HLA-)?(DPB1)\*(\d{2}):(\d{2})$', allele)
    if match:
        beta = f"DPB1{match.group(3)}{match.group(4)}"
        # Default alpha chain DPA1*01:03 (most common)
        return f"HLA-DPA10103-{beta}"

    # Mouse MHC Class II
    if 'H-2-I' in allele or 'H2-I' in allele:
        # Return normalized format
        return allele.replace('H2-', 'H-2-')

    return None


def convert_allele_mhcflurry(allele: str) -> Optional[str]:
    """
    Convert allele name to MHCflurry format.

    MHCflurry accepts alleles in format: HLA-A0201 (no colon)

    Args:
        allele: Input allele name

    Returns:
        Formatted allele string for MHCflurry, or None if conversion fails
    """
    if not allele or pd.isna(allele):
        return None

    allele = str(allele).strip().upper()

    # Already in MHCflurry format: HLA-A0201
    if re.match(r'^HLA-[ABC]\d{4}$', allele):
        return allele

    # Format: HLA-A*02:01 -> HLA-A0201
    match = re.match(r'^HLA-([ABC])\*(\d{2}):(\d{2})$', allele)
    if match:
        gene = match.group(1)
        g1 = match.group(2)
        g2 = match.group(3)
        return f"HLA-{gene}{g1}{g2}"

    # Format: A*02:01 -> HLA-A0201
    match = re.match(r'^([ABC])\*(\d{2}):(\d{2})$', allele)
    if match:
        gene = match.group(1)
        g1 = match.group(2)
        g2 = match.group(3)
        return f"HLA-{gene}{g1}{g2}"

    # Format: A0201 -> HLA-A0201
    match = re.match(r'^([ABC])(\d{4})$', allele)
    if match:
        gene = match.group(1)
        digits = match.group(2)
        return f"HLA-{gene}{digits}"

    # Format: HLA-A*0201 -> HLA-A0201
    match = re.match(r'^HLA-([ABC])\*(\d{4})$', allele)
    if match:
        gene = match.group(1)
        digits = match.group(2)
        return f"HLA-{gene}{digits}"

    return None


def format_for_netmhcpan(
    df: pd.DataFrame,
    peptide_col: str = "peptide",
    allele_col: str = "mhc_one_id",
    output_dir: Path = None,
) -> Tuple[Path, List[str], Dict[str, str]]:
    """
    Format data for netMHCpan input.

    Creates a peptide file (one peptide per line) and extracts unique alleles.

    Args:
        df: DataFrame with peptide and allele columns
        peptide_col: Name of peptide column
        allele_col: Name of allele column
        output_dir: Directory to write peptide file

    Returns:
        Tuple of (peptide_file_path, list_of_alleles, allele_mapping)
        allele_mapping maps original allele names to converted names
    """
    if output_dir is None:
        output_dir = Path("/tmp")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get unique peptides
    peptides = df[peptide_col].dropna().unique().tolist()

    # Write peptide file
    peptide_file = output_dir / "peptides.txt"
    with open(peptide_file, 'w') as f:
        for pep in peptides:
            f.write(f"{pep}\n")

    # Get unique alleles and convert to netMHCpan format
    original_alleles = df[allele_col].dropna().unique().tolist()
    allele_mapping = {}
    converted_alleles = []

    for orig in original_alleles:
        converted = convert_allele_netmhcpan(orig)
        if converted:
            allele_mapping[orig] = converted
            if converted not in converted_alleles:
                converted_alleles.append(converted)

    return peptide_file, converted_alleles, allele_mapping


def format_for_netmhciipan(
    df: pd.DataFrame,
    peptide_col: str = "peptide",
    allele_col: str = "mhc_one_id",
    output_dir: Path = None,
) -> Tuple[Path, List[str], Dict[str, str]]:
    """
    Format data for netMHCIIpan input.

    Creates a peptide file (one peptide per line) and extracts unique alleles.

    Args:
        df: DataFrame with peptide and allele columns
        peptide_col: Name of peptide column
        allele_col: Name of allele column (typically mhc_one_id for DR beta chain)
        output_dir: Directory to write peptide file

    Returns:
        Tuple of (peptide_file_path, list_of_alleles, allele_mapping)
    """
    if output_dir is None:
        output_dir = Path("/tmp")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get unique peptides
    peptides = df[peptide_col].dropna().unique().tolist()

    # Write peptide file
    peptide_file = output_dir / "peptides.txt"
    with open(peptide_file, 'w') as f:
        for pep in peptides:
            f.write(f"{pep}\n")

    # Get unique alleles and convert to netMHCIIpan format
    original_alleles = df[allele_col].dropna().unique().tolist()
    allele_mapping = {}
    converted_alleles = []

    for orig in original_alleles:
        converted = convert_allele_netmhciipan(orig)
        if converted:
            allele_mapping[orig] = converted
            if converted not in converted_alleles:
                converted_alleles.append(converted)

    return peptide_file, converted_alleles, allele_mapping


def format_for_mhcflurry(
    df: pd.DataFrame,
    peptide_col: str = "peptide",
    allele_col: str = "mhc_one_id",
    output_dir: Path = None,
) -> Tuple[Path, Dict[str, str]]:
    """
    Format data for MHCflurry input.

    Creates a CSV file with 'allele' and 'peptide' columns.

    Args:
        df: DataFrame with peptide and allele columns
        peptide_col: Name of peptide column
        allele_col: Name of allele column
        output_dir: Directory to write input file

    Returns:
        Tuple of (input_csv_path, allele_mapping)
    """
    if output_dir is None:
        output_dir = Path("/tmp")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create allele mapping
    original_alleles = df[allele_col].dropna().unique().tolist()
    allele_mapping = {}

    for orig in original_alleles:
        converted = convert_allele_mhcflurry(orig)
        if converted:
            allele_mapping[orig] = converted

    # Create input dataframe with converted alleles
    input_df = df[[peptide_col, allele_col]].copy()
    input_df.columns = ['peptide', 'original_allele']
    input_df['allele'] = input_df['original_allele'].map(allele_mapping)

    # Filter out rows where allele conversion failed
    input_df = input_df.dropna(subset=['allele'])

    # Select final columns for MHCflurry
    output_df = input_df[['allele', 'peptide']].drop_duplicates()

    # Write CSV
    input_file = output_dir / "mhcflurry_input.csv"
    output_df.to_csv(input_file, index=False)

    return input_file, allele_mapping


def create_allele_batches(
    alleles: List[str],
    batch_size: int = 10,
) -> List[List[str]]:
    """
    Split alleles into batches for processing.

    Some tools work better with fewer alleles per run.

    Args:
        alleles: List of allele strings
        batch_size: Maximum alleles per batch

    Returns:
        List of allele batches
    """
    batches = []
    for i in range(0, len(alleles), batch_size):
        batches.append(alleles[i:i + batch_size])
    return batches


def get_supported_alleles_netmhcpan(tool_path: str) -> List[str]:
    """
    Get list of alleles supported by netMHCpan.

    Args:
        tool_path: Path to netMHCpan executable

    Returns:
        List of supported allele names
    """
    import subprocess

    try:
        result = subprocess.run(
            [tool_path, "-listMHC"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        alleles = []
        for line in result.stdout.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                alleles.append(line)

        return alleles
    except Exception as e:
        print(f"Warning: Could not get supported alleles from netMHCpan: {e}")
        return []


def get_supported_alleles_netmhciipan(tool_path: str) -> List[str]:
    """
    Get list of alleles supported by netMHCIIpan.

    Args:
        tool_path: Path to netMHCIIpan executable

    Returns:
        List of supported allele names
    """
    import subprocess

    try:
        result = subprocess.run(
            [tool_path, "-list"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        alleles = []
        for line in result.stdout.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                alleles.append(line)

        return alleles
    except Exception as e:
        print(f"Warning: Could not get supported alleles from netMHCIIpan: {e}")
        return []


def filter_to_supported_alleles(
    alleles: List[str],
    supported_alleles: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Filter alleles to only those supported by the tool.

    Args:
        alleles: List of alleles to check
        supported_alleles: List of alleles supported by the tool

    Returns:
        Tuple of (supported_list, unsupported_list)
    """
    supported_set = set(supported_alleles)
    supported = []
    unsupported = []

    for allele in alleles:
        if allele in supported_set:
            supported.append(allele)
        else:
            unsupported.append(allele)

    return supported, unsupported
