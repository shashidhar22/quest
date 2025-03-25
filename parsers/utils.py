import os
import re
import glob
import numpy as np
import pandas as pd

from itertools import product
from collections import OrderedDict

def parse_imgt_four_digit(hla_directory):
    """
    Parses IMGT/HLA FASTA files to extract unique HLA alleles at four-digit resolution.

    Args:
        hla_directory (str): Path to the directory containing FASTA files.

    Returns:
        dict: A dictionary with four-digit HLA allele identifiers as keys
              and nucleotide sequences as values.
    """
    # Collect all relevant FASTA file paths
    file_paths = glob.glob(f"{hla_directory}/*_prot.fasta")
    fasta_dict = OrderedDict()

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            header, sequence = None, []
            for line in file:
                line = line.strip()
                if line.startswith('>'):  # Process header lines
                    if header and header not in fasta_dict:
                        fasta_dict[header] = ''.join(sequence)

                    # Extract HLA ID at four-digit resolution
                    hla_id_full = line.split()[1]  # E.g., "A*01:01:01:06"
                    hla_id_four_digit = ":".join(hla_id_full.split(":")[:2])  # "A*01:01"

                    # Only process if this four-digit HLA ID is new
                    if hla_id_four_digit not in fasta_dict:
                        header = hla_id_four_digit
                    else:
                        header = None  # Skip if already processed

                    sequence = []  # Reset sequence buffer for the new record
                else:
                    sequence.append(line)  # Collect sequence lines

            # Save the last sequence if valid
            if header and header not in fasta_dict:
                fasta_dict[header] = ''.join(sequence)
    return fasta_dict
    
def get_mhc_sequence(value, fasta_dict):
    """
    Get MHC sequence given MHC allele value. If value contain a statndard HLA allele with
    8-digit resolution, returns the IMGT sequence. If the value also contains an annotation 
    with mutation coordinates (e.g., "mA76Q"), then returns mutated sequence. Returns 
    pd.NA if no higher-resolution matches exist
    """
    try:
        allele, _ = value.split(';', 1)
    except (ValueError, AttributeError):
        allele = value

    return fasta_dict.get(allele, None)

def process_mhc_restriction(mhc):
    """
    Process MHC restriction to split alpha and beta chains or reassign class I alleles.

    Args:
        mhc (str): The MHC restriction value.

    Returns:
        tuple: (mhc_restriction, mhc_restriction_two)
    """
    beta_chain_prefixes = ['DRB', 'DPB', 'DQB']
    alpha_chain_prefixes = ['DRA', 'DPA', 'DQA']
    
    if pd.isna(mhc):  # This handles None or NaN values
        return np.nan, np.nan
    elif '/' in mhc:
        alpha, beta = mhc.split('/')
        if any(prefix in beta for prefix in beta_chain_prefixes):
            return alpha, beta
    elif any(prefix in mhc for prefix in beta_chain_prefixes):
        return np.nan, mhc
    else:
        return mhc, np.nan


def transform_mhc_restriction(values, fasta_dict):
    """
    Partition-wise transformation of MHC restriction values.

    Args:
        values (pd.Series): Column of MHC restriction values.
        fasta_dict (dict): Dictionary of known HLA alleles.

    Returns:
        pd.Series: Transformed values.
    """
    sorted_fasta_keys = sorted(fasta_dict.keys())

    def normalize_input(value):
        try:
            base, annotations = extract_annotations(value)
            if base.startswith("HLA-"):
                base = base.replace("HLA-", "")
            base = base.replace("-", ":")
            if re.match(r'^[ABC]\*?\d{1,2}$', base):
                base = re.sub(r'^([ABC])(\d{1,2})$', r'\1*\2', base)
                base = re.sub(r'([ABC])\*(\d{1})$', r'\1*0\2', base)
            elif re.match(r'^[ABC]\*?\d{1,2}:\d{1,2}$', base):
                base = re.sub(r'([ABC])\*(\d{1}):(\d{1,2})$', r'\1*0\2:\3', base)
                base = re.sub(r'([ABC])\*(\d{2}):(\d{1})$', r'\1*\2:0\3', base)
            return base, annotations
        except (ValueError, AttributeError):
            return pd.NA, None

    def extract_annotations(value):
        match = re.search(r'\s+([A-Za-z0-9, ]+)\s+mutant$', value)
        if match:
            annotations = "_".join([f"m{m.strip()}" for m in match.group(1).split(",")])
            return value[:match.start()].strip(), annotations
        return value.strip(), None

    def find_highest_resolution(base):
        for key in sorted_fasta_keys:
            if key.startswith(base):
                return key
        return pd.NA

    def process_value(value):
        try:
            normalized_value, annotations = normalize_input(value)
            patterns = {
                r'^[ABC]\*\d{2}(:\d{2}){1,3}$': normalized_value,
                r'^DR[AB][1345]?\*\d{2}(:\d{2}){0,2}$': normalized_value,
                r'^D[PQ][AB][12]?\*\d{2}(:\d{2}){0,2}$': normalized_value,
                r'^[ABC]\*\d{2}$': normalized_value,
                r'^DR[AB][1345]?\*\d{2}$': normalized_value,
                r'^D[PQ][AB][12]?\*\d{2}$': normalized_value,
            }

            resolved = pd.NA
            for pattern, allele in patterns.items():
                if re.match(pattern, allele):
                    resolved = find_highest_resolution(allele)
                    break

            if pd.notna(resolved) and annotations:
                return f"{resolved}_{annotations}"
            return resolved
        except TypeError:
            return pd.NA

    return values.map(process_value)


def format_combined_tcell(barcode, index, tra, trb):
    """
    Row-wise wrapper for formatting a combined T-cell for Dask compatibility.

    Args:
        row (pd.Series): A single row from a DataFrame.

    Returns:
        dict: Formatted T-cell information as a dictionary.
    """
    #barcode = row['barcode']
    #index = row['index']
    #tra = (row['tra'], row['trav_gene'], row['trad_gene'], row['traj_gene']) if pd.notna(row['tra']) else None
    #trb = (row['trb'], row['trbv_gene'], row['trbd_gene'], row['trbj_gene']) if pd.notna(row['trb']) else None

    tid = f"{barcode}_{index}"
    result_dict = {'tid': tid}

    # Populate TRA information
    if tra:
        result_dict.update({
            'trav_gene': tra[1], 'trad_gene': tra[2], 'traj_gene': tra[3],
            'tra': tra[0]
        })

    # Populate TRB information
    if trb:
        result_dict.update({
            'trbv_gene': trb[1], 'trbd_gene': trb[2], 'trbj_gene': trb[3],
            'trb': trb[0]
        })

   
    # Create the sequence column
    result_dict['sequence'] = ' '.join(
        filter(None, [result_dict.get('tra'), result_dict.get('trb')])
    ) + ';'

    return result_dict

def parse_junction_aa(row):
    """
    Parse the `cdr3s_aa` column and extract TRA and TRB sequences.

    Args:
        row (str): Row value from `cdr3s_aa`.

    Returns:
        dict: Extracted TRA and TRB sequences.
    """
    result_dict = {}
    for chains in row.split(';'):
        if ':' not in chains:
            continue
        chain_name, chain_seq = chains.split(':', 1)
        if chain_name == "TRA":
            result_dict['tra'] = chain_seq
        elif chain_name == "TRB":
            result_dict['trb'] = chain_seq
    return result_dict

# Group by barcode and process TRA and TRB combinations
def process_barcode_group(df):
    formatted_contigs = []
    for barcode, group in df.groupby('barcode'):
        tra_seqs = group[group['chain'] == 'TRA'][['cdr3', 'v_gene', 'd_gene', 'j_gene']].apply(tuple, axis=1).tolist()
        trb_seqs = group[group['chain'] == 'TRB'][['cdr3', 'v_gene', 'd_gene', 'j_gene']].apply(tuple, axis=1).tolist()

        # Combine TRA and TRB sequences
        if tra_seqs and trb_seqs:
            for index, (tra, trb) in enumerate(product(tra_seqs, trb_seqs), start=1):
                result_dict = format_combined_tcell(barcode, index, tra, trb, tra_only=False)
                formatted_contigs.append(result_dict)
        elif tra_seqs:  # TRA only
            for index, tra in enumerate(tra_seqs, start=1):
                result_dict = format_combined_tcell(barcode, index, tra, '', tra_only=True)
                formatted_contigs.append(result_dict)
        elif trb_seqs:  # TRB only
            for index, trb in enumerate(trb_seqs, start=1):
                result_dict = format_combined_tcell(barcode, index, '', trb, tra_only=False)
                formatted_contigs.append(result_dict)
    return pd.DataFrame(formatted_contigs)

def standardize_sequence(df):
    """
    Ensure the Dask DataFrame has the fixed set of columns.
    Missing columns are added with default values (None).
    Columns are reordered to match the fixed set.

    Args:
        df (dd.DataFrame): Input Dask DataFrame.

    Returns:
        dd.DataFrame: Standardized Dask DataFrame.
    """
    fixed_columns = [
        'source', 'tid', 'tra', 'trad_gene', 'traj_gene', 'trav_gene',
        'trb', 'trbd_gene', 'trbj_gene', 'trbv_gene', 'peptide', 'mhc_one',
        'mhc_two', 'sequence'
    ]

    # Define metadata for all columns
    meta = OrderedDict({col: 'string[pyarrow]' for col in fixed_columns})

    # Ensure all columns are present by adding missing columns with default value None
    def add_missing_columns(partition):
        for col in fixed_columns:
            if col not in partition.columns:
                partition[col] = ''
        partition = partition[meta.keys()]
        return partition

    def reorder_columns(df, columns):
    # Reindex with the correct order
        return df[columns]

    # Apply the function to each partition and enforce fixed columns
    df = df.map_partitions(add_missing_columns, meta=meta)

    # Reorder columns to match the fixed set
    df = df.map_partitions(reorder_columns, columns=fixed_columns)

    return df

def standardize_mri(df):
    """
    Ensure the Dask DataFrame has the fixed set of columns.
    Missing columns are added with default values (None).
    Columns are reordered to match the fixed set.

    Args:
        df (dd.DataFrame): Input Dask DataFrame.

    Returns:
        dd.DataFrame: Standardized Dask DataFrame.
    """
    fixed_columns = ['tid', 'tra', 'trad_gene', 'traj_gene', 'trav_gene', 'trb',
        'trbd_gene', 'trbj_gene', 'trbv_gene', 'peptide', 'mhc_one', 'mhc_two', 
        'sequence', 'repertoire_id', 'study_id', 'category', 'molecule_type',
        'host_organism', 'source']

    # Define metadata for all columns
    meta = OrderedDict({col: 'string[pyarrow]' for col in fixed_columns})

    # Ensure all columns are present by adding missing columns with default value None
    def add_missing_columns(partition):
        for col in fixed_columns:
            if col not in partition.columns:
                partition[col] = ''
        partition = partition[meta.keys()]
        return partition

    def reorder_columns(df, columns):
    # Reindex with the correct order
        return df[columns]

    # Apply the function to each partition and enforce fixed columns
    df = df.map_partitions(add_missing_columns, meta=meta)

    # Reorder columns to match the fixed set
    df = df.map_partitions(reorder_columns, columns=fixed_columns)

    return df