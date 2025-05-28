import os
import re
import glob
import numpy as np
import pandas as pd

from dask import delayed
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
    file_paths = glob.glob(f"{hla_directory}/*_prot.fasta")
    fasta_dict = OrderedDict()

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            header, sequence = None, []
            for line in file:
                line = line.strip()
                if line.startswith('>'):
                    if header and header not in fasta_dict:
                        fasta_dict[header] = ''.join(sequence)
                    # Extract HLA ID at four-digit resolution
                    hla_id_full = line.split()[1]  # e.g. "A*01:01:01:06"
                    hla_id_four_digit = ":".join(hla_id_full.split(":")[:2])  # "A*01:01"

                    if hla_id_four_digit not in fasta_dict:
                        header = hla_id_four_digit
                    else:
                        header = None  # Skip if already processed

                    sequence = []
                else:
                    sequence.append(line)

            # Save the last sequence if valid
            if header and header not in fasta_dict:
                fasta_dict[header] = ''.join(sequence)

    return fasta_dict


def get_mhc_sequence(value, fasta_dict):
    """
    Given an MHC allele value (e.g. "A*02:01"), return the IMGT sequence from fasta_dict.
    If not found, return None.
    """
    try:
        allele, _ = value.split(';', 1)
    except (ValueError, AttributeError):
        allele = value

    return fasta_dict.get(allele, None)


def process_mhc_restriction(mhc):
    """
    Split MHC restriction into alpha/beta if needed.
    
    Args:
        mhc (str): e.g. "HLA-DRB1*04:05" or "A*01:01"
    Returns:
        tuple(str or nan, str or nan)
    """
    beta_chain_prefixes = ['DRB', 'DPB', 'DQB']
    alpha_chain_prefixes = ['DRA', 'DPA', 'DQA']
    
    if pd.isna(mhc):
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
    Apply a normalization + highest-resolution lookup to each MHC allele in a pandas Series.
    
    Args:
        values (pd.Series): MHC restriction values
        fasta_dict (dict): dictionary of known HLA alleles
    Returns:
        pd.Series: resolved or NA
    """
    sorted_fasta_keys = sorted(fasta_dict.keys())

    def extract_annotations(value):
        match = re.search(r'\s+([A-Za-z0-9, ]+)\s+mutant$', value)
        if match:
            annotations = "_".join([f"m{m.strip()}" for m in match.group(1).split(",")])
            return value[:match.start()].strip(), annotations
        return value.strip(), None

    def normalize_input(value):
        try:
            base, annotations = extract_annotations(value)
            if base.startswith("HLA-"):
                base = base.replace("HLA-", "")
            base = base.replace("-", ":")
            # e.g. "A01:01" -> "A*01:01"
            if re.match(r'^[ABC]\*?\d{1,2}$', base):
                base = re.sub(r'^([ABC])(\d{1,2})$', r'\1*\2', base)
                base = re.sub(r'([ABC])\*(\d{1})$', r'\1*0\2', base)
            elif re.match(r'^[ABC]\*?\d{1,2}:\d{1,2}$', base):
                base = re.sub(r'([ABC])\*(\d{1}):(\d{1,2})$', r'\1*0\2:\3', base)
                base = re.sub(r'([ABC])\*(\d{2}):(\d{1})$', r'\1*\2:0\3', base)
            return base, annotations
        except (ValueError, AttributeError):
            return pd.NA, None

    def find_highest_resolution(base):
        # Linear scan for the first matching key that starts with base
        for key in sorted_fasta_keys:
            if key.startswith(base):
                return key
        return pd.NA

    def process_value(value):
        if pd.isna(value) or not value.strip():
            return pd.NA
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
            if re.match(pattern, str(allele)):
                resolved = find_highest_resolution(allele)
                break

        if pd.notna(resolved) and annotations:
            return f"{resolved}_{annotations}"
        return resolved

    return values.apply(process_value)


def format_combined_tcell(barcode, index, tra, trb):
    """
    Combine a TRA tuple and TRB tuple into a single row dict with a 'sequence' column.
    
    Args:
        barcode (str): e.g. cell barcode
        index (int): sub-clone index for that cell
        tra (tuple or empty string): (cdr3, v_gene, d_gene, j_gene) for alpha
        trb (tuple or empty string): (cdr3, v_gene, d_gene, j_gene) for beta
    Returns:
        dict: e.g. {'tid': 'cell_1', 'tra': 'CASSL...', 'trb': 'CASSE...','sequence': '...'}
    """
    tid = f"{barcode}_{index}"
    result_dict = {'tid': tid}

    # If `tra` is non-empty tuple
    if tra and isinstance(tra, tuple) and len(tra) == 4:
        result_dict.update({
            'tra': tra[0],
            'trav_gene': tra[1],
            'trad_gene': tra[2],
            'traj_gene': tra[3],
        })
    # If `trb` is non-empty tuple
    if trb and isinstance(trb, tuple) and len(trb) == 4:
        result_dict.update({
            'trb': trb[0],
            'trbv_gene': trb[1],
            'trbd_gene': trb[2],
            'trbj_gene': trb[3],
        })

    tra_seq = result_dict.get('tra', '')
    trb_seq = result_dict.get('trb', '')
    joined = " ".join([x for x in [tra_seq, trb_seq] if x])
    result_dict['sequence'] = (joined + ';') if joined else ''

    return result_dict


def parse_junction_aa(row_value):
    """
    Given a string like 'TRA:CATAAA;TRB:CAGGG...' split out 'tra' and 'trb' keys.
    """
    result_dict = {}
    if not row_value or pd.isna(row_value):
        return result_dict
    for pair in row_value.split(';'):
        if ':' not in pair:
            continue
        chain_name, chain_seq = pair.split(':', 1)
        chain_name = chain_name.strip().upper()
        chain_seq = chain_seq.strip()
        if chain_name == "TRA":
            result_dict['tra'] = chain_seq
        elif chain_name == "TRB":
            result_dict['trb'] = chain_seq
    return result_dict


def process_barcode_group(df):
    """
    Example function that groups by 'barcode' in a normal pandas DataFrame
    and forms all TRA x TRB combos using product().
    """
    from itertools import product

    formatted_contigs = []
    for barcode, group in df.groupby('barcode'):
        tra_seqs = group.loc[group['chain'] == 'TRA', ['cdr3','v_gene','d_gene','j_gene']] \
            .apply(tuple, axis=1).tolist()
        trb_seqs = group.loc[group['chain'] == 'TRB', ['cdr3','v_gene','d_gene','j_gene']] \
            .apply(tuple, axis=1).tolist()

        if tra_seqs and trb_seqs:
            for idx, (tra, trb) in enumerate(product(tra_seqs, trb_seqs), start=1):
                result_dict = format_combined_tcell(barcode, idx, tra, trb)
                formatted_contigs.append(result_dict)
        elif tra_seqs:
            for idx, tra in enumerate(tra_seqs, start=1):
                result_dict = format_combined_tcell(barcode, idx, tra, '')
                formatted_contigs.append(result_dict)
        elif trb_seqs:
            for idx, trb in enumerate(trb_seqs, start=1):
                result_dict = format_combined_tcell(barcode, idx, '', trb)
                formatted_contigs.append(result_dict)

    return pd.DataFrame(formatted_contigs)


def standardize_sequence(df):
    """
    Standardize an in-memory pandas DataFrame to ensure it has a fixed set of columns
    for sequence data. Missing columns are added with default values (empty string).
    Columns are reordered to the canonical order.
    
    Decorated with @delayed so that it returns a Delayed object if used in a Dask pipeline.
    """
    fixed_columns = [
        'source', 'tid', 'tra', 'trad_gene', 'traj_gene', 'trav_gene',
        'trb', 'trbd_gene', 'trbj_gene', 'trbv_gene', 'peptide', 'mhc_one',
        'mhc_two']

    # Add missing columns
    for col in fixed_columns:
        if col not in df.columns:
            df[col] = ''

    # Reindex to canonical order
    df = df[fixed_columns].copy()
    df = df.fillna("").astype(str)
    return df


def standardize_mri(df):
    """
    Standardize an in-memory pandas DataFrame to ensure it has a fixed set of columns
    for MRI data. Missing columns are added with default values (empty string).
    Columns are reordered to the canonical order.
    
    Decorated with @delayed so that it returns a Delayed object if used in a Dask pipeline.
    """
    fixed_columns = [
        'tid', 'tra', 'trad_gene', 'traj_gene', 'trav_gene',
        'trb', 'trbd_gene', 'trbj_gene', 'trbv_gene',
        'peptide', 'mhc_one', 'mhc_two', 'sequence', 
        'repertoire_id', 'study_id', 'category', 
        'molecule_type', 'host_organism', 'source'
    ]

    for col in fixed_columns:
        if col not in df.columns:
            df[col] = ''

    df = df[fixed_columns].copy()
    df = df.fillna("").astype(str)
    return df
