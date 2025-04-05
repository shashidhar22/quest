import os
import pandas as pd
from itertools import product
from collections import OrderedDict

from .utils import (
    parse_junction_aa,
    standardize_sequence,
    format_combined_tcell,
    standardize_mri
)

class PairedFileParser:
    def __init__(self, paired_file, test=False):
        """
        A parser for single-cell TCR data in 'paired' files (e.g., contigs, clonotypes, rearrangements).
        Now uses only pandas, no dask.
        """
        self.paired_file = paired_file
        self.test = test

        self.repertoire_id = os.path.splitext(os.path.basename(self.paired_file))[0]
        self.file_type = os.path.basename(os.path.dirname(self.paired_file))
        self.molecule_type = os.path.basename(os.path.dirname(os.path.dirname(self.paired_file)))
        self.study_id = os.path.basename(
            os.path.dirname(
                os.path.dirname(os.path.dirname(self.paired_file))
            )
        )
        self.category = os.path.basename(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(self.paired_file)
                    )
                )
            )
        )
        self.extension = os.path.splitext(self.paired_file)[1]
        self.source = 'single_cell'
        self.host_organism = 'human'

    def parse(self):
        """
        Decide which parse function to call based on the 'file_type' directory:
        - "contigs" => _parse_contigs
        - "clonotypes" => _parse_clonotypes
        - "airr" => _parse_rearrangements
        Returns:
            (mri_table, sequence_table) as pandas DataFrames
        """
        if self.file_type == "contigs":
            return self._parse_contigs()
        elif self.file_type == "clonotypes":
            return self._parse_clonotypes()
        elif self.file_type == "airr":
            return self._parse_rearrangements()
        else:
            raise ValueError(f"Unrecognized single-cell file format: {self.file_type}")

    # ----------------------------------------------------------------------
    #                           _parse_contigs
    # ----------------------------------------------------------------------
    def _parse_contigs(self):
        """
        Reads a 'contigs' file, filters to TCR alpha/beta contigs that are productive
        and high_confidence == True, then groups by 'barcode' to form TRA/TRB combos.
        Returns:
            (mri_table, sequence_table)
        """
        try:
            df = pd.read_csv(self.paired_file, dtype=str, na_filter=False)
        except FileNotFoundError:
            print(f"File not found: {self.paired_file}")
            return pd.DataFrame(), pd.DataFrame()

        # Filter for TCR alpha/beta only
        df = df[df['chain'].str.contains('TR[AB]', na=False)].reset_index(drop=True)

        # Filter for productive + high_confidence
        df = df[
            df['productive'].isin(["True", "true", "TRUE"]) &
            df['high_confidence'].isin(["True", "true", "TRUE"])
        ].copy()

        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        if self.test:
            df = df.sample(frac=0.1, random_state=21)

        # Group by barcode
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
        # Add metadata
        mri_table['repertoire_id'] = self.repertoire_id
        mri_table['study_id'] = self.study_id
        mri_table['category'] = self.category
        mri_table['molecule_type'] = self.molecule_type
        mri_table['host_organism'] = self.host_organism
        mri_table['source'] = self.source

        # Build sequence table
        seq_cols = [
            'source', 'tra', 'trad_gene', 'traj_gene', 'trav_gene',
            'trb', 'trbd_gene', 'trbj_gene', 'trbv_gene', 'sequence'
        ]
        sequence_table = mri_table[seq_cols].drop_duplicates()

        # Standardize
        sequence_table = standardize_sequence(sequence_table)
        mri_table = standardize_mri(mri_table)
        return mri_table, sequence_table

    # ----------------------------------------------------------------------
    #                           _parse_clonotypes
    # ----------------------------------------------------------------------
    def _parse_clonotypes(self):
        """
        Reads a 'clonotypes' CSV. Expects 'cdr3s_aa' column that we parse,
        plus either 'barcode' or 'clonotype_id' used as tid. 
        Returns:
            (mri_table, sequence_table)
        """
        try:
            df = pd.read_csv(self.paired_file, sep=",", dtype=str, na_filter=False)
        except FileNotFoundError:
            print(f"File not found: {self.paired_file}")
            return pd.DataFrame(), pd.DataFrame()

        if self.test:
            df = df.sample(frac=0.1, random_state=21)

        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        parsed_rows = []
        for idx, row in df.iterrows():
            # parse_junction_aa returns dict with keys like 'tra','trb','trav_gene','trbv_gene', etc.
            result = parse_junction_aa(row['cdr3s_aa'])
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

        parsed_table = pd.DataFrame(parsed_rows)

        # Add metadata
        parsed_table['repertoire_id'] = self.repertoire_id
        parsed_table['study_id'] = self.study_id
        parsed_table['category'] = self.category
        parsed_table['molecule_type'] = self.molecule_type
        parsed_table['host_organism'] = self.host_organism
        parsed_table['source'] = self.source

        # Build MRI table
        mri_table = parsed_table.copy()

        # Build sequence table
        # If 'tra' or 'trb' is present, build combined sequence
        def build_sequence(row):
            parts = []
            if 'tra' in row and row['tra']:
                parts.append(str(row['tra']))
            if 'trb' in row and row['trb']:
                parts.append(str(row['trb']))
            return ' '.join(parts) + ';' if parts else ''

        parsed_table['sequence'] = parsed_table.apply(build_sequence, axis=1)
        seq_cols = ['source', 'tra', 'trb', 'sequence']
        for gene_col in ['trav_gene','trad_gene','traj_gene','trbv_gene','trbd_gene','trbj_gene']:
            if gene_col in parsed_table.columns:
                seq_cols.insert(-1, gene_col)
        # Keep only columns that exist
        sequence_table = parsed_table[[c for c in seq_cols if c in parsed_table.columns]].drop_duplicates()

        # Standardize
        sequence_table = standardize_sequence(sequence_table)
        mri_table = standardize_mri(mri_table)
        return mri_table, sequence_table

    # ----------------------------------------------------------------------
    #                           _parse_rearrangements
    # ----------------------------------------------------------------------
    def _parse_rearrangements(self):
        """
        Reads an AIRR rearrangements TSV, filters for is_cell == T and productive == T,
        then groups by cell_id to form TRA/TRB combos. 
        Returns:
            (mri_table, sequence_table)
        """
        try:
            df = pd.read_csv(self.paired_file, sep="\t", dtype=str, na_filter=False)
        except FileNotFoundError:
            print(f"File not found: {self.paired_file}")
            return pd.DataFrame(), pd.DataFrame()

        if self.test:
            df = df.sample(frac=0.1, random_state=21)

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

        result_df = pd.DataFrame(formatted_results)
        # Add metadata
        result_df['repertoire_id'] = self.repertoire_id
        result_df['study_id'] = self.study_id
        result_df['host_organism'] = self.host_organism
        result_df['source'] = self.source
        result_df['category'] = self.category
        result_df['molecule_type'] = self.molecule_type

        # MRI
        mri_table = result_df.copy()

        # Sequence table
        seq_cols = [
            'source', 'trav_gene', 'trad_gene', 'traj_gene', 'tra',
            'trbv_gene', 'trbd_gene', 'trbj_gene', 'trb', 'sequence'
        ]
        sequence_table = result_df[seq_cols].drop_duplicates()

        # Standardize
        sequence_table = standardize_sequence(sequence_table)
        mri_table = standardize_mri(mri_table)

        return mri_table, sequence_table
