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
        # Build sequence table
        sequence_table = mri_table.drop_duplicates()
        # Add metadata
        mri_table['repertoire_id'] = self.repertoire_id
        mri_table['study_id'] = self.study_id
        mri_table['category'] = self.category
        mri_table['molecule_type'] = self.molecule_type
        mri_table['host_organism'] = self.host_organism
        mri_table['source'] = self.source

        # Annotate sequence table 
        sequence_table['source'] = self.source

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

        mri_table = pd.DataFrame(parsed_rows)
        sequence_table = mri_table.copy().drop_duplicates()
        # Add metadata
        mri_table['repertoire_id'] = self.repertoire_id
        mri_table['study_id'] = self.study_id
        mri_table['category'] = self.category
        mri_table['molecule_type'] = self.molecule_type
        mri_table['host_organism'] = self.host_organism
        mri_table['source'] = self.source

        # Add metadata to sequence table
        sequence_table['source'] = self.source
        
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

        mri_table = pd.DataFrame(formatted_results)
        sequence_table = mri_table.copy().drop_duplicates()
        # Add metadata
        mri_table['repertoire_id'] = self.repertoire_id
        mri_table['study_id'] = self.study_id
        mri_table['host_organism'] = self.host_organism
        mri_table['source'] = self.source
        mri_table['category'] = self.category
        mri_table['molecule_type'] = self.molecule_type

        # Annotate sequence table
        sequence_table['source'] = self.source
        
        # Standardize
        sequence_table = standardize_sequence(sequence_table)
        mri_table = standardize_mri(mri_table)

        return mri_table, sequence_table
