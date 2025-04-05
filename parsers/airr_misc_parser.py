import os
import csv
import yaml
import pandas as pd

from itertools import product
from collections import OrderedDict

from .utils import standardize_sequence, standardize_mri


class MiscFileParser:
    def __init__(self, misc_file, format_config, test=False):
        """
        A parser for 'misc' file formats, now using only pandas in memory.
        """
        self.misc_file = misc_file
        self.format_config = format_config
        self.test = test
        self.format_dict = self._load_format_config()

        self.separator = self._detect_delimiter()
        self.misc_table = self._load_misc_table()

        self.repertoire_id = os.path.splitext(os.path.basename(self.misc_file))[0]
        self.file_type = os.path.basename(os.path.dirname(self.misc_file))
        self.molecule_type = os.path.basename(os.path.dirname(os.path.dirname(self.misc_file)))
        self.study_id = os.path.basename(
            os.path.dirname(
                os.path.dirname(os.path.dirname(self.misc_file))
            )
        )
        self.category = os.path.basename(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(self.misc_file)
                    )
                )
            )
        )
        self.extension = os.path.splitext(self.misc_file)[1]
        self.source = 'misc_format'
        self.host_organism = 'human'

    def _load_format_config(self):
        with open(self.format_config, 'r') as f:
            return yaml.safe_load(f)

    def _detect_delimiter(self):
        """
        Use Python's csv.Sniffer to detect delimiter from a sample of the file.
        """
        if not os.path.isfile(self.misc_file):
            return None
        with open(self.misc_file, 'r') as file:
            sample = file.read(20000)
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)
        return dialect.delimiter

    def _load_misc_table(self):
        """
        Load the file into a pandas DataFrame using the discovered delimiter.
        """
        if not os.path.isfile(self.misc_file):
            return pd.DataFrame()

        # If we didn't detect a standard delimiter, fallback to None -> default guess
        sep = self.separator if self.separator in ["\t", ","] else None
        try:
            df = pd.read_csv(self.misc_file, sep=sep, dtype=str, na_filter=False, na_values=[''])
        except Exception as e:
            print(f"Error reading {self.misc_file}: {e}")
            return pd.DataFrame()

        if self.test and not df.empty:
            df = df.sample(frac=0.1, random_state=21)

        # Convert all columns to string explicitly
        df = df.astype(str)
        df = df.fillna('')  # Replace NaN with empty string
        df = df.replace('None', '', regex=True)  # Replace 'None' string with empty string
        df = df.replace('nan', '', regex=True)  # Replace 'nan' string with empty string
        df = df.replace('NA', '', regex=True)  # Replace 'NA' string with empty string
        df = df.replace('N/A', '', regex=True)  # Replace 'N/A' string with empty string
        df = df.replace('na', '', regex=True)  # Replace 'na' string with empty string
        df = df.replace('NaN', '', regex=True)  # Replace 'NaN' string with empty string
        return df

    def parse(self):
        """
        Decide which parse function to call based on recognized columns,
        then return the resulting (mri_table, sequence_table).
        """
        if self.misc_table is None or self.misc_table.empty:
            # Return empty DataFrames
            return pd.DataFrame(), pd.DataFrame()

        known_formats = self.format_dict['misc']
        columns_set = set(self.misc_table.columns)

        # Compare with known format columns
        if columns_set == set(known_formats['format_one']):
            return self._parse_format_one()
        elif columns_set == set(known_formats['format_two']):
            return self._parse_format_two()
        elif columns_set == set(known_formats['format_three']):
            return self._parse_format_three()
        elif columns_set == set(known_formats['format_four']):
            return self._parse_format_four()
        elif columns_set == set(known_formats['format_five']):
            return self._parse_format_five()
        elif columns_set == set(known_formats['format_six']):
            return self._parse_format_six()
        else:
            raise ValueError("Unrecognized file format!")

    # ----------------------------------------------------------------------
    #                          FORMAT ONE
    # ----------------------------------------------------------------------
    def _parse_format_one(self):
        """
        Parse 'format_one' with pure pandas. Return (mri_table, sequence_table).
        e.g., each row has cdr3s_nt='TRA:xxxx;TRB:xxxx' and cdr3s_aa='TRA:xxxx;TRB:xxxx'
        """
        df = self.misc_table.copy()
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        parsed_rows = []
        for idx, row in df.iterrows():
            # cdr3s_nt => Nucleotide sequences
            # cdr3s_aa => Amino acid sequences
            cdr3_nt = row['cdr3s_nt'].split(';')
            cdr3_aa = row['cdr3s_aa'].split(';')

            tra_list, trb_list = [], []
            # We'll pair them up by index
            for nt, aa in zip(cdr3_nt, cdr3_aa):
                chain_nt, nuc_seq = nt.split(':', 1)
                chain_aa, aa_seq = aa.split(':', 1)
                # Must match chain type (assuming they do). We'll check chain_nt
                if chain_nt == 'TRA':
                    tra_list.append(aa_seq)
                elif chain_nt == 'TRB':
                    trb_list.append(aa_seq)

            # Expand combos
            if tra_list and trb_list:
                for (tra, trb) in product(tra_list, trb_list):
                    parsed_rows.append({
                        'tid': str(idx),
                        'tra': tra,
                        'trb': trb,
                        'sequence': f"{tra} {trb};"
                    })
            elif tra_list:
                for tra in tra_list:
                    parsed_rows.append({
                        'tid': str(idx),
                        'tra': tra,
                        'sequence': f"{tra};"
                    })
            elif trb_list:
                for trb in trb_list:
                    parsed_rows.append({
                        'tid': str(idx),
                        'trb': trb,
                        'sequence': f"{trb};"
                    })

        if not parsed_rows:
            return pd.DataFrame(), pd.DataFrame()

        mri_table = pd.DataFrame(parsed_rows)
        # Add metadata
        mri_table['repertoire_id'] = self.repertoire_id
        mri_table['study_id'] = self.study_id
        mri_table['category'] = self.category
        mri_table['molecule_type'] = self.molecule_type
        mri_table['host_organism'] = self.host_organism
        mri_table['source'] = self.source

        # Build the sequence table
        sequence_table = mri_table[['source', 'tra', 'trb', 'sequence']].drop_duplicates()

        # Standardize
        sequence_table = standardize_sequence(sequence_table)
        mri_table = standardize_mri(mri_table)

        return mri_table, sequence_table

    # ----------------------------------------------------------------------
    #                          FORMAT TWO
    # ----------------------------------------------------------------------
    def _parse_format_two(self):
        """
        Similar row-by-row logic for 'format_two'.
        e.g., cdr3s_aa => "TRA:xxx;TRB:yyy", plus 'barcode' col for cell ID.
        """
        df = self.misc_table.copy()
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        parsed_rows = []
        for _, row in df.iterrows():
            tid = row['barcode']
            cdr3_aa_list = row['cdr3s_aa'].split(';')
            tra_list, trb_list = [], []
            for cdr_str in cdr3_aa_list:
                chain, seq = cdr_str.split(':', 1)
                if chain == 'TRA':
                    tra_list.append(seq)
                elif chain == 'TRB':
                    trb_list.append(seq)

            if tra_list and trb_list:
                for (tra, trb) in product(tra_list, trb_list):
                    parsed_rows.append({
                        'tid': tid,
                        'tra': tra,
                        'trb': trb,
                        'sequence': f"{tra} {trb};"
                    })
            elif tra_list:
                for tra in tra_list:
                    parsed_rows.append({
                        'tid': tid,
                        'tra': tra,
                        'sequence': f"{tra};"
                    })
            elif trb_list:
                for trb in trb_list:
                    parsed_rows.append({
                        'tid': tid,
                        'trb': trb,
                        'sequence': f"{trb};"
                    })

        if not parsed_rows:
            return pd.DataFrame(), pd.DataFrame()

        mri_table = pd.DataFrame(parsed_rows)
        mri_table['repertoire_id'] = self.repertoire_id
        mri_table['study_id'] = self.study_id
        mri_table['category'] = self.category
        mri_table['molecule_type'] = self.molecule_type
        mri_table['host_organism'] = self.host_organism
        mri_table['source'] = self.source

        seq_table = mri_table[['source', 'tra', 'trb', 'sequence']].drop_duplicates()
        seq_table = standardize_sequence(seq_table)
        mri_table = standardize_mri(mri_table)

        return mri_table, seq_table

    # ----------------------------------------------------------------------
    #                          FORMAT THREE
    # ----------------------------------------------------------------------
    def _parse_format_three(self):
        """
        e.g., columns: 'CDR3.aa' => 'cdr3_aa', 'V.name'=>'v_gene', 'D.name'=>'d_gene',
                      'J.name'=>'j_gene', 'chain'=>'TRA'/'TRB', 'sample_id'=>'repertoire_id', etc.
        """
        df = self.misc_table.copy()
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

        combined = pd.concat([tra_df, trb_df], ignore_index=True)
        if combined.empty:
            return pd.DataFrame(), pd.DataFrame()

        combined['repertoire_id'] = self.repertoire_id
        combined['study_id'] = self.study_id
        combined['category'] = self.category
        combined['molecule_type'] = self.molecule_type
        combined['host_organism'] = self.host_organism
        combined['source'] = self.source

        def build_sequence(row):
            parts = []
            if 'tra' in row and row['tra']:
                parts.append(str(row['tra']))
            if 'trb' in row and row['trb']:
                parts.append(str(row['trb']))
            return ' '.join(parts) + ';' if parts else ''

        combined['sequence'] = combined.apply(build_sequence, axis=1)

        seq_table = combined[[
            'source', 'trav_gene', 'trad_gene', 'traj_gene', 'tra',
            'trbv_gene', 'trbd_gene', 'trbj_gene', 'trb', 'sequence'
        ]].drop_duplicates()

        seq_table = standardize_sequence(seq_table)
        combined = standardize_mri(combined)
        return combined, seq_table

    # ----------------------------------------------------------------------
    #                          FORMAT FOUR
    # ----------------------------------------------------------------------
    def _parse_format_four(self):
        """
        e.g., columns: 'orig.ident' => 'sample_name', 't_cdr3s_aa' => 'cdr3s_aa',
                       'Unnamed: 0' => 'barcode', 'Group' => 'condition'
        """
        df = self.misc_table.copy()
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        df.rename(columns={
            'orig.ident': 'sample_name',
            't_cdr3s_aa': 'cdr3s_aa',
            'Unnamed: 0': 'barcode',
            'Group': 'condition'
        }, inplace=True)

        # filter out rows where cdr3s_aa is empty or 'NA'
        df = df[df['cdr3s_aa'].notnull() & df['cdr3s_aa'].ne('NA')]

        parsed_rows = []
        for _, row in df.iterrows():
            barcode = row['barcode']
            cdr_data = row['cdr3s_aa'].split(';') if row['cdr3s_aa'] else []
            chain_map = {}

            for tcr in cdr_data:
                chain, seq = tcr.split(':', 1)
                chain_map.setdefault(chain, []).append(seq)

            tra_list = chain_map.get('TRA', [])
            trb_list = chain_map.get('TRB', [])

            if tra_list and trb_list:
                for index, (tra, trb) in enumerate(product(tra_list, trb_list), start=1):
                    tid = f"tcr_{barcode}_{index}"
                    parsed_rows.append({
                        'tid': tid,
                        'tra': tra,
                        'trb': trb,
                        'sequence': f"{tra} {trb};",
                        'repertoire_id': row['sample_name'],
                        'condition': row['condition']
                    })
            elif tra_list:
                for index, tra in enumerate(tra_list, start=1):
                    tid = f"tcr_{barcode}_{index}"
                    parsed_rows.append({
                        'tid': tid,
                        'tra': tra,
                        'sequence': f"{tra};",
                        'repertoire_id': row['sample_name'],
                        'condition': row['condition']
                    })
            elif trb_list:
                for index, trb in enumerate(trb_list, start=1):
                    tid = f"tcr_{barcode}_{index}"
                    parsed_rows.append({
                        'tid': tid,
                        'trb': trb,
                        'sequence': f"{trb};",
                        'repertoire_id': row['sample_name'],
                        'condition': row['condition']
                    })

        if not parsed_rows:
            return pd.DataFrame(), pd.DataFrame()

        mri_table = pd.DataFrame(parsed_rows)
        mri_table['repertoire_id'] = self.repertoire_id
        mri_table['study_id'] = self.study_id
        mri_table['source'] = 'single_cell'
        mri_table['host_organism'] = 'human'

        seq_table = mri_table.drop(columns=['tid', 'condition']).copy()
        seq_table = standardize_sequence(seq_table)
        mri_table = standardize_mri(mri_table)
        return mri_table, seq_table

    # ----------------------------------------------------------------------
    #                          FORMAT FIVE
    # ----------------------------------------------------------------------
    def _parse_format_five(self):
        """
        e.g., columns: 'orig.ident' => 'repertoire_id', 'barcode' => 'tid', 'Tissue'=>'source_tissue',
        'PatientID'=>'patient_id', 'CDR3A'=>'tra', 'TRAV'=>'trav_gene', 'TRAD'=>'trad_gene', 'TRAJ'=>'traj_gene',
        'CDR3B'=>'trb', 'TRBV'=>'trbv_gene', 'TRBD'=>'trbd_gene', 'TRBJ'=>'trbj_gene'
        """
        df = self.misc_table.copy()
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        df.rename(columns={
            'orig.ident': 'repertoire_id',
            'barcode': 'tid',
            'Tissue': 'source_tissue',
            'PatientID': 'patient_id',
            'CDR3A': 'tra',
            'TRAV': 'trav_gene',
            'TRAD': 'trad_gene',
            'TRAJ': 'traj_gene',
            'CDR3B': 'trb',
            'TRBV': 'trbv_gene',
            'TRBD': 'trbd_gene',
            'TRBJ': 'trbj_gene'
        }, inplace=True)

        df['repertoire_id'] = self.repertoire_id
        df['study_id'] = self.study_id
        df['category'] = self.category
        df['molecule_type'] = self.molecule_type
        df['host_organism'] = self.host_organism
        df['source'] = self.source

        def build_sequence(row):
            parts = []
            if pd.notna(row['tra']) and row['tra']:
                parts.append(str(row['tra']))
            if pd.notna(row['trb']) and row['trb']:
                parts.append(str(row['trb']))
            return ' '.join(parts) + ';' if parts else ''

        df['sequence'] = df.apply(build_sequence, axis=1)

        seq_cols = [
            'source', 'trav_gene', 'trad_gene', 'traj_gene', 'tra',
            'trbv_gene', 'trbd_gene', 'trbj_gene', 'trb', 'sequence'
        ]
        sequence_table = df[seq_cols].drop_duplicates()

        sequence_table = standardize_sequence(sequence_table)
        df = standardize_mri(df)
        return df, sequence_table

    # ----------------------------------------------------------------------
    #                          FORMAT SIX
    # ----------------------------------------------------------------------
    def _parse_format_six(self):
        """
        e.g., columns: 'cdr3aa', 'v', 'd', 'j'
        We'll figure out chain type from 'v' (TRAV or TRBV).
        """
        df = self.misc_table.copy()
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        keep_cols = ['cdr3aa', 'v', 'd', 'j']
        for c in keep_cols:
            if c not in df.columns:
                df[c] = ''
        df = df[keep_cols]

        parsed_rows = []
        for idx, row in df.iterrows():
            v_gene = row['v']
            if 'TRAV' in v_gene:
                parsed_rows.append({
                    'trav_gene': row['v'],
                    'trad_gene': row['d'],
                    'traj_gene': row['j'],
                    'tra': row['cdr3aa'],
                    'trbv_gene': '',
                    'trbd_gene': '',
                    'trbj_gene': '',
                    'trb': ''
                })
            elif 'TRBV' in v_gene:
                parsed_rows.append({
                    'trav_gene': '',
                    'trad_gene': '',
                    'traj_gene': '',
                    'tra': '',
                    'trbv_gene': row['v'],
                    'trbd_gene': row['d'],
                    'trbj_gene': row['j'],
                    'trb': row['cdr3aa']
                })
            # else skip

        if not parsed_rows:
            return pd.DataFrame(), pd.DataFrame()

        mri_table = pd.DataFrame(parsed_rows)
        mri_table['repertoire_id'] = self.repertoire_id
        mri_table['study_id'] = self.study_id
        mri_table['host_organism'] = self.host_organism
        mri_table['source'] = self.source
        mri_table['category'] = self.category
        mri_table['molecule_type'] = self.molecule_type

        def build_sequence(row):
            parts = []
            if row.get('tra') and row['tra']:
                parts.append(str(row['tra']))
            if row.get('trb') and row['trb']:
                parts.append(str(row['trb']))
            return ' '.join(parts) + ';' if parts else ''

        mri_table['sequence'] = mri_table.apply(build_sequence, axis=1)
        mri_table.drop_duplicates(inplace=True)

        seq_cols = [
            'source', 'trav_gene', 'trad_gene', 'traj_gene', 'tra',
            'trbv_gene', 'trbd_gene', 'trbj_gene', 'trb', 'sequence'
        ]
        sequence_table = mri_table[seq_cols].drop_duplicates()

        sequence_table = standardize_sequence(sequence_table)
        mri_table = standardize_mri(mri_table)
        return mri_table, sequence_table
