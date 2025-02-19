import os
import csv
import yaml
import pandas as pd
import dask.dataframe as dd

from itertools import product
from collections import OrderedDict

from .utils import standardize_sequence, standardize_mri


class MiscFileParser:
    def __init__(self, misc_file, format_config, test=False):
        self.misc_file = misc_file
        self.format_config = format_config
        self.test = test
        self.format_dict = self._load_format_config()
        self.separator = self._detect_delimiter()
        self.misc_table = self._load_misc_table()
        self.repertoire_id = os.path.splitext(os.path.basename(self.misc_file))[0]
        self.file_type = os.path.basename(os.path.dirname(self.misc_file))
        self.molecule_type = os.path.basename(os.path.dirname(os.path.dirname(self.misc_file)))   
        self.study_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(self.misc_file))))
        self.category = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(self.misc_file)))))
        self.extension = os.path.splitext(self.misc_file)[1]
        self.source = 'misc_format'
        self.host_organism = 'human'
        

    def _load_format_config(self):
        with open(self.format_config, 'r') as f:
            return yaml.safe_load(f)

    def _detect_delimiter(self):
        with open(self.misc_file, 'r') as file:
            sample = file.read(20000)
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)
        return dialect.delimiter

    def _load_misc_table(self):
        if self.separator in ["\t", "s", "t"]:
            misc_table = dd.read_csv(self.misc_file, sep="\t", dtype=str, na_filter=False)
        elif self.separator == ",":
            misc_table = dd.read_csv(self.misc_file, dtype=str, na_filter=False)
        else:
            return None
        
        if self.test:
            misc_table = misc_table.sample(frac=0.1, random_state=21)
        misc_table = misc_table.map_partitions(lambda pdf: pdf.astype("string[pyarrow]"))

        return misc_table

    def parse(self):
        if self.misc_table is None:
            return None, None
        if set(self.misc_table.columns) == set(self.format_dict['misc']['format_one']):
            return self._parse_format_one()
        elif set(self.misc_table.columns) == set(self.format_dict['misc']['format_two']):
            return self._parse_format_two()
        elif set(self.misc_table.columns) == set(self.format_dict['misc']['format_three']):
            return self._parse_format_three()
        elif set(self.misc_table.columns) == set(self.format_dict['misc']['format_four']):
            return self._parse_format_four()
        elif set(self.misc_table.columns) == set(self.format_dict['misc']['format_five']):
            return self._parse_format_five()
        elif set(self.misc_table.columns) == set(self.format_dict['misc']['format_six']):
            return self._parse_format_six()
        else:
            raise ValueError("Unrecognized file format!")

    def _parse_format_one(self):
        """
        Parses AIRR data in format one using Dask to handle large datasets efficiently.

        Returns:
            tuple: Dask DataFrames for MRI table and sequence table.
        """
        # Metadata for the resulting Dask DataFrame
        meta = OrderedDict({
            'tid': 'string[pyarrow]',
            'tra': 'string[pyarrow]',
            'trad_gene': 'string[pyarrow]',
            'traj_gene': 'string[pyarrow]',
            'trav_gene': 'string[pyarrow]',
            'trb': 'string[pyarrow]',
            'trbd_gene': 'string[pyarrow]',
            'trbj_gene': 'string[pyarrow]',
            'trbv_gene': 'string[pyarrow]',
            'sequence': 'string[pyarrow]'
        })

        # Function to process each partition
        def process_partition(df):
            def process_row(row):
                nucleotide = row['cdr3s_nt'].split(';')
                amino_acids = row['cdr3s_aa'].split(';')
                tid = row.name

                tra_list, trb_list = [], []
                for nuc, amino in zip(nucleotide, amino_acids):
                    chain = nuc.split(":")[0]
                    nuc_seq = nuc.split(":")[1]
                    amino_seq = amino.split(":")[1]
                    if chain == "TRA":
                        tra_list.append((nuc_seq, amino_seq))
                    elif chain == "TRB":
                        trb_list.append((nuc_seq, amino_seq))

                formatted = []
                if tra_list and trb_list:
                    for tcell in product(tra_list, trb_list):
                        tra, trb = tcell[0][1], tcell[1][1]
                        sequence = f"{tra} {trb};"
                        formatted.append({'tid': tid, 'tra': tra, 'trb': trb, 'sequence': sequence})
                elif tra_list:
                    for tcell in tra_list:
                        tra = tcell[1]
                        sequence = f"{tra};"
                        formatted.append({'tid': tid, 'tra': tra, 'sequence': sequence})
                elif trb_list:
                    for tcell in trb_list:
                        trb = tcell[1]
                        sequence = f"{trb};"
                        formatted.append({'tid': tid, 'trb': trb, 'sequence': sequence})
                return formatted

            # Process each row and flatten the results
            results = []
            for _, row in df.iterrows():
                results.extend(process_row(row))
            return pd.DataFrame(results, columns=meta.keys())

        # Process partitions using Dask
        mri_table = self.misc_table.map_partitions(process_partition, meta=meta)

        # Add metadata lazily
        mri_table = mri_table.assign(
            repertoire_id=self.repertoire_id,
            study_id=self.study_id,
            category=self.category,
            molecule_type=self.molecule_type,
            host_organism=self.host_organism,
            source=self.source
        )

        # Create the sequence table lazily
        sequence_table = mri_table[
            ['source', 'tra', 'trb', 'sequence']
        ].drop_duplicates()

        # Standardize the sequence table
        sequence_table = standardize_sequence(sequence_table)
        mri_table = standardize_mri(mri_table)
        
        return mri_table, sequence_table


    def _parse_format_two(self):
        """
        Parses AIRR data in format two using Dask.

        Returns:
            tuple: Dask DataFrames for MRI table and sequence table.
        """
        # Define metadata for the resulting DataFrame
        meta = OrderedDict({
            'tid': 'string[pyarrow]',
            'tra': 'string[pyarrow]',
            'trad_gene': 'string[pyarrow]',
            'traj_gene': 'string[pyarrow]',
            'trav_gene': 'string[pyarrow]',
            'trb': 'string[pyarrow]',
            'trbd_gene': 'string[pyarrow]',
            'trbj_gene': 'string[pyarrow]',
            'trbv_gene': 'string[pyarrow]',
            'sequence': 'string[pyarrow]'
        })

        # Function to process each row
        def process_partition(partition_df):
            def process_row(row):
                tid = row['barcode']
                amino_acids = row['cdr3s_aa'].split(';')
                tra_list, trb_list = [], []

                for amino in amino_acids:
                    chain, sequence = amino.split(':')
                    if chain == "TRA":
                        tra_list.append(sequence)
                    elif chain == "TRB":
                        trb_list.append(sequence)

                formatted = []
                if tra_list and trb_list:
                    for tcell in product(tra_list, trb_list):
                        tra, trb = tcell
                        sequence = f"{tra} {trb};"
                        formatted.append({'tid': tid, 'tra': tra, 'trb': trb, 'sequence': sequence})
                elif tra_list:
                    for tra in tra_list:
                        sequence = f"{tra};"
                        formatted.append({'tid': tid, 'tra': tra, 'sequence': sequence})
                elif trb_list:
                    for trb in trb_list:
                        sequence = f"{trb};"
                        formatted.append({'tid': tid, 'trb': trb, 'sequence': sequence})
                return formatted

            # Process each row and flatten the results
            results = []
            for _, row in partition_df.iterrows():
                results.extend(process_row(row))
            return pd.DataFrame(results, columns=meta.keys())

        # Process the partitions lazily
        mri_table = self.misc_table.map_partitions(process_partition, meta=meta)

        # Add metadata lazily
        mri_table = mri_table.assign(
            repertoire_id=self.repertoire_id,
            study_id=self.study_id,
            category=self.category,
            molecule_type=self.molecule_type,
            host_organism=self.host_organism,
            source=self.source
        )

        # Create and deduplicate the sequence table
        sequence_table = mri_table[['source', 'tra', 'trb', 'sequence']].drop_duplicates()
        sequence_table = standardize_sequence(sequence_table)
        mri_table = standardize_mri(mri_table)
        
        return mri_table, sequence_table


    def _parse_format_three(self):
        """
        Parse the dataset in format three using Dask, ensuring compatibility for large datasets.

        Returns:
            tuple: Dask DataFrames for MRI table and sequence table.
        """
        # Rename and select relevant columns
        processed = self.misc_table.rename(columns={
            'sample_id': 'repertoire_id',
            'TR_chain': 'chain',
            'subject_id': 'patient_id'
        })
        processed = processed[['CDR3.aa', 'V.name', 'D.name', 'J.name', 'chain',
                            'repertoire_id', 'patient_id']]

        # Split into TRA and TRB tables
        tra_table = processed[processed['chain'] == "TRA"].rename(columns={
            'CDR3.aa': 'tra',
            'V.name': 'trav_gene',
            'D.name': 'trad_gene',
            'J.name': 'traj_gene'
        })

        trb_table = processed[processed['chain'] == "TRB"].rename(columns={
            'CDR3.aa': 'trb',
            'V.name': 'trbv_gene',
            'D.name': 'trbd_gene',
            'J.name': 'trbj_gene'
        })

        # Combine TRA and TRB tables
        mri_table = dd.concat([tra_table, trb_table], axis=0, interleave_partitions=True)

        # Add metadata columns lazily
        mri_table = mri_table.assign(
            repertoire_id=self.repertoire_id,
            study_id=self.study_id,
            category=self.category,
            molecule_type=self.molecule_type,
            host_organism=self.host_organism,
            source=self.source
        )

        # Create a combined sequence column lazily
        mri_table['sequence'] = mri_table[['tra', 'trb']].map_partitions(
            lambda df: df.apply(lambda row: ' '.join(row.dropna().astype('string[pyarrow]')) + ';', axis=1),
            meta=('sequence', 'string[pyarrow]')
        )

        # Prepare the sequence table
        sequence_table = mri_table[['source', 'trav_gene', 'trad_gene', 'traj_gene', 'tra',
                                    'trbv_gene', 'trbd_gene', 'trbj_gene', 'trb', 'sequence']]

        # Drop duplicates and standardize columns
        sequence_table = sequence_table.drop_duplicates()
        sequence_table = standardize_sequence(sequence_table)
        mri_table = standardize_mri(mri_table)
        
        return mri_table, sequence_table

    
    def _parse_format_four(self):
        """
        Parse the data using Dask to handle large datasets efficiently.

        Returns:
            tuple: Dask DataFrames for MRI table and sequence table.
        """
        # Rename columns lazily
        misc_table = self.misc_table.rename(columns={
            'orig.ident': 'sample_name',
            't_cdr3s_aa': 'cdr3s_aa',
            'Unnamed: 0': 'barcode',
            'Group': 'condition'
        })

        # Filter rows where 'cdr3s_aa' is not null
        misc_table = misc_table[misc_table['cdr3s_aa'].notnull()]

        # Define a function to process partitions
        def process_partition(df):
            formatted_contigs = []
            for _, row in df.iterrows():
                barcode = row['barcode']
                cdr_data = row['cdr3s_aa'].split(';')
                chain_dict = {}

                for tcr in cdr_data:
                    if tcr == 'NA':
                        continue
                    else:
                        chain, seq = tcr.split(':')
                        chain_dict.setdefault(chain, []).append(seq)

                if 'TRA' in chain_dict and 'TRB' in chain_dict:
                    tra_seqs = chain_dict['TRA']
                    trb_seqs = chain_dict['TRB']
                    for index, tcell in enumerate(product(tra_seqs, trb_seqs), start=1):
                        tra, trb = tcell
                        tid = f"tcr_{barcode}_{index}"
                        sequence = f"{tra} {trb};"
                        formatted_contigs.append({
                            'tid': tid, 'tra': tra, 'trb': trb,
                            'sequence': sequence, 'repertoire_id': row['sample_name'],
                            'condition': row['condition']
                        })
                elif 'TRA' in chain_dict:
                    tra_seqs = chain_dict['TRA']
                    for index, tra in enumerate(tra_seqs, start=1):
                        tid = f"tcr_{barcode}_{index}"
                        sequence = f"{tra};"
                        formatted_contigs.append({
                            'tid': tid, 'tra': tra,
                            'sequence': sequence, 'repertoire_id': row['sample_name'],
                            'condition': row['condition']
                        })
                elif 'TRB' in chain_dict:
                    trb_seqs = chain_dict['TRB']
                    for index, trb in enumerate(trb_seqs, start=1):
                        tid = f"tcr_{barcode}_{index}"
                        sequence = f"{trb};"
                        formatted_contigs.append({
                            'tid': tid, 'trb': trb,
                            'sequence': sequence, 'repertoire_id': row['sample_name'],
                            'condition': row['condition']
                        })
            return pd.DataFrame(formatted_contigs)

        meta = OrderedDict({
            'tid': 'string[pyarrow]',
            'tra': 'string[pyarrow]',
            'trb': 'string[pyarrow]',
            'sequence': 'string[pyarrow]',
            'repertoire_id': 'string[pyarrow]',
            'condition': 'string[pyarrow]'
        })

        # Apply the processing function lazily
        mri_table = misc_table.map_partitions(process_partition, meta=meta)
        # Add metadata lazily
        mri_table = mri_table.assign(
            repertoire_id=os.path.splitext(os.path.basename(self.misc_file))[0],
            study_id=os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(self.misc_file)))),
            source='single_cell',
            host_organism='human'
        )

        # Prepare the sequence table
        sequence_table = mri_table.drop(columns=['tid', 'repertoire_id', 'condition'])
        sequence_table = standardize_sequence(sequence_table)
        mri_table = standardize_mri(mri_table)
        
        return mri_table, sequence_table


    def _parse_format_five(self):
        """
        Parse format five using Dask to handle large datasets.

        Returns:
            tuple: Dask DataFrames for MRI table and sequence table.
        """
        # Rename columns lazily
        mri_table = self.misc_table.rename(columns={
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
        })

        # Add metadata columns lazily
        mri_table = mri_table.assign(
            repertoire_id=self.repertoire_id,
            study_id=self.study_id,
            category=self.category,
            molecule_type=self.molecule_type,
            host_organism=self.host_organism,
            source=self.source
        )

        # Select relevant columns for the sequence table lazily
        sequence_table = mri_table[[
            'source', 'trav_gene', 'trad_gene', 'traj_gene', 'tra',
            'trbv_gene', 'trbd_gene', 'trbj_gene', 'trb'
        ]]

        # Add a 'sequence' column by concatenating 'tra' and 'trb', skipping NA values
        def generate_sequence(df):
            df['sequence'] = df[['tra', 'trb']].apply(
                lambda row: ' '.join(filter(lambda x: x != '', row)) + ';',
                axis=1
            )

            return df

        sequence_table = sequence_table.map_partitions(generate_sequence, meta=OrderedDict({
            'source': 'string[pyarrow]',
            'trav_gene': 'string[pyarrow]',
            'trad_gene': 'string[pyarrow]',
            'traj_gene': 'string[pyarrow]',
            'tra': 'string[pyarrow]',
            'trbv_gene': 'string[pyarrow]',
            'trbd_gene': 'string[pyarrow]',
            'trbj_gene': 'string[pyarrow]',
            'trb': 'string[pyarrow]',
            'sequence': 'string[pyarrow]'
        }))

        # Standardize the sequence table lazily
        sequence_table = standardize_sequence(sequence_table)
        mri_table = standardize_mri(mri_table)
        
        return mri_table, sequence_table


    def _parse_format_six(self):
        """
        Processes a Dask DataFrame containing bulk survey data to extract metadata
        and sequence information.

        Returns:
            tuple: Processed MRI table and sequence table as Dask DataFrames.
        """
        # Select relevant columns
        misc_table = self.misc_table[['cdr3aa', 'v', 'd', 'j']]

        # Define the processing function for each partition
        def process_partition(df):
            def process_row(row):
                if "TRAV" in row['v']:
                    return {
                        'trav_gene': row['v'],
                        'traj_gene': row['j'],
                        'trad_gene': row['d'],
                        'tra': row['cdr3aa'],
                        'trbv_gene': '',
                        'trbj_gene': '',
                        'trbd_gene': '',
                        'trb': ''
                    }
                elif "TRBV" in row['v']:
                    return {
                        'trav_gene': '',
                        'traj_gene': '',
                        'trad_gene': '',
                        'tra': '',
                        'trbv_gene': row['v'],
                        'trbj_gene': row['j'],
                        'trbd_gene': row['d'],
                        'trb': row['cdr3aa']
                    }
                else:
                    return {}

            # Apply row-wise processing
            result = pd.DataFrame([process_row(row) for _, row in df.iterrows()])
            return result

        meta = OrderedDict({
            'trav_gene': 'string[pyarrow]',
            'traj_gene': 'string[pyarrow]',
            'trad_gene': 'string[pyarrow]',
            'tra': 'string[pyarrow]',
            'trbv_gene': 'string[pyarrow]',
            'trbj_gene': 'string[pyarrow]',
            'trbd_gene': 'string[pyarrow]',
            'trb': 'string[pyarrow]'
        })

        # Process the misc_table partition-wise
        formatted_sequences = misc_table.map_partitions(process_partition, meta=meta)

        # Drop duplicates and empty rows
        mri_table = formatted_sequences.drop_duplicates().dropna(how='all')

        # Add metadata lazily
        mri_table = mri_table.assign(
            repertoire_id=self.repertoire_id,
            study_id=self.study_id,
            host_organism=self.host_organism,
            source=self.source,
            category=self.category,
            molecule_type=self.molecule_type
        )

        sequence_table = mri_table[[
            'source', 'trav_gene', 'trad_gene', 'traj_gene', 'tra',
            'trbv_gene', 'trbd_gene', 'trbj_gene', 'trb'
        ]]
        # Prepare the sequence table
        sequence_table = sequence_table.assign(
            sequence = sequence_table[['tra', 'trb']].map_partitions(
                lambda df: df.apply(
                    lambda row: ' '.join(
                        filter(lambda x: x != '', [row.get('tra'), row.get('trb')])
                    ) + ';',
                    axis=1
                ),
                meta=('sequence', 'string[pyarrow]')
            )
        )
        seq_meta = OrderedDict({
            'source': 'string[pyarrow]',
            'trav_gene': 'string[pyarrow]',
            'trad_gene': 'string[pyarrow]',
            'traj_gene': 'string[pyarrow]',
            'tra': 'string[pyarrow]',
            'trbv_gene': 'string[pyarrow]',
            'trbd_gene': 'string[pyarrow]',
            'trbj_gene': 'string[pyarrow]',
            'trb': 'string[pyarrow]',
            'sequence': 'string[pyarrow]'
            })
        # Ensure sequences for single-chain data
        sequence_table = sequence_table.map_partitions(
            lambda df: df.assign(
                sequence=df['tra'] + ';' if 'tra' in df.columns and 'trb' not in df.columns else (
                    df['trb'] + ';' if 'trb' in df.columns and 'tra' not in df.columns else df['sequence']
                )
            ),
            meta=seq_meta
        )

        # Standardize the sequence table
        sequence_table = sequence_table.drop_duplicates()
        sequence_table = standardize_sequence(sequence_table)
        mri_table = standardize_mri(mri_table)
        
        return mri_table, sequence_table
