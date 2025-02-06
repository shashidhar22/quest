import os
import pandas as pd
import dask.dataframe as dd

from itertools import product

from .utils import parse_junction_aa, standardize_sequence, format_combined_tcell

class PairedFileParser:
    def __init__(self, paired_file):
        self.paired_file = paired_file
        self.repertoire_id = os.path.splitext(os.path.basename(self.paired_file))[0]
        self.file_type = os.path.basename(os.path.dirname(self.paired_file))
        self.molecule_type = os.path.basename(os.path.dirname(os.path.dirname(self.paired_file)))   
        self.study_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(self.paired_file))))
        self.category = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(self.paired_file)))))
        self.extension = os.path.splitext(self.paired_file)[1]
        self.source = 'single_cell'
        self.host_organism = 'human'

    def _parse_contigs(self):
        """
        Parse paired contig files using only Dask.

        Returns:
            tuple: Dask DataFrames for MRI table and sequence table.
        """
        # Read the contig file into a Dask DataFrame
        contig_table = dd.read_csv(self.paired_file, dtype="str").fillna("unknown")

        try:
            # Filter for productive, high-confidence contigs
            productive_contigs = contig_table[
                (contig_table['productive'] == "True") &
                (contig_table['high_confidence'] == "True")
            ]
        except KeyError:
            print(f"Missing required columns in file: {self.paired_file}")
            return None, None

        # Define metadata for the output DataFrame
        meta = {
            'tid': 'object',
            'tra': 'object',
            'trad_gene': 'object',
            'traj_gene': 'object',
            'trav_gene': 'object',
            'trb': 'object',
            'trbd_gene': 'object',
            'trbj_gene': 'object',
            'trbv_gene': 'object',
            'sequence': 'object'
        }

        def process_partition(partition):
            """
            Process a single partition of the DataFrame to format TRA and TRB contigs.
            """
            formatted_contigs = []

            for barcode, group in partition.groupby('barcode'):
                tra_seqs = (
                    group[group['chain'] == 'TRA'][['cdr3', 'v_gene', 'd_gene', 'j_gene']]
                    .apply(tuple, axis=1)
                    .tolist()
                )
                trb_seqs = (
                    group[group['chain'] == 'TRB'][['cdr3', 'v_gene', 'd_gene', 'j_gene']]
                    .apply(tuple, axis=1)
                    .tolist()
                )

                # Combine TRA and TRB sequences
                if tra_seqs and trb_seqs:
                    for index, (tra, trb) in enumerate(product(tra_seqs, trb_seqs), start=1):
                        result_dict = format_combined_tcell(barcode, index, tra, trb, tra_only=False)
                        formatted_contigs.append(result_dict)
                elif tra_seqs:  # TRA only
                    for index, tra in enumerate(tra_seqs, start=1):
                        result_dict = format_combined_tcell(barcode, index, tra, None, tra_only=True)
                        formatted_contigs.append(result_dict)
                elif trb_seqs:  # TRB only
                    for index, trb in enumerate(trb_seqs, start=1):
                        result_dict = format_combined_tcell(barcode, index, None, trb, tra_only=False)
                        formatted_contigs.append(result_dict)

            return pd.DataFrame(formatted_contigs)

        # Apply processing to each partition
        formatted_contigs = productive_contigs.map_partitions(
            process_partition, meta=meta
        )

        # Add metadata to the MRI table
        mri_table = formatted_contigs.copy()
        mri_table = mri_table.assign(
            repertoire_id=self.repertoire_id,
            study_id=self.study_id,
            category=self.category,
            molecule_type=self.molecule_type,
            host_organism=self.host_organism,
            source=self.source
        )

        # Create the sequence table by selecting relevant columns
        sequence_table = mri_table[[
            'source', 'trav_gene', 'trad_gene', 'traj_gene', 'tra',
            'trbv_gene', 'trbd_gene', 'trbj_gene', 'trb', 'sequence'
        ]]
        sequence_table = sequence_table.drop_duplicates()
        sequence_table = standardize_sequence(sequence_table)

        return mri_table, sequence_table

    
    def _parse_clonotypes(self):
        # Load the clonotype file as a Dask DataFrame
        clonotype_table = dd.read_csv(self.paired_file, sep=",", dtype="str").fillna("unknown")

        # Metadata extracted from the file path
        repertoire_id = os.path.splitext(os.path.basename(self.paired_file))[0]
        study_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(self.paired_file))))

        fixed_columns = [
            'tid', 'tra', 'trad_gene', 'traj_gene', 'trav_gene',
            'trb', 'trbd_gene', 'trbj_gene', 'trbv_gene', 'sequence'
        ]
        meta = {
            'tid': 'object',
            'tra': 'object',
            'trad_gene': 'object',
            'traj_gene': 'object',
            'trav_gene': 'object',
            'trb': 'object',
            'trbd_gene': 'object',
            'trbj_gene': 'object',
            'trbv_gene': 'object',
            'sequence': 'object'
        }

        def process_partition(df):
            # Parse `cdr3s_aa` column to extract TRA and TRB sequences
            parsed_rows = []
            for _, row in df.iterrows():
                result = parse_junction_aa(row['cdr3s_aa'])
                result['tid'] = row['barcode']
                parsed_rows.append(result)

            # Create a DataFrame from parsed rows and ensure fixed columns
            partition_result = pd.DataFrame(parsed_rows)
            for col in fixed_columns:
                if col not in partition_result.columns:
                    partition_result[col] = None  # Add missing columns with default values
            return partition_result[fixed_columns]

        # Apply parsing to each partition lazily
        parsed_table = clonotype_table.map_partitions(
            lambda df: process_partition(df), meta=meta
        )

        # Add metadata lazily
        parsed_table = parsed_table.assign(
            repertoire_id=repertoire_id,
            study_id=study_id,
            category=self.category,
            molecule_type=self.molecule_type,
            host_organism=self.host_organism,
            source=self.source
        )

        # Create the MRI table with relevant metadata
        mri_table = parsed_table[['category', 'molecule_type', 'study_id', 
                                  'repertoire_id', 'source', 'host_organism', 
                                  'tra', 'trb']]

        # Create the sequence table
        sequence_table = parsed_table.copy()
        sequence_table['sequence'] = sequence_table[['tra', 'trb']].map_partitions(
            lambda df: df.apply(lambda x: ' '.join(x.dropna()) + ';', axis=1),
            meta=('sequence', 'str')
        )
        sequence_table = sequence_table[['source', 'tra', 'trb', 'sequence']]
        sequence_table = sequence_table.drop_duplicates()

        # Standardize sequence table to ensure fixed columns
        sequence_table = standardize_sequence(sequence_table)

        return mri_table, sequence_table

    
    def _parse_rearrangements(self):
        """
        Parses AIRR rearrangements file to generate MRI and sequence tables using Dask.

        Returns:
            tuple: Dask DataFrames for MRI table and sequence table.
        """
        # Read the AIRR file as a Dask DataFrame
        airr_table = dd.read_csv(self.paired_file, sep="\t", dtype="str").fillna("unknown")

        # Filter for productive cells
        airr_table = airr_table[
            (airr_table['is_cell'] == "T") & (airr_table['productive'] == "T")
        ]

        # Define fixed columns and metadata
        fixed_columns = [
            'tid', 'tra', 'trad_gene', 'traj_gene', 'trav_gene',
            'trb', 'trbd_gene', 'trbj_gene', 'trbv_gene', 'sequence'
        ]
        meta = {col: 'object' for col in fixed_columns}

        # Define the partition processing function
        def process_partition(df):
            # Group by barcode and generate TRA/TRB combinations
            grouped = df.groupby('cell_id')
            formatted_results = []

            for barcode, group in grouped:
                tra_seqs = group[group['chain'] == 'TRA'][
                    ['cdr3', 'v_gene', 'd_gene', 'j_gene']
                ].apply(tuple, axis=1).tolist()

                trb_seqs = group[group['chain'] == 'TRB'][
                    ['cdr3', 'v_gene', 'd_gene', 'j_gene']
                ].apply(tuple, axis=1).tolist()

                if tra_seqs and trb_seqs:
                    for idx, (tra, trb) in enumerate(product(tra_seqs, trb_seqs), start=1):
                        formatted_results.append(format_combined_tcell(barcode, idx, tra, trb, tra_only=False))
                elif tra_seqs:
                    for idx, tra in enumerate(tra_seqs, start=1):
                        formatted_results.append(format_combined_tcell(barcode, idx, tra, None, tra_only=True))
                elif trb_seqs:
                    for idx, trb in enumerate(trb_seqs, start=1):
                        formatted_results.append(format_combined_tcell(barcode, idx, None, trb, tra_only=False))

            # Create a DataFrame from the results and ensure fixed columns
            formatted_df = pd.DataFrame(formatted_results)
            for col in fixed_columns:
                if col not in formatted_df.columns:
                    formatted_df[col] = None  # Add missing columns with default values
            return formatted_df[fixed_columns]  # Reorder columns

        # Apply the partition processing function lazily
        formatted_results = airr_table.map_partitions(process_partition, meta=meta)

        # Add metadata to the results
        formatted_results = formatted_results.assign(
            repertoire_id=self.repertoire_id,
            source=self.source,
            study_id=self.study_id,
            host_organism=self.host_organism,
            category=self.category,
            molecule_type=self.molecule_type
        )

        # Split into MRI and sequence tables
        mri_table = formatted_results.copy()
        sequence_table = formatted_results[
            ['source', 'trav_gene', 'trad_gene', 'traj_gene', 'tra',
            'trbv_gene', 'trbd_gene', 'trbj_gene', 'trb', 'sequence']
        ].drop_duplicates()
        sequence_table = standardize_sequence(sequence_table)

        return mri_table, sequence_table

    def parse(self):
        if self.file_type == "contigs":
            return self._parse_contigs()
        elif self.file_type == "clonotypes":
            return self._parse_clonotypes()
        elif self.file_type == "airr":
            return self._parse_rearrangements()
        else:
            raise ValueError("Unrecognized single-cell file format!")