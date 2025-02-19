import os
import pandas as pd
import dask.dataframe as dd

from itertools import product
from collections import OrderedDict
from .utils import parse_junction_aa, standardize_sequence, format_combined_tcell, standardize_mri

class PairedFileParser:
    def __init__(self, paired_file, test=False):
        self.paired_file = paired_file
        self.test = test
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
        contig_table = dd.read_csv(
            self.paired_file,
            dtype=str, na_filter=False,    # or "str"
            include_path_column=True)
        contig_table = contig_table[contig_table['chain'].str.contains('TR[AB]', na=False)]
        contig_table = contig_table.reset_index(drop=True)
        contig_table = contig_table.repartition(partition_size="100MB")

        if self.test:
            contig_table = contig_table.sample(frac=0.1, random_state=21)
        contig_table = contig_table.map_partitions(lambda pdf: pdf.astype("string[pyarrow]"))
        
        try:
            # Filter for productive, high-confidence contigs
            productive_contigs = contig_table[
                (contig_table['productive'].isin(["True", "true", "TRUE"])) &
                (contig_table['high_confidence'].isin(["True", "true", "TRUE"]))
            ]
        except KeyError:
            print(f"Missing required columns in file: {self.paired_file}")
            return None, None
        if productive_contigs.head(1).empty:
            return None, None
        # Define metadata for the output DataFrame
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
                        result_dict = format_combined_tcell(barcode, index, tra, trb) #, tra_only=False)
                        formatted_contigs.append(result_dict)
                elif tra_seqs:  # TRA only
                    for index, tra in enumerate(tra_seqs, start=1):
                        result_dict = format_combined_tcell(barcode, index, tra, '') #, tra_only=True)
                        formatted_contigs.append(result_dict)
                elif trb_seqs:  # TRB only
                    for index, trb in enumerate(trb_seqs, start=1):
                        result_dict = format_combined_tcell(barcode, index, '', trb) #, tra_only=False)
                        formatted_contigs.append(result_dict)
        
            formatted_table = pd.DataFrame(formatted_contigs)
            #formatted_table  = formatted_table.drop(columns=['path'])
            
            formatted_table = formatted_table[meta.keys()]
            
            formatted_table = formatted_table.astype('string[pyarrow]')
            return formatted_table

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
            'source', 'tra', 'trad_gene', 'traj_gene', 'trav_gene',
            'trb', 'trbd_gene', 'trbj_gene', 'trbv_gene', 'sequence'

        ]]
        sequence_table = sequence_table.drop_duplicates()
        sequence_table = standardize_sequence(sequence_table)
        mri_table = standardize_mri(mri_table)
        
        return mri_table, sequence_table

    
    def _parse_clonotypes(self):
        # Load the clonotype file as a Dask DataFrame
        clonotype_table = dd.read_csv(self.paired_file, sep=",", dtype=str, na_filter=False)

        if self.test:
            clonotype_table = clonotype_table.sample(frac=0.1, random_state=21)
        clonotype_table = clonotype_table.map_partitions(lambda pdf: pdf.astype("string[pyarrow]"))
        # Metadata extracted from the file path
        repertoire_id = os.path.splitext(os.path.basename(self.paired_file))[0]
        study_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(self.paired_file))))

        fixed_columns = [
            'tid', 'tra', 'trad_gene', 'traj_gene', 'trav_gene',
            'trb', 'trbd_gene', 'trbj_gene', 'trbv_gene', 'sequence'
        ]
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

        def process_partition(df):
            # Parse `cdr3s_aa` column to extract TRA and TRB sequences
            parsed_rows = []
            for _, row in df.iterrows():
                result = parse_junction_aa(row['cdr3s_aa'])
                try:
                    if 'barcode' in row:
                        result['tid'] = row['barcode']
                    elif 'clonotype_id' in row:
                        result['tid'] = row['clonotype_id']
                except KeyError:
                    result['tid'] = f"row_index_{_}"
                parsed_rows.append(result)

            # Create a DataFrame from parsed rows and ensure fixed columns
            partition_result = pd.DataFrame(parsed_rows)
            for col in fixed_columns:
                if col not in partition_result.columns:
                    partition_result[col] = ''  # Add missing columns with default values
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
            meta=('sequence', 'string[pyarrow]')
        )
        sequence_table = sequence_table[['source', 'tra', 'trb', 'sequence']]
        sequence_table = sequence_table.drop_duplicates()

        # Standardize sequence table to ensure fixed columns
        sequence_table = standardize_sequence(sequence_table)
        mri_table = standardize_mri(mri_table)
        
        return mri_table, sequence_table

    
    def _parse_rearrangements(self):
        """
        Parses AIRR rearrangements file to generate MRI and sequence tables using Dask.

        Returns:
            tuple: Dask DataFrames for MRI table and sequence table.
        """
        # Read the AIRR file as a Dask DataFrame
        airr_table = dd.read_csv(self.paired_file, sep="\t", dtype=str, na_filter=False)
        if self.test:
            airr_table = airr_table.sample(frac=0.1, random_state=21)
        airr_table = airr_table.map_partitions(lambda pdf: pdf.astype("string[pyarrow]"))
        # Filter for productive cells
        airr_table = airr_table[
            (airr_table['is_cell'] == "T") & (airr_table['productive'] == "T")
        ]

        # Define fixed columns and metadata
        fixed_columns = [
            'tid', 'tra', 'trad_gene', 'traj_gene', 'trav_gene',
            'trb', 'trbd_gene', 'trbj_gene', 'trbv_gene', 'sequence'
        ]
        meta = OrderedDict({col: 'string[pyarrow]' for col in fixed_columns})

        # Define the partition processing function
        def process_partition(df):
            # Group by barcode and generate TRA/TRB combinations
            grouped = df.groupby('cell_id')
            formatted_results = []

            for barcode, group in grouped:

                tra_seqs = group[
                    group['v_call'].str.contains("TRA", na=False) |
                    group['j_call'].str.contains("TRA", na=False) |
                    group['d_call'].str.contains("TRA", na=False)
                ][['junction_aa', 'v_call', 'd_call', 'j_call']].apply(tuple, axis=1).tolist()

                trb_seqs = group[
                    group['v_call'].str.contains("TRB", na=False) |
                    group['j_call'].str.contains("TRB", na=False) |
                    group['d_call'].str.contains("TRB", na=False)
                ][['junction_aa', 'v_call', 'd_call', 'j_call']].apply(tuple, axis=1).tolist()

                if tra_seqs and trb_seqs:
                    for idx, (tra, trb) in enumerate(product(tra_seqs, trb_seqs), start=1):
                        formatted_results.append(format_combined_tcell(barcode, idx, tra, trb))
                elif tra_seqs:
                    for idx, tra in enumerate(tra_seqs, start=1):
                        formatted_results.append(format_combined_tcell(barcode, idx, tra, ''))
                elif trb_seqs:
                    for idx, trb in enumerate(trb_seqs, start=1):
                        formatted_results.append(format_combined_tcell(barcode, idx, '', trb))

            # Create a DataFrame from the results and ensure fixed columns
            formatted_df = pd.DataFrame(formatted_results)
            for col in fixed_columns:
                if col not in formatted_df.columns:
                    formatted_df[col] = ''  # Add missing columns with default values
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
        mri_table = standardize_mri(mri_table)
        
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