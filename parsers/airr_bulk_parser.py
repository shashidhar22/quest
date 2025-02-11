import os
import re
import csv
import yaml
import dask.dataframe as dd
import pandas as pd

from collections import OrderedDict

from .utils import standardize_sequence


class BulkFileParser:
    def __init__(self, bulk_file, format_config):
        self.bulk_file = bulk_file
        self.format_config = format_config
        self.repertoire_id = os.path.splitext(os.path.basename(self.bulk_file))[0]
        self.study_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(self.bulk_file))))
        self.extension = os.path.splitext(self.bulk_file)[1]
        self.format_dict = self._load_format_config()
        self.separator = self._detect_delimiter()
        self.bulk_table = self._load_bulk_table()
        self.file_type = os.path.basename(os.path.dirname(self.bulk_file))
        self.molecule_type = os.path.basename(os.path.dirname(os.path.dirname(self.bulk_file)))   
        self.study_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(self.bulk_file))))
        self.category = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(self.bulk_file)))))


    def _load_format_config(self):
        with open(self.format_config, 'r') as f:
            return yaml.safe_load(f)

    def _detect_delimiter(self):
        with open(self.bulk_file, 'r') as file:
            sample = file.read(20000)
            if "MiTCRFullExportV1.1" in sample:
                separator = "M"
            else:
                sniffer = csv.Sniffer()
                try:
                    dialect = sniffer.sniff(sample)
                    separator = dialect.delimiter
                except csv.Error:
                    if "\t" in sample: 
                        separator = "\t"
                    else:
                        separator = ","
        if separator == "\t" or self.extension == ".tsv":
            delimiter = "\t"
        elif separator == "," or self.extension == ".csv":
            delimiter = ","
        elif separator == "M":
            delimiter = "M"
        return delimiter


    def _load_bulk_table(self):
        if self.separator == "\t":
            try:
                return dd.read_csv(self.bulk_file, sep="\t", dtype=str, na_filter=False).fillna("unknown")            
            except ValueError:
                return None
        elif self.separator == ",":
            return dd.read_csv(self.bulk_file, dtype=str, na_filter=False).fillna("unknown")
        elif self.separator == "M":
            return dd.read_csv(self.bulk_file, sep="\t", dtype=str, na_filter=False, skiprows=1).fillna("unknown")

    def parse(self):
        if self.bulk_table is None:
            return None, None
        if set(self.bulk_table.columns) == set(self.format_dict['bulk']['format_one']):
            return self._parse_format_one()
        elif set(self.bulk_table.columns) == set(self.format_dict['bulk']['format_two']):
            return self._parse_format_two()
        elif set(self.bulk_table.columns) == set(self.format_dict['bulk']['format_three']):
            return self._parse_format_three()
        elif set(self.bulk_table.columns) == set(self.format_dict['bulk']['format_four']):
            return self._parse_format_four()
        elif set(self.bulk_table.columns) == set(self.format_dict['bulk']['format_five']):
            return self._parse_format_five()
        elif set(self.bulk_table.columns) == set(self.format_dict['bulk']['format_six']):
            return self._parse_format_six()
        elif (set(self.bulk_table.columns) == set(self.format_dict['bulk']['format_seven']) or
              set(self.bulk_table.columns) == set(self.format_dict['bulk']['format_eight']) or
              set(self.bulk_table.columns) == set(self.format_dict['bulk']['format_ten'])): 
            return self._parse_format_seven()
        elif set(self.bulk_table.columns) == set(self.format_dict['bulk']['format_nine']):
            return self._parse_format_nine()
        elif set(self.bulk_table.columns) == set(self.format_dict['bulk']['format_eleven']):
            return self._parse_format_eleven()
        else:
            raise ValueError("Unrecognized file format!")
    
    def _parse_format_one(self):
        """
        Parse format one using Dask DataFrame operations only.
        """
        # Drop unnecessary columns
        processed = self.bulk_table.drop(columns=['VJCombo', 'Copy', 'ntCDR3', 'NetInsertionLength'])

        # Detect chain type from the first entry in 'VGene' using map_partitions
        def detect_chain_type(df):
            if not df.empty:
                return df.iloc[0]['VGene']
            return None

        # Ensure that `detect_chain_type` returns a single-column Dask Series
        # Detect chain type and drop missing values
        chain_type_series = processed.map_partitions(detect_chain_type, meta=('VGene', 'str'))

        # Compute the first non-null value from the series
        chain_type = chain_type_series.compute().iloc[0]
        
        # Process based on chain type
        if 'TRAV' in chain_type:
            # Rename columns for TRA chain and add trad_gene as NaN
            mri_table = processed.rename(columns={
                'VGene': 'trav_gene',
                'JGene': 'traj_gene',
                'aaCDR3': 'tra'
            }).assign(trad_gene=None)

            # Create sequence table
            sequence_table = mri_table[['trav_gene', 'trad_gene', 'traj_gene', 'tra']]

        elif 'TRBV' in chain_type:
            # Rename columns for TRB chain and add trbd_gene as NaN
            mri_table = processed.rename(columns={
                'VGene': 'trbv_gene',
                'JGene': 'trbj_gene',
                'aaCDR3': 'trb'
            }).assign(trbd_gene=None)

            # Create sequence table
            sequence_table = mri_table[['trbv_gene', 'trbd_gene', 'trbj_gene', 'trb']]

        else:
            raise ValueError(f"Unrecognized chain type in VGene: {chain_type}")

        # Add metadata using assign
        mri_table = mri_table.assign(
            repertoire_id=self.repertoire_id,
            study_id=self.study_id,
            category=self.category,
            molecule_type=self.molecule_type,
            host_organism='human',
            source='bulk_survey'
        )
        sequence_table = sequence_table.assign(source='bulk_survey')

        # Standardize sequence table columns
        sequence_table = standardize_sequence(sequence_table)

        return mri_table, sequence_table

    
    def _parse_format_two(self):
        """
        Parse the AIRR dataset in format two using Dask without computing prematurely.

        Returns:
            tuple: Dask DataFrames for MRI and sequence tables.
        """
        # Drop unnecessary columns
        columns_to_drop = [
            'Read count', 'Percentage', 'CDR3 nucleotide sequence', 
            'CDR3 nucleotide quality', 'Min quality', 'V segments', 
            'J segments', 'D segments', 'Last V nucleotide position ', 
            'First D nucleotide position', 'Last D nucleotide position', 
            'First J nucleotide position', 'VD insertions', 'DJ insertions', 
            'Total insertions'
        ]

        mri_table = self.bulk_table.drop(columns=columns_to_drop, errors='ignore')

        # Define the final metadata explicitly
        meta = OrderedDict({
            'trav_gene': 'object',
            'traj_gene': 'object',
            'trad_gene': 'object',
            'tra': 'object',
            'trbv_gene': 'object',
            'trbj_gene': 'object',
            'trbd_gene': 'object',
            'trb': 'object',
            'sequence': 'object'
        })

        # Process each partition to detect chain type and transform columns
        def process_partition(partition):
            if partition.empty:
                return pd.DataFrame(columns=meta.keys())

            # Detect chain type from the first row in the partition
            chain_type = partition.iloc[0]['V alleles']

            if "TRAV" in chain_type:
                # Rename columns for TRA chain
                partition = partition.rename(columns={
                    'V alleles': 'trav_gene',
                    'J alleles': 'traj_gene',
                    'D alleles': 'trad_gene',
                    'CDR3 amino acid sequence': 'tra'
                })
                partition['sequence'] = partition['tra'] + ';'
                partition['trb'] = None
                partition['trbv_gene'] = None
                partition['trbj_gene'] = None
                partition['trbd_gene'] = None
            elif "TRBV" in chain_type:
                # Rename columns for TRB chain
                partition = partition.rename(columns={
                    'V alleles': 'trbv_gene',
                    'J alleles': 'trbj_gene',
                    'D alleles': 'trbd_gene',
                    'CDR3 amino acid sequence': 'trb'
                })
                partition['sequence'] = partition['trb'] + ';'
                partition['tra'] = None
                partition['trav_gene'] = None
                partition['traj_gene'] = None
                partition['trad_gene'] = None
            else:
                raise ValueError(f"Unrecognized chain type in V alleles: {chain_type}")

            # Ensure the output matches the metadata
            return partition[meta.keys()]

        # Apply the transformation lazily to all partitions
        mri_table = mri_table.map_partitions(process_partition, meta=meta)

        # Deduplicate the sequence table and standardize it
        sequence_table = mri_table.drop_duplicates(subset=['sequence'])
        sequence_table = standardize_sequence(sequence_table)

        # Add metadata lazily
        mri_table = mri_table.assign(
            repertoire_id=self.repertoire_id,
            study_id=self.study_id,
            host_organism='human',
            category=self.category,
            molecule_type=self.molecule_type,
            source='bulk_survey'
        )
        sequence_table['source'] = 'bulk_survey'

        return mri_table, sequence_table

    
    def _parse_format_three(self):
        """
        Parse the AIRR dataset in format three using Dask without premature computations.

        Returns:
            tuple: Dask DataFrames for MRI and sequence tables.
        """
        # Drop unnecessary columns
        columns_to_drop = [
            'Count', 'Percentage', 'CDR3 nucleotide sequence', 
            'Last V nucleotide position', 'First D nucleotide position', 
            'Last D nucleotide position', 'First J nucleotide position', 
            'Good events', 'Total events', 'Good reads', 'Total reads'
        ]
        mri_table = self.bulk_table.drop(columns=columns_to_drop, errors='ignore')

        # Define meta for the resulting DataFrame
        meta = OrderedDict({
            'trav_gene': 'object', 'trad_gene': 'object', 'traj_gene': 'object', 'tra': 'object',
            'trbv_gene': 'object', 'trbd_gene': 'object', 'trbj_gene': 'object', 'trb': 'object',
            'sequence': 'object'
        })

        # Process each partition to detect chain type and transform columns
        def process_partition(partition):
            if partition.empty:
                # Return an empty DataFrame with the expected columns
                return pd.DataFrame(columns=meta.keys())

            # Detect chain type from the first non-null value in 'V segments'
            chain_type = partition['V segments'].dropna().iloc[0]

            if "TRAV" in chain_type:
                # Rename columns for TRA chain
                partition = partition.rename(columns={
                    'V segments': 'trav_gene',
                    'J segments': 'traj_gene',
                    'D segments': 'trad_gene',
                    'CDR3 amino acid sequence': 'tra'
                })
                partition['sequence'] = partition['tra'] + ';'
                partition['trb'] = None
                partition['trbv_gene'] = None
                partition['trbj_gene'] = None
                partition['trbd_gene'] = None
            elif "TRBV" in chain_type:
                # Rename columns for TRB chain
                partition = partition.rename(columns={
                    'V segments': 'trbv_gene',
                    'J segments': 'trbj_gene',
                    'D segments': 'trbd_gene',
                    'CDR3 amino acid sequence': 'trb'
                })
                partition['sequence'] = partition['trb'] + ';'
                partition['tra'] = None
                partition['trav_gene'] = None
                partition['traj_gene'] = None
                partition['trad_gene'] = None
            else:
                raise ValueError(f"Unrecognized chain type in V segments: {chain_type}")

            # Reorder and return the partition with fixed columns
            return partition[meta.keys()]

        # Apply the transformation lazily to all partitions
        mri_table = mri_table.map_partitions(process_partition, meta=meta)

        # Deduplicate the sequence table and standardize it
        sequence_table = mri_table.drop_duplicates(subset=['sequence'])
        sequence_table = standardize_sequence(sequence_table)

        # Add metadata lazily
        mri_table = mri_table.assign(
            repertoire_id=self.repertoire_id,
            study_id=self.study_id,
            category=self.category,
            molecule_type=self.molecule_type,
            host_organism='human',
            source='bulk_survey'
        )
        sequence_table = sequence_table.assign(source='bulk_survey')

        return mri_table, sequence_table

    
    def _parse_format_four(self):
        """
        Parse the AIRR dataset in format four using Dask without premature computations.

        Returns:
            tuple: Dask DataFrames for MRI and sequence tables.
        """
        # Filter rows with frame_type == "In" and select relevant columns
        processed = self.bulk_table[self.bulk_table['frame_type'] == "In"]
        processed = processed[['amino_acid', 'v_resolved', 'd_resolved', 'j_resolved']]

        # Fixed column names and metadata for the final DataFrame
        fixed_columns = [
            'tid', 'tra', 'trad_gene', 'traj_gene', 'trav_gene',
            'trb', 'trbd_gene', 'trbj_gene', 'trbv_gene', 'sequence'
        ]
        meta = OrderedDict({
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
        })

        # Function to process each partition
        def process_partition(partition):
            if partition.empty:
                return pd.DataFrame(columns=fixed_columns)  # Return an empty DataFrame with fixed columns

            formatted_rows = []
            for _, row in partition.iterrows():
                if "TCRA" in row['v_resolved']:
                    formatted_rows.append({
                        'trav_gene': row['v_resolved'],
                        'traj_gene': row['j_resolved'],
                        'trad_gene': row['d_resolved'],
                        'tra': row['amino_acid']
                    })
                elif "TCRB" in row['v_resolved']:
                    formatted_rows.append({
                        'trbv_gene': row['v_resolved'],
                        'trbj_gene': row['j_resolved'],
                        'trbd_gene': row['d_resolved'],
                        'trb': row['amino_acid']
                    })
                # Skip "TCRD" rows

            # Create a DataFrame from the formatted rows
            result = pd.DataFrame(formatted_rows)

            # Ensure all fixed columns are present
            for col in fixed_columns:
                if col not in result.columns:
                    result[col] = None  # Add missing columns with default value

            # Return DataFrame with reordered columns
            return result[fixed_columns]

        # Apply processing lazily to each partition
        mri_table = processed.map_partitions(process_partition, meta=meta)

        # Deduplicate the sequence table and standardize it
        sequence_table = mri_table.drop_duplicates(subset=['sequence'])
        sequence_table = standardize_sequence(sequence_table)

        # Add the 'sequence' column
        sequence_table['sequence'] = sequence_table[['tra', 'trb']].map_partitions(
            lambda df: df.apply(lambda x: ' '.join(x.dropna()) + ';', axis=1),
            meta=('sequence', 'str')
        )

        # Add metadata lazily to MRI and sequence tables
        mri_table = mri_table.assign(
            repertoire_id=self.repertoire_id,
            study_id=self.study_id,
            category=self.category,
            molecule_type=self.molecule_type,
            host_organism='human',
            source='bulk_survey'
        )
        sequence_table = sequence_table.assign(source='bulk_survey')

        return mri_table, sequence_table

    
    def _parse_format_five(self):
        """
        Parse the AIRR dataset in format five using Dask, ensuring lazy processing.

        Returns:
            tuple: Dask DataFrames for MRI and sequence tables.
        """
        # Filter rows where 'fuction' equals 'in-frame' and select relevant columns
        processed = self.bulk_table[self.bulk_table['fuction'] == "in-frame"]
        processed = processed[['#ID', 'V_ref', 'D_ref', 'J_ref', 'CDR3(aa)', 'amino_acid']]

        # Define the fixed columns and metadata
        fixed_columns = [
            'tid', 'tra', 'trad_gene', 'traj_gene', 'trav_gene',
            'trb', 'trbd_gene', 'trbj_gene', 'trbv_gene', 'sequence'
        ]
        meta = OrderedDict({col: 'object' for col in fixed_columns})

        # Define the row-level processing function
        def process_row(row):
            if "TRAV" in row['V_ref']:
                return {
                    'trav_gene': row['V_ref'],
                    'traj_gene': row['J_ref'],
                    'trad_gene': row['D_ref'],
                    'tra': row['CDR3(aa)']
                }
            elif "TRBV" in row['V_ref']:
                return {
                    'trbv_gene': row['V_ref'],
                    'trbj_gene': row['J_ref'],
                    'trbd_gene': row['D_ref'],
                    'trb': row['CDR3(aa)']
                }
            return None

        # Partition-level processing function
        def process_partition(partition):
            formatted_rows = []
            for _, row in partition.iterrows():
                result = process_row(row)
                if result is not None:
                    formatted_rows.append(result)

            # Create a DataFrame and ensure all fixed columns are present
            result_df = pd.DataFrame(formatted_rows)
            for col in fixed_columns:
                if col not in result_df.columns:
                    result_df[col] = None  # Add missing columns
            return result_df[fixed_columns]  # Reorder columns

        # Apply partition processing lazily
        mri_table = processed.map_partitions(process_partition, meta=meta)

        # Deduplicate the sequence table
        sequence_table = mri_table.drop_duplicates()

        # Add the 'sequence' column
        def generate_sequence(df):
            df['sequence'] = df[['tra', 'trb']].apply(
                lambda x: ' '.join(x.dropna()) + ';', axis=1
            )
            return df

        sequence_table = sequence_table.map_partitions(generate_sequence, meta=meta)

        # Add metadata lazily
        mri_table = mri_table.assign(
            repertoire_id=self.repertoire_id,
            study_id=self.study_id,
            host_organism='human',
            category=self.category,
            molecule_type=self.molecule_type,
            source='bulk_survey'
        )
        sequence_table = sequence_table.assign(source='bulk_survey')

        # Standardize the sequence table
        sequence_table = standardize_sequence(sequence_table)

        return mri_table, sequence_table

    
    def _parse_format_six(self):
        """
        Parse the AIRR dataset in format six using Dask, ensuring lazy processing.

        Returns:
            tuple: Dask DataFrames for MRI and sequence tables.
        """
        # Filter rows where 'fuction' equals 'in-frame' and select relevant columns
        airr_table = self.bulk_table[self.bulk_table['fuction'] == "in-frame"]
        airr_table = airr_table[['aminoAcid(CDR3 in lowercase)', 'vGene', 'dGene', 'jGene']]

        # Function to find the longest stretch of lowercase letters
        def longest_lowercase(s):
            lowercase_stretches = re.findall(r'[a-z]+', s)
            return max(lowercase_stretches, key=len, default='')

        # Apply the longest_lowercase function lazily
        airr_table['cdr3'] = airr_table['aminoAcid(CDR3 in lowercase)'].map_partitions(
            lambda partition: partition.apply(longest_lowercase), meta=('cdr3', 'str')
        )

        # Define the processing function for each partition
        def process_partition(partition):
            formatted_rows = []

            for _, row in partition.iterrows():
                if "TRAV" in row['vGene']:
                    formatted_rows.append({
                        'trav_gene': row['vGene'],
                        'traj_gene': row['jGene'],
                        'trad_gene': row['dGene'],
                        'tra': row['cdr3'],
                        'trb': None,
                        'trbv_gene': None,
                        'trbj_gene': None,
                        'trbd_gene': None
                    })
                elif "TRBV" in row['vGene']:
                    formatted_rows.append({
                        'trbv_gene': row['vGene'],
                        'trbj_gene': row['jGene'],
                        'trbd_gene': row['dGene'],
                        'trb': row['cdr3'],
                        'tra': None,
                        'trav_gene': None,
                        'traj_gene': None,
                        'trad_gene': None
                    })

            # Create a DataFrame and ensure consistent columns
            result = pd.DataFrame(formatted_rows)
            for col in self.fixed_columns:
                if col not in result.columns:
                    result[col] = None
            return result[self.fixed_columns]

        # Apply partition-wise processing
        meta = {col: 'object' for col in self.fixed_columns}
        mri_table = airr_table.map_partitions(process_partition, meta=meta)

        # Deduplicate the sequence table
        sequence_table = mri_table.drop_duplicates()

        # Add a 'sequence' column lazily
        def generate_sequence(df):
            df['sequence'] = df[['tra', 'trb']].apply(
                lambda x: ' '.join(x.dropna()) + ';', axis=1
            )
            return df

        sequence_table = sequence_table.map_partitions(generate_sequence, meta=meta)

        # Add metadata lazily
        mri_table = mri_table.assign(
            repertoire_id=self.repertoire_id,
            study_id=self.study_id,
            category=self.category,
            molecule_type=self.molecule_type,
            host_organism='human',
            source='bulk_survey'
        )
        sequence_table = sequence_table.assign(source='bulk_survey')

        # Standardize sequence table
        sequence_table = standardize_sequence(sequence_table)

        return mri_table, sequence_table



    def _parse_format_seven(self):
        """
        Parse the AIRR dataset in format seven using Dask for efficient processing.

        Returns:
            tuple: Dask DataFrames for MRI and sequence tables.
        """
        # Filter rows where 'sequenceStatus' is "In" and select relevant columns
        airr_table = self.bulk_table[self.bulk_table['sequenceStatus'] == "In"]
        airr_table = airr_table[['aminoAcid', 'vMaxResolved', 'dMaxResolved', 'jMaxResolved']]

        # Fill NaN values with a placeholder string
        airr_table = airr_table.fillna("NAN")

        # Define fixed columns and metadata
        fixed_columns = [
            'tid', 'tra', 'trad_gene', 'traj_gene', 'trav_gene',
            'trb', 'trbd_gene', 'trbj_gene', 'trbv_gene', 'sequence'
        ]
        meta = OrderedDict({col: 'object' for col in fixed_columns})

        # Define a function to process each partition
        def process_partition(partition):
            processed_rows = []
            for _, row in partition.iterrows():
                if "TCRAV" in row['vMaxResolved'] or "TCRAJ" in row['jMaxResolved']:
                    processed_rows.append({
                        'trav_gene': row['vMaxResolved'],
                        'traj_gene': row['jMaxResolved'],
                        'trad_gene': row['dMaxResolved'],
                        'tra': row['aminoAcid'],
                        'trb': None,
                        'trbv_gene': None,
                        'trbj_gene': None,
                        'trbd_gene': None,
                    })
                elif "TCRBV" in row['vMaxResolved'] or "TCRBJ" in row['jMaxResolved'] or "TCRBD" in row['dMaxResolved']:
                    processed_rows.append({
                        'trbv_gene': row['vMaxResolved'],
                        'trbj_gene': row['jMaxResolved'],
                        'trbd_gene': row['dMaxResolved'],
                        'trb': row['aminoAcid'],
                        'tra': None,
                        'trav_gene': None,
                        'traj_gene': None,
                        'trad_gene': None,
                    })

            result = pd.DataFrame(processed_rows)
            # Ensure all fixed columns are present
            for col in fixed_columns:
                if col not in result.columns:
                    result[col] = None
            return result[fixed_columns]

        # Apply the processing function to each partition lazily
        mri_table = airr_table.map_partitions(process_partition, meta=meta)

        # Drop duplicates in the sequence table
        sequence_table = mri_table.drop_duplicates()

        # Generate the 'sequence' column
        def generate_sequence(partition):
            partition['sequence'] = partition[['tra', 'trb']].apply(
                lambda row: ' '.join(filter(None, [row['tra'], row['trb']])) + ';', axis=1
            )
            return partition

        sequence_table = sequence_table.map_partitions(generate_sequence, meta=meta)

        # Add metadata lazily
        mri_table = mri_table.assign(
            repertoire_id=self.repertoire_id,
            study_id=self.study_id,
            category=self.category,
            molecule_type=self.molecule_type,
            host_organism='human',
            source='bulk_survey'
        )
        sequence_table = sequence_table.assign(source='bulk_survey')

        # Standardize sequence table
        sequence_table = standardize_sequence(sequence_table)

        return mri_table, sequence_table

    
    def _parse_format_nine(self):
        """
        Parse the AIRR dataset in format nine using Dask for efficient processing.

        Returns:
            tuple: Dask DataFrames for MRI and sequence tables.
        """
        # Filter for rows with frame_type "In" and select relevant columns
        airr_table = self.bulk_table[self.bulk_table['frame_type'] == "In"]
        airr_table = airr_table[['amino_acid', 'v_gene', 'd_gene', 'j_gene']]

        # Fill NaN values with a placeholder string
        airr_table = airr_table.fillna("NAN")

        # Define fixed columns and metadata
        fixed_columns = [
            'tid', 'tra', 'trad_gene', 'traj_gene', 'trav_gene',
            'trb', 'trbd_gene', 'trbj_gene', 'trbv_gene', 'sequence'
        ]
        meta = OrderedDict({col: 'object' for col in fixed_columns})

        # Define a function to process each partition
        def process_partition(partition):
            processed_rows = []
            for _, row in partition.iterrows():
                if "TCRAV" in row['v_gene']:
                    processed_rows.append({
                        'trav_gene': row['v_gene'],
                        'traj_gene': row['j_gene'],
                        'trad_gene': row['d_gene'],
                        'tra': row['amino_acid'],
                        'trb': None,
                        'trbv_gene': None,
                        'trbj_gene': None,
                        'trbd_gene': None,
                    })
                elif "TCRBV" in row['v_gene']:
                    processed_rows.append({
                        'trbv_gene': row['v_gene'],
                        'trbj_gene': row['j_gene'],
                        'trbd_gene': row['d_gene'],
                        'trb': row['amino_acid'],
                        'tra': None,
                        'trav_gene': None,
                        'traj_gene': None,
                        'trad_gene': None,
                    })

            result = pd.DataFrame(processed_rows)
            # Ensure all fixed columns are present
            for col in fixed_columns:
                if col not in result.columns:
                    result[col] = None
            return result[fixed_columns]

        # Apply the transformation to each partition
        mri_table = airr_table.map_partitions(process_partition, meta=meta)

        # Deduplicate the sequence table
        sequence_table = mri_table.drop_duplicates()

        # Generate the 'sequence' column
        def generate_sequence(partition):
            partition['sequence'] = partition[['tra', 'trb']].apply(
                lambda row: ' '.join(filter(None, [row['tra'], row['trb']])) + ';', axis=1
            )
            return partition

        sequence_table = sequence_table.map_partitions(generate_sequence, meta=meta)

        # Add metadata lazily
        mri_table = mri_table.assign(
            repertoire_id=self.repertoire_id,
            study_id=self.study_id,
            host_organism='human',
            category=self.category,
            molecule_type=self.molecule_type,
            source='bulk_survey'
        )
        sequence_table = sequence_table.assign(source='bulk_survey')

        # Standardize the sequence table
        sequence_table = standardize_sequence(sequence_table)

        return mri_table, sequence_table

    
    def _parse_format_eleven(self):
        """
        Parse the AIRR dataset in format eleven using Dask without relying on pandas.

        Returns:
            tuple: Dask DataFrames for MRI and sequence tables.
        """
        # Select relevant columns
        airr_table = self.bulk_table[['cdr3_b_aa', 'v_b_gene', 'j_b_gene']]

        # Define fixed columns and metadata
        fixed_columns = [
            'tid', 'tra', 'trad_gene', 'traj_gene', 'trav_gene',
            'trb', 'trbd_gene', 'trbj_gene', 'trbv_gene', 'sequence'
        ]
        meta = OrderedDict({col: 'object' for col in fixed_columns})

        # Define a function to process rows
        def process_row(row):
            if "TRAV" in row['v_b_gene']:
                return {
                    'tid': None,
                    'trav_gene': row['v_b_gene'],
                    'traj_gene': row['j_b_gene'],
                    'trad_gene': None,
                    'tra': row['cdr3_b_aa'],
                    'trbv_gene': None,
                    'trbj_gene': None,
                    'trbd_gene': None,
                    'trb': None,
                    'sequence': f"{row['cdr3_b_aa']};",
                }
            elif "TRBV" in row['v_b_gene']:
                return {
                    'tid': None,
                    'trav_gene': None,
                    'traj_gene': None,
                    'trad_gene': None,
                    'tra': None,
                    'trbv_gene': row['v_b_gene'],
                    'trbj_gene': row['j_b_gene'],
                    'trbd_gene': None,
                    'trb': row['cdr3_b_aa'],
                    'sequence': f"{row['cdr3_b_aa']};",
                }
            return None

        # Define a partition-level processing function
        def process_partition(partition):
            # Iterate over rows and process
            rows = [
                process_row(row) for _, row in partition.iterrows()
                if process_row(row) is not None
            ]
            # Convert to a list of dictionaries matching fixed_columns
            if not rows:
                return dd.utils.make_meta(meta)
            return rows

        # Apply the processing function to each partition
        formatted_sequences = airr_table.map_partitions(
            lambda partition: process_partition(partition), meta=meta
        )

        # Convert the list of dictionaries directly to a Dask DataFrame
        mri_table = dd.from_delayed(
            formatted_sequences.to_delayed(), meta=meta
        )

        # Drop duplicates lazily
        sequence_table = mri_table.drop_duplicates(subset='sequence')

        # Add metadata lazily
        mri_table = mri_table.assign(
            repertoire_id=self.repertoire_id,
            study_id=self.study_id,
            host_organism='human',
            category=self.category,
            molecule_type=self.molecule_type,
            source='bulk_survey'
        )
        sequence_table = sequence_table.assign(source='bulk_survey')

        # Ensure the sequence table has the fixed columns
        sequence_table = standardize_sequence(sequence_table)

        return mri_table, sequence_table
