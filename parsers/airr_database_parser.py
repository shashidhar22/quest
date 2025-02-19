import yaml
import ijson
import pandas as pd
import dask.dataframe as dd

from pathlib import Path
from dask import delayed 
from collections import OrderedDict

from .utils import parse_imgt_four_digit, transform_mhc_restriction, get_mhc_sequence

class DatabaseParser:
    def __init__(self, config_path, test=False):
        self.config_path = config_path
        self.test = test
        self.config = self._load_config()
        self.hla_dictionary = parse_imgt_four_digit(self.config['databases']['imgt']['hla_fasta'])
        self.output_path = self.config['outputs']['output_path']
        

    def _load_config(self):
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    

    def parse(self):
        """
        Parse the databases specified in the configuration file.

        Returns:
            tuple: A tuple containing:
                - pd.DataFrame: Combined sequence data table from CEDAR and IEDB.
                - pd.DataFrame: Combined MRI (minimum required information) table from CEDAR and IEDB.
        """
        mri_tables = []
        sequence_tables = []
        # Parse each database
        vdjdb_mri, vdjdb_sequence = self._parse_vdjdb()
        mcpas_mri, mcpas_sequence = self._parse_mcpas()
        tcrdb_mri, tcrdb_sequence = self._parse_tcrdb()
        iedb_mri, iedb_sequence = self._parse_iedb()
        ireceptor_mri, ireceptor_sequence = self._parse_ireceptor()
        # Combine tables
        mri_tables.extend([vdjdb_mri, mcpas_mri, tcrdb_mri, iedb_mri, ireceptor_mri])    
        sequence_tables.extend([vdjdb_sequence, mcpas_sequence, tcrdb_sequence, iedb_sequence, ireceptor_sequence]) 
        # Concatenate all DataFrames in the list into a single DataFrame in Dask
        mri_table = dd.concat(mri_tables, axis=0, interleave_partitions=True)
        sequence_table = dd.concat(sequence_tables, axis=0, interleave_partitions=True)
        return mri_table, sequence_table
        
    def _parse_vdjdb(self):
        """
        Parse the VDJdb dataset using Dask, ensuring large-scale compatibility.

        Returns:
            tuple: Dask DataFrames for MRI table and sequence table.
        """
        database_path = self.config['databases']['vdjdb']

        # Read the VDJdb data lazily
        vdjdb_table = dd.read_csv(
            database_path, sep="\t",
            assume_missing=True,
            dtype={'d.beta': 'str', 'meta.epitope.id': 'str'}
        )
        if self.test:
            vdjdb_table = vdjdb_table.sample(frac=0.1, random_state=21)
        # Define column renaming mapping
        relevant_columns = OrderedDict({
            'cdr3.alpha': 'tra',
            'v.alpha': 'trav_gene',
            'j.alpha': 'traj_gene',
            'cdr3.beta': 'trb',
            'v.beta': 'trbv_gene',
            'd.beta': 'trbd_gene',
            'j.beta': 'trbj_gene',
            'species': 'host_organism',
            'mhc.a': 'mhc_restriction',
            'mhc.b': 'mhc_restriction_two',
            'mhc.class': 'mhc_class',
            'antigen.epitope': 'peptide',
            'antigen.gene': 'epitope_source_molecule',
            'antigen.species': 'epitope_source_organism',
            'reference.id': 'study_id',
            'method.verification': 'assay_method',
            'meta.epitope.id': 'epitope_reference_name',
            'meta.tissue': 'source_tissue',
            'meta.donor.MHC': 'mhc_profile',
        })

        # Filter and rename columns lazily
        vdjdb_table = vdjdb_table.loc[
            (vdjdb_table['species'] == 'HomoSapiens') & (vdjdb_table['vdjdb.score'] > 0),
            list(relevant_columns.keys()) + ['vdjdb.score']
        ].rename(columns=relevant_columns)

        # Drop unnecessary columns
        mri_table = vdjdb_table.drop(columns=['vdjdb.score'])

        # Select relevant columns for sequence table
        sequence_columns = [
            'trav_gene', 'traj_gene', 'tra',
            'trbv_gene', 'trbd_gene', 'trbj_gene', 'trb',
            'peptide', 'mhc_restriction', 'mhc_restriction_two'
        ]
        sequence_table = mri_table[sequence_columns]

        # Transform MHC restriction values lazily
        sequence_table['mhc_restriction'] = sequence_table['mhc_restriction'].map_partitions(
            lambda partition_series: transform_mhc_restriction(partition_series, fasta_dict=self.hla_dictionary),
            meta=('mhc_restriction', 'str')
        )
        sequence_table['mhc_restriction_two'] = sequence_table['mhc_restriction_two'].map_partitions(
            lambda partition_series: transform_mhc_restriction(partition_series, fasta_dict=self.hla_dictionary),
            meta=('mhc_restriction_two', 'str')
        )

        # Map MHC sequences lazily
        sequence_table['mhc_one'] = sequence_table['mhc_restriction'].map_partitions(
            lambda col: col.apply(lambda x: get_mhc_sequence(x, fasta_dict=self.hla_dictionary)),
            meta=('mhc_one', 'str')
        )
        sequence_table['mhc_two'] = sequence_table['mhc_restriction_two'].map_partitions(
            lambda col: col.apply(lambda x: get_mhc_sequence(x, fasta_dict=self.hla_dictionary)),
            meta=('mhc_two', 'str')
        )

        # Rename and create the 'sequence' column lazily
        sequence_table = sequence_table.rename(columns={
            'mhc_restriction': 'mhc_one_id', 'mhc_restriction_two': 'mhc_two_id'
        })
        sequence_table['sequence'] = sequence_table[['tra', 'trb', 'peptide', 'mhc_one', 'mhc_two']].map_partitions(
            lambda df: df.apply(lambda row: ' '.join(str(x) for x in row if pd.notna(x)) + ';', axis=1),
            meta=('sequence', 'str')
        )

        # Add metadata lazily
        sequence_table['source'] = 'vdjdb'

        # Drop duplicates lazily
        sequence_table = sequence_table.drop_duplicates()

        return mri_table, sequence_table
        
    def _parse_tcrdb(self):
        """
        Parse the TCRdb dataset using Dask, ensuring compatibility with large datasets.

        Returns:
            tuple: Dask DataFrames for MRI table and sequence table.
        """
        database_path = Path(self.config['databases']['tcrdb'])
        
        # Gather all .tsv files in the directory and its subdirectories
        tcrdb_file_list = list(database_path.rglob("*.tsv"))

        if not tcrdb_file_list:
            raise FileNotFoundError(f"No TSV files found in {database_path}")

        # Define metadata for Dask
        meta = OrderedDict({
            'study_id': 'str',
            'repertoire_id': 'str',
            'trbv_gene': 'str',
            'trbd_gene': 'str',
            'trbj_gene': 'str',
            'trb': 'str',
            'sequence': 'str'
        })

        # Load all files lazily using Dask
        tcrdb_tables = []
        for file_path in tcrdb_file_list:
            study_id = file_path.stem  # Extract study_id from filename

            # Read the file lazily
            tcr_table = dd.read_csv(file_path, sep="\t", dtype="str", assume_missing=True)
            if self.test:
                tcr_table = tcr_table.sample(frac=0.1, random_state=21)
            # Rename columns for consistency
            tcr_table = tcr_table.rename(columns={
                'RunId': 'repertoire_id',
                'Vregion': 'trbv_gene',
                'Dregion': 'trbd_gene',
                'Jregion': 'trbj_gene',
                'AASeq': 'trb'
            })

            # Drop unnecessary columns
            tcr_table = tcr_table.drop(columns=['cloneFraction'], errors='ignore')

            # Replace 'Unknown' with NaN
            tcr_table = tcr_table.replace('Unknown', '')

            # Assign metadata lazily
            tcr_table = tcr_table.assign(study_id=study_id)

            # Select only required columns
            tcr_table = tcr_table[['study_id', 'repertoire_id', 'trbv_gene', 'trbd_gene', 'trbj_gene', 'trb']]
            
            # Append processed table to list
            tcrdb_tables.append(tcr_table)

        # Concatenate all processed tables lazily
        mri_table = dd.concat(tcrdb_tables, axis=0, interleave_partitions=True)

        # Create the sequence table
        sequence_table = mri_table[['trbv_gene', 'trbd_gene', 'trbj_gene', 'trb']].copy()
        sequence_table = sequence_table.assign(
            sequence=sequence_table['trb'] + ';',
            source='tcrdb'
        )

        # Drop duplicates lazily
        sequence_table = sequence_table.drop_duplicates()

        return mri_table, sequence_table

    def _parse_mcpas(self):
        """
        Parse the McPAS-TCR dataset using Dask, ensuring compatibility with large datasets.

        Returns:
            tuple: Dask DataFrames for MRI table and sequence table.
        """
        database_path = self.config['databases']['mcpas_tcr']
        
        # Read the McPAS-TCR data using Dask
        mcpastcr_table = dd.read_csv(database_path, assume_missing=True, dtype="str")
        if self.test:
                mcpastcr_table = mcpastcr_table.sample(frac=0.1, random_state=21)
        # Rename columns for consistency
        rename_columns = {
            'TRAV': 'trav_gene',
            'TRAJ': 'traj_gene',
            'CDR3.alpha.aa': 'tra_junction_aa',
            'TRBV': 'trbv_gene',
            'TRBD': 'trbd_gene',
            'TRBJ': 'trbj_gene',
            'CDR3.beta.aa': 'trb_junction_aa',
            'PubMed.ID': 'study_id',
            'Category': 'host_condition',
            'Species': 'host_organism',
            'Epitope.peptide': 'peptide',
            'Antigen.identification.method': 'assay_method',
            'Antigen.protein': 'epitope_source_molecule',
            'Protein.ID': 'epitope_reference_name',
            'Pathology': 'epitope_source_organism',
            'MHC': 'mhc_restriction'
        }
        mcpastcr_table = mcpastcr_table.rename(columns=rename_columns)

        # Add metadata columns lazily
        mcpastcr_table = mcpastcr_table.assign(data_source='McPAS-TCR', study_id_type='PMID')

        # Standardized MRI table with selected metadata columns
        metadata_columns = [
            'data_source', 'study_id', 'study_id_type', 'host_organism', 'trav_gene', 'traj_gene',
            'tra_junction_aa', 'trbv_gene', 'trbd_gene', 'trbj_gene', 'trb_junction_aa',
            'peptide', 'epitope_reference_name', 'epitope_source_molecule', 'epitope_source_organism',
            'mhc_restriction', 'assay_method'
        ]
        mri_table = mcpastcr_table[metadata_columns]

        # Sequence Table Columns
        sequence_columns = [
            'trav_gene', 'traj_gene', 'tra_junction_aa', 'trbv_gene', 'trbd_gene',
            'trbj_gene', 'trb_junction_aa', 'peptide', 'mhc_restriction'
        ]
        sequence_table = mri_table[sequence_columns].copy()

        # Process MHC restriction using map_partitions
        sequence_table['mhc_restriction'] = sequence_table['mhc_restriction'].map_partitions(
            lambda partition_series: transform_mhc_restriction(partition_series, fasta_dict=self.hla_dictionary),
            meta=('mhc_restriction', 'str')
        )

        # Map MHC sequences based on restriction
        sequence_table['mhc_one'] = sequence_table['mhc_restriction'].map_partitions(
            lambda df: df.apply(lambda x: get_mhc_sequence(x, fasta_dict=self.hla_dictionary)),
            meta=('mhc_one', 'str')
        )

        # Expand peptides lazily
        sequence_table = sequence_table.assign(peptide=sequence_table['peptide'].str.split('/')).explode('peptide')

        # Rename junction columns for consistency
        sequence_table = sequence_table.rename(
            columns={'tra_junction_aa': 'tra', 'trb_junction_aa': 'trb'}
        )

        # Generate the 'sequence' column lazily
        sequence_table['sequence'] = sequence_table[['tra', 'trb', 'peptide', 'mhc_one']].map_partitions(
            lambda df: df.apply(lambda row: ' '.join(str(x) for x in row if pd.notna(x)) + ';', axis=1),
            meta=('sequence', 'str')
        )

        # Final selection and deduplication
        sequence_columns_final = [
            'trav_gene', 'traj_gene', 'tra', 'trbv_gene', 'trbj_gene', 'trbd_gene',
            'trb', 'peptide', 'mhc_restriction', 'mhc_one', 'sequence'
        ]
        sequence_table = sequence_table[sequence_columns_final].assign(source='McPAS-TCR')
        sequence_table = sequence_table.drop_duplicates()

        return mri_table, sequence_table
    
    def _parse_iedb_tcr(self, source):
        """
        Parses IEDB T-cell assay data using Dask for efficient handling of large datasets.

        Args:
            source (str): Data source key from the config file.

        Returns:
            tuple: Dask DataFrames for MRI table and sequence table.
        """
        database_path = self.config['databases'][source]['tcell_assay']

        # Load T-cell assay data lazily
        tcell_table = dd.read_csv(
            database_path, 
            sep="\t",
            dtype="str",
            names=[
                'study_id', 'epitope_type', 'peptide', 'epitope_reference_name',
                'epitope_source_molecule', 'epitope_source_organism',
                'epitope_source_species', 'host_organism', 'host_population',
                'host_sex', 'host_age', 'host_mhc_profile', 'assay_method',
                'assay_response', 'assay_outcome', 'assay_subject_count',
                'assay_positive_count', 'source_tissue', 'mhc_restriction'
            ],
            header=0
        )
        if self.test:
                tcell_table = tcell_table.sample(frac=0.1, random_state=21)
        # Drop rows where 'study_id' is missing and keep only positive assay outcomes
        tcell_table = tcell_table.dropna(subset=['study_id'])
        tcell_table = tcell_table[tcell_table['assay_outcome'] == "Positive"]

        # Define MRI Table with selected metadata
        mri_table = tcell_table.assign(
            data_source=f"{source}_tcell_assay",
            study_id_type="PMID"
        )[[  # Keep only relevant columns
            'data_source', 'study_id', 'study_id_type', 'host_organism', 
            'host_population', 'host_age', 'host_sex', 'host_mhc_profile', 
            'source_tissue', 'epitope_type', 'peptide', 'epitope_reference_name', 
            'epitope_source_molecule', 'epitope_source_organism', 'mhc_restriction', 
            'assay_method', 'assay_response', 'assay_outcome', 'assay_subject_count', 
            'assay_positive_count'
        ]]

        # Create Sequence Table from relevant columns
        sequence_table = mri_table[['peptide', 'mhc_restriction']].copy()

        # Process MHC alleles (split multiple MHCs if present)
        def process_mhc(df):
            df['mhc_restriction_two'] = df['mhc_restriction'].str.split('/').str[1]
            df['mhc_restriction'] = df['mhc_restriction'].str.split('/').str[0]
            return df

        sequence_table = sequence_table.map_partitions(process_mhc, meta={
            'peptide': 'str',
            'mhc_restriction': 'str',
            'mhc_restriction_two': 'str'
        })

        sequence_table['mhc_restriction'] = sequence_table['mhc_restriction'].map_partitions(
            lambda partition_series: transform_mhc_restriction(partition_series, fasta_dict=self.hla_dictionary),
            meta=('mhc_restriction', 'str')
        )


        sequence_table['mhc_restriction_two'] = sequence_table['mhc_restriction_two'].map_partitions(
            lambda partition_series: transform_mhc_restriction(partition_series, fasta_dict=self.hla_dictionary),
            meta=('mhc_restriction_two', 'str')
        )

        # Map MHC alleles to their sequences
        sequence_table['mhc_one'] = sequence_table['mhc_restriction'].map_partitions(
            lambda df: df.apply(lambda x: get_mhc_sequence(x, fasta_dict=self.hla_dictionary)),
            meta=('mhc_one', 'str')
        )
        sequence_table['mhc_two'] = sequence_table['mhc_restriction_two'].map_partitions(
            lambda df: df.apply(lambda x: get_mhc_sequence(x, fasta_dict=self.hla_dictionary)),
            meta=('mhc_two', 'str')
        )

        # Standardize peptide sequences by extracting the first peptide when multiple exist
        sequence_table['peptide'] = sequence_table['peptide'].str.split('+').str[0].str.strip()

        # Generate 'sequence' column lazily
        sequence_table['sequence'] = sequence_table[['peptide', 'mhc_one', 'mhc_two']].map_partitions(
            lambda df: df.apply(lambda row: ' '.join(str(x) for x in row if pd.notna(x)) + ';', axis=1),
            meta=('sequence', 'str')
        )

        # Rename MHC columns for clarity
        sequence_table = sequence_table.rename(columns={
            'mhc_restriction': 'mhc_one_id',
            'mhc_restriction_two': 'mhc_two_id'
        })[['peptide', 'mhc_one_id', 'mhc_one', 'mhc_two_id', 'mhc_two', 'sequence']]

        return mri_table, sequence_table
    
    def _parse_iedb_mhc(self, source):
        """
        Parses MHC ligand assay data from IEDB or CEDAR using Dask for efficient large-scale processing.

        Args:
            source (str): Data source key from the config file.

        Returns:
            tuple: Dask DataFrames for MRI table and sequence table.
        """
        # Define column names based on the database source
        col_names = {
            "iedb": [
                'study_id', 'epitope_type', 'peptide', 'epitope_reference_name',
                'epitope_source_molecule', 'epitope_source_organism',
                'epitope_source_species', 'host_organism', 'host_population',
                'host_sex', 'host_age', 'host_mhc_profile', 'assay_method',
                'assay_response', 'assay_outcome', 'assay_subject_count',
                'assay_positive_count', 'source_tissue', 'mhc_restriction'
            ],
            "cedar": [
                'study_id', 'epitope_type', 'peptide', 'epitope_source_molecule',
                'epitope_source_organism', 'epitope_source_species',
                'host_organism', 'host_population', 'host_sex', 'host_age',
                'host_mhc_profile', 'assay_method', 'assay_response',
                'assay_subject_count', 'assay_positive_count',
                'source_tissue', 'mhc_restriction'
            ]
        }
        
        # Validate source type
        column_names = col_names.get(source.lower())
        if not column_names:
            raise ValueError(f"Unsupported database: {source}")

        # Load the dataset lazily
        database_path = self.config['databases'][source]['mhc_ligand']
        mhc_ligand_table = dd.read_csv(
            database_path,
            sep="\t",
            names=column_names,
            header=0,
            dtype="str"
        ).dropna(subset=['study_id'])

        if self.test:
                mhc_ligand_table = mhc_ligand_table.sample(frac=0.1, random_state=21)

        # Filter for positive assay outcomes (if column exists)
        if 'assay_outcome' in mhc_ligand_table.columns:
            mhc_ligand_table = mhc_ligand_table[mhc_ligand_table['assay_outcome'] == "Positive"]

        # Assign metadata lazily
        mri_table = mhc_ligand_table.assign(
            data_source=f"{source}_mhc_ligand_assay",
            study_id_type="PMID"
        )[[  # Select relevant columns
            'data_source', 'study_id', 'study_id_type', 'host_organism',
            'host_population', 'host_age', 'host_sex', 'host_mhc_profile',
            'source_tissue', 'epitope_type', 'peptide', 'epitope_source_molecule',
            'epitope_source_organism', 'mhc_restriction', 'assay_method',
            'assay_response', 'assay_subject_count', 'assay_positive_count'
        ]]

        # Create sequence table
        sequence_table = mri_table[['peptide', 'mhc_restriction']]

        # Efficiently split and process MHC chains using Dask `.str.split()`
        sequence_table['mhc_restriction_two'] = sequence_table['mhc_restriction'].str.split('/').str[1]
        sequence_table['mhc_restriction'] = sequence_table['mhc_restriction'].str.split('/').str[0]

        # Transform MHC alleles into standardized IMGT HLA notation
        sequence_table['mhc_restriction'] = sequence_table['mhc_restriction'].map_partitions(
            lambda partition_series: transform_mhc_restriction(partition_series, fasta_dict=self.hla_dictionary),
            meta=('mhc_restriction', 'str')
        )
        sequence_table['mhc_restriction_two'] = sequence_table['mhc_restriction_two'].map_partitions(
            lambda partition_series: transform_mhc_restriction(partition_series, fasta_dict=self.hla_dictionary),
            meta=('mhc_restriction_two', 'str')
        )

        # Map MHC alleles to sequences
        sequence_table['mhc_one'] = sequence_table['mhc_restriction'].map_partitions(
            lambda df: df.apply(lambda x: get_mhc_sequence(x, fasta_dict=self.hla_dictionary)),
            meta=('mhc_one', 'str')
        )
        sequence_table['mhc_two'] = sequence_table['mhc_restriction_two'].map_partitions(
            lambda df: df.apply(lambda x: get_mhc_sequence(x, fasta_dict=self.hla_dictionary)),
            meta=('mhc_two', 'str')
        )

        # Generate the 'sequence' column lazily
        sequence_table['sequence'] = sequence_table[['peptide', 'mhc_one', 'mhc_two']].map_partitions(
            lambda df: df.apply(lambda row: ' '.join(str(x) for x in row if pd.notna(x)) + ';', axis=1),
            meta=('sequence', 'str')
        )

        # Rename and select final columns
        sequence_table = sequence_table.rename(columns={
            'mhc_restriction': 'mhc_one_id',
            'mhc_restriction_two': 'mhc_two_id'
        })[['peptide', 'mhc_one_id', 'mhc_one', 'mhc_two_id', 'mhc_two', 'sequence']]

        return mri_table, sequence_table
    
    def _parse_iedb_receptor(self, source):
        """
        Parses receptor assay data from IEDB or CEDAR using Dask for efficient large-scale processing.

        Args:
            source (str): Data source key from the config file.

        Returns:
            tuple: Dask DataFrames for MRI table and sequence table.
        """
        # Define column names based on the database source
        col_names = {
            "iedb": [
                'study_id', 'recepetor_reference_name', 'receptor_type', 'peptide',
                'epitope_source_molecule', 'epitope_source_organism',
                'assay_method', 'mhc_restriction', 'chain_one_type', 'trav_gene',
                'trad_gene', 'traj_gene', 'tra_protein_sequence', 'tra_junction_aa',
                'tra_cdr1', 'tra_cdr2', 'chain_two_type', 'trbv_gene', 'trbd_gene',
                'trbj_gene', 'trb_protein_sequence', 'trb_junction_aa', 'trb_cdr1',
                'trb_cdr2'
            ],
            "cedar": [
                'recepetor_reference_name', 'study_id', 'peptide',
                'epitope_source_molecule', 'epitope_source_organism',
                'assay_method', 'chain_one_type', 'trav_gene', 'trad_gene',
                'traj_gene', 'tra_protein_sequence', 'tra_junction_aa',
                'tra_cdr1', 'tra_cdr2', 'chain_two_type', 'trbv_gene', 'trbd_gene',
                'trbj_gene', 'trb_protein_sequence', 'trb_junction_aa', 'trb_cdr1',
                'trb_cdr2'
            ]
        }

        # Validate source type
        column_names = col_names.get(source.lower())
        if not column_names:
            raise ValueError(f"Unsupported database: {source}")

        # Load the dataset lazily
        database_path = self.config['databases'][source]['receptor']
        receptor_table = dd.read_csv(
            database_path,
            sep="\t",
            names=column_names,
            header=0,
            dtype="str"
        )

        if self.test:
                receptor_table = receptor_table.sample(frac=0.1, random_state=21)

        # Process `study_id` for CEDAR datasets (extract numeric ID)
        if source.lower() == "cedar":
            receptor_table['study_id'] = receptor_table['study_id'].str.extract(r'(\d{7})')[0]

        # Standardize `study_id` format lazily
        receptor_table['study_id'] = receptor_table['study_id'].astype(str).map_partitions(
            lambda df: df.apply(lambda x: f"{source.upper()}{x}" if pd.notna(x) else x),
            meta=('study_id', 'str')
        )

        # Define the MRI table
        mri_table = receptor_table.assign(
            data_source=f"{source}_receptor_table",
            study_id_type=source.upper(),
            host_organism="human"  # Assume host organism is human
        )[[  # Keep only relevant columns
            'data_source', 'study_id', 'study_id_type', 'host_organism',
            'trav_gene', 'trad_gene', 'traj_gene', 'tra_junction_aa',
            'trbv_gene', 'trbd_gene', 'trbj_gene', 'trb_junction_aa',
            'peptide', 'epitope_source_molecule', 'epitope_source_organism'
        ]]

        # Define the sequence table
        sequence_table = mri_table[[
            'trav_gene', 'trad_gene', 'traj_gene', 'trbv_gene', 'trbd_gene',
            'trbj_gene', 'peptide', 'tra_junction_aa', 'trb_junction_aa'
        ]].rename(columns={
            'tra_junction_aa': 'tra',
            'trb_junction_aa': 'trb'
        })

        # Standardize peptide sequences using `.str.split()`
        sequence_table['peptide'] = sequence_table['peptide'].str.split('+').str[0].str.strip()

        # Drop duplicates lazily
        sequence_table = sequence_table.drop_duplicates()

        # Generate the 'sequence' column lazily
        sequence_table['sequence'] = sequence_table[['tra', 'trb', 'peptide']].map_partitions(
            lambda df: df.apply(lambda row: ' '.join(str(x) for x in row if pd.notna(x)) + ';', axis=1),
            meta=('sequence', 'str')
        )

        # Select final columns for the sequence table
        sequence_table = sequence_table[[
            'trav_gene', 'trad_gene', 'traj_gene', 'tra',
            'trbv_gene', 'trbd_gene', 'trbj_gene', 'trb',
            'peptide', 'sequence'
        ]]

        return mri_table, sequence_table
    
    def _parse_iedb(self):
        """
        Parses data from IEDB and CEDAR using Dask, ensuring efficient large-scale processing.

        Returns:
            tuple: Dask DataFrames for MRI table and sequence table.
        """
        # Load and parse data from CEDAR
        cedar_tcell_mri_table, cedar_tcell_sequence_table = self._parse_iedb_tcr('cedar')
        cedar_mhc_mri_table, cedar_mhc_sequence_table = self._parse_iedb_mhc('cedar')
        cedar_tcr_mri_table, cedar_tcr_sequence_table = self._parse_iedb_receptor('cedar')
        print('Parsed CEDAR')

        # Concatenate CEDAR MRI and sequence tables lazily
        cedar_mri_table = dd.concat([cedar_tcell_mri_table, cedar_tcr_mri_table, cedar_mhc_mri_table]).drop_duplicates()
        cedar_sequence_table = dd.concat([cedar_tcell_sequence_table, cedar_tcr_sequence_table, cedar_mhc_sequence_table]).drop_duplicates()
        cedar_sequence_table = cedar_sequence_table.assign(source='cedar')

        # Load and parse data from IEDB
        iedb_tcell_mri_table, iedb_tcell_sequence_table = self._parse_iedb_tcr('iedb')
        iedb_mhc_mri_table, iedb_mhc_sequence_table = self._parse_iedb_mhc('iedb')
        iedb_tcr_mri_table, iedb_tcr_sequence_table = self._parse_iedb_receptor('iedb')
        print('Parsed IEDB')

        # Concatenate IEDB MRI and sequence tables lazily
        iedb_mri_table = dd.concat([iedb_tcell_mri_table, iedb_tcr_mri_table, iedb_mhc_mri_table]).drop_duplicates()
        iedb_sequence_table = dd.concat([iedb_tcell_sequence_table, iedb_tcr_sequence_table, iedb_mhc_sequence_table]).drop_duplicates()
        iedb_sequence_table = iedb_sequence_table.assign(source='iedb')

        # Combine data from both CEDAR and IEDB databases lazily
        sequence_table = dd.concat([cedar_sequence_table, iedb_sequence_table]).reset_index(drop=True)
        mri_table = dd.concat([cedar_mri_table, iedb_mri_table]).reset_index(drop=True)

        return mri_table, sequence_table

    def _parse_paired_ireceptor(self, source):
        """
        Parses iReceptor paired-chain data using Dask, ensuring large-scale compatibility.

        Args:
            source (str): Source identifier (e.g., 'tcr', 'bcr').

        Returns:
            tuple: Dask DataFrames for MRI table and sequence table.
        """
        database_path = self.config['databases']['ireceptor'][source]['database']
        
        # Read iReceptor data lazily
        database_table = dd.read_csv(
            database_path,
            sep="\t",
            dtype=str,
            usecols=[
                'data_processing_id', 'repertoire_id', 'cell_id', 'clone_id', 'productive',
                'locus', 'v_call', 'd_call', 'j_call', 'junction_aa'
            ]
        )

        if self.test:
                database_table = database_table.sample(frac=0.1, random_state=21)

        # Filter productive rows lazily
        database_table = database_table[database_table['productive'] == 'T']

        # Function to process TRA and TRB chains
        def process_chain(df, chain_type, rename_map):
            df = df[df['locus'] == chain_type].rename(columns=rename_map)
            return df.drop(columns=['locus', 'repertoire_id', 'productive'], errors='ignore')

        # Column mappings
        tra_map = {'v_call': 'trav_gene', 'd_call': 'trad_gene', 'j_call': 'traj_gene', 'junction_aa': 'tra'}
        trb_map = {'v_call': 'trbv_gene', 'd_call': 'trbd_gene', 'j_call': 'trbj_gene', 'junction_aa': 'trb'}

        # Process TRA and TRB chains lazily
        tra_table = database_table.map_partitions(process_chain, chain_type='TRA', 
            rename_map=tra_map, meta={'trav_gene': 'str', 'trad_gene': 'str', 
                                      'traj_gene': 'str', 'tra': 'str', 
                                      'cell_id': 'str', 'clone_id': 'str', 'data_processing_id': 'str'})
        
        trb_table = database_table.map_partitions(process_chain, chain_type='TRB',
            rename_map=trb_map, meta={'trbv_gene': 'str', 'trbd_gene': 'str',
                                      'trbj_gene': 'str', 'trb': 'str',
                                      'cell_id': 'str', 'clone_id': 'str', 'data_processing_id': 'str'})

        # Merge TRA and TRB tables lazily
        mri_table = dd.merge(tra_table, trb_table, on=['data_processing_id', 'cell_id', 'clone_id'], how='outer').rename(columns={'data_processing_id': 'repertoire_id'})

        # Drop unnecessary columns
        mri_table = mri_table.drop(columns=['cell_id', 'clone_id'], errors='ignore')

        # Create sequence table lazily
        sequence_table = mri_table[['trav_gene', 'traj_gene', 'tra', 'trbv_gene', 'trbd_gene', 'trbj_gene', 'trb']]

        # Generate 'sequence' column lazily
        sequence_table['sequence'] = sequence_table[['tra', 'trb']].map_partitions(
            lambda df: df.apply(lambda row: ' '.join(str(x) for x in row if pd.notna(x)) + ';', axis=1),
            meta=('sequence', 'str')
        )

        # Drop duplicates lazily
        sequence_table = sequence_table.drop_duplicates()
        # Add source column lazily
        sequence_table = sequence_table.assign(source=f"ireceptor_{source}")

        return mri_table, sequence_table
    
    def _parse_bulk_ireceptor(self, source):
        """
        Parses bulk iReceptor data using Dask, ensuring large-scale compatibility.

        Args:
            source (str): Source identifier (e.g., 'tcr', 'bcr').

        Returns:
            tuple: Dask DataFrames for MRI table and sequence table.
        """
        database_path = self.config['databases']['ireceptor'][source]['database']
        
        # Read the bulk iReceptor data lazily
        tcr_table = dd.read_csv(
            database_path,
            sep="\t",
            dtype=str,
            usecols=[
                'repertoire_id', 'cell_id', 'clone_id', 'productive', 'locus',
                'v_call', 'd_call', 'j_call', 'junction_aa'
            ]
        )

        if self.test:
                tcr_table = tcr_table.sample(frac=0.1, random_state=21)
        # Filter for productive sequences lazily
        mri_table = tcr_table[tcr_table['productive'] == 'T'].drop(columns=['productive'])

        # Define renaming maps for TRA and TRB chains
        rename_maps = {
            'TRA': {'v_call': 'trav_gene', 'd_call': 'trad_gene', 'j_call': 'traj_gene', 'junction_aa': 'tra'},
            'TRB': {'v_call': 'trbv_gene', 'd_call': 'trbd_gene', 'j_call': 'trbj_gene', 'junction_aa': 'trb'}
        }

        # Function to process TRA and TRB chains
        def process_chain(df, locus, rename_map):
            if locus == 'TRA':
                correct_order = [
                    'tra', 
                    'trav_gene', 
                    'trad_gene', 
                    'traj_gene', 
                    'repertoire_id', 
                    'cell_id', 
                    'clone_id',
                    'locus'
                ]
                df = df.reindex(columns=correct_order)
            elif locus == 'TRB':
                correct_order = [
                    'trb', 
                    'trbv_gene', 
                    'trbd_gene', 
                    'trbj_gene', 
                    'repertoire_id', 
                    'cell_id', 
                    'clone_id',
                    'locus'
                ]
                df = df.reindex(columns=correct_order)
            return df[df['locus'] == locus].rename(columns=rename_map).drop(columns=['locus'], errors='ignore')
        

        # Process TRA and TRB lazily using map_partitions
        tra_table = mri_table.map_partitions(process_chain, locus='TRA', rename_map=rename_maps['TRA'], meta={'tra': 'str', 'trav_gene': 'str', 'trad_gene': 'str', 'traj_gene': 'str', 'repertoire_id': 'str', 'cell_id': 'str', 'clone_id': 'str'})
        trb_table = mri_table.map_partitions(process_chain, locus='TRB', rename_map=rename_maps['TRB'], meta={'trb': 'str', 'trbv_gene': 'str', 'trbd_gene': 'str', 'trbj_gene': 'str', 'repertoire_id': 'str', 'cell_id': 'str', 'clone_id': 'str'})

        # Merge TRA and TRB tables lazily
        sequence_table = dd.concat([tra_table, trb_table], interleave_partitions=True)

        # Generate 'sequence' column lazily
        sequence_table['sequence'] = sequence_table[['tra', 'trb']].map_partitions(
            lambda df: df.apply(lambda row: ' '.join(str(x) for x in row if pd.notna(x)) + ';', axis=1),
            meta=('sequence', 'str')
        )

        # Drop unnecessary columns
        sequence_table = sequence_table.drop(columns=['repertoire_id', 'cell_id', 'clone_id'], errors='ignore')

        # Add source column lazily
        sequence_table = sequence_table.assign(source=f"ireceptor_{source}")

        return mri_table, sequence_table
    
    def _parse_json_ireceptor(self, source):
        """
        Parses iReceptor JSON metadata lazily using Dask, ensuring large-scale compatibility.

        Args:
            source (str): Source identifier (e.g., 'ireceptor', 'other_db').

        Returns:
            dd.DataFrame: Dask DataFrame with parsed metadata.
        """
        metadata_path = self.config['databases']['ireceptor'][source]['metadata']

        # Define function to process a single JSON entry
        def process_repertoire(repertoire, database):
            """
            Parses a single repertoire JSON str.
            """
            try:
                # Extract primary metadata fields
                repertoire_dict = {
                    'study_id': repertoire['study']['study_id'],
                    'repertoire_id': (
                        repertoire['repertoire_id']
                        if database == "ireceptor"
                        else repertoire['data_processing'][0]['data_processing_id']
                    ),
                    'host_organism': repertoire['subject']['species']['label'],
                    'condition_studies': (
                        repertoire['subject']['diagnosis'][0]['study_group_description']
                        + (' ' + repertoire['subject']['diagnosis'][0]['disease_diagnosis']['label']
                        if repertoire['subject']['diagnosis'][0]['disease_diagnosis']['label'] else '')
                    ),
                    'age': repertoire['subject'].get('age', ''),
                    'sex': repertoire['subject'].get('sex', ''),
                    'population_surveyed': repertoire['subject'].get('race', ''),
                    'source_tissue': repertoire['sample'][0]['tissue']['label'] if repertoire.get('sample') else '',
                }

                # Extract MHC profile (if available)
                mhc_list = set()
                if 'genotype' in repertoire['subject']:
                    for mhc_genotypes in repertoire['subject']['genotype'].get('mhc_genotype_set', {}).get('mhc_genotype_list', []):
                        for allele in mhc_genotypes.get('mhc_alleles', []):
                            if allele.get('allele_designation'):
                                mhc_list.add(allele['allele_designation'])

                repertoire_dict['mhc_profile'] = ','.join(sorted(mhc_list)) if mhc_list else ''
                return repertoire_dict
            except Exception as e:
                print(f"Error processing repertoire: {e}")
                return None  # Skip invalid entries

        # Stream JSON lazily and process repertoires in parallel
        with open(metadata_path, 'r') as meta_file:
            repertoires = [
                delayed(process_repertoire)(item, source)
                for item in ijson.items(meta_file, "Repertoire.item")
            ]

        # Convert processed JSON strs to a Dask DataFrame lazily
        metadata_table = dd.from_delayed([
            delayed(pd.DataFrame)([r]) for r in repertoires if r is not None
        ])

        # Drop duplicate rows lazily
        metadata_table = metadata_table.drop_duplicates()

        return metadata_table
    
    def _parse_ireceptor(self):
        """
        Parses multiple iReceptor datasets using Dask, ensuring scalable and efficient processing.

        Returns:
            tuple: (mri_table, sequence_table) as Dask DataFrames.
        """
        # Load HLA dictionary lazily
        hla_dictionary = parse_imgt_four_digit(self.config['databases']['imgt']['hla_fasta'])

        # Define databases and parsing functions
        databases = {
            "airr_covid": (self._parse_paired_ireceptor, 'airr_covid'),
            "paone": (self._parse_bulk_ireceptor, 'paone'),
            "patwo": (self._parse_bulk_ireceptor, 'patwo'),
            "pathree": (self._parse_paired_ireceptor, 'pathree'),
            "umunster": (self._parse_bulk_ireceptor, 'umunster'),
            "vdjserver": (self._parse_bulk_ireceptor, 'vdjserver'),
        }

        sequence_tables = []
        mri_tables = []

        def process_data(parse_function, db_name):
            """
            Helper function to process each database.
            """
            # Parse TCR and sequence tables
            tcr_table, sequence_table = parse_function(db_name)

            # Parse metadata using Dask
            metadata_table = self._parse_json_ireceptor(db_name)
            
            # Ensure `repertoire_id` is consistent
            metadata_table['repertoire_id'] = metadata_table['repertoire_id'].astype(str)
            tcr_table['repertoire_id'] = tcr_table['repertoire_id'].astype(str)

            # Merge metadata with TCR table lazily
            merged_table = dd.merge(metadata_table, tcr_table, on="repertoire_id", how="inner")

            print(f"{db_name} formatted")
            return sequence_table, merged_table

        # Process each database in parallel
        for db_key, (parse_function, db_name) in databases.items():
            db_path = self.config['databases']['ireceptor'][db_key]['database']
            metadata_path = self.config['databases']['ireceptor'][db_key]['metadata']

            seq_table, mri_table = process_data(parse_function, db_name)

            sequence_tables.append(seq_table)
            mri_tables.append(mri_table)

        # Define standardized columns
        standardized_columns = ['trav_gene', 'traj_gene', 'tra', 'trbv_gene', 'trbd_gene', 'trbj_gene', 'trb', 'sequence', 'source']

        # Ensure all sequence tables conform to the standardized format
        sequence_tables = [seq_table[standardized_columns] for seq_table in sequence_tables]

        # Define metadata schema for lazy concatenation
        meta = {col: 'str' for col in standardized_columns}

        # Concatenate all sequence tables lazily
        sequence_table = dd.concat(sequence_tables, meta=meta).reset_index(drop=True)
        
        # Concatenate all MRI tables lazily
        mri_table = dd.concat(mri_tables).reset_index(drop=True)

        return mri_table, sequence_table