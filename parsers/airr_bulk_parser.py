import os
import re
import csv
import yaml
import pandas as pd
from collections import OrderedDict

# If these come from another module, just import them.
# Otherwise, define them here or adapt as needed.
from .utils import standardize_sequence, standardize_mri


class BulkFileParser:
    def __init__(self, bulk_file, format_config, test=False):
        self.bulk_file = bulk_file
        self.format_config = format_config
        self.test = test

        self.repertoire_id = os.path.splitext(os.path.basename(self.bulk_file))[0]
        self.study_id = os.path.basename(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(self.bulk_file)
                )
            )
        )
        self.extension = os.path.splitext(self.bulk_file)[1]
        self.format_dict = self._load_format_config()

        self.separator = self._detect_delimiter()
        self.bulk_table = self._load_bulk_table()

        # These folder-based attributes may vary based on your directory structure
        self.file_type = os.path.basename(os.path.dirname(self.bulk_file))
        self.molecule_type = os.path.basename(os.path.dirname(os.path.dirname(self.bulk_file)))
        self.study_id = os.path.basename(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(self.bulk_file)
                )
            )
        )
        self.category = os.path.basename(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(self.bulk_file)
                    )
                )
            )
        )

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

        # Standardize the detected separator into the delimiter we will use
        if separator == "\t" or self.extension == ".tsv":
            delimiter = "\t"
        elif separator == "," or self.extension == ".csv":
            delimiter = ","
        elif separator == "M":
            # This was a special case in the original code for a MiTCR file
            delimiter = "M"

        return delimiter

    def _load_bulk_table(self):
        """
        Load the file into a pandas DataFrame, applying the detected delimiter.
        If self.test == True, we sample 10% of the data.
        """
        if self.separator == "\t":
            try:
                df = pd.read_csv(self.bulk_file, sep="\t", dtype=str, na_filter=False)
            except ValueError:
                return None
        elif self.separator == ",":
            df = pd.read_csv(self.bulk_file, dtype=str, na_filter=False)
        elif self.separator == "M":
            # Skip the first row for MiTCR
            df = pd.read_csv(self.bulk_file, sep="\t", dtype=str, na_filter=False, skiprows=1)

        if self.test and df is not None and not df.empty:
            df = df.sample(frac=0.1, random_state=21)

        # Convert all columns to string explicitly
        df = df.astype(str)

        return df

    def parse(self):
        """
        Choose the right parse function based on the known column sets.
        """
        if self.bulk_table is None or self.bulk_table.empty:
            return None, None

        known_formats = self.format_dict['bulk']

        if set(self.bulk_table.columns) == set(known_formats['format_one']):
            return self._parse_format_one()
        elif set(self.bulk_table.columns) == set(known_formats['format_two']):
            return self._parse_format_two()
        elif set(self.bulk_table.columns) == set(known_formats['format_three']):
            return self._parse_format_three()
        elif set(self.bulk_table.columns) == set(known_formats['format_four']):
            return self._parse_format_four()
        elif set(self.bulk_table.columns) == set(known_formats['format_five']):
            return self._parse_format_five()
        elif set(self.bulk_table.columns) == set(known_formats['format_six']):
            return self._parse_format_six()
        elif (
            set(self.bulk_table.columns) == set(known_formats['format_seven']) or
            set(self.bulk_table.columns) == set(known_formats['format_eight']) or
            set(self.bulk_table.columns) == set(known_formats['format_ten'])
        ):
            return self._parse_format_seven()
        elif set(self.bulk_table.columns) == set(known_formats['format_nine']):
            return self._parse_format_nine()
        elif set(self.bulk_table.columns) == set(known_formats['format_eleven']):
            return self._parse_format_eleven()
        else:
            raise ValueError("Unrecognized file format!")

    def _parse_format_one(self):
        """
        Example: 
        Remove columns ['VJCombo', 'Copy', 'ntCDR3', 'NetInsertionLength'], 
        detect chain type by the first row's VGene, rename accordingly, 
        return (mri_table, sequence_table).
        """

        # Drop unnecessary columns
        df = self.bulk_table.drop(
            columns=['VJCombo', 'Copy', 'ntCDR3', 'NetInsertionLength'],
            errors='ignore'
        ).copy()

        if df.empty:
            return None, None

        # Detect chain type from the first row
        first_vgene = df.iloc[0]['VGene']
        if 'TRAV' in first_vgene:
            # This is an alpha chain
            df = df.rename(
                columns={
                    'VGene': 'trav_gene',
                    'JGene': 'traj_gene',
                    'aaCDR3': 'tra'
                }
            )
            # Fill missing columns
            df['trad_gene'] = ''
            df['trb'] = ''
            df['trbv_gene'] = ''
            df['trbj_gene'] = ''
            df['trbd_gene'] = ''
            # Make a single sequence column
            df['sequence'] = df['tra'] + ';'
        elif 'TRBV' in first_vgene:
            # This is a beta chain
            df = df.rename(
                columns={
                    'VGene': 'trbv_gene',
                    'JGene': 'trbj_gene',
                    'aaCDR3': 'trb'
                }
            )
            df['tra'] = ''
            df['trav_gene'] = ''
            df['traj_gene'] = ''
            df['trad_gene'] = ''
            df['trbd_gene'] = ''
            df['sequence'] = df['trb'] + ';'
        else:
            raise ValueError(f"Unrecognized chain type in VGene: {first_vgene}")

        # Reorder columns consistently
        columns_order = [
            'trav_gene', 'traj_gene', 'trad_gene', 'tra',
            'trbv_gene', 'trbj_gene', 'trbd_gene', 'trb', 'sequence'
        ]
        df = df[columns_order]

        # Build the MRI table
        mri_table = df.copy()
        mri_table['repertoire_id'] = self.repertoire_id
        mri_table['study_id'] = self.study_id
        mri_table['category'] = self.category
        mri_table['molecule_type'] = self.molecule_type
        mri_table['host_organism'] = 'human'
        mri_table['source'] = 'bulk_survey'

        # Build the sequence table
        sequence_table = mri_table[[
            'trav_gene', 'trad_gene', 'traj_gene', 'tra',
            'trbv_gene', 'trbd_gene', 'trbj_gene', 'trb',
            'sequence', 'source'
        ]].copy()

        # Standardize
        sequence_table = standardize_sequence(sequence_table)
        mri_table = standardize_mri(mri_table)
        return mri_table, sequence_table

    def _parse_format_two(self):
        """
        Format two example.
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
        df = self.bulk_table.drop(columns=columns_to_drop, errors='ignore').copy()
        if df.empty:
            return None, None

        # Detect chain type from the first row
        first_v = df.iloc[0]['V alleles']

        if 'TRAV' in first_v:
            # alpha chain
            df = df.rename(columns={
                'V alleles': 'trav_gene',
                'J alleles': 'traj_gene',
                'D alleles': 'trad_gene',
                'CDR3 amino acid sequence': 'tra'
            })
            df['sequence'] = df['tra'] + ';'
            df['trb'] = ''
            df['trbv_gene'] = ''
            df['trbj_gene'] = ''
            df['trbd_gene'] = ''
        elif 'TRBV' in first_v:
            # beta chain
            df = df.rename(columns={
                'V alleles': 'trbv_gene',
                'J alleles': 'trbj_gene',
                'D alleles': 'trbd_gene',
                'CDR3 amino acid sequence': 'trb'
            })
            df['sequence'] = df['trb'] + ';'
            df['tra'] = ''
            df['trav_gene'] = ''
            df['traj_gene'] = ''
            df['trad_gene'] = ''
        else:
            raise ValueError(f"Unrecognized chain type in V alleles: {first_v}")

        # Reorder columns
        columns_order = [
            'trav_gene', 'traj_gene', 'trad_gene', 'tra',
            'trbv_gene', 'trbj_gene', 'trbd_gene', 'trb', 'sequence'
        ]
        df = df[columns_order]

        # Build MRI and sequence tables
        mri_table = df.copy()
        mri_table['repertoire_id'] = self.repertoire_id
        mri_table['study_id'] = self.study_id
        mri_table['host_organism'] = 'human'
        mri_table['category'] = self.category
        mri_table['molecule_type'] = self.molecule_type
        mri_table['source'] = 'bulk_survey'

        sequence_table = mri_table.drop_duplicates(subset=['sequence']).copy()

        # Standardize
        mri_table = standardize_mri(mri_table)
        sequence_table = standardize_sequence(sequence_table)
        return mri_table, sequence_table

    def _parse_format_three(self):
        """
        Format three example.
        """
        # Drop unnecessary columns
        columns_to_drop = [
            'Count', 'Percentage', 'CDR3 nucleotide sequence',
            'Last V nucleotide position', 'First D nucleotide position',
            'Last D nucleotide position', 'First J nucleotide position',
            'Good events', 'Total events', 'Good reads', 'Total reads'
        ]
        df = self.bulk_table.drop(columns=columns_to_drop, errors='ignore').copy()
        if df.empty:
            return None, None

        # Detect chain type from the first row's V segments
        first_v = df.iloc[0]['V segments']
        if 'TRAV' in first_v:
            df = df.rename(columns={
                'V segments': 'trav_gene',
                'J segments': 'traj_gene',
                'D segments': 'trad_gene',
                'CDR3 amino acid sequence': 'tra'
            })
            df['sequence'] = df['tra'] + ';'
            df['trb'] = ''
            df['trbv_gene'] = ''
            df['trbj_gene'] = ''
            df['trbd_gene'] = ''
        elif 'TRBV' in first_v:
            df = df.rename(columns={
                'V segments': 'trbv_gene',
                'J segments': 'trbj_gene',
                'D segments': 'trbd_gene',
                'CDR3 amino acid sequence': 'trb'
            })
            df['sequence'] = df['trb'] + ';'
            df['tra'] = ''
            df['trav_gene'] = ''
            df['traj_gene'] = ''
            df['trad_gene'] = ''
        else:
            raise ValueError(f"Unrecognized chain type in V segments: {first_v}")

        columns_order = [
            'trav_gene', 'traj_gene', 'trad_gene', 'tra',
            'trbv_gene', 'trbd_gene', 'trbj_gene', 'trb', 'sequence'
        ]
        df = df[columns_order]

        # Build MRI and sequence tables
        mri_table = df.copy()
        mri_table['repertoire_id'] = self.repertoire_id
        mri_table['study_id'] = self.study_id
        mri_table['category'] = self.category
        mri_table['molecule_type'] = self.molecule_type
        mri_table['host_organism'] = 'human'
        mri_table['source'] = 'bulk_survey'

        sequence_table = df.drop_duplicates(subset=['sequence']).copy()
        sequence_table['source'] = 'bulk_survey'

        # Standardize
        mri_table = standardize_mri(mri_table)
        sequence_table = standardize_sequence(sequence_table)
        return mri_table, sequence_table

    def _parse_format_four(self):
        """
        Format four example.
        """
        df = self.bulk_table[self.bulk_table['frame_type'] == "In"].copy()
        if df.empty:
            return None, None

        # We only keep the relevant columns
        df = df[['amino_acid', 'v_resolved', 'd_resolved', 'j_resolved']]

        # For each row, figure out if it’s TCRA or TCRB, then place data accordingly
        transformed_rows = []
        for _, row in df.iterrows():
            if 'TCRA' in row['v_resolved']:
                transformed_rows.append({
                    'trav_gene': row['v_resolved'],
                    'traj_gene': row['j_resolved'],
                    'trad_gene': row['d_resolved'],
                    'tra': row['amino_acid'],
                    'trbv_gene': '',
                    'trbj_gene': '',
                    'trbd_gene': '',
                    'trb': '',
                })
            elif 'TCRB' in row['v_resolved']:
                transformed_rows.append({
                    'trbv_gene': row['v_resolved'],
                    'trbj_gene': row['j_resolved'],
                    'trbd_gene': row['d_resolved'],
                    'trb': row['amino_acid'],
                    'trav_gene': '',
                    'traj_gene': '',
                    'trad_gene': '',
                    'tra': '',
                })
            # Possibly skip TCRD rows or add logic if needed

        out_df = pd.DataFrame(transformed_rows)
        if out_df.empty:
            return None, None

        # Create a 'sequence' column from tra/trb
        out_df['sequence'] = out_df[['tra', 'trb']].apply(
            lambda x: ' '.join(y for y in x if y) + ';',
            axis=1
        )

        # Build MRI
        mri_table = out_df.copy()
        mri_table['repertoire_id'] = self.repertoire_id
        mri_table['study_id'] = self.study_id
        mri_table['category'] = self.category
        mri_table['molecule_type'] = self.molecule_type
        mri_table['host_organism'] = 'human'
        mri_table['source'] = 'bulk_survey'

        # Build sequence table
        sequence_table = mri_table.drop_duplicates(subset=['sequence']).copy()

        # Standardize
        mri_table = standardize_mri(mri_table)
        sequence_table = standardize_sequence(sequence_table)
        return mri_table, sequence_table

    def _parse_format_five(self):
        """
        Format five example.
        """
        df = self.bulk_table[self.bulk_table['fuction'] == "in-frame"].copy()
        if df.empty:
            return None, None

        # Relevant columns
        df = df[['#ID', 'V_ref', 'D_ref', 'J_ref', 'CDR3(aa)', 'amino_acid']]

        transformed_rows = []
        for _, row in df.iterrows():
            if 'TRAV' in row['V_ref']:
                transformed_rows.append({
                    'trav_gene': row['V_ref'],
                    'traj_gene': row['J_ref'],
                    'trad_gene': row['D_ref'],
                    'tra': row['CDR3(aa)'],
                    'trbv_gene': '',
                    'trbj_gene': '',
                    'trbd_gene': '',
                    'trb': '',
                })
            elif 'TRBV' in row['V_ref']:
                transformed_rows.append({
                    'trbv_gene': row['V_ref'],
                    'trbj_gene': row['J_ref'],
                    'trbd_gene': row['D_ref'],
                    'trb': row['CDR3(aa)'],
                    'trav_gene': '',
                    'traj_gene': '',
                    'trad_gene': '',
                    'tra': '',
                })

        out_df = pd.DataFrame(transformed_rows)
        if out_df.empty:
            return None, None

        out_df['sequence'] = out_df[['tra', 'trb']].apply(
            lambda x: ' '.join(y for y in x if y) + ';',
            axis=1
        )

        # Build MRI
        mri_table = out_df.copy()
        mri_table['repertoire_id'] = self.repertoire_id
        mri_table['study_id'] = self.study_id
        mri_table['category'] = self.category
        mri_table['molecule_type'] = self.molecule_type
        mri_table['host_organism'] = 'human'
        mri_table['source'] = 'bulk_survey'

        # Build sequence table
        sequence_table = mri_table.drop_duplicates().copy()

        # Standardize
        mri_table = standardize_mri(mri_table)
        sequence_table = standardize_sequence(sequence_table)
        return mri_table, sequence_table

    def _parse_format_six(self):
        """
        Format six example.
        """
        df = self.bulk_table[self.bulk_table['fuction'] == "in-frame"].copy()
        if df.empty:
            return None, None

        df = df[['aminoAcid(CDR3 in lowercase)', 'vGene', 'dGene', 'jGene']]

        # Extract the longest lowercase stretch from 'aminoAcid(CDR3 in lowercase)'
        def longest_lowercase(s):
            matches = re.findall(r'[a-z]+', s)
            return max(matches, key=len, default='')

        df['cdr3'] = df['aminoAcid(CDR3 in lowercase)'].apply(longest_lowercase)

        transformed_rows = []
        for _, row in df.iterrows():
            if 'TRAV' in row['vGene']:
                transformed_rows.append({
                    'trav_gene': row['vGene'],
                    'traj_gene': row['jGene'],
                    'trad_gene': row['dGene'],
                    'tra': row['cdr3'],
                    'trbv_gene': '',
                    'trbj_gene': '',
                    'trbd_gene': '',
                    'trb': '',
                })
            elif 'TRBV' in row['vGene']:
                transformed_rows.append({
                    'trbv_gene': row['vGene'],
                    'trbj_gene': row['jGene'],
                    'trbd_gene': row['dGene'],
                    'trb': row['cdr3'],
                    'trav_gene': '',
                    'traj_gene': '',
                    'trad_gene': '',
                    'tra': '',
                })

        out_df = pd.DataFrame(transformed_rows)
        if out_df.empty:
            return None, None

        out_df['sequence'] = out_df[['tra', 'trb']].apply(
            lambda x: ' '.join(y for y in x if y) + ';',
            axis=1
        )

        # Build MRI
        mri_table = out_df.copy()
        mri_table['repertoire_id'] = self.repertoire_id
        mri_table['study_id'] = self.study_id
        mri_table['category'] = self.category
        mri_table['molecule_type'] = self.molecule_type
        mri_table['host_organism'] = 'human'
        mri_table['source'] = 'bulk_survey'

        # Build sequence table
        sequence_table = mri_table.drop_duplicates().copy()

        # Standardize
        mri_table = standardize_mri(mri_table)
        sequence_table = standardize_sequence(sequence_table)
        return mri_table, sequence_table

    def _parse_format_seven(self):
        """
        Format seven (also used by format eight/ten in original code).
        """
        df = self.bulk_table[self.bulk_table['sequenceStatus'] == "In"].copy()
        if df.empty:
            return None, None

        df = df[['aminoAcid', 'vMaxResolved', 'dMaxResolved', 'jMaxResolved']]
        df = df.fillna('')

        transformed_rows = []
        for _, row in df.iterrows():
            v, d, j = row['vMaxResolved'], row['dMaxResolved'], row['jMaxResolved']
            if 'TCRAV' in v or 'TCRAJ' in j:
                transformed_rows.append({
                    'trav_gene': v,
                    'traj_gene': j,
                    'trad_gene': d,
                    'tra': row['aminoAcid'],
                    'trbv_gene': '',
                    'trbj_gene': '',
                    'trbd_gene': '',
                    'trb': '',
                })
            elif 'TCRBV' in v or 'TCRBJ' in j or 'TCRBD' in d:
                transformed_rows.append({
                    'trbv_gene': v,
                    'trbj_gene': j,
                    'trbd_gene': d,
                    'trb': row['aminoAcid'],
                    'trav_gene': '',
                    'traj_gene': '',
                    'trad_gene': '',
                    'tra': '',
                })

        out_df = pd.DataFrame(transformed_rows)
        if out_df.empty:
            return None, None

        out_df['sequence'] = out_df[['tra', 'trb']].apply(
            lambda x: ' '.join(y for y in x if y) + ';',
            axis=1
        )

        # Build MRI
        mri_table = out_df.copy()
        mri_table['repertoire_id'] = self.repertoire_id
        mri_table['study_id'] = self.study_id
        mri_table['category'] = self.category
        mri_table['molecule_type'] = self.molecule_type
        mri_table['host_organism'] = 'human'
        mri_table['source'] = 'bulk_survey'

        # Build sequence table
        sequence_table = mri_table.drop_duplicates().copy()

        # Standardize
        mri_table = standardize_mri(mri_table)
        sequence_table = standardize_sequence(sequence_table)
        return mri_table, sequence_table

    def _parse_format_nine(self):
        """
        Format nine example.
        """
        df = self.bulk_table[self.bulk_table['frame_type'] == "In"].copy()
        if df.empty:
            return None, None

        df = df[['amino_acid', 'v_gene', 'd_gene', 'j_gene']]
        df = df.fillna('')

        transformed_rows = []
        for _, row in df.iterrows():
            if 'TCRAV' in row['v_gene']:
                transformed_rows.append({
                    'trav_gene': row['v_gene'],
                    'traj_gene': row['j_gene'],
                    'trad_gene': row['d_gene'],
                    'tra': row['amino_acid'],
                    'trbv_gene': '',
                    'trbj_gene': '',
                    'trbd_gene': '',
                    'trb': '',
                })
            elif 'TCRBV' in row['v_gene']:
                transformed_rows.append({
                    'trbv_gene': row['v_gene'],
                    'trbj_gene': row['j_gene'],
                    'trbd_gene': row['d_gene'],
                    'trb': row['amino_acid'],
                    'trav_gene': '',
                    'traj_gene': '',
                    'trad_gene': '',
                    'tra': '',
                })

        out_df = pd.DataFrame(transformed_rows)
        if out_df.empty:
            return None, None

        out_df['sequence'] = out_df[['tra', 'trb']].apply(
            lambda x: ' '.join(y for y in x if y) + ';',
            axis=1
        )

        mri_table = out_df.copy()
        mri_table['repertoire_id'] = self.repertoire_id
        mri_table['study_id'] = self.study_id
        mri_table['host_organism'] = 'human'
        mri_table['category'] = self.category
        mri_table['molecule_type'] = self.molecule_type
        mri_table['source'] = 'bulk_survey'

        sequence_table = mri_table.drop_duplicates().copy()

        # Standardize
        mri_table = standardize_mri(mri_table)
        sequence_table = standardize_sequence(sequence_table)
        return mri_table, sequence_table

    def _parse_format_eleven(self):
        """
        Format eleven example.
        """
        df = self.bulk_table[['cdr3_b_aa', 'v_b_gene', 'j_b_gene']].copy()
        if df.empty:
            return None, None

        transformed_rows = []
        for _, row in df.iterrows():
            v_gene = row['v_b_gene']
            if 'TRAV' in v_gene:
                transformed_rows.append({
                    'trav_gene': row['v_b_gene'],
                    'traj_gene': row['j_b_gene'],
                    'trad_gene': '',
                    'tra': row['cdr3_b_aa'],
                    'trbv_gene': '',
                    'trbj_gene': '',
                    'trbd_gene': '',
                    'trb': '',
                    'sequence': f"{row['cdr3_b_aa']};",
                })
            elif 'TRBV' in v_gene:
                transformed_rows.append({
                    'trav_gene': '',
                    'traj_gene': '',
                    'trad_gene': '',
                    'tra': '',
                    'trbv_gene': row['v_b_gene'],
                    'trbj_gene': row['j_b_gene'],
                    'trbd_gene': '',
                    'trb': row['cdr3_b_aa'],
                    'sequence': f"{row['cdr3_b_aa']};",
                })

        out_df = pd.DataFrame(transformed_rows)
        if out_df.empty:
            return None, None

        # Build MRI
        mri_table = out_df.copy()
        mri_table['repertoire_id'] = self.repertoire_id
        mri_table['study_id'] = self.study_id
        mri_table['host_organism'] = 'human'
        mri_table['category'] = self.category
        mri_table['molecule_type'] = self.molecule_type
        mri_table['source'] = 'bulk_survey'

        sequence_table = mri_table.drop_duplicates(subset='sequence').copy()

        # Standardize
        mri_table = standardize_mri(mri_table)
        sequence_table = standardize_sequence(sequence_table)
        return mri_table, sequence_table
