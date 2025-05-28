import yaml
import ijson
import pandas as pd

from pathlib import Path
from collections import OrderedDict
from dask import delayed  # <-- NEW: We'll use Dask Delayed for final concatenation
import logging
import time
from functools import partial   # only if you need it elsewhere
import gc
import os
from typing import Optional, Tuple
from .utils import (
    parse_imgt_four_digit,
    transform_mhc_restriction,
    get_mhc_sequence
)
import dask.dataframe as dd
from dask.dataframe import merge as dd_merge

@delayed
def delayed_concat(dataframe_list):
    """
    A small helper function (decorated with @delayed) that uses
    pandas.concat under the hood. This returns a Delayed object;
    calling .compute() will yield the concatenated pandas DataFrame.
    """
    if not dataframe_list:
        return pd.DataFrame()
    # Filter out any empty (None or empty) dataframes to avoid concat errors
    real_dfs = [df for df in dataframe_list if df is not None and not df.empty]
    if not real_dfs:
        return pd.DataFrame()
    return pd.concat(real_dfs, axis=0, ignore_index=True)


class DatabaseParser:
    def __init__(self, config_path, test=False):
        self.config_path = config_path
        self.test = test
        self.config = self._load_config()
        self.hla_dictionary = parse_imgt_four_digit(self.config['databases']['imgt']['hla_fasta'])
        self.output_path = self.config['outputs']['output_path']
        self._mri_dir = Path(self.output_path) / "mri"
        self._seq_dir = Path(self.output_path) / "seq"

    def _load_config(self):
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def parse(
        self,
        out_prefix: Optional[Path] = None,
        compression: str = "snappy",
    ) -> Tuple[Path, Path]:
        """
        Parse each of the individual databases and return two *dask.delayed* objects:
            - mri_delayed : combined MRI table
            - seq_delayed : combined sequence table
        """
        log = logging.getLogger(__name__)
        log.info("⇢ Starting parse()")

        # name, function pairs in the order you want them run
        parse_fns = [
            ("vdjdb",   self._parse_vdjdb),
            ("mcpas",   self._parse_mcpas),
            ("tcrdb",   self._parse_tcrdb),
            ("iedb",    self._parse_iedb),
        ]

        mri_list, seq_list = [], []

        for label, fn in parse_fns:
            if os.path.exists(self._mri_dir / f"{label}_mri.parquet"):
                log.info("  • %s: skipped (already parsed)", label)
                continue
            log.info("  ↳ Parsing %s …", label)
            t0 = time.time()
            mri_df, seq_df = fn()  # each helper returns *combined* tables
            dt = time.time() - t0
            log.info(
                "    ✓ %s done in %.2fs (MRI rows=%s, Seq rows=%s)",
                label,
                dt,
                f"{len(mri_df):,}",
                f"{len(seq_df):,}",
            )

            # Write the aggregated result as well (handy for global analyses)
            if not mri_df.empty:
                mri_path = self._mri_dir / f"{label}_mri.parquet"
                mri_df.to_parquet(mri_path, engine="pyarrow", compression=compression, index=False)
            if not seq_df.empty:
                seq_path = self._seq_dir / f"{label}_seq.parquet"
                seq_df.to_parquet(seq_path, engine="pyarrow", compression=compression, index=False)

            # Free RAM ASAP
            del mri_df, seq_df
            gc.collect()

        # --- iReceptor ---
        log.info("  ↳ Parsing iReceptor …")
        self._parse_ireceptor()

        #log.info("⇢ Building delayed concatenations")
        #mri_delayed = delayed_concat(mri_list)
        #seq_delayed = delayed_concat(seq_list)

        #log.info("⇢ parse() finished—returning Delayed objects")
        #return mri_delayed, seq_delayed

    # ----------------------------------------------------------------------
    #                           VDJdb
    # ----------------------------------------------------------------------
    def _parse_vdjdb(self):
        """
        Parse the VDJdb dataset using pandas in memory.
        Returns:
            (mri_table, sequence_table)
        """
        database_path = self.config['databases']['vdjdb']

        try:
            df = pd.read_csv(database_path, sep="\t", dtype=str, na_filter=False)
        except FileNotFoundError:
            print(f"File not found: {database_path}")
            return pd.DataFrame(), pd.DataFrame()

        if self.test and not df.empty:
            df = df.sample(frac=0.1, random_state=21)

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
            'vdjdb.score': 'vdjdb_score'
        })

        # Filter for HomoSapiens and 'vdjdb.score' != '0'
        df = df[
            (df['species'] == 'HomoSapiens') &
            (df['vdjdb.score'] != '0')
        ]

        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        df = df[list(relevant_columns.keys())].rename(columns=relevant_columns)

        # MRI table
        mri_table = df.drop(columns=['vdjdb_score'], errors='ignore').copy()

        # Sequence table
        seq_cols = [
            'trav_gene', 'traj_gene', 'tra',
            'trbv_gene', 'trbd_gene', 'trbj_gene', 'trb',
            'peptide', 'mhc_restriction', 'mhc_restriction_two'
        ]
        sequence_table = mri_table[seq_cols].copy()
        # Transform MHC
        try:
            sequence_table['mhc_restriction'] = transform_mhc_restriction(
                sequence_table['mhc_restriction'], 
                fasta_dict=self.hla_dictionary)
            
        except AttributeError:
            breakpoint()
        sequence_table['mhc_restriction_two'] =  transform_mhc_restriction(
                sequence_table['mhc_restriction_two'], 
                fasta_dict=self.hla_dictionary)

        sequence_table['mhc_one'] = sequence_table['mhc_restriction'].apply(
            lambda x: get_mhc_sequence(x, fasta_dict=self.hla_dictionary)
        )
        sequence_table['mhc_two'] = sequence_table['mhc_restriction_two'].apply(
            lambda x: get_mhc_sequence(x, fasta_dict=self.hla_dictionary)
        )

        # Rename columns and build final sequence
        sequence_table.rename(columns={
            'mhc_restriction': 'mhc_one_id',
            'mhc_restriction_two': 'mhc_two_id'
        }, inplace=True)

        

        sequence_table['source'] = 'vdjdb'
        sequence_table.drop_duplicates(inplace=True)

        return mri_table, sequence_table

    # ----------------------------------------------------------------------
    #                           TCRdb
    # ----------------------------------------------------------------------
    def _parse_tcrdb(self):
        """
        Parse the TCRdb dataset (all *.tsv files) using pandas in memory.
        Returns:
            (mri_table, sequence_table)
        """
        database_path = Path(self.config['databases']['tcrdb'])
        tsv_files = list(database_path.rglob("*.tsv"))
        if not tsv_files:
            print(f"No TSV files found in {database_path}")
            return pd.DataFrame(), pd.DataFrame()

        all_mri = []
        for file_path in tsv_files:
            try:
                tcr = pd.read_csv(file_path, sep="\t", dtype=str, na_filter=False)
            except FileNotFoundError:
                continue
            if tcr.empty:
                continue

            if self.test:
                tcr = tcr.sample(frac=0.1, random_state=21)

            study_id = file_path.stem
            tcr.rename(columns={
                'RunId': 'repertoire_id',
                'Vregion': 'trbv_gene',
                'Dregion': 'trbd_gene',
                'Jregion': 'trbj_gene',
                'AASeq': 'trb'
            }, inplace=True)
            tcr.replace('Unknown', '', inplace=True)

            keep_cols = ['repertoire_id', 'trbv_gene', 'trbd_gene', 'trbj_gene', 'trb']
            tcr = tcr[keep_cols]
            tcr['study_id'] = study_id
            all_mri.append(tcr)

        if not all_mri:
            return pd.DataFrame(), pd.DataFrame()

        mri_table = pd.concat(all_mri, ignore_index=True)

        seq_cols = ['trbv_gene', 'trbd_gene', 'trbj_gene', 'trb']
        sequence_table = mri_table[seq_cols].copy()
        sequence_table['source'] = 'tcrdb'
        sequence_table.drop_duplicates(inplace=True)

        return mri_table, sequence_table

    # ----------------------------------------------------------------------
    #                           McPAS-TCR
    # ----------------------------------------------------------------------
    def _parse_mcpas(self):
        """
        Parse McPAS-TCR with pandas in memory.
        Returns:
            (mri_table, sequence_table)
        """
        database_path = self.config['databases']['mcpas_tcr']
        try:
            df = pd.read_csv(database_path, dtype=str, na_filter=False)
        except FileNotFoundError:
            print(f"File not found: {database_path}")
            return pd.DataFrame(), pd.DataFrame()

        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        if self.test:
            df = df.sample(frac=0.1, random_state=21)

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
        df.rename(columns=rename_columns, inplace=True)

        meta_cols = [
            'study_id', 'host_organism', 'trav_gene', 'traj_gene',
            'tra_junction_aa', 'trbv_gene', 'trbd_gene', 'trbj_gene',
            'trb_junction_aa', 'peptide', 'epitope_reference_name',
            'epitope_source_molecule', 'epitope_source_organism',
            'mhc_restriction', 'assay_method'
        ]
        for c in meta_cols:
            if c not in df.columns:
                df[c] = ""

        mri_table = df[meta_cols].copy()
        mri_table['data_source'] = 'McPAS-TCR'
        mri_table['study_id_type'] = 'PMID'

        seq_cols = [
            'trav_gene', 'traj_gene', 'tra_junction_aa',
            'trbv_gene', 'trbd_gene', 'trbj_gene',
            'trb_junction_aa', 'peptide', 'mhc_restriction'
        ]
        sequence_table = mri_table[seq_cols].copy()

        sequence_table['mhc_restriction'] = transform_mhc_restriction(
            sequence_table['mhc_restriction'], fasta_dict=self.hla_dictionary)
        sequence_table['mhc_one'] = sequence_table['mhc_restriction'].apply(
            lambda x: get_mhc_sequence(x, fasta_dict=self.hla_dictionary)
        )

        # Expand multiple peptides
        sequence_table['peptide'] = sequence_table['peptide'].apply(lambda x: x.split('/') if x else [])
        sequence_table = sequence_table.explode('peptide').reset_index(drop=True)

        sequence_table.rename(columns={
            'tra_junction_aa': 'tra',
            'trb_junction_aa': 'trb'
        }, inplace=True)

        final_cols = [
            'trav_gene', 'traj_gene', 'tra',
            'trbv_gene', 'trbd_gene', 'trbj_gene', 'trb',
            'peptide', 'mhc_restriction', 'mhc_one']
        for c in final_cols:
            if c not in sequence_table.columns:
                sequence_table[c] = ''

        sequence_table = sequence_table[final_cols].copy()
        sequence_table['source'] = 'McPAS-TCR'
        sequence_table.drop_duplicates(inplace=True)

        return mri_table, sequence_table

    # ----------------------------------------------------------------------
    #                           IEDB / CEDAR
    # ----------------------------------------------------------------------
    def _parse_iedb(self):
        """
        Parses data from IEDB and CEDAR, combining T-cell, MHC-ligand, and receptor data.
        Returns: (mri_table, sequence_table)
        """
        # --- CEDAR ---
        cedar_tcell_mri, cedar_tcell_seq = self._parse_iedb_tcr('cedar')
        cedar_mhc_mri, cedar_mhc_seq = self._parse_iedb_mhc('cedar')
        cedar_tcr_mri, cedar_tcr_seq = self._parse_iedb_receptor('cedar')

        cedar_mri = pd.concat([cedar_tcell_mri, cedar_tcr_mri, cedar_mhc_mri], ignore_index=True).drop_duplicates()
        cedar_seq = pd.concat([cedar_tcell_seq, cedar_tcr_seq, cedar_mhc_seq], ignore_index=True).drop_duplicates()
        cedar_seq['source'] = 'cedar'

        # --- IEDB ---
        iedb_tcell_mri, iedb_tcell_seq = self._parse_iedb_tcr('iedb')
        iedb_mhc_mri, iedb_mhc_seq = self._parse_iedb_mhc('iedb')
        iedb_tcr_mri, iedb_tcr_seq = self._parse_iedb_receptor('iedb')

        iedb_mri = pd.concat([iedb_tcell_mri, iedb_tcr_mri, iedb_mhc_mri], ignore_index=True).drop_duplicates()
        iedb_seq = pd.concat([iedb_tcell_seq, iedb_tcr_seq, iedb_mhc_seq], ignore_index=True).drop_duplicates()
        iedb_seq['source'] = 'iedb'

        # Combine
        combined_seq = pd.concat([cedar_seq, iedb_seq], ignore_index=True).drop_duplicates()
        combined_mri = pd.concat([cedar_mri, iedb_mri], ignore_index=True).drop_duplicates()

        return combined_mri, combined_seq

    def _parse_iedb_tcr(self, source):
        """
        Parses T-cell assay data from IEDB or CEDAR (similar structure).
        Returns: (mri_table, sequence_table)
        """
        db_cfg = self.config['databases'].get(source, {})
        if 'tcell_assay' not in db_cfg:
            return pd.DataFrame(), pd.DataFrame()

        file_path = db_cfg['tcell_assay']
        col_names = [
            'study_id', 'epitope_type', 'peptide', 'epitope_reference_name',
            'epitope_source_molecule', 'epitope_source_organism',
            'epitope_source_species', 'host_organism', 'host_population',
            'host_sex', 'host_age', 'host_mhc_profile', 'assay_method',
            'assay_response', 'assay_outcome', 'assay_subject_count',
            'assay_positive_count', 'source_tissue', 'mhc_restriction'
        ]
        try:
            df = pd.read_csv(file_path, sep="\t", names=col_names, header=0, dtype=str, na_filter=False)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return pd.DataFrame(), pd.DataFrame()
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        if self.test:
            df = df.sample(frac=0.1, random_state=21)

        df.dropna(subset=['study_id'], inplace=True)
        df = df[df['assay_outcome'].eq("Positive")]

        df['data_source'] = f"{source}_tcell_assay"
        df['study_id_type'] = "PMID"

        keep_mri = [
            'data_source', 'study_id', 'study_id_type', 'host_organism',
            'host_population', 'host_age', 'host_sex', 'host_mhc_profile',
            'source_tissue', 'epitope_type', 'peptide', 'epitope_reference_name',
            'epitope_source_molecule', 'epitope_source_organism',
            'mhc_restriction', 'assay_method', 'assay_response',
            'assay_outcome', 'assay_subject_count', 'assay_positive_count'
        ]
        mri_table = df[keep_mri].copy()

        seq_table = df[['peptide', 'mhc_restriction']].copy()

        # Split MHC on '/'
        mhc_1, mhc_2 = [], []
        for val in seq_table['mhc_restriction']:
            parts = val.split('/') if val else []
            mhc_1.append(parts[0] if len(parts) > 0 else '')
            mhc_2.append(parts[1] if len(parts) > 1 else '')
        seq_table['mhc_restriction'] = mhc_1
        seq_table['mhc_restriction_two'] = mhc_2

        seq_table['mhc_restriction'] = transform_mhc_restriction(
            seq_table['mhc_restriction'],  self.hla_dictionary)
        seq_table['mhc_restriction_two'] = transform_mhc_restriction(
            seq_table['mhc_restriction_two'], self.hla_dictionary)

        seq_table['mhc_one'] = seq_table['mhc_restriction'].apply(
            lambda x: get_mhc_sequence(x, self.hla_dictionary)
        )
        seq_table['mhc_two'] = seq_table['mhc_restriction_two'].apply(
            lambda x: get_mhc_sequence(x, self.hla_dictionary)
        )

        seq_table['peptide'] = seq_table['peptide'].apply(lambda x: x.split('+')[0] if x else '')

        
        seq_table.rename(columns={
            'mhc_restriction': 'mhc_one_id',
            'mhc_restriction_two': 'mhc_two_id'
        }, inplace=True)
        seq_cols = ['peptide', 'mhc_one_id', 'mhc_one', 'mhc_two_id', 'mhc_two']
        seq_table = seq_table[seq_cols].drop_duplicates()

        return mri_table, seq_table

    def _parse_iedb_mhc(self, source):
        """
        Parses MHC-ligand data from IEDB or CEDAR.
        Returns: (mri_table, sequence_table)
        """
        db_cfg = self.config['databases'].get(source, {})
        if 'mhc_ligand' not in db_cfg:
            return pd.DataFrame(), pd.DataFrame()

        file_path = db_cfg['mhc_ligand']
        col_map = {
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
                'assay_subject_count', 'assay_positive_count', 'source_tissue',
                'mhc_restriction'
            ]
        }
        columns = col_map.get(source.lower(), [])
        if not columns:
            return pd.DataFrame(), pd.DataFrame()

        try:
            df = pd.read_csv(
                file_path,
                sep="\t",
                names=columns,
                header=0,
                dtype=str,
                na_filter=False
            )
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return pd.DataFrame(), pd.DataFrame()

        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        if self.test:
            df = df.sample(frac=0.1, random_state=21)

        if 'assay_outcome' in df.columns:
            df = df[df['assay_outcome'] == "Positive"]

        df['data_source'] = f"{source}_mhc_ligand_assay"
        df['study_id_type'] = "PMID"

        keep_cols = [
            'data_source', 'study_id', 'study_id_type', 'host_organism',
            'host_population', 'host_age', 'host_sex', 'host_mhc_profile',
            'source_tissue', 'epitope_type', 'peptide', 'epitope_source_molecule',
            'epitope_source_organism', 'mhc_restriction', 'assay_method',
            'assay_response', 'assay_subject_count', 'assay_positive_count'
        ]
        for c in keep_cols:
            if c not in df.columns:
                df[c] = ""

        mri_table = df[keep_cols].copy()

        seq_table = df[['peptide', 'mhc_restriction']].copy()
        mhc_1, mhc_2 = [], []
        for val in seq_table['mhc_restriction']:
            parts = val.split('/') if val else []
            mhc_1.append(parts[0] if len(parts) > 0 else '')
            mhc_2.append(parts[1] if len(parts) > 1 else '')
        seq_table['mhc_restriction'] = mhc_1
        seq_table['mhc_restriction_two'] = mhc_2

        seq_table['mhc_restriction'] = transform_mhc_restriction(
            seq_table['mhc_restriction'], self.hla_dictionary)
        seq_table['mhc_restriction_two'] = transform_mhc_restriction(
            seq_table['mhc_restriction_two'], self.hla_dictionary)
        seq_table['mhc_one'] = seq_table['mhc_restriction'].apply(
            lambda x: get_mhc_sequence(x, self.hla_dictionary)
        )
        seq_table['mhc_two'] = seq_table['mhc_restriction_two'].apply(
            lambda x: get_mhc_sequence(x, self.hla_dictionary)
        )

        
        seq_table.rename(columns={
            'mhc_restriction': 'mhc_one_id',
            'mhc_restriction_two': 'mhc_two_id'
        }, inplace=True)
        final_cols = ['peptide', 'mhc_one_id', 'mhc_one', 'mhc_two_id', 'mhc_two']
        seq_table = seq_table[final_cols].drop_duplicates()

        return mri_table, seq_table

    def _parse_iedb_receptor(self, source):
        """
        Parses receptor assay data from IEDB or CEDAR.
        Returns: (mri_table, sequence_table)
        """
        db_cfg = self.config['databases'].get(source, {})
        if 'receptor' not in db_cfg:
            return pd.DataFrame(), pd.DataFrame()

        file_path = db_cfg['receptor']
        col_map = {
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
        columns = col_map.get(source.lower(), [])
        if not columns:
            return pd.DataFrame(), pd.DataFrame()

        try:
            df = pd.read_csv(
                file_path,
                sep="\t",
                names=columns,
                header=0,
                dtype=str,
                na_filter=False
            )
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return pd.DataFrame(), pd.DataFrame()
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        if self.test:
            df = df.sample(frac=0.1, random_state=21)

        if source.lower() == 'cedar':
            df['study_id'] = df['study_id'].str.extract(r'(\d{7})')[0]

        def fix_study_id(x):
            if pd.notna(x) and x:
                return f"{source.upper()}{x}"
            return x
        df['study_id'] = df['study_id'].apply(fix_study_id)

        df['data_source'] = f"{source}_receptor_table"
        df['study_id_type'] = source.upper()
        df['host_organism'] = "human"

        keep_mri = [
            'data_source', 'study_id', 'study_id_type', 'host_organism',
            'trav_gene', 'trad_gene', 'traj_gene', 'tra_junction_aa',
            'trbv_gene', 'trbd_gene', 'trbj_gene', 'trb_junction_aa',
            'peptide', 'epitope_source_molecule', 'epitope_source_organism'
        ]
        for c in keep_mri:
            if c not in df.columns:
                df[c] = ""

        mri_table = df[keep_mri].copy()

        seq_cols = [
            'trav_gene', 'trad_gene', 'traj_gene',
            'trbv_gene', 'trbd_gene', 'trbj_gene',
            'peptide', 'tra_junction_aa', 'trb_junction_aa'
        ]
        sequence_table = mri_table[seq_cols].copy()
        sequence_table.rename(columns={
            'tra_junction_aa': 'tra',
            'trb_junction_aa': 'trb'
        }, inplace=True)

        # Keep first peptide if multiple
        sequence_table['peptide'] = sequence_table['peptide'].apply(
            lambda x: x.split('+')[0] if x else ''
        )

        
        sequence_table.drop_duplicates(inplace=True)

        final_cols = [
            'trav_gene', 'trad_gene', 'traj_gene',
            'tra', 'trbv_gene', 'trbd_gene', 'trbj_gene',
            'trb', 'peptide']
        for c in final_cols:
            if c not in sequence_table.columns:
                sequence_table[c] = ''

        sequence_table = sequence_table[final_cols]
        return mri_table, sequence_table

    # ----------------------------------------------------------------------
    #                           iReceptor
    # ----------------------------------------------------------------------
    def _parse_ireceptor(self):
        """
        Same high-level logic as before, but the per-database frames are now
        dask.dataframe.DataFrame objects.  We only call .compute() when we
        actually *need* an in-memory result (e.g. checking for emptiness).
        """
        log = logging.getLogger(__name__)

        databases = {
            "airr_covid": (self._parse_paired_ireceptor, "airr_covid"),
            "paone":      (self._parse_bulk_ireceptor,   "paone"),
            "patwo":      (self._parse_bulk_ireceptor,   "patwo"),
            "pathree":    (self._parse_paired_ireceptor, "pathree"),
            "umunster":   (self._parse_bulk_ireceptor,   "umunster"),
            "vdjserver":  (self._parse_bulk_ireceptor,   "vdjserver"),
        }

        # canonical column order for sequence Parquet files
        seq_cols = [
            "trav_gene", "traj_gene", "tra",
            "trbv_gene", "trbd_gene", "trbj_gene",
            "trb", "source",
        ]

        for db_key, (parse_func, db_name) in databases.items():
            if os.path.exists(self._mri_dir / f"ireceptor_{db_name}_mri.parquet"):
                log.info("  • %s: skipped (already parsed)", db_key)
                continue
            log.info("    ↪ %s …", db_key)
            t0 = time.time()

            # ---- TSV → Dask --------------------------------------------------
            db_mri, db_seq = parse_func(db_name)          # dask frames now

            # ---- tiny JSON metadata is still small → pandas, then dask ------
            meta_pdf = self._parse_json_ireceptor(db_name)   # pandas
            if meta_pdf.empty:
                log.warning("      • %s: skipped (no metadata)", db_key)
                continue
            meta_dd = dd.from_pandas(meta_pdf.astype({"repertoire_id": "string[pyarrow]"}),
                                    npartitions=1)

            if db_mri.npartitions == 0:                     # nothing parsed
                log.warning("      • %s: skipped (empty)", db_key)
                continue

            # ensure ‘repertoire_id’ is string on both sides
            db_mri  = db_mri.astype({"repertoire_id": "string[pyarrow]"})
            merged  = dd_merge(meta_dd, db_mri,
                            on="repertoire_id", how="inner")

            # ---------- Parquet dump (single_file=True keeps the old naming) -
            mri_path = self._mri_dir / f"ireceptor_{db_name}_mri.parquet"
            seq_path = self._seq_dir / f"ireceptor_{db_name}_seq.parquet"
            breakpoint
            if merged.map_partitions(len).sum().compute() > 0:
                merged.to_parquet(mri_path,
                                engine="pyarrow",
                                compression="snappy",
                                write_index=False)

            # pad missing cols once, then write
            for c in seq_cols:
                if c not in db_seq.columns:
                    db_seq[c] = ""
            db_seq = db_seq[seq_cols]

            if db_seq.map_partitions(len).sum().compute() > 0:
                db_seq.to_parquet(seq_path,
                                engine="pyarrow",
                                compression="snappy",
                                write_index=False)

            log.info("      ✓ %s parsed & saved (%.2fs)", db_key, time.time() - t0)

            # GC not really needed with dask, but keep it symmetrical
            del db_mri, db_seq, merged, meta_dd, meta_pdf
            gc.collect()

    # ──────────────────────────────────────────────────────────────────────────
    # helper: TRA + TRB (paired) TSV                                              
    # ──────────────────────────────────────────────────────────────────────────
    def _parse_paired_ireceptor(self, source):
        """
        Reads a *tab-separated* paired-chain file with Dask.
        Returns: (mri_ddf, seq_ddf)
        """
        cfg = self.config["databases"].get("ireceptor", {})
        db_path = cfg.get(source, {}).get("database", "")
        if not db_path:
            return dd.from_pandas(pd.DataFrame(), 1), dd.from_pandas(pd.DataFrame(), 1)

        # 1 Lazy read --------------------------------------------------------
        df = dd.read_csv(db_path,
                        sep="\t",
                        dtype=str,
                        na_filter=False,
                        blocksize="64 MB",
                        assume_missing=True)

        if self.test:
            df = df.sample(frac=0.10, random_state=21)

        df = df[df["productive"] == "T"]

        tra = df[df.locus == "TRA"]
        trb = df[df.locus == "TRB"]

        tra = tra.drop("repertoire_id", axis=1, errors="ignore").rename(columns={
            "v_call": "trav_gene",
            "d_call": "trad_gene",
            "j_call": "traj_gene",
            "junction_aa": "tra",
            "data_processing_id": "repertoire_id",
        })

        trb = trb.drop("repertoire_id", axis=1, errors="ignore").rename(columns={
            "v_call": "trbv_gene",
            "d_call": "trbd_gene",
            "j_call": "trbj_gene",
            "junction_aa": "trb",
            "data_processing_id": "repertoire_id",
        })

        keep_tra = ["repertoire_id", "cell_id", "clone_id",
                    "trav_gene", "trad_gene", "traj_gene", "tra"]
        keep_trb = ["repertoire_id", "cell_id", "clone_id",
                    "trbv_gene", "trbd_gene", "trbj_gene", "trb"]

        tra = tra[keep_tra]
        trb = trb[keep_trb]

        mri   = dd_merge(tra, trb, how="outer",
                        on=["repertoire_id", "cell_id", "clone_id"])
        seq_cols = ["trav_gene", "traj_gene", "tra",
                    "trbv_gene", "trbd_gene", "trbj_gene", "trb"]

        seq = mri[seq_cols].copy()

        
        seq["source"] = f"ireceptor_{source}"
        seq = seq.drop_duplicates()

        # MRI table doesn’t need cell/clone ids downstream
        mri = mri.drop(["cell_id", "clone_id"], axis=1, errors="ignore")

        return mri, seq

    # ──────────────────────────────────────────────────────────────────────────
    # helper: bulk (single-chain) TSV                                            
    # ──────────────────────────────────────────────────────────────────────────
    def _parse_bulk_ireceptor(self, source):
        """
        Bulk TRA or TRB file → Dask.
        """
        cfg = self.config["databases"].get("ireceptor", {})
        db_path = cfg.get(source, {}).get("database", "")
        if not db_path:
            return dd.from_pandas(pd.DataFrame(), 1), dd.from_pandas(pd.DataFrame(), 1)

        df = dd.read_csv(db_path,
                        sep="\t",
                        dtype=str,
                        na_filter=False,
                        blocksize="64 MB",
                        assume_missing=True)

        if self.test:
            df = df.sample(frac=0.10, random_state=21)

        df  = df[df.productive == "T"]

        keep = ["repertoire_id", "cell_id", "clone_id",
                "locus", "v_call", "d_call", "j_call", "junction_aa"]
        for c in keep:
            if c not in df.columns:
                df[c] = ""

        df = df[keep]

        # build long → wide exactly as before (but vectorised)
        tra = df[df.locus == "TRA"].assign(
            trb="", trbv_gene="", trbd_gene="", trbj_gene=""
        ).rename(columns={
            "junction_aa": "tra",
            "v_call": "trav_gene",
            "d_call": "trad_gene",
            "j_call": "traj_gene",
        })

        trb = df[df.locus == "TRB"].assign(
            tra="", trav_gene="", trad_gene="", traj_gene=""
        ).rename(columns={
            "junction_aa": "trb",
            "v_call": "trbv_gene",
            "d_call": "trbd_gene",
            "j_call": "trbj_gene",
        })

        final = dd.concat([tra, trb], axis=0)

        def _build_seq(row):
            parts = []
            for v in (row["tra"], row["trb"]):
                if v:
                    parts.append(str(v))
            return " ".join(parts) + ";"

        
        final["source"] = f"ireceptor_{source}"

        mri_cols = ["repertoire_id", "cell_id", "clone_id"]
        mri  = final[mri_cols].copy()

        seq_cols = ["tra", "trb", "trav_gene", "trad_gene", "traj_gene",
                    "trbv_gene", "trbd_gene", "trbj_gene", "source"]
        seq  = final[seq_cols].drop_duplicates()

        return mri, seq

    def _parse_json_ireceptor(self, source):
        """
        Parses iReceptor JSON metadata using ijson, building a single
        pandas DataFrame in memory.
        """
        db_cfg = self.config['databases'].get('ireceptor', {})
        if source not in db_cfg:
            return pd.DataFrame()
        meta_info = db_cfg[source].get('metadata', '')
        if not meta_info:
            return pd.DataFrame()

        repertoires_data = []
        with open(meta_info, 'r') as f:
            parser = ijson.items(f, 'Repertoire.item')
            for item in parser:
                rep = self._process_repertoire_json(item, source)
                if rep:
                    repertoires_data.append(rep)

        if not repertoires_data:
            return pd.DataFrame()
        df = pd.DataFrame(repertoires_data).drop_duplicates()
        return df

    def _process_repertoire_json(self, repertoire, database):
        """
        Helper to parse a single repertoire JSON object from iReceptor metadata.
        """
        try:
            study_id = repertoire['study']['study_id']
            if database == "ireceptor":
                rep_id = repertoire['repertoire_id']
            else:
                data_proc = repertoire.get('data_processing', [])
                rep_id = data_proc[0]['data_processing_id'] if data_proc else ''

            subject = repertoire.get('subject', {})
            host_organism = subject.get('species', {}).get('label', '')
            diagnosis = subject.get('diagnosis', [{}])[0] if 'diagnosis' in subject else {}
            condition_studies = diagnosis.get('study_group_description', '')
            disease_label = diagnosis.get('disease_diagnosis', {}).get('label', '')
            if disease_label:
                condition_studies += f" {disease_label}"

            age = subject.get('age', '')
            sex = subject.get('sex', '')
            population_surveyed = subject.get('race', '')
            sample_info = repertoire.get('sample', [{}])
            source_tissue = sample_info[0].get('tissue', {}).get('label', '') if sample_info else ''

            mhc_list = set()
            genotype = subject.get('genotype', {})
            mhc_set = genotype.get('mhc_genotype_set', {})
            for mhc_obj in mhc_set.get('mhc_genotype_list', []):
                for allele in mhc_obj.get('mhc_alleles', []):
                    allele_designation = allele.get('allele_designation', '')
                    if allele_designation:
                        mhc_list.add(allele_designation)

            return {
                'study_id': study_id,
                'repertoire_id': str(rep_id),
                'host_organism': host_organism,
                'condition_studies': condition_studies,
                'age': age,
                'sex': sex,
                'population_surveyed': population_surveyed,
                'source_tissue': source_tissue,
                'mhc_profile': ','.join(sorted(mhc_list)) if mhc_list else ''
            }
        except Exception as e:
            print(f"Error processing repertoire: {e}")
            return None