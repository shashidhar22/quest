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
from tqdm import tqdm
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
    ) -> dict:
        """
        Parse each of the individual databases and write to parquet files.
        
        Returns:
            dict: Statistics including total_mri_rows, total_seq_rows, databases_parsed
        """
        log = logging.getLogger(__name__)
        
        # Verify config has required keys
        if 'databases' not in self.config:
            log.error("  ✗ Config missing 'databases' key")
            return {'total_mri_rows': 0, 'total_seq_rows': 0, 'databases_parsed': 0}

        # name, function pairs in the order you want them run
        parse_fns = [
            ("imgt",    self._parse_imgt),
            ("vdjdb",   self._parse_vdjdb),
            ("mcpas",   self._parse_mcpas),
            ("tcrdb",   self._parse_tcrdb),
            ("iedb",    self._parse_iedb),
        ]

        mri_list, seq_list = [], []
        total_mri_rows = 0
        total_seq_rows = 0
        databases_parsed = 0

        # Create progress bar for databases
        pbar = tqdm(parse_fns, desc="Parsing databases", unit="db")
        
        for label, fn in pbar:
            pbar.set_description(f"Parsing {label}")
            
            if os.path.exists(self._mri_dir / f"{label}_mri.parquet"):
                log.info("  • %s: skipped (already parsed)", label)
                continue
            
            t0 = time.time()
            
            try:
                result = fn()  # each helper returns *combined* tables
                
                # Handle case where function returns None
                if result is None:
                    log.warning(f"    ✗ {label} returned None - likely missing file")
                    continue
                
                mri_df, seq_df = result
                
                # Handle empty dataframes
                if mri_df is None or seq_df is None:
                    log.warning(f"    ✗ {label} returned None dataframes - likely missing file")
                    continue
                    
            except Exception as e:
                log.error(f"    ✗ {label} failed with error: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            dt = time.time() - t0
            
            # Count rows before deletion
            mri_rows = len(mri_df) if not mri_df.empty else 0
            seq_rows = len(seq_df) if not seq_df.empty else 0
            total_mri_rows += mri_rows
            total_seq_rows += seq_rows
            
            if mri_rows > 0 or seq_rows > 0:
                databases_parsed += 1
            
            log.info(
                "    ✓ %s done in %.2fs (MRI rows=%s, Seq rows=%s)",
                label,
                dt,
                f"{mri_rows:,}",
                f"{seq_rows:,}",
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

        log.info("⇢ parse() finished")
        return {
            'total_mri_rows': total_mri_rows,
            'total_seq_rows': total_seq_rows,
            'databases_parsed': len(parse_fns) + 1  # +1 for iReceptor
        }

    # ----------------------------------------------------------------------
    #                           IMGT
    # ----------------------------------------------------------------------
    def _parse_imgt(self):
        """
        Parse IMGT/HLA FASTA files and convert to parsed output format.
        Each FASTA file contains protein sequences for HLA alleles.
        
        Returns:
            (mri_table, sequence_table)
        """
        log = logging.getLogger(__name__)
        hla_directory = self.config['databases']['imgt']['hla_fasta']
        
        try:
            import glob
            file_paths = glob.glob(f"{hla_directory}/*_prot.fasta")
            
            if not file_paths:
                log.error(f"    ✗ No IMGT FASTA files found in: {hla_directory}")
                return pd.DataFrame(), pd.DataFrame()
            
            all_records = []
            
            # Parse each FASTA file
            with tqdm(file_paths, desc="      • Processing IMGT FASTA files", leave=False) as pbar:
                for file_path in pbar:
                    gene_name = Path(file_path).stem.replace('_prot', '')  # e.g., "A", "B", "C"
                    pbar.set_postfix(gene=gene_name)
                    
                    with open(file_path, 'r') as f:
                        header, sequence = None, []
                        allele_id = None
                        bp_length = None
                        
                        for line in f:
                            line = line.strip()
                            if line.startswith('>'):
                                # Save previous record
                                if header and sequence and allele_id:
                                    all_records.append({
                                        'hla_id': header,
                                        'allele_id': allele_id,
                                        'gene': gene_name,
                                        'sequence': ''.join(sequence),
                                        'bp_length': bp_length
                                    })
                                
                                # Parse header: >HLA:HLA00001 A*01:01:01:01 365 bp
                                parts = line[1:].split()
                                if len(parts) >= 3:
                                    hla_code = parts[0]  # "HLA:HLA00001"
                                    allele_id = parts[1]  # "A*01:01:01:01"
                                    bp_length = parts[2]  # "365"
                                    
                                    # Extract four-digit resolution
                                    allele_parts = allele_id.split(':')
                                    if len(allele_parts) >= 2:
                                        header = ':'.join(allele_parts[:2])  # "A*01:01"
                                    else:
                                        header = allele_id
                                else:
                                    header = None
                                    allele_id = None
                                    
                                sequence = []
                            else:
                                sequence.append(line)
                        
                        # Save last record
                        if header and sequence and allele_id:
                            all_records.append({
                                'hla_id': header,
                                'allele_id': allele_id,
                                'gene': gene_name,
                                'sequence': ''.join(sequence),
                                'bp_length': bp_length
                            })
            
            if not all_records:
                log.warning("    ⚠ No records extracted from IMGT FASTA files")
                return pd.DataFrame(), pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(all_records)
            
            if self.test and not df.empty:
                df = df.sample(frac=0.1, random_state=21)
            
            # Create MRI table (metadata about the alleles)
            mri_table = pd.DataFrame({
                'source': 'imgt',
                'hla_id': df['hla_id'],
                'gene': df['gene'],
                'allele_id': df['allele_id'],
                'bp_length': df['bp_length'],
                'mhc_class': df['gene'].apply(lambda g: 'I' if g in ['A', 'B', 'C'] else 'II')
            })
            
            # Create sequence table (the actual protein sequences)
            seq_table = pd.DataFrame({
                'source': 'imgt',
                'hla_id': df['hla_id'],
                'mhc_sequence': df['sequence']
            })
            
            return mri_table, seq_table
            
        except Exception as e:
            log.error(f"    ✗ IMGT parsing failed: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(), pd.DataFrame()

    # ----------------------------------------------------------------------
    #                           VDJdb
    # ----------------------------------------------------------------------
    def _parse_vdjdb(self):
        """
        Parse the VDJdb dataset using pandas in memory.
        VDJdb format: Long format with one row per chain (TRA or TRB).
        Returns:
            (mri_table, sequence_table)
        """
        log = logging.getLogger(__name__)
        database_path = self.config['databases']['vdjdb']

        try:
            df = pd.read_csv(database_path, sep="\t", dtype=str, na_filter=False)
        except FileNotFoundError:
            log.error(f"    ✗ VDJdb file not found: {database_path}")
            return pd.DataFrame(), pd.DataFrame()
        except Exception as e:
            log.error(f"    ✗ VDJdb error reading file: {e}")
            return pd.DataFrame(), pd.DataFrame()

        if self.test and not df.empty:
            df = df.sample(frac=0.1, random_state=21)

        # Filter for HomoSapiens and vdjdb.score != '0'
        if 'species' in df.columns and 'vdjdb.score' in df.columns:
            df = df[
                (df['species'] == 'HomoSapiens') &
                (df['vdjdb.score'] != '0')
            ]

        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Parse metadata columns
        meta_cols = {
            'species': 'host_organism',
            'mhc.a': 'mhc_restriction',
            'mhc.b': 'mhc_restriction_two',
            'mhc.class': 'mhc_class',
            'antigen.epitope': 'peptide',
            'antigen.gene': 'epitope_source_molecule',
            'antigen.species': 'epitope_source_organism',
            'reference.id': 'study_id'
        }
        
        # Add optional metadata columns
        optional_meta = {
            'method': 'assay_method',
            'vdjdb.score': 'vdjdb_score'
        }
        
        # Rename columns that exist
        rename_map = {}
        for old_col, new_col in {**meta_cols, **optional_meta}.items():
            if old_col in df.columns:
                rename_map[old_col] = new_col
        
        df.rename(columns=rename_map, inplace=True)

        # Separate TRA and TRB rows
        tra_df = df[df['gene'] == 'TRA'].copy()
        trb_df = df[df['gene'] == 'TRB'].copy()

        # Rename chain-specific columns
        tra_df.rename(columns={
            'cdr3': 'tra',
            'v.segm': 'trav_gene',
            'j.segm': 'traj_gene'
        }, inplace=True)

        trb_df.rename(columns={
            'cdr3': 'trb',
            'v.segm': 'trbv_gene',
            'j.segm': 'trbj_gene'
        }, inplace=True)

        # Add empty columns for missing chain data
        for col in ['trb', 'trbv_gene', 'trbj_gene', 'trbd_gene']:
            if col not in tra_df.columns:
                tra_df[col] = ""
        
        for col in ['tra', 'trav_gene', 'traj_gene', 'trad_gene']:
            if col not in trb_df.columns:
                trb_df[col] = ""

        # Merge on complex.id to reconstruct paired chains
        common_cols = ['complex.id'] + list(rename_map.values())
        common_cols = [c for c in common_cols if c in tra_df.columns and c in trb_df.columns]
        
        # Select relevant columns for merge
        tra_merge_cols = common_cols + ['tra', 'trav_gene', 'traj_gene']
        trb_merge_cols = common_cols + ['trb', 'trbv_gene', 'trbj_gene']
        
        tra_merge_cols = [c for c in tra_merge_cols if c in tra_df.columns]
        trb_merge_cols = [c for c in trb_merge_cols if c in trb_df.columns]

        # Merge paired chains
        if 'complex.id' in df.columns:
            # Separate paired (complex.id != '0') from unpaired (complex.id == '0')
            # complex.id='0' is used in VDJdb for unpaired/single-chain sequences
            # Handle both string '0' and numeric 0, as well as empty/NA values
            tra_paired = tra_df[
                (tra_df['complex.id'] != '0') & 
                (tra_df['complex.id'] != 0) & 
                (tra_df['complex.id'] != '') &
                (tra_df['complex.id'].notna())
            ].copy()
            tra_unpaired = tra_df[
                (tra_df['complex.id'] == '0') | 
                (tra_df['complex.id'] == 0) | 
                (tra_df['complex.id'] == '') |
                (tra_df['complex.id'].isna())
            ].copy()
            
            trb_paired = trb_df[
                (trb_df['complex.id'] != '0') & 
                (trb_df['complex.id'] != 0) & 
                (trb_df['complex.id'] != '') &
                (trb_df['complex.id'].notna())
            ].copy()
            trb_unpaired = trb_df[
                (trb_df['complex.id'] == '0') | 
                (trb_df['complex.id'] == 0) | 
                (trb_df['complex.id'] == '') |
                (trb_df['complex.id'].isna())
            ].copy()
            
            log.info(f"      • Paired: {len(tra_paired):,} TRA + {len(trb_paired):,} TRB")
            log.info(f"      • Unpaired: {len(tra_unpaired):,} TRA + {len(trb_unpaired):,} TRB")
            
            # Merge only the paired sequences
            if len(tra_paired) > 0 and len(trb_paired) > 0:
                merge_on_cols = ['complex.id']
                # Add metadata columns that should be the same for both chains
                metadata_merge_cols = ['peptide', 'mhc_restriction', 'mhc_restriction_two', 
                                       'study_id', 'host_organism', 'epitope_source_molecule',
                                       'epitope_source_organism']
                
                for col in metadata_merge_cols:
                    if col in tra_paired.columns and col in trb_paired.columns:
                        merge_on_cols.append(col)
                
                df_paired = pd.merge(
                    tra_paired[tra_merge_cols],
                    trb_paired[trb_merge_cols],
                    on=merge_on_cols,
                    how='outer',
                    suffixes=('', '_y')
                )
                
                log.info(f"      • After merge: {len(df_paired):,} paired rows")
                
                # Check for explosion - should be roughly len(tra_paired) + len(trb_paired)
                expected_max = len(tra_paired) + len(trb_paired)
                if len(df_paired) > expected_max * 2:
                    log.warning(f"      ⚠ Merge explosion: {len(df_paired):,} rows from {len(tra_paired):,} TRA + {len(trb_paired):,} TRB")
                    log.warning(f"      ⚠ Deduplicating...")
                    df_paired = df_paired.drop_duplicates()
                    log.info(f"      • After dedup: {len(df_paired):,} rows")
                
                # Merge metadata columns that got duplicated
                for col in rename_map.values():
                    if f'{col}_y' in df_paired.columns:
                        df_paired[col] = df_paired[col].fillna(df_paired[f'{col}_y'])
                        df_paired.drop(columns=[f'{col}_y'], inplace=True)
            else:
                df_paired = pd.DataFrame()
            
            # Concatenate paired + unpaired sequences
            df = pd.concat([df_paired, tra_unpaired[tra_merge_cols], trb_unpaired[trb_merge_cols]], 
                          axis=0, ignore_index=True)
            
            log.info(f"      • Final VDJdb table: {len(df):,} rows (paired + unpaired)")
        else:
            # If no complex.id, concatenate all rows
            df = pd.concat([tra_df, trb_df], axis=0, ignore_index=True)

        # Ensure all required columns exist
        required_cols = [
            'tra', 'trav_gene', 'traj_gene', 
            'trb', 'trbv_gene', 'trbj_gene', 'trbd_gene',
            'peptide', 'mhc_restriction', 'mhc_restriction_two',
            'host_organism', 'study_id', 'mhc_class',
            'epitope_source_molecule', 'epitope_source_organism'
        ]
        
        for col in required_cols:
            if col not in df.columns:
                df[col] = ""

        # MRI table
        mri_cols = [c for c in df.columns if c not in ['vdjdb_score', 'gene', 'complex.id']]
        mri_table = df[mri_cols].copy()
        mri_table['data_source'] = 'vdjdb'

        # Sequence table
        seq_cols = [
            'trav_gene', 'traj_gene', 'tra',
            'trbv_gene', 'trbd_gene', 'trbj_gene', 'trb',
            'peptide', 'mhc_restriction', 'mhc_restriction_two'
        ]
        sequence_table = df[seq_cols].copy()
        
        # Transform MHC (vectorized)
        try:
            sequence_table['mhc_restriction'] = transform_mhc_restriction(
                sequence_table['mhc_restriction'], 
                fasta_dict=self.hla_dictionary)
        except AttributeError:
            breakpoint()
        
        sequence_table['mhc_restriction_two'] = transform_mhc_restriction(
            sequence_table['mhc_restriction_two'], 
            fasta_dict=self.hla_dictionary)

        # Vectorized lookup - much faster than apply!
        sequence_table['mhc_one'] = sequence_table['mhc_restriction'].map(self.hla_dictionary)
        sequence_table['mhc_two'] = sequence_table['mhc_restriction_two'].map(self.hla_dictionary)

        # Rename columns and build final sequence
        sequence_table.rename(columns={
            'mhc_restriction': 'mhc_one_id',
            'mhc_restriction_two': 'mhc_two_id'
        }, inplace=True)

        # Filter out sequences shorter than 4 characters
        if 'tra' in sequence_table.columns:
            sequence_table.loc[sequence_table['tra'].str.len() < 4, 'tra'] = ''
        if 'trb' in sequence_table.columns:
            sequence_table.loc[sequence_table['trb'].str.len() < 4, 'trb'] = ''

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
        log = logging.getLogger(__name__)
        database_path = Path(self.config['databases']['tcrdb'])
        tsv_files = list(database_path.rglob("*.tsv"))
        if not tsv_files:
            log.error(f"    ✗ No TSV files found in {database_path}")
            return pd.DataFrame(), pd.DataFrame()

        all_mri = []
        total_records = 0
        
        for file_path in tqdm(tsv_files, desc="      TCRdb files", unit="file", leave=False):
            try:
                tcr = pd.read_csv(file_path, sep="\t", dtype=str, na_filter=False)
                total_records += len(tcr)
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

        log.info(f"      • Loaded {total_records:,} total records from all files")
        mri_table = pd.concat(all_mri, ignore_index=True)
        log.info(f"      • Combined into {len(mri_table):,} records")

        seq_cols = ['trbv_gene', 'trbd_gene', 'trbj_gene', 'trb']
        sequence_table = mri_table[seq_cols].copy()
        
        # Filter out sequences shorter than 4 characters
        if 'trb' in sequence_table.columns:
            sequence_table.loc[sequence_table['trb'].str.len() < 4, 'trb'] = ''
        
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
        log = logging.getLogger(__name__)
        database_path = self.config['databases']['mcpas_tcr']
        try:
            df = pd.read_csv(database_path, dtype=str, na_filter=False)
            log.info(f"      • Loaded {len(df):,} raw records")
        except FileNotFoundError:
            log.error(f"    ✗ McPAS file not found: {database_path}")
            return pd.DataFrame(), pd.DataFrame()
        except Exception as e:
            log.error(f"    ✗ McPAS error reading file: {e}")
            return pd.DataFrame(), pd.DataFrame()

        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        if self.test:
            df = df.sample(frac=0.1, random_state=21)
            log.info(f"      • Test mode: sampled to {len(df):,} records")

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
        
        # Vectorized lookup - much faster!
        sequence_table['mhc_one'] = sequence_table['mhc_restriction'].map(self.hla_dictionary)

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
        
        # Filter out sequences shorter than 4 characters
        if 'tra' in sequence_table.columns:
            sequence_table.loc[sequence_table['tra'].str.len() < 4, 'tra'] = ''
        if 'trb' in sequence_table.columns:
            sequence_table.loc[sequence_table['trb'].str.len() < 4, 'trb'] = ''
        
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
            log = logging.getLogger(__name__)
            log.info(f"      • {source} tcell: Loaded {len(df):,} raw records")
        except FileNotFoundError:
            log = logging.getLogger(__name__)
            log.error(f"    ✗ {source} tcell_assay file not found: {file_path}")
            return pd.DataFrame(), pd.DataFrame()
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        if self.test:
            df = df.sample(frac=0.1, random_state=21)
            log.info(f"      • {source} tcell: Test mode - sampled to {len(df):,} records")

        initial_count = len(df)
        df.dropna(subset=['study_id'], inplace=True)
        df = df[df['assay_outcome'].eq("Positive")]
        log.info(f"      • {source} tcell: Filtered to {len(df):,} positive records (removed {initial_count - len(df):,})")

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

        # Transform MHC alleles with progress bar
        with tqdm(total=2, desc=f"      • {source} tcell MHC transform", leave=False) as pbar:
            seq_table['mhc_restriction'] = transform_mhc_restriction(
                seq_table['mhc_restriction'],  self.hla_dictionary)
            pbar.update(1)
            seq_table['mhc_restriction_two'] = transform_mhc_restriction(
                seq_table['mhc_restriction_two'], self.hla_dictionary)
            pbar.update(1)

        # Vectorized lookup - much faster!
        with tqdm(total=2, desc=f"      • {source} tcell MHC lookup", leave=False) as pbar:
            seq_table['mhc_one'] = seq_table['mhc_restriction'].map(self.hla_dictionary)
            pbar.update(1)
            seq_table['mhc_two'] = seq_table['mhc_restriction_two'].map(self.hla_dictionary)
            pbar.update(1)

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
            log = logging.getLogger(__name__)
            log.info(f"      • {source} mhc: Loaded {len(df):,} raw records")
        except FileNotFoundError:
            log = logging.getLogger(__name__)
            log.error(f"    ✗ {source} mhc_ligand file not found: {file_path}")
            return pd.DataFrame(), pd.DataFrame()

        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        if self.test:
            df = df.sample(frac=0.1, random_state=21)
            log.info(f"      • {source} mhc: Test mode - sampled to {len(df):,} records")

        if 'assay_outcome' in df.columns:
            initial_count = len(df)
            df = df[df['assay_outcome'] == "Positive"]
            log.info(f"      • {source} mhc: Filtered to {len(df):,} positive records (removed {initial_count - len(df):,})")

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

        # Transform MHC alleles with progress bar
        with tqdm(total=2, desc=f"      • {source} mhc_ligand MHC transform", leave=False) as pbar:
            seq_table['mhc_restriction'] = transform_mhc_restriction(
                seq_table['mhc_restriction'], self.hla_dictionary)
            pbar.update(1)
            seq_table['mhc_restriction_two'] = transform_mhc_restriction(
                seq_table['mhc_restriction_two'], self.hla_dictionary)
            pbar.update(1)
        
        # Vectorized lookup - much faster!
        with tqdm(total=2, desc=f"      • {source} mhc_ligand MHC lookup", leave=False) as pbar:
            seq_table['mhc_one'] = seq_table['mhc_restriction'].map(self.hla_dictionary)
            pbar.update(1)
            seq_table['mhc_two'] = seq_table['mhc_restriction_two'].map(self.hla_dictionary)
            pbar.update(1)

        
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
            log = logging.getLogger(__name__)
            log.info(f"      • {source} receptor: Loaded {len(df):,} raw records")
        except FileNotFoundError:
            log = logging.getLogger(__name__)
            log.error(f"    ✗ {source} receptor file not found: {file_path}")
            return pd.DataFrame(), pd.DataFrame()
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        if self.test:
            df = df.sample(frac=0.1, random_state=21)
            log.info(f"      • {source} receptor: Test mode - sampled to {len(df):,} records")

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
        
        # Filter out sequences shorter than 4 characters
        if 'tra' in sequence_table.columns:
            sequence_table.loc[sequence_table['tra'].str.len() < 4, 'tra'] = ''
        if 'trb' in sequence_table.columns:
            sequence_table.loc[sequence_table['trb'].str.len() < 4, 'trb'] = ''
        
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
        
        # Enable Dask progress bar
        from dask.diagnostics import ProgressBar
        ProgressBar().register()

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

        # Progress bar for iReceptor databases
        pbar = tqdm(databases.items(), desc="  iReceptor databases", unit="db")
        
        for db_key, (parse_func, db_name) in pbar:
            pbar.set_description(f"  iReceptor: {db_key}")
            
            if os.path.exists(self._mri_dir / f"ireceptor_{db_name}_mri.parquet"):
                log.info("  • %s: skipped (already parsed)", db_key)
                continue
            
            t0 = time.time()

            # ---- TSV → Dask or PyArrow streaming --------------------------------------------------
            db_mri, db_seq = parse_func(db_name)          # dask frames or empty if PyArrow already wrote
            
            # Check if PyArrow streaming already wrote the files (returns empty dask df with 1 partition)
            if db_mri.npartitions == 1:
                # Check if this was a PyArrow streaming operation (already complete)
                if os.path.exists(self._mri_dir / f"ireceptor_{db_name}_mri.parquet"):
                    log.info(f"      ✓ {db_key} completed via streaming (%.2fs)", time.time() - t0)
                    continue
            
            # Skip premature computation - just log that we're processing
            log.info(f"      • Processing {db_key} with {db_mri.npartitions} partitions...")

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
            
            # Write MRI data (this is where actual computation happens)
            log.info(f"      • Writing MRI data to parquet...")
            merged.to_parquet(mri_path,
                            engine="pyarrow",
                            compression="snappy",
                            write_index=False)

            # pad missing cols once, then write
            for c in seq_cols:
                if c not in db_seq.columns:
                    db_seq[c] = ""
            db_seq = db_seq[seq_cols]

            # Write sequence data (computation happens here too)
            log.info(f"      • Writing sequence data to parquet...")
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

        # Filter out sequences shorter than 4 characters (Dask-compatible)
        seq['tra'] = seq['tra'].where(seq['tra'].str.len() >= 4, '')
        seq['trb'] = seq['trb'].where(seq['trb'].str.len() >= 4, '')
        
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
        Bulk TRA or TRB file → PyArrow streaming for massive files.
        Optimized for very large files (e.g., 112GB paone).
        """
        cfg = self.config["databases"].get("ireceptor", {})
        db_path = cfg.get(source, {}).get("database", "")
        if not db_path:
            return dd.from_pandas(pd.DataFrame(), 1), dd.from_pandas(pd.DataFrame(), 1)

        import os
        import pyarrow as pa
        import pyarrow.csv as pa_csv
        import pyarrow.parquet as pq
        from pathlib import Path
        
        log = logging.getLogger(__name__)
        file_size_gb = os.path.getsize(db_path) / (1024**3)
        
        # For very large files (>50GB), use PyArrow streaming instead of Dask
        if file_size_gb > 50:
            log.info(f"      • Using PyArrow streaming for {file_size_gb:.1f}GB file...")
            
            # Output paths
            mri_path = self._mri_dir / f"ireceptor_{source}_mri.parquet"
            seq_path = self._seq_dir / f"ireceptor_{source}_seq.parquet"
            
            # Create temp directory for partitioned output
            temp_mri_dir = Path(str(mri_path) + "_temp")
            temp_seq_dir = Path(str(seq_path) + "_temp")
            temp_mri_dir.mkdir(exist_ok=True)
            temp_seq_dir.mkdir(exist_ok=True)
            
            # Stream and process in chunks
            # Use streaming table reader instead of open_csv to avoid type inference issues
            try:
                log.info(f"      • Opening CSV stream reader with pandas chunking...")
                
                chunk_num = 0
                total_rows = 0
                
                # Use pandas read_csv with chunksize for better control
                chunk_size_rows = 5_000_000  # ~5M rows per chunk for 10GB blocks
                
                with tqdm(desc=f"      • Streaming {source}", unit="chunk") as pbar:
                    for df_chunk in pd.read_csv(
                        db_path,
                        sep='\t',
                        dtype=str,
                        chunksize=chunk_size_rows,
                        na_filter=False,
                        low_memory=False
                    ):
                        chunk_num += 1
                        total_rows += len(df_chunk)
                        
                        # Rename for consistency
                        df = df_chunk
                        
                        if df.empty:
                            pbar.update(1)
                            continue
                        
                        # Process TRA/TRB
                        keep = ["repertoire_id", "cell_id", "clone_id",
                               "locus", "v_call", "d_call", "j_call", "junction_aa"]
                        for c in keep:
                            if c not in df.columns:
                                df[c] = ""
                        df = df[keep]
                        
                        # Build long → wide
                        tra = df[df['locus'] == "TRA"].assign(
                            trb="", trbv_gene="", trbd_gene="", trbj_gene=""
                        ).rename(columns={
                            "junction_aa": "tra",
                            "v_call": "trav_gene",
                            "d_call": "trad_gene",
                            "j_call": "traj_gene",
                        })
                        
                        trb = df[df['locus'] == "TRB"].assign(
                            tra="", trav_gene="", trad_gene="", traj_gene=""
                        ).rename(columns={
                            "junction_aa": "trb",
                            "v_call": "trbv_gene",
                            "d_call": "trbd_gene",
                            "j_call": "trbj_gene",
                        })
                        
                        final = pd.concat([tra, trb], axis=0, ignore_index=True)
                        final["source"] = f"ireceptor_{source}"
                        
                        # MRI
                        mri_cols = ["repertoire_id", "cell_id", "clone_id"]
                        mri_chunk = final[mri_cols].copy()
                        
                        # Sequence
                        seq_cols = ["tra", "trb", "trav_gene", "trad_gene", "traj_gene",
                                   "trbv_gene", "trbd_gene", "trbj_gene", "source"]
                        seq_chunk = final[seq_cols].copy()
                        
                        # Filter short sequences
                        seq_chunk.loc[seq_chunk['tra'].str.len() < 4, 'tra'] = ''
                        seq_chunk.loc[seq_chunk['trb'].str.len() < 4, 'trb'] = ''
                        seq_chunk = seq_chunk.drop_duplicates()
                        
                        # Write chunks to temp parquet files
                        if not mri_chunk.empty:
                            mri_chunk.to_parquet(
                                temp_mri_dir / f"chunk_{chunk_num:06d}.parquet",
                                engine="pyarrow",
                                compression="snappy",
                                index=False
                            )
                        
                        if not seq_chunk.empty:
                            seq_chunk.to_parquet(
                                temp_seq_dir / f"chunk_{chunk_num:06d}.parquet",
                                engine="pyarrow",
                                compression="snappy",
                                index=False
                            )
                        
                        pbar.update(1)
                        pbar.set_postfix({"rows": f"{total_rows:,}", "chunks": chunk_num})
                        
                        # Clean up memory
                        del df, tra, trb, final, mri_chunk, seq_chunk
                        if chunk_num % 10 == 0:
                            gc.collect()
            
                log.info(f"      • Processed {total_rows:,} rows in {chunk_num} chunks")
                
            except Exception as e:
                log.error(f"      ✗ Pandas chunked reading failed: {e}")
                log.info(f"      • Falling back to Dask for this file...")
                # Clean up any partial temp files
                import shutil
                shutil.rmtree(temp_mri_dir, ignore_errors=True)
                shutil.rmtree(temp_seq_dir, ignore_errors=True)
                # Return empty to trigger skip
                return dd.from_pandas(pd.DataFrame(), 1), dd.from_pandas(pd.DataFrame(), 1)
            
            # Consolidate temp files into final parquet
            log.info(f"      • Consolidating MRI chunks...")
            mri_files = sorted(temp_mri_dir.glob("*.parquet"))
            if mri_files:
                mri_table = pq.ParquetDataset(temp_mri_dir).read()
                pq.write_table(mri_table, mri_path, compression='snappy')
            
            log.info(f"      • Consolidating sequence chunks...")
            seq_files = sorted(temp_seq_dir.glob("*.parquet"))
            if seq_files:
                seq_table = pq.ParquetDataset(temp_seq_dir).read()
                pq.write_table(seq_table, seq_path, compression='snappy')
            
            # Clean up temp directories
            import shutil
            shutil.rmtree(temp_mri_dir, ignore_errors=True)
            shutil.rmtree(temp_seq_dir, ignore_errors=True)
            
            log.info(f"      ✓ {source} completed via PyArrow streaming")
            
            # Return empty Dask dataframes since we already wrote the files
            return dd.from_pandas(pd.DataFrame(), 1), dd.from_pandas(pd.DataFrame(), 1)
        
        # Original Dask code for smaller files - define blocksize first
        blocksize = "64 MB"
        
        if self.test:
            # Read first 10% of blocks instead of sampling entire file
            df = dd.read_csv(db_path,
                            sep="\t",
                            dtype=str,
                            na_filter=False,
                            blocksize=blocksize,
                            assume_missing=True)
            # Only take first few partitions for testing
            n_partitions = max(1, df.npartitions // 10)
            df = df.head(n_partitions * 200_000, npartitions=-1, compute=False)
        else:
            log = logging.getLogger(__name__)
            log.info(f"      • Reading {file_size_gb:.1f}GB file with {blocksize} blocks...")
            df = dd.read_csv(db_path,
                            sep="\t",
                            dtype=str,
                            na_filter=False,
                            blocksize=blocksize,
                            assume_missing=True)
            log.info(f"      • Created {df.npartitions} partitions")

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
        
        # Filter out sequences shorter than 4 characters (Dask-compatible)
        seq['tra'] = seq['tra'].where(seq['tra'].str.len() >= 4, '')
        seq['trb'] = seq['trb'].where(seq['trb'].str.len() >= 4, '')

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
            # Wrap in tqdm for progress tracking
            for item in tqdm(parser, desc=f"      Loading {source} metadata", unit="repertoire"):
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