"""
TCR Full-Length Sequence Stitcher

Uses stitchr to generate full-length TCR sequences from CDR3 + gene segments.
This module processes CDR3 sequences with V/J gene annotations to produce complete TCR alpha
and beta chain sequences.

Dependencies:
    - stitchr: Reconstructs full-length TCR sequences from CDR3 + gene segments
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import warnings
import re
import os
import sys

# Import format_to_imgt from same package
try:
    from parsers.format_to_imgt import standardize_to_imgt
    IMGT_FORMATTER_AVAILABLE = True
except ImportError:
    try:
        # Fallback for when running from parsers directory
        from format_to_imgt import standardize_to_imgt
        IMGT_FORMATTER_AVAILABLE = True
    except ImportError:
        IMGT_FORMATTER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Suppress warnings from stitchr
warnings.filterwarnings('ignore')

try:
    from Stitchr import stitchrfunctions as fxn
    from Stitchr import stitchr as st
    STITCHR_AVAILABLE = True
except ImportError:
    STITCHR_AVAILABLE = False
    logger.warning("stitchr not available. Install with: pip install stitchr")

try:
    import tcrconvert
    TCRCONVERT_AVAILABLE = True
except ImportError:
    TCRCONVERT_AVAILABLE = False
    logger.warning("tcrconvert not available. Install with: pip install tcrconvert")


class TCRStitcher:
    """
    Handles full-length TCR sequence generation from CDR3 + gene segments.
    
    Workflow:
        1. Normalize gene names to be compatible with stitchr
        2. Generate full-length sequences using stitchr
        3. Return complete TCR alpha/beta sequences
    """
    
    def __init__(self, species: str = "HUMAN"):
        """
        Initialize TCR stitcher.
        
        Args:
            species: Species for stitching (default: "HUMAN")
        """
        self.species = species.upper()
        self.enabled = STITCHR_AVAILABLE
        self.use_imgt_formatter = IMGT_FORMATTER_AVAILABLE
        
        if not self.enabled:
            logger.warning("TCR stitching disabled. Missing: stitchr")
        else:
            # Initialize stitchr data for both TRA and TRB chains
            try:
                self.tra_data = self._init_chain_data('TRA')
                self.trb_data = self._init_chain_data('TRB')
                logger.info("TCR stitcher initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize stitchr data: {e}")
                self.enabled = False
    
    def _init_chain_data(self, chain: str) -> dict:
        """
        Initialize reference data for a specific chain.
        
        Args:
            chain: 'TRA' or 'TRB'
        
        Returns:
            Dictionary containing initialized data structures
        """
        tcr_dat, functionality, partial = fxn.get_ref_data(chain, st.gene_types, self.species)
        codons = fxn.get_optimal_codons('', self.species)
        j_res, low_conf_js = fxn.get_j_motifs(self.species)
        c_res = fxn.get_c_motifs(self.species)
        
        return {
            'tcr_dat': tcr_dat,
            'functionality': functionality,
            'partial': partial,
            'codons': codons,
            'j_res': j_res,
            'low_conf_js': low_conf_js,
            'c_res': c_res
        }
    
    def convert_to_imgt(self, gene: str, chain: str) -> Optional[str]:
        """
        Convert gene name to IMGT format using tcrconvert.
        
        Args:
            gene: Gene name in any format
            chain: 'TRA' or 'TRB'
        
        Returns:
            Gene name in IMGT format or None if conversion fails
        """
        if not TCRCONVERT_AVAILABLE or not gene:
            return None
        
        try:
            # Determine gene type (V, D, or J)
            if 'V' in gene.upper():
                gene_type = 'V'
            elif 'D' in gene.upper():
                gene_type = 'D'
            elif 'J' in gene.upper():
                gene_type = 'J'
            else:
                return None
            
            # Convert to IMGT format
            result = tcrconvert.convert_gene_name(
                gene_name=gene,
                species='human',
                chain_type=chain.lower(),  # 'tra' or 'trb'
                target_format='imgt',
                gene_type=gene_type.lower()
            )
            
            return result if result else None
            
        except Exception as e:
            logger.debug(f"tcrconvert failed for {gene}: {e}")
            return None
    
    def normalize_gene_name(self, gene: str, chain: str = None) -> Optional[str]:
        """
        Normalize gene name to IMGT format using format_to_imgt.py.
        Falls back to tcrconvert, then manual normalization if needed.
        
        Handles variations like:
        - "TCRBV13-01*01" -> "TRBV13-1*01" (IMGT format)
        - "TCRAV1-2" -> "TRAV1-2"
        - "TRBV06-05" -> "TRBV6-5"
        - "TRAJ10 56 0" -> "TRAJ10" (removes trailing text)
        - Multiple alleles: "TRAV1-2,TRAV1-3" -> "TRAV1-2"
        
        Args:
            gene: Gene name
            chain: Chain type ('TRA' or 'TRB') - for logging/fallback
        
        Returns:
            Normalized gene name or None if invalid
        """
        if not gene or pd.isna(gene) or gene == '':
            return None
        
        try:
            # Clean up gene name
            gene = str(gene).strip()
            
            # Handle multiple alleles (take first)
            if ',' in gene:
                gene = gene.split(',')[0].strip()
            if ';' in gene:
                gene = gene.split(';')[0].strip()
            
            # Try format_to_imgt first (most robust)
            if self.use_imgt_formatter:
                imgt_gene = standardize_to_imgt(gene)
                if imgt_gene:
                    if gene != imgt_gene:
                        logger.debug(f"IMGT format: {gene} -> {imgt_gene}")
                    return imgt_gene
            
            # Fallback to tcrconvert if available and chain is specified
            if chain and TCRCONVERT_AVAILABLE:
                imgt_gene = self.convert_to_imgt(gene, chain)
                if imgt_gene:
                    logger.debug(f"tcrconvert: {gene} -> {imgt_gene}")
                    return imgt_gene
            
            # Last resort: manual normalization
            # Remove "TCR" prefix if present (e.g., TCRAV -> TRAV)
            gene = re.sub(r'^TCR([AB][VDJ])', r'TR\1', gene)
            
            # Normalize numbering (remove leading zeros)
            # TRAV01-02 -> TRAV1-2
            gene = re.sub(r'([TRAV|TRBV|TRAJ|TRBJ|TRAD|TRBD])0*(\d+)-0*(\d+)', r'\1\2-\3', gene)
            
            logger.debug(f"Manual normalization: {gene}")
            return gene
            
        except Exception as e:
            logger.debug(f"Gene normalization failed for {gene}: {e}")
            return None
    
    def stitch_tcr(
        self,
        cdr3: str,
        v_gene: str,
        j_gene: str,
        chain: str,
        c_gene: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate full-length TCR sequence using stitchr.
        
        Args:
            cdr3: CDR3 amino acid sequence
            v_gene: V gene name (will be normalized to IMGT format)
            j_gene: J gene name (will be normalized to IMGT format)
            chain: 'TRA' or 'TRB'
            c_gene: Optional constant region gene (auto-selected if None)
        
        Returns:
            Full-length TCR amino acid sequence or None if stitching fails
        """
        if not STITCHR_AVAILABLE or not all([cdr3, v_gene, j_gene]):
            return None
        
        try:
            # Clean inputs
            cdr3 = str(cdr3).strip()
            v_gene = str(v_gene).strip()
            j_gene = str(j_gene).strip()
            
            # Normalize gene names to IMGT format
            norm_v_gene = self.normalize_gene_name(v_gene, chain)
            norm_j_gene = self.normalize_gene_name(j_gene, chain)
            
            if not norm_v_gene or not norm_j_gene:
                return None
            
            # Auto-select constant region if not provided
            if c_gene is None:
                if chain == 'TRA':
                    c_gene = 'TRAC*01'
                elif chain == 'TRB':
                    c_gene = 'TRBC1*01'  # Default to TRBC1
            
            # Get chain data
            chain_data = self.tra_data if chain == 'TRA' else self.trb_data
            
            # Build tcr_bits dictionary for stitchr
            tcr_bits = {
                'v': norm_v_gene,
                'j': norm_j_gene,
                'cdr3': cdr3,
                'l': norm_v_gene,  # Use V gene for leader
                'c': c_gene,
                'mode': '',
                'skip_c_checks': False,
                'skip_n_checks': False,
                'no_leader': False,  # Include leader sequence
                'species': self.species,
                'seamless': False,
                '5_prime_seq': '',
                '3_prime_seq': '',
                'name': f'{chain}-{cdr3}'
            }
            
            # Call stitchr
            stitched = st.stitch(
                tcr_bits,
                chain_data['tcr_dat'],
                chain_data['functionality'],
                chain_data['partial'],
                chain_data['codons'],
                3,  # codon_warning_threshold
                '',  # aa_file
                chain_data['c_res'],
                chain_data['j_res'],
                chain_data['low_conf_js']
            )
            
            # Extract and translate nucleotide sequence to amino acid
            if stitched and 'stitched_nt' in stitched:
                # Translate the full nucleotide sequence to amino acid
                aa_seq = self._translate_dna(stitched['stitched_nt'])
                if aa_seq and len(aa_seq) > len(cdr3):
                    return aa_seq
            
            return None
            
        except Exception as e:
            logger.debug(f"TCR stitching failed for {chain} {cdr3}: {e}")
            return None
    
    def process_tcr_pair(
        self,
        tra_cdr3: str,
        tra_v: str,
        tra_j: str,
        trb_cdr3: str,
        trb_v: str,
        trb_j: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Process a TCR alpha/beta pair to generate full-length sequences.
        
        Args:
            tra_cdr3: Alpha chain CDR3
            tra_v: Alpha V gene
            tra_j: Alpha J gene
            trb_cdr3: Beta chain CDR3
            trb_v: Beta V gene
            trb_j: Beta J gene
        
        Returns:
            (tra_full_sequence, trb_full_sequence) - either may be None
        """
        tra_full = None
        trb_full = None
        
        # Process TRA
        if tra_cdr3 and tra_v and tra_j:
            norm_tra_v = self.normalize_gene_name(tra_v, 'TRA')
            norm_tra_j = self.normalize_gene_name(tra_j, 'TRA')
            
            if norm_tra_v and norm_tra_j:
                tra_full = self.stitch_tcr(
                    cdr3=tra_cdr3,
                    v_gene=norm_tra_v,
                    j_gene=norm_tra_j,
                    chain='TRA'
                )
        
        # Process TRB
        if trb_cdr3 and trb_v and trb_j:
            norm_trb_v = self.normalize_gene_name(trb_v, 'TRB')
            norm_trb_j = self.normalize_gene_name(trb_j, 'TRB')
            
            if norm_trb_v and norm_trb_j:
                trb_full = self.stitch_tcr(
                    cdr3=trb_cdr3,
                    v_gene=norm_trb_v,
                    j_gene=norm_trb_j,
                    chain='TRB'
                )
        
        return tra_full, trb_full
    
    def _translate_dna(self, dna_seq: str) -> str:
        """
        Translate DNA sequence to amino acid sequence.
        
        Args:
            dna_seq: DNA sequence string
            
        Returns:
            Amino acid sequence string
        """
        if not dna_seq:
            return ''
        
        try:
            # Simple translation table (standard genetic code)
            codon_table = {
                'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
                'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
                'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
                'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
                'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
                'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
                'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
                'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
                'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
                'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
                'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
                'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
                'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
                'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
                'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
                'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
            }
            
            dna_seq = dna_seq.upper()
            protein = []
            
            # Translate in triplets
            for i in range(0, len(dna_seq) - 2, 3):
                codon = dna_seq[i:i+3]
                aa = codon_table.get(codon, 'X')  # X for unknown
                if aa == '*':  # Stop codon
                    break
                protein.append(aa)
            
            return ''.join(protein)
            
        except Exception as e:
            logger.debug(f"Translation failed: {e}")
            return ''
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add full-length TCR sequences to a DataFrame.
        
        Expected columns: tra, trav_gene, traj_gene, trb, trbv_gene, trbj_gene
        Updates gene columns with IMGT-formatted names
        Adds columns: tra_full, trb_full (amino acid sequences)
        
        Args:
            df: DataFrame with TCR data
        
        Returns:
            DataFrame with formatted gene names and full-length amino acid sequences
        """
        if not self.enabled or df.empty:
            # Just format gene names if stitching is disabled
            df['tra_full'] = ''
            df['trb_full'] = ''
            # Still format gene names
            df['trav_gene'] = df.get('trav_gene', '').apply(lambda x: self.normalize_gene_name(x, 'TRA') if x else '')
            df['traj_gene'] = df.get('traj_gene', '').apply(lambda x: self.normalize_gene_name(x, 'TRA') if x else '')
            df['trbv_gene'] = df.get('trbv_gene', '').apply(lambda x: self.normalize_gene_name(x, 'TRB') if x else '')
            df['trbj_gene'] = df.get('trbj_gene', '').apply(lambda x: self.normalize_gene_name(x, 'TRB') if x else '')
            return df
        
        # Initialize columns
        tra_full_list = []
        trb_full_list = []
        tra_v_fmt_list = []
        tra_j_fmt_list = []
        trb_v_fmt_list = []
        trb_j_fmt_list = []
        
        # Process dataframe
        for _, row in df.iterrows():
            tra_cdr3 = row.get('tra', '')
            tra_v = row.get('trav_gene', '')
            tra_j = row.get('traj_gene', '')
            trb_cdr3 = row.get('trb', '')
            trb_v = row.get('trbv_gene', '')
            trb_j = row.get('trbj_gene', '')
            
            # Normalize gene names using IMGT formatter
            norm_tra_v = self.normalize_gene_name(tra_v, 'TRA') if tra_v else ''
            norm_tra_j = self.normalize_gene_name(tra_j, 'TRA') if tra_j else ''
            norm_trb_v = self.normalize_gene_name(trb_v, 'TRB') if trb_v else ''
            norm_trb_j = self.normalize_gene_name(trb_j, 'TRB') if trb_j else ''
            
            # Store formatted gene names
            tra_v_fmt_list.append(norm_tra_v if norm_tra_v else '')
            tra_j_fmt_list.append(norm_tra_j if norm_tra_j else '')
            trb_v_fmt_list.append(norm_trb_v if norm_trb_v else '')
            trb_j_fmt_list.append(norm_trb_j if norm_trb_j else '')
            
            # Process TRA
            tra_aa = ''
            if tra_cdr3 and norm_tra_v and norm_tra_j:
                try:
                    # Build tcr_bits for TRA
                    tcr_bits = {
                        'v': norm_tra_v,
                        'j': norm_tra_j,
                        'cdr3': tra_cdr3,
                        'l': norm_tra_v,  # Use V gene for leader
                        'c': 'TRAC*01',
                        'mode': '',
                        'skip_c_checks': False,
                        'skip_n_checks': False,
                        'no_leader': False,  # Include leader sequence
                        'species': self.species,
                        'seamless': False,
                        '5_prime_seq': '',
                        '3_prime_seq': '',
                        'name': f'TRA-{tra_cdr3}'
                    }
                    
                    # Stitch TRA
                    stitched = st.stitch(
                        tcr_bits,
                        self.tra_data['tcr_dat'],
                        self.tra_data['functionality'],
                        self.tra_data['partial'],
                        self.tra_data['codons'],
                        3,  # codon_warning_threshold
                        '',  # aa_file
                        self.tra_data['c_res'],
                        self.tra_data['j_res'],
                        self.tra_data['low_conf_js']
                    )
                    
                    if stitched and 'stitched_nt' in stitched:
                        # Translate the full nucleotide sequence to amino acid
                        tra_aa = self._translate_dna(stitched['stitched_nt'])
                            
                except Exception as e:
                    logger.debug(f"TRA stitching failed for {tra_cdr3}: {e}")
            
            tra_full_list.append(tra_aa)
            
            # Process TRB
            trb_aa = ''
            if trb_cdr3 and norm_trb_v and norm_trb_j:
                try:
                    # Build tcr_bits for TRB
                    tcr_bits = {
                        'v': norm_trb_v,
                        'j': norm_trb_j,
                        'cdr3': trb_cdr3,
                        'l': norm_trb_v,  # Use V gene for leader
                        'c': 'TRBC1*01',  # Default to TRBC1
                        'mode': '',
                        'skip_c_checks': False,
                        'skip_n_checks': False,
                        'no_leader': False,  # Include leader sequence
                        'species': self.species,
                        'seamless': False,
                        '5_prime_seq': '',
                        '3_prime_seq': '',
                        'name': f'TRB-{trb_cdr3}'
                    }
                    
                    # Stitch TRB
                    stitched = st.stitch(
                        tcr_bits,
                        self.trb_data['tcr_dat'],
                        self.trb_data['functionality'],
                        self.trb_data['partial'],
                        self.trb_data['codons'],
                        3,  # codon_warning_threshold
                        '',  # aa_file
                        self.trb_data['c_res'],
                        self.trb_data['j_res'],
                        self.trb_data['low_conf_js']
                    )
                    
                    if stitched and 'stitched_nt' in stitched:
                        # Translate the full nucleotide sequence to amino acid
                        trb_aa = self._translate_dna(stitched['stitched_nt'])
                            
                except Exception as e:
                    logger.debug(f"TRB stitching failed for {trb_cdr3}: {e}")
            
            trb_full_list.append(trb_aa)
        
        # Update gene columns with formatted names
        df['trav_gene'] = tra_v_fmt_list
        df['traj_gene'] = tra_j_fmt_list
        df['trbv_gene'] = trb_v_fmt_list
        df['trbj_gene'] = trb_j_fmt_list
        
        # Add full-length amino acid sequences
        df['tra_full'] = tra_full_list
        df['trb_full'] = trb_full_list
        
        # Log statistics
        tra_success = sum(1 for x in tra_full_list if x)
        trb_success = sum(1 for x in trb_full_list if x)
        total_rows = len(df)
        
        if total_rows > 0:
            logger.info(
                f"TCR stitching: {tra_success}/{total_rows} TRA "
                f"({100*tra_success/total_rows:.1f}%), "
                f"{trb_success}/{total_rows} TRB "
                f"({100*trb_success/total_rows:.1f}%)"
            )
        
        return df


def add_full_tcr_sequences(
    df: pd.DataFrame,
    species: str = "HUMAN",
    enabled: bool = True
) -> pd.DataFrame:
    """
    Convenience function to add full-length TCR sequences to a DataFrame.
    
    Args:
        df: DataFrame with CDR3 and gene annotations
            Expected columns: tra, trav_gene, traj_gene, trb, trbv_gene, trbj_gene
        species: Species for gene conversion (default: "HUMAN")
        enabled: Whether to enable stitching (False = only format gene names)
    
    Returns:
        DataFrame with:
            - Gene columns (trav_gene, traj_gene, trbv_gene, trbj_gene) updated with IMGT format
            - tra_full: Full-length TRA amino acid sequence
            - trb_full: Full-length TRB amino acid sequence
    """
    if not enabled:
        df['tra_full'] = ''
        df['trb_full'] = ''
        return df
    
    stitcher = TCRStitcher(species=species)
    return stitcher.process_dataframe(df)
