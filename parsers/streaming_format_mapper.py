"""
Streaming Format Mapper - Maps all formats to unified schema
Handles 2TB+ datasets efficiently with minimal memory footprint
"""
import yaml
from typing import Dict, Optional, Tuple, List
from collections import OrderedDict


class FormatMapper:
    """Maps various input formats to unified output schema"""
    
    # Unified output schema
    OUTPUT_SCHEMA = [
        'tra', 'trav_gene', 'trad_gene', 'traj_gene',
        'trb', 'trbv_gene', 'trbd_gene', 'trbj_gene',
        'peptide', 'mhc_one', 'mhc_two',
        'mhc_one_id', 'mhc_two_id',
        'repertoire_id', 'study_id', 'source', 
        'host_organism', 'category', 'molecule_type'
    ]
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to header_config.yaml
        """
        with open(config_path, 'r') as f:
            self.format_config = yaml.safe_load(f)
        
        # Build mapping rules for each format
        self.bulk_mappings = self._build_bulk_mappings()
        self.misc_mappings = self._build_misc_mappings()
    
    def _build_bulk_mappings(self) -> Dict[str, Dict]:
        """Build column mapping rules for bulk formats"""
        mappings = {}
        
        # Format One: ['VJCombo', 'Copy', 'VGene', 'JGene', 'aaCDR3', 'ntCDR3', 'NetInsertionLength']
        mappings['format_one'] = {
            'columns': self.format_config['bulk']['format_one'],
            'drop': ['VJCombo', 'Copy', 'ntCDR3', 'NetInsertionLength'],
            'chain_detect': 'VGene',  # Use this to detect TRA vs TRB
            'TRA_map': {'VGene': 'trav_gene', 'JGene': 'traj_gene', 'aaCDR3': 'tra'},
            'TRB_map': {'VGene': 'trbv_gene', 'JGene': 'trbj_gene', 'aaCDR3': 'trb'}
        }
        
        # Format Two: V/D/J alleles with CDR3 amino acid
        mappings['format_two'] = {
            'columns': self.format_config['bulk']['format_two'],
            'drop': ['Read count', 'Percentage', 'CDR3 nucleotide sequence',
                    'CDR3 nucleotide quality', 'Min quality', 'V segments',
                    'J segments', 'D segments', 'Last V nucleotide position ',
                    'First D nucleotide position', 'Last D nucleotide position',
                    'First J nucleotide position', 'VD insertions', 'DJ insertions',
                    'Total insertions'],
            'chain_detect': 'V alleles',
            'TRA_map': {'V alleles': 'trav_gene', 'J alleles': 'traj_gene', 
                       'D alleles': 'trad_gene', 'CDR3 amino acid sequence': 'tra'},
            'TRB_map': {'V alleles': 'trbv_gene', 'J alleles': 'trbj_gene',
                       'D alleles': 'trbd_gene', 'CDR3 amino acid sequence': 'trb'}
        }
        
        # Format Three: Similar to format two
        mappings['format_three'] = {
            'columns': self.format_config['bulk']['format_three'],
            'drop': ['Count', 'Percentage', 'CDR3 nucleotide sequence',
                    'Last V nucleotide position', 'First D nucleotide position',
                    'Last D nucleotide position', 'First J nucleotide position',
                    'Good events', 'Total events', 'Good reads', 'Total reads'],
            'chain_detect': 'V segments',
            'TRA_map': {'V segments': 'trav_gene', 'J segments': 'traj_gene',
                       'D segments': 'trad_gene', 'CDR3 amino acid sequence': 'tra'},
            'TRB_map': {'V segments': 'trbv_gene', 'J segments': 'trbj_gene',
                       'D segments': 'trbd_gene', 'CDR3 amino acid sequence': 'trb'}
        }
        
        # Format Four: Adaptive format
        mappings['format_four'] = {
            'columns': self.format_config['bulk']['format_four'],
            'keep': ['amino_acid', 'v_resolved', 'd_resolved', 'j_resolved'],
            'chain_detect': 'v_resolved',
            'TRA_map': {'v_resolved': 'trav_gene', 'j_resolved': 'traj_gene',
                       'd_resolved': 'trad_gene', 'amino_acid': 'tra'},
            'TRB_map': {'v_resolved': 'trbv_gene', 'j_resolved': 'trbj_gene',
                       'd_resolved': 'trbd_gene', 'amino_acid': 'trb'}
        }
        
        # Format Five: IMonitor format
        mappings['format_five'] = {
            'columns': self.format_config['bulk']['format_five'],
            'keep': ['CDR3(aa)', 'V_ref', 'D_ref', 'J_ref'],
            'chain_detect': 'V_ref',
            'TRA_map': {'V_ref': 'trav_gene', 'J_ref': 'traj_gene',
                       'D_ref': 'trad_gene', 'CDR3(aa)': 'tra'},
            'TRB_map': {'V_ref': 'trbv_gene', 'J_ref': 'trbj_gene',
                       'D_ref': 'trbd_gene', 'CDR3(aa)': 'trb'}
        }
        
        # Format Six: MiXCR format
        mappings['format_six'] = {
            'columns': self.format_config['bulk']['format_six'],
            'keep': ['aminoAcid(CDR3 in lowercase)', 'vGene', 'dGene', 'jGene'],
            'chain_detect': 'vGene',
            'TRA_map': {'vGene': 'trav_gene', 'jGene': 'traj_gene',
                       'dGene': 'trad_gene', 'aminoAcid(CDR3 in lowercase)': 'tra'},
            'TRB_map': {'vGene': 'trbv_gene', 'jGene': 'trbj_gene',
                       'dGene': 'trbd_gene', 'aminoAcid(CDR3 in lowercase)': 'trb'}
        }
        
        # Formats 7, 8, 10: Similar MiXCR-style formats
        for fmt in ['format_seven', 'format_eight', 'format_ten']:
            mappings[fmt] = {
                'columns': self.format_config['bulk'][fmt],
                'keep': ['aminoAcid', 'vGeneName', 'dGeneName', 'jGeneName'],
                'chain_detect': 'vGeneName',
                'TRA_map': {'vGeneName': 'trav_gene', 'jGeneName': 'traj_gene',
                           'dGeneName': 'trad_gene', 'aminoAcid': 'tra'},
                'TRB_map': {'vGeneName': 'trbv_gene', 'jGeneName': 'trbj_gene',
                           'dGeneName': 'trbd_gene', 'aminoAcid': 'trb'}
            }
        
        # Format Nine: Adaptive full export
        mappings['format_nine'] = {
            'columns': self.format_config['bulk']['format_nine'],
            'keep': ['amino_acid', 'v_gene', 'd_gene', 'j_gene'],
            'chain_detect': 'v_gene',
            'TRA_map': {'v_gene': 'trav_gene', 'j_gene': 'traj_gene',
                       'd_gene': 'trad_gene', 'amino_acid': 'tra'},
            'TRB_map': {'v_gene': 'trbv_gene', 'j_gene': 'trbj_gene',
                       'd_gene': 'trbd_gene', 'amino_acid': 'trb'}
        }
        
        # Format Eleven: Simplified format
        mappings['format_eleven'] = {
            'columns': self.format_config['bulk']['format_eleven'],
            'keep': ['cdr3_b_aa', 'v_b_gene', 'j_b_gene'],
            'direct_map': {'v_b_gene': 'trbv_gene', 'j_b_gene': 'trbj_gene',
                          'cdr3_b_aa': 'trb'}
        }
        
        return mappings
    
    def _build_misc_mappings(self) -> Dict[str, Dict]:
        """Build column mapping rules for misc formats"""
        mappings = {}
        
        # Format One: TRA:seq;TRB:seq format
        mappings['format_one'] = {
            'columns': self.format_config['misc']['format_one'],
            'special': 'paired_chains',  # Needs special parsing
            'nt_col': 'cdr3s_nt',
            'aa_col': 'cdr3s_aa'
        }
        
        # Format Two: Barcode with CDR3
        mappings['format_two'] = {
            'columns': self.format_config['misc']['format_two'],
            'special': 'single_cell',
            'keep': ['barcode', 'cdr3s_aa']
        }
        
        # Format Three: Single cell with VDJ
        mappings['format_three'] = {
            'columns': self.format_config['misc']['format_three'],
            'keep': ['CDR3.aa', 'V.name', 'D.name', 'J.name', 'TR_chain'],
            'chain_col': 'TR_chain',
            'TRA_map': {'V.name': 'trav_gene', 'J.name': 'traj_gene',
                       'D.name': 'trad_gene', 'CDR3.aa': 'tra'},
            'TRB_map': {'V.name': 'trbv_gene', 'J.name': 'trbj_gene',
                       'D.name': 'trbd_gene', 'CDR3.aa': 'trb'}
        }
        
        # Format Four: Seurat/scRNA-seq
        mappings['format_four'] = {
            'columns': self.format_config['misc']['format_four'],
            'special': 'single_cell',
            'keep': ['t_clonotype_id', 't_cdr3s_aa']
        }
        
        # Format Five: Complex single-cell
        mappings['format_five'] = {
            'columns': self.format_config['misc']['format_five'],
            'keep': ['CDR3A', 'CDR3B', 'TRAV', 'TRAJ', 'TRBV', 'TRBJ', 'TRBD'],
            'direct_map': {
                'CDR3A': 'tra', 'CDR3B': 'trb',
                'TRAV': 'trav_gene', 'TRAJ': 'traj_gene',
                'TRBV': 'trbv_gene', 'TRBJ': 'trbj_gene', 'TRBD': 'trbd_gene'
            }
        }
        
        # Format Six: MiXCR-style
        mappings['format_six'] = {
            'columns': self.format_config['misc']['format_six'],
            'keep': ['cdr3aa', 'v', 'd', 'j'],
            'chain_detect': 'v',
            'TRA_map': {'v': 'trav_gene', 'j': 'traj_gene', 'd': 'trad_gene', 'cdr3aa': 'tra'},
            'TRB_map': {'v': 'trbv_gene', 'j': 'trbj_gene', 'd': 'trbd_gene', 'cdr3aa': 'trb'}
        }
        
        return mappings
    
    def detect_format(self, columns: List[str], category: str = 'bulk') -> Optional[Tuple[str, Dict]]:
        """
        Detect which format a file is based on its columns
        
        Args:
            columns: List of column names from file
            category: 'bulk' or 'misc'
        
        Returns:
            (format_name, mapping_dict) or (None, None) if not recognized
        """
        column_set = set(columns)
        
        mappings = self.bulk_mappings if category == 'bulk' else self.misc_mappings
        
        for format_name, mapping in mappings.items():
            if column_set == set(mapping['columns']):
                return format_name, mapping
        
        return None, None
    
    def map_row(self, row: Dict, mapping: Dict, metadata: Dict) -> Dict:
        """
        Map a single row from input format to output schema
        
        Args:
            row: Dictionary of column->value from input file
            mapping: Mapping rules from detect_format
            metadata: Dict with repertoire_id, study_id, etc.
        
        Returns:
            Dictionary with output schema fields
        """
        output = {col: '' for col in self.OUTPUT_SCHEMA}
        
        # Add metadata
        output.update(metadata)
        
        # Handle special formats
        if mapping.get('special') == 'paired_chains':
            return self._map_paired_chains(row, mapping, output)
        elif mapping.get('special') == 'single_cell':
            return self._map_single_cell(row, mapping, output)
        
        # Handle direct mapping (no chain detection)
        if 'direct_map' in mapping:
            for src, dst in mapping['direct_map'].items():
                if src in row:
                    output[dst] = row[src]
            return output
        
        # Handle chain-specific mapping
        if 'chain_detect' in mapping:
            chain_col = mapping['chain_detect']
            chain_value = row.get(chain_col, '')
            
            # Detect if TRA or TRB
            if 'TRAV' in chain_value or 'TRA' in chain_value:
                col_map = mapping['TRA_map']
            elif 'TRBV' in chain_value or 'TRB' in chain_value:
                col_map = mapping['TRB_map']
            else:
                # Unknown chain, skip
                return None
            
            for src, dst in col_map.items():
                if src in row:
                    output[dst] = row[src]
        
        # Handle explicit chain column (misc format_three)
        if 'chain_col' in mapping:
            chain = row.get(mapping['chain_col'], '')
            if 'TRA' in chain:
                col_map = mapping['TRA_map']
            elif 'TRB' in chain:
                col_map = mapping['TRB_map']
            else:
                return None
            
            for src, dst in col_map.items():
                if src in row:
                    output[dst] = row[src]
        
        return output
    
    def _map_paired_chains(self, row: Dict, mapping: Dict, output: Dict) -> List[Dict]:
        """Handle misc format_one with paired TRA:seq;TRB:seq"""
        results = []
        
        nt_field = row.get(mapping['nt_col'], '')
        aa_field = row.get(mapping['aa_col'], '')
        
        if not nt_field or not aa_field:
            return [output]
        
        # Parse paired chains
        nt_chains = nt_field.split(';')
        aa_chains = aa_field.split(';')
        
        tra_seqs = []
        trb_seqs = []
        
        for nt, aa in zip(nt_chains, aa_chains):
            if ':' not in nt or ':' not in aa:
                continue
            
            chain_nt, _ = nt.split(':', 1)
            chain_aa, seq = aa.split(':', 1)
            
            if 'TRA' in chain_nt:
                tra_seqs.append(seq)
            elif 'TRB' in chain_nt:
                trb_seqs.append(seq)
        
        # Create all combinations
        if tra_seqs and trb_seqs:
            for tra in tra_seqs:
                for trb in trb_seqs:
                    result = output.copy()
                    result['tra'] = tra
                    result['trb'] = trb
                    results.append(result)
        elif tra_seqs:
            for tra in tra_seqs:
                result = output.copy()
                result['tra'] = tra
                results.append(result)
        elif trb_seqs:
            for trb in trb_seqs:
                result = output.copy()
                result['trb'] = trb
                results.append(result)
        else:
            results.append(output)
        
        return results
    
    def _map_single_cell(self, row: Dict, mapping: Dict, output: Dict) -> Dict:
        """Handle single-cell formats with simple CDR3 extraction"""
        keep_cols = mapping.get('keep', [])
        
        for col in keep_cols:
            if col in row and row[col]:
                # Try to parse if it's a paired format
                if ';' in row[col]:
                    parts = row[col].split(';')
                    for part in parts:
                        if ':' in part:
                            chain, seq = part.split(':', 1)
                            if 'TRA' in chain:
                                output['tra'] = seq
                            elif 'TRB' in chain:
                                output['trb'] = seq
                else:
                    # Assume it's TRB if not specified
                    output['trb'] = row[col]
        
        return output
