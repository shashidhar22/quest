"""
Test duplication tracking functionality in streaming parser.
"""

import sys
import pandas as pd
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parsers.streaming_parser_complete import StreamingParserComplete


def test_duplication_stats():
    """Test that duplication statistics are calculated correctly."""
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    
    try:
        # Create test data with known duplication patterns
        test_file = Path(temp_dir) / "bulk_survey_trb" / "test_sample.tsv"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create data with 50% duplication in TRB sequences
        # Using format_four columns: amino_acid, v_resolved, d_resolved, j_resolved, frame_type
        test_data = pd.DataFrame({
            'rearrangement': ['TGTGCCAGCAGTTTGGGACAGGCATATGAGCAGTTCTTC'] * 10,
            'amino_acid': ['CASSLGQAYEQYF', 'CASSLGQAYEQYF', 'CASSFTGELFF', 'CASSFTGELFF', 
                          'CASSDAGELF', 'CASSDAGELF', 'CASSQGLSTDTQYF', 'CASSQGLSTDTQYF',
                          'CASSLGGTYNEQFF', 'CASSLGGTYNEQFF'],
            'bio_identity': [''] * 10,
            'templates': ['100'] * 10,
            'frame_type': ['In'] * 10,
            'rearrangement_type': ['VDJ'] * 10,
            'cdr3_length': ['13'] * 10,
            'frequency': ['0.01'] * 10,
            'productive_frequency': ['0.01'] * 10,
            'v_resolved': ['TCRBV05-01*01', 'TCRBV05-01*01', 'TCRBV06-01*01', 'TCRBV06-01*01',
                          'TCRBV07-02*01', 'TCRBV07-02*01', 'TCRBV12-03*01', 'TCRBV12-03*01',
                          'TCRBV28*01', 'TCRBV28*01'],
            'd_resolved': ['TCRBD02*01', 'TCRBD02*01', 'TCRBD01*01', 'TCRBD01*01',
                          'TCRBD02*01', 'TCRBD02*01', 'TCRBD01*01', 'TCRBD01*01',
                          'TCRBD02*01', 'TCRBD02*01'],
            'j_resolved': ['TCRBJ02-07*01', 'TCRBJ02-07*01', 'TCRBJ02-02*01', 'TCRBJ02-02*01',
                          'TCRBJ02-02*01', 'TCRBJ02-02*01', 'TCRBJ02-03*01', 'TCRBJ02-03*01',
                          'TCRBJ02-01*01', 'TCRBJ02-01*01'],
            'v_family': ['TCRBV05'] * 10,
            'v_family_ties': [''] * 10,
            'v_gene': ['TCRBV05-01'] * 10,
            'v_gene_ties': [''] * 10,
            'v_allele': ['TCRBV05-01*01'] * 10,
            'v_allele_ties': [''] * 10,
            'd_family': ['TCRBD02'] * 10,
            'd_family_ties': [''] * 10,
            'd_gene': ['TCRBD02'] * 10,
            'd_gene_ties': [''] * 10,
            'd_allele': ['TCRBD02*01'] * 10,
            'd_allele_ties': [''] * 10,
            'j_family': ['TCRBJ02'] * 10,
            'j_family_ties': [''] * 10,
            'j_gene': ['TCRBJ02-07'] * 10,
            'j_gene_ties': [''] * 10,
            'j_allele': ['TCRBJ02-07*01'] * 10,
            'j_allele_ties': [''] * 10
        })
        
        test_data.to_csv(test_file, sep='\t', index=False)
        
        # Get config path relative to project root
        project_root = Path(__file__).parent.parent
        config_path = project_root / 'config' / 'header_config.yaml'
        
        # Initialize parser
        parser = StreamingParserComplete(
            format_config=str(config_path),
            output_dir=output_dir,
            chunk_size=100,
            test_mode=False
        )
        
        # Process the file
        format_type, format_name, delimiter = parser.detect_format(str(test_file))
        mri_rows, seq_rows, combined_df = parser._parse_bulk_streaming(
            str(test_file), format_name, delimiter
        )
        
        # Calculate stats
        stats = parser._calculate_duplication_stats(combined_df, str(test_file))
        
        print("\n" + "="*60)
        print("DUPLICATION TRACKING TEST RESULTS")
        print("="*60)
        print(f"\nTest file: {test_file.name}")
        print(f"Total rows: {stats['total_rows']}")
        
        # Check TRB duplication (should be 50%)
        if 'trb_dup_rate' in stats:
            dup_rate = stats['trb_dup_rate'] * 100
            print(f"\nTRB CDR3 sequences:")
            print(f"  Total: {stats['trb_total']}")
            print(f"  Unique: {stats['trb_unique']}")
            print(f"  Duplication rate: {dup_rate:.1f}%")
            
            # Verify expected 50% duplication (10 total, 5 unique)
            assert stats['trb_total'] == 10, f"Expected 10 total TRB, got {stats['trb_total']}"
            assert stats['trb_unique'] == 5, f"Expected 5 unique TRB, got {stats['trb_unique']}"
            assert 49.0 <= dup_rate <= 51.0, f"Expected ~50% duplication, got {dup_rate:.1f}%"
            
            print("  ✓ Duplication rate correct!")
        
        # Check V gene duplication
        if 'trbv_gene_dup_rate' in stats:
            v_dup_rate = stats['trbv_gene_dup_rate'] * 100
            print(f"\nTRBV genes:")
            print(f"  Total: {stats['trbv_gene_total']}")
            print(f"  Unique: {stats['trbv_gene_unique']}")
            print(f"  Duplication rate: {v_dup_rate:.1f}%")
        
        # Check J gene duplication
        if 'trbj_gene_dup_rate' in stats:
            j_dup_rate = stats['trbj_gene_dup_rate'] * 100
            print(f"\nTRBJ genes:")
            print(f"  Total: {stats['trbj_gene_total']}")
            print(f"  Unique: {stats['trbj_gene_unique']}")
            print(f"  Duplication rate: {j_dup_rate:.1f}%")
        
        print("\n" + "="*60)
        print("✓ TEST PASSED: Duplication tracking working correctly!")
        print("="*60 + "\n")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        shutil.rmtree(output_dir)


if __name__ == "__main__":
    test_duplication_stats()
