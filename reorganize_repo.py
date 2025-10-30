#!/usr/bin/env python3
"""
Repository reorganization script for QUEST project.
This script moves files to a better organized structure.
"""

import os
import shutil
from pathlib import Path

# Get the repository root
REPO_ROOT = Path(__file__).parent

# Define the reorganization mapping
REORGANIZATION = {
    # Training scripts
    'scripts/ray_train.py': 'scripts/training/ray_train.py',
    'scripts/ray_fine_tune.py': 'scripts/training/ray_fine_tune.py',
    'scripts/ray_evaluator.py': 'scripts/training/ray_evaluator.py',
    
    # Inference scripts
    'run_inference.py': 'scripts/inference/run_inference.py',
    'inference_examples.py': 'scripts/inference/inference_examples.py',
    'inference_with_dataset.py': 'scripts/inference/inference_with_dataset.py',
    'quick_inference.py': 'scripts/inference/quick_inference.py',
    
    # Data processing scripts
    'scripts/ray_datawriter.py': 'scripts/data_processing/ray_datawriter.py',
    'reformat_data.py': 'scripts/data_processing/reformat_data.py',
    'scripts/reformat_batch.py': 'scripts/data_processing/reformat_batch.py',
    'run_parser_production.py': 'scripts/data_processing/run_parser_production.py',
    'run_parser_test.py': 'scripts/data_processing/run_parser_test.py',
    'reparse_databases.py': 'scripts/data_processing/reparse_databases.py',
    
    # Analysis scripts
    'analyze_duplication_parallel.py': 'scripts/analysis/analyze_duplication_parallel.py',
    'calculate_perplexity.py': 'scripts/analysis/calculate_perplexity.py',
    'evaluate_masked_predictions.py': 'scripts/analysis/evaluate_masked_predictions.py',
    'export_embeddings.py': 'scripts/analysis/export_embeddings.py',
    'best_worst_by_molecule.py': 'scripts/analysis/best_worst_by_molecule.py',
    'show_best_worst.py': 'scripts/analysis/show_best_worst.py',
    'summarize_data.py': 'scripts/analysis/summarize_data.py',
    'summarize_source_data.py': 'scripts/analysis/summarize_source_data.py',
    'create_source_mapping.py': 'scripts/analysis/create_source_mapping.py',
    
    # Utility scripts
    'run_duplication_analysis.sh': 'scripts/utils/run_duplication_analysis.sh',
    'setup_stitchr.sh': 'scripts/utils/setup_stitchr.sh',
    
    # Integration tests
    'test_tcrconvert_integration.py': 'tests/integration/test_tcrconvert_integration.py',
    'test_tcr_stitching.py': 'tests/integration/test_tcr_stitching.py',
    'test_stitchr_direct.py': 'tests/integration/test_stitchr_direct.py',
    'test_stitcher_diagnostic.py': 'tests/integration/test_stitcher_diagnostic.py',
    
    # Output files - reports
    'duplication_report_final.txt': 'outputs/reports/duplication_report_final.txt',
    'final_duplication_report2.txt': 'outputs/reports/final_duplication_report2.txt',
    'final_duplication_report2.csv': 'outputs/reports/final_duplication_report2.csv',
    'test_dup_report.txt': 'outputs/reports/test_dup_report.txt',
    'test_sorted_report.txt': 'outputs/reports/test_sorted_report.txt',
    'test_sorted_fixed.txt': 'outputs/reports/test_sorted_fixed.txt',
    'data_structure_summary.txt': 'outputs/reports/data_structure_summary.txt',
    
    # Output files - CSV
    'contigs_paired_analysis.csv': 'outputs/csv/contigs_paired_analysis.csv',
    'paired_sequences_analysis.csv': 'outputs/csv/paired_sequences_analysis.csv',
    'source_data_by_category.csv': 'outputs/csv/source_data_by_category.csv',
    'source_data_by_file.csv': 'outputs/csv/source_data_by_file.csv',
    'source_data_by_study.csv': 'outputs/csv/source_data_by_study.csv',
    'file_duplication_stats_final.csv': 'outputs/csv/file_duplication_stats_final.csv',
    'data_structure_stats.csv': 'outputs/csv/data_structure_stats.csv',
    'test_dup_stats.csv': 'outputs/csv/test_dup_stats.csv',
    'test_sorted_stats.csv': 'outputs/csv/test_sorted_stats.csv',
    'test_sorted_fixed.csv': 'outputs/csv/test_sorted_fixed.csv',
    
    # Output files - logs
    'evaluation_001.log': 'outputs/logs/evaluation_001.log',
    'parser_production.log': 'outputs/logs/parser_production.log',
    'perplexity_001.log': 'outputs/logs/perplexity_001.log',
    'perplexity_0001.log': 'outputs/logs/perplexity_0001.log',
    'molecule_analysis_001.log': 'outputs/logs/molecule_analysis_001.log',
    'streaming_test.log': 'outputs/logs/streaming_test.log',
    
    # Data files
    'source_file_mapping.json': 'data/source_file_mapping.json',
}

# Directories to create __init__.py files in
INIT_DIRS = [
    'scripts',
    'scripts/training',
    'scripts/inference',
    'scripts/data_processing',
    'scripts/analysis',
    'scripts/utils',
    'tests/integration',
]

def create_directories():
    """Create all necessary directories."""
    dirs_to_create = [
        'scripts/training',
        'scripts/inference',
        'scripts/data_processing',
        'scripts/analysis',
        'scripts/utils',
        'tests/integration',
        'outputs/reports',
        'outputs/csv',
        'outputs/logs',
        'data/raw',
        'data/processed',
        'docs',
    ]
    
    for dir_path in dirs_to_create:
        full_path = REPO_ROOT / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def create_init_files():
    """Create __init__.py files in script directories."""
    for dir_path in INIT_DIRS:
        init_file = REPO_ROOT / dir_path / '__init__.py'
        if not init_file.exists():
            init_file.touch()
            print(f"✓ Created {dir_path}/__init__.py")

def move_files():
    """Move files according to the reorganization mapping."""
    moved_count = 0
    skipped_count = 0
    
    for old_path, new_path in REORGANIZATION.items():
        old_full = REPO_ROOT / old_path
        new_full = REPO_ROOT / new_path
        
        if old_full.exists():
            # Ensure parent directory exists
            new_full.parent.mkdir(parents=True, exist_ok=True)
            
            # Move the file
            shutil.move(str(old_full), str(new_full))
            print(f"✓ Moved: {old_path} → {new_path}")
            moved_count += 1
        else:
            print(f"⚠ Skipped (not found): {old_path}")
            skipped_count += 1
    
    print(f"\nSummary: {moved_count} files moved, {skipped_count} files skipped")

def create_readme_in_outputs():
    """Create a README in outputs directory explaining the structure."""
    readme_content = """# Outputs Directory

This directory contains all analysis outputs, reports, and logs generated by the QUEST pipeline.

## Structure

- `reports/`: Text and analysis reports
- `csv/`: CSV data files with analysis results
- `logs/`: Log files from various pipeline runs

**Note**: This directory is excluded from version control. Add important results to your documentation or separate results repository.
"""
    
    readme_path = REPO_ROOT / 'outputs' / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print("✓ Created outputs/README.md")

def main():
    """Main reorganization routine."""
    print("=" * 60)
    print("QUEST Repository Reorganization")
    print("=" * 60)
    print()
    
    print("Step 1: Creating directory structure...")
    create_directories()
    print()
    
    print("Step 2: Creating __init__.py files...")
    create_init_files()
    print()
    
    print("Step 3: Moving files...")
    move_files()
    print()
    
    print("Step 4: Creating documentation...")
    create_readme_in_outputs()
    print()
    
    print("=" * 60)
    print("✓ Reorganization complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Review the changes")
    print("2. Update import statements (run update_imports.py)")
    print("3. Update .gitignore (run update_gitignore.py)")
    print("4. Test that scripts still work")
    print("5. Commit the changes")

if __name__ == '__main__':
    main()
