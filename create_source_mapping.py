#!/usr/bin/env python3
"""
Create a mapping file that maps parsed output filenames back to their source metadata.

This script scans the source data directory structure and creates a JSON mapping
that includes:
- category (e.g., viral_studies, acute_illness, etc.)
- study_id (e.g., GSE123456, PRJNA123456, etc.)

The mapping is used by the duplication analyzer to provide detailed breakdowns
of the "other" category by study and category.
"""

import os
import json
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def create_mapping(data_dir='/mnt/ephemeral/data', output_file='source_file_mapping.json'):
    """
    Scan the data directory and create a mapping of output filenames to source metadata.
    
    Directory structure expected:
    /mnt/ephemeral/data/{category}/{study_id}/{molecule_type}/{file_type}/file.tsv
    
    Example:
    /mnt/ephemeral/data/viral_studies/GSE165080/tcr/contigs/sample.csv
    -> category: viral_studies
    -> study_id: GSE165080
    """
    
    data_path = Path(data_dir)
    mapping = {}
    
    # Categories to scan (skip databases, test_output)
    categories = [
        'acute_illness',
        'adaptive_bulk',
        'autoimmune_studies',
        'cancer_studies',
        'chronic_illness',
        'healthy_studies',
        'multiple_classifications',
        'unclassified_data',
        'viral_studies'
    ]
    
    logger.info(f"Scanning {data_dir} for source files...")
    logger.info(f"Categories: {', '.join(categories)}")
    logger.info("")
    
    total_files = 0
    
    for category in tqdm(categories, desc="Categories"):
        category_path = data_path / category
        if not category_path.exists():
            continue
        
        # Get all study directories
        study_dirs = [d for d in category_path.iterdir() if d.is_dir()]
        
        for study_dir in study_dirs:
            study_id = study_dir.name
            
            # Find all TSV and CSV files recursively
            for file_path in study_dir.rglob('*.[tc]sv'):
                # The output filename is the stem of the source file
                # with _mri.parquet or _seq.parquet appended
                file_stem = file_path.stem
                
                # Store mapping (without the _mri or _seq suffix)
                mapping[file_stem] = {
                    'category': category,
                    'study_id': study_id,
                    'source_path': str(file_path.relative_to(data_path))
                }
                total_files += 1
    
    logger.info(f"\n✓ Mapped {total_files:,} source files")
    logger.info(f"  - From {len(categories)} categories")
    logger.info(f"  - Across {len(set(m['study_id'] for m in mapping.values()))} studies")
    
    # Write to JSON
    with open(output_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    logger.info(f"\n✓ Mapping saved to {output_file}")
    
    # Print summary statistics
    category_counts = {}
    for entry in mapping.values():
        cat = entry['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    logger.info("\nFiles per category:")
    for cat in sorted(category_counts.keys()):
        logger.info(f"  {cat:30s}: {category_counts[cat]:>6,} files")
    
    return mapping


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create source file mapping for duplication analysis"
    )
    parser.add_argument('--data-dir', default='/mnt/ephemeral/data',
                       help='Source data directory (default: /mnt/ephemeral/data)')
    parser.add_argument('--output', default='source_file_mapping.json',
                       help='Output JSON file (default: source_file_mapping.json)')
    
    args = parser.parse_args()
    
    create_mapping(args.data_dir, args.output)


if __name__ == '__main__':
    main()
