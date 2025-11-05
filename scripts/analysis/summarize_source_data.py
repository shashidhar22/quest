#!/usr/bin/env python3
"""
Quick summary of source data directory:
- Total records by category
- Total records by study
- Number of studies
- File counts
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json

def count_lines(file_path):
    """Count lines in a TSV/CSV file (fast)."""
    try:
        # Quick line count
        with open(file_path, 'rb') as f:
            # Skip header
            next(f)
            count = sum(1 for _ in f)
        return count
    except:
        return 0

def analyze_source_data(data_dir='/mnt/ephemeral/data'):
    """Analyze source data directory."""
    
    data_path = Path(data_dir)
    
    # Categories to scan
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
    
    print("=" * 80)
    print("SOURCE DATA SUMMARY")
    print("=" * 80)
    print()
    
    # Statistics
    category_stats = {}
    study_stats = {}
    all_files = []
    
    print("Scanning directories...")
    for category in tqdm(categories, desc="Categories"):
        category_path = data_path / category
        if not category_path.exists():
            continue
        
        category_files = 0
        category_records = 0
        
        # Get all study directories
        study_dirs = [d for d in category_path.iterdir() if d.is_dir()]
        
        for study_dir in study_dirs:
            study_id = study_dir.name
            study_files = 0
            study_records = 0
            
            # Find all TSV and CSV files recursively
            for file_path in study_dir.rglob('*.[tc]sv'):
                file_records = count_lines(file_path)
                
                study_files += 1
                study_records += file_records
                category_files += 1
                category_records += file_records
                
                all_files.append({
                    'category': category,
                    'study_id': study_id,
                    'file': file_path.name,
                    'records': file_records,
                    'path': str(file_path.relative_to(data_path))
                })
            
            if study_records > 0:
                if study_id not in study_stats:
                    study_stats[study_id] = {
                        'category': category,
                        'files': 0,
                        'records': 0
                    }
                study_stats[study_id]['files'] += study_files
                study_stats[study_id]['records'] += study_records
        
        if category_records > 0:
            category_stats[category] = {
                'files': category_files,
                'records': category_records,
                'studies': len([s for s in study_stats if study_stats[s]['category'] == category])
            }
    
    print()
    print("=" * 80)
    print("SUMMARY BY CATEGORY")
    print("=" * 80)
    print()
    print(f"{'Category':<30} {'Studies':>10} {'Files':>10} {'Records':>15}")
    print("-" * 80)
    
    total_files = 0
    total_records = 0
    total_studies = 0
    
    for cat in sorted(category_stats.keys()):
        stats = category_stats[cat]
        print(f"{cat:<30} {stats['studies']:>10,} {stats['files']:>10,} {stats['records']:>15,}")
        total_files += stats['files']
        total_records += stats['records']
        total_studies += stats['studies']
    
    print("-" * 80)
    print(f"{'TOTAL':<30} {total_studies:>10,} {total_files:>10,} {total_records:>15,}")
    
    print()
    print("=" * 80)
    print(f"TOP 30 STUDIES BY RECORD COUNT")
    print("=" * 80)
    print()
    print(f"{'Study ID':<20} {'Category':<25} {'Files':>8} {'Records':>15}")
    print("-" * 80)
    
    sorted_studies = sorted(study_stats.items(), key=lambda x: -x[1]['records'])[:30]
    for study_id, stats in sorted_studies:
        print(f"{study_id:<20} {stats['category']:<25} {stats['files']:>8,} {stats['records']:>15,}")
    
    print()
    print("=" * 80)
    print(f"STUDY DISTRIBUTION BY CATEGORY")
    print("=" * 80)
    print()
    
    for cat in sorted(category_stats.keys()):
        studies_in_cat = [s for s in study_stats if study_stats[s]['category'] == cat]
        print(f"\n{cat.upper()} ({len(studies_in_cat)} studies):")
        print("-" * 80)
        
        for study_id in sorted(studies_in_cat)[:10]:  # Show first 10
            stats = study_stats[study_id]
            print(f"  {study_id:<30} {stats['files']:>6,} files, {stats['records']:>12,} records")
        
        if len(studies_in_cat) > 10:
            print(f"  ... and {len(studies_in_cat) - 10} more studies")
    
    # Export detailed stats
    print()
    print("=" * 80)
    print("Exporting detailed statistics...")
    
    # Study-level CSV
    study_df = pd.DataFrame([
        {
            'study_id': study_id,
            'category': stats['category'],
            'files': stats['files'],
            'records': stats['records']
        }
        for study_id, stats in study_stats.items()
    ])
    study_df = study_df.sort_values('records', ascending=False)
    study_df.to_csv('source_data_by_study.csv', index=False)
    print(f"✓ Saved: source_data_by_study.csv ({len(study_df)} studies)")
    
    # Category-level CSV
    cat_df = pd.DataFrame([
        {
            'category': cat,
            'studies': stats['studies'],
            'files': stats['files'],
            'records': stats['records']
        }
        for cat, stats in category_stats.items()
    ])
    cat_df = cat_df.sort_values('records', ascending=False)
    cat_df.to_csv('source_data_by_category.csv', index=False)
    print(f"✓ Saved: source_data_by_category.csv ({len(cat_df)} categories)")
    
    # File-level CSV (detailed)
    file_df = pd.DataFrame(all_files)
    file_df = file_df.sort_values('records', ascending=False)
    file_df.to_csv('source_data_by_file.csv', index=False)
    print(f"✓ Saved: source_data_by_file.csv ({len(file_df)} files)")
    
    print()
    print("=" * 80)
    print("✓ Analysis Complete!")
    print("=" * 80)
    
    return {
        'total_studies': total_studies,
        'total_files': total_files,
        'total_records': total_records,
        'category_stats': category_stats,
        'study_stats': study_stats
    }

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Summarize source data directory")
    parser.add_argument('--data-dir', default='/mnt/ephemeral/data',
                       help='Source data directory (default: /mnt/ephemeral/data)')
    
    args = parser.parse_args()
    
    analyze_source_data(args.data_dir)
