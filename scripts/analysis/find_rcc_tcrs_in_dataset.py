#!/usr/bin/env python3
"""
Find RCC TCR sequences in parsed dataset.

This script:
1. Loads RCC TCR sequences from CSV
2. Uses TCR stitcher to generate full-length TRA and TRB sequences
3. Scans through the parsed output to find matches for:
   - CDR3 sequences (TRA, TRB)
   - Full-length sequences (TRA_full, TRB_full)
   - Paired sequences

Usage:
    python find_rcc_tcrs_in_dataset.py \
        --rcc-csv data/processed/rcc_tcrs.csv \
        --parsed-dir /mnt/ephemeral/parsed_output/seq \
        --output-report outputs/rcc_tcr_matches.txt
"""

import argparse
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm
import time
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from quest.parsers.tcr_stitcher import TCRStitcher

def load_rcc_sequences(csv_path):
    """Load RCC TCR sequences and generate full-length sequences using stitcher."""
    print(f"📂 Loading RCC sequences from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"✓ Loaded {len(df)} RCC TCR entries")
    print()
    
    # Initialize TCR stitcher
    print("🧬 Generating full-length TCR sequences using stitchr...")
    stitcher = TCRStitcher(species="HUMAN")
    
    # Prepare dataframe for stitcher
    # Map CSV columns to stitcher expected columns
    df_stitch = pd.DataFrame({
        'tra': df['TRA_cdr3'],
        'trav_gene': df['TRAV'],
        'traj_gene': df['TRAJ'],
        'trb': df['TRB_cdr3'],
        'trbv_gene': df['TRBV'],
        'trbj_gene': df['TRBJ']
    })
    
    # Generate full-length sequences
    df_stitched = stitcher.process_dataframe(df_stitch)
    
    # Count successes
    tra_full_count = df_stitched['tra_full'].notna().sum()
    trb_full_count = df_stitched['trb_full'].notna().sum()
    
    print(f"✓ Generated full-length sequences:")
    print(f"  - {tra_full_count}/{len(df)} TRA sequences ({tra_full_count/len(df)*100:.1f}%)")
    print(f"  - {trb_full_count}/{len(df)} TRB sequences ({trb_full_count/len(df)*100:.1f}%)")
    print()
    
    # Create sets for fast lookup
    tra_cdr3_sequences = set()
    trb_cdr3_sequences = set()
    tra_full_sequences = set()
    trb_full_sequences = set()
    paired_cdr3_sequences = set()
    paired_full_sequences = set()
    
    for _, row in df_stitched.iterrows():
        tra_cdr3 = row['tra']
        trb_cdr3 = row['trb']
        tra_full = row['tra_full']
        trb_full = row['trb_full']
        
        # CDR3 sequences
        if pd.notna(tra_cdr3) and tra_cdr3:
            tra_cdr3_sequences.add(tra_cdr3)
        if pd.notna(trb_cdr3) and trb_cdr3:
            trb_cdr3_sequences.add(trb_cdr3)
        
        # Full-length sequences
        if pd.notna(tra_full) and tra_full:
            tra_full_sequences.add(tra_full)
        if pd.notna(trb_full) and trb_full:
            trb_full_sequences.add(trb_full)
        
        # Paired CDR3 sequences
        if pd.notna(tra_cdr3) and pd.notna(trb_cdr3) and tra_cdr3 and trb_cdr3:
            paired_cdr3_sequences.add((tra_cdr3, trb_cdr3))
        
        # Paired full-length sequences
        if pd.notna(tra_full) and pd.notna(trb_full) and tra_full and trb_full:
            paired_full_sequences.add((tra_full, trb_full))
    
    print(f"✓ Created lookup sets:")
    print(f"  - {len(tra_cdr3_sequences)} unique TRA CDR3 sequences")
    print(f"  - {len(trb_cdr3_sequences)} unique TRB CDR3 sequences")
    print(f"  - {len(tra_full_sequences)} unique TRA full-length sequences")
    print(f"  - {len(trb_full_sequences)} unique TRB full-length sequences")
    print(f"  - {len(paired_cdr3_sequences)} unique paired CDR3 sequences")
    print(f"  - {len(paired_full_sequences)} unique paired full-length sequences")
    print()
    
    return {
        'tra_cdr3': tra_cdr3_sequences,
        'trb_cdr3': trb_cdr3_sequences,
        'tra_full': tra_full_sequences,
        'trb_full': trb_full_sequences,
        'paired_cdr3': paired_cdr3_sequences,
        'paired_full': paired_full_sequences,
        'df': df_stitched
    }

def scan_parquet_files(parsed_dir, rcc_data, sample_limit=None):
    """Scan through parquet files and find matches for both CDR3 and full-length sequences."""
    print(f"🔍 Scanning parquet files in {parsed_dir}...")
    
    # Get all parquet files
    parquet_files = list(Path(parsed_dir).glob("*.parquet"))
    
    if sample_limit:
        parquet_files = parquet_files[:sample_limit]
        print(f"⚠️  Sampling first {sample_limit} files for testing")
    
    print(f"📊 Found {len(parquet_files):,} parquet files to scan")
    print()
    
    # Track matches - use dict instead of defaultdict for faster lookups
    matches = {
        'tra_cdr3': {},  # tra_cdr3_seq -> set([file_paths])
        'trb_cdr3': {},  # trb_cdr3_seq -> set([file_paths])
        'tra_full': {},  # tra_full_seq -> set([file_paths])
        'trb_full': {},  # trb_full_seq -> set([file_paths])
        'paired_cdr3': {},  # (tra_cdr3, trb_cdr3) -> set([file_paths])
        'paired_full': {},  # (tra_full, trb_full) -> set([file_paths])
    }
    
    # Pre-initialize with empty sets for all RCC sequences
    for seq in rcc_data['tra_cdr3']:
        matches['tra_cdr3'][seq] = set()
    for seq in rcc_data['trb_cdr3']:
        matches['trb_cdr3'][seq] = set()
    for seq in rcc_data['tra_full']:
        matches['tra_full'][seq] = set()
    for seq in rcc_data['trb_full']:
        matches['trb_full'][seq] = set()
    for pair in rcc_data['paired_cdr3']:
        matches['paired_cdr3'][pair] = set()
    for pair in rcc_data['paired_full']:
        matches['paired_full'][pair] = set()
    
    # Track statistics
    stats = {
        'files_scanned': 0,
        'total_rows': 0,
        'tra_cdr3_matches': 0,
        'trb_cdr3_matches': 0,
        'tra_full_matches': 0,
        'trb_full_matches': 0,
        'paired_cdr3_matches': 0,
        'paired_full_matches': 0,
        'errors': 0
    }
    
    # Scan files with progress bar
    print("🔍 Scanning files for matches...")
    with tqdm(total=len(parquet_files), desc="Scanning files", unit="files") as pbar:
        for parquet_file in parquet_files:
            try:
                # Read parquet file - only read needed columns
                columns_to_read = []
                table = pq.read_table(parquet_file)
                available_cols = table.schema.names
                
                if 'tra' in available_cols:
                    columns_to_read.append('tra')
                if 'trb' in available_cols:
                    columns_to_read.append('trb')
                if 'tra_full' in available_cols:
                    columns_to_read.append('tra_full')
                if 'trb_full' in available_cols:
                    columns_to_read.append('trb_full')
                
                if not columns_to_read:
                    pbar.update(1)
                    continue
                
                # Read only needed columns
                table = pq.read_table(parquet_file, columns=columns_to_read)
                df = table.to_pandas()
                
                stats['files_scanned'] += 1
                stats['total_rows'] += len(df)
                
                filename = str(parquet_file.name)
                
                # Vectorized check for TRA CDR3 matches
                if 'tra' in df.columns:
                    tra_matches = df['tra'].isin(rcc_data['tra_cdr3'])
                    if tra_matches.any():
                        for seq in df.loc[tra_matches, 'tra'].unique():
                            if seq in matches['tra_cdr3']:
                                matches['tra_cdr3'][seq].add(filename)
                                stats['tra_cdr3_matches'] += tra_matches.sum()
                
                # Vectorized check for TRB CDR3 matches
                if 'trb' in df.columns:
                    trb_matches = df['trb'].isin(rcc_data['trb_cdr3'])
                    if trb_matches.any():
                        for seq in df.loc[trb_matches, 'trb'].unique():
                            if seq in matches['trb_cdr3']:
                                matches['trb_cdr3'][seq].add(filename)
                                stats['trb_cdr3_matches'] += trb_matches.sum()
                
                # Vectorized check for TRA full-length matches
                if 'tra_full' in df.columns:
                    tra_full_matches = df['tra_full'].isin(rcc_data['tra_full'])
                    if tra_full_matches.any():
                        for seq in df.loc[tra_full_matches, 'tra_full'].unique():
                            if seq in matches['tra_full']:
                                matches['tra_full'][seq].add(filename)
                                stats['tra_full_matches'] += tra_full_matches.sum()
                
                # Vectorized check for TRB full-length matches
                if 'trb_full' in df.columns:
                    trb_full_matches = df['trb_full'].isin(rcc_data['trb_full'])
                    if trb_full_matches.any():
                        for seq in df.loc[trb_full_matches, 'trb_full'].unique():
                            if seq in matches['trb_full']:
                                matches['trb_full'][seq].add(filename)
                                stats['trb_full_matches'] += trb_full_matches.sum()
                
                # Check paired CDR3 matches (optimize by only checking rows with both)
                if 'tra' in df.columns and 'trb' in df.columns:
                    paired_df = df[['tra', 'trb']].dropna()
                    if len(paired_df) > 0:
                        # Create tuples and check against set
                        pairs = list(zip(paired_df['tra'], paired_df['trb']))
                        for pair in set(pairs):  # Use set to get unique pairs
                            if pair in matches['paired_cdr3']:
                                matches['paired_cdr3'][pair].add(filename)
                                stats['paired_cdr3_matches'] += pairs.count(pair)
                
                # Check paired full-length matches
                if 'tra_full' in df.columns and 'trb_full' in df.columns:
                    paired_full_df = df[['tra_full', 'trb_full']].dropna()
                    if len(paired_full_df) > 0:
                        # Create tuples and check against set
                        pairs = list(zip(paired_full_df['tra_full'], paired_full_df['trb_full']))
                        for pair in set(pairs):  # Use set to get unique pairs
                            if pair in matches['paired_full']:
                                matches['paired_full'][pair].add(filename)
                                stats['paired_full_matches'] += pairs.count(pair)
                
                pbar.update(1)
                pbar.set_postfix({
                    'TRA_cdr3': stats['tra_cdr3_matches'],
                    'TRB_cdr3': stats['trb_cdr3_matches'],
                    'TRA_full': stats['tra_full_matches'],
                    'TRB_full': stats['trb_full_matches']
                }, refresh=False)
                
            except Exception as e:
                stats['errors'] += 1
                tqdm.write(f"⚠️  Error reading {parquet_file.name}: {e}")
                pbar.update(1)
    
    # Convert sets back to lists for compatibility with report generation
    for key in matches:
        for seq_key in matches[key]:
            matches[key][seq_key] = list(matches[key][seq_key])
    
    print()
    return matches, stats

def generate_csv_output(rcc_data, matches, output_path):
    """
    Generate CSV output with original data plus match information.
    
    Adds columns:
    - tra_full: Full-length TRA sequence
    - trb_full: Full-length TRB sequence
    - n_datasets_tra_cdr3: Number of unique files where TRA CDR3 was found
    - n_datasets_trb_cdr3: Number of unique files where TRB CDR3 was found
    - n_datasets_paired_cdr3: Number of unique files where paired CDR3 was found
    """
    print("\n📊 Generating CSV output with match information...")
    
    # Start with the stitched dataframe (has tra_full, trb_full columns)
    output_df = rcc_data['df'].copy()
    
    # Vectorized approach - create lookup dictionaries for fast mapping
    tra_count_map = {seq: len(files) for seq, files in matches['tra_cdr3'].items()}
    trb_count_map = {seq: len(files) for seq, files in matches['trb_cdr3'].items()}
    paired_count_map = {pair: len(files) for pair, files in matches['paired_cdr3'].items()}
    
    # Map counts using vectorized operations
    output_df['n_datasets_tra_cdr3'] = output_df['tra'].map(tra_count_map).fillna(0).astype(int)
    output_df['n_datasets_trb_cdr3'] = output_df['trb'].map(trb_count_map).fillna(0).astype(int)
    
    # For paired, need to create a temporary column with tuple
    output_df['_temp_pair'] = list(zip(output_df['tra'].fillna(''), output_df['trb'].fillna('')))
    output_df['n_datasets_paired_cdr3'] = output_df['_temp_pair'].map(paired_count_map).fillna(0).astype(int)
    output_df.drop(columns=['_temp_pair'], inplace=True)
    
    # Reorder columns to put new columns at the end but in logical order
    # Get original columns from input
    original_cols = ['clonotype_type', 'clonotype_id', 'TRB_cdr3', 'TRA_cdr3', 
                     'peptide_target', 'patient_id', 'TRAV', 'TRAJ', 'TRBV', 'TRBJ']
    
    # Add stitched/formatted gene columns if they exist
    gene_cols = ['trav_gene', 'traj_gene', 'trbv_gene', 'trbj_gene']
    existing_gene_cols = [col for col in gene_cols if col in output_df.columns]
    
    # New columns in desired order
    new_cols = ['tra_full', 'trb_full', 
                'n_datasets_tra_cdr3', 'n_datasets_trb_cdr3', 'n_datasets_paired_cdr3']
    
    # Build final column order
    final_cols = original_cols + existing_gene_cols + new_cols
    
    # Only include columns that exist
    final_cols = [col for col in final_cols if col in output_df.columns]
    
    # Add any remaining columns not in our list
    remaining_cols = [col for col in output_df.columns if col not in final_cols]
    final_cols.extend(remaining_cols)
    
    output_df = output_df[final_cols]
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    
    print(f"✓ CSV output saved to: {output_path}")
    print(f"  - {len(output_df)} rows")
    print(f"  - {len(output_df.columns)} columns")
    print(f"  - Columns: {', '.join(final_cols)}")
    print()
    
    # Print summary statistics
    print("📈 Match Statistics:")
    print(f"  TRA CDR3 found in dataset:    {(output_df['n_datasets_tra_cdr3'] > 0).sum()}/{len(output_df)} sequences")
    print(f"  TRB CDR3 found in dataset:    {(output_df['n_datasets_trb_cdr3'] > 0).sum()}/{len(output_df)} sequences")
    print(f"  Paired CDR3 found in dataset: {(output_df['n_datasets_paired_cdr3'] > 0).sum()}/{len(output_df)} sequences")
    print(f"  TRA full-length generated:    {output_df['tra_full'].notna().sum()}/{len(output_df)} sequences")
    print(f"  TRB full-length generated:    {output_df['trb_full'].notna().sum()}/{len(output_df)} sequences")
    print()

def generate_report(rcc_data, matches, stats, output_path):
    """Generate detailed report of findings."""
    report_lines = []
    
    report_lines.append("="*80)
    report_lines.append("RCC TCR SEQUENCES - DATASET MATCH REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Scanning statistics
    report_lines.append("📊 SCANNING STATISTICS")
    report_lines.append("-"*80)
    report_lines.append(f"Files scanned:     {stats['files_scanned']:>15,}")
    report_lines.append(f"Total rows:        {stats['total_rows']:>15,}")
    report_lines.append(f"Errors:            {stats['errors']:>15,}")
    report_lines.append("")
    
    # TRA CDR3 matches
    report_lines.append("🔵 TRA CDR3 SEQUENCE MATCHES")
    report_lines.append("-"*80)
    report_lines.append(f"RCC TRA CDR3 sequences: {len(rcc_data['tra_cdr3']):>15}")
    report_lines.append(f"Found in dataset:       {len(matches['tra_cdr3']):>15}")
    if len(rcc_data['tra_cdr3']) > 0:
        report_lines.append(f"Match rate:             {len(matches['tra_cdr3'])/len(rcc_data['tra_cdr3'])*100:>14.1f}%")
    report_lines.append(f"Total occurrences:      {stats['tra_cdr3_matches']:>15,}")
    report_lines.append("")
    
    if matches['tra_cdr3']:
        report_lines.append("Found TRA CDR3 sequences:")
        for tra_seq, files in sorted(matches['tra_cdr3'].items(), key=lambda x: len(x[1]), reverse=True):
            report_lines.append(f"  {tra_seq}")
            report_lines.append(f"    → Found in {len(files)} file(s): {', '.join(files[:5])}")
            if len(files) > 5:
                report_lines.append(f"      ... and {len(files)-5} more files")
        report_lines.append("")
    
    # TRB CDR3 matches
    report_lines.append("🟢 TRB CDR3 SEQUENCE MATCHES")
    report_lines.append("-"*80)
    report_lines.append(f"RCC TRB CDR3 sequences: {len(rcc_data['trb_cdr3']):>15}")
    report_lines.append(f"Found in dataset:       {len(matches['trb_cdr3']):>15}")
    if len(rcc_data['trb_cdr3']) > 0:
        report_lines.append(f"Match rate:             {len(matches['trb_cdr3'])/len(rcc_data['trb_cdr3'])*100:>14.1f}%")
    report_lines.append(f"Total occurrences:      {stats['trb_cdr3_matches']:>15,}")
    report_lines.append("")
    
    if matches['trb_cdr3']:
        report_lines.append("Found TRB CDR3 sequences:")
        for trb_seq, files in sorted(matches['trb_cdr3'].items(), key=lambda x: len(x[1]), reverse=True):
            report_lines.append(f"  {trb_seq}")
            report_lines.append(f"    → Found in {len(files)} file(s): {', '.join(files[:5])}")
            if len(files) > 5:
                report_lines.append(f"      ... and {len(files)-5} more files")
        report_lines.append("")
    
    # TRA full-length matches
    report_lines.append("🔷 TRA FULL-LENGTH SEQUENCE MATCHES")
    report_lines.append("-"*80)
    report_lines.append(f"RCC TRA full sequences: {len(rcc_data['tra_full']):>15}")
    report_lines.append(f"Found in dataset:       {len(matches['tra_full']):>15}")
    if len(rcc_data['tra_full']) > 0:
        report_lines.append(f"Match rate:             {len(matches['tra_full'])/len(rcc_data['tra_full'])*100:>14.1f}%")
    report_lines.append(f"Total occurrences:      {stats['tra_full_matches']:>15,}")
    report_lines.append("")
    
    if matches['tra_full']:
        report_lines.append("Found TRA full-length sequences:")
        for tra_seq, files in sorted(matches['tra_full'].items(), key=lambda x: len(x[1]), reverse=True):
            report_lines.append(f"  {tra_seq[:60]}... (length: {len(tra_seq)} aa)")
            report_lines.append(f"    → Found in {len(files)} file(s): {', '.join(files[:5])}")
            if len(files) > 5:
                report_lines.append(f"      ... and {len(files)-5} more files")
        report_lines.append("")
    
    # TRB full-length matches
    report_lines.append("🟩 TRB FULL-LENGTH SEQUENCE MATCHES")
    report_lines.append("-"*80)
    report_lines.append(f"RCC TRB full sequences: {len(rcc_data['trb_full']):>15}")
    report_lines.append(f"Found in dataset:       {len(matches['trb_full']):>15}")
    if len(rcc_data['trb_full']) > 0:
        report_lines.append(f"Match rate:             {len(matches['trb_full'])/len(rcc_data['trb_full'])*100:>14.1f}%")
    report_lines.append(f"Total occurrences:      {stats['trb_full_matches']:>15,}")
    report_lines.append("")
    
    if matches['trb_full']:
        report_lines.append("Found TRB full-length sequences:")
        for trb_seq, files in sorted(matches['trb_full'].items(), key=lambda x: len(x[1]), reverse=True):
            report_lines.append(f"  {trb_seq[:60]}... (length: {len(trb_seq)} aa)")
            report_lines.append(f"    → Found in {len(files)} file(s): {', '.join(files[:5])}")
            if len(files) > 5:
                report_lines.append(f"      ... and {len(files)-5} more files")
        report_lines.append("")
    
    # Paired CDR3 matches
    report_lines.append("🟣 PAIRED CDR3 MATCHES")
    report_lines.append("-"*80)
    report_lines.append(f"RCC paired CDR3 TCRs:   {len(rcc_data['paired_cdr3']):>15}")
    report_lines.append(f"Found in dataset:       {len(matches['paired_cdr3']):>15}")
    if len(rcc_data['paired_cdr3']) > 0:
        report_lines.append(f"Match rate:             {len(matches['paired_cdr3'])/len(rcc_data['paired_cdr3'])*100:>14.1f}%")
    report_lines.append(f"Total occurrences:      {stats['paired_cdr3_matches']:>15,}")
    report_lines.append("")
    
    if matches['paired_cdr3']:
        report_lines.append("Found paired CDR3 sequences:")
        for (tra, trb), files in sorted(matches['paired_cdr3'].items(), key=lambda x: len(x[1]), reverse=True):
            report_lines.append(f"  TRA: {tra}")
            report_lines.append(f"  TRB: {trb}")
            report_lines.append(f"    → Found in {len(files)} file(s): {', '.join(files[:5])}")
            if len(files) > 5:
                report_lines.append(f"      ... and {len(files)-5} more files")
        report_lines.append("")
    
    # Paired full-length matches
    report_lines.append("🟪 PAIRED FULL-LENGTH MATCHES")
    report_lines.append("-"*80)
    report_lines.append(f"RCC paired full TCRs:   {len(rcc_data['paired_full']):>15}")
    report_lines.append(f"Found in dataset:       {len(matches['paired_full']):>15}")
    if len(rcc_data['paired_full']) > 0:
        report_lines.append(f"Match rate:             {len(matches['paired_full'])/len(rcc_data['paired_full'])*100:>14.1f}%")
    report_lines.append(f"Total occurrences:      {stats['paired_full_matches']:>15,}")
    report_lines.append("")
    
    if matches['paired_full']:
        report_lines.append("Found paired full-length sequences:")
        for (tra, trb), files in sorted(matches['paired_full'].items(), key=lambda x: len(x[1]), reverse=True):
            report_lines.append(f"  TRA: {tra[:60]}... (length: {len(tra)} aa)")
            report_lines.append(f"  TRB: {trb[:60]}... (length: {len(trb)} aa)")
            report_lines.append(f"    → Found in {len(files)} file(s): {', '.join(files[:5])}")
            if len(files) > 5:
                report_lines.append(f"      ... and {len(files)-5} more files")
        report_lines.append("")
    
    # Missing sequences
    report_lines.append("❌ SEQUENCES NOT FOUND")
    report_lines.append("-"*80)
    
    missing_tra_cdr3 = rcc_data['tra_cdr3'] - set(matches['tra_cdr3'].keys())
    if missing_tra_cdr3:
        report_lines.append(f"\nMissing TRA CDR3 sequences ({len(missing_tra_cdr3)}):")
        for seq in sorted(missing_tra_cdr3):
            report_lines.append(f"  {seq}")
    
    missing_trb_cdr3 = rcc_data['trb_cdr3'] - set(matches['trb_cdr3'].keys())
    if missing_trb_cdr3:
        report_lines.append(f"\nMissing TRB CDR3 sequences ({len(missing_trb_cdr3)}):")
        for seq in sorted(missing_trb_cdr3):
            report_lines.append(f"  {seq}")
    
    missing_tra_full = rcc_data['tra_full'] - set(matches['tra_full'].keys())
    if missing_tra_full:
        report_lines.append(f"\nMissing TRA full-length sequences ({len(missing_tra_full)}):")
        for seq in sorted(missing_tra_full):
            report_lines.append(f"  {seq[:60]}... (length: {len(seq)} aa)")
    
    missing_trb_full = rcc_data['trb_full'] - set(matches['trb_full'].keys())
    if missing_trb_full:
        report_lines.append(f"\nMissing TRB full-length sequences ({len(missing_trb_full)}):")
        for seq in sorted(missing_trb_full):
            report_lines.append(f"  {seq[:60]}... (length: {len(seq)} aa)")
    
    missing_paired_cdr3 = rcc_data['paired_cdr3'] - set(matches['paired_cdr3'].keys())
    if missing_paired_cdr3:
        report_lines.append(f"\nMissing paired CDR3 sequences ({len(missing_paired_cdr3)}):")
        for tra, trb in sorted(missing_paired_cdr3):
            report_lines.append(f"  TRA: {tra}")
            report_lines.append(f"  TRB: {trb}")
    
    missing_paired_full = rcc_data['paired_full'] - set(matches['paired_full'].keys())
    if missing_paired_full:
        report_lines.append(f"\nMissing paired full-length sequences ({len(missing_paired_full)}):")
        for tra, trb in sorted(missing_paired_full):
            report_lines.append(f"  TRA: {tra[:60]}... (length: {len(tra)} aa)")
            report_lines.append(f"  TRB: {trb[:60]}... (length: {len(trb)} aa)")
    
    report_lines.append("")
    report_lines.append("="*80)
    
    # Write report
    report_text = "\n".join(report_lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"✓ Report saved to: {output_path}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if len(rcc_data['tra_cdr3']) > 0:
        print(f"TRA CDR3:         {len(matches['tra_cdr3'])}/{len(rcc_data['tra_cdr3'])} found ({len(matches['tra_cdr3'])/len(rcc_data['tra_cdr3'])*100:.1f}%)")
    if len(rcc_data['trb_cdr3']) > 0:
        print(f"TRB CDR3:         {len(matches['trb_cdr3'])}/{len(rcc_data['trb_cdr3'])} found ({len(matches['trb_cdr3'])/len(rcc_data['trb_cdr3'])*100:.1f}%)")
    if len(rcc_data['tra_full']) > 0:
        print(f"TRA full-length:  {len(matches['tra_full'])}/{len(rcc_data['tra_full'])} found ({len(matches['tra_full'])/len(rcc_data['tra_full'])*100:.1f}%)")
    if len(rcc_data['trb_full']) > 0:
        print(f"TRB full-length:  {len(matches['trb_full'])}/{len(rcc_data['trb_full'])} found ({len(matches['trb_full'])/len(rcc_data['trb_full'])*100:.1f}%)")
    if len(rcc_data['paired_cdr3']) > 0:
        print(f"Paired CDR3:      {len(matches['paired_cdr3'])}/{len(rcc_data['paired_cdr3'])} found ({len(matches['paired_cdr3'])/len(rcc_data['paired_cdr3'])*100:.1f}%)")
    if len(rcc_data['paired_full']) > 0:
        print(f"Paired full:      {len(matches['paired_full'])}/{len(rcc_data['paired_full'])} found ({len(matches['paired_full'])/len(rcc_data['paired_full'])*100:.1f}%)")
    print("="*80)
    
    return report_text

def main():
    parser = argparse.ArgumentParser(description="Find RCC TCR sequences in parsed dataset")
    parser.add_argument("--rcc-csv", required=True, help="Path to RCC TCR CSV file")
    parser.add_argument("--parsed-dir", required=True, help="Directory containing parsed parquet files")
    parser.add_argument("--output-report", default=None, help="Path to save text report (optional)")
    parser.add_argument("--output-csv", default=None, help="Path to save CSV with matches (optional)")
    parser.add_argument("--sample", type=int, default=None, help="Sample N files for testing")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("RCC TCR SEQUENCE FINDER")
    print("="*80)
    print()
    
    start_time = time.time()
    
    # Load RCC sequences
    rcc_data = load_rcc_sequences(args.rcc_csv)
    
    # Scan dataset
    matches, stats = scan_parquet_files(args.parsed_dir, rcc_data, args.sample)
    
    # Generate CSV output with match information
    if args.output_csv:
        generate_csv_output(rcc_data, matches, args.output_csv)
    
    # Generate text report
    report = generate_report(rcc_data, matches, stats, args.output_report)
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print()

if __name__ == "__main__":
    main()
