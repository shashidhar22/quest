#!/usr/bin/env python3
"""
Parallel duplication analyzer optimized for high-memory systems.

Strategy:
1. Parallel extraction: Process multiple fields simultaneously
2. Memory-efficient extraction: Use vectorized pandas operations
3. Parallel sorting: Sort multiple fields at once
4. Fast streaming count: Same O(1) memory approach

With 500GB RAM available, we can:
- Run 4-8 fields in parallel
- Use larger sort buffers (50GB per sort)
- Keep file stats in memory

Expected speedup: 4-8x faster than sequential version
"""

import pyarrow.parquet as pq
from pathlib import Path
from collections import defaultdict
import logging
import subprocess
from tqdm import tqdm
import os
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ParallelStreamAnalyzer:
    """Parallel analyzer optimized for high-memory systems."""
    
    def __init__(self, parsed_dir: str, temp_dir: str = '/mnt/ephemeral', n_parallel_fields: int = 6, use_seq: bool = False, mapping_file: str = None):
        self.parsed_dir = Path(parsed_dir)
        self.data_dir = self.parsed_dir / ("seq" if use_seq else "mri")
        self.use_seq = use_seq
        self.temp_dir = Path(temp_dir) / "dup_parallel_temp"
        self.temp_dir.mkdir(exist_ok=True)
        self.n_parallel_fields = n_parallel_fields
        
        # Load source file mapping
        self.source_mapping = {}
        if mapping_file and os.path.exists(mapping_file):
            import json
            with open(mapping_file, 'r') as f:
                self.source_mapping = json.load(f)
            logger.info(f"Loaded mapping for {len(self.source_mapping)} source files")
        else:
            logger.warning("No source mapping file provided - will use basic database detection")
        
        # Set Unix sort to use ephemeral storage
        os.environ['TMPDIR'] = temp_dir
        
        logger.info(f"Using data directory: {self.data_dir} ({'seq' if use_seq else 'mri'})")
        logger.info(f"Using temp directory: {self.temp_dir}")
        logger.info(f"Unix sort TMPDIR: {temp_dir}")
        logger.info(f"Parallel fields: {n_parallel_fields}")
        
        # File-level stats
        self.file_stats = []
        
        # Comprehensive field definitions - all possible fields across all databases
        # We'll check which ones exist in each file dynamically
        self.single_fields = [
            'tra', 'trb', 'peptide', 'mhc_one', 'mhc_two',
            'trav_gene', 'trbv_gene',  # Keep these for combinations
        ]
        
        # Exact combinations as specified - 27 fields total
        # Using OrderedDict to preserve order for reporting
        from collections import OrderedDict
        self.combo_fields = OrderedDict([
            # 2. TCR combinations
            ('tra_trav', ['tra', 'trav_gene']),
            ('trb_trbv', ['trb', 'trbv_gene']),
            ('tra_trb', ['tra', 'trb']),
            ('tra_trav_trb_trbv', ['tra', 'trav_gene', 'trb', 'trbv_gene']),
            
            # 10. MHC combinations
            ('mhc_one_mhc_two', ['mhc_one', 'mhc_two']),
            
            # 11-13. Peptide + MHC combinations
            ('peptide_mhc_one', ['peptide', 'mhc_one']),
            ('peptide_mhc_two', ['peptide', 'mhc_two']),
            ('peptide_mhc_one_mhc_two', ['peptide', 'mhc_one', 'mhc_two']),
            
            # 14-17. TRA + peptide + MHC
            ('tra_peptide_mhc_one', ['tra', 'peptide', 'mhc_one']),
            ('tra_trav_peptide_mhc_one', ['tra', 'trav_gene', 'peptide', 'mhc_one']),
            ('trb_peptide_mhc_one', ['trb', 'peptide', 'mhc_one']),
            ('trb_trbv_peptide_mhc_one', ['trb', 'trbv_gene', 'peptide', 'mhc_one']),
            
            # 18-21. TRA/TRB + peptide + mhc_two
            ('tra_peptide_mhc_two', ['tra', 'peptide', 'mhc_two']),
            ('tra_trav_peptide_mhc_two', ['tra', 'trav_gene', 'peptide', 'mhc_two']),
            ('trb_peptide_mhc_two', ['trb', 'peptide', 'mhc_two']),
            ('trb_trbv_peptide_mhc_two', ['trb', 'trbv_gene', 'peptide', 'mhc_two']),
            
            # 22-27. Full TCR pair + peptide + MHC combinations
            ('tra_trb_peptide_mhc_one', ['tra', 'trb', 'peptide', 'mhc_one']),
            ('tra_trav_trb_trbv_peptide_mhc_one', ['tra', 'trav_gene', 'trb', 'trbv_gene', 'peptide', 'mhc_one']),
            ('tra_trb_peptide_mhc_two', ['tra', 'trb', 'peptide', 'mhc_two']),
            ('tra_trav_trb_trbv_peptide_mhc_two', ['tra', 'trav_gene', 'trb', 'trbv_gene', 'peptide', 'mhc_two']),
            ('tra_trb_peptide_mhc_one_mhc_two', ['tra', 'trb', 'peptide', 'mhc_one', 'mhc_two']),
            ('tra_trav_trb_trbv_peptide_mhc_one_mhc_two', ['tra', 'trav_gene', 'trb', 'trbv_gene', 'peptide', 'mhc_one', 'mhc_two']),
        ])
    
    def process_file_stats_fast(self, data_file: Path):
        """Ultra-fast file-level statistics - just metadata and basic row counts.
        Skip detailed per-file duplication analysis to speed up with 26k+ files."""
        try:
            # Read metadata only - no data loading
            parquet_file = pq.ParquetFile(data_file)
            total_rows = parquet_file.metadata.num_rows
            
            if total_rows == 0:
                return None
            
            # Determine folder based on file name
            file_name = data_file.stem.replace('_mri', '').replace('_seq', '')
            
            # Check if it's a database file
            if 'vdjdb' in file_name.lower():
                folder = 'vdjdb'
                category = 'database'
                study_id = 'vdjdb'
            elif 'mcpas' in file_name.lower():
                folder = 'mcpas'
                category = 'database'
                study_id = 'mcpas'
            elif 'tcrdb' in file_name.lower():
                folder = 'tcrdb'
                category = 'database'
                study_id = 'tcrdb'
            elif 'iedb' in file_name.lower():
                folder = 'iedb'
                category = 'database'
                study_id = 'iedb'
            elif 'cedar' in file_name.lower():
                folder = 'cedar'
                category = 'database'
                study_id = 'cedar'
            elif 'ireceptor' in file_name.lower():
                folder = 'ireceptor'
                category = 'database'
                study_id = 'ireceptor'
            else:
                folder = 'other'
                # Try to get category and study_id from mapping
                if file_name in self.source_mapping:
                    category = self.source_mapping[file_name]['category']
                    study_id = self.source_mapping[file_name]['study_id']
                else:
                    category = 'unknown'
                    study_id = 'unknown'
            
            file_stats = {
                'file': file_name,
                'folder': folder,
                'category': category,
                'study_id': study_id,
                'total_rows': total_rows,
                'available_columns': ','.join(parquet_file.schema.names)
            }
            
            return file_stats
            
        except Exception as e:
            return None
    
    def process_file_stats(self, data_file: Path):
        """Compute file-level statistics - handles both MRI and SEQ files with different schemas.
        Optimized to process in chunks to avoid loading huge files into memory."""
        try:
            # Read metadata first to get row count without loading data
            parquet_file = pq.ParquetFile(data_file)
            total_rows = parquet_file.metadata.num_rows
            
            if total_rows == 0:
                return None
            
            # Determine folder based on file name
            file_name = data_file.stem
            if 'vdjdb' in file_name.lower():
                folder = 'vdjdb'
            elif 'mcpas' in file_name.lower():
                folder = 'mcpas'
            elif 'tcrdb' in file_name.lower():
                folder = 'tcrdb'
            elif 'iedb' in file_name.lower():
                folder = 'iedb'
            elif 'cedar' in file_name.lower():
                folder = 'cedar'
            elif 'ireceptor' in file_name.lower():
                folder = 'ireceptor'
            else:
                folder = 'other'
            
            file_stats = {
                'file': file_name,
                'folder': folder,
                'total_rows': total_rows,
                'available_columns': ','.join(parquet_file.schema.names)
            }
            
            # For large files, use sampling instead of full scan for stats
            # This is much faster and gives good estimates
            if total_rows > 100_000:
                # Sample 100k rows for statistics (much faster)
                sample_size = min(100_000, total_rows)
                table = pq.read_table(data_file)
                df = table.to_pandas()
                
                # Random sample for better representation
                if len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
                
                sample_scale = total_rows / len(df)
            else:
                # For smaller files, read all data
                table = pq.read_table(data_file)
                df = table.to_pandas()
                sample_scale = 1.0
            
            # Single field stats - only process fields that exist
            for field in self.single_fields:
                if field in df.columns:
                    col = df[field]
                    non_empty = col[(col.notna()) & (col != '')]
                    
                    if len(non_empty) > 0:
                        unique_count = non_empty.nunique()
                        total = len(non_empty)
                        
                        # Scale estimates for sampled data
                        if sample_scale > 1:
                            unique_count = int(unique_count * sample_scale)
                            total = int(total * sample_scale)
                        
                        dup_rate = 1.0 - (unique_count / total) if total > 0 else 0
                        
                        file_stats[f'{field}_total'] = total
                        file_stats[f'{field}_unique'] = unique_count
                        file_stats[f'{field}_dup_rate'] = dup_rate * 100
            
            # Combination stats - only process combinations where all fields exist
            for combo_name, cols in self.combo_fields.items():
                if all(col in df.columns for col in cols):
                    combo_df = df[cols]
                    
                    # Filter: all columns must be non-null and non-empty
                    mask = (combo_df != '').all(axis=1) & combo_df.notna().all(axis=1)
                    filtered = combo_df[mask]
                    
                    if len(filtered) > 0:
                        unique_count = len(filtered.drop_duplicates())
                        total = len(filtered)
                        
                        # Scale estimates for sampled data
                        if sample_scale > 1:
                            unique_count = int(unique_count * sample_scale)
                            total = int(total * sample_scale)
                        
                        dup_rate = 1.0 - (unique_count / total) if total > 0 else 0
                        
                        file_stats[f'{combo_name}_total'] = total
                        file_stats[f'{combo_name}_unique'] = unique_count
                        file_stats[f'{combo_name}_dup_rate'] = dup_rate * 100
            
            # Explicitly delete dataframe to free memory
            del df
            if 'table' in locals():
                del table
            
            return file_stats
            
        except Exception as e:
            logger.warning(f"Failed to process {data_file}: {e}")
            return None
    
    def extract_field_to_file(self, data_files: list, field_name: str, columns: list, output_file: Path):
        """Extract field values to text file. Format: folder\tvalue1\tvalue2\t..."""
        with open(output_file, 'w') as out:
            for data_file in tqdm(data_files, desc=f"Extracting {field_name}", unit="file", position=0, leave=True):
                try:
                    # First check if columns exist in this file
                    table = pq.read_table(data_file)
                    if not all(col in table.column_names for col in columns):
                        continue
                    
                    # Now read only the columns we need
                    table = pq.read_table(data_file, columns=columns)
                    df = table.to_pandas()
                    
                    # Determine folder from filename
                    # Database files are named like "vdjdb_mri.parquet", "mcpas_mri.parquet", etc.
                    # Study files are named like "GSE123456_sample_mri.parquet"
                    file_name = data_file.stem
                    
                    # Extract database/folder name
                    if 'vdjdb' in file_name.lower():
                        folder = 'vdjdb'
                    elif 'mcpas' in file_name.lower():
                        folder = 'mcpas'
                    elif 'tcrdb' in file_name.lower():
                        folder = 'tcrdb'
                    elif 'iedb' in file_name.lower():
                        folder = 'iedb'
                    elif 'cedar' in file_name.lower():
                        folder = 'cedar'
                    elif 'ireceptor' in file_name.lower():
                        folder = 'ireceptor'
                    else:
                        folder = 'other'
                    
                    # Filter: all columns must be non-null and non-empty
                    mask = (df != '').all(axis=1) & df.notna().all(axis=1)
                    filtered = df[mask]
                    
                    if len(filtered) == 0:
                        continue
                    
                    # Add folder column and write in bulk (MUCH faster than iterrows!)
                    filtered.insert(0, 'folder', folder)
                    filtered.to_csv(out, sep='\t', header=False, index=False, mode='a')
                
                except Exception as e:
                    continue
        
        return output_file
    
    def sort_file(self, input_file: Path, field_name: str) -> Path:
        """Sort file using Unix sort (disk-based) with larger buffer for speed."""
        sorted_file = input_file.with_suffix('.sorted')
        
        logger.info(f"  [{field_name}] Sorting ({input_file.stat().st_size / (1024**3):.2f} GB)...")
        
        # Use Unix sort with 50GB buffer (we have 500GB available, run 6 in parallel = ~300GB for sort)
        cmd = [
            'sort',
            '-S', '50G',  # 50GB buffer per sort
            '--parallel=8',  # 8 threads per sort (6 sorts * 8 threads = 48 cores used)
            '-o', str(sorted_file),
            str(input_file)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        logger.info(f"  [{field_name}] ✓ Sorted")
        
        # Delete unsorted to save space
        input_file.unlink()
        
        return sorted_file
    
    def count_unique_streaming(self, sorted_file: Path, field_name: str) -> tuple[int, dict]:
        """
        Stream through sorted file and count unique values.
        Returns: (dataset_unique_count, {folder: unique_count})
        """
        dataset_unique = 0
        folder_unique = defaultdict(int)
        
        prev_line = None
        prev_folder = None
        prev_value = None
        
        logger.info(f"  [{field_name}] Counting unique values...")
        
        with open(sorted_file, 'r') as f:
            for line in f:
                line = line.rstrip('\n')
                
                # Parse line: folder\tvalue1\tvalue2\t...
                parts = line.split('\t', 1)  # Split only on first tab (faster)
                folder = parts[0]
                value = parts[1] if len(parts) > 1 else ''
                
                # Count unique per folder
                if folder != prev_folder or value != prev_value:
                    if folder != prev_folder:
                        prev_folder = folder
                        prev_value = None
                    
                    if value != prev_value:
                        folder_unique[folder] += 1
                        prev_value = value
                
                # Count unique across dataset
                if line != prev_line:
                    dataset_unique += 1
                    prev_line = line
        
        logger.info(f"  [{field_name}] ✓ Found {dataset_unique:,} unique values")
        
        # Cleanup sorted file
        sorted_file.unlink()
        
        return dataset_unique, dict(folder_unique)
    
    def process_field_worker(self, args):
        """Worker function for parallel field processing."""
        field_name, columns, data_files, temp_dir = args
        
        logger.info(f"[{field_name}] Starting extraction...")
        
        # Extract
        extract_file = Path(temp_dir) / f"{field_name}.txt"
        
        # Extraction with proper file writing
        with open(extract_file, 'w') as out:
            for data_file in tqdm(data_files, desc=f"Extracting {field_name}", unit="file", position=0, leave=True):
                try:
                    # First check if columns exist
                    table = pq.read_table(data_file)
                    if not all(col in table.column_names for col in columns):
                        continue
                    
                    # Read only needed columns
                    table = pq.read_table(data_file, columns=columns)
                    df = table.to_pandas()
                    
                    # Determine folder from filename
                    # Database files are named like "vdjdb_mri.parquet", "mcpas_mri.parquet", etc.
                    # Study files are named like "GSE123456_sample_mri.parquet"
                    file_name = data_file.stem
                    
                    # Extract database/folder name
                    if 'vdjdb' in file_name.lower():
                        folder = 'vdjdb'
                    elif 'mcpas' in file_name.lower():
                        folder = 'mcpas'
                    elif 'tcrdb' in file_name.lower():
                        folder = 'tcrdb'
                    elif 'iedb' in file_name.lower():
                        folder = 'iedb'
                    elif 'cedar' in file_name.lower():
                        folder = 'cedar'
                    elif 'ireceptor' in file_name.lower():
                        folder = 'ireceptor'
                    else:
                        folder = 'other'
                    
                    # Filter: all columns must be non-null and non-empty
                    mask = (df != '').all(axis=1) & df.notna().all(axis=1)
                    filtered = df[mask]
                    
                    if len(filtered) == 0:
                        continue
                    
                    # Add folder column and write in bulk
                    filtered.insert(0, 'folder', folder)
                    filtered.to_csv(out, sep='\t', header=False, index=False, mode='a')
                
                except Exception as e:
                    continue
        
        if not extract_file.exists() or extract_file.stat().st_size == 0:
            logger.info(f"  [{field_name}] No data")
            return field_name, 0, 0, {}, {}
        
        logger.info(f"  [{field_name}] Extracted {extract_file.stat().st_size / (1024**3):.2f} GB")
        
        # Sort
        sorted_file = extract_file.with_suffix('.sorted')
        logger.info(f"  [{field_name}] Sorting ({extract_file.stat().st_size / (1024**3):.2f} GB)...")
        
        cmd = [
            'sort',
            '-S', '50G',
            '--parallel=8',
            '-o', str(sorted_file),
            str(extract_file)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"  [{field_name}] ✓ Sorted")
        extract_file.unlink()
        
        # Count unique and total (only non-empty values)
        # Note: The extraction phase already filtered out empty values,
        # so every line we read here represents a valid, non-empty record
        dataset_unique = 0
        dataset_total = 0
        folder_unique = defaultdict(int)
        folder_total = defaultdict(int)
        prev_folder = None
        prev_value = None
        prev_dataset_value = None  # Track dataset-level uniqueness separately
        
        logger.info(f"  [{field_name}] Counting unique values...")
        
        with open(sorted_file, 'r') as f:
            for line in f:
                line = line.rstrip('\n')
                
                parts = line.split('\t', 1)
                folder = parts[0]
                value = parts[1] if len(parts) > 1 else ''
                
                # Count total per folder (all records are already non-empty due to filtering)
                folder_total[folder] += 1
                dataset_total += 1
                
                # Count unique per folder
                if folder != prev_folder or value != prev_value:
                    if folder != prev_folder:
                        prev_folder = folder
                        prev_value = None
                    
                    if value != prev_value:
                        folder_unique[folder] += 1
                        prev_value = value
                
                # Count unique across dataset (ignoring folder/database)
                # This counts each biological combination only once
                if value != prev_dataset_value:
                    dataset_unique += 1
                    prev_dataset_value = value
        
        logger.info(f"  [{field_name}] ✓ Found {dataset_unique:,} unique / {dataset_total:,} total")
        sorted_file.unlink()
        
        return field_name, dataset_unique, dataset_total, dict(folder_unique), dict(folder_total)
    
    def analyze_all(self):
        """Full analysis pipeline with parallel processing."""
        data_files = list(self.data_dir.glob("*.parquet"))
        
        logger.info("=" * 80)
        logger.info("PHASE 1: Computing file-level statistics")
        logger.info("=" * 80)
        logger.info(f"Found {len(data_files)} {'SEQ' if self.use_seq else 'MRI'} files")
        
        # With 26k+ files, skip detailed per-file stats and just get metadata
        # The real duplication analysis happens in Phase 2 anyway
        if len(data_files) > 1000:
            logger.info("Large dataset detected - using fast metadata-only mode for Phase 1")
            logger.info("(Detailed duplication stats will be computed in Phase 2)")
            logger.info("")
            
            # Use ProcessPoolExecutor with more workers for metadata reading (very fast)
            with ProcessPoolExecutor(max_workers=60) as executor:
                futures = [executor.submit(self.process_file_stats_fast, f) for f in data_files]
                for future in tqdm(as_completed(futures), total=len(futures), desc="File metadata", unit="file"):
                    stats = future.result()
                    if stats:
                        self.file_stats.append(stats)
        else:
            logger.info("Small dataset - computing detailed per-file statistics")
            logger.info("")
            
            # For smaller datasets, use the original detailed processing
            with ThreadPoolExecutor(max_workers=32) as executor:
                futures = [executor.submit(self.process_file_stats, f) for f in data_files]
                for future in tqdm(as_completed(futures), total=len(futures), desc="File-level stats", unit="file"):
                    stats = future.result()
                    if stats:
                        self.file_stats.append(stats)
        
        logger.info(f"\n✓ Processed {len(self.file_stats)} files")
        
        # Initialize stats
        dataset_stats = {'total_rows': sum(s['total_rows'] for s in self.file_stats)}
        folder_stats = defaultdict(lambda: {'total_rows': 0})
        for fstat in self.file_stats:
            folder_stats[fstat['folder']]['total_rows'] += fstat['total_rows']
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("PHASE 2: Dataset-level unique counts (parallel processing)")
        logger.info("=" * 80)
        logger.info("")
        
        # Prepare all fields to process
        all_fields = []
        for field in self.single_fields:
            all_fields.append((field, [field], data_files, str(self.temp_dir)))
        for combo_name, columns in self.combo_fields.items():
            all_fields.append((combo_name, columns, data_files, str(self.temp_dir)))
        
        logger.info(f"Processing {len(all_fields)} fields in parallel (batch size: {self.n_parallel_fields})...")
        
        # Process fields in batches (to avoid overwhelming the system)
        for i in range(0, len(all_fields), self.n_parallel_fields):
            batch = all_fields[i:i+self.n_parallel_fields]
            batch_names = [f[0] for f in batch]
            
            logger.info(f"\n--- Batch {i//self.n_parallel_fields + 1}/{(len(all_fields)-1)//self.n_parallel_fields + 1}: {', '.join(batch_names)} ---")
            
            # Process batch in parallel
            with ProcessPoolExecutor(max_workers=self.n_parallel_fields) as executor:
                futures = [executor.submit(self.process_field_worker, args) for args in batch]
                
                for future in as_completed(futures):
                    field_name, dataset_unique, dataset_total, folder_unique, folder_total = future.result()
                    
                    # Store results
                    dataset_stats[f'unique_{field_name}'] = dataset_unique
                    dataset_stats[f'total_{field_name}'] = dataset_total
                    for folder, count in folder_unique.items():
                        folder_stats[folder][f'unique_{field_name}'] = count
                    for folder, count in folder_total.items():
                        folder_stats[folder][f'total_{field_name}'] = count
        
        logger.info("\n✓ All fields processed")
        
        return dict(folder_stats), dataset_stats
    
    def generate_report(self, folder_stats, dataset_stats, output_file: str):
        """Generate comprehensive report."""
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TCR DUPLICATION ANALYSIS REPORT (Parallel Streaming)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("📊 DATASET LEVEL SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total rows: {dataset_stats['total_rows']:,}\n\n")
            
            f.write("TCR Single Fields:\n")
            for field in ['tra', 'trb', 'tra_junction_aa', 'trb_junction_aa', 
                         'trav_gene', 'trad_gene', 'traj_gene', 
                         'trbv_gene', 'trbd_gene', 'trbj_gene']:
                unique_key = f'unique_{field}'
                total_key = f'total_{field}'
                if unique_key in dataset_stats and dataset_stats[unique_key] > 0:
                    unique = dataset_stats[unique_key]
                    total = dataset_stats.get(total_key, 0)
                    dup_rate = (1 - unique/total) * 100 if total > 0 else 0
                    f.write(f"  {field:18s}: {unique:>12,} unique / {total:>15,} total ({dup_rate:5.1f}% dup)\n")
            
            f.write("\nPeptide/MHC Single Fields:\n")
            for field in ['peptide', 'mhc_one', 'mhc_two', 'mhc_one_id', 'mhc_two_id']:
                unique_key = f'unique_{field}'
                total_key = f'total_{field}'
                if unique_key in dataset_stats and dataset_stats[unique_key] > 0:
                    unique = dataset_stats[unique_key]
                    total = dataset_stats.get(total_key, 0)
                    dup_rate = (1 - unique/total) * 100 if total > 0 else 0
                    f.write(f"  {field:15s}: {unique:>12,} unique / {total:>15,} total ({dup_rate:5.1f}% dup)\n")
            
            f.write("\nTCR Combinations:\n")
            for combo in ['tra_trav', 'trb_trbv', 'tra_trb', 'tra_trav_trb_trbv']:
                unique_key = f'unique_{combo}'
                total_key = f'total_{combo}'
                if unique_key in dataset_stats and dataset_stats[unique_key] > 0:
                    unique = dataset_stats[unique_key]
                    total = dataset_stats.get(total_key, 0)
                    dup_rate = (1 - unique/total) * 100 if total > 0 else 0
                    f.write(f"  {combo:25s}: {unique:>12,} unique / {total:>15,} total ({dup_rate:5.1f}% dup)\n")
            
            f.write("\nPeptide/MHC Combinations:\n")
            for combo in ['mhc_one_mhc_two', 'peptide_mhc_one', 'peptide_mhc_two', 'peptide_mhc_one_mhc_two']:
                unique_key = f'unique_{combo}'
                total_key = f'total_{combo}'
                if unique_key in dataset_stats and dataset_stats[unique_key] > 0:
                    unique = dataset_stats[unique_key]
                    total = dataset_stats.get(total_key, 0)
                    dup_rate = (1 - unique/total) * 100 if total > 0 else 0
                    f.write(f"  {combo:30s}: {unique:>12,} unique / {total:>15,} total ({dup_rate:5.1f}% dup)\n")
            
            f.write("\nTCR + Peptide/MHC Combinations:\n")
            for combo in ['tra_peptide_mhc_one', 'tra_trav_peptide_mhc_one',
                         'tra_peptide_mhc_two', 'tra_trav_peptide_mhc_two',
                         'trb_peptide_mhc_one', 'trb_trbv_peptide_mhc_one',
                         'trb_peptide_mhc_two', 'trb_trbv_peptide_mhc_two']:
                unique_key = f'unique_{combo}'
                total_key = f'total_{combo}'
                if unique_key in dataset_stats and dataset_stats[unique_key] > 0:
                    unique = dataset_stats[unique_key]
                    total = dataset_stats.get(total_key, 0)
                    dup_rate = (1 - unique/total) * 100 if total > 0 else 0
                    f.write(f"  {combo:35s}: {unique:>12,} unique / {total:>15,} total ({dup_rate:5.1f}% dup)\n")
            
            f.write("\nFull Combinations:\n")
            for combo in ['tra_trb_peptide_mhc_one', 'tra_trav_trb_trbv_peptide_mhc_one',
                         'tra_trb_peptide_mhc_two', 'tra_trav_trb_trbv_peptide_mhc_two',
                         'tra_trb_peptide_mhc_one_mhc_two', 'tra_trav_trb_trbv_peptide_mhc_one_mhc_two']:
                unique_key = f'unique_{combo}'
                total_key = f'total_{combo}'
                if unique_key in dataset_stats and dataset_stats[unique_key] > 0:
                    unique = dataset_stats[unique_key]
                    total = dataset_stats.get(total_key, 0)
                    dup_rate = (1 - unique/total) * 100 if total > 0 else 0
                    f.write(f"  {combo:45s}: {unique:>12,} unique / {total:>15,} total ({dup_rate:5.1f}% dup)\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("📁 DATABASE/FOLDER LEVEL SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            for folder in sorted(folder_stats.keys()):
                stats = folder_stats[folder]
                f.write(f"\n{folder.upper()} ({stats['total_rows']:,} rows)\n")
                f.write("-" * 80 + "\n")
                
                # Show key fields for this database
                key_fields = [
                    ('tra', 'TRA'),
                    ('trb', 'TRB'),
                    ('tra_trav', 'TRA + TRAV'),
                    ('trb_trbv', 'TRB + TRBV'),
                    ('tra_trb', 'TRA + TRB'),
                    ('peptide', 'Peptide'),
                    ('mhc_one', 'MHC-I'),
                    ('peptide_mhc_one', 'Peptide + MHC-I'),
                    ('tra_trb_peptide_mhc_one', 'TRA + TRB + Peptide + MHC-I'),
                ]
                
                shown_any = False
                for field_key, field_label in key_fields:
                    key = f'unique_{field_key}'
                    if key in stats and stats[key] > 0:
                        total = stats['total_rows']
                        unique = stats[key]
                        dup_rate = (1 - unique/total) * 100 if total > 0 else 0
                        f.write(f"  {field_label:35s}: {unique:>10,} unique ({dup_rate:5.1f}% dup)\n")
                        shown_any = True
                
                if not shown_any:
                    f.write("  (No field data available)\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("📄 FILE DISTRIBUTION BY DATABASE\n")
            f.write("=" * 80 + "\n\n")
            
            # Count files per database
            from collections import Counter
            file_counts = Counter(s['folder'] for s in self.file_stats)
            row_counts = defaultdict(int)
            for s in self.file_stats:
                row_counts[s['folder']] += s['total_rows']
            
            f.write(f"{'Database':<15} {'Files':>10}  {'Total Rows':>15}\n")
            f.write("-" * 80 + "\n")
            for db in sorted(file_counts.keys()):
                f.write(f"{db:<15} {file_counts[db]:>10,}  {row_counts[db]:>15,}\n")
            
            # Add category and study breakdowns for "other" folder
            other_files = [s for s in self.file_stats if s['folder'] == 'other']
            if other_files and len(other_files) > 0:
                f.write("\n" + "=" * 80 + "\n")
                f.write("📂 'OTHER' CATEGORY BREAKDOWN BY STUDY CATEGORY\n")
                f.write("=" * 80 + "\n\n")
                
                category_counts = Counter(s['category'] for s in other_files)
                category_rows = defaultdict(int)
                for s in other_files:
                    category_rows[s['category']] += s['total_rows']
                
                f.write(f"{'Category':<30} {'Files':>10}  {'Total Rows':>15}\n")
                f.write("-" * 80 + "\n")
                for cat in sorted(category_counts.keys()):
                    f.write(f"{cat:<30} {category_counts[cat]:>10,}  {category_rows[cat]:>15,}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("📂 'OTHER' CATEGORY BREAKDOWN BY STUDY ID (Top 30)\n")
                f.write("=" * 80 + "\n\n")
                
                study_counts = Counter(s['study_id'] for s in other_files)
                study_rows = defaultdict(int)
                study_category = {}
                for s in other_files:
                    study_rows[s['study_id']] += s['total_rows']
                    study_category[s['study_id']] = s['category']
                
                f.write(f"{'Study ID':<20} {'Category':<25} {'Files':>8}  {'Total Rows':>15}\n")
                f.write("-" * 80 + "\n")
                for study, count in sorted(study_counts.items(), key=lambda x: -x[1])[:30]:
                    cat = study_category.get(study, 'unknown')
                    rows = study_rows[study]
                    f.write(f"{study:<20} {cat:<25} {count:>8,}  {rows:>15,}\n")
            
            # Only show top files if we have per-file duplication stats (small datasets)
            trb_stats = [s for s in self.file_stats if 'trb_dup_rate' in s]
            if trb_stats:
                f.write("\n" + "=" * 80 + "\n")
                f.write("📄 TOP 20 FILES BY TRB DUPLICATION RATE\n")
                f.write("=" * 80 + "\n\n")
                
                trb_sorted = sorted(trb_stats, key=lambda x: x['trb_dup_rate'], reverse=True)[:20]
                
                for i, stat in enumerate(trb_sorted, 1):
                    f.write(f"{i:2d}. {stat['file']:50s} "
                           f"{stat['trb_dup_rate']:5.1f}% dup "
                           f"({stat['trb_unique']:,}/{stat['trb_total']:,} unique)\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        logger.info(f"Report written to {output_file}")
    
    def export_file_stats_csv(self, output_file: str):
        """Export file-level stats to CSV."""
        import pandas as pd
        df = pd.DataFrame(self.file_stats)
        df.to_csv(output_file, index=False)
        logger.info(f"File statistics exported to {output_file}")
    
    def cleanup(self):
        """Clean up temp directory."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up: {self.temp_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Parallel streaming duplication analysis (optimized for high-memory systems)"
    )
    parser.add_argument('--parsed-dir', default='/mnt/ephemeral/parsed_output')
    parser.add_argument('--temp-dir', default='/mnt/ephemeral')
    parser.add_argument('--output-report', default='duplication_report_final.txt')
    parser.add_argument('--output-csv', default='file_duplication_stats_final.csv')
    parser.add_argument('--parallel', type=int, default=6, 
                       help='Number of fields to process in parallel (default: 6, uses ~300GB RAM)')
    parser.add_argument('--use-seq', action='store_true',
                       help='Use seq files instead of mri files (recommended for iReceptor data)')
    parser.add_argument('--mapping-file', default='/home/ubuntu/quest/source_file_mapping.json',
                       help='JSON file mapping output filenames to source metadata')
    
    args = parser.parse_args()
    
    analyzer = ParallelStreamAnalyzer(
        args.parsed_dir, 
        temp_dir=args.temp_dir,
        n_parallel_fields=args.parallel,
        use_seq=args.use_seq,
        mapping_file=args.mapping_file
    )
    
    try:
        logger.info("Starting parallel streaming duplication analysis...")
        logger.info(f"Optimized for high-memory systems (using up to {args.parallel * 50}GB for sorting)")
        logger.info("")
        
        folder_stats, dataset_stats = analyzer.analyze_all()
        
        logger.info("")
        logger.info("Generating report...")
        analyzer.generate_report(folder_stats, dataset_stats, args.output_report)
        
        logger.info("Exporting CSV...")
        analyzer.export_file_stats_csv(args.output_csv)
        
        logger.info("")
        logger.info("✓ Analysis complete!")
        
    finally:
        analyzer.cleanup()


if __name__ == "__main__":
    main()
