import logging
import yaml
import argparse
import os
import math
from pathlib import Path
from tqdm import tqdm

import pandas as pd
from dask import delayed, compute

from parsers.airr_bulk_parser import BulkFileParser
from parsers.airr_misc_parser import MiscFileParser
from parsers.airr_paired_parser import PairedFileParser
from parsers.airr_database_parser import DatabaseParser
from parsers.utils import standardize_mri, standardize_sequence

###############################################################################
# Utility / Helper Functions
###############################################################################

def setup_logger(verbose):
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO if verbose else logging.ERROR
    )

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_file_sync(file_path, config_path):
    """
    Synchronously parse one file into (mri_df, seq_df), both pandas DataFrames.
    Returns (None, None) if skip/fail.
    """
    extension = file_path.suffix
    file_type = file_path.parent.name
    filename = file_path.name

    # Skip irrelevant or empty
    if (
        extension in {".sh", ".zip"}
        or file_type in {
            "fasta", "consensus_fasta", "consensus", "raw_contigs",
            "contig_fasta", "unfiltered_contigs", "gd"
        }
        or filename.startswith(".")
        or os.path.getsize(file_path) == 0
    ):
        logging.debug(f"Skipped file: {file_path}")
        return None, None

    logging.info(f"Parsing file: {file_path}")
    if file_type in {"contigs", "clonotypes", "airr"}:
        mri_df, seq_df = PairedFileParser(file_path, test=False).parse()
    elif file_type == "misc":
        mri_df, seq_df = MiscFileParser(file_path, config_path, test=False).parse()
    else:
        mri_df, seq_df = BulkFileParser(file_path, config_path, test=False).parse()

    if mri_df is None or seq_df is None:
        logging.warning(f"Failed to parse file: {file_path}")
        return None, None

    # Reindex to known columns
    mri_cols = [
        'tid','tra','trad_gene','traj_gene','trav_gene','trb',
        'trbd_gene','trbj_gene','trbv_gene','sequence',
        'repertoire_id','study_id','category','molecule_type',
        'host_organism','source'
    ]
    seq_cols = [
        'trav_gene','traj_gene','trad_gene','tra',
        'trbv_gene','trbj_gene','trbd_gene','trb',
        'peptide','mhc_one','mhc_two','sequence'
    ]
    # Only keep columns that exist
    mri_keep = [c for c in mri_cols if c in mri_df.columns]
    seq_keep = [c for c in seq_cols if c in seq_df.columns]
    mri_df = mri_df[mri_keep]
    seq_df = seq_df[seq_keep]

    # Standardize
    mri_df = standardize_mri(mri_df)
    seq_df = standardize_sequence(seq_df)
    return mri_df, seq_df

@delayed
def parse_file_delayed(file_path, config_path):
    """Dask-delayed wrapper around parse_file_sync."""
    return parse_file_sync(file_path, config_path)

@delayed
def delayed_concat(list_of_dfs):
    """Delayed function to concatenate a list of Pandas DataFrames."""
    real_dfs = [df for df in list_of_dfs if df is not None and not df.empty]
    if not real_dfs:
        return pd.DataFrame()
    return pd.concat(real_dfs, ignore_index=True)

def parse_database(config_path):
    """
    Parse the AIRR-Seq database (ex: VDJDB, McPAS, IEDB, iReceptor, etc.)
    Return (db_mri, db_seq) as pandas DataFrames.
    """
    tqdm.write("Parsing AIRR-Seq database...")
    db_mri, db_seq = DatabaseParser(config_path).parse()  # both are pandas DataFrames
    return db_mri, db_seq

###############################################################################
# Main logic with chunk-based approach
###############################################################################

def main(config_path, verbose=True, approx_chunk_size=50_000_000):
    """
    approx_chunk_size: approximate # of rows or some heuristic to keep memory under ~50GB.
    You might refine this logic by actual measurement or file size grouping.
    """
    setup_logger(verbose)
    config = load_config(config_path)
    study_path = Path(config["ngs"]["study_path"])
    output_path = Path(config["outputs"]["output_path"])

    logging.info("Beginning parse with chunk-based approach to limit memory usage...")

    # 1) Parse the entire database (pandas DataFrames in memory).
    #    Usually this isn't huge, so keep it around to merge with NGS data.
    db_mri, db_seq = parse_database(config_path)
    if db_mri is None or db_seq is None:
        db_mri = pd.DataFrame()
        db_seq = pd.DataFrame()
    else:
        logging.info(f"Database parse complete. MRI shape={db_mri.shape}, SEQ shape={db_seq.shape}")

    # 2) Gather all categories
    categories = [
        p for p in study_path.iterdir()
        if p.is_dir() and p.name not in {'parquet', 'pickle', 'formatted'}
    ]

    # 3) Collect all TCR file paths across all categories to track overall progress
    all_tcr_files = []
    cat_file_map = {}  # {category -> list_of_paths}
    for category in categories:
        tcr_files = list(category.rglob('*/tcr/*/*'))
        if tcr_files:
            cat_file_map[category] = tcr_files
            all_tcr_files.extend(tcr_files)

    logging.info(f"Total TCR files across all categories: {len(all_tcr_files)}")

    # Create a single tqdm progress bar for all TCR files
    pbar = tqdm(total=len(all_tcr_files), desc="Reading TCR data files", unit="file", leave=True)

    # We'll store partial outputs per category chunk, then do a final combine
    for category in categories:
        tcr_files = cat_file_map.get(category, [])
        if not tcr_files:
            continue

        logging.info(f"Category '{category.name}' has {len(tcr_files)} input files")

        # We'll chunk the file list so that each chunk doesn't blow up memory.
        chunk_size = 100  # or user can guess an approximate # of files for ~50GB
        file_chunks = [
            tcr_files[i : i + chunk_size]
            for i in range(0, len(tcr_files), chunk_size)
        ]

        cat_mri_frames = []
        cat_seq_frames = []

        # For each chunk, parse in delayed, concat, then compute => partial chunk in memory
        for chunk_idx, chunk_files in enumerate(file_chunks):
            if not chunk_files:
                continue

            # Build parse tasks
            parse_tasks = [parse_file_delayed(cf, config["ngs"]["format_path"]) for cf in chunk_files]

            # Split them into MRI/SEQ
            chunk_mri_delayed = delayed_concat([pt.getitem(0) for pt in parse_tasks])
            chunk_seq_delayed = delayed_concat([pt.getitem(1) for pt in parse_tasks])

            logging.info(f"Computing category={category.name} chunk={chunk_idx} ...")
            mri_chunk_df, seq_chunk_df = compute(chunk_mri_delayed, chunk_seq_delayed)

            # Combine chunk-level data with DB data if desired
            if not db_mri.empty or not db_seq.empty:
                mri_chunk_df = pd.concat([db_mri, mri_chunk_df], ignore_index=True)
                seq_chunk_df = pd.concat([db_seq, seq_chunk_df], ignore_index=True)

            # We can now write chunk outputs to disk or keep them in memory for final concat
            chunk_out_dir = output_path / "chunks" / category.name
            chunk_out_dir.mkdir(parents=True, exist_ok=True)

            # Write partial chunk
            mri_chunk_path = chunk_out_dir / f"mri_chunk_{chunk_idx}.parquet"
            seq_chunk_path = chunk_out_dir / f"seq_chunk_{chunk_idx}.parquet"
            mri_chunk_df.to_parquet(mri_chunk_path, index=False)
            seq_chunk_df.to_parquet(seq_chunk_path, index=False)

            logging.info(f"Wrote partial chunk {chunk_idx} for category={category.name}")

            # **Update the progress bar** by however many files we just processed
            pbar.update(len(chunk_files))

            # Optionally store these paths for a final category-level combine
            cat_mri_frames.append(mri_chunk_path)
            cat_seq_frames.append(seq_chunk_path)

            # Then promptly free memory
            del mri_chunk_df
            del seq_chunk_df

        # After finishing all chunks for this category, we can read all partial chunk files
        # and combine them if we want a single category-level file
        if cat_mri_frames:
            partial_mris = [pd.read_parquet(p) for p in cat_mri_frames]
            cat_mri = pd.concat(partial_mris, ignore_index=True) if partial_mris else pd.DataFrame()

            partial_seqs = [pd.read_parquet(p) for p in cat_seq_frames]
            cat_seq = pd.concat(partial_seqs, ignore_index=True) if partial_seqs else pd.DataFrame()

            category_out_dir = output_path / "final" / category.name
            category_out_dir.mkdir(parents=True, exist_ok=True)

            cat_mri_path = category_out_dir / f"{category.name}_mri.parquet"
            cat_seq_path = category_out_dir / f"{category.name}_seq.parquet"
            cat_mri.to_parquet(cat_mri_path, index=False)
            cat_seq.to_parquet(cat_seq_path, index=False)

            logging.info(f"Category {category.name} final MRI => {cat_mri_path}")
            logging.info(f"Category {category.name} final SEQ => {cat_seq_path}")

            # discard everything to free memory
            del cat_mri
            del cat_seq
            del partial_mris
            del partial_seqs

    pbar.close()
    logging.info("All categories processed with chunk-based approach. Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch parser for AIRR-Seq data with chunk-limited memory usage.")
    parser.add_argument("--config_path", required=True, type=str, help="Path to configuration YAML file")
    parser.add_argument("--verbose", action='store_true', default=False, help="Display detailed logging")
    args = parser.parse_args()

    setup_logger(args.verbose)
    main(args.config_path, args.verbose)
