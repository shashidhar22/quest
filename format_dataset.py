import logging
import yaml
import argparse
import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import dask.dataframe as dd

from parsers.airr_bulk_parser import BulkFileParser
from parsers.airr_misc_parser import MiscFileParser
from parsers.airr_paired_parser import PairedFileParser
from parsers.utils import standardize_mri, standardize_sequence

def setup_logger(verbose):
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO if verbose else logging.ERROR
    )

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_file(file_path, config_path):
    """
    Parses one file into two DataFrames: mri_df, seq_df
    Returns (None, None) if file is skipped or parsing fails.
    """
    extension = file_path.suffix
    file_type = file_path.parent.name
    filename = file_path.name

    # Skip irrelevant or empty files
    if (
        extension in {".sh", ".zip"}
        or file_type in {"fasta", "consensus_fasta", "consensus", "raw_contigs",
                         "contig_fasta", "unfiltered_contigs", "gd"}
        or filename.startswith(".")
        or os.path.getsize(file_path) == 0
    ):
        logging.debug(f"Skipped irrelevant or empty file: {file_path}")
        return None, None

    logging.info(f"Parsing file: {file_path}")

    # Determine which parser to use
    if file_type in {"contigs", "clonotypes", "airr"}:
        mri_df, seq_df = PairedFileParser(file_path, test=False).parse()
    elif file_type == "misc":
        mri_df, seq_df = MiscFileParser(file_path, config_path, test=False).parse()
    else:
        mri_df, seq_df = BulkFileParser(file_path, config_path, test=False).parse()

    if mri_df is None or seq_df is None:
        logging.warning(f"Failed to parse file: {file_path}")
        return None, None

    # Reorder & standardize columns
    mri_order = [
        'tid','tra','trad_gene','traj_gene','trav_gene','trb',
        'trbd_gene','trbj_gene','trbv_gene','sequence',
        'repertoire_id','study_id','category','molecule_type',
        'host_organism','source'
    ]
    seq_order = [
        'trav_gene','traj_gene','trad_gene','tra',
        'trbv_gene','trbj_gene','trbd_gene','trb',
        'peptide','mhc_one','mhc_two','sequence'
    ]

    # Reindex columns and standardize
    mri_df = mri_df[mri_order]
    seq_df = seq_df[seq_order]
    mri_df = standardize_mri(mri_df)
    seq_df = standardize_sequence(seq_df)

    return mri_df, seq_df

def parse_in_batches(file_paths, config_path, output_mri_path, output_seq_path, batch_size=1):
    """
    Process the given file_paths in batches of 10 files, parse them into
    MRI and SEQ DataFrames, then write each batch to a separate Parquet file
    in the same folder.
    """
    # Ensure output directories exist
    os.makedirs(output_mri_path, exist_ok=True)
    os.makedirs(output_seq_path, exist_ok=True)

    batch_idx = 0

    for start_idx in range(0, len(file_paths), batch_size):
        # Grab up to 10 files in this batch
        chunk_paths = file_paths[start_idx : start_idx + batch_size]

        mri_frames = []
        seq_frames = []

        for file_path in chunk_paths:
            mri_df, seq_df = parse_file(file_path, config_path)
            if mri_df is not None and seq_df is not None:
                mri_frames.append(mri_df)
                seq_frames.append(seq_df)

        if not mri_frames and not seq_frames:
            continue

        # Combine & write if we have any MRI data in this batch
        if mri_frames:
            combined_mri = dd.concat(mri_frames, interleave_partitions=True)
            # Reset index so it won't complain about divisions
            combined_mri = combined_mri.reset_index(drop=True)
            mri_batch_file = os.path.join(output_mri_path, f"mri_batch_{batch_idx}.parquet")
            combined_mri.to_parquet(
                mri_batch_file,
                engine="pyarrow",
                write_index=False
            )
            logging.info(f"Wrote MRI batch {batch_idx} to {mri_batch_file}")

        # Combine & write if we have any Sequence data in this batch
        if seq_frames:
            combined_seq = dd.concat(seq_frames, interleave_partitions=True)
            combined_seq = combined_seq.reset_index(drop=True)
            seq_batch_file = os.path.join(output_seq_path, f"seq_batch_{batch_idx}.parquet")
            combined_seq.to_parquet(
                seq_batch_file,
                engine="pyarrow",
                write_index=False
            )
            logging.info(f"Wrote SEQ batch {batch_idx} to {seq_batch_file}")

        batch_idx += 1

def main(config_path, verbose=True):
    setup_logger(verbose)

    # Load config
    config = load_config(config_path)
    study_path = Path(config["ngs"]["study_path"])
    output_path = config["outputs"]["output_path"]
    format_path = config["ngs"]["format_path"]
    logging.info("Starting AIRR-Seq data parsing")

    # Collect all categories
    study_categories = [
        p for p in study_path.iterdir()
        if p.is_dir() and p.name not in {'parquet', 'pickle'}
    ]

    for category in tqdm(study_categories, desc="Study Categories"):
        # All TCR files for this category
        tcr_files = list(category.rglob('*/tcr/*/*'))
        if not tcr_files:
            continue

        logging.info(f"Processing category '{category.name}' with {len(tcr_files)} files")

        # Output directories for MRI and sequence parquet
        output_mri_path = os.path.join(output_path, "mri/airr_ngs_data", category.name)
        output_seq_path = os.path.join(output_path, "sequence/airr_ngs_data", category.name)

        # Process the files in batches of 1000 and write results
        parse_in_batches(tcr_files, format_path, output_mri_path, output_seq_path)

    logging.info("Parsing completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch parser for AIRR-Seq data.")
    parser.add_argument("--config_path", required=True, type=str, help="Path to configuration YAML file")
    parser.add_argument("--verbose", action='store_true', default=False, help="Display detailed logging")
    args = parser.parse_args()

    setup_logger(args.verbose)
    main(args.config_path, args.verbose)
