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

def parse_files_individually(file_paths, config_path, output_mri_path, output_seq_path):
    os.makedirs(output_mri_path, exist_ok=True)
    os.makedirs(output_seq_path, exist_ok=True)

    for file_path in file_paths:
        mri_dd, seq_dd = parse_file(file_path, config_path)
        if mri_dd is None or seq_dd is None:
            continue

        # Convert to Dask DataFrame if they aren’t already
        # (Only necessary if parse_file returns pandas DataFrames)
        #mri_dd = parse_file(mri_df, npartitions=1)
        #seq_dd = dd.from_pandas(seq_df, npartitions=1)

        # Build output filenames that match the input file stem
        # e.g. /path/to/input/tcr_data.csv -> "tcr_data_mri.parquet"
        parquet_basename = file_path.stem
        mri_parquet_name = f"{parquet_basename}_mri.parquet"
        seq_parquet_name = f"{parquet_basename}_seq.parquet"

        # Full path to output locations
        mri_output_file = os.path.join(output_mri_path, mri_parquet_name)
        seq_output_file = os.path.join(output_seq_path, seq_parquet_name)

        # Convert schema to PyArrow for Parquet writing
        mri_column_types = {"tid": "string", "tra": "string", "trad_gene": "string",
            "traj_gene": "string", "trav_gene": "string", "trb": "string",
            "trbd_gene": "string", "trbj_gene": "string", "trbv_gene": "string",
            "peptide": "string", "mhc_one": "string", "mhc_two": "string",
            "sequence": "string", "repertoire_id": "string", "study_id": "string",
            "category": "string", "molecule_type": "string", "host_organism": "string",
            "source": "string"}
        
        seq_column_types = {"source": "string", "tid": "string",
            "tra": "string", "trad_gene": "string", "traj_gene": "string",
            "trav_gene": "string", "trb": "string", "trbd_gene": "string",
            "trbj_gene": "string", "trbv_gene": "string", "peptide": "string",
            "mhc_one": "string", "mhc_two": "string", "sequence": "string"}

        mri_dd = mri_dd.astype(mri_column_types)
        seq_dd = seq_dd.astype(seq_column_types)
        # Write each DataFrame to its own Parquet
        mri_dd.to_parquet(mri_output_file, engine="pyarrow", write_index=False)
        seq_dd.to_parquet(seq_output_file, engine="pyarrow", write_index=False)

        logging.info(f"Wrote MRI to {mri_output_file}")
        logging.info(f"Wrote SEQ to {seq_output_file}")


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
        output_mri_path = os.path.join(output_path, "mri/airr_ngs_data")
        output_seq_path = os.path.join(output_path, "sequence/airr_ngs_data")

        # Process the files individually and write results
        parse_files_individually(tcr_files, format_path, output_mri_path, output_seq_path)


    logging.info("Parsing completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch parser for AIRR-Seq data.")
    parser.add_argument("--config_path", required=True, type=str, help="Path to configuration YAML file")
    parser.add_argument("--verbose", action='store_true', default=False, help="Display detailed logging")
    args = parser.parse_args()

    setup_logger(args.verbose)
    main(args.config_path, args.verbose)
