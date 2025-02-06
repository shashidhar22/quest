import os
import yaml
import dask
import argparse
import dask.dataframe as dd

from tqdm import tqdm
from pathlib import Path
from dask.diagnostics import ProgressBar


from parsers.airr_bulk_parser import BulkFileParser
from parsers.airr_misc_parser import MiscFileParser
from parsers.airr_database_parser import DatabaseParser
from parsers.airr_paired_parser import PairedFileParser



def load_config(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

def parse_airrseq(study_path, config_path):
    """
    Parse AIRR-Seq datasets in the study path and write results to Parquet files in batches.

    Args:
        study_path (str): Path to the study dataset.
        config_path (str): Path to the configuration file.
        output_path (str): Path to save Parquet files.
        batch_size (int): Number of iterations to process before writing to Parquet.
    """
    ignore = {'parquet', 'pickle'}
    study_path = Path(study_path)

    mri_tables = []
    sequence_tables = []

    study_categories = list(study_path.iterdir())

    for study_category in tqdm(study_categories, desc="Parsing AIRR-Seq studies"):
        if study_category.is_dir() and study_category.name not in ignore:
            tcr_files = list(study_category.rglob('*/tcr/*/*'))

            for file_path in tqdm(tcr_files, desc=f"Processing files in {study_category.name}", leave=False):
                extension = file_path.suffix
                file_type = file_path.parent.name
                filename = file_path.name

                # Skip irrelevant files
                if (extension in {".sh", ".zip"} or
                    file_type in {"fasta", "consensus_fasta", "consensus", "raw_contigs", "contig_fasta", "unfiltered_contigs", "gd"} or
                    filename.startswith(".")):
                    continue

                file_size = os.path.getsize(file_path)
                if file_type == "contigs":
                    mri_table, sequence_table = PairedFileParser(file_path).parse()
                elif file_type == "clonotypes":
                    mri_table, sequence_table = PairedFileParser(file_path).parse()
                elif file_type == "airr":
                    mri_table, sequence_table = PairedFileParser(file_path).parse()
                elif file_type == "misc":
                    mri_table, sequence_table = MiscFileParser(file_path, config_path).parse()
                elif file_size > 0:
                    mri_table, sequence_table = BulkFileParser(file_path, config_path).parse()
                else:
                    open('missing_data.txt', 'a').write(f'{file_path}\n')
                    continue
                if mri_table is not None and sequence_table is not None:
                    mri_tables.append(mri_table)
                    sequence_tables.append(sequence_table)
                else:
                    open('missing_data.txt', 'a').write(f'{file_path}\n')
    mri_table = dd.concat(mri_tables, axis=0, interleave_partitions=True)
    sequence_table = dd.concat(sequence_tables, axis=0, interleave_partitions=True)
    return mri_table, sequence_table


def parse_database(config_path):
    """
    Parse the AIRR-Seq database.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        tuple: Dask DataFrames for MRI table and sequence table.
    """
    tqdm.write("Parsing AIRR-Seq database...")
    mri_table, sequence_table = DatabaseParser(config_path).parse()
    return mri_table, sequence_table

def main(config_path):
    config = load_config(config_path)
    study_path = config["ngs"]["study_path"]
    format_path = config["ngs"]["format_path"]
    tqdm.write(f"Study path: {study_path}")
    tqdm.write(f"Format path: {format_path}")
    tqdm.write(f"Output path: {config["outputs"]["output_path"]}")
    tqdm.write(f"Temp path: {config["outputs"]["temp_path"]}")
    tqdm.write("Starting AIRR-Seq data parsing...")
    mri_seq_table, sequence_seq_table = parse_airrseq(study_path, format_path)

    tqdm.write("Starting AIRR-Seq database parsing...")
    mri_db_table, sequence_db_table = parse_database(config_path)

    # Optional: Repartition. Only if you want to reduce partitions or set a size.
    mri_seq_table = mri_seq_table.repartition(partition_size="1024MB")
    mri_db_table = mri_db_table.repartition(partition_size="1024MB")

    # Concat them as Dask DataFrames
    mri_table = dd.concat([mri_seq_table, mri_db_table], interleave_partitions=True)

    # Same for sequence tables
    sequence_seq_table = sequence_seq_table.repartition(partition_size="1024MB")
    sequence_db_table = sequence_db_table.repartition(partition_size="1024MB")

    sequence_table = dd.concat([sequence_seq_table, sequence_db_table], interleave_partitions=True)

    # Write to Parquet using Dask (lazy -> triggered compute in parallel)
    output_path = Path(config["outputs"]["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)
    dask.config.set(temporary_directory=config["outputs"]["temp_path"])
    tqdm.write(f"Saving MRI and Sequence table to {output_path}")
    with ProgressBar():
        mri_table.to_parquet(str(output_path / "mri_table.parquet"), engine="pyarrow",  compute=True)
    with ProgressBar():
        sequence_table.to_parquet(str(output_path / "sequence_table.parquet"), engine="pyarrow",  compute=True)

    tqdm.write("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse AIRR-Seq data.")
    
    # Add arguments
    parser.add_argument(
        "--config_path",
        required=True,
        type=str,
        help="Path to the configuration file (YAML)."
    )
    
    # Parse arguments
    args = parser.parse_args()

    # Call main with parsed arguments
    main(args.config_path)
