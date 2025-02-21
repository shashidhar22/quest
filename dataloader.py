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
from parsers.utils import standardize_mri, standardize_sequence
import dask.dataframe as dd

def write_parquet_append(ddfs, output_path, overwrite=True, engine="pyarrow"):
    """
    Write a list of Dask DataFrames to a single Parquet dataset incrementally,
    avoiding a massive task graph by appending each DataFrame in a loop.
    
    Parameters
    ----------
    ddfs : list of dd.DataFrame
        The list of Dask DataFrames (all must share the same schema).
    output_path : str
        The directory (or path) for the resulting Parquet dataset.
    overwrite : bool, optional (default True)
        If True, the first DataFrame overwrites any existing data at `output_path`.
        Subsequent DataFrames are appended.
    engine : str, optional (default "pyarrow")
        The Parquet engine to use for writing.
    """
    first_write = True
    for df in tqdm(ddfs, desc="Writing AIRR-Seq studies"):
        # Optional: Repartition each DF if you want a certain # of partitions
        # df = df.repartition(npartitions=10)
        
        # On the first iteration, possibly overwrite existing dataset
        df = df.map_partitions(lambda pdf: pdf.astype("string[pyarrow]"))
        if first_write and overwrite:
            df.to_parquet(
                output_path,
                engine=engine,
                write_index=False,
                overwrite=True
            )
            first_write = False
        else:
            # On subsequent iterations, append to the existing dataset
            try:
                df.to_parquet(
                    output_path,
                    engine=engine,
                    write_index=False,
                    append=True
                )
            except ValueError:
                breakpoint()


def load_config(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

def parse_airrseq(study_path, config_path, output_path):
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
                    mri_table, sequence_table = PairedFileParser(file_path, test=True).parse()
                elif file_type == "clonotypes":
                    mri_table, sequence_table = PairedFileParser(file_path, test=True).parse()
                elif file_type == "airr":
                    mri_table, sequence_table = PairedFileParser(file_path, test=True).parse()
                elif file_type == "misc":
                    mri_table, sequence_table = MiscFileParser(file_path, config_path, test=True).parse()
                elif file_size > 0:
                    mri_table, sequence_table = BulkFileParser(file_path, config_path, test=True).parse()
                else:
                    open('missing_data.txt', 'a').write(f'{file_path}\n')
                    continue
                if mri_table is not None and sequence_table is not None:
                    mri_order = ['tid', 'tra', 'trad_gene', 'traj_gene', 
                                 'trav_gene', 'trb', 'trbd_gene', 'trbj_gene', 
                                 'trbv_gene', 'sequence', 'repertoire_id', 
                                 'study_id', 'category', 'molecule_type', 
                                 'host_organism', 'source']
                    mri_table = mri_table[mri_order]
                    mri_table = standardize_mri(mri_table)
                    sequence_table = standardize_sequence(sequence_table)
                    # try:
                    #     mri_table.head(1)
                    # except ValueError:
                    #     breakpoint()
                    mri_tables.append(mri_table)
                    sequence_tables.append(sequence_table)
                else:
                    open('missing_data.txt', 'a').write(f'{file_path}\n')
    
    write_parquet_append(sequence_tables, f"{output_path}/sequence/airr_seq_data.parquet", overwrite=True)
    write_parquet_append(mri_tables, f"{output_path}/mri/airr_seq_data.parquet", overwrite=True)
    
    return mri_tables, sequence_tables


def parse_database(config_path, output_path):
    """
    Parse the AIRR-Seq database.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        tuple: Dask DataFrames for MRI table and sequence table.
    """
    tqdm.write("Parsing AIRR-Seq database...")
    mri_table, sequence_table = DatabaseParser(config_path, test=True).parse()
    with ProgressBar():
        mri_table.to_parquet(f"{output_path}/mri/databases.parquet", engine="pyarrow",  compute=True)
    with ProgressBar():
        sequence_table.to_parquet(f"{output_path}/sequence/databases.parquet", engine="pyarrow",  compute=True)
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
    dask.config.set(temporary_directory=config["outputs"]["temp_path"])
    mri_seq_table, sequence_seq_table = parse_airrseq(study_path, format_path, config["outputs"]["output_path"])

    tqdm.write("Starting AIRR-Seq database parsing...")
    mri_db_table, sequence_db_table = parse_database(config_path, config["outputs"]["output_path"])

    # Optional: Repartition. Only if you want to reduce partitions or set a size.
    #mri_seq_table = mri_seq_table.repartition(partition_size="1024MB")
    #mri_db_table = mri_db_table.repartition(partition_size="1024MB")

    # Concat them as Dask DataFrames
    #mri_table = dd.concat([mri_seq_table, mri_db_table], interleave_partitions=True)

    # Same for sequence tables
    #sequence_seq_table = sequence_seq_table.repartition(partition_size="1024MB")
    #sequence_db_table = sequence_db_table.repartition(partition_size="1024MB")

    #sequence_table = dd.concat([sequence_seq_table, sequence_db_table], interleave_partitions=True)

    # Write to Parquet using Dask (lazy -> triggered compute in parallel)
    #output_path = Path(config["outputs"]["output_path"])
    #output_path.mkdir(parents=True, exist_ok=True)
    
    #tqdm.write(f"Saving MRI and Sequence table to {output_path}")
    #with ProgressBar():
    #    mri_table.to_parquet(str(output_path / "mri_table.parquet"), engine="pyarrow",  compute=True)
    #with ProgressBar():
    #    sequence_table.to_parquet(str(output_path / "sequence_table.parquet"), engine="pyarrow",  compute=True)

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
