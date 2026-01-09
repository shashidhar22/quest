
import argparse
import pyarrow.feather as pf
import pyarrow as pa
from pathlib import Path
from tqdm import tqdm
import numpy as np

def add_length_field(input_dir: Path, output_dir: Path):
    """
    Adds a 'length' field to tokenized Arrow datasets.
    The length is calculated as the sum of the 'attention_mask' for each sequence.

    Args:
        input_dir: Path to the input directory containing tokenized Arrow files.
        output_dir: Path to the output directory where processed Arrow files
                    with the new 'length' field will be saved.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    arrow_files = sorted(list(input_dir.glob("*.arrow")))
    if not arrow_files:
        print(f"No .arrow files found in {input_dir}. Exiting.")
        return

    print(f"Found {len(arrow_files)} arrow files in {input_dir}")

    for input_file in tqdm(arrow_files, desc="Processing files"):
        try:
            # Read the Arrow file
            table = pf.read_table(input_file)

            # Check if 'attention_mask' column exists
            if 'attention_mask' not in table.column_names:
                print(f"Warning: '{input_file.name}' does not contain 'attention_mask'. Skipping.")
                # If attention_mask is missing, we still want to copy the file
                # so the output directory is consistent. We'll add a 'length' column
                # with nulls or zeros if needed, but for now just copying.
                # For this specific request, it's about adding a *calculated* length.
                # So if attention_mask is missing, we can't calculate it.
                # Just copy the original file content.
                pf.write_table(table, output_dir / input_file.name)
                continue

            # Calculate lengths from attention_mask
            # The attention_mask is a list of integers (0 or 1)
            # Summing it gives the actual sequence length (excluding padding)
            attention_mask_array = table['attention_mask'].to_numpy()
            
            # Use a list comprehension to sum each attention mask list
            # Ensure it handles potential None values if the column has them
            lengths = [np.sum(mask) if mask is not None else 0 for mask in attention_mask_array]
            
            # Create a new PyArrow array for lengths
            lengths_array = pa.array(lengths, type=pa.int32())

            # Add the new 'length' column to the table
            new_table = table.set_column(
                len(table.column_names),  # Insert at the end
                pa.field('length', pa.int32()),
                lengths_array
            )

            # Write the new table to the output directory
            output_file = output_dir / input_file.name
            pf.write_table(new_table, output_file)

        except Exception as e:
            print(f"Error processing file {input_file.name}: {e}")
            # Optionally, copy the file as is if processing fails
            # This ensures all files are present in output_dir even if some fail
            try:
                table = pf.read_table(input_file)
                pf.write_table(table, output_dir / input_file.name)
                print(f"Copied original file {input_file.name} due to error.")
            except Exception as copy_e:
                print(f"Could not even copy original file {input_file.name}: {copy_e}")


def main():
    parser = argparse.ArgumentParser(
        description="Add a 'length' field to tokenized Arrow datasets."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Path to the input directory containing tokenized Arrow files."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Path to the output directory for processed Arrow files."
    )
    args = parser.parse_args()

    add_length_field(args.input_dir, args.output_dir)
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
