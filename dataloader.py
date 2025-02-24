import pandas as pd
import torch
import random
from torch.utils.data import Dataset, random_split
import argparse
import pickle
from itertools import permutations
import os
from quest.dataset import AminoAcidDataset
from torch.utils.data import Dataset
import torch
import pandas as pd
import torch
import random
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, random_split
from itertools import permutations
import argparse
import os


def parse_args():
    """Parse command-line arguments for the script."""
    parser = argparse.ArgumentParser(description="Train an LSTM model on amino acid sequences.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to the dataset file in parquet format.")
    parser.add_argument('-t', '--temp_path', type=str, required=True, help="Path to the output folder.")
    parser.add_argument('-o', '--output', type=str, required=True, help="Path to the output folder.")
    parser.add_argument('-p', '--percentage', type=float, default=100, help="Percentage of sequences to select.")
    parser.add_argument('-s', '--seq_length', type=int, default=21, help="Percentage of sequences to select.")
    return parser.parse_args()


def generate_combinations_and_individuals(seq):
    """
    Generate individual molecules and their combinations from a sequence.

    Args:
        seq (str): The input sequence.

    Returns:
        list: A list of individual molecules and their combinations.
    """
    if seq.endswith(";"):
        seq  = seq.rstrip(';')
    molecules = seq.split()
    result = [molecule + ";" for molecule in molecules]
    for r in range(2, len(molecules) + 1):
        result.extend([" ".join(perm) + ";" for perm in permutations(molecules, r)])
    return result


def main(seed=1234):
    """Main function for preprocessing and saving the dataset."""
    # Parse arguments and set random seed for reproducibility
    args = parse_args()
    random.seed(seed)

    # Load dataset
    dataset = pd.read_parquet(args.input)
    sequences = dataset['sequence'].unique().tolist()

    # Expand sequences to include combinations
    expanded_sequences = []
    for seq in sequences:
        if ' ' in seq:
            expanded_sequences.extend(generate_combinations_and_individuals(seq))
        else:
            if not seq.endswith(";"):
                seq += ";"
            expanded_sequences.append(seq)

    expanded_sequences = list(set(expanded_sequences))
    print(f"Expanded {len(sequences)} sequences to {len(expanded_sequences)}.")

    # Sample sequences based on the specified percentage
    selected_sequences = random.sample(expanded_sequences, int(len(expanded_sequences) * args.percentage / 100))
    print(f"Selected {len(selected_sequences)} sequences out of {len(expanded_sequences)}.")

    # Encode sequences and split into train, val, and test datasets
    token_to_idx = {token: idx for idx, token in enumerate(set("".join(selected_sequences)))}
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}
    encoded_sequences = [[token_to_idx[char] for char in seq] for seq in selected_sequences]

    train_size = int(0.8 * len(encoded_sequences))
    val_size = int(0.1 * len(encoded_sequences))
    test_size = len(encoded_sequences) - train_size - val_size
    train_data, val_data, test_data = random_split(encoded_sequences, [train_size, val_size, test_size])

    print(f"Train data size: {len(train_data)}; Val data size: {len(val_data)}; Test data size: {len(test_data)}.")

    # Create dataset instances
    train_dataset = AminoAcidDataset(train_data, args.seq_length)
    val_dataset = AminoAcidDataset(val_data, args.seq_length)
    test_dataset = AminoAcidDataset(test_data, args.seq_length)

    print(f"Train dataset size: {len(train_dataset)}; Val dataset size: {len(val_dataset)}; Test dataset size: {len(test_dataset)}.")

    # Prepare data for saving
    input_basename = os.path.splitext(os.path.basename(args.input))[0]
    data_to_save = {
        "train_data": train_dataset,
        "val_data": val_dataset,
        "test_data": test_dataset,
        "token_to_idx": token_to_idx,
        "idx_to_token": idx_to_token,
        "percentage": args.percentage,
        "source": input_basename,
        "temp_path": args.temp_path,
        "output_path": args.output
    }

    # Save the preprocessed dataset
    output_path = os.path.join(args.temp_path, f"{input_basename}_{args.percentage}_P_{args.seq_length}_L.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(data_to_save, f)

    print(f"Preprocessed dataset saved to {output_path}.")


if __name__ == "__main__":
    main()
