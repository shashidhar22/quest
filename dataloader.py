import os
import argparse
import random
import glob
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from tqdm import tqdm
from itertools import permutations
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from torch.utils.data import random_split
from datasets import Dataset, DatasetDict

# Import your updated class with RNN/transformer logic
from quest.dataset import AminoAcidDataset

# Constants from quest.constants
from quest.constants import (
    TRA_START, TRB_START, PEP_START, MHC1_START, MHC2_START,
    END_TOKEN, PAD_TOKEN_STR, UNK_TOKEN_STR
)

# -----------------------------------------------------------------------------
# 1) Argument Parsing
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Load multiple Parquet files, generate permutations, sample, encode, and save as HF dataset."
    )
    parser.add_argument("-i", "--input", type=str, required=True, nargs='+',
                        help="Directory containing multiple .parquet files.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output folder where HF dataset + dicts will be stored.")
    parser.add_argument("-p", "--percentage", type=float, default=100,
                        help="Percentage of sequences to select from the expanded set.")
    parser.add_argument("-s", "--seq_length", type=int, default=21,
                        help="Max sequence length for model training.")
    parser.add_argument("--batch_size", type=int, default=500000,
                        help="Rows per chunk to read from Parquet.")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed.")

    # Additional arguments
    parser.add_argument("--model_type", type=str, default="transformer",
                        choices=["transformer", "rnn"],
                        help="Choose 'transformer' for entire-sequence causal LM, or 'rnn' for sliding-window samples.")
    parser.add_argument("--truncate_long_sequences", action="store_true",
                        help="Truncate sequences longer than seq_length in transformer mode.")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Fraction of data for training.")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of data for validation. Remainder is test.")
    parser.add_argument("--build_tokenizer", action="store_true", default=False,
                        help="Write a tokenizer")
    parser.add_argument("-v", "--vocab_size", type=int, default=200,
                        help="Total number of iterations used in BPE, limits the vocab size to 200 token by default (needs to be optimized)")
    return parser.parse_args()


def gather_parquet_files(paths):
    """
    Given a list of directory paths, gather all *.parquet files from each.
    """
    all_files = []
    for path in paths:
        found = glob.glob(os.path.join(path, '*.parquet'))
        all_files.extend(found)
    return all_files

# -----------------------------------------------------------------------------
# 2) Data Reading + Permutation Functions
# -----------------------------------------------------------------------------
def is_valid_sequence(seq):
    if not isinstance(seq, str) or not seq:
        return False
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    return all(aa in valid_aas for aa in seq)

def safe_get(row, key):
    val = row.get(key, "")
    if pd.isna(val):
        return ""
    return val

def row_to_tagged_list(row):
    mapping = [
        ("tra", TRA_START),
        ("trb", TRB_START),
        ("peptide", PEP_START),
        ("mhc_one", MHC1_START),
        ("mhc_two", MHC2_START),
    ]
    tagged = []
    for col, start_tk in mapping:
        val = safe_get(row, col)
        if val and is_valid_sequence(val):
            tagged.append(f"{start_tk} {val} {END_TOKEN}")
    return tagged

def generate_permutations_of_tagged(tagged):
    if not tagged:
        return []
    results = []
    n = len(tagged)
    for r in range(1, n + 1):
        for perm in permutations(tagged, r):
            results.append(" ".join(perm))
    return results

# -----------------------------------------------------------------------------
# 3) Tokenizer
# -----------------------------------------------------------------------------
def train_bpe_tokenizer(train_sequences, vocab_size=200):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    special_tokens = [
        PAD_TOKEN_STR, UNK_TOKEN_STR,
        TRA_START, TRB_START, PEP_START, MHC1_START, MHC2_START, END_TOKEN
    ]
    trainer = trainers.BpeTrainer(special_tokens=special_tokens, vocab_size = vocab_size)
    tokenizer.train_from_iterator(train_sequences, trainer)
    return tokenizer

# -----------------------------------------------------------------------------
# 4) High-level Pipeline Functions
# -----------------------------------------------------------------------------
def read_all_parquet_in_chunks(parquet_files, batch_size):
    """
    Yields dataframes in 'chunk_size' from all .parquet files in 'parquet_dir'.
    """
    dataset = ds.dataset(parquet_files, format="parquet")
    scanner = dataset.scanner(batch_size=batch_size)

    for record_batch in scanner.to_batches():
        df_chunk = pa.Table.from_batches([record_batch]).to_pandas()
        yield df_chunk

def build_all_sequences(parquet_files, batch_size):
    """
    Reads all Parquet in chunks, for each row generates permutations,
    and accumulates in a list. Returns that list of expansions.
    """
    all_sequences = []
    total_rows = 0

    # Count total rows to show progress bar
    ds_ = ds.dataset(parquet_files, format="parquet")
    total_rows_est = ds_.count_rows()

    with tqdm(total=total_rows_est, desc="Reading & permutations") as pbar:
        for df_chunk in read_all_parquet_in_chunks(parquet_files, batch_size):
            chunk_size = len(df_chunk)
            total_rows += chunk_size
            local_expansions = []
            for _, row in df_chunk.iterrows():
                tagged = row_to_tagged_list(row)
                if tagged:
                    perms = generate_permutations_of_tagged(tagged)
                    local_expansions.extend(perms)

            all_sequences.extend(local_expansions)
            pbar.update(chunk_size)
            pbar.set_postfix({
                "Rows": total_rows,
                "TotalSeqs": len(all_sequences),
            })

    return all_sequences

def sample_sequences(all_sequences, percentage):
    """
    Randomly sample a given percentage of sequences if 0<percentage<100
    """
    if 0 < percentage < 100:
        sample_size = int(len(all_sequences) * (percentage / 100.0))
        all_sequences = random.sample(all_sequences, sample_size)
    return all_sequences

def split_data(all_sequences, train_ratio, val_ratio, seed):
    """
    Shuffle + split into train/val/test based on ratios.
    """
    random.seed(seed)
    random.shuffle(all_sequences)
    total = len(all_sequences)
    train_end = int(total * train_ratio)
    val_end   = int(total * (train_ratio + val_ratio))

    train_texts = all_sequences[:train_end]
    val_texts   = all_sequences[train_end:val_end]
    test_texts  = all_sequences[val_end:]
    return train_texts, val_texts, test_texts

def convert_to_hf_dataset(torch_dataset):
    """
    Convert a PyTorch dataset returning (x, y) -> HF Dataset with 'input_ids' and 'target_ids'.
    """
    data = [torch_dataset[i] for i in range(len(torch_dataset))]
    input_ids, target_ids = [], []
    for (x, y) in data:
        input_ids.append(x.tolist())
        target_ids.append(y.tolist())
    return Dataset.from_dict({"input_ids": input_ids, "target_ids": target_ids})

# -----------------------------------------------------------------------------
# 5) Main Workflow
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    # 1) Collect all .parquet files from each path in args.input
    parquet_files = gather_parquet_files(args.input)
    if not parquet_files:
        raise ValueError(f"No .parquet files found in {args.input} directories.")

    # (A) Read + generate expansions
    all_sequences = build_all_sequences(parquet_files, args.batch_size)
    if not all_sequences:
        raise ValueError("No expansions generated from parquet.")

    # (B) Percentage-based sampling
    if 0 < args.percentage < 100:
        all_sequences = sample_sequences(all_sequences, args.percentage)
        print(f"After sampling {args.percentage}%, sequences = {len(all_sequences)}")

    # (C) Split data
    train_texts, val_texts, test_texts = split_data(
        all_sequences, args.train_ratio, args.val_ratio, args.seed
    )
    print(f"Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")

    # (D) Train tokenizer on train set only
    if not os.path.exists(os.path.join(args.output, "tokenizer.json")) or args.build_tokenizer:
        tokenizer = train_bpe_tokenizer(train_texts, args.vocab_size)
        tokenizer.save(os.path.join(args.output, "tokenizer.json"))
        print(f"Tokenizer saved to file: {os.path.join(args.output, "tokenizer.json")}")
    else:
        print("Skipping BPE")

    # (E) Build PyTorch datasets (RNN or Transformer mode)
    train_ds = AminoAcidDataset(
        sequences=train_texts,
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        model_type=args.model_type,
        step=1,
        truncate_long_sequences=args.truncate_long_sequences
    )
    val_ds = AminoAcidDataset(
        sequences=val_texts,
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        model_type=args.model_type,
        step=1,
        truncate_long_sequences=args.truncate_long_sequences
    )
    test_ds = AminoAcidDataset(
        sequences=test_texts,
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        model_type=args.model_type,
        step=1,
        truncate_long_sequences=args.truncate_long_sequences
    )
    print(f"train_ds: {len(train_ds)}, val_ds: {len(val_ds)}, test_ds: {len(test_ds)}")

    # (F) (Optional) Convert to Hugging Face dataset
    hf_train = convert_to_hf_dataset(train_ds)
    hf_val   = convert_to_hf_dataset(val_ds)
    hf_test  = convert_to_hf_dataset(test_ds)
    ds_dict  = DatasetDict({"train": hf_train, "validation": hf_val, "test": hf_test})
    ds_dict.save_to_disk(args.output)
    print("HF dataset saved.")

    print("All done!")

if __name__ == "__main__":
    main()
