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
from concurrent.futures import ProcessPoolExecutor

from datasets import DatasetDict, load_dataset
from datasets import Features, Sequence, Value

# Transformers for BERT-based tokenizer
from transformers import AutoTokenizer  # NEW import

# Custom dataset logic for amino acid sequences
from quest.dataset import AminoAcidDataset

# Constants
from quest.constants import (
    BPE_TOKENS, PROT_BERT_TOKENS, BERT_TOKENS,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Dataloader with incremental arrow writing using AminoAcidDataset."
    )
    parser.add_argument("-i", "--input", type=str, required=True, nargs='+',
                        help="Directory(ies) containing .parquet files.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output folder for final HF dataset.")
    parser.add_argument("-p", "--percentage", type=float, default=100,
                        help="Percentage of expansions to keep per chunk (0-100).")
    parser.add_argument("-s", "--seq_length", type=int, default=21,
                        help="Max sequence length for model training.")
    parser.add_argument("--batch_size", type=int, default=500000,
                        help="Rows per Parquet chunk.")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed.")
    parser.add_argument("-m", "--model_type", type=str, default="transformer",
                        choices=["transformer", "rnn", "protbert", "bert", "esm", "llama"],
                        help="Transformer or RNN mode in AminoAcidDataset.")
    parser.add_argument("--truncate_long_sequences", action="store_true",
                        help="Truncate sequences longer than seq_length (transformer mode).")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Fraction of expansions in train split.")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of expansions in val split (test is remainder).")
    parser.add_argument("--build_tokenizer", action="store_true", default=False,
                        help="If true, train a new tokenizer from the expansions in the train set.")
    parser.add_argument("-v", "--vocab_size", type=int, default=200,
                        help="Vocabulary size for BPE tokenizer.")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Number of parallel processes (defaults to all cores).")
    return parser.parse_args()

def gather_parquet_files(paths):
    """Collect all *.parquet files from each directory in `paths`."""
    all_files = []
    for path in paths:
        found = glob.glob(os.path.join(path, '*.parquet'))
        all_files.extend(found)
    return all_files

def is_valid_sequence(seq):
    if not isinstance(seq, str) or not seq:
        return False
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    return all(aa in valid_aas for aa in seq)

def safe_get(row, key):
    val = getattr(row, key, "")
    if pd.isna(val) or val == "NA":
        return ""
    return val

def tag_bpe(row):
    """Convert relevant columns into a list of tagged sub-sequences."""
    mapping = [
        ("tra", BPE_TOKENS["tra_tokens"]),
        ("trb", BPE_TOKENS["trb_tokens"]),
        ("peptide", BPE_TOKENS["pep_tokens"]),
        ("mhc_one", BPE_TOKENS["mho_tokens"]),
        ("mhc_two", BPE_TOKENS["mht_tokens"]),
    ]
    tagged = []
    for col, tokens in mapping:
        start_tk, end_tk = tokens
        val = safe_get(row, col)
        if val and is_valid_sequence(val):
            if 'O' in val or '[' in val or ']' in val:
                breakpoint()
            val = " ".join(list(val))
            tagged.append(f"{start_tk} {val} {end_tk}")
    return tagged

def tag_protbert(row):
    tagged = []
    sep_tk = PROT_BERT_TOKENS["sep_token"][0]
    for col in ["tra", "trb", "peptide", "mhc_one", "mhc_two"]:
        val = safe_get(row, col)
        if val and is_valid_sequence(val):
            val = " ".join(list(val))
            tagged.append(f"{val} {sep_tk}")
    return tagged

def tag_bert(row):
    tagged = []
    sep_tk = BERT_TOKENS["sep_token"][0]
    for col in ["tra", "trb", "peptide", "mhc_one", "mhc_two"]:
        val = safe_get(row, col)
        if val and is_valid_sequence(val):
            val = " ".join(list(val))
            tagged.append(f"{val} {sep_tk}")
    return tagged

def tag_esm2(row):

def generate_permutations_of_tagged(tagged, model_tags):
    """Generate permutations of the tagged sequences."""
    if not tagged:
        return []
    results = []
    n = len(tagged)
    cls_token = model_tags["cls_token"][0]
    end_token = model_tags["end_token"][0]
    for r in range(1, n + 1):
        for perm in permutations(tagged, r):
            sequence = " ".join(perm)
            if cls_token:
                sequence = f"{cls_token} " + sequence
            if end_token:
                sequence += f" {end_token}"
            results.append(sequence)
    return results

def process_chunk(df_chunk, model_type="rnn"):
    """
    Expand a single chunk:
      - row_to_tagged_list()
      - generate_permutations_of_tagged()
    Return a list of raw text expansions (strings).
    """
    if "tra" in df_chunk.columns:
        df_chunk.loc[df_chunk["tra"].str.len() < 8, "tra"] = pd.NA
    if "trb" in df_chunk.columns:
        df_chunk.loc[df_chunk["trb"].str.len() < 8, "trb"] = pd.NA
    if "peptide" in df_chunk.columns:
        df_chunk.loc[df_chunk["peptide"].str.len() < 4, "peptide"] = pd.NA
        
        # Identify rows that have *only* peptide non-null
        other_cols = [c for c in df_chunk.columns if c != "peptide"]
        mask_peptide_only = df_chunk["peptide"].notna() & df_chunk[other_cols].isna().all(axis=1)
        
        # Drop those rows
        df_chunk = df_chunk[~mask_peptide_only]

    local_expansions = []
    for row in df_chunk.itertuples(index=False):
        if model_type == "rnn" or model_type == "transformer":
            tagged = tag_bpe(row)
            model_tags = BPE_TOKENS
        elif model_type == "protbert":
            # For ProtBERT, we use a different tagging scheme
            tagged = tag_protbert(row)
            model_tags = PROT_BERT_TOKENS
        elif model_type == "bert":
            # For BERT, we use a different tagging scheme
            tagged = tag_bert(row)
            model_tags = BERT_TOKENS
        #elif model_type == "esm":
        #    # For ESM, we use a different tagging scheme
        #    tagged = tag_esm(row)
        #elif model_type == "llama":
        #    # For Llama, we use a different tagging scheme
        #    tagged = tag_llama(row)
        if tagged:
            perms = generate_permutations_of_tagged(tagged, model_tags)
            local_expansions.extend(perms)
    return local_expansions


def train_bpe_tokenizer(train_sequences, vocab_size=200):
    # 1) Pre-check pass (optional)
    debug_pretok = pre_tokenizers.Split(pattern=r" ", behavior="removed")
    # 2) Now train
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = debug_pretok
    special_tokens = [token for tklist in BPE_TOKENS.values() for token in tklist if token is not None]
    trainer = trainers.BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size)

    tokenizer.train_from_iterator(train_sequences, trainer)
    return tokenizer

def chunked_store_raw_expansions(expansions, arrow_dir, shard_prefix, shard_index):
    """
    Stores the raw expansions (strings) in a single 'sequence' column in Arrow.
    """
    # expansions is a list of strings
    arrow_rows = []
    for seq in expansions:
        arrow_rows.append({
            "sequence": seq
        })

    table = pa.Table.from_pylist(arrow_rows, schema=pa.schema([
        pa.field("sequence", pa.string())
    ]))

    os.makedirs(arrow_dir, exist_ok=True)
    shard_path = os.path.join(arrow_dir, f"{shard_prefix}_{shard_index}.arrow")
    with pa.OSFile(shard_path, "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)

    return len(arrow_rows)

def chunked_encode_and_write(
    expansions, tokenizer, seq_length, model_type, truncate_long_sequences,
    arrow_dir, shard_prefix, shard_index
):
    """
    1) Build an AminoAcidDataset from expansions.
    2) Iterate over that dataset to produce (input_ids, target_ids).
    3) Write them to an Arrow file: arrow_dir/shard_prefix_{shard_index}.arrow
    """
    # Build the custom dataset
    ds = AminoAcidDataset(
        sequences=expansions,
        tokenizer=tokenizer,
        seq_length=seq_length,
        model_type=model_type,
        step=1,
        truncate_long_sequences=truncate_long_sequences
    )

    if not ds:  # empty dataset
        return 0

    # Convert each (x, y) to a dictionary for Arrow
    arrow_rows = []
    for i in range(len(ds)):
        input_tensor, target_tensor = ds[i]  # each is a torch.Tensor
        arrow_rows.append({
            "input_ids": input_tensor.tolist(),
            "target_ids": target_tensor.tolist()
        })

    # Create a PyArrow table
    table = pa.Table.from_pylist(arrow_rows, schema=pa.schema([
        pa.field("input_ids", pa.list_(pa.int32())),
        pa.field("target_ids", pa.list_(pa.int32()))
    ]))

    # Write to arrow file
    os.makedirs(arrow_dir, exist_ok=True)
    shard_path = os.path.join(arrow_dir, f"{shard_prefix}_{shard_index}.arrow")
    with pa.OSFile(shard_path, "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)

    return len(ds)

def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    # 1) Gather parquet files
    parquet_files = gather_parquet_files(args.input)
    if not parquet_files:
        raise ValueError(f"No .parquet files found in {args.input} directories.")

    # 2) Arrow directories for each split to store chunked shards
    train_dir = os.path.join(args.output, "train_shards")
    val_dir   = os.path.join(args.output, "val_shards")
    test_dir  = os.path.join(args.output, "test_shards")

    # 3) Count total rows for progress
    ds_ = ds.dataset(parquet_files, format="parquet")
    total_rows_est = ds_.count_rows()

    # 4) Parallel chunk expansions
    parquet_dataset = ds.dataset(parquet_files, format="parquet")
    scanner = parquet_dataset.scanner(batch_size=args.batch_size)

    pbar = tqdm(total=total_rows_est, desc="Reading & Permutations")
    futures = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        for record_batch in scanner.to_batches():
            df_chunk = pa.Table.from_batches([record_batch]).to_pandas()
            fut = executor.submit(process_chunk, df_chunk, model_type=args.model_type)
            futures.append((fut, len(df_chunk)))

        expansions_buffer = []
        for chunk_idx, (fut, chunk_size) in enumerate(futures):
            expansions = fut.result()
            pbar.update(chunk_size)

            # 4a) If 0<percentage<100, sample expansions
            if 0 < args.percentage < 100:
                sample_size = int(len(expansions) * (args.percentage / 100.0))
                expansions = random.sample(expansions, sample_size)

            # 4b) Shuffle expansions and split them
            random.shuffle(expansions)
            n = len(expansions)
            train_end = int(n * args.train_ratio)
            val_end   = int(n * (args.train_ratio + args.val_ratio))

            chunk_train = expansions[:train_end]
            chunk_val   = expansions[train_end:val_end]
            chunk_test  = expansions[val_end:]

            # We store train expansions if we need to build tokenizer from them later
            expansions_buffer.extend(chunk_train)

        pbar.close()

    # 5) Build or load tokenizer
    tokenizer_path = os.path.join(args.output, "tokenizer.json")


    if args.model_type == "rnn" or args.model_type == "transformer":  # "bpe"
        if not os.path.exists(tokenizer_path) or args.build_tokenizer:
            print("Building a new BPE tokenizer from the collected train expansions...")
            tokenizer = train_bpe_tokenizer(expansions_buffer, vocab_size=args.vocab_size)
            tokenizer.save(tokenizer_path)
            print(f"BPE tokenizer saved at {tokenizer_path}")
        else:
            print("Tokenizer file found. Loading existing BPE tokenizer...")
            tokenizer = Tokenizer.from_file(tokenizer_path)

    # 6) We must now re-process the chunks *again*, but this time we have the tokenizer.
    #    Because we never wrote expansions out to disk in step 4, we must do it once more
    #    to produce the final arrow shards with (input_ids, target_ids).
    #    To avoid reading the entire expansions into memory, let's do chunk expansions again, 
    #    but now we do on-the-fly encoding & arrow writing for each chunk.

    # We'll repeat the scanning logic, but each chunk is *immediately* turned into 
    # train/val/test arrow shards (input_ids + target_ids).

    print("Second pass: building final arrow shards with tokenized (input_ids, target_ids).")
    # Reset the scanner
    parquet_dataset = ds.dataset(parquet_files, format="parquet")
    scanner2 = parquet_dataset.scanner(batch_size=args.batch_size)

    # We'll track how many shards we write per split
    train_shard_count = 0
    val_shard_count   = 0
    test_shard_count  = 0

    # In parallel or single-thread? Typically single-thread is simpler 
    # because we must write arrow files from each chunk. 
    # We'll do single-thread to avoid collisions writing shards. 
    # (One can do it with e.g. locked counters, but let's keep it simpler.)

    pbar2 = tqdm(total=total_rows_est, desc="Second Pass: Tokenizing & Writing Shards")
    for record_batch in scanner2.to_batches():
        df_chunk = pa.Table.from_batches([record_batch]).to_pandas()
        expansions = process_chunk(df_chunk, model_type=args.model_type)  # expansions is a list of raw text
        pbar2.update(len(df_chunk))

        # sample expansions
        if 0 < args.percentage < 100:
            sample_size = int(len(expansions) * (args.percentage / 100.0))
            expansions = random.sample(expansions, sample_size)

        # split
        random.shuffle(expansions)
        n = len(expansions)
        train_end = int(n * args.train_ratio)
        val_end   = int(n * (args.train_ratio + args.val_ratio))

        chunk_train = expansions[:train_end]
        chunk_val   = expansions[train_end:val_end]
        chunk_test  = expansions[val_end:]

        # 6a) Encode train chunk & write arrow
        if args.model_type == "rnn" or args.model_type == "transformer":  # "bpe"
            # ================ BPE branch: use your existing custom logic ================
            if chunk_train:
                n_train = chunked_encode_and_write(
                    expansions=chunk_train,
                    tokenizer=tokenizer,
                    seq_length=args.seq_length,
                    model_type=args.model_type,
                    truncate_long_sequences=args.truncate_long_sequences,
                    arrow_dir=train_dir,
                    shard_prefix="train",
                    shard_index=train_shard_count
                )
                train_shard_count += 1

            if chunk_val:
                n_val = chunked_encode_and_write(
                    expansions=chunk_val,
                    tokenizer=tokenizer,
                    seq_length=args.seq_length,
                    model_type=args.model_type,
                    truncate_long_sequences=args.truncate_long_sequences,
                    arrow_dir=val_dir,
                    shard_prefix="val",
                    shard_index=val_shard_count
                )
                val_shard_count += 1

            if chunk_test:
                n_test = chunked_encode_and_write(
                    expansions=chunk_test,
                    tokenizer=tokenizer,
                    seq_length=args.seq_length,
                    model_type=args.model_type,
                    truncate_long_sequences=args.truncate_long_sequences,
                    arrow_dir=test_dir,
                    shard_prefix="test",
                    shard_index=test_shard_count
                )
                test_shard_count += 1
        else:
            # ================ For pre-trained models: store raw expansions only ================
            if chunk_train:
                n_train = chunked_store_raw_expansions(
                    expansions=chunk_train,
                    arrow_dir=train_dir,
                    shard_prefix="train",
                    shard_index=train_shard_count
                )
                train_shard_count += 1

            if chunk_val:
                n_val = chunked_store_raw_expansions(
                    expansions=chunk_val,
                    arrow_dir=val_dir,
                    shard_prefix="val",
                    shard_index=val_shard_count
                )
                val_shard_count += 1

            if chunk_test:
                n_test = chunked_store_raw_expansions(
                    expansions=chunk_test,
                    arrow_dir=test_dir,
                    shard_prefix="test",
                    shard_index=test_shard_count
                )
                test_shard_count += 1
    pbar2.close()

    # 7) Now we have multiple shard files in train_dir, val_dir, test_dir.
    #    We'll load them into a huggingface DatasetDict.
    print("Loading final shards into a Hugging Face DatasetDict...")

    from datasets import load_dataset

    train_shards = sorted(glob.glob(os.path.join(train_dir, "train_*.arrow")))
    val_shards   = sorted(glob.glob(os.path.join(val_dir,   "val_*.arrow")))
    test_shards  = sorted(glob.glob(os.path.join(test_dir,  "test_*.arrow")))

    # load_dataset(..., split="train") merges them
    train_ds = load_dataset("arrow", data_files=train_shards, split="train")
    val_ds   = load_dataset("arrow", data_files=val_shards,   split="train")
    test_ds  = load_dataset("arrow", data_files=test_shards,  split="train")

    ds_dict = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds
    })

    # 8) Save final dataset
    ds_dict.save_to_disk(args.output)
    print("All done! Final dataset saved to:", args.output)

if __name__ == "__main__":
    main()
