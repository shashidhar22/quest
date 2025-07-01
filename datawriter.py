#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_writer_tokeniser.py  *map-only version*
──────────────────────────────────────────────
• Reads Parquet shards (local or s3://)
• Splits 80/10/10 → train / val / test
• Two branches controlled by --model-name

  1.  bert | protbert | esm | llama
        explode → HF tokenizer → DatasetDict

  2.  custom-rnn | custom-transformer
        ▸ train a BPE tokenizer on **train** split
        ▸ tag sequences with custom boundary tokens
        ▸ encode via AminoAcidDataset **inside datasets.map**
        ▸ returns DatasetDict with {input_ids,target_ids}

No manual pyarrow writing — all handled by hf dataset `map`.
"""

import os, argparse, multiprocessing as mp
from itertools import combinations
from pathlib  import Path
from typing   import List, Dict

import pandas as pd
import pyarrow.dataset as pds
from datasets import (load_dataset, concatenate_datasets, 
                      Dataset, DatasetDict, Features, Value)
from datasets.data_files import DataFilesList
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from tqdm.auto import tqdm
import s3fs, torch
from tokenizers import Tokenizer, pre_tokenizers, trainers
from torch.utils.data import Dataset as TorchDataset
from concurrent.futures import ThreadPoolExecutor

# -- model cards ------------------------------------------------
MODEL_CARDS = {
    "bert": "google-bert/bert-base-uncased",
    "protbert": "Rostlab/prot_bert",
    "esm": "EvolutionaryScale/esm3-sm-open-v1",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "lstm": "rnn",
    "transformer": "tranformer",
}

# ── universal constants ────────────────────────────────────────────────
FIELDS = ["tra","trb","peptide","mhc_one","mhc_two"]
START_STYLE = {m: (lambda s: "[CLS] "+s) for m in ("bert","protbert")}|{"esm":lambda s:s,"llama":lambda s:s}
JOIN_STYLE  = {m: (lambda t: " [SEP] ".join(t)) for m in ("bert","protbert")}|{"esm":lambda t:" ".join(t),"llama":lambda t:" ".join(t)}
END_STYLE   = {m: (lambda s: f"{s} [SEP]") for m in ("bert","protbert")}|{"esm":lambda s:s,"llama":lambda s:s}

# BPE boundary tokens ----------------------------------------------------
BPE_TOKENS = {
    "pad_token": ["[PAD]"], "unk_token": ["[UNK]"], "end_token": ["[END]"],
    "tra_tokens": ["[TRA]", "[ETRA]"], "trb_tokens": ["[TRB]", "[ETRB]"],
    "pep_tokens": ["[PEP]", "[EPEP]"], "mho_tokens": ["[MHO]", "[EMHO]"],
    "mht_tokens": ["[MHT]", "[EMHT]"],
}
PAD_TOKEN_STR = "[PAD]"; PAD_TOKEN_ID = 0

# ── helper: ProtBERT‑style explode ─────────────────────────────────────-

def debug_row(dataset, index):
    """
    Prints the content and type of key fields for a specific row index.
    """
    print(f"\n--- 🕵️  Debugging Row at Index: {index} ---")
    try:
        example = dataset[index]
        present_count = 0
        for field in ["tra", "trb", "peptide", "mhc_one", "mhc_two"]:
            value = example.get(field)
            is_valid_str = isinstance(value, str) and value and value != "NA"

            print(f"  - Field: '{field}'")
            print(f"    Value: {repr(value)}")
            print(f"    Type: {type(value)}")
            print(f"    Passes Check: {is_valid_str}")
            if is_valid_str:
                present_count += 1
        print(f"--- Total Valid Features Found in this Row: {present_count} ---")
    except IndexError:
        print(f"--- Error: Index {index} is out of bounds for this dataset ---")
        
        
def explode_example(ex, join_fn, start_fn, end_fn, model_name="bert"):
    # Step 1: Gather all valid, non-empty features from the input row.
    present_features = []
    for f in FIELDS:
        val = ex.get(f)
        if isinstance(val, str) and val and val != "NA":
            present_features.append((val, f))

    # If there are no valid features in this row, exit early.
    if not present_features:
        return {"combo_id": [], "combo_feats": []}

    # Step 2: Generate ALL possible combinations first.
    generated_ids = []
    generated_feats = []
    for r in range(1, len(present_features) + 1):
        for combo in combinations(present_features, r):
            vals, fs = zip(*combo) # fs is a tuple like ('tra', 'trb')
            vals = [val.split('+')[0].strip() if '+' in val else val for val in vals]
            if model_name == "protbert":
                vals = tuple(' '.join(list(v)) for v in vals)

            generated_ids.append(end_fn(start_fn(join_fn(vals))))
            generated_feats.append(fs)

    # Step 3: Now, in a separate, clear step, filter out the results we don't want.
    final_ids = []
    final_feats = []
    for i, feats_tuple in enumerate(generated_feats):
        # The only condition we want to filter is where the combo is ONLY the peptide
        if feats_tuple == ('peptide',):
            continue  # Skip this one

        # Otherwise, keep the generated combination
        final_ids.append(generated_ids[i])
        final_feats.append(feats_tuple)
    return {"combo_id": final_ids, "combo_feats": final_feats}
# ── helpers for custom BPE path ─────────────────────────────────────────

_valid_aa = set("ACDEFGHIKLMNPQRSTVWY")

def is_valid_sequence(seq:str):
    return isinstance(seq,str) and seq and all(c in _valid_aa for c in seq)

def safe_get(row,key):
    v=row.get(key,"")
    return "" if v in (None,"NA") or (isinstance(v,float) and pd.isna(v)) else v

def tag_bpe(row):
    mapping=[("tra","tra_tokens"),("trb","trb_tokens"),("peptide","pep_tokens"),
             ("mhc_one","mho_tokens"),("mhc_two","mht_tokens")]
    tagged=[]
    for col,tkkey in mapping:
        val=safe_get(row,col)
        if val and is_valid_sequence(val):
            start_tk,end_tk=BPE_TOKENS[tkkey]
            tagged.append(f"{start_tk} {' '.join(val)} {end_tk}")
    return tagged

def train_bpe_tokenizer(seqs:List[str], vocab_size:int):
    tok=Tokenizer(models.BPE())
    tok.pre_tokenizer=pre_tokenizers.Split(pattern=r" ", behavior="removed")
    special=[t for pair in BPE_TOKENS.values() for t in pair if t]
    tok.train_from_iterator(seqs, trainers.BpeTrainer(special_tokens=special,vocab_size=vocab_size))
    return tok

class AminoAcidDataset(TorchDataset):
    def __init__(self, sequences, tokenizer, seq_len=128, model="rnn", step=1, tr_long=True):
        self.tok, self.seq_len=tokenizer,seq_len; self.model=model; self.step=step; self.tr_long=tr_long
        self.pad_id=tokenizer.token_to_id(PAD_TOKEN_STR) or PAD_TOKEN_ID; self.samples=[]; self._build(sequences)
    def _build(self, seqs):
        return self._build_rnn(seqs) if self.model=="rnn" else self._build_tx(seqs)
    def _build_rnn(self, seqs):
        for s in seqs:
            ids=self.tok.encode(s).ids
            if len(ids)>=self.seq_len+1:
                n=(len(ids)-(self.seq_len+1))//self.step+1
                for i in range(n):
                    chunk=ids[i*self.step:i*self.step+self.seq_len+1]
                    self.samples.append((chunk[:-1],chunk[1:]))
            else:
                pad=ids+[self.pad_id]*((self.seq_len+1)-len(ids))
                self.samples.append((pad[:-1],pad[1:]))
    def _build_tx(self, seqs):
        for s in seqs:
            ids=self.tok.encode(s).ids
            if len(ids)>self.seq_len and self.tr_long:
                ids=ids[:self.seq_len]
            if len(ids)<self.seq_len:
                ids+=[self.pad_id]*(self.seq_len-len(ids))
            if len(ids)>=2:
                self.samples.append((ids[:-1],ids[1:]))
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        x,y=self.samples[i]; return torch.tensor(x),torch.tensor(y)
        
def load_single_dataset(data_file, storage_options):
    """
    Worker function to load a single dataset file. This will be executed
    in parallel by the ThreadPoolExecutor.
    """
    return load_dataset(
        "parquet",
        data_files=str(data_file),
        split="train",
        storage_options=storage_options
    )
    
# ─── Worker function for MLM masking ─────────────────────────────────────-
def apply_masking_to_batch(batch, collator):
    list_of_examples = [{k: v[i] for k, v in batch.items() if k in ("input_ids", "attention_mask")} for i in range(len(batch["input_ids"]))]
    collated_batch = collator(list_of_examples)
    return collated_batch
    
def encode_bpe_batch(batch,*,tokenizer,seq_len,model_type,trunc_long):
    out_in,out_tgt=[],[]
    for tagged_seq in batch["tagged"]:
        ds=AminoAcidDataset([tagged_seq],tokenizer,seq_len,model_type,tr_long=trunc_long)
        for x,y in ds:
            out_in.append(x.tolist()); out_tgt.append(y.tolist())
    return {"input_ids":out_in,"target_ids":out_tgt}

# ── CLI -----------------------------------------------------------------

def cli():
    p=argparse.ArgumentParser()
    p.add_argument("--path",nargs="+",required=True)
    p.add_argument("--model-name",required=True,
                   choices=["bert","protbert","esm","llama","lstm","transformer"])
    p.add_argument("--output-raw", required=True, help="Path to save the tokenized dataset with text columns.")
    p.add_argument("--output-processed", required=True, help="Path to save the final dataset with MLM labels and no text.")

    p.add_argument("--max-len",type=int,default=1024)
    p.add_argument("--bpe-vocab",type=int,default=200)
    p.add_argument("--truncate-long",action="store_true")
    p.add_argument("--mlm-prob",type=float,default=0.15,
                   help="Probability of masking tokens in MLM. Default is 0.15.")
    p.add_argument("--s3-key",default=os.getenv("AWS_ACCESS_KEY_ID"))
    p.add_argument("--s3-secret",default=os.getenv("AWS_SECRET_ACCESS_KEY"))
    p.add_argument("--s3-token",default=os.getenv("AWS_SESSION_TOKEN"))
    return p.parse_args()

# ── MAIN ----------------------------------------------------------------

def main():
    args=cli()
    is_remote=any(p.startswith("s3://") for p in args.path)
    s3_options = {
        "key": args.s3_key,
        "secret": args.s3_secret,
        "token": args.s3_token,
    } if is_remote else None

    num_cores = max(1, os.cpu_count() // 4)

    # 1. Expand the glob pattern to get a definitive list of all data files/dirs.
    #    `args.path` is a list of patterns, e.g., ["s3://.../**/*.parquet"]
    print("Resolving data files from glob pattern...")
    resolved_data_files = DataFilesList.from_patterns(args.path)
    print(f"Found {len(resolved_data_files)} individual dataset files/directories.")

    # 2. Load each file/dir as a separate Dataset object.
    #    This allows each one to have its own unique schema initially.
    all_datasets = []
    # Use a ThreadPoolExecutor to manage a pool of threads for I/O tasks.
    # The `with` statement ensures threads are properly closed.
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        # Create a partial function to pass the constant `s3_options` to the worker
        from functools import partial
        worker_fn = partial(load_single_dataset, storage_options=s3_options)
        # executor.map applies the worker function to each file in parallel.
        # We wrap it in tqdm for a progress bar and list() to gather all results.
        print(f"Loading files in parallel with up to {executor._max_workers} workers...")
        all_datasets = list(
            tqdm(
                executor.map(worker_fn, resolved_data_files),
                total=len(resolved_data_files),
                desc="Loading files in parallel"
            )
        )
    # 3. Concatenate the list of datasets into one.
    #    This function  unifies the schemas.
    print("Concatenating datasets and unifying schemas...")
    base = concatenate_datasets(all_datasets)
    print("Finished concatenating.")


    # split 80/10/10
    step1=base.train_test_split(test_size=0.10,shuffle=True,seed=42)
    tr_val=step1["train"].train_test_split(test_size=0.111111,shuffle=True,seed=42)
    splits={"train":tr_val["train"],"val":tr_val["test"],"test":step1["test"]}

    out_raw = Path(args.output_raw)
    out_processed = Path(args.output_processed)
    out_raw.mkdir(parents=True, exist_ok=True)
    out_processed.mkdir(parents=True, exist_ok=True)

    model_name=MODEL_CARDS[args.model_name]
    print(f"Using model {model_name} for tokenization")
    # ------------------------------------------------------------------
    if args.model_name in {"bert","protbert","esm","llama"}:
        join,start,end=JOIN_STYLE[args.model_name],START_STYLE[args.model_name],END_STYLE[args.model_name]
       
        hf_tok=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
        def tok_fn(batch, tokenizer, max_len):
            """
            This function performs two actions in one pass:
            1. Flattens the malformed `combo_id` column.
            2. Tokenizes the clean, flattened data.
            3. Overwrites the old `combo_id` with the clean version.
            """
            # batch["combo_id"] is a list of lists, e.g., [['str_a'], ['str_b']]
            # Flatten it, ensuring we handle potential empty lists gracefully.
            clean_ids = [item[0] for item in batch["combo_id"] if item]

            # Tokenize the clean list of strings
            tokenized_output = tokenizer(clean_ids, truncation=True, max_length=max_len)

            # Add the clean, flat list back into the output. This will
            # OVERWRITE the old, messy `combo_id` column.
            tokenized_output["combo_id"] = clean_ids

            return tokenized_output
        def proc(ds, split):
            """
            Manually explodes the dataset to ensure each feature combination
            becomes its own unique row, then tokenizes the result.
            """
            # Step 1: Manually build the exploded lists in Python
            all_combo_ids = []
            all_combo_feats = []
            print(f"Manually exploding {len(ds):,} rows for the '{split}' split...")
            for example in tqdm(ds, desc=f"Exploding {ds.split}"):
                exploded_data = explode_example(example, join, start, end, model_name=args.model_name)
                all_combo_ids.extend(exploded_data["combo_id"])
                all_combo_feats.extend(exploded_data["combo_feats"])

            # Step 2: Create a new, correctly structured dataset from the lists
            exploded_dataset = Dataset.from_dict({
                "combo_id": all_combo_ids,
                "combo_feats": all_combo_feats
            })
            print(f"Explosion complete. New size for '{split}': {len(exploded_dataset):,} rows")

            # Step 3: Tokenize the new, clean dataset
            def tokenize_clean_batch(batch):
                return hf_tok(batch["combo_id"], truncation=True, max_length=args.max_len)

            tokenized_dataset = exploded_dataset.map(
                tokenize_clean_batch,
                batched=True,
                batch_size=2000,
                num_proc=num_cores,
            )
            return tokenized_dataset
        # 1. Process all splits to get the "raw" tokenized data
        print("Step 1: Generating 'raw' tokenized dataset with text columns...")
        raw_final = {n: proc(ds, n) for n, ds in splits.items()}
        raw_dataset_dict = DatasetDict(raw_final)
        
    
    else:  # custom BPE -------------------------------------------------
        # train tokenizer on train split
        train_tag=[s for row in splits["train"].map(lambda r:{"tag":tag_bpe(r)},num_proc=max(1,mp.cpu_count()-2))["tag"] for s in row]
        bpe_tok=train_bpe_tokenizer(train_tag, vocab_size=args.bpe_vocab)
        def proc(ds):
            # tag each row
            tagged=ds.map(lambda r:{"tagged":tag_bpe(r)},remove_columns=[c for c in ds.column_names if c not in FIELDS],
                         num_proc=num_cores)
            return tagged.map(
                encode_bpe_batch,
                batched=True,
                batch_size=1000,
                writer_batch_size=1000,
                fn_kwargs=dict(tokenizer=bpe_tok,seq_len=args.max_len,model_type=model_name,trunc_long=args.truncate_long),
                remove_columns=["tagged"],
                num_proc=num_cores,
                desc="encode")
        raw_final = {n: proc(ds, n) for n, ds in splits.items()}
        raw_dataset_dict = DatasetDict(raw_final)

    

    # 2. Save the "raw" dataset
    print(f"💾 Saving raw tokenized dataset to {out_raw}")
    raw_dataset_dict.save_to_disk(str(out_raw))

    # 3. Create the "processed" dataset for training/evaluation
    print("Step 2: Creating 'processed' dataset with MLM labels...")
    processed_dataset_dict = raw_dataset_dict.filter(lambda x: True, num_proc=num_cores) # A simple way to copy

    # Create the MLM collator for masking
    mlm_collator = DataCollatorForLanguageModeling(
        tokenizer=hf_tok, mlm=True, mlm_probability=args.mlm_prob
    )
    masking_fn = partial(apply_masking_to_batch, collator=mlm_collator)

    # Apply masking to val and test splits
    for split_name in ["val", "test"]:
        print(f"Applying MLM masking to '{split_name}' split...")
        processed_dataset_dict[split_name] = processed_dataset_dict[split_name].map(
            masking_fn,
            batched=True,
            batch_size=1024,
            num_proc=num_cores
        )
    
    # 4. Remove text columns from all splits for the final processed version
    print("Removing text columns from processed dataset...")
    text_columns_to_remove = ["combo_id", "combo_feats", "token_type_ids"]
    for split_name in ["train", "val", "test"]:
         processed_dataset_dict[split_name] = processed_dataset_dict[split_name].remove_columns(text_columns_to_remove)

    # 5. Save the final "processed" dataset
    print(f"💾 Saving processed dataset to {out_processed}")
    processed_dataset_dict.save_to_disk(str(out_processed))
    
    print("✅ All tasks complete.")


if __name__=="__main__":
    main()
