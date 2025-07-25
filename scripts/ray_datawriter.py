 #!/usr/bin/env python
# -*- coding: utf-8 -*-
# type: ignore
"""
datawriter.py  *map-only version*
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

import os
import argparse
from itertools import combinations
from pathlib import Path
from typing import List, Dict, Any
import s3fs
from datasets import (load_dataset, Dataset, DatasetDict)
from datasets.data_files import DataFilesList
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch
from torch.utils.data import Dataset as TorchDataset
import subprocess
import ray
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from datasets import load_from_disk, concatenate_datasets  # type: ignore
import pandas as pd  # type: ignore



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
START_STYLE = {m: (lambda s: s) for m in ("bert","protbert")}|{"esm":lambda s:s,"llama":lambda s:s}  # type: ignore
JOIN_STYLE  = {m: (lambda t: " [SEP] ".join(t)) for m in ("bert","protbert")}|{"esm":lambda t:" ".join(t),"llama":lambda t:" ".join(t)}  # type: ignore
END_STYLE   = {m: (lambda s: s) for m in ("bert","protbert")}|{"esm":lambda s:s,"llama":lambda s:s}  # type: ignore

# BPE boundary tokens ----------------------------------------------------
BPE_TOKENS = {
    "pad_token": ["[PAD]"], "unk_token": ["[UNK]"], "end_token": ["[END]"],
    "tra_tokens": ["[TRA]", "[ETRA]"], "trb_tokens": ["[TRB]", "[ETRB]"],
    "pep_tokens": ["[PEP]", "[EPEP]"], "mho_tokens": ["[MHO]", "[EMHO]"],
    "mht_tokens": ["[MHT]", "[EMHT]"],
}
PAD_TOKEN_STR = "[PAD]"; PAD_TOKEN_ID = 0

# ── helper: ProtBERT‑style explode ─────────────────────────────────────-
        
def explode_example(ex: Dict[str, Any], join_fn: Any, start_fn: Any, end_fn: Any, model_name: str = "bert") -> Dict[str, List[str]]:
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

def safe_get(row: Dict[str, Any], key: str) -> str:
    v=row.get(key,"")
    return "" if v in (None,"NA") or (isinstance(v,float) and pd.isna(v)) else v

def tag_bpe(row: Dict[str, Any]) -> List[str]:
    mapping=[("tra","tra_tokens"),("trb","trb_tokens"),("peptide","pep_tokens"),
             ("mhc_one","mho_tokens"),("mhc_two","mht_tokens")]
    tagged=[]
    for col,tkkey in mapping:
        val=safe_get(row,col)
        if val and is_valid_sequence(val):
            start_tk,end_tk=BPE_TOKENS[tkkey]
            tagged.append(f"{start_tk} {' '.join(val)} {end_tk}")
    return tagged

def train_bpe_tokenizer(seqs: List[str], vocab_size: int) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Split(pattern=r" ", behavior="removed")
    special = [t for pair in BPE_TOKENS.values() for t in pair if t]
    trainer = trainers.BpeTrainer(special_tokens=special, vocab_size=vocab_size)
    tok.train_from_iterator(seqs, trainer)
    return tok

class AminoAcidDataset(TorchDataset):
    def __init__(self, sequences: List[str], tokenizer: Tokenizer, seq_len: int = 128, model: str = "rnn", step: int = 1, tr_long: bool = True):
        self.tok, self.seq_len=tokenizer,seq_len; self.model=model; self.step=step; self.tr_long=tr_long
        self.pad_id=tokenizer.token_to_id(PAD_TOKEN_STR) or PAD_TOKEN_ID; self.samples=[]; self._build(sequences)
    def _build(self, seqs: List[str]) -> None:
        return self._build_rnn(seqs) if self.model=="rnn" else self._build_tx(seqs)
    def _build_rnn(self, seqs: List[str]) -> None:
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
    def _build_tx(self, seqs: List[str]) -> None:
        for s in seqs:
            ids=self.tok.encode(s).ids
            if len(ids)>self.seq_len and self.tr_long:
                ids=ids[:self.seq_len]
            if len(ids)<self.seq_len:
                ids+=[self.pad_id]*(self.seq_len-len(ids))
            if len(ids)>=2:
                self.samples.append((ids[:-1],ids[1:]))
    def __len__(self) -> int: return len(self.samples)
    def __getitem__(self, i: int):
        x,y=self.samples[i]; return torch.tensor(x),torch.tensor(y)
        
def load_single_dataset(data_file: Any, storage_options: Dict[str, Any]):
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

    
def encode_bpe_batch(batch: Dict[str, Any], *, tokenizer: Tokenizer, seq_len: int, model_type: str, trunc_long: bool) -> Dict[str, List[List[int]]]:
    out_in,out_tgt=[],[]
    
    # Process each sequence in the batch
    for tagged_seq in batch["tagged"]:
        ds=AminoAcidDataset(tagged_seq,tokenizer,seq_len,model_type,tr_long=trunc_long)
        for x,y in ds:
            # Ensure all sequences have the same length
            x_list = x.tolist()
            y_list = y.tolist()
            
            # Pad or truncate to seq_len
            if len(x_list) < seq_len:
                x_list.extend([tokenizer.token_to_id(PAD_TOKEN_STR) or PAD_TOKEN_ID] * (seq_len - len(x_list)))
                y_list.extend([tokenizer.token_to_id(PAD_TOKEN_STR) or PAD_TOKEN_ID] * (seq_len - len(y_list)))
            elif len(x_list) > seq_len:
                x_list = x_list[:seq_len]
                y_list = y_list[:seq_len]
            
            out_in.append(x_list)
            out_tgt.append(y_list)
    
    # Ensure consistent batch size by padding with dummy examples if needed
    # This is a workaround for the ArrowInvalid error
    if len(out_in) == 0:
        # If no examples were generated, create dummy examples
        pad_id = tokenizer.token_to_id(PAD_TOKEN_STR) or PAD_TOKEN_ID
        dummy_input = [pad_id] * seq_len
        dummy_target = [pad_id] * seq_len
        out_in = [dummy_input]
        out_tgt = [dummy_target]
    
    return {"input_ids":out_in,"target_ids":out_tgt}

# ── CLI -----------------------------------------------------------------

def cli():
    p=argparse.ArgumentParser()
    p.add_argument("--path",nargs="+", help="Path to original Parquet files (for a full run).")
    p.add_argument("--model-name",required=True,
                   choices=["bert","protbert","esm","llama","lstm","transformer"])
    p.add_argument("--input_raw_dir", default=None, help="Optional: Path to an existing 'raw' dataset to start masking from.")
    p.add_argument("--output-raw", required=True, help="Path to save the tokenized dataset with text columns.")
    p.add_argument("--max-len",type=int,default=1024)
    p.add_argument("--bpe-vocab",type=int,default=200)
    p.add_argument("--truncate-long",action="store_true")
    p.add_argument("--mlm-prob",type=float,default=0.15,
                   help="Probability of masking tokens in MLM. Default is 0.15.")
    p.add_argument("--s3-key",default=os.getenv("AWS_ACCESS_KEY_ID"))
    p.add_argument("--s3-secret",default=os.getenv("AWS_SECRET_ACCESS_KEY"))
    p.add_argument("--s3-token",default=os.getenv("AWS_SESSION_TOKEN"))
    p.add_argument("--use-ray", action="store_true", help="Use Ray for distributed data loading and processing.")
    p.add_argument("--num-workers", type=int, default=8, help="Number of Ray workers (default: 8)")
    p.add_argument("--s3-output-path", type=str, default=None, help="S3 path to sync the output directory after processing.")
    return p.parse_args()

# Ray remote wrapper for data loading and processing
@ray.remote
def ray_process_and_save_dataset(data_file: Any, storage_options: Dict[str, Any], output_dir: str, model_name: str, max_len: int, worker_idx: int) -> str:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import os
    # Load the dataset
    ds = load_dataset(
        "parquet",
        data_files=str(data_file),
        split="train",
        storage_options=storage_options
    )
    # Tokenize/process as needed (example for BERT/protbert/esm/llama)
    model_hf_id = MODEL_CARDS.get(model_name, model_name)
    hf_tok = AutoTokenizer.from_pretrained(model_hf_id, trust_remote_code=True)
    def tokenize_clean_batch(batch):
        return hf_tok(batch["combo_id"], truncation=True, max_length=max_len)
    # If combo_id not present, just pass through
    if "combo_id" in ds.column_names:
        ds = ds.map(tokenize_clean_batch, batched=True)
    # Save to disk (each worker saves to a unique subdir)
    out_path = os.path.join(output_dir, f"processed_{worker_idx}")
    ds.save_to_disk(out_path)
    return out_path

# Ray remote function to extract sequences for BPE training
@ray.remote
def ray_extract_sequences(data_file: Any, storage_options: Dict[str, Any], model_name: str, worker_idx: int) -> List[str]:
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files=str(data_file), split="train", storage_options=storage_options)
    seqs = []
    if "combo_id" in ds.column_names:
        for ex in ds:
            seqs.extend(ex["combo_id"])
    return seqs

# Ray remote function to encode data using trained BPE tokenizer
@ray.remote
def ray_encode_with_bpe(data_file: Any, storage_options: Dict[str, Any], output_dir: str, bpe_tokenizer_path: str, max_len: int, worker_idx: int) -> str:
    
    ds = load_dataset("parquet", data_files=str(data_file), split="train", storage_options=storage_options)
    bpe_tokenizer = Tokenizer.from_file(bpe_tokenizer_path)
    def encode_bpe_batch(batch: Dict[str, Any]) -> Dict[str, List[List[int]]]:
        input_ids = [bpe_tokenizer.encode(seq).ids for seq in batch["combo_id"]]
        return {"input_ids": input_ids}
    if "combo_id" in ds.column_names:
        ds = ds.map(encode_bpe_batch, batched=True)
    out_path = os.path.join(output_dir, f"processed_{worker_idx}")
    ds.save_to_disk(out_path)
    return out_path

# ── MAIN ----------------------------------------------------------------

def main():
    args = cli()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    is_remote = any(p.startswith("s3://") for p in args.path)
    s3_options = {
        "key": args.s3_key, "secret": args.s3_secret, "token": args.s3_token
    }
    #num_cores = int(max(1, os.cpu_count() - 2) * 0.75)
    num_cores = 1
    print("Resolving data files...")
    resolved_data_files = DataFilesList.from_patterns(args.path)
    # Remove unused variable
    # is_remote = any(p.startswith("s3://") for p in args.path)
    if args.use_ray:
        if not ray.is_initialized():
            ray.init()
        tmp_output_dir = "tmp_processed"
        os.makedirs(tmp_output_dir, exist_ok=True)
        if args.model_name in {"lstm", "transformer", "rnn"}:
            print("Stage 1: Extracting sequences for BPE training in parallel with Ray...")
            seq_object_refs = [
                ray_extract_sequences.remote(
                    str(f), s3_options, args.model_name, idx
                )
                for idx, f in enumerate(resolved_data_files)
            ]
            seq_lists = list(tqdm(ray.get(seq_object_refs), total=len(resolved_data_files), desc="Extracting sequences"))
            all_seqs = [seq for sublist in seq_lists for seq in sublist]
            print(f"Collected {len(all_seqs):,} sequences for BPE training.")
            # Train BPE tokenizer
            print("Training BPE tokenizer...")
            bpe_tokenizer = Tokenizer(models.BPE())
            bpe_tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern=r" ", behavior="removed")
            trainer = trainers.BpeTrainer(vocab_size=args.bpe_vocab, special_tokens=["[PAD]", "[UNK]", "[END]"])
            bpe_tokenizer.train_from_iterator(all_seqs, trainer)
            bpe_tokenizer_path = os.path.join(tmp_output_dir, "bpe_tokenizer.json")
            bpe_tokenizer.save(bpe_tokenizer_path)
            print(f"Saved BPE tokenizer to {bpe_tokenizer_path}")
            print("Stage 2: Encoding data with BPE tokenizer in parallel with Ray...")
            encode_object_refs = [
                ray_encode_with_bpe.remote(
                    str(f), s3_options, tmp_output_dir, bpe_tokenizer_path, args.max_len, idx
                )
                for idx, f in enumerate(resolved_data_files)
            ]
            processed_paths = list(tqdm(ray.get(encode_object_refs), total=len(resolved_data_files), desc="Encoding files"))
            from datasets import load_from_disk, concatenate_datasets  # type: ignore
            all_datasets = [load_from_disk(p) for p in processed_paths]
        else:
            print(f"Processing {len(resolved_data_files)} files in parallel with Ray ({args.num_workers} workers)...")
            object_refs = [
                ray_process_and_save_dataset.remote(
                    str(f), s3_options, tmp_output_dir, args.model_name, args.max_len, idx
                )
                for idx, f in enumerate(resolved_data_files)
            ]
            processed_paths = list(tqdm(ray.get(object_refs), total=len(resolved_data_files), desc="Processing files"))
            all_datasets = [load_from_disk(p) for p in processed_paths]
    else:
        with ThreadPoolExecutor(max_workers=16) as executor:
            from functools import partial
            worker_fn = partial(load_single_dataset, storage_options=s3_options)
            print(f"Loading {len(resolved_data_files)} files in parallel...")
            all_datasets = list(tqdm(executor.map(worker_fn, resolved_data_files), total=len(resolved_data_files), desc="Loading files"))
    print("Concatenating and splitting dataset...")
    from datasets import load_from_disk, concatenate_datasets  # type: ignore

    base = concatenate_datasets(all_datasets)
    step1 = base.train_test_split(test_size=0.1, shuffle=True, seed=42)
    tr_val = step1["train"].train_test_split(test_size=0.111111, shuffle=True, seed=42)
    splits = {"train": tr_val["train"], "val": tr_val["test"], "test": step1["test"]}


    # --- 2. Process Data Based on Model Type ---
    raw_dataset_dict = None
    model_hf_id = MODEL_CARDS[args.model_name]
    raw_dataset_dict = None
    
    if args.model_name in {"bert", "protbert", "esm", "llama"}:
        print(f"Using Hugging Face model path for: {model_hf_id}")
        hf_tok = AutoTokenizer.from_pretrained(model_hf_id, trust_remote_code=True)
        
        def proc_hf(ds: Dataset, split_name_str: str) -> Dataset:
            join, start, end = JOIN_STYLE[args.model_name], START_STYLE[args.model_name], END_STYLE[args.model_name]
            all_combo_ids, all_combo_feats = [], []
            print(f"Manually exploding {len(ds):,} rows for the '{split_name_str}' split...")
            for example in tqdm(ds, desc=f"Exploding {split_name_str}"):
                exploded_data = explode_example(example, join, start, end, model_name=args.model_name)
                all_combo_ids.extend(exploded_data["combo_id"])
                all_combo_feats.extend(exploded_data["combo_feats"])

            exploded_dataset = Dataset.from_dict({"combo_id": all_combo_ids, "combo_feats": all_combo_feats})
            
            def tokenize_clean_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
                return hf_tok(batch["combo_id"], truncation=True, max_length=args.max_len)
            
            return exploded_dataset.map(tokenize_clean_batch, batched=True, batch_size=2000, num_proc=num_cores)

        raw_final = {n: proc_hf(ds, n) for n, ds in splits.items()}
        raw_dataset_dict = DatasetDict(raw_final)

    else:  # custom BPE path for lstm/transformer
        print("Using custom BPE model path...")
        # Train tokenizer on the train split
        train_tagged_seqs = (s for row in splits["train"].map(lambda r:{"tagged": tag_bpe(r)}, num_proc=num_cores)["tagged"] for s in row)
        bpe_tok = train_bpe_tokenizer(train_tagged_seqs, vocab_size=args.bpe_vocab)
        
        def proc_custom(ds: Dataset) -> Dataset:
            # First, tag the sequences
            tagged = ds.map(lambda r: {"tagged": tag_bpe(r)}, num_proc=num_cores)
            
            model_type = "rnn" if args.model_name == "lstm" else "transformer"
            
            # Process each example individually to avoid batch size issues
            def encode_single_example(example: Dict[str, Any]) -> Dict[str, List[List[int]]]:
                tagged_seqs = example["tagged"]
                all_input_ids = []
                all_target_ids = []
                
                for seq in tagged_seqs:
                    ds = AminoAcidDataset([seq], bpe_tok, args.max_len, model_type, tr_long=args.truncate_long)
                    for x, y in ds:
                        x_list = x.tolist()
                        y_list = y.tolist()
                        
                        # Pad or truncate to seq_len
                        if len(x_list) < args.max_len:
                            x_list.extend([bpe_tok.token_to_id(PAD_TOKEN_STR) or PAD_TOKEN_ID] * (args.max_len - len(x_list)))
                            y_list.extend([bpe_tok.token_to_id(PAD_TOKEN_STR) or PAD_TOKEN_ID] * (args.max_len - len(y_list)))
                        elif len(x_list) > args.max_len:
                            x_list = x_list[:args.max_len]
                            y_list = y_list[:args.max_len]
                        
                        all_input_ids.append(x_list)
                        all_target_ids.append(y_list)
                
                return {"input_ids": all_input_ids, "target_ids": all_target_ids}
            
            return tagged.map(
                encode_single_example,
                remove_columns=["tagged"],
                num_proc=num_cores
            )
            
        raw_final = {n: proc_custom(ds) for n, ds in splits.items()}
        raw_dataset_dict = DatasetDict(raw_final)

    # --- 3. Save the Raw Dataset ---
    out_raw_path = args.output_raw
    print(f"Saving raw tokenized dataset to {out_raw_path}")
    
    if out_raw_path.startswith("s3://"):
        s3_filesystem = s3fs.S3FileSystem(key=args.s3_key, secret=args.s3_secret, token=args.s3_token)
        raw_dataset_dict.save_to_disk(out_raw_path, storage_options=s3_filesystem.storage_options)
    else:
        Path(out_raw_path).mkdir(parents=True, exist_ok=True)
        raw_dataset_dict.save_to_disk(out_raw_path)

    # --- S3 Sync if requested ---
    if args.s3_output_path:
        print(f"Syncing output directory to S3: {args.s3_output_path}")
        sync_cmd = ["aws", "s3", "sync", out_raw_path, args.s3_output_path]
        result = subprocess.run(sync_cmd)
        if result.returncode == 0:
            print(f"Successfully synced to {args.s3_output_path}")
        else:
            print(f"[ERROR] Failed to sync to {args.s3_output_path}")

    print("Raw data generation complete.")

    

if __name__=="__main__":
    main()