#!/usr/bin/env python
"""
Evaluate any Hugging Face *masked-language-model* checkpoint by streaming
a dataset folder from an S3 bucket.

Outputs:
  • average MLM loss
  • perplexity  (exp(loss))
  • masked-token top-1 accuracy
"""
import s3fs

import argparse, math, torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
)
from tqdm.auto import tqdm

from datasets import load_from_disk

# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    # MODIFIED: Argument now points to the S3 folder, not a single file.
    p.add_argument("--s3_folder_path", required=True,
                   help="S3 path to the dataset folder, e.g. 's3://my-bucket/my-hf-dataset/'.")
    p.add_argument("--model_name",  required=True,
                   help="Model card / Hub ID, e.g. 'Rostlab/prot_bert' or 'bert-base-uncased'.")
    p.add_argument("--max_len", type=int, required=True,
                   help="Maximum sequence length for truncation.")
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--mlm_prob",    type=float, default=0.15)
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------
def main():
    args   = parse_args()
    device = torch.device(args.device)

    # 1️⃣  Hub pull
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model     = AutoModelForMaskedLM.from_pretrained(args.model_name, trust_remote_code=True)
    model.to(device).eval()

    # 2️⃣  Dataset → Stream from S3 folder and tokenize on the fly
    # MODIFIED: Load dataset directly from the S3 folder path.
    # The `datasets` library infers the format from the folder's contents.

    # --- START DEBUGGING BLOCK ---
    #print("--- DEBUGGING ---")
    #print(f"The type of s3_folder_path is: {type(args.s3_folder_path)}")
    #print(f"The value of s3_folder_path is: '{args.s3_folder_path}'")
    #exit()
    # --- END DEBUGGING BLOCK ---
    s3_root_path = args.s3_folder_path.rstrip('/')
    data_files = {
        "train":      f"{s3_root_path}/train/*.arrow",
        "validation": f"{s3_root_path}/validation/*.arrow",   # adjust if you have it
        "test":       f"{s3_root_path}/test/*.arrow",
    }
    raw_ds = load_dataset("arrow", data_files=data_files, streaming=True, storage_options={"anon": False})
    test_dataset = raw_ds["test"]  # Use the test split for evaluation
    print(f"✅  Streaming dataset from folder: {args.s3_folder_path}")
    print(f"✂️  Truncating sequences to {args.max_len} tokens")

    # This tokenization function assumes your dataset has a 'sequence' column.
    # If your column is named differently (e.g., 'text'), change it here.
    def tokenize_function(examples):
        return tokenizer(
            examples["sequence"],
            truncation=True,
            max_length=args.max_len,
        )

    # Get column names from the first example to correctly remove them later
    # This is safe as it only fetches one item from the stream.
    first_example = next(iter(test_dataset))
    
    ds = test_dataset.map(
        tokenize_function,
        remove_columns=first_example.keys(), # drop raw text columns
    )
    ds = ds.with_format("torch")


    # 3️⃣  Loader + collator
    collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=True, mlm_probability=args.mlm_prob
    )
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collator)


    # 4️⃣  Eval
    # Note: tqdm will not show a progress bar total because the stream length is unknown
    losses, correct, masked = [], 0, 0
    for batch in tqdm(loader, desc="evaluating stream"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch)
        losses.append(out.loss.item())

        m = batch["labels"] != -100
        preds = out.logits.argmax(dim=-1)
        correct += (preds[m] == batch["labels"][m]).sum().item()
        masked  += m.sum().item()

    avg_loss = sum(losses) / len(losses)
    print("\n----- MLM evaluation -----")
    print(f"model            : {args.model_name}")
    print(f"avg loss         : {avg_loss:.4f}")
    print(f"perplexity       : {math.exp(avg_loss):.2f}")
    print(f"mask accuracy    : {100*correct/masked:.2f} %")
    print("--------------------------------")


if __name__ == "__main__":
    main()