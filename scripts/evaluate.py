#!/usr/bin/env python
"""
Evaluate any Hugging Face *masked-language-model* checkpoint by streaming
a dataset folder from an S3 bucket with a full progress bar.
"""
import argparse, math, torch, json, s3fs
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
)
from tqdm.auto import tqdm

# CLI section remains the same...
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--s3_folder_path", required=True,
                   help="S3 path to the PARENT dataset folder, e.g. 's3://my-bucket/my-hf-dataset/'.")
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
    print(f"🔄  Loading model and tokenizer for '{args.model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model     = AutoModelForMaskedLM.from_pretrained(args.model_name, trust_remote_code=True)
    model.to(device).eval()

    s3_root_path = args.s3_folder_path.rstrip('/')

    # NEW: Step 2️⃣a - Get the total number of examples for the progress bar
    print("🔎  Reading dataset metadata for total count...")
    s3 = s3fs.S3FileSystem()
    info_path = f"{s3_root_path}/test/dataset_info.json" # Path to the specific split's info
    with s3.open(info_path, 'r') as f:
        metadata = json.load(f)
    
    num_examples = metadata['splits']['train']['num_examples']
    num_batches = math.ceil(num_examples / args.batch_size)
    print(f"✅  Found {num_examples} examples, which will be processed in {num_batches} batches.")


    # 2️⃣b  Dataset → Define file paths and stream from S3
    data_files = {
        "test": f"{s3_root_path}/test/*.arrow",
    }
    
    raw_ds_dict = load_dataset("arrow", data_files=data_files, streaming=True)
    dataset_to_eval = raw_ds_dict["test"]
    
    print(f"▶️   Starting streaming of 'test' split from: {args.s3_folder_path}")

    def tokenize_function(examples):
        return tokenizer(
            examples["sequence"],
            truncation=True,
            max_length=args.max_len,
        )

    first_example = next(iter(dataset_to_eval))
    tokenized_ds = dataset_to_eval.map(
        tokenize_function,
        remove_columns=first_example.keys(),
    )
    tokenized_ds = tokenized_ds.with_format("torch")

    # 3️⃣  Loader + collator
    collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=True, mlm_probability=args.mlm_prob
    )
    loader = DataLoader(tokenized_ds, batch_size=args.batch_size, collate_fn=collator)

    # 4️⃣  Eval
    losses, correct, masked = [], 0, 0
    # MODIFIED: Pass the calculated `num_batches` to tqdm's `total` argument
    for batch in tqdm(loader, total=num_batches, desc="Evaluating test data"):
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
    print(f"dataset          : {args.s3_folder_path}")
    print(f"avg loss         : {avg_loss:.4f}")
    print(f"perplexity       : {math.exp(avg_loss):.2f}")
    print(f"mask accuracy    : {100*correct/masked:.2f} %")
    print("--------------------------------")


if __name__ == "__main__":
    main()