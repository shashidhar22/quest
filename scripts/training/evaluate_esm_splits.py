#!/usr/bin/env python3
"""
evaluate_esm_splits.py
──────────────────────────────────────────────────────────────────────────
Evaluate a PEFT LoRA ESM2 model on multiple validation splits with wandb logging.

Loads a LoRA adapter, merges it into the base model, and evaluates on each
specified split using on-the-fly MLM masking (same collator as training).

Usage:
    python scripts/training/evaluate_esm_splits.py \
        --model_path data/tcrcbench2/A1/final_model \
        --data_dir /home/ubuntu/quest/data/tokenized/mlm_full/esm2 \
        --splits val_singles val_pairs val_triplets val_quartets val_quintets \
        --wandb_project tcrbench2-eval \
        --batch_size 64
"""

import argparse
import math
import os

import torch
import numpy as np
from datasets import load_from_disk
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_SPLITS = [
    "val_singles",
    "val_pairs",
    "val_triplets",
    "val_quartets",
    "val_quintets",
]


class DataCollatorForMLMDynamic:
    """MLM collator with dynamic padding, separator token protection, and BERT masking."""

    def __init__(self, tokenizer, mlm_probability=0.15, pad_to_multiple_of=8, separator_token_id=None):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = len(tokenizer)
        self.pad_to_multiple_of = pad_to_multiple_of

        special_ids = [
            tokenizer.pad_token_id,
            tokenizer.cls_token_id,
            tokenizer.eos_token_id,
            tokenizer.sep_token_id,
            tokenizer.unk_token_id,
        ]
        if separator_token_id is not None:
            special_ids.append(separator_token_id)
        self.special_token_ids = torch.tensor([x for x in special_ids if x is not None], dtype=torch.long)

    def __call__(self, examples):
        sequences = []
        for ex in examples:
            ids = ex.get("input_ids", ex) if isinstance(ex, dict) else ex
            if isinstance(ids, list):
                sequences.append(torch.tensor(ids, dtype=torch.long))
            else:
                sequences.append(ids.long() if ids.dtype != torch.long else ids)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=self.pad_token_id
        )

        lengths = torch.tensor([len(s) for s in sequences])
        max_len = input_ids.size(1)
        attention_mask = (torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)).long()

        if self.pad_to_multiple_of and max_len % self.pad_to_multiple_of != 0:
            pad_len = self.pad_to_multiple_of - (max_len % self.pad_to_multiple_of)
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=self.pad_token_id)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=0)

        input_ids, labels = self._apply_mlm_masking(input_ids, attention_mask)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def _apply_mlm_masking(self, input_ids, attention_mask):
        labels = input_ids.clone()
        rand_vals = torch.rand(input_ids.shape[0], input_ids.shape[1], 2)
        rand_mask = rand_vals[..., 0]
        mask_type = rand_vals[..., 1]

        special_mask = torch.isin(input_ids, self.special_token_ids)
        valid_mask = (attention_mask == 1) & ~special_mask
        masked_indices = valid_mask & (rand_mask < self.mlm_probability)
        labels[~masked_indices] = -100

        input_ids = input_ids.clone()
        input_ids[masked_indices & (mask_type < 0.8)] = self.mask_token_id

        random_token_indices = masked_indices & (mask_type >= 0.8) & (mask_type < 0.9)
        if random_token_indices.any():
            input_ids[random_token_indices] = torch.randint(
                self.vocab_size, (random_token_indices.sum(),), dtype=torch.long
            )
        return input_ids, labels


def evaluate_split(model, dataloader, device):
    """Evaluate model on a single split. Returns loss, accuracy, perplexity, num_samples."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            logits = outputs.logits.float()
            mask = labels != -100
            if mask.sum() == 0:
                continue

            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions[mask] == labels[mask]).sum().item()
            num_masked = mask.sum().item()

            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)).item()

            total_loss += loss * num_masked
            total_correct += correct
            total_masked += num_masked

    if total_masked == 0:
        return {"loss": 0.0, "accuracy": 0.0, "perplexity": float("inf"), "num_masked_tokens": 0}

    avg_loss = total_loss / total_masked
    accuracy = total_correct / total_masked
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "perplexity": perplexity,
        "num_masked_tokens": int(total_masked),
        "num_correct": int(total_correct),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA ESM2 on multiple val splits")
    parser.add_argument("--model_path", type=str, required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--data_dir", type=str, required=True, help="Parent dir containing split subdirs")
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS, help="Split names to evaluate")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name (disables wandb if not set)")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--separator_token_id", type=int, default=30, help="Separator token ID to protect from masking")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load base model + LoRA adapter, merge and unload
    print(f"Loading LoRA adapter from: {args.model_path}")
    base_model = AutoModelForMaskedLM.from_pretrained(
        "facebook/esm2_t12_35M_UR50D",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model = model.merge_and_unload()
    model = model.to(device)
    model.eval()
    print("Model loaded and merged.")

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")

    # Init wandb
    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"eval-{os.path.basename(args.model_path)}",
            config={
                "model_path": args.model_path,
                "data_dir": args.data_dir,
                "splits": args.splits,
                "batch_size": args.batch_size,
                "mlm_probability": args.mlm_probability,
                "seed": args.seed,
                "separator_token_id": args.separator_token_id,
            },
        )

    collator = DataCollatorForMLMDynamic(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability,
        separator_token_id=args.separator_token_id,
    )

    all_results = {}

    for split_name in args.splits:
        split_path = os.path.join(args.data_dir, split_name)
        if not os.path.exists(split_path):
            print(f"WARNING: Split {split_name} not found at {split_path}, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {split_name}")
        print(f"{'='*60}")

        dataset = load_from_disk(split_path)
        dataset.set_format("torch", columns=["input_ids"])
        print(f"  Samples: {len(dataset):,}")

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collator,
        )

        metrics = evaluate_split(model, dataloader, device)
        all_results[split_name] = metrics

        print(f"  Loss:       {metrics['loss']:.6f}")
        print(f"  Accuracy:   {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Perplexity: {metrics['perplexity']:.4f}")
        print(f"  Masked tokens: {metrics['num_masked_tokens']:,}")

        if wandb_run:
            wandb.log({
                f"{split_name}/loss": metrics["loss"],
                f"{split_name}/accuracy": metrics["accuracy"],
                f"{split_name}/perplexity": metrics["perplexity"],
                f"{split_name}/num_masked_tokens": metrics["num_masked_tokens"],
            })

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Split':<20} {'Loss':>10} {'Accuracy':>10} {'Perplexity':>12} {'Tokens':>12}")
    print(f"{'-'*20} {'-'*10} {'-'*10} {'-'*12} {'-'*12}")
    for split_name, metrics in all_results.items():
        print(
            f"{split_name:<20} {metrics['loss']:>10.6f} {metrics['accuracy']:>10.4f} "
            f"{metrics['perplexity']:>12.4f} {metrics['num_masked_tokens']:>12,}"
        )

    # Log summary table to wandb
    if wandb_run:
        import wandb
        table = wandb.Table(
            columns=["split", "loss", "accuracy", "perplexity", "num_masked_tokens"],
            data=[
                [name, m["loss"], m["accuracy"], m["perplexity"], m["num_masked_tokens"]]
                for name, m in all_results.items()
            ],
        )
        wandb.log({"summary_table": table})

        # Also log as summary metrics
        for split_name, metrics in all_results.items():
            for metric_name, value in metrics.items():
                wandb.run.summary[f"{split_name}/{metric_name}"] = value

        wandb.finish()

    print("\nDone.")


if __name__ == "__main__":
    main()
