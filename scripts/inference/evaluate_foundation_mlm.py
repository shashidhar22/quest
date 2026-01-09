#!/usr/bin/env python3
"""
evaluate_foundation_mlm.py
────────────────────────────────────────────────────────────────────────────────
Evaluation script for PEFT ESM2 MLM model trained on TCR, peptide, MHC sequences.

This script evaluates the fine-tuned ESM2 650M model with LoRA adapters on
the validation dataset, computing:
- Overall accuracy, loss, and perplexity
- Per-batch metrics for monitoring
- Sample predictions with masked/true/predicted sequences

Output:
- Writes a detailed report with metrics and 20 example predictions

Usage:
    python scripts/inference/evaluate_foundation_mlm.py \
        --checkpoint_path data/model/esm2t33_foundation_100M/best_model.pt \
        --dataset_path data/foundation_100M/validation \
        --batch_size 32 \
        --output_report evaluation_report.txt

Multi-GPU:
    torchrun --nproc_per_node=4 scripts/inference/evaluate_foundation_mlm.py \
        --checkpoint_path data/model/esm2t33_foundation_100M/best_model.pt \
        --dataset_path data/foundation_100M/validation \
        --batch_size 64
"""

import argparse
import glob
import json
import math
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_from_disk, concatenate_datasets
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DataCollatorForMLMDynamic:
    """
    Simple MLM collator with dynamic padding to batch max length.
    Applies 15% masking on-the-fly following BERT masking strategy:
    - 80% -> [MASK]
    - 10% -> random token
    - 10% -> keep original
    """

    def __init__(self, tokenizer, mlm_probability: float = 0.15, pad_to_multiple_of: int = 8):
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
        self.special_token_ids = torch.tensor([x for x in special_ids if x is not None], dtype=torch.long)

    def __call__(self, examples):
        """Dynamically pad to batch max length using vectorized operations."""
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

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _apply_mlm_masking(self, input_ids, attention_mask):
        """Apply MLM masking using vectorized operations."""
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


def load_sharded_dataset(dataset_path: str):
    """Load a sharded HuggingFace dataset."""
    shard_dirs = sorted(glob.glob(os.path.join(dataset_path, "shard_*")))

    if not shard_dirs:
        # Try loading as single dataset
        return load_from_disk(dataset_path)

    print(f"Found {len(shard_dirs)} shards")
    datasets = []
    for shard_dir in tqdm(shard_dirs, desc="Loading shards"):
        ds = load_from_disk(shard_dir)
        datasets.append(ds)

    return concatenate_datasets(datasets)


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    use_flash_attention: bool = True,
) -> tuple:
    """
    Load the PEFT model from a native PyTorch checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
        use_flash_attention: Whether to use Flash Attention 2

    Returns:
        (model, tokenizer, config)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = checkpoint.get("config", {})
    model_name = config.get("model_name", "facebook/esm2_t33_650M_UR50D")

    print(f"Base model: {model_name}")
    print(f"Best validation loss from training: {checkpoint.get('best_val_loss', 'N/A'):.6f}")
    print(f"Trained for {checkpoint.get('global_step', 'N/A')} steps")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    # Load base model
    if use_flash_attention:
        try:
            model = AutoModelForMaskedLM.from_pretrained(
                model_name,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
            )
            print("Using Flash Attention 2")
        except Exception as e:
            print(f"Flash Attention 2 not available: {e}")
            model = AutoModelForMaskedLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
            )
    else:
        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )

    # Apply LoRA with same config as training
    if config.get("use_lora", True):
        lora_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.05),
            target_modules=["query", "key", "value", "dense"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        print(f"Applied LoRA (r={config.get('lora_r', 16)}, alpha={config.get('lora_alpha', 32)})")

    # Load state dict
    state_dict = checkpoint["model_state_dict"]

    # Remove DDP wrapper prefix if present (module.)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    print("Loaded model weights from checkpoint")

    model = model.to(device)
    model.eval()

    return model, tokenizer, config


def compute_mlm_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Compute MLM metrics for a batch.

    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        labels: Ground truth labels [batch_size, seq_len] (-100 for non-masked)

    Returns:
        Dictionary with accuracy, loss, perplexity, and counts
    """
    mask = labels != -100

    if mask.sum() == 0:
        return {
            "accuracy": 0.0,
            "loss": 0.0,
            "perplexity": float('inf'),
            "num_masked": 0,
            "num_correct": 0,
        }

    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    accuracy = correct / total if total > 0 else 0.0

    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    ).item()

    perplexity = math.exp(loss) if loss < 100 else float('inf')

    return {
        "accuracy": accuracy,
        "loss": loss,
        "perplexity": perplexity,
        "num_masked": total,
        "num_correct": correct,
    }


def evaluate(
    model,
    dataloader: DataLoader,
    device: torch.device,
    tokenizer,
    is_distributed: bool = False,
    local_rank: int = 0,
    num_samples: int = 20,
) -> Dict[str, Any]:
    """
    Run evaluation on the validation dataset.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for validation data
        device: Device to run on
        tokenizer: Tokenizer for decoding sequences
        is_distributed: Whether using distributed evaluation
        local_rank: Local rank for progress bar display
        num_samples: Number of example predictions to collect

    Returns:
        Dictionary with overall metrics and sample predictions
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    num_batches = 0

    # Collect sample predictions
    sample_predictions = []
    samples_collected = 0

    is_main = local_rank == 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=not is_main):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            logits = outputs.logits.float()
            predictions = torch.argmax(logits, dim=-1)
            metrics = compute_mlm_metrics(logits, labels)

            if metrics["num_masked"] > 0:
                total_loss += metrics["loss"] * metrics["num_masked"]
                total_correct += metrics["num_correct"]
                total_masked += metrics["num_masked"]
                num_batches += 1

            # Collect sample predictions (only on main process, only if we need more)
            if is_main and samples_collected < num_samples:
                batch_size = input_ids.size(0)
                for i in range(batch_size):
                    if samples_collected >= num_samples:
                        break

                    # Get mask positions for this example
                    mask = labels[i] != -100
                    if mask.sum() == 0:
                        continue

                    # Decode sequences
                    input_seq = input_ids[i].cpu().tolist()
                    label_seq = labels[i].cpu().tolist()
                    pred_seq = predictions[i].cpu().tolist()

                    # Create the "true" sequence by replacing masked positions with true labels
                    true_seq = input_seq.copy()
                    for j, (lbl, is_masked) in enumerate(zip(label_seq, mask.cpu().tolist())):
                        if is_masked:
                            true_seq[j] = lbl

                    # Decode (remove padding)
                    seq_len = attention_mask[i].sum().item()
                    masked_decoded = tokenizer.decode(input_seq[:seq_len], skip_special_tokens=False)
                    true_decoded = tokenizer.decode(true_seq[:seq_len], skip_special_tokens=False)
                    pred_decoded_seq = true_seq.copy()  # Start with true, replace masked with predictions
                    for j, is_masked in enumerate(mask.cpu().tolist()):
                        if is_masked:
                            pred_decoded_seq[j] = pred_seq[j]
                    pred_decoded = tokenizer.decode(pred_decoded_seq[:seq_len], skip_special_tokens=False)

                    # Count correct/incorrect for this example
                    mask_positions = mask.cpu().tolist()
                    example_correct = sum(
                        1 for j, m in enumerate(mask_positions) if m and pred_seq[j] == label_seq[j]
                    )
                    example_total = sum(mask_positions)

                    sample_predictions.append({
                        "example_id": samples_collected,
                        "masked_sequence": masked_decoded,
                        "true_sequence": true_decoded,
                        "predicted_sequence": pred_decoded,
                        "num_masked_tokens": example_total,
                        "num_correct": example_correct,
                        "accuracy": example_correct / example_total if example_total > 0 else 0.0,
                    })
                    samples_collected += 1

    # Aggregate across GPUs if distributed
    if is_distributed:
        stats = torch.tensor([total_loss, total_correct, total_masked, num_batches],
                           dtype=torch.float64, device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, total_correct, total_masked, num_batches = stats.tolist()

    # Compute final metrics
    if total_masked > 0:
        overall_accuracy = total_correct / total_masked
        overall_loss = total_loss / total_masked
        overall_perplexity = math.exp(overall_loss) if overall_loss < 100 else float('inf')
    else:
        overall_accuracy = 0.0
        overall_loss = 0.0
        overall_perplexity = float('inf')

    return {
        "metrics": {
            "accuracy": overall_accuracy,
            "loss": overall_loss,
            "perplexity": overall_perplexity,
            "total_masked_tokens": int(total_masked),
            "total_correct": int(total_correct),
            "num_batches": int(num_batches),
        },
        "sample_predictions": sample_predictions,
    }


def write_report(
    results: Dict[str, Any],
    config: Dict[str, Any],
    args: argparse.Namespace,
    output_path: str,
) -> None:
    """
    Write evaluation report to file.

    Args:
        results: Evaluation results including metrics and sample predictions
        config: Model training configuration
        args: Command line arguments
        output_path: Path to write report
    """
    metrics = results["metrics"]
    samples = results["sample_predictions"]

    with open(output_path, "w") as f:
        # Header
        f.write("=" * 100 + "\n")
        f.write("ESM2 MLM EVALUATION REPORT\n")
        f.write("=" * 100 + "\n\n")

        # Metadata
        f.write("EVALUATION DETAILS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoint: {args.checkpoint_path}\n")
        f.write(f"Dataset: {args.dataset_path}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        if args.num_samples:
            f.write(f"Num Samples: {args.num_samples}\n")
        f.write("\n")

        # Model config
        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 50 + "\n")
        f.write(f"Base Model: {config.get('model_name', 'N/A')}\n")
        f.write(f"LoRA Rank (r): {config.get('lora_r', 'N/A')}\n")
        f.write(f"LoRA Alpha: {config.get('lora_alpha', 'N/A')}\n")
        f.write(f"MLM Probability: {config.get('mlm_probability', 'N/A')}\n")
        f.write(f"Training Steps: {config.get('global_step', 'N/A')}\n")
        f.write("\n")

        # Overall metrics
        f.write("=" * 100 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"Loss:               {metrics['loss']:.6f}\n")
        f.write(f"Perplexity:         {metrics['perplexity']:.4f}\n")
        f.write(f"Total Masked Tokens: {metrics['total_masked_tokens']:,}\n")
        f.write(f"Total Correct:      {metrics['total_correct']:,}\n")
        f.write(f"Number of Batches:  {metrics['num_batches']:,}\n")
        f.write("\n")

        # Sample predictions
        f.write("=" * 100 + "\n")
        f.write(f"SAMPLE PREDICTIONS ({len(samples)} examples)\n")
        f.write("=" * 100 + "\n\n")

        for sample in samples:
            f.write(f"--- Example {sample['example_id'] + 1} ---\n")
            f.write(f"Accuracy: {sample['accuracy']:.2%} ({sample['num_correct']}/{sample['num_masked_tokens']} tokens)\n\n")
            f.write(f"MASKED SEQUENCE:\n{sample['masked_sequence']}\n\n")
            f.write(f"TRUE SEQUENCE:\n{sample['true_sequence']}\n\n")
            f.write(f"PREDICTED SEQUENCE:\n{sample['predicted_sequence']}\n\n")
            f.write("-" * 80 + "\n\n")

        f.write("=" * 100 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 100 + "\n")

    print(f"\nReport written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PEFT ESM2 MLM model")

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/ubuntu/quest/data/model/esm2t33_foundation_100M/best_model.pt",
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/ubuntu/quest/data/foundation_100M/validation",
        help="Path to validation dataset (sharded HuggingFace dataset)"
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default="evaluation_report.txt",
        help="Path to write the evaluation report"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of validation examples to use (default: use all). "
             "Randomly samples from the dataset if specified."
    )
    parser.add_argument(
        "--num_example_predictions",
        type=int,
        default=20,
        help="Number of example predictions to include in the report"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no_flash_attention",
        action="store_true",
        help="Disable Flash Attention 2"
    )

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup distributed training
    is_distributed = "LOCAL_RANK" in os.environ

    if is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")

        if local_rank == 0:
            print(f"Distributed evaluation with {world_size} GPUs")
    else:
        local_rank = 0
        global_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = local_rank == 0

    # Load model
    model, tokenizer, config = load_model_from_checkpoint(
        args.checkpoint_path,
        device,
        use_flash_attention=not args.no_flash_attention,
    )

    # Wrap with DDP if distributed
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Load dataset
    if is_main:
        print(f"\nLoading dataset from: {args.dataset_path}")

    dataset = load_sharded_dataset(args.dataset_path)

    if is_main:
        print(f"Dataset size: {len(dataset):,} examples")
        print(f"Columns: {dataset.column_names}")

    # Sample if num_samples is specified
    if args.num_samples is not None and len(dataset) > args.num_samples:
        if is_main:
            print(f"Sampling {args.num_samples:,} examples from dataset")
        dataset = dataset.shuffle(seed=args.seed).select(range(args.num_samples))

    # Create data collator (applies MLM masking on-the-fly)
    mlm_probability = config.get("mlm_probability", 0.15)
    data_collator = DataCollatorForMLMDynamic(
        tokenizer=tokenizer,
        mlm_probability=mlm_probability,
    )

    if is_main:
        print(f"MLM probability: {mlm_probability}")

    # Create dataloader
    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=False)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=data_collator,
    )

    # Run evaluation
    if is_main:
        print("\n" + "=" * 80)
        print("STARTING EVALUATION")
        print("=" * 80 + "\n")

    results = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        tokenizer=tokenizer,
        is_distributed=is_distributed,
        local_rank=local_rank,
        num_samples=args.num_example_predictions,
    )

    # Print results and write report (main process only)
    if is_main:
        metrics = results["metrics"]
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"  Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Loss:         {metrics['loss']:.6f}")
        print(f"  Perplexity:   {metrics['perplexity']:.4f}")
        print(f"  Total tokens: {metrics['total_masked_tokens']:,}")
        print(f"  Correct:      {metrics['total_correct']:,}")
        print(f"  Batches:      {metrics['num_batches']:,}")
        print("=" * 80)

        # Write report
        write_report(results, config, args, args.output_report)

        print("\nEvaluation complete!")

    # Cleanup
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
